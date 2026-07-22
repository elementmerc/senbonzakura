#!/usr/bin/env python3
"""Multi-direction refusal abliteration for transformer language models, with an Optuna search.

Handles dense transformers and mixture-of-experts models, including the *fused* expert layout
(one batched 3D weight per layer) that recent transformers releases use for speed. Tools that
target only the *unfused* per-expert ModuleList layout can't reach the fused expert weights; this
one resolves the residual-writing down-projection across both, so the ablation applies to every
matrix that writes the residual stream. Optuna does the quality work: it finds the layers and
strengths that drive refusals toward zero without wrecking the model, guarded by KL divergence.

Method (Arditi et al., "refusal is mediated by a direction"), with Heretic's refinements plus two
additions here (5 multi-directional, 6 interpolated index):
  1. Extract per-layer refusal directions: the difference-of-means (bad - good), good-
     orthogonalised, PLUS up to KMAX-1 secondary axes from PCA of the bad residual cloud,
     giving an orthonormal basis of the refusal SUBSPACE at every layer.
  2. SEARCH (Optuna): a candidate is a windowed strength profile (peak position + strengths +
     width) AND how the directions are chosen (num_directions K; per-layer own directions vs a
     single interpolated direction_index shared across layers). Each trial applies the REAL
     norm-preserving weight bake (step 3) and restores from a pristine snapshot afterwards, so
     the search scores the exact model it will save (no activation-hook proxy, no proxy/bake
     gap). Score = refusals on the bad-eval set co-minimised with KL vs the original on the
     harmless set, with incoherence penalised directly.
  3. BAKE: for the winning config, norm-preservingly orthogonalise that direction span out of
     every residual-*writing* weight (each attn o_proj, each fused expert down_proj) so the
     ablation is permanent, then save.

The code is organised in three parts: pure module-level helpers (the refusal classifier and the
weight math, testable without a model), the Abliterator class (everything that needs the loaded
model: direction extraction, the reversible bake, evaluation, and the search), and a thin main().
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

import optuna

from .metrics import (
    REFUSAL,                # hard-refusal markers (kept in metrics.py, the single source)
    broken_rate,            # fraction of a batch that is wrecked output
    heretic_keyword_rate,   # Heretic-comparable refusal metric (the axis Heretic wins)
    is_broken,              # wrecked-output detector (empty / garbage / repetition)
    is_refusal,             # hard-refusal detector
    is_soft_refusal,        # hedged-compliance detector (the moralising lecture)
)
from .resources import ResourceGovernor, SearchProgress  # adaptive VRAM throttle + ETA
from .crashsafe import (  # crash-resilience: persist by default, recover a lost save, fail loud early
    MIN_TORCH, torch_version_ok, study_db_path, search_already_done, winning_config, config_to_bake_args)

__version__ = "0.3.0"

# Coherence guard thresholds. KL_TARGET is the "comfortably intact" mark used by the knee
# scalariser (a config under it pays no coherence surcharge); KL_CEIL is the hard "too damaged"
# line above which a trial is excluded outright.
KL_TARGET = 0.1
KL_CEIL = 0.25

# Knee-selection weights (Tier-1 P2 fix): the final pick minimises a WEIGHTED scalar of the three
# search objectives, not a lexicographic tuple that let the keyword axis fall to a tiebreaker. The
# keyword/hedging term carries real weight so the saved model actually reflects the axis Heretic wins.
KNEE_W_NONCOMPLIANCE = 1.0   # hard refusal + hedged compliance
KNEE_W_KEYWORD = 1.0         # Heretic keyword rate (its own axis, now steered at selection time)
KNEE_W_KL = 0.5             # coherence surcharge, applied only above KL_TARGET


def knee_scalar(ref, soft, heretic, kl):
    # The weighted knee score (P2): lower is better. The keyword/hedging axis carries real weight so
    # the final pick reflects the axis the search already optimises, instead of the old lexicographic
    # tuple where it only broke exact ties. KL surcharges only above the comfortably-intact target.
    return (KNEE_W_NONCOMPLIANCE * (ref + soft)
            + KNEE_W_KEYWORD * heretic
            + KNEE_W_KL * max(0.0, kl - KL_TARGET))


# A candidate PCA axis is kept as a refusal direction only if it separates the harmful and harmless
# residual clouds by at least this standardised mean difference (Cohen's d). Below it, the axis is
# within-harmful content/topic variance, not refusal, and ablating it strips capability (Tier-1 P1).
MIN_AXIS_SEPARATION = 0.5

# The "worse than anything real" score, used to keep damaged / unmeasured trials out of the running
# for best. A true infinity so no finite objective can ever tie or beat it.
WORST_SCORE = float("inf")


# ── weight math (pure) ───────────────────────────────────────────────────────────
def _orth_to(vec, basis):
    # Remove from `vec` its component along each (unit) row in `basis`.
    for u in basis:
        vec = vec - (vec @ u) * u
    return vec


def _axis_separation(bad, good, v):
    # Cohen's d: the standardised mean difference between the harmful and harmless residual
    # projections onto unit axis `v`. A genuine refusal axis separates the two clouds (large d);
    # a within-harmful topic/phrasing axis does not (small d). Used to keep only PCA axes that
    # actually carry refusal, so multi-direction ablation cuts refusal and not capability (P1).
    pb = bad @ v
    pg = good @ v
    md = (pb.mean() - pg.mean()).abs()
    pooled = ((pb.var(unbiased=False) + pg.var(unbiased=False)) / 2).clamp_min(1e-12).sqrt()
    return (md / pooled).item()


def _signal_certificate(bad, good, n_perm=25, seed=0):
    # Post-ablation "is the refusal SIGNAL actually gone" check on ONE layer's residual stream.
    # A robust, self-calibrating reimplementation of OBLITERATUS's RMT spectral-certification idea:
    # instead of a fragile Marchenko-Pastur constant, measure the linear harmful-vs-harmless
    # separation (Cohen's d along the mean-difference direction) on the REAL labels, then against
    # n_perm random label SHUFFLES, and report how many null-sigmas above chance the real
    # separation sits. This is orthogonal to KL: KL says "the model is still capable", this says
    # "the refusal signal is un-decodable". A refusal that survives spread thinly across many weak
    # directions (the extended-refusal / KAUST failure mode) shows here as a modest-but-real z that
    # a keyword refusal rate misses.
    #   bad, good : [N, H] post-ablation residuals for harmful / harmless prompts at one layer.
    # Returns (verdict, z): GREEN (z < 2, inside the shuffled null, refusal linearly gone),
    # YELLOW (2..5, real but weak/distributed), RED (>= 5, a clear surviving signal).
    Xb, Xg = bad.float(), good.float()

    def sep(a, b):
        dm = a.mean(0) - b.mean(0)
        u = dm / dm.norm().clamp_min(1e-8)               # empirical mean-difference direction
        pa, pb = a @ u, b @ u
        pooled = ((pa.var(unbiased=False) + pb.var(unbiased=False)) / 2).clamp_min(1e-12).sqrt()
        return ((pa.mean() - pb.mean()).abs() / pooled).item()

    real = sep(Xb, Xg)
    X = torch.cat([Xb, Xg], 0)
    nb, ntot = Xb.shape[0], Xb.shape[0] + Xg.shape[0]
    gen = torch.Generator().manual_seed(seed)
    null = torch.tensor([
        sep(X[p[:nb]], X[p[nb:]])
        for p in (torch.randperm(ntot, generator=gen) for _ in range(n_perm))
    ])
    # Cohen's d along a FITTED direction is inflated under the null too, so the shuffled separations
    # are the correct baseline; z measures the real separation against exactly that fitting bias.
    z = ((real - null.mean()) / null.std().clamp_min(1e-6)).item()
    verdict = "GREEN" if z < 2.0 else ("YELLOW" if z < 5.0 else "RED")
    return verdict, z


def _available_ram_bytes():
    # Best-effort available host RAM in bytes, for the snapshot pre-flight. Returns None when it
    # can't be determined (non-Linux / unreadable /proc), so the caller treats it as "unknown,
    # proceed" rather than refusing on a machine it simply couldn't measure.
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def layer_weight(idx, P, wmax, wmin, D):
    # Refinement 1 (Heretic windowed strength): the ablation strength PEAKS at layer
    # position P (wmax) and tapers linearly to wmin at distance D, and is ZERO beyond D.
    # Ablating early/late layers, where this direction is not the refusal direction, is
    # what destroyed coherence (KL 12-19) under a uniform-all-layers strength.
    dist = abs(idx - P)
    if dist > D:
        return 0.0
    return wmax + (dist / D) * (wmin - wmax)


@torch.no_grad()
def _sparsify_rows_(delta, sparsity):
    # Sparse surgery (OBLITERATUS-style, adapted to the norm-preserving projection): only KEEP the
    # top (1-sparsity) output-rows by edit magnitude, zero the edit on the rest. The rows with the
    # largest projection delta are the ones that actually WRITE the refusal direction; leaving the
    # low-projection rows pristine spares whatever capability they carry, for less collateral at the
    # same refusal removal on the rows that matter. `delta` is [..., out, in]; rows are the -2 dim.
    if sparsity <= 0.0:
        return delta
    mag = delta.norm(dim=-1)                            # [..., out] per-row edit magnitude
    keep = max(1, int(round((1.0 - sparsity) * mag.shape[-1])))
    thr = torch.topk(mag, keep, dim=-1).values.amin(dim=-1, keepdim=True)   # [..., 1]
    return delta * (mag >= thr).unsqueeze(-1)           # zero the untouched rows


def orthogonalize_np_(W, R, s, sparsity=0.0):
    # Refinement 4 (norm-preserving ablation; Heretic row_normalization=full / grimjim):
    # ablate on the row-normalized weight, renormalize, then RESTORE the original row norms.
    # Raw orthogonalization changed the norms and wrecked calibration (KL 12-19); preserving
    # them keeps the model intact. R is [K, H] (refinement 5): removes the whole span.
    # sparsity>0 restricts the edit to the top-magnitude rows (sparse surgery).
    Rf = R.to(W.device).float()                         # [K, H]
    Wf = W.float()                                      # [out=H, in]
    rn = Wf.norm(dim=1, keepdim=True).clamp_min(1e-8)   # [out,1] original row norms
    Wn = Wf / rn
    delta = s * (Rf.T @ (Rf @ Wn))                      # each column's projection onto span(R)
    Wn = Wn - _sparsify_rows_(delta, sparsity)
    Wn = Wn / Wn.norm(dim=1, keepdim=True).clamp_min(1e-8)
    W.copy_((Wn * rn).to(W.dtype))


@torch.no_grad()
def orthogonalize_np_3d_(W, R, s, sparsity=0.0):
    # Norm-preserving, fused experts [E, out, in]; row norms per (expert, out-row). R is [K, H].
    Rf = R.to(W.device).float()                         # [K, H]
    Wf = W.float()
    rn = Wf.norm(dim=2, keepdim=True).clamp_min(1e-8)   # [E,out,1]
    Wn = Wf / rn
    proj = torch.einsum("kh,ehi->eki", Rf, Wn)          # [E,K,in]
    delta = s * torch.einsum("kh,eki->ehi", Rf, proj)   # [E,out,in]
    Wn = Wn - _sparsify_rows_(delta, sparsity)          # per (expert, out-row) sparsify
    Wn = Wn / Wn.norm(dim=2, keepdim=True).clamp_min(1e-8)
    W.copy_((Wn * rn).to(W.dtype))


def _decoder_layers(model):
    # The list of decoder blocks, resolved across the common architecture trees rather than
    # assuming `model.model.layers`. Raises loud if none matches, so an unsupported model fails
    # at load with a clear message instead of an opaque AttributeError deep in the search.
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            return obj
    raise ValueError(
        f"could not find the decoder layer stack on {type(model).__name__}; looked for "
        "model.layers, transformer.h, gpt_neox.layers, model.decoder.layers.")


def _real_tensor(owner, name):
    # The REAL, editable weight tensor for owner.<name>, transparently unwrapping accelerate offload.
    # When a model is larger than VRAM, accelerate dispatches some layers to CPU (or disk): their
    # parameter becomes a META tensor with no data, and the resident copy lives in that module's
    # AlignDevicesHook.weights_map. weights_map returns a STABLE tensor object (so id()-based dirty
    # tracking holds) and in-place edits to it propagate to the next forward, so the norm-preserving
    # bake works on offloaded layers with no other change. Non-offloaded weights return unchanged.
    t = getattr(owner, name)
    if not getattr(t, "is_meta", False):
        return t
    hook = getattr(owner, "_hf_hook", None)
    wm = getattr(hook, "weights_map", None)
    if wm is not None:
        try:
            real = wm[name]
        except (KeyError, TypeError):
            real = None
        if real is not None and not getattr(real, "is_meta", False):
            return real
    raise ValueError(
        f"weight {name!r} on {type(owner).__name__} is on the meta device with no resident offload "
        "copy, so it cannot be abliterated. This usually means disk-offload without an in-memory "
        "cache; load with more VRAM or host-RAM headroom so the weights stay resident.")


def _owned_weight(parent, name):
    # parent.<name> is either a submodule that owns a `.weight` (a Linear) or a raw parameter/tensor
    # held directly under `name`. Resolve to the real editable tensor in both cases, unwrapping
    # accelerate offload wherever the resident copy actually lives (on the submodule, or on parent).
    obj = getattr(parent, name)
    if hasattr(obj, "weight"):
        return _real_tensor(obj, "weight")
    return _real_tensor(parent, name)


def _attn_outproj(layer):
    # The attention OUTPUT projection (the matrix that writes attention back into the residual
    # stream), across naming conventions. Linear-based only: GPT-2-style Conv1D out-projections
    # (transposed weight) are deliberately not returned, since the row-wise norm-preserving bake
    # assumes a [out, in] Linear weight.
    attn = (getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            or getattr(layer, "self_attention", None) or getattr(layer, "attn", None))
    if attn is not None:
        for name in ("o_proj", "out_proj", "dense"):
            p = getattr(attn, name, None)
            if p is not None and hasattr(p, "weight") and _real_tensor(p, "weight").dim() == 2:
                return _real_tensor(p, "weight")
    raise ValueError(
        f"could not locate a Linear attention output projection on this layer "
        f"(type {type(layer).__name__}); architecture not supported.")


def _classify(w):
    # Infer the bake kind from the tensor rank: 3D = fused expert stack [E, out, in], else 2D dense.
    return "fused3d" if w.dim() == 3 else "dense"


def _mlp_downprojs(mlp):
    # Every residual-WRITING down-projection inside one MLP / MoE block, as (kind, obj) entries.
    # kind is "dense" (2D weight), "fused3d" (batched [E, out, in] expert weight) or "list" (a list
    # of 2D per-expert weights). Covers dense, fused MoE (Qwen3-MoE / Granite output_linear),
    # unfused expert lists (OLMoE down_proj, Mixtral w2), and always-on shared experts.
    out = []
    if mlp is None:
        return out
    ol = getattr(mlp, "output_linear", None)                     # Granite-MoE parallel experts
    if ol is not None:
        w = _owned_weight(mlp, "output_linear")
        out.append((_classify(w), w))
    experts = getattr(mlp, "experts", None)
    if experts is not None:
        if hasattr(experts, "down_proj"):                        # Qwen3-MoE fused single tensor
            w = _owned_weight(experts, "down_proj")
            out.append((_classify(w), w))
        elif hasattr(experts, "w2"):                             # fused Mixtral-style single tensor
            w = _owned_weight(experts, "w2")
            out.append((_classify(w), w))
        else:                                                    # unfused per-expert Linear list
            ex = list(experts)
            if ex and hasattr(ex[0], "down_proj"):               # OLMoE
                out.append(("list", [_owned_weight(e, "down_proj") for e in ex]))
            elif ex and hasattr(ex[0], "w2"):                    # Mixtral unfused
                out.append(("list", [_owned_weight(e, "w2") for e in ex]))
    for attr in ("shared_expert", "shared_experts"):             # Qwen2-MoE / DeepSeek-MoE
        sh = getattr(mlp, attr, None)
        if sh is not None and hasattr(sh, "down_proj"):
            out.append(("dense", _owned_weight(sh, "down_proj")))
    if not out and hasattr(mlp, "down_proj"):                    # plain dense MLP
        out.append(("dense", _owned_weight(mlp, "down_proj")))
    return out


def layer_downproj(layer):
    # ALL residual-writing down-projections in this decoder layer, as (kind, obj) entries. A layer
    # may have more than one (a routed expert stack PLUS an always-on shared expert), and every one
    # writes the residual, so every one must be ablated. Granite exposes its experts under
    # block_sparse_moe; the dense / Qwen-family / Mixtral / OLMoE layouts under mlp. Raises loud on
    # an architecture whose down-projection can't be found, so an unsupported model fails at load
    # rather than silently leaving a live refusal write-path.
    entries = _mlp_downprojs(getattr(layer, "block_sparse_moe", None))
    entries += _mlp_downprojs(getattr(layer, "mlp", None))
    if not entries:
        raise ValueError(
            f"could not locate a residual-writing down-projection on this decoder layer "
            f"(type {type(layer).__name__}); architecture not supported. Supported: dense, "
            "Qwen3-MoE / Granite-MoE (fused), Mixtral (fused or unfused), OLMoE (unfused), and "
            "shared-expert MoE (Qwen2-MoE / DeepSeek).")
    return entries


def _profiles_from_params(p):
    # Reconstruct (oP,owmax,owmin,oD, dP,dwmax,dwmin,dD) from a stored trial's params, for either
    # the per-component or the uniform schema, so the frontier/knee/final-bake are schema-agnostic.
    if "o_max_weight" in p:
        if "d_max_weight" in p:
            d = (p["d_max_weight_position"], max(0.0, p["d_max_weight"]), p["d_min_weight"], p["d_min_weight_distance"])
        else:  # --mlp-off: the d params were never suggested; MLP stays untouched (zero strength)
            d = (p["o_max_weight_position"], 0.0, 0.0, p["o_min_weight_distance"])
        return (p["o_max_weight_position"], p["o_max_weight"], p["o_min_weight"], p["o_min_weight_distance"], *d)
    P, wmax, wmin, D = p["max_weight_position"], p["max_weight"], p["min_weight"], p["min_weight_distance"]
    return (P, wmax, wmin, D, P, wmax, wmin, D)


def _scalar_of(t):
    # Scalarise a trial for early-stop / candidate ranking: non-compliance (hard + hedged) plus
    # half the keyword rate, but only for INTACT trials (KL under ceiling, coherent). A damaged or
    # unmeasured trial scores +inf so it can never look "best".
    ua = t.user_attrs
    if not ua or ua.get("kl", WORST_SCORE) > KL_CEIL or ua.get("broken", 1.0) > 0.1:
        return WORST_SCORE
    return ua["refusals"] + ua.get("soft", 0.0) + 0.5 * ua.get("heretic", 0.0)


def _apply_kageyoshi(args, model, arch, ne, NL, log):
    # BANKAI — "ultimate balanced-effort" preset. The user asked for the best abliteration
    # we can produce with no knob-twiddling. So: read the detected architecture + parameter
    # count, auto-scale the search budget to the model's size, and switch on every quality
    # lever we have (3-objective NSGA-II front + knee, multi-direction refusal subspace,
    # hedging-contrast direction when a hedged set is present, larger-eval knee re-score,
    # early stop). "Balanced" is load-bearing: the KL ceiling + broken penalty + the knee
    # keep it intact rather than scorched, so this is the best UNCENSORING that stays coherent,
    # not the most aggressive one. Only path/device/dataset flags are honoured; kageyoshi owns
    # the search budget itself.
    total = sum(p.numel() for p in model.parameters())
    b = total / 1e9
    # Bigger models generate slower per trial, so fewer trials + smaller evals; the snapshot
    # also holds a CPU copy of every o_proj + down_proj, which is heavy past ~20B (see the
    # snapshot_weights note), hence the trimmed direction/eval counts in the top tier.
    if b < 5:
        args.trials, args.dir_prompts, args.eval_refusal, args.eval_kl, args.eval_refusal_final = 100, 256, 64, 64, 128
    elif b < 20:
        args.trials, args.dir_prompts, args.eval_refusal, args.eval_kl, args.eval_refusal_final = 80, 256, 64, 48, 96
    else:
        args.trials, args.dir_prompts, args.eval_refusal, args.eval_kl, args.eval_refusal_final = 64, 192, 48, 32, 96
    args.search = "pareto"          # map the whole refusals/keyword/KL front, pick the balanced knee
    args.per_component = True        # tune attn.o_proj and mlp.down_proj apart (the MLP may stay untouched)
    args.mlp_off = False             # let the search decide, don't force attention-only
    args.max_directions = 3          # ablate the refusal SUBSPACE, not just the difference-of-means
    args.kl_scale = 4.0              # the coherence guard that makes it balanced
    args.top_rescore = 6
    args.patience = max(20, args.trials // 3)   # stop once the front is mapped
    # Fold the hedging axis only if a hedged-compliance set is present (the lever that closes
    # the residual keyword gap Heretic wins on); silently skip it when absent.
    if not args.hedge_ds and os.path.isdir(f"{args.track}/hedge_ds"):
        args.hedge_ds = f"{args.track}/hedge_ds"
    hedge_note = f"hedging={args.hedge_ds}" if args.hedge_ds else "hedging=none (no hedge_ds in track)"
    log("BANKAI. Senbonzakura Kageyoshi — scatter, a thousand blades.")
    log(f"  {b:.1f}B params, down-proj={arch}{'' if ne is None else f'/{ne}e'}, {NL} layers -> "
        f"{args.trials} trials, K<={args.max_directions}, eval {args.eval_refusal}/{args.eval_refusal_final}, "
        f"patience={args.patience}, {hedge_note}")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura",
        description="Multi-direction refusal abliteration for transformer language models, with a quality-guarded Optuna (NSGA-II) search over windowed, per-component, multi-directional weight ablations.",
        epilog="bankai: run `senbonzakura kageyoshi [--model ... --out ... --track ... --device ...]` "
               "for the ultimate balanced-effort abliteration. It auto-detects the architecture "
               "(dense / fused MoE / expert-list) and parameter count, scales the search budget, and "
               "turns on every quality lever, so you set only the paths. It owns the search knobs; "
               "manual --trials / --max-directions / etc. are ignored in this mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    try:   # optional shell completion; degrade gracefully if shtab is not installed
        import shtab
        shtab.add_argument_to(ap, ["--print-completion"],
                              help="print a bash/zsh/tcsh shell completion script and exit")
    except ImportError:
        pass
    ap.add_argument("--model", required=True, help="HF model id or local path to abliterate")
    ap.add_argument("--out", default="abliterated", help="directory to write the abliterated model to")
    ap.add_argument("--dir-prompts", type=int, default=256, help="contrast prompts per side for direction extraction")
    ap.add_argument("--eval-refusal", type=int, default=64, help="bad-eval prompts for the refusal score")
    ap.add_argument("--eval-kl", type=int, default=64, help="harmless prompts for the KL score")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--kl-scale", type=float, default=4.0, help="weight on KL in the objective (higher = protect quality more)")
    ap.add_argument("--layer-lo", type=float, default=0.3, help="search layers from this fraction of depth")
    ap.add_argument("--layer-hi", type=float, default=0.8)
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--gen-batch", type=int, default=16, dest="gen_batch",
                    help="max prompts per generation batch (the ceiling the adaptive VRAM throttle "
                         "ramps up to; it shrinks below this automatically when the card is busy).")
    ap.add_argument("--gpu-min-free-frac", type=float, default=0.06, dest="gpu_min_free_frac",
                    help="pause generation while USABLE VRAM (free plus senbon's own reclaimable "
                         "cache) is below this fraction of the card. Because it counts senbon's own "
                         "cache, a model that simply fills the card does not pause; only another app "
                         "(a game, a browser) taking the GPU triggers a pause, with resume on free-up.")
    ap.add_argument("--max-pause", type=float, default=None, dest="max_pause_s",
                    help="safety cap (seconds) on how long to wait for VRAM headroom before pushing on "
                         "regardless; default waits indefinitely so a busy card never crashes the run.")
    ap.add_argument("--no-throttle", action="store_true", dest="no_throttle",
                    help="disable the adaptive VRAM throttle (fixed batch, no pause/resume). Use only "
                         "when senbon has the card to itself and you want maximum, unpaced throughput.")
    ap.add_argument("--background", action="store_true", dest="background_mode",
                    help="good-gaming-citizen mode: run in the background and YIELD the GPU (pause "
                         "generation) whenever a foreground app (a game) is on the card, resuming when "
                         "it closes. Frees compute, not just VRAM, so the game stays smooth.")
    ap.add_argument("--external-pressure-mb", type=int, default=500, dest="external_pressure_mb",
                    help="in --background mode, how much VRAM a non-senbon process must hold to count "
                         "as a foreground app worth yielding to (default 500 MB).")
    ap.add_argument("--bench-only", action="store_true", help="load, extract, run 1 default-strength ablation + print refusals, no search")
    ap.add_argument("--track", default="track", help="dir holding bad_ds / good_ds / bad_eval_ds")
    ap.add_argument("--good-ds", default=None, help="override the harmless dataset dir (for a matched-form contrast)")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
    ap.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true",
                    help="allow models that ship custom modelling code (some Hub models need it); off by default.")
    ap.add_argument("--attn-impl", dest="attn_impl", default=None,
                    help="attention implementation to request (eager / sdpa / flash_attention_2); "
                         "default lets transformers choose (sdpa).")
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true",
                    help="NOT supported by the abliterator: the weight bake needs full precision. Use it "
                         "with the scorer (python -m senbonzakura.score --load-in-4bit) to measure a model "
                         "on low VRAM.")
    ap.add_argument("--inspect", nargs=2, type=float, default=None, metavar=("LAYER", "STRENGTH"),
                    help="print real harmful+harmless generations at (layer, strength), pre and post ablation, then exit")
    ap.add_argument("--inspect-n", type=int, default=8, help="prompts per side to print in --inspect")
    ap.add_argument("--max-directions", type=int, default=3,
                    help="upper bound on refusal directions per layer the search may ablate "
                         "(1 = single-direction, the original method; >1 enables multi-directional)")
    ap.add_argument("--sparsity", type=float, default=0.0,
                    help="sparse surgery: fraction of output-rows to LEAVE untouched per weight, "
                         "editing only the top-magnitude (most refusal-writing) rows. 0.0 (default) "
                         "edits every row as before; e.g. 0.3 leaves the quietest 30%% of rows pristine "
                         "for less collateral. A/B against 0.0 per model to see if coherence improves "
                         "at equal refusal removal.")
    ap.add_argument("--no-good-orth", action="store_true", dest="no_good_orth",
                    help="ablation study: do NOT orthogonalise the refusal direction against the "
                         "harmless mean (Refinement 3). Uses the raw difference-of-means instead. This "
                         "toggles off the projection grimjim calls 'projected abliteration'; on by "
                         "default. For measuring whether the projection helps or hurts the search.")
    ap.add_argument("--search", choices=["pareto", "scalar"], default="pareto",
                    help="pareto: NSGA-II maps the whole refusals-vs-KL frontier, we pick the knee "
                         "(intact + most uncensored). scalar: the old single weighted objective (TPE).")
    ap.add_argument("--per-component", dest="per_component", action="store_true", default=True,
                    help="tune attn.o_proj and mlp.down_proj SEPARATELY (Heretic-style). The MLP "
                         "profile may go to zero (leave the MLP untouched), which often preserves "
                         "intelligence. This is the default.")
    ap.add_argument("--uniform", dest="per_component", action="store_false",
                    help="apply ONE strength profile to both components (the pre-decouple behaviour).")
    ap.add_argument("--mlp-off", dest="mlp_off", action="store_true",
                    help="pin mlp.down_proj ablation to zero (attention-only). Tests the "
                         "'attention carries refusal, MLP carries capability' hypothesis and "
                         "removes the d-profile dimensions from the search entirely.")
    ap.add_argument("--hedge-ds", default=None,
                    help="dir of a HEDGED-compliance dataset (moralising-but-complying answers). "
                         "When given, a hedged-vs-clean contrast direction is folded into the "
                         "ablated basis, so the search can remove the disclaimer/hedging axis that "
                         "the difference-of-means (hard-refusal) direction misses.")
    ap.add_argument("--clean-ds", default=None,
                    help="dir of CLEAN (disclaimer-free) compliance for the hedged contrast; "
                         "defaults to --good-ds / <track>/good_ds.")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop the search early if no trial improves the best scalarised score for "
                         "this many consecutive trials (0 = run all --trials).")
    ap.add_argument("--eval-refusal-final", type=int, default=0,
                    help="re-score the top frontier candidates on this many bad-eval prompts before "
                         "picking the knee, so the choice isn't overfit to the small search eval "
                         "(0 = skip, use the search-eval numbers).")
    ap.add_argument("--top-rescore", type=int, default=6,
                    help="how many frontier candidates to re-score with --eval-refusal-final.")
    ap.add_argument("--study-db", default=None,
                    help="persist the Optuna study to this SQLite file (default: <track>/senbon-study.db, "
                         "so a killed run resumes with --resume instead of re-searching).")
    ap.add_argument("--no-persist-study", action="store_true", dest="no_persist_study",
                    help="do NOT persist the Optuna study (in-memory only). A crash then loses the "
                         "search; the persistent default is the safer choice for a long paid run.")
    ap.add_argument("--resume", action="store_true",
                    help="resume a persisted study; continues where an interrupted search left off, "
                         "and if the study already finished, skips straight to bake+save.")
    ap.add_argument("--bake-config", default=None, dest="bake_config",
                    help="skip the search entirely: load a saved best-config.json and bake+save that "
                         "config directly. Recovers a crashed save in minutes instead of re-searching.")
    ap.add_argument("--version", action="version", version=f"senbonzakura {__version__}")
    return ap


def load_model_and_tokenizer(model_id, device="cuda", load_in_4bit=False,
                             trust_remote_code=False, attn_impl=None, log=None):
    # Shared model loader for the abliterator and the scorer. Left-pads the tokenizer and sets a pad
    # token, loads in bf16 (or 4-bit via bitsandbytes when asked), and honours trust_remote_code and
    # a chosen attention implementation. Placement uses accelerate's device_map so multi-GPU and a
    # specific cuda:N both work. 4-bit is for the pure-forward paths only (scoring / measurement):
    # the weight bake rewrites tensors in place and needs full precision.
    _log = log or (lambda m: None)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    if attn_impl:
        kw["attn_implementation"] = attn_impl
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = "auto"
        _log("loading in 4-bit (nf4, double-quant); measurement path only, the bake needs full precision")
    elif device == "cuda":
        kw["device_map"] = "auto"                       # accelerate places / shards; supports big models
    elif device.startswith("cuda:"):
        kw["device_map"] = {"": int(device.split(":", 1)[1])}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if "device_map" not in kw:                          # cpu path
        model = model.to(device)
    model.eval()
    return model, tok


class Abliterator:
    """The loaded model plus everything that operates on it: direction extraction, the
    reversible norm-preserving bake (shared by the search and the final save), evaluation
    (refusals + KL), and the Optuna search. Constructing it loads the model and detects the
    architecture; `run()` does the extract -> search -> bake -> save pipeline. A model + tokenizer
    may be injected (skipping the load) so the class can be exercised against a tiny model in tests."""

    def __init__(self, args, log, model=None, tok=None):
        self.args = args
        self.log = log
        self.dev = args.device
        if model is None or tok is None:
            log(f"loading {args.model} on {self.dev}")
            model, tok = load_model_and_tokenizer(
                args.model, device=self.dev,
                load_in_4bit=getattr(args, "load_in_4bit", False),
                trust_remote_code=getattr(args, "trust_remote_code", False),
                attn_impl=getattr(args, "attn_impl", None), log=log)
        self.tok = tok
        self.model = model
        self.layers = _decoder_layers(model)             # the decoder blocks, resolved defensively
        self.H = model.config.hidden_size
        self.NL = model.config.num_hidden_layers
        # Architecture label = the distinct down-proj kinds present (e.g. "fused3d" or "fused3d+dense"
        # when a routed stack sits beside a shared expert). Raises loud here if the arch is unsupported.
        _dp = layer_downproj(self.layers[0])
        self.arch = "+".join(dict.fromkeys(k for k, _ in _dp)) or "dense"
        self.ne = getattr(model.config, "num_experts", None) or getattr(model.config, "num_local_experts", None)
        self.KMAX = max(1, args.max_directions)
        log(f"model up: hidden={self.H} layers={self.NL} down-proj={self.arch} experts={self.ne}")

        # Offload awareness: when the model is bigger than VRAM, accelerate places some layers on CPU
        # (or disk). The bake handles those transparently (see _real_tensor); we just report it so the
        # slower, throttled run is not a surprise. dmap values are ints (a cuda device) or "cpu"/"disk".
        dmap = getattr(model, "hf_device_map", None) or {}
        self.offloaded = sum(1 for v in dmap.values() if not isinstance(v, int))
        if self.offloaded:
            log(f"  low-VRAM mode: {self.offloaded}/{len(dmap)} module groups offloaded off the GPU; "
                "the adaptive throttle will pace generation to the card")

        # The resource governor paces GPU generation to live free VRAM: it shrinks the batch (or pauses
        # and resumes) when another app grabs the card, and grows back when it frees. No-op on CPU.
        self.gov = ResourceGovernor(
            self.dev, log,
            max_batch=getattr(args, "gen_batch", 16),
            min_free_frac=getattr(args, "gpu_min_free_frac", 0.08),
            max_pause_s=getattr(args, "max_pause_s", None),
            background_mode=getattr(args, "background_mode", False),
            external_pressure_mb=getattr(args, "external_pressure_mb", 500),
            enabled=not getattr(args, "no_throttle", False))

        # Search-window layer bounds + reversible-bake / current-direction state.
        self.lo = int(self.NL * args.layer_lo)
        self.hi = int(self.NL * args.layer_hi)
        self._cur = {}            # current ablation config (mode + interpolated set), read by active_dirs
        self._pristine = {}       # id(W) -> (W, cpu clone); the pristine snapshot
        self._dirty = set()       # id(W)s touched by the last bake, restored between trials
        self.dirs_multi = None    # [NL+1, KMAX, H]; filled by extract_directions
        # Eval state, populated by run(); declared up-front so the object's shape is visible and
        # methods that read them fail predictably rather than with a surprise AttributeError.
        self.bad_eval = []
        self.kl_eval = []
        self.orig_lp = None

    # ── data + prompt formatting ────────────────────────────────────────────────
    def load(self, d, n):
        # Load the first n prompts from a datasets.save_to_disk directory (column "text").
        # Fails with a readable message on a missing dir or a missing column, and warns rather
        # than silently truncating when the dataset is smaller than requested.
        try:
            ds = load_from_disk(d)
        except Exception as e:
            raise FileNotFoundError(
                f"could not load dataset at {d}: {e}. Expected a datasets.save_to_disk directory "
                f"with a 'text' column (build one with tools/build_hedge_set.py or datasets.Dataset."
                f"from_dict({{'text': [...]}}).save_to_disk('{d}')).") from e
        avail = len(ds)
        if avail < n:
            self.log(f"  note: {d} holds {avail} prompts, fewer than the {n} requested; using all {avail}")
        take = min(n, avail)
        try:
            return [ds[i]["text"] for i in range(take)]
        except KeyError as e:
            raise KeyError(
                f"dataset at {d} has no 'text' column (columns: {getattr(ds, 'column_names', '?')})") from e

    def chat(self, p):
        msgs = [{"role": "user", "content": p}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            # A tokenizer that doesn't accept enable_thinking; retry without it.
            try:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except ValueError:
                return self._fallback_prompt(p)
        except ValueError:
            # A base/foundation model that ships no chat template at all; don't die on a raw ValueError.
            return self._fallback_prompt(p)

    @staticmethod
    def _fallback_prompt(p):
        # Minimal instruction wrapper for base models with no chat template: not a real template, just
        # enough structure to elicit a completion so the run proceeds instead of crashing.
        return f"User: {p}\nAssistant:"

    # ── direction extraction: per-prompt last-token residuals, bad vs good ────────
    @torch.no_grad()
    def collect_resid(self, prompts, bs=16):
        # Per-prompt last-token residual at every layer -> [NL+1, N, H] on CPU (float32).
        # We keep the whole cloud, not just its mean, so secondary refusal directions can be
        # recovered by PCA (multi-directional ablation), not only the difference-of-means.
        chunks = []
        for i in range(0, len(prompts), bs):
            ch = [self.chat(p) for p in prompts[i:i+bs]]
            enc = self.tok(ch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            out = self.model(**enc, output_hidden_states=True, use_cache=False)
            # The tokenizer is LEFT-padded (see __init__), so the real prompt always ends at the
            # final column and the last real token is at index -1 for every row. Indexing by
            # attention_mask.sum()-1 would point into the pad region for every prompt shorter than
            # the batch max, averaging pad-token activations into the refusal subspace (the old bug).
            per = torch.stack([h[:, -1, :].float().cpu() for h in out.hidden_states], 0)  # [NL+1, b, H]
            chunks.append(per)
        return torch.cat(chunks, 1)  # [NL+1, N, H]

    def extract_directions(self, bad_dir, good_dir_path, hedge_ds, clean_src):
        # Build up to KMAX ORTHONORMAL refusal directions per layer into self.dirs_multi.
        args, NL, H, KMAX, log = self.args, self.NL, self.H, self.KMAX, self.log
        log("extracting refusal directions (primary + secondary)")
        bad = self.load(bad_dir, args.dir_prompts)
        good = self.load(good_dir_path, args.dir_prompts)
        if not bad or not good:
            raise ValueError(
                f"empty contrast set (bad={len(bad)} good={len(good)} prompts): need non-empty "
                f"datasets at {bad_dir} and {good_dir_path}. Direction extraction needs both.")
        Rb = self.collect_resid(bad)                         # [NL+1, Nb, H] cpu float32
        Rg = self.collect_resid(good)                        # [NL+1, Ng, H]
        mb = Rb.mean(1); mg = Rg.mean(1)                     # [NL+1, H]
        # Refinement 3 (Heretic `orthogonalize_direction`): keep only the component ORTHOGONAL to
        # the good direction, so ablation does not tear out good behaviour itself (a major cause of
        # our early high harmless KL). clamp_min guards a degenerate (near-zero) mean from producing
        # NaN directions that would then run the whole search on garbage.
        good_dir = mg / mg.norm(dim=-1, keepdim=True).clamp_min(1e-8)   # unit good direction, per layer

        # Optional hedging contrast (lever 2): the difference-of-means captures HARD refusal, not the
        # disclaimer/"I must warn you" hedging that the keyword metric flags on complying answers. If a
        # hedged-compliance set is supplied, extract mean(hedged) - mean(clean) at each layer and fold it
        # in as a GUARANTEED ablated direction, so the search can strip the hedging axis the hard-refusal
        # direction never sees. Build the set with tools/build_hedge_set.py.
        hedge_md = None
        if hedge_ds:
            hedged = self.load(hedge_ds, args.dir_prompts)
            cleanc = self.load(clean_src, args.dir_prompts)
            hedge_md = self.collect_resid(hedged).mean(1) - self.collect_resid(cleanc).mean(1)   # [NL+1, H]
            log(f"hedging contrast: {len(hedged)} hedged vs {len(cleanc)} clean (folded as a guaranteed direction)")

        # Refinement 5 (multi-directional): d0 is the difference-of-means (the canonical Arditi
        # direction), good-orthogonalized; d1.. are the top principal axes of the bad residual cloud
        # AFTER projecting out good_dir and the earlier directions, so the set spans the refusal
        # SUBSPACE (refusal is not always a single direction). The search picks how many (num_directions)
        # to actually ablate.
        dirs_multi = torch.zeros(NL + 1, KMAX, H)
        for li in range(NL + 1):
            gd = good_dir[li]
            if getattr(args, "no_good_orth", False):
                # Ablation study: raw difference-of-means, NOT orthogonalised to the harmless mean,
                # and the harmless direction is left out of the basis so the PCA axes are not
                # good-orthogonalised either. This is the toggle that isolates Refinement 3 (the
                # projection grimjim proposes) so its effect on the search can be measured.
                d0 = mb[li] - mg[li]
                d0 = d0 / d0.norm().clamp_min(1e-8)
                basis = [d0]; kept = [d0]
            else:
                d0 = _orth_to(mb[li] - mg[li], [gd])
                d0 = d0 / d0.norm().clamp_min(1e-8)
                basis = [gd, d0]; kept = [d0]
            # Guaranteed hedging direction (lever 2), good- and d0-orthogonalised, before PCA fills the rest.
            if hedge_md is not None and len(kept) < KMAX:
                hv = _orth_to(hedge_md[li], basis)
                n = hv.norm()
                if n > 1e-6:
                    hv = hv / n
                    kept.append(hv); basis.append(hv)
            if KMAX > len(kept):
                Xc = Rb[li] - Rb[li].mean(0, keepdim=True)   # centre the bad cloud, [N, H]
                for u in basis:
                    Xc = Xc - torch.outer(Xc @ u, u)         # project out good_dir + d0 (+ hedge)
                try:
                    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)  # rows of Vh = principal axes
                except torch.linalg.LinAlgError as e:
                    # A non-converged SVD must fail LOUD, not silently degrade to a weaker basis.
                    log(f"  layer {li}: SVD did not converge ({e}); using the primary direction only here")
                    Vh = Xc.new_zeros(0, H)
                dropped = 0
                for j in range(Vh.size(0)):
                    if len(kept) >= KMAX:
                        break
                    v = _orth_to(Vh[j], basis)
                    n = v.norm()
                    if n < 1e-6:
                        continue
                    v = v / n
                    # P1: a principal axis of the harmful cloud is kept as a refusal direction ONLY
                    # if it actually separates harmful from harmless (Cohen's d over the projections).
                    # Below the threshold it is within-harmful content/topic variance, not refusal, and
                    # ablating it would strip capability rather than refusal.
                    if _axis_separation(Rb[li], Rg[li], v) < MIN_AXIS_SEPARATION:
                        dropped += 1
                        continue
                    kept.append(v); basis.append(v)
                if dropped and li == self.lo:   # one representative log line, not NL of them
                    log(f"  layer {li}: dropped {dropped} PCA axis/axes below the refusal-separation "
                        f"threshold (d<{MIN_AXIS_SEPARATION}); they carried content, not refusal")
            for j, v in enumerate(kept):
                dirs_multi[li, j] = v
        self.dirs_multi = dirs_multi.to(torch.bfloat16)      # [NL+1, KMAX, H]; unused rows stay 0 (ablate nothing)
        log(f"directions ready: {tuple(self.dirs_multi.shape)} (<= {KMAX}/layer, orthonormal, "
            f"good-orthogonalized, refusal-separation filtered)")

    def _interp_multi(self, fidx):
        # Refinement 6 (Heretic float `direction_index`): linearly interpolate the K-direction set
        # between the two nearest residual-stack indices and re-orthonormalize. A float index reaches
        # refusal directions that are not aligned to any single layer.
        NL = self.NL
        lo_i = int(fidx); hi_i = min(lo_i + 1, NL); frac = fidx - lo_i
        M = (1 - frac) * self.dirs_multi[lo_i].float() + frac * self.dirs_multi[hi_i].float()  # [KMAX, H]
        out = torch.zeros_like(M)
        for j in range(M.size(0)):                        # Gram-Schmidt to restore orthonormality
            v = M[j].clone()
            for i in range(j):
                v = v - (v @ out[i]) * out[i]
            n = v.norm()
            out[j] = v / n if n > 1e-6 else v * 0.0
        return out.to(torch.bfloat16)                     # [KMAX, H]

    def active_dirs(self, idx, K):
        # The K unit directions to ablate at layer `idx` under the current dir_mode:
        #   per_layer -> that layer's own set (dirs_multi[idx+1])
        #   single    -> one interpolated set shared across all layers (Heretic-style)
        M = self._cur["single_set"] if self._cur.get("mode") == "single" and self._cur.get("single_set") is not None \
            else self.dirs_multi[idx + 1]
        return M[:K]   # [K, H]

    # ── reversible norm-preserving bake (used by BOTH the search and the final save) ──
    def snapshot_weights(self):
        # Snapshot the pristine residual-writing weights ONCE so the search can apply the real
        # bake, score it, then restore exactly between trials. This unifies the search and the
        # save: the search optimises the same norm-preserving surgery we ultimately keep, so there
        # is no proxy/bake gap (the old activation-hook proxy over-estimated damage, e.g. proxy
        # KL 4.5 vs baked KL 0.30 on Qwen3-1.7B). NOTE: holds one CPU copy of every o_proj +
        # down_proj; trivial for small models, heavy for the 30B, hence the RAM pre-flight below.
        self._pristine.clear(); self._dirty.clear()
        targets = []
        for layer in self.layers:
            targets.append(_attn_outproj(layer))
            for kind, obj in layer_downproj(layer):
                targets += obj if kind == "list" else [obj]
        need = sum(W.numel() * W.element_size() for W in targets)
        avail = _available_ram_bytes()
        if avail is not None and need > 0.9 * avail:
            raise MemoryError(
                f"the reversible search needs {need/1e9:.1f} GB of host RAM to hold the pristine copy "
                f"of every o_proj + down_proj, but only {avail/1e9:.1f} GB is available. Free memory, "
                f"pick a smaller model, or run on a box with more RAM.")
        if need > 8e9:
            self.log(f"  snapshot: holding {need/1e9:.1f} GB of pristine weights in host RAM")
        for W in targets:
            self._pristine[id(W)] = (W, W.detach().clone().to("cpu"))

    def _mark_dirty(self, W):
        self._dirty.add(id(W))

    @torch.no_grad()
    def restore_weights(self):
        # Copy the pristine values back into just the weights the last bake touched.
        for i in list(self._dirty):
            W, c = self._pristine[i]
            W.copy_(c.to(W.device))
        self._dirty.clear()

    @torch.no_grad()
    def bake_pc(self, oP, owmax, owmin, oD, dP, dwmax, dwmin, dD, K=1, mode="per_layer", didx=None):
        # PER-COMPONENT windowed ablation (Heretic decouples attn.o_proj from mlp.down_proj):
        # attn.o_proj follows the o-profile, mlp.down_proj the d-profile. The d-profile may be
        # all-zero (dwmax==0) to leave the MLP entirely untouched, which Heretic's issue #202 finds
        # often preserves intelligence (ablating the MLP hurts more than it helps). Windowed per
        # layer (ref 1), directions per config (ref 2/5/6), norm-preserving (ref 4). embed_tokens is
        # deliberately left alone (Heretic notes the benefit is unclear + a norm-preserving embed edit
        # is awkward).
        self._cur["mode"] = mode
        self._cur["single_set"] = self._interp_multi(didx) if (mode == "single" and didx is not None) else None
        sp = float(getattr(self.args, "sparsity", 0.0))         # sparse surgery: 0 = edit every row
        for idx, layer in enumerate(self.layers):
            wo = layer_weight(idx, oP, owmax, owmin, oD)
            wd = layer_weight(idx, dP, dwmax, dwmin, dD)
            if wo == 0.0 and wd == 0.0:
                continue
            R = self.active_dirs(idx, K).to(self.dev).float()   # [K, H]
            R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-8)  # renormalize (interp/GS drift); zero rows stay ~0
            if wo > 0.0:
                op = _attn_outproj(layer)
                self._mark_dirty(op); orthogonalize_np_(op, R, wo, sp)
            if wd > 0.0:
                for kind, obj in layer_downproj(layer):         # every residual-writing down-proj
                    if kind == "fused3d":
                        self._mark_dirty(obj); orthogonalize_np_3d_(obj, R, wd, sp)
                    elif kind == "list":
                        for W in obj:
                            self._mark_dirty(W); orthogonalize_np_(W, R, wd, sp)
                    else:  # dense
                        self._mark_dirty(obj); orthogonalize_np_(obj, R, wd, sp)

    @torch.no_grad()
    def bake(self, P, wmax, wmin, D, K=1, mode="per_layer", didx=None):
        # Uniform wrapper: the same profile on both components (the pre-decouple behaviour). Used by
        # --inspect / --bench-only and the --uniform search path.
        self.bake_pc(P, wmax, wmin, D, P, wmax, wmin, D, K, mode, didx)

    def signal_certificate(self):
        # Scan the searched layer band POST-BAKE and return the WORST (highest surviving-signal)
        # refusal-signal certificate, since a single layer that still carries the refusal signal is
        # enough for the capability to leak. KL-independent: complements "still capable" (KL) with
        # "actually un-refused". Returns (verdict, z, layer). Cheap: one residual pass + a handful of
        # label-permutation separations per layer.
        Rb = self.collect_resid(self.bad_eval)
        Rg = self.collect_resid(self.kl_eval)
        lo = max(1, int(self.NL * self.args.layer_lo))
        hi = min(self.NL, int(self.NL * self.args.layer_hi))
        worst = ("GREEN", 0.0, lo)
        for li in range(lo, hi + 1):
            v, z = _signal_certificate(Rb[li], Rg[li])
            if z == z and z > worst[1]:          # z == z skips a NaN
                worst = (v, z, li)
        return worst

    # ── eval: refusals (generation) + KL vs original (first-token) ───────────────
    @torch.no_grad()
    def gen_batch(self, prompts):
        # The governor drives the chunking: it picks the batch size from live free VRAM, shrinks on an
        # out-of-memory instead of crashing, and pauses/resumes when another app takes the card.
        def _do(chunk):
            ch = [self.chat(p) for p in chunk]
            enc = self.tok(ch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            g = self.model.generate(**enc, max_new_tokens=self.args.gen_tokens, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
            return [self.tok.decode(g[j, enc.input_ids.size(1):], skip_special_tokens=True)
                    for j in range(g.size(0))]
        return self.gov.run(_do, list(prompts))

    @torch.no_grad()
    def refusal_rate(self, prompts):
        return sum(is_refusal(t) for t in self.gen_batch(prompts)) / max(1, len(prompts))

    @torch.no_grad()
    def first_token_logprobs(self, prompts):
        # Same governor-paced chunking as gen_batch; each chunk yields a list of per-row [V] log-prob
        # tensors, which the governor concatenates in order for a final stack.
        def _do(chunk):
            ch = [self.chat(p) for p in chunk]
            enc = self.tok(ch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            lo = self.model(**enc, use_cache=False).logits[:, -1, :].float()
            return list(F.log_softmax(lo, dim=-1).cpu())   # list of [V] tensors, one per prompt
        rows = self.gov.run(_do, list(prompts))
        return torch.stack(rows, 0)  # [N, V]

    @torch.no_grad()
    def kl_vs_orig(self, prompts):
        lp = self.first_token_logprobs(prompts)
        p = self.orig_lp.exp()
        return (p * (self.orig_lp - lp)).sum(-1).mean().item()  # KL(orig || ablated)

    # ── the search ───────────────────────────────────────────────────────────────
    def _suggest_profiles(self, trial):
        # Return (oP,owmax,owmin,oD, dP,dwmax,dwmin,dD). Per-component (default) tunes attn.o_proj
        # and mlp.down_proj separately; the MLP max_weight lower bound is NEGATIVE (clamped to 0) so
        # "leave the MLP alone" is a reachable, probable point in the search (Heretic's asymmetry).
        args, lo, hi, NL = self.args, self.lo, self.hi, self.NL
        if args.per_component:
            oP = trial.suggest_int("o_max_weight_position", lo, hi)
            owmax = trial.suggest_float("o_max_weight", 0.2, 1.4)
            owmin = trial.suggest_float("o_min_weight", 0.0, 0.6)
            oD = trial.suggest_int("o_min_weight_distance", 2, max(3, NL // 3))
            if args.mlp_off:
                # Attention-only: leave mlp.down_proj untouched and don't spend search
                # dimensions on it (lever 3).
                return (oP, owmax, owmin, oD, oP, 0.0, 0.0, oD)
            dP = trial.suggest_int("d_max_weight_position", lo, hi)
            dwmax = max(0.0, trial.suggest_float("d_max_weight", -0.25, 1.4))  # <=0 -> MLP untouched
            dwmin = trial.suggest_float("d_min_weight", 0.0, 0.6)
            dD = trial.suggest_int("d_min_weight_distance", 2, max(3, NL // 3))
            return (oP, owmax, owmin, oD, dP, dwmax, dwmin, dD)
        P = trial.suggest_int("max_weight_position", lo, hi)
        wmax = trial.suggest_float("max_weight", 0.2, 1.4)
        wmin = trial.suggest_float("min_weight", 0.0, 0.6)
        D = trial.suggest_int("min_weight_distance", 2, max(3, NL // 3))
        return (P, wmax, wmin, D, P, wmax, wmin, D)

    def objective(self, trial):
        # Search the windowed profile PER COMPONENT (peak position + peak/edge strengths + window
        # width for attn.o_proj and mlp.down_proj independently), not a single (layer, strength).
        args, log = self.args, self.log
        K = trial.suggest_int("num_directions", 1, self.KMAX)     # refinement 5: how many directions
        mode = trial.suggest_categorical("dir_mode", ["per_layer", "single"])
        didx = trial.suggest_float("direction_index", self.lo, self.hi) if mode == "single" else None  # refinement 6
        pr = self._suggest_profiles(trial)
        self.restore_weights()                             # undo the previous trial's bake
        self.bake_pc(*pr, K, mode, didx)                   # apply the REAL norm-preserving bake
        gens = self.gen_batch(self.bad_eval)
        ref = sum(is_refusal(t) for t in gens) / max(1, len(gens))
        soft = sum(is_soft_refusal(t) for t in gens) / max(1, len(gens))   # hedged compliance (lever 1/2)
        hk = heretic_keyword_rate(gens)                                 # Heretic-comparable rate (the axis Heretic wins)
        brk = broken_rate(gens)                                         # coherence on the harmful set
        hbrk = broken_rate(self.gen_batch(self.kl_eval[:min(16, len(self.kl_eval))]))  # coherence on harmless
        kl = self.kl_vs_orig(self.kl_eval)
        broken = max(brk, hbrk)
        noncomp = ref + soft                                            # strict non-compliance: hard refusal + hedging
        trial.set_user_attr("refusals", ref); trial.set_user_attr("soft", soft); trial.set_user_attr("kl", kl)
        trial.set_user_attr("broken", broken); trial.set_user_attr("heretic", hk)
        # Drive non-compliance (hard AND hedged) to 0 while keeping the model intact: penalise KL past
        # the ceiling AND penalise incoherence directly, so a wrecked config (which scores LOW refusals
        # since broken != refusal) can never win. The keyword rate rides as a separate Pareto axis.
        scal = noncomp + 0.5 * hk + args.kl_scale * max(0.0, kl - KL_CEIL) + 0.3 * kl + 2.0 * broken
        log(f"  trial {trial.number}: o(P={pr[0]},wmax={pr[1]:.2f}) d(P={pr[4]},wmax={pr[5]:.2f}) K={K} {mode}"
            f"{'' if didx is None else f' di={didx:.1f}'} -> refusals={ref*100:.1f}% soft={soft*100:.1f}% "
            f"heretic={hk*100:.1f}% broken={broken*100:.0f}% KL={kl:.4f} obj={scal:.4f}")
        if args.search == "pareto":
            # THREE objectives (lever 1): strict non-compliance (hard+hedge, with a broken penalty),
            # the Heretic keyword rate (its own axis so the search actually targets what Heretic wins),
            # and KL. NSGA-II maps the frontier; we pick the knee afterwards.
            return noncomp + 2.0 * broken, hk, kl
        return scal

    # ── the full pipeline ─────────────────────────────────────────────────────────
    def run(self):
        args, log, NL = self.args, self.log, self.NL
        TR = args.track
        GOOD_DS = args.good_ds or f"{TR}/good_ds"
        clean_src = args.clean_ds or GOOD_DS
        self.extract_directions(f"{TR}/bad_ds", GOOD_DS, args.hedge_ds, clean_src)

        # eval sets: refusals (generation) on the bad-eval set, KL vs original on a harmless set that
        # is DISJOINT from the direction-extraction prompts (the first dir_prompts of good), so
        # coherence is measured on prompts the directions were not fit on. Falls back to the harmless
        # tail (with a warning) only when the dataset is too small to spare a disjoint slice.
        self.bad_eval = self.load(f"{TR}/bad_eval_ds", args.eval_refusal)
        _kl_all = self.load(GOOD_DS, args.dir_prompts + args.eval_kl)
        if len(_kl_all) > args.dir_prompts:
            self.kl_eval = _kl_all[args.dir_prompts:args.dir_prompts + args.eval_kl]
        else:
            log(f"  note: {GOOD_DS} too small for a KL set disjoint from extraction; reusing the harmless tail")
            self.kl_eval = _kl_all[-args.eval_kl:]

        log("caching original first-token distribution (KL reference) + baseline refusals")
        self.orig_lp = self.first_token_logprobs(self.kl_eval)
        base_ref = self.refusal_rate(self.bad_eval)
        log(f"BASELINE refusals: {base_ref*100:.1f}%  on {len(self.bad_eval)} bad-eval prompts")

        self.snapshot_weights()   # pristine copy taken now, on the untouched model; enables reversible search/inspect/bench

        lo, hi = self.lo, self.hi

        if args.inspect is not None:
            # Eyeball what the ablation actually does to real generations: harmful (should
            # comply after) and harmless (should stay coherent). The KL number alone hides
            # whether high KL = "wrecked" or just "a few benign first-tokens flipped".
            ilayer = int(args.inspect[0]); istr = float(args.inspect[1]); n = args.inspect_n
            iD = max(2, NL // 4)
            hprompts = self.bad_eval[:n]; gprompts = self.kl_eval[:n]
            log(f"INSPECT window P={ilayer} wmax={istr} wmin=0 D={iD}: {n} harmful + {n} harmless, pre vs post")
            pre_h = self.gen_batch(hprompts); pre_g = self.gen_batch(gprompts)
            self.bake(ilayer, istr, 0.0, iD)
            post_h = self.gen_batch(hprompts); post_g = self.gen_batch(gprompts)
            kl = self.kl_vs_orig(self.kl_eval); self.restore_weights()
            def show(tag, prompts, pre, post):
                for p, a, b in zip(prompts, pre, post):
                    print(f"\n### {tag}: {p[:110].strip()}")
                    print(f"  PRE : {a[:220].strip()!r}")
                    print(f"  POST: {b[:220].strip()!r}")
            show("HARMFUL", hprompts, pre_h, post_h)
            show("HARMLESS", gprompts, pre_g, post_g)
            pct = lambda xs, f: 100 * sum(f(t) for t in xs) / max(1, len(xs))
            log(f"INSPECT harmful:   refusals {pct(pre_h,is_refusal):.0f}%->{pct(post_h,is_refusal):.0f}%  "
                f"broken {pct(pre_h,is_broken):.0f}%->{pct(post_h,is_broken):.0f}%")
            log(f"INSPECT harmless:  refusals {pct(pre_g,is_refusal):.0f}%->{pct(post_g,is_refusal):.0f}%  "
                f"broken {pct(pre_g,is_broken):.0f}%->{pct(post_g,is_broken):.0f}%  first-token KL={kl:.4f}")
            log("INSPECT verdict: want harmful refusals DOWN with harmless broken≈0 and KL low")
            return   # --inspect is a diagnostic; nothing to search or save

        if args.bench_only:
            self.bake(int(NL*0.6), 1.0, 0.0, max(2, NL//4), K=self.KMAX); r = self.refusal_rate(self.bad_eval); k = self.kl_vs_orig(self.kl_eval); self.restore_weights()
            log(f"BENCH-ONLY default window (P={int(NL*0.6)}, wmax=1.0, K={self.KMAX}): refusals={r*100:.1f}% KL={k:.4f}")
            return   # --bench-only is a one-shot probe; nothing to search or save

        # Direct re-bake: skip the search entirely, bake a saved best-config.json and save. Recovers a
        # crashed save (or re-issues an output) in minutes instead of paying for the whole search again.
        # Directions and evals are already prepared above, so the bake reproduces the searched result.
        if args.bake_config:
            cfg = json.load(open(args.bake_config, encoding="utf-8"))
            bpr, b_K, b_mode, b_di = config_to_bake_args(cfg)
            log(f"direct bake from {args.bake_config} (skipping the search)")
            self._bake_and_save(bpr, b_K, b_mode, b_di, base_ref, TR)
            return

        log(f"searching {args.trials} trials over layers [{lo},{hi}] ({args.search})")
        # NSGA-II needs a population to exert selection pressure; Optuna's default of 50 is far larger
        # than the small trial budgets here (a 60-trial run would be barely one generation), so pin the
        # population to a fraction of the budget so evolution actually happens instead of degenerating
        # to random sampling.
        pop = max(4, min(50, args.trials // 4))
        storage = None
        study_name = f"senbon-{args.search}"
        db = study_db_path(args.study_db, args.no_persist_study, TR)
        if db:
            storage = f"sqlite:///{db}"
            log(f"persistent study at {db} ({'resuming' if args.resume else 'fresh'}); "
                f"a killed run resumes with --resume (or --no-persist-study to disable)")
        if args.search == "pareto":
            study = optuna.create_study(
                directions=["minimize", "minimize", "minimize"],
                sampler=optuna.samplers.NSGAIISampler(seed=42, population_size=pop),
                storage=storage, study_name=study_name, load_if_exists=args.resume)
        else:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=12),
                storage=storage, study_name=study_name, load_if_exists=args.resume)

        # --resume on a study that already finished its search (ran the budget or early-stopped)
        # must skip straight to bake+save, not re-search it. Without this, resume re-runs the whole
        # search on a completed study (the trap that wasted ~34 min of GPU on the first large run).
        skip_search = args.resume and search_already_done(study.user_attrs)
        if skip_search:
            log("resumed study already finished its search; skipping to bake + save")

        # Early stop (lever 4): stop once the best scalarised score (non-compliance + 0.5*keyword under
        # the KL/broken guards) hasn't improved for --patience consecutive trials, on the theory that the
        # frontier is mapped and more sampling of the same space won't help.
        _stall = {"best": WORST_SCORE, "since": 0}
        def _patience_cb(study, trial):
            if args.patience <= 0:
                return
            s = _scalar_of(trial)
            if s < _stall["best"] - 1e-4:
                _stall["best"] = s; _stall["since"] = 0
            else:
                _stall["since"] += 1
                if _stall["since"] >= args.patience:
                    log(f"early stop: no improvement in {args.patience} trials (best scalar {_stall['best']:.4f})")
                    study.stop()

        # Per-trial progress with an ETA that discounts any time the governor spent paused for VRAM,
        # so the estimate stays honest even when a game is opened mid-run.
        progress = SearchProgress(args.trials, log, governor=self.gov)
        def _progress_cb(study, trial):
            progress.tick()

        # A single flaky trial (a transient CUDA OOM on an unlucky batch, say) must not kill the whole
        # search; catch it so Optuna marks that trial failed and moves on. Whatever the outcome, the
        # last trial's bake is left applied, so restore to pristine before any downstream read/mutate.
        if not skip_search:
            try:
                study.optimize(self.objective, n_trials=args.trials, callbacks=[_patience_cb, _progress_cb],
                               catch=(RuntimeError,))
            finally:
                self.restore_weights()
            # Mark the search finished (budget spent or early-stopped) so a later --resume skips it
            # instead of re-searching. Persisted with the study, so it survives a crash after the search.
            study.set_user_attr("search_done", True)

        def _row(t):
            pr = _profiles_from_params(t.params)
            return {"oP": pr[0], "owmax": round(pr[1], 3), "oD": pr[3],
                    "dP": pr[4], "dwmax": round(pr[5], 3), "dD": pr[7],
                    "K": t.params.get("num_directions", 1), "mode": t.params.get("dir_mode", "per_layer"),
                    "di": (round(t.params["direction_index"], 2) if "direction_index" in t.params else None),
                    "refusals": round(t.user_attrs["refusals"], 4), "heretic": round(t.user_attrs.get("heretic", 0.0), 4),
                    "kl": round(t.user_attrs["kl"], 4), "broken": round(t.user_attrs.get("broken", 0.0), 4)}

        rows = sorted([_row(t) for t in study.trials if t.user_attrs], key=lambda r: (r["refusals"], r["kl"]))
        with open(f"{TR}/trials.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        log("frontier (lowest refusals first, intact = KL under ceiling AND broken≈0):")
        for r in [x for x in rows if x["kl"] <= KL_CEIL and x["broken"] <= 0.1][:8]:
            di_s = "" if r["di"] is None else f" di={r['di']}"
            log(f"   o(P={r['oP']},wmax={r['owmax']},D={r['oD']}) d(P={r['dP']},wmax={r['dwmax']},D={r['dD']}) "
                f"K={r['K']} {r['mode']}{di_s} refusals={r['refusals']*100:.1f}% heretic={r['heretic']*100:.1f}% "
                f"broken={r['broken']*100:.0f}% KL={r['kl']:.4f}")

        if args.search == "pareto":
            log(f"Pareto front ({len(study.best_trials)} non-dominated configs, refusals vs KL):")
            for t in sorted(study.best_trials, key=lambda t: t.user_attrs["refusals"])[:10]:
                log(f"   refusals={t.user_attrs['refusals']*100:.1f}% broken={t.user_attrs.get('broken',0)*100:.0f}% "
                    f"KL={t.user_attrs['kl']:.4f}")

        # Pick the KNEE: the most-uncensored config that is still intact (KL under ceiling, coherent).
        # Reads the MEASURED user_attrs, so it works for both search modes (and for multi-objective,
        # where study.best_trial is undefined). Fall back to the whole set if nothing is under ceiling.
        # Preference now weighs hedging (soft) and the Heretic keyword rate, not hard refusals alone.
        _cand = [t for t in study.trials if t.user_attrs]
        _intact = [t for t in _cand if t.user_attrs["kl"] <= KL_CEIL and t.user_attrs.get("broken", 0.0) <= 0.1]

        # Re-score the top candidates on a LARGER bad-eval before choosing, so the knee isn't overfit to
        # the small search eval (lever 5). Held in a local dict; the trials themselves aren't mutated.
        _final = {}
        _pool = _intact or _cand
        if args.eval_refusal_final and args.eval_refusal_final > len(self.bad_eval):
            big = self.load(f"{TR}/bad_eval_ds", args.eval_refusal_final)
            ranked = sorted(_pool, key=_scalar_of)[:max(1, args.top_rescore)]
            log(f"re-scoring top {len(ranked)} candidates on {len(big)} bad-eval prompts (lever 5)")
            for t in ranked:
                pr = _profiles_from_params(t.params)
                K = t.params.get("num_directions", 1); mode = t.params.get("dir_mode", "per_layer")
                di = t.params.get("direction_index")
                self.restore_weights(); self.bake_pc(*pr, K, mode, di)
                g = self.gen_batch(big)
                r = sum(is_refusal(x) for x in g) / max(1, len(g))
                s = sum(is_soft_refusal(x) for x in g) / max(1, len(g))
                h = heretic_keyword_rate(g)
                _final[t.number] = {"refusals": r, "soft": s, "heretic": h}
                log(f"   trial {t.number}: refusals={r*100:.1f}% soft={s*100:.1f}% heretic={h*100:.1f}% KL={t.user_attrs['kl']:.4f}")
            self.restore_weights()
            _pool = ranked

        def _knee_key(t):
            # P2 fix: minimise a WEIGHTED scalar of the three objectives so the Heretic keyword /
            # hedging axis actually STEERS the pick, instead of the old lexicographic tuple where it
            # was a mere tiebreaker behind hard+soft refusal and so almost never decided the winner
            # (the whole point of making it a first-class search objective was defeated at selection
            # time). KL only surcharges above the "comfortably intact" target.
            f = _final.get(t.number)
            ua = t.user_attrs
            ref = f["refusals"] if f else ua["refusals"]
            soft = f["soft"] if f else ua.get("soft", 0.0)
            her = f["heretic"] if f else ua.get("heretic", 0.0)
            return knee_scalar(ref, soft, her, ua["kl"])

        best = min(_pool, key=_knee_key)
        bp = best.params
        b_K = bp.get("num_directions", 1); b_mode = bp.get("dir_mode", "per_layer")
        b_di = bp.get("direction_index")
        bpr = _profiles_from_params(bp)
        log(f"BEST: o(P={bpr[0]},wmax={bpr[1]:.3f},wmin={bpr[2]:.3f},D={bpr[3]}) "
            f"d(P={bpr[4]},wmax={bpr[5]:.3f},wmin={bpr[6]:.3f},D={bpr[7]}) K={b_K} mode={b_mode}"
            f"{'' if b_di is None else f' di={b_di:.2f}'} refusals={best.user_attrs['refusals']*100:.1f}% "
            f"heretic={best.user_attrs.get('heretic',0)*100:.1f}% "
            f"broken={best.user_attrs.get('broken',0)*100:.0f}% KL={best.user_attrs['kl']:.4f}")

        self._bake_and_save(bpr, b_K, b_mode, b_di, base_ref, TR)

    def _bake_and_save(self, bpr, b_K, b_mode, b_di, base_ref, track):
        """Bake the winning config into the weights and save. Extracted from run() so a
        --bake-config recovery can save a known config WITHOUT re-searching. Writes
        best-config.json BEFORE the (crash-prone) save, so even a failed save leaves a
        re-bakeable artefact and a lost save becomes a minutes-long re-bake, not a re-search."""
        args, log = self.args, self.log
        with open(f"{track}/best-config.json", "w", encoding="utf-8") as f:
            json.dump(winning_config(bpr, b_K, b_mode, b_di), f, indent=2)
        log(f"wrote winning config to {track}/best-config.json (re-bakeable with --bake-config)")

        # ── BAKE the winner into the weights + save ──────────────────────────────────
        # The search left the LAST trial's bake applied; restore to pristine, then bake the winner.
        # Because the search scored this exact operation, POST-BAKE should reproduce the best trial's
        # numbers (that equality is the check that search and bake are unified, no proxy gap).
        self.restore_weights()
        log("baking best config (per-component windowed, per-config directions, norm-preserving)")
        self.bake_pc(*bpr, b_K, b_mode, b_di)
        # POST-BAKE re-measure on the actual weights; should match the best trial (same operation).
        post_gens = self.gen_batch(self.bad_eval)
        post_ref = sum(is_refusal(t) for t in post_gens) / max(1, len(post_gens))
        post_heretic = heretic_keyword_rate(post_gens)
        post_kl = self.kl_vs_orig(self.kl_eval)
        post_brk = broken_rate(self.gen_batch(self.kl_eval[:min(16, len(self.kl_eval))]))
        log(f"POST-BAKE (weights, no hooks): refusals={post_ref*100:.1f}% heretic={post_heretic*100:.1f}% "
            f"broken={post_brk*100:.0f}% KL={post_kl:.4f}")
        # KL-independent verification: is the refusal SIGNAL actually gone (not just the keyword rate)?
        cert, cert_z, cert_layer = self.signal_certificate()
        log(f"REFUSAL-SIGNAL CERT: {cert} (z={cert_z:.1f} above chance at layer {cert_layer}); "
            f"GREEN=refusal linearly gone, YELLOW=weak/distributed (extended-refusal risk), RED=survives")

        log(f"saving to {args.out}")
        self.model.save_pretrained(args.out, safe_serialization=True)
        self.tok.save_pretrained(args.out)
        with open(f"{args.out}/abliteration.json", "w", encoding="utf-8") as f:
            json.dump({"per_component": args.per_component,
                       "o_profile": {"max_weight_position": bpr[0], "max_weight": bpr[1], "min_weight": bpr[2], "min_weight_distance": bpr[3]},
                       "d_profile": {"max_weight_position": bpr[4], "max_weight": bpr[5], "min_weight": bpr[6], "min_weight_distance": bpr[7]},
                       "num_directions": b_K, "dir_mode": b_mode, "direction_index": b_di,
                       "max_directions": self.KMAX,
                       "baseline_refusals": base_ref, "post_bake_refusals": post_ref,
                       "post_bake_heretic": post_heretic, "post_bake_broken": post_brk, "post_bake_kl": post_kl,
                       "refusal_signal_cert": cert, "refusal_signal_cert_z": round(cert_z, 3), "refusal_signal_cert_layer": cert_layer,
                       "sparsity": float(getattr(args, "sparsity", 0.0))},
                      f, indent=2)
        log("DONE")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Easter egg / bankai: `senbonzakura kageyoshi ...` strips the release word and runs the
    # auto-scaled best-effort preset (resolved after the model loads, once the architecture and
    # size are known). Everything after it still parses, so paths/device flags work as usual.
    bankai = bool(argv) and argv[0] == "kageyoshi"
    if bankai:
        argv = argv[1:]
    args = build_parser().parse_args(argv)

    if args.load_in_4bit:
        # The abliterator rewrites weights in place (the norm-preserving bake), which needs full
        # precision; 4-bit Params4bit can't be orthogonalised. Reject early with a clear pointer
        # rather than failing cryptically at bake time.
        raise SystemExit(
            "senbonzakura abliterates by rewriting weights, which needs full precision, so "
            "--load-in-4bit is not supported here. Use it with the scorer to measure a model on "
            "low VRAM: python -m senbonzakura.score --load-in-4bit --model <dir> --eval <ds> --out r.json")

    # Fail loud on an incompatible torch BEFORE the model download (a fused-MoE class imports
    # torch.distributed.tensor.DTensor, torch >= 2.5); otherwise the run dies only after pulling
    # tens of GB. The runpod pytorch 2.4 image tripped exactly this.
    if not torch_version_ok(torch.__version__):
        raise SystemExit(
            f"senbonzakura needs torch >= {MIN_TORCH[0]}.{MIN_TORCH[1]} (the transformers MoE path "
            f"imports torch.distributed.tensor.DTensor); found {torch.__version__}. "
            f"Install a compatible build, e.g. torch==2.5.1.")

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    abl = Abliterator(args, log)
    if bankai:
        _apply_kageyoshi(args, abl.model, abl.arch, abl.ne, abl.NL, log)
    abl.run()


if __name__ == "__main__":
    main()
