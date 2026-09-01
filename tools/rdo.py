#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
r"""Refusal Direction Optimisation: find directions that ABLATE refusal, not directions that
describe how harmful prompts differ.

Every direction this project has ever used is descriptive. The difference-of-means says where the
harmful cloud sits relative to the harmless one; the cluster means say the same thing per cluster.
Neither is chosen for what happens when you remove it. That is the gap the 2026-08-03 experiments
measured: at matched refusal removal, extra cluster directions cost monotonically more KL and
bought nothing, and random directions beat them.

Wollschlaeger et al. (arXiv:2502.17420) optimise instead. Each direction is a free parameter, and
the loss is what the model DOES once that direction is projected out of its residual stream: drive
refusal down on harmful prompts, hold behaviour still on harmless ones. For more than one
direction they add a representational-independence term, so the second direction is not a
rediscovery of the first wearing a different basis.

The mechanical difference from everything here so far: the ablation is applied as a forward HOOK
on the residual stream, which is differentiable, so gradients reach the direction. Nothing is
baked into a weight during the search, and the model's own parameters never move.

  loss = refusal        per-token log-probability of a refusal opener on harmful prompts, minimised
       + induce   * I   refusal the same directions ADD back when they are applied to a harmless
                        prompt, maximised: a direction that only breaks the model cannot do this
       + preserve * KL  first-token divergence from the untouched model on harmless prompts
       + indep    * L   cosine-similarity coupling between the directions being learned

The induction term is the paper's second refusal property and this tool ran without it until
2026-08-04. Without it the objective only ever asks a direction to REMOVE refusal when it is
subtracted, and a direction that removes refusal by damaging the model scores perfectly. It is the
term that makes "this is the refusal direction" different from "this direction wrecks the model".

Usage:
  python tools/rdo.py --model <hf-id> --track <dir> --k 4 --out dirs.pt \
      [--steps 150] [--layer-frac auto] [--preserve 1.0] [--induce 0.2] [--indep 0.5]

The output is a [NL+1, K, H] tensor in the same layout as `Abliterator.dirs_multi`, so the
existing evaluation harness can score it against the cluster directions and against single
without any of them getting a different measuring stick.
"""
import argparse
import contextlib
import json
import sys

import torch
import torch.nn.functional as F

from senbonzakura import cli


# ── the differentiable ablation ───────────────────────────────────────────────────────
@contextlib.contextmanager
def ablation_hooks(layers, dirs, start=0):
    """Project `dirs` out of the residual stream at every layer from `start` onward.

    A hook rather than a weight edit, because a weight edit is not differentiable with respect
    to the direction and the whole point here is to take its gradient. The projection is the same
    operation the bake performs, so a direction optimised under this hook is optimised for what
    the bake will actually do to the model.

    `dirs` is [K, H] and must be orthonormal: the projection onto a span is R^T R only for an
    orthonormal basis, and correlated rows over-subtract along whatever they share.
    """
    handles = []

    def make(idx):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Compute in float32 and cast back. Real models run in bf16 while the directions are a
            # float32 parameter, so multiplying the two directly raises; casting the DIRECTION down
            # instead would round the thing being optimised to bf16 at every step. The fixtures are
            # float32, so this only ever shows up on a real model.
            hf = h.float()
            df = dirs.float()
            hf = hf - (hf @ df.T) @ df            # remove the component lying in span(dirs)
            hf = hf.to(h.dtype)
            return (hf, *out[1:]) if isinstance(out, tuple) else hf
        return hook

    try:
        for i, layer in enumerate(layers):
            if i >= start:
                handles.append(layer.register_forward_hook(make(i)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def addition_hook(layers, vec, alpha, layer_idx):
    """Add `alpha * vec` to the residual stream at ONE layer.

    The counterpart to `ablation_hooks`, and deliberately a single layer rather than a window: the
    claim being tested is that this direction CARRIES refusal, and a direction that only induces
    refusal once it has been added at twenty layers at once has not shown that.
    """
    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hf = h.float() + alpha * vec.float()
        hf = hf.to(h.dtype)
        return (hf, *out[1:]) if isinstance(out, tuple) else hf

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def orthonormalise(raw):
    """Gram-Schmidt the rows of `raw` into an orthonormal [K, H].

    Done every step rather than penalised, because the bake's projection is only correct for an
    orthonormal basis and a direction set that drifts off the constraint would be scored under one
    geometry and applied under another.
    """
    q, _ = torch.linalg.qr(raw.T)
    return q.T[: raw.shape[0]]


# ── the losses ────────────────────────────────────────────────────────────────────────
def opener_logprob(model, enc, openers, ctx, per_token=True):
    r"""How much probability the model puts on actually SAYING a refusal opener, under `ctx`.

    Scored over whole openers under teacher forcing, not over their first tokens. Scoring first
    tokens looked equivalent and is not: "I cannot", "I cant" and "I am unable" all begin with
    "I", so nine openers collapsed to four token ids and the loss became mostly "do not begin with
    the word I". That is an ordinary English word, so the optimiser would have been pushed to
    damage plain language rather than to remove refusal, with the preservation term fighting it
    the whole way.

    `openers` is a list of [T] id tensors. Each contributes its MEAN PER-TOKEN log-probability;
    logsumexp across them reads as "how likely is ANY of these refusals".

    Per-token rather than summed, and the difference is not cosmetic. Summed, this term ran from
    -22 to -700 over a real optimisation while the preservation KL sat between 0.03 and 3 at
    weight 1.0, so preservation contributed almost nothing to the gradient and the optimiser was
    free to buy refusal removal with unlimited damage. Every RDO direction produced before
    2026-08-04 was learned under that imbalance, which fits their KL being an order of magnitude
    worse than a plain difference-of-means. The paper's term is a cross-entropy, which is a mean
    over tokens, so this is the faithful reading as well as the balanced one.

    `per_token=False` restores the summed form. It exists so an attribution run can hold the other
    two 2026-08-04 fixes fixed and vary this one, rather than reporting that "the fixes helped"
    without knowing which fix did the work.
    """
    reduce = (lambda x: x.mean(-1)) if per_token else (lambda x: x.sum(-1))
    scores = []
    ids_in = enc["input_ids"]
    for opener in openers:
        cont = opener.to(ids_in.device).unsqueeze(0).expand(ids_in.shape[0], -1)
        kw = {"input_ids": torch.cat([ids_in, cont], dim=1), "use_cache": False}
        if "attention_mask" in enc:
            kw["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(cont)], dim=1)
        with ctx():
            out = model(**kw)
        logp = F.log_softmax(out.logits.float(), dim=-1)
        # Position i predicts token i+1, so the opener's first token is predicted at the last
        # prompt position.
        t = cont.shape[1]
        picked = logp[:, -t - 1:-1, :].gather(-1, cont.unsqueeze(-1)).squeeze(-1)   # [b, t]
        scores.append(reduce(picked))                                              # [b]
    return torch.logsumexp(torch.stack(scores, dim=-1), dim=-1).mean()


def refusal_loss(model, enc, dirs, layers, start, openers, per_token=True):
    """Refusal still available on HARMFUL prompts once the directions are projected out.

    Minimised: the whole point of ablating is that this goes down.
    """
    return opener_logprob(model, enc, openers, lambda: ablation_hooks(layers, dirs, start),
                          per_token=per_token)


def induction_loss(model, enc, direction, layers, layer_idx, alpha, openers, per_token=True):
    """Refusal the direction ADDS to a HARMLESS prompt when it is added rather than removed.

    Returned negated, so minimising the total loss maximises induced refusal. This is
    Wollschlaeger's second refusal property and the term this tool was missing until 2026-08-04:
    without it, "removing this direction stops the model refusing" is satisfied just as well by a
    direction that stops the model doing anything. Adding it back is the check that the direction
    is about refusal specifically, and no amount of preservation weight substitutes for it,
    because preservation only says the undamaged model stayed undamaged.
    """
    return -opener_logprob(model, enc, openers,
                           lambda: addition_hook(layers, direction, alpha, layer_idx),
                           per_token=per_token)


def preserve_loss(model, enc, dirs, layers, start, base_logp):
    """First-token KL from the untouched model on harmless prompts.

    Without this the optimiser is free to find a direction that removes refusal by destroying the
    model, which is the failure the harmless arm exists to catch and which a refusal rate alone
    cannot distinguish from success.
    """
    with ablation_hooks(layers, dirs, start):
        out = model(**enc, use_cache=False)
    logp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    return F.kl_div(logp, base_logp, log_target=True, reduction="batchmean")


def independence_loss(dirs):
    """Penalise the directions for pointing at the same thing.

    Orthogonality alone does not give independence: two orthogonal directions can still be
    rediscoveries of one mechanism seen from different angles. This is the cheap surrogate for
    Wollschlaeger's representational-independence term: push the off-diagonal Gram entries of the
    RAW (pre-orthonormalisation) parameters towards zero, so the constraint is not doing all the
    work and hiding a collapse.
    """
    n = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    gram = n @ n.T
    off = gram - torch.diag(torch.diag(gram))
    return (off ** 2).sum() / max(1, off.numel() - off.shape[0])


# ── the optimiser ─────────────────────────────────────────────────────────────────────
def bake_window_start(NL):
    """The lowest layer the evaluation's bake actually touches.

    `_measure` bakes at P = int(NL * 0.6) with half-width D = max(2, NL // 4), so the window runs
    from P - D. Derived rather than written down: RDO optimised from layer 14 while the bake
    reached down to layer 9, so five layers inside the window held no directions at all and
    gemma's RDO arms could not remove enough refusal to reach any comparison band. Two
    hand-tuned constants that must agree is how that happens.
    """
    return max(0, int(NL * 0.6) - max(2, NL // 4))


def resolve_layer_frac(spec, NL):
    """`auto` ties the optimisation window to the bake window; a float pins it by hand."""
    if str(spec).strip().lower() == "auto":
        return bake_window_start(NL) / max(1, NL)
    return float(spec)


def bake_centre_layer(NL):
    """The layer the evaluation's bake hits hardest, and so the one worth anchoring to.

    `_measure` centres its profile at int(NL * 0.6); the strength falls away from there. A single
    layer has to be picked for both the warm start and the induction term, and this is the one
    where the intervention is strongest and therefore where a direction's claim to carry refusal
    is most testable.
    """
    return min(max(int(NL * 0.6), 0), NL)


def mean_diff_init(a, harmful, harmless, k, g, log):
    """Warm-start row 0 from the good-orthogonalised difference-of-means; fill the rest randomly.

    Gram-Schmidt preserves the first row up to sign, and a projection is sign-invariant, so after
    orthonormalisation the K-direction set provably CONTAINS the difference-of-means at step zero.
    That makes RDO unable to start worse than the baseline it is being compared against, which is
    what a random init gives up: every RDO result this project has produced started from noise at
    200 steps, and "optimised directions lose to the simplest possible direction" is the shape of
    an optimisation that has not converged rather than of a method that does not work.

    Taken at the bake's centre layer because `to_dirs_multi` writes one direction set to every
    layer in the window, so exactly one layer's geometry can be honoured and this is the one that
    matters most. The good-orthogonalisation matches `extract_directions`, so the warm start is
    the baseline direction itself rather than something near it.
    """
    li = bake_centre_layer(a.NL)
    mb = a.collect_resid(harmful)[li].mean(0)                       # [H]
    mg = a.collect_resid(harmless)[li].mean(0)
    gd = mg / mg.norm().clamp_min(1e-8)
    d0 = mb - mg
    d0 = d0 - (d0 @ gd) * gd
    n = d0.norm()
    if n < 1e-6:
        # Identical class means at this layer. Layer 0 does exactly this (every prompt's last
        # token is the same chat-template token), and a silent fallback would look like a warm
        # start that simply did not help.
        log(f"  WARNING: the class means at layer {li} are identical (norm {float(n):.2e}), so "
            f"there is no difference-of-means to warm-start from. Falling back to a random init; "
            f"this run is NOT warm-started and must not be compared with one that is.")
        return torch.randn(k, a.H, generator=g), False
    raw = torch.randn(k, a.H, generator=g)
    raw[0] = (d0 / n).float().cpu()
    log(f"  warm start: row 0 is the difference-of-means at layer {li} (norm {float(n):.3f})")
    return raw, True


def optimise(a, k, steps, layer_frac, preserve_w, indep_w, lr, batch, log, seed=0,
             induce_w=0.2, induce_alpha=1.0, init="mean-diff", per_token=True):
    """Learn K orthonormal directions that ablate refusal while holding harmless behaviour."""
    args = a.args
    model, tok = a.model, a.tok
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    harmful = a.load(f"{args.track}/bad_ds", args.dir_prompts)
    harmless = a.load(args.good_ds or f"{args.track}/good_ds", args.dir_prompts)
    start = int(a.NL * resolve_layer_frac(layer_frac, a.NL))
    window_lo = bake_window_start(a.NL)
    if start > window_lo:
        log(f"  NOTE: optimising from layer {start} while the evaluation bake reaches down to "
            f"layer {window_lo}. Layers {window_lo} to {start - 1} will be inside the bake window "
            f"with no directions in them, so this direction set cannot remove as much refusal as "
            f"one that covers the window. Pass --layer-frac auto to tie the two together.")

    # The refusal vocabulary: the first tokens of the openers the scorer already counts. Reusing
    # them means the optimiser is pushing on the same quantity the evaluation reads, rather than
    # on a proxy that might move independently of it.
    opener_text = ["I cannot help with that", "I can't assist with that", "I'm sorry, but I can't",
                   "I am unable to provide", "As an AI, I cannot"]
    # Distinct SEQUENCES. Two openers that tokenise identically would be counted twice and the
    # logsumexp would read as though there were more refusal mass than there is.
    seen, openers = set(), []
    for text in opener_text:
        ids = tuple(tok(text, add_special_tokens=False)["input_ids"])
        if ids and ids not in seen:
            seen.add(ids)
            openers.append(torch.tensor(ids))
    log(f"refusal openers: {len(openers)} distinct sequences, lengths {[len(o) for o in openers]}")

    g = torch.Generator(device="cpu").manual_seed(seed)
    if init == "mean-diff":
        raw0, warm_started = mean_diff_init(a, harmful, harmless, k, g, log)
    else:
        raw0, warm_started = torch.randn(k, a.H, generator=g), False
    raw = raw0.to(a.dev, torch.float32)
    raw = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    raw.requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    induce_layer = bake_centre_layer(a.NL) - 1   # a hook on block i edits the stream entering i+1
    induce_layer = min(max(induce_layer, 0), len(a.layers) - 1)

    def encode(prompts):
        return tok([a.chat(p) for p in prompts], return_tensors="pt", padding=True,
                   add_special_tokens=False).to(a.dev)

    # The untouched harmless distribution, measured once. It is the reference the preservation
    # term is measured against, so it must come from the model with no hooks attached.
    with torch.no_grad():
        base_cache = {}
        for i in range(0, min(len(harmless), 4 * batch), batch):
            enc = encode(harmless[i:i + batch])
            base_cache[i] = F.log_softmax(
                model(**enc, use_cache=False).logits[:, -1, :].float(), dim=-1).detach()

    history = []
    for step in range(steps):
        i = (step * batch) % max(1, len(harmful) - batch)
        j = (step * batch) % max(1, min(len(harmless), 4 * batch) - batch)
        j -= j % batch

        dirs = orthonormalise(raw)
        enc_good = encode(harmless[j:j + batch])
        l_ref = refusal_loss(model, encode(harmful[i:i + batch]), dirs, a.layers,
                             start, openers, per_token=per_token)
        l_pre = preserve_loss(model, enc_good, dirs, a.layers, start, base_cache[j])
        l_ind = independence_loss(raw) if k > 1 else torch.zeros((), device=a.dev)
        # One direction per step, round-robin, rather than all K. Scoring every direction's
        # induction each step costs K extra forward passes for what is a mean over directions
        # anyway; cycling gives an unbiased stochastic estimate of that mean at the price of one.
        # Over 200 steps with K=4 each direction is exercised about 50 times.
        l_add = (induction_loss(model, enc_good, dirs[step % k], a.layers, induce_layer,
                                induce_alpha, openers, per_token=per_token)
                 if induce_w else torch.zeros((), device=a.dev))
        loss = l_ref + induce_w * l_add + preserve_w * l_pre + indep_w * l_ind

        if step == 0 and l_ref.grad_fn is None:
            # Fail loudly rather than optimise nothing. The hooks attach to decoder blocks, so a
            # model whose forward reaches inside its blocks instead of calling them never runs
            # them, the refusal term detaches, and the run completes with the directions sitting
            # exactly where they were initialised. Nothing in the output would say so: the
            # independence term still falls and still prints.
            raise RuntimeError(
                "the refusal loss is not connected to the directions, so this optimisation would "
                "move nothing. The ablation is applied by forward hooks on the decoder blocks, so "
                "this happens when the model's forward does not call those blocks as modules. "
                "Check that _decoder_layers() returns the modules this architecture actually "
                "invokes.")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            row = {"step": step, "refusal": round(float(l_ref.detach()), 4),
                   "induced": round(float(-l_add.detach()), 4),
                   "preserve_kl": round(float(l_pre.detach()), 4),
                   "independence": round(float(l_ind.detach()), 5)}
            history.append(row)
            log(f"  step {step:>4}  refusal {row['refusal']:>8}  induced {row['induced']:>8}  "
                f"kl {row['preserve_kl']:>7}  indep {row['independence']}")

    return orthonormalise(raw).detach(), history, start, warm_started


def to_dirs_multi(dirs, NL, H, start):
    """Lay the learned directions out as [NL+1, K, H], the shape the rest of the tool expects.

    The same set at every layer in the window and zeros below it. Zeros ablate nothing, which is
    how a layer outside the window says "not here" without a special case downstream.
    """
    k = dirs.shape[0]
    out = torch.zeros(NL + 1, k, H)
    for li in range(start, NL + 1):
        out[li] = dirs.detach().float().cpu()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", default="track")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--layer-frac", default="auto",
                    help="where to start optimising, as a fraction of depth. 'auto' (default) "
                         "starts at the lowest layer the evaluation's bake touches, so the "
                         "directions cover the window they will be applied in.")
    ap.add_argument("--preserve", type=float, default=1.0)
    ap.add_argument("--induce", type=float, default=0.2,
                    help="weight on the term that requires the directions to INDUCE refusal when "
                         "added to a harmless prompt. 0 disables it, reproducing the behaviour "
                         "before 2026-08-04, which could not tell a refusal direction from a "
                         "direction that merely breaks the model. The paper's value is 0.2.")
    ap.add_argument("--induce-alpha", type=float, default=1.0,
                    help="how much of the direction to add when testing induced refusal")
    ap.add_argument("--init", choices=("mean-diff", "random"), default="mean-diff",
                    help="'mean-diff' (default) warm-starts the first direction from the "
                         "difference-of-means, so the set cannot start worse than the baseline it "
                         "is compared against. 'random' reproduces the paper's initialisation.")
    ap.add_argument("--score", choices=("per-token", "summed"), default="per-token",
                    help="how a refusal opener is scored. 'per-token' (default) is the "
                         "paper's cross-entropy. 'summed' reproduces the behaviour before "
                         "2026-08-04, where this term reached -700 against a preservation KL "
                         "near 1 and preservation stopped mattering; it exists for "
                         "attribution runs, not for producing directions.")
    ap.add_argument("--indep", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dir-prompts", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    own = ap.parse_args(argv)

    args = cli.build_parser().parse_args([
        "--model", own.model, "--track", own.track, "--device", own.device,
        "--dir-prompts", str(own.dir_prompts), "--seed", str(own.seed)])

    log = lambda m: print(m, flush=True)   # noqa: E731
    a = cli.Abliterator(args, log)
    dirs, history, start, warm_started = optimise(
        a, own.k, own.steps, own.layer_frac, own.preserve, own.indep, own.lr, own.batch, log,
        seed=own.seed, induce_w=own.induce, induce_alpha=own.induce_alpha, init=own.init,
        per_token=(own.score == "per-token"))

    # Everything that changes what the directions MEAN, recorded with them. Two runs whose
    # metadata differ here are two different experiments, and a resume guard or a cross-seed table
    # that treats them as one is the defect that wasted 2026-08-04: a run optimised from layer 15
    # sat in a table with four optimised from layer 9 and nothing in the file said so.
    meta = {"k": own.k, "start_layer": start, "model": own.model,
            "bake_window_start": bake_window_start(a.NL), "layers": a.NL,
            "steps": own.steps, "lr": own.lr, "batch": own.batch,
            "preserve": own.preserve, "induce": own.induce, "induce_alpha": own.induce_alpha,
            "indep": own.indep, "init": own.init, "warm_started": warm_started,
            "score": own.score,
            "dir_prompts": own.dir_prompts, "layer_frac": str(own.layer_frac),
            "history": history, "seed": own.seed}
    torch.save({"dirs_multi": to_dirs_multi(dirs, a.NL, a.H, start), **meta}, own.out)
    log(f"written to {own.out}")
    with cli.atomic_write(own.out + ".json") as f:
        json.dump(meta, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
