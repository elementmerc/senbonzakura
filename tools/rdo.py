#!/usr/bin/env python3
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

  loss = refusal        mean log-probability of a refusal opener on harmful prompts, minimised
       + preserve * KL  first-token divergence from the untouched model on harmless prompts
       + indep    * L   cosine-similarity coupling between the directions being learned

Usage:
  python tools/rdo.py --model <hf-id> --track <dir> --k 4 --out dirs.pt \
      [--steps 150] [--layer-frac 0.6] [--preserve 1.0] [--indep 0.5]

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
            # h: [batch, seq, H]. Remove the component of h lying in span(dirs).
            coef = h @ dirs.T                     # [b, s, K]
            h = h - coef @ dirs                   # [b, s, H]
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    try:
        for i, layer in enumerate(layers):
            if i >= start:
                handles.append(layer.register_forward_hook(make(i)))
        yield
    finally:
        for handle in handles:
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
def refusal_loss(model, enc, dirs, layers, start, refusal_ids):
    """How much probability mass the model still puts on refusing, with `dirs` ablated."""
    with ablation_hooks(layers, dirs, start):
        out = model(**enc, use_cache=False)
    logp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    # logsumexp over the refusal vocabulary: "how likely is ANY refusal opener", which is what
    # the metric downstream actually counts, rather than one hand-picked token.
    return torch.logsumexp(logp[:, refusal_ids], dim=-1).mean()


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
def optimise(a, k, steps, layer_frac, preserve_w, indep_w, lr, batch, log, seed=0):
    """Learn K orthonormal directions that ablate refusal while holding harmless behaviour."""
    args = a.args
    model, tok = a.model, a.tok
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    harmful = a.load(f"{args.track}/bad_ds", args.dir_prompts)
    harmless = a.load(args.good_ds or f"{args.track}/good_ds", args.dir_prompts)
    start = int(a.NL * layer_frac)

    # The refusal vocabulary: the first tokens of the openers the scorer already counts. Reusing
    # them means the optimiser is pushing on the same quantity the evaluation reads, rather than
    # on a proxy that might move independently of it.
    openers = ["I cannot", "I can't", "I'm sorry", "Sorry", "I am unable", "I apologise",
               "As an AI", "I must decline", "Unfortunately"]
    refusal_ids = sorted({tok(o, add_special_tokens=False)["input_ids"][0] for o in openers})
    log(f"refusal vocabulary: {len(refusal_ids)} first-token ids from {len(openers)} openers")

    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(k, a.H, generator=g).to(a.dev, torch.float32)
    raw = raw / raw.norm(dim=-1, keepdim=True)
    raw.requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)

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
        l_ref = refusal_loss(model, encode(harmful[i:i + batch]), dirs, a.layers, start, refusal_ids)
        l_pre = preserve_loss(model, encode(harmless[j:j + batch]), dirs, a.layers, start,
                              base_cache[j])
        l_ind = independence_loss(raw) if k > 1 else torch.zeros((), device=a.dev)
        loss = l_ref + preserve_w * l_pre + indep_w * l_ind

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
                   "preserve_kl": round(float(l_pre.detach()), 4),
                   "independence": round(float(l_ind.detach()), 5)}
            history.append(row)
            log(f"  step {step:>4}  refusal {row['refusal']:>8}  kl {row['preserve_kl']:>7}  "
                f"indep {row['independence']}")

    return orthonormalise(raw).detach(), history, start


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
    ap.add_argument("--layer-frac", type=float, default=0.6)
    ap.add_argument("--preserve", type=float, default=1.0)
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
    dirs, history, start = optimise(a, own.k, own.steps, own.layer_frac, own.preserve,
                                    own.indep, own.lr, own.batch, log, seed=own.seed)

    torch.save({"dirs_multi": to_dirs_multi(dirs, a.NL, a.H, start),
                "k": own.k, "start_layer": start, "model": own.model,
                "history": history, "seed": own.seed}, own.out)
    log(f"written to {own.out}")
    with cli.atomic_write(own.out + ".json") as f:
        json.dump({"k": own.k, "start_layer": start, "model": own.model,
                   "history": history, "seed": own.seed}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
