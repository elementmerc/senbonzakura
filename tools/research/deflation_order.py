#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""MEASUREMENT HARNESS: how far apart do the two deflation orders land, on the same weights?

It measures and does not decide. Nothing in the tool calls it, and it names no order canonical.

THE OPEN QUESTION. `cli.orthogonalize_np_` removes the refusal directions one at a time, each
projection recomputed against the partly updated weight. `cli._orthogonalize_np_3d_block_` removes
all of them at once from one batched projection. For an orthonormal basis those are the same
operation, which is why they agree with `--sparsity 0` or `--ablation-rounds 0` or one direction.
A masked subtraction is not a projection, so with the mask and the rounds both on, the order
matters. `tests/test_the_expert_path_on_both_layouts.py` pins that this divergence exists. This
reports HOW BIG it is, across a grid, so a decision can be made on a number.

The operator wants it measured across architectures before an order is named canonical, and
suspects the choice may end up per model. If it does, the chosen order has to be recorded in the
artefact's provenance, or the figure is not reproducible. That is why every report here carries
its own provenance block.

TWO SOURCES OF WEIGHTS, and the difference matters when quoting a result.
  synthetic   seeded Gaussian stacks. Proves the harness and bounds the effect; says nothing about
              any model.
  --model     the residual writers of a real checkpoint, read one tensor at a time through
              `streaming`. Two dimensional writers go to the one-direction-at-a-time routine and
              the same matrix, given a leading axis of one, goes to the batched one.
The refusal basis is ALWAYS a seeded random orthonormal one: this measures arithmetic divergence on
a model's real weights, not the divergence a measured refusal direction would produce.

    python tools/research/deflation_order.py --out divergence.json
    python tools/research/deflation_order.py --model ./Qwen3-0.6B --max-tensors 8
"""
import argparse
import itertools
import json
import platform
import sys

import torch

from senbonzakura import cli

DEFAULT_DIRECTIONS = (1, 2, 3, 4)
DEFAULT_SPARSITIES = (0.0, 0.3, 0.6)
DEFAULT_ROUNDS = (0, 2, 4)
SYNTHETIC_SHAPE = (4, 64, 96)           # [experts, hidden, inner]
SCHEMA = 1


def basis(hidden, k, seed):
    """A seeded orthonormal `[k, hidden]` basis, the shape the bake removes."""
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(hidden, k, generator=gen, dtype=torch.float32))
    return q.T.contiguous()


def divergence(weight, rows, *, sparsity, rounds, strength=1.0):
    """Edit `weight` ([E, out, in] or [out, in]) both ways and compare. Inputs are not modified.

    Order A is the per-direction routine and order B the batched one. Both run on float32 copies,
    so the figure is the orders' disagreement and not a dtype artefact.
    """
    w = weight.detach().clone().float()
    stack = w if w.ndim == 3 else w[None]
    a = stack.clone()
    for e in range(a.shape[0]):
        cli.orthogonalize_np_(a[e], rows, strength, sparsity=sparsity, rounds=rounds)
    b = stack.clone()
    cli._orthogonalize_np_3d_block_(b, rows, strength, sparsity, rounds, restore_norms=True)
    gap = (a - b).abs()
    scale = a.norm().clamp_min(1e-12)
    return {"max_abs": float(gap.max()), "relative_frobenius": float((a - b).norm() / scale),
            "elements_differing": int((gap > 0).sum()), "elements": int(gap.numel())}


def sweep(weights, *, directions, sparsities, rounds, seed):
    """One row per (tensor, K, sparsity, rounds). Order of rows is the order of the inputs."""
    rows = []
    for name, weight in weights:
        hidden = weight.shape[-2]
        for k, sparsity, n in itertools.product(directions, sparsities, rounds):
            result = divergence(weight, basis(hidden, k, seed), sparsity=sparsity, rounds=n)
            rows.append({"tensor": name, "directions": k, "sparsity": sparsity, "rounds": n,
                         **result})
    return rows


def synthetic_weights(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return [("synthetic", torch.randn(*shape, generator=gen, dtype=torch.float32))]


def checkpoint_weights(model_dir, max_tensors):
    """The residual writers of a checkpoint, a tensor at a time, at most `max_tensors` of them."""
    from senbonzakura import streaming
    index = streaming.index_layers(model_dir)
    found = []
    for layer in range(index.count):
        for name, spec in sorted(index.writers(layer).items()):
            if len(found) >= max_tensors:
                return found
            found.append((name, streaming.load_tensor(*spec)))
    return found


def summarise(rows):
    """What the grid shows, in the terms the question is asked in."""
    active = [r for r in rows if r["sparsity"] > 0 and r["rounds"] > 0 and r["directions"] > 1]
    quiet = [r for r in rows if r not in active]
    return {
        "worst_when_both_flags_on": max((r["relative_frobenius"] for r in active), default=None),
        "worst_everywhere_else": max((r["relative_frobenius"] for r in quiet), default=None),
        "cells": len(rows),
    }


def provenance(args, weights):
    return {"schema": SCHEMA, "seed": args.seed, "torch": torch.__version__,
            "python": platform.python_version(),
            "source": args.model or {"synthetic_shape": list(args.shape)},
            "tensors": [name for name, _ in weights],
            "basis": "seeded random orthonormal; not a measured refusal direction",
            "decides": "nothing; no order is named canonical by this report"}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=None, help="checkpoint directory to read writers from")
    p.add_argument("--max-tensors", type=int, default=4)
    p.add_argument("--shape", type=int, nargs=3, default=SYNTHETIC_SHAPE,
                   metavar=("E", "HIDDEN", "INNER"))
    p.add_argument("--directions", type=int, nargs="+", default=list(DEFAULT_DIRECTIONS))
    p.add_argument("--sparsities", type=float, nargs="+", default=list(DEFAULT_SPARSITIES))
    p.add_argument("--rounds", type=int, nargs="+", default=list(DEFAULT_ROUNDS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_tensors < 1:
        print("--max-tensors must be at least 1", file=sys.stderr)
        return 2
    if args.model:
        from senbonzakura import streaming
        try:
            weights = checkpoint_weights(args.model, args.max_tensors)
        except (streaming.ShardError, OSError) as exc:
            print(f"cannot read {args.model}: {exc}", file=sys.stderr)
            return 2
        if not weights:
            print(f"no residual writers found under {args.model}", file=sys.stderr)
            return 2
    else:
        weights = synthetic_weights(tuple(args.shape), args.seed)
    rows = sweep(weights, directions=args.directions, sparsities=args.sparsities,
                 rounds=args.rounds, seed=args.seed)
    report = {"provenance": provenance(args, weights), "summary": summarise(rows), "rows": rows}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        from senbonzakura import crashsafe
        with crashsafe.atomic_write(args.out) as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
