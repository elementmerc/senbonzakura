#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""How much of an ablated direction the norm restore puts back, as a file rather than a memory.

WHY THIS EXISTS

The standard abliteration edit removes a refusal direction from a weight matrix and then restores
each output row's original length, so the model's calibration survives. Those two steps do not
commute. The projection acts ACROSS rows; the restore scales each row ON ITS OWN. A matrix whose
rows were orthogonal to the refusal span stops being orthogonal the moment the lengths go back.

    W  ->  normalise rows  ->  subtract the projection  ->  put the lengths back
                                        ^                            ^
                                 exact, to 5e-07            and here it comes back

The size of what comes back depends on how uneven the row lengths are, which varies by model and
which nothing has ever recorded. This measures it.

WHY IT IS A SCRIPT AND NOT A PARAGRAPH

The numbers were quoted in a handoff and lived nowhere a machine could reproduce them. A figure
drawn from prose is a picture of somebody's memory. This writes a JSON, and anything published
renders that file, so a reader who disbelieves the chart can rerun the measurement in a minute on
a laptop.

TWO MEASUREMENTS, AND THE SECOND IS THE ONE THAT BOUNDS THE ANSWER

  --sweep   synthetic matrices at a controlled row-length spread. Clean, and its weakness is that
            the spread is UNIFORM. Real weights are not: they have a tight bulk and long tails,
            so a real model at "7x max/min" leaks less than a synthetic one at 7x.
  --model   the row-length spread of the tensors this tool actually edits, in a real checkpoint.
            This is what says where on the sweep a real model sits, and it is why the honest
            answer is a range rather than a number.

USAGE

  tools/leak_sweep.py --out leak.json                     # the synthetic sweep
  tools/leak_sweep.py --model path/to/weights --out m.json # a real model's row-length spread
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Row-length spreads to measure at. The bottom of the range is below 1.0 on purpose: a matrix
#: whose rows are MORE even than average leaks least, and showing that the curve goes down as well
#: as up is what makes it a measurement rather than an argument.
SPREADS = (0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)

#: Repeats per point. The spread between seeds is reported, not averaged away: a curve with no
#: sense of its own noise cannot be argued with, which is a defect and not a feature.
SEEDS = 5

#: Direction counts. Refusal is not one direction in every model, and the leak should be shown to
#: behave the same way whether one is removed or several.
DIRECTIONS = (1, 2, 4)

#: The tensors this tool edits: the ones that write into the residual stream. Suffix-matched so
#: the same list covers attention output, MLP output and the mixer paths of hybrid models.
WRITER_SUFFIXES = ("o_proj.weight", "down_proj.weight", "out_proj.weight", "w2.weight")


def measure(spread, k, seed, rounds, *, h=256, cols=512):
    """One point: how much of the direction survives the edit, as a fraction of what was there.

    Returns (leak_fraction, row_length_error). The second number matters as much as the first,
    because the whole reason for restoring the lengths is that the model needs them: a cleaner cut
    that got there by abandoning the lengths would not be a fix, it would be a different edit.
    """
    import torch

    from senbonzakura.cli import orthogonalize_np_

    torch.manual_seed(seed)
    w = torch.randn(h, cols) * 0.02
    w *= (1.0 + torch.rand(h, 1) * spread)
    r = torch.linalg.qr(torch.randn(h, k))[0].T.contiguous()

    before = float((r @ w).norm())
    lengths = w.norm(dim=1).clone()
    orthogonalize_np_(w, r, 1.0, rounds=rounds)
    after = float((r @ w).norm())
    err = float(((w.norm(dim=1) - lengths).abs() / lengths.clamp_min(1e-8)).max())
    return (after / before if before else 0.0), err


def sweep(spreads=SPREADS, seeds=SEEDS, directions=DIRECTIONS):
    """The whole synthetic curve, single pass against four alternating rounds."""
    from senbonzakura.cli import ABLATION_ROUNDS

    rows = []
    for spread in spreads:
        for k in directions:
            for rounds in (0, ABLATION_ROUNDS):
                got = [measure(spread, k, s, rounds) for s in range(seeds)]
                leaks = sorted(x for x, _e in got)
                errs = [e for _x, e in got]
                rows.append({
                    "row_length_spread": spread,
                    "directions": k,
                    "rounds": rounds,
                    "leak_fraction_median": leaks[len(leaks) // 2],
                    "leak_fraction_min": leaks[0],
                    "leak_fraction_max": leaks[-1],
                    "row_length_error_max": max(errs),
                    "seeds": seeds,
                })
    return rows


def _row_norm_stats(t):
    """max/min and p99/p1 of a matrix's row lengths.

    Both, because they say different things and only the pair is honest. max/min is what a
    synthetic sweep controls; p99/p1 describes the bulk, and a real matrix with a tight bulk and a
    few long rows leaks far less than a synthetic one with the same max/min.
    """
    n = t.norm(dim=1).float()
    n = n[n > 0]
    if n.numel() < 100:
        return None
    s = n.sort().values
    p1 = float(s[int(0.01 * (s.numel() - 1))])
    p99 = float(s[int(0.99 * (s.numel() - 1))])
    return {
        "rows": int(n.numel()),
        "max_over_min": float(s[-1] / s[0]),
        "p99_over_p1": (p99 / p1) if p1 else None,
    }


def model_spreads(path):
    """The row-length spread of every residual writer in a checkpoint on disk.

    Reads the tensors it needs and nothing else. No GPU, no model class, no config: this is a
    statement about numbers in a file, and loading a model to make it would let an architecture
    the loader does not know refuse a measurement that does not depend on the architecture.
    """
    import torch  # noqa: F401  (safetensors returns torch tensors)
    from safetensors import safe_open

    shards = sorted(Path(path).glob("*.safetensors"))
    if not shards:
        raise SystemExit(
            f"no .safetensors files in {path}. This measures a checkpoint on disk; point it at "
            f"the directory holding the weights.")
    out = []
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():                                # noqa: SIM118 (safetensors API)
                if not name.endswith(WRITER_SUFFIXES):
                    continue
                t = f.get_tensor(name)
                if t.ndim != 2:
                    # Fused expert stacks are [E, out, in] and each expert is its own matrix.
                    # Skipped rather than flattened: flattening would mix experts and report a
                    # spread no single edited matrix ever has.
                    continue
                stats = _row_norm_stats(t.float())
                if stats:
                    out.append({"tensor": name, **stats})
    return out


def build_parser():
    ap = argparse.ArgumentParser(
        prog="leak_sweep",
        description="Measure how much of an ablated direction the norm restore puts back.")
    ap.add_argument("--model", default=None,
                    help="a directory of safetensors: report the row-length spread of the "
                         "tensors this tool edits, which is what says where a real model sits on "
                         "the synthetic curve")
    ap.add_argument("--seeds", type=int, default=SEEDS,
                    help=f"repeats per point (default {SEEDS})")
    ap.add_argument("--out", default="", help="write the measurements here as JSON")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.seeds < 1:
        raise SystemExit("--seeds must be at least 1; a point measured zero times is not a point.")

    record = {"schema": "senbonzakura-leak-sweep/1"}
    if a.model:
        stats = model_spreads(a.model)
        record["model"] = {"path": str(a.model), "tensors": stats}
        if stats:
            worst = max(stats, key=lambda s: s["max_over_min"])
            bulk = [s["p99_over_p1"] for s in stats if s["p99_over_p1"]]
            print(f"{len(stats)} edited tensors")
            print(f"  row length max/min: worst {worst['max_over_min']:.2f}x "
                  f"({worst['tensor']})")
            if bulk:
                print(f"  bulk p99/p1:        median {sorted(bulk)[len(bulk) // 2]:.2f}x")
            print("  The bulk figure is the one to read beside the synthetic curve: a matrix with "
                  "a tight bulk\n  and a few long rows leaks far less than a synthetic one at the "
                  "same max/min.")
    else:
        record["sweep"] = sweep(seeds=a.seeds)
        print(f"{'spread':>8} {'K':>3} {'rounds':>7} {'leak':>8} {'length err':>11}")
        for r in record["sweep"]:
            print(f"{r['row_length_spread']:>8.1f} {r['directions']:>3} {r['rounds']:>7} "
                  f"{r['leak_fraction_median']:>7.1%} {r['row_length_error_max']:>11.2e}")

    if a.out:
        Path(a.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
