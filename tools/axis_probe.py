#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Ask why a layer got the direction count it did, without running a search.

The headline claim of this project is that refusal occupies a subspace rather than a single
direction. On 2026-08-03 a five-seed comparison of K=1 against K=3 produced a clean, separated,
pre-registered result, and it had to be withdrawn: both arms had ablated exactly one direction
per layer, because no second principal axis cleared `MIN_AXIS_SEPARATION = 0.5`. The arms
performed identical surgery, so the numbers compared two search trajectories rather than two
direction budgets.

Three explanations survive that, and the direction count cannot tell them apart:

  1. the threshold, a constant chosen once and never validated, is set above a real second
     direction;
  2. the model, Qwen3-1.7B specifically, genuinely carries refusal in one direction;
  3. the corpus, whose harmful side may vary too little to give the PCA anything past the
     first axis.

This probe answers all three, because it reports the separation of every candidate axis rather
than only whether it passed. Point it at a different model to test (2) and a different track to
test (3). It never edits weights and never generates, so it costs one forward pass over the
contrast prompts rather than the hours a search costs.

It drives `cli.Abliterator.extract_directions` itself rather than reimplementing the extraction.
That is deliberate and it is the lesson of the withdrawn result: a probe that re-derives the
maths measures the probe, and the only claim worth making is about the code that ships.

Usage:
  python tools/axis_probe.py --model <hf-id> --track <dir with bad_ds/good_ds> \
      [--out probe.json] [--device cuda] [--max-directions 8] [--dir-prompts 128]

Reading the output: `best_rejected_separation` is the number to look at first. Close under the
threshold means the constant decided the count; far under it means the second direction is not
there to find.
"""
import argparse
import json
import sys

from senbonzakura import cli

# The probe reads the same relative criterion the extractor uses rather than carrying its own, so
# the two cannot disagree about whether a run was broken. It was an independent absolute constant
# for a few hours and immediately drifted: 1e-6, calibrated on a synthetic case, against a real
# model whose residue peaked at 1.07e-5 across 3,556 axes.
STRUCTURAL_ZERO = cli.MIN_AXIS_SEPARATION * cli.STRUCTURAL_ZERO_FRACTION


def build_args(argv=None):
    """Parse into the REAL parser, then override, so no default is invented here.

    A hand-written namespace is how five tests died for eleven days in this repository: it drifts
    from the parser silently, and a probe whose defaults differ from the tool's defaults is
    measuring a configuration nobody runs.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", default="track")
    ap.add_argument("--out", default=None, help="write the record here as JSON (default: stdout only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-directions", type=int, default=8,
                    help="how many candidate axes to pursue per layer. Higher than a real run's K "
                         "on purpose: the question is what is THERE, not what would be applied.")
    ap.add_argument("--dir-prompts", type=int, default=128)
    ap.add_argument("--good-ds", default=None,
                    help="override the harmless side, which is half of varying the fit corpus")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--label", default=None, help="name for this probe in the record")
    own = ap.parse_args(argv)

    args = cli.build_parser().parse_args(
        ["--model", own.model, "--track", own.track, "--device", own.device,
         "--max-directions", str(own.max_directions), "--dir-prompts", str(own.dir_prompts)]
        + (["--good-ds", own.good_ds] if own.good_ds else [])
        + (["--load-in-4bit"] if own.load_in_4bit else [])
        + (["--trust-remote-code"] if own.trust_remote_code else []))
    return own, args


def summarise(a):
    """The record, and the reading of it, as data rather than prose in a log."""
    seps = a.axis_separations
    measured = [d for layer in seps for d in layer]
    rejected = [d for d in measured if d < cli.MIN_AXIS_SEPARATION]
    best = a.best_rejected_separation
    threshold = cli.MIN_AXIS_SEPARATION

    # An exact zero is not a small measurement, it is the signature of a broken one. The candidate
    # axes are orthogonalised against a basis spanning both class means, so the numerator of
    # Cohen's d vanishes by construction and the filter cannot be passed at any positive threshold.
    # Reported first because every other reading below assumes the instrument works.
    # From the extractor's exact totals, not from `measured`, which is a bounded per-layer sample.
    # One real layer measured 127 candidates and recorded 8, so a count taken from the sample
    # understates the evidence by more than an order of magnitude.
    total = getattr(a, "axes_measured_total", len(measured))
    peak = getattr(a, "max_axis_separation", None)
    if peak is None:
        peak = max((abs(d) for d in measured), default=None)
    structural = bool(total) and peak is not None and peak < STRUCTURAL_ZERO

    if structural:
        verdict = (f"every one of the {total} candidate axes scored a separation "
                   "indistinguishable from zero, which is what the geometry forces rather than "
                   "anything about this model or corpus: the axes are orthogonalised against a "
                   "basis spanning both class means, and Cohen's d is a difference of class means. "
                   "The filter cannot be passed at any positive threshold. Nothing here measures "
                   "direction count")
    elif best is None:
        verdict = ("no candidate axis was rejected, so the threshold did not bind here and the "
                   "direction count is a property of the model and the corpus")
    elif best >= threshold * 0.8:
        verdict = (f"the best rejected axis reached {best:.4f} against a threshold of {threshold}, "
                   "which is close enough that the constant decided the direction count. The "
                   "single-direction reading is a property of MIN_AXIS_SEPARATION until it is varied")
    else:
        verdict = (f"the best rejected axis reached only {best:.4f} against a threshold of "
                   f"{threshold}, so lowering the threshold to any defensible value would not add "
                   "a direction. The second direction is absent, not filtered")

    return {
        "model": a.args.model,
        "track": a.args.track,
        # Recorded even when unset. A probe that varies only the harmless side is otherwise
        # indistinguishable in its own artefact from one that did not, and two records that look
        # identical while measuring different things is how this project loses a result.
        "good_ds_override": a.args.good_ds,
        "hidden_size": a.H,
        "layers": a.NL,
        "max_directions_requested": a.KMAX,
        "dir_prompts": a.args.dir_prompts,
        "axis_separation_threshold": threshold,
        "directions_per_layer": a.dirs_per_layer,
        "axis_separations": seps,
        "axes_measured": total,
        "axes_recorded": len(measured),
        "axes_rejected": len(rejected),
        "best_rejected_separation": best,
        "max_separation_any_axis": peak,
        "filter_is_unsatisfiable": structural,
        "verdict": verdict,
    }


def main(argv=None):
    own, args = build_args(argv)
    a = cli.Abliterator(args, lambda m: print(m, flush=True))
    good_ds = args.good_ds or f"{args.track}/good_ds"
    a.extract_directions(f"{args.track}/bad_ds", good_ds, args.hedge_ds,
                         args.clean_ds or good_ds)

    record = summarise(a)
    if own.label:
        record["label"] = own.label

    print("\n── axis probe ──────────────────────────────────────────────")
    print(f"  model                     {record['model']}")
    print(f"  track                     {record['track']}")
    print(f"  candidate axes measured   {record['axes_measured']}")
    print(f"  rejected below threshold  {record['axes_rejected']}")
    print(f"  best rejected separation  {record['best_rejected_separation']}")
    print(f"  directions per layer      min {min(record['directions_per_layer'])} "
          f"max {max(record['directions_per_layer'])}")
    print(f"\n  {record['verdict']}.")

    if own.out:
        with cli.atomic_write(own.out) as f:
            json.dump(record, f, indent=2)
        print(f"\n  written to {own.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
