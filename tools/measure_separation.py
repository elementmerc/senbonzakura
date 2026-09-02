#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The Q-14 measurement: does any statistic tell a refusal axis from a topic axis?

Pre-registration: `private/plans/pre-registration-2026-09-02-separation-statistic.md`, written
before any of the code it measures. Read that first. **Nothing in this file may be tuned in the
light of its own output**, and a statistic invented after seeing these numbers is a new
pre-registration rather than a continuation of that one.

WHAT IT MEASURES

The primary, per Q-18: the held-out rejection rate on candidate axes, per statistic, against that
statistic's measured null floor, on one model, across five seeds. Reported as five independent
rates and their spread, never as a mean, because three of this project's withdrawn results were a
mean hiding a range.

TWO THINGS THIS HARNESS DOES THAT DRIVING THE CLI EIGHT TIMES WOULD NOT

**It captures once per seed and scores every arm on that capture.** The pre-registration requires
the statistics to be compared PAIRED, on the same candidate axes from the same extraction, so the
comparison is between statistics and not between two runs that happened to differ. Running the CLI
once per arm would recapture each time and quietly compare eight extractions.

**It draws a different prompt subset per seed.** `Abliterator.load` returns `rows[:n]`, the head of
the dataset, deterministically. Five CLI runs at five seeds would therefore see the IDENTICAL
prompts and identical residuals, and would vary only in how the rows were clustered and split. The
spread across such seeds measures re-splitting, not sampling, which the pre-registration names in
advance as one of the things that would make this measurement worthless. So the harness samples its
own rows per seed and each seed is a genuinely independent extraction.

Everything else runs through the shipped code path unchanged: the real clustering, the real
scoring, the real null floor, the real matching. This file arranges runs and records results; it
does not reimplement any of the arithmetic under test.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from senbonzakura import cli, separation

#: Every arm. All four statistics against both comparisons, so the choice of statistic and the
#: choice of comparison can be read apart from each other rather than confounded.
ARMS = [(name, matched) for name in separation.CHOICES for matched in (False, True)]

#: The real labelling and two nulls. `shuffled` destroys refusal but makes both sides mixtures,
#: which inflates within-group variance and so flatters any statistic that divides by it;
#: `harmless-split` destroys refusal without touching the variance and is the cleaner of the two.
WORLDS = ("real", "shuffled", "harmless-split")


def _sampled_loader(base_load, seed, n_per_side, bad_dir, good_dir, world):
    """Replace `load` with a seeded SAMPLE, optionally with the two labels shuffled together.

    TWO JOBS, AND THE SECOND IS THE CONTROL THIS WHOLE MEASUREMENT NEEDS.

    The sampling: `Abliterator.load` returns `rows[:n]`, the head of the dataset, deterministically.
    Five CLI runs at five seeds would see the identical prompts and identical residuals and vary
    only in how the rows were split, so their spread would measure re-splitting rather than
    sampling. That is named in the pre-registration in advance as a way to get this wrong.

    The controls, of which there are two because the first one has a confound the second does not.

    `shuffled` pools the harmful and harmless prompts and deals them back out at random. Subject
    structure survives, since both sides still hold every subject, and the refusal contrast is
    destroyed, since neither side is harmful. It is a permutation test.

    **Its confound:** each side becomes a 50/50 mixture, so within-group variance rises, and every
    statistic here except the AUC divides by exactly that. Some of the control's extra rejection is
    therefore mechanical rather than an absence of refusal, and a gap measured against it is an
    upper bound on the real discrimination rather than an estimate of it.

    `harmless-split` has no such confound. The harmless prompts alone are split in two and one half
    is labelled harmful. Subject structure is intact, there is no refusal anywhere, and neither
    side is a mixture, so within-group variance is what it would ordinarily be. **This is the
    cleaner null**, and it is the one to read when the two disagree.
    """
    def load(directory, n):
        want = min(n_per_side, n)
        is_bad = Path(directory).name == Path(bad_dir).name
        if world == "real":
            rows = base_load(directory, 10**9)
            g = torch.Generator().manual_seed((seed * 7919 + len(rows)) & 0x7FFFFFFF)
            idx = torch.randperm(len(rows), generator=g)[:min(want, len(rows))].tolist()
            return [rows[i] for i in idx]
        if world == "shuffled":
            pool = base_load(bad_dir, 10**9) + base_load(good_dir, 10**9)
            salt = 104729
        elif world == "harmless-split":
            pool = base_load(good_dir, 10**9)
            salt = 15485863
        else:
            raise ValueError(f"unknown world {world!r}")
        g = torch.Generator().manual_seed((seed * salt + len(pool)) & 0x7FFFFFFF)
        order = torch.randperm(len(pool), generator=g).tolist()
        half = len(order) // 2
        side = order[:half] if is_bad else order[half:]
        return [pool[i] for i in side[:min(want, len(side))]]
    return load


def _capture_once(abl):
    """Memoise `collect_resid` so every arm at this seed scores the SAME residuals."""
    original = abl.collect_resid
    cache = {}

    def collect(prompts):
        key = (len(prompts), prompts[0] if prompts else "", prompts[-1] if prompts else "")
        if key not in cache:
            cache[key] = original(prompts)
        return cache[key]
    abl.collect_resid = collect
    return cache


def _arm_result(abl):
    """The recorded fields of one extraction, as the artefact would carry them."""
    measured = int(getattr(abl, "axes_measured_total", 0) or 0)
    rejected = int(getattr(abl, "axes_rejected_total", 0) or 0)
    floors = [f for f in (getattr(abl, "layer_null_floors", None) or []) if f is not None]
    per_layer = getattr(abl, "dirs_per_layer", None) or []
    return {
        "axes_measured": measured,
        "axes_rejected": rejected,
        # THE PRIMARY. None rather than 0.0 when nothing was measured, because a rate over zero
        # candidates is not a rate and must not average into anything.
        "rejection_rate": (rejected / measured) if measured else None,
        "rejected_by_null": int(getattr(abl, "axes_rejected_by_null", 0) or 0),
        "null_floor_max": max(floors) if floors else None,
        "null_floor_min": min(floors) if floors else None,
        "max_axis_separation": getattr(abl, "max_axis_separation", None),
        "best_rejected_separation": getattr(abl, "best_rejected_separation", None),
        "matching_quality": getattr(abl, "matching_quality", None),
        "directions_per_layer_max": max(per_layer) if per_layer else None,
        "directions_per_layer_mean": (sum(per_layer) / len(per_layer)) if per_layer else None,
        "filter_is_unsatisfiable": bool(getattr(abl, "filter_is_unsatisfiable", False)),
    }


def run_seed(args, seed, log, model, tok, world):

    run_args = cli.build_parser().parse_args([
        "--model", args.model, "--track", args.track, "--out", str(Path(args.out).parent / "unused"),
        "--device", args.device, "--dir-prompts", str(args.dir_prompts),
        "--max-directions", str(args.max_directions),
        "--direction-clusters", str(args.direction_clusters), "--seed", str(seed),
    ])
    abl = cli.Abliterator(run_args, log, model=model, tok=tok)
    bad_dir = f"{args.track}/bad_ds"
    good_dir = args.good_ds or f"{args.track}/good_ds"
    abl.load = _sampled_loader(abl.load, seed, args.dir_prompts, bad_dir, good_dir, world)
    cache = _capture_once(abl)

    out, captures_after_first = {}, None
    for name, matched in ARMS:
        abl.args.separation_statistic = name
        abl.args.matched_scoring = matched
        t0 = time.time()
        abl.extract_directions(bad_dir, good_dir, None, good_dir)
        arm = _arm_result(abl)
        arm["seconds"] = round(time.time() - t0, 2)
        out[f"{name}{'+matched' if matched else ''}"] = arm
        rate = arm["rejection_rate"]
        log(f"    {name:>14} {'matched' if matched else 'unmatched':>9}: "
            f"rejected {arm['axes_rejected']}/{arm['axes_measured']}"
            f" = {'n/a' if rate is None else f'{rate:.1%}'}")
        # Every arm must score the SAME candidate axes, or this compares extractions rather than
        # statistics, which is exactly the confound that got the 2026-08-03 K comparison withdrawn.
        # Asserted as "the first arm captured, and no arm after it did" rather than as a count,
        # because a count would still pass if arm five captured while arm two did not.
        if captures_after_first is None:
            captures_after_first = len(cache)
        if len(cache) != captures_after_first:
            # Loud rather than an assert, per the baseline: this is the integrity check the
            # measurement rests on, and it must fire the same way with optimisations on.
            raise RuntimeError(
                f"arm {name}/matched={matched} triggered a fresh capture ({captures_after_first} "
                f"-> {len(cache)}), so the arms no longer share one extraction and this run "
                f"compares extractions rather than statistics. Refusing to record it.")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--track", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dir-prompts", type=int, default=256, dest="dir_prompts")
    p.add_argument("--max-directions", type=int, default=8, dest="max_directions")
    p.add_argument("--direction-clusters", type=int, default=8, dest="direction_clusters")
    p.add_argument("--good-ds", default=None, dest="good_ds",
                   help="override the harmless set, for a matched-form contrast")
    a = p.parse_args(argv)

    def log(m):
        print(m, flush=True)

    results = {
        "pre_registration": "private/plans/pre-registration-2026-09-02-separation-statistic.md",
        "code_version": cli.code_version(),
        "model": a.model, "track": a.track, "device": a.device,
        "dir_prompts": a.dir_prompts, "max_directions": a.max_directions,
        "direction_clusters": a.direction_clusters,
        "torch": torch.__version__, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "note": ("Each seed draws its own prompt subset, because Abliterator.load returns the head "
                 "of the dataset and five seeds over one fixed set of rows would measure "
                 "re-splitting rather than sampling. Within a seed every arm scores one capture, "
                 "so the statistics are compared paired."),
        "seeds": {},
    }
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16,
        device_map=None if a.device == "cpu" else a.device)
    for seed in range(a.seeds):
        for world in WORLDS:
            log(f"\n=== seed {seed}, {world} labels ===")
            got = run_seed(a, seed, log, model, tok, world)
            results["seeds"].setdefault(str(seed), {})[world] = got
            Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    def rates(key, world):
        return [results["seeds"][s][world][key]["rejection_rate"] for s in results["seeds"]]

    log("\n" + "=" * 130)
    log("REJECTION RATE. The shuffled column is the control: the labels are dealt at random, so")
    log("no candidate there carries refusal and every one of them should be rejected.")
    log("=" * 130)
    log(f"{'arm':>24} | {'real labels, per seed':>32} | {'SHUFFLED control':>32} | "
        f"{'HARMLESS-SPLIT control':>32}")
    log("-" * 130)
    for name, matched in ARMS:
        key = f"{name}{'+matched' if matched else ''}"
        f = lambda rs: " ".join("  n/a" if r is None else f"{r:5.1%}" for r in rs)  # noqa: E731
        log(f"{key:>24} | {f(rates(key, 'real')):>32} | "
            f"{f(rates(key, 'shuffled')):>32} | {f(rates(key, 'harmless-split')):>32}")
    log("=" * 130)
    # The verdict, stated per arm rather than left for a reader to derive. An arm that rejects the
    # shuffled world no harder than the real one has not been shown to measure refusal at all.
    for name, matched in ARMS:
        key = f"{name}{'+matched' if matched else ''}"
        real = [r for r in rates(key, "real") if r is not None]
        clean = [r for r in rates(key, "harmless-split") if r is not None]
        if not real or not clean:
            continue
        # Read against the harmless-split control, because the shuffled one inflates within-group
        # variance and so overstates the gap for every statistic that divides by it.
        gap = statistics.median(clean) - statistics.median(real)
        verdict = ("rejects the clean control harder than the real labels" if gap > 0.1 else
                   "CANNOT TELL THE CLEAN CONTROL FROM THE REAL LABELS")
        log(f"{key:>24} | real {statistics.median(real):5.1%} | clean control "
            f"{statistics.median(clean):5.1%} | gap {gap:+6.1%} | {verdict}")
    log("-" * 130)
    for name, matched in ARMS:
        if not matched:
            continue
        key = f"{name}+matched"
        qualities = [results["seeds"][s]["real"][key]["matching_quality"]
                     for s in results["seeds"]]
        qualities = [q for q in qualities if q is not None]
        if qualities:
            median = statistics.median(qualities)
            verdict = ("MATCHING ACHIEVED NOTHING" if median > cli.MATCHING_USELESS_RATIO
                       else "matching found on-subject controls")
            log(f"{key:>24} | matching quality {median:.3f} | {verdict}")
    log(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
