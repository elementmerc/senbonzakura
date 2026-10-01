#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The sampled, budgeted mutation report, and the ratchet that cannot fire for sampling reasons.

This is Q-51 option B. Option C, the mutation requirement on new guards, is a record kept by hand
at the moment a guard is written, and it is proven: it is where the 29% miss rate came from. B is
the backstop for when C is forgotten, and it answers a different question.

WHAT THE TWO INSTRUMENTS EACH ANSWER, because running them as one would be the
two-mechanisms-deriving-one-fact defect this project keeps finding

    tools/research/mutate.py   21 hand-written cases over specific literals, each paired with the
                               one test file that claims to guard it. It is a RECORD of the
                               2026-09-27 measurement and a regression test over those 21 guards.
                               It samples nothing and it is not a mutation engine.

    this file                  samples from the 48,822 mutants mutmut generates over our own
                               modules, inside a wall-clock budget, and reports what survived.

Pretending the first is the second would be dishonest in the direction that flatters us: 21
chosen cases that were all fixed the day they were found is not a sample of the tree.

THE NUMBER THAT WAS NEVER MEASURED, measured 2026-10-01 on chronos

    mutants over our own code, vendor/ excluded    48,822   (63 modules, 25,427 executable lines)
    mutant generation for the whole package        53.6 s   (0.85 s per module, a fixed cost)
    mutants on the changed lines of one commit     median 44, mean 148, p90 367, max 737
    measured cost per mutant, `say.py`             0.046 s  (21.5 mutants/second, 4 children)

That last figure is the one the plan guessed at 5 s, and it is wrong by a factor of about a
hundred, because mutmut 3 does not spawn a pytest process per mutant. It `os.fork()`s a child
from a parent that has already imported the suite, runs only the tests the coverage map says
touch the mutated line, sorts them fastest-first, and passes `-x` so it stops at the first test
that notices. So the per-mutant cost is the runtime of the quickest covering test, not of a
pytest startup, and certainly not of the suite.

**The 0.046 s figure is the fast end and must not be quoted as the tree's rate.** It was measured
on a module whose covering tests are pure Python and run in 0.55 s. A module whose tests import
torch will cost far more per mutant, and that could not be measured on this machine because it
has no torch. The job therefore measures its own rate every night and reports it, rather than
trusting a constant: see `--calibrate`.

WHY THE BUDGET IS ENFORCED AND NOT HOPED FOR

mutmut has no time budget and no sampling. `mutmut run --help` offers exactly one option,
`--max-children`. What it does accept is an explicit list of mutant names, which is the whole
lever this file pulls: it enumerates candidates, samples them under a seed, and hands mutmut a
list sized to the budget at the measured rate. The deadline is then enforced here, and a run that
hits it reports PARTIAL rather than a score.

THE RATCHET, AND THE ANSWER TO "WHAT IF IT FAILS FOR SAMPLING REASONS"

This is the part that decides whether the job survives a fortnight. A ratchet on a sampled
PERCENTAGE fails for sampling reasons roughly half the time it is level: at a true kill rate of
0.8 and a sample of 100, the standard error is about 4 percentage points, so "must not fall below
last night" fires on noise, somebody disables it, and the gate is gone.

So the ratchet is not on the rate. **It is on identified mutants, where there is no sampling error
at all.** The rule is:

    A mutant that was KILLED, whose function has not changed since, may never come back SURVIVED.

That comparison is like with like on a named mutant, and it is deterministic: the same mutant
against the same code gives the same verdict every time. mutmut hands us
`hash_by_function_name`, so "has not changed since" is a fact rather than a guess, and a mutant
whose function was edited is retired from the ledger instead of being compared across a rewrite.

What such a regression means is narrow and worth catching: a test got weaker. Nothing else can
produce it.

A mutant that survives and was never seen before is a DISCOVERY. It is recorded and reported, and
it does NOT fail the ratchet, because the sample reaching new ground is the job working rather
than the tree getting worse. That asymmetry is the whole design: discoveries accumulate, and only
regressions ring.

The kill rate is still reported, because it is informative, and it is reported as a sample with an
interval and a refusal to be called a score. It gates nothing.

WHY THE SAMPLE ROTATES OVER THE WHOLE TREE RATHER THAN THE CHANGED MODULES

A deviation from the plan, argued rather than assumed. The plan says to sample from the modules
that changed since the last run. But freshly written code is exactly what option C already covers,
at the moment it is written, by the person writing it. B's distinct value is the BACKLOG: the
48,822 mutants in code nobody has ever mutation-tested, where the unknown lives. Scoping B to the
diff would point both instruments at the same ground and leave the backlog unmeasured for ever.

So the sample is drawn least-recently-tested first across the whole package, with changed modules
promoted to the front of the queue so new code is still seen quickly. At 2,000 mutants a night the
backlog is covered in about 25 nights, and every module's worst case is bounded.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "private" / "reviews" / "mutation-ledger.json"
PACKAGE = ROOT / "src" / "senbonzakura"

#: Vendored upstream code is not ours to mutate: a surviving mutant in llama.cpp's `gguf-py` is a
#: finding about llama.cpp's tests, which we neither own nor ship a fix for.
EXCLUDE = ("/vendor/",)

#: The three outcomes, as everywhere else in this repository's gates. `no_tests` is mutmut's own
#: name for "the coverage map says nothing touches this line", which is a DID NOT RUN about that
#: mutant and must never be counted as a survivor: it would inflate the miss rate with mutants
#: nothing even attempted.
KILLED, SURVIVED, NO_TESTS = "killed", "survived", "no_tests"


def modules() -> list[Path]:
    return [p for p in sorted(PACKAGE.rglob("*.py"))
            if not any(x in str(p) for x in EXCLUDE)]


def changed_modules(since: str | None) -> set[str]:
    """Modules touched since a git ref, promoted to the front of the sampling queue."""
    if not since:
        return set()
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{since}..HEAD", "--", "src/senbonzakura"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"warning: could not diff against {since}: {proc.stderr.strip()}", file=sys.stderr)
        return set()
    return {line for line in proc.stdout.split()
            if line.endswith(".py") and not any(x in line for x in EXCLUDE)}


def load_ledger() -> dict:
    if not LEDGER.exists():
        return {"version": 1, "runs": [], "mutants": {}}
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def save_ledger(ledger: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".json.part")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(LEDGER)                       # atomic: never a half-written ledger


def wilson_interval(killed: int, total: int) -> tuple[float, float]:
    """A 95% interval on the kill rate, so the sampled figure is never printed bare.

    Wilson rather than the textbook normal approximation, because the normal one is badly wrong
    near 0 and 1 and a kill rate near 1 is exactly where this job will usually sit.
    """
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    p = killed / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def enumerate_candidates(paths: list[Path]) -> dict[str, dict]:
    """Every mutant mutmut would generate for these modules, with its function's hash.

    The hash is what makes the ratchet honest later: it says whether the code under a named
    mutant is the same code that was judged last time.
    """
    # `get_mutant_name` is imported rather than reimplemented. The rule is "dotted path, suffix
    # stripped, leading `src.` removed, `.__init__.` collapsed", and writing that out here would
    # be a second mechanism deriving one fact: the day upstream changes it, our names would stop
    # matching and the sample would silently run nothing. It matters because
    # `mutate_file_contents` returns BARE names (`x_width__mutmut_1`) while `mutmut run` and the
    # `.meta` files both want the qualified form (`senbonzakura.say.x_width__mutmut_1`), and that
    # mismatch was a real bug in this function until it was checked against a run.
    #
    # Imported inside the function, not at module scope, so this tool still imports, and its
    # tests still run, on a machine with no mutmut: the ratchet logic is pure and testable, and
    # only enumeration needs the engine.
    try:
        from mutmut.mutation.file_mutation import mutate_file_contents
        from mutmut.utils.format_utils import get_mutant_name
    except ImportError as exc:
        raise RuntimeError(
            "mutmut is not importable, so no mutant could be enumerated. This is DID NOT RUN, "
            "not a clean report. Install the pinned version and re-run.",
        ) from exc

    out: dict[str, dict] = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        try:
            mutated = mutate_file_contents(str(rel), path.read_text(encoding="utf-8"))
        except Exception as exc:
            # A module mutmut cannot parse is reported, never skipped silently: a quiet skip is
            # how a sample over nine modules gets read as a sample over ten.
            print(f"warning: {rel} could not be mutated: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        hashes = dict(mutated.hash_by_function_name)
        for name in mutated.mutant_names:
            qualified = get_mutant_name(rel, name)
            out[qualified] = {
                "module": str(rel),
                "mutant": qualified,
                # The hash is per generated FUNCTION: every `x_width__mutmut_N` shares `x_width`.
                "function_hash": hashes.get(name.rsplit("__mutmut_", 1)[0], ""),
            }
    return out


#: mutmut's per-mutant exit codes, read from `mutants/<module>.meta`. Established by reading its
#: source and confirmed against a real run: 153 mutants came back {0: 30, 1: 123}.
#:   0  the suite passed with the mutant in place      -> SURVIVED
#:   1  a test failed                                  -> KILLED
#:  33  no test covers the mutated line                -> NO_TESTS, a DID NOT RUN about that
#:                                                        mutant, never a survivor
#:  37  a type checker rejected it                     -> KILLED, by a different instrument
EXIT_VERDICT = {0: SURVIVED, 1: KILLED, 33: NO_TESTS, 37: KILLED}


#: How much of the sample has to reach a verdict before the run counts as having happened.
#: Not 100%: a mutant can legitimately be lost to the per-mutant CPU limit, and failing the whole
#: night's report over one of those would be a gate that cries wolf. Not low either, because the
#: whole point is to refuse a run that evaluated a handful and read as clean.
COMPLETENESS_FRACTION = 0.9


def completeness_floor(sampled: int) -> int:
    """The minimum number of verdicts a run must produce to be reported at all.

    Extracted from `main` so it can be tested. A guard that only its author has ever seen fire,
    by hand, on one afternoon, is exactly what Q-51 option C exists to stop shipping.
    """
    return max(1, int(COMPLETENESS_FRACTION * sampled))


def read_meta(sample_keys: set[str]) -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    """Per-mutant verdicts, per-mutant function hashes and per-mutant durations.

    `mutmut results` prints only the survivors, so the killed set cannot be recovered from it and
    the ratchet needs both. The `.meta` files carry every mutant's exit code, which is why this
    reads them instead.
    """
    verdicts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    durations: dict[str, float] = {}
    for meta in sorted((ROOT / "mutants").rglob("*.py.meta")):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: {meta} unreadable: {exc}", file=sys.stderr)
            continue
        fn_hashes = data.get("hash_by_function_name", {})
        for mutant, code in data.get("exit_code_by_key", {}).items():
            # The `.meta` already keys by the qualified name, which is the same key
            # `enumerate_candidates` produces via `get_mutant_name`. No second derivation.
            if sample_keys and mutant not in sample_keys:
                continue              # a verdict left over from an earlier run, not ours
            verdicts[mutant] = EXIT_VERDICT.get(code, f"unknown-exit-{code}")
            base = mutant.rsplit("__mutmut_", 1)[0].rsplit(".", 1)[-1]
            hashes[mutant] = fn_hashes.get(base, "")
            if mutant in data.get("durations_by_key", {}):
                durations[mutant] = float(data["durations_by_key"][mutant])
    return verdicts, hashes, durations


def ratchet(ledger: dict, results: dict[str, str], hashes: dict[str, str]) -> list[str]:
    """Regressions only. A new survivor is a discovery, not a failure. See the module docstring."""
    regressions = []
    known = ledger.get("mutants", {})
    for key, verdict in results.items():
        if verdict != SURVIVED:
            continue
        prior = known.get(key)
        if not prior or prior.get("verdict") != KILLED:
            continue                      # never seen, or already known to survive: a discovery
        if prior.get("function_hash") and prior["function_hash"] != hashes.get(key, ""):
            continue                      # the code changed, so this is not a like-for-like
        regressions.append(
            f"{key}: was KILLED on {prior.get('last_seen', 'an earlier run')} and now SURVIVES, "
            f"with its function unchanged. A test got weaker; nothing else does this.",
        )
    return regressions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sampled, budgeted mutation report (Q-51 B).")
    parser.add_argument("--budget-seconds", type=int, default=5400,
                        help="hard wall-clock ceiling for the mutation phase (default 90 min)")
    parser.add_argument("--rate", type=float, default=None,
                        help="measured mutants per second; omit to read the ledger's last "
                             "measured rate, which is how the sample size self-calibrates")
    parser.add_argument("--seed", type=int, default=None,
                        help="sampling seed; defaults to the run date so a day is reproducible")
    parser.add_argument("--since", default=None,
                        help="git ref: modules changed since it go to the front of the queue")
    parser.add_argument("--max-sample", type=int, default=4000,
                        help="never hand mutmut more than this many mutants in one run")
    parser.add_argument("--enumerate-only", action="store_true",
                        help="count and sample, run nothing. Needs no test dependencies.")
    parser.add_argument("--max-children", type=int, default=4)
    parser.add_argument("--recheck-fraction", type=float, default=0.25,
                        help="share of the sample spent re-running already-judged mutants. "
                             "This is what makes the ratchet able to fire at all: without it "
                             "the rotation samples disjoint sets and nothing is ever compared.")
    parser.add_argument("--allow-first-run", action="store_true",
                        help="permit an empty ledger to exit 0. Set it for the genuine first "
                             "run and nowhere else: without it, a lost ledger is loud, which "
                             "is the point, because an empty ledger cannot fail.")
    args = parser.parse_args(argv)

    started = time.time()
    ledger = load_ledger()

    # The rate: measured, inherited, or refused. A guessed rate is how a budget overruns.
    rate = args.rate
    if rate is None:
        prior_runs = ledger.get("runs", [])
        measured = [r["rate"] for r in prior_runs if r.get("rate")]
        if measured:
            rate = sum(measured[-3:]) / len(measured[-3:])
            print(f"rate: {rate:.2f} mutants/s, the mean of the last "
                  f"{len(measured[-3:])} run(s) in the ledger")
        else:
            rate = 2.0
            print(f"rate: no measured rate in the ledger yet, so using a DELIBERATELY "
                  f"PESSIMISTIC {rate} mutants/s for the first run. The measured figure on "
                  f"`say.py` was 21.5/s, and the first run reports its own rate for the next "
                  f"one to use.")
    else:
        print(f"rate: {rate:.2f} mutants/s, given on the command line")

    try:
        candidates = enumerate_candidates(modules())
    except RuntimeError as exc:
        print(f"DID NOT RUN: {exc}", file=sys.stderr)
        return 2
    if not candidates:
        print("DID NOT RUN: no mutants were enumerated at all. A sample of nothing is not a "
              "clean report.", file=sys.stderr)
        return 2

    # Sample size from the budget and the rate, never the other way round.
    affordable = max(1, int(args.budget_seconds * rate))
    sample_size = min(affordable, args.max_sample, len(candidates))

    promoted = changed_modules(args.since)
    seen = ledger.get("mutants", {})

    # THE ROTATION AND THE RATCHET PULL AGAINST EACH OTHER, and the first version of this got it
    # wrong. Ordering by least-recently-tested means two consecutive runs sample almost disjoint
    # sets, so no mutant in tonight's sample has a prior verdict, so there is nothing to compare
    # and the ratchet never evaluates. Measured: a second run reported "0 prior verdicts covered
    # this sample" with 25 verdicts sitting in the ledger.
    #
    # So the sample is split. A recheck cohort re-runs mutants already judged, which is the only
    # thing that can ever trip the ratchet, and the remainder explores new ground to work through
    # the backlog. Oldest judged first, so every mutant is eventually revisited.
    seed = args.seed if args.seed is not None else int(time.strftime("%Y%m%d"))
    # The suppression below is the right call rather than a shortcut. The sampling MUST be
    # reproducible: baseline section 2.1 requires two runs on the same input to agree, and the
    # seed is recorded in the ledger so a night's sample can be re-drawn exactly. A
    # cryptographic generator would make it unseedable, which is the one property this needs.
    # Same reason as the `tests/*` entry in pyproject's per-file-ignores, and inline because
    # this file is under tools/, which that entry does not cover.
    #
    # Written this way round deliberately: a comment that BEGINS with the directive token is
    # itself parsed as a directive, which is how this line first produced an unused-suppression
    # warning. The same trap as a comment opening with the shellcheck token, met earlier today.
    rng = random.Random(seed)  # noqa: S311

    judged = [k for k in candidates if k in seen]
    fresh = [k for k in candidates if k not in seen]
    recheck_target = int(sample_size * args.recheck_fraction)

    # Oldest-judged first for the recheck cohort, so the revisit interval is bounded.
    judged.sort(key=lambda k: seen[k].get("last_seen_epoch", 0.0))
    recheck = judged[:recheck_target]

    # New ground: changed modules first, then arbitrary-but-seeded among the rest.
    fresh.sort(key=lambda k: (0 if candidates[k]["module"] in promoted else 1, k))
    head = fresh[: (sample_size - len(recheck)) * 2]
    rng.shuffle(head)
    explore = head[: sample_size - len(recheck)]

    # If there is no new ground left, spend the whole budget rechecking rather than idling.
    if len(recheck) + len(explore) < sample_size:
        spare = [k for k in judged[recheck_target:]
                 if k not in set(recheck)][: sample_size - len(recheck) - len(explore)]
        recheck += spare

    chosen = recheck + explore
    rng.shuffle(chosen)

    print()
    print(f"mutants available : {len(candidates)}")
    print(f"budget            : {args.budget_seconds}s at {rate:.2f}/s = {affordable} affordable")
    print(f"sample size       : {len(chosen)}  (cap {args.max_sample})")
    print(f"seed              : {seed}")
    print(f"modules promoted  : {len(promoted)}")
    print(f"coverage of tree  : {100 * len(chosen) / len(candidates):.1f}% of mutants this run")
    print()
    print("THIS IS A SAMPLE, NOT A SCORE. The figure below describes the mutants drawn this")
    print("run and nothing else. It is not this project's mutation score, it may not be")
    print("published as one, and it is not comparable with another project's full-tree figure.")

    if args.enumerate_only:
        print("\n--enumerate-only: nothing was executed, so there is no verdict to report.")
        return 0

    # Running mutmut is deliberately a separate, explicit step rather than an import, because it
    # rewrites a copy of the tree into `mutants/` and forks; wrapping that in-process would make
    # a crash here indistinguishable from a crash in the suite.
    names = [candidates[k]["mutant"] for k in chosen]
    listing = ROOT / "mutants-to-run.txt"
    listing.write_text("\n".join(names) + "\n", encoding="utf-8")
    deadline = started + args.budget_seconds
    remaining = max(1, int(deadline - time.time()))
    proc = subprocess.run(
        ["mutmut", "run", "--max-children", str(args.max_children), *names],
        cwd=ROOT, capture_output=True, text=True, timeout=remaining, check=False,
    )
    phase_seconds = time.time() - started

    # The per-mutant verdicts ARE the aggregate: counting them here rather than also reading
    # mutmut's `export-cicd-stats` output keeps one mechanism behind one fact. The first draft
    # read both and the two could have disagreed, which is this project's most repeated defect.
    verdicts, hashes, durations = read_meta(set(chosen))
    if not verdicts:
        print(f"DID NOT RUN: mutmut exited {proc.returncode} and no per-mutant verdict matched "
              f"the sample, so nothing was measured and the ratchet had nothing to compare. "
              f"Last stderr line: "
              f"{(proc.stderr.strip().splitlines() or ['(none)'])[-1]}", file=sys.stderr)
        return 2

    stats = {
        "killed": sum(1 for v in verdicts.values() if v == KILLED),
        "survived": sum(1 for v in verdicts.values() if v == SURVIVED),
        "no_tests": sum(1 for v in verdicts.values() if v == NO_TESTS),
        "unknown": sum(1 for v in verdicts.values() if v.startswith("unknown-exit-")),
    }
    attempted = stats["killed"] + stats["survived"]

    # THE COUNT ASSERTION, and it is the most important line in this file.
    #
    # It exists because this job had the exact defect it is built to find. While testing the
    # ratchet, a deliberately weakened test file was left syntactically invalid; mutmut collected
    # no tests, printed "Failed to collect list of tests" and exited 1. This tool then read the
    # 13 stale verdicts left in the `.meta` from the PREVIOUS run, found no regression among
    # them, printed "RATCHET HELD" and exited 0. A run that evaluated nothing reported as clean,
    # which is this repository's single most repeated failure and the reason every gate here has
    # three outcomes.
    #
    # `exit_code_by_key` carries a null for a mutant that has not been run, so a stale `.meta`
    # is indistinguishable from a fresh one by content alone. The only honest check is whether
    # the number of mutants that actually reached a verdict matches the number sampled.
    evaluated = attempted + stats["no_tests"]
    floor = completeness_floor(len(chosen))
    if evaluated < floor:
        print(
            f"DID NOT RUN: {evaluated} of {len(chosen)} sampled mutants reached a verdict, "
            f"below the floor of {floor}. mutmut exited {proc.returncode}. "
            f"{stats['unknown']} mutant(s) carry no exit code, which means they were never run "
            f"and any verdict read for them is left over from an earlier run. Nothing is "
            f"reported and the ratchet is NOT evaluated, because a ratchet over a stale subset "
            f"would say 'held' for the wrong reason.\n"
            f"Last stderr line: {(proc.stderr.strip().splitlines() or ['(none)'])[-1]}",
            file=sys.stderr)
        return 2
    if stats["unknown"]:
        print(f"warning: {stats['unknown']} mutant(s) came back with an exit code this tool "
              f"does not recognise or were never run. They are counted in neither killed nor "
              f"survived, and they are excluded from the ledger.", file=sys.stderr)
        for key in [k for k, v in verdicts.items() if v.startswith("unknown-exit-")]:
            verdicts.pop(key)
    measured_rate = attempted / phase_seconds if phase_seconds > 0 else 0.0
    lo, hi = wilson_interval(stats["killed"], attempted)

    print()
    print(f"attempted   : {attempted} of {len(chosen)} sampled "
          f"({stats.get('no_tests', 0)} had no covering test: DID NOT RUN, not survived)")
    print(f"killed      : {stats['killed']}")
    print(f"survived    : {stats['survived']}")
    print(f"elapsed     : {phase_seconds:.0f}s of {args.budget_seconds}s budget")
    print(f"measured rate: {measured_rate:.2f} mutants/s  (recorded for the next run)")
    if attempted:
        print(f"kill rate on THIS SAMPLE: {100 * stats['killed'] / attempted:.1f}% "
              f"(95% interval {100 * lo:.1f}% to {100 * hi:.1f}%)")
    print()
    print("Again: a sample, not a score.")

    # ── the ratchet, and the ledger it compares against ─────────────────────────────────────
    prior_known = dict(ledger.get("mutants", {}))
    # How many of the mutants judged this run had a prior verdict to be compared against. If it
    # is zero the ratchet did not run, whatever it would otherwise have reported.
    comparable = sum(1 for key in verdicts if key in prior_known)
    regressions = ratchet(ledger, verdicts, hashes)

    now = time.time()
    known = ledger.setdefault("mutants", {})
    discoveries = []
    for key, verdict in verdicts.items():
        prior = known.get(key)
        if verdict == SURVIVED and (prior is None or prior.get("verdict") != SURVIVED):
            discoveries.append(key)
        known[key] = {
            "verdict": verdict,
            "module": candidates[key]["module"] if key in candidates else "",
            "function_hash": hashes.get(key, ""),
            "last_seen": time.strftime("%Y-%m-%d", time.gmtime(now)),
            "last_seen_epoch": now,
            "seconds": round(durations.get(key, 0.0), 4),
        }
    ledger.setdefault("runs", []).append({
        "date": time.strftime("%Y-%m-%d", time.gmtime(now)),
        "seed": seed,
        "sampled": len(chosen),
        "attempted": attempted,
        "killed": stats["killed"],
        "survived": stats["survived"],
        "no_tests": stats.get("no_tests", 0),
        "seconds": round(phase_seconds, 1),
        "rate": round(measured_rate, 3),
        "mutants_available": len(candidates),
        "regressions": len(regressions),
        "discoveries": len(discoveries),
    })
    save_ledger(ledger)

    print()
    print(f"ledger            : {LEDGER.relative_to(ROOT)}")
    print(f"known mutants     : {len(known)} of {len(candidates)} "
          f"({100 * len(known) / len(candidates):.1f}% of the tree ever examined)")
    print(f"new survivors     : {len(discoveries)}  (discoveries, which do NOT fail this job)")
    for key in discoveries[:20]:
        print(f"    + {key}")
    if len(discoveries) > 20:
        print(f"    ... and {len(discoveries) - 20} more, all in the ledger")

    if regressions:
        print()
        print(f"RATCHET FAILED: {len(regressions)} mutant(s) that were killed now survive, with "
              f"their function unchanged.")
        for line in regressions:
            print(f"  - {line}")
        print()
        print("This cannot be a sampling artefact. It compares named mutants against the same "
              "code, so the only thing that produces it is a test that stopped noticing.")
        return 1

    # A ratchet with nothing to compare against has not held; it has not run. This matters more
    # than it looks, because `private/` is excluded from this repository's remotes, so a CI
    # clone has no ledger and the file is carried between runs in a cache. A cache can be
    # evicted, and an evicted cache would otherwise read as a clean night, for ever, silently.
    if not comparable:
        print()
        print(f"RATCHET NOT EVALUATED: the ledger held {len(prior_known)} prior verdict(s) and "
              f"none of them covered a mutant in this sample, so no like-for-like comparison "
              f"was possible. This is NOT a pass. If this is the first ever run that is "
              f"expected; if it is not, the ledger has been lost and needs restoring, because "
              f"an empty ledger cannot fail.")
        return 0 if args.allow_first_run else 2

    print()
    print(f"RATCHET HELD: none of the {comparable} previously-judged mutant(s) in this sample "
          f"regressed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
