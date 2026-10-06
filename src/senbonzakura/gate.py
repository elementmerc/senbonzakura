# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura gate`: fail a build when a measured property moved outside its interval.

This is the caller `baseline.py` was written for. The module holds the rules; this holds the
command, the exit status, and the words a reader meets when it refuses.

A MEASUREMENT AND A BASELINE ARE THE SAME SHAPE, deliberately. Both are `baseline.record`
artefacts, and a baseline is simply a measurement somebody decided to keep. Inventing a second
format for "the new number" would have meant two schemas that must agree about what a measurement
is, and the first thing that drifts between two schemas is the part neither side checks.

WHY THIS RUNS WITHOUT TORCH. The gate's whole value is that it runs on every change rather than
when somebody wonders, so it has to be affordable on whatever a CI runner has. It reads two JSON
files and does arithmetic: no model, no corpus, no network. `tests/test_gate.py` pins that.

EXIT STATUS IS THE PRODUCT HERE, not the printed text, because a gate is read by a build system
before it is read by a person:

    0  the property did not move outside its interval, or moved in the better direction
    1  it regressed
    2  the comparison was refused, because the two measurements were never comparable

Two and one are distinct on purpose. A refusal is not a regression: nothing has been shown about
the model, and a build that treats "I could not compare these" as "this got worse" teaches its
reader to ignore the difference.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

from . import argresolve, baseline
from .crashsafe import atomic_write

#: Exit statuses, named so the tests and the docs cannot drift from the code.
OK = 0
REGRESSED = 1
REFUSED = 2

#: What a written run record declares itself to be, so a reader of the directory knows.
RUN_SCHEMA = "senbonzakura-gate-run/1"

#: What each verdict is called in a written record, so the file reads the way the log does.
VERDICT_NAMES = {OK: "OK", REGRESSED: "REGRESSED", REFUSED: "REFUSED"}


def build_parser():
    p = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog="senbonzakura gate",
        description="Compare a measurement against a recorded baseline and fail on a regression.")
    p.add_argument("--baseline", required=True,
                   help="the recorded measurement to judge against, written by an earlier run")
    # `--current`, NOT `--measurement`, and the old name still works.
    #
    # `senbonzakura baseline --measurement <file>` takes a RESULT ARTEFACT, straight out of a run.
    # This command's `--measurement` took a BASELINE-SHAPED file, which is what `baseline` writes.
    # One flag name, two incompatible meanings, in two commands people use together, and `baseline`
    # printed `gate --baseline ... --measurement <new>` on success, so following the hint exited 2 on
    # a schema error. A reader met exactly that on 2026-09-26 and could not tell which of the two
    # files was wrong.
    #
    # Operator decision 2026-09-27: rename here and keep the old spelling as an alias, so nothing
    # anybody has scripted stops working. `--current` rather than `--against`, which was the name
    # first proposed and reads like the thing being compared TO, which is the baseline.
    p.add_argument("--current", "--measurement", dest="current", required=True,
                   help="THIS run's measurement, recorded in baseline shape. That means run "
                        "`senbonzakura baseline` on the new run's result artefact first: this gate "
                        "compares two recorded measurements, not a recorded one against a raw "
                        "artefact. `--measurement` is accepted as the old name for this flag.")
    p.add_argument("--quiet", action="store_true",
                   help="print only the verdict line, for a build log that already has enough in "
                        "it. The refusal path ignores this and always says why")
    p.add_argument("--history", metavar="DIR",
                   help="record this run, pass or fail, under DIR. A regression caught in March "
                        "is only evidence in June if March wrote something down. Records are "
                        "named after their own contents, so re-running the same comparison "
                        "records it once and overwrites nothing")
    p.add_argument("--store", metavar="DIR",
                   help="where measurements recorded by address live, so --baseline and --current "
                        "may be given as `sha256:...` addresses instead of paths")
    return p


def _now():
    """One timestamp per run, captured once and reused, per baseline section 2.1."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_history(directory, *, recorded, current, status, headline, detail, at, log):
    """Write what this run compared and what it concluded. Return the path, or None.

    THE RECORD IS NAMED AFTER THE COMPARISON AND NOT AFTER THE CLOCK. Two runs of the same
    comparison are the same evidence, so they land on the same file and the first one's recording
    stands; two different comparisons can never collide. That makes an interrupted job safe to
    re-run, which is the property a history kept by a build system needs most, and it means a
    directory of these can be merged from several runners without anybody choosing a winner.

    A FAILURE TO WRITE IS REPORTED AND DOES NOT CHANGE THE VERDICT. The gate's answer is about
    the model; whether this machine could write a file is about the machine, and swallowing one
    inside the other in either direction would be wrong. A full disk must not turn a regression
    into a pass, and it must not turn a pass into a regression either.
    """
    body = {
        "schema": RUN_SCHEMA,
        "baseline": recorded.get(baseline.ADDRESS_FIELD) or baseline.address(recorded),
        "current": current.get(baseline.ADDRESS_FIELD) or baseline.address(current),
        "metric": recorded.get("metric"),
        "partition": recorded.get("partition"),
        "status": status,
        "verdict": VERDICT_NAMES.get(status, f"unknown status {status}"),
        "headline": headline,
        "detail": detail,
        # THE WHOLE MEASUREMENT, not a copy of its number. A history holding the figure and not
        # the conditions it was taken under is a history nobody can re-judge, and re-judging is
        # what somebody disputing a verdict in June will want to do.
        "measurement": current,
        "recorded_at": at,
    }
    canonical = json.dumps({k: v for k, v in body.items() if k != "recorded_at"},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = Path(directory) / digest[:2] / digest[2:4] / f"{digest}.json"
    try:
        if path.exists():
            log(f"  this comparison was already recorded at {path}")
            return path
        with atomic_write(path) as f:
            json.dump(body, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as e:
        log(f"  WARNING: the verdict above is correct and could NOT be recorded under "
            f"{directory}: {type(e).__name__}: {e}")
        log("  Nothing was written, so this run leaves no evidence behind. The exit status is "
            "still the verdict.")
        return None
    log(f"  recorded at {path}")
    return path


def run(argv=None, log=print):
    """Compare, and return an exit status. Never raises on an ordinary refusal.

    A refusal is an expected outcome of this command rather than a crash: the two measurements
    were not comparable, which is information, and it is reported the same way a regression is.
    """
    a = build_parser().parse_args(argv)
    at = _now()

    def record_run(status, headline, detail):
        """Write the history entry, when one was asked for and there is something to write."""
        if a.history:
            _write_history(a.history, recorded=recorded, current=current, status=status,
                           headline=headline, detail=detail, at=at, log=log)

    try:
        recorded = baseline.read(baseline.resolve(a.baseline, a.store))
        current = baseline.read(baseline.resolve(a.current, a.store))
    except baseline.BaselineError as e:
        # NOTHING IS RECORDED HERE, AND THAT IS NOT AN OVERSIGHT. One of the two files could not
        # be read, so there is no measurement to write down and no comparison to describe. A
        # history entry saying "a run happened and we cannot say what it compared" is a row that
        # looks like evidence and is not.
        log(f"gate REFUSED: {e}")
        if a.history:
            log("  nothing was recorded: there is no measurement here to record.")
        return REFUSED

    try:
        baseline.refuse_if_incomparable(recorded, current)
        # AND SEPARATELY, whether this run was precise enough for an overlap to mean anything.
        # Comparability asks "were these measured under the same conditions"; this asks "could
        # this measurement have seen the thing move at all". A run that could not is refused
        # rather than passed, because a pass here is the reassurance a reader takes away.
        baseline.refuse_if_too_blunt(recorded, baseline.interval_of(current))
    except baseline.BaselineError as e:
        # NOT `--quiet`-able. The one thing a reader must never have to go looking for is the
        # reason a gate declined to answer, because the alternative reading of a silent refusal
        # is that nothing was wrong.
        log(f"gate REFUSED: {e}")
        # A REFUSAL IS RECORDED, because it is a finding about the measurements rather than an
        # absence of one: six weeks later, "the gate refused every day that fortnight" is the
        # answer to why nothing was caught, and without a row for it the record reads as though
        # the gate had been passing.
        record_run(REFUSED, "refused", str(e))
        return REFUSED

    # A MALFORMED MEASUREMENT IS A REFUSAL, NOT A REGRESSION. `baseline.read` validates the
    # schema string and nothing else, so a file with the right schema and no `point` used to raise
    # KeyError out of here and exit 1, which is this command's code for "it regressed". A CI gate
    # reading exit 1 would report a regression that was never measured, and the docstring above
    # argues that conflating those two teaches its reader to ignore the difference. `verdict`
    # itself can also raise on an inverted interval, which was outside both try blocks.
    #
    # `BaselineError` IS IN THIS TUPLE, since 2026-09-25, and its absence was the same defect at
    # one remove. It subclasses `Exception`, not `ValueError`, so the two refusals `verdict` raises
    # for a determinism mismatch went straight past this handler and out of `run` as a traceback,
    # which the shell reports as exit 1. Exit 1 is this command's code for REGRESSED. So a CI job
    # upgrading `coherence` to the `deterministic=True` stamp this release introduces, against a
    # baseline recorded a fortnight earlier, would have reported a regression that was never
    # measured: exactly the conflation the docstring above argues teaches a reader to ignore the
    # difference. The comment was written about `KeyError` and the fix stopped at the exception
    # families that existed that day.
    try:
        ok, headline, detail = baseline.verdict(
            recorded, current["point"], baseline.interval_of(current))
    except baseline.BaselineError as e:
        log(f"gate REFUSED: {e}")
        record_run(REFUSED, "refused", str(e))
        return REFUSED
    except (KeyError, TypeError, ValueError, IndexError) as e:
        log(f"gate REFUSED: {a.current} is not a measurement this gate can read: "
            f"{type(e).__name__}: {e}")
        return REFUSED
    status = OK if ok else REGRESSED
    log(f"gate {'OK' if ok else 'FAIL'}: {headline}")
    if not a.quiet or not ok:
        log(f"  {detail}")
        # WHAT WAS LOOKED AT, printed beside the verdict and not only in the file.
        #
        # Loophole 5: somebody will read a pass as a claim that the model is safe, and the
        # sentence below says it is not. That sentence is useless on its own, because "one
        # measured property, on one track" does not tell a reader WHICH property or WHICH track,
        # and a reader who cannot see the slice cannot tell a gate over 200 rows of one partition
        # from a gate over everything. So the slice is named: the metric, the partition, the
        # sample size, and the seeds the baseline was taken over.
        log(f"  Checked: {recorded.get('metric')} on the {recorded.get('partition')} partition, "
            f"n={recorded.get('n')}, seeds {recorded.get('seeds')}, "
            f"estimator {recorded.get('estimator')!r} at {recorded.get('precision')}.")
        log("  Not checked: every other property, every other partition, and anything this "
            "metric does not measure.")
        # THE CAVEAT IS WORDED FOR THE VERDICT IT FOLLOWS. It used to read "a pass means ..."
        # under a FAIL as well, which is a sentence about an outcome that did not happen sitting
        # directly beneath the one that did.
        log(f"  {baseline.VERDICT_CAVEAT if ok else baseline.FAILURE_CAVEAT}")
    record_run(status, headline, detail)
    return status


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    # The guard eight modules once lacked, so `python -m senbonzakura.<module>` executed nothing
    # and exited 0 while the documentation said otherwise.
    sys.exit(main())
