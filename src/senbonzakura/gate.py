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
import argparse
import sys

from . import baseline

#: Exit statuses, named so the tests and the docs cannot drift from the code.
OK = 0
REGRESSED = 1
REFUSED = 2


def build_parser():
    p = argparse.ArgumentParser(
        prog="senbonzakura gate",
        description="Compare a measurement against a recorded baseline and fail on a regression.")
    p.add_argument("--baseline", required=True,
                   help="the recorded measurement to judge against, written by an earlier run")
    p.add_argument("--measurement", required=True,
                   help="this run's measurement, in the same shape as a baseline")
    p.add_argument("--quiet", action="store_true",
                   help="print only the verdict line, for a build log that already has enough in "
                        "it. The refusal path ignores this and always says why")
    return p


def run(argv=None, log=print):
    """Compare, and return an exit status. Never raises on an ordinary refusal.

    A refusal is an expected outcome of this command rather than a crash: the two measurements
    were not comparable, which is information, and it is reported the same way a regression is.
    """
    a = build_parser().parse_args(argv)

    try:
        recorded = baseline.read(a.baseline)
        current = baseline.read(a.measurement)
    except baseline.BaselineError as e:
        log(f"gate REFUSED: {e}")
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
        return REFUSED

    # A MALFORMED MEASUREMENT IS A REFUSAL, NOT A REGRESSION. `baseline.read` validates the
    # schema string and nothing else, so a file with the right schema and no `point` used to raise
    # KeyError out of here and exit 1, which is this command's code for "it regressed". A CI gate
    # reading exit 1 would report a regression that was never measured, and the docstring above
    # argues that conflating those two teaches its reader to ignore the difference. `verdict`
    # itself can also raise on an inverted interval, which was outside both try blocks.
    try:
        ok, headline, detail = baseline.verdict(
            recorded, current["point"], baseline.interval_of(current))
    except (KeyError, TypeError, ValueError, IndexError) as e:
        log(f"gate REFUSED: {a.measurement} is not a measurement this gate can read: "
            f"{type(e).__name__}: {e}")
        return REFUSED
    log(f"gate {'OK' if ok else 'FAIL'}: {headline}")
    if not a.quiet or not ok:
        log(f"  {detail}")
        log(f"  {baseline.VERDICT_CAVEAT}")
    return OK if ok else REGRESSED


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    # The guard eight modules once lacked, so `python -m senbonzakura.<module>` executed nothing
    # and exited 0 while the documentation said otherwise.
    sys.exit(main())
