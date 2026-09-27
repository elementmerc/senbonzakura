#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Break a fix on purpose and check that its test notices.

WHY THIS EXISTS, 2026-09-27

A self-review pass over a cluster of twenty-eight fixes asked one question of each: if somebody
deleted this, would anything fail? Twenty-one were checkable mechanically and **six were not
guarded at all**.

Five of the six were fixes wired to call sites nobody tested. `say.some_of` had thorough tests of its
own and nothing asserted that `doctor`, `convert` or `dataset` actually called it, so putting the bare
slices back left the whole suite green: `doctor` could go back to reporting "7 failed to import" and
naming six of them with nothing complaining. The sixth was worse. Deleting the two lines in
`margin.main` that print the compass reading in words broke nothing, because every test for that
feature exercised the function directly. A correct function wired to nothing is the defect this
project keeps finding in its own work: the null-panel diagnostic was computed, recorded, printed and
gated on nothing, and `_build.py` was generated, shipped and read by nothing.

A passing suite says the tests agree with the code. It does not say the tests would notice the code
going away, and those are different questions. This asks the second one.

HOW TO USE IT

Write a CASES table for the cluster you just finished: a name, the file, the exact text to replace,
what to replace it with, and the test that should fail. Then run it. Every row should print NOTICED.
A MISSED row is a fix that could be deleted silently.

    python tools/dev/mutate_guards.py --cases my_cluster.py

The table is deliberately hand-written per cluster rather than generated. A generated mutation is
usually either trivial or nonsense; the useful mutation is the one that expresses "somebody undid
this particular decision", and only the person who made the decision knows what that looks like.

IT RESTORES EVERY FILE IT TOUCHES, in a `finally`, including when a test run times out or the
process is interrupted. Run it on a clean tree anyway: an interrupted mutation leaves a source file
edited, and `git status` is the check that costs nothing.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

#: Example table. Replace it with your own, or pass one in with `--cases`. Each row is
#: (description, path relative to the repo root, text to find, text to put there, test to run).
CASES: list[tuple[str, str, str, str, str]] = [
    ("say.shorten marks a cut value", "src/senbonzakura/say.py",
     "    return text[:limit - len(CUT)] + CUT", "    return text[:limit]",
     "tests/test_say_wraps_prose_and_nothing_else.py"),
]


def run(cases, root, timeout):
    """Mutate, test, restore. Returns the rows nothing noticed."""
    missed = []
    for name, path, old, new, test in cases:
        target = root / path
        original = target.read_text(encoding="utf-8")
        if old not in original:
            print(f"SKIPPED  {name}: the text to mutate is not in {path}")
            missed.append(f"{name} (text not found: the case has gone stale)")
            continue
        target.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test, "-q", "-p", "no:cacheprovider",
                 "--no-cov", "-x"],
                cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
            noticed = result.returncode != 0
        except subprocess.TimeoutExpired:
            # A hang is not a pass. Reported as missed so somebody looks at it.
            noticed = False
            print(f"TIMEOUT  {name}: {test} did not finish in {timeout}s")
        finally:
            target.write_text(original, encoding="utf-8")
        print(f"{'NOTICED ' if noticed else 'MISSED  '} {name}  ({test})")
        if not noticed:
            missed.append(name)
    return missed


def main(argv=None):
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        prog="mutate_guards",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="repo root to mutate inside (default: this checkout)")
    ap.add_argument("--cases", default=None,
                    help="a python file defining CASES, for a cluster other than the example")
    ap.add_argument("--timeout", type=int, default=600,
                    help="seconds allowed per test run (default: 600)")
    a = ap.parse_args(argv)

    root = pathlib.Path(a.root) if a.root else pathlib.Path(__file__).resolve().parent.parent.parent
    cases = CASES
    if a.cases:
        namespace: dict = {}
        exec(compile(pathlib.Path(a.cases).read_text(encoding="utf-8"), a.cases, "exec"),  # noqa: S102
             namespace)
        cases = namespace["CASES"]

    if not cases:
        print("no cases given, so nothing was checked. That is not a pass.", file=sys.stderr)
        return 2

    missed = run(cases, root, a.timeout)
    print()
    if missed:
        print("THESE FIXES COULD BE DELETED AND NOTHING WOULD FAIL:", file=sys.stderr)
        for name in missed:
            print(f"  {name}", file=sys.stderr)
        return 1
    print(f"all {len(cases)} guards noticed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
