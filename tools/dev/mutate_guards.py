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

EVERY ROW IS MEASURED TWICE, and that is the part worth knowing. The named test is run once with
the tree as it is, to establish that it passes and how many tests that takes, and once with the
mutation in place. A guard NOTICED only if the second run failed having run the same tests. Anything
else is a BROKEN row rather than a verdict: a stale path, a file that collects nothing, a test that
was already red, a mutation that broke collection instead of a test.

That distinction is the whole reason this file changed on 2026-09-28. It read `returncode != 0` as
"the guard noticed", and pytest exits 4 for a path that matches nothing and 5 for a file that
collects nothing, so a renamed test printed NOTICED and the run ended with "all guards noticed"
having executed no test at all. The tool written to answer "would anything fail if this were
deleted" was answering it with a run that never happened, which is the same defect it exists to
find, one level up.

IT RESTORES EVERY FILE IT TOUCHES, in a `finally`, including when a test run times out or the
process is interrupted. `finally` does not run on a SIGKILL, so it also drops a
`.mutate-in-progress.json` naming the file and holding its original text before editing anything,
and removes it afterwards. A leftover one means a mutation is still in place and says what to put
back. Run it on a clean tree anyway.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

#: Example table. Replace it with your own, or pass one in with `--cases`. Each row is
#: (description, path relative to the repo root, text to find, text to put there, test to run).
CASES: list[tuple[str, str, str, str, str]] = [
    ("say.shorten marks a cut value", "src/senbonzakura/say.py",
     "    return text[:limit - len(CUT)] + CUT", "    return text[:limit]",
     "tests/test_say_wraps_prose_and_nothing_else.py"),
]


#: pytest's own exit codes. 1 is the only one that means "tests ran and something failed", which is
#: the only one that is evidence about a guard. The rest are facts about the invocation.
PYTEST_PASSED, PYTEST_FAILED = 0, 1
PYTEST_NOT_A_RUN = {2: "interrupted", 3: "internal error", 4: "usage error, e.g. no such path",
                    5: "no tests collected"}

#: `-q` prints e.g. "3 passed in 0.10s" or "1 failed, 2 passed in 0.11s". The count this cares about
#: is how many tests RAN, so every outcome word is summed.
_TALLY = re.compile(r"(\d+) (passed|failed|xfailed|xpassed|error|errors)\b")

#: Written before a source file is edited and removed after it is put back, because `finally` does
#: not run on a SIGKILL and this project lost half an hour to a machine level OOM the day this tool
#: was written. A leftover file names what to restore instead of leaving `git status` to explain it.
BREADCRUMB = ".mutate-in-progress.json"


def collected(output):
    """How many tests actually ran, from pytest's summary line, or 0 if it never said.

    THE WHOLE POINT OF THIS FUNCTION. The harness used to read `returncode != 0` as "the guard
    noticed". pytest exits 4 for a path that matches nothing and 5 for a file that collects nothing,
    so a renamed test file printed NOTICED and the run finished with "all guards noticed" having
    executed no test at all. That is the failure this tool was built the same week to prevent, and
    it was inside the tool.
    """
    return sum(int(n) for n, _word in _TALLY.findall(output or ""))


def _pytest(test, root, timeout, *, cache=None):
    """Run one test path. Returns (returncode, how many tests ran, the output).

    A FRESH BYTECODE CACHE PER RUN, and it is not housekeeping. CPython decides a `.pyc` is current
    from the source's size and mtime, so a mutation that changes neither can be skipped entirely and
    the second run executes the ORIGINAL code from cache. Every interesting mutation of an operator
    is that shape: `>=` to `<=`, `and` to `or`, `min` to `max`. The row then prints MISSED, somebody
    goes looking for a guard that already exists, and the real answer was that the code never
    changed. Found while writing this tool's own tests, where the two writes fall inside one second.
    """
    env = dict(os.environ)
    if cache is not None:
        env["PYTHONPYCACHEPREFIX"] = str(cache)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test, "-q", "-p", "no:cacheprovider", "--no-cov", "-x"],
        cwd=root, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    out = (result.stdout or "") + (result.stderr or "")
    return result.returncode, collected(out), out


def run(cases, root, timeout):
    """Mutate, test, restore. Returns the rows nothing noticed, and the rows that proved nothing.

    Each case is measured TWICE: once with the tree as it is, to establish that the named test
    passes and how many tests that takes, and once mutated. A guard noticed only if the second run
    failed while running the same number of tests as the first. Anything else is a broken case
    rather than a verdict, and it is reported as one, because a harness that cannot tell "your fix
    is unguarded" from "your test path is wrong" answers neither question.
    """
    missed, broken = [], []
    for case in cases:
        verdict, detail = _one_case(case, root, timeout)
        if verdict == "broken":
            broken.append(detail)
        elif verdict == "missed":
            missed.append(detail)
    return missed, broken


def _one_case(case, root, timeout):
    """Measure one row. Returns ("noticed" | "missed" | "broken", the name or the reason)."""
    name, path, old, new, test = case
    target = root / path
    original = target.read_text(encoding="utf-8")
    if old not in original:
        print(f"BROKEN   {name}: the text to mutate is not in {path}")
        return "broken", f"{name} (text not found: the case has gone stale)"

    with tempfile.TemporaryDirectory(prefix="mutate-pyc-") as caches:
        cache = pathlib.Path(caches)
        try:
            code, before, _out = _pytest(test, root, timeout, cache=cache / "before")
        except subprocess.TimeoutExpired:
            print(f"BROKEN   {name}: {test} did not finish in {timeout}s unmutated")
            return "broken", f"{name} (the test hangs before anything was changed)"
        if code in PYTEST_NOT_A_RUN or before == 0:
            why = PYTEST_NOT_A_RUN.get(code, "the summary line reported no tests")
            print(f"BROKEN   {name}: {test} is not a run ({why})")
            return "broken", f"{name} ({test}: {why})"
        if code != PYTEST_PASSED:
            print(f"BROKEN   {name}: {test} already fails before anything was mutated")
            return "broken", f"{name} ({test} was not green to begin with)"

        crumb = root / BREADCRUMB
        crumb.write_text(json.dumps({"path": path, "case": name, "original": original}),
                         encoding="utf-8")
        try:
            target.write_text(original.replace(old, new, 1), encoding="utf-8")
            try:
                code, after, _out = _pytest(test, root, timeout, cache=cache / "after")
            except subprocess.TimeoutExpired:
                # A hang is not a pass, and it is not evidence either.
                print(f"BROKEN   {name}: {test} did not finish in {timeout}s while mutated")
                return "broken", f"{name} (the mutated test hangs)"
        finally:
            target.write_text(original, encoding="utf-8")
            crumb.unlink(missing_ok=True)

    if code in PYTEST_NOT_A_RUN:
        print(f"BROKEN   {name}: the mutation stopped {test} running ({PYTEST_NOT_A_RUN[code]})")
        return "broken", f"{name} (the mutation broke collection rather than a test)"
    # `-x` stops at the first failure, so FEWER tests in the mutated run is ordinary. MORE means the
    # mutation changed what gets collected, and then the two runs are not of the same thing.
    if after > before:
        print(f"BROKEN   {name}: {test} ran {after} tests mutated and {before} before")
        return "broken", f"{name} (the mutation changed what was collected)"

    noticed = code == PYTEST_FAILED
    print(f"{'NOTICED ' if noticed else 'MISSED  '} {name}  ({test}, {before} test(s))")
    return ("noticed", name) if noticed else ("missed", name)


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

    missed, broken = run(cases, root, a.timeout)
    print()
    if broken:
        # SEPARATE EXIT STATUS, because these are different sentences. A missed row says the code
        # is unguarded; a broken row says this tool learned nothing about that row, and reporting
        # the second as the first sends somebody to write a test that already exists.
        print("THESE ROWS MEASURED NOTHING, so they are neither a pass nor a finding:",
              file=sys.stderr)
        for name in broken:
            print(f"  {name}", file=sys.stderr)
    if missed:
        print("THESE FIXES COULD BE DELETED AND NOTHING WOULD FAIL:", file=sys.stderr)
        for name in missed:
            print(f"  {name}", file=sys.stderr)
    if missed:
        return 1
    if broken:
        return 3
    print(f"all {len(cases)} guards noticed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
