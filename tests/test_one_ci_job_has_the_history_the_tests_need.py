# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Nine guards skip without git history, and one workflow line is what stops them skipping always.

WHAT PROMPTED IT, 2026-09-27

Nine test files in this suite skip when the clone has no usable history, and they are right to: with
`fetch-depth: 1` the single fetched commit appears to introduce the entire tree, so
`git log -1 -- <path>` returns the tip's date for every file. A check that compared a recorded date
against that would fail every day after it was written while looking like a licence finding.

`test_package_imports.py` documents the arrangement in its own comment: "One CI job checks out full
history so this runs somewhere real; everywhere else it skips and says why."

**That sentence is an assumption about a workflow file, and nothing checked it.** The main `test`
job, which is the whole OS and Python matrix, uses the default shallow checkout. Exactly one job
that runs the suite passes `fetch-depth: 0`. Delete that one line and all nine guards skip in every
job, forever, silently, while the comment explaining the arrangement still reads as true.

This is the shape this project keeps finding in itself, and this time it is one layer up from the
code: a correct diagnostic wired to nothing. A skip is not a pass, but a skip everywhere is
indistinguishable from a pass everywhere in a summary line, and the whole point of a guard is the
day it fires.

WHAT THIS DOES NOT ASSERT

It does not demand full history on every job. Fetching it is slow and most of the suite has no use
for it, so one job is the right design. It asserts only that the count is at least one, and names
which job is currently carrying it so the failure message tells whoever broke it what they broke.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"

#: A job needs history only if it RUNS the suite. Matched on the command rather than on a job name,
#: because a renamed job would otherwise drop out of this check silently.
_RUNS_PYTEST = re.compile(r"\bpytest\b")


def _jobs():
    """Every job in ci.yml, with whether it runs pytest and what checkout depth it asks for.

    Parsed by indentation rather than with a YAML library, deliberately: `yaml` is present here
    through transformers rather than by declaration, and a test that silently skipped when it was
    absent would be the very failure this file is about.
    """
    if not CI.is_file():
        pytest.skip(f"{CI} is not in this checkout")
    jobs, name = {}, None
    for line in CI.read_text(encoding="utf-8").splitlines():
        top = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if top:
            name = top.group(1)
            jobs[name] = {"pytest": False, "checkout": False, "full_history": False}
            continue
        if name is None:
            continue
        if _RUNS_PYTEST.search(line):
            jobs[name]["pytest"] = True
        if "actions/checkout" in line:
            jobs[name]["checkout"] = True
        if "fetch-depth" in line:
            depth = line.split("fetch-depth", 1)[1].lstrip(": ").strip()
            # 0 means all history. Any other value is a depth, and a depth is a shallow clone.
            if depth == "0":
                jobs[name]["full_history"] = True
    return jobs


def test_the_workflow_declares_jobs_at_all():
    """If the parse returns nothing, every assertion below would pass by checking nothing."""
    jobs = _jobs()
    assert jobs, f"no jobs parsed out of {CI.name}; the assertions below would be vacuous"
    assert any(d["checkout"] for d in jobs.values()), "no job checks the repository out at all"


def test_at_least_one_job_that_runs_the_suite_fetches_full_history():
    """The one line that keeps nine history-dependent guards from skipping everywhere."""
    jobs = _jobs()
    running = {n: d for n, d in jobs.items() if d["pytest"]}
    assert running, (
        "no job in ci.yml appears to run pytest, so either the suite is not run in CI or this "
        "test's detection is wrong. Both are worth knowing.")

    with_history = sorted(n for n, d in running.items() if d["full_history"])
    assert with_history, (
        "NO job that runs the suite checks out full history (`fetch-depth: 0`).\n"
        "  Nine test files skip when history is missing, so all nine now skip in every job, and a "
        "skip everywhere reads like a pass everywhere in the summary.\n"
        f"  Jobs that run the suite: {sorted(running)}.\n"
        "  Restore `fetch-depth: 0` on one of them. It does not need to be on all of them; "
        "fetching history is slow and most of the suite has no use for it.")


def test_the_history_dependent_guards_still_exist():
    """If they have all gone, the workflow line above is now cost with no benefit.

    Asserted so the two halves are maintained together: a guard whose reason has disappeared should
    be noticed, rather than leaving a slow checkout in place for nothing.
    """
    wants_history = re.compile(r"skip\(.{0,200}?(shallow|git history|git checkout)",
                               re.IGNORECASE | re.DOTALL)
    needs_history = sorted(
        p.name for p in (ROOT / "tests").glob("test_*.py")
        if wants_history.search(p.read_text(encoding="utf-8"))
    )
    assert needs_history, (
        "no test skips for want of git history any more. If that is deliberate, the "
        "`fetch-depth: 0` this file protects is now buying nothing and can go, and so can this "
        "file.")


def test_the_shallow_matrix_job_is_the_expected_arrangement():
    """Records the design so a change to it is a decision rather than a surprise.

    The matrix job is deliberately shallow. This asserts that at least one job that runs the suite
    is NOT fetching history, so that if somebody "fixes" this by turning it on everywhere, the cost
    of that choice is visible here rather than only in the billing.
    """
    jobs = _jobs()
    running = {n: d for n, d in jobs.items() if d["pytest"]}
    shallow = sorted(n for n, d in running.items() if not d["full_history"])
    if not shallow:
        pytest.skip("every job that runs the suite now fetches full history; slower, not wrong")
    assert shallow
