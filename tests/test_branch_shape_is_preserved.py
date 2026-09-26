# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""No job may write to `main`, because `main` is only ever a fast-forward of `dev`.

WHY THIS IS A TEST AND NOT A COMMENT

The branch workflow is two branches: work lands on `dev`, and `main` moves forward only at a
release, as a fast-forward from `dev`. That invariant is what makes a promotion safe to do and
cheap to verify, and it is held by nothing except everybody remembering it.

WHAT PROMPTED IT, measured 2026-09-26

The tests-badge job pushes a one-line README correction with `git push origin
"HEAD:${GITHUB_REF_NAME}"`. On a push to `dev` that is what it was built for. On a push to `main`,
which is what a promotion is, it committed to `main` alone, and two invariants broke in the same
second:

  * `main` gained a commit that `dev` did not have, so the NEXT promotion was no longer a
    fast-forward. The only way to make it one again is a force-push to a shared branch, which the
    workflow forbids, so the fault compounds rather than clears.
  * It pushes to `origin` and not to the private remote, so the two remotes' `main` branches held
    different trees while both looked promoted.

The job was already careful about the thing people expect to guard: it refuses to commit a diff
that touches anything but the badge number, and it refuses a fork's pull request. It said nothing
about WHICH BRANCH it writes to, which is the same shape this project keeps finding: **a guard that
covers one spelling of a defect reports clean on the others.**

This file asks the branch question of every job in every workflow, so a new job that writes to a
branch has to answer it too.
"""
import re
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).resolve().parent.parent / ".github" / "workflows").glob("*.yml"))

#: A push of a ref to a remote, as written in a run step. Matches `git push origin HEAD:main`,
#: `git push origin "HEAD:${GITHUB_REF_NAME}"` and `git push olympus dev`.
_PUSHES = re.compile(r"git\s+push\s+\S+\s+[\"']?\S*[\"']?", re.MULTILINE)

#: The release workflow legitimately writes tags and release assets rather than branches. Recorded
#: by filename so that adding a workflow does not quietly inherit an exemption.
_MAY_WRITE_OUTSIDE_DEV = {"publish.yml", "release.yml"}


def test_there_are_workflows_to_check():
    """An empty glob passes every assertion below while checking nothing."""
    assert WORKFLOWS, "no workflow files were found, so this file measures nothing"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_job_that_pushes_a_branch_is_confined_to_dev(path):
    """Any workflow that pushes a branch says, in its own condition, that it means `dev`.

    The check is deliberately coarse: it asks whether a workflow containing a branch push also
    contains the ref condition. A workflow can defeat that by structuring itself unusually, and
    that is acceptable; the defect this exists for was a job nobody had asked the question of at
    all, not a job that lied about the answer.
    """
    text = path.read_text(encoding="utf-8")
    pushes = [m.group(0) for m in _PUSHES.finditer(text)]
    # A tag push is not a branch push and does not touch the promotion path.
    branch_pushes = [p for p in pushes if "refs/tags" not in p and "--tags" not in p]
    if not branch_pushes or path.name in _MAY_WRITE_OUTSIDE_DEV:
        pytest.skip(f"{path.name} pushes no branch")

    assert "github.ref_name == 'dev'" in text, (
        f"{path.name} pushes a branch ({branch_pushes}) and never states that it means `dev` "
        f"only. On a push to `main`, which is what a promotion is, this writes a commit to `main` "
        f"that `dev` does not have, and the next promotion stops being a fast-forward. Add "
        f"`github.ref_name == 'dev'` to the job's `if:`.")


def test_the_badge_job_names_dev_specifically():
    """The job the rule was written for, pinned by name so a rename cannot drop it silently."""
    ci = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml")
    text = ci.read_text(encoding="utf-8")
    assert "tests-badge:" in text, (
        "the tests-badge job has been renamed or removed; if it still pushes a branch, keep the "
        "`dev` confinement with it")
    after = text.split("tests-badge:", 1)[1]
    # The condition sits in the job's own block, above the first step.
    condition_block = after.split("steps:", 1)[0]
    assert "github.ref_name == 'dev'" in condition_block, (
        "tests-badge holds a write token and pushes the branch it was triggered on. Without the "
        "`dev` condition, promoting `main` leaves a badge commit on `main` alone.")
