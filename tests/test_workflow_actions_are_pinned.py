# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Every third-party action in every workflow is pinned to a commit, not to a tag.

WHY THIS IS A TEST AND NOT A COMMENT

`ci.yml` opens with a comment saying actions are pinned by SHA. A comment is a statement of
intent that the next person adding a step does not have to read. This is the same rule expressed
as something that fails.

The rule is baseline section 5. A tag is a movable label: whoever controls the action's
repository can repoint `v7` at different code tomorrow, and that code runs inside this
workflow with this workflow's token. A commit SHA names content that cannot change underneath
you.

WHAT PROMPTED IT

Adding `codecov/codecov-action` to publish the coverage figure. Codecov is the textbook case
for this rule rather than an exception to it: in 2021 its bash uploader was modified to
exfiltrate environment variables from everyone who curled it, and everyone who curled it was
following the documented instructions at the time. Pinning is the mitigation that would have
bounded that, so the pin gets a gate the same day the dependency arrives.

WHAT IT DOES NOT DO

It does not check that the SHA exists, or that it belongs to the tag in the trailing comment.
That needs the network on every run. What it guarantees is the property that was actually
at stake: nothing in this repository's CI resolves an action through a name someone else can
move.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

#: `uses: owner/repo@ref` or `uses: owner/repo/path@ref`, with an optional trailing comment.
USES = re.compile(r"^\s*-?\s*uses:\s*(?P<action>[^@\s]+)@(?P<ref>\S+)", re.MULTILINE)

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _steps():
    """Every `uses:` line across every workflow, as (file, action, ref)."""
    found = []
    for wf in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        found.extend((wf.name, m.group("action"), m.group("ref"))
                     for m in USES.finditer(wf.read_text(encoding="utf-8")))
    return found


def test_there_are_workflows_with_steps_to_check():
    """Zero matches would make the assertion below vacuously true."""
    assert _steps(), f"no `uses:` steps found under {WORKFLOWS}; has the layout moved?"


@pytest.mark.parametrize(("wf", "action", "ref"), _steps())
def test_every_action_is_pinned_to_a_commit(wf, action, ref):
    """A tag or a branch here is a name its owner can repoint at different code.

    Local actions (`./.github/actions/...`) and reusable workflows inside this repository are
    not third-party and do not reach this assertion, because they carry no `@ref`.
    """
    assert FULL_SHA.match(ref), (
        f"{wf} uses {action}@{ref}, which is a tag or a branch rather than a commit. "
        f"Pin it to the full 40-character SHA that tag currently points at, and leave the "
        f"version in a trailing comment for humans: whoever owns {action.split('/')[0]} can "
        f"move a tag onto different code, and that code runs with this workflow's token.")


def test_the_codecov_upload_cannot_fail_the_build():
    """Coverage REPORTING is a third party; the coverage GATE is pytest, and it stays ours.

    If `fail_ci_if_error` is ever flipped to true, an outage at Codecov turns this repository's
    build red for a reason that says nothing about this repository's code. The real gate is
    `--cov-fail-under`, which runs in the step above the upload on every matrix row.
    """
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    if "codecov-action" not in ci:
        pytest.skip("no Codecov step in ci.yml")
    assert "fail_ci_if_error: false" in ci, (
        "the Codecov step does not set `fail_ci_if_error: false`. Coverage is gated by "
        "`--cov-fail-under` in the test step; the upload only reports, and a reporting "
        "service must not be able to fail this build.")


def test_the_coverage_badge_is_measured_rather_than_typed():
    """The badge reads a figure from Codecov instead of quoting a number someone maintains.

    The previous badge said `coverage-95%` as static text. Nothing verified it, and CI's real
    figure is lower than 95 because a runner has none of the release artefacts. A hardcoded
    number that drifts in the flattering direction is the defect class this project keeps
    finding in its own published figures.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "img.shields.io/codecov/c/github/" in readme, (
        "the README coverage badge no longer reads from Codecov")
    assert not re.search(r"badge/coverage-\d+", readme), (
        "the README carries a hardcoded coverage badge again; nothing verifies that number")
