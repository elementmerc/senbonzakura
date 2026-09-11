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


#: `action.yml` at the repository root is the action this project PUBLISHES, and its composite
#: steps carry `uses:` lines exactly like a workflow's. Leaving it out would mean the one file
#: strangers execute in their own CI was the one file not checked for pinned actions.
def _yaml_files():
    return (sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
            + [p for p in (ROOT / "action.yml", ROOT / "action.yaml") if p.exists()])


def _steps():
    """Every `uses:` line across every workflow and the published action."""
    found = []
    for wf in _yaml_files():
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


# ── a workflow that GitHub will actually accept ──────────────────────────────────────────────

def _workflow_files():
    return _yaml_files()


#: Keys whose value GitHub requires to be a real mapping. A null here is refused outright.
#:
#: This is a deny-list rather than "nothing may be null", and the first version of this test got
#: that wrong. `on: workflow_dispatch:` with nothing under it is legal and ordinary, and so are
#: the other trigger keys: they mean "this event, with no filters". Asserting on every key at
#: once flagged `docs.yml` for something GitHub accepts happily, which would have been a gate
#: that cries wolf, and a gate that cries wolf gets deleted.
MUST_NOT_BE_NULL = ("env", "with", "jobs", "steps", "strategy", "matrix", "defaults", "outputs")


@pytest.mark.parametrize("wf", [p.name for p in _workflow_files()])
def test_no_mapping_key_that_needs_a_value_is_left_empty(wf):
    """A key with nothing under it is null, not an empty mapping, and GitHub REJECTS the file.

    THE INCIDENT, 2026-09-11. Deleting the last variable out of `ci.yml` left `env:` followed by
    a comment block and nothing else. `yaml.safe_load` parsed it happily and returned
    `{"env": None}`, so the local check passed. GitHub refused the whole workflow: the run
    completed as a FAILURE with zero jobs, no annotations, and `gh run view` printing nothing
    under the job list. That looks nothing like a normal red build, and the first instinct is to
    go hunting for a broken test.

    The lesson is narrow and worth keeping: parsing is not validation. PyYAML answers "is this
    YAML", and the question that mattered was "will GitHub take it".
    """
    import yaml

    path = next(p for p in _yaml_files() if p.name == wf)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{wf} does not parse to a mapping at all"

    def _check(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in MUST_NOT_BE_NULL:
                    assert v, (
                        f"{wf}: `{path}{k}:` has nothing under it, so it parses as null and "
                        f"GitHub refuses the whole workflow, failing the run with zero jobs. "
                        f"Delete the key, or give it a value.")
                _check(v, f"{path}{k}.")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _check(v, f"{path}[{i}].")

    _check(doc, "")


def test_the_empty_key_gate_would_actually_fire():
    """The negative control. A gate only ever seen agreeing has not been shown to work.

    This project's own discipline, arrived at after a dead-flag audit passed a tree with the bug
    reinstated: build the broken thing and require the check to refuse it.
    """
    import yaml

    broken = yaml.safe_load("name: x\non:\n  push:\njobs:\n  a:\n    env:\n    steps:\n      - run: x\n")
    assert broken["jobs"]["a"]["env"] is None, "the fixture is not the shape being guarded against"

    found = []

    def _check(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in MUST_NOT_BE_NULL and not v:
                    found.append(f"{path}{k}")
                _check(v, f"{path}{k}.")

    _check(broken, "")
    assert "jobs.a.env" in found, "the empty `env:` that broke CI would not be caught"


@pytest.mark.parametrize("wf", [p.name for p in _workflow_files()])
def test_every_job_has_steps_to_run(wf):
    """A job with no steps is accepted and does nothing, which is a green tick for no work.

    Cheaper to assert than to notice. This project has twice shipped a check that passed because
    nothing happened rather than because everything did.
    """
    import yaml

    path = next(p for p in _yaml_files() if p.name == wf)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, job in (doc.get("jobs") or {}).items():
        if "uses" in job:
            continue                      # a reusable workflow call carries no steps of its own
        assert job.get("steps"), f"{wf}: job `{name}` has no steps"
