# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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

THE SECOND HALF, ADDED 2026-10-01

This file used to end with a section headed "what it does not do", saying it did not check that
the SHA exists or that it belongs to the tag in the trailing comment, because that needs the
network. The gap was real and it was costly: the comment beside a hash is prose, nothing read
it, and a wrong one is worse than none because it licenses the next reader to trust it. An
audit of all fourteen distinct pins found one wrong, `actions/setup-node` pinned to the v6.0.0
commit under a comment saying v5.0.0, and three more claiming `tip of the v3 series` or `v6`,
which are claims that were true on the day they were written and go false on their own.

It is closed now without the network, by `.github/action-pins.lock`: a checked-in record of
what each pinned SHA resolves to, produced by `tools/ci/resolve_action_pins.py --resolve`
against the GitHub API and reviewed in a diff like any other change. The tests below assert the
workflows and their comments agree with that record. A scheduled job re-resolves it and reports
when a tag has been repointed underneath us.

Why the record rather than a live lookup, in one line: resolving live would check the pin
against whatever the tag says today, and the movable tag is the thing the pinning rule exists
to distrust, so an attacker who repointed `v7` would have made the test agree with them. The
full reasoning is in the tool's docstring.
"""
import re
import sys
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


# ── the comment beside the hash is a claim, and these check it ───────────────────────────────

#: Imported rather than reimplemented. Two parsers for the same line is how the audit and the
#: gate drift apart, and then the gate is measuring its own copy of the question.
sys.path.insert(0, str(ROOT / "tools" / "ci"))
import resolve_action_pins as pins  # noqa: E402


def test_the_pin_lockfile_exists_and_covers_every_pinned_reference():
    """Every `uses: owner/repo@<sha>` in the tree has a resolved record behind it.

    The failure this forbids is a pin added without resolving it, which would leave its version
    comment unchecked and look exactly like a pin that had been verified.
    """
    lock = pins.load_lockfile()
    sites = pins.pin_sites()
    assert sites, "no pinned references found at all; has the workflow layout moved?"
    locked = {(p["action"], p["sha"]) for p in lock["pins"]}
    missing = sorted({(s["repo"], s["sha"]) for s in sites} - locked)
    assert not missing, (
        "these pins have no resolved record, so the version in their trailing comment is an "
        f"unverified claim: {missing}. Run `python tools/ci/resolve_action_pins.py --resolve`.")


def test_every_version_comment_agrees_with_the_resolved_record():
    """The actual gate. A comment naming a version the SHA is not is a finding, not a typo."""
    findings = pins.check(pins.pin_sites(), pins.load_lockfile())
    assert not findings, "action pin comments disagree with the resolved record:\n" + "\n".join(
        f"  - {f}" for f in findings)


def test_the_published_action_is_in_scope():
    """`action.yml` carries a pin too, and an audit scoped to "the workflows" misses it.

    It is the composite action this project PUBLISHES, so it is the one file strangers execute
    in their own CI. The first draft of the audit tool enumerated `.github/workflows/*.yml` and
    would have left this pin unresolved.
    """
    assert (ROOT / "action.yml").exists(), "action.yml has moved; this test's premise is stale"
    assert any(s["file"] == "action.yml" for s in pins.pin_sites()), (
        "no pinned reference was found in action.yml, but the file exists. Either the pin was "
        "removed, or the parser stopped reaching the published action.")


def test_no_version_comment_claims_a_moving_name():
    """A comment saying `v3` or `tip of the v6 series` cannot be held to account by anything.

    It was true when written and goes false when upstream tags again, with nothing in this
    repository changing. Four pins read that way until 2026-10-01. The rule is that a comment
    names the exact release, which is a statement that either stays true or becomes a finding.
    """
    loose = [
        f"{s['file']}:{s['line']} ({s['repo']}) -> {s['comment']!r}"
        for s in pins.pin_sites()
        if (t := pins._claimed_tag(str(s["comment"]))) and re.fullmatch(r"v\d+", t)
    ]
    assert not loose, (
        "these comments claim a moving major version rather than a release:\n"
        + "\n".join(f"  - {line}" for line in loose))


def test_the_comment_gate_would_actually_fire():
    """The negative control, per this project's own discipline: build the defect, demand refusal.

    Three mutants, each a real shape the audit met: a comment naming the wrong version, a pin
    with no version at all, and a pin absent from the lockfile.
    """
    lock = {"pins": [{
        "action": "actions/setup-node",
        "sha": "2028fbc5c25fe9cf00d9f06a71cc4710d4507903",
        "exists": True,
        "tags_pointing_here": ["v6.0.0"],
        "tag": "v6.0.0",
    }]}
    site = {
        "file": "ci.yml", "line": 809,
        "repo": "actions/setup-node",
        "sha": "2028fbc5c25fe9cf00d9f06a71cc4710d4507903",
    }

    wrong = pins.check([{**site, "comment": "v5.0.0"}], lock)
    assert len(wrong) == 1 and "v5.0.0" in wrong[0] and "v6.0.0" in wrong[0], (
        f"the real 2026-10-01 defect would not be caught: {wrong}")

    silent = pins.check([{**site, "comment": ""}], lock)
    assert len(silent) == 1 and "no version" in silent[0], (
        f"a pin with no version comment would pass: {silent}")

    unknown = pins.check([{**site, "sha": "0" * 40, "comment": "v6.0.0"}], lock)
    assert len(unknown) == 1 and "not in the pin lockfile" in unknown[0], (
        f"a pin with no resolved record would pass: {unknown}")

    # And the positive control: the true comment must NOT be flagged, or the gate cries wolf.
    assert not pins.check([{**site, "comment": "v6.0.0"}], lock), (
        "the correct comment is being reported as a finding, which makes the gate noise")


def test_a_sha_the_api_could_not_find_is_a_finding_in_its_own_right():
    """`exists: false` outranks every comment question. It means the pin resolves to nothing.

    Kept separate because the handling differs: a wrong comment is a documentation fix, and a
    SHA that is not in the repository means the workflow cannot run at all, or that the commit
    was removed from upstream after we pinned it.
    """
    lock = {"pins": [{
        "action": "actions/checkout", "sha": "a" * 40, "exists": False,
        "tags_pointing_here": [], "tag": None,
    }]}
    found = pins.check(
        [{"file": "ci.yml", "line": 1, "repo": "actions/checkout",
          "sha": "a" * 40, "comment": "v7.0.1"}], lock)
    assert len(found) == 1 and "DOES NOT EXIST" in found[0], found

    locked = pins.load_lockfile()
    ghosts = [p["action"] for p in locked["pins"] if not p.get("exists")]
    assert not ghosts, f"the lockfile records pins whose SHA resolves to nothing: {ghosts}"


def test_the_lockfile_records_when_it_was_resolved():
    """A record with no date cannot be judged stale, and staleness is the whole risk here."""
    lock = pins.load_lockfile()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(lock.get("resolved_at", ""))), (
        "the pin lockfile carries no ISO date, so nothing can say how old its claims are")


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
