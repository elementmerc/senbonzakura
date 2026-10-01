# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The gates that must run on every push, asserted against the workflow rather than against a name.

WHAT PROMPTED IT, 2026-09-28

`tools/ci/check_prompt_artefacts.py` is the one control between a harmful prompt and a public push.
A reviewer asked what would happen if the `run:` line that invokes it were deleted, and the answer
was that the whole suite stays green.

`test_tools_layout.py` looks like it covers this and does not. It asks whether anything in a list of
invoker locations mentions the script's filename, and `tools/hooks/precommit-prompt-artefacts.sh` is
in that list, so the assertion is satisfied by the LOCAL hook wiring whatever CI does. Those are
different guarantees: a hook protects a clone that ran the installer, and CI protects the push. It
also matches a bare filename anywhere in a file, so a surviving comment mentioning the script
satisfies it too.

The second gate here has the opposite history. `tools/packaging/add_license_headers.py --check` was
written months ago, `pyproject.toml` points at it as "the same assertion in a form CI and a
pre-commit hook can run", and until 2026-09-28 nothing ran it anywhere. It was also failing, on the
build stamp that ships in every wheel.

SO THIS READS THE PARSED WORKFLOW. Not a substring of the file: a step's `run` command, in a job
that triggers on push, with no `if:` narrowing it to one matrix row. A comment cannot satisfy it and
a step moved behind a condition fails it.
"""
from __future__ import annotations

import pathlib
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is needed to read the workflow as data")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"

#: (the script, why it must run on every push). A gate that runs on one matrix row protects that row.
MUST_RUN_ON_EVERY_PUSH = {
    "tools/ci/check_prompt_artefacts.py":
        "the one control between a harmful prompt and a public push",
    "tools/packaging/add_license_headers.py":
        "every file this project ships owes a licence notice, and the generated build stamp had none",
}


@pytest.fixture(scope="module")
def workflow():
    if not CI.is_file():
        pytest.skip("no .github/workflows/ci.yml in this checkout")
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def test_the_workflow_is_readable_and_has_jobs(workflow):
    """Without this every test below passes on a parse that returned nothing."""
    assert isinstance(workflow, dict), type(workflow)
    jobs = workflow.get("jobs") or {}
    assert len(jobs) > 3, f"only found jobs {sorted(jobs)}"
    steps = [s for j in jobs.values() for s in (j.get("steps") or [])]
    assert len(steps) > 20, f"only {len(steps)} steps across {len(jobs)} jobs"


def _push_jobs(workflow):
    """Jobs that run on a push to a branch, which is the event these gates exist for.

    `on` parses as the boolean True under YAML 1.1, because that is what `on` means there. Both
    spellings are accepted rather than guessed at.
    """
    triggers = workflow.get("on", workflow.get(True)) or {}
    assert "push" in triggers, f"this workflow does not run on push: {sorted(triggers)}"
    for name, job in (workflow.get("jobs") or {}).items():
        # A job gated on its own condition is not a job that runs on every push.
        if job.get("if"):
            continue
        yield name, job


@pytest.mark.parametrize("script", sorted(MUST_RUN_ON_EVERY_PUSH))
def test_a_gate_that_must_run_on_every_push_does(workflow, script):
    reason = MUST_RUN_ON_EVERY_PUSH[script]
    unconditional, conditional = [], []
    for job_name, job in _push_jobs(workflow):
        for step in job.get("steps") or []:
            if script not in str(step.get("run") or ""):
                continue
            where = f"{job_name}: {step.get('name') or 'unnamed step'}"
            (conditional if step.get("if") else unconditional).append(where)

    assert unconditional, (
        f"nothing in this workflow runs {script} unconditionally on a push, and it is {reason}.\n"
        f"  Steps that run it behind a condition: {conditional or 'none'}\n"
        f"  A gate behind an `if:` protects the rows the condition happens to match. A gate no step "
        f"runs protects nothing, and deleting its line would leave the suite green, which is how "
        f"this came to be written.")


@pytest.mark.parametrize("script", sorted(MUST_RUN_ON_EVERY_PUSH))
def test_the_gate_is_given_something_to_check(workflow, script):
    """A gate invoked with no argument may inspect nothing and still exit 0.

    `check_prompt_artefacts.py` takes a path and `add_license_headers.py` takes `--check`; either one
    called bare is a green tick that means the script imported.
    """
    calls = [str(step.get("run")) for _n, job in _push_jobs(workflow)
             for step in (job.get("steps") or []) if script in str(step.get("run") or "")]
    assert calls, f"no call to {script} to inspect"
    for call in calls:
        after = call.split(script, 1)[1].strip()
        assert after and not after.startswith("#"), (
            f"{script} is called with no argument, so it may check nothing and exit 0: {call!r}")


def test_the_reader_would_notice_a_gate_being_removed():
    """Mutation test: the predicate has to fail on a workflow that does not run the gate.

    Built here rather than by editing the real file, because a guard whose own proof needs the
    subject broken on disk is a guard nobody runs twice.
    """
    without = {"on": {"push": {"branches": ["dev"]}},
               "jobs": {"test": {"steps": [{"name": "Lint", "run": "ruff check src/"}]}}}
    with pytest.raises(AssertionError, match="nothing in this workflow runs"):
        test_a_gate_that_must_run_on_every_push_does(
            without, "tools/ci/check_prompt_artefacts.py")

    behind_a_condition = {
        "on": {"push": {}},
        "jobs": {"test": {"steps": [
            {"name": "gate", "if": "matrix.os == 'ubuntu-latest'",
             "run": "python tools/ci/check_prompt_artefacts.py ."}]}}}
    with pytest.raises(AssertionError, match="nothing in this workflow runs"):
        test_a_gate_that_must_run_on_every_push_does(
            behind_a_condition, "tools/ci/check_prompt_artefacts.py")


def test_a_bare_invocation_would_be_noticed():
    """Mutation test for the second predicate, for the same reason."""
    bare = {"on": {"push": {}},
            "jobs": {"test": {"steps": [
                {"name": "gate", "run": "python tools/ci/check_prompt_artefacts.py"}]}}}
    with pytest.raises(AssertionError, match="no argument"):
        test_the_gate_is_given_something_to_check(
            bare, "tools/ci/check_prompt_artefacts.py")


# ── the Action this repository publishes ─────────────────────────────────────────────────────────

def test_the_published_action_is_executed_on_every_push(workflow):
    """Parsing `action.yml` is not running it, and for six days that difference shipped.

    Four unit tests in `test_check_cli.py` asserted the action's metadata and all four passed while
    the action could not run at all: it installed `--no-deps senbonzakura`, which brings the
    launcher without the `senbonzakura_check` package it dispatches to. The one thing that would
    have caught it is a job that says `uses: ./`.
    """
    runners = [(name, step) for name, job in _push_jobs(workflow)
               for step in (job.get("steps") or [])
               if str(step.get("uses") or "").strip() in {"./", ".github", "."}]
    assert runners, (
        "no job on the push path runs this repository's own action with `uses: ./`, so the action "
        "is published metadata that nothing executes. That is how it shipped broken in v0.4.0.")


def test_the_action_is_exercised_on_a_failing_case_and_not_only_a_passing_one(workflow):
    """An action that can only be seen to pass is an action whose failure path is untested.

    The break was silent precisely because the step exited 0. A job that runs the action once on
    clean input would have stayed green through all of it.
    """
    invocations = [step for _n, job in _push_jobs(workflow)
                   for step in (job.get("steps") or [])
                   if str(step.get("uses") or "").strip() == "./"]
    assert len(invocations) >= 2, (
        f"the action is run {len(invocations)} time(s) on the push path. It needs at least a "
        "passing case and a failing one, or nothing distinguishes 'it works' from 'it exits 0'.")
    assert any(step.get("continue-on-error") for step in invocations), (
        "every invocation of the action is expected to succeed, so the case where it MUST fail, "
        "which is the entire purpose of the action, is never exercised.")


def test_the_regression_gate_is_exercised_through_the_action_and_not_only_as_a_command(workflow):
    """A gate nobody has wired in runs exactly as often as no gate.

    `gate.py` was complete, tested and reachable by no CI job and no action input for a fortnight,
    which is the failure mode that is easiest to mistake for closed because the code is there. The
    unit tests drive the action's shell with stubs, which is where the verdict-to-annotation
    mapping lives; this asserts that the real thing runs, against the real install, on every push.

    ALL THREE VERDICTS, not just the failure. A refusal and a regression are different findings and
    the exit statuses exist to keep them apart, so an action exercised on two of the three can map
    the third onto either of the others and nothing would notice.
    """
    gated = [step for _n, job in _push_jobs(workflow)
             for step in (job.get("steps") or [])
             if str(step.get("uses") or "").strip() == "./" and (step.get("with") or {}).get("baseline")]
    assert len(gated) >= 3, (
        f"the published action is run with a baseline {len(gated)} time(s) on the push path. It "
        f"needs the pass, the regression and the refusal, because those are three outcomes and a "
        f"job that sees two of them cannot tell the third from either.")
    measurements = {Path(str((step.get("with") or {}).get("measurement") or "")).name
                    for step in gated}
    assert len(measurements) >= 3, (
        f"the gate is run against {sorted(measurements)}. Three invocations of the same comparison "
        f"exercise one verdict three times.")
    assert any(step.get("continue-on-error") for step in gated), (
        "every gated invocation is expected to succeed, so the case where the gate MUST fail the "
        "build, which is the only reason a gate exists, is never exercised.")


# ── the badge, and a self-correcting job that stood down when it was needed ──────────
#
# Two reviewers reached the same finding on 2026-09-28: the `tests-NNNN` badge was stale, and the
# job that corrects it is skipped precisely when the suite is red. `needs: test` is the mechanism,
# and it was a guess rather than a reason. The number is how many tests COLLECT, and a suite with
# ten failures collects exactly as many as a green one, so nothing about it depends on the result.
#
# The property these hold is not "the condition reads a certain way". It is that neither the
# measurement nor the commit is gated on the suite having PASSED, which is the shape the finding
# was about, and that the one job holding a write token is still fenced everywhere else.

BADGE_JOB = "tests-badge"
BADGE_SCRIPT = "tools/ci/check_tests_badge.py"


def _step_named(workflow, job, needle):
    for step in (workflow["jobs"][job].get("steps") or []):
        if needle.lower() in str(step.get("name", "")).lower():
            return step
    return None


def _badge_steps(workflow):
    """Every step of any job that invokes the badge script, plus the job's name."""
    return [(name, job, step)
            for name, job in (workflow.get("jobs") or {}).items()
            for step in (job.get("steps") or [])
            if BADGE_SCRIPT in str(step.get("run", ""))]


def test_the_badge_is_measured_somewhere(workflow):
    """Without this the two tests below pass on a workflow that stopped measuring it at all."""
    found = _badge_steps(workflow)
    assert found, f"no step in this workflow runs {BADGE_SCRIPT}, so nothing measures the badge"


def test_the_badge_measurement_is_not_gated_on_the_suite_passing(workflow):
    """A collection count does not depend on the suite's verdict, and used to be gated on it."""
    for job_name, _job, step in _badge_steps(workflow):
        condition = str(step.get("if", ""))
        assert "always()" in condition, (
            f"the step measuring the badge in `{job_name}` has `if: {condition or '<none>'}`, so a "
            f"failing step earlier in that job skips it. The count is how many tests collect, "
            f"which is the same number whether the suite passed or not, and the badge therefore "
            f"stays stale on exactly the runs somebody is already looking at")


def test_the_badge_artefact_is_handed_over_on_a_red_run_too(workflow):
    """Measuring it and then not uploading it leaves the committing job nothing to commit."""
    step = _step_named(workflow, "test", "Hand the corrected badge")
    assert step is not None, "the test job no longer uploads the measured badge"
    assert "always()" in str(step.get("if", "")), (
        "the badge is measured on a red run and then not handed over, so the correction stops "
        "one step short of the job that can commit it")


def test_the_badge_job_runs_when_the_suite_is_red_and_stops_when_it_is_cancelled(workflow):
    """`needs:` alone skipped it on every red run. A cancelled run is still a stop."""
    job = (workflow.get("jobs") or {}).get(BADGE_JOB)
    assert job is not None, f"there is no `{BADGE_JOB}` job to check"
    condition = str(job.get("if", ""))
    assert "always()" in condition, (
        f"`{BADGE_JOB}` has `needs: {job.get('needs')}` and no `always()`, so it is skipped "
        f"whenever any matrix row fails, which is when the badge is most likely to be wrong")
    assert "cancelled()" in condition, (
        f"`{BADGE_JOB}` runs on `always()` and does not exclude a cancelled run, which may not "
        f"have reached the measurement at all")


def test_the_badge_job_still_installs_nothing_and_runs_no_project_code(workflow):
    """The reason it is allowed a write token at all, per baseline section 5.

    Loosening WHEN it runs must not loosen WHAT it runs. Checked here rather than left implied,
    because the condition above is the sort of change that invites a convenience step afterwards.
    """
    job = (workflow.get("jobs") or {}).get(BADGE_JOB)
    assert job is not None
    for step in (job.get("steps") or []):
        uses = str(step.get("uses", ""))
        if uses:
            assert uses.startswith(("actions/checkout@", "actions/download-artifact@")), (
                f"`{BADGE_JOB}` holds a write token and now uses `{uses}`, which puts a third "
                f"party in the same process as that token")
        run = str(step.get("run", ""))
        for forbidden in ("pip install", "uv pip", "pytest", "python tools/", "python -m"):
            assert forbidden not in run, (
                f"`{BADGE_JOB}` holds a write token and its step `{step.get('name')}` now runs "
                f"`{forbidden}`. It is allowed that token because it executes nothing")


def test_the_badge_job_cannot_commit_more_than_the_badge_line(workflow):
    """The guard that makes running it on a red run safe. It must survive this change."""
    step = _step_named(workflow, BADGE_JOB, "Commit the corrected badge")
    assert step is not None, f"`{BADGE_JOB}` no longer has its commit step"
    run = str(step.get("run", ""))
    assert "img\\.shields\\.io/badge/tests-" in run or "img.shields.io/badge/tests-" in run, (
        "the commit step no longer checks that the handed-over diff touches only the badge URL, "
        "which is the check that stops this job committing whatever an earlier job produced")
    assert "numstat" in run, (
        "the commit step no longer checks that exactly one line changed")


def test_the_badge_job_says_so_when_no_count_was_handed_over(workflow):
    """A missing artefact must read as a missing artefact, not as a badge already correct.

    `test` can go red before the measurement, for instance on a failed install, and the download
    is `continue-on-error` so this job does not add a second red to an already red run. Silence
    there would be indistinguishable from `cmp -s` finding the badge correct.
    """
    data = workflow
    download = None
    for step in (data["jobs"][BADGE_JOB].get("steps") or []):
        if "download-artifact" in str(step.get("uses", "")):
            download = step
    assert download is not None, f"`{BADGE_JOB}` no longer downloads the measured badge"
    assert download.get("continue-on-error") is True, (
        "the download is not `continue-on-error`, so a run that never reached the measurement "
        "fails this job for an input it was never promised")
    commit = _step_named(data, BADGE_JOB, "Commit the corrected badge")
    run = str(commit.get("run", ""))
    assert "badge/README.md" in run and "nothing to correct" in run, (
        "the commit step does not say, in words, that no count was handed over. A silent exit "
        "reads exactly like a badge that was already right")
