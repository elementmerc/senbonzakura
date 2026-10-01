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
