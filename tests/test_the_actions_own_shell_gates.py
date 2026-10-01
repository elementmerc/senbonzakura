# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The action's shell, run as a shell, because reading it is not running it.

THE DEFECT THIS MODULE EXISTS FOR, in the words `action.yml`'s own header uses: the report the
command could not write parsed to an empty string rather than to a zero, three shell variables
held the empty string, every gate below them is an integer test, `[ "" -eq 0 ]` errors rather than
being false, a failing test is a skipped branch, so all three gates were jumped and the step
reached `exit 0`. A broken install reported a clean run.

Four unit tests asserted that action's metadata at the time. All four passed.

So this module takes each step's `run:` script out of `action.yml` and executes it under bash,
with the commands it calls replaced by stubs whose exit status and output the test chooses. What
is asserted is the step's own exit status and the annotations it emits, which is the whole of what
a workflow can observe.

WHY STUBS RATHER THAN THE REAL COMMANDS. The question here is not whether the checker and the gate
are correct, which is what the rest of the suite is for. It is whether the shell around them turns
each of their outcomes into the right verdict, and the outcomes that matter most are the ones that
are awkward to produce on demand: a crash, an unwritable report, an exit status nobody expected. A
stub produces those exactly, every time, which is what a test of a failure path needs.

NOTHING HERE READS `${{ }}`. Every input reaches these scripts through `env:`, so the script a
runner executes and the script this module executes are the same text. A `${{ }}` expansion is
pasted in before bash starts, which both makes it untestable here and makes a path containing a
quote and a semicolon a command on somebody's runner, so the last test in this module keeps it
that way.
"""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
ACTION = ROOT / "action.yml"

#: What the action declares as defaults, so a test only states the inputs it is about.
_ACTION = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
DEFAULTS = {name: str(spec.get("default", "")) for name, spec in _ACTION["inputs"].items()}

TEMPLATE = re.compile(r"\$\{\{\s*inputs\.([a-z0-9-]+)\s*\}\}")


def _step(name: str) -> dict:
    for step in _ACTION["runs"]["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(
        f"no step named {name!r} in action.yml. The steps are: "
        f"{[s.get('name') or s.get('uses') for s in _ACTION['runs']['steps']]}. This module drives "
        f"them by name, so a rename has to reach here rather than silently testing nothing.")


def _stub(directory: Path, name: str, *, status: int, stdout: str = "", stderr: str = "") -> Path:
    """A command on PATH that prints what the test wants and exits how the test wants."""
    path = directory / name
    path.write_text(
        "#!/usr/bin/env bash\n"
        f"cat <<'STUB_OUT'\n{stdout}\nSTUB_OUT\n"
        f"cat >&2 <<'STUB_ERR'\n{stderr}\nSTUB_ERR\n"
        f"exit {status}\n",
        encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class Result:
    """What a workflow can see of a step: its status, its log, and what it wrote back."""

    def __init__(self, proc, outputs: dict, summary: str):
        self.status = proc.returncode
        self.log = proc.stdout + proc.stderr
        self.outputs = outputs
        self.summary = summary

    @property
    def errors(self) -> list[str]:
        return [ln for ln in self.log.splitlines() if ln.startswith("::error::")]

    @property
    def notices(self) -> list[str]:
        return [ln for ln in self.log.splitlines() if ln.startswith("::notice::")]

    @property
    def warnings(self) -> list[str]:
        return [ln for ln in self.log.splitlines() if ln.startswith("::warning::")]


def _run_step_with_env(step_name: str, tmp_path: Path, env_values: dict) -> Result:
    """Execute a step whose `env:` reads something other than an input.

    The step that fails the build reads `steps.gate.outputs.status`, which this harness cannot
    resolve from the action's own metadata because it is produced by a previous step at runtime.
    The test supplies it, and `_run_step` below is deliberately strict about the rest so that a
    step whose inputs the harness CANNOT resolve fails loudly rather than running with holes.
    """
    return _run_step(step_name, tmp_path, supplied_env=env_values)


def _run_step(step_name: str, tmp_path: Path, inputs: dict | None = None,
              stubs: dict | None = None, supplied_env: dict | None = None) -> Result:
    """Execute one step's script, and hand back everything a workflow would be able to read."""
    step = _step(step_name)
    resolved = {**DEFAULTS, **(inputs or {})}

    tmp_path.mkdir(parents=True, exist_ok=True)
    binaries = tmp_path / "bin"
    binaries.mkdir(exist_ok=True)
    for name, spec in (stubs or {}).items():
        _stub(binaries, name, **spec)
    # `python`, NOT `python3`. The scripts call `python`, which is what `setup-python` leaves on
    # the PATH of a runner, and a machine where only `python3` exists would otherwise make every
    # assertion below a measurement of a missing interpreter rather than of the step.
    shim = binaries / "python"
    if not shim.exists():
        shim.symlink_to(sys.executable)

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir(exist_ok=True)
    out_file = tmp_path / "github-output"
    summary_file = tmp_path / "github-step-summary"
    out_file.write_text("", encoding="utf-8")
    summary_file.write_text("", encoding="utf-8")

    env = {
        "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_OUTPUT": str(out_file),
        "GITHUB_STEP_SUMMARY": str(summary_file),
    }
    for name, value in (step.get("env") or {}).items():
        if supplied_env is not None and name in supplied_env:
            env[name] = supplied_env[name]
            continue
        match = TEMPLATE.fullmatch(str(value).strip())
        assert match, (
            f"step {step_name!r} sets env {name} to {value!r}, which this harness cannot resolve. "
            f"Only `${{{{ inputs.<name> }}}}` is understood, because anything else would be a "
            f"value the test invents rather than one the action supplies.")
        key = match.group(1)
        assert key in resolved, f"step {step_name!r} reads an input {key!r} the action does not declare"
        env[name] = resolved[key]

    proc = subprocess.run(["bash", "-c", step["run"]], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=120, check=False)
    outputs = dict(
        line.split("=", 1) for line in out_file.read_text(encoding="utf-8").splitlines() if "=" in line)
    return Result(proc, outputs, summary_file.read_text(encoding="utf-8"))


# ── the gate, which is what a build system branches on ──────────────────────────────────────────

GATE_STEP = "Run the regression gate"
GATE_INPUTS = {"baseline": "base.json", "measurement": "now.json"}


def _gate(tmp_path, status: int, said: str = "gate said something") -> Result:
    return _run_step(GATE_STEP, tmp_path, GATE_INPUTS,
                     {"senbonzakura": {"status": status, "stdout": said}})


class TestTheGatesThreeVerdictsReachTheWorkflow:
    def test_a_pass_is_a_notice_and_not_a_failure(self, tmp_path):
        res = _gate(tmp_path, 0, "gate OK: refusal_rate: within interval")
        assert res.status == 0, res.log
        assert res.notices, f"a pass emitted no annotation at all:\n{res.log}"
        assert not res.errors, res.errors
        assert res.outputs["status"] == "0"

    def test_a_pass_says_what_it_does_not_mean(self, tmp_path):
        """Loophole 5. Somebody will read a green tick as a claim the model is safe, and the
        annotation is where they will read it rather than in a README they never opened.
        """
        res = _gate(tmp_path, 0)
        assert "not a statement that the model is safe" in res.notices[0], res.notices

    def test_a_regression_is_reported_and_recorded(self, tmp_path):
        """THE VERDICT STEP DOES NOT FAIL, AND THAT IS DELIBERATE: see the step below it.

        A composite action's outputs are mapped from its steps' outputs, and whether a step that
        exited non-zero still has its output propagated to the caller is not a guarantee worth
        resting a gate on. A consumer reading `gate-status` would get the empty string from
        exactly the runs that matter, and an empty string compared as an integer is how this
        action went green on a broken install once already. So the verdict is recorded by a step
        that cannot fail, and `TestWhereTheBuildActuallyFails` below is what fails on it.
        """
        res = _gate(tmp_path, 1, "gate FAIL: refusal_rate: REGRESSED")
        assert res.status == 0, f"the recording step must not fail:\n{res.log}"
        assert res.errors, res.log
        assert "outside its interval" in res.errors[0], res.errors
        assert res.outputs["status"] == "1"

    def test_a_refusal_is_recorded_and_says_nothing_was_shown(self, tmp_path):
        """A refusal is not a regression. Both fail the build, and they must not read alike:
        one says the model got worse, the other says the comparison never happened.
        """
        res = _gate(tmp_path, 2, "gate REFUSED: prompt_format disagreed")
        assert res.status == 0, f"the recording step must not fail:\n{res.log}"
        assert "NOTHING was shown" in res.errors[0], res.errors
        assert "not a regression" in res.errors[0], res.errors
        assert res.outputs["status"] == "2"

    def test_the_three_annotations_are_different_sentences(self, tmp_path):
        """Three statuses mapped onto one message is the same thing as one status."""
        said = {}
        for s in (0, 1, 2):
            res = _gate(tmp_path / f"s{s}", s)
            said[s] = "\n".join(res.errors + res.notices)
        assert len({said[0], said[1], said[2]}) == 3, said
        assert all(said.values()), (
            f"a verdict produced no annotation at all, so this comparison is between empty "
            f"strings: {said}")

    def test_an_unrecognised_status_is_a_broken_gate_rather_than_a_result(self, tmp_path):
        """The case the integer gates got wrong last time, in its general form.

        A status this script has no branch for means the command did not reach a verdict. Reading
        it as either verdict is how a crash becomes a measurement.
        """
        res = _gate(tmp_path, 7)
        assert res.status == 0, res.log
        assert "did not reach one" in res.errors[0], res.errors
        assert res.outputs["status"] == "7", (
            "the status was not reported, so a workflow reading `gate-status` would see nothing "
            "and could not tell a broken gate from an absent one")

    def test_a_missing_command_is_not_a_passing_gate(self, tmp_path):
        """The step runs under `always()`, so it is reached even when its install failed.

        `command not found` is exit 127 from bash, and bash's message does not say which of the
        job's commands was missing. The step says it, and fails.
        """
        res = _run_step(GATE_STEP, tmp_path, GATE_INPUTS, stubs={})
        assert res.status == 1, f"a gate that was never installed reported {res.status}:\n{res.log}"
        assert "not installed" in res.errors[0], res.errors
        assert "not a passing gate" in res.errors[0], res.errors

    def test_the_gates_own_words_reach_the_step_summary(self, tmp_path):
        """A CI failure nobody can read is a CI failure somebody disables."""
        res = _gate(tmp_path, 1, "gate FAIL: refusal_rate: REGRESSED by 0.1900")
        assert "REGRESSED by 0.1900" in res.summary, res.summary


class TestAskingForHalfAGate:
    """A half-configured gate is the one outcome that must never be quiet.

    Skipping it would leave a workflow whose author believes their numbers are gated and whose
    build has never compared them, and the missing line reads as a pass.
    """

    PREFLIGHT = "Say what this run was asked to do"

    def test_a_baseline_with_no_measurement_stops_the_run(self, tmp_path):
        res = _run_step(self.PREFLIGHT, tmp_path, {"baseline": "base.json"})
        assert res.status == 1, res.log
        assert "no measurement" in res.errors[0], res.errors

    def test_a_measurement_with_no_baseline_stops_the_run(self, tmp_path):
        res = _run_step(self.PREFLIGHT, tmp_path, {"measurement": "now.json"})
        assert res.status == 1, res.log
        assert "no baseline" in res.errors[0], res.errors

    def test_both_together_is_fine(self, tmp_path):
        assert _run_step(self.PREFLIGHT, tmp_path, GATE_INPUTS).status == 0

    def test_neither_is_fine_because_the_gate_is_optional(self, tmp_path):
        assert _run_step(self.PREFLIGHT, tmp_path).status == 0


# ── the checker's own gates, driven the same way ─────────────────────────────────────────────────

CHECK_STEP = "Check the artefacts"


def _report(entries) -> str:
    return json.dumps(entries)


def _check(tmp_path, report: str, inputs: dict | None = None, status: int = 0) -> Result:
    """Drive the checker step with a report of the test's choosing.

    The stub writes the report to stdout, which is where `--json` goes and therefore where the
    step's redirection puts it. The second invocation, the one that renders for the log, writes the
    same thing and is ignored, exactly as a real run's two formats are.
    """
    return _run_step(CHECK_STEP, tmp_path, inputs,
                     {"senbonzakura-check": {"status": status, "stdout": report}})


ONE_CLEAN = [{"unchecked": False, "not_a_result": False, "applied": True, "findings": []}]
ONE_FINDING = [{"unchecked": False, "not_a_result": False, "applied": True,
                "findings": [{"id": "a-rate-with-no-partition-beside-it"}]}]
ONE_UNREADABLE = [{"unchecked": True, "not_a_result": False, "applied": False, "findings": []}]
ONE_NOT_A_RESULT = [{"unchecked": False, "not_a_result": True, "applied": False, "findings": []}]


class TestTheCheckersThreeOutcomesReachTheWorkflow:
    def test_a_clean_sweep_passes_and_counts_the_file(self, tmp_path):
        res = _check(tmp_path, _report(ONE_CLEAN))
        assert res.status == 0, res.log
        assert res.outputs["checked"] == "1" and res.outputs["findings"] == "0"

    def test_a_finding_fails_by_default(self, tmp_path):
        res = _check(tmp_path, _report(ONE_FINDING), status=1)
        assert res.status == 1, res.log
        assert res.outputs["findings"] == "1"

    def test_a_finding_can_be_advisory(self, tmp_path):
        """The sensible first week on an existing repository, and it has to actually work."""
        res = _check(tmp_path, _report(ONE_FINDING), {"fail-on-findings": "false"}, status=1)
        assert res.status == 0, res.log
        assert res.outputs["findings"] == "1"

    def test_an_unreadable_file_fails_and_is_never_counted_as_clean(self, tmp_path):
        res = _check(tmp_path, _report(ONE_UNREADABLE), status=2)
        assert res.status == 1, res.log
        assert res.outputs["unchecked"] == "1"
        assert res.outputs["checked"] == "0", (
            "a file that could not be read was counted as checked, which is the arithmetic that "
            "makes an unreadable artefact indistinguishable from a clean one")

    def test_nothing_checked_warns_and_can_be_made_fatal(self, tmp_path):
        res = _check(tmp_path, _report(ONE_NOT_A_RESULT))
        assert res.status == 0, res.log
        assert res.warnings, "nothing was checked and the log did not say so"
        fatal = _check(tmp_path / "fatal", _report(ONE_NOT_A_RESULT), {"fail-on-empty": "true"})
        assert fatal.status == 1, fatal.log

    def test_an_empty_report_is_a_broken_install_rather_than_a_clean_run(self, tmp_path):
        """THE ORIGINAL DEFECT, pinned where it happened.

        A command that could not run writes nothing, and nothing is not zero. The step has to say
        that it knows nothing about the run rather than branch on three empty strings.
        """
        res = _check(tmp_path, "", status=127)
        assert res.status == 1, f"an unwritable report reported success:\n{res.log}"
        assert "no readable report" in res.errors[0], res.errors
        assert "NOT a clean result" in res.errors[0], res.errors

    def test_a_report_the_checker_could_not_finish_writing_is_not_a_clean_run(self, tmp_path):
        """The same defect one step along: truncated JSON, which `json.load` refuses."""
        res = _check(tmp_path, '[{"unchecked": false, "find', status=1)
        assert res.status == 1, res.log
        assert "no readable report" in res.errors[0], res.errors

    def test_a_checker_without_the_applied_field_says_so(self, tmp_path):
        """`version:` lets somebody pin an older checker, and a count that silently changes
        definition with the pin is what this project exists to catch in other people's tooling.
        """
        legacy = [{"unchecked": False, "not_a_result": False, "findings": []}]
        res = _check(tmp_path, _report(legacy))
        assert res.status == 0, res.log
        assert any("predates" in w for w in res.warnings), res.warnings




class TestWhereTheBuildActuallyFails:
    """The step that turns a recorded verdict into a failed build.

    Split from the step that records it, so the output a consumer reads is always there. The two
    halves have to agree about which statuses are failures, and this is the half a reader of the
    workflow sees, so each status is driven through it separately rather than inferred.
    """

    STEP = "Fail the build on the gate's verdict"

    def test_the_condition_skips_a_passing_gate(self):
        """Read off the step's own `if`, because nothing else in this module can: a skipped step
        is the runner's decision and there is no script to execute for it.
        """
        condition = str(_step(self.STEP).get("if") or "")
        assert "steps.gate.outputs.status != '0'" in condition, condition
        assert "inputs.baseline != ''" in condition, (
            f"the failing step is not conditional on the gate having been asked for, so a run "
            f"that gave no baseline would be failed by it: {condition}")
        assert "always()" in condition, (
            f"the step is skipped when the checker ahead of it failed, so a repository that gates "
            f"on findings would never have its regression failure delivered: {condition}")
        assert "steps.gate.outcome == 'success'" in condition, (
            f"the step speaks even when the gate above it never reached a verdict, which puts a "
            f"second error under a failure that already said why: {condition}")

    @pytest.mark.parametrize("status", ["1", "2", "7"])
    def test_every_non_zero_verdict_fails_the_build(self, tmp_path, status):
        res = _run_step_with_env(self.STEP, tmp_path / f"env{status}", {"STATUS": status})
        assert res.status == 1, f"status {status} did not fail the build:\n{res.log}"
        assert status in res.log, res.log

    def test_an_absent_status_is_a_broken_gate_and_not_a_pass(self, tmp_path):
        """The case the split exists for. If the recording step could not run at all there is no
        status, and the one thing that must not happen is this reading as a pass.
        """
        res = _run_step_with_env(self.STEP, tmp_path, {"STATUS": ""})
        assert res.status == 1, res.log
        assert "no status at all" in res.errors[0], res.errors
        assert "not a pass" in res.errors[0], res.errors


# ── the property that keeps this module able to test anything at all ───────────────────────────

def test_no_step_pastes_an_expression_into_its_own_script():
    """Two reasons, and the second one is why this is a test rather than a convention.

    A `${{ }}` expansion is substituted into the script before bash starts, so a value containing
    a quote and a semicolon is a command on the runner with whatever that job's token can reach.
    It is also untestable here, because nothing outside GitHub can supply it, so a step that goes
    back to pasting its inputs silently drops out of every test above while still being driven by
    this harness through its `env:` block.
    """
    offenders = {}
    for step in _ACTION["runs"]["steps"]:
        script = step.get("run")
        if not script:
            continue
        pasted = re.findall(r"\$\{\{[^}]*\}\}", script)
        if pasted:
            offenders[step.get("name")] = pasted
    assert not offenders, (
        f"these steps paste an expression into their own script rather than passing it through "
        f"`env:`: {offenders}")


@pytest.mark.parametrize("name", ["Run the regression gate", "Check the artefacts",
                                  "Say what this run was asked to do",
                                  "Fail the build on the gate's verdict"])
def test_every_driven_step_still_exists(name):
    """`_step` raises on a rename, and a renamed step would otherwise take its tests with it."""
    assert _step(name)["run"].strip()
