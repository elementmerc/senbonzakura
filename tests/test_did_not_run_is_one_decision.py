# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The third outcome, and the one switch that decides whether it is yellow or red.

WHAT Q-71 ACTUALLY DECIDED

A check has three outcomes: PASS, FAIL and DID NOT RUN. GitHub gives a plain workflow job four
conclusions and no neutral, so DID NOT RUN has to borrow one of the other two. It borrows success
today and shouts about it, which is right while a person reads the run and wrong the day a
required status check reads it instead, because a branch rule reads green as "verified" and cannot
see an annotation.

So the posture is a switch rather than a position, and this file is what makes the switch real
rather than documented. Two surfaces carry it and both are exercised here:

  tools/ci/did_not_run.sh    the workflow-level sites (a binary that would not download, an
                             unreachable index), via REQUIRE_EVERY_CHECK_TO_RUN
  tools/ci/audit_env.py      the three dependency audits, via --required

THE PROPERTY WORTH TESTING IS NOT "IT PRINTS SOMETHING"

It is that the exit status changes with the switch and the wording does not. A reader comparing
two runs should be able to tell that the consequence changed without having to work out whether
the finding did, and a caller that forgets to wire the switch through must not quietly inherit
the permissive branch, which is the failure mode every one of this project's recorded gate
defects shares.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools" / "ci" / "did_not_run.sh"
AUDIT = ROOT / "tools" / "ci" / "audit_env.py"

TIMEOUT_S = 60


def _helper(tmp_path, *args, flag="unset"):
    """Run the helper with the switch set, unset or malformed, and return (status, output, summary)."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    if flag != "unset":
        env["REQUIRE_EVERY_CHECK_TO_RUN"] = flag
    summary = tmp_path / "summary.md"
    summary.touch()
    env["GITHUB_STEP_SUMMARY"] = str(summary)
    proc = subprocess.run(
        ["bash", str(HELPER), *args],
        capture_output=True, text=True, env=env, timeout=TIMEOUT_S, check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr, summary.read_text(encoding="utf-8")


def test_the_helper_exists_and_is_executable():
    """A helper the workflow calls by path has to be present and runnable in a fresh clone."""
    assert HELPER.is_file(), f"{HELPER} is missing, so every call site in the workflow fails"
    assert HELPER.stat().st_mode & 0o111, (
        f"{HELPER} is not executable. The workflow invokes it directly rather than through "
        f"`bash`, so the mode bit is load-bearing and does survive a git checkout.")


def test_off_is_green_and_loud(tmp_path):
    """The posture today: report it, name it, and do not fail the build."""
    status, out, summary = _helper(tmp_path, "actionlint", "the workflows were NOT linted",
                                   flag="false")
    assert status == 0, f"the permissive branch failed the caller: {out}"
    assert "::warning::" in out, f"nothing would appear beside the run: {out!r}"
    assert "DID NOT RUN" in out and "actionlint" in out, (
        f"the annotation does not say what did not run: {out!r}")
    assert "did not check anything" in summary, (
        f"the summary does not say the green tick verified nothing: {summary!r}")


def test_on_is_red_with_the_same_sentence(tmp_path):
    """The posture after branch protection lands, and the wording must not drift with it."""
    off_status, off_out, _ = _helper(tmp_path, "actionlint", "the workflows were NOT linted",
                                     flag="false")
    on_status, on_out, on_summary = _helper(tmp_path, "actionlint",
                                            "the workflows were NOT linted", flag="true")
    assert off_status == 0 and on_status == 1, (
        f"the switch did not change the exit status: off={off_status}, on={on_status}")
    assert "::error::" in on_out, f"the strict branch is still only a warning: {on_out!r}"
    assert "fails" in on_summary, (
        f"the summary does not say the job fails: {on_summary!r}")

    # The FACT is the same fact. Only its consequence changes, and that is deliberate.
    fact = "actionlint did not run, so the workflows were NOT linted"
    assert fact in off_out and fact in on_out, (
        f"the two postures describe the same event differently, which leaves a reader working "
        f"out whether the wording moved or the finding did.\noff: {off_out!r}\non: {on_out!r}")


def test_an_unset_switch_is_refused_rather_than_defaulted(tmp_path):
    """The one that matters most, because it is the failure that looks like success.

    A caller that forgets to pass the environment through would, under a default, silently get
    the permissive branch and no sign that it had chosen it. Every recorded gate defect in this
    project has that shape: the narrower question answered, reported clean.
    """
    status, out, _ = _helper(tmp_path, "actionlint", "the workflows were NOT linted", flag="unset")
    assert status == 1, f"an unset switch was treated as a posture rather than as a mistake: {out}"
    assert "REQUIRE_EVERY_CHECK_TO_RUN" in out, (
        f"the refusal does not name the variable somebody has to set: {out!r}")


@pytest.mark.parametrize("value", ["1", "yes", "TRUE", "on", " true"])
def test_a_value_that_is_neither_true_nor_false_is_refused(tmp_path, value):
    """Guessing which was meant is how a safety switch ends up half on."""
    status, out, _ = _helper(tmp_path, "actionlint", "not linted", flag=value)
    assert status == 1, f"{value!r} was accepted as a posture: {out}"
    assert "Refusing to guess" in out, f"the refusal does not say why: {out!r}"


@pytest.mark.parametrize("args", [
    (),
    ("actionlint",),
    ("actionlint", "not linted", "extra"),
])
def test_the_wrong_number_of_arguments_is_refused(tmp_path, args):
    """A miswired call must not produce a half-written annotation and exit 0."""
    status, out, _ = _helper(tmp_path, *args, flag="false")
    assert status == 1, f"{args!r} produced exit 0: {out}"
    assert "two arguments" in out, f"the refusal does not say what the contract is: {out!r}"


@pytest.mark.parametrize("args", [("", "not linted"), ("actionlint", "")])
def test_an_empty_argument_is_refused(tmp_path, args):
    """An annotation that names neither the tool nor the gap tells a reader nothing."""
    status, out, _ = _helper(tmp_path, *args, flag="false")
    assert status == 1, f"{args!r} produced exit 0: {out}"
    assert "empty" in out, f"the refusal does not say which argument was empty: {out!r}"


def test_it_works_with_no_step_summary(tmp_path):
    """Outside a runner there is no GITHUB_STEP_SUMMARY, and the helper must not die on it.

    This is how the helper gets run by hand while somebody is debugging a workflow, which is
    exactly when a crash in the reporting path is most expensive.
    """
    proc = subprocess.run(
        ["bash", str(HELPER), "actionlint", "not linted"],
        capture_output=True, text=True, timeout=TIMEOUT_S, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "REQUIRE_EVERY_CHECK_TO_RUN": "false"},
    )
    assert proc.returncode == 0, f"{proc.stdout}{proc.stderr}"
    assert "::warning::" in proc.stdout


# ── the dependency audit's half of the same switch ───────────────────────────────────────────────

def _audit(tmp_path, *args):
    summary = tmp_path / "summary.md"
    summary.touch()
    proc = subprocess.run(
        [sys.executable, str(AUDIT), *args],
        capture_output=True, text=True, timeout=TIMEOUT_S, check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "GITHUB_STEP_SUMMARY": str(summary)},
    )
    return proc.returncode, proc.stdout + proc.stderr, summary.read_text(encoding="utf-8")


@pytest.mark.parametrize("value", ["", "1", "yes", "TRUE"])
def test_the_audit_refuses_a_required_value_it_does_not_recognise(tmp_path, value):
    """The empty string is the case that matters: it is what an unset workflow variable becomes.

    The call site is `--required "$AUDIT_REQUIRED"`, quoted so shellcheck is satisfied, which
    means an unset variable arrives as an empty argument rather than disappearing. argparse has to
    refuse it by name rather than fall back to the permissive branch.
    """
    status, out, _ = _audit(tmp_path, "--label", "x", "--min-packages", "1", "--required", value)
    assert status != 0, f"--required {value!r} was accepted: {out}"
    assert "--required" in out and "invalid choice" in out, (
        f"the refusal does not name the option and the bad value: {out!r}")


def _audit_module():
    """The tool imported as a module, so the unreachable branch can be forced rather than waited for.

    Driving it by subprocess would make the test depend on whether the machine running it happens
    to have a network, which is the opposite of a test: it would pass here and exercise nothing on
    a runner, and exercise nothing here on a machine that does have one.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_audit_env_under_test", AUDIT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("flag", "expected", "level"), [("false", 0, "::warning::"),
                                                         ("true", 1, "::error::")])
def test_an_unreachable_advisory_service_follows_the_switch(tmp_path, monkeypatch, capsys,
                                                            flag, expected, level):
    """The branch the switch exists for, forced rather than waited for.

    This is the hole D33 described: with the switch off the job is green having audited nothing,
    which a person can see from the annotation and a branch rule cannot. With it on, the same
    event is a failure.
    """
    module = _audit_module()
    monkeypatch.setattr(module, "service_reachable", lambda *_a, **_k: False)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))

    status = module.main(["--label", "a test environment", "--min-packages", "1",
                          "--required", flag])
    out = capsys.readouterr().out
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")

    assert status == expected, f"--required {flag} returned {status}:\n{out}"
    assert level in out, f"the annotation level does not match the posture: {out!r}"
    assert "DID NOT RUN" in out and "a test environment" in out, (
        f"the annotation does not name the environment that went unaudited: {out!r}")
    assert "audited nothing" in summary or "fails" in summary, (
        f"the summary does not state the consequence: {summary!r}")


def test_the_audits_default_posture_is_the_permissive_one(tmp_path):
    """Q-71 chose green-plus-warning NOW, so the default has to be that and not the other one."""
    status, out, _ = _audit(tmp_path, "--help")
    assert status == 0, out
    assert "--required {true,false}" in out, (
        f"--required is not a true/false option any more, so the call sites that quote it are "
        f"wrong: {out!r}")
