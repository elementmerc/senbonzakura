# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The history secret scan blocks, and the proof is the script running on a planted finding.

WHY THIS IS NOT A GREP

Q-75 flipped `.github/workflows/supply-chain.yml`'s `secrets` job from reporting to blocking, and
the whole change is one `exit 1` inside a `case` branch. A test that searched the file for the
string `exit 1` would pass on a file where that line sat in the wrong branch, or after an `echo`
that had already swallowed the status, and this repository has met both shapes: a non-zero status
in an `elif` condition is just "false" and does not trip `set -e`, and a pipeline's exit status is
the last command's unless `PIPESTATUS` is read.

So this extracts the step's actual `run:` script and EXECUTES it, under the same shell options
GitHub uses, with a stub `gitleaks` that returns each of the three statuses the branch
distinguishes. A script that reports a finding and exits 0 fails here.

WHAT IS STUBBED AND WHY IT IS HONEST TO STUB IT

Two commands, both for reasons that are about the harness rather than about the gate:

  gitleaks   the real binary is fetched in CI at a pinned sha256 and is not a test dependency.
             The branch under test is selected purely by its exit status, so the stub controls
             exactly the input the gate reads.
  git        the script asserts `git rev-list --count HEAD` is at least 100, to catch a shallow
             clone. Building a 100-commit repository per case would cost seconds per test for no
             extra coverage, so the count is supplied and the shallow-clone branch gets a case of
             its own with a small number.

Everything else in the script is the real text from the real file.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is needed to read the workflow as data")

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "supply-chain.yml"

#: The step whose script is under test, matched on its name in the workflow.
STEP = "Scan every commit"

#: Long enough to catch a wedge, short enough that a wedged test is not mistaken for a slow one.
TIMEOUT_S = 60


@pytest.fixture(scope="module")
def workflow():
    if not WORKFLOW.is_file():
        pytest.skip("no .github/workflows/supply-chain.yml in this checkout")
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), type(doc)
    return doc


@pytest.fixture(scope="module")
def secrets_job(workflow):
    job = (workflow.get("jobs") or {}).get("secrets")
    assert job, f"no `secrets` job in the workflow; jobs are {sorted(workflow.get('jobs') or {})}"
    return job


@pytest.fixture(scope="module")
def scan_script(secrets_job):
    """The real `run:` text of the scanning step, so nothing here tests a paraphrase."""
    for step in secrets_job.get("steps") or []:
        if step.get("name") == STEP:
            script = str(step.get("run") or "")
            assert script.strip(), f"the {STEP!r} step has an empty run block"
            return script
    names = [s.get("name") for s in secrets_job.get("steps") or []]
    pytest.fail(f"no step named {STEP!r} in the secrets job; found {names}")
    return ""  # unreachable; keeps the return type honest for a reader


def _run(script, tmp_path, *, gitleaks_status, commits=854, write_report=True):
    """Execute the step's script with the two commands stubbed, and return the completed process.

    `bash -e -o pipefail` is what GitHub wraps a `run:` block in. Running it under anything more
    forgiving would let a script pass here and fail there, or the reverse, which makes the test a
    different question from the one the gate answers.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "rev-list" ]; then echo {commits}; exit 0; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    (bin_dir / "git").chmod(0o755)

    (tmp_path / "gitleaks").write_text(
        "#!/bin/sh\n"
        f"exit {gitleaks_status}\n",
        encoding="utf-8",
    )
    (tmp_path / "gitleaks").chmod(0o755)

    if write_report:
        # Two findings, so a test asserting the count cannot pass on an empty list.
        (tmp_path / "gitleaks.json").write_text(
            json.dumps([{"RuleID": "generic-api-key"}, {"RuleID": "aws-access-token"}]),
            encoding="utf-8",
        )

    summary = tmp_path / "step-summary.md"
    summary.touch()
    script_path = tmp_path / "step.sh"
    script_path.write_text(script, encoding="utf-8")

    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", str(script_path)],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "GITHUB_STEP_SUMMARY": str(summary),
            "HOME": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )


def test_the_job_no_longer_describes_itself_as_reporting_only(secrets_job):
    """The name is what a reader sees in the checks list, so it has to match the behaviour.

    A blocking gate labelled "reporting only" is worse than either: somebody who sees it red
    assumes the label and looks for the real cause somewhere else.
    """
    name = str(secrets_job.get("name") or "")
    assert "reporting" not in name.lower(), (
        f"the secrets job is named {name!r} while its scan step blocks. One of the two is wrong.")


def test_a_finding_fails_the_step(scan_script, tmp_path):
    """THE WHOLE POINT. gitleaks exits 1 when it finds something, and the step must not survive it."""
    proc = _run(scan_script, tmp_path, gitleaks_status=1)
    assert proc.returncode != 0, (
        "gitleaks reported findings and the step exited 0, so the secret scan is reporting rather "
        f"than blocking.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "::error::" in proc.stdout, (
        f"a finding produced no error annotation, so it would not be visible: {proc.stdout!r}")
    assert "::warning::" not in proc.stdout, (
        "a finding is still emitting a warning annotation, which renders as yellow beside a red "
        f"job and reads as though the finding were advisory: {proc.stdout!r}")


def test_the_failure_message_carries_the_recovery_path(scan_script, tmp_path):
    """A gate that blocks without saying what to do next gets bypassed, not obeyed.

    Three things have to be reachable from the run itself, because a reader who has to find a
    document first will reach for the bypass instead: where the evidence is, how to deal with a
    real secret, and how to deal with a fixture, including that the allowance expires.
    """
    proc = _run(scan_script, tmp_path, gitleaks_status=1)
    summary = (tmp_path / "step-summary.md").read_text(encoding="utf-8")
    message = proc.stdout + summary

    for needle, why in (
        ("gitleaks-report", "the artefact holding the evidence is not named"),
        ("rotate", "a real secret needs rotating first and the message does not say so"),
        (".gitleaksignore", "the ignore file is not named, so a fixture has no documented route"),
        ("expire", "the message does not say an ignore entry expires"),
        ("commit:path:rule:line", "the fingerprint format an entry needs is not given"),
    ):
        assert needle in message, f"{why}. Message was:\n{message}"


def test_the_report_is_still_uploaded_when_the_scan_fails(secrets_job):
    """The error message sends a reader to an artefact, so the upload cannot be skipped on failure.

    A step with a plain `if: steps.fetch.outputs.ran == 'true'` is skipped once an earlier step in
    the job has failed, which is precisely the run where the artefact is wanted. This asserts the
    condition carries `always()`, which is the only thing that keeps the recovery path reachable.
    """
    uploads = [s for s in secrets_job.get("steps") or []
               if "upload-artifact" in str(s.get("uses") or "")]
    assert uploads, "the secrets job uploads no report, so a finding leaves nothing to read"
    for step in uploads:
        condition = str(step.get("if") or "")
        assert "always()" in condition, (
            f"the report upload is gated on {condition!r}, so it is skipped on the run where the "
            f"scan failed, which is the only run whose report anybody needs.")


def test_a_clean_history_still_passes(scan_script, tmp_path):
    """The positive control. A gate that fails on everything is noise, not a gate."""
    proc = _run(scan_script, tmp_path, gitleaks_status=0, write_report=False)
    assert proc.returncode == 0, (
        f"a clean scan failed the step:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "PASS" in proc.stdout, f"a clean scan said nothing affirmative: {proc.stdout!r}"


def test_the_tool_failing_is_not_a_pass(scan_script, tmp_path):
    """DID NOT RUN is the third outcome and it is as loud as a finding.

    An exit status other than 0 or 1 means gitleaks itself failed: a bad flag, a corrupt
    repository, an out-of-memory kill. None of those is evidence that the history is clean, and
    this repository has already shipped the two-outcome version of this mistake, where a tool
    printing usage and exiting 2 was read as zero findings.
    """
    proc = _run(scan_script, tmp_path, gitleaks_status=2, write_report=False)
    assert proc.returncode != 0, (
        f"gitleaks exited 2 and the step passed:\nstdout:\n{proc.stdout}")
    assert "DID NOT RUN" in proc.stdout, (
        f"a tool failure was not named as DID NOT RUN: {proc.stdout!r}")


def test_a_shallow_clone_is_refused_before_anything_is_scanned(scan_script, tmp_path):
    """The defect this job was written for: a scan of one commit reporting no leaks reads clean.

    The 2026-09-25 claim that nothing had leaked was checked against the working tree, where the
    files were absent, rather than the history, where they remained. A shallow clone reproduces
    that exactly, so the count assertion has to fire before the scan rather than after it.
    """
    proc = _run(scan_script, tmp_path, gitleaks_status=0, commits=1, write_report=False)
    assert proc.returncode != 0, (
        f"a one-commit checkout was scanned and passed:\nstdout:\n{proc.stdout}")
    assert "fetch-depth" in proc.stdout, (
        f"the shallow-clone refusal does not name the cause a reader has to fix: {proc.stdout!r}")


def test_an_unparseable_report_still_fails(scan_script, tmp_path):
    """The count is read with a fallback, and the fallback must not become an escape hatch.

    `n="$(python3 -c ... || echo ...)"` cannot fail the script, by construction. So the thing to
    prove is that the `exit 1` sits outside that substitution: a finding with a report nobody can
    parse is still a finding.
    """
    proc = _run(scan_script, tmp_path, gitleaks_status=1, write_report=False)
    assert proc.returncode != 0, (
        f"a finding with no readable report passed the step:\nstdout:\n{proc.stdout}")
    assert "unknown number" in proc.stdout, (
        f"the count fell back without saying it had: {proc.stdout!r}")
