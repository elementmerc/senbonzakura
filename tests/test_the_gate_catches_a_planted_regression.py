# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A planted regression, driven through a shell, because that is the number a build reads.

`tests/test_gate.py` already builds measurements through `baseline.record` and asserts all three
statuses, which proves the arithmetic. It proves nothing about the status a shell sees: it calls
`gate.run` in-process and reads its return value, and the whole history of this repository's gates
is that the wiring between a correct function and the thing that invokes it is where the defect
lives. A quiet mode once suppressed warnings over a false premise about which stream carried what;
an action once reported a clean run because three shell variables held the empty string.

So this module does the other half. The measurements are committed files rather than built in a
temporary directory, the command is a subprocess rather than a function call, and the assertion is
on `returncode` rather than on a value Python handed back.

THE STATUSES ARE NOT INTERCHANGEABLE, and each is pinned separately:

    0  the property did not move outside its interval
    1  it regressed
    2  the two measurements were never comparable, so nothing was shown about the model

A gate that collapses 1 into 2 teaches its reader to ignore both.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from senbonzakura import baseline as b
from senbonzakura import gate

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "gate"
BASELINE = FIXTURES / "baseline-refusal-rate.json"

#: The three measurements and the status each one must produce. Read off the files' own
#: `_provenance.expected_status` as well, below, so the table and the fixtures cannot drift.
EXPECTED = {
    "measurement-steady.json": gate.OK,
    "measurement-regressed.json": gate.REGRESSED,
    "measurement-incomparable.json": gate.REFUSED,
}


def _shell(measurement: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run the gate the way a build system runs it, and bound the wait.

    THE CHECKOUT IS PUT ON THE PATH DELIBERATELY, rather than relying on an install. The gate is
    meant to be affordable in any CI job, and `python -m senbonzakura gate` on a bare checkout is
    the cheapest way anybody will ever invoke it. Shadowing an installed copy is normally the
    defect this project chases, so it is stated rather than implied: the installed-console-script
    path is `tests/test_exit_status.py`'s job, and this module's job is the status a shell reads.

    A missing module would otherwise arrive here as exit 1, which is this command's code for a
    regression, so the import is checked separately below rather than left to look like a verdict.
    """
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = (src + os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else src
    return subprocess.run(
        [sys.executable, "-m", "senbonzakura", "gate",
         "--baseline", str(BASELINE), "--current", str(measurement), *extra],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120, check=False, env=env)


class TestTheShellSeesTheRightNumber:
    def test_the_subprocess_can_reach_the_command_at_all(self):
        """Checked first, because a missing module exits 1 and 1 is this command's REGRESSED.

        Without this, an environment where `senbonzakura` cannot be imported makes the planted
        regression pass for the wrong reason, and the only honest-looking test in the module is
        the one asserting a failure.
        """
        proc = _shell(FIXTURES / "measurement-steady.json")
        assert "No module named" not in proc.stderr, (
            "the gate could not be imported, so every status below would be the import's rather "
            f"than the gate's:\n{proc.stderr}")

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_each_measurement_produces_its_own_status(self, name):
        proc = _shell(FIXTURES / name)
        assert proc.returncode == EXPECTED[name], (
            f"{name} reached the shell as {proc.returncode} rather than {EXPECTED[name]}.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    def test_the_planted_regression_names_what_moved(self):
        """A CI failure that does not say which property moved is a failure somebody disables
        rather than investigates.
        """
        proc = _shell(FIXTURES / "measurement-regressed.json")
        assert "REGRESSED" in proc.stdout, proc.stdout
        assert "refusal_rate" in proc.stdout, proc.stdout
        assert "0.1900" in proc.stdout, (
            "the failure did not say by how much the property moved:\n" + proc.stdout)

    def test_the_pass_prints_how_much_room_there_was(self):
        """Loophole 3. A gate whose interval is wide enough that nothing trips it reads as safety,
        and the only way a reader can tell the difference is by seeing the interval.
        """
        proc = _shell(FIXTURES / "measurement-steady.json")
        assert "within interval" in proc.stdout, proc.stdout
        assert "0.061" in proc.stdout and "0.133" in proc.stdout, (
            "a pass was printed without the baseline's interval:\n" + proc.stdout)

    def test_the_refusal_names_the_field_that_disagreed(self):
        proc = _shell(FIXTURES / "measurement-incomparable.json")
        assert "REFUSED" in proc.stdout, proc.stdout
        assert "prompt_format" in proc.stdout, proc.stdout
        assert "renderer:v3" in proc.stdout and "renderer:v4" in proc.stdout, proc.stdout

    def test_a_refusal_is_never_quietable(self):
        """The alternative reading of a silent refusal is that nothing was wrong."""
        proc = _shell(FIXTURES / "measurement-incomparable.json", "--quiet")
        assert proc.returncode == gate.REFUSED
        assert "prompt_format" in proc.stdout, proc.stdout


class TestTheFixturesCannotDriftFromTheTest:
    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_the_file_states_the_status_this_module_expects(self, name):
        """The table above and the files are two statements of one fact, so they are compared.

        Without this, renaming a fixture's role or regenerating it with different numbers leaves
        the test asserting a status the file was no longer built to produce, and a green run.
        """
        doc = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        stated = doc["extra"]["_provenance"]["expected_status"]
        assert stated == EXPECTED[name], (
            f"{name} says it should produce status {stated} and this module expects "
            f"{EXPECTED[name]}")

    @pytest.mark.parametrize("name", sorted([*EXPECTED, BASELINE.name]))
    def test_every_fixture_is_what_the_writer_would_have_written(self, name):
        """A hand-edited baseline passes the reader and would never have passed the writer.

        `baseline.read` validates the schema string and little else, so a fixture edited in place
        to produce a convenient number can carry a shape `record` refuses. Rebuilding each file
        from its own fields catches that, and catches a field this project later makes mandatory
        quietly staying absent here.
        """
        doc = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        rebuilt = b.record(
            model=doc["model"], metric=doc["metric"], direction=doc["direction"],
            point=doc["point"],
            interval=None if doc["deterministic"] else tuple(doc["interval"]),
            deterministic=doc["deterministic"],
            input_digest=doc["input_digest"], partition=doc["partition"],
            prompt_format=doc["prompt_format"], tool_version=doc["tool_version"],
            estimator=doc["estimator"], precision=doc["precision"],
            seeds=doc["seeds"], n=doc["n"], extra=doc["extra"])
        assert rebuilt == doc, f"{name} is not what `baseline.record` produces from its own fields"

    @pytest.mark.parametrize("name", sorted([*EXPECTED, BASELINE.name]))
    def test_every_fixture_says_nobody_measured_it(self, name):
        """These numbers describe no run, and the file says so rather than only the README,
        because a file travels and a README does not.
        """
        doc = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        prov = doc["extra"]["_provenance"]
        assert prov["kind"] == "invented", (
            f"{name} claims a provenance stronger than invented, and these figures describe "
            f"no run")
        assert prov["not_an_incident"].strip(), f"{name} does not say what it may not be quoted as"
