# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura gate`: the command that makes a baseline do something.

`baseline.py` holds the rules and was written with no caller, which in this project is a green
suite waiting to happen: a correct function nobody invokes has been shipped twice here. These
tests drive the command, and the one that matters most is the wiring test, not the arithmetic.

EXIT STATUS IS THE PRODUCT. A build system reads the number before a person reads the words, so
each of the three is pinned separately, and the distinctness of 1 from 2 is pinned on its own,
because collapsing "it regressed" into "I could not compare these" is the failure that teaches a
reader to ignore both.
"""
import pytest

from senbonzakura import baseline as b
from senbonzakura import gate

COMMON = {
    "model": "Qwen/Qwen3-1.7B",
    "metric": "refusal_rate",
    "direction": b.LOWER_IS_BETTER,
    "input_digest": "sha256:aaa",
    "partition": "measure",
    "prompt_format": "renderer:v3",
    "tool_version": "0.4.0",
    "seeds": [42],
    "n": 200,
}


@pytest.fixture
def files(tmp_path):
    """A baseline plus three measurements: comparable and steady, regressed, and incomparable."""
    made = {}
    for name, over in (
        ("base", {"point": 0.10, "interval": (0.05, 0.18)}),
        ("steady", {"point": 0.12, "interval": (0.07, 0.20)}),
        ("worse", {"point": 0.40, "interval": (0.33, 0.47)}),
        ("better", {"point": 0.01, "interval": (0.00, 0.03)}),
        ("drifted", {"point": 0.12, "interval": (0.07, 0.20), "prompt_format": "renderer:v4"}),
    ):
        path = tmp_path / f"{name}.json"
        b.write(path, b.record(**{**COMMON, **over}))
        made[name] = path
    return made


def _run(files, which, *extra):
    lines = []
    status = gate.run(["--baseline", str(files["base"]),
                       "--measurement", str(files[which]), *extra], log=lines.append)
    return status, "\n".join(lines)


class TestTheThreeOutcomes:
    def test_a_steady_measurement_passes(self, files):
        status, said = _run(files, "steady")
        assert status == gate.OK
        assert "within interval" in said

    def test_a_regression_fails(self, files):
        status, said = _run(files, "worse")
        assert status == gate.REGRESSED
        assert "REGRESSED" in said

    def test_an_improvement_beyond_the_interval_still_passes(self, files):
        status, said = _run(files, "better")
        assert status == gate.OK
        assert "improved" in said

    def test_an_incomparable_measurement_is_refused_not_failed(self, files):
        """A refusal says nothing about the model, and a build that reads it as a regression
        teaches its reader that the two are the same thing.
        """
        status, said = _run(files, "drifted")
        assert status == gate.REFUSED
        assert "prompt_format" in said

    def test_the_three_statuses_are_distinct(self):
        assert len({gate.OK, gate.REGRESSED, gate.REFUSED}) == 3
        assert gate.OK == 0, "a shell reads anything non-zero as failure"


class TestWhatItSaysAndWhenItStaysQuiet:
    def test_a_pass_shows_the_interval_and_the_caveat(self, files):
        """Loopholes 3 and 5 together: how much room there was, and what a pass does not mean."""
        _, said = _run(files, "steady")
        assert "0.05" in said and "0.18" in said
        assert "not a statement that the model is safe" in said

    def test_quiet_shortens_a_pass(self, files):
        _, loud = _run(files, "steady")
        _, quiet = _run(files, "steady", "--quiet")
        assert len(quiet) < len(loud)
        assert "within interval" in quiet

    def test_quiet_does_not_shorten_a_failure(self, files):
        """A build log that hid why it failed would send its reader to the wrong place."""
        _, said = _run(files, "worse", "--quiet")
        assert "0.33" in said and "not a statement that the model is safe" in said

    def test_quiet_never_silences_a_refusal(self, files):
        """The alternative reading of a silent refusal is that nothing was wrong."""
        status, said = _run(files, "drifted", "--quiet")
        assert status == gate.REFUSED
        assert "prompt_format" in said


class TestUnreadableInputs:
    def test_a_missing_baseline_refuses_rather_than_crashes(self, files, tmp_path):
        lines = []
        status = gate.run(["--baseline", str(tmp_path / "absent.json"),
                           "--measurement", str(files["steady"])], log=lines.append)
        assert status == gate.REFUSED
        assert "no baseline at" in "\n".join(lines)

    def test_an_unparseable_measurement_refuses(self, files, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json", encoding="utf-8")
        lines = []
        status = gate.run(["--baseline", str(files["base"]),
                           "--measurement", str(bad)], log=lines.append)
        assert status == gate.REFUSED
        assert "not readable JSON" in "\n".join(lines)


class TestItIsActuallyReachable:
    def test_the_command_is_registered(self):
        """THE WIRING. `baseline.py` had no caller for an hour, and this is what would have
        caught that: a module can be perfect and unreachable, and the suite stays green.
        """
        from senbonzakura import entry
        assert entry.DELEGATED["gate"] == ("gate", "main")

    def test_dispatch_returns_the_entry_point(self):
        from senbonzakura import entry
        assert entry.dispatch("gate") is gate.main

    def test_main_returns_the_status_for_the_shell(self, files):
        assert gate.main(["--baseline", str(files["base"]),
                          "--measurement", str(files["worse"])]) == gate.REGRESSED

    def test_exit_status_passes_the_verdict_through_unchanged(self, files):
        """`exit_status` maps a command's return value to a shell status, and it once turned
        every successful run of five commands into a failure. A gate returning 1 must stay 1.
        """
        from senbonzakura import entry
        assert entry.exit_status(gate.REGRESSED) == 1
        assert entry.exit_status(gate.REFUSED) == 2
        assert entry.exit_status(gate.OK) == 0

    def test_it_pulls_nothing_heavy(self):
        """The gate runs on every change, so it has to be affordable on a CI runner.

        Asserted on the module's imports rather than by uninstalling torch, which the torch-free
        suite already does at a higher level.
        """
        import inspect
        source = inspect.getsource(gate)
        for heavy in ("import torch", "import optuna", "import transformers", "import datasets"):
            assert heavy not in source
