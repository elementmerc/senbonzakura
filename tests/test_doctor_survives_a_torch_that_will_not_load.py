# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
r"""`doctor` has to survive the most likely way an install is broken, because that is when it runs.

FOUND ON REAL WINDOWS HARDWARE, 2026-09-17, on the project's own laptop, in a disposable venv
built the way a stranger would build one.

    pip install senbonzakura          succeeded, both distributions, exit 0
    senbonzakura --version            worked, exit 0
    senbonzakura doctor               thirty line traceback, exit 1, no diagnosis

    OSError: [WinError 4551] An Application Control policy has blocked this file.
    Error loading ...\\torch\\lib\\shm.dll or one of its dependencies.

Smart App Control is ON by default on a new Windows 11 installation and blocks unsigned
binaries; torch ships many. The machine reported `VerifiedAndReputablePolicyState = 1`, which is
the stock setting, so this is the ordinary Windows experience rather than an exotic one.

WHY IT GOT THROUGH. Every guard here read `except ImportError`. A package that is absent raises
ImportError; a package that is PRESENT and whose native libraries will not load raises OSError,
which is not an ImportError and went straight past four handlers and out through `main`.

WHY IT MATTERS MORE THAN AN ORDINARY CRASH. `doctor` exists to say whether this install can do
the job, and a user runs it precisely because something is already wrong. Every cheap signal said
the install was fine: pip reported success, the version printed, the entry point parses without
importing torch by design. The one command built to find the fault was the one that could not
survive it. That is the same shape as the missing device pre-flight found a day earlier: the tool
knew, in the command built to say so, and nothing carried it to the user as a sentence.
"""
import builtins

import pytest

from senbonzakura import doctor


def _torch_raising(error):
    """Make `import torch` raise, and leave every other import alone."""
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "torch":
            raise error
        return real(name, *args, **kwargs)

    return fake


def _app_control_error():
    """The exact shape Windows raised, including the winerror the message keys on."""
    error = OSError(22, "An Application Control policy has blocked this file")
    error.winerror = doctor._WINDOWS_APP_CONTROL
    return error


@pytest.fixture
def torch_that_will_not_load(monkeypatch):
    monkeypatch.setattr(builtins, "__import__", _torch_raising(_app_control_error()))


class TestTheDiagnosisSurvivesTheThingItDiagnoses:

    def test_check_torch_reports_rather_than_raising(self, torch_that_will_not_load):
        result = doctor.check_torch()
        assert result.status == "fail", f"expected a failed check, got {result.status}"

    def test_it_says_the_package_is_present_rather_than_missing(self, torch_that_will_not_load):
        """"not installed" would send the user to reinstall something already there."""
        text = " ".join(str(v) for v in vars(doctor.check_torch()).values())
        assert "installed" in text and "not installed" not in text

    def test_it_names_smart_app_control(self, torch_that_will_not_load):
        text = " ".join(str(v) for v in vars(doctor.check_torch()).values())
        assert "Smart App Control" in text, (
            "the remedy does not name the thing doing the blocking, so the user cannot act on it")

    def test_it_warns_that_the_setting_is_one_way(self, torch_that_will_not_load):
        """Smart App Control cannot be re-enabled without reinstalling Windows.

        A remedy that says "turn it off" and omits that is advice the reader cannot take back.
        """
        text = " ".join(str(v) for v in vars(doctor.check_torch()).values())
        assert "cannot be turned back" in text

    def test_it_offers_the_option_that_costs_nothing_first(self, torch_that_will_not_load):
        text = " ".join(str(v) for v in vars(doctor.check_torch()).values())
        assert text.index("WSL2") < text.index("turn Smart App Control off"), (
            "the irreversible remedy is offered before the reversible one")

    def test_the_pinned_memory_check_degrades_rather_than_crashing(self, torch_that_will_not_load):
        """The second site. Fixing only the first moved the traceback eighty lines down."""
        result = doctor.check_pinned_memory()
        assert result.status in {"warn", "fail"}

    def test_a_plain_missing_torch_still_says_so(self, monkeypatch):
        """The original message must not be lost to the new branch."""
        monkeypatch.setattr(builtins, "__import__", _torch_raising(ImportError("no torch")))
        text = " ".join(str(v) for v in vars(doctor.check_torch()).values())
        assert "not installed" in text

    def test_an_unexplained_load_failure_still_reports_something_useful(self, monkeypatch):
        """Not every OSError is Smart App Control. A missing system library is the Linux case."""
        monkeypatch.setattr(builtins, "__import__",
                            _torch_raising(OSError("libgomp.so.1: cannot open shared object file")))
        result = doctor.check_torch()
        text = " ".join(str(v) for v in vars(result).values())
        assert result.status == "fail"
        assert "will not load" in text
        assert "Smart App Control" not in text, "a Linux fault was blamed on a Windows setting"


def test_run_checks_completes_end_to_end(torch_that_will_not_load):
    """THE PROPERTY THAT ACTUALLY MATTERS, and the one a per-function test cannot see.

    Fixing `check_torch` alone still left `doctor` dying, eighty lines further down, in
    `check_pinned_memory`. The thing the user needs is not that one check behaves; it is that the
    command RETURNS A REPORT. So this drives the whole sweep, which is what would have caught
    both sites at once.
    """
    checks = doctor.run_checks(deep=False)
    assert checks, "run_checks returned nothing"
    statuses = {c.status for c in checks}
    assert "fail" in statuses, "a torch that cannot load was not reported as a failure"
