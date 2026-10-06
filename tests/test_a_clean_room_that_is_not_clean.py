# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A clean room that is not clean is the one kind of test whose pass means nothing.

TWO DEFECTS, ONE FILE, AND THEY ARE THE SAME DEFECT TWICE.

The first: the harness on one machine shadowed an installed 0.3.0 with a `PYTHONPATH` pointing at
a checkout, so what it tested was not what it installed. Every check then described the checkout,
including the ones about what a wheel ships.

The second: the checks reported every binary that would not start as "a packaging fault", and the
commonest reason a vendored `llama-quantize` will not start in a slim image is that the image has
no `libgomp1`. The binary is correct. `doctor` says so, and the clean room contradicted it.

Both are a check answering a narrower question than the one it was asked and reporting the answer
with full confidence, which is the failure shape this repository keeps finding in its own gates.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "ci" / "clean_room_checks.py"

#: The refusal's own status. One means the install misbehaved; two means the room was not sealed.
NOT_A_CLEAN_ROOM = 2


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("clean_room_checks", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _run(**env_overrides):
    """Run the harness with a chosen environment, from a directory that is not the checkout.

    `cwd` matters: run from the repository root, `senbonzakura` would resolve out of `./src` on an
    editable install and the origin check would fire for a reason this test is not about.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update({k: v for k, v in env_overrides.items() if v is not None})
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True,
                          cwd=str(ROOT.parent), env=env, timeout=180, check=False)


class TestItRefusesRatherThanRepairingItself:
    def test_a_shadowing_pythonpath_stops_the_run(self):
        proc = _run(PYTHONPATH=str(ROOT / "src"))
        assert proc.returncode == NOT_A_CLEAN_ROOM, (
            f"exit {proc.returncode}, so a shadowed run was reported as an ordinary result:\n"
            f"{proc.stdout}")
        assert "REFUSED" in proc.stdout, proc.stdout
        assert "PYTHONPATH" in proc.stdout, proc.stdout
        assert "nothing was checked" in proc.stdout, proc.stdout

    def test_it_does_not_unset_the_variable_for_you(self):
        """The decision, and the only part of this that changes anybody's behaviour.

        Quietly dropping the variable makes the run correct and leaves the caller believing they
        measured the thing they pointed it at, so the next person wires it up the same way.
        """
        proc = _run(PYTHONPATH=str(ROOT / "src"))
        assert "Nothing here was unset for you" in proc.stdout, proc.stdout

    def test_no_check_result_is_printed_at_all(self):
        """A refusal that still prints thirty results invites somebody to read them."""
        proc = _run(PYTHONPATH=str(ROOT / "src"))
        assert "  ok    " not in proc.stdout, proc.stdout
        assert "  FAIL  " not in proc.stdout, proc.stdout

    def test_the_refusal_names_the_variable_rather_than_the_symptom(self):
        """A reader who is told "this is not a clean room" and not which setting made it one has
        to guess, and the guess is usually that the harness is broken.
        """
        proc = _run(PYTHONPATH=str(ROOT / "src"))
        assert str(ROOT / "src") in proc.stdout, proc.stdout
        assert "shadows the installed wheel" in proc.stdout, proc.stdout

    def test_the_refusal_is_told_apart_from_a_failing_install(self):
        """Exit 2, not 1. The gate this project ships makes exactly that distinction, and a
        caller that treats them alike learns to ignore both.
        """
        proc = _run(PYTHONPATH=str(ROOT / "src"))
        assert proc.returncode != 1, (
            "the room being unsealed was reported with the status that means the install "
            "misbehaved")
        assert "Exit 2" in proc.stdout, proc.stdout


class TestWhoseFaultAQuantiserThatWillNotStartIs:
    @pytest.mark.parametrize("verdict", [
        ("✗ llama-quantize  present at /x/llama-quantize and cannot start: a shared library "
         "is missing (libgomp.so.1)"),
        "✗ llama-quantize  error while loading shared libraries: libgomp.so.1",
        "✗ llama-quantize  CANNOT OPEN SHARED OBJECT file",
    ])
    def test_a_missing_library_is_the_images_fault(self, module, verdict):
        assert module.blames_the_image(verdict), verdict

    @pytest.mark.parametrize("verdict", [
        "✗ llama-quantize  ran and printed no usage; the binary is not what we think",
        "✗ llama-quantize  present at /x/llama-quantize and will not run (OSError)",
        "✓ llama-quantize  vendored, runs",
        "",
    ])
    def test_everything_else_is_not(self, module, verdict):
        assert not module.blames_the_image(verdict), verdict

    def test_more_than_one_phrasing_is_recognised(self, module):
        """`doctor` has several ways to say a binary could not start, and chasing prose one
        phrase at a time is what made the earlier version of this wrong.
        """
        assert len(module.IMAGE_IS_AT_FAULT) >= 2
