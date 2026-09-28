# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Fixtures for the journeys. The harness is in `harness.py`; this is the wiring.

Journeys are deselected from an ordinary `pytest` run by `-m "not journey"` in the project's
addopts, and run with `pytest -m journey`. They are slower than unit tests by two orders of
magnitude, because each one starts the real program on a real terminal, and a suite people avoid
running is a suite that rots.
"""
import shutil

import pytest

from .harness import drive


@pytest.fixture
def session(tmp_path):
    """Run the tool, and return what a person would have seen.

    Every session gets its own working directory and its own Hugging Face cache, so no journey can
    see what another left behind and none of them can reach the network.
    """
    def _run(*argv, answers=(), **kw):
        return drive(tmp_path, list(argv), answers, **kw)

    return _run


@pytest.fixture
def work(tmp_path):
    """The directory the tool is run in, so a journey can prepare what it expects to find."""
    path = tmp_path / "work"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture(autouse=True)
def _a_terminal_and_nothing_else():
    """Journeys need a pty, which is POSIX. On anything else they skip rather than lie."""
    if not hasattr(shutil, "get_terminal_size"):          # pragma: no cover - shape guard
        pytest.skip("no terminal support")
    pytest.importorskip("pty", reason="journeys drive a real terminal")
