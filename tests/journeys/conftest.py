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
import pytest

#: THE GUARD GOES BEFORE THE IMPORT IT GUARDS, which is the whole of this defect.
#:
#: A journey drives the program through a pty, and `pty` is POSIX only. There was already a fixture
#: that skipped for exactly this reason, but a fixture runs after collection, and collection is
#: what imports this file, which imports the harness, which imports `pty`. So the skip never got
#: the chance to fire and Windows CI ended in four collection errors instead. A check placed
#: downstream of the thing it checks reports nothing.
#:
#: The probe is the import, not the platform and not `find_spec`: Windows ships `Lib/pty.py` and it
#: is found, so the only question anybody can answer honestly is whether it loads.
try:
    import pty as _pty
except ImportError:                                       # pragma: no cover - POSIX runs the suite
    collect_ignore_glob = ["*.py"]
else:
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
    """Journeys need a pty, and a machine without one never reaches here: the module-level guard
    above drops the whole directory from collection. This stays as the second layer, for a machine
    that imports `pty` but cannot fork one.
    """
    import os
    try:
        fds = _pty.openpty()
    except OSError:                                       # pragma: no cover - a pty-less POSIX box
        pytest.skip("this machine has `pty` but will not open one, so there is no terminal to drive")
    for fd in fds:
        os.close(fd)
