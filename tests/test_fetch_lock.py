# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for the fetch lock (tools/fetch_model.py).

A sister project lost a day to a 987 MB fragment of a 5.16 GB checkpoint that passed an "exists
and is non-empty" check, and the cause was two fetchers writing one path. Hashing catches that
after the fact; the lock stops the second writer starting.
"""
import importlib.util
import multiprocessing
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_model", Path(__file__).resolve().parent.parent / "tools" / "fetch_model.py")
fm = importlib.util.module_from_spec(_SPEC)
sys.modules["fetch_model"] = fm
_SPEC.loader.exec_module(fm)


def test_the_lock_is_taken_and_released(tmp_path):
    d = str(tmp_path / "out")
    with fm.exclusive(d, log=lambda *a: None):
        assert (Path(d) / ".senbon-fetch.lock").exists()
    # Released, so a second acquisition succeeds.
    with fm.exclusive(d, log=lambda *a: None):
        pass


def test_the_lock_creates_the_destination(tmp_path):
    d = str(tmp_path / "does-not-exist-yet")
    with fm.exclusive(d, log=lambda *a: None):
        assert Path(d).is_dir()


def _hold(path, ready, done):
    spec = importlib.util.spec_from_file_location(
        "fetch_model", Path(__file__).resolve().parent.parent / "tools" / "fetch_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with mod.exclusive(path, log=lambda *a: None):
        ready.set()
        done.wait(10)


def test_a_second_fetcher_refuses_rather_than_joining_in(tmp_path):
    """The whole point. flock is per-process, so this needs a real second process."""
    pytest.importorskip("fcntl")
    ctx = multiprocessing.get_context("spawn")
    ready, done = ctx.Event(), ctx.Event()
    d = str(tmp_path / "out")
    p = ctx.Process(target=_hold, args=(d, ready, done))
    p.start()
    try:
        assert ready.wait(15), "the holder never acquired the lock"
        with pytest.raises(SystemExit) as e, fm.exclusive(d, log=lambda *a: None):
            pass
        msg = str(e.value)
        assert "already writing" in msg
        assert "truncated checkpoint" in msg
    finally:
        done.set()
        p.join(15)


def test_a_platform_without_locking_warns_instead_of_pretending(tmp_path, monkeypatch):
    real_import = __import__

    def no_fcntl(name, *a, **k):
        if name == "fcntl":
            raise ImportError("no fcntl here")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", no_fcntl)
    said = []
    with fm.exclusive(str(tmp_path / "out"), log=said.append):
        pass
    assert any("locking is unavailable" in s for s in said)
