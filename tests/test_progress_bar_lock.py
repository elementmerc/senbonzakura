# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The progress bars use a thread lock, and we find out at test time if tqdm moves the API.

WHY THE FIX EXISTS

`tqdm` builds a global write lock on the first progress bar, and it builds a
`multiprocessing.RLock`: one POSIX semaphore, registered with the resource tracker, held as a class
attribute and therefore never released. Interrupt a run and the tracker prints

    resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown

Traced 2026-09-27 to `tqdm/std.py`'s `create_mp_lock`, by instrumenting
`resource_tracker.register`, after a reader interrupted a two-hour run and met it. The bars belong
to `transformers` and `huggingface_hub`, not to this package, and nothing actually leaks: the tracker
says it cleaned up and it did.

It is still worth fixing, because this project claims an interrupted run leaves the dataset
recoverable and the temp directories wiped, and that claim is TRUE. The same reader checked and found
no partial weights and no `.part` files. A warning about leaked resources at the moment somebody
kills a long run undercuts a property that holds, and the reader cannot tell which.

WHY THIS FILE IS THE INTERESTING HALF

The fix reaches into a third-party global and swallows any error, because suppressing a cosmetic
warning must never stop an abliteration from starting. A swallowed error is exactly how a fix rots
into a no-op that nobody notices, so the operator asked for something that tells us when tqdm
changes. These tests are that: they assert the API the fix depends on still exists and still does
what the fix assumes, so a tqdm release that moves it fails here rather than quietly restoring the
warning years later.
"""
import multiprocessing

import pytest

tqdm_module = pytest.importorskip("tqdm", reason="tqdm arrives with transformers")


class TestTheApiTheFixDependsOn:
    """Three assumptions, each pinned, so a tqdm change names which one moved."""

    def test_set_lock_still_exists_and_is_callable(self):
        assert callable(getattr(tqdm_module.tqdm, "set_lock", None)), (
            "tqdm.tqdm.set_lock has gone. The semaphore-warning fix in cli.py depends on it and "
            "swallows failures, so without this test the warning would come back unnoticed.")

    def test_get_lock_still_exists(self):
        """The reader half, which is how the test below can check the fix took effect."""
        assert callable(getattr(tqdm_module.tqdm, "get_lock", None)), (
            "tqdm.tqdm.get_lock has gone, so nothing can verify which lock is installed")

    def test_the_default_lock_is_still_the_multiprocessing_kind(self):
        """The reason the fix is needed at all.

        If a future tqdm stops building a `multiprocessing.RLock` by default, the warning goes away
        on its own and the two lines in `cli.py` become dead weight worth deleting. This test failing
        is GOOD NEWS and the message says so, because a fix nobody can tell is unnecessary stays
        forever.
        """
        import inspect

        # READ OFF THE METHOD'S SOURCE, not off a class attribute: `mp_lock` is set lazily by
        # `create_mp_lock` and does not exist until the first progress bar, so `hasattr` answers
        # False on a fresh interpreter and says nothing about what tqdm would do. That was this
        # test's own first bug.
        source = inspect.getsource(tqdm_module.std.TqdmDefaultWriteLock.create_mp_lock)
        source_has_mp = "RLock" in source
        assert source_has_mp, (
            "tqdm no longer appears to build a multiprocessing lock for its write lock. If that is "
            "true, the semaphore warning cannot happen any more and "
            "`_progress_bars_use_a_thread_lock` in cli.py is dead weight: delete it and this file.")


class TestTheFixTakesEffect:
    def test_it_installs_a_thread_lock(self):
        """The observable behaviour: after calling it, the lock is not a POSIX semaphore."""
        from senbonzakura.cli import _progress_bars_use_a_thread_lock

        _progress_bars_use_a_thread_lock()
        installed = tqdm_module.tqdm.get_lock()
        # tqdm wraps what it is given in a TqdmDefaultWriteLock, so the lock lives in `locks`.
        held = getattr(installed, "locks", [installed])
        assert held, "tqdm reported a lock holding nothing"
        for lock in held:
            assert not isinstance(lock, type(multiprocessing.RLock())), (
                f"{lock!r} is a multiprocessing lock, which is the POSIX semaphore this fix exists "
                f"to avoid")

    def test_it_is_safe_to_call_twice(self):
        """It runs at the top of every run, and a resumed run calls it again in a fresh process."""
        from senbonzakura.cli import _progress_bars_use_a_thread_lock

        _progress_bars_use_a_thread_lock()
        _progress_bars_use_a_thread_lock()

    def test_it_never_raises_when_tqdm_misbehaves(self, monkeypatch):
        """The swallow is deliberate and is worth a test, because it is load bearing.

        A two-hour abliteration must not fail to start because a cosmetic warning could not be
        suppressed. This asserts the swallow catches rather than that it is absent.
        """
        from senbonzakura import cli

        def explode(*_args, **_kwargs):
            raise RuntimeError("set_lock moved house")

        monkeypatch.setattr(tqdm_module.tqdm, "set_lock", explode)
        cli._progress_bars_use_a_thread_lock()

    def test_help_does_not_pay_for_it(self):
        """`--help` was taken from 2.80s to 0.06s by keeping imports out of that path.

        The fix is called from `run_parsed`, not from `main`, so parsing a help request never imports
        tqdm. Asserted on the call graph rather than by timing, which would be flaky.
        """
        import inspect

        from senbonzakura import cli

        main_source = inspect.getsource(cli.main)
        assert "_progress_bars_use_a_thread_lock" not in main_source, (
            "the lock fix is called from main(), so `senbonzakura --help` now imports tqdm. Call it "
            "from run_parsed, which only runs once there is real work to do.")
        assert "_progress_bars_use_a_thread_lock()" in inspect.getsource(cli.run_parsed), (
            "the lock fix is no longer called from run_parsed, so nothing installs the thread lock")
