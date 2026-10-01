# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura track` says it cannot write a track before it reads anybody's prompts.

WHAT PROMPTED IT, 2026-10-01

Driven through the real entry point on a machine with neither pyarrow nor datasets installed,
`senbonzakura track --harmful h.txt --harmless n.txt --out tr` read both files, filtered them,
printed the kept counts for each side, and then exited 1 with a thirty-line traceback whose last
line was:

    senbonzakura.trackio.TrackIOError: writing a track needs either pyarrow or datasets, and this
    install has neither: pip install pyarrow

The sentence in that traceback is the right sentence. Two things about it were wrong anyway. The
dependency was knowable before any work started, and a user whose actual problem is one `pip
install` was handed a stack of frames through four of our files, which reads like our bug.

WHY THE ORDER IS ASSERTED AND NOT ONLY THE MESSAGE

A refusal that arrives after the work is not a pre-flight, and the difference is invisible in the
text of the message. So the test that matters here is the one asserting nothing was printed before
the refusal: that is the half a message test cannot see, and it is the half that would rot first if
somebody moved the check down the function to be near the write it protects.

THE SECOND HALF: THE CLASS, NOT THE CASE

The pre-flight can only name the fault it knows about. Every other `TrackIOError` reached the
terminal the same way, so the write is wrapped as well, and a mistyped `SENBONZAKURA_TRACK_BACKEND`
is the cheapest of those to drive.
"""
from __future__ import annotations

import pytest

from senbonzakura import track, trackio


@pytest.fixture
def prompts(tmp_path):
    """Two sides of thirty prompts each, enough to partition with --fit 10 --search 10."""
    harmful = tmp_path / "h.txt"
    harmless = tmp_path / "n.txt"
    harmful.write_text("".join(f"harmful prompt number {i}\n" for i in range(30)), encoding="utf-8")
    harmless.write_text("".join(f"harmless prompt number {i}\n" for i in range(30)),
                        encoding="utf-8")
    return harmful, harmless


def _side_prompts(path):
    """Thirty prompts whose text differs per side.

    One list for both sides is not a shortcut here: `check` refuses a track whose two sides share
    a prompt, so a stub that returns the same rows twice is rejected before the write is reached,
    and a test built on it proves nothing about the write.
    """
    side = "harmful" if "h.txt" in str(path) else "harmless"
    return [f"{side} prompt number {i}" for i in range(30)]


def _argv(prompts, tmp_path):
    harmful, harmless = prompts
    return ["--harmful", str(harmful), "--harmless", str(harmless),
            "--out", str(tmp_path / "tr"), "--fit", "10", "--search", "10"]


def test_no_writer_is_a_refusal_rather_than_a_traceback(prompts, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trackio, "writable", lambda: None)
    with pytest.raises(SystemExit) as exit_info:
        track.main(_argv(prompts, tmp_path))
    message = str(exit_info.value)
    assert "pyarrow" in message and "datasets" in message
    assert "pip install pyarrow" in message
    # The exception type's own name is the tell that a traceback escaped rather than a refusal
    # being raised, because `SystemExit(str)` is how every other refusal in this module exits.
    assert "TrackIOError" not in message
    assert not isinstance(exit_info.value.code, int), (
        "the refusal exited with a bare status and no words")


def test_the_refusal_comes_before_any_prompt_is_read(prompts, tmp_path, monkeypatch, capsys):
    """The property that makes this a pre-flight rather than a tidier crash.

    Before the fix the two `harmful: 30 kept ...` lines were already on stdout when the write
    failed, which is the evidence the check was downstream of the work.

    MUTATION-CHECKED, 2026-10-01: with the pre-flight neutered this is the ONLY test in the file
    that fails, because the wrapper around the write catches the same fault a moment later and
    produces a message that satisfies every assertion about the words. The two layers mask each
    other everywhere except in the ordering, so this test is the whole guard for the first one.
    """
    reads = []
    real_read = track.read_prompts
    monkeypatch.setattr(track, "read_prompts",
                        lambda path, *a, **kw: (reads.append(path), real_read(path, *a, **kw))[1])
    monkeypatch.setattr(trackio, "writable", lambda: None)
    with pytest.raises(SystemExit):
        track.main(_argv(prompts, tmp_path))
    assert reads == [], f"the refusal read {len(reads)} prompt file(s) before refusing: {reads}"
    assert capsys.readouterr().out == "", "the refusal printed progress before refusing"


def test_nothing_is_left_on_disk_when_the_writer_is_missing(prompts, tmp_path, monkeypatch):
    """The refusal claims there is no half-built track to clear up, so that is asserted."""
    monkeypatch.setattr(trackio, "writable", lambda: None)
    out = tmp_path / "tr"
    with pytest.raises(SystemExit):
        track.main(_argv(prompts, tmp_path))
    assert not out.exists(), f"{out} was created by a command that refused to run"


def test_a_writer_that_is_present_is_not_refused(prompts, tmp_path, monkeypatch):
    """The guard must not fire on an install that can write, which is every real one.

    Driven by asserting the command gets past the pre-flight: whether the rest of the build
    succeeds here depends on the backend installed, and that is `test_track.py`'s subject, not
    this file's. Reaching `read_prompts` is the whole claim.
    """
    monkeypatch.setattr(trackio, "writable", lambda: "pyarrow")
    seen = []
    monkeypatch.setattr(track, "read_prompts", lambda path, *a, **kw: seen.append(path) or
                        _side_prompts(path))
    monkeypatch.setattr(track, "write_track", lambda *a, **kw: {
        "counts": {"harmful": 30, "harmless": 30}, "skip_harmful": 20, "skip_harmless": 20,
        "n_harmful": 10, "n_harmless": 10})
    track.main(_argv(prompts, tmp_path))
    assert len(seen) == 2, f"expected both sides to be read, read {len(seen)}"


def test_a_write_fault_that_is_not_the_missing_writer_is_also_a_refusal(
        prompts, tmp_path, monkeypatch):
    """The class, not the case. `chosen_backend` raises this for a mistyped environment variable.

    Asserted on the message rather than on the type, because what is under test is that the
    reason survives into something a reader meets at the terminal.
    """
    monkeypatch.setattr(trackio, "writable", lambda: "pyarrow")
    monkeypatch.setattr(track, "read_prompts", lambda path, *a, **kw: _side_prompts(path))

    def explode(*_a, **_kw):
        raise trackio.TrackIOError("SENBONZAKURA_TRACK_BACKEND='pyarow' is not a backend")

    monkeypatch.setattr(track, "write_track", explode)
    with pytest.raises(SystemExit) as exit_info:
        track.main(_argv(prompts, tmp_path))
    message = str(exit_info.value)
    assert "pyarow" in message, "the underlying reason was dropped on the way out"
    assert "half-built" in message, "the refusal no longer says whether a partial track survives"


# ── the capability check itself ──────────────────────────────────────────────────
def test_writable_names_pyarrow_when_pyarrow_is_there(monkeypatch):
    monkeypatch.setattr(trackio, "_pyarrow", lambda: object())
    assert trackio.writable() == "pyarrow"


def test_writable_agrees_with_the_writer_about_which_backend_wins():
    """`writable` mirrors `_write_shard`'s precedence, and a drift between them is the whole bug.

    Asserted against the real environment rather than a stub: whatever this machine has, the
    answer must be one of the two names or None, and never a name for a library absent here.
    """
    answer = trackio.writable()
    assert answer in {"pyarrow", "datasets", None}
    if answer == "pyarrow":
        import pyarrow as pa  # noqa: F401
    elif answer == "datasets":
        assert trackio._pyarrow() is None, (
            "datasets was named while pyarrow is installed, so this disagrees with _write_shard, "
            "which reaches for pyarrow first")
        import datasets  # noqa: F401
