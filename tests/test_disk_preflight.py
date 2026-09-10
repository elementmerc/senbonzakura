# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""There is room for every arm, asked of the volume underneath as well as of this filesystem.

WHAT HAPPENED, 2026-09-10

A ten-arm head-to-head reached nine complete arms and died in the tenth. The preflight had checked
the corpus, the images, the mounts and that the output directory was writable, and had never asked
whether there was anywhere to put ten copies of a model. Its own docstring says a run that dies
forty minutes in on something knowable at the start is the most expensive kind of failure; this one
died eight hours in.

THE PART THAT MAKES IT WORTH A FILE OF ITS OWN

Somebody did check the disk. `df -h` inside the WSL guest reported 534 GB free and that was written
down as plenty. On WSL2 the Linux root is a sparse virtual disk living on the Windows volume, so
that figure is the virtual disk's APPARENT size: a promise the host can only keep while it has room
to grow the file. The Windows volume hit zero bytes, the disk could not grow, and every operation
inside the guest began returning EIO, down to `getpwuid` failing to read `/etc/passwd`.

The rule, which is more useful than the instance: a measurement taken inside the thing you are
measuring cannot see the constraint that contains it. The same shape produced a container listing a
directory full of files it could not open, because a bind mount carries a directory and not the
things its entries point at.

So the check asks two questions, and an unanswerable one must never read as a reassuring answer.
"""
from pathlib import Path

import pytest

from senbonzakura import headtohead

GB = 1024 ** 3


@pytest.fixture
def model(tmp_path):
    """A model directory shaped like a HuggingFace snapshot: real bytes in blobs, links beside."""
    repo = tmp_path / "models--x"
    blobs, snap = repo / "blobs", repo / "snapshots" / "abc"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (blobs / "weights").write_bytes(b"\0" * (4 * 1024 * 1024))
    (snap / "model.safetensors").symlink_to(Path("../../blobs") / "weights")
    (snap / "config.json").write_text("{}", encoding="utf-8")
    return snap


def complain(out, model, arms, free, host_free):
    return headtohead.disk_complaints(out=Path(out), model=str(model), arms=arms,
                                      free=free, host_free=host_free)


# ── sizing the model, which is where a symlinked cache reads as empty ────────────────

def test_a_snapshot_of_symlinks_is_sized_by_what_they_point_at(model):
    """A size that does not follow links calls a multi-gigabyte model a few kilobytes, and then
    every arm fits in any amount of space at all.
    """
    assert headtohead._tree_bytes(model) >= 4 * 1024 * 1024


def test_ten_arms_of_a_model_that_does_not_fit_is_refused(model):
    got = complain("/", model, arms=10, free=1 * GB, host_free=None)
    assert got, "ten copies of the model do not fit and nothing said so"
    assert "10 arms" in got[0]
    assert "full copy of the model" in got[0], "the reader needs to know why it is that large"


def test_room_for_every_arm_is_silent(model):
    assert complain("/", model, arms=10, free=500 * GB, host_free=500 * GB) == []


# ── the question that was never asked ────────────────────────────────────────────────

def test_the_backing_volume_is_asked_even_when_this_filesystem_says_yes(model):
    """THE DEFECT. `df` here reported hundreds of gigabytes while the host had none."""
    got = complain("/", model, arms=10, free=500 * GB, host_free=1 * GB)
    assert got, "the guest had room, the host did not, and the run went ahead"
    assert "Windows volume" in got[0]
    assert "apparent size" in got[0], "the reader has to understand why df disagreed"


def test_an_unreachable_host_volume_is_silent_rather_than_reassuring(model, monkeypatch):
    """None means the question could not be asked. It must not be reported as a pass.

    It is also not an error: off WSL there is no backing volume to ask about, and a complaint
    there would make the check noise that gets ignored, which is how a gate stops being read.
    """
    monkeypatch.setattr(headtohead, "host_free_bytes", lambda: None)
    assert headtohead.disk_complaints(out=Path("/"), model=str(model), arms=10,
                                      free=500 * GB) == []


def test_both_volumes_short_is_reported_as_two_separate_problems(model):
    got = complain("/", model, arms=10, free=1 * GB, host_free=1 * GB)
    assert len(got) == 2, "they are different facts and either alone would stop the run"


# ── refusing to guess ────────────────────────────────────────────────────────────────

def test_a_model_that_is_not_a_directory_is_left_to_the_checks_that_own_it(tmp_path):
    """No size to work from means no arithmetic, not an invented estimate."""
    assert complain(tmp_path, tmp_path / "nope", arms=10, free=0, host_free=0) == []


def test_zero_arms_asks_nothing(model):
    assert complain("/", model, arms=0, free=0, host_free=0) == []


# ── the WSL detection itself ─────────────────────────────────────────────────────────

def test_a_kernel_that_is_not_wsl_reports_no_backing_volume(monkeypatch, tmp_path):
    fake = tmp_path / "osrelease"
    fake.write_text("6.11.0-generic\n", encoding="utf-8")
    monkeypatch.setattr(headtohead, "Path", _PathStub(fake))
    assert headtohead.host_free_bytes() is None


def test_a_missing_osrelease_reports_no_backing_volume(monkeypatch, tmp_path):
    monkeypatch.setattr(headtohead, "Path", _PathStub(tmp_path / "absent"))
    assert headtohead.host_free_bytes() is None


class _PathStub:
    """Redirects only `/proc/sys/kernel/osrelease`; every other path behaves normally."""

    def __init__(self, target):
        self.target = target

    def __call__(self, p):
        return Path(self.target) if str(p) == "/proc/sys/kernel/osrelease" else Path(p)


# ── the wiring, because a check nothing calls is not a check ─────────────────────────

def test_the_preflight_reports_the_shortfall_when_there_is_one(tmp_path, model, monkeypatch):
    """Driven with a host volume that has nothing left, which is the state that caused this."""
    (tmp_path / "corpus").mkdir()
    monkeypatch.setattr(headtohead, "host_free_bytes", lambda: 1 * GB)
    problems = headtohead.preflight(
        tools=["senbon", "heretic"], track=tmp_path / "corpus", out=tmp_path / "out",
        model=str(model), isolate="none", images={}, arms=10)
    assert any("Windows volume" in p for p in problems), (
        f"preflight ran the disk check and did not surface it: {problems}")


def test_a_run_that_does_not_say_how_many_arms_is_not_second_guessed(tmp_path, model):
    """`arms` defaults to 0, so a caller that does not say gets no estimate rather than a guess
    built from a number nobody passed.
    """
    (tmp_path / "corpus").mkdir()
    problems = headtohead.preflight(
        tools=["senbon", "heretic"], track=tmp_path / "corpus", out=tmp_path / "out",
        model=str(model), isolate="none", images={})
    assert not any("arms need about" in p for p in problems)


def test_the_arm_count_is_tools_times_seeds():
    """Five seeds of two tools is ten models on disk, not five."""
    text = Path(headtohead.__file__).read_text(encoding="utf-8")
    assert "arms=len(tools) * len(seeds)" in text, (
        "the estimate must count every arm; counting seeds alone halves it")
