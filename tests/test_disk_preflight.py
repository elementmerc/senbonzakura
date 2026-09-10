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
    total, complete = headtohead._tree_bytes(model)
    assert total >= 4 * 1024 * 1024
    assert complete, "the walk finished, so the figure is a size rather than a lower bound"


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

def test_a_model_it_cannot_size_says_so_instead_of_staying_silent(tmp_path):
    """AN UNANSWERABLE QUESTION MUST NOT READ AS A REASSURING ANSWER.

    This returned [] for any --model that is not a local directory, which is every Hub id, which is
    the documented invocation (`--model Qwen/Qwen3-1.7B` in docs/guide/benchmark.md). So the gate
    added because a ten-arm run filled the disk eight hours in was silent on the exact command line
    the docs teach, and silent about being silent. `host_free_bytes` twelve lines above it gets the
    same question right by returning None rather than a comfortable zero.
    """
    got = complain(tmp_path, tmp_path / "nope", arms=10, free=0, host_free=0)
    assert got, "no answer was available and the caller was told everything was fine"
    assert "could not run" in got[0]
    assert "10 copies" in got[0]


def test_a_hub_id_with_no_local_snapshot_is_reported_as_unsizeable(tmp_path, monkeypatch):
    """The documented invocation, on a machine that has not downloaded the model yet."""
    monkeypatch.setattr(headtohead, "_cached_model_dir", lambda m: None)
    got = complain(tmp_path, "Qwen/Qwen3-1.7B", arms=10, free=0, host_free=0)
    assert got and "could not run" in got[0]


def test_a_hub_id_already_in_the_cache_is_sized_from_it(tmp_path, model, monkeypatch):
    """Having found the snapshot, the arithmetic is the same as for a local path."""
    monkeypatch.setattr(headtohead, "_cached_model_dir", lambda m: model)
    got = complain(tmp_path, "Qwen/Qwen3-1.7B", arms=10, free=1 * GB, host_free=None)
    assert got and "10 arms" in got[0]
    assert "could not run" not in got[0]


def test_the_same_weights_in_two_formats_are_counted_once(tmp_path):
    """A Llama-3 snapshot ships model-*.safetensors beside original/consolidated.*.pth.

    Summing everything doubles the estimate, and an over-count makes the gate refuse a run that
    would have fitted, which is the failure `arms_to_run` was written to avoid arriving through the
    other input.
    """
    snap = tmp_path / "snap"
    (snap / "original").mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"\0" * (4 * 1024 * 1024))
    (snap / "original" / "consolidated.00.pth").write_bytes(b"\0" * (4 * 1024 * 1024))
    total, _ = headtohead._tree_bytes(snap)
    assert total == 4 * 1024 * 1024, "the same weights twice is one model, not two"


def test_a_truncated_walk_reaches_the_reader_as_a_lower_bound(tmp_path, monkeypatch):
    """The complaint, not just the flag. Driven through the real `disk_complaints` with a walk
    reported as incomplete, because the cap is 200,000 entries and building that many files to
    exercise one branch would trade minutes of test time for nothing.
    """
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"\0" * 1024)
    monkeypatch.setattr(headtohead, "_tree_bytes", lambda p, cap=None: (1024, False))
    got = headtohead.disk_complaints(out=tmp_path, model=str(snap), arms=2,
                                     free=0, host_free=None)
    assert any("LOWER BOUND" in g for g in got), (
        "a truncated measurement presented as a measurement is the defect this file exists for")


def test_a_truncated_walk_is_flagged_by_the_walker(tmp_path):
    """It used to break at the cap and hand back the partial total as though it were the size."""
    snap = tmp_path / "snap"
    snap.mkdir()
    for i in range(20):
        (snap / f"model-{i:05d}.safetensors").write_bytes(b"\0" * 1024)
    total, complete = headtohead._tree_bytes(snap, cap=5)
    assert not complete, "the walk hit the cap"
    assert total < 20 * 1024, "and the total it returned is therefore partial"


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

def _flight(tmp_path, model, **kw):
    (tmp_path / "corpus").mkdir(exist_ok=True)
    return headtohead.preflight(
        tools=["senbon", "heretic"], track=tmp_path / "corpus", out=tmp_path / "out",
        model=str(model), isolate="none", images={}, **kw)


def test_the_preflight_reports_the_shortfall_when_there_is_one(tmp_path, model, monkeypatch):
    """Driven with a host volume that has nothing left, which is the state that caused this."""
    monkeypatch.setattr(headtohead, "host_free_bytes", lambda: 1 * GB)
    problems = _flight(tmp_path, model, seeds=[42, 43, 44, 45, 46], trials=200)
    assert any("Windows volume" in p for p in problems), (
        f"preflight ran the disk check and did not surface it: {problems}")
    assert any("10 arms" in p for p in problems), "five seeds of two tools is ten arms, not five"


def test_a_run_that_names_no_seeds_is_not_second_guessed(tmp_path, model):
    """No seeds means no arm count, so no estimate rather than a guess from a number nobody gave."""
    assert not any("arms need about" in p for p in _flight(tmp_path, model))


# ── the recovery case, which the first version of this gate would have blocked ───────

def test_arms_already_on_disk_are_not_sized_for_again(tmp_path, model, monkeypatch):
    """THE DEFECT IN THE FIX. A run that died with nine of ten complete needs room for ONE arm.

    Sizing the resumed run against all ten would refuse the recovery, which is the moment the
    check matters most. Counted through the runner's own `arm_is_done` so the two cannot drift.
    """
    monkeypatch.setattr(headtohead, "arm_is_done",
                        lambda arm, expected, adapter: (Path(arm).name != "heretic-seed46",
                                                        "pretend"))
    n = headtohead.arms_to_run(tools=["senbon", "heretic"], seeds=[42, 43, 44, 45, 46],
                               model=str(model), out=tmp_path / "out", trials=200)
    assert n == 1, f"nine arms are already done and it wants room for {n}"


def test_force_sizes_for_every_arm_again(tmp_path, model, monkeypatch):
    """`--force` re-runs what is on disk, so the space has to be there for all of it."""
    monkeypatch.setattr(headtohead, "arm_is_done", lambda *a: (True, "done"))
    n = headtohead.arms_to_run(tools=["senbon", "heretic"], seeds=[42, 43], model=str(model),
                               out=tmp_path / "out", trials=200, force=True)
    assert n == 4


def test_a_run_with_nothing_on_disk_sizes_for_all_of_it(tmp_path, model):
    n = headtohead.arms_to_run(tools=["senbon", "heretic"], seeds=[42, 43, 44],
                               model=str(model), out=tmp_path / "nothing-here", trials=200)
    assert n == 6


def test_an_unknown_tool_is_left_to_the_check_that_owns_it(tmp_path, model):
    """It is already a complaint elsewhere; counting it here would double-report and mis-size."""
    n = headtohead.arms_to_run(tools=["senbon", "nonsense"], seeds=[42],
                               model=str(model), out=tmp_path / "nothing", trials=200)
    assert n == 1
