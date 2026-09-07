# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Deleting the base model to make room, and every reason not to.

WHY THIS EXISTS

The disk pre-flight has told operators to "delete the base model directory first if the output is
going somewhere else" since a 57 GB base plus a 61 GB output met a 120 GB volume and safetensors
died partway through the shards, hours into a rented GPU. That was advice in an error message,
which does not help a volume that is already full at the moment the save starts.

Now it is behaviour, and it is opt-in and it always will be. The action is irreversible, it fires
at the end of a long run when nobody is watching, and the thing being deleted is re-downloadable
while the run is not.

WHAT THESE ACTUALLY GUARD

The refusals, mostly. Each one is a way this could destroy something that costs more than the disk
space it frees, and the worst is the shared Hugging Face cache: deleting one to finish this run
breaks the next run, and every other tool on the machine that reads the same cache. A test that
only checked the happy path would pass against code that did exactly that.
"""
from __future__ import annotations

import pytest

from senbonzakura.crashsafe import base_release_verdict

GB = 10 ** 9


@pytest.fixture
def pair(tmp_path):
    """A plain local base model directory and a separate output directory."""
    src = tmp_path / "base"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "model.safetensors").write_bytes(b"\x00" * 16)
    return src, out


# ── the one case that is allowed ──────────────────────────────────────────────────
def test_a_plain_local_copy_with_no_room_may_be_released(pair):
    src, out = pair
    ok, reason = base_release_verdict(str(src), str(out), need_bytes=100 * GB, free_bytes=1 * GB)
    assert ok
    assert str(src.resolve()) in reason


def test_the_verdict_is_readable_without_the_numbers(pair):
    """Called without a size check it answers the eligibility question alone."""
    src, out = pair
    ok, _reason = base_release_verdict(str(src), str(out))
    assert ok


# ── the refusals, which are the point ─────────────────────────────────────────────
def test_a_shared_hub_cache_is_refused(tmp_path):
    """THE WORST CASE. A Hub cache is shared: freeing it to finish this run breaks the next one
    and every other tool on the machine that reads it.
    """
    src = tmp_path / "huggingface" / "hub" / "models--Qwen--Qwen3-1.7B"
    src.mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    ok, reason = base_release_verdict(str(src), str(out), need_bytes=100 * GB, free_bytes=1 * GB)
    assert not ok
    assert "shared" in reason
    assert "other" in reason, "the message has to say what else it would break"


@pytest.mark.parametrize("shape", ["huggingface/hub/models--x", ".cache/huggingface/x",
                                   "hf_hub/x", "somewhere/models--org--name"])
def test_every_cache_shape_is_refused(tmp_path, shape):
    src = tmp_path / shape
    src.mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    ok, _r = base_release_verdict(str(src), str(out), need_bytes=100 * GB, free_bytes=1 * GB)
    assert not ok, f"{shape} was not recognised as a cache"


def test_the_source_being_the_output_is_refused(pair):
    """Releasing it would delete what was just written."""
    src, _out = pair
    ok, reason = base_release_verdict(str(src), str(src), need_bytes=100 * GB, free_bytes=1 * GB)
    assert not ok
    assert "just written" in reason


def test_an_output_inside_the_source_is_refused(pair):
    src, _out = pair
    inner = src / "abliterated"
    inner.mkdir()
    ok, reason = base_release_verdict(str(src), str(inner), need_bytes=100 * GB, free_bytes=1 * GB)
    assert not ok
    assert "just written" in reason


def test_a_symlinked_source_is_still_recognised_as_the_same_directory(tmp_path):
    """A string comparison would miss this, and the consequence is deleting the output."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not do symlinks")
    ok, reason = base_release_verdict(str(link), str(real), need_bytes=100 * GB, free_bytes=1 * GB)
    assert not ok
    assert "just written" in reason


def test_a_hub_id_is_refused_because_there_is_nothing_local(pair):
    _src, out = pair
    ok, reason = base_release_verdict("Qwen/Qwen3-1.7B", str(out),
                                      need_bytes=100 * GB, free_bytes=1 * GB)
    assert not ok
    assert "not a local directory" in reason


def test_no_recorded_source_is_refused(pair):
    _src, out = pair
    ok, reason = base_release_verdict("", str(out))
    assert not ok
    assert "nothing to release" in reason


def test_a_source_that_does_not_exist_is_refused(tmp_path):
    ok, _reason = base_release_verdict(str(tmp_path / "gone"), str(tmp_path / "out"))
    assert not ok


def test_room_already_being_sufficient_is_refused(pair):
    """A deletion that buys nothing costs a re-download, so it is pure loss."""
    src, out = pair
    ok, reason = base_release_verdict(str(src), str(out), need_bytes=1 * GB, free_bytes=500 * GB)
    assert not ok
    assert "already room" in reason


def test_an_unmeasurable_disk_does_not_licence_a_deletion(pair):
    """CAUGHT BY THIS TEST, in the code it was written against.

    `disk_verdict` reads an unmeasurable disk as "proceed", which is right for a save and wrong
    here. The first version skipped the size check entirely when the disk could not be measured,
    so an unmeasurable volume LICENSED the deletion. Proceeding with a save on a hunch costs a
    failed write; proceeding with a deletion on a hunch costs the model.
    """
    src, out = pair
    ok, reason = base_release_verdict(str(src), str(out), need_bytes=100 * GB, free_bytes=None)
    assert not ok
    assert "not deleted on a guess" in reason


# ── the flag, and its default ─────────────────────────────────────────────────────
def test_the_flag_is_off_by_default():
    """It is irreversible and it fires unattended. There is no argument for a different default."""
    from senbonzakura.parser import build_parser

    assert build_parser().parse_args(["--model", "x"]).free_base_model is False


def test_the_flag_help_says_what_it_refuses():
    """An operator reaching for this is short of disk and in a hurry, which is exactly when the
    refusals need to be readable before the run rather than discovered after it.
    """
    from senbonzakura.parser import build_parser

    action, = [a for a in build_parser()._actions if a.dest == "free_base_model"]
    for word in ("irreversible", "cache", "--out"):
        assert word in action.help


# ── the method on the abliterator, which is where the deletion actually happens ────
class _FakeModel:
    def parameters(self):
        import torch
        return iter([torch.zeros(4, 4)])


def _abl(tmp_path, model_path, out):
    """An Abliterator shaped just enough to call `_release_base_model`.

    Built by hand rather than through the constructor, which loads a model. The method under test
    reads exactly four things and this supplies exactly those four, so a field it starts using
    later shows up as an AttributeError rather than passing silently.
    """
    import types

    from senbonzakura import cli

    abl = cli.Abliterator.__new__(cli.Abliterator)
    abl.args = types.SimpleNamespace(model=str(model_path), out=str(out))
    abl.model = _FakeModel()
    abl.log = lambda m: said.append(m)
    abl.events = types.SimpleNamespace(emit=lambda *a, **k: None)
    return abl


said: list[str] = []


def test_the_method_deletes_only_when_the_verdict_allows_it(tmp_path, monkeypatch):
    """The whole point: a refusal leaves the directory exactly where it was."""
    from senbonzakura import cli

    said.clear()
    src = tmp_path / "base"
    src.mkdir()
    (src / "w.safetensors").write_bytes(b"\x00" * 8)
    out = tmp_path / "out"
    out.mkdir()

    monkeypatch.setattr(cli, "free_bytes_for", lambda _p: 10 ** 15)   # plenty of room
    _abl(tmp_path, src, out)._release_base_model()
    assert src.exists(), "it deleted the base model when there was already room"
    assert any("NOT releasing" in m for m in said)


def test_the_method_deletes_when_the_disk_is_tight(tmp_path, monkeypatch):
    from senbonzakura import cli

    said.clear()
    src = tmp_path / "base"
    src.mkdir()
    (src / "w.safetensors").write_bytes(b"\x00" * 8)
    out = tmp_path / "out"
    out.mkdir()

    monkeypatch.setattr(cli, "free_bytes_for", lambda _p: 1)          # no room at all
    _abl(tmp_path, src, out)._release_base_model()
    assert not src.exists(), "the base model was not released"
    assert any("removed" in m for m in said)


def test_a_deletion_that_fails_does_not_kill_the_run(tmp_path, monkeypatch):
    """The save is about to be attempted either way, and its own pre-flight reports the room.
    Losing an expensive run to a failed `rmtree` would be the larger failure.
    """
    import shutil

    from senbonzakura import cli

    said.clear()
    src = tmp_path / "base"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(cli, "free_bytes_for", lambda _p: 1)
    monkeypatch.setattr(shutil, "rmtree", lambda *_a, **_k: (_ for _ in ()).throw(OSError("nope")))

    _abl(tmp_path, src, out)._release_base_model()          # must not raise
    assert any("could not remove it" in m for m in said)
    assert any("continuing to the save" in m for m in said)
