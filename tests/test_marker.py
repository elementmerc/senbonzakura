# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for the checkpoint marker (senbonzakura.marker).

The property that matters is not that a field is written. It is that the field cannot be
separated from the weights: `abliteration.json` already records everything and is shed the moment
somebody copies one shard out of a directory. So these tests care about two things above all
others, and both were the reason the module was written after a review on 2026-08-15:

- **the tensors come through byte for byte.** A metadata edit that perturbs a weight would be a
  far worse defect than the one it fixes;
- **a partial abliteration cannot be saved unmarked.** That model's refusal behaviour is only half
  removed and it must never be able to pass for a whole one.
"""
import json
import os
import struct
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from senbonzakura import marker


@pytest.fixture
def saved(tmp_path):
    """The shape `save_pretrained(safe_serialization=True)` leaves behind."""
    tensors = {"a": torch.randn(16, 8), "b": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    save_file(tensors, str(tmp_path / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file({"c": torch.randn(4, 4)}, str(tmp_path / "model-00002-of-00002.safetensors"),
              metadata={"format": "pt"})
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    return tmp_path, tensors


# ── the fields ────────────────────────────────────────────────────────────────────────
def test_a_whole_abliteration_says_so_in_the_negative():
    f = marker.fields(version="0.3.0", ablate_conv=True, partial_layers=[])
    assert f["partial"] == "false"
    assert "unedited_layers" not in f and "warning" not in f


def test_a_partial_one_names_the_layers_and_carries_a_warning():
    f = marker.fields(version="0.3.0", ablate_conv=False, partial_layers=[3, 1, 0])
    assert f["partial"] == "true"
    assert f["unedited_layers"] == "0,1,3", "sorted, so two runs of one config agree"
    assert "PARTIAL ABLITERATION" in f["warning"]
    assert "control, not a result" in f["warning"]


def test_every_value_is_a_string():
    """The safetensors header allows nothing else, and config.json uses the same map so the two
    cannot drift into saying different things.
    """
    f = marker.fields(version="0.3.0", ablate_conv=True, partial_layers=[1],
                      num_directions=2, dir_mode="single", seed=42, base_model="Qwen/Qwen3-1.7B")
    assert all(isinstance(v, str) for v in f.values()), f


def test_the_optional_fields_are_left_out_rather_than_written_as_none():
    f = marker.fields(version="0.3.0", ablate_conv=True, partial_layers=[])
    for absent in ("num_directions", "dir_mode", "seed", "base_model"):
        assert absent not in f


# ── the header rewrite, which must not touch a weight ─────────────────────────────────
def test_the_tensors_come_through_byte_for_byte(saved):
    directory, before = saved
    marker.stamp_safetensors(str(directory), {"partial": "true"}, log=lambda _s: None)
    after = load_file(str(directory / "model-00001-of-00002.safetensors"))
    assert sorted(after) == sorted(before)
    for k in before:
        assert torch.equal(before[k], after[k]), k


def test_the_marker_lands_in_every_shard(saved):
    directory, _ = saved
    n = marker.stamp_safetensors(str(directory), {"partial": "true"}, log=lambda _s: None)
    assert n == 2
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        with safe_open(str(directory / name), framework="pt") as f:
            assert f.metadata()["senbonzakura.partial"] == "true"


def test_the_existing_metadata_is_kept(saved):
    """`format: pt` is what tells a loader how to read the file. Replacing rather than merging
    the metadata would produce a marked checkpoint nothing could load.
    """
    directory, _ = saved
    marker.stamp_safetensors(str(directory), {"partial": "false"}, log=lambda _s: None)
    with safe_open(str(directory / "model-00002-of-00002.safetensors"), framework="pt") as f:
        assert f.metadata()["format"] == "pt"


def test_the_data_block_stays_eight_byte_aligned(saved):
    """Readers expect the tensor data to start on an 8-byte boundary, and the header is padded
    with spaces to keep it there. JSON ignores trailing whitespace, so nothing else notices.
    """
    directory, _ = saved
    marker.stamp_safetensors(str(directory), {"partial": "true", "seed": "42"},
                             log=lambda _s: None)
    with open(directory / "model-00001-of-00002.safetensors", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
    assert n % marker.ALIGNMENT == 0


def test_stamping_twice_updates_rather_than_accumulates(saved):
    """A re-stamped checkpoint must not end up with two contradictory answers in one header."""
    directory, before = saved
    marker.stamp_safetensors(str(directory), {"partial": "true"}, log=lambda _s: None)
    marker.stamp_safetensors(str(directory), {"partial": "false"}, log=lambda _s: None)
    with safe_open(str(directory / "model-00001-of-00002.safetensors"), framework="pt") as f:
        meta = f.metadata()
    assert meta["senbonzakura.partial"] == "false"
    after = load_file(str(directory / "model-00001-of-00002.safetensors"))
    for k in before:
        assert torch.equal(before[k], after[k]), "two rewrites must still not move a weight"


def test_no_partial_file_is_left_behind(saved):
    directory, _ = saved
    marker.stamp_safetensors(str(directory), {"partial": "true"}, log=lambda _s: None)
    assert not [n for n in os.listdir(directory) if n.endswith(".stamping")]


# ── config.json, the copy every loader reads ──────────────────────────────────────────
def test_the_config_keeps_what_was_already_in_it(saved):
    directory, _ = saved
    marker.stamp_config(str(directory), {"partial": "true"})
    doc = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    assert doc["model_type"] == "qwen3", "the model would not load without it"
    assert doc["senbonzakura"]["partial"] == "true"


# ── reading it back ───────────────────────────────────────────────────────────────────
def test_the_marker_reads_back_from_the_header(saved):
    directory, _ = saved
    marker.stamp(str(directory), marker.fields(version="0.3.0", ablate_conv=False,
                                               partial_layers=[0, 1]), log=lambda _s: None)
    got = marker.read(str(directory))
    assert got["partial"] == "true" and got["unedited_layers"] == "0,1"


def test_a_lone_shard_still_answers_for_itself(tmp_path):
    """The failure this module exists for: somebody copies one file out of the directory. The
    sibling JSON is gone, config.json is gone, and the marker has to survive that.
    """
    save_file({"a": torch.randn(4, 4)}, str(tmp_path / "model.safetensors"),
              metadata={"format": "pt"})
    marker.stamp_safetensors(str(tmp_path), {"partial": "true"}, log=lambda _s: None)
    lone = tmp_path / "elsewhere"
    lone.mkdir()
    os.replace(tmp_path / "model.safetensors", lone / "model.safetensors")
    assert marker.read(str(lone))["partial"] == "true"


def test_an_unmarked_checkpoint_reads_as_nothing(saved):
    directory, _ = saved
    assert marker.read(str(directory)) is None


# ── how strict it is, and why that is asymmetric ──────────────────────────────────────
def test_a_partial_model_that_cannot_be_marked_takes_the_run_down(saved, monkeypatch):
    """An unmarked partial abliteration is indistinguishable from a whole one, which is the
    entire failure being guarded. Better a dead save than that model in the world.
    """
    directory, _ = saved
    monkeypatch.setattr(marker, "stamp_safetensors",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(RuntimeError) as e:
        marker.stamp(str(directory), {"partial": "true"}, required=True, log=lambda _s: None)
    assert "PARTIAL abliteration" in str(e.value) and "disk full" in str(e.value)


def test_a_whole_model_that_cannot_be_marked_warns_and_lives(saved, monkeypatch):
    """A provenance line is not worth killing a save whose GPU work is already spent."""
    directory, _ = saved
    monkeypatch.setattr(marker, "stamp_safetensors",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    said = []
    assert marker.stamp(str(directory), {"partial": "false"}, required=False,
                        log=said.append) is False
    joined = " ".join(said)
    assert "WARNING" in joined and "saved and usable" in joined


def test_the_happy_path_reports_success(saved):
    directory, _ = saved
    assert marker.stamp(str(directory), {"partial": "false"}, log=lambda _s: None) is True


def test_the_config_answers_when_the_shards_are_gone(tmp_path):
    """The other half of the pair: a directory whose weights were converted away (to GGUF, say)
    still says what the original was.
    """
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "lfm2", "senbonzakura": {"partial": "true"}}), encoding="utf-8")
    assert marker.read(str(tmp_path))["partial"] == "true"


# ── the stamp needs room, and a half-stamped directory must say so ───────────────────────────

def _no_room(monkeypatch):
    import shutil as _shutil
    monkeypatch.setattr(marker.shutil, "disk_usage",
                        lambda _p: _shutil._ntuple_diskusage(total=1 << 40, used=1 << 40, free=1))


def test_stamping_refuses_before_the_first_shard_when_there_is_no_room(saved, monkeypatch):
    """THE GAP: `_rewrite_header` writes a COMPLETE copy of a shard before renaming over it,
    and nothing reserved that space. The model save's own pre-flight reserves 5% of the model
    size, which on a 30B with 5 GB shards is not one shard. With `--free-base-model` the base
    is already deleted by then, so recovering means downloading it again.
    """
    directory, _ = saved
    _no_room(monkeypatch)
    with pytest.raises(OSError) as e:
        marker.stamp_safetensors(str(directory), {"partial": "true"})
    said = str(e.value)
    assert "largest shard" in said
    assert "already written and are not affected" in said, (
        "the message must say the weights survived, or a reader assumes the save was lost")


def test_nothing_is_stamped_when_the_preflight_refuses(saved, monkeypatch):
    """A pre-flight that refuses AFTER touching a shard would create the state it prevents."""
    directory, _ = saved
    _no_room(monkeypatch)
    with pytest.raises(OSError):
        marker.stamp_safetensors(str(directory), {"partial": "true"})
    assert marker.read(str(directory)) is None, "a refused stamp left a marker behind"


def test_a_stamp_that_dies_mid_copy_leaves_no_partial_file(saved, monkeypatch):
    """Baseline 2.1: no `.part`, `.tmp` or half-written file survives a failure.

    This has to drive the failure INSIDE the copy. Asserting it after a pre-flight refusal
    proves nothing, because the pre-flight stops before any temp file exists: that version of
    this assertion passed with the cleanup deleted.
    """
    directory, _ = saved
    def dies_after_creating_the_temp(path, tmp, fields):
        Path(tmp).write_bytes(b"half a shard")
        raise OSError("No space left on device")

    monkeypatch.setattr(marker, "_write_stamped_copy", dies_after_creating_the_temp)
    with pytest.raises(OSError):
        marker.stamp_safetensors(str(directory), {"partial": "true"})
    assert not list(Path(directory).glob("*.stamping")), (
        "a failed stamp left its half-written copy behind, which on a full disk is part of "
        "what filled it")


def test_an_unmeasurable_filesystem_says_the_preflight_was_skipped(saved, monkeypatch):
    """Proceeding is right; proceeding QUIETLY is not, which is the rule the host-RAM
    pre-flight already follows.
    """
    directory, _ = saved
    lines = []
    monkeypatch.setattr(marker.shutil, "disk_usage",
                        lambda _p: (_ for _ in ()).throw(OSError("no statvfs here")))
    marker.stamp_safetensors(str(directory), {"partial": "true"}, log=lines.append)
    assert any("skipped, not passed" in line for line in lines)


def test_a_half_stamped_directory_refuses_to_report_itself_as_stamped(saved):
    """THE DEFECT `read` HAD. It returned the first shard carrying a marker, so a directory
    where stamping died part way reported as fully marked. For a PARTIAL abliteration, a model
    whose refusal behaviour is only half removed, that is the exact failure this module was
    written to prevent.
    """
    directory, _ = saved
    shards = sorted(Path(directory).glob("*.safetensors"))
    assert len(shards) >= 2, "the fixture must have more than one shard for this to be reachable"
    marker._rewrite_header(str(shards[0]), {"partial": "true"})      # one only
    with pytest.raises(marker.PartialStampError) as e:
        marker.read(str(directory))
    said = str(e.value)
    assert "cannot say what was done to it" in said
    assert shards[1].name in said, "the message must name a shard the reader can go and look at"


def test_a_fully_stamped_directory_still_reads_back(saved):
    """The other half: the new check must not refuse the normal case."""
    directory, _ = saved
    marker.stamp_safetensors(str(directory), {"partial": "true"})
    assert marker.read(str(directory))["partial"] == "true"
