# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Reading and rewriting a checkpoint one tensor at a time.

Two things are being asserted and they fail in different ways, so they are tested apart.

**It is correct.** A rewrite that changes nothing leaves the file byte for byte identical, a
rewrite that changes one tensor changes exactly that tensor, and the header, including any marker
another part of the tool stamped into it, survives untouched.

**It is bounded.** Peak memory is one tensor and not one file. That is the entire reason this
module exists, and every correctness test in here passes just as happily against an implementation
that reads the whole shard into memory first, so the bound is measured directly with `tracemalloc`.
It can see these allocations because they are Python `bytes`, which is not true of the torch
tensors elsewhere in this project.

The header validation gets its own group. The failure it guards against is not a corrupt file,
which announces itself, but a header whose byte ranges disagree with the shapes beside them: a
rewrite driven by those ranges writes a correct tensor into the wrong place and produces a
checkpoint that still loads.
"""
import json
import struct
import tracemalloc

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from senbonzakura import streaming


@pytest.fixture
def shard(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"a": torch.randn(4, 8), "b": torch.randn(16, 2), "c": torch.arange(6)},
              str(path), metadata={"format": "pt", "senbonzakura.marker": "kept"})
    return path


def _corrupt_header(path, mutate):
    """Rewrite a shard's header JSON through `mutate`, leaving the tensor bytes alone."""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
        rest = fh.read()
    mutate(header)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % streaming.ALIGNMENT)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        out.write(rest)


# ── reading ──────────────────────────────────────────────────────────────────────────────────

def test_tensors_are_returned_in_the_order_their_bytes_appear(shard):
    """Offset order, not header order, because the rewrite copies the file forwards in one pass."""
    found = streaming.tensors(shard)
    assert [t.name for t in found] == sorted((t.name for t in found),
                                             key=lambda n: {t.name: t.start for t in found}[n])
    assert [t.start for t in found] == sorted(t.start for t in found)


def test_a_tensor_reports_the_shape_and_dtype_the_header_declares(shard):
    by_name = {t.name: t for t in streaming.tensors(shard)}
    assert by_name["a"].shape == (4, 8)
    assert by_name["a"].dtype == "F32"
    assert by_name["a"].nbytes == 4 * 8 * 4
    assert by_name["b"].shape == (16, 2)
    assert "a" in repr(by_name["a"])


def test_read_tensor_returns_the_bytes_that_tensor_holds(shard):
    by_name = {t.name: t for t in streaming.tensors(shard)}
    with safe_open(str(shard), framework="pt") as fh:
        expected = fh.get_tensor("a")
    raw = streaming.read_tensor(shard, by_name["a"])
    assert raw == expected.numpy().tobytes()


def test_a_missing_file_is_refused_with_its_name(tmp_path):
    with pytest.raises(streaming.ShardError, match="cannot be read"):
        streaming.read_header(tmp_path / "absent.safetensors")


# ── header validation ────────────────────────────────────────────────────────────────────────

def test_a_file_too_short_to_hold_a_header_is_refused(tmp_path):
    path = tmp_path / "stub.safetensors"
    path.write_bytes(b"\x01\x02")
    with pytest.raises(streaming.ShardError, match="too short"):
        streaming.read_header(path)


def test_a_header_longer_than_the_file_is_refused(tmp_path):
    path = tmp_path / "lying.safetensors"
    path.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(streaming.ShardError, match="header says it is"):
        streaming.read_header(path)


def test_a_zero_length_header_is_refused(tmp_path):
    path = tmp_path / "empty.safetensors"
    path.write_bytes(struct.pack("<Q", 0))
    with pytest.raises(streaming.ShardError, match="header says it is"):
        streaming.read_header(path)


def test_a_header_that_is_not_json_is_refused(tmp_path):
    path = tmp_path / "garbage.safetensors"
    blob = b"not json"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    with pytest.raises(streaming.ShardError, match="not valid JSON"):
        streaming.read_header(path)


def test_a_header_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / "list.safetensors"
    blob = b"[1,2,3]"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    with pytest.raises(streaming.ShardError, match="not a JSON object"):
        streaming.read_header(path)


def test_a_truncated_header_is_refused(tmp_path):
    path = tmp_path / "cut.safetensors"
    blob = b'{"a":'
    # The length is honest about the file, and the JSON inside it is not complete.
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    with pytest.raises(streaming.ShardError, match="not valid JSON"):
        streaming.read_header(path)


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda h: h.update(a="not an object"), "is not an object"),
    (lambda h: h["a"].update(dtype="WAT"), "cannot size"),
    (lambda h: h["a"].update(shape="four by eight"), "has shape"),
    (lambda h: h["a"].update(shape=[4, -8]), "has shape"),
    (lambda h: h["a"].update(data_offsets=[0]), "data_offsets"),
    (lambda h: h["a"].update(data_offsets="0 to 128"), "data_offsets"),
    (lambda h: h["a"].update(data_offsets=[64, 16]), "spans"),
    (lambda h: h["a"].update(data_offsets=[-4, 128]), "spans"),
])
def test_a_malformed_entry_is_refused_with_the_reason(shard, mutate, match):
    _corrupt_header(shard, mutate)
    with pytest.raises(streaming.ShardError, match=match):
        streaming.tensors(shard)


def test_a_range_that_disagrees_with_the_shape_is_refused(shard):
    """THE ONE THAT MATTERS MOST.

    Every other malformation here is caught by any reader. This one is not: the file opens, the
    tensor loads, and only the arithmetic is wrong. A rewrite trusting the range would write a
    correctly edited tensor into a span belonging to something else, and the result is a checkpoint
    that loads and is wrong, which is the worst outcome available.
    """
    _corrupt_header(shard, lambda h: h["a"].update(data_offsets=[0, 64]))
    with pytest.raises(streaming.ShardError, match="disagrees with itself"):
        streaming.tensors(shard)


def test_a_tensor_ending_past_the_file_is_refused(shard):
    def mutate(header):
        header["z"] = {"dtype": "F32", "shape": [1 << 20], "data_offsets": [0, 4 << 20]}
    _corrupt_header(shard, mutate)
    with pytest.raises(streaming.ShardError, match="past the end"):
        streaming.tensors(shard)


def test_two_tensors_sharing_bytes_are_refused(shard):
    """Editing either would corrupt the other, so the rewrite must not start."""
    def mutate(header):
        first = min(header[k]["data_offsets"][0] for k in header if k != "__metadata__")
        header["overlap"] = {"dtype": "U8", "shape": [8], "data_offsets": [first, first + 8]}
    _corrupt_header(shard, mutate)
    with pytest.raises(streaming.ShardError, match="overlap"):
        streaming.tensors(shard)


def test_a_tensor_shorter_on_disk_than_its_header_claims_is_refused(shard):
    last = max(streaming.tensors(shard), key=lambda t: t.end)
    with open(shard, "r+b") as fh:
        fh.truncate(fh.seek(0, 2) - 4)
    with pytest.raises(streaming.ShardError, match=r"shorter on disk|past the end"):
        streaming.read_tensor(shard, last)


# ── rewriting ────────────────────────────────────────────────────────────────────────────────

def test_a_rewrite_that_changes_nothing_leaves_the_file_identical(shard):
    """The control. Without it, every test below could pass against a rewrite that mangles the
    bytes it is not asked to touch, as long as it mangles them consistently.
    """
    before = shard.read_bytes()
    assert streaming.rewrite_shard(shard, lambda tensor, raw: None) == 0
    assert shard.read_bytes() == before


def test_a_rewrite_replaces_exactly_the_tensors_it_was_given(shard):
    original = load_file(str(shard))
    replacement = torch.full((4, 8), 7.0)

    def edit(tensor, raw):
        return replacement.numpy().tobytes() if tensor.name == "a" else None

    assert streaming.rewrite_shard(shard, edit) == 1
    after = load_file(str(shard))
    assert torch.equal(after["a"], replacement)
    assert torch.equal(after["b"], original["b"])
    assert torch.equal(after["c"], original["c"])


def test_the_header_and_its_marker_survive_a_rewrite(shard):
    """`marker.py` stamps provenance into `__metadata__`. A bake that dropped it would produce a
    checkpoint that no longer says what was done to it, which is the one thing this tool promises.
    """
    with safe_open(str(shard), framework="pt") as fh:
        before = fh.metadata()
    streaming.rewrite_shard(shard, lambda tensor, raw: bytes(tensor.nbytes))
    with safe_open(str(shard), framework="pt") as fh:
        assert fh.metadata() == before
        assert before["senbonzakura.marker"] == "kept"


def test_every_tensor_can_be_rewritten_in_one_pass(shard):
    seen = []

    def edit(tensor, raw):
        seen.append(tensor.name)
        return bytes(tensor.nbytes)

    assert streaming.rewrite_shard(shard, edit) == 3
    assert sorted(seen) == ["a", "b", "c"]
    assert all(torch.count_nonzero(t) == 0 for t in load_file(str(shard)).values())


def test_a_replacement_of_the_wrong_size_is_refused_and_the_shard_survives(shard):
    """A tensor cannot change size without moving every tensor after it, so this is refused rather
    than accommodated, and refused before the original has been replaced.
    """
    before = shard.read_bytes()
    with pytest.raises(streaming.ShardError, match=r"cannot change a tensor's size"):
        streaming.rewrite_shard(shard, lambda tensor, raw: b"short")
    assert shard.read_bytes() == before
    assert not list(shard.parent.glob("*.rewriting"))


def test_an_edit_that_raises_leaves_the_original_shard_and_no_leftovers(shard):
    """Baseline 2.1: a partial-state file is cleaned up on the way out. The case this matters for
    is the disk filling during the copy, where the leftover is part of what filled it.
    """
    before = shard.read_bytes()

    def edit(tensor, raw):
        raise ZeroDivisionError("the bake fell over")

    with pytest.raises(ZeroDivisionError):
        streaming.rewrite_shard(shard, edit)
    assert shard.read_bytes() == before
    assert not list(shard.parent.glob("*.rewriting"))


def test_the_log_names_each_tensor_it_replaced(shard):
    lines = []
    streaming.rewrite_shard(shard, lambda tensor, raw: bytes(tensor.nbytes), log=lines.append)
    assert len(lines) == 3
    assert any("rewrote a" in line for line in lines)


def test_bytes_after_the_last_tensor_are_carried_through(shard):
    """Trailing bytes survive the rewrite.

    safetensors does not append a trailer today. A reader that assumed so would silently drop
    whatever a future writer decides to put there, and silently is the problem.
    """
    with open(shard, "ab") as fh:
        fh.write(b"TRAILER!")
    streaming.rewrite_shard(shard, lambda tensor, raw: None)
    assert shard.read_bytes().endswith(b"TRAILER!")


def test_a_copy_larger_than_one_buffer_is_still_exact(shard, monkeypatch):
    """The chunked copy, exercised across many chunks rather than one."""
    monkeypatch.setattr(streaming, "CHUNK", 7)
    before = shard.read_bytes()
    streaming.rewrite_shard(shard, lambda tensor, raw: None)
    assert shard.read_bytes() == before


# ── the bound ────────────────────────────────────────────────────────────────────────────────

def test_a_rewrite_never_holds_more_than_a_tensor_at_a_time(tmp_path):
    """THE HALF THAT WOULD SURVIVE READING THE WHOLE FILE INTO MEMORY.

    Every correctness test above passes against an implementation that slurps the shard, edits it
    and writes it back, which is precisely the implementation this module exists instead of. So the
    bound is measured: eight tensors of 1 MB each, and the peak allocation stays near one of them
    rather than near the file.
    """
    each = 256 * 1024                                   # 1 MB of float32
    path = tmp_path / "big.safetensors"
    save_file({f"t{i}": torch.zeros(each) for i in range(8)}, str(path))
    one = each * 4
    assert path.stat().st_size > 6 * one, "the fixture is not large enough to tell the two apart"

    tracemalloc.start()
    try:
        streaming.rewrite_shard(path, lambda tensor, raw: bytes(tensor.nbytes))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Room for the tensor read, its replacement, and the interpreter's own noise, and still far
    # below the file. A slurping implementation lands at eight times a tensor or more.
    assert peak < 4 * one, (
        f"peak allocation was {peak / 1024:.0f} KiB against a {one / 1024:.0f} KiB tensor, so the "
        f"rewrite is holding more of the shard than one tensor at a time")
