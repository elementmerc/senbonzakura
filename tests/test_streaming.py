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


# ── the layer index ──────────────────────────────────────────────────────────────────────────
#
# The thing being guarded here is NOT "does it find the layers on a checkpoint we already
# support". Every version of this reader does that. It is what happens on a layout it does not
# know, because the failure that costs a five-day run is the silent one: a reader that matches
# `layers` and meets `h`, indexes nothing, and reports a checkpoint with no layers to edit as a
# checkpoint edited successfully. So most of these build a layout the reader has never seen and
# assert it says so.


def _checkpoint(root, tensors_by_name, *, layers=None, config=None, shards=1):
    """A model directory: a config, and the tensors spread over `shards` files."""
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(config or {})
    if layers is not None:
        cfg.setdefault("num_hidden_layers", layers)
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    names = sorted(tensors_by_name)
    for s in range(shards):
        chunk = {n: tensors_by_name[n] for n in names[s::shards]}
        if chunk:
            name = "model.safetensors" if shards == 1 else f"model-{s + 1:05d}.safetensors"
            save_file(chunk, str(root / name), metadata={"format": "pt"})
    return root


def _dense(prefix="model.layers", n=3):
    """A plain decoder stack, two tensors a layer, plus the usual unlayered company."""
    out = {"model.embed_tokens.weight": torch.randn(8, 4), "lm_head.weight": torch.randn(8, 4)}
    for i in range(n):
        out[f"{prefix}.{i}.self_attn.o_proj.weight"] = torch.randn(4, 4)
        out[f"{prefix}.{i}.mlp.down_proj.weight"] = torch.randn(4, 6)
    return out


def test_it_indexes_a_plain_decoder_stack(tmp_path):
    root = _checkpoint(tmp_path / "m", _dense(n=3), layers=3)
    idx = streaming.index_layers(root)

    assert idx.count == 3
    assert [sorted(layer) for layer in idx.layers] == [
        [f"model.layers.{i}.mlp.down_proj.weight",
         f"model.layers.{i}.self_attn.o_proj.weight"] for i in range(3)]
    # Everything outside the stack, and nothing from inside it.
    assert sorted(idx.shared) == ["lm_head.weight", "model.embed_tokens.weight"]


@pytest.mark.parametrize("prefix", ["model.layers", "transformer.h", "model.decoder.layers",
                                    "backbone.blocks", "encoder.layer"])
def test_the_block_can_be_called_anything(tmp_path, prefix):
    """THE POINT OF DECIDING ON THE CONFIG'S COUNT RATHER THAN ON THE WORD "layers".

    `layers` on most architectures, `h` on GPT-2 descendants, `blocks` elsewhere. A reader that
    knows three spellings reports clean on the fourth, which is this project's oldest defect and
    the reason nothing here matches a block name at all.
    """
    root = _checkpoint(tmp_path / prefix.replace(".", "_"), _dense(prefix, 4), layers=4)
    idx = streaming.index_layers(root)
    assert idx.count == 4
    assert all(len(layer) == 2 for layer in idx.layers)


def test_the_layer_is_the_first_index_and_not_the_expert(tmp_path):
    """A mixture-of-experts name carries two indices, and picking the wrong one is silent.

    Built with AS MANY EXPERTS AS LAYERS on purpose, so the expert position covers exactly the
    same set of values as the layer position and nothing but the ordering rule separates them.
    Choosing the expert would produce an index with the right number of layers, the right number
    of tensors, and every tensor filed under the wrong layer.
    """
    both = 4
    weights = {"model.embed_tokens.weight": torch.randn(8, 4)}
    for i in range(both):
        weights[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(4, 4)
        for e in range(both):
            weights[f"model.layers.{i}.mlp.experts.{e}.w2.weight"] = torch.randn(4, 6)
    root = _checkpoint(tmp_path / "moe", weights, layers=both)
    idx = streaming.index_layers(root)

    for i, layer in enumerate(idx.layers):
        assert all(f".layers.{i}." in name for name in layer), (
            f"layer {i} holds {sorted(layer)}, which is grouped by expert rather than by layer")
        assert len(layer) == both + 1


def test_a_fused_expert_stack_indexes_like_any_other_tensor(tmp_path):
    """Granite stores the fused stack, LFM stores per-expert tensors, and both are real.

    Settled against real checkpoints on 2026-09-22. The index does not care, and this pins that:
    a fused rank-3 weight carries one index, the layer's, and lands in exactly one layer.
    """
    weights = {"model.embed_tokens.weight": torch.randn(8, 4)}
    for i in range(3):
        weights[f"model.layers.{i}.block_sparse_moe.output_linear.weight"] = torch.randn(4, 4, 6)
    root = _checkpoint(tmp_path / "granite", weights, layers=3)
    idx = streaming.index_layers(root)
    assert [len(layer) for layer in idx.layers] == [1, 1, 1]
    assert not idx.shared or sorted(idx.shared) == ["model.embed_tokens.weight"]


def test_tensors_are_found_across_several_shards(tmp_path):
    """A real checkpoint of the size this module exists for is always sharded."""
    root = _checkpoint(tmp_path / "split", _dense(n=6), layers=6, shards=3)
    idx = streaming.index_layers(root)
    assert idx.count == 6
    assert len(idx.shards()) == 3
    assert all(len(layer) == 2 for layer in idx.layers)


def test_it_reads_the_layer_count_from_a_nested_text_config(tmp_path):
    """A multimodal checkpoint keeps the part this tool edits one level down."""
    root = _checkpoint(tmp_path / "mm", _dense(n=2),
                       config={"text_config": {"num_hidden_layers": 2}})
    assert streaming.index_layers(root).count == 2


@pytest.mark.parametrize("key", ["n_layer", "num_layers", "num_decoder_layers"])
def test_the_count_can_be_spelled_several_ways(tmp_path, key):
    root = _checkpoint(tmp_path / key, _dense(n=2), config={key: 2})
    assert streaming.index_layers(root).count == 2


# ── the refusals, which are the reason it is worth writing ───────────────────────────────────


def test_a_config_with_no_layer_count_is_refused(tmp_path):
    root = _checkpoint(tmp_path / "nocount", _dense(n=2), config={"model_type": "mystery"})
    with pytest.raises(streaming.ShardError, match="no config declares a layer count"):
        streaming.index_layers(root)


def test_a_naming_layout_the_reader_does_not_know_is_refused_not_guessed(tmp_path):
    """THE FAILURE THIS WHOLE GROUP EXISTS FOR.

    The config says four layers and no index position covers 0 to 3, because this checkpoint
    spells its stack with a separator the reader does not split on. The wrong answer is an index
    with four empty layers and a successful return; a run on that edits nothing and reports DONE.
    """
    weights = {f"model_layers_{i}_o_proj.weight": torch.randn(4, 4) for i in range(4)}
    root = _checkpoint(tmp_path / "odd", weights, layers=4)
    with pytest.raises(streaming.ShardError, match="naming layout this reader does not know"):
        streaming.index_layers(root)


def test_a_config_claiming_more_layers_than_the_weights_hold_is_refused(tmp_path):
    """The config and the weights are two accounts of the same thing, and they disagree."""
    root = _checkpoint(tmp_path / "short", _dense(n=3), layers=8)
    with pytest.raises(streaming.ShardError, match="naming layout this reader does not know"):
        streaming.index_layers(root)


def test_a_tensor_in_two_shards_is_refused(tmp_path):
    """Editing one copy leaves the other, and a loader may read either."""
    root = tmp_path / "dup"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"num_hidden_layers": 1}), encoding="utf-8")
    shared = {"model.layers.0.o_proj.weight": torch.randn(4, 4)}
    save_file(shared, str(root / "model-00001.safetensors"))
    save_file(shared, str(root / "model-00002.safetensors"))
    with pytest.raises(streaming.ShardError, match="appears in more than one shard"):
        streaming.index_layers(root)


def test_a_directory_with_no_shards_is_refused(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"num_hidden_layers": 2}), encoding="utf-8")
    with pytest.raises(streaming.ShardError, match="no safetensors shards"):
        streaming.index_layers(root)


# ── what the preflight asks it ───────────────────────────────────────────────────────────────


def test_the_budget_is_the_widest_layer_and_not_the_average(tmp_path):
    """Budgeting the mean fits every layer but one, and fails hours into the run.

    Architectures that put a dense block on some layers and a much larger mixture-of-experts
    block on others are real, so the layer sizes are deliberately uneven here.
    """
    weights = {"model.embed_tokens.weight": torch.randn(8, 4)}
    weights["model.layers.0.mlp.down_proj.weight"] = torch.randn(4, 4)
    weights["model.layers.1.mlp.down_proj.weight"] = torch.randn(64, 64)
    weights["model.layers.2.mlp.down_proj.weight"] = torch.randn(4, 4)
    root = _checkpoint(tmp_path / "uneven", weights, layers=3)
    idx = streaming.index_layers(root)

    widest, nbytes = idx.widest()
    assert widest == 1
    assert nbytes == 64 * 64 * 4
    assert nbytes > sum(idx.nbytes(i) for i in range(3)) / 3


def test_the_index_reads_no_weights(tmp_path):
    """A preflight has to answer before the user has committed to anything.

    The whole index of a real checkpoint is a few hundred kilobytes of JSON, and reading a single
    layer's weights instead would make it cost gigabytes on exactly the machine that cannot spare
    them. Measured rather than asserted in a comment, because every other test in this group
    passes against a version that loads the lot.
    """
    weights = {"model.embed_tokens.weight": torch.randn(8, 4)}
    for i in range(4):
        # 1 MiB a layer, so slurping is unmistakable against a header of a few hundred bytes.
        weights[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(256, 1024)
    root = _checkpoint(tmp_path / "big", weights, layers=4)
    one_layer = 256 * 1024 * 4

    tracemalloc.start()
    try:
        idx = streaming.index_layers(root)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert idx.nbytes(0) == one_layer
    assert peak < one_layer // 4, (
        f"indexing peaked at {peak / 1024:.0f} KiB against a {one_layer / 1024:.0f} KiB layer, "
        f"so it is reading weights rather than headers")


def test_the_writers_come_from_the_same_list_the_editor_walks(tmp_path):
    """ONE LIST. A streaming bake with its own idea of what to edit is the third copy.

    The second copy already drifted far enough that the guard refused a Qwen3.5 the editor could
    edit perfectly well, which is why `writers.py` exists and why this asserts identity with it
    rather than re-stating the names.
    """
    from senbonzakura import writers

    weights = {"model.embed_tokens.weight": torch.randn(8, 4)}
    for i in range(2):
        weights[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(4, 4)
        weights[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(4, 6)
        # Read by the residual stream, never written to it, so never edited.
        weights[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(6, 4)
        weights[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(4)
    root = _checkpoint(tmp_path / "w", weights, layers=2)
    idx = streaming.index_layers(root)

    for i in range(2):
        assert sorted(idx.writers(i)) == [
            f"model.layers.{i}.mlp.down_proj.weight",
            f"model.layers.{i}.self_attn.o_proj.weight"]
        # Not a restatement of the names: the same predicate, over the same tensors.
        assert set(idx.writers(i)) == {
            n for n in idx.layers[i] if writers.is_writer_tensor(n)}


def test_the_conv_control_arm_reaches_the_streaming_path_too(tmp_path):
    """`--skip-conv-ablation` is a control arm, and a control that only half applies is not one."""
    weights = {}
    for i in range(2):
        weights[f"model.layers.{i}.conv.out_proj.weight"] = torch.randn(4, 4)
        weights[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(4, 6)
    root = _checkpoint(tmp_path / "conv", weights, layers=2)
    idx = streaming.index_layers(root)

    assert len(idx.writers(0, ablate_conv=True)) == 2
    assert sorted(idx.writers(0, ablate_conv=False)) == ["model.layers.0.mlp.down_proj.weight"]


def test_the_module_needs_no_torch(tmp_path):
    """THE PROPERTY THAT MAKES THIS MODULE TESTABLE ANYWHERE, so it is asserted rather than hoped.

    `streaming.py` reads a file format. Importing it must not drag in torch, transformers or
    optuna, because the machine this whole rung exists for is the one that cannot afford them and
    because a bake half that needs a GPU to test is a bake half nobody tests.

    Checked in a subprocess with a fresh interpreter: asserting `'torch' not in sys.modules` in
    this one would pass trivially, since the test suite imported torch at the top of this file
    long before the assertion ran.
    """
    import subprocess
    import sys
    from pathlib import Path

    probe = ("import sys; import senbonzakura.streaming; "
             "heavy = [m for m in ('torch', 'transformers', 'optuna') if m in sys.modules]; "
             "print(','.join(heavy))")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1], timeout=120, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "", (
        f"importing senbonzakura.streaming pulled in {r.stdout.strip()}, which is what this "
        f"module is shaped to avoid")
