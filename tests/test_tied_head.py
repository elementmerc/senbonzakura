# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""One logical model must convert to one file, whatever tensors its source happened to ship.

WHY THIS FILE EXISTS

`llama-quantize` decides the embedding's precision from whether a separate output head is present.
A lone `token_embd` is also the output projection, so it is kept accurate; when a separate head
exists the embedding is only an embedding and is quantised cheaply. That is sensible behaviour
given what it is handed, and it means a checkpoint that ships an `lm_head.weight` its own config
calls redundant gets a materially different file from one that does not.

MEASURED on a tied model at Q3_K_L, 2026-09-06, which is what settled the fix:

    duplicate present   token_embd Q3_K   embedding error vs f16   0.1509   1007 KiB
    duplicate absent    token_embd Q6_K   embedding error vs f16   0.0177    897 KiB

The output projection is Q6_K in BOTH. So this was never a trade between size and accuracy: the
duplicate costs both at once, and which you got was decided by an accident of the source.

WHAT IS NOT AFFECTED, and it is worth stating because the fix was nearly aimed at it: a model whose
config declares `tie_word_embeddings` FALSE has a real, distinct head, and `output.weight` at Q6_K
beside a cheap `token_embd` is correct there. Qwen3-30B-A3B is such a model.

THE GUARD

Dropping a tensor because a config says it is redundant is only safe if it IS redundant. A
checkpoint whose head differs from its embedding while the config claims tying is telling two
different stories, and this code cannot know which is true, so it refuses instead of choosing.
"""
import json
import pathlib

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from senbonzakura import convert

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

VOCAB, HIDDEN = 64, 32


def _checkpoint(path, *, tie, head="same"):
    """A minimal checkpoint. `head`: "none", "same" (a duplicate) or "different" (a contradiction)."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    embed = rng.standard_normal((VOCAB, HIDDEN), dtype=np.float32)
    sd = {convert.TIED_EMBED: embed,
          "model.layers.0.input_layernorm.weight": np.ones(HIDDEN, dtype=np.float32)}
    if head == "same":
        sd[convert.TIED_HEAD] = embed.copy()
    elif head == "different":
        sd[convert.TIED_HEAD] = embed + 0.5
    save_file(sd, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "tie_word_embeddings": tie,
        "vocab_size": VOCAB, "hidden_size": HIDDEN}), encoding="utf-8")
    return path


# ── what the state check decides ─────────────────────────────────────────────────────

def test_an_untied_config_is_left_alone(tmp_path):
    """THE 30B'S CASE, and the one this fix must not touch.

    `tie_word_embeddings` false means the head is real. Its separate `output.weight` is correct and
    removing it would change the model.
    """
    d = _checkpoint(tmp_path / "m", tie=False, head="different")
    verdict, _shard, detail = convert.tied_head_state(d)
    assert verdict == "absent", detail


def test_a_tied_config_with_no_head_has_nothing_to_do(tmp_path):
    d = _checkpoint(tmp_path / "m", tie=True, head="none")
    verdict, _shard, _detail = convert.tied_head_state(d)
    assert verdict == "absent"


def test_a_duplicate_head_is_recognised_as_redundant(tmp_path):
    d = _checkpoint(tmp_path / "m", tie=True, head="same")
    verdict, shard, detail = convert.tied_head_state(d)
    assert verdict == "redundant"
    assert shard.name == "model.safetensors"
    assert "byte-identical" in detail


def test_a_head_that_differs_is_a_contradiction_not_a_duplicate(tmp_path):
    """THE GUARD. The config says these are one tensor and the weights say otherwise."""
    d = _checkpoint(tmp_path / "m", tie=True, head="different")
    verdict, _shard, detail = convert.tied_head_state(d)
    assert verdict == "contradicts"
    assert "bytes differ" in detail
    assert "offset" in detail, "the message must locate the disagreement, not just assert it"


def test_equality_is_exact_and_not_a_tolerance(tmp_path):
    """A tensor that merely rounds to another is not the same tensor.

    The claim being checked is identity. A near-miss is a contradiction with a small number in it,
    and treating it as a duplicate would silently discard whatever the difference encoded.
    """
    d = _checkpoint(tmp_path / "m", tie=True, head="same")
    sd = load_file(d / "model.safetensors")
    sd[convert.TIED_HEAD] = sd[convert.TIED_HEAD] + np.float32(1e-7)
    save_file(sd, d / "model.safetensors", metadata={"format": "pt"})
    verdict, _shard, _detail = convert.tied_head_state(d)
    assert verdict == "contradicts"


def test_a_head_with_no_embedding_to_tie_to_is_a_contradiction(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    save_file({convert.TIED_HEAD: np.zeros((VOCAB, HIDDEN), dtype=np.float32)},
              d / "model.safetensors", metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps(
        {"architectures": ["LlamaForCausalLM"], "tie_word_embeddings": True}), encoding="utf-8")
    verdict, _shard, _detail = convert.tied_head_state(d)
    assert verdict == "contradicts"


def test_shapes_that_disagree_are_a_contradiction(tmp_path):
    d = _checkpoint(tmp_path / "m", tie=True, head="none")
    sd = load_file(d / "model.safetensors")
    sd[convert.TIED_HEAD] = np.zeros((VOCAB + 1, HIDDEN), dtype=np.float32)
    save_file(sd, d / "model.safetensors", metadata={"format": "pt"})
    verdict, _shard, detail = convert.tied_head_state(d)
    assert verdict == "contradicts"
    assert str(VOCAB + 1) in detail


# ── building the view ────────────────────────────────────────────────────────────────

def test_the_view_drops_the_head_and_keeps_everything_else(tmp_path):
    d = _checkpoint(tmp_path / "m", tie=True, head="same")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    view = convert.without_tied_head(d, tmp_path / "view", d / "model.safetensors",
                                     log=lambda _m: None)
    sd = load_file(view / "model.safetensors")
    assert convert.TIED_HEAD not in sd
    assert convert.TIED_EMBED in sd
    assert (view / "config.json").is_file()
    assert (view / "tokenizer.json").is_file()


def test_the_view_symlinks_rather_than_copying_the_weights(tmp_path):
    """The reason this approach is affordable at all.

    Only the shard holding the duplicate is rewritten; every other file is a link. A checkpoint
    that had to be copied to be converted would double the disk a conversion needs, which on a
    rented pod is the difference between a job running and a job dying.
    """
    d = _checkpoint(tmp_path / "m", tie=True, head="same")
    other = d / "model-00002-of-00002.safetensors"
    save_file({"model.layers.1.input_layernorm.weight": np.ones(HIDDEN, dtype=np.float32)},
              other, metadata={"format": "pt"})
    view = convert.without_tied_head(d, tmp_path / "view", d / "model.safetensors",
                                     log=lambda _m: None)
    assert (view / other.name).is_symlink(), "an untouched shard was copied instead of linked"
    assert not (view / "model.safetensors").is_symlink(), "the rewritten shard must be a real file"


def test_the_index_is_rewritten_so_the_converter_stays_consistent(tmp_path):
    """The converter checks the tensors it found against the index and raises on a mismatch, so an
    index still promising the dropped tensor would break the conversion it is meant to fix.
    """
    d = _checkpoint(tmp_path / "m", tie=True, head="same")
    head_bytes = VOCAB * HIDDEN * 4
    (d / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 999_999},
        "weight_map": {convert.TIED_HEAD: "model.safetensors",
                       convert.TIED_EMBED: "model.safetensors"}}), encoding="utf-8")
    view = convert.without_tied_head(d, tmp_path / "view", d / "model.safetensors",
                                     log=lambda _m: None)
    doc = json.loads((view / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert convert.TIED_HEAD not in doc["weight_map"]
    assert convert.TIED_EMBED in doc["weight_map"]
    assert doc["metadata"]["total_size"] == 999_999 - head_bytes


def test_the_view_keeps_the_source_directory_name(tmp_path):
    """The converter derives `general.name` from the directory it is pointed at.

    A view built in a randomly named temp directory stamped that into the model's metadata: a
    published GGUF called "Senbonzakura Tied Zkxhdz01". Found by comparing two files that should
    have been byte-identical and were not.
    """
    d = _checkpoint(tmp_path / "my-model", tie=True, head="same")
    view = convert.without_tied_head(d, tmp_path / "scratch" / d.name, d / "model.safetensors",
                                     log=lambda _m: None)
    assert view.name == "my-model"


# ── dtypes, which is where the first version broke ───────────────────────────────────
# The first version read tensors through `safetensors.safe_open(framework="np")` and crashed with
# "data type 'bfloat16' not understood" on the very checkpoints it existed for: numpy has no
# bfloat16, and bf16 is what most modern checkpoints ship. The fixtures here were all float32, so
# nothing caught it until a peer converted a stock Qwen3-1.7B on real hardware. Comparing and
# copying raw bytes rather than decoded tensors is what makes the dtype irrelevant.

def _raw_checkpoint(path, dtype, *, tie=True, head="same", nbytes=2):
    """A checkpoint written by hand, so a dtype numpy cannot represent can still be built."""
    import struct
    path.mkdir(parents=True, exist_ok=True)
    n = 64 * 32
    embed = bytes((i * 7 + 3) % 256 for i in range(n * nbytes))
    other = embed if head == "same" else bytes((b + 1) % 256 for b in embed)
    entries, blob, cur = {}, b"", 0
    for name, payload in ((convert.TIED_EMBED, embed),
                          *(((convert.TIED_HEAD, other),) if head != "none" else ())):
        entries[name] = {"dtype": dtype, "shape": [64, 32],
                         "data_offsets": [cur, cur + len(payload)]}
        blob += payload
        cur += len(payload)
    hdr = json.dumps(entries, separators=(",", ":")).encode()
    hdr += b" " * ((-len(hdr)) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr + blob)
    (path / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": tie}), encoding="utf-8")
    return path


@pytest.mark.parametrize(("dtype", "nbytes"), [("BF16", 2), ("F16", 2), ("F32", 4), ("F8_E4M3", 1)])
def test_a_duplicate_is_recognised_whatever_the_dtype(tmp_path, dtype, nbytes):
    """THE REGRESSION THIS SECTION IS NAMED FOR.

    BF16 is the one that shipped broken. F8_E4M3 is here because no numeric library in this
    process can represent it either, and the check must not start caring.
    """
    d = _raw_checkpoint(tmp_path / dtype, dtype, head="same", nbytes=nbytes)
    verdict, _shard, _detail = convert.tied_head_state(d)
    assert verdict == "redundant"


@pytest.mark.parametrize(("dtype", "nbytes"), [("BF16", 2), ("F8_E4M3", 1)])
def test_a_contradiction_is_caught_whatever_the_dtype(tmp_path, dtype, nbytes):
    d = _raw_checkpoint(tmp_path / dtype, dtype, head="different", nbytes=nbytes)
    verdict, _shard, detail = convert.tied_head_state(d)
    assert verdict == "contradicts"
    assert dtype in detail


def test_dtypes_that_disagree_are_a_contradiction(tmp_path):
    """Same bytes, different declared types, is not one tensor."""
    d = _raw_checkpoint(tmp_path / "m", "BF16", head="same")
    hdr = convert.safetensors_header(d / "model.safetensors")
    assert hdr[convert.TIED_HEAD]["dtype"] == "BF16"
    raw = (d / "model.safetensors").read_bytes()
    n = int.from_bytes(raw[:8], "little")
    doc = json.loads(raw[8:8 + n])
    doc[convert.TIED_HEAD]["dtype"] = "F16"
    blob = json.dumps(doc, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    (d / "model.safetensors").write_bytes(
        len(blob).to_bytes(8, "little") + blob + raw[8 + n:])
    verdict, _shard, detail = convert.tied_head_state(d)
    assert verdict == "contradicts"
    assert "BF16" in detail and "F16" in detail


def test_the_copy_preserves_every_surviving_tensor_byte_for_byte(tmp_path):
    """The rewrite decodes nothing, so what it keeps is identical by construction, not by luck."""
    d = _raw_checkpoint(tmp_path / "m", "BF16", head="same")
    src = d / "model.safetensors"
    before = convert.safetensors_header(src)
    dropped = convert.copy_shard_without(src, tmp_path / "out.safetensors", convert.TIED_HEAD)
    after = convert.safetensors_header(tmp_path / "out.safetensors")
    assert convert.TIED_HEAD not in after
    assert set(after) == set(before) - {convert.TIED_HEAD}
    assert after[convert.TIED_EMBED]["dtype"] == "BF16"
    same, where = convert.raw_bytes_equal(src, before[convert.TIED_EMBED],
                                          tmp_path / "out.safetensors", after[convert.TIED_EMBED])
    assert same, f"the surviving embedding changed at byte {where}"
    assert dropped["shape"] == [64, 32]


def test_the_rewritten_header_is_eight_byte_aligned(tmp_path):
    """A misaligned header is valid JSON that some readers accept and others do not, which is the
    worst of both. The reference writer pads with spaces; so does this.
    """
    d = _raw_checkpoint(tmp_path / "m", "BF16", head="same")
    convert.copy_shard_without(d / "model.safetensors", tmp_path / "out.safetensors",
                               convert.TIED_HEAD)
    n = int.from_bytes((tmp_path / "out.safetensors").read_bytes()[:8], "little")
    assert n % 8 == 0


def test_no_part_file_is_left_behind(tmp_path):
    """The shard is written to .part and renamed, so an interrupted copy cannot be mistaken for a
    finished one by anything that comes later.
    """
    d = _raw_checkpoint(tmp_path / "m", "BF16", head="same")
    convert.copy_shard_without(d / "model.safetensors", tmp_path / "out.safetensors",
                               convert.TIED_HEAD)
    assert not list(tmp_path.glob("*.part"))


def test_the_header_reader_refuses_a_file_that_is_not_safetensors(tmp_path):
    """Arbitrary bytes read as a length field give a huge number, so the size guard catches this
    before the JSON parse does. Either refusal is correct; what matters is that it is a
    ConvertError with the filename in it and not a MemoryError from a 7-exabyte read.
    """
    junk = tmp_path / "j.safetensors"
    junk.write_bytes(b"not a safetensors file at all")
    with pytest.raises(convert.ConvertError, match=r"j\.safetensors"):
        convert.safetensors_header(junk)


def test_the_header_reader_refuses_a_header_that_is_not_json(tmp_path):
    junk = tmp_path / "j.safetensors"
    body = b"this is not json"
    junk.write_bytes(len(body).to_bytes(8, "little") + body)
    with pytest.raises(convert.ConvertError, match="not readable as safetensors"):
        convert.safetensors_header(junk)


def test_the_header_reader_refuses_an_implausible_header_length(tmp_path):
    """A corrupt length field must not become a multi-gigabyte read."""
    junk = tmp_path / "j.safetensors"
    junk.write_bytes((1 << 40).to_bytes(8, "little") + b"{}")
    with pytest.raises(convert.ConvertError, match="implausible header"):
        convert.safetensors_header(junk)


def test_convert_needs_neither_torch_nor_numpy_for_any_of_this(tmp_path):
    """The clean room requires `convert` to refuse cleanly on an install with neither.

    The first version imported numpy to compare the tensors, which would have made this module's
    behaviour depend on an optional dependency for a question answerable from the file header.
    """
    import ast
    src = pathlib.Path(convert.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names = [n.name.split(".")[0] for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert "torch" not in names and "numpy" not in names, (
            f"convert.py imports {names} at line {node.lineno}; this module has to work on an "
            f"install that has neither")
