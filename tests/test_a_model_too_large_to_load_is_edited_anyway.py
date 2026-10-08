# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The streamed bake: the same edit as the resident one, with the model never resident.

WHY THIS FILE EXISTS

`streaming.py` shipped a shard reader, a layer index, a single-tensor read and a shard rewriter,
and nothing in the package called any of them. This covers the thing that calls them, and the two
properties that make it trustworthy rather than merely working:

**One: it must not touch the source.** `rewrite_shard` replaces a shard by renaming over it, so a
bake pointed at a Hugging Face cache would consume the base model and leave an abliterated one
wearing its name. The output is a hard-linked copy where the filesystem allows it, which is free
and is safe only because a rename swaps a directory entry rather than writing through to the
inode. That is a real property of `os.replace` and not a hope, and the test below is what holds
it: if anything in this path ever opens an output shard for in-place writing, the hard link
becomes a route into the source and this is what notices.

**Two: it must produce the same weights as the resident editor.** The whole value of a streamed
bake is that it is the same arithmetic from a different place. So the comparison is byte for byte
against `orthogonalize_np_` applied directly, not an assertion that something changed.

And the one that is neither: re-running must not project twice. Twice-projected weights are not a
stronger edit, they are a different and undocumented one, and nothing in the file would say so.
"""
import json
import pathlib
import struct

import numpy as np
import pytest
import torch

from senbonzakura import streambake, streaming

HIDDEN = 8
LAYERS = 3


def _write_shard(path, arrays):
    """A minimal safetensors file, written by hand so the test owns its own fixture."""
    # CAST FIRST, then measure. Measuring the float64 array `standard_normal` returns and writing
    # its float32 cast declared every tensor at twice its length, and the header's own
    # self-consistency check caught it on the first run.
    arrays = {name: np.asarray(arr, dtype=np.float32) for name, arr in arrays.items()}
    header, offset = {}, 0
    for name, arr in arrays.items():
        nbytes = arr.nbytes
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode("utf-8")
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for arr in arrays.values():
            f.write(arr.astype(np.float32).tobytes())


def _checkpoint(root, *, layers=LAYERS, hidden=HIDDEN, seed=0):
    """A tiny Llama-shaped checkpoint on disk: two shards, one residual writer per layer."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    (root / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": layers, "hidden_size": hidden}), encoding="utf-8")

    per_shard = [{}, {}]
    index = {}
    for i in range(layers):
        which = i % 2
        for suffix in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
            name = f"model.layers.{i}.{suffix}"
            per_shard[which][name] = rng.standard_normal((hidden, hidden))
            index[name] = f"model-{which}.safetensors"
    per_shard[0]["model.embed_tokens.weight"] = rng.standard_normal((16, hidden))
    index["model.embed_tokens.weight"] = "model-0.safetensors"
    per_shard[1]["model.norm.weight"] = rng.standard_normal((hidden,))
    index["model.norm.weight"] = "model-1.safetensors"

    for which, arrays in enumerate(per_shard):
        _write_shard(root / f"model-{which}.safetensors", arrays)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": index}), encoding="utf-8")
    return root


def _directions(layers=LAYERS, hidden=HIDDEN, k=2, seed=7):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((layers + 1, k, hidden)).astype(np.float32)
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


def _read(path, name):
    for t in streaming.tensors(str(path)):
        if t.name == name:
            return streaming.load_tensor(str(path), t).clone()
    raise AssertionError(f"{name} not in {path}")


@pytest.fixture
def model(tmp_path):
    return _checkpoint(tmp_path / "source")


# ── the source survives, which is the property that protects somebody's cache ────────

def _digest_tree(root):
    import hashlib
    out = {}
    for p in sorted(pathlib.Path(root).iterdir()):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_the_source_checkpoint_is_byte_identical_after_a_bake(model, tmp_path):
    """THE ONE THAT PROTECTS DATA.

    The output is hard-linked from the source where the filesystem allows it, which is free and
    correct only because `rewrite_shard` RENAMES over the shard rather than writing into it. If
    that ever changes, the hard link becomes a route into the source and this fails.
    """
    before = _digest_tree(model)
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, _directions(), log=lambda _m: None)
    assert _digest_tree(model) == before, (
        "the bake changed the source checkpoint, which is what the hard-linked output exists to "
        "make impossible")


def test_the_output_actually_differs_from_the_source(model, tmp_path):
    """The other half, because a bake that changed nothing would also pass the test above."""
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, _directions(), log=lambda _m: None)
    name = "model.layers.0.self_attn.o_proj.weight"
    assert not torch.equal(_read(model / "model-0.safetensors", name),
                           _read(out / "model-0.safetensors", name))


@pytest.mark.parametrize("where", ["same", "inside-source", "contains-source"])
def test_an_output_that_could_reach_the_source_is_refused(model, tmp_path, where):
    target = {"same": model, "inside-source": model / "out",
              "contains-source": model.parent}[where]
    with pytest.raises(streambake.StreamBakeError):
        streambake.check_output_is_safe(model, target)


def test_an_output_inside_a_model_cache_is_refused(model, tmp_path):
    """A cache entry edited in place is indistinguishable from the base model it claims to be, for
    every tool on the machine, which makes it the worst of the four cases rather than the mildest.
    """
    cache = tmp_path / "huggingface" / "hub" / "models--org--name" / "snapshots" / "abc"
    cache.mkdir(parents=True)
    with pytest.raises(streambake.StreamBakeError, match="model cache"):
        streambake.check_output_is_safe(model, cache)


def test_an_ordinary_working_directory_is_allowed(model, tmp_path):
    assert streambake.check_output_is_safe(model, tmp_path / "work") is None


# ── the same arithmetic as the resident editor, byte for byte ────────────────────────

def test_the_streamed_edit_equals_the_resident_edit(model, tmp_path):
    """THE CLAIM THIS MODULE MAKES. Not "the weights changed": the same weights the resident
    editor would have produced, from a path where the model was never loaded.
    """
    from senbonzakura.cli import orthogonalize_np_

    dirs = _directions()
    name = "model.layers.1.mlp.down_proj.weight"
    expected = _read(model / "model-1.safetensors", name)
    orthogonalize_np_(expected, torch.from_numpy(dirs[2]), 1.0, 0.0, 0)

    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, dirs, log=lambda _m: None)
    assert torch.equal(_read(out / "model-1.safetensors", name), expected)


def test_single_mode_applies_one_set_to_every_layer(model, tmp_path):
    """`single` is how the resident path spells Heretic's formulation, and the difference between
    the two modes has to be the direction used rather than anything else.
    """
    from senbonzakura.cli import orthogonalize_np_

    dirs = _directions()
    name = "model.layers.2.self_attn.o_proj.weight"
    expected = _read(model / "model-0.safetensors", name)
    # Position 1's set, not position 3's, even though this is layer 2.
    orthogonalize_np_(expected, torch.from_numpy(dirs[1]), 1.0, 0.0, 0)

    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, dirs, mode="single", log=lambda _m: None)
    assert torch.equal(_read(out / "model-0.safetensors", name), expected)


def test_only_the_residual_writers_are_touched(model, tmp_path):
    """The embedding and the final norm are not residual writers, and an edit that reached them
    would be a different intervention reported under this one's name.
    """
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, _directions(), log=lambda _m: None)
    for shard, name in (("model-0.safetensors", "model.embed_tokens.weight"),
                        ("model-1.safetensors", "model.norm.weight")):
        assert torch.equal(_read(model / shard, name), _read(out / shard, name)), name


def test_k_is_honoured_and_clamped_to_what_the_set_holds(model, tmp_path):
    from senbonzakura.cli import orthogonalize_np_

    dirs = _directions(k=2)
    name = "model.layers.0.self_attn.o_proj.weight"
    expected = _read(model / "model-0.safetensors", name)
    orthogonalize_np_(expected, torch.from_numpy(dirs[1][:1]), 1.0, 0.0, 0)

    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    stats = streambake.bake(out, dirs, K=1, log=lambda _m: None)
    assert stats["K"] == 2, "the set still holds two; K selects from it"
    assert torch.equal(_read(out / "model-0.safetensors", name), expected)


# ── resuming, and the twice-projected weights it exists to prevent ──────────────────

def test_a_second_run_is_a_no_op_rather_than_a_second_projection(model, tmp_path):
    """THE DEFECT WITH NO SYMPTOM. Twice-projected weights are not a stronger edit, they are a
    different one, and the checkpoint would carry nothing saying it happened.
    """
    dirs = _directions()
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, dirs, log=lambda _m: None)
    after_first = _digest_tree(out)

    stats = streambake.bake(out, dirs, log=lambda _m: None)
    assert stats["shards"] == 0
    assert _digest_tree(out) == after_first, "the second run edited the weights again"


def test_resuming_with_a_different_strength_is_refused(model, tmp_path):
    """Half a checkpoint at one strength and half at another, with no field recording it. It would
    load, run and score, and every number from it would describe a model nobody designed.
    """
    dirs = _directions()
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, dirs, strength=1.0, log=lambda _m: None)
    with pytest.raises(streambake.StreamBakeError, match="part-finished"):
        streambake.bake(out, dirs, strength=0.5, log=lambda _m: None)


def test_an_interrupted_bake_resumes_from_the_shard_it_reached(model, tmp_path):
    """The resumable unit is the SHARD, because that is what `rewrite_shard` makes atomic. A
    layer-level marker would claim a granularity the writer does not have.
    """
    dirs = _directions()
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)

    real = streaming.rewrite_shard
    calls = []

    def one_then_stop(path, edit, **kw):
        calls.append(path)
        if len(calls) > 1:
            raise KeyboardInterrupt
        return real(path, edit, **kw)

    streaming.rewrite_shard = one_then_stop
    try:
        with pytest.raises(KeyboardInterrupt):
            streambake.bake(out, dirs, log=lambda _m: None)
    finally:
        streaming.rewrite_shard = real

    assert len(streambake.read_progress(out)["done"]) == 1
    stats = streambake.bake(out, dirs, log=lambda _m: None)
    assert stats["resumed"] is True
    assert stats["shards"] == 1, "the finished shard was edited a second time"


def test_an_unreadable_progress_file_refuses_rather_than_starting_over(model, tmp_path):
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    (out / streambake.PROGRESS_NAME).write_text("{ not json", encoding="utf-8")
    with pytest.raises(streambake.StreamBakeError, match="project them twice"):
        streambake.bake(out, _directions(), log=lambda _m: None)


def test_a_progress_file_from_another_version_is_refused(model, tmp_path):
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    (out / streambake.PROGRESS_NAME).write_text(
        json.dumps({"version": 999, "done": []}), encoding="utf-8")
    with pytest.raises(streambake.StreamBakeError, match="different version"):
        streambake.bake(out, _directions(), log=lambda _m: None)


# ── the pre-flight, which is where a long job earns its keep ─────────────────────────

def test_directions_for_a_different_model_are_refused_before_any_byte_is_written(
        model, tmp_path):
    """A hidden size that does not match is directions for another model. The projection would
    still run if the arithmetic allowed it, and the checkpoint would load and mean nothing.
    """
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    before = _digest_tree(out)
    with pytest.raises(streambake.StreamBakeError, match="different model"):
        streambake.bake(out, _directions(hidden=HIDDEN + 1), log=lambda _m: None)
    assert _digest_tree(out) == before, "the refusal came after something had been written"


def test_a_direction_set_too_short_for_the_stack_is_refused(model, tmp_path):
    """The failure it prevents: the deepest layers silently unedited, run reporting success."""
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    short = _directions()[:LAYERS - 1]
    with pytest.raises(streambake.StreamBakeError, match="residual positions"):
        streambake.bake(out, short, log=lambda _m: None)


def test_a_two_dimensional_direction_set_is_refused_with_the_shape_it_wanted(model, tmp_path):
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    with pytest.raises(streambake.StreamBakeError, match=r"\[positions, K, hidden\]"):
        streambake.bake(out, np.zeros((4, HIDDEN), dtype=np.float32), log=lambda _m: None)


def test_a_checkpoint_with_no_recognised_writer_is_refused_rather_than_reported_as_baked(
        tmp_path):
    """Editing nothing and announcing success is the failure mode this project keeps meeting."""
    root = tmp_path / "odd"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "mystery", "num_hidden_layers": 2, "hidden_size": HIDDEN}),
        encoding="utf-8")
    # BOTH indices present, so the layer axis resolves and the refusal under test is the one
    # about writers rather than `streaming`'s earlier one about an unknown naming layout.
    names = [f"mystery.block.{i}.unknown_proj.weight" for i in range(2)]
    _write_shard(root / "model-0.safetensors",
                 {n: np.zeros((HIDDEN, HIDDEN)) for n in names})
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": dict.fromkeys(names, "model-0.safetensors")}),
        encoding="utf-8")
    out = streambake.prepare_output(root, tmp_path / "out", log=lambda _m: None)
    with pytest.raises(streambake.StreamBakeError, match="residual writer"):
        streambake.bake(out, _directions(layers=2), log=lambda _m: None)


def test_the_preflight_names_layers_it_cannot_reach(model, tmp_path, capsys):
    """A layer with no writer is normal for some architectures and is also what a name predicate
    looks like when it has met one it does not know. The bake cannot tell those apart, so it says
    so rather than choosing.
    """
    said = []
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    index = streaming.index_layers(out)
    plan = streambake.plan_edits(index)
    plan.pop("model.layers.0.self_attn.o_proj.weight")
    plan.pop("model.layers.0.mlp.down_proj.weight")
    streambake.preflight(index, _directions(), out=out, plan=plan, log=said.append)
    joined = " ".join(said)
    assert "hold no recognised residual writer" in joined
    assert "[0]" in joined


def test_the_plan_covers_every_layer_of_an_ordinary_checkpoint(model, tmp_path):
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    index = streaming.index_layers(out)
    assert streambake.unreachable_layers(index, streambake.plan_edits(index)) == ()


def test_the_peak_memory_reported_is_one_tensor_and_not_one_layer(model, tmp_path):
    """The claim that makes this worth building. If the figure ever becomes a layer's or a
    model's, the module has stopped doing what its docstring says.
    """
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    stats = streambake.bake(out, _directions(), log=lambda _m: None)
    index = streaming.index_layers(out)
    biggest_tensor = max(t.nbytes for layer in index.layers for _s, t in layer.values())
    assert stats["peak_tensor_bytes"] == biggest_tensor
    assert stats["peak_tensor_bytes"] < index.nbytes(0), "a layer is more than its largest tensor"


# ── the output copy ─────────────────────────────────────────────────────────────────

def test_preparing_the_output_brings_the_config_across(model, tmp_path):
    """A directory of edited shards with no config is not a model anyone can load."""
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    assert (out / "config.json").is_file()
    assert (out / "model.safetensors.index.json").is_file()


def test_preparing_an_output_twice_does_not_overwrite_edited_shards(model, tmp_path):
    """The resume path calls this again, and re-linking would undo the work it is resuming."""
    dirs = _directions()
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, dirs, log=lambda _m: None)
    edited = _digest_tree(out)
    streambake.prepare_output(model, out, log=lambda _m: None)
    assert _digest_tree(out) == edited


def test_a_missing_model_directory_is_named(tmp_path):
    with pytest.raises(streambake.StreamBakeError, match="does not exist"):
        streambake.prepare_output(tmp_path / "absent", tmp_path / "out", log=lambda _m: None)


# ── the digest, which is what makes a resume safe ───────────────────────────────────

def test_every_edit_parameter_changes_the_digest():
    """If a parameter is missing from the key, a resume can mix two different edits in one
    checkpoint with nothing recording it.
    """
    dirs = _directions()
    base = {"strength": 1.0, "sparsity": 0.0, "rounds": 0, "restore_norms": True,
            "ablate_conv": True}
    first = streambake.directions_digest(dirs, **base)
    for key, other in (("strength", 0.5), ("sparsity", 0.1), ("rounds", 2),
                       ("restore_norms", False), ("ablate_conv", False)):
        assert streambake.directions_digest(dirs, **{**base, key: other}) != first, key


def test_the_digest_follows_the_directions_themselves():
    base = {"strength": 1.0, "sparsity": 0.0, "rounds": 0, "restore_norms": True,
            "ablate_conv": True}
    assert (streambake.directions_digest(_directions(seed=1), **base)
            != streambake.directions_digest(_directions(seed=2), **base))


def test_the_digest_is_stable_across_calls():
    """A digest that moved between runs would refuse every resume it exists to allow."""
    base = {"strength": 1.0, "sparsity": 0.0, "rounds": 0, "restore_norms": True,
            "ablate_conv": True}
    dirs = _directions()
    assert (streambake.directions_digest(dirs, **base)
            == streambake.directions_digest(dirs, **base))


# ── the directions file, which is the seam between a rented card and a laptop ───────

def test_a_directions_file_round_trips(tmp_path):
    dirs = _directions()
    path = streambake.save_directions(
        tmp_path / "d.safetensors", dirs, model="org/model", mode="per_layer",
        provenance={"num_directions": 2, "seed": 0})
    back, meta = streambake.load_directions(path)
    assert np.array_equal(back, dirs)
    assert meta["model"] == "org/model"
    assert meta["mode"] == "per_layer"
    assert meta["num_directions"] == "2", "metadata values are strings in safetensors"


def test_the_file_records_the_geometry_so_a_reader_can_refuse_early(tmp_path):
    path = streambake.save_directions(
        tmp_path / "d.safetensors", _directions(), model="m", mode="single")
    _back, meta = streambake.load_directions(path)
    assert (meta["positions"], meta["K"], meta["hidden"]) == (str(LAYERS + 1), "2", str(HIDDEN))


def test_the_file_is_small_whatever_the_model(tmp_path):
    """THE PROPERTY THAT MAKES THE SPLIT WORTH HAVING. A direction set is positions x K x hidden
    floats and carries no weights, so a 200 GB checkpoint's directions still fit in an email.
    """
    path = streambake.save_directions(
        tmp_path / "d.safetensors", _directions(layers=80, hidden=8192, k=8),
        model="m", mode="per_layer")
    assert path.stat().st_size < 25 * 1024 * 1024


def test_a_bake_consumes_what_a_save_produced(tmp_path, model):
    """End to end across the seam: nothing in between touches the array."""
    path = streambake.save_directions(
        tmp_path / "d.safetensors", _directions(), model="m", mode="per_layer")
    arr, meta = streambake.load_directions(path)
    out = streambake.prepare_output(model, tmp_path / "out", log=lambda _m: None)
    stats = streambake.bake(out, arr, mode=meta["mode"], log=lambda _m: None)
    assert stats["tensors"] == LAYERS * 2


def test_some_other_safetensors_file_is_not_mistaken_for_directions(tmp_path):
    from safetensors.torch import save_file
    path = tmp_path / "weights.safetensors"
    save_file({"model.embed_tokens.weight": torch.zeros(4, 4)}, str(path))
    with pytest.raises(streambake.StreamBakeError, match="some other safetensors file"):
        streambake.load_directions(path)


def test_a_file_that_is_not_safetensors_at_all_is_named(tmp_path):
    path = tmp_path / "nope.safetensors"
    path.write_text("not safetensors", encoding="utf-8")
    with pytest.raises(streambake.StreamBakeError, match="could not be read"):
        streambake.load_directions(path)


def test_saving_a_two_dimensional_set_is_refused(tmp_path):
    with pytest.raises(streambake.StreamBakeError, match=r"\[positions, K, hidden\]"):
        streambake.save_directions(tmp_path / "d.safetensors", np.zeros((3, 8)),
                                   model="m", mode="per_layer")


# ── the command line, where the exit code is the interface ──────────────────────────

def test_the_command_is_registered_under_a_name_a_person_would_type():
    from senbonzakura.entry import DELEGATED
    assert DELEGATED["stream-bake"] == ("streambake", "main")


def test_the_parser_declares_what_its_help_describes():
    dests = {a.dest for a in streambake.build_parser()._actions}
    for flag in ("model", "directions", "out", "strength", "sparsity", "rounds", "K", "mode",
                 "restore_norms", "ablate_conv"):
        assert flag in dests, f"{flag} is described and not declared"


def test_the_command_bakes_and_exits_zero(tmp_path, model, capsys):
    path = streambake.save_directions(
        tmp_path / "d.safetensors", _directions(), model=str(model), mode="per_layer")
    assert streambake.main(["--model", str(model), "--directions", str(path),
                            "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "config.json").is_file()


def test_the_command_refuses_an_output_that_could_reach_the_model(tmp_path, model):
    path = streambake.save_directions(
        tmp_path / "d.safetensors", _directions(), model=str(model), mode="per_layer")
    with pytest.raises(SystemExit, match="model directory itself"):
        streambake.main(["--model", str(model), "--directions", str(path), "--out", str(model)])


def test_a_model_path_that_differs_from_the_recorded_one_is_a_note_and_not_a_refusal(
        tmp_path, model, capsys):
    """The two steps run on different machines by design, so the paths legitimately differ. The
    geometry check in the pre-flight is the one that can actually prove a mismatch.
    """
    path = streambake.save_directions(
        tmp_path / "d.safetensors", _directions(), model="somewhere/else", mode="per_layer")
    assert streambake.main(["--model", str(model), "--directions", str(path),
                            "--out", str(tmp_path / "out")]) == 0
    assert "different machines" in capsys.readouterr().out


def test_the_mode_comes_from_the_file_when_the_flag_is_absent(tmp_path, model):
    from senbonzakura.cli import orthogonalize_np_

    dirs = _directions()
    name = "model.layers.2.self_attn.o_proj.weight"
    expected = _read(model / "model-0.safetensors", name)
    orthogonalize_np_(expected, torch.from_numpy(dirs[1]), 1.0, 0.0, 0)

    path = streambake.save_directions(tmp_path / "d.safetensors", dirs, model=str(model),
                                      mode="single")
    streambake.main(["--model", str(model), "--directions", str(path),
                     "--out", str(tmp_path / "out")])
    assert torch.equal(_read(tmp_path / "out" / "model-0.safetensors", name), expected)


def test_an_unreadable_directions_file_exits_rather_than_tracebacks(tmp_path, model):
    bad = tmp_path / "bad.safetensors"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="stream-bake:"):
        streambake.main(["--model", str(model), "--directions", str(bad),
                         "--out", str(tmp_path / "out")])


# ── a prediction head inside the stack, which the reader used to refuse outright ─────

def _checkpoint_with_head(root, *, layers=LAYERS, hidden=HIDDEN):
    """A DeepSeek V3 shaped stack: `layers` decoder blocks plus one prediction-head block.

    The head is marked the way real ones are, by `eh_proj` beside `enorm` and `hnorm`, which is
    what `modelmap.prediction_head` reads. It also carries an `o_proj` and a `down_proj`, because
    a head that held no residual writer would make the `edit` policy untestable.
    """
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    (root / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"],
        "num_hidden_layers": layers, "hidden_size": hidden,
        "num_nextn_predict_layers": 1}), encoding="utf-8")

    arrays, index = {}, {}
    for i in range(layers):
        for suffix in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
            name = f"model.layers.{i}.{suffix}"
            arrays[name] = rng.standard_normal((hidden, hidden))
    # The head, at the index one past the declared count.
    for suffix in ("eh_proj.weight", "enorm.weight", "hnorm.weight",
                   "self_attn.o_proj.weight", "mlp.down_proj.weight"):
        name = f"model.layers.{layers}.{suffix}"
        arrays[name] = rng.standard_normal(
            (hidden, hidden) if "proj" in suffix else (hidden,))
    arrays["model.norm.weight"] = rng.standard_normal((hidden,))
    for name in arrays:
        index[name] = "model-0.safetensors"
    _write_shard(root / "model-0.safetensors", arrays)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index}), encoding="utf-8")
    return root


@pytest.fixture
def model_with_head(tmp_path):
    return _checkpoint_with_head(tmp_path / "source-head")


def test_a_stack_holding_a_prediction_head_is_indexed_rather_than_refused(model_with_head):
    """WHAT THIS REPLACES. The strict axis search wants indices of exactly 0..count-1 and a head
    puts one more there, so the reader refused and blamed "a naming layout this reader does not
    know". The layout is ordinary, the checkpoint is editable, and fifteen models in the
    architecture corpus are in this position.
    """
    index = streaming.index_layers(model_with_head)
    assert index.count == LAYERS
    assert index.head_indices == (LAYERS,)
    assert index.head, "the head's tensors were not kept"


def test_the_head_is_kept_apart_from_the_layers_and_from_shared(model_with_head):
    """Apart from the layers because the declared count is right to exclude it; apart from shared
    because whether to edit it is a flag, and a thing filed under "everything else" cannot be one.
    """
    index = streaming.index_layers(model_with_head)
    assert len(index.layers) == LAYERS
    for layer in index.layers:
        assert not any(f".{LAYERS}." in n for n in layer)
    assert not any(f".{LAYERS}." in n for n in index.shared)
    assert all(f".{LAYERS}." in n for n in index.head)


def test_the_repr_says_a_head_is_there(model_with_head):
    assert "prediction head" in repr(streaming.index_layers(model_with_head))


def test_skipping_the_head_is_the_default_and_leaves_its_weights_alone(
        model_with_head, tmp_path):
    """The recorded position: the config's declared layer count excludes the head, so excluding it
    from the edit is what that declaration means.
    """
    out = streambake.prepare_output(model_with_head, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, _directions(), log=lambda _m: None)
    for suffix in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
        name = f"model.layers.{LAYERS}.{suffix}"
        assert torch.equal(_read(model_with_head / "model-0.safetensors", name),
                           _read(out / "model-0.safetensors", name)), name


def test_editing_the_head_reaches_its_residual_writers(model_with_head, tmp_path):
    out = streambake.prepare_output(model_with_head, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, _directions(), prediction_head="edit", log=lambda _m: None)
    for suffix in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
        name = f"model.layers.{LAYERS}.{suffix}"
        assert not torch.equal(_read(model_with_head / "model-0.safetensors", name),
                               _read(out / "model-0.safetensors", name)), name


def test_editing_the_head_does_not_touch_its_norms_or_its_projection_marker(
        model_with_head, tmp_path):
    """`eh_proj`, `enorm` and `hnorm` are what mark the block as a head. None of them is a
    residual writer, and an edit that reached them would be a different intervention.
    """
    out = streambake.prepare_output(model_with_head, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, _directions(), prediction_head="edit", log=lambda _m: None)
    for suffix in ("enorm.weight", "hnorm.weight"):
        name = f"model.layers.{LAYERS}.{suffix}"
        assert torch.equal(_read(model_with_head / "model-0.safetensors", name),
                           _read(out / "model-0.safetensors", name)), name


def test_the_two_policies_are_different_edits_and_cannot_resume_each_other(
        model_with_head, tmp_path):
    """One checkpoint holding both answers to the question the flag exists to ask."""
    out = streambake.prepare_output(model_with_head, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, _directions(), prediction_head="skip", log=lambda _m: None)
    with pytest.raises(streambake.StreamBakeError, match="part-finished"):
        streambake.bake(out, _directions(), prediction_head="edit", log=lambda _m: None)


def test_an_unknown_policy_is_refused_by_name(model_with_head):
    index = streaming.index_layers(model_with_head)
    with pytest.raises(streambake.StreamBakeError, match="must be one of"):
        streambake.plan_edits(index, prediction_head="maybe")


def test_the_bake_says_a_head_is_present_and_which_policy_it_used(model_with_head, tmp_path):
    said = []
    out = streambake.prepare_output(model_with_head, tmp_path / "out", log=lambda _m: None)
    streambake.bake(out, _directions(), prediction_head="edit", log=said.append)
    joined = " ".join(said)
    assert "prediction head" in joined
    assert "'edit'" in joined


def test_the_command_exposes_both_policies():
    dests = {a.dest: a for a in streambake.build_parser()._actions}
    assert set(dests["prediction_head"].choices) == set(streambake.PREDICTION_HEAD_POLICIES)
    assert dests["prediction_head"].default == "skip"


def test_an_extra_index_that_is_not_a_head_is_still_refused(tmp_path):
    """THE STRICTNESS THAT HAD TO SURVIVE. An extra index with no head markers means the config
    disagrees with the weights, which has a different fix from a prediction head, and accepting
    any superset would turn that into an off-by-one applied to every block.
    """
    root = tmp_path / "mismatch"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 2, "hidden_size": HIDDEN}), encoding="utf-8")
    arrays = {f"model.layers.{i}.self_attn.o_proj.weight": np.zeros((HIDDEN, HIDDEN))
              for i in range(3)}
    _write_shard(root / "model-0.safetensors", arrays)
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": dict.fromkeys(arrays, "model-0.safetensors")}),
        encoding="utf-8")
    with pytest.raises(streaming.ShardError):
        streaming.index_layers(root)


def test_a_shard_holding_only_the_head_is_still_walked(tmp_path):
    """A head can be the only thing in its shard. A bake that walked the layer-derived shard list
    would never open that file, leave the head untouched under `edit`, and report success.
    """
    root = _checkpoint_with_head(tmp_path / "split")
    index = streaming.index_layers(root)
    assert set(index.shards()) >= {shard for shard, _t in index.head.values()}


# ── parity across the dtypes a real checkpoint actually stores ───────────────────────

TORCH_DTYPE = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def _write_typed_shard(path, arrays, dtype_name):
    """A safetensors file in a named dtype, written through torch so BF16 is reachable.

    numpy has no bfloat16, and bf16 is what most published checkpoints are stored in, so a parity
    test that only covered float32 would be testing the one dtype the field does not use.
    """
    header, offset, blobs = {}, 0, []
    for name, arr in arrays.items():
        t = torch.as_tensor(arr, dtype=torch.float32).to(TORCH_DTYPE[dtype_name]).contiguous()
        raw = t.view(torch.uint8).reshape(-1).numpy().tobytes()
        header[name] = {"dtype": dtype_name, "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for raw in blobs:
            f.write(raw)


def _typed_checkpoint(root, dtype_name, *, layers=2, hidden=HIDDEN):
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    (root / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": layers, "hidden_size": hidden}), encoding="utf-8")
    arrays, index = {}, {}
    for i in range(layers):
        name = f"model.layers.{i}.self_attn.o_proj.weight"
        arrays[name] = rng.standard_normal((hidden, hidden))
        index[name] = "model-0.safetensors"
    _write_typed_shard(root / "model-0.safetensors", arrays, dtype_name)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index}), encoding="utf-8")
    return root


@pytest.mark.parametrize("dtype_name", ["F32", "F16", "BF16"])
def test_the_streamed_edit_matches_the_resident_one_in_every_stored_dtype(
        dtype_name, tmp_path):
    """BF16 is the one that matters: it is what most published checkpoints store, and a parity
    claim demonstrated only in float32 would be about the one dtype the field does not use.

    The projection upcasts to float32 and casts back, on both paths, so the result should be
    identical rather than merely close. Asserted as identical.
    """
    from senbonzakura.cli import orthogonalize_np_

    model = _typed_checkpoint(tmp_path / f"src-{dtype_name}", dtype_name, layers=2)
    dirs = _directions(layers=2)
    name = "model.layers.1.self_attn.o_proj.weight"

    expected = _read(model / "model-0.safetensors", name)
    assert expected.dtype == TORCH_DTYPE[dtype_name]
    orthogonalize_np_(expected, torch.from_numpy(dirs[2]), 1.0, 0.0, 0)

    out = streambake.prepare_output(model, tmp_path / f"out-{dtype_name}",
                                   log=lambda _m: None)
    streambake.bake(out, dirs, log=lambda _m: None)
    got = _read(out / "model-0.safetensors", name)
    assert got.dtype == TORCH_DTYPE[dtype_name], "the edit changed the stored dtype"
    assert torch.equal(got, expected), (
        f"{dtype_name}: the streamed edit and the resident one disagree, so the parity claim "
        f"holds in float32 only")


@pytest.mark.parametrize("dtype_name", ["F16", "BF16"])
def test_the_parity_check_can_be_made_to_fail(dtype_name, tmp_path):
    """THE FORCED-FAIL CONTROL the rung's exit gate asks every parity instrument to carry.

    An equality assertion that has only ever been seen passing has not been shown to detect
    anything. This perturbs one path by a single direction's worth of strength and confirms the
    comparison notices, which is what makes the passing case above evidence rather than decoration.
    """
    from senbonzakura.cli import orthogonalize_np_

    model = _typed_checkpoint(tmp_path / f"src-{dtype_name}", dtype_name, layers=2)
    dirs = _directions(layers=2)
    name = "model.layers.1.self_attn.o_proj.weight"

    expected = _read(model / "model-0.safetensors", name)
    orthogonalize_np_(expected, torch.from_numpy(dirs[2]), 1.0, 0.0, 0)

    out = streambake.prepare_output(model, tmp_path / f"out-{dtype_name}", log=lambda _m: None)
    # The SAME bake at a different strength. Nothing else changes.
    streambake.bake(out, dirs, strength=0.95, log=lambda _m: None)
    got = _read(out / "model-0.safetensors", name)
    assert not torch.equal(got, expected), (
        "a 5% strength difference went undetected, so this comparison would pass a streamed path "
        "that applied the wrong edit")


def test_an_unsupported_dtype_is_refused_rather_than_reinterpreted(tmp_path):
    """A dtype the projection has no meaning for. Reinterpreting the bytes as something it can
    multiply would produce a checkpoint that loads and is noise.
    """
    model = _typed_checkpoint(tmp_path / "src", "F32", layers=2)
    # An integer tensor in a residual-writer position, which no real checkpoint has and which is
    # the cheapest way to reach the refusal.
    tensors = list(streaming.tensors(str(model / "model-0.safetensors")))
    assert tensors, "the fixture produced no tensors"
    # 16 elements of F32 is 64 bytes, so the reshape succeeds and the REFUSAL is what is reached.
    # The first version passed 8 bytes, which failed on the reshape instead and tested nothing.
    with pytest.raises(streambake.StreamBakeError, match="neither a 2-D residual writer"):
        streambake.apply_direction(
            b"\0" * 64, streaming.Tensor("x", "F32", (2, 2, 2, 2), 0, 64),
            torch.from_numpy(_directions()[1]), strength=1.0, sparsity=0.0, rounds=0,
            restore_norms=True)


# ── peak memory, asserted rather than observed once ─────────────────────────────────

def _wide_checkpoint(root, *, layers, hidden):
    """A checkpoint whose shard grows with `layers` while each tensor stays the same size."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    (root / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": layers, "hidden_size": hidden}), encoding="utf-8")
    arrays, index = {}, {}
    for i in range(layers):
        name = f"model.layers.{i}.self_attn.o_proj.weight"
        arrays[name] = rng.standard_normal((hidden, hidden))
        index[name] = "model-0.safetensors"
    _write_shard(root / "model-0.safetensors", arrays)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index}), encoding="utf-8")
    return root


def _peak_bytes_of_a_bake(model, out, dirs):
    import tracemalloc

    prepared = streambake.prepare_output(model, out, log=lambda _m: None)
    tracemalloc.start()
    try:
        streambake.bake(prepared, dirs, log=lambda _m: None)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_peak_memory_does_not_grow_when_the_shard_does(tmp_path):
    """THE CLAIM THE WHOLE MODULE IS FOR, asserted rather than observed once.

    Stated as a comparison rather than an absolute, deliberately. An absolute bound would have to
    account for `streaming.CHUNK`, the 4 MB buffer the pass-through copy uses, which is a fixed
    cost and not part of the claim. What the claim actually says is that cost scales with the
    largest TENSOR and not with the shard, so the test quadruples the shard while holding the
    tensor size fixed and asserts the peak barely moves.

    If the bake ever starts holding a layer, or a shard, or the model, this is what notices.
    """
    hidden = 256                      # 256 x 256 float32 is 256 KB per tensor
    small = _wide_checkpoint(tmp_path / "small", layers=4, hidden=hidden)
    large = _wide_checkpoint(tmp_path / "large", layers=16, hidden=hidden)
    dirs_small, dirs_large = _directions(layers=4, hidden=hidden), _directions(
        layers=16, hidden=hidden)

    small_shard = (small / "model-0.safetensors").stat().st_size
    large_shard = (large / "model-0.safetensors").stat().st_size
    assert large_shard > small_shard * 3.5, "the fixture did not actually grow the shard"

    peak_small = _peak_bytes_of_a_bake(small, tmp_path / "out-small", dirs_small)
    peak_large = _peak_bytes_of_a_bake(large, tmp_path / "out-large", dirs_large)

    assert peak_large < peak_small * 2, (
        f"peak went from {peak_small / 1e6:.2f} MB to {peak_large / 1e6:.2f} MB when the shard "
        f"went from {small_shard / 1e6:.2f} MB to {large_shard / 1e6:.2f} MB, so it is tracking "
        f"the shard rather than one tensor")


def test_peak_memory_is_a_small_multiple_of_one_tensor(tmp_path):
    """And the absolute form, with the copy buffer accounted for rather than ignored.

    A tensor's bytes are held three times at the peak by design: the raw read, the torch view the
    projection writes into, and the bytes handed back to the writer. Plus `streaming.CHUNK` for
    the pass-through copy. The bound is that sum with headroom, and it is asserted so that a
    fourth copy appearing is a failure rather than a slow drift.
    """
    hidden = 512                      # 512 x 512 float32 is 1 MB per tensor
    model = _wide_checkpoint(tmp_path / "wide", layers=12, hidden=hidden)
    one_tensor = hidden * hidden * 4
    shard = (model / "model-0.safetensors").stat().st_size

    peak = _peak_bytes_of_a_bake(model, tmp_path / "out",
                                 _directions(layers=12, hidden=hidden))
    budget = one_tensor * 4 + streaming.CHUNK
    assert peak < budget, (
        f"peak {peak / 1e6:.2f} MB is over the {budget / 1e6:.2f} MB budget: three copies of a "
        f"{one_tensor / 1e6:.2f} MB tensor plus the {streaming.CHUNK / 1e6:.2f} MB copy buffer, "
        f"with one copy of headroom")
    assert peak < shard / 2, (
        f"peak {peak / 1e6:.2f} MB is more than half the {shard / 1e6:.2f} MB shard, which is the "
        f"shape of a reader that is holding the whole file")
