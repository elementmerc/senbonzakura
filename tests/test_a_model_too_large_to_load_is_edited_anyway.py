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
