# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Round trip: a recipe senbonzakura writes, applied by the reference applier, reproduces the bake.

THE CLAIM UNDER TEST is the one the recipe standard rests on. Given the same base checkpoint and
the same directions, the reference applier (`recipes/apply_recipe.py`, Apache-2.0, written from
the specification) must produce the same weights as senbonzakura's own bake. If it does not, the
specification or the applier is wrong, and that is the finding.

Tolerance, stated before the run: for float32 checkpoints the two outputs agree to within
`FLOAT32_TOL` in absolute value, which is float32 rounding in a different summation order. For
bfloat16 checkpoints they agree to within one bfloat16 unit in the last place on every element,
because both sides round the same float32 value.
"""

import hashlib
import importlib.util
import json
import pathlib
import struct

import numpy as np
import pytest

from senbonzakura import recipe as recipe_mod
from senbonzakura import streambake

APPLIER_PATH = pathlib.Path(__file__).resolve().parents[1] / "recipes" / "apply_recipe.py"

#: Absolute tolerance for float32 outputs, see the module docstring.
FLOAT32_TOL = 1e-5

HIDDEN, INTER, VOCAB, LAYERS = 8, 12, 16, 2


def load_applier():
    spec = importlib.util.spec_from_file_location("apply_recipe", APPLIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode_st(path, tensors):
    """Write a safetensors file from `{name: (dtype, float32 array)}`, dtype F32 or BF16."""
    applier = load_applier()
    header, blobs, offset = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = applier.encode(arr, dtype)
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    pathlib.Path(path).write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))


def make_checkpoint(root, *, seed=0, dtype="F32"):
    """A two-layer Llama-shaped checkpoint, small enough to read in one glance.

    Returns the float32 arrays that were written, keyed by tensor name, so a test can recompute
    the edit independently of both the applier and the bake.
    """
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    kv = 4
    arrays = {"model.embed_tokens.weight": rng.standard_normal((VOCAB, HIDDEN))}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        arrays[p + "input_layernorm.weight"] = 1 + 0.1 * rng.standard_normal(HIDDEN)
        arrays[p + "self_attn.q_proj.weight"] = rng.standard_normal((HIDDEN, HIDDEN))
        arrays[p + "self_attn.k_proj.weight"] = rng.standard_normal((kv, HIDDEN))
        arrays[p + "self_attn.v_proj.weight"] = rng.standard_normal((kv, HIDDEN))
        arrays[p + "self_attn.o_proj.weight"] = rng.standard_normal((HIDDEN, HIDDEN))
        arrays[p + "post_attention_layernorm.weight"] = 1 + 0.1 * rng.standard_normal(HIDDEN)
        arrays[p + "mlp.gate_proj.weight"] = rng.standard_normal((INTER, HIDDEN))
        arrays[p + "mlp.up_proj.weight"] = rng.standard_normal((INTER, HIDDEN))
        arrays[p + "mlp.down_proj.weight"] = rng.standard_normal((HIDDEN, INTER))
    arrays["model.norm.weight"] = np.ones(HIDDEN)
    arrays["lm_head.weight"] = rng.standard_normal((VOCAB, HIDDEN))
    arrays = {k: v.astype(np.float32) for k, v in arrays.items()}

    _encode_st(root / "model.safetensors", {name: (dtype, arr) for name, arr in arrays.items()})
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": HIDDEN,
                "num_hidden_layers": LAYERS,
                "intermediate_size": INTER,
                "vocab_size": VOCAB,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            }
        ),
        encoding="utf-8",
    )
    return arrays


def make_directions(path, *, seed=1):
    """Orthonormal directions for layers 0 and 1, with one unused slot to exercise padding."""
    rng = np.random.default_rng(seed)
    dirs = np.zeros((LAYERS + 1, 2, HIDDEN), dtype=np.float32)
    q, _ = np.linalg.qr(rng.standard_normal((HIDDEN, 2)))
    dirs[1, :2] = q.T
    dirs[2, :1] = q[:, 0]
    streambake.save_directions(path, dirs, model="fixture", mode="per_layer", provenance={"fixture": "roundtrip"})
    return dirs


def _load_all(path):
    applier = load_applier()
    header, data_start, _ = applier.read_safetensors_header(path)
    out = {}
    with pathlib.Path(path).open("rb") as f:
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            raw = applier.read_tensor_bytes(f, data_start, entry)
            out[name] = (entry["dtype"], applier.decode(raw, entry["dtype"], entry["shape"]))
    return out


def bake_reference(model, out, directions_path):
    """Senbonzakura's own edit, into `out`, with the defaults the recipe will record."""
    arr, _meta = streambake.load_directions(directions_path)
    prepared = streambake.prepare_output(model, out, log=lambda *_: None)
    streambake.bake(prepared, arr, strength=1.0, restore_norms=True, log=lambda *_: None)
    return prepared


def write_recipe_for(model, directions_path, path, *, name="roundtrip"):
    recipe = recipe_mod.build(
        name=name,
        model_dir=model,
        model_id="fixture/tiny",
        revision="fixture",
        directions_path=directions_path,
        layer_count=LAYERS,
        strength=1.0,
        restore_norms=True,
        measured=None,
        command="senbonzakura stream-bake --model MODEL --directions DIRS --out OUT",
    )
    return recipe_mod.write(path, recipe)


def test_recipe_from_senbonzakura_applied_by_the_reference_applier_matches_the_bake(tmp_path):
    model = tmp_path / "model"
    make_checkpoint(model)
    dirs = tmp_path / "directions.safetensors"
    make_directions(dirs)
    reference = bake_reference(model, tmp_path / "bake", dirs)
    recipe_path = write_recipe_for(model, dirs, tmp_path / "recipe.json")

    applier = load_applier()
    applier.apply(recipe_path, model, dirs, tmp_path / "applied", log=lambda *_: None)

    baked = _load_all(reference / "model.safetensors")
    applied = _load_all(tmp_path / "applied" / "model.safetensors")
    assert baked.keys() == applied.keys()
    worst = 0.0
    for name in baked:
        assert baked[name][0] == applied[name][0], name
        diff = float(np.abs(baked[name][1] - applied[name][1]).max())
        worst = max(worst, diff)
        assert diff <= FLOAT32_TOL, f"{name} differs by {diff}"
    # The edit must have done something, or the agreement above is between two no-ops.
    untouched = _load_all(model / "model.safetensors")
    moved = max(
        float(np.abs(untouched[n][1] - applied[n][1]).max())
        for n in ("model.layers.0.self_attn.o_proj.weight", "model.layers.1.mlp.down_proj.weight")
    )
    assert moved > 1e-3
    assert worst <= FLOAT32_TOL


def test_bfloat16_outputs_are_bit_identical_to_the_bake(tmp_path):
    model = tmp_path / "model"
    make_checkpoint(model, dtype="BF16")
    dirs = tmp_path / "directions.safetensors"
    make_directions(dirs)
    reference = bake_reference(model, tmp_path / "bake", dirs)
    recipe_path = write_recipe_for(model, dirs, tmp_path / "recipe.json")
    load_applier().apply(recipe_path, model, dirs, tmp_path / "applied", log=lambda *_: None)

    baked = _load_all(reference / "model.safetensors")
    applied = _load_all(tmp_path / "applied" / "model.safetensors")
    for name in baked:
        assert baked[name][0] == applied[name][0], name
        if baked[name][0] != "BF16":
            continue
        # Exact, not within a tolerance. Both sides round the same float32 to bfloat16, and a
        # tolerance of one unit would accept truncation in place of rounding, which is a different
        # edit. Measured on this fixture: zero elements differ.
        assert np.array_equal(baked[name][1], applied[name][1]), name


def test_the_recipe_senbonzakura_writes_records_what_the_bake_did(tmp_path):
    model = tmp_path / "model"
    make_checkpoint(model)
    dirs = tmp_path / "directions.safetensors"
    make_directions(dirs)
    recipe_path = write_recipe_for(model, dirs, tmp_path / "recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["schema"] == "abliteration-recipe/1"
    assert recipe["edit"]["directions"]["directions_found"] == 3
    assert recipe["edit"]["directions"]["directions_found_per_position"] == [0, 2, 1]
    assert recipe["edit"]["directions"]["shape"] == [3, 2, HIDDEN]
    assert recipe["edit"]["layers"] == {"from": 0, "to": LAYERS - 1}
    assert recipe["measured"]["status"] == "not_measured"
    assert recipe["base"]["weights_sha256"] is not None
    expected = hashlib.sha256(dirs.read_bytes()).hexdigest()
    assert recipe["edit"]["directions"]["sha256"] == expected


def test_the_writer_and_the_applier_measure_the_same_residual(tmp_path):
    """Two independent implementations of the residual definition must agree on one run.

    The writer computes before and after from the two checkpoints. The applier computes them while
    it edits and refuses a recipe whose recorded figures disagree. A recipe the writer measures and
    the applier accepts is the cross-check; neither implementation is trusted alone.
    """
    model = tmp_path / "model"
    make_checkpoint(model)
    dirs = tmp_path / "directions.safetensors"
    make_directions(dirs)
    baked = bake_reference(model, tmp_path / "bake", dirs)
    residual = recipe_mod.measure_residual(
        model, baked, dirs, layer_count=LAYERS, targets=[recipe_mod.ATTN_TARGET, recipe_mod.MLP_TARGET]
    )
    recipe = recipe_mod.build(
        name="cross-check",
        model_dir=model,
        model_id="fixture/tiny",
        revision="fixture",
        directions_path=dirs,
        layer_count=LAYERS,
        strength=1.0,
        restore_norms=True,
        measured=None,
        command="cmd",
        residual=residual,
    )
    path = recipe_mod.write(tmp_path / "recipe.json", recipe)
    summary = load_applier().apply(path, model, dirs, tmp_path / "applied", log=lambda *_: None)
    assert summary["residual"]["measured"] is True
    assert summary["residual"]["before"] == pytest.approx(residual["before"], rel=1e-3)
    assert summary["residual"]["after"] == pytest.approx(residual["after"], rel=1e-3, abs=1e-6)
