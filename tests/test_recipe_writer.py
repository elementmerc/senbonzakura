# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The recipe writer records what the bake did, and refuses to write a recipe it cannot stand behind."""

import json

import numpy as np
import pytest

from senbonzakura import recipe as recipe_mod
from tests.test_recipe_roundtrip import HIDDEN, LAYERS, make_checkpoint, make_directions


def _build(model, dirs, **overrides):
    kwargs = {
        "name": "t",
        "model_dir": model,
        "model_id": "fixture/tiny",
        "revision": "r",
        "directions_path": dirs,
        "layer_count": LAYERS,
        "strength": 1.0,
        "restore_norms": True,
        "measured": None,
        "command": "cmd",
    }
    kwargs.update(overrides)
    return recipe_mod.build(**kwargs)


def test_no_measurement_is_recorded_as_not_measured_with_a_reason(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    doc = _build(model, dirs)
    assert doc["measured"]["status"] == "not_measured"
    assert doc["measured"]["why"]


def test_a_supplied_measurement_is_carried_through_unchanged(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    block = {
        "status": "measured",
        "refusal_before": {"point": 0.4, "n": 100, "ci": [0.3, 0.5]},
        "refusal_after": {"point": 0.0, "n": 100, "ci": [0.0, 0.04]},
        "kl": {"point": 0.02, "n": 64, "ci": [0.01, 0.03]},
    }
    assert _build(model, dirs, measured=block)["measured"] == block


def test_a_several_shard_checkpoint_without_an_index_gets_a_null_hash_and_a_reason(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    (model / "model-2.safetensors").write_bytes((model / "model.safetensors").read_bytes())
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    doc = _build(model, dirs)
    assert doc["base"]["weights_sha256"] is None
    assert "shards" in doc["base"]["weights_sha256_reason"]


def test_an_index_file_is_the_hash_when_present(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {}}), encoding="utf-8")
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    doc = _build(model, dirs)
    assert doc["base"]["weights_sha256"] is not None
    assert "weights_sha256_reason" not in doc["base"]


def test_a_direction_set_too_short_for_the_layers_is_refused(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    with pytest.raises(recipe_mod.RecipeError, match="positions"):
        _build(model, dirs, layer_count=LAYERS + 5)


def test_a_directions_file_that_disagrees_with_its_own_count_is_refused(tmp_path):
    from senbonzakura import streambake

    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    arr = np.zeros((LAYERS + 1, 2, HIDDEN), np.float32)
    arr[1, 0, 0] = 1.0
    streambake.save_directions(dirs, arr, model="x", mode="per_layer", provenance={"directions_found": 7})
    with pytest.raises(recipe_mod.RecipeError, match="records 7"):
        _build(model, dirs)


def test_an_attention_only_or_mlp_only_edit_names_only_its_targets(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    assert _build(model, dirs, edit_tensors=("mlp",))["edit"]["targets"] == [recipe_mod.MLP_TARGET]
    assert _build(model, dirs, edit_tensors=("attn",))["edit"]["targets"] == [recipe_mod.ATTN_TARGET]


def test_write_is_atomic_and_leaves_no_part_file(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    target = tmp_path / "recipe.json"
    recipe_mod.write(target, _build(model, dirs))
    assert json.loads(target.read_text(encoding="utf-8"))["schema"] == "abliteration-recipe/1"
    assert not list(tmp_path.glob("*.part"))


def test_a_bake_with_sparsity_is_refused_rather_than_recorded_as_dense(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    with pytest.raises(recipe_mod.RecipeError, match="sparsity"):
        _build(model, dirs, sparsity=0.3)


def test_a_bake_with_rounds_is_refused_rather_than_recorded_as_single_pass(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    with pytest.raises(recipe_mod.RecipeError, match="rounds"):
        _build(model, dirs, rounds=4)


def test_a_recipe_without_a_residual_says_it_was_not_measured(tmp_path):
    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    doc = _build(model, dirs)
    assert doc["profile"] == "core/1"
    assert doc["extensions"] == []
    assert doc["edit"]["direction_residual"]["status"] == "not_measured"
    assert doc["edit"]["direction_residual"]["why"]


def test_the_residual_measurement_records_the_definition_it_used(tmp_path):
    from tests.test_recipe_roundtrip import bake_reference

    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    baked = bake_reference(model, tmp_path / "baked", dirs)
    residual = recipe_mod.measure_residual(
        model, baked, dirs, layer_count=LAYERS, targets=[recipe_mod.ATTN_TARGET, recipe_mod.MLP_TARGET]
    )
    assert residual["status"] == "measured"
    assert residual["metric"] == recipe_mod.RESIDUAL_METRIC
    assert residual["definition"] == recipe_mod.RESIDUAL_DEFINITION
    assert residual["rounds"] == 0
    # The edit must remove most of the direction, and the figures must be internally consistent.
    assert 0 <= residual["after"] < residual["before"]
    assert residual["fraction_remaining"] == pytest.approx(residual["after"] / residual["before"])


def test_the_residual_measurement_refuses_a_checkpoint_with_nothing_to_measure(tmp_path):
    from tests.test_recipe_roundtrip import bake_reference

    model = tmp_path / "m"
    make_checkpoint(model)
    dirs = tmp_path / "d.safetensors"
    make_directions(dirs)
    baked = bake_reference(model, tmp_path / "baked", dirs)
    with pytest.raises(recipe_mod.RecipeError, match="residual stream"):
        recipe_mod.measure_residual(
            model, baked, dirs, layer_count=LAYERS, targets=["model.layers.*.mlp.gate_proj.weight"]
        )
