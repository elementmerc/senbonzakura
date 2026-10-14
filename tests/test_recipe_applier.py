# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The reference applier refuses every recipe it cannot apply exactly as written.

Each test writes one defective recipe or input, and asserts the applier refuses it with a message
that names the problem. A refusal that goes missing would turn a different edit into a success
report, which is the failure this format exists to prevent.
"""

import hashlib
import json

import numpy as np
import pytest

from tests.test_recipe_roundtrip import (
    HIDDEN,
    LAYERS,
    load_applier,
    make_checkpoint,
    make_directions,
    write_recipe_for,
)

applier = load_applier()


@pytest.fixture
def world(tmp_path):
    model = tmp_path / "model"
    make_checkpoint(model)
    dirs = tmp_path / "directions.safetensors"
    make_directions(dirs)
    recipe_path = write_recipe_for(model, dirs, tmp_path / "recipe.json")
    return {"model": model, "dirs": dirs, "recipe": recipe_path, "tmp": tmp_path}


def _edit_recipe(world, mutate):
    doc = json.loads(world["recipe"].read_text(encoding="utf-8"))
    mutate(doc)
    path = world["tmp"] / "defective.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _refused(world, recipe_path, match):
    with pytest.raises(applier.RecipeError, match=match):
        applier.apply(recipe_path, world["model"], world["dirs"], world["tmp"] / "out", log=lambda *_: None)


def test_bfloat16_rounds_to_nearest_with_ties_to_even_and_does_not_truncate():
    # Exact halfway points between two bfloat16 values, where the rule is unambiguous. Truncation
    # would return the lower value in both cases; round-to-nearest-even returns the even one.
    x = np.array([1.0 + 2.0**-8, 1.0 + 3 * 2.0**-8], dtype=np.float32)
    rounded = np.frombuffer(applier.encode(x, "BF16"), dtype="<u2").astype(np.uint32) << 16
    values = rounded.view(np.float32)
    assert values[0] == np.float32(1.0)
    assert values[1] == np.float32(1.0 + 2.0**-6)


def test_a_clean_recipe_is_accepted_in_check_mode(world):
    summary = applier.apply(world["recipe"], world["model"], world["dirs"], None, check_only=True, log=lambda *_: None)
    assert summary["tensors_to_edit"] == 2 * LAYERS
    assert summary["status"] == "checked"


@pytest.mark.parametrize("schema", ["abliteration-recipe/2", "some-other-format/1", 7])
def test_an_unknown_schema_major_or_name_is_refused(world, schema):
    path = _edit_recipe(world, lambda d: d.__setitem__("schema", schema))
    with pytest.raises(applier.RecipeError, match="schema"):
        applier.load_recipe(path)


def test_an_unknown_key_inside_edit_is_refused_not_ignored(world):
    # A sparsity knob the applier does not implement would otherwise be silently dropped, and the
    # output would be a dense edit reported as a sparse one.
    path = _edit_recipe(world, lambda d: d["edit"].__setitem__("sparsity", 0.3))
    _refused(world, path, "sparsity")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("projection_side", "input"),
        ("sign_convention", "reflection"),
        ("tensor_naming", "gguf"),
        ("method", "lora"),
        ("dtype", "float16"),
    ],
)
def test_a_convention_the_applier_does_not_implement_is_refused(world, key, value):
    path = _edit_recipe(world, lambda d: d["edit"].__setitem__(key, value))
    _refused(world, path, key)


def test_a_negative_strength_is_refused(world):
    path = _edit_recipe(world, lambda d: d["edit"].__setitem__("strength", -1.0))
    _refused(world, path, "strength")


def test_restore_norms_must_be_a_boolean(world):
    path = _edit_recipe(world, lambda d: d["edit"].__setitem__("restore_norms", "false"))
    _refused(world, path, "restore_norms")


def test_a_target_without_exactly_one_layer_wildcard_is_refused(world):
    path = _edit_recipe(world, lambda d: d["edit"].__setitem__("targets", ["lm_head.weight"]))
    _refused(world, path, "exactly one")


def test_a_target_that_matches_nothing_is_refused_rather_than_reported_as_success(world):
    # The pattern is well formed and names real tensors, but none of them sits in the layers the
    # recipe asks for. Reporting success here would be an unedited model with a clean report.
    def mutate(d):
        d["edit"]["targets"] = ["model.layers.*.mlp.down_proj.weight"]
        d["edit"]["layers"] = {"from": 0, "to": 0}
        d["edit"]["targets"] = ["model.layers.*.mlp.gate_proj.bias"]

    _refused(world, _edit_recipe(world, mutate), "matches no tensor")


def test_a_pattern_naming_no_real_tensor_is_refused(world):
    path = _edit_recipe(world, lambda d: d["edit"].__setitem__("targets", ["model.layers.*.mlp.gate_proj_typo.weight"]))
    _refused(world, path, "matches no tensor")


def test_a_layer_range_past_the_direction_set_is_refused(world):
    path = _edit_recipe(world, lambda d: d["edit"].__setitem__("layers", {"from": 0, "to": 9}))
    _refused(world, path, "positions")


def test_directions_with_a_changed_hash_are_refused(world):
    path = _edit_recipe(world, lambda d: d["edit"]["directions"].__setitem__("sha256", "0" * 64))
    _refused(world, path, "hashes to")


def test_a_direction_count_that_disagrees_with_the_file_is_refused(world):
    path = _edit_recipe(world, lambda d: d["edit"]["directions"].__setitem__("directions_found", 99))
    _refused(world, path, "directions_found|non-empty rows")


def test_non_orthonormal_directions_are_refused(world):
    # Scale one row so the basis is no longer orthonormal, then re-hash so only the basis fails.
    arr = np.zeros((LAYERS + 1, 2, HIDDEN), np.float32)
    rng = np.random.default_rng(3)
    arr[1, :2] = rng.standard_normal((2, HIDDEN)).astype(np.float32)
    arr[2, :1] = arr[1, :1]
    from senbonzakura import streambake

    streambake.save_directions(world["dirs"], arr, model="x", mode="per_layer")
    doc = json.loads(world["recipe"].read_text(encoding="utf-8"))
    doc["edit"]["directions"]["sha256"] = hashlib.sha256(world["dirs"].read_bytes()).hexdigest()
    doc["edit"]["directions"]["directions_found"] = 3
    doc["edit"]["directions"]["directions_found_per_position"] = [0, 2, 1]
    doc["edit"]["directions"]["shape"] = [LAYERS + 1, 2, HIDDEN]
    path = world["tmp"] / "bad-basis.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    _refused(world, path, "orthonormal")


def test_a_base_hash_that_does_not_match_the_checkpoint_is_refused(world):
    path = _edit_recipe(world, lambda d: d["base"].__setitem__("weights_sha256", "f" * 64))
    _refused(world, path, "hashes to")


def test_a_null_base_hash_without_a_reason_is_refused(world):
    def mutate(d):
        d["base"]["weights_sha256"] = None
        d["base"].pop("weights_sha256_reason", None)

    _refused(world, _edit_recipe(world, mutate), "weights_sha256_reason")


@pytest.mark.parametrize(
    "measured",
    [
        {"status": "maybe"},
        {"status": "not_measured"},
        {
            "status": "measured",
            "refusal_before": {"point": 0.4, "ci": [0.3, 0.5]},
            "refusal_after": {"point": 0.0, "n": 10, "ci": [0.0, 0.1]},
            "kl": {"point": 0.01, "n": 10, "ci": [0.0, 0.02]},
        },
    ],
)
def test_a_measured_block_must_say_what_was_measured_and_with_how_many(world, measured):
    path = _edit_recipe(world, lambda d: d.__setitem__("measured", measured))
    with pytest.raises(applier.RecipeError):
        applier.load_recipe(path)


def test_the_output_may_not_overwrite_the_model(world):
    with pytest.raises(applier.RecipeError, match="overlaps"):
        applier.apply(world["recipe"], world["model"], world["dirs"], world["model"], log=lambda *_: None)


def test_an_output_directory_that_already_holds_files_is_refused(world):
    out = world["tmp"] / "busy"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(applier.RecipeError, match="already holds files"):
        applier.apply(world["recipe"], world["model"], world["dirs"], out, log=lambda *_: None)


def test_a_tensor_whose_offsets_leave_a_gap_is_refused(world, tmp_path):
    raw = (world["model"] / "model.safetensors").read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + n])
    first = next(k for k in header if k != "__metadata__")
    header[first]["data_offsets"][1] -= 4
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    bad = tmp_path / "gap.safetensors"
    bad.write_bytes(len(blob).to_bytes(8, "little") + blob + raw[8 + n :])
    with pytest.raises(applier.RecipeError):
        applier.read_safetensors_header(bad)


def test_a_non_finite_source_weight_is_refused_during_the_edit(world, tmp_path):
    from tests.test_recipe_roundtrip import _encode_st

    arrays = {}
    header, data_start, _ = applier.read_safetensors_header(world["model"] / "model.safetensors")
    with (world["model"] / "model.safetensors").open("rb") as f:
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            raw = applier.read_tensor_bytes(f, data_start, entry)
            arrays[name] = ("F32", applier.decode(raw, "F32", entry["shape"]))
    arrays["model.layers.0.mlp.down_proj.weight"][1][0, 0] = np.nan
    model = tmp_path / "nan-model"
    model.mkdir()
    _encode_st(model / "model.safetensors", arrays)
    (model / "config.json").write_text((world["model"] / "config.json").read_text())
    copy_recipe = world["recipe"]
    doc = json.loads(copy_recipe.read_text(encoding="utf-8"))
    doc["base"]["weights_sha256"] = None
    doc["base"]["weights_sha256_reason"] = "test fixture with a planted NaN"
    path = tmp_path / "nan-recipe.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(applier.RecipeError, match="NaN"):
        applier.apply(path, model, world["dirs"], tmp_path / "out", log=lambda *_: None)


def test_the_check_does_not_write_anything(world):
    out = world["tmp"] / "never"
    applier.apply(world["recipe"], world["model"], world["dirs"], out, check_only=True, log=lambda *_: None)
    assert not out.exists()


def test_the_default_ignores_unknown_top_level_keys(world):
    path = _edit_recipe(world, lambda d: d.__setitem__("x_lab_notes", {"anything": 1}))
    applier.load_recipe(path)


def test_partial_output_is_never_left_behind_on_success(world):
    applier.apply(world["recipe"], world["model"], world["dirs"], world["tmp"] / "clean", log=lambda *_: None)
    assert not list((world["tmp"] / "clean").glob("*.part"))
    assert (world["tmp"] / "clean" / "model.safetensors").is_file()
    assert copy_equal(world["model"] / "config.json", world["tmp"] / "clean" / "config.json")


def copy_equal(a, b):
    return a.read_bytes() == b.read_bytes()


def test_the_edit_leaves_unnamed_tensors_bit_identical(world):
    applier.apply(world["recipe"], world["model"], world["dirs"], world["tmp"] / "same", log=lambda *_: None)
    before = applier.read_safetensors_header(world["model"] / "model.safetensors")
    after = applier.read_safetensors_header(world["tmp"] / "same" / "model.safetensors")
    assert before[0] == after[0]
    src = (world["model"] / "model.safetensors").read_bytes()
    dst = (world["tmp"] / "same" / "model.safetensors").read_bytes()
    assert src[: before[1]] == dst[: after[1]]
    untouched = "model.layers.0.mlp.up_proj.weight"
    s0 = before[0][untouched]["data_offsets"]
    assert src[before[1] + s0[0] : before[1] + s0[1]] == dst[after[1] + s0[0] : after[1] + s0[1]]
    assert copy_equal(world["model"] / "config.json", world["tmp"] / "same" / "config.json")
