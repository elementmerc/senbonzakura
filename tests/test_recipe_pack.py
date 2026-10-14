# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A pack is written once, holds exactly what its manifest lists, and is checked before anyone applies it.

Each refusal below is fed the defect it names and must give the code it names. The planted-defect runs recorded
in the progress ledger show that each check fails when its rule is removed.
"""

import json
import os

import pytest

from senbonzakura import recipe as recipe_mod
from senbonzakura import recipe_pack
from tests.test_recipe_roundtrip import LAYERS, load_applier, make_checkpoint, make_directions

PACK_FILES = {"recipe.json", "directions.safetensors", "README.md", "MANIFEST.json"}


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "model"
    make_checkpoint(model)
    dirs = tmp_path / "directions.safetensors"
    make_directions(dirs)
    doc = recipe_mod.build(
        name="roundtrip-tiny",
        model_dir=model,
        model_id="tiny",
        revision=None,
        revision_reason="a local checkpoint has no hub revision, so weights_sha256 identifies it",
        directions_path=dirs,
        layer_count=LAYERS,
        strength=1.0,
        restore_norms=True,
        measured=None,
        command="senbonzakura stream-bake --model model --directions directions.safetensors --out out",
        provenance=recipe_mod.provenance_block({"commit": "abc1234", "dirty": False, "source": "git"}),
    )
    recipe_path = tmp_path / "in" / "recipe.json"
    recipe_path.parent.mkdir()
    recipe_mod.write(recipe_path, doc)
    return {"model": model, "dirs": dirs, "doc": doc, "recipe": recipe_path, "tmp": tmp_path}


def _measured_for(edit, **overrides):
    block = {
        "status": "measured",
        "instrument": {
            "scorer": "scorer/1",
            "corpus": {"id": "held-out", "revision": "rev-1"},
            "partition": "test",
            "seed": 7,
        },
        "refusal_before": {"point": 0.391, "n": 305, "ci": [0.337, 0.447]},
        "refusal_after": {"point": 0.0, "n": 305, "ci": [0.0, 0.012]},
        "kl": {"point": 0.016, "n": 256, "ci": [0.012, 0.021]},
        "capability": [
            {
                "name": "GSM8K",
                "before": {"point": 0.42, "n": 150, "ci": [0.34, 0.5]},
                "after": {"point": 0.41, "n": 150, "ci": [0.33, 0.49]},
            }
        ],
        "self_invalidated": None,
        "ci_level": 0.95,
        "limits": ["Measured on one held-out partition, with one judge."],
        "edit_sha256": recipe_mod.edit_sha256(edit),
    }
    block.update(overrides)
    return block


def _recipe_with(inputs, mutate, name="recipe-mutated.json"):
    doc = json.loads(inputs["recipe"].read_text(encoding="utf-8"))
    mutate(doc)
    path = inputs["tmp"] / name
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _code_of(callable_, *args, **kwargs):
    with pytest.raises(recipe_mod.RecipeError) as refused:
        callable_(*args, **kwargs)
    return refused.value.code


def test_a_pack_holds_exactly_the_four_files_and_verifies(inputs):
    out = recipe_pack.build_pack(inputs["recipe"], inputs["dirs"], inputs["tmp"] / "pack", log=lambda *_: None)
    assert {p.name for p in out.iterdir()} == PACK_FILES
    summary = recipe_pack.verify_pack(out)
    assert summary["status"] == "verified"
    assert summary["measured"] == "not_measured"
    assert summary["base_sha256"] == inputs["doc"]["base"]["weights_sha256"]


def test_two_packs_from_the_same_inputs_are_byte_identical(inputs):
    first = recipe_pack.build_pack(inputs["recipe"], inputs["dirs"], inputs["tmp"] / "one", log=lambda *_: None)
    second = recipe_pack.build_pack(inputs["recipe"], inputs["dirs"], inputs["tmp"] / "two", log=lambda *_: None)
    for name in PACK_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_a_measured_result_is_carried_and_its_figures_are_written_in_the_readme(inputs):
    block = _measured_for(inputs["doc"]["edit"])
    out = recipe_pack.build_pack(
        inputs["recipe"], inputs["dirs"], inputs["tmp"] / "pack", measured=block, log=lambda *_: None
    )
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "0.391 (n = 305, 0.95 interval 0.337 to 0.447)" in readme
    assert "Measured on one held-out partition, with one judge." in readme
    assert "Every interval is at the 0.95 level." in readme
    assert recipe_pack.verify_pack(out)["measured"] == "measured"


def test_a_measurement_bound_to_another_edit_is_refused(inputs):
    other = recipe_mod.build(
        name="other",
        model_dir=inputs["model"],
        model_id="tiny",
        revision=None,
        revision_reason="local",
        directions_path=inputs["dirs"],
        layer_count=LAYERS,
        strength=0.5,
        restore_norms=True,
        measured=None,
        command="cmd",
    )
    block = _measured_for(other["edit"])
    code = _code_of(
        recipe_pack.build_pack,
        inputs["recipe"],
        inputs["dirs"],
        inputs["tmp"] / "pack",
        measured=block,
        log=lambda *_: None,
    )
    assert code == "measured-mismatch"
    assert not (inputs["tmp"] / "pack").exists()


def test_a_pack_will_not_show_a_second_measured_result_beside_the_first(inputs):
    measured_path = _recipe_with(inputs, lambda d: d.__setitem__("measured", _measured_for(d["edit"])))
    code = _code_of(
        recipe_pack.build_pack,
        measured_path,
        inputs["dirs"],
        inputs["tmp"] / "pack",
        measured=_measured_for(inputs["doc"]["edit"]),
        log=lambda *_: None,
    )
    assert code == "pack-measured-conflict"


def test_a_pack_refuses_a_recipe_whose_base_is_not_pinned(inputs):
    path = _recipe_with(
        inputs,
        lambda d: (d["base"].__setitem__("weights_sha256", None), d["base"].__setitem__("weights_sha256_reason", "x")),
    )
    code = _code_of(recipe_pack.build_pack, path, inputs["dirs"], inputs["tmp"] / "pack", log=lambda *_: None)
    assert code == "pack-unpinned"


def test_a_pack_refuses_a_direction_file_that_is_not_the_one_the_recipe_means(inputs):
    other = inputs["tmp"] / "other.safetensors"
    make_directions(other)
    other.write_bytes(other.read_bytes() + b"\x00")
    code = _code_of(recipe_pack.build_pack, inputs["recipe"], other, inputs["tmp"] / "pack", log=lambda *_: None)
    assert code == "pack-mismatch"


def test_a_pack_refuses_a_direction_name_that_leaves_the_pack(inputs):
    path = _recipe_with(inputs, lambda d: d["edit"]["directions"].__setitem__("file", "../escape.safetensors"))
    code = _code_of(recipe_pack.build_pack, path, inputs["dirs"], inputs["tmp"] / "pack", log=lambda *_: None)
    assert code == "pack-invalid"


def test_a_pack_refuses_a_recipe_that_records_a_local_path(inputs):
    path = _recipe_with(
        inputs,
        lambda d: d["reproduce"].__setitem__("command", "senbonzakura stream-bake --model /home/someone/model"),
    )
    code = _code_of(recipe_pack.build_pack, path, inputs["dirs"], inputs["tmp"] / "pack", log=lambda *_: None)
    assert code == "pack-local-path"


def test_a_pack_is_never_written_over_an_existing_directory(inputs):
    existing = inputs["tmp"] / "pack"
    existing.mkdir()
    (existing / "keep.txt").write_text("someone else's", encoding="utf-8")
    code = _code_of(recipe_pack.build_pack, inputs["recipe"], inputs["dirs"], existing, log=lambda *_: None)
    assert code == "pack-exists"
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "someone else's"


def test_a_failed_pack_leaves_no_directory_and_no_staging_directory_behind(inputs, monkeypatch):
    def fail(path, recipe):
        raise OSError("disk full, simulated")

    monkeypatch.setattr(recipe_mod, "write", fail)
    with pytest.raises(OSError):
        recipe_pack.build_pack(inputs["recipe"], inputs["dirs"], inputs["tmp"] / "pack", log=lambda *_: None)
    assert not (inputs["tmp"] / "pack").exists()
    assert not [p for p in inputs["tmp"].iterdir() if p.name.startswith(".pack-")]


def _built(inputs, name="pack"):
    return recipe_pack.build_pack(inputs["recipe"], inputs["dirs"], inputs["tmp"] / name, log=lambda *_: None)


def _rewrite_entry(root, name, text_or_bytes):
    """Change one file and update the manifest to match, so only the rule under test can refuse the pack."""
    target = root / name
    if isinstance(text_or_bytes, bytes):
        target.write_bytes(text_or_bytes)
    else:
        target.write_text(text_or_bytes, encoding="utf-8")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    digest, size = recipe_pack._digest(target)
    manifest["files"][name] = {"sha256": digest, "bytes": size}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_verify_refuses_a_direction_file_altered_after_the_pack_was_written(inputs):
    out = _built(inputs)
    (out / "directions.safetensors").write_bytes((out / "directions.safetensors").read_bytes() + b"\x00")
    assert _code_of(recipe_pack.verify_pack, out) == "pack-mismatch"


def test_verify_refuses_a_file_the_manifest_does_not_list(inputs):
    out = _built(inputs)
    (out / "extra.txt").write_text("not covered by the manifest", encoding="utf-8")
    assert _code_of(recipe_pack.verify_pack, out) == "pack-mismatch"


def test_verify_refuses_a_pack_without_a_manifest(inputs):
    out = _built(inputs)
    (out / "MANIFEST.json").unlink()
    assert _code_of(recipe_pack.verify_pack, out) == "pack-invalid"


def test_verify_refuses_a_readme_that_says_more_than_the_recipe_records(inputs):
    out = _built(inputs)
    _rewrite_entry(out, "README.md", "# roundtrip-tiny\n\nAll refusal was removed, measured on everything.\n")
    assert _code_of(recipe_pack.verify_pack, out) == "pack-mismatch"


def test_verify_refuses_a_recipe_edited_under_its_measured_result(inputs):
    block = _measured_for(inputs["doc"]["edit"])
    out = recipe_pack.build_pack(
        inputs["recipe"], inputs["dirs"], inputs["tmp"] / "pack", measured=block, log=lambda *_: None
    )
    doc = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
    doc["edit"]["strength"] = 0.5
    _rewrite_entry(out, "recipe.json", json.dumps(doc, indent=2, sort_keys=True) + "\n")
    assert _code_of(recipe_pack.verify_pack, out) == "measured-mismatch"


def test_verify_refuses_a_link_standing_in_for_a_file(inputs):
    out = _built(inputs)
    elsewhere = inputs["tmp"] / "elsewhere.md"
    elsewhere.write_text((out / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "README.md").unlink()
    os.symlink(elsewhere, out / "README.md")
    assert _code_of(recipe_pack.verify_pack, out) == "pack-mismatch"


def test_the_reference_applier_accepts_a_packs_recipe_in_check_mode(inputs):
    out = _built(inputs)
    applier = load_applier()
    summary = applier.apply(
        out / "recipe.json",
        inputs["model"],
        out / "directions.safetensors",
        inputs["tmp"] / "never-written",
        check_only=True,
        log=lambda *_: None,
    )
    assert summary["status"] == "checked"
    assert not (inputs["tmp"] / "never-written").exists()


def test_a_failed_cleanup_is_reported_and_the_original_failure_is_what_surfaces(inputs, monkeypatch):
    def fail_write(path, recipe):
        raise OSError("disk full, simulated")

    def fail_remove(path):
        raise OSError("staging directory is busy, simulated")

    messages = []
    monkeypatch.setattr(recipe_mod, "write", fail_write)
    monkeypatch.setattr(recipe_pack.shutil, "rmtree", fail_remove)
    with pytest.raises(OSError, match="disk full"):
        recipe_pack.build_pack(inputs["recipe"], inputs["dirs"], inputs["tmp"] / "pack", log=messages.append)
    assert any("could not remove the unfinished pack" in m for m in messages)


def test_a_pack_appearing_during_the_write_is_not_replaced(inputs, monkeypatch):
    real_write = recipe_mod.write

    def race(path, recipe):
        (inputs["tmp"] / "pack").mkdir()
        return real_write(path, recipe)

    monkeypatch.setattr(recipe_mod, "write", race)
    code = _code_of(
        recipe_pack.build_pack, inputs["recipe"], inputs["dirs"], inputs["tmp"] / "pack", log=lambda *_: None
    )
    assert code == "pack-exists"


def test_verify_refuses_something_that_is_not_a_pack_directory(inputs):
    assert _code_of(recipe_pack.verify_pack, inputs["tmp"] / "no-such-pack") == "pack-invalid"


def test_verify_refuses_a_manifest_of_another_schema(inputs):
    out = _built(inputs)
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["schema"] = "abliteration-recipe-pack/9"
    (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _code_of(recipe_pack.verify_pack, out) == "pack-invalid"


def test_verify_refuses_a_manifest_that_does_not_list_the_readme(inputs):
    out = _built(inputs)
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    del manifest["files"]["README.md"]
    (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _code_of(recipe_pack.verify_pack, out) == "pack-invalid"


def test_verify_refuses_a_manifest_entry_that_is_not_a_hash_and_a_size(inputs):
    out = _built(inputs)
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["files"]["README.md"] = "not an entry"
    (out / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _code_of(recipe_pack.verify_pack, out) == "pack-invalid"


def test_verify_refuses_a_recipe_that_names_a_direction_file_the_pack_does_not_hold(inputs):
    out = _built(inputs)
    doc = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
    doc["edit"]["directions"]["file"] = "elsewhere.safetensors"
    _rewrite_entry(out, "recipe.json", json.dumps(doc, indent=2, sort_keys=True) + "\n")
    assert _code_of(recipe_pack.verify_pack, out) == "pack-mismatch"


def test_verify_refuses_a_recipe_whose_direction_hash_is_not_the_packed_file(inputs):
    out = _built(inputs)
    doc = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
    doc["edit"]["directions"]["sha256"] = "0" * 64
    _rewrite_entry(out, "recipe.json", json.dumps(doc, indent=2, sort_keys=True) + "\n")
    assert _code_of(recipe_pack.verify_pack, out) == "pack-mismatch"


def test_a_recipe_file_that_is_not_a_json_object_is_refused(inputs):
    path = inputs["tmp"] / "list.json"
    path.write_text("[]", encoding="utf-8")
    assert (
        _code_of(recipe_pack.build_pack, path, inputs["dirs"], inputs["tmp"] / "pack", log=lambda *_: None)
        == "pack-invalid"
    )


def test_a_direction_path_that_is_a_directory_is_refused(inputs):
    assert (
        _code_of(recipe_pack.build_pack, inputs["recipe"], inputs["tmp"], inputs["tmp"] / "pack", log=lambda *_: None)
        == "pack-invalid"
    )


def test_the_readme_says_what_a_real_revision_and_a_schedule_and_a_residual_look_like(inputs):
    doc = json.loads(inputs["recipe"].read_text(encoding="utf-8"))
    doc["base"]["revision"] = "3f2a9c1"
    doc["base"]["revision_reason"] = None
    del doc["edit"]["strength"]
    doc["edit"]["schedule"] = [{"target": t, "strengths": [1.0, 1.0]} for t in doc["edit"]["targets"]]
    doc["edit"]["direction_residual"] = {
        "status": "measured",
        "metric": "max_abs_projection/1",
        "definition": "the definition text",
        "before": 1.0,
        "after": 0.5,
        "fraction_remaining": 0.5,
        "rounds": 0,
    }
    readme = recipe_pack.render_readme(doc)
    assert "- Revision: 3f2a9c1" in readme
    assert "- Strength: a schedule" in readme
    assert "That leaves a fraction of 0.5 (metric max_abs_projection/1" in readme


def test_the_readme_reports_a_capability_that_was_not_measured_and_a_run_that_invalidated_itself(inputs):
    doc = json.loads(inputs["recipe"].read_text(encoding="utf-8"))
    doc["measured"] = _measured_for(
        doc["edit"],
        capability={"status": "not_measured", "why": "no benchmark was run"},
        self_invalidated="the seed was reused across arms",
    )
    readme = recipe_pack.render_readme(doc)
    assert "- Capability: not measured. no benchmark was run" in readme
    assert "recorded that its measurement is invalid: the seed was reused across arms" in readme


def test_the_readme_says_so_when_no_commit_could_be_named(inputs):
    doc = json.loads(inputs["recipe"].read_text(encoding="utf-8"))
    doc["provenance"]["source"] = {"commit": None, "dirty": None, "via": None, "reason": "no checkout was present"}
    assert "No commit could be named, because no checkout was present." in recipe_pack.render_readme(doc)


def test_a_pack_from_a_real_bake_verifies_and_the_reference_applier_checks_it(tmp_path):
    from senbonzakura import streambake

    model = tmp_path / "model"
    make_checkpoint(model)
    dirs = tmp_path / "directions.safetensors"
    make_directions(dirs)
    bake = tmp_path / "bake"
    assert streambake.main(["--model", str(model), "--directions", str(dirs), "--out", str(bake)]) == 0
    pack = recipe_pack.build_pack(bake / "recipe.json", dirs, tmp_path / "pack", log=lambda *_: None)
    assert recipe_pack.verify_pack(pack)["status"] == "verified"
    summary = load_applier().apply(
        pack / "recipe.json",
        model,
        pack / "directions.safetensors",
        tmp_path / "never-written",
        check_only=True,
        log=lambda *_: None,
    )
    assert summary["status"] == "checked"
