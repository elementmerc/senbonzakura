# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`stream-bake` writes a recipe beside every bake it can express, measuring the residual by default.

A recipe is the better artefact, so these run through the real command. A bake the profile cannot
express still succeeds, and says plainly why no recipe was written.
"""

import json
import subprocess
import sys

import pytest

from senbonzakura import streambake
from tests.test_recipe_roundtrip import LAYERS, load_applier, make_checkpoint, make_directions


@pytest.fixture
def bake_inputs(tmp_path):
    model = tmp_path / "model"
    make_checkpoint(model)
    dirs = tmp_path / "directions.safetensors"
    make_directions(dirs)
    return model, dirs


def _bake(model, dirs, out, *extra):
    return streambake.main(["--model", str(model), "--directions", str(dirs), "--out", str(out), *extra])


def test_a_default_bake_writes_a_recipe_with_a_measured_residual(bake_inputs, tmp_path):
    model, dirs = bake_inputs
    out = tmp_path / "out"
    assert _bake(model, dirs, out) == 0
    recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
    assert recipe["profile"] == "core/1"
    residual = recipe["edit"]["direction_residual"]
    assert residual["status"] == "measured"
    assert residual["metric"] == "max_abs_projection/1"
    assert 0 <= residual["after"] < residual["before"]


def test_the_recipe_a_bake_writes_is_accepted_by_the_reference_applier(bake_inputs, tmp_path):
    model, dirs = bake_inputs
    out = tmp_path / "out"
    _bake(model, dirs, out)
    summary = load_applier().apply(out / "recipe.json", model, dirs, tmp_path / "again", log=lambda *_: None)
    assert summary["residual"]["measured"] is True


def test_the_recipe_records_the_command_that_made_it(bake_inputs, tmp_path):
    model, dirs = bake_inputs
    out = tmp_path / "out"
    _bake(model, dirs, out, "--strength", "0.8")
    recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
    assert recipe["reproduce"]["command"].startswith("senbonzakura stream-bake ")
    assert "--strength 0.8" in recipe["reproduce"]["command"]
    assert recipe["edit"]["strength"] == pytest.approx(0.8)


def test_skip_residual_records_not_measured_with_the_reason_named(bake_inputs, tmp_path):
    model, dirs = bake_inputs
    out = tmp_path / "out"
    _bake(model, dirs, out, "--skip-residual")
    recipe = json.loads((out / "recipe.json").read_text(encoding="utf-8"))
    residual = recipe["edit"]["direction_residual"]
    assert residual["status"] == "not_measured"
    assert "--skip-residual" in residual["why"]
    # The skipped recipe is still valid: the applier accepts a not_measured residual.
    load_applier().apply(out / "recipe.json", model, dirs, tmp_path / "again", log=lambda *_: None)


def test_a_bake_with_sparsity_succeeds_and_says_no_recipe_was_written(bake_inputs, tmp_path, capsys):
    model, dirs = bake_inputs
    out = tmp_path / "out"
    assert _bake(model, dirs, out, "--sparsity", "0.3") == 0
    assert not (out / "recipe.json").exists()
    assert (out / streambake.PROGRESS_NAME).exists()
    assert "no recipe written" in capsys.readouterr().out


def test_a_bake_that_used_a_subset_of_directions_gets_no_recipe(bake_inputs, tmp_path, capsys):
    model, dirs = bake_inputs
    out = tmp_path / "out"
    _bake(model, dirs, out, "--max-directions", "1")
    assert not (out / "recipe.json").exists()
    assert "no recipe written" in capsys.readouterr().out


def test_the_command_line_bake_writes_the_recipe_too(bake_inputs, tmp_path):
    model, dirs = bake_inputs
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "senbonzakura",
            "stream-bake",
            "--model",
            str(model),
            "--directions",
            str(dirs),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((out / "recipe.json").read_text(encoding="utf-8"))["profile"] == "core/1"
    assert LAYERS == 2


def test_a_bake_that_edits_a_tensor_the_targets_do_not_name_gets_no_recipe(tmp_path, capsys):
    """The bake plans a linear-attention output projection that the recipe's targets do not name.

    Without the set-equality check the recipe would omit that edit and describe a different model.
    The bake still succeeds, and the message says why no recipe was written.
    """
    import numpy as np

    from tests.test_recipe_roundtrip import HIDDEN, _encode_st, _load_all

    model = tmp_path / "model"
    make_checkpoint(model)
    arrays = {name: ("F32", arr) for name, (_dtype, arr) in _load_all(model / "model.safetensors").items()}
    rng = np.random.default_rng(7)
    arrays["model.layers.0.linear_attn.out_proj.weight"] = (
        "F32",
        rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32),
    )
    _encode_st(model / "model.safetensors", arrays)
    dirs = tmp_path / "directions.safetensors"
    make_directions(dirs)
    out = tmp_path / "out"
    assert _bake(model, dirs, out) == 0
    assert not (out / "recipe.json").exists()
    assert "linear_attn.out_proj" in capsys.readouterr().out
