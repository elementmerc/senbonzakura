# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`stream-bake` refuses a direction set that is not orthonormal, before it writes a byte.

The defect: a basis with a row of squared norm 9 was baked without error and reported success, and
the result was a different model. These tests reproduce that exact case through the real command,
not only through the check, and show that a valid basis still bakes.
"""

import subprocess
import sys

import numpy as np
import pytest

from senbonzakura import streambake
from tests.test_recipe_roundtrip import HIDDEN, LAYERS, make_checkpoint, make_directions


def _scaled_row_directions(path, *, layer=0, factor=3.0):
    """A direction set identical to `make_directions` except one row scaled, so its norm is `factor`."""
    arr, _meta = streambake.load_directions(_orthonormal_path(path))
    bad = np.array(arr, copy=True)
    bad[layer + 1, 0] *= factor
    streambake.save_directions(path, bad, model="test", mode="per_layer")
    return path


def _orthonormal_path(path):
    ok = path.with_name("ok-" + path.name)
    if not ok.exists():
        make_directions(ok)
    return ok


def test_the_check_names_the_layer_and_the_deviation(tmp_path):
    arr = np.zeros((LAYERS + 1, 2, HIDDEN), np.float32)
    arr[1, 0, 0] = 3.0
    arr[1, 1, 1] = 1.0
    with pytest.raises(streambake.StreamBakeError, match="layer 0") as err:
        streambake.check_orthonormal(arr, [0], K=2)
    assert "orthonormal" in str(err.value)


def test_the_check_passes_an_orthonormal_set(tmp_path):
    arr, _ = streambake.load_directions(_orthonormal_path(tmp_path / "d.safetensors"))
    streambake.check_orthonormal(arr, range(LAYERS), K=2)


def test_the_check_ignores_empty_slots_and_rows_beyond_k(tmp_path):
    # A zero row is an unused slot. A bad row beyond K is not used, so it must not be refused.
    arr = np.zeros((LAYERS + 1, 3, HIDDEN), np.float32)
    arr[1, 0, 0] = 1.0
    arr[1, 2, 1] = 9.0
    streambake.check_orthonormal(arr, [0], K=2)


def test_the_check_only_looks_at_layers_the_bake_edits(tmp_path):
    arr = np.zeros((LAYERS + 1, 2, HIDDEN), np.float32)
    arr[1, 0, 0] = 1.0
    arr[2, 0, 0] = 5.0  # layer 1's direction is bad, but the bake is asked to edit layer 0 only
    streambake.check_orthonormal(arr, [0], K=1)


def test_a_real_bake_refuses_the_scaled_row_before_writing_anything(tmp_path):
    model = tmp_path / "model"
    make_checkpoint(model)
    bad = _scaled_row_directions(tmp_path / "bad.safetensors")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as err:
        streambake.main(["--model", str(model), "--directions", str(bad), "--out", str(out)])
    message = str(err.value)
    assert "layer 0" in message and "orthonormal" in message
    # Nothing was rewritten: no progress record, and the shards are the source's bytes.
    assert not (out / streambake.PROGRESS_NAME).exists()
    assert (out / "model.safetensors").read_bytes() == (model / "model.safetensors").read_bytes()


def test_a_real_bake_of_an_orthonormal_set_still_succeeds(tmp_path):
    model = tmp_path / "model"
    make_checkpoint(model)
    good = _orthonormal_path(tmp_path / "d.safetensors")
    assert streambake.main(["--model", str(model), "--directions", str(good), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / streambake.PROGRESS_NAME).exists()


def test_the_command_line_refuses_the_bad_set_with_a_nonzero_exit(tmp_path):
    model = tmp_path / "model"
    make_checkpoint(model)
    bad = _scaled_row_directions(tmp_path / "bad.safetensors")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "senbonzakura",
            "stream-bake",
            "--model",
            str(model),
            "--directions",
            str(bad),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode != 0
    assert "not orthonormal" in result.stderr
    assert "layer 0" in result.stderr
