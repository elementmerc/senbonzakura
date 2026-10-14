# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The committed conformance fixture passes its own checker, and the checker can actually fail.

The fixture lives in `recipes/conformance/` and is built from senbonzakura's bake by
`tools/recipe/build_conformance.py`. The checker, `recipes/conformance/check.py`, is the piece a
third party runs; it does not import senbonzakura. These tests run the reference applier on the
fixture and hand the result to that checker, then plant defects to show the checker refuses them.
"""

import importlib.util
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = REPO / "recipes" / "conformance"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


applier = _load("apply_recipe", REPO / "recipes" / "apply_recipe.py")
checker = _load("conformance_check", FIXTURE / "check.py")


def _applied(tmp_path):
    """The reference applier's output for the fixture, in a fresh directory."""
    model = tmp_path / "model"
    model.mkdir(parents=True)
    shutil.copyfile(FIXTURE / "model" / "model.safetensors", model / "model.safetensors")
    shutil.copyfile(FIXTURE / "model" / "config.json", model / "config.json")
    out = tmp_path / "applied"
    applier.apply(FIXTURE / "recipe.json", model, FIXTURE / "directions.safetensors", out, log=lambda *_: None)
    return out


def test_the_reference_applier_passes_the_conformance_fixture(tmp_path):
    out = _applied(tmp_path)
    actual = checker.read_tensors(out)
    expected = checker.read_tensors(FIXTURE / "expected")
    assert checker.compare(actual, expected) == []


def test_the_checker_passes_its_own_fixture_through_the_command_line(tmp_path):
    out = _applied(tmp_path)
    assert checker.main(["check.py", str(out), str(FIXTURE / "expected")]) == 0


def test_the_checker_refuses_a_float32_value_that_is_off_by_more_than_the_tolerance(tmp_path):
    out = _applied(tmp_path)
    actual = checker.read_tensors(out)
    expected = checker.read_tensors(FIXTURE / "expected")
    name = "model.layers.1.mlp.down_proj.weight"
    dtype, values, shape = actual[name]
    tampered = values.copy()
    tampered[0, 0] += 1e-3
    actual[name] = (dtype, tampered, shape)
    problems = checker.compare(actual, expected)
    assert any(name in p and "differs" in p for p in problems)


def test_the_checker_refuses_a_bfloat16_value_that_is_off_by_more_than_one_unit(tmp_path):
    out = _applied(tmp_path)
    actual = checker.read_tensors(out)
    expected = checker.read_tensors(FIXTURE / "expected")
    name = "model.layers.0.self_attn.o_proj.weight"
    dtype, values, shape = actual[name]
    assert dtype == "BF16"
    tampered = values.copy()
    tampered[3, 3] = tampered[3, 3] * 1.5
    actual[name] = (dtype, tampered, shape)
    problems = checker.compare(actual, expected)
    assert any(name in p for p in problems)


def test_the_checker_refuses_a_tensor_whose_dtype_changed(tmp_path):
    out = _applied(tmp_path)
    actual = checker.read_tensors(out)
    expected = checker.read_tensors(FIXTURE / "expected")
    name = "model.layers.1.mlp.down_proj.weight"
    _dtype, values, shape = actual[name]
    actual[name] = ("F16", values, shape)
    problems = checker.compare(actual, expected)
    assert any("dtype" in p and name in p for p in problems)


def test_the_checker_refuses_a_missing_tensor(tmp_path):
    out = _applied(tmp_path)
    actual = checker.read_tensors(out)
    expected = checker.read_tensors(FIXTURE / "expected")
    del actual["model.norm.weight"]
    problems = checker.compare(actual, expected)
    assert any("missing" in p for p in problems)


def test_an_unedited_checkpoint_is_refused_as_not_the_edit(tmp_path):
    # The sharpest failure a third party can make: returning the base model untouched and
    # reporting success. The fixture's expected output differs from the base, so the checker must
    # say so.
    base = checker.read_tensors(FIXTURE / "model")
    expected = checker.read_tensors(FIXTURE / "expected")
    problems = checker.compare(base, expected)
    assert problems, "the base checkpoint should not pass as the edited one"


def test_the_checker_exit_status_is_one_on_a_failure(tmp_path):
    out = _applied(tmp_path)
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    path = bad / "model.safetensors"
    raw = bytearray(path.read_bytes())
    header, data_start, _size = applier.read_safetensors_header(path)
    # The sign bit of the last byte of one float32 value: a change that cannot hide under the
    # tolerance, so the exit status is the only thing being tested.
    begin = header["model.layers.1.mlp.down_proj.weight"]["data_offsets"][0]
    raw[data_start + begin + 3] ^= 0x80
    path.write_bytes(bytes(raw))
    assert checker.main(["check.py", str(bad), str(FIXTURE / "expected")]) == 1


def test_the_fixture_is_small_enough_to_read_in_a_glance():
    total = sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())
    assert total < 64 * 1024, total


def test_the_applier_output_is_bit_identical_on_every_run(tmp_path):
    # Reproducibility: a published output hash has to be a fact about the recipe, not about one
    # run, so two runs must write the same bytes.
    first = _applied(tmp_path / "a")
    second = _applied(tmp_path / "b")
    assert (first / "model.safetensors").read_bytes() == (second / "model.safetensors").read_bytes()
