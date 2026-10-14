# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The conformance fixture and harness: a tool's conformance level is measured, not asserted.

`recipes/conformance/run.py` drives any tool through a command template and reports the level it
reaches. These tests run it on the reference applier, one named test per case, and show that the
harness refuses to award a level to a tool that does not earn it.
"""

import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = REPO / "recipes" / "conformance"
TOOL = f"{sys.executable} {REPO / 'recipes' / 'apply_recipe.py'} {{recipe}} {{model}} {{directions}} {{out}}"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load("conformance_run", FIXTURE / "run.py")
CASES = json.loads((FIXTURE / "cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_the_reference_applier_passes_every_conformance_case(case, tmp_path):
    result = harness.run_case(TOOL, case, tmp_path)
    assert result["ok"], result["problem"]


def test_the_reference_applier_reaches_the_core_profile_level(tmp_path):
    results = [harness.run_case(TOOL, case, tmp_path) for case in CASES]
    assert harness.level_of(results, CASES) == "core/1 core"


def _write_fake_tool(path, *, status, code=None, exit_code=0, make_output=False):
    body = f"""
import json, pathlib, sys
status = {status!r}
if {make_output!r} and status == "applied":
    pathlib.Path(sys.argv[4]).mkdir(parents=True, exist_ok=True)
print(json.dumps({{"status": status, "code": {code!r}, "ignored_extensions": []}}))
sys.exit({exit_code})
"""
    path.write_text(body, encoding="utf-8")
    return path


def test_a_tool_that_applies_everything_is_not_awarded_a_level(tmp_path):
    fake = _write_fake_tool(tmp_path / "applies_everything.py", status="applied", make_output=True)
    tool = f"{sys.executable} {fake} {{recipe}} {{model}} {{directions}} {{out}}"
    results = [harness.run_case(tool, case, tmp_path) for case in CASES]
    assert harness.level_of(results, CASES) == "none"
    refused = [r for r in results if r["id"] == "profile-unsupported"]
    assert refused and not refused[0]["ok"]


def test_a_tool_that_refuses_everything_reaches_no_level_because_the_valid_case_fails(tmp_path):
    fake = _write_fake_tool(
        tmp_path / "refuses_everything.py", status="refused", code="profile-unsupported", exit_code=2
    )
    tool = f"{sys.executable} {fake} {{recipe}} {{model}} {{directions}} {{out}}"
    results = [harness.run_case(tool, case, tmp_path) for case in CASES]
    assert harness.level_of(results, CASES) == "none"


def test_a_refusal_with_the_wrong_code_is_counted_as_a_failure(tmp_path):
    fake = _write_fake_tool(tmp_path / "wrong_code.py", status="refused", code="something-else", exit_code=2)
    tool = f"{sys.executable} {fake} {{recipe}} {{model}} {{directions}} {{out}}"
    case = next(c for c in CASES if c["id"] == "profile-missing")
    result = harness.run_case(tool, case, tmp_path)
    assert not result["ok"]
    assert "expected code" in result["problem"]


def test_the_harness_reports_a_patch_that_does_not_apply(tmp_path):
    broken = {
        "id": "broken",
        "expect": "refused",
        "code": "x",
        "patch": [{"op": "delete", "path": ["edit", "no_such_field"]}],
    }
    result = harness.run_case(TOOL, broken, tmp_path)
    assert not result["ok"]
    assert "does not apply" in result["problem"]


def test_the_fixture_stays_small_enough_to_read(tmp_path):
    # Counts the fixture and both harness scripts: two checkpoints of about 7 kB each, the directions,
    # the valid and defective recipes as patches, and the harness and checker source. Excludes the
    # untracked stale copies the operator has yet to clear, and any cache directory.
    tracked = [
        p
        for p in FIXTURE.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.relative_to(FIXTURE).as_posix() not in ("config.json", "model.safetensors")
    ]
    total = sum(p.stat().st_size for p in tracked)
    assert total < 48 * 1024, total


def test_the_applier_output_is_bit_identical_on_every_run(tmp_path):
    first = harness.run_case(TOOL, next(c for c in CASES if c["id"] == "valid"), tmp_path / "a")
    second = harness.run_case(TOOL, next(c for c in CASES if c["id"] == "valid"), tmp_path / "b")
    assert first["ok"] and second["ok"]
    out_a = pathlib.Path(tmp_path / "a" / "valid" / "model.safetensors").read_bytes()
    out_b = pathlib.Path(tmp_path / "b" / "valid" / "model.safetensors").read_bytes()
    assert out_a == out_b


def test_a_refused_run_leaves_no_output_checkpoint(tmp_path):
    case = next(c for c in CASES if c["id"] == "residual-mismatch")
    harness.run_case(TOOL, case, tmp_path)
    assert not (tmp_path / "residual-mismatch" / "model.safetensors").exists()
