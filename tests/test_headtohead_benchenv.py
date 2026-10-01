# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The environment record, and the check that would have caught the numpy move.

WHY THESE TESTS EXIST AT ALL

`head-to-head/Dockerfile.tool` pinned `numpy==2.1.3` under a comment saying a benchmark whose
numeric stack moves under it is not a benchmark. Heretic v1.4.0 declares `numpy~=2.2`, which that
version does not satisfy, so the Heretic image upgraded numpy on every build while the senbonzakura
image did not. Both images asserted a version at build time and both asserted `torch`, which was
never the package that moved.

So the thing under test is not "does the record have the right keys". It is **does a second package
moving get caught**, which is the question the old assertion could not ask.

WHY THIS FILE IMPORTS BY PATH

`head-to-head/` has hyphens and no `__init__.py`, so it is not an importable package, and its other
scripts import `heretic` or `optuna` and cannot be imported on a developer machine at all. That is
exactly why `test_headtohead_argv_matches_scripts.py` reads them with `ast` instead. `benchenv.py`
is deliberately standard-library-only, for the same reason the sealed box needs it to be, so it
*can* be imported, and this file is the test that keeps it that way.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "head-to-head" / "benchenv.py"


def _load():
    spec = importlib.util.spec_from_file_location("benchenv_undertest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchenv = _load()


# ── the record itself ───────────────────────────────────────────────────────────────────
def test_the_record_sees_this_interpreter_and_sorts_its_output():
    """Sorted at the boundary, per baseline 2.1: iteration order must not reach a result."""
    loaded = benchenv.loaded_distributions()
    assert loaded, "importlib.metadata saw no distributions, so the record cannot describe anything"
    assert list(loaded) == sorted(loaded)
    # pytest is installed in any environment running this, so the record is demonstrably live
    # rather than a fixture that happens to parse.
    assert "pytest" in loaded


def test_names_are_normalised_so_a_lookup_cannot_miss_by_a_hyphen():
    assert benchenv.normalise("huggingface_hub") == "huggingface-hub"
    assert benchenv.normalise("  NumPy  ") == "numpy"


def test_a_missing_freeze_is_none_rather_than_empty(tmp_path):
    """None and {} are different claims: 'nothing said' against 'it contained no packages'."""
    assert benchenv.baked_freeze(tmp_path / "absent.txt") is None
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert benchenv.baked_freeze(empty) == {}


def test_the_freeze_parser_skips_the_forms_that_are_not_name_equals_version(tmp_path):
    """Editable and direct-URL lines are skipped: a URL in a provenance field is worse than a gap.

    `pip freeze` emits `-e git+...` and `name @ file:///...` as well as `name==version`.
    """
    path = tmp_path / "f.txt"
    path.write_text("# a comment\nnumpy==2.5.3\n-e git+https://example/x#egg=x\n"
                    "thing @ file:///tmp/thing.whl\nTorch==2.5.1+cu124\n\n", encoding="utf-8")
    assert benchenv.baked_freeze(path) == {"numpy": "2.5.3", "torch": "2.5.1+cu124"}


# ── the local-version rule ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("declared", "installed", "ok"), [
    ("2.5.1", "2.5.1", True),
    # The pin this image cares most about. An exact string compare fails here, which is why the
    # rule exists; constraints.txt makes the same argument in prose about not pinning +cu130.
    ("2.5.1", "2.5.1+cu124", True),
    ("2.5.1", "2.5.2", False),
    ("2.5.1", "2.5.10", False),
    # A declared local version is held exactly, so somebody who wants the CUDA build pinned can.
    ("2.5.1+cu124", "2.5.1+cu124", True),
    ("2.5.1+cu124", "2.5.1+cu121", False),
    ("2.5.1+cu124", "2.5.1", False),
])
def test_a_local_version_suffix_is_the_build_not_the_version(declared, installed, ok):
    assert benchenv.version_matches(declared, installed) is ok


def test_pins_parse_from_the_one_authored_string():
    got = benchenv.parse_pins("torch==2.5.1  numpy==2.5.3\ntransformers==5.14.1")
    assert got == {"torch": "2.5.1", "numpy": "2.5.3", "transformers": "5.14.1"}
    assert benchenv.parse_pins(None) == {}
    assert benchenv.parse_pins("garbage notapin") == {}


# ── the check that would have caught it ─────────────────────────────────────────────────
def _record(**loaded):
    return {"loaded": dict(loaded), "baked_freeze": dict(loaded), "loaded_count": len(loaded)}


def test_the_numpy_move_is_caught():
    """THE REGRESSION THIS FILE IS FOR.

    The base pinned 2.1.3 and the tool's own range forced an upgrade. Under the old build-time
    check, which asserted torch alone, this image passed.
    """
    record = _record(torch="2.5.1+cu124", numpy="2.5.3")
    with pytest.raises(benchenv.DeclaredPinError) as caught:
        benchenv.verify_declared(record, pins="torch==2.5.1 numpy==2.1.3", required_present="")
    message = str(caught.value)
    assert "numpy" in message
    assert "declared 2.1.3" in message and "installed 2.5.3" in message
    # The refusal has to say what to do about it, not merely that something is wrong.
    assert "re-measurement" in message


def test_torch_alone_passing_is_not_enough_to_pass():
    """The exact shape of the old guard: torch right, a second package wrong."""
    record = _record(torch="2.5.1+cu124", numpy="2.1.3")
    with pytest.raises(benchenv.DeclaredPinError):
        benchenv.verify_declared(record, pins="torch==2.5.1 numpy==2.5.3", required_present="")


def test_a_kept_environment_passes_and_reports_what_it_checked():
    record = _record(torch="2.5.1+cu124", numpy="2.5.3", tokenizers="0.22.2")
    got = benchenv.verify_declared(record, pins="torch==2.5.1 numpy==2.5.3",
                                   required_present="tokenizers")
    assert got["declared_pins"] == {"torch": "2.5.1", "numpy": "2.5.3"}
    assert got["required_present"] == ["tokenizers"]
    # Recorded, so an artefact can say what was checked rather than only that something was.
    assert got["checked"] == {"numpy": "2.5.3", "tokenizers": "0.22.2", "torch": "2.5.1+cu124"}


def test_a_required_package_that_is_absent_is_refused():
    """Presence is the honest check for an unpinned package, and it still has to be a check.

    `tokenizers` decides what the model is shown and its version is not pinned here.
    """
    record = _record(torch="2.5.1+cu124", numpy="2.5.3")
    with pytest.raises(benchenv.DeclaredPinError, match="tokenizers"):
        benchenv.verify_declared(record, pins="torch==2.5.1", required_present="tokenizers")


def test_a_declared_pin_for_a_package_that_is_not_installed_is_refused():
    record = _record(torch="2.5.1+cu124")
    with pytest.raises(benchenv.DeclaredPinError, match="not installed at all"):
        benchenv.verify_declared(record, pins="numpy==2.5.3", required_present="")


def test_an_image_declaring_nothing_is_refused_rather_than_passing_vacuously():
    """A check with nothing to check is the failure mode this replaces, not an acceptable state."""
    with pytest.raises(benchenv.DeclaredPinError, match="declares no pins"):
        benchenv.verify_declared(_record(torch="2.5.1"), pins="", required_present="")


# ── the fail-loud rule, at both ends ────────────────────────────────────────────────────
def test_an_empty_environment_refuses_rather_than_writing_a_null():
    with pytest.raises(benchenv.EnvironmentRecordError, match="empty"):
        benchenv.require_complete({"loaded": {}, "baked_freeze": {}})


def test_a_wrong_environment_is_named_as_wrong_not_as_unreadable():
    record = {"loaded": {"pytest": "8.0.0"}, "baked_freeze": {}, "loaded_count": 1}
    with pytest.raises(benchenv.EnvironmentRecordError, match="outside the image"):
        benchenv.require_complete(record)


def test_an_image_with_no_baked_freeze_refuses():
    record = {"loaded": {"torch": "2.5.1", "numpy": "2.5.3"}, "baked_freeze": None,
              "baked_freeze_path": "/opt/bench-env.txt"}
    with pytest.raises(benchenv.EnvironmentRecordError, match="no baked freeze"):
        benchenv.require_complete(record)


def test_a_budget_without_an_environment_block_is_refused():
    """The second end of the rule. A record that is built and then not written is the same gap."""
    with pytest.raises(benchenv.EnvironmentRecordError, match="no usable"):
        benchenv.require_recorded({"tool": "heretic", "seed": 42})
    with pytest.raises(benchenv.EnvironmentRecordError):
        benchenv.require_recorded({"environment": {"loaded": {}}})
    with pytest.raises(benchenv.EnvironmentRecordError):
        benchenv.require_recorded({"environment": None})
    # And accepts the real shape, so the test is not merely asserting that it raises.
    benchenv.require_recorded({"environment": {"loaded": {"torch": "2.5.1"}}})


def test_disagreement_between_baked_and_loaded_is_recorded_not_resolved():
    got = benchenv.disagreements({"numpy": "2.1.3", "torch": "2.5.1"},
                                 {"numpy": "2.5.3", "torch": "2.5.1"})
    assert got == {"numpy": {"baked": "2.1.3", "loaded": "2.5.3"}}
    # No baked freeze means no comparison to make, which is not the same as agreement.
    assert benchenv.disagreements(None, {"numpy": "2.5.3"}) == {}


# ── the property the sealed box depends on ──────────────────────────────────────────────
def test_the_module_imports_nothing_outside_the_standard_library():
    """IT RUNS INSIDE SOMEBODY ELSE'S IMAGE, where our dependencies are absent.

    `senbonzakura.metrics` and `firsttoken` are import-light for the same reason and
    `head-to-head/selftest.py` checks that property from inside the box. A third-party import added
    here would not fail on a developer machine; it would fail inside a tool's container, which is
    the one place it cannot be debugged cheaply.
    """
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    outside = sorted(r for r in roots if r not in sys.stdlib_module_names)
    assert not outside, f"benchenv.py imports {outside}, which the tool images do not carry"


def test_the_cli_emits_json_a_reader_can_parse(capsys):
    """`python /work/bench/benchenv.py` answers 'what is actually in here' without an arm."""
    assert benchenv.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["loaded_count"] == len(payload["loaded"])
    assert payload["baked_freeze_path"] == benchenv.BAKED_FREEZE
