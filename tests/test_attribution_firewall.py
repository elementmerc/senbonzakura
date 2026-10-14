# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Where the tool wrote is not evidence of where the behaviour lived, enforced rather than promised.

The abliteration edit has a footprint: the blocks whose weights it changed. Reporting that as
attribution would be cheap, invalid and believed. So the attribution modules are forbidden from
reaching the things that carry the footprint, and this test fails the moment one of them does.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "senbonzakura"
MODULES = sorted(SRC.glob("attribution*.py"))

#: Modules that hold or decide the edit. Importing any of them is a route to the footprint.
FORBIDDEN_MODULES = {"cli", "streambake", "streamextract", "tamper", "modeldiff", "writers",
                     "checkpoint", "runrecord", "validate", "residualleak"}
#: Names that ARE the footprint: the per layer direction sets, the layer weight profile, the bake,
#: and the record the tool writes about what it edited.
FORBIDDEN_NAMES = {"dirs_multi", "dirs_per_layer", "dirs_per_position", "layer_weight",
                   "directions_per_layer", "bake", "bake_pc", "restore_weights"}
FORBIDDEN_STRINGS = ("abliteration.json", "directions_per_layer", "best-config", "trials.json")


def test_there_are_attribution_modules_to_guard():
    assert MODULES, "no attribution modules found, so this test would pass over nothing"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_attribution_never_reaches_the_edit_footprint(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    seen_modules, seen_names, seen_strings = set(), set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            seen_modules.update((node.module or "").split("."))
            seen_names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                seen_modules.update(a.name.split("."))
        elif isinstance(node, ast.Name):
            seen_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen_names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            seen_strings.append(node.value)
    # `from . import cli` shows up as an imported NAME, so both sets are checked against both lists.
    assert not (seen_modules | seen_names) & FORBIDDEN_MODULES, path.name
    assert not seen_names & FORBIDDEN_NAMES, path.name
    for text in seen_strings:
        assert not any(bad in text for bad in FORBIDDEN_STRINGS), (path.name, text[:60])


def test_the_guard_does_catch_a_module_that_reaches_for_the_footprint(tmp_path):
    bad = tmp_path / "attribution_bad.py"
    bad.write_text("from . import cli\nlayers = run.dirs_per_layer\n", encoding="utf-8")
    tree = ast.parse(bad.read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert names & FORBIDDEN_MODULES and attrs & FORBIDDEN_NAMES
