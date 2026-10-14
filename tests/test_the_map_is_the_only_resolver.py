# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The map owns what a tensor is. A second answer anywhere fails this test.

Two answers to "what is this tensor" is the defect class this project fixed when two places
counted the directions in one file. The map (`senbonzakura.modelmap`) is the one place that
classifies a tensor, names the layer-count and expert keys, and maps GGUF names. This test reads
every module under `src/senbonzakura` and, when it is visible, the structure probe, and fails when
any of them:

  1. defines a name the map owns (a function such as `classify_block_path`, or a table such as
     `LAYER_COUNT_KEYS`) with its own value, not an alias of the map's, or
  2. defines a top-level table whose literal strings overlap the map's own vocabulary in two or
     more places. That is a parallel key table under a new name, which rule 1 alone would miss.

A violation is either removed or listed in KNOWN_EXCEPTIONS with the reason it is still there. An
exception that no longer matches anything fails too, so the list shrinks as the duplicates are
paid off rather than growing as they are forgotten.

The structure probe lives in an unversioned `private/` tree. It is checked when the path is given
through SENBONZAKURA_PROBES_DIR, and that check is skipped with a reason otherwise, because a fresh
clone does not have it. The probe is the thing most likely to drift, so run this with the variable
set before merging anything that touches the map.
"""
import ast
import os
from pathlib import Path

import pytest

from senbonzakura import modelmap

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "senbonzakura"
MAP_FILE = PACKAGE / "modelmap.py"

OWNED_FUNCTIONS = {"classify_block_path", "tensor_role", "block_of", "segments",
                   "collapse_expert_index", "_is_norm"}
OWNED_NAMES = {"LAYER_COUNT_KEYS", "EXPERT_COUNT_KEYS", "EXPERT_TOPK_KEYS", "ATTENTION_SEGMENTS",
               "MLP_SEGMENTS", "ROUTER_SEGMENTS", "CONV_SEGMENTS", "SSM_SEGMENTS", "NORM_MARKERS",
               "GGUF_BLOCK_ROLES", "GGUF_METADATA_SPELLINGS"}


def _table_strings(path):
    """Every string literal held in a top-level table of one module, read from its source."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    strings = set()
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            strings.update(e.value for e in value.elts
                           if isinstance(e, ast.Constant) and isinstance(e.value, str))
        elif isinstance(value, ast.Dict):
            strings.update(k.value for k in value.keys
                           if isinstance(k, ast.Constant) and isinstance(k.value, str))
    return strings


#: The map's own vocabulary: every key spelling and segment name in the map's module-level tables.
#: Read from the map's source, so a spelling the map learns later is guarded without editing this
#: test, and a parallel table anywhere else is caught by overlapping that vocabulary.
VOCABULARY = _table_strings(MAP_FILE)

#: Duplicates that exist today, each with the reason it is still there and what would remove it.
#: Keyed by (module path relative to its root, top-level name). Adding to this list is a decision
#: that needs a reason; the stale check below makes sure an entry is deleted when it is paid off.
KNOWN_EXCEPTIONS = {
    ("src/senbonzakura/writers.py", "ATTN_BLOCKS"): (
        "the writer module owns the block-kind vocabulary and the map imports it from here. "
        "Moving it into the map inverts that dependency, so it is paid off by making the writer "
        "module take its names from the map once the writers no longer need them."),
    ("src/senbonzakura/writers.py", "MIXER_BLOCKS"): (
        "same as ATTN_BLOCKS: the writers' mixer vocabulary, imported by the map."),
    ("src/senbonzakura/writers.py", "MLP_BLOCKS"): (
        "same as ATTN_BLOCKS: the writers' MLP vocabulary, imported by the map."),
    ("src/senbonzakura/writers.py", "WRITER_PROJECTIONS"): (
        "the projections that write into the residual stream, named a third time here. The "
        "diff command and the map's output-projection lookup each keep their own copy. Paid off "
        "by one table in the map that both import."),
    ("src/senbonzakura/modeldiff.py", "WRITER_HINTS"): (
        "the diff command's list of residual writers, a second copy of WRITER_PROJECTIONS. "
        "Paid off by importing the map's table."),
}

#: The probe's remaining tables, which the map does not own yet. Each one is the next thing to move
#: into the map, in the order the probe's own comments describe them. Keyed by (file, name).
PROBE_EXCEPTIONS = {
    ("structure_of_a_checkpoint.py", "EXTRA_BLOCK_KEYS"): (
        "extra blocks stored past the declared stack. Moves into the map with block counting."),
    ("structure_of_a_checkpoint.py", "SHARED_EXPERT_KEYS"): (
        "shared-expert keys, a mix of a count and a width. Moves with the expert resolver."),
    ("structure_of_a_checkpoint.py", "BLOCK_TYPE_LIST_KEYS"): (
        "the per-block mechanism list spellings. The map has the same spellings under its own "
        "names; these move into the map's mechanism section."),
    ("structure_of_a_checkpoint.py", "NESTED_CONFIG_KEYS"): (
        "where a multi-tower config hides the decoder's config. Moves with the map's decoder "
        "scope, which already reads these spellings under its own name."),
    ("structure_of_a_checkpoint.py", "PATTERN_LETTERS"): (
        "Nemotron-H's one-letter block codes. Moves into the map's mechanism table."),
}

#: Known gap, stated rather than hidden: a table holding ONE spelling, such as the probe's
#: `BLOCK_TYPE_PATTERN_KEY = "hybrid_override_pattern"`, is below the two-spelling threshold above,
#: so this guard cannot see it. Single-key duplicates are caught by review, not by this test.


def _module_tables_and_names(path):
    """Top-level function names, assigned names and literal key tables in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions, names, tables = [], [], []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target.id], node.value
        # An alias such as `LAYER_COUNT_KEYS = modelmap.LAYER_COUNT_KEYS` is the same object, not a
        # second copy, so it is not a redefinition.
        if isinstance(value, (ast.Attribute, ast.Name)):
            continue
        names.extend(targets)
        if value is not None and targets:
            strings = set()
            if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
                strings = {e.value for e in value.elts
                           if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            elif isinstance(value, ast.Dict):
                strings = {k.value for k in value.keys
                           if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            overlap = strings & VOCABULARY
            if len(overlap) >= 2:
                tables.append((",".join(targets), sorted(overlap)))
    return functions, names, tables


def _violations(path, root_label):
    functions, names, tables = _module_tables_and_names(path)
    found = {}
    for fn in functions:
        if fn in OWNED_FUNCTIONS:
            found[(root_label, fn)] = f"redefines the map's {fn}()"
    for name in names:
        if name in OWNED_NAMES:
            found[(root_label, name)] = f"redefines the map's {name}"
    for joined, overlap in tables:
        for name in joined.split(","):
            found.setdefault((root_label, name),
                             f"a literal table repeats {len(overlap)} of the map's key spellings "
                             f"({overlap[:3]}...)")
    return found


def _package_violations():
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if path == MAP_FILE:
            continue
        label = path.relative_to(REPO).as_posix()
        found.update(_violations(path, label))
    return found


def _probe_violations():
    probes = os.environ.get("SENBONZAKURA_PROBES_DIR")
    if not probes:
        return None
    found = {}
    for path in sorted(Path(probes).glob("*.py")):
        found.update(_violations(path, path.name))
    return found


def test_no_package_module_defines_a_second_resolver_outside_the_known_exceptions():
    found = _package_violations()
    unknown = {key: why for key, why in found.items() if key not in KNOWN_EXCEPTIONS}
    assert not unknown, (
        "a second answer to what a tensor is, outside the map. Remove the duplicate and let the "
        "map answer, or list it in KNOWN_EXCEPTIONS with a reason:\n"
        + "\n".join(f"  {path}:{name} - {why}" for (path, name), why in sorted(unknown.items())))


def test_every_known_package_exception_still_matches_a_real_duplicate():
    found = _package_violations()
    stale = [key for key in KNOWN_EXCEPTIONS if key not in found]
    assert not stale, (
        "these exceptions no longer match a duplicate, so they should be deleted: " + repr(stale))


def test_every_known_exception_has_a_reason():
    for key, reason in {**KNOWN_EXCEPTIONS, **PROBE_EXCEPTIONS}.items():
        assert reason.strip(), key


@pytest.mark.skipif(not os.environ.get("SENBONZAKURA_PROBES_DIR"),
                    reason="the structure probe lives in an unversioned private/ tree; set "
                           "SENBONZAKURA_PROBES_DIR to its probes directory to check it")
def test_the_structure_probe_holds_no_second_resolver_outside_its_known_exceptions():
    found = _probe_violations()
    unknown = {key: why for key, why in found.items() if key not in PROBE_EXCEPTIONS}
    assert not unknown, (
        "the structure probe answers a tensor question the map should answer:\n"
        + "\n".join(f"  {path}:{name} - {why}" for (path, name), why in sorted(unknown.items())))


@pytest.mark.skipif(not os.environ.get("SENBONZAKURA_PROBES_DIR"),
                    reason="needs SENBONZAKURA_PROBES_DIR, see the test above")
def test_every_probe_exception_still_matches_a_real_duplicate():
    probe = Path(os.environ["SENBONZAKURA_PROBES_DIR"]) / "structure_of_a_checkpoint.py"
    assert probe.exists(), (f"SENBONZAKURA_PROBES_DIR does not hold structure_of_a_checkpoint.py "
                            f"({probe}), so the probe exceptions cannot be checked against it")
    found = _probe_violations()
    stale = [key for key in PROBE_EXCEPTIONS if key not in found]
    assert not stale, "these probe exceptions no longer match, so delete them: " + repr(stale)
