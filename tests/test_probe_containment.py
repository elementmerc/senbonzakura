# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A contributed probe must not read a file outside its own directory.

A probe is a thing somebody hands you. `tools/ci/check_probes.py` calls `probe._check_items` as
its only gate on that, so whatever this function accepts, CI accepts.

THE HOLE, until 2026-09-25. Containment was tested with `str(path).startswith(str(directory))`,
which compares TEXT rather than path structure. A sibling directory whose name merely extends the
probe's name satisfies it:

    probe directory   /x/p
    manifest items    ../p-evil/items.jsonl
    resolves to       /x/p-evil/items.jsonl
    startswith("/x/p")   True      <- accepted
    is_relative_to       False     <- correct

The sibling case is the one a naive prefix check always misses, and it needs no unusual filename:
`p` and `p-evil` are both ordinary directory names.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura import probe  # noqa: E402


def _probe_dir(tmp_path, items_name):
    d = tmp_path / "p"
    (d / "sub").mkdir(parents=True)
    (d / "items.jsonl").write_text(
        json.dumps({"problem": "2+2?", "reference": "#### 4"}) + "\n", encoding="utf-8")
    (d / "sub" / "items.jsonl").write_text(
        json.dumps({"problem": "2+2?", "reference": "#### 4"}) + "\n", encoding="utf-8")
    return d, {"items": items_name}


def _outside(tmp_path):
    """A sibling whose name EXTENDS the probe directory's, which is the case that escaped."""
    d = tmp_path / "p-evil"
    d.mkdir(parents=True, exist_ok=True)
    (d / "items.jsonl").write_text(
        json.dumps({"problem": "secret", "reference": "#### 1"}) + "\n", encoding="utf-8")
    return d


@pytest.mark.parametrize("escape", [
    "../p-evil/items.jsonl",
    "../../p-evil/items.jsonl",
    "sub/../../p-evil/items.jsonl",
])
def test_a_manifest_cannot_read_a_sibling_that_extends_the_directory_name(tmp_path, escape):
    _outside(tmp_path)
    d, decl = _probe_dir(tmp_path, escape)
    problems = []
    probe._check_items(d, decl, problems)
    assert problems, (
        f"{escape!r} resolved outside the probe directory and was accepted. A string prefix test "
        f"passes this; a path containment test does not.")


def test_the_ordinary_case_still_works(tmp_path):
    d, decl = _probe_dir(tmp_path, "items.jsonl")
    problems = []
    probe._check_items(d, decl, problems)
    assert not problems, f"a probe reading its own items was refused: {problems}"


def test_a_subdirectory_of_the_probe_is_still_allowed(tmp_path):
    """Containment means inside, not immediately inside."""
    d, decl = _probe_dir(tmp_path, "sub/items.jsonl")
    problems = []
    probe._check_items(d, decl, problems)
    assert not problems, f"a probe reading its own subdirectory was refused: {problems}"


def test_the_check_does_not_use_a_string_prefix():
    """The regression guard: the defect was the technique, so the technique is what is pinned.

    Comment lines are stripped before looking, because the comment explaining the old technique
    necessarily names it. The first version of this test failed on its own explanation, which is
    the same shape as a guard that greps a module for a string and finds it in a docstring.
    """
    src = Path(probe.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _check_items"):]
    body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    code = "\n".join(line for line in body.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "startswith" not in code, (
        "_check_items is testing containment with a string prefix again. A sibling directory "
        "whose name extends the probe's name passes that test.")
    assert "is_relative_to" in code, "the containment test is no longer a path containment test"
