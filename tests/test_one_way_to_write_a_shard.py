# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""There is exactly one way `streaming.py` puts a shard into place.

A second writer (a 4-bit path wanting a different byte layout is the obvious temptation) means a
crash leaves a checkpoint whose recovery depends on which writer made it. This pins the shape
rather than a behaviour: one function renames into place, and it is `rewrite_shard`.
"""
import ast
import pathlib

from senbonzakura import streaming

SOURCE = pathlib.Path(streaming.__file__)
PLACERS = {"replace", "rename", "move", "copyfile", "copy", "copy2"}


def _placing_calls(tree):
    """`(enclosing function, call name)` for every call that could put a file into place."""
    found = []

    def walk(node, owner):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner
            if isinstance(child, ast.Call):
                fn = child.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in PLACERS:
                    found.append((owner, name))
            walk(child, inner)

    walk(tree, "<module>")
    return found


def test_exactly_one_function_renames_a_shard_into_place():
    assert _placing_calls(ast.parse(SOURCE.read_text())) == [("rewrite_shard", "replace")]


def test_the_scan_sees_a_second_writer_when_one_is_added():
    planted = ast.parse(SOURCE.read_text() + "\n\ndef write_quantised(tmp, path):\n"
                        "    import os\n    os.rename(tmp, path)\n")
    assert ("write_quantised", "rename") in _placing_calls(planted)
