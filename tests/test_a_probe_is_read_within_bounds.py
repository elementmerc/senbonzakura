# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A probe is a stranger's file, so reading it is bounded in size and in depth.

WHAT WAS WRONG

`probe.py` is the ingestion point whose entire purpose is loading files somebody else wrote, and
it had neither bound. It read the whole items file into memory before anything looked at it, and
it guarded `json.loads` for `ValueError` alone, so a line of `[[[[...]]]]` (a few kilobytes)
raised `RecursionError`, which is neither a `ValueError` nor an `OSError`, and escaped as a
traceback. `tools/ci/check_probes.py` calls `problems_with` as its only gate, so a crafted probe
crashed CI rather than being refused by it.

The checker's own ingestion point received exactly these two bounds the same cycle. This one did
not, which is the recurring shape here: a guard covers one spelling of a defect and reports clean
on the others.

WHAT THESE ASSERT

That both limits are the checker's own numbers rather than a second pair, and that every refusal
joins the module's "every reason at once" list instead of raising.
"""
from __future__ import annotations

import json

from senbonzakura_check import registry

from senbonzakura import probe as probefmt

MANIFEST = """
[probe]
name = "p"
measures = "arithmetic"
task = "numeric"
items = "items.jsonl"
version = "1"

[content]
class = "benign"

[source]
licence = "MIT"
author = "somebody"
"""


def _probe(tmp_path, *, manifest=MANIFEST, items=None):
    d = tmp_path / "p"
    d.mkdir(parents=True)
    (d / "probe.toml").write_text(manifest, encoding="utf-8")
    rows = items if items is not None else [
        json.dumps({"problem": "2 + 2", "reference": "4"}),
    ]
    (d / "items.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return d


def test_the_limits_are_the_checkers_own_and_not_a_second_pair():
    """Two numbers in two places drift, and this project has paid for that already."""
    assert probefmt.MAX_ARTEFACT_BYTES is registry.MAX_ARTEFACT_BYTES
    assert probefmt.MAX_ARTEFACT_DEPTH is registry.MAX_ARTEFACT_DEPTH


def test_a_good_probe_still_loads(tmp_path):
    """The guard must not be the thing that refuses every probe."""
    assert probefmt.problems_with(_probe(tmp_path)) == []
    assert probefmt.load(_probe(tmp_path / "second")).pairs() == [("2 + 2", "4")]


def test_an_items_file_past_the_size_cap_is_refused_without_being_read(tmp_path, monkeypatch):
    """`stat` before `read_text`, so an over-large file is never held in memory even briefly."""
    d = _probe(tmp_path)
    monkeypatch.setattr(probefmt, "MAX_ARTEFACT_BYTES", 8)
    read = []
    original = probefmt.pathlib.Path.read_text

    def _watch(self, *args, **kwargs):
        read.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(probefmt.pathlib.Path, "read_text", _watch)
    problems = probefmt.problems_with(d)
    assert any("reads at most" in p for p in problems), problems
    assert "items.jsonl" not in read, "the over-large file was read before it was refused"


def test_a_manifest_past_the_size_cap_is_refused(tmp_path, monkeypatch):
    d = _probe(tmp_path)
    monkeypatch.setattr(probefmt, "MAX_ARTEFACT_BYTES", 4)
    problems = probefmt.problems_with(d)
    assert any("probe.toml" in p and "reads at most" in p for p in problems), problems


def test_a_deeply_nested_item_is_a_problem_and_not_a_traceback(tmp_path):
    """THE DEFECT: a few kilobytes of brackets crashed the only gate CI runs."""
    deep = "[" * 40_000 + "]" * 40_000
    d = _probe(tmp_path, items=[deep])
    problems = probefmt.problems_with(d)
    assert any("nests deeper" in p for p in problems), problems


def test_an_item_nested_under_the_interpreter_ceiling_is_still_refused(tmp_path):
    """Between the documented limit and the parser's own ceiling nothing raises, so the depth is
    checked as well as caught. Without this, a 500-level document parses and passes.
    """
    deep = json.dumps({"problem": "x", "reference": _nest(registry.MAX_ARTEFACT_DEPTH + 20)})
    problems = probefmt.problems_with(_probe(tmp_path, items=[deep]))
    assert any("nests deeper" in p for p in problems), problems


def test_a_deeply_nested_manifest_is_a_problem_and_not_a_traceback(tmp_path):
    manifest = MANIFEST + "\nnested = " + "[" * 20_000 + "]" * 20_000 + "\n"
    problems = probefmt.problems_with(_probe(tmp_path, manifest=manifest))
    assert any("nests deeper" in p for p in problems), problems


def test_every_reason_is_still_reported_at_once(tmp_path):
    """The bounded read must not short-circuit the format's promise: a contributor fixing one
    problem should already know about the rest.
    """
    manifest = MANIFEST.replace('licence = "MIT"', "")
    problems = probefmt.problems_with(_probe(tmp_path, manifest=manifest,
                                             items=[json.dumps({"prompt": "x"})]))
    assert len(problems) >= 2, problems


def _nest(depth):
    obj = "x"
    for _ in range(depth):
        obj = [obj]
    return obj
