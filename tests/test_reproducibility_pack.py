# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every published figure traces to a committed artefact.

A v1.0 exit-gate box asks that every claim in the README trace to a command in the
reproducibility pack, and until 2026-09-22 there was no pack, so the box could not be met or
even measured.

The value is not the document. It is that a claim which cannot be traced now fails a test,
rather than being noticed by a reader who went looking and found a paragraph where a file should
have been. That has happened here: the p-value this project was quoted on most often traced to a
working note excluded from the repository, and nothing caught it.
"""
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The figures the README and the paper state in public. Each must appear in the pack AND in a
#: committed artefact, because a pack that merely repeats a number has moved the untraceable
#: claim rather than closed it.
#: figure -> (artefact, the dotted path to the FIELD it is the value of).
#:
#: THE FIELD, NOT JUST THE FILE, and the difference is not pedantry. The first version of this
#: accepted any number anywhere in the artefact that rounded to the published figure, and a
#: mutation proved it worthless: changing the 0.6B model's `auc` to something else left the test
#: passing, because `controls.canonical_auc` happens to hold the same value for that model. A
#: trace that a coincidence can satisfy is not a trace.
PUBLISHED = {
    "0.9887": ("evidence/compass-2026-07-30/base-qwen3-1.7b.json", "auc"),
    "0.6564": ("evidence/compass-2026-07-30/base-qwen3-1.7b.json", "controls.length_only_auc"),
    "0.6616": ("evidence/compass-2026-07-30/base-qwen3-0.6b.json", "auc"),
}


def _flat(path):
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


def test_the_pack_exists():
    assert (ROOT / "REPRODUCING.md").is_file()
    assert (ROOT / "METHOD.md").is_file()


@pytest.mark.parametrize(("figure", "trace"), sorted(PUBLISHED.items()))
def test_every_published_figure_is_in_the_pack(figure, trace):
    assert figure in _flat("REPRODUCING.md"), (
        f"{figure} is published and REPRODUCING.md does not mention it, so a reader cannot find "
        f"out where it came from")


def _at(doc, dotted):
    """The value at a dotted path, or None if the path does not lead anywhere."""
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


@pytest.mark.parametrize(("figure", "trace"), sorted(PUBLISHED.items()))
def test_every_published_figure_is_supported_by_a_committed_artefact(figure, trace):
    """THE HALF THAT MATTERS. A pack naming a figure and a file that does not support it is a
    citation to nothing, which is the failure it exists to fix.

    Matched to the PRECISION PUBLISHED rather than as a string. The artefacts store full
    precision: the length-only control is 0.6563985752549933 and the README says 0.6564. A string
    match called that a broken trace when the trace was fine, which would have taught whoever met
    it to loosen the test rather than read the file.
    """
    artefact, field = trace
    path = ROOT / artefact
    assert path.is_file(), f"{artefact} is named in the pack and is not in the repository"
    value = _at(json.loads(path.read_text(encoding="utf-8")), field)
    assert isinstance(value, (int, float)), (
        f"{artefact} has no number at {field!r}, which is where {figure} is supposed to come from")
    places = len(figure.split(".")[1])
    assert f"{value:.{places}f}" == figure, (
        f"{artefact}:{field} is {value}, which is not {figure} at {places} decimal places. "
        f"Either the figure moved and the pack was not updated, or the pack cites the wrong "
        f"field")


@pytest.mark.parametrize(("figure", "trace"), sorted(PUBLISHED.items()))
def test_the_pack_names_the_artefact_for_each_figure(figure, trace):
    assert trace[0] in _flat("REPRODUCING.md"), (
        f"REPRODUCING.md quotes {figure} without naming {trace[0]}")


def test_the_readme_points_at_the_pack_and_the_method():
    """Two documents nobody is sent to are two documents nobody reads."""
    readme = _flat("README.md")
    for name in ("REPRODUCING.md", "METHOD.md"):
        assert name in readme, f"README.md does not link {name}"


def test_the_committed_evidence_is_still_stripped():
    """The pack invites people to open these files, so it is worth asserting afresh that they
    carry no prompts and no generations. `tools/ci/check_prompt_artefacts.py` is the real gate;
    this is the cheap one that runs with the suite.
    """
    for path in (ROOT / "evidence").rglob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        flat = json.dumps(doc)
        for banned in ('"prompt"', '"generation"', '"completion"', '"response"'):
            assert banned not in flat, f"{path} carries a {banned} field"


def test_the_pack_states_what_cannot_be_reproduced():
    """A reproducibility document that omits the limits is marketing. The corpus is not in this
    repository on purpose and that is the largest single limit on reproducing anything here.
    """
    pack = _flat("REPRODUCING.md")
    assert "What you cannot reproduce" in pack
    assert "corpus" in pack
    assert re.search(r"above 3B parameters", pack), "the pack does not state the size ceiling"
