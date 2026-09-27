# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Pointing the checker at the output directory of a run must never raise.

WHAT HAPPENED, 2026-09-26

A first-time user with a GPU finished an abliteration, did the obvious next thing, and got a
twelve-frame traceback:

    senbonzakura check abl06/      ->  exit 1
    TypeError: unhashable type: 'dict'   in adapters/senbonzakura_log.py, _agreed

`--skip-unknown`, the documented escape hatch for files the checker does not recognise, did not
help, because the failure was not in recognising a file. Every single file in that directory
checked cleanly when named on its own; only the directory sweep died.

The cause was a set comprehension. `_agreed` asks whether every stamped metric agrees on a field,
and it built a set of the values to find out. It guarded that the metric BLOCK was a dict and never
that the block's VALUE was hashable, so a `prompt_format` recorded as an object rather than a string
raised out of the set literal. Validated one thing; the thing that needed it was one level down.

WHY THIS PROGRAM IN PARTICULAR

A traceback is the worst failure available to this tool. Its whole value is that it never reports a
result clean unless it can stand behind it, and it is documented as safe to point at other people's
files. A reader who meets a stack trace cannot tell a bug from a finding, which is the one
distinction the program exists to make. `SECURITY.md` already commits to refusing rather than
crashing on a hostile file; crashing on OUR OWN output is worse than crashing on a crafted one.

These tests do not check the field is interpreted correctly. They check that no shape of it can stop
the checker running.
"""
from pathlib import Path

import pytest

ADAPTER = (Path(__file__).resolve().parent.parent / "checker" / "src" / "senbonzakura_check"
           / "adapters" / "senbonzakura_log.py")


@pytest.fixture(scope="module")
def agreed():
    """`_agreed` lifted out by source, so this needs neither the installed checker nor its imports.

    The helper is self-contained. Importing the module would drag the registry in and make a unit
    test of six lines depend on the whole package resolving.
    """
    source = ADAPTER.read_text(encoding="utf-8")
    start = source.index("def _agreed")
    end = source.index("def _older_shapes")
    namespace = {}
    exec(compile(source[start:end], str(ADAPTER), "exec"), namespace)  # noqa: S102
    return namespace["_agreed"]


@pytest.mark.parametrize("value", [
    pytest.param({"template": "chatml", "variant": 2}, id="a-dict"),
    pytest.param(["a", "b"], id="a-list"),
    pytest.param({"nested": {"deeper": [1, 2]}}, id="a-nested-dict"),
    pytest.param("chatml", id="a-string"),
    pytest.param(7, id="a-number"),
    pytest.param(True, id="a-bool"),
])
def test_no_field_shape_can_raise(agreed, value):
    """The regression itself: an unhashable value used to raise out of a set literal."""
    agreed({"refusal": {"prompt_format": value}}, "prompt_format")


def test_one_agreeing_value_is_returned(agreed):
    assert agreed({"a": {"prompt_format": "chatml"}, "b": {"prompt_format": "chatml"}},
                  "prompt_format") == "chatml"


def test_agreeing_unhashable_values_are_returned(agreed):
    """The fix must not have bought survival by giving up the answer."""
    assert agreed({"a": {"prompt_format": {"t": 1}}, "b": {"prompt_format": {"t": 1}}},
                  "prompt_format") == {"t": 1}


@pytest.mark.parametrize("blocks", [
    pytest.param({"a": {"prompt_format": "chatml"}, "b": {"prompt_format": "raw"}}, id="strings"),
    pytest.param({"a": {"prompt_format": {"t": 1}}, "b": {"prompt_format": {"t": 2}}}, id="dicts"),
])
def test_disagreement_still_answers_none(agreed, blocks):
    """DISAGREEMENT IS NOT A TIE TO BREAK, and the docstring on the helper says why.

    A document whose metrics were rendered in two prompt formats has no single prompt format.
    Returning either hands a check a fact about half the file while looking like a fact about the
    file, and the check then reports clean on a question it never asked. The equality rewrite had to
    preserve this, and for dicts it did not previously get the chance to.
    """
    assert agreed(blocks, "prompt_format") is None


@pytest.mark.parametrize("blocks", [
    pytest.param({}, id="no-metrics"),
    pytest.param({"a": {}}, id="field-absent"),
    pytest.param({"a": {"prompt_format": None}}, id="field-null"),
    pytest.param({"a": "not a dict at all"}, id="block-is-a-string"),
    pytest.param({"a": None}, id="block-is-null"),
])
def test_nothing_to_agree_on_answers_none(agreed, blocks):
    assert agreed(blocks, "prompt_format") is None
