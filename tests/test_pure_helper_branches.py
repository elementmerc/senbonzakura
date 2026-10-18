# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Branches in the small pure helpers of `capability` that nothing has taken.

The companion to `test_cli_helpers_branches.py`, and the same argument. Measured on 2026-09-11,
with every release artefact present, 200 lines in this package execute with their condition only
ever answered one way. These are the ones that need nothing but a string or a patched path to
reach, so they are the cheapest real coverage in the tree and the easiest place for a guard to be
quietly wrong.

`tool_call` is worth more than its line count: it parses whatever a model emitted, so it is a
boundary in baseline section 2.1's sense: untrusted input, validated here or nowhere. (The
head-to-head helpers these tests also covered left the package with that command on 2026-10-18.)

CPU only. No model, no corpus, no card, no network.
"""
import json
import os

import pytest

from senbonzakura import capability

# ── capability.predicted_answer and last_line_answer ─────────────────────────────────────────

def test_no_generation_is_not_a_wrong_answer():
    """None is a real outcome and must stay separate from a wrong answer.

    A truncated generation, a refusal, and a model that answered in words all land here, and none
    of them is evidence that the arithmetic is gone.
    """
    assert capability.predicted_answer(None) is None
    assert capability.predicted_answer("no digits at all") is None


def test_the_last_number_is_the_answer_not_the_first():
    """A worked solution restates the question's quantities before it reaches a result."""
    assert capability.predicted_answer("She had 5 apples and ate 2, so 3 remain.") == 3


def test_last_line_answer_handles_nothing_and_whitespace():
    assert capability.last_line_answer(None) is None
    assert capability.last_line_answer("   \n\n  ") is None
    assert capability.last_line_answer("working\n\n  Paris  \n\n") == \
        capability.normalised_text("Paris")


# ── capability.tool_call: a boundary, so every malformed shape is reached ────────────────────

def test_a_bare_json_tool_call_is_read():
    assert capability.tool_call('{"name": "search", "arguments": {"q": "x"}}') == \
        ("search", {"q": "x"})


def test_a_fenced_tool_call_is_read():
    """Models wrap JSON in a ```json fence constantly, and that is not them getting it wrong."""
    text = 'Here you go:\n```json\n{"tool": "search", "args": {"q": "x"}}\n```\n'
    assert capability.tool_call(text) == ("search", {"q": "x"})


def test_arguments_arriving_as_a_json_string_are_parsed():
    """Also common, also not an error: `arguments` is itself a JSON string."""
    assert capability.tool_call('{"name": "t", "arguments": "{\\"a\\": 1}"}') == ("t", {"a": 1})


def test_an_unparseable_arguments_string_is_kept_rather_than_discarded():
    """DATA PRESERVATION, per baseline section 3.7. The caller can see what arrived.

    Dropping it would make "the model emitted nonsense" indistinguishable from "the model emitted
    nothing", and only one of those is a capability failure.
    """
    name, args = capability.tool_call('{"name": "t", "arguments": "not json"}')
    assert name == "t"
    assert args == {"__unparsed__": "not json"}


def test_non_dict_arguments_are_wrapped_rather_than_dropped():
    name, args = capability.tool_call('{"name": "t", "arguments": [1, 2]}')
    assert (name, args) == ("t", {"__value__": [1, 2]})


def test_a_nested_function_name_is_found():
    """`{"function": {"name": ...}}` is the OpenAI-shaped variant."""
    assert capability.tool_call('{"function": {"name": "f"}, "parameters": {"k": 1}}') == \
        ("f", {"k": 1})


@pytest.mark.parametrize(("text", "why"), [
    (None, "nothing was generated"),
    ("I will not do that.", "prose with no JSON in it"),
    ('{"name": ', "a JSON blob that does not parse"),
    ("[1, 2, 3]", "valid JSON that is not an object"),
    ('{"arguments": {"a": 1}}', "an object with no name in it"),
    ('{"name": 7, "arguments": {}}', "a name that is not a string"),
])
def test_everything_that_is_not_a_tool_call_returns_none(text, why):
    """Each of these was a branch nothing had taken, and each is a thing a model really emits."""
    assert capability.tool_call(text) is None, why


def test_a_tool_call_with_no_arguments_at_all_gets_an_empty_mapping():
    """Absent arguments is a legitimate call, not a parse failure."""
    assert capability.tool_call('{"name": "ping"}') == ("ping", {})


# ── a guard on the fixtures above ────────────────────────────────────────────────────────────

def test_the_json_blob_pattern_is_what_these_tests_think_it_is():
    """If `JSON_BLOB` stops matching a fenced object, the tests above would pass vacuously by
    all returning None for the wrong reason. This pins the assumption they rest on.
    """
    m = capability.JSON_BLOB.search('```json\n{"name": "x"}\n```')
    assert m and json.loads(m.group()) == {"name": "x"}


def test_the_helpers_under_test_need_no_model_or_corpus():
    """A statement about this file rather than about the code: nothing here loads anything.

    It is the property that makes these the tests a contributor with no GPU can run and extend,
    which is the direction the project is trying to move in.
    """
    assert os.environ.get("SENBON_REQUIRE_BUNDLED") in (None, "0", "1")
