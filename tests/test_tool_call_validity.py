# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The mechanical half of tool-call measurement, and the distinctions it must not collapse.

WHY THIS FILE IS MOSTLY ABOUT DISTINCTIONS RATHER THAN COUNTS

The roadmap had this measure gated on labelled data, and that gate is real for one half of it only.
"Was calling that tool the right move, with the right values" needs a certified grader. "Is this a
syntactically valid call, naming a function on offer, with its required arguments present and of
the declared types" needs nothing but the schema, which the harness already has because it is what
went into the prompt.

So the value of this measure is entirely in telling failures apart. A single "invalid" rate would
be worth very little: a model answering in prose, a model emitting a call cut off mid-object, a
model inventing a function and a model passing a string where an integer was declared need four
different fixes, and a tool that reports one number for all four has told its user nothing they
can act on. Every test below asserts on WHICH failure was found, not only that one was.

THE TRAP, which is the same one the rest of `capability.py` is built around

A reply that could not be graded must not be scored as a failure. A call truncated by the token
budget looks exactly like malformed JSON in the text, so if truncation counted as invalid, setting
`--max-new` too low would read as a tool-use regression, in the same direction as the claim the
measure exists to test.
"""
import json

import pytest

from senbonzakura import capability as cap

# A schema in the shape a harness puts in a prompt: one required string, one optional integer.
WEATHER = {
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "verbose": {"type": "boolean"},
        },
        "required": ["city"],
    },
}

#: The same tool as OpenAI writes it, to prove the reader is not coupled to one provider's wrapper.
WEATHER_WRAPPED = {"type": "function", "function": WEATHER}

#: The same tool flattened, which is how a terse fixture or a hand-written spec tends to look.
WEATHER_FLAT = {
    "name": "get_weather",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city"],
}

GOOD = '{"name": "get_weather", "arguments": {"city": "Leeds"}}'


# ── did it emit a call at all, and if not, which way did it fail ─────────────────────

@pytest.mark.parametrize(("reply", "why"), [
    ("", cap.NO_CALL_NO_JSON),
    (None, cap.NO_CALL_NO_JSON),
    ("I'd check the weather in Leeds if I were you.", cap.NO_CALL_NO_JSON),
    # Cut off mid-object. Object-shaped, so it is NOT the same failure as answering in prose.
    ('{"name": "get_weather", "arguments": {"city": "Lee', cap.NO_CALL_UNPARSEABLE),
    # Trailing comma: the single most common way a model's JSON fails to parse.
    ('{"name": "get_weather", "arguments": {"city": "Leeds",}}', cap.NO_CALL_UNPARSEABLE),
    # Parses perfectly and is simply not a call.
    ('{"result": 42}', cap.NO_CALL_NAMES_NOTHING),
    # A name that is not a string names no function.
    ('{"name": 7, "arguments": {}}', cap.NO_CALL_NAMES_NOTHING),
    # A JSON array carries nothing object-shaped, so it lands with the prose rather than with the
    # replies that tried to emit a call and got the syntax wrong.
    ("[1, 2, 3]", cap.NO_CALL_NO_JSON),
])
def test_the_three_ways_there_is_no_call_stay_apart(reply, why):
    call, got = cap.emitted_tool_call(reply)
    assert call is None
    assert got == why
    assert cap.tool_call_validity(reply, WEATHER)["no_call_because"] == why


def test_prose_and_malformed_json_are_not_the_same_failure():
    """The distinction the whole measure rests on, asserted directly rather than by code name."""
    prose = cap.tool_call_validity("You should look that up.", WEATHER)
    broken = cap.tool_call_validity('{"name": "get_weather", "arguments": {', WEATHER)
    assert prose["no_call_because"] != broken["no_call_because"]
    # Both are failures of the thing being measured, and neither is ungradeable. A model asked for
    # a call that answers in prose has failed; calling that indeterminate would hide the most
    # common way tool use breaks.
    assert prose["valid"] is False
    assert broken["valid"] is False
    assert prose["indeterminate"] is False
    assert broken["indeterminate"] is False


@pytest.mark.parametrize("reply", [
    GOOD,
    f"Sure, I'll look that up.\n```json\n{GOOD}\n```",
    f"```\n{GOOD}\n```",
    '{"tool": "get_weather", "args": {"city": "Leeds"}}',
    '{"function": "get_weather", "parameters": {"city": "Leeds"}}',
    # `arguments` as a JSON string, which is common and is not the model getting it wrong.
    '{"name": "get_weather", "arguments": "{\\"city\\": \\"Leeds\\"}"}',
])
def test_the_shapes_models_actually_emit_all_count_as_a_call(reply):
    report = cap.tool_call_validity(reply, WEATHER)
    assert report["emitted_call"] is True
    assert report["name"] == "get_weather"
    assert report["valid"] is True


def test_the_last_balanced_object_wins_so_reasoning_aloud_does_not_hide_the_call():
    """A model that thinks in prose with braces in it, then calls. The call is at the END.

    This is the case the regex-based reader in `tool_call` gets wrong, and it matters here because
    a reasoning model is exactly the kind that writes braces before its call.
    """
    reply = f"First I should use {{the weather tool}} for this. {GOOD}"
    assert cap.emitted_tool_call(reply)[0] == ("get_weather", {"city": "Leeds"})
    assert cap.tool_call_validity(reply, WEATHER)["valid"] is True


def test_a_quotation_mark_in_the_prose_does_not_swallow_the_call():
    reply = f'He said "check the forecast" so here it is. {GOOD}'
    assert cap.tool_call_validity(reply, WEATHER)["valid"] is True


# ── is the named function one the schema offers ──────────────────────────────────────

def test_an_invented_function_is_its_own_failure_and_not_an_argument_failure():
    report = cap.tool_call_validity(
        '{"name": "get_forecast", "arguments": {"city": "Leeds"}}', WEATHER)
    assert report["emitted_call"] is True
    assert report["name"] == "get_forecast"
    assert report["known_function"] is False
    assert report["valid"] is False
    # The argument checks are WITHHELD, not answered. There is no schema for a function the schema
    # does not offer, so an empty missing-argument list would read as "all arguments present".
    assert report["missing_required"] is None
    assert report["wrong_typed"] is None
    assert report["undeclared"] is None


def test_a_function_is_known_if_any_offered_tool_names_it():
    box = [WEATHER, {"name": "send_email",
                     "parameters": {"properties": {"to": {"type": "string"}},
                                    "required": ["to"]}}]
    assert cap.tool_call_validity(GOOD, box)["known_function"] is True
    second = cap.tool_call_validity('{"name": "send_email", "arguments": {"to": "a@b.c"}}', box)
    assert second["valid"] is True


@pytest.mark.parametrize("schema", [WEATHER, WEATHER_WRAPPED, WEATHER_FLAT])
def test_the_provider_wrappers_read_the_same_tool(schema):
    assert cap.tool_schema(schema)[0] == "get_weather"
    assert cap.tool_schema(schema)[2] == ["city"]
    assert cap.tool_call_validity(GOOD, schema)["valid"] is True


@pytest.mark.parametrize("schema", [
    None,
    "get_weather",
    {"name": ""},
    {"name": 7},
    {"parameters": {"properties": {}}},
    {"name": "f", "parameters": 5},                      # parameters is not an object
    {"name": "f", "parameters": {"properties": 5}},      # properties is not an object
])
def test_a_schema_the_reader_cannot_use_is_rejected_rather_than_half_read(schema):
    """Half-reading a malformed schema is how a harness bug becomes a model failure.

    Each of these returns None, which `tool_call_validity` turns into an indeterminate item. The
    alternative, reading what parts it can and checking against those, would grade the model
    against a parameter list that is not the one it was shown.
    """
    assert cap.tool_schema(schema) is None


def test_required_entries_that_are_not_parameter_names_are_dropped():
    schema = {"name": "f", "parameters": {"properties": {"x": {"type": "string"}},
                                          "required": ["x", 7, None]}}
    assert cap.tool_schema(schema)[2] == ["x"]
    schema["parameters"]["required"] = "x"               # a string where a list was expected
    assert cap.tool_schema(schema)[2] == []


def test_a_parsed_payload_that_is_not_an_object_names_no_function():
    """The reader's own guard, exercised directly because no reply text can reach it.

    Both callers only ever hand it a substring starting with a brace, so a list or a bare number
    cannot arrive by that route. The guard stays because the helper is shared and the next caller
    may not have that property, and this test is what stops it being silently deleted as dead.
    """
    assert cap._call_from_object([1, 2]) is None
    assert cap._call_from_object(7) is None
    assert cap._call_from_object(None) is None


# ── required, mistyped and invented arguments, kept apart ────────────────────────────

def test_a_missing_required_argument_is_named_not_merely_counted():
    report = cap.tool_call_validity(
        '{"name": "get_weather", "arguments": {"days": 3}}', WEATHER)
    assert report["missing_required"] == ["city"]
    assert report["undeclared"] == []
    assert report["wrong_typed"] == []
    assert report["valid"] is False


def test_an_optional_argument_left_out_is_not_a_failure():
    report = cap.tool_call_validity(GOOD, WEATHER)
    assert report["missing_required"] == []
    assert report["valid"] is True


@pytest.mark.parametrize(("value", "declared", "got"), [
    ('"three"', "integer", "str"),
    ("3.5", "integer", "float"),
    ("[3]", "integer", "list"),
    ("null", "integer", "NoneType"),
    # THE ONE PYTHON HIDES: `isinstance(True, int)` is True, so a bool passed where an integer was
    # declared would be scored as correctly typed by a naive check.
    ("true", "integer", "bool"),
])
def test_a_wrong_typed_argument_reports_what_was_declared_and_what_arrived(value, declared, got):
    reply = f'{{"name": "get_weather", "arguments": {{"city": "Leeds", "days": {value}}}}}'
    report = cap.tool_call_validity(reply, WEATHER)
    assert report["wrong_typed"] == [{"argument": "days", "declared": declared, "got": got}]
    assert report["missing_required"] == []
    assert report["undeclared"] == []
    assert report["valid"] is False


def test_an_integer_is_admitted_where_a_number_was_declared():
    schema = {"name": "f", "parameters": {"properties": {"x": {"type": "number"}},
                                          "required": ["x"]}}
    assert cap.tool_call_validity('{"name": "f", "arguments": {"x": 3}}', schema)["valid"] is True
    assert cap.tool_call_validity('{"name": "f", "arguments": {"x": 3.5}}', schema)["valid"] is True
    # Still not a bool, for the same reason as above.
    boolean = cap.tool_call_validity('{"name": "f", "arguments": {"x": true}}', schema)
    assert boolean["wrong_typed"][0]["got"] == "bool"


def test_a_union_type_admits_either_member():
    schema = {"name": "f", "parameters": {"properties": {"x": {"type": ["string", "null"]}},
                                          "required": ["x"]}}
    assert cap.tool_call_validity('{"name": "f", "arguments": {"x": "a"}}', schema)["valid"] is True
    assert cap.tool_call_validity('{"name": "f", "arguments": {"x": null}}', schema)["valid"] is True
    assert cap.tool_call_validity('{"name": "f", "arguments": {"x": 1}}', schema)["valid"] is False


@pytest.mark.parametrize("spec", [{}, {"type": "colour"}, {"type": 7}, "not-a-spec"])
def test_an_undeclared_or_unknown_type_is_never_called_a_mismatch(spec):
    """A schema keyword this reader does not know is not a licence to fail the model.

    Reporting a mismatch against a type nobody can check would turn our own gap into a capability
    loss, which is the error this module exists to avoid making.
    """
    schema = {"name": "f", "parameters": {"properties": {"x": spec}, "required": ["x"]}}
    assert cap.tool_call_validity('{"name": "f", "arguments": {"x": 1}}', schema)["valid"] is True


def test_an_undeclared_argument_is_a_different_failure_from_a_missing_one():
    extra = cap.tool_call_validity(
        '{"name": "get_weather", "arguments": {"city": "Leeds", "units": "C"}}', WEATHER)
    missing = cap.tool_call_validity(
        '{"name": "get_weather", "arguments": {"days": 1}}', WEATHER)
    assert extra["undeclared"] == ["units"]
    assert extra["missing_required"] == []
    assert missing["undeclared"] == []
    assert missing["missing_required"] == ["city"]


def test_both_halves_of_a_double_failure_are_reported():
    report = cap.tool_call_validity(
        '{"name": "get_weather", "arguments": {"units": "C", "days": "two"}}', WEATHER)
    assert report["missing_required"] == ["city"]
    assert report["undeclared"] == ["units"]
    assert report["wrong_typed"] == [{"argument": "days", "declared": "integer", "got": "str"}]


def test_a_zero_parameter_tool_is_usable_and_any_argument_is_undeclared():
    schema = {"name": "ping", "parameters": {"type": "object", "properties": {}}}
    assert cap.tool_call_validity('{"name": "ping", "arguments": {}}', schema)["valid"] is True
    noisy = cap.tool_call_validity('{"name": "ping", "arguments": {"x": 1}}', schema)
    assert noisy["undeclared"] == ["x"]
    assert noisy["valid"] is False


def test_malformed_arguments_are_not_reported_as_an_invented_parameter():
    """An `arguments` string that is not JSON, and one that parses to the wrong kind of thing.

    The reader stores these under sentinel keys so the failure survives. Letting those keys reach
    the undeclared-argument list would describe a malformed payload as a hallucinated parameter.
    """
    unparsed = cap.tool_call_validity(
        '{"name": "get_weather", "arguments": "city=Leeds"}', WEATHER)
    assert unparsed["arguments_malformed"] == ["__unparsed__"]
    assert unparsed["undeclared"] is None
    assert unparsed["missing_required"] is None
    assert unparsed["valid"] is False

    listed = cap.tool_call_validity(
        '{"name": "get_weather", "arguments": ["Leeds"]}', WEATHER)
    assert listed["arguments_malformed"] == ["__value__"]
    assert listed["valid"] is False


# ── the indeterminate-versus-wrong discipline ────────────────────────────────────────

def test_truncation_is_indeterminate_and_not_invalid():
    cut = '{"name": "get_weather", "arguments": {"city": "Lee'
    scored = cap.tool_call_validity(cut, WEATHER, truncated=False)
    held = cap.tool_call_validity(cut, WEATHER, truncated=True)
    assert scored["valid"] is False
    assert held["indeterminate"] is True
    # `None`, not False. A token budget running out is not a statement about the model, and a False
    # here would be counted in the valid-rate denominator as a failure.
    assert held["valid"] is None
    assert held["indeterminate_because"]


@pytest.mark.parametrize("schema", [None, {}, [], "get_weather", {"parameters": {}},
                                    {"name": ""}, [{"name": None}]])
def test_an_unusable_schema_makes_the_item_indeterminate_not_failed(schema):
    """Our own defect must never be scored against the model. `grade_one` says the same of a
    corpus row it cannot read.
    """
    report = cap.tool_call_validity(GOOD, schema)
    assert report["indeterminate"] is True
    assert report["valid"] is None
    assert "schema" in report["indeterminate_because"]


def test_every_record_carries_the_mechanical_caveat():
    for record in (cap.tool_call_validity(GOOD, WEATHER),
                   cap.tool_call_validity("prose", WEATHER),
                   cap.tool_call_validity(GOOD, WEATHER, truncated=True),
                   cap.tool_call_validity_block([GOOD], WEATHER)):
        assert "NOT whether" in record["measures"]


# ── the aggregate ────────────────────────────────────────────────────────────────────

def test_no_replies_is_none_rather_than_a_block_of_zeroes():
    assert cap.tool_call_validity_block([], WEATHER) is None


def test_the_aggregate_keeps_the_failure_kinds_apart_over_a_mix():
    replies = [
        GOOD,                                                                   # valid
        f"Here you go.\n```json\n{GOOD}\n```",                                  # valid, fenced
        '{"name": "get_weather", "arguments": "{\\"city\\": \\"Hull\\"}"}',      # valid, arg string
        "I'd just look out of the window.",                                     # no json
        '{"name": "get_weather", "arguments": {"city": ',                        # unparseable
        '{"answer": "Leeds"}',                                                  # names nothing
        '{"name": "get_forecast", "arguments": {"city": "Leeds"}}',             # invented
        '{"name": "get_weather", "arguments": {"days": 2}}',                    # missing required
        '{"name": "get_weather", "arguments": {"city": "Leeds", "days": "two"}}',  # wrong type
        '{"name": "get_weather", "arguments": {"city": "Leeds", "units": "C"}}',   # undeclared
    ]
    block = cap.tool_call_validity_block(replies, WEATHER)

    assert block["n"] == 10
    assert block["indeterminate"] == 0
    assert block["graded"] == 10

    # Three different reasons there was no call, each counted as itself.
    assert block["no_call_because"] == {
        cap.NO_CALL_NO_JSON: 1,
        cap.NO_CALL_UNPARSEABLE: 1,
        cap.NO_CALL_NAMES_NOTHING: 1,
    }

    # Seven replies emitted a call; six of them named a function on offer.
    assert block["emitted_call"]["count"] == 7
    assert block["emitted_call"]["n"] == 10
    assert block["invented_function"]["count"] == 1
    assert block["invented_function"]["n"] == 7
    assert block["checkable"] == 6

    # THE DENOMINATORS DIFFER AND THAT IS THE POINT: the argument checks are over the six calls
    # naming a real function, not over all ten replies.
    assert block["valid"]["count"] == 3
    assert block["valid"]["n"] == 10
    for key, count in (("missing_required", 1), ("wrong_typed", 1), ("undeclared", 1),
                       ("arguments_malformed", 0)):
        assert block[key]["count"] == count, key
        assert block[key]["n"] == 6, key

    assert len(block["reports"]) == 10


def test_a_small_sample_withholds_the_rate_and_gives_the_counts():
    """Ten replies is below the reportable floor, so no rate is stated. The counts still are."""
    block = cap.tool_call_validity_block([GOOD] * 10, WEATHER)
    assert block["valid"]["count"] == 10
    assert block["valid"]["rate"] is None
    assert block["valid"]["reportable"] is False
    assert block["valid"]["why_not"]


def test_a_sample_over_the_floor_states_the_rate():
    replies = [GOOD] * 30 + ["no call here"] * 10
    block = cap.tool_call_validity_block(replies, WEATHER)
    assert block["valid"]["n"] == 40
    assert block["valid"]["rate"] == pytest.approx(0.75)
    assert block["valid"]["reportable"] is True
    assert block["valid"]["ci"]


def test_the_argument_rates_are_withheld_when_nothing_was_checkable():
    """Every reply answered in prose, so there is no set the argument checks are defined over.

    A zero here would say "no arguments were missing", which is true the way an empty list is true
    and reads as a pass.
    """
    block = cap.tool_call_validity_block(["prose"] * 5, WEATHER)
    assert block["checkable"] == 0
    for key in ("missing_required", "wrong_typed", "undeclared", "arguments_malformed"):
        assert block[key]["n"] == 0
        assert block[key]["rate"] is None


def test_truncated_replies_leave_the_graded_denominator_and_raise_the_ceiling_flag():
    replies = [GOOD] * 9 + ['{"name": "get_weather", "arguments": {']
    flags = [False] * 9 + [True]
    block = cap.tool_call_validity_block(replies, WEATHER, truncated=flags)
    assert block["indeterminate"] == 1
    assert block["graded"] == 9
    assert block["valid"]["count"] == 9
    assert block["valid"]["n"] == 9
    # One in ten is not over a ten percent ceiling; two in ten is.
    assert block["budget_suspect"] is False
    worse = cap.tool_call_validity_block(replies, WEATHER, truncated=[False] * 8 + [True, True])
    assert worse["budget_suspect"] is True
    assert worse["budget_threshold"] == cap.MAX_INDETERMINATE


def test_a_truncated_flag_list_of_the_wrong_length_is_a_loud_failure():
    """Silently zipping to the shorter of the two would drop replies from the denominator."""
    with pytest.raises(ValueError, match="shorter"):
        cap.tool_call_validity_block([GOOD, GOOD], WEATHER, truncated=[False])


def test_a_model_that_always_calls_the_wrong_tool_still_scores_perfectly():
    """The measure's own blind spot, asserted so that nobody can read the figure as correctness.

    Both tools are on offer, so every one of these calls is mechanically perfect. Whether the
    weather tool was the right answer to an email request is exactly the question this does not
    ask, and the test exists so that a future reader meets that fact in the suite.
    """
    box = [WEATHER, {"name": "send_email",
                     "parameters": {"properties": {"to": {"type": "string"}},
                                    "required": ["to"]}}]
    block = cap.tool_call_validity_block([GOOD] * 40, box)
    assert block["valid"]["count"] == 40
    assert block["valid"]["rate"] == 1.0
    assert "NOT whether" in block["measures"]


# ── the refactor underneath, which must not have moved the graded task ───────────────

@pytest.mark.parametrize(("reply", "want"), [
    (GOOD, ("get_weather", {"city": "Leeds"})),
    ('{"tool": "get_weather", "args": {"city": "Leeds"}}', ("get_weather", {"city": "Leeds"})),
    ('{"function": "get_weather", "parameters": {"city": "Leeds"}}',
     ("get_weather", {"city": "Leeds"})),
    ('{"function": {"name": "get_weather"}}', ("get_weather", {})),
    ('{"name": "get_weather", "arguments": "{\\"city\\": \\"Leeds\\"}"}',
     ("get_weather", {"city": "Leeds"})),
    ('{"name": "get_weather", "arguments": "city=Leeds"}',
     ("get_weather", {"__unparsed__": "city=Leeds"})),
    ('{"name": "get_weather", "arguments": ["Leeds"]}',
     ("get_weather", {"__value__": ["Leeds"]})),
    ('{"name": "get_weather"}', ("get_weather", {})),
    ("no json at all", None),
    ('{"result": 1}', None),
    ("[1, 2]", None),
    (None, None),
    # FIXED 2026-10-08, and this row used to pin the defect. The regex took everything between the
    # FIRST brace and the LAST, so a call preceded by braces in the model's reasoning did not
    # parse and, because the task sets `unparseable_is_wrong=True`, scored WRONG rather than
    # unparsed. The bias ran the flattering way, exactly as the `choice_answer` defect above does:
    # a safety-tuned model reasons aloud and writes braces, an abliterated one is terser, so the
    # stock arm collected more false failures and abliteration would have read as IMPROVING tool
    # use. `tool_call` now scans for balanced objects and takes the last one that parses.
    ('I will use {the tool}. {"name": "get_weather", "arguments": {"city": "Leeds"}}',
     ("get_weather", {"city": "Leeds"})),
    # The same defect from the other side: the call must still be found when the prose is clean.
    ('Let me check the weather. {"name": "get_weather", "arguments": {"city": "Leeds"}}',
     ("get_weather", {"city": "Leeds"})),
    # FIXED 2026-10-08. OpenAI writes the arguments BESIDE the name inside the nested object, and
    # the lookup only ever checked the top level, so every argument was silently dropped and the
    # item scored wrong on a correct call.
    ('{"function": {"name": "get_weather", "arguments": {"city": "Leeds"}}}',
     ("get_weather", {"city": "Leeds"})),
    # THE SHAPE THE FIRST ATTEMPT AT THAT FIX BROKE, caught by an existing test in
    # `test_pure_helper_branches.py` and pinned here too so the next person meets it in the same
    # table. The name is nested and the arguments are OUTSIDE it, so taking the nested object
    # wholesale finds no arguments. The nested object is preferred and the outer one is a
    # fallback, which is not the same thing as a replacement.
    ('{"function": {"name": "get_weather"}, "parameters": {"city": "Leeds"}}',
     ("get_weather", {"city": "Leeds"})),
])
def test_the_reference_graded_extractor_is_unchanged(reply, want):
    """`tool_call` grades against a gold call, so every shape it reads is pinned here.

    THE TWO FIXED ROWS ARE DELIBERATE CHANGES, not drift. Checked before changing them: `tool-call`
    is one of five GRADING RULES and the only dataset this package ships is a 256 row GSM8K subset
    graded `numeric`, so no published figure was taken through this extractor and no re-run is
    owed. `docs/comparison.md` is exact about that distinction.
    """
    assert cap.tool_call(reply) == want


# ── offering the tools, which is what made the measures above reachable ───────────────
#
# ADDED 2026-10-08. `tool_call_validity_block` was written, tested, and CALLED BY NOTHING. These
# cover the wiring that fixed that: the loader, the prompt preamble, the renderer and the exam
# fingerprint. The defect class is this project's most repeated one, so the tests here are about
# the join between the pieces rather than about any one of them.

TWO_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"]}}},
    {"name": "send_email", "parameters": {"type": "object", "properties": {}}},
]


def _schema_file(tmp_path, payload):
    p = tmp_path / "tools.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_a_provider_request_body_is_accepted_as_it_is_found(tmp_path):
    """The shape somebody already has, rather than one they have to retype."""
    items, box = cap.load_tool_schema(_schema_file(tmp_path, {"tools": TWO_TOOLS}))
    assert sorted(box) == ["get_weather", "send_email"]
    assert items == TWO_TOOLS


def test_a_bare_list_and_a_single_object_both_load(tmp_path):
    _, box = cap.load_tool_schema(_schema_file(tmp_path, TWO_TOOLS))
    assert sorted(box) == ["get_weather", "send_email"]
    _, one = cap.load_tool_schema(_schema_file(tmp_path, TWO_TOOLS[0]))
    assert sorted(one) == ["get_weather"]


@pytest.mark.parametrize(("payload", "fragment"), [
    ({"tools": []}, "no usable tool declaration"),
    ([{"parameters": {"type": "object"}}], "no usable tool declaration"),
    ({"name": ""}, "no usable tool declaration"),
])
def test_an_unusable_schema_is_refused_before_a_model_loads(tmp_path, payload, fragment):
    """REFUSED, NOT RUN. Every item would be ungradeable against a schema that names no function,
    and a generation pass over hundreds of prompts spent to discover a typo in a path is the cost
    this refusal exists to avoid.
    """
    with pytest.raises(SystemExit) as e:
        cap.load_tool_schema(_schema_file(tmp_path, payload))
    assert fragment in str(e.value)


def test_unreadable_and_unparseable_files_name_which_one_it_was(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cap.load_tool_schema(str(bad))
    assert "not valid JSON" in str(e.value)

    with pytest.raises(SystemExit) as e:
        cap.load_tool_schema(str(tmp_path / "absent.json"))
    assert "cannot be read" in str(e.value)


def test_the_offer_names_every_tool_and_marks_what_is_required(tmp_path):
    _, box = cap.load_tool_schema(_schema_file(tmp_path, TWO_TOOLS))
    offered = cap.offered_tools(box)
    assert "get_weather: city (string) required, days (integer) optional." in offered
    assert "send_email: takes no arguments." in offered


def test_the_offer_is_deterministic_whatever_order_the_tools_arrive_in(tmp_path):
    """IT GOES INTO THE EXAM FINGERPRINT. A dictionary's iteration order leaking in here would
    make two identical runs look like two different exams and refuse to pair them.
    """
    forward = cap.offered_tools(cap.load_tool_schema(
        _schema_file(tmp_path, TWO_TOOLS))[1])
    backward = cap.offered_tools(cap.load_tool_schema(
        _schema_file(tmp_path, list(reversed(TWO_TOOLS))))[1])
    assert forward == backward


def test_no_tools_offers_nothing_rather_than_an_empty_heading():
    assert cap.offered_tools({}) == ""


def test_a_union_type_is_shown_as_a_union(tmp_path):
    schema = {"name": "f", "parameters": {"type": "object", "properties": {
        "x": {"type": ["string", "integer"]}}, "required": ["x"]}}
    _, box = cap.load_tool_schema(_schema_file(tmp_path, schema))
    assert "x (string or integer) required" in cap.offered_tools(box)


def test_a_parameter_with_no_declared_type_is_named_without_inventing_one(tmp_path):
    schema = {"name": "f", "parameters": {"type": "object", "properties": {"x": {}}}}
    _, box = cap.load_tool_schema(_schema_file(tmp_path, schema))
    assert "- f: x optional." in cap.offered_tools(box)


def test_the_fingerprint_is_unchanged_when_no_tools_were_offered():
    """Every artefact already on disk was written without this argument, so the no-tool digest has
    to be byte-identical or every existing paired comparison refuses itself.
    """
    questions = ["one", "two"]
    assert cap.items_digest(questions) == cap.items_digest(questions, "")


def test_two_runs_offered_different_tools_are_different_exams(tmp_path):
    """WITHOUT THIS they pair item by item and report the difference as a change in capability."""
    questions = ["one", "two"]
    _, both = cap.load_tool_schema(_schema_file(tmp_path, TWO_TOOLS))
    _, one = cap.load_tool_schema(_schema_file(tmp_path, TWO_TOOLS[:1]))
    assert (cap.items_digest(questions, cap.offered_tools(both))
            != cap.items_digest(questions, cap.offered_tools(one)))
    assert cap.items_digest(questions, cap.offered_tools(both)) != cap.items_digest(questions)


def test_the_toolbox_cannot_collide_with_one_more_question():
    """The separator earns its place: without it an exam of three questions and an exam of two
    plus a toolbox spelled the same way would hash alike.
    """
    assert cap.items_digest(["a", "b"], "c") != cap.items_digest(["a", "b", "c"])


def test_the_report_prints_a_withheld_rate_as_withheld(tmp_path):
    """NEVER AS ZERO. `reportable_rate` returns None under the floor and a renderer that formats
    that as 0.0% tells a reader the model never once passed an undeclared argument, when what
    happened is that too few replies could be checked to say.
    """
    items, _ = cap.load_tool_schema(_schema_file(tmp_path, TWO_TOOLS))
    block = cap.tool_call_validity_block(
        ['{"name": "get_weather", "arguments": {"city": "Leeds"}}'] * 3, items)
    out = "\n".join(cap.tool_validity_report(block))
    assert "0.0%" not in out
    assert "below the floor" in out
    assert "and no rate" in out


def test_the_report_gives_a_rate_with_its_counts_and_interval_once_there_are_enough(tmp_path):
    items, _ = cap.load_tool_schema(_schema_file(tmp_path, TWO_TOOLS))
    replies = (['{"name": "get_weather", "arguments": {"city": "Leeds"}}'] * 30
               + ["no call here at all"] * 10)
    out = "\n".join(cap.tool_validity_report(cap.tool_call_validity_block(replies, items)))
    assert "emitted a call at all: 75.0% (30/40" in out
    assert "95% CI" in out
    assert "no call because: no_json 10" in out


def test_the_report_says_the_measures_are_mechanical_in_its_own_output(tmp_path):
    """The caveat travels with the figures. A reader meeting 100% valid has to meet, in the same
    block, the fact that a model calling the wrong tool cleanly every time scores exactly that.
    """
    items, _ = cap.load_tool_schema(_schema_file(tmp_path, TWO_TOOLS))
    out = "\n".join(cap.tool_validity_report(cap.tool_call_validity_block(
        ['{"name": "get_weather", "arguments": {"city": "Leeds"}}'], items)))
    assert "MECHANICAL ONLY" in out
    assert "chose the right tool" in out


def test_the_report_warns_when_the_budget_ate_the_sample(tmp_path):
    items, _ = cap.load_tool_schema(_schema_file(tmp_path, TWO_TOOLS))
    replies = ['{"name": "get_weather", "arguments": {"city": "Leeds"}}'] * 10
    block = cap.tool_call_validity_block(replies, items, [True] * 9 + [False])
    out = "\n".join(cap.tool_validity_report(block))
    assert "TOOL_CALL_BUDGET_SUSPECT" in out
    assert "9 of 10 replies could not be graded" in out


def test_nothing_to_grade_says_so_rather_than_printing_a_block_of_zeros():
    assert "nothing to grade" in cap.tool_validity_report(None)[0]


# ── and through main(), because the artefact is what gets quoted ──────────────────────

def _jsonl_eval(tmp_path, rows):
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


def test_a_tool_schema_on_the_wrong_task_is_refused_before_anything_loads(tmp_path):
    """REFUSED, NOT IGNORED. A flag silently dead on the task it was passed with is how a run gets
    described as having measured tool validity when nothing measured anything, and this command has
    shipped exactly that defect before: `--chat-template` was dead on the abliterate path and
    `--capability-eval` read a flag that did not exist.

    Before the dataset, before the model, because a flag-combination mistake should cost a second.
    """
    schema = _schema_file(tmp_path, TWO_TOOLS)
    with pytest.raises(SystemExit) as e:
        cap.main(["--model", "m", "--task", "numeric", "--tool-schema", schema,
                  "--eval", _jsonl_eval(tmp_path, [{"question": "q", "answer": "1"}]),
                  "--out", str(tmp_path / "c.json")])
    assert "--task tool-call" in str(e.value)
    assert not (tmp_path / "c.json").exists(), "it refused after writing something"


def test_the_offered_tools_and_their_measures_reach_the_artefact(tmp_path, monkeypatch,
                                                                 tiny_model, tiny_tok):
    """THE WIRING ITSELF. `tool_call_validity_block` existed, passed its own tests, and was called
    by nothing for the whole of the day it was written. This is the test that would have noticed.
    """
    import senbonzakura.cli as _cli

    monkeypatch.setattr(_cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)
    monkeypatch.setattr(
        cap, "generate_with_truncation",
        lambda *a, **k: (['{"name": "get_weather", "arguments": {"city": "Leeds"}}'] * 2,
                         [False] * 2))
    schema = _schema_file(tmp_path, TWO_TOOLS)
    out = tmp_path / "c.json"
    rows = [{"question": "weather in Leeds?",
             "answer": '{"name": "get_weather", "arguments": {"city": "Leeds"}}'}] * 2
    cap.main(["--model", "m", "--device", "cpu", "--eval", _jsonl_eval(tmp_path, rows),
              "--n", "2", "--max-new", "8", "--task", "tool-call",
              "--tool-schema", schema, "--out", str(out)])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["tools_offered"] == ["get_weather", "send_email"]
    assert "get_weather: city (string) required" in doc["tools_offered_text"]
    assert doc["tool_schema"] == schema
    assert doc["tool_calls"]["valid"]["count"] == 2
    assert len(doc["tool_calls"]["reports"]) == 2


def test_the_model_is_actually_shown_the_tools(tmp_path, monkeypatch, tiny_model, tiny_tok):
    """AND THIS IS THE ONE THAT MATTERS MOST. Grading a reply against a schema the model never saw
    scores it on a guess: "passed an argument the tool does not declare" is only a failure if the
    model was told what the tool declares. A hidden schema would report the harness's own silence
    as malformed tool use, and malformed tool use is the direction an edited model is expected to
    move in, so the error would have flattered exactly the claim this module exists to distrust.
    """
    import senbonzakura.cli as _cli

    seen = {}
    monkeypatch.setattr(_cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)

    def _capture(model, tok, prompts, device, **k):
        seen["prompts"] = list(prompts)
        return ["no call"] * len(prompts), [False] * len(prompts)

    monkeypatch.setattr(cap, "generate_with_truncation", _capture)
    rows = [{"question": "weather in Leeds?", "answer": "{}"}] * 2
    cap.main(["--model", "m", "--device", "cpu", "--eval", _jsonl_eval(tmp_path, rows),
              "--n", "2", "--max-new", "8", "--task", "tool-call",
              "--tool-schema", _schema_file(tmp_path, TWO_TOOLS),
              "--out", str(tmp_path / "c.json")])
    for prompt in seen["prompts"]:
        assert prompt.startswith("You have these tools:")
        assert "get_weather: city (string) required, days (integer) optional." in prompt
        assert "weather in Leeds?" in prompt


def test_a_run_without_a_schema_writes_the_absence_rather_than_an_empty_block(
        tmp_path, monkeypatch, tiny_model, tiny_tok):
    """None, not {}. An empty block would read as "measured, and found nothing wrong"."""
    import senbonzakura.cli as _cli

    monkeypatch.setattr(_cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)
    monkeypatch.setattr(cap, "generate_with_truncation",
                        lambda *a, **k: (["no call"] * 2, [False] * 2))
    out = tmp_path / "c.json"
    rows = [{"question": "q", "answer": "{}"}] * 2
    cap.main(["--model", "m", "--device", "cpu", "--eval", _jsonl_eval(tmp_path, rows),
              "--n", "2", "--max-new", "8", "--task", "tool-call", "--out", str(out)])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["tool_calls"] is None
    assert doc["tools_offered"] is None
    assert doc["tool_schema"] is None


def test_the_fingerprint_in_the_artefact_covers_the_toolbox(tmp_path, monkeypatch, tiny_model,
                                                            tiny_tok):
    """So the two runs cannot be paired against each other by a later `--compare-to`."""
    import senbonzakura.cli as _cli

    monkeypatch.setattr(_cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)
    monkeypatch.setattr(cap, "generate_with_truncation",
                        lambda *a, **k: (["no call"] * 2, [False] * 2))
    rows = [{"question": "q", "answer": "{}"}] * 2
    bench = _jsonl_eval(tmp_path, rows)

    digests = []
    for i, tools in enumerate((TWO_TOOLS, TWO_TOOLS[:1])):
        out = tmp_path / f"c{i}.json"
        # A directory each, because `_schema_file` writes one fixed filename and the second run
        # would otherwise be measured against the first run's toolbox.
        holder = tmp_path / f"s{i}"
        holder.mkdir()
        cap.main(["--model", "m", "--device", "cpu", "--eval", bench, "--n", "2",
                  "--max-new", "8", "--task", "tool-call", "--out", str(out),
                  "--tool-schema", _schema_file(holder, tools)])
        digests.append(json.loads(out.read_text(encoding="utf-8"))["items_digest"])
    assert digests[0] != digests[1], (
        "the same questions with a different toolbox are a different exam, and without this a "
        "later --compare-to pairs them and reports the difference as a change in capability")
