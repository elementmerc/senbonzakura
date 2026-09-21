# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The three rule operators added on 2026-09-21, and the checks that needed each one.

The engine's vocabulary is deliberately small, and `registry.py` sets the condition for growing
it: extend when a real check cannot be written otherwise, and say in the commit which check
needed the extension. These three each have one:

    less_than    `quoted-at-a-budget-below-the-visibility-floor` reads a generation budget at a
                 fixed path, outside the per-harness metrics mapping the numeric quantifier works
                 over, and the vocabulary had no numeric comparison anywhere else.
    disagrees    `an-arm-labelled-by-a-setting-it-did-not-apply` and
                 `a-count-that-is-not-the-count-that-was-applied` both ask whether two fields of
                 one artefact that describe the same quantity actually agree.
    any_entry    `null-reported-as-zero` and `a-rate-with-no-partition-beside-it` both need SEVERAL
                 conditions to hold of ONE metric entry, which a conjunction of two existing
                 quantifiers cannot express: that asks instead for each condition to hold of some
                 entry, possibly a different one each time.

Every operator in this engine is TOTAL, so most of what is asserted below is that an unfamiliar
shape answers false rather than raising. The input is somebody else's artefact and a rule that
raised would turn "this check does not apply" into a crash.
"""
import pytest
from senbonzakura_check.registry import CheckError, evaluate

# ── less_than ────────────────────────────────────────────────────────────────────────────────

BUDGET = {"raw": {"generation": {"max_new_tokens": 48}},
          "zero": 0, "float": 95.5, "text": "48", "flag": True, "nothing": None}


@pytest.mark.parametrize(("path", "limit", "expected"), [
    ("raw.generation.max_new_tokens", 96, True),
    ("raw.generation.max_new_tokens", 48, False),
    ("raw.generation.max_new_tokens", 32, False),
    ("float", 96, True),
    ("zero", 96, True),
    # ABSENT IS NOT SHORT. A missing budget is a different finding from a short one, and reading
    # absence as zero would fire this on every artefact that records no budget, which is most.
    ("missing", 96, False),
    ("nothing", 96, False),
    ("deeply.nested.absent", 96, False),
    # A string that looks like a number is not one. Coercing here would let a harness's free-text
    # field be compared numerically on a guess about its format.
    ("text", 96, False),
    # A bool is an int in Python and is never a measurement.
    ("flag", 96, False),
])
def test_less_than_compares_only_real_numbers(path, limit, expected):
    assert evaluate({"op": "less_than", "path": path, "value": limit}, BUDGET) is expected


@pytest.mark.parametrize("rule", [
    {"op": "less_than", "path": "zero"},
    {"op": "less_than", "path": "zero", "value": "96"},
    {"op": "less_than", "path": "zero", "value": True},
    {"op": "less_than", "value": 96},
])
def test_a_less_than_with_no_usable_limit_is_refused(rule):
    """An authoring error has to stop the run. Treated as false it would sit in the registry
    reporting nothing forever, which is the failure the checker itself exists to find.
    """
    with pytest.raises(CheckError):
        evaluate(rule, BUDGET)


# ── disagrees ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("a", "b", "expected"), [
    (3, 1, True),
    (1, 1, False),
    ("per_layer", "single", True),
    ("single", "single", False),
    # THE LIST CASE, which is why this is not a `not_equals` over two paths. The per-layer
    # direction list is compared elementwise against the scalar count, because the finding is
    # that ANY layer disagrees, and every summary of that list has been the wrong one before.
    ([3, 2, 1], 1, True),
    ([1, 1, 1], 1, False),
    (1, [1, 2], True),
    (1, [1, 1], False),
    ([], 1, False),
    ([1, 1], [1, 1], False),
    ([1, 1], [1, 2], True),
])
def test_disagrees_answers_whether_two_recorded_values_match(a, b, expected):
    doc = {"a": a, "b": b}
    assert evaluate({"op": "disagrees", "paths": ["a", "b"]}, doc) is expected


@pytest.mark.parametrize("doc", [
    {"a": 3},
    {"b": 1},
    {},
    {"a": 3, "b": None},
    {"a": None, "b": 1},
])
def test_a_missing_side_is_not_a_disagreement(doc):
    """"These two fields disagree" is a claim about two recorded values. An artefact that
    records only one of them has not made it, and saying it has would turn every incomplete
    artefact into a finding about a comparison nobody attempted.
    """
    assert evaluate({"op": "disagrees", "paths": ["a", "b"]}, doc) is False


@pytest.mark.parametrize("rule", [
    {"op": "disagrees", "paths": ["only-one"]},
    {"op": "disagrees", "paths": ["a", "b", "c"]},
    {"op": "disagrees"},
])
def test_a_disagrees_without_exactly_two_paths_is_refused(rule):
    with pytest.raises(CheckError):
        evaluate(rule, {"a": 1, "b": 2})


# ── any_entry ────────────────────────────────────────────────────────────────────────────────

def _entry(**over):
    return {"op": "any_entry", "path": "metrics", **over}


def test_any_entry_needs_every_condition_true_of_one_entry():
    """THE WHOLE REASON THIS OPERATOR EXISTS, asserted directly.

    Two metrics, one with a zero value and a denominator, one with a real value and none. Every
    condition holds of SOME entry, so a conjunction of two per-mapping quantifiers would fire.
    Nothing here is wrong, and the check that fired would be reporting a defect built out of
    halves of two different measurements.
    """
    doc = {"metrics": {"measured_zero": {"value": 0, "n": 132},
                       "no_denominator": {"value": 0.094}}}
    conditions = [{"field": "value", "is": "zero"},
                  {"field": "n", "is": "unset_or_falsy"}]
    assert evaluate(_entry(all=conditions), doc) is False

    doc["metrics"]["both_at_once"] = {"value": 0}
    assert evaluate(_entry(all=conditions), doc) is True


@pytest.mark.parametrize(("entry", "condition", "expected"), [
    # ZERO IS NOT ABSENT, and the two are opposite claims about a measurement: one says it was
    # taken and came out at nothing, the other says nobody looked.
    ({"value": 0}, {"field": "value", "is": "zero"}, True),
    ({"value": 0.0}, {"field": "value", "is": "zero"}, True),
    ({"value": None}, {"field": "value", "is": "zero"}, False),
    ({}, {"field": "value", "is": "zero"}, False),
    ({"value": 0.094}, {"field": "value", "is": "zero"}, False),
    # A bool is an int in Python. `higher_is_better: false` must never read as the number zero.
    ({"value": False}, {"field": "value", "is": "zero"}, False),
    # TWO MEANINGS, TWO WORDS, SPLIT 2026-09-21. `falsy` meant "missing or falsy" here while
    # the fixed-path operator of the same name meant "present and falsy", so the two vocabularies
    # disagreed on the only case that distinguishes them, and a reader of a check file met a
    # second grammar without being told. Both meanings are wanted; they no longer share a name.
    ({"n": None}, {"field": "n", "is": "falsy"}, True),
    ({}, {"field": "n", "is": "falsy"}, False),
    ({"n": 0}, {"field": "n", "is": "falsy"}, True),
    ({"n": 132}, {"field": "n", "is": "falsy"}, False),
    ({"n": None}, {"field": "n", "is": "unset_or_falsy"}, True),
    ({}, {"field": "n", "is": "unset_or_falsy"}, True),
    ({"n": 0}, {"field": "n", "is": "unset_or_falsy"}, True),
    ({"n": 132}, {"field": "n", "is": "unset_or_falsy"}, False),
    # `number` is what applicability asks when a rule reads a denominator. A null, a missing
    # field and a bool are all "no measurement here".
    ({"n": 132}, {"field": "n", "is": "number"}, True),
    ({"n": 0}, {"field": "n", "is": "number"}, True),
    ({"n": None}, {"field": "n", "is": "number"}, False),
    ({}, {"field": "n", "is": "number"}, False),
    ({"n": True}, {"field": "n", "is": "number"}, False),
    ({"n": 132}, {"field": "n", "is": "truthy"}, True),
    ({"n": None}, {"field": "n", "is": "present"}, True),
    ({}, {"field": "n", "is": "present"}, False),
    ({}, {"field": "n", "is": "absent"}, True),
    ({"units": "proportion"}, {"field": "units", "equals": "proportion"}, True),
    ({"units": "nats"}, {"field": "units", "equals": "proportion"}, False),
    ({}, {"field": "units", "equals": "proportion"}, False),
])
def test_each_entry_predicate_answers_the_question_it_claims_to(entry, condition, expected):
    doc = {"metrics": {"only": entry}}
    assert evaluate(_entry(all=[condition]), doc) is expected


@pytest.mark.parametrize("doc", [
    {},
    {"metrics": None},
    {"metrics": []},
    {"metrics": "not a mapping"},
    {"metrics": {}},
    {"metrics": {"not a block": 0.5}},
])
def test_any_entry_is_total_over_shapes_it_did_not_expect(doc):
    assert evaluate(_entry(all=[{"field": "value", "is": "zero"}]), doc) is False


@pytest.mark.parametrize("rule", [
    _entry(),
    _entry(all=[]),
    _entry(all="not a list"),
    _entry(all=[{"is": "zero"}]),
    _entry(all=[{"field": "value"}]),
    _entry(all=[{"field": "value", "is": "no-such-predicate"}]),
    _entry(all=["not an object"]),
])
def test_a_malformed_any_entry_is_refused_loudly(rule):
    with pytest.raises(CheckError):
        evaluate(rule, {"metrics": {"a": {"value": 0}}})
