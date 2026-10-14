# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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


# ── differs_by_more_than ─────────────────────────────────────────────────────────────────────
#
# Added for `a-headline-that-its-own-control-does-not-reproduce`. The compass published a headline
# AUC of 0.9636 beside a canonical control of 0.8132, and no operator could say "these should agree
# to a stated tolerance". `disagrees` tests equality, which no measurement satisfies, and
# `differs_in_more_than` counts keys. The boundary is strict: a gap of exactly the tolerance agrees.

_TOL = {"op": "differs_by_more_than", "paths": ["a", "b"], "value": 0.01}


@pytest.mark.parametrize(("a", "b", "expected"), [
    (0.9636, 0.8132, True),      # the broken compass pair: 0.15 apart
    (0.7263, 0.3663, True),      # the 0.6B pair, below chance on the control
    (0.9887, 0.98875, False),    # the corrected 1.7B pair, agreeing to 0.00005
    (0.5, 0.45, True),           # 0.05 apart, well past the tolerance
    (0.5, 0.55, True),           # the sign of the gap does not matter
    (0, 0, False),
])
def test_differs_by_more_than_compares_how_far_apart_two_numbers_are(a, b, expected):
    assert evaluate(_TOL, {"a": a, "b": b}) is expected


# THE BOUNDARY IS TESTED ON VALUES A BINARY FLOAT REPRESENTS EXACTLY. 0.51 minus 0.5 is not 0.01
# in IEEE arithmetic, so a boundary case written in decimal would test the float, not the rule.
@pytest.mark.parametrize(("a", "b", "expected"), [
    (0.5, 0.75, False),          # exactly the tolerance: agreement, because the comparison is strict
    (0.5, 0.7500001, True),      # a hair past it
    (0.75, 0.5, False),          # and the same pair the other way round
])
def test_the_differs_by_more_than_boundary_is_strict(a, b, expected):
    rule = {"op": "differs_by_more_than", "paths": ["a", "b"], "value": 0.25}
    assert evaluate(rule, {"a": a, "b": b}) is expected


@pytest.mark.parametrize("doc", [
    {},
    {"a": 0.5},
    {"b": 0.5},
    {"a": None, "b": 0.5},
    {"a": "0.5", "b": 0.9},
    {"a": True, "b": 0.0},
    {"a": float("nan"), "b": 0.5},
    {"a": float("inf"), "b": 0.5},
    {"a": [0.5], "b": 0.9},
])
def test_a_missing_or_non_numeric_side_is_not_a_difference(doc):
    """Total over unfamiliar shapes: a claim about two recorded numbers needs both recorded."""
    assert evaluate(_TOL, doc) is False


@pytest.mark.parametrize("rule", [
    {"op": "differs_by_more_than", "paths": ["a"], "value": 0.01},
    {"op": "differs_by_more_than", "paths": "ab", "value": 0.01},
    {"op": "differs_by_more_than", "paths": ["a", "b"]},
    {"op": "differs_by_more_than", "paths": ["a", "b"], "value": -0.01},
    {"op": "differs_by_more_than", "paths": ["a", "b"], "value": "0.01"},
    {"op": "differs_by_more_than", "paths": ["a", "b"], "value": True},
])
def test_a_differs_by_more_than_without_a_usable_tolerance_is_refused(rule):
    """A tolerance that is missing, negative or a string would make the check silently fire on
    everything, or on nothing, so it is refused rather than guessed at.
    """
    with pytest.raises(CheckError):
        evaluate(rule, {"a": 0.5, "b": 0.9})


def test_a_tolerance_of_zero_is_allowed_and_means_exact_agreement():
    assert evaluate({"op": "differs_by_more_than", "paths": ["a", "b"], "value": 0},
                    {"a": 0.5, "b": 0.5}) is False
    assert evaluate({"op": "differs_by_more_than", "paths": ["a", "b"], "value": 0},
                    {"a": 0.5, "b": 0.5000001}) is True


# ── totality over shapes the operators did not expect ─────────────────────────────────────────
#
# Found by the coverage measurement of 2026-10-14: these were the operator branches no test
# reached, and they are exactly where an unfamiliar artefact would go. Each expectation below was
# read from the operator's stated contract, then checked against the code, not taken from it.

def test_any_missing_answers_false_when_the_mapping_is_not_a_mapping():
    assert evaluate({"op": "any_missing", "path": "p", "field": "f"}, {"p": "x"}) is False


def test_any_missing_counts_a_non_object_entry_as_missing_the_field():
    """An entry that is not an object cannot carry the field, so it is missing it."""
    assert evaluate({"op": "any_missing", "path": "p", "field": "f"}, {"p": {"a": 3}}) is True


def test_any_missing_is_false_when_every_entry_carries_the_field():
    assert evaluate({"op": "any_missing", "path": "p", "field": "f"},
                    {"p": {"a": {"f": 1}}}) is False


def test_any_outside_answers_false_when_the_mapping_is_not_a_mapping():
    rule = {"op": "any_outside", "path": "p", "field": "v", "min": 0}
    assert evaluate(rule, {"p": "x"}) is False


def test_any_outside_skips_entries_the_when_clause_does_not_select():
    rule = {"op": "any_outside", "path": "p", "field": "v", "min": 0,
            "when": {"field": "k", "equals": "z"}}
    assert evaluate(rule, {"p": {"a": {"k": "y", "v": -1}}}) is False
    assert evaluate(rule, {"p": {"a": {"k": "z", "v": -1}}}) is True


def test_any_outside_never_reads_a_boolean_as_a_number():
    rule = {"op": "any_outside", "path": "p", "field": "v", "min": 0}
    assert evaluate(rule, {"p": {"a": {"v": True}}}) is False


def test_any_outside_skips_a_non_object_entry_rather_than_reading_it():
    rule = {"op": "any_outside", "path": "p", "field": "v", "min": 0}
    assert evaluate(rule, {"p": {"a": 5}}) is False


def test_any_outside_with_only_a_maximum_fires_above_it_and_not_below():
    rule = {"op": "any_outside", "path": "p", "field": "v", "max": 1}
    assert evaluate(rule, {"p": {"a": {"v": 2}}}) is True
    assert evaluate(rule, {"p": {"a": {"v": 0.5}}}) is False


def test_intersects_answers_false_for_a_scalar_side_rather_than_comparing_it():
    """A scalar is not a collection, so it has no members to share with the other side."""
    assert evaluate({"op": "intersects", "paths": ["s", "t"]}, {"s": 3, "t": [3]}) is False


def test_intersects_is_true_when_both_sides_are_collections_that_share_a_member():
    assert evaluate({"op": "intersects", "paths": ["s", "t"]}, {"s": [3], "t": [3]}) is True
