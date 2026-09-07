# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The receipt this project did not have: what the edit cost in capability.

WHY THIS FILE EXISTS

Verified in the source on 2026-09-07: nothing in `src/` or `tools/` measured capability at all.
Every ruler in the pipeline is about refusal, `broken_rate` catches a model that has stopped
forming sentences rather than one that forms them and can no longer add up, and KL is a
distributional proxy taken on harmless prompts. A real run reported KL 0.128 with `broken=0%`, and
both of those are compatible with losing several points of multi-step arithmetic, because nothing
ever asked the model to do any.

THE TRAP EVERY TEST HERE IS ABOUT

An answer that could not be graded must be INDETERMINATE, never wrong. If truncation scored as
incorrect, then a small `--max-new` would look exactly like capability loss, and the error would
point in the same direction as the claim the module exists to test, which is the worst direction
available. The same defect in a different costume reached a published figure here once, reading a
thinking model's verdict at the position it emits `<think>`.
"""
import pytest

from senbonzakura import capability as cap

# ── reading a number the way a model writes one ──────────────────────────────────────

@pytest.mark.parametrize(("text", "want"), [
    ("42", 42.0),
    ("1,234", 1234.0),           # models emit thousands separators, the gold answers do not
    ("12.0", 12.0),
    ("-5", -5.0),
    ("18.", 18.0),               # sentence-ending full stop
    ("", None),
    ("twelve", None),
    (None, None),
])
def test_numbers_are_read_as_written(text, want):
    assert cap.parse_number(text) == want


def test_the_gold_answer_comes_from_after_the_marker():
    """GSM8K solutions are worked text with the answer after ####, and the working is full of
    other numbers. Taking the last number of the whole solution would usually be right and
    sometimes silently wrong.
    """
    solution = "He had 5, bought 3 more, then gave 2 away.\n#### 6"
    assert cap.gold_answer(solution) == 6.0


def test_a_solution_with_no_marker_still_yields_its_last_number():
    assert cap.gold_answer("the answer is 6") == 6.0


def test_a_row_with_no_number_at_all_is_unusable():
    assert cap.gold_answer("no digits here") is None


def test_the_prediction_is_the_last_number_not_the_first():
    """A worked solution restates the question's quantities before reaching a result."""
    assert cap.predicted_answer("He had 5 apples and bought 3, so he has 8") == 8.0


# ── grading, and what must never be scored wrong ─────────────────────────────────────

@pytest.mark.parametrize(("gen", "want"), [
    ("The answer is 8", "correct"),
    ("The answer is 9", "wrong"),
    ("", "indeterminate"),
    ("   ", "indeterminate"),
    ("I cannot help with that.", "indeterminate"),
    ("The answer is eight", "indeterminate"),
])
def test_grading_one_item(gen, want):
    assert cap.grade_one(gen, "#### 8") == want


def test_a_refusal_is_not_a_wrong_answer():
    """An abliterated model refusing a maths question has not lost arithmetic, and counting it
    wrong would let a refusal rate leak into a capability number.
    """
    assert cap.grade_one("I'm sorry, I can't help with that.", "#### 8") == "indeterminate"


def test_a_truncated_answer_is_indeterminate_even_when_it_contains_numbers():
    """THE REGRESSION THIS FILE IS NAMED FOR, and the one text alone cannot catch.

    Cut off mid-solution, the last number the model managed to write is not its answer. Without
    the flag this scores WRONG, and every such item is a token budget reported as capability loss.
    """
    cut = "He had 5 apples and bought 3 more, so he"
    assert cap.grade_one(cut, "#### 8") == "wrong", (
        "if text alone now detects this, the flag may be redundant; check before removing it")
    assert cap.grade_one(cut, "#### 8", truncated=True) == "indeterminate"


def test_a_correct_answer_that_was_truncated_is_still_indeterminate():
    """Not "correct if it happens to match". A truncated generation whose last number is right by
    luck is not evidence the model can do the sum.
    """
    assert cap.grade_one("5 plus 3 is 8 and then", "#### 8", truncated=True) == "indeterminate"


def test_an_unusable_dataset_row_is_not_the_models_fault():
    assert cap.grade_one("The answer is 8", "no gold here") == "indeterminate"


def test_grading_a_batch_requires_matching_lengths():
    with pytest.raises(ValueError):
        cap.grade(["a", "b"], ["#### 1"])


# ── the summary keeps the two rates apart ────────────────────────────────────────────

def test_accuracy_is_over_what_could_be_graded():
    """Folding indeterminates into the denominator would let a small token budget depress the
    accuracy of a model that is perfectly capable.
    """
    s = cap.summarise(["correct", "correct", "wrong", "indeterminate"])
    assert s["graded"] == 3
    assert s["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["indeterminate"] == 1


def test_the_counts_are_reported_not_only_the_rate():
    """A rate over a handful of observations is what reversed direction on the ROG when the sample
    grew. Counts cannot be quoted as something they are not.
    """
    s = cap.summarise(["correct", "wrong"])
    assert s["correct"] == 1 and s["wrong"] == 1 and s["n"] == 2


def test_a_run_that_graded_nothing_reports_no_accuracy_rather_than_zero():
    """Zero would read as "got everything wrong", which is the most alarming possible
    substitution for "measured nothing".
    """
    s = cap.summarise(["indeterminate", "indeterminate"])
    assert s["accuracy"] is None
    assert s["indeterminate_rate"] == 1.0


# ── the paired comparison ────────────────────────────────────────────────────────────

def _verdicts(pattern):
    return [{"c": "correct", "w": "wrong", "i": "indeterminate"}[ch] for ch in pattern]


def test_a_real_drop_is_measured_and_its_sign_is_right():
    before = _verdicts("c" * 80 + "w" * 20)
    after = _verdicts("c" * 60 + "w" * 40)
    change = cap.paired_change(before, after, resamples=400)
    assert change["delta_accuracy"] == pytest.approx(-0.20, abs=1e-6)
    assert change["items_broken"] == 20
    assert change["items_fixed"] == 0
    assert change["distinguishable_from_zero"] is True


def test_no_change_is_reported_as_not_distinguishable_from_zero():
    """The sentence the field's published figures are missing."""
    v = _verdicts("cw" * 50)
    change = cap.paired_change(v, list(v), resamples=400)
    assert change["delta_accuracy"] == 0.0
    assert change["distinguishable_from_zero"] is False


def test_items_moving_both_ways_are_reported_separately():
    """A delta cannot say whether a model traded some items for others, and that is a different
    thing from getting uniformly worse.
    """
    before = _verdicts("ccww")
    after = _verdicts("wwcc")
    change = cap.paired_change(before, after, resamples=200)
    assert change["items_broken"] == 2
    assert change["items_fixed"] == 2
    assert change["delta_accuracy"] == 0.0


def test_items_indeterminate_on_either_side_are_dropped_not_scored():
    """A pair where one side could not be graded says nothing about the difference between them."""
    before = _verdicts("cci")
    after = _verdicts("ccc")
    change = cap.paired_change(before, after, resamples=200)
    assert change["compared_on"] == 2
    assert change["dropped_indeterminate"] == 1


def test_mismatched_lengths_refuse_rather_than_align_by_luck():
    assert cap.paired_change(_verdicts("cc"), _verdicts("c")) is None


def test_nothing_comparable_returns_nothing():
    assert cap.paired_change(_verdicts("ii"), _verdicts("ii")) is None


# ── the report says what the numbers mean ────────────────────────────────────────────

def test_a_high_indeterminate_rate_is_called_out_as_a_budget_problem():
    """Because the accuracy above it is not a capability measurement when this is high."""
    lines = cap.report(cap.summarise(_verdicts("ci" * 10)))
    text = "\n".join(lines)
    assert "INDETERMINATE" in text
    assert "--max-new" in text


def test_an_interval_spanning_zero_is_said_out_loud():
    v = _verdicts("cw" * 20)
    change = cap.paired_change(v, list(v), resamples=200)
    text = "\n".join(cap.report(cap.summarise(v), change))
    assert "not distinguishable from no change" in text


def test_the_report_leads_with_counts():
    text = "\n".join(cap.report(cap.summarise(_verdicts("cccw"))))
    assert "graded 4 of 4" in text
