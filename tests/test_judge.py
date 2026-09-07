# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A judge is validated before it grades, or nothing it grades may be published.

WHY THIS FILE EXISTS BEFORE ANY JUDGE DOES

Every task in `capability` is graded by code against a reference answer, and each records why it
needs no judge. That runs out: tool use, open-ended instruction following and anything agentic
cannot be marked by string equality, and the usual answer is to ask a model. From that moment every
number downstream inherits the judge's reliability, and this project has withdrawn results over an
unvalidated grader before.

THE FAILURE THIS IS BUILT AROUND

A judge that answers "correct" to everything agrees 90% of the time with a set that is 90% correct.
It looks excellent and it is useless, and raw agreement cannot see it. Agreement above chance can,
and so can per-class recall.
"""
import json

import pytest

from senbonzakura import judge


def _set(correct=90, wrong=10):
    return ["correct"] * correct + ["wrong"] * wrong


def test_the_lazy_judge_is_refused_despite_ninety_percent_agreement():
    """THE REGRESSION THIS FILE IS NAMED FOR."""
    reference = _set()
    lazy = ["correct"] * 100
    v = judge.validate(lazy, reference)
    assert v["certified"] is False
    assert v["agreement"] == pytest.approx(0.90), "the misleading number is still reported"
    assert v["kappa"] == pytest.approx(0.0), "and the honest one shows it learned nothing"


def test_the_refusal_says_both_reasons_not_just_one():
    """Two independent grounds catch it, and a reader should see both: the kappa says it learned
    nothing, the per-class recall says which half it is blind to.
    """
    v = judge.validate(["correct"] * 100, _set())
    text = " ".join(v["reasons"])
    assert "above chance" in text
    assert "blind to one class" in text


def test_a_good_judge_is_certified():
    reference = _set()
    good = ["correct"] * 88 + ["wrong"] * 2 + ["wrong"] * 9 + ["correct"]
    v = judge.validate(good, reference)
    assert v["certified"] is True
    assert v["kappa"] > judge.MIN_KAPPA


def test_a_judge_blind_to_the_rare_class_is_refused_even_at_high_kappa():
    """The class it misses is usually the one that would have caught the model out, so this is
    checked separately rather than being folded into a single score.
    """
    reference = ["correct"] * 60 + ["wrong"] * 40
    blind = ["correct"] * 60 + ["correct"] * 30 + ["wrong"] * 10
    v = judge.validate(blind, reference)
    assert v["certified"] is False
    assert any("blind to one class" in r for r in v["reasons"])


# ── the maths behind the verdict ─────────────────────────────────────────────────────

def test_perfect_agreement_on_a_balanced_set_is_kappa_one():
    labels = ["correct", "wrong"] * 50
    assert judge.cohens_kappa(labels, list(labels)) == pytest.approx(1.0)


def test_chance_level_agreement_is_kappa_zero():
    """Two graders that both answer the common label nine times in ten agree 81% by accident."""
    reference = _set()
    assert judge.cohens_kappa(["correct"] * 100, reference) == pytest.approx(0.0)


def test_a_judge_worse_than_guessing_gets_a_negative_kappa():
    """Not clamped to zero. "Worse than chance" is worth seeing rather than hiding."""
    reference = ["correct"] * 50 + ["wrong"] * 50
    inverted = ["wrong"] * 50 + ["correct"] * 50
    assert judge.cohens_kappa(inverted, reference) < 0


def test_a_set_with_only_one_label_cannot_certify_anything():
    """A set that could not have caught the judge being wrong must not certify it. Reporting 1.0
    here is the trap: perfect agreement on an impossible test.
    """
    assert judge.cohens_kappa(["correct"] * 50, ["correct"] * 50) is None
    v = judge.validate(["correct"] * 50, ["correct"] * 50)
    assert v["certified"] is False
    assert any("could not have caught the judge being wrong" in r for r in v["reasons"])


def test_per_class_recall_carries_an_interval():
    """Ten items of a class is a thin basis, and the width is what says so."""
    stats = judge.per_class(["correct"] * 100, _set())
    assert stats["wrong"]["n"] == 10
    assert stats["wrong"]["recall"] == 0.0
    lo, hi = stats["wrong"]["ci"]
    assert hi > 0.2, "an interval that excludes doubt on 10 items would be the defect"


# ── the guards ───────────────────────────────────────────────────────────────────────

def test_a_tiny_validation_set_certifies_nothing():
    """A judge validated on a handful has not been validated, and this is the same floor the
    rate reporting uses.
    """
    v = judge.validate(["correct", "wrong"], ["correct", "wrong"])
    assert v["certified"] is False
    assert any("below the floor" in r for r in v["reasons"])


def test_mismatched_lengths_refuse_rather_than_align_by_luck():
    v = judge.validate(["correct"], ["correct", "wrong"])
    assert v["certified"] is False
    assert any("different numbers of items" in r for r in v["reasons"])


def test_ungraded_items_are_dropped_from_both_sides_together():
    """A pair where either side has no label says nothing about agreement."""
    reference = _set() + [None] * 5
    j = ["correct"] * 88 + ["wrong"] * 2 + ["wrong"] * 9 + ["correct"] + [None] * 5
    v = judge.validate(j, reference)
    assert v["n"] == 100


def test_the_report_names_raw_agreement_as_the_wrong_number():
    """Somebody will look for it, so it is shown beside kappa and labelled rather than omitted."""
    text = "\n".join(judge.report(judge.validate(["correct"] * 100, _set())))
    assert "NOT the number that matters" in text
    assert "NOT CERTIFIED" in text


# ── the command ──────────────────────────────────────────────────────────────────────

def _write(path, labels):
    path.write_text("\n".join(json.dumps({"verdict": v}) for v in labels), encoding="utf-8")
    return path


def test_the_command_exits_non_zero_when_a_judge_is_not_certified(tmp_path):
    """So a pipeline cannot go on to grade with it by accident."""
    j = _write(tmp_path / "j.jsonl", ["correct"] * 100)
    r = _write(tmp_path / "r.jsonl", _set())
    assert judge.main(["--judge", str(j), "--reference", str(r)]) == 1


def test_the_command_exits_zero_for_a_good_judge(tmp_path):
    good = ["correct"] * 88 + ["wrong"] * 2 + ["wrong"] * 9 + ["correct"]
    j = _write(tmp_path / "j.jsonl", good)
    r = _write(tmp_path / "r.jsonl", _set())
    assert judge.main(["--judge", str(j), "--reference", str(r)]) == 0


def test_an_unreadable_file_is_a_clear_refusal(tmp_path):
    with pytest.raises(SystemExit, match="could not read"):
        judge.main(["--judge", str(tmp_path / "nope.jsonl"),
                    "--reference", str(tmp_path / "also-nope.jsonl")])


def test_the_verdict_can_be_written_for_a_run_to_record(tmp_path):
    j = _write(tmp_path / "j.jsonl", ["correct"] * 100)
    r = _write(tmp_path / "r.jsonl", _set())
    out = tmp_path / "v.json"
    judge.main(["--judge", str(j), "--reference", str(r), "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["certified"] is False
