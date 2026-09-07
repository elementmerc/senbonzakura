# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The card beside the weights states what the artefacts support and nothing else.

WHY THIS FILE EXISTS

Abliterated models get published with a sentence like "minimal degradation" and nothing behind it.
The field ships on a claimed one to three percent with no benchmarks; a comparative study puts one
family at eight points on grade-school arithmetic. Neither side offers an interval and nobody can
check either.

Everything needed to do better is already on disk when a run finishes. What was missing is the
page, and the discipline that a page is where overclaiming happens.

THE FOUR RULES UNDER TEST

A number not in an artefact does not appear. A rate the sample cannot carry is printed as counts.
An interval spanning zero is written as "not distinguishable from no change" rather than as a
delta with a caveat further down. And a section with no evidence says NOT MEASURED in those words,
because an absent section reads as nothing to report and that is a different claim.
"""
import json

import pytest

from senbonzakura import modelcard


def _abl(**over):
    d = {"model": "M", "method": "searched", "baseline_refusal": 0.4, "post_bake_refusal": 0.0,
         "post_bake_kl": 0.12, "track": "default"}
    d.update(over)
    return d


def _cap(n=200, graded=190, correct=120, change=None, **over):
    d = {"task": "numeric", "eval": "gsm8k",
         "summary": {"n": n, "graded": graded, "correct": correct, "wrong": graded - correct,
                     "indeterminate": n - graded},
         "change": change}
    d.update(over)
    return d


def _text(**kw):
    return "\n".join(modelcard.build(**kw))


# ── the rule that matters most: absence is stated ────────────────────────────────────

def test_a_card_with_no_capability_evidence_declares_it_unmeasured():
    """THE RULE THIS FILE EXISTS FOR. Omitting the section would read as nothing to report."""
    text = _text(abl=_abl())
    assert "NOT MEASURED" in text
    assert "Capability" in text


def test_the_missing_capability_section_explains_why_the_other_numbers_do_not_cover_it():
    """A reader who sees a low KL and no capability section will assume the KL covered it."""
    text = _text(abl=_abl())
    assert "cannot see reasoning loss" in text
    assert "no claim about capability is supported" in text


def test_every_section_appears_even_with_no_artefacts_at_all():
    text = _text()
    for heading in ("What was done", "Refusal", "Capability", "Corpus", "Reproducing"):
        assert heading in text, f"{heading} was dropped rather than declared"


def test_a_run_with_no_command_says_it_cannot_be_reproduced():
    assert "NOT RECORDED" in _text(abl=_abl())
    assert "senbonzakura --method x" in _text(abl=_abl(), command="senbonzakura --method x")


# ── rates the sample cannot carry ────────────────────────────────────────────────────

def test_a_small_capability_run_is_reported_as_counts_not_a_rate():
    """The reporting floor reaches the card, or the card is where the defect comes back."""
    text = _text(cap=_cap(n=5, graded=5, correct=3))
    assert "3/5" in text
    assert "not stated as a rate" in text
    assert "60.0%" not in text


def test_a_large_run_gets_its_rate_with_the_counts_beside_it():
    text = _text(cap=_cap())
    assert "120/190" in text
    assert "95% CI" in text


# ── the change, and what an overlapping interval must read as ────────────────────────

def test_a_real_drop_is_stated_with_its_interval():
    change = {"compared_on": 185, "delta_accuracy": -0.078, "delta_ci": [-0.121, -0.036],
              "distinguishable_from_zero": True, "items_broken": 20, "items_fixed": 6}
    text = _text(cap=_cap(change=change))
    assert "-7.8%" in text
    assert "95% CI" in text


def test_an_interval_spanning_zero_is_not_written_as_a_delta():
    """A point estimate with a caveat three lines below is how a null result gets quoted as a
    finding. The words come first here, and the number after them.
    """
    change = {"compared_on": 100, "delta_accuracy": -0.01, "delta_ci": [-0.06, 0.04],
              "distinguishable_from_zero": False, "items_broken": 5, "items_fixed": 4}
    text = _text(cap=_cap(change=change))
    assert "not distinguishable from no change" in text
    line = next(ln for ln in text.splitlines() if "change in accuracy" in ln)
    assert "not distinguishable" in line, "the caveat must be on the same line as the number"


def test_items_moving_both_ways_is_called_out_as_a_different_finding():
    change = {"compared_on": 100, "delta_accuracy": 0.0, "delta_ci": [-0.05, 0.05],
              "distinguishable_from_zero": False, "items_broken": 10, "items_fixed": 10}
    assert "answers differently" in _text(cap=_cap(change=change))


def test_a_capability_run_with_no_reference_says_it_is_not_a_cost():
    """An absolute score is not a statement about what the edit did, and a reader will take it
    as one unless told.
    """
    text = _text(cap=_cap())
    assert "not a statement about what the edit cost" in text


def test_a_high_indeterminate_count_is_surfaced_beside_the_accuracy():
    text = _text(cap=_cap(n=200, graded=100, correct=60))
    assert "100 of 200 answers could not be graded" in text


# ── loading, and refusing to make something up ───────────────────────────────────────

def test_a_missing_artefact_is_a_gap_rather_than_an_error(tmp_path):
    assert modelcard.load(tmp_path / "nope.json") is None
    assert modelcard.load("") is None


def test_an_unreadable_artefact_is_a_gap_rather_than_a_crash(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert modelcard.load(p) is None


def test_the_command_refuses_to_build_a_card_from_nothing():
    """A card with no artefacts behind it is a template, and this exists to stop those being
    published.
    """
    with pytest.raises(SystemExit, match="nothing to report on"):
        modelcard.main([])


def test_the_command_writes_a_file_when_asked(tmp_path):
    abl = tmp_path / "a.json"
    abl.write_text(json.dumps(_abl()), encoding="utf-8")
    out = tmp_path / "card.md"
    assert modelcard.main(["--abliteration", str(abl), "--out", str(out)]) == 0
    assert "Abliteration report" in out.read_text(encoding="utf-8")


def test_the_card_never_invents_a_number_that_is_not_in_an_artefact():
    """The load-bearing property. Every figure printed has to trace to a key that was supplied."""
    text = _text(abl={"model": "M", "method": "single-pass"})
    for absent in ("refusal before", "KL drift", "broken output"):
        assert absent not in text, f"{absent} was printed from an artefact that did not carry it"
