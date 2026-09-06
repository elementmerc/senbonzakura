# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A model that stopped answering must never outrank one that still works.

WHY THIS FILE EXISTS

`refusal_rate` counts refusals, and an empty string is not a refusal, so a model that has been
damaged into silence scores 0.0% refusal: a perfect abliteration by the headline number. The
search already knew that and penalised brokenness at 2.0 in its objective. The FINAL PICK did
not: `knee_scalar` took four numbers and brokenness was not among them, and `--eval-refusal-final`
re-scored refusals, soft and heretic on a larger eval while leaving brokenness at its stale value
from the small one. So the one step that exists to stop the knee being overfit to the small eval
was blind to exactly the axis that says whether the model still works.

Found the long way round, 2026-09-06. A peer session's determinism run on the ROG reported every
generation IDENTICAL while measuring nothing: the model spent its whole token budget in
`reasoning_content` and returned an empty `content` five times, and hashing the empty string five
times agrees with itself. They caught it only because a transcript column printed blank. The same
shape is what makes a broken model look abliterated, so the ledger entry and these tests treat it
as one defect class rather than two coincidences.
"""
import pytest

from senbonzakura import metrics
from senbonzakura.metrics import KNEE_W_BROKEN, knee_scalar


def test_a_wrecked_model_does_not_beat_a_working_one_on_a_lower_refusal_rate():
    """THE REGRESSION THIS FILE IS NAMED FOR.

    The wrecked candidate looks strictly better on every axis the old scalar could see: it refuses
    less, hedges less, trips the keyword ruler less, and drifts no further. It is also producing
    nothing. Before the brokenness term it won.
    """
    wrecked = knee_scalar(0.0, 0.0, 0.0, 0.05, broken=1.0)
    working = knee_scalar(0.2, 0.0, 0.0, 0.05, broken=0.0)
    assert working < wrecked, (
        "a model that emits nothing scored better than one that still answers; the brokenness "
        "term is not reaching the selection")


def test_the_old_scalar_would_have_picked_the_wrecked_one():
    """The mutation test: show the defect is real, not hypothetical.

    Recomputes both candidates with the brokenness weight removed, which is exactly the rule that
    shipped, and requires that rule to get it wrong. A gate only ever seen passing has not been
    shown to work.
    """
    def old(ref, soft, heretic, kl):
        return (metrics.KNEE_W_NONCOMPLIANCE * (ref + soft)
                + metrics.KNEE_W_KEYWORD * heretic
                + metrics.KNEE_W_KL * max(0.0, kl - metrics.KL_TARGET))

    assert old(0.0, 0.0, 0.0, 0.05) < old(0.2, 0.0, 0.0, 0.05), (
        "if the old rule no longer prefers the wrecked candidate, this test is guarding nothing "
        "and needs rewriting around a case that still separates them")


def test_brokenness_is_required_and_not_defaulted():
    """A default of 0.0 would reintroduce the defect for any caller who forgot the argument.

    Both real callers already HAD the number and neither passed it, which is the whole argument
    for making it required: a silent default would have left both of them scoring wrecked models
    as perfect ones while looking correct at the call site.
    """
    with pytest.raises(TypeError):
        knee_scalar(0.1, 0.0, 0.0, 0.05)


def test_brokenness_is_positional_proof_against_being_passed_by_accident():
    """Keyword-only, so a caller cannot slide a value into it by argument order."""
    with pytest.raises(TypeError):
        knee_scalar(0.1, 0.0, 0.0, 0.05, metrics.KL_TARGET, 1.0)


def test_the_weight_matches_the_search_objective():
    """The rule that RANKS trials and the rule that PICKS among them must agree.

    The search objective carries `2.0 * broken` (cli.py, `_objective`). If these two drift apart,
    the search optimises one definition of "wrecked" and the final pick applies another, which is
    the class of defect that produced this file.
    """
    assert KNEE_W_BROKEN == 2.0
    delta = knee_scalar(0.0, 0.0, 0.0, 0.0, broken=0.5) - knee_scalar(0.0, 0.0, 0.0, 0.0, broken=0.0)
    assert delta == pytest.approx(2.0 * 0.5)


def test_an_all_empty_batch_reads_as_broken_and_not_as_compliant():
    """The measurement underneath the whole thing, stated as an assertion.

    Empty output is 0% refusal and 100% broken. Both are true; the danger is reading only the
    first. This pins the pair so a change to `is_refusal` or `is_broken` that quietly makes an
    empty batch look like a clean abliteration fails here.
    """
    empties = ["", "   ", "\n"]
    assert metrics.refusal_rate(empties) == 0.0
    assert metrics.broken_rate(empties) == 1.0
    assert knee_scalar(metrics.refusal_rate(empties), 0.0, 0.0, 0.0,
                       broken=metrics.broken_rate(empties)) == pytest.approx(KNEE_W_BROKEN)


def test_a_partly_broken_candidate_is_ranked_between_the_two_extremes():
    """Monotone in brokenness, so the term grades rather than merely gating at a threshold."""
    scores = [knee_scalar(0.0, 0.0, 0.0, 0.05, broken=b) for b in (0.0, 0.25, 0.5, 1.0)]
    assert scores == sorted(scores)
    assert len(set(scores)) == 4, "brokenness must change the score at every step, not just once"
