# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Three judges, and the rule that makes having three worth the trouble.

The v0.7 gate: each arm is scored by at least three judges that differ in a way that matters,
every judge's own verdict is kept, the reported interval carries the between-judge spread, and
**where the judges disagree about which tool won, there is no winner**.

The last clause is the one under test here, because it is the one a panel implementation
naturally loses. Averaging three judges into one number produces a more expensive single judge
and hides the only finding a panel can make that a single ruler cannot.
"""
from __future__ import annotations

import pytest

from senbonzakura import panel


def _j(name, a, b, *, higher_is_better=False, sd=None, n=None):
    return panel.judge_verdict(name, a, b, higher_is_better=higher_is_better,
                               sd=sd, n_per_arm=n)


# ── one judge ────────────────────────────────────────────────────────────────────
def test_lower_wins_when_lower_is_better():
    """Refusal rate: arm A at 0.02 against arm B at 0.17 means A won."""
    assert _j("keyword", 0.02, 0.17)["verdict"] == panel.A


def test_higher_wins_when_higher_is_better():
    """Harm recognition: the compass reads the other way round, and a panel that got this
    backwards would report every arm's loss as a win.
    """
    assert _j("compass", 0.81, 0.62, higher_is_better=True)["verdict"] == panel.A


def test_an_exact_tie_is_a_tie():
    assert _j("keyword", 0.1, 0.1)["verdict"] == panel.TIE


def test_a_gap_the_judge_cannot_resolve_is_a_tie_not_a_narrow_win():
    """THE COMPOSITION WITH `power`, and the reason it is not cosmetic.

    A judge reporting a gap smaller than its own comparison can resolve has found noise. Letting
    it vote would allow an underpowered judge to break a tie between two judges that can see.
    """
    v = _j("keyword", 0.50, 0.49, sd=0.05, n=5)
    assert v["resolvable"] is False
    assert v["verdict"] == panel.TIE


def test_a_gap_the_judge_can_resolve_is_kept():
    """Lower is better here, so the arm at 0.10 wins. Asserted in the direction that caught my
    own slip: the first version of this test expected A because A was written first, which is
    exactly the mistake a panel reading refusal and harm recognition together will make.
    """
    v = _j("keyword", 0.50, 0.10, sd=0.05, n=5)
    assert v["resolvable"] is True
    assert v["verdict"] == panel.B


def test_a_judge_with_no_spread_reports_its_resolving_power_as_unknown():
    """Unknown is not yes. A judge that cannot say whether it could see the gap it reported has
    not established that it did.
    """
    assert _j("held-out-model", 0.2, 0.4)["resolvable"] is None


# ── the panel ────────────────────────────────────────────────────────────────────
def test_three_agreeing_judges_produce_a_winner():
    p = panel.panel_verdict([_j("keyword", 0.02, 0.17),
                             _j("compass", 0.30, 0.55),
                             _j("held-out", 0.10, 0.40)])
    assert p["verdict"] == panel.A
    assert p["agreed"] is True


def test_judges_that_disagree_produce_no_winner():
    """THE RULE THE PANEL EXISTS FOR. Two judges for A, one for B, and the answer is not "A by
    majority": it is that this comparison depends on which ruler you pick.
    """
    p = panel.panel_verdict([_j("keyword", 0.02, 0.17),
                             _j("compass", 0.55, 0.30),
                             _j("held-out", 0.10, 0.40)])
    assert p["verdict"] == panel.NO_WINNER
    assert p["agreed"] is False


def test_a_majority_does_not_override_a_dissenter():
    """Stated separately from the test above because majority voting is the obvious
    implementation and it is the one that destroys the finding. Four to one is still no winner.
    """
    judges = [_j("k", 0.02, 0.17), _j("c", 0.02, 0.17),
              _j("h", 0.02, 0.17), _j("x", 0.02, 0.17), _j("dissent", 0.17, 0.02)]
    assert panel.panel_verdict(judges)["verdict"] == panel.NO_WINNER


def test_a_dissenting_judge_that_cannot_resolve_its_gap_does_not_create_a_disagreement():
    """The other side of the composition. A dissenter whose gap is below its own resolving power
    is reporting noise, and noise must not be able to veto a result any more than it can create
    one.
    """
    judges = [_j("k", 0.02, 0.17, sd=0.02, n=5),
              _j("c", 0.02, 0.17, sd=0.02, n=5),
              _j("noisy", 0.101, 0.100, sd=0.05, n=5)]
    p = panel.panel_verdict(judges)
    assert p["verdict"] == panel.A
    assert p["tie_votes"] == ["noisy"]


def test_every_judge_reporting_a_tie_is_a_tie_not_a_no_winner():
    """A tie and a disagreement are different results and must not render alike: one says the
    arms are indistinguishable, the other says the rulers are.
    """
    p = panel.panel_verdict([_j("k", 0.1, 0.1), _j("c", 0.2, 0.2), _j("h", 0.3, 0.3)])
    assert p["verdict"] == panel.TIE


def test_fewer_than_three_judges_is_not_a_panel():
    with pytest.raises(panel.PanelError, match="not a panel"):
        panel.panel_verdict([_j("k", 0.02, 0.17), _j("c", 0.02, 0.17)])


def test_the_same_judge_twice_is_refused():
    """A panel's value is that its members are different instruments. A duplicate name is either
    a mistake or one ruler counted twice, and both would inflate agreement.
    """
    with pytest.raises(panel.PanelError, match="share a name"):
        panel.panel_verdict([_j("k", 0.02, 0.17), _j("k", 0.02, 0.17), _j("c", 0.02, 0.17)])


# ── the spread, which is the measurement a single judge cannot make ──────────────
def test_the_between_judge_spread_is_reported_rather_than_averaged():
    """A panel that agrees on the winner and disagrees on the size of the win has still found
    something, and a mean would discard it.
    """
    p = panel.panel_verdict([_j("k", 0.02, 0.17),     # gap 0.15
                             _j("c", 0.30, 0.55),     # gap 0.25
                             _j("h", 0.10, 0.40)])    # gap 0.30
    assert p["gap_min"] == pytest.approx(0.15)
    assert p["gap_max"] == pytest.approx(0.30)
    assert p["gap_spread"] == pytest.approx(0.15)


def test_the_report_names_who_voted_which_way_when_there_is_no_winner():
    """A disagreement nobody can attribute is not actionable: which ruler dissented is the
    finding.
    """
    lines = "\n".join(panel.report(panel.panel_verdict(
        [_j("keyword", 0.02, 0.17), _j("compass", 0.55, 0.30), _j("held-out", 0.10, 0.40)])))
    assert "NO WINNER" in lines
    assert "keyword" in lines and "compass" in lines


def test_the_report_calls_a_disagreement_a_finding_rather_than_a_failure():
    lines = "\n".join(panel.report(panel.panel_verdict(
        [_j("k", 0.02, 0.17), _j("c", 0.55, 0.30), _j("h", 0.10, 0.40)])))
    assert "finding rather than a failure" in lines


def test_a_win_is_reported_with_the_range_rather_than_one_judges_number():
    lines = "\n".join(panel.report(panel.panel_verdict(
        [_j("k", 0.02, 0.17), _j("c", 0.30, 0.55), _j("h", 0.10, 0.40)])))
    assert "wins" in lines
    assert "NOT agreed" in lines
    assert "Quote the range" in lines


def test_the_report_says_which_judges_resolving_power_was_never_checked():
    """The silent-pass trap again: a judge with no spread must not read as a verified vote."""
    lines = "\n".join(panel.report(panel.panel_verdict(
        [_j("k", 0.02, 0.17, sd=0.02, n=5), _j("c", 0.02, 0.17, sd=0.02, n=5),
         _j("h", 0.02, 0.17)])))
    assert "NOT CHECKED" in lines
    assert "Unknown is not the same as yes" in lines


def test_the_gate_floor_is_three():
    """Pinned so that lowering it becomes a visible decision rather than an edit."""
    assert panel.MIN_JUDGES == 3


# ── the minority verdict, found by two panel reviewers independently ─────────────────────────

def _reading(name, verdict, gap, resolvable=True):
    """A judge reading, built directly rather than through judge_verdict.

    Built by hand on purpose: the defect is in how `panel_verdict` COMBINES readings, and
    routing through `judge_verdict` would make the test depend on the power arithmetic as well.
    """
    return {"judge": name, "score_a": 0.0, "score_b": 0.0, "gap": gap,
            "higher_is_better": True, "verdict": verdict, "resolvable": resolvable}


def test_one_decisive_judge_and_two_ties_is_not_a_win():
    """THE DEFECT TWO REVIEWERS REACHED SEPARATELY ON 2026-09-21.

    Judges that tie drop out of the `decided` set, so one decisive judge satisfied "nobody
    disagreed" and the panel reported that judge's verdict with `agreed: True`. Unanimity among
    one is not unanimity, and the other two did not abstain: they looked and could not find a
    difference, which makes the claim weaker than that judge alone, not stronger.
    """
    p = panel.panel_verdict([
        _reading("keyword", panel.B, 0.20),
        _reading("compass", panel.TIE, 0.01, resolvable=False),
        _reading("llm", panel.TIE, 0.01, resolvable=False),
    ])
    assert p["verdict"] == panel.NO_WINNER
    assert p["agreed"] is False
    assert p["no_winner_because"] == "under-supported"
    assert p["n_decided"] == 1


def test_the_report_never_claims_more_judges_agreed_than_reached_a_verdict():
    """The sentence a reader actually sees. It said "all 3 judges that reached a verdict agree"
    using the ROSTER size, which is false on any panel where somebody tied.
    """
    p = panel.panel_verdict([
        _reading("keyword", panel.B, 0.20),
        _reading("compass", panel.B, 0.15),
        _reading("llm", panel.TIE, 0.01, resolvable=False),
    ])
    assert p["verdict"] == panel.B and p["n_decided"] == 2
    text = "\n".join(panel.report(p))
    assert "2 of 3 judges" in text, text
    assert "all 3" not in text, text
    assert "llm" in text, "the judge that could not resolve it is not named"


def test_the_size_of_the_win_is_measured_over_the_judges_that_decided():
    """A tie-voter's gap is the number this module just ruled to be noise. Including it in the
    range reports a lower bound the panel itself does not believe.
    """
    p = panel.panel_verdict([
        _reading("keyword", panel.B, 0.20),
        _reading("compass", panel.B, 0.15),
        _reading("llm", panel.TIE, 0.001, resolvable=False),
    ])
    assert p["gap_min"] == 0.15 and p["gap_max"] == 0.20
    assert p["gap_spread"] == pytest.approx(0.05)
    assert "0.001" not in "\n".join(panel.report(p))


def test_an_all_tie_panel_sizes_no_win_at_all():
    """With nobody deciding there is no win to size, so the fields are absent rather than a
    range computed over noise.
    """
    p = panel.panel_verdict([_reading(n, panel.TIE, 0.01, resolvable=False)
                             for n in ("keyword", "compass", "llm")])
    assert p["verdict"] == panel.TIE
    assert p["gap_min"] is None and p["gap_max"] is None and p["gap_spread"] is None


def test_genuine_disagreement_still_reads_as_disagreement_not_as_thin_support():
    """The two NO WINNER reasons are different findings and must not print the same sentence."""
    p = panel.panel_verdict([
        _reading("keyword", panel.A, 0.20),
        _reading("compass", panel.B, 0.15),
        _reading("llm", panel.A, 0.10),
    ])
    assert p["verdict"] == panel.NO_WINNER
    assert p["no_winner_because"] == "disagree"
    assert "disagree" in "\n".join(panel.report(p))
