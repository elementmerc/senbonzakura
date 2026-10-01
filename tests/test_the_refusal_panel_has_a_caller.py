# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The refusal panel, and the single-ruler headline it replaces.

WHAT WAS WRONG BEFORE THIS.

`panel.py` shipped complete: the disagreement rule, the deciding floor, the between-judge spread,
and a test file over all of it. It had no caller anywhere in the package. A grep for
`panel_verdict` and `judge_verdict` across `src/` returned one comment. So the logic that decides
"the judges disagree, therefore there is no winner" existed, was tested, and could not affect any
report this tool produces.

Meanwhile the head-to-head report printed a refusal table with THREE rulers in it and a verdict
computed from the first column alone, immediately beneath. The other two instruments were on the
page for the reader to not combine. That is the single-ruler headline the panel exists to prevent,
in the shipped reporter, with the panel sitting unused one import away.

WHAT THESE CHECKS ARE FOR.

Two different failures, so they are tested apart. **The panel is reached**: the report runs it,
and a disagreement between the rulers produces no winner rather than a sentence picked from
whichever column said something. **The panel is honest about itself**: these three rulers are
related, so the spread they produce is a lower bound, and the output says that every time rather
than in a comment nobody reads. A panel that understates its own spread reads as more rigour than
one judge, which is worse than one judge honestly labelled.
"""
import json

import pytest

from senbonzakura import headtohead_report as rh
from senbonzakura import panel


def _arm(root, tool, seed, *, refusal, noncompliant, keyword):
    """One arm's artefacts: the three refusal rulers, plus the files `collect` needs to see it."""
    (root / f"scored-{tool}-seed{seed}.json").write_text(
        json.dumps({"auc": 0.987, "controls": {"length_only_auc": 0.5}}), encoding="utf-8")
    # None means the ruler has no figure for this arm, which is how a ruler drops out of the
    # panel rather than being quietly averaged over a smaller n.
    payload = {"n": 128, "broken": 0.0} | {
        key: value for key, value in (("refusal", refusal), ("noncompliant", noncompliant),
                                      ("heretic", keyword)) if value is not None}
    (root / f"refusal-{tool}-seed{seed}.json").write_text(json.dumps(payload), encoding="utf-8")
    arm = root / f"{tool}-seed{seed}"
    arm.mkdir(exist_ok=True)
    name = "abliteration.json" if tool.startswith("senbon") else "best_of_n.json"
    body = ({"post_bake_refusals": refusal, "post_bake_kl": 0.1}
            if tool.startswith("senbon") else {"winner": {"refusals": refusal, "kl": 0.1}})
    (arm / name).write_text(json.dumps(body), encoding="utf-8")


def _run(root, by_tool):
    """A run directory from `{tool: [(refusal, noncompliant, keyword), ...]}`, one per seed."""
    for tool, rows in by_tool.items():
        for i, (refusal, noncompliant, keyword) in enumerate(rows):
            _arm(root, tool, 42 + i, refusal=refusal, noncompliant=noncompliant, keyword=keyword)
    return rh.collect(str(root))


def _panel(root, by_tool):
    return rh.refusal_panel(_run(root, by_tool))


def _text(lines):
    return "\n".join(lines)


# ── the panel is reached, and it decides ─────────────────────────────────────────────────────


def test_three_rulers_that_agree_produce_a_winner_with_the_arms_named(tmp_path):
    """Senbon removes far more refusals on every ruler, and the panel says so once.

    The arms are named in the output because `panel.report` speaks in A and B, and a reader of a
    head-to-head needs to know which tool is which without counting back through the table.
    """
    out = _panel(tmp_path, {
        "senbon": [(0.02, 0.04, 0.01), (0.03, 0.05, 0.02), (0.02, 0.04, 0.01)],
        "heretic": [(0.40, 0.46, 0.38), (0.42, 0.48, 0.39), (0.41, 0.47, 0.40)],
    })
    text = _text(out)

    assert "Arm A is heretic, arm B is senbon" in text, (
        "the panel speaks in A and B, so the output has to say which tool each one is")
    # Sorted, so `heretic` is A and `senbon` is B. B is the arm with fewer refusals.
    assert "B WINS" in text.upper(), f"the panel did not find the clear winner:\n{text}"
    assert "3 of 3 judges reached a verdict" in text


def test_rulers_that_disagree_about_the_winner_produce_no_winner(tmp_path):
    """THE RULE THE WHOLE MODULE EXISTS FOR, now reachable from a report.

    Hard refusal and noncompliance favour one tool, the keyword rate favours the other, each by a
    margin its own spread can resolve. Before this wiring the report would have printed the hard
    refusal column's verdict as the answer and said nothing about the keyword column contradicting
    it.
    """
    out = _panel(tmp_path, {
        "senbon": [(0.02, 0.04, 0.40), (0.03, 0.05, 0.42), (0.02, 0.04, 0.41)],
        "heretic": [(0.40, 0.46, 0.01), (0.42, 0.48, 0.02), (0.41, 0.47, 0.01)],
    })
    text = _text(out)

    assert "NO WINNER" in text, f"the rulers disagree and the panel named a winner:\n{text}"
    assert "disagree about which arm won" in text
    # Each side of the disagreement is attributed, so a reader can see which instrument said what.
    assert "heretic-keyword" in text
    assert "hard-refusal" in text


def test_rulers_that_all_see_nothing_produce_a_tie(tmp_path):
    """Identical arms. No ruler can resolve a gap, and a tie is the honest answer, not a winner."""
    rows = [(0.20, 0.25, 0.18), (0.24, 0.30, 0.23), (0.28, 0.34, 0.27)]
    out = _panel(tmp_path, {"senbon": rows, "heretic": list(rows)})
    text = _text(out)

    assert "TIE" in text, f"identical arms did not tie:\n{text}"
    assert "no judge found a difference it could resolve" in text


def test_one_decisive_ruler_against_two_that_cannot_see_is_not_a_win(tmp_path):
    """`MIN_DECIDING` reached from a report, which is where it matters.

    One ruler separates the tools cleanly; the other two are flat, so their gaps are inside what
    they can resolve and they tie. One instrument finding a gap the others looked for and did not
    find is weaker evidence than that instrument alone, and the panel refuses to call it a win.
    """
    out = _panel(tmp_path, {
        "senbon": [(0.02, 0.22, 0.16), (0.03, 0.28, 0.22), (0.02, 0.34, 0.28)],
        "heretic": [(0.40, 0.22, 0.16), (0.42, 0.28, 0.22), (0.41, 0.34, 0.28)],
    })
    text = _text(out)

    assert "NO WINNER" in text, f"one decisive ruler was read as a win:\n{text}"
    assert "could not see a difference at all" in text


# ── the panel declines, loudly, rather than reporting a tie ──────────────────────────────────


def test_one_tool_alone_is_not_a_panel_and_is_not_a_tie(tmp_path):
    out = _panel(tmp_path, {"senbon": [(0.02, 0.04, 0.01)] * 3})
    text = _text(out)

    assert "NO PANEL on refusals" in text
    assert "needs two" in text
    assert "Nothing is being reported as a tie" in text, (
        "a comparison that could not be made must not read as a comparison that found nothing")


def test_too_few_seeds_declines_and_names_what_was_short(tmp_path):
    """Two seeds is arithmetic dressed as statistics, and the panel says which rulers were short.

    It also has to say that the single-ruler verdict printed above it is all the run supports,
    because a section that merely goes quiet leaves that verdict reading as the answer.
    """
    out = _panel(tmp_path, {
        "senbon": [(0.02, 0.04, 0.01), (0.03, 0.05, 0.02)],
        "heretic": [(0.40, 0.46, 0.38), (0.42, 0.48, 0.39)],
    })
    text = _text(out)

    assert "NO PANEL on refusals: 0 of 3 rulers" in text
    assert "hard-refusal (2 and 2 seeds)" in text
    assert "The single-ruler verdict above is all this run supports" in text


def test_a_ruler_missing_its_figures_drops_out_and_the_panel_declines(tmp_path):
    """Below three judges is not a panel, even when the rulers that remain agree.

    The keyword rate is absent from every arm, so two rulers are left. Two agreeing rulers is a
    better measurement than one and it is still not what the gate asks for, and the distinction
    is the whole reason `MIN_JUDGES` is a constant rather than a preference.
    """
    out = _panel(tmp_path, {
        "senbon": [(0.02, 0.04, None), (0.03, 0.05, None), (0.02, 0.06, None)],
        "heretic": [(0.40, 0.46, None), (0.42, 0.48, None), (0.41, 0.47, None)],
    })
    text = _text(out)

    assert "NO PANEL on refusals" in text
    assert "0 tool(s) carry a heretic-keyword figure" in text


def test_a_ruler_that_returned_the_same_figure_on_every_seed_does_not_vote(tmp_path):
    """A FLAT COLUMN IS NOT A PRECISE ONE, and it would otherwise take the report down.

    `power.detectable_gap` refuses a standard deviation of exactly zero, deliberately: it would
    make every gap resolvable, including a gap of nothing, and in this project's own experience a
    column that is identical across seeds means the scorer returned a constant rather than that
    the instrument is noiseless. `judge_verdict` passes `sd` straight through, so a flat ruler
    reaching it raises `PowerError` out of the middle of rendering a report.

    This was found by writing the checks above with constant fixtures, which is exactly the shape
    a real run with a stuck scorer would have. So the flat ruler is excluded and named, and the
    other two still produce a verdict if they can.
    """
    out = _panel(tmp_path, {
        "senbon": [(0.02, 0.04, 0.18), (0.03, 0.05, 0.18), (0.02, 0.06, 0.18)],
        "heretic": [(0.40, 0.46, 0.18), (0.42, 0.48, 0.18), (0.41, 0.47, 0.18)],
    })
    text = _text(out)

    assert "NO PANEL on refusals: 2 of 3 rulers could vote" in text
    assert "Flat across every seed, so no spread and no vote: heretic-keyword" in text


def test_rulers_describing_different_pairs_are_not_a_panel(tmp_path):
    """A panel over two different comparisons is not a panel, and agreement across them is not
    agreement.

    Three arms are present. Every ruler sees exactly two tools, and they are not the same two:
    the keyword rate is missing from `senbon` and present on `senbon-k1`, so that ruler compares a
    different pair from the one the first two rulers compared. Combining their verdicts would
    report agreement between judges that answered different questions.
    """
    for i in range(3):
        _arm(tmp_path, "heretic", 42 + i, refusal=0.40 + 0.01 * i,
             noncompliant=0.46 + 0.01 * i, keyword=0.38 + 0.01 * i)
        _arm(tmp_path, "senbon", 42 + i, refusal=0.02 + 0.01 * i,
             noncompliant=0.04 + 0.01 * i, keyword=None)
        _arm(tmp_path, "senbon-k1", 42 + i, refusal=None, noncompliant=None,
             keyword=0.01 + 0.01 * i)
    out = rh.refusal_panel(rh.collect(str(tmp_path)))
    text = _text(out)

    assert "NO PANEL on refusals" in text
    assert "A panel over different pairs is not a panel" in text


# ── the panel is honest about what it is ─────────────────────────────────────────────────────


def test_the_panel_always_says_its_own_spread_is_a_lower_bound(tmp_path):
    """THE CAVEAT THAT KEEPS THIS FROM BEING WORSE THAN ONE JUDGE.

    `noncompliance` is `is_refusal(t) or is_soft_refusal(t)`, so it CONTAINS `hard-refusal` rather
    than standing beside it, and the keyword rate is a keyword list too. Three related rulers
    agree more often than three unrelated ones, so the between-judge spread they produce is too
    small, and a too-small spread presented as the measurement reads as rigour. The gate's third
    judge reads meaning rather than strings and does not exist yet.
    """
    out = _panel(tmp_path, {
        "senbon": [(0.02, 0.04, 0.01), (0.03, 0.05, 0.02), (0.02, 0.04, 0.01)],
        "heretic": [(0.40, 0.46, 0.38), (0.42, 0.48, 0.39), (0.41, 0.47, 0.40)],
    })
    text = _text(out)

    assert "THE GATE IS NOT MET BY THIS PANEL" in text
    assert "LOWER BOUND" in text
    assert "noncompliance is hard refusal plus soft lectures" in text


def test_a_short_ruler_is_named_rather_than_dropped_in_silence(tmp_path):
    """Three rulers present, one of them short, and the panel still runs on the other two.

    It does not: two is below `MIN_JUDGES`. What this pins is that the short ruler is NAMED in the
    refusal rather than vanishing, because a panel that quietly shrinks is a panel whose reported
    judge count means nothing.
    """
    for i in range(3):
        _arm(tmp_path, "senbon", 42 + i, refusal=0.02 + 0.01 * i, noncompliant=0.04 + 0.01 * i,
             keyword=0.01 + 0.01 * i if i < 2 else None)
        _arm(tmp_path, "heretic", 42 + i, refusal=0.40 + 0.01 * i,
             noncompliant=0.46 + 0.01 * i, keyword=0.38 + 0.01 * i if i < 2 else None)
    text = _text(rh.refusal_panel(rh.collect(str(tmp_path))))

    assert "NO PANEL on refusals: 2 of 3 rulers" in text
    assert "heretic-keyword (2 and 2 seeds)" in text


# ── and it is actually in the report ─────────────────────────────────────────────────────────


def test_the_rendered_report_runs_the_panel_and_not_only_the_first_column(tmp_path):
    """THE WIRING, end to end, because a function nothing calls is what this row existed to fix.

    The rendered report has to carry the panel's section. Asserted on a disagreeing run, so the
    check fails if `render` keeps printing the hard-refusal verdict alone: that verdict names a
    winner on this data and the panel does not.
    """
    _run(tmp_path, {
        "senbon": [(0.02, 0.04, 0.40), (0.03, 0.05, 0.42), (0.02, 0.04, 0.41)],
        "heretic": [(0.40, 0.46, 0.01), (0.42, 0.48, 0.02), (0.41, 0.47, 0.01)],
    })
    text = rh.render(rh.collect(str(tmp_path)))[0]

    assert "The same comparison, put to every refusal ruler at once" in text
    assert "NO WINNER" in text, (
        "the report named a winner from one ruler while the rulers disagreed, which is the "
        "single-ruler headline this section exists to replace")
    assert "THE GATE IS NOT MET BY THIS PANEL" in text


def test_the_panel_module_now_has_a_caller(tmp_path):
    """The ledger item is "panel.py has no caller", so the absence of one is the thing to pin.

    Asserted by observing that the reporter's output could not have been produced without it: the
    no-winner-because reasoning and the `MIN_DECIDING` sentence are `panel.report`'s words, and
    nothing else in the package writes them.
    """
    assert panel.MIN_JUDGES == 3
    out = _panel(tmp_path, {
        "senbon": [(0.02, 0.22, 0.16), (0.03, 0.28, 0.22), (0.02, 0.34, 0.28)],
        "heretic": [(0.40, 0.22, 0.16), (0.42, 0.28, 0.22), (0.41, 0.34, 0.28)],
    })
    assert any(f"{panel.MIN_DECIDING} are required" in line for line in out), (
        f"the output does not carry panel.report's reasoning, so the panel was not reached:\n"
        f"{_text(out)}")


@pytest.mark.parametrize("field", [f for f, _n in rh.REFUSAL_JUDGES])
def test_every_declared_ruler_is_a_field_the_collector_actually_produces(tmp_path, field):
    """A ruler named here that `one_ruler_refusal` never writes would drop out on every run.

    It would drop out silently too: the panel would report "2 of 3 rulers" for ever and read as a
    data problem rather than a typo. So each declared field is checked against what the collector
    emits, which is the same one-list discipline the writer names follow.
    """
    _arm(tmp_path, "senbon", 42, refusal=0.02, noncompliant=0.04, keyword=0.01)
    _arm(tmp_path, "heretic", 42, refusal=0.40, noncompliant=0.46, keyword=0.38)
    arms = rh.collect(str(tmp_path))
    assert arms, "the fixture produced no arms, so this check verifies nothing"
    assert all(a.get(field) is not None for a in arms), (
        f"{field} is declared in REFUSAL_JUDGES and the collector did not produce it")
