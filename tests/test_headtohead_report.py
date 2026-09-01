# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for `senbonzakura head-to-head report`, which turns a finished head-to-head into a claim.

The tests are about the claims, not the layout: that a gap inside the noise is called a tie, that
two seeds cannot buy a verdict, that a missing arm fails the report rather than shrinking the
table quietly, and above all that the two tools' own KL figures are never presented as one
comparable column. That last one is the error this project withdrew four claims for.
"""
import json

import pytest

from senbonzakura import headtohead_report as rh


@pytest.fixture
def run_dir(tmp_path):
    def build(senbon_aucs, heretic_aucs, own_pick=None, unreadable=False):
        for tool, aucs in (("senbon", senbon_aucs), ("heretic", heretic_aucs)):
            for i, auc in enumerate(aucs):
                seed = 42 + i
                (tmp_path / f"scored-{tool}-seed{seed}.json").write_text(json.dumps({
                    "auc": auc, "auc_ci": [auc - 0.02, auc + 0.02],
                    "controls": {"length_only_auc": 0.60},
                }), encoding="utf-8")
                arm = tmp_path / f"{tool}-seed{seed}"
                arm.mkdir(exist_ok=True)
                if tool == "senbon":
                    (arm / "abliteration.json").write_text(json.dumps({
                        "post_bake_refusals": 0.03, "post_bake_kl": 0.21}), encoding="utf-8")
                else:
                    (arm / "best_of_n.json").write_text(json.dumps({
                        "winner": {"refusals": 0.05, "kl": 0.004}}), encoding="utf-8")
        if own_pick is not None:
            (tmp_path / "scored-heretic-seed42-own-pick.json").write_text(json.dumps({
                "auc": own_pick, "auc_ci": [own_pick - 0.02, own_pick + 0.02],
                "controls": {"length_only_auc": 0.60}}), encoding="utf-8")
        if unreadable:
            (tmp_path / "scored-senbon-seed99.json").write_text("{}", encoding="utf-8")
        return tmp_path
    return build


def report(directory, *args):
    arms = rh.collect(str(directory))
    return rh.render(arms)[0]


# ── the tie rule, which is the gate's own words ───────────────────────────────────────
def test_a_gap_smaller_than_the_spread_is_a_tie(run_dir):
    """The exit gate says so explicitly, and "suggestive with a spread" is publishable."""
    d = run_dir([0.90, 0.80, 0.85, 0.95, 0.75], [0.88, 0.78, 0.83, 0.93, 0.73])
    assert "TIE on harm recognition" in report(d)


def test_a_gap_equal_to_the_spread_is_a_tie(run_dir):
    """The regression from 2026-08-13, where the K experiment named a winner on a tie.

    Both arms have a standard deviation of 0.01 and their means are 0.01 apart, so the gap and the
    pooled spread are the same number. Floating point put the gap a few parts in a quadrillion
    above the spread, `gap < spread` was therefore false, and the report printed "gap 0.0010
    against a pooled spread of 0.0010" and declared a result.
    """
    d = run_dir([0.90, 0.91, 0.92], [0.89, 0.90, 0.91])
    assert "TIE on harm recognition" in report(d)


def test_a_margin_too_small_to_print_is_a_tie(run_dir):
    """A win the reader cannot verify from the figures shown is not a win.

    The gap clears the spread here, but by far less than the fourth decimal both are printed to, so
    the verdict sentence would read as two identical numbers with a winner between them.
    """
    d = run_dir([0.900001, 0.910001, 0.920001], [0.89, 0.90, 0.91])
    assert "TIE on harm recognition" in report(d)


def test_a_gap_clearing_the_spread_names_a_winner(run_dir):
    d = run_dir([0.95, 0.94, 0.96, 0.95, 0.94], [0.60, 0.61, 0.59, 0.60, 0.62])
    out = report(d)
    assert "TIE" not in out
    assert "senbon scores higher on harm recognition than heretic" in out


def test_the_winner_can_be_either_tool(run_dir):
    d = run_dir([0.60, 0.61, 0.59, 0.60, 0.62], [0.95, 0.94, 0.96, 0.95, 0.94])
    assert "heretic scores higher on harm recognition than senbon" in report(d)


def test_a_zero_spread_is_flagged_rather_than_trusted(run_dir):
    """Five seeds landing on one number is usually a seed that varied nothing."""
    d = run_dir([0.95, 0.95, 0.95], [0.60, 0.60, 0.60])
    out = report(d)
    assert "spread is exactly" in out and "never varied anything" in out


# ── refusing to answer is an answer ───────────────────────────────────────────────────
def test_two_seeds_cannot_buy_a_verdict(run_dir):
    """A spread from two points is arithmetic dressed as statistics."""
    d = run_dir([0.90, 0.80], [0.70, 0.60])
    out = report(d)
    assert "NO VERDICT" in out and "spread is not an estimate" in out


def test_identical_scores_with_no_gap_are_reported_as_suspicious(run_dir):
    """Both tools on one number, seed after seed, is a measurement fault rather than a tie."""
    d = run_dir([0.9, 0.9, 0.9], [0.9, 0.9, 0.9])
    out = report(d)
    assert "NO VERDICT" in out and "investigate" in out


def test_one_tool_alone_is_not_a_head_to_head(tmp_path):
    for seed in (42, 43, 44):
        (tmp_path / f"scored-senbon-seed{seed}.json").write_text(
            json.dumps({"auc": 0.9, "controls": {}}), encoding="utf-8")
    assert "a head-to-head needs two" in report(tmp_path)


# ── the error this file exists to avoid ───────────────────────────────────────────────
def test_the_two_kl_figures_are_never_offered_as_a_comparison(run_dir):
    """Different estimators, different slices. One column would repeat the withdrawn claims."""
    out = report(run_dir([0.9, 0.9, 0.9], [0.8, 0.8, 0.8]))
    assert "NOT a comparison" in out
    assert "senbonzakura, our coherence slice" in out
    assert "Heretic, its own evaluation" in out


def test_every_self_reported_row_carries_its_estimator(run_dir):
    out = report(run_dir([0.9, 0.9, 0.9], [0.8, 0.8, 0.8]))
    assert out.count("KL estimator:") == 6
    assert out.count("refusal estimator:") == 6


def test_the_verdict_reads_the_compass_not_the_self_reported_numbers(run_dir):
    """The compass is one instrument over both tools; the self-reported figures are not."""
    d = run_dir([0.95, 0.95, 0.95], [0.60, 0.60, 0.60])
    assert "senbon scores higher" in report(d)


# ── an incomplete run must not read as a complete one ─────────────────────────────────
def test_an_unreadable_arm_fails_the_report(run_dir):
    d = run_dir([0.9, 0.9, 0.9], [0.8, 0.8, 0.8], unreadable=True)
    assert rh.main([str(d)]) == 1


def test_an_unreadable_arm_can_be_waived_explicitly(run_dir):
    d = run_dir([0.9, 0.9, 0.9], [0.8, 0.8, 0.8], unreadable=True)
    assert rh.main([str(d), "--allow-unreadable"]) == 0


def test_an_unreadable_arm_is_named_rather_than_dropped(run_dir):
    out = report(run_dir([0.9, 0.9, 0.9], [0.8, 0.8, 0.8], unreadable=True))
    assert "produced nothing readable" in out and "scored-senbon-seed99" in out


def test_an_empty_directory_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        rh.main([str(tmp_path)])
    assert "no scored arms" in str(e.value)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        rh.main([str(tmp_path / "nowhere")])
    assert "no directory" in str(e.value)


# ── Heretic's own pick is shown but never voted with ──────────────────────────────────
def test_heretics_own_pick_is_shown_but_kept_out_of_the_verdict(run_dir):
    """It is Heretic as shipped, not the arm the equal budget defines.

    Given a wildly better own-pick, the verdict must not move: it answers a different question.
    """
    d = run_dir([0.95, 0.95, 0.95], [0.60, 0.60, 0.60], own_pick=0.99)
    out = report(d)
    assert "scored-heretic-seed42-own-pick" in out
    assert "senbon scores higher" in out
    assert "  heretic    n=3" in out


def test_a_run_with_no_own_pick_still_reports(run_dir):
    """The selection pass writes no second model when it agreed with Heretic's own choice."""
    out = report(run_dir([0.9, 0.9, 0.9], [0.8, 0.8, 0.8]))
    assert "own-pick" not in out


# ── the arithmetic ────────────────────────────────────────────────────────────────────
def test_stdev_of_one_point_is_zero_and_never_divided_by(run_dir):
    assert rh.stdev([0.9]) == 0.0
    d = run_dir([0.9], [0.8])
    assert "NO VERDICT" in report(d)


def test_the_pooled_spread_uses_both_arms(run_dir):
    """A silent arm beside a noisy one must not report the silent arm's spread as the spread."""
    d = run_dir([0.90, 0.90, 0.90], [0.60, 0.95, 0.75])
    out = report(d)
    # One arm has zero spread; the pooled figure must still carry the other arm's noise.
    assert "pooled spread of 0.1" in out or "pooled spread 0.1" in out


def test_a_spread_of_identical_floats_is_treated_as_zero():
    """`stdev([0.95, 0.95, 0.95])` is 2.7e-16, not 0.0.

    The mean of three identical floats is not exactly that float, so an `== 0.0` check never
    fired and the verdict printed "pooled spread of 0.0000" beside a winner it had no business
    naming. Comparing a computed float with equality is the bug; the tolerance is the fix.
    """
    assert rh.stdev([0.95, 0.95, 0.95]) != 0.0
    assert rh.stdev([0.95, 0.95, 0.95]) < rh.SPREAD_IS_ZERO


# ── control arms: the number escapes more easily than the weights ─────────────────────
def _make_control(directory, tool, seed, layers=(0, 1)):
    """An arm whose model directory declares itself a partial abliteration."""
    import os
    d = directory / f"{tool}-seed{seed}"
    os.makedirs(d, exist_ok=True)
    (d / "abliteration.json").write_text(
        json.dumps({"ablate_conv": False, "partially_ablated_layers": list(layers)}),
        encoding="utf-8")


def test_a_partial_ablation_is_kept_out_of_every_table_and_the_verdict(run_dir):
    """Guarding the weights is not enough. A refusal rate that reaches a comparison table is read
    later by somebody holding neither the flag, the warning nor the run log.
    """
    d = run_dir([0.95, 0.94, 0.96, 0.95, 0.94], [0.60, 0.61, 0.59, 0.60, 0.62])
    _make_control(d, "heretic", 42)
    out = report(d)
    assert "scored-heretic-seed42" not in out.split("=== Harm recognition")[1]
    assert "EXCLUDED from every table and the verdict" in out
    # The per-tool line counts four, not five: the control's AUC reached no mean and no spread.
    assert "heretic    n=4" in out
    assert "senbon     n=5" in out


def test_an_excluded_control_is_announced_rather_than_dropped(run_dir):
    """A silently shorter table is its own defect: the reader has to see that the arm ran."""
    d = run_dir([0.95, 0.94, 0.96], [0.60, 0.61, 0.59])
    _make_control(d, "senbon", 43, layers=(0, 1, 3, 4))
    out = report(d)
    assert "scored-senbon-seed43" in out
    assert "4 layer(s) unedited: 0, 1, 3, 4" in out


def test_a_whole_abliteration_is_not_mistaken_for_a_control(run_dir):
    """`ablate_conv: true`, and the absence of the file at all, both mean a whole model."""
    import os
    d = run_dir([0.95, 0.94, 0.96], [0.60, 0.61, 0.59])
    whole = d / "senbon-seed42"
    os.makedirs(whole, exist_ok=True)
    (whole / "abliteration.json").write_text(json.dumps({"ablate_conv": True}), encoding="utf-8")
    out = report(d)
    assert "EXCLUDED" not in out
    assert "scored-senbon-seed42" in out


def _with_numbers(directory, tool, seed, refusal, noncompliant, drift):
    """The refusal and drift artefacts an arm produces, which the pair section reads."""
    (directory / f"refusal-{tool}-seed{seed}.json").write_text(
        json.dumps({"refusal": refusal, "noncompliant": noncompliant, "n": 200}), encoding="utf-8")
    (directory / f"drift-{tool}-seed{seed}.json").write_text(
        json.dumps({"kl": drift, "prompts": "p.txt", "n_prompts": 200}), encoding="utf-8")


def test_a_control_and_its_pair_are_shown_together_somewhere(run_dir):
    """The regression found by the 2026-08-16 rehearsal.

    Excluding partial arms from every table is right, and on an experiment whose whole design is
    one arm against its own control it left one tool standing and printed "a head-to-head needs
    two". The safety property was correct and it made the experiment unreadable.
    """
    d = run_dir([0.95, 0.94, 0.96], [0.60, 0.61, 0.59])
    _make_control(d, "heretic", 42)
    _with_numbers(d, "senbon", 42, 0.05, 0.07, 0.0623)
    _with_numbers(d, "heretic", 42, 0.005, 0.005, 0.1154)
    out = report(d)

    assert "The partial-ablation comparison, which is NOT a result about any model" in out
    block = out.split("The partial-ablation comparison")[1].split("=== Harm recognition")[0]
    assert "scored-senbon-seed42" in block and "scored-heretic-seed42" in block
    assert "CONTROL" in block
    assert "5.0%" in block and "0.5%" in block
    assert "0.0623" in block and "0.1154" in block


def test_the_pair_section_says_what_the_numbers_are_not(run_dir):
    """A reader taking a half-abliterated model's refusal rate out of this block and putting it
    in a sentence has been warned in the only place the number appears.
    """
    d = run_dir([0.95, 0.94, 0.96], [0.60, 0.61, 0.59])
    _make_control(d, "heretic", 42)
    out = report(d)
    assert "half-abliterated by construction" in out
    assert "only a comparison at matched drift separates the two" in out


def test_no_pair_section_when_nothing_is_partial(run_dir):
    out = report(run_dir([0.95, 0.94, 0.96], [0.60, 0.61, 0.59]))
    assert "partial-ablation comparison" not in out
