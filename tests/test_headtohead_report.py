# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for `senbonzakura head-to-head report`, which turns a finished head-to-head into a claim.

The tests are about the claims, not the layout: that a gap inside the noise is called a tie, that
two seeds cannot buy a verdict, that a missing arm fails the report rather than shrinking the
table quietly, and above all that the two tools' own KL figures are never presented as one
comparable column. That last one is the error this project withdrew four claims for.
"""
import json
import re

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
    # FIVE seeds per arm, not three. At three there are twenty ways to split six observations,
    # so no permutation test can clear 0.05 and the report now says the comparison could not have
    # concluded rather than calling it a tie. The regression this test is named for is about the
    # tie path, so it needs enough seeds to reach it.
    d = run_dir([0.90, 0.91, 0.92, 0.905, 0.915], [0.89, 0.90, 0.91, 0.895, 0.905])
    assert "TIE on harm recognition" in report(d)


def test_a_margin_too_small_to_print_is_a_tie(run_dir):
    """A win the reader cannot verify from the figures shown is not a win.

    The gap clears the spread here, but by far less than the fourth decimal both are printed to, so
    the verdict sentence would read as two identical numbers with a winner between them.
    """
    # Five seeds per arm: at three, no permutation test can clear 0.05, so the report
    # correctly refuses a verdict and this test's subject is unreachable.
    d = run_dir([0.900001, 0.910001, 0.920001, 0.905001, 0.915001],
                [0.89, 0.90, 0.91, 0.895, 0.905])
    assert "TIE on harm recognition" in report(d)


def test_a_gap_clearing_the_spread_names_a_winner(run_dir):
    d = run_dir([0.95, 0.94, 0.96, 0.95, 0.94], [0.60, 0.61, 0.59, 0.60, 0.62])
    out = report(d)
    assert "TIE" not in out
    assert "senbon is ahead of heretic on harm recognition" in out


def test_the_winner_can_be_either_tool(run_dir):
    d = run_dir([0.60, 0.61, 0.59, 0.60, 0.62], [0.95, 0.94, 0.96, 0.95, 0.94])
    assert "heretic is ahead of senbon on harm recognition" in report(d)


def test_a_zero_spread_is_flagged_rather_than_trusted(run_dir):
    """Five seeds landing on one number is usually a seed that varied nothing."""
    # Five seeds per arm: at three, no permutation test can clear 0.05, so the report
    # correctly refuses a verdict and this test's subject is unreachable.
    d = run_dir([0.95] * 5, [0.60] * 5)
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
    # Five seeds per arm: at three, no permutation test can clear 0.05, so the report
    # correctly refuses a verdict and this test's subject is unreachable.
    d = run_dir([0.9] * 5, [0.9] * 5)
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
    out = report(run_dir([0.9] * 5, [0.8] * 5))
    assert "NOT a comparison" in out
    assert "senbonzakura, our coherence slice" in out
    # These fixtures carry no `kl_source`, so the Heretic row is reported as UNLABELLED rather
    # than given a plausible owner. Before the selection pass existed, hardcoding "Heretic, its
    # own evaluation" here was right; afterwards it was printed over a figure our own estimator
    # had produced. An unstamped number is now named as unstamped, which is the only honest
    # answer available from the artefact alone.
    assert "UNRECORDED" in out
    assert "Heretic, its own evaluation" not in out


def test_every_self_reported_row_carries_its_estimator(run_dir):
    out = report(run_dir([0.9] * 5, [0.8] * 5))
    # Five seeds per tool, two tools: one line per arm.
    assert out.count("KL estimator:") == 10
    assert out.count("refusal estimator:") == 10


def test_the_verdict_reads_the_compass_not_the_self_reported_numbers(run_dir):
    """The compass is one instrument over both tools; the self-reported figures are not."""
    # Five seeds per arm: at three, no permutation test can clear 0.05, so the report
    # correctly refuses a verdict and this test's subject is unreachable.
    d = run_dir([0.95] * 5, [0.60] * 5)
    assert "senbon is ahead" in report(d)


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
    assert "senbon is ahead" in out
    assert "  heretic    n=3" in out


def test_a_run_with_no_own_pick_still_reports(run_dir):
    """The selection pass writes no second model when it agreed with Heretic's own choice."""
    out = report(run_dir([0.9] * 5, [0.8] * 5))
    assert "own-pick" not in out


# ── the arithmetic ────────────────────────────────────────────────────────────────────
def test_stdev_of_one_point_is_zero_and_never_divided_by(run_dir):
    assert rh.stdev([0.9]) == 0.0
    d = run_dir([0.9], [0.8])
    assert "NO VERDICT" in report(d)


def test_the_pooled_spread_uses_both_arms(run_dir):
    """A silent arm beside a noisy one must not report the silent arm's spread as the spread."""
    # Five per arm: at three no verdict is reachable, so the sentence carrying the spread is
    # never printed and this test would be asserting on a string that cannot appear.
    d = run_dir([0.90] * 5, [0.60, 0.95, 0.75, 0.70, 0.85])
    out = report(d)
    # One arm has zero spread; the pooled figure must still carry the other arm's noise.
    # Asserted on the VALUE rather than on a literal string, because the value depends on how
    # many seeds the test uses and the property does not.
    m = re.search(r"pooled spread(?: of)? ([0-9.]+)", out)
    assert m, f"the verdict printed no pooled spread:\n{out[-400:]}"
    assert float(m.group(1)) > 0.05, (
        f"the pooled spread came back {m.group(1)}, which is the silent arm's spread rather "
        f"than both arms'")


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


def _arms(tool, values):
    return [{"tool": tool, "seed": i, "auc": v} for i, v in enumerate(values)]


def test_more_seeds_no_longer_make_a_real_effect_harder_to_declare():
    """THE GATE THAT GOT STRICTER AS EVIDENCE ACCUMULATED.

    The old rule was `gap > pooled_spread`, which never looks at how many seeds there are. Since
    the standard error of a difference shrinks as sqrt(n), that rule fires at |t| > 1.58 with
    five seeds per arm and |t| > 5.0 with fifty. Running ten times as many seeds therefore made a
    real effect HARDER to declare, while the report described the rule as a conservative gate
    against noise.

    Here the same effect and the same spread are measured with five seeds and with twenty. A
    correct gate is at least as willing to call it with more data; the old one was less.
    """
    from senbonzakura import headtohead_report as hh

    small = {"alpha": _arms("alpha", [0.50, 0.52, 0.54, 0.56, 0.58]),
             "beta": _arms("beta", [0.60, 0.62, 0.64, 0.66, 0.68])}
    big_a = [0.50 + 0.02 * (i % 5) for i in range(20)]
    big_b = [0.60 + 0.02 * (i % 5) for i in range(20)]
    large = {"alpha": _arms("alpha", big_a), "beta": _arms("beta", big_b)}

    v_small, v_large = hh.verdict(small), hh.verdict(large)
    assert "TIE" not in v_large, (
        f"twenty seeds per arm on a ten-point gap reported a tie:\n{v_large}")
    if "TIE" not in v_small:
        assert "TIE" not in v_large, "more evidence must not turn a win into a tie"


def test_the_verdict_reports_a_p_value_rather_than_a_ratio_of_two_summaries():
    from senbonzakura import headtohead_report as hh

    by_tool = {"alpha": _arms("alpha", [0.50, 0.52, 0.54, 0.56, 0.58]),
               "beta": _arms("beta", [0.60, 0.62, 0.64, 0.66, 0.68])}
    said = hh.verdict(by_tool)
    assert "permutation p=" in said, said
    assert "pooled spread" in said, "the spread is still worth showing a reader"


def test_a_gap_inside_the_noise_is_still_a_tie():
    """The fix must not turn the gate into a rubber stamp."""
    from senbonzakura import headtohead_report as hh

    by_tool = {"alpha": _arms("alpha", [0.50, 0.60, 0.70, 0.40, 0.55]),
               "beta": _arms("beta", [0.52, 0.61, 0.68, 0.43, 0.57])}
    assert "TIE" in hh.verdict(by_tool)


def test_a_win_that_does_not_clear_the_spread_says_so():
    """The two comparators can disagree, and the disagreement is printed rather than resolved."""
    from senbonzakura import headtohead_report as hh

    by_tool = {"alpha": _arms("alpha", [0.50, 0.51, 0.52, 0.53, 0.54]),
               "beta": _arms("beta", [0.56, 0.57, 0.58, 0.59, 0.60])}
    said = hh.verdict(by_tool)
    if "TIE" not in said:
        assert "permutation p=" in said


@pytest.mark.parametrize(("n", "possible"), [(2, False), (3, False), (4, True), (5, True)])
def test_a_comparison_that_could_not_have_concluded_says_so(n, possible):
    """AN INCONCLUSIVE RESULT AND AN IMPOSSIBLE ONE ARE DIFFERENT THINGS.

    At three seeds per arm there are twenty ways to split six observations, so the smallest
    two-sided p reachable is 0.10. Nothing in the data can clear 0.05. Calling that a tie would
    let a reader believe the two tools had been found equal, when the truth is that the run was
    never capable of finding anything.
    """
    from senbonzakura import headtohead_report as hh

    a = [0.50 + 0.001 * i for i in range(n)]
    b = [0.90 + 0.001 * i for i in range(n)]      # separated as cleanly as data can be
    said = hh.verdict({"alpha": _arms("alpha", a), "beta": _arms("beta", b)})
    if possible:
        assert "NO VERDICT POSSIBLE" not in said, said
    else:
        assert "NO VERDICT" in said, said


def test_a_p_that_rounds_to_zero_is_printed_as_a_bound():
    """A permutation test never returns zero: the observed arrangement is one of them."""
    from senbonzakura import headtohead_report as hh

    a = [0.10 + 0.001 * i for i in range(20)]
    b = [0.90 + 0.001 * i for i in range(20)]
    said = hh.verdict({"alpha": _arms("alpha", a), "beta": _arms("beta", b)})
    assert "p=0.000" not in said, "printing p=0.000 claims a certainty the method cannot express"
    assert "p=<0.001" in said


# ─────────────────────────────────────────────────────────────────────────────────────
# The two axes that printed a mean under the word "comparable" and had no test at all.
#
# Reached independently by two personas on 2026-09-10. Only the compass AUC ever went
# through `verdict()`, and our own AUC spans 0.9860 to 0.9887 across ten arms and a
# 32.8-point refusal swing, so the report was structurally guaranteed to return TIE on
# the axis it tested and to hand the reader an untested mean-versus-mean table on the
# two axes where a claim actually gets made. That is the route by which "roughly half
# the collateral damage" reached the published site.
# ─────────────────────────────────────────────────────────────────────────────────────

def _run_with_drift(tmp_path, sen, her, refusals=None):
    """A run directory carrying compass, drift and refusal artefacts for each arm."""
    for tool, xs in (("senbon", sen), ("heretic", her)):
        for i, kl in enumerate(xs):
            seed = 42 + i
            (tmp_path / f"scored-{tool}-seed{seed}.json").write_text(json.dumps(
                {"auc": 0.987, "controls": {"length_only_auc": 0.5}}), encoding="utf-8")
            (tmp_path / f"drift-{tool}-seed{seed}.json").write_text(json.dumps(
                {"kl": kl, "n_prompts": 200, "kl_ci": [kl * 0.8, kl * 1.2],
                 "kl_ci_method": "percentile bootstrap over prompts", "precision_ok": True}),
                encoding="utf-8")
            r = (refusals or {}).get(tool, [0.0] * len(xs))[i]
            (tmp_path / f"refusal-{tool}-seed{seed}.json").write_text(json.dumps(
                {"refusal": r, "n": 128, "heretic": 0.02, "broken": 0.0, "noncompliant": 0.03}),
                encoding="utf-8")
            arm = tmp_path / f"{tool}-seed{seed}"
            arm.mkdir(exist_ok=True)
            if tool == "senbon":
                (arm / "abliteration.json").write_text(json.dumps(
                    {"post_bake_refusals": r, "post_bake_kl": kl}), encoding="utf-8")
            else:
                (arm / "best_of_n.json").write_text(json.dumps(
                    {"winner": {"refusals": r, "kl": kl}}), encoding="utf-8")
    return rh.render(rh.collect(str(tmp_path)))[0]


def test_the_drift_axis_now_gets_a_test_and_not_just_two_means(tmp_path):
    """THE DEFECT, with the real per-seed values from the 2026-09-10 run.

    Means 0.0545 against 0.1227 read as "half the collateral damage". One seed at 0.3591 against
    siblings of 0.0439 to 0.1075 carries all of it, and a permutation test says so: p = 0.238.

    These ten numbers were APPROXIMATED FROM MEMORY when this test was first written, because the
    arms were still on the ROG and unreachable. The report driven by them printed p = 0.405, and
    that figure went onto the public benchmark page as a measurement. It was a property of my
    fixture, not of the run. The arms were fetched the same evening and these are the real values;
    the conclusion was unchanged and the strength of it was overstated. A fixture is a claim about
    the world, and a report driven by one tells you about your claim.
    """
    out = _run_with_drift(tmp_path,
                          [0.0363, 0.0406, 0.0510, 0.0936, 0.0512],
                          [0.1075, 0.3591, 0.0439, 0.0573, 0.0456])
    assert "TIE on coherence drift" in out, (
        "a gap this size across these seeds is not evidence, and the report has to say so where "
        "the reader meets the means")
    assert "permutation p=" in out


def test_a_real_drift_difference_is_still_reported(tmp_path):
    """The gate must not be so blunt that nothing can ever be found."""
    out = _run_with_drift(tmp_path,
                          [0.0510, 0.0498, 0.0505, 0.0512, 0.0501],
                          [0.3010, 0.2998, 0.3005, 0.3012, 0.3001])
    assert "senbon is ahead of heretic on coherence drift" in out
    assert "TIE on coherence drift" not in out


def test_lower_drift_is_the_better_direction(tmp_path):
    """Unlike AUC, less is better here. Getting this backwards would invert the headline."""
    out = _run_with_drift(tmp_path,
                          [0.3010, 0.2998, 0.3005, 0.3012, 0.3001],
                          [0.0510, 0.0498, 0.0505, 0.0512, 0.0501])
    assert "heretic is ahead of senbon on coherence drift" in out


def test_the_refusal_axis_gets_the_same_treatment(tmp_path):
    out = _run_with_drift(tmp_path, [0.05] * 5, [0.05] * 5,
                          refusals={"senbon": [0.00, 0.00, 0.00, 0.00, 0.005],
                                    "heretic": [0.0] * 5})
    assert "refusals removed" in out
    assert "TIE on refusals removed" in out


def test_the_median_is_printed_beside_the_mean(tmp_path):
    """The outlier only became visible when somebody looked past the mean."""
    out = _run_with_drift(tmp_path,
                          [0.0512, 0.0498, 0.0605, 0.0533, 0.0577],
                          [0.0439, 0.0573, 0.1075, 0.0457, 0.3591])
    assert "median 0.0573" in out, "heretic's median, against a mean of 0.1227"
    assert "median 0.0533" in out


def test_the_interval_drift_already_wrote_finally_reaches_the_reader(tmp_path):
    """`drift.py` has written kl_ci into every artefact since the axis was found to have no
    uncertainty at all, and until 2026-09-10 nothing outside that file read it.
    """
    out = _run_with_drift(tmp_path, [0.05] * 5, [0.06] * 5)
    assert "95% interval" in out
    assert "[0.0400,0.0600]" in out, "the per-arm interval, not just the point estimate"


def test_an_arm_missing_its_figure_is_counted_rather_than_silently_dropped(tmp_path):
    """Five arms and two arms printed side by side with no note that the n's differed."""
    _run_with_drift(tmp_path, [0.05] * 5, [0.06] * 5)
    for seed in (44, 45, 46):
        (tmp_path / f"drift-heretic-seed{seed}.json").unlink()
    out = rh.render(rh.collect(str(tmp_path)))[0]
    assert "3 arm(s) had no figure and are NOT in this mean" in out
