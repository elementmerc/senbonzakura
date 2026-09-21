# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What a comparison can see, and the plan's own figure being 14% optimistic.

The v0.7 rung asks for "a stated effect size that five seeds can actually resolve" and quotes
roughly 1.8 standard deviations. That is the normal approximation. With five seeds the
distribution is t with eight degrees of freedom and the answer is 2.02, so the plan understates
what five seeds need by about 14%, in the direction that turns a tie into a published win.

Both of the rung's arithmetic claims are checked here against the module rather than taken on
trust, because a plan is a document and documents rot. They both hold.
"""
from __future__ import annotations

import math

import pytest

from senbonzakura import power


# ── the two figures the plan asserts ─────────────────────────────────────────────
def test_the_plans_figure_is_reproduced_exactly_under_its_own_assumption():
    """1.8 standard deviations at five seeds, which is the z-based number the rung quotes."""
    got = power.detectable_gap(1.0, 5, normal=True)
    assert got == pytest.approx(1.772, abs=0.001)


def test_using_t_rather_than_z_makes_the_requirement_larger_not_smaller():
    """THE CORRECTION, and the direction is the point.

    Five seeds do not give a normal distribution. Using t moves the detectable gap from 1.77 to
    2.02 standard deviations, so a comparison sized on the plan's figure is underpowered for a
    gap it believes it can see. Erring the other way would merely waste GPU time; erring this way
    publishes ties as wins.
    """
    z = power.detectable_gap(1.0, 5, normal=True)
    t = power.detectable_gap(1.0, 5)
    assert t == pytest.approx(2.021, abs=0.001)
    assert t > z
    assert (t - z) / z == pytest.approx(0.14, abs=0.01)


def test_the_plans_three_seed_warning_is_arithmetically_true():
    """The rung says three seeds give a variance interval running "from about half to six times
    the point estimate". Checked rather than repeated: 0.52 to 6.29.
    """
    lo, hi = power.sd_interval(1.0, 3)
    assert lo == pytest.approx(0.52, abs=0.01)
    assert hi == pytest.approx(6.29, abs=0.01)


# ── the shape of the answer ──────────────────────────────────────────────────────
def test_more_seeds_resolve_a_smaller_gap():
    gaps = [power.detectable_gap(1.0, n) for n in (3, 5, 7, 11)]
    assert gaps == sorted(gaps, reverse=True), gaps


def test_the_detectable_gap_scales_with_the_spread():
    """Linear in sd, which is what makes the figure quotable in units of the spread."""
    assert power.detectable_gap(2.0, 5) == pytest.approx(2 * power.detectable_gap(1.0, 5))


def test_a_zero_spread_resolves_any_gap():
    """The degenerate case, and it is real: three byte-identical re-sampled Optuna points were
    recovered on 2026-09-21, giving a measured within-arm noise floor of exactly zero.
    """
    assert power.detectable_gap(0.0, 5) == 0.0
    assert power.can_resolve(0.001, 0.0, 5)["resolvable"] is True


# ── the refusals, which are the reason this is a module and not a formula ────────
def test_an_untabulated_seed_count_is_refused_rather_than_interpolated():
    """An interpolated critical value looks authoritative and is not, which is the property every
    wrong number this project has published also had.
    """
    with pytest.raises(power.PowerError, match="no tabulated critical value"):
        power.detectable_gap(1.0, 22)   # df 42, between the contiguous run and the sparse tail


def test_one_seed_cannot_produce_a_spread_and_says_so():
    with pytest.raises(power.PowerError, match="cannot produce a spread"):
        power.detectable_gap(1.0, 1)


def test_a_negative_spread_is_refused():
    with pytest.raises(power.PowerError, match="cannot be negative"):
        power.detectable_gap(-1.0, 5)


# ── the verdict ──────────────────────────────────────────────────────────────────
def test_an_underpowered_comparison_is_told_how_many_seeds_it_would_need():
    """A bare False sends somebody to re-run the same underpowered comparison."""
    v = power.can_resolve(gap=0.5, sd=1.0, n_per_arm=5)
    assert v["resolvable"] is False
    assert v["seeds_needed"] and v["seeds_needed"] > 5
    assert power.detectable_gap(1.0, v["seeds_needed"]) <= 0.5


def test_a_gap_smaller_than_any_practical_sample_returns_no_seed_count():
    """Past a point the honest answer is "not with seeds": the instrument is the problem."""
    v = power.can_resolve(gap=0.001, sd=1.0, n_per_arm=5)
    assert v["resolvable"] is False
    assert v["seeds_needed"] is None
    assert "better instrument" in "\n".join(power.report(v))


def test_the_verdict_carries_the_interval_on_its_own_spread():
    """A comparison resolvable only at the optimistic end of the spread is not resolvable, and
    a single detectable-gap figure hides that.
    """
    v = power.can_resolve(gap=2.5, sd=1.0, n_per_arm=5)
    lo, hi = v["detectable_gap_interval"]
    assert lo < v["detectable_gap"] < hi


def test_the_hephaestus_kv_case_comes_back_not_resolvable():
    """THE REAL CASE THIS MODULE WAS BUILT FOR.

    Three KV configurations looked cleanly ordered across ECE 0.123 to 0.140, an arm spread of
    0.017. Two runs of the IDENTICAL baseline then scored 0.103 and 0.157, so the within-arm
    spread is around 0.038 at the crudest reading. Asked whether a 0.017 gap is visible against
    that, the answer has to be no, and it has to be no loudly enough that nobody reports the
    ordering.
    """
    v = power.can_resolve(gap=0.017, sd=0.038, n_per_arm=5)
    assert v["resolvable"] is False
    lines = "\n".join(power.report(v))
    assert "NOT resolvable" in lines
    assert "reported as a tie" in lines


def test_the_report_says_what_it_has_not_established():
    """A resolvable gap is not a real one, and a reader who takes it as one has been misled by
    a number that was correct.
    """
    lines = "\n".join(power.report(power.can_resolve(gap=3.0, sd=1.0, n_per_arm=5)))
    assert "resolvable" in lines
    assert "not what is true" in lines
    assert "replication of the control arm" in lines


def test_every_tabulated_critical_pair_is_ordered_and_plausible():
    """A transcription slip in the table would silently change every figure this module reports.

    Both quantiles fall with degrees of freedom and converge on their normal limits, so the table
    is checked for that shape rather than value by value against a source it cannot reach.
    """
    dfs = sorted(power._T)
    firsts = [power._T[d][0] for d in dfs]
    seconds = [power._T[d][1] for d in dfs]
    assert firsts == sorted(firsts, reverse=True), firsts
    assert seconds == sorted(seconds, reverse=True), seconds
    assert firsts[-1] == pytest.approx(1.96, abs=0.03)
    assert seconds[-1] == pytest.approx(0.84, abs=0.02)
    assert pytest.approx(1.959964 + 0.841621, abs=1e-6) == power.NORMAL_CONSTANT


def test_every_tabulated_chi_square_pair_brackets_its_degrees_of_freedom():
    """The mean of a chi-square is its df, so each interval must contain it. A swapped pair or a
    misplaced decimal fails here rather than in a published interval.
    """
    for df, (lo, hi) in power._CHI2.items():
        assert lo < df < hi, (df, lo, hi)


def test_the_interval_on_the_spread_narrows_as_seeds_are_added():
    widths = []
    for n in (3, 5, 7, 11):
        lo, hi = power.sd_interval(1.0, n)
        widths.append(hi - lo)
    assert widths == sorted(widths, reverse=True), widths


def test_the_gap_and_the_seeds_needed_agree_with_each_other():
    """The two entry points must not disagree: whatever `seeds_for` returns has to actually
    resolve the gap, or one of them is wrong and both look right.
    """
    for gap in (0.5, 1.0, 1.5, 2.5):
        n = power.seeds_for(gap, 1.0)
        if n is None:
            continue
        assert abs(gap) >= power.detectable_gap(1.0, n)
        if 2 * (n - 2) in power._T and n > 2:
            assert abs(gap) < power.detectable_gap(1.0, n - 1) or n == 2


def test_the_module_needs_nothing_heavier_than_the_standard_library():
    """It is imported by the reporting path, which must stay light. scipy would give exact
    quantiles and would also make this unusable where it is needed.
    """
    import inspect
    src = inspect.getsource(power)
    for heavy in ("import scipy", "import numpy", "import torch"):
        assert heavy not in src, heavy
    assert math  # the one dependency, and it is in the standard library


# ── the refusals, which had never been executed on any runner ────────────────────

def test_one_seed_is_refused_because_there_is_no_spread_to_bound():
    """A standard deviation from a single observation is not small, it does not exist. Returning
    an interval anyway would hand a reader a width computed from nothing, and a width is exactly
    what this function is consulted for.
    """
    with pytest.raises(power.PowerError, match="no spread"):
        power.sd_interval(1.0, 1)


def test_an_untabulated_seed_count_is_refused_rather_than_interpolated():
    """The chi-square quantiles are a table, and a count the table does not hold is a question
    this module cannot answer. Interpolating between tabulated rows would produce a plausible
    interval with no distribution behind it, which is the failure mode the whole module exists
    to argue against: the plan's own 1.8s figure was the normal approximation wearing a t label.
    """
    untabulated = next(n for n in range(2, 400) if n - 1 not in power._CHI2)
    with pytest.raises(power.PowerError, match="no tabulated chi-square quantiles"):
        power.sd_interval(1.0, untabulated)


def test_a_gap_of_zero_needs_no_seeds_because_it_is_not_a_gap():
    """Asked how many seeds resolve a difference of nothing, the honest answer is that no number
    of seeds resolves it. Falling through the table would return the smallest tabulated count,
    which reads as "five seeds will do" for a comparison that has nothing to detect.
    """
    assert power.seeds_for(0, 1.0) is None


def test_a_gap_beyond_the_table_is_out_of_reach_rather_than_rounded_to_the_largest_count():
    """The other end of the same honesty. A gap far below what the largest tabulated seed count
    resolves gets None, which the report renders as "a better instrument, not a longer night".
    """
    tiny = power.detectable_gap(1.0, max(power.TABULATED_SEEDS)) / 100
    assert power.seeds_for(tiny, 1.0) is None
