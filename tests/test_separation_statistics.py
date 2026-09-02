# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Candidate separation statistics (Q-14, Candidate A): the `separation` module.

Distinct from `test_separation.py`, which covers the extractor's held-out machinery in `cli`
(`_halves`, `_held_out_separation`, `_null_separation_floor`). That file tests HOW a candidate is
scored; this one tests WHAT it is scored with.


The pre-registration is `private/plans/pre-registration-2026-09-02-separation-statistic.md`. These
tests are the "verified when" line of sprint 02 task A: a synthetic refusal axis separates, a
synthetic topic axis does not, and two draws from one distribution do not.

The size-calibration tests are the ones that matter. Candidate A's whole claim over the incumbent
is that its null does not move with the group size, so a test that only checks "big effect scores
big" would pass for a statistic with none of the property the candidate exists for.
"""
import pytest
import torch

from senbonzakura import separation


def _projections(n_bad, n_good, shift, seed=0, bad_scale=1.0):
    """Two synthetic clouds already projected onto an axis, separated by `shift` standard errors."""
    g = torch.Generator().manual_seed(seed)
    pb = torch.randn(n_bad, generator=g) * bad_scale + shift
    pg = torch.randn(n_good, generator=g)
    return pb, pg


def _null_keep_rate(stat, n_bad, n_good=64, draws=4000, bad_scale=1.0):
    """How often a meaningless axis clears this statistic's fixed threshold, at this group size."""
    kept = 0
    for i in range(draws):
        pb, pg = _projections(n_bad, n_good, 0.0, seed=1000 + i, bad_scale=bad_scale)
        if stat.fn(pb, pg) >= stat.threshold:
            kept += 1
    return kept / draws


# ── the sprint's verification line ─────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["cohens-d", "variance-ratio"])
def test_separates_a_refusal_axis_from_a_topic_axis(name):
    stat = separation.get(name)
    # A "refusal axis": the harmful cloud sits well away from the harmless one along it.
    refusal = stat.score(*_projections(64, 64, shift=1.5, seed=1))
    # A "topic axis": the harmful rows vary along it (twice the spread) but their MEAN does not
    # move, which is exactly the within-harmful content variance the filter is meant to reject.
    topic = stat.score(*_projections(64, 64, shift=0.0, bad_scale=2.0, seed=1))
    assert refusal >= stat.threshold, f"{name} failed to keep a real refusal axis: {refusal}"
    assert topic < stat.threshold, f"{name} kept a pure topic axis: {topic}"
    assert refusal > topic


@pytest.mark.parametrize("name", ["cohens-d", "variance-ratio"])
def test_does_not_separate_two_draws_from_one_distribution(name):
    stat = separation.get(name)
    # The control that makes the test above mean something: no effect at all, so a statistic that
    # simply returns a large number cannot pass.
    for seed in range(20):
        score = stat.score(*_projections(64, 64, shift=0.0, seed=200 + seed))
        assert score < stat.threshold, f"{name} kept a null axis at seed {seed}: {score}"


# ── Candidate A's actual claim: the null does not move with the group size ──────────
def test_variance_ratio_null_is_flat_across_group_sizes():
    """The property Candidate A exists for, and the reason it is not a reparametrisation.

    Cluster sizes in a real run differ by more than an order of magnitude (MIN_CLUSTER_ROWS is 8;
    a dominant cluster can hold hundreds), so a statistic whose null moves with size applies a
    different test to each cluster while appearing to apply one.
    """
    stat = separation.get("variance-ratio")
    rates = {n: _null_keep_rate(stat, n) for n in (8, 16, 32, 64, 128, 256)}
    for n, rate in rates.items():
        # The threshold is pre-registered as the null's 95th percentile, so the target is 5%.
        assert 0.03 <= rate <= 0.07, f"false-keep rate {rate:.1%} at n={n}, expected ~5%: {rates}"
    assert max(rates.values()) - min(rates.values()) < 0.02, f"null is not flat: {rates}"


def test_cohens_d_null_moves_with_group_size():
    """The control for the test above: the incumbent really does have the defect claimed.

    Without this, "Candidate A's null is flat" is an unfalsifiable compliment. This asserts the
    contrast is REAL and would fail if the incumbent were already size-calibrated, in which case
    Candidate A would have no reason to exist and the write-up would be wrong.
    """
    stat = separation.get("cohens-d")
    small = _null_keep_rate(stat, 8)
    large = _null_keep_rate(stat, 256)
    assert small > 0.15, f"expected the incumbent to keep many null axes at n=8, got {small:.1%}"
    assert large < 0.01, f"expected the incumbent to keep almost none at n=256, got {large:.1%}"
    assert small > large * 20


# ── the algebraic relationship, pinned so a future edit cannot quietly break it ─────
def test_variance_ratio_is_a_monotone_function_of_cohens_d_at_fixed_sizes():
    """F = d^2 (n-1)/2 for balanced groups, exactly.

    Pinned because it is the honest limit of Candidate A: among candidates of the SAME size it
    ranks identically to the incumbent, and the write-up says so. A change that broke this identity
    would mean the statistic had stopped being the ANOVA F without anyone noticing.
    """
    n = 64
    for shift in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        pb, pg = _projections(n, n, shift, seed=int(shift * 100))
        d = separation.cohens_d(pb, pg)
        f = separation.variance_ratio(pb, pg)
        assert f == pytest.approx(d**2 * (n - 1) / 2, rel=1e-4), f"shift={shift}"


def test_variance_ratio_null_is_flat_under_unequal_spread_when_the_groups_are_the_same_size():
    """The size-invariance survives unequal spread only while the groups are BALANCED.

    The companion to the test below. Together they say exactly where Candidate A's null holds.
    """
    stat = separation.get("variance-ratio")
    for scale in (0.5, 1.0, 2.0, 4.0):
        rate = _null_keep_rate(stat, 32, n_good=32, draws=2000, bad_scale=scale)
        assert 0.03 <= rate <= 0.07, f"sd ratio {scale} broke the balanced null: {rate:.1%}"


def test_variance_ratio_null_breaks_when_the_groups_are_both_unbalanced_and_unequally_spread():
    """THE KNOWN LIMIT OF CANDIDATE A, pinned so nobody has to rediscover it.

    The one-way ANOVA F assumes the two groups share a variance. It tolerates that assumption
    being false while the groups are the same size, and stops tolerating it as they diverge: this
    is the Behrens-Fisher problem, and it is not an implementation defect.

    It matters here because this extractor's comparison is unbalanced BY DESIGN. A candidate is
    one cluster of harmful rows, which can be as few as MIN_CLUSTER_ROWS, judged against half the
    harmless set, which is `--dir-prompts / 2` and defaults to 128. So the small-group case is the
    normal case, not the corner.

    Measured: a 4-row group four times as dispersed as its 32-row comparison clears the threshold
    43.5% of the time when the truth is no separation at all. The pre-registered 5% is a
    statement about equal variances, and this test is where that qualification lives.

    It is asserted rather than described because a silent future change to the pooling, to Welch's
    correction for instance, would alter this number, and that must be a decision someone makes
    rather than a test that quietly starts passing differently.
    """
    stat = separation.get("variance-ratio")
    over = _null_keep_rate(stat, 4, n_good=32, draws=2000, bad_scale=4.0)
    assert over > 0.25, (
        f"expected the unbalanced unequal-variance null to be badly anti-conservative, got "
        f"{over:.1%}. If this now passes, the statistic changed and the pre-registration's "
        f"stated 5% needs re-deriving.")
    # And over-conservative in the other direction, for the same reason with the sign flipped.
    under = _null_keep_rate(stat, 128, n_good=32, draws=2000, bad_scale=0.5)
    assert under > 0.10, f"expected an inflated keep-rate for a large tight group, got {under:.1%}"


def test_variance_ratio_grows_with_evidence_at_a_fixed_effect_size():
    """The same true effect, measured on more rows, scores higher. Cohen's d does not do this.

    This is the other half of the size calibration: the null stays put while the signal grows, so
    a big cluster's separation counts for more than a small one's, which is what "better evidenced"
    ought to mean.
    """
    scores = []
    for n in (8, 32, 128):
        trials = [separation.variance_ratio(*_projections(n, 64, 0.6, seed=300 + i))
                  for i in range(200)]
        scores.append(sum(trials) / len(trials))
    assert scores == sorted(scores), f"F did not grow with sample size: {scores}"
    assert scores[-1] > scores[0] * 2


# ── boundaries ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["cohens-d", "variance-ratio"])
def test_score_returns_none_when_a_group_is_too_small(name):
    stat = separation.get(name)
    pb, pg = _projections(3, 64, 2.0)
    assert stat.score(pb, pg) is None
    pb, pg = _projections(64, 3, 2.0)
    assert stat.score(pb, pg) is None
    pb, pg = _projections(separation.MIN_GROUP_ROWS, separation.MIN_GROUP_ROWS, 2.0)
    assert stat.score(pb, pg) is not None


def test_variance_ratio_survives_a_zero_variance_cloud():
    # Every projection identical: within-group variance is exactly zero, so the ratio is a
    # division by zero unless it is clamped. It must return a number rather than inf or NaN.
    pb = torch.full((16,), 3.0)
    pg = torch.full((16,), 3.0)
    score = separation.variance_ratio(pb, pg)
    assert score == pytest.approx(0.0)
    # Identical clouds are the extreme null: no separation, so it must not clear the threshold.
    assert score < separation.get("variance-ratio").threshold


def test_variance_ratio_returns_the_null_when_there_are_no_degrees_of_freedom():
    one = torch.tensor([1.0])
    assert separation.variance_ratio(one, torch.tensor([2.0])) == 1.0


def test_thresholds_are_the_pre_registered_numbers():
    # These are pre-registered and may not be moved in the light of a rejection rate. A change here
    # is a new pre-registration, and this test is where that has to be argued.
    assert separation.get("cohens-d").null == 0.0
    assert separation.get("cohens-d").threshold == 0.5
    assert separation.get("variance-ratio").null == 1.0
    assert separation.get("variance-ratio").threshold == 4.0


def test_default_is_the_incumbent():
    # Implementing a candidate must not change any existing result.
    assert separation.DEFAULT_STATISTIC == "cohens-d"


def test_unknown_statistic_names_what_it_knows():
    with pytest.raises(ValueError, match="variance-ratio"):
        separation.get("varience-ratio")


def test_cohens_d_matches_the_incumbent_implementation():
    """Byte-for-byte the same number as `cli._axis_separation`, or the comparison is confounded."""
    from senbonzakura import cli
    torch.manual_seed(5)
    bad, good = torch.randn(32, 8), torch.randn(48, 8)
    v = torch.randn(8)
    v = v / v.norm()
    assert separation.cohens_d(bad @ v, good @ v) == pytest.approx(
        cli._axis_separation(bad, good, v), rel=1e-9)
