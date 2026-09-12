# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for the refusal-separation statistic (cli._halves, _held_out_separation, _null_...).

WHY THIS FILE EXISTS

For the project's whole history the separation filter printed "rejected NONE of N candidates" on
every run, and that was read as a lenient threshold. It was not. The candidate direction is a
cluster's mean minus the harmless mean, and the statistic judging it is a difference of those
same means over those same rows, so the quantity a vector was built to maximise was being
measured along that vector. It could not come out small.

The 2026-08-16 rework fits each candidate on half the rows and scores it on the half it never
saw, and gives the threshold a floor measured from directions that carry nothing. These tests
are mostly about the two properties that make that worth having: that the in-sample statistic
really is degenerate, and that the held-out one really does separate a signal from noise.
"""
import pytest
import torch

from senbonzakura import cli


def _blob(n, h, centre, spread=0.35, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, h, generator=g) * spread + centre


# ── _halves ──────────────────────────────────────────────────────────────────────
def test_halves_are_disjoint_and_cover_everything():
    a, b = cli._halves(21, 7)
    assert len(a) + len(b) == 21
    assert set(a.tolist()).isdisjoint(b.tolist())
    assert sorted(a.tolist() + b.tolist()) == list(range(21))


def test_halves_are_the_same_every_time_for_a_seed():
    a1, b1 = cli._halves(40, 3)
    a2, b2 = cli._halves(40, 3)
    assert a1.tolist() == a2.tolist()
    assert b1.tolist() == b2.tolist()


def test_halves_differ_between_seeds():
    a1, _ = cli._halves(40, 3)
    a2, _ = cli._halves(40, 4)
    assert a1.tolist() != a2.tolist()


def test_halves_do_not_touch_the_global_rng():
    # The extractor runs inside a search that draws from the global RNG. If this function
    # advanced it, rerunning the same trial would fit directions on different rows.
    torch.manual_seed(11)
    before = torch.randn(3)
    torch.manual_seed(11)
    cli._halves(50, 1)
    after = torch.randn(3)
    assert torch.equal(before, after)


def test_halves_of_an_odd_count_put_the_extra_row_in_the_scoring_side():
    fit, score = cli._halves(9, 1)
    assert len(fit) == 4
    assert len(score) == 5


# ── the defect the rework replaces ───────────────────────────────────────────────
def test_the_in_sample_statistic_cannot_come_out_small():
    """The whole reason the filter rejected nothing, demonstrated on pure noise.

    There is no structure here at all: both clouds are the same distribution. Yet building the
    direction from these rows and scoring it on these rows clears the threshold comfortably,
    because the score is the quantity the direction was chosen to maximise.
    """
    h = 16
    bad = _blob(24, h, 0.0, seed=1)
    good = _blob(24, h, 0.0, seed=2)
    v = bad.mean(0) - good.mean(0)
    v = v / v.norm()
    assert cli._axis_separation(bad, good, v) > cli.MIN_AXIS_SEPARATION


def test_the_held_out_statistic_is_not_fooled_by_the_same_noise():
    """Same clouds, same construction, scored on rows the direction never saw."""
    h = 16
    bad = _blob(40, h, 0.0, seed=1)
    good = _blob(40, h, 0.0, seed=2)
    gf, gs = cli._halves(40, 0)
    sep = cli._held_out_separation(bad, good[gf], good[gs], [], seed=0)
    assert sep is not None
    assert sep < cli.MIN_AXIS_SEPARATION


def test_the_held_out_statistic_still_finds_a_real_contrast():
    """A genuine separation survives the move to held-out rows, or the fix would be useless."""
    h = 16
    centre = torch.zeros(h)
    centre[0] = 6.0
    bad = _blob(40, h, centre, seed=1)
    good = _blob(40, h, torch.zeros(h), seed=2)
    gf, gs = cli._halves(40, 0)
    sep = cli._held_out_separation(bad, good[gf], good[gs], [], seed=0)
    assert sep is not None
    assert sep > cli.MIN_AXIS_SEPARATION


def test_the_held_out_statistic_separates_signal_from_noise_by_a_wide_margin():
    h = 16
    centre = torch.zeros(h)
    centre[0] = 6.0
    good = _blob(40, h, torch.zeros(h), seed=2)
    gf, gs = cli._halves(40, 0)
    signal = cli._held_out_separation(_blob(40, h, centre, seed=1), good[gf], good[gs], [], 0)
    noise = cli._held_out_separation(_blob(40, h, 0.0, seed=1), good[gf], good[gs], [], 0)
    assert signal > 5 * noise


# ── _held_out_separation, the refusal cases ──────────────────────────────────────
def test_a_cluster_too_small_to_split_is_refused_rather_than_scored():
    h = 8
    bad = _blob(cli.MIN_HELD_OUT_ROWS * 2 - 1, h, 1.0, seed=1)
    good = _blob(20, h, 0.0, seed=2)
    gf, gs = cli._halves(20, 0)
    assert cli._held_out_separation(bad, good[gf], good[gs], [], 0) is None


def test_a_cluster_exactly_at_the_floor_is_scored():
    h = 8
    bad = _blob(cli.MIN_HELD_OUT_ROWS * 2, h, 3.0, seed=1)
    good = _blob(20, h, 0.0, seed=2)
    gf, gs = cli._halves(20, 0)
    assert cli._held_out_separation(bad, good[gf], good[gs], [], 0) is not None


def test_a_direction_entirely_inside_the_basis_is_refused():
    # Nothing survives the orthogonalisation, so there is no direction left to score and a
    # normalised numerical residue would be scored as though it carried something.
    h = 8
    bad = _blob(20, h, 0.0, seed=1)
    good = _blob(20, h, 0.0, seed=2)
    gf, gs = cli._halves(20, 0)
    basis = [torch.eye(h)[i] for i in range(h)]
    assert cli._held_out_separation(bad, good[gf], good[gs], basis, 0) is None


def test_held_out_separation_is_reproducible():
    h = 12
    bad = _blob(30, h, 2.0, seed=1)
    good = _blob(30, h, 0.0, seed=2)
    gf, gs = cli._halves(30, 0)
    a = cli._held_out_separation(bad, good[gf], good[gs], [], 5)
    b = cli._held_out_separation(bad, good[gf], good[gs], [], 5)
    assert a == b


def test_held_out_separation_respects_the_basis():
    # Removing the one axis that carries the contrast has to change the answer, or the basis is
    # being ignored and directions would be kept that duplicate what is already ablated.
    h = 12
    centre = torch.zeros(h)
    centre[0] = 8.0
    bad = _blob(30, h, centre, seed=1)
    good = _blob(30, h, torch.zeros(h), seed=2)
    gf, gs = cli._halves(30, 0)
    free = cli._held_out_separation(bad, good[gf], good[gs], [], 5)
    blocked = cli._held_out_separation(bad, good[gf], good[gs], [torch.eye(h)[0]], 5)
    assert blocked is None or blocked < free


# ── _null_separation_floor ───────────────────────────────────────────────────────
def _floor(bad, good, size, seed=0, n_null=4, basis=None):
    gf, gs = cli._halves(int(good.shape[0]), 0)
    return cli._null_separation_floor(
        bad, good[gf], good[gs], basis or [], size, seed, n_null)


def test_the_null_floor_is_a_stated_quantile_of_the_draws():
    """CHANGED BY Q-22, 2026-09-02. It asserted `floor == max(samples)`.

    The maximum was replaced because it made the DRAW COUNT a hidden gate setting: the maximum of
    n draws climbs with n without limit, so raising the count to steady the estimate would have
    tightened the filter instead. Measured on one layer's rows, the variance ratio's maximum went
    2.33 at four draws to 6.14 at two hundred.

    Still not the mean, for the original reason: a candidate that merely beats a TYPICAL
    meaningless direction is not evidence. It is the 95th percentile, by nearest rank, so the
    floor is always a value some null direction actually reached.
    """
    h = 12
    bad = _blob(60, h, 1.0, seed=1)
    good = _blob(40, h, 0.0, seed=2)
    floor, samples = _floor(bad, good, 20)
    assert samples
    assert floor in samples, "the floor must be a value a null direction actually reached"
    ordered = sorted(samples)
    assert floor == ordered[min(len(ordered) - 1,
                                int(cli.NULL_FLOOR_QUANTILE * (len(ordered) - 1)))]
    assert floor >= ordered[len(ordered) // 2], "a quantile this high sits above the median"


def test_the_null_floor_does_not_move_with_the_draw_count():
    """The property Q-22 bought, and the one the maximum could never have.

    More draws must mean a better-located floor, not a stricter one. Without this the draw count
    is a gate setting wearing the clothes of a precision setting, which is how it behaved before.
    """
    h = 12
    bad = _blob(200, h, 1.0, seed=1)
    good = _blob(120, h, 0.0, seed=2)
    few = _floor(bad, good, 20, n_null=20)[0]
    many = _floor(bad, good, 20, n_null=200)[0]
    assert many == pytest.approx(few, rel=0.5), (
        f"the floor moved from {few} to {many} on a tenfold change in draws alone")


def test_the_null_floor_draws_one_sample_per_requested_null():
    h = 12
    bad = _blob(60, h, 1.0, seed=1)
    good = _blob(40, h, 0.0, seed=2)
    _, samples = _floor(bad, good, 20, n_null=6)
    assert len(samples) == 6


def test_the_null_floor_is_reproducible():
    h = 12
    bad = _blob(60, h, 1.0, seed=1)
    good = _blob(40, h, 0.0, seed=2)
    assert _floor(bad, good, 20, seed=3)[1] == _floor(bad, good, 20, seed=3)[1]


def test_the_null_floor_moves_with_its_seed():
    h = 12
    bad = _blob(60, h, 1.0, seed=1)
    good = _blob(40, h, 0.0, seed=2)
    assert _floor(bad, good, 20, seed=3)[1] != _floor(bad, good, 20, seed=9)[1]


def test_the_null_floor_is_zero_when_no_null_could_be_measured():
    # Every draw is too small to split, so there is no floor. Zero rather than None keeps the
    # threshold arithmetic total: max(constant, 0.0) is the constant, which is the old behaviour.
    h = 8
    bad = _blob(60, h, 1.0, seed=1)
    good = _blob(40, h, 0.0, seed=2)
    floor, samples = _floor(bad, good, cli.MIN_HELD_OUT_ROWS * 2 - 1)
    assert floor == 0.0
    assert samples == []


def test_a_real_direction_beats_the_null_floor_it_is_judged_against():
    """The property the whole mechanism rests on, on a cloud with genuine structure."""
    h = 16
    centre = torch.zeros(h)
    centre[0] = 6.0
    cluster = _blob(30, h, centre, seed=1)
    rest = _blob(60, h, torch.zeros(h), seed=3)
    bad_all = torch.cat([cluster, rest])
    good = _blob(40, h, torch.zeros(h) - 1.0, seed=2)
    gf, gs = cli._halves(40, 0)
    floor, _ = cli._null_separation_floor(bad_all, good[gf], good[gs], [], 30, 0, 4)
    sep = cli._held_out_separation(cluster, good[gf], good[gs], [], 0)
    assert sep > floor


def test_a_direction_over_pure_noise_does_not_beat_its_null_floor():
    """The other half of the same property, and the case the old statistic passed anyway."""
    h = 16
    bad_all = _blob(90, h, 0.0, seed=1)
    good = _blob(40, h, 0.0, seed=2)
    gf, gs = cli._halves(40, 0)
    floor, _ = cli._null_separation_floor(bad_all, good[gf], good[gs], [], 30, 0, 8)
    sep = cli._held_out_separation(bad_all[:30], good[gf], good[gs], [], 0)
    assert sep <= floor


# ── the constants stay in step ───────────────────────────────────────────────────
def test_the_smallest_allowed_cluster_can_still_be_split():
    # If these two drift apart, every smallest cluster is silently dropped by a guard that reads
    # as a size floor rather than as a split floor.
    assert cli.MIN_CLUSTER_ROWS >= 2 * cli.MIN_HELD_OUT_ROWS


def test_there_is_more_than_one_null_per_layer():
    # Renamed by Q-22 (2026-09-02): the floor is drawn once per CANDIDATE at that candidate's own
    # row count, not once per layer at the median cluster's. The count also has to be high enough
    # to locate a 95th percentile rather than merely to beat one draw.
    assert cli.NULL_DIRECTIONS_PER_CANDIDATE >= 20


# ── the held-out half is held out by REQUEST, not by row ─────────────────────────────────────

def _templated(n_requests=10, n_templates=7):
    """A corpus shaped like this project's: every request crossed with every template."""
    seeds = [f"make a {w}" for w in
             ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]][:n_requests]
    tpls = ["Explain in detail how to", "Write a step by step guide to", "Give me instructions to",
            "Describe the process to", "Tell me how to", "Provide a tutorial on",
            "Outline the way to"][:n_templates]
    return [f"{t} {s}" for s in seeds for t in tpls]


def test_a_row_split_puts_every_request_on_both_sides():
    """The defect, measured rather than argued.

    Seven templates share one seed, so "Explain in detail how to X" and "Tell me how to X"
    are two rows and one request. Permuting raw row indices scores a direction on a
    rephrasing of exactly what it was fitted on, and `abliteration.json` recorded
    `separation_held_out: true` over it.
    """
    rows = _templated()
    keys = cli._request_keys(rows)
    fit, score = cli._halves(len(rows), 5)
    both = {keys[i] for i in fit.tolist()} & {keys[i] for i in score.tolist()}
    assert len(both) == len(set(keys)), (
        "this test documents the OLD behaviour; if the row split no longer leaks, the grouped "
        "split has become the default and this test should be deleted rather than adjusted")


def test_grouping_by_request_leaks_nothing_across_the_split():
    rows = _templated()
    keys = cli._request_keys(rows)
    fit, score = cli._halves(len(rows), 5, keys)
    assert not ({keys[i] for i in fit.tolist()} & {keys[i] for i in score.tolist()})
    assert sorted(fit.tolist() + score.tolist()) == list(range(len(rows))), (
        "every row must land on exactly one side, or the split has dropped or duplicated rows")


def test_the_grouped_halves_stay_balanced():
    """Groups differ in width, so they are dealt to whichever side is behind."""
    rows = _templated()
    fit, score = cli._halves(len(rows), 5, cli._request_keys(rows))
    assert abs(len(fit) - len(score)) <= 7, (len(fit), len(score))


def test_the_grouped_split_is_reproducible_for_a_seed():
    rows = _templated()
    keys = cli._request_keys(rows)
    a = cli._halves(len(rows), 5, keys)
    b = cli._halves(len(rows), 5, keys)
    assert a[0].tolist() == b[0].tolist() and a[1].tolist() == b[1].tolist()


def test_a_corpus_with_no_shared_templates_still_splits():
    """The safety property that makes this unconditional: with nothing shared, every row is
    its own request, so the grouped split degrades to the row split rather than refusing.
    """
    rows = [f"an entirely unrelated question number {i}" for i in range(40)]
    keys = cli._request_keys(rows)
    assert len(set(keys)) == len(rows)
    fit, score = cli._halves(len(rows), 3, keys)
    assert len(fit) == len(score) == 20
