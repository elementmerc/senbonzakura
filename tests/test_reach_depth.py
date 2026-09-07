# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The reach reading blamed an architecture for a depth effect. These are the case that caught it.

WHAT HAPPENED

`validate --experiment reach` on LFM2.5-8B-A1B returned `architectures_failing: ['conv']` off a
pooled mean. A peer session read the per-layer output and pointed out that reduction climbs
monotonically with DEPTH in both LFM2 models measured, and stops caring about layer kind entirely
past a threshold. LFM2 puts full attention at layers 2, 6, 10, 14, 18 and 20, so conv blocks fill
the shallow positions that score badly for every kind AND the deep positions that score well. The
conv mean carries a shallow tail the attention mean does not have.

The pooled means were answering a question about depth and printing an answer about architecture,
and the same verdict would have fired on any hybrid whose non-attention blocks sit shallow,
including Qwen3.5 where 30 of 40 layers are mixers.

THE PART OF THE REPORT THAT WAS ITSELF OVERSTATED

The report said conv is never worse at matched depth, and listed four pairs. Over the FULL matched
set that is not true: on the 1.2B, conv layers 6 and 7 both lose to attention layer 8. The four
pairs were the ones supporting the claim. What survives is the depth trend itself, which is 0.9636
on the 8B and 0.75 on the 1.2B, and the fact that holding depth still moves conv from behind
attention to ahead of it on the 8B. Both of those are enough to sink the architecture verdict
without needing the stronger claim, and the stronger claim has its own test here so it cannot come
back.

WHAT THESE ASSERT

The real numbers from both models, as fixtures, so the regression is the actual case rather than a
constructed one. Plus the thing that must survive the fix: a layer that moved BACKWARDS is still
shouted about, because that is a fact about a layer and not a claim about a type.
"""
from __future__ import annotations

import pytest

from senbonzakura import validate

#: LFM2.5-8B-A1B, measured on the ROG 2026-09-07. Layer, kind, reduction.
EIGHT_B = [
    (9, "conv", -0.0246), (10, "attention", 0.2482), (11, "conv", 0.3475),
    (12, "conv", 0.4193), (13, "conv", 0.5704), (14, "attention", 0.7802),
    (15, "conv", 0.7814), (16, "conv", 0.7903), (17, "conv", 0.8388),
    (18, "attention", 0.7985), (19, "conv", 0.7982),
]

#: LFM2.5-1.2B, same run family.
ONE_TWO_B = [
    (6, "conv", 0.1010), (7, "conv", 0.3593), (8, "attention", 0.5599),
    (9, "conv", 0.8221), (10, "attention", 0.8505), (11, "conv", 0.8269),
    (12, "attention", 0.7871),
]

#: LFM2.5-350M, measured 2026-08-18. THE FIGURES ALREADY CITED IN THIS CODEBASE as evidence about
#: the conv path: conv 0.4947 against attention 0.7593, which is the largest gap of the three
#: models and reads as conv landing at two thirds of attention's strength. Held at shared depth it
#: reverses to conv 0.8109 against attention 0.7343. The whole unrestricted gap is layers 6 and 7,
#: which sit below every attention layer in the model.
THREE_FIFTY_M = [
    (6, "conv", 0.0755), (7, "conv", 0.2816), (8, "attention", 0.7020),
    (9, "conv", 0.8075), (10, "attention", 0.7666), (11, "conv", 0.8143),
    (12, "attention", 0.8094),
]


def rows(spec):
    return [{"layer": li, "kind": k, "reduction": r} for li, k, r in spec]


# ── the depth trend, which is the simpler explanation ─────────────────────────────
@pytest.mark.parametrize(("name", "spec", "expected"),
                         [("8B", EIGHT_B, 0.9636), ("1.2B", ONE_TWO_B, 0.75),
                          ("350M", THREE_FIFTY_M, 0.9286)])
def test_reduction_climbs_with_depth_in_both_models(name, spec, expected):
    """If this is not true, the whole re-reading is wrong and should be reverted.

    The measured values are pinned rather than compared against a round threshold, because a
    threshold is a number somebody picked and these are numbers the models produced. Note the
    8B is much the stronger of the two at 0.96 against 0.75: eleven layers against seven, and the
    1.2B's shallowest conv layer is the one dragging it.
    """
    r = rows(spec)
    trend = validate._spearman([x["layer"] for x in r], [x["reduction"] for x in r])
    assert trend == pytest.approx(expected, abs=1e-4), f"{name}: {trend}"
    assert trend > 0.7, f"{name}: depth no longer explains the ordering ({trend})"


def test_the_rank_correlation_handles_ties_and_short_inputs():
    assert validate._spearman([1, 2], [1, 2]) is None, "two points is not a correlation"
    assert validate._spearman([1, 2, 3], [5, 5, 5]) is None, "a flat series has no rank order"
    assert validate._spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert validate._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


# ── the confound itself ───────────────────────────────────────────────────────────
def test_the_matched_pairs_are_mixed_and_that_is_the_honest_reading():
    """A CORRECTION to the report that raised this, kept as a test so it cannot be re-lost.

    The peer's message said there is no depth at which attention beats conv in either model, and
    listed four pairs supporting it. The FULL matched set does not say that. On the 1.2B, conv
    layer 7 (0.359) pairs with attention layer 8 (0.560) and loses, and conv 6 loses too. The four
    quoted pairs were the ones that supported the claim.

    The general point survives and the specific claim does not, which is worth a test of its own:
    a re-reading that overshoots is how a correct correction turns into the next wrong verdict.
    """
    pairs = validate._matched_depth_pairs(rows(ONE_TWO_B))
    winners = {p["stronger"] for p in pairs}
    assert winners == {"conv", "attention"}, (
        f"the 1.2B matched pairs are mixed, not one-sided: {pairs}")


def test_holding_depth_still_removes_the_conv_deficit_on_the_8b():
    """THE REGRESSION, in the numbers that produced it.

    Unrestricted, conv reads 0.565 against attention's 0.609 and the verdict named conv. Attention
    only occupies layers 10, 14 and 18, while conv also fills 9 and 19, and reduction climbs with
    depth. Restricted to the range both kinds occupy, conv comes out AHEAD.
    """
    r = rows(EIGHT_B)
    span = validate._shared_depth_range(r)
    assert span == (10, 18)
    held = validate._by_kind_in_range(r, span)
    assert held["conv"]["mean_reduction"] > held["attention"]["mean_reduction"], held
    # And the unrestricted mean says the opposite, which is the whole point.
    raw_conv = sum(x["reduction"] for x in r if x["kind"] == "conv") / 8
    raw_attn = sum(x["reduction"] for x in r if x["kind"] == "attention") / 3
    assert raw_conv < raw_attn, "the fixture no longer reproduces the defect"


def test_the_350m_figures_this_codebase_already_cites_reverse_the_same_way():
    """THE ONE THAT WAS ALREADY IN PRINT.

    conv 0.495 against attention 0.759 sits in a source comment as evidence about the conv path,
    and it is the largest apparent gap of the three models. Held at the depth both kinds occupy it
    reverses: conv 0.811 against attention 0.734. The whole gap is layers 6 and 7, below every
    attention layer in the model.

    Arithmetic re-derived here from the per-layer numbers rather than taken from the report that
    supplied them, because a second copy of somebody's mean is not a check on it.
    """
    r = rows(THREE_FIFTY_M)
    raw_conv = sum(x["reduction"] for x in r if x["kind"] == "conv") / 4
    raw_attn = sum(x["reduction"] for x in r if x["kind"] == "attention") / 3
    assert raw_conv == pytest.approx(0.4947, abs=1e-3), "not the figures the comment cites"
    assert raw_attn == pytest.approx(0.7593, abs=1e-3)

    span = validate._shared_depth_range(r)
    assert span == (8, 11)
    held = validate._by_kind_in_range(r, span)
    assert held["conv"]["mean_reduction"] > held["attention"]["mean_reduction"], held
    assert held["conv"]["mean_reduction"] == pytest.approx(0.8109, abs=1e-3)
    # Two layers of each kind is the bare minimum this reading will compare on, and the reading
    # says so rather than presenting four layers as though they were forty.
    assert held["conv"]["layers"] == held["attention"]["layers"] == validate.MIN_DEPTH_OVERLAP


def test_all_three_models_reverse_in_the_same_direction():
    """One model reversing is a curiosity. Three is the reason the verdict changed."""
    for spec in (EIGHT_B, ONE_TWO_B, THREE_FIFTY_M):
        r = rows(spec)
        held = validate._by_kind_in_range(r, validate._shared_depth_range(r))
        assert held["conv"]["mean_reduction"] > held["attention"]["mean_reduction"], (spec, held)


def test_the_shallow_layer_outside_the_shared_range_is_what_moved_the_mean():
    """Layer 9, the one that went backwards, sits below every attention layer. It belongs in the
    per-layer finding and not in a comparison between kinds.
    """
    span = validate._shared_depth_range(rows(EIGHT_B))
    assert span[0] > 9
    assert span[1] < 19


def test_a_single_kind_has_no_shared_range():
    assert validate._shared_depth_range([{"layer": 1, "kind": "attention", "reduction": 0.5}]) is None
    assert validate._by_kind_in_range([], None) is None


def test_a_single_kind_has_no_overlap_to_compute():
    assert validate._depth_overlap([{"layer": 1, "kind": "attention", "reduction": 0.5}]) is None


def test_kinds_interleaved_at_the_same_depths_are_comparable():
    """The fix must not refuse every comparison. Two kinds alternating are genuinely comparable."""
    spec = [(i, "conv" if i % 2 else "attention", 0.5) for i in range(8)]
    overlap = validate._depth_overlap(rows(spec))
    assert all(c >= validate.MIN_DEPTH_OVERLAP for c in overlap.values()), overlap


def test_matched_pairs_are_skipped_when_nothing_is_near_enough():
    """A kind with no neighbour inside the window contributes no pair rather than a distant one."""
    spec = [(0, "conv", 0.1), (1, "conv", 0.2), (20, "attention", 0.9)]
    assert validate._matched_depth_pairs(rows(spec)) == []


def test_matched_pairs_need_exactly_two_kinds():
    spec = [(0, "conv", 0.1), (1, "attention", 0.2), (2, "linear_attn", 0.3)]
    assert validate._matched_depth_pairs(rows(spec)) == []


def test_a_pair_names_the_stronger_side():
    spec = [(4, "attention", 0.20), (5, "conv", 0.60)]
    p = validate._matched_depth_pairs(rows(spec))[0]
    assert p["stronger"] == "conv"
    assert p["conv"]["layer"] == 5
    assert p["attention"]["reduction"] == 0.20


def test_the_nearest_neighbour_is_the_one_chosen():
    """A pair against a distant layer of the right kind would smuggle the depth effect back in."""
    spec = [(4, "attention", 0.2), (5, "conv", 0.6), (6, "attention", 0.9)]
    pairs = validate._matched_depth_pairs(rows(spec))
    # conv is the rarer kind, so it is the one paired, and layer 4 and 6 are equidistant.
    assert len(pairs) == 1
    assert pairs[0]["attention"]["layer"] in (4, 6)
    assert abs(pairs[0]["attention"]["layer"] - 5) == 1
