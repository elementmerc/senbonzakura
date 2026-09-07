# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Two numerical points from the ORBA write-up, checked against our own code.

WHY THIS FILE EXISTS

A peer handover flagged two cheap adoptions from a geometric refinement of abliteration: run the
Gram-Schmidt orthogonalisation TWICE, and subtract before normalising rather than the reverse.
Both are the kind of suggestion that is easy to adopt on faith and pointless if the code already
does it or if the case never arises here. So both were measured.

WHAT THE MEASUREMENT SAID

Twice is a real improvement, on exactly one input shape, and that shape is the one this project
actually produces: eight near-parallel rows, which is `--max-directions 8` asked of a corpus that
cannot support eight distinct axes.

    ordinary rows                    4.4e-16 -> 4.4e-16   no change
    near-parallel rows               6.7e-13 -> 4.4e-16   1500x better
    gain-folded, 13 orders           8.9e-16 -> 8.9e-16   no change
    near-parallel AND gain-folded    1.0     -> 1.0       neither pass saves it

The last line is why this is an improvement and not a fix. A basis that degenerate is not rescued
by arithmetic, and the orthonormality check refuses it rather than baking with it.

Subtract-before-normalise turned out to be what the direction construction already does, so the
tests below pin that rather than changing it. Confirming a suggestion is already satisfied is a
result, and pinning it stops a later edit from quietly reversing the order.
"""
import pytest
import torch

from senbonzakura.cli import _modified_gram_schmidt, _orth_to


def _orth_err(Q):
    k = Q.shape[0]
    return float((Q @ Q.T - torch.eye(k, dtype=Q.dtype)).abs().max())


def _near_parallel(k=8, h=512, spread=1e-4, seed=0):
    """The input that produces the case: many directions asked of one real axis."""
    torch.manual_seed(seed)
    base = torch.randn(1, h)
    return base.repeat(k, 1) + torch.randn(k, h) * spread


def test_near_parallel_rows_come_out_orthonormal_to_machine_precision():
    """THE CASE THIS WAS ADOPTED FOR. One pass leaves 6.7e-13 here, which is far above the
    tolerance the bake needs for `R^T (R W)` to be a projection.
    """
    Q = _modified_gram_schmidt(_near_parallel())
    assert _orth_err(Q) < 1e-14


def test_ordinary_rows_are_unchanged_by_the_second_pass():
    """It costs one extra inner loop and buys nothing here, which is worth knowing: the win is
    specific, not general, and claiming otherwise would oversell it.
    """
    torch.manual_seed(0)
    Q = _modified_gram_schmidt(torch.randn(8, 512))
    assert _orth_err(Q) < 1e-14


def test_gain_folded_rows_spanning_many_orders_stay_orthonormal():
    """Rows arrive gain-weighted on the architectures with a post-sublayer norm, and a small gain
    can leave a row many orders of magnitude below its neighbours.
    """
    torch.manual_seed(0)
    scale = torch.tensor([1e3, 1e1, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]).unsqueeze(1)
    Q = _modified_gram_schmidt(torch.randn(8, 512) * scale)
    assert _orth_err(Q) < 1e-14


def test_a_row_inside_the_span_of_its_predecessors_becomes_zero_not_noise():
    """An arbitrary unit vector there would be a direction nothing asked to ablate."""
    torch.manual_seed(0)
    a = torch.randn(1, 64)
    M = torch.cat([a, a * 2.0])          # the second row carries nothing new
    Q = _modified_gram_schmidt(M)
    assert float(Q[1].norm()) < 1e-8


def test_the_pathological_case_is_not_silently_rescued():
    """Near-parallel AND badly scaled stays broken, and the honest record of that is here rather
    than in a claim that the second pass fixes degeneracy. `_orthonormal_rows` refuses it.
    """
    torch.manual_seed(0)
    scale = torch.tensor([1e4, 1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]).unsqueeze(1)
    Q = _modified_gram_schmidt(_near_parallel(spread=1e-5) * scale)
    assert _orth_err(Q) > 1e-6, (
        "if this now comes out orthonormal, the second pass is doing more than measured and the "
        "docstring's table needs re-taking rather than the assertion being relaxed")


# ── subtract before normalise, which was already the order ───────────────────────────

def test_the_projection_subtracts_and_does_not_normalise():
    """`_orth_to` removes a component and returns the result at whatever length it has. A version
    that normalised on the way out would change the magnitude of every direction it touched, and
    the caller normalises deliberately afterwards.
    """
    v = torch.tensor([3.0, 4.0, 0.0])
    u = torch.tensor([1.0, 0.0, 0.0])
    out = _orth_to(v, [u])
    assert torch.allclose(out, torch.tensor([0.0, 4.0, 0.0]))
    assert float(out.norm()) == pytest.approx(4.0), "the projection normalised its output"


def test_the_direction_construction_subtracts_first_then_normalises():
    """The order ORBA prefers, and the order already used. Reversing it would normalise a vector
    and then remove a component from it, leaving a direction of unpredictable length that the
    subsequent unit-normalisation would have to rescue.
    """
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "senbonzakura" / "cli.py"
    text = src.read_text(encoding="utf-8")
    assert "d0 = _orth_to(mb[li] - mg[li], [gd])\n" in text
    subtract_at = text.index("d0 = _orth_to(mb[li] - mg[li], [gd])")
    normalise_at = text.index("d0 = d0 / d0.norm().clamp_min(1e-8)", subtract_at)
    assert subtract_at < normalise_at, "the direction is normalised before it is projected"


# ── the reach reading must not hide a near-miss (peer finding, 2026-09-07) ───────────
# On LFM2.5-1.2B one conv layer reduced by 0.1010 against a floor of 0.100: it cleared by one part
# in a thousand, and the reading sentence said "comparable, so the bake treats both positions
# alike" without mentioning it. `below_floor` is binary, so 0.101 and 0.99 print identically, and
# the reading is the line that gets pasted into a table.
#
# This is the same failure as the one that moved this check from comparing MEANS to counting
# layers, one level up: a summary that hides a per-item result.

def test_a_layer_that_barely_cleared_the_floor_is_named_in_the_reading():
    """THE REGRESSION THIS SECTION IS NAMED FOR, with the real number that forced it."""
    from senbonzakura.validate import _thin_margin_note
    note = _thin_margin_note([{"layer": 3, "kind": "conv", "reduction": 0.1010},
                              {"layer": 7, "kind": "attention", "reduction": 0.7325}])
    assert "0.1010" in note
    assert "layer 3" in note or "is 3" in note
    assert "1.01" in note, "the margin over the floor is the number a reader needs"


def test_a_healthy_run_gets_no_note():
    """A warning that fires on every run is one nobody reads."""
    from senbonzakura.validate import _thin_margin_note
    assert _thin_margin_note([{"layer": 1, "kind": "conv", "reduction": 0.42},
                              {"layer": 2, "kind": "attention", "reduction": 0.73}]) == ""


def test_the_note_says_it_passed_so_it_is_not_mistaken_for_a_failure():
    """It IS a pass. Wording it as a failure would make a real failure unreadable by comparison."""
    from senbonzakura.validate import _thin_margin_note
    note = _thin_margin_note([{"layer": 3, "kind": "conv", "reduction": 0.11}])
    assert "It passed" in note


def test_no_rows_produce_no_note():
    from senbonzakura.validate import _thin_margin_note
    assert _thin_margin_note([]) == ""


def test_the_threshold_is_a_named_constant_with_the_number_behind_it():
    """0.101 against 0.100 is the case; the constant has to be wide enough to catch it and narrow
    enough that an ordinary run stays quiet.
    """
    from senbonzakura.validate import MIN_REACH, THIN_MARGIN, _thin_margin_note
    assert THIN_MARGIN > 1.0
    assert _thin_margin_note([{"layer": 0, "kind": "conv",
                               "reduction": MIN_REACH * THIN_MARGIN * 1.01}]) == ""
    assert _thin_margin_note([{"layer": 0, "kind": "conv",
                               "reduction": MIN_REACH * 1.01}]) != ""
