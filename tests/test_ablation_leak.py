# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Putting the row lengths back undoes part of the ablation. Measured, then fixed.

WHY THIS FILE EXISTS

The edit removes a refusal direction, then restores each output row's original length so the
model's calibration survives. Those two steps do not commute. The projection acts across rows and
the restore scales each row on its own, so a matrix whose columns were orthogonal to the refusal
span stops being orthogonal the moment the lengths go back.

Measured on 2026-09-07, after a peer session worked out the algebra and asked to be checked rather
than believed:

    ||R W|| after the subtraction        5e-07     the ablation itself is exact
    ||R W|| after restoring the lengths  32% of the original, at a 4x row-length spread
                                         46% at 10x, 5% at 0.2x

So the edit has always been approximate, and HOW approximate depended on the model, which nothing
recorded and nobody knew.

Alternating "restore the lengths" and "remove the direction again" converges to a matrix that has
both properties, in four rounds, across every shape and direction count tried.

WHY IT IS OFF BY DEFAULT

A cleaner cut is not obviously a better model. The leak may be part of why quality held up here,
and turning this on changes every number a run produces. It is an arm to measure, not an upgrade
to apply.
"""
import pytest
import torch

from senbonzakura.cli import ABLATION_ROUNDS, orthogonalize_np_, orthogonalize_np_3d_


def _case(H=256, IN=512, K=2, spread=3.0, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(H, IN) * 0.02
    W *= (1.0 + torch.rand(H, 1) * spread)
    R = torch.linalg.qr(torch.randn(H, K))[0].T.contiguous()
    return W, R


def _leak(W, R):
    return float((R @ W).norm())


def _length_error(W, original):
    return float(((W.norm(dim=1) - original.norm(dim=1)).abs()
                  / original.norm(dim=1)).mean())


def test_the_single_pass_leaves_a_large_part_of_the_direction_behind():
    """THE DEFECT, asserted so it cannot be quietly fixed without this test noticing.

    This asserts a defect is PRESENT in the default path. If it ever fails, the default has
    changed and every recorded separation number needs re-reading, so the test should be updated
    deliberately rather than adjusted away.
    """
    W, R = _case()
    before = _leak(W, R)
    edited = W.clone()
    orthogonalize_np_(edited, R, 1.0)
    assert _leak(edited, R) / before > 0.20, (
        "the single pass no longer leaks; the default behaviour has changed")


def test_the_subtraction_itself_is_exact():
    """The leak is the restore, not the maths. Worth pinning so a future reader does not go
    looking for a bug in the projection.
    """
    W, R = _case()
    rn = W.norm(dim=1, keepdim=True)
    Wn = W / rn
    ablated = Wn - (R.T @ (R @ Wn))
    assert _leak(ablated, R) < 1e-5


@pytest.mark.parametrize(("H", "IN", "K", "spread"), [
    (256, 512, 1, 3.0),
    (256, 512, 8, 3.0),
    (512, 1024, 4, 10.0),      # very uneven lengths: where the leak is worst
    (512, 1024, 4, 0.2),       # nearly uniform: where it is mildest
])
def test_rounds_remove_the_leak_across_shapes_and_spreads(H, IN, K, spread):
    W, R = _case(H, IN, K, spread)
    before = _leak(W, R)
    edited = W.clone()
    orthogonalize_np_(edited, R, 1.0, rounds=ABLATION_ROUNDS)
    assert _leak(edited, R) / before < 1e-4


@pytest.mark.parametrize("spread", [0.2, 3.0, 10.0])
def test_rounds_keep_the_row_lengths_which_is_the_whole_point_of_restoring_them(spread):
    """Removing the leak by dropping the restore would be no fix at all: the restore exists
    because raw orthogonalisation wrecked calibration at KL 12 to 19.
    """
    W, R = _case(spread=spread)
    edited = W.clone()
    orthogonalize_np_(edited, R, 1.0, rounds=ABLATION_ROUNDS)
    assert _length_error(edited, W) < 1e-4


def test_the_default_is_the_old_behaviour():
    """Turning this on changes every number a run produces, so it must never arrive by surprise."""
    W, R = _case()
    a, b = W.clone(), W.clone()
    orthogonalize_np_(a, R, 1.0)
    orthogonalize_np_(b, R, 1.0, rounds=0)
    assert torch.equal(a, b)


def test_more_rounds_than_needed_change_nothing():
    """It converges rather than drifting, so a caller cannot make it worse by asking for more."""
    W, R = _case()
    four, twenty = W.clone(), W.clone()
    orthogonalize_np_(four, R, 1.0, rounds=4)
    orthogonalize_np_(twenty, R, 1.0, rounds=20)
    assert torch.allclose(four, twenty, atol=1e-5)


# ── the fused-MoE path has the same defect and the same fix ──────────────────────────

def test_the_fused_expert_path_leaks_too():
    torch.manual_seed(0)
    E, H, IN, K = 4, 128, 256, 2
    W = torch.randn(E, H, IN) * 0.02 * (1.0 + torch.rand(E, H, 1) * 3.0)
    R = torch.linalg.qr(torch.randn(H, K))[0].T.contiguous()

    def leak3(M):
        return float(torch.einsum("kh,ehi->eki", R, M).norm())

    before = leak3(W)
    single, many = W.clone(), W.clone()
    orthogonalize_np_3d_(single, R, 1.0)
    orthogonalize_np_3d_(many, R, 1.0, rounds=ABLATION_ROUNDS)
    assert leak3(single) / before > 0.20, "the fused path no longer leaks; default changed"
    assert leak3(many) / before < 1e-4
    lengths = float(((many.norm(dim=2) - W.norm(dim=2)).abs() / W.norm(dim=2)).mean())
    assert lengths < 1e-4


def test_the_flag_defaults_to_the_old_behaviour():
    from senbonzakura.parser import build_parser
    assert build_parser().parse_args(["--model", "m"]).ablation_rounds == 0
