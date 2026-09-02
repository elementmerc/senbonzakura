# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The subspace-comparison instrument.

An instrument whose whole output is "these two agree" is worthless until it can say no, so every
property here is asserted in both directions: the things that must NOT move the number, and the
things that must. The second half is the control, and it is the half that would catch a change
turning this into a function that always agrees.

The sprint's verification line is the control test at the bottom: it must report a non-zero
difference and refuse.
"""
import math

import pytest
import torch

from senbonzakura import subspace

H, K = 256, 8


def _span(seed=0, h=H, k=K):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, h, generator=g)


# ── things that must NOT move the number ────────────────────────────────────────────
#: The floor a perfect comparison actually reaches, and it is not zero.
#:
#: The distance is a SQUARE ROOT of a sum of squares, so a rounding residual of 1e-16 in the
#: squared quantity surfaces as 1e-8 in the reported number. Anyone reading these tests will
#: otherwise wonder why identical inputs score 1.8e-08, and anyone tightening the tolerance
#: towards machine epsilon will make them flaky for a reason that is arithmetic, not drift.
EXACT = 1e-7


def test_a_subspace_equals_itself():
    d = _span()
    c = subspace.compare(d, d)
    assert c.same
    assert c.distance == pytest.approx(0.0, abs=EXACT)
    assert c.disagreeing == pytest.approx(0.0, abs=EXACT)


def test_flipping_every_sign_changes_nothing():
    """A direction and its negation ablate identically, so they are the same direction here.

    This is the first reason per-vector cosines are the wrong instrument: they would call this
    total disagreement.
    """
    d = _span()
    flipped = d * torch.tensor([1.0, -1.0] * (K // 2)).unsqueeze(1)
    c = subspace.compare(d, flipped)
    assert c.same
    assert c.distance == pytest.approx(0.0, abs=EXACT)


def test_rotating_the_basis_changes_nothing():
    """The second reason. A near-degenerate subspace comes back arbitrarily rotated, and that is
    bookkeeping rather than a difference in what gets ablated.
    """
    d = _span()
    g = torch.Generator().manual_seed(5)
    rotation = torch.linalg.qr(torch.randn(K, K, generator=g).double())[0]
    c = subspace.compare(d, (rotation @ d.double()))
    assert c.same
    assert c.distance == pytest.approx(0.0, abs=EXACT)


def test_reordering_the_directions_changes_nothing():
    d = _span()
    c = subspace.compare(d, d[torch.randperm(K, generator=torch.Generator().manual_seed(2))])
    assert c.same


def test_scaling_a_direction_changes_nothing():
    # Only the span matters; a longer arrow points the same way.
    d = _span()
    scaled = d.clone()
    scaled[3] *= 17.0
    assert subspace.compare(d, scaled).same


def test_padding_rows_are_dropped_rather_than_compared():
    """`dirs_multi` pads unfilled slots with exact zeros, and they are not directions.

    A layer that filled three of eight slots must compare as a three-dimensional subspace. Without
    this, the padding becomes five shared axes and every comparison looks better than it is.
    """
    d = _span(k=3)
    padded = torch.cat([d, torch.zeros(5, H)])
    c = subspace.compare(d, padded)
    assert c.same
    assert c.rank_a == 3 and c.rank_b == 3, "the zero rows must not count as dimensions"


def test_a_bfloat16_round_trip_stays_within_tolerance():
    """The noise the project actually incurs, since `dirs_multi` is stored in bfloat16."""
    d = _span()
    c = subspace.compare(d, d.to(torch.bfloat16).float())
    assert c.same, f"bf16 round trip scored {c.distance}, above the {c.tolerance} tolerance"
    assert c.distance < subspace.SAME_SUBSPACE_TOL / 3, (
        "the tolerance is supposed to sit comfortably above bf16 noise, not just barely")


# ── things that MUST move the number: the controls ──────────────────────────────────
def test_replacing_one_direction_is_detected():
    """THE CONTROL THE SPRINT REQUIRES. One wrong direction of eight, and it must say so."""
    d = _span()
    broken = d.clone()
    broken[0] = _span(seed=99)[0]
    c = subspace.compare(d, broken)
    assert not c.same, "an instrument that cannot report this difference cannot report any"
    assert c.distance > 0.1
    assert c.disagreeing > 0.5, f"about one direction's worth expected, got {c.disagreeing}"
    assert "DIFFERENT" in c.describe()


def test_two_unrelated_subspaces_are_nearly_orthogonal():
    c = subspace.compare(_span(seed=1), _span(seed=2))
    assert not c.same
    assert c.distance > 0.9, f"unrelated random spans should be near 1.0, got {c.distance}"


def test_a_completely_orthogonal_pair_scores_exactly_one():
    eye = torch.eye(H)
    c = subspace.compare(eye[:4], eye[4:8])
    assert c.distance == pytest.approx(1.0)
    assert c.disagreeing == pytest.approx(4.0), "four whole directions unshared"


def test_a_rank_difference_is_a_disagreement_no_tolerance_covers():
    """One side carries a direction the other has nowhere to put, however close the rest are."""
    d = _span(k=8)
    c = subspace.compare(d, d[:7])
    assert not c.same
    assert c.rank_a == 8 and c.rank_b == 7
    assert "rank difference" in c.describe()
    # And it stays a disagreement however lenient the caller is.
    assert not subspace.compare(d, d[:7], tolerance=0.99).same


def test_a_small_perturbation_of_every_entry_is_detected():
    d = _span()
    g = torch.Generator().manual_seed(7)
    c = subspace.compare(d, d + torch.randn(K, H, generator=g) * 0.05)
    assert not c.same, f"1-in-20 noise on every entry should not pass, scored {c.distance}"


def test_the_tolerance_sits_between_the_noise_and_the_smallest_real_difference():
    """Pins the derivation in the module docstring, so a future edit has to argue with a number."""
    d = _span()
    noise = subspace.compare(d, d.to(torch.bfloat16).float()).distance
    broken = d.clone()
    broken[0] = _span(seed=99)[0]
    real = subspace.compare(d, broken).distance
    assert noise < subspace.SAME_SUBSPACE_TOL < real
    assert noise * 3 < subspace.SAME_SUBSPACE_TOL, "too close to the noise floor to be safe"
    assert real > subspace.SAME_SUBSPACE_TOL * 10, "too close to the signal to be discriminating"


# ── boundaries ──────────────────────────────────────────────────────────────────────
def test_two_empty_spans_agree():
    empty = torch.zeros(4, H)
    c = subspace.compare(empty, empty)
    assert c.same
    assert c.rank_a == 0


def test_an_empty_span_against_a_real_one_disagrees():
    c = subspace.compare(torch.zeros(4, H), _span(k=4))
    assert not c.same
    assert c.rank_a == 0 and c.rank_b == 4


def test_duplicate_directions_do_not_inflate_the_rank():
    d = _span(k=3)
    c = subspace.compare(d, torch.cat([d, d]))
    assert c.same
    assert c.rank_b == 3, "a repeated direction adds no dimension"


def test_angles_are_reported_in_degrees_and_are_sane():
    eye = torch.eye(H)
    c = subspace.compare(eye[:2], eye[2:4])
    assert c.max_angle_degrees == pytest.approx(90.0, abs=1e-6)
    # Same amplification: arccos near 1 turns a 1e-16 residual into ~1e-6 degrees.
    assert subspace.compare(_span(), _span()).max_angle_degrees == pytest.approx(0.0, abs=1e-4)


def test_principal_angles_do_not_return_nan_for_identical_spans():
    """A singular value can land at 1 + 1e-16 and arccos it into a NaN, from two identical spans."""
    a = subspace.orthonormalise(_span())
    angles = subspace.principal_angles(a, a)
    assert not torch.isnan(angles).any()
    assert float(angles.max()) == pytest.approx(0.0, abs=1e-7)


def test_a_non_matrix_input_is_refused_with_a_readable_message():
    with pytest.raises(ValueError, match="2-D"):
        subspace.orthonormalise(torch.zeros(4))


def test_the_distance_is_symmetric():
    a, b = _span(seed=1), _span(seed=2)
    assert subspace.compare(a, b).distance == pytest.approx(subspace.compare(b, a).distance)


def test_the_distance_never_leaves_its_stated_range():
    """It is documented as living in [0, 1], and a reader will rely on that."""
    for seed in range(12):
        a, b = _span(seed=seed), _span(seed=seed + 100)
        d = subspace.compare(a, b).distance
        assert 0.0 <= d <= 1.0 and not math.isnan(d)


# ── the runnable self-check ─────────────────────────────────────────────────────────
def test_the_self_check_passes_with_no_control(capsys):
    """A rotated, bfloat16 round-tripped copy of one span must compare as the same subspace.

    If this fails the instrument is too strict to be useful, which is a different failure from
    being too lenient and needs a different message.
    """
    assert subspace.main([]) == 0
    out = capsys.readouterr().out
    assert "same subspace" in out
    assert "OK:" in out


@pytest.mark.parametrize("control", ["replace", "drop", "noise"])
def test_every_control_forces_a_failure_and_a_non_zero_exit(control, capsys):
    """THE SPRINT'S VERIFICATION LINE.

    An instrument whose whole output is "these two agree" has to be shown saying no, once per
    shape of disagreement it claims to catch.
    """
    assert subspace.main(["--control", control]) == 1
    out = capsys.readouterr().out
    assert "DIFFERENT" in out
    assert "was caught" in out


def test_a_lenient_tolerance_can_be_asked_for_and_is_reported(capsys):
    # The tolerance is an input, so the output has to say which one produced the verdict.
    subspace.main(["--tolerance", "0.5"])
    assert "5e-01" in capsys.readouterr().out


def test_an_unknown_control_is_refused():
    with pytest.raises(ValueError, match="unknown control"):
        subspace._reference_pair("wobble")


def test_the_self_check_would_notice_an_instrument_that_always_agrees(monkeypatch, capsys):
    """The mutation this file exists to catch, run rather than described.

    If `compare` were replaced by something that always says yes, every control must start
    reporting FAILED rather than quietly returning success.
    """
    always_same = subspace.Comparison(distance=0.0, disagreeing=0.0, rank_a=8, rank_b=8,
                                      max_angle_degrees=0.0, tolerance=1e-2)
    monkeypatch.setattr(subspace, "compare", lambda *a, **k: always_same)
    assert subspace.main(["--control", "replace"]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "cannot say no" in out
