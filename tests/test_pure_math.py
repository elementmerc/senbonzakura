"""Tests for the pure weight-math helpers in senbonzakura.cli (no model needed)."""
import types

import pytest
import torch

from senbonzakura import cli


# ── layer_weight: the windowed strength profile ────────────────────────────────────
def test_layer_weight_peak_taper_and_zero():
    # Peaks at P, tapers linearly to wmin at distance D, zero beyond D.
    assert cli.layer_weight(10, 10, 1.0, 0.2, 4) == pytest.approx(1.0)     # at the peak
    assert cli.layer_weight(14, 10, 1.0, 0.2, 4) == pytest.approx(0.2)     # at distance D
    assert cli.layer_weight(6, 10, 1.0, 0.2, 4) == pytest.approx(0.2)      # symmetric
    assert cli.layer_weight(20, 10, 1.0, 0.2, 4) == 0.0                    # beyond D
    mid = cli.layer_weight(12, 10, 1.0, 0.2, 4)                            # halfway
    assert 0.2 < mid < 1.0


# ── _orth_to ────────────────────────────────────────────────────────────────────────
def test_orth_to_removes_component():
    b = torch.tensor([1.0, 0.0, 0.0])
    v = torch.tensor([3.0, 2.0, 1.0])
    out = cli._orth_to(v, [b])
    assert out[0] == pytest.approx(0.0)     # component along b removed
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(1.0)


# ── orthogonalize_np_ (2D norm-preserving ablation) ────────────────────────────────
def test_orthogonalize_np_preserves_row_norms_and_attenuates_direction():
    # The ablation removes a direction from the OUTPUT (row) space, so R is [K, out_dim].
    torch.manual_seed(1)
    out_dim, in_dim = 6, 4
    W = torch.randn(out_dim, in_dim).double()
    W_orig = W.clone()
    before_norms = W.norm(dim=1).clone()
    r = torch.randn(out_dim).double(); r = r / r.norm()
    R = r.view(1, out_dim)
    before_proj = (r @ W_orig).abs().sum()
    cli.orthogonalize_np_(W, R, 1.0)
    # Row norms are preserved EXACTLY (that's the guarantee the norm-preserving bake makes).
    assert torch.allclose(before_norms, W.norm(dim=1), atol=1e-6)
    # The component along r is clearly attenuated. It is not driven to exactly zero, because the
    # row-norm restoration and exact orthogonality are in mild tension (an accepted trade-off).
    after_proj = (r @ W).abs().sum()
    assert after_proj < before_proj


def test_orthogonalize_np_3d_matches_2d_per_expert():
    torch.manual_seed(2)
    E, out, inn = 3, 5, 4
    W3 = torch.randn(E, out, inn).double()
    r = torch.randn(out).double(); r = r / r.norm()   # R is [K, out_dim]
    R = r.view(1, out)
    W3c = W3.clone()
    cli.orthogonalize_np_3d_(W3, R, 1.0)
    # Each expert slab equals the 2D op applied to that slab.
    for e in range(E):
        slab = W3c[e].clone()
        cli.orthogonalize_np_(slab, R, 1.0)
        assert torch.allclose(slab, W3[e], atol=1e-6)


# ── _axis_separation (Cohen's d, the P1 filter) ────────────────────────────────────
def test_axis_separation_high_for_separable_clouds():
    v = torch.tensor([1.0, 0.0])
    bad = torch.stack([torch.tensor([5.0, 0.0])] * 10) + torch.randn(10, 2) * 0.1
    good = torch.stack([torch.tensor([-5.0, 0.0])] * 10) + torch.randn(10, 2) * 0.1
    d = cli._axis_separation(bad, good, v)
    assert d > 2.0     # well-separated along v


def test_axis_separation_low_for_overlapping_clouds():
    v = torch.tensor([1.0, 0.0])
    torch.manual_seed(3)
    bad = torch.randn(50, 2)
    good = torch.randn(50, 2)
    d = cli._axis_separation(bad, good, v)
    assert d < cli.MIN_AXIS_SEPARATION   # no separation -> would be dropped


# ── the filter cannot be passed, and that is the bug ──────────────────────────────────
def _candidate_axes(Rb, Rg, n=5):
    """Build candidate axes exactly as `extract_directions` does, and return them with the basis."""
    mb, mg = Rb.mean(0), Rg.mean(0)
    gd = mg / mg.norm()
    d0 = cli._orth_to(mb - mg, [gd]); d0 = d0 / d0.norm()
    basis = [gd, d0]

    Xc = Rb - Rb.mean(0, keepdim=True)
    for u in basis:
        Xc = Xc - torch.outer(Xc @ u, u)
    Vh = torch.linalg.svd(Xc, full_matrices=False)[2]

    out = []
    for j in range(n):
        v = cli._orth_to(Vh[j], basis)
        out.append(v / v.norm())
    return out


def _clouds_with_a_real_second_direction(seed=0, N=200, H=64):
    torch.manual_seed(seed)
    Rg = torch.randn(N, H)
    Rb = torch.randn(N, H)
    Rb[:, 0] += 4.0                      # the difference-of-means direction
    Rb[:, 1] += 3.0                      # a genuine SECOND separating direction
    Rb[:, 2] += torch.randn(N) * 5.0     # within-harmful spread that separates nothing
    return Rb, Rg


def test_the_data_really_does_carry_a_second_separating_direction():
    """The control for the test below: without this, a zero would prove nothing."""
    Rb, Rg = _clouds_with_a_real_second_direction()
    e1 = torch.zeros(Rb.shape[1]); e1[1] = 1.0
    assert cli._axis_separation(Rb, Rg, e1) > 2.0


def test_every_candidate_axis_scores_zero_because_it_is_orthogonal_to_both_means(recwarn):
    """The structural defect, pinned (2026-08-03).

    Candidates are orthogonalised against a basis spanning both class means, and Cohen's d is a
    difference of class means, so the numerator is exactly zero however separable the data is.
    The filter is unsatisfiable rather than strict, and no positive threshold changes that.

    This test asserts the CURRENT broken behaviour on purpose, so that a fix has to come here
    and say what it changed. Its partner below asserts the behaviour we actually want.
    """
    Rb, Rg = _clouds_with_a_real_second_direction()
    for v in _candidate_axes(Rb, Rg):
        assert cli._axis_separation(Rb, Rg, v) < 1e-6


@pytest.mark.xfail(strict=True, reason=(
    "The refusal-separation filter cannot pass any axis: candidates are orthogonalised against a "
    "basis spanning both class means and the statistic is a difference of class means, so d is "
    "structurally 0. Documented in private/research/"
    "2026-08-03-the-separation-filter-can-never-pass.md. strict=True so that whichever replacement "
    "statistic is chosen, this test goes red the moment it starts working and has to be un-xfailed "
    "deliberately rather than drifting green unnoticed."))
def test_a_genuinely_separating_second_axis_can_be_kept():
    """The property the multi-direction claim depends on, and which has never held.

    No test in this suite ever required a candidate axis to PASS the filter, which is why a
    filter that rejects everything survived every review. A rejection path tested only with
    rejections is not tested.
    """
    Rb, Rg = _clouds_with_a_real_second_direction()
    assert any(cli._axis_separation(Rb, Rg, v) >= cli.MIN_AXIS_SEPARATION
               for v in _candidate_axes(Rb, Rg))


# ── _profiles_from_params ───────────────────────────────────────────────────────────
def test_profiles_per_component():
    p = {"o_max_weight_position": 10, "o_max_weight": 0.5, "o_min_weight": 0.1,
         "o_min_weight_distance": 3, "d_max_weight_position": 12, "d_max_weight": 0.7,
         "d_min_weight": 0.2, "d_min_weight_distance": 4}
    prof = cli._profiles_from_params(p)
    assert prof == (10, 0.5, 0.1, 3, 12, 0.7, 0.2, 4)


def test_profiles_mlp_off():
    p = {"o_max_weight_position": 10, "o_max_weight": 0.5, "o_min_weight": 0.1,
         "o_min_weight_distance": 3}   # no d_* keys
    prof = cli._profiles_from_params(p)
    assert prof[4] == 10 and prof[5] == 0.0    # MLP profile pinned to zero strength


def test_profiles_uniform():
    p = {"max_weight_position": 8, "max_weight": 0.6, "min_weight": 0.15, "min_weight_distance": 2}
    prof = cli._profiles_from_params(p)
    assert prof == (8, 0.6, 0.15, 2, 8, 0.6, 0.15, 2)


def test_profiles_clamps_negative_d_max():
    p = {"o_max_weight_position": 10, "o_max_weight": 0.5, "o_min_weight": 0.1,
         "o_min_weight_distance": 3, "d_max_weight_position": 12, "d_max_weight": -0.3,
         "d_min_weight": 0.2, "d_min_weight_distance": 4}
    prof = cli._profiles_from_params(p)
    assert prof[5] == 0.0    # negative d_max_weight clamped to 0 (MLP untouched)


# ── _scalar_of ──────────────────────────────────────────────────────────────────────
def _trial(attrs):
    return types.SimpleNamespace(user_attrs=attrs, number=0)


def test_scalar_of_worst_for_damaged_or_unmeasured():
    assert cli._scalar_of(_trial({})) == cli.WORST_SCORE
    assert cli._scalar_of(_trial({"kl": 9.9, "broken": 0.0, "refusals": 0.0})) == cli.WORST_SCORE
    assert cli._scalar_of(_trial({"kl": 0.01, "broken": 0.9, "refusals": 0.0})) == cli.WORST_SCORE


def test_scalar_of_intact():
    s = cli._scalar_of(_trial({"kl": 0.05, "broken": 0.0, "refusals": 0.1, "soft": 0.05, "heretic": 0.2}))
    assert s == pytest.approx(0.1 + 0.05 + 0.5 * 0.2)
    assert float("inf") == cli.WORST_SCORE


# ── _available_ram_bytes ─────────────────────────────────────────────────────────────
def test_available_ram_bytes():
    v = cli._available_ram_bytes()
    assert v is None or (isinstance(v, int) and v > 0)
