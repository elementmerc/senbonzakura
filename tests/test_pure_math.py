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
    assert cli.WORST_SCORE == float("inf")


# ── _available_ram_bytes ─────────────────────────────────────────────────────────────
def test_available_ram_bytes():
    v = cli._available_ram_bytes()
    assert v is None or (isinstance(v, int) and v > 0)
