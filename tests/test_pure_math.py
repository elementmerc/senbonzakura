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


# ── why PCA candidates were abandoned, and what replaced them ─────────────────────────
def _pca_candidate_axes(Rb, Rg, n=5):
    """Build candidates the way the extractor did BEFORE 2026-08-03: principal axes of the
    harmful cloud, orthogonalised against the harmless direction and the difference of means.

    Kept as executable documentation. It is the construction the project shipped for its whole
    history, and the reason it was replaced is a fact about geometry that a comment cannot
    demonstrate and this can.
    """
    mb, mg = Rb.mean(0), Rg.mean(0)
    gd = mg / mg.norm()
    d0 = cli._orth_to(mb - mg, [gd]); d0 = d0 / d0.norm()
    basis = [gd, d0]

    Xc = Rb - Rb.mean(0, keepdim=True)
    for u in basis:
        Xc = Xc - torch.outer(Xc @ u, u)
    Vh = torch.linalg.svd(Xc, full_matrices=False)[2]
    return [cli._orth_to(Vh[j], basis) / cli._orth_to(Vh[j], basis).norm() for j in range(n)]


def _clouds_with_two_refusal_modes(seed=0, per_mode=100, H=64):
    """Harmful prompts in two distinct refusal modes, harmless prompts in neither.

    Both modes sit away from the harmless cloud, in different directions. Any method that claims
    to find a refusal SUBSPACE has to find two directions here; a method that finds one has found
    the average of two things that are not the same thing.
    """
    torch.manual_seed(seed)
    Rg = torch.randn(2 * per_mode, H)
    Rb = torch.randn(2 * per_mode, H)
    Rb[:per_mode, 0] += 6.0        # mode one
    Rb[per_mode:, 1] += 6.0        # mode two, independent of the first
    return Rb, Rg


def test_the_data_really_does_carry_two_separating_directions():
    """The control. Without it, a method finding nothing would prove nothing."""
    Rb, Rg = _clouds_with_two_refusal_modes()
    for axis in (0, 1):
        e = torch.zeros(Rb.shape[1]); e[axis] = 1.0
        assert cli._axis_separation(Rb, Rg, e) > 1.0


def test_pca_candidates_score_zero_because_they_are_orthogonal_to_both_means():
    """The defect that made the multi-direction feature inert for the project's whole history.

    Candidates were orthogonalised against a basis spanning both class means, and the statistic
    judging them is a difference of class means, so the numerator is exactly zero however
    separable the data is. Not strict: unsatisfiable. Kept as a regression test so nobody
    reintroduces the construction, and as the evidence behind the rewrite.
    """
    Rb, Rg = _clouds_with_two_refusal_modes()
    for v in _pca_candidate_axes(Rb, Rg):
        assert cli._axis_separation(Rb, Rg, v) < 1e-4


def test_a_cluster_candidate_separates_where_a_pca_candidate_cannot():
    """The replacement, on the same data, measured the same way.

    Each cluster's own difference-of-means is separating BY CONSTRUCTION, and what survives
    orthogonalisation against d0 is the part of that cluster's refusal the global mean difference
    misses. That residue is what a second direction is, and it is what the old construction threw
    away before measuring.
    """
    Rb, Rg = _clouds_with_two_refusal_modes()
    mg = Rg.mean(0)
    gd = mg / mg.norm()
    d0 = cli._orth_to(Rb.mean(0) - mg, [gd]); d0 = d0 / d0.norm()
    basis = [gd, d0]

    labels = cli._kmeans_labels(Rb, 2, seed=42)
    best = 0.0
    for c in labels.unique():
        rows = Rb[labels == c]
        if rows.size(0) < cli.MIN_CLUSTER_ROWS:
            continue
        v = cli._orth_to(rows.mean(0) - mg, basis)
        if v.norm() < 1e-6:
            continue
        v = v / v.norm()
        # Scored against the cluster's own rows. Against the whole harmful cloud the global mean
        # is orthogonal to v by construction and this collapses to zero again.
        best = max(best, cli._axis_separation(rows, Rg, v))

    assert best >= cli.MIN_AXIS_SEPARATION, (
        f"the replacement kept nothing either: best cluster separation was {best:.4f} against a "
        f"threshold of {cli.MIN_AXIS_SEPARATION}")


def test_scoring_a_cluster_direction_against_the_whole_cloud_reproduces_the_bug():
    """The one line that would silently undo the rewrite.

    Measuring a cluster's direction against every harmful row instead of that cluster's rows
    puts the global mean back in the numerator, where it is orthogonal to the candidate by
    construction. The score returns to zero and the filter is unsatisfiable again.
    """
    Rb, Rg = _clouds_with_two_refusal_modes()
    mg = Rg.mean(0)
    gd = mg / mg.norm()
    d0 = cli._orth_to(Rb.mean(0) - mg, [gd]); d0 = d0 / d0.norm()

    labels = cli._kmeans_labels(Rb, 2, seed=42)
    c = labels.unique()[0]
    rows = Rb[labels == c]
    v = cli._orth_to(rows.mean(0) - mg, [gd, d0])
    v = v / v.norm()

    assert cli._axis_separation(rows, Rg, v) >= cli.MIN_AXIS_SEPARATION   # correct scoring
    assert cli._axis_separation(Rb, Rg, v) < 1e-4                          # the trap


# ── k-means ───────────────────────────────────────────────────────────────────────────
def test_kmeans_recovers_planted_clusters():
    torch.manual_seed(3)
    X = torch.cat([torch.randn(40, 6) + 10.0, torch.randn(40, 6) - 10.0])
    labels = cli._kmeans_labels(X, 2, seed=1)
    assert len(labels.unique()) == 2
    assert len(labels[:40].unique()) == 1 and len(labels[40:].unique()) == 1


def test_kmeans_is_deterministic_for_a_seed():
    """Two runs on the same input must produce byte-identical output, directions included."""
    torch.manual_seed(3)
    X = torch.randn(60, 8)
    assert torch.equal(cli._kmeans_labels(X, 4, seed=7), cli._kmeans_labels(X, 4, seed=7))


def test_kmeans_never_returns_more_clusters_than_rows():
    X = torch.randn(3, 5)
    assert len(cli._kmeans_labels(X, 10, seed=0).unique()) <= 3


def test_kmeans_survives_identical_rows():
    """Every squared distance is zero, so the ++ seeding has nothing to sample from."""
    X = torch.ones(20, 4)
    labels = cli._kmeans_labels(X, 5, seed=0)
    assert labels.shape == (20,)
    assert len(labels.unique()) >= 1


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
