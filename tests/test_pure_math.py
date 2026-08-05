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


# ── the two geometric facts that were misdiagnosed on 2026-08-03 ──────────────────────
def test_orthogonalising_the_extras_does_not_change_the_ablated_subspace():
    """Stated wrongly, out loud, before it was checked. Pinned so it is not restated.

    It looked as though orthogonalising each cluster direction against d0 was throwing away the
    refusal component and leaving only topic, and that removing the orthogonalisation would be
    the fix. Gram-Schmidt preserves the span, so the subspace ablated is identical either way and
    "stop orthogonalising" is a no-op on the surgery. The real difference between this method and
    the published ones is WHICH subspace is chosen, not how its basis is written down.
    """
    torch.manual_seed(0)
    H = 64
    d0 = torch.randn(H); d0 = d0 / d0.norm()
    c1 = 0.8 * d0 + 0.6 * torch.randn(H); c1 = c1 / c1.norm()
    c2 = 0.7 * d0 + 0.7 * torch.randn(H); c2 = c2 / c2.norm()

    v1 = cli._orth_to(c1, [d0]); v1 = v1 / v1.norm()
    v2 = cli._orth_to(c2, [d0, v1]); v2 = v2 / v2.norm()

    def projector(M):
        q, _ = torch.linalg.qr(M.T)
        return q @ q.T

    raw = projector(torch.stack([d0, c1, c2]))
    orth = projector(torch.stack([d0, v1, v2]))
    assert torch.allclose(raw, orth, atol=1e-5), "Gram-Schmidt changed the span, which it cannot"


def test_the_bake_requires_an_orthonormal_direction_set():
    """Why the orthogonalisation cannot simply be dropped, whatever a paper does.

    `orthogonalize_np_` computes R^T (R W), which equals the projection onto span(R) only when R
    is orthonormal. Feed it correlated rows and it over-subtracts along the shared component,
    which is a silent increase in ablation strength wearing the costume of "more directions".
    Anyone tempted to pass raw cluster directions has to defeat this test first.
    """
    torch.manual_seed(1)
    # W is [H, in]: `orthogonalize_np_` projects each COLUMN onto span(R), so the residual
    # dimension is the first axis.
    H, cols = 32, 16
    d0 = torch.randn(H); d0 = d0 / d0.norm()
    c1 = 0.9 * d0 + 0.2 * torch.randn(H); c1 = c1 / c1.norm()   # strongly correlated with d0

    W = torch.randn(H, cols)
    correlated = torch.stack([d0, c1])
    q, _ = torch.linalg.qr(correlated.T)
    orthonormal = q.T

    a, b = W.clone(), W.clone()
    cli.orthogonalize_np_(a, correlated, 1.0)
    cli.orthogonalize_np_(b, orthonormal, 1.0)
    assert not torch.allclose(a, b, atol=1e-4), (
        "a correlated basis and its orthonormalisation produced the same edit, so this test no "
        "longer demonstrates why the orthonormalisation is load-bearing")


def test_a_correlated_basis_removes_more_than_the_subspace_contains():
    """The over-subtraction, measured rather than asserted.

    With a correctly orthonormal basis, projecting twice changes nothing after the first time
    (the projector is idempotent). With correlated rows it keeps eating into the weight, which is
    what makes the extra "direction" look effective when it is only extra strength.
    """
    torch.manual_seed(2)
    H = 24
    d0 = torch.randn(H); d0 = d0 / d0.norm()
    c1 = 0.95 * d0 + 0.1 * torch.randn(H); c1 = c1 / c1.norm()
    correlated = torch.stack([d0, c1])
    q, _ = torch.linalg.qr(correlated.T)

    x = torch.randn(H)
    once_ok = q.T.T @ (q.T @ x)
    twice_ok = q.T.T @ (q.T @ once_ok)
    assert torch.allclose(once_ok, twice_ok, atol=1e-5), "the orthonormal projector is idempotent"

    once_bad = correlated.T @ (correlated @ x)
    twice_bad = correlated.T @ (correlated @ once_bad)
    assert not torch.allclose(once_bad, twice_bad, atol=1e-3), (
        "the correlated 'projector' is idempotent, so it is not over-subtracting after all")


# ── the post-sublayer-norm bake fix (added 2026-08-05) ────────────────────────────────
class _GemmaStyleNorm(torch.nn.Module):
    """Gemma-2's RMSNorm: scales by (1 + weight), with weight initialised to zeros."""

    def __init__(self, H, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(H, generator=g) * 0.5)
        self.eps = 1e-6

    def forward(self, x):
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (n * (1.0 + self.weight.float())).type_as(x)


class _LlamaStyleNorm(torch.nn.Module):
    """Llama's RMSNorm: scales by weight directly, initialised to ones."""

    def __init__(self, H, seed=1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(1.0 + torch.randn(H, generator=g) * 0.3)
        self.eps = 1e-6

    def forward(self, x):
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (n * self.weight.float()).type_as(x)


@pytest.mark.parametrize("norm_cls", [_GemmaStyleNorm, _LlamaStyleNorm])
def test_norm_gain_is_recovered_without_reading_the_weight(norm_cls):
    """Probing with ones recovers the gain for BOTH RMSNorm conventions.

    Reading `norm.weight` directly is silently wrong by exactly one on Gemma-style norms, which
    multiply by (1 + weight) where Llama-style multiply by weight. Since rms(ones) == 1, the
    output on a ones vector IS the gain, whatever the convention.
    """
    H = 32
    norm = norm_cls(H)
    g = cli.norm_gain(norm, H, "cpu", torch.float32)
    expected = norm(torch.ones(1, 1, H)).float().reshape(-1)
    assert torch.allclose(g, expected, atol=1e-5)


def test_folding_the_gain_cuts_the_post_norm_leak_across_seeds():
    """The property the fix exists for, on a Gemma-shaped write path.

    The bake edits the weight producing x; the residual actually receives norm(x). Removing
    span(R) from x leaves R present in norm(x), because the per-channel gain rotates the vector
    off the plane. Folding the gain in targets the right subspace.

    It does not reach zero, and that is expected rather than a shortfall: `orthogonalize_np_` is
    norm-preserving by design (it restores each row's original norm after projecting, which is
    what stopped raw orthogonalisation wrecking KL), and row scaling does not commute with a
    left-side projection. So roughly a seventh of any direction survives on EVERY architecture.
    What this asserts is the part the fold is responsible for: the extra leak caused by the norm.

    Several seeds, because a single one would let a lucky gain vector stand in for the mechanism.
    """
    H, K = 48, 3
    for seed in range(5):
        torch.manual_seed(seed)
        norm = _GemmaStyleNorm(H, seed=seed)
        W = torch.randn(H, 20)
        q, _ = torch.linalg.qr(torch.randn(H, K))
        R = q.T[:K]
        g = cli.norm_gain(norm, H, "cpu", torch.float32)

        def leak(weight, norm=norm, R=R, H=H):
            with torch.no_grad():
                x = weight.T.unsqueeze(1)                  # [cols, 1, H]
                y = norm(x).reshape(x.shape[0], H)
                return float((y @ R.T).norm())

        naive = W.clone()
        cli.orthogonalize_np_(naive, R, 1.0)               # what the bake did before
        folded = W.clone()
        cli.orthogonalize_np_(folded, cli.fold_norm_gain(R, g), 1.0)

        assert leak(folded) < leak(naive) / 1.5, (
            f"seed {seed}: folding did not cut the post-norm leak "
            f"(naive {leak(naive):.4f}, folded {leak(folded):.4f})")


def test_norm_preservation_leaves_part_of_the_direction_standing():
    """A documented trade-off, asserted so it stays visible and cannot drift silently.

    `orthogonalize_np_` restores each row's original norm after projecting, which is what keeps
    KL sane (raw orthogonalisation measured KL 12 to 19). The cost is that row scaling does not
    commute with a left-side projection, so ablating a direction removes most of it and not all
    of it, on every architecture. Anyone reading "the direction was ablated" should know it means
    roughly seven eighths of it.
    """
    torch.manual_seed(0)
    H, K = 64, 3
    W = torch.randn(H, 32)
    q, _ = torch.linalg.qr(torch.randn(H, K))
    R = q.T[:K]

    before = float((R @ W).norm())
    pure = W - R.T @ (R @ W)
    baked = W.clone()
    cli.orthogonalize_np_(baked, R, 1.0)
    after = float((R @ baked).norm())

    assert float((R @ pure).norm()) < 1e-4, "a plain projection should remove the direction fully"
    assert after > 1e-3, "norm preservation is expected to leave a residual; it did not"
    assert after < before * 0.25, (
        f"norm preservation left {after / before:.1%} of the direction standing, which is more "
        f"than the roughly one seventh this trade-off has historically cost")


def test_the_fold_is_a_no_op_when_the_gain_is_uniform():
    """A gain of all-ones cannot rotate anything, so the folded set must span what R spans."""
    torch.manual_seed(2)
    H, K = 24, 2
    q, _ = torch.linalg.qr(torch.randn(H, K))
    R = q.T[:K]
    folded = cli.fold_norm_gain(R, torch.ones(H))

    def projector(M):
        b, _ = torch.linalg.qr(M.T)
        return b @ b.T

    assert torch.allclose(projector(R), projector(folded), atol=1e-5)


def test_the_folded_rows_are_orthonormal_so_the_bake_stays_a_projection():
    """R * g is not orthonormal even when R is, and R^T(RW) is a projection only if it is."""
    torch.manual_seed(3)
    H, K = 40, 4
    q, _ = torch.linalg.qr(torch.randn(H, K))
    R = q.T[:K]
    g = 1.0 + torch.rand(H) * 3.0
    folded = cli.fold_norm_gain(R, g)
    assert torch.allclose(folded @ folded.T, torch.eye(K), atol=1e-5)


def test_an_unused_direction_slot_stays_zero_after_folding():
    """A zero row means "no direction here"; QR would otherwise fill it with an arbitrary one."""
    H = 16
    R = torch.zeros(3, H)
    R[0, 0] = 1.0
    folded = cli.fold_norm_gain(R, 1.0 + torch.rand(H))
    assert folded[1].abs().sum() == pytest.approx(0.0, abs=1e-6)
    assert folded[2].abs().sum() == pytest.approx(0.0, abs=1e-6)
    assert folded[0].norm() == pytest.approx(1.0, abs=1e-5)


def test_the_two_layer_shapes_are_told_apart():
    """Both shapes have a `post_attention_layernorm` and it means opposite things."""
    class Standard(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = _LlamaStyleNorm(8)

    class PostSublayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = _GemmaStyleNorm(8)
            self.post_feedforward_layernorm = _GemmaStyleNorm(8, seed=5)

    assert cli.post_sublayer_norms(Standard()) == (None, None)
    attn, mlp = cli.post_sublayer_norms(PostSublayer())
    assert attn is not None and mlp is not None
