# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The branches of `cli.py`'s small helpers that the suite has never taken.

WHY THIS FILE EXISTS

Measured on 2026-09-11, three runs on the same tree. With every release artefact present the
suite covers 93.57% of lines by Codecov's reckoning, and what is left is 351 lines that never
run and **200 lines that do run but whose condition only ever goes one way**. `cli.py` alone
holds 54 of those partial branches and 74 cold lines.

A partial branch is the cheaper defect and the more interesting one. The line executes, so the
code is reachable and something already exercises it; what has never happened is the OTHER
answer to its question. That is precisely where a guard can be wrong without anything noticing:
`if x is None: return None` is tested by no test that never passes None.

WHAT IS IN SCOPE HERE

The helpers small enough to drive directly: tensor surgery, the residual-write predicate, the
clusterer, the matched-control scorer, the track digests. Deep guards inside `extract_directions`
and `_run_search` are NOT here; they need the whole search driven around them and they belong
with that fixture rather than in a helper file.

Everything runs on CPU, on tensors of a few rows. Nothing here needs a card, a model, a corpus
or a network, which is the point: this is the part of the gate that can be held on any machine.
"""
import math
import os

import pytest
import torch

from senbonzakura import cli

# ── _sparsify_rows_: the sparsity<=0 short circuit, and the surgery itself ───────────────────

def test_sparsify_returns_the_delta_untouched_when_sparsity_is_off():
    """Zero sparsity must be the identity, not "keep everything by coincidence".

    The default is 0.0, so this branch is the one nearly every run takes, and it returns the
    SAME object rather than an equal one. A copy here would be a silent per-layer allocation on
    the hot path.
    """
    delta = torch.randn(4, 6)
    assert cli._sparsify_rows_(delta, 0.0) is delta
    assert cli._sparsify_rows_(delta, -1.0) is delta


def test_sparsify_keeps_only_the_rows_that_write_hardest():
    """The branch the default never takes: rows below the magnitude cut are zeroed entirely."""
    delta = torch.zeros(4, 3)
    delta[0] = 10.0      # loudest
    delta[1] = 5.0
    delta[2] = 1.0
    delta[3] = 0.5       # quietest
    out = cli._sparsify_rows_(delta, 0.5)               # keep half of 4 rows == 2

    assert torch.equal(out[0], delta[0]), "the loudest row must survive untouched"
    assert torch.equal(out[1], delta[1])
    assert float(out[2].abs().sum()) == 0.0, "a row below the cut keeps none of its edit"
    assert float(out[3].abs().sum()) == 0.0


def test_sparsify_never_zeroes_every_row():
    """`max(1, ...)` is load bearing: sparsity 1.0 would otherwise make the edit a no-op.

    An edit that silently does nothing is the failure this project has met most often, so the
    floor of one kept row is asserted rather than assumed.
    """
    delta = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out = cli._sparsify_rows_(delta, 1.0)
    surviving = [i for i in range(4) if float(out[i].abs().sum()) > 0]
    assert len(surviving) == 1, "exactly the single loudest row survives"
    assert surviving == [3], "and it is the one with the largest norm"


# ── orthogonalize_np_3d_: the restore_norms=False path, and the extra rounds ─────────────────

#: The residual-writing axis of a fused stack is axis 1, so the layout is [experts, hidden,
#: inter] and NOT [experts, out, inter] with the hidden size trailing. `_assert_writes_on_axis`
#: refuses the transposed reading loudly, and it refused the first version of these tests, which
#: is the guard doing exactly what it was written for.
H = 4


def _fused_stack(experts=2, hidden=H, inter=3):
    torch.manual_seed(0)
    return torch.randn(experts, hidden, inter)


def test_fused_expert_ablation_removes_the_direction_from_every_expert():
    """The ordinary path. Named so the contrast with the branch below is visible."""
    W = _fused_stack()
    R = torch.eye(H)[:1]                                  # one direction, the first basis vector
    cli.orthogonalize_np_3d_(W, R, 1.0)
    left = torch.einsum("kh,ehi->eki", R, W).abs().max()
    before = torch.einsum("kh,ehi->eki", R, _fused_stack()).abs().max()
    assert float(left) < float(before), (
        "restoring per-row norms leaks some of the projection back, which is documented and "
        f"measured at 5 to 46%, but the component must fall; {float(left):.3f} against "
        f"{float(before):.3f}")


def test_fused_expert_ablation_without_norm_restoration_is_a_clean_projection():
    """`restore_norms=False` returns early, and is the ONLY path that projects exactly.

    This branch is why the flag exists: the norm restore undoes part of the ablation, so the
    unrestored path is the one that can be checked against the arithmetic rather than against a
    tolerance.
    """
    W = _fused_stack()
    R = torch.eye(H)[:1]
    cli.orthogonalize_np_3d_(W, R, 1.0, restore_norms=False)
    left = torch.einsum("kh,ehi->eki", R, W).abs().max()
    assert float(left) == pytest.approx(0.0, abs=1e-5), (
        "with no norm restore this is an exact projection and the component must be zero")


def test_sparsity_and_no_norm_restore_leave_the_quiet_rows_alone():
    """Both optional branches at once, which nothing had taken.

    With sparsity on, the unrestored path sparsifies `raw` rather than `delta`, and a row below
    the magnitude cut must come back byte-identical.
    """
    W = _fused_stack()
    before = W.clone()
    R = torch.eye(H)[:1]
    cli.orthogonalize_np_3d_(W, R, 1.0, sparsity=0.75, restore_norms=False)
    unchanged = [(e, r) for e in range(W.shape[0]) for r in range(W.shape[1])
                 if torch.equal(W[e, r], before[e, r])]
    assert unchanged, "at sparsity 0.75 most rows should have been left pristine"


def test_extra_rounds_drive_the_leak_down_rather_than_up():
    """`rounds` re-projects after each restore. Never exercised above zero before this test.

    THE DIRECTION HAS TO MIX COMPONENTS, and that is the interesting part. With an axis-aligned
    direction the projection zeroes one whole row of the stack, the per-row norm restore divides
    a zero row by its clamped norm and leaves it zero, so the removal is already exact and there
    is no leak for a second round to reduce. The documented 5 to 46% leak needs a direction that
    spreads across rows, which is what a real difference-of-means direction is.
    """
    R = torch.zeros(1, H)
    R[0] = torch.tensor([0.5, 0.5, 0.5, 0.5])       # unit, and spread across every row
    a, b = _fused_stack(), _fused_stack()
    cli.orthogonalize_np_3d_(a, R, 1.0, rounds=0)
    cli.orthogonalize_np_3d_(b, R, 1.0, rounds=3)
    left_a = float(torch.einsum("kh,ehi->eki", R, a).abs().max())
    left_b = float(torch.einsum("kh,ehi->eki", R, b).abs().max())
    assert left_a > 0, (
        "a direction spread across rows must leak through the norm restore; if this is zero the "
        "fixture has gone back to being axis-aligned and the test proves nothing")
    assert left_b < left_a, f"three rounds should leave less behind, not more ({left_b} vs {left_a})"


def test_a_transposed_expert_stack_is_refused_rather_than_edited():
    """THE GUARD THAT CAUGHT THE FIRST DRAFT OF THIS FILE.

    Some architectures store the projection transposed. Editing one anyway removes a direction
    from the wrong axis: the run completes, the artefact says it succeeded, and the model's
    refusal behaviour has not moved. That silent success is the failure mode this project has met
    most often, so the refusal is asserted rather than trusted.
    """
    W = torch.randn(2, H + 1, H)            # hidden size trailing, which is the wrong layout
    with pytest.raises(SystemExit):
        cli.orthogonalize_np_3d_(W, torch.eye(H)[:1], 1.0)


# ── _writes_residual: every rank, including the depthwise convolution trap ───────────────────

class _FakeParam:
    """Something with a `.shape` and a `.dim()`, which is all the predicate reads."""

    def __init__(self, *shape):
        self.shape = tuple(shape)

    def dim(self):
        return len(self.shape)


def test_a_thing_with_no_dim_is_not_a_residual_writer():
    """The `dim is None` guard. Anything that is not a tensor reaches here."""
    assert cli._writes_residual(object(), 8) is False


@pytest.mark.parametrize(("shape", "expected", "why"), [
    ((8, 32), True, "a dense down-projection [hidden, inter]"),
    ((16, 32), False, "2-D but the leading axis is not the hidden size"),
    ((4, 8, 32), True, "a fused expert stack [experts, hidden, inter]"),
    ((8, 1, 4), False, "a depthwise Conv1d kernel over the residual width, NOT an expert stack"),
    ((1, 8, 32), False, "3-D with a single leading element is not a stack of experts"),
    ((8,), False, "a 1-D norm weight"),
    ((2, 3, 4, 5), False, "rank 4 is nothing this tool edits"),
])
def test_residual_writer_by_rank_and_shape(shape, expected, why):
    """THE CONV TRAP IS THE REASON THIS IS PARAMETRISED.

    A Conv1d kernel is [channels, in/groups, kernel] and is also 3-D. Reading the hidden size
    "anywhere" rather than in the middle would admit [hidden, 1, L] as an expert stack and edit a
    convolution as though it were an MoE layer.
    """
    assert cli._writes_residual(_FakeParam(*shape), 8) is expected, why


# ── _kmeans_labels: the degenerate corpora that break the seeding loop early ─────────────────

def test_clustering_separates_two_obvious_groups():
    """The ordinary path, so the degenerate ones below are read as the exceptions they are."""
    X = torch.cat([torch.zeros(6, 4), torch.full((6, 4), 10.0)])
    labels = cli._kmeans_labels(X, 2, seed=0)
    assert len(set(labels[:6].tolist())) == 1
    assert len(set(labels[6:].tolist())) == 1
    assert labels[0] != labels[6], "the two clusters must not collapse into one label"


def test_asking_for_more_clusters_than_rows_is_clamped():
    """`k = max(1, min(k, n))`. A small corpus must not ask for empty centres."""
    X = torch.randn(3, 4)
    labels = cli._kmeans_labels(X, 99, seed=0)
    assert len(labels) == 3
    assert set(labels.tolist()) <= {0, 1, 2}


def test_identical_rows_stop_the_seeding_instead_of_picking_noise():
    """THE BRANCH THAT HAD NEVER RUN: every remaining row coincides with a chosen centre.

    `total > 0` is false once all squared distances are zero. Without the break, `multinomial`
    would be handed an all-zero weight vector. A corpus of duplicated prompts is not exotic; it
    is what a badly deduplicated harmful pool looks like.
    """
    X = torch.ones(8, 4)
    labels = cli._kmeans_labels(X, 4, seed=0)
    assert len(labels) == 8
    assert len(set(labels.tolist())) == 1, "identical rows cannot be split into real clusters"


def test_clustering_is_deterministic_for_a_seed():
    """Two runs on the same input must be byte-identical, per baseline section 2.1."""
    X = torch.randn(20, 5)
    assert torch.equal(cli._kmeans_labels(X, 3, seed=7), cli._kmeans_labels(X, 3, seed=7))


# ── _matched_held_out_separation: the three ways it declines to score ────────────────────────

def test_matched_separation_refuses_a_cluster_too_small_to_halve():
    """Below `MIN_HELD_OUT_ROWS` in either half there is nothing to hold out."""
    small = torch.randn(cli.MIN_HELD_OUT_ROWS, 6)
    assert cli._matched_held_out_separation(small, torch.randn(50, 6), torch.randn(50, 6),
                                            [], seed=0) is None


def test_matched_separation_refuses_when_there_are_no_controls_to_match():
    """`_matched_harmless_idx` returns None on an empty harmless pool, and that must propagate.

    Scoring against nothing is how a comparison ends up being made against rows that are not
    controls, which is the defect `matching_quality` exists to detect.
    """
    bad = torch.randn(40, 6)
    empty = torch.zeros(0, 6)
    assert cli._matched_held_out_separation(bad, empty, empty, [], seed=0) is None


def test_matched_separation_refuses_a_direction_with_no_length():
    """The `norm < 1e-6` guard: the cluster mean and its controls coincide.

    There is no axis to score, and normalising would divide by roughly zero. Identical harmful
    and harmless rows is the cleanest way to reach it.
    """
    same = torch.ones(40, 6)
    assert cli._matched_held_out_separation(same, same.clone(), same.clone(), [],
                                            seed=0) is None


def test_matched_separation_scores_a_real_split():
    """The path that returns a number, so the three refusals above are not the only behaviour."""
    torch.manual_seed(0)
    bad = torch.randn(40, 6) + 3.0
    good_fit = torch.randn(40, 6)
    good_score = torch.randn(40, 6)
    got = cli._matched_held_out_separation(bad, good_fit, good_score, [], seed=0)
    assert got is not None and math.isfinite(float(got))


# ── _track_digests: absent, unreadable, and real ─────────────────────────────────────────────

def test_no_track_digests_to_nothing():
    """`--track` is optional, and provenance must not invent a field for a run without one."""
    assert cli._track_digests(None) is None
    assert cli._track_digests("") is None


def test_a_track_path_that_is_not_a_directory_digests_to_nothing(tmp_path):
    """A run recorded a path nobody else can resolve. Returning None is how that is admitted."""
    missing = tmp_path / "not-here"
    assert cli._track_digests(str(missing)) is None

    afile = tmp_path / "a-file"
    afile.write_text("not a track", encoding="utf-8")
    assert cli._track_digests(str(afile)) is None


def test_a_directory_with_no_known_splits_digests_to_nothing(tmp_path):
    """`out or None`: an empty dict is not a provenance record, so it must not become one."""
    (tmp_path / "something-else").mkdir()
    assert cli._track_digests(str(tmp_path)) is None


def test_an_empty_split_directory_still_digests(tmp_path):
    """Learned while writing this: `dataset_digest` walks file bytes, so an empty directory is
    not an error. It hashes to the sha256 of nothing, which is a real answer to a real question.
    """
    (tmp_path / "bad_ds").mkdir()
    got = cli._track_digests(str(tmp_path))
    assert list(got) == ["bad_ds"]
    assert got["bad_ds"] != "UNREADABLE"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file, so nothing raises")
def test_an_unreadable_split_is_recorded_rather_than_raising(tmp_path):
    """PROVENANCE MUST NEVER FAIL A RUN, and it must not quietly omit the split either.

    The except branch had never been taken. Omitting the key would make an unreadable split
    indistinguishable from an absent one in the artefact; "UNREADABLE" says which it was.
    """
    split = tmp_path / "bad_ds"
    split.mkdir()
    blocked = split / "data.arrow"
    blocked.write_bytes(b"rows")
    blocked.chmod(0o000)
    try:
        got = cli._track_digests(str(tmp_path))
    finally:
        blocked.chmod(0o600)
    assert got == {"bad_ds": "UNREADABLE"}


# ── _fold_for: the no-norm short circuit against the folded path ─────────────────────────────

class _Gemma(torch.nn.Module):
    """Gemma-style: multiplies by `1 + weight`, initialised to zeros."""

    def __init__(self, h):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(h))

    def forward(self, x):
        return x * (1.0 + self.weight)


class _Bakerish:
    """Only the three things `_fold_for` reads."""

    H = H
    dev = "cpu"
    _fold_for = cli.Abliterator._fold_for


def test_fold_for_returns_the_directions_unchanged_when_nothing_intercepts():
    """No norm between the edited weight and the residual, so R is already the right basis.

    The `norm is None` short circuit, which is the common case and was only ever taken one way.
    """
    R = torch.eye(H)[:2]
    assert _Bakerish()._fold_for(R, None) is R


def test_fold_for_folds_the_gain_when_a_norm_intercepts_the_write():
    """The other side: a norm is present, so the directions must be re-expressed through it."""
    R = torch.eye(H)[:2]
    got = _Bakerish()._fold_for(R, _Gemma(H))
    assert got.shape == R.shape
    assert torch.isfinite(got).all()


def test_norm_gain_is_probed_rather_than_read_off_the_weight():
    """Gemma-style norms multiply by `1 + weight` where Llama-style multiply by `weight`.

    Reading the attribute directly is silently wrong by exactly one on half the supported
    architectures, which is why the gain is recovered by pushing a vector of ones through.
    """
    g = cli.norm_gain(_Gemma(H), H, "cpu", torch.float32)
    assert torch.allclose(g, torch.ones(H)), (
        "a Gemma-style norm at its initial weights applies a gain of one; reading .weight "
        "directly would have said zero")


def test_folding_a_non_uniform_gain_moves_the_directions():
    """LEARNED WHILE WRITING THIS, and worth stating because it looks like a bug and is not.

    `fold_norm_gain` re-orthonormalises, so a gain that scales an already-orthogonal basis
    uniformly comes back unchanged: QR preserves the span, and the span did not move. Only a gain
    that makes the rows non-orthogonal to each other actually changes the basis, so the test has
    to use directions that mix components rather than axis-aligned ones.
    """
    R = torch.zeros(2, H)
    R[0, 0] = R[0, 1] = 1 / math.sqrt(2)
    R[1, 0], R[1, 1] = 1 / math.sqrt(2), -1 / math.sqrt(2)

    unchanged = cli.fold_norm_gain(R, torch.full((H,), 3.0))
    assert torch.allclose(unchanged.abs(), R.abs(), atol=1e-5), (
        "a uniform gain scales the rows without tilting them, so the span is the same")

    g = torch.tensor([1.0, 5.0, 1.0, 1.0])
    moved = cli.fold_norm_gain(R, g)
    assert not torch.allclose(moved.abs(), R.abs(), atol=1e-3), (
        "a gain that differs across components tilts the rows, so the basis must move")

    gram = moved @ moved.T
    assert torch.allclose(gram, torch.eye(2), atol=1e-5), (
        "the bake's R^T (R W) is a projection only for an orthonormal basis, so the folded rows "
        "must still be orthonormal")


def test_an_unused_direction_slot_stays_zero_after_folding():
    """A zero row must not be replaced by whatever QR puts in an empty column.

    `keep = M.norm(dim=1) > 1e-8` exists for this and had only ever been true. If it were
    dropped, an unused slot would come back holding an arbitrary direction, and the bake would
    ablate something nothing had selected.
    """
    R = torch.zeros(3, H)
    R[0, 0] = 1.0
    R[1, 1] = 1.0
    # row 2 left empty, as an unfilled direction slot
    got = cli.fold_norm_gain(R, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert float(got[2].abs().sum()) == 0.0, "the unused slot came back carrying a direction"
    assert float(got[0].norm()) == pytest.approx(1.0, abs=1e-5)
