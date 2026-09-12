# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`--sparsity` and `--ablation-rounds` do what each of them says when used together.

THE DEFECT, found 2026-09-09 and warned about rather than fixed until 2026-09-12. `--sparsity`
promises to leave the low-projection output rows pristine. The refinement rounds re-projected
EVERY row with no mask, so the rows sparsity was holding back were edited anyway, one round
later. Measured on a 64-row matrix before the fix: sparsity 0.9 edited the 6 rows it should at
rounds 0, and all 64 at rounds 4.

Recomputing the mask per round would not have fixed it. The top rows by edit magnitude change
once the first edit lands, so the "untouched" set would drift and the promise would still be
broken, only less visibly. The mask is chosen once, from the first delta.
"""
import torch

from senbonzakura import cli

H, IN = 64, 96


def _setup(seed=0):
    torch.manual_seed(seed)
    R = torch.nn.functional.normalize(torch.randn(1, H), dim=1)
    return R, torch.randn(H, IN)


def _edited(W0, W, tol=1e-4):
    """Rows that moved by more than float32 round-trip noise."""
    return int(((W - W0).norm(dim=1) / W0.norm(dim=1) > tol).sum())


def _leak(R, W0, W):
    return (R @ W).norm().item() / (R @ W0).norm().item()


def test_sparsity_edits_the_same_rows_whether_or_not_rounds_are_on():
    """The property the flags promise together, and the one that was broken."""
    R, W0 = _setup()
    counts = []
    for rounds in (0, 4):
        W = W0.clone()
        cli.orthogonalize_np_(W, R, 1.0, sparsity=0.9, rounds=rounds)
        counts.append(_edited(W0, W))
    keep = max(1, round(0.1 * H))
    assert counts == [keep, keep], (
        f"sparsity 0.9 edited {counts[0]} rows with no rounds and {counts[1]} with four; it "
        f"should edit {keep} in both cases")


def test_the_rounds_still_do_their_job_inside_the_mask():
    """Fixing the mask must not turn the rounds off: they still close the norm-restore leak on
    the rows they ARE allowed to touch.
    """
    R, W0 = _setup()
    leaks = []
    for rounds in (0, 4):
        W = W0.clone()
        cli.orthogonalize_np_(W, R, 1.0, sparsity=0.9, rounds=rounds)
        leaks.append(_leak(R, W0, W))
    assert leaks[1] < leaks[0] / 4, (
        f"the refinement rounds left the leak at {leaks[1]:.3f} against {leaks[0]:.3f}, so "
        f"masking them has stopped them working rather than scoping them")


def test_sparsity_zero_is_untouched_by_the_change():
    """Every run this project has published used sparsity 0. That path must be identical."""
    R, W0 = _setup()
    W = W0.clone()
    cli.orthogonalize_np_(W, R, 1.0, sparsity=0.0, rounds=4)
    assert _edited(W0, W) == H
    assert _leak(R, W0, W) < 1e-4, "with no mask, four rounds should close the leak entirely"


def test_the_fused_expert_path_has_the_same_property():
    """MoE takes `orthogonalize_np_3d_`, and a fix applied to one path and not its twin is the
    mistake this project made three times in two days.
    """
    torch.manual_seed(1)
    E, OUT, IN3 = 3, 32, 48
    R = torch.nn.functional.normalize(torch.randn(1, OUT), dim=1)
    W0 = torch.randn(E, OUT, IN3)
    counts = []
    for rounds in (0, 4):
        W = W0.clone()
        cli.orthogonalize_np_3d_(W, R, 1.0, sparsity=0.9, rounds=rounds)
        rel = (W - W0).norm(dim=2) / W0.norm(dim=2)
        counts.append(int((rel > 1e-4).sum()))
    assert counts[0] == counts[1], (
        f"the fused path edited {counts[0]} (expert, row) pairs with no rounds and {counts[1]} "
        f"with four")
