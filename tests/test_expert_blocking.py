# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The fused-expert rewrite edits a block of experts at a time, and that must not change it.

`orthogonalize_np_3d_` built four [E, out, in] float32 intermediates for a rewrite that is
separable per expert, so editing a 128-expert stack cost several gigabytes of working set for a
tensor the checkpoint stores in under one. That is a capacity wall on the laptop card the
streaming milestone is aiming at, and chunking it is the fix the v0.5 plan asks for.

Chunking a numerical routine is easy to get subtly wrong and hard to notice, because the output
stays plausible. So the claim is split into the part that is exact and the part that is not, and
the split was MEASURED rather than assumed. The first version of this file asserted that blocking
was bit-exact at every dtype, which is false, and the honest reading took three diagnostics:

- Slicing the stack by hand outside the routine and slicing it inside agree **bit-exactly**. That
  is the structural claim, and it is the one that would break if blocking changed semantics.
- The row norms and the first einsum are bit-exact at any block size.
- The second einsum is not, by about 1.5 float32 units in the last place. Its contraction index
  runs over the refusal span and never over experts, so this is the BLAS kernel taking a different
  path for a different batch size, not a reduction that crosses a block boundary.
- Cast back to bfloat16 or float16 and the difference disappears entirely.

The last point is the one that decides whether any of this matters: those are the dtypes real
checkpoints are stored in, so for the models this tool actually edits, blocking changes nothing
that reaches the file.
"""
import pytest
import torch

from senbonzakura import cli

#: A block wider than any stack here, so the routine runs in one pass: the pre-blocking behaviour,
#: kept reachable so the blocked path has something to be compared against.
UNBLOCKED = 10 ** 6

#: How far the float32 paths may disagree, as a multiple of float32's unit in the last place.
#: Measured at about 1.5; set with headroom for a different BLAS on a different CI row, and still
#: some seven orders of magnitude tighter than any logic error could hide under, because the edit
#: it is checking is O(1) relative to the weights.
ULP_BUDGET = 8

FLAGS = [
    (0.0, 0, True),
    (0.0, 3, True),
    (0.75, 0, True),
    (0.75, 3, True),
    (0.0, 0, False),
    (0.75, 0, False),
]


def _stack(experts, out, inn, seed, dtype=torch.float32):
    torch.manual_seed(seed)
    return torch.randn(experts, out, inn).to(dtype)


def _span(out, k, seed):
    """A [K, out] unit-row span to ablate, the shape the bake extracts."""
    torch.manual_seed(seed + 500)
    R = torch.randn(k, out)
    return R / R.norm(dim=1, keepdim=True)


# 8 is the block, so 8 fits exactly, 9 and 13 leave a short final block, 3 never reaches a
# boundary, and 1 is the degenerate stack.
@pytest.mark.parametrize("experts", [1, 3, 8, 9, 13, 16])
@pytest.mark.parametrize(("sparsity", "rounds", "restore_norms"), FLAGS)
def test_blocking_is_a_faithful_slice_of_the_stack(experts, sparsity, rounds, restore_norms):
    """THE STRUCTURAL CLAIM, and it is exact.

    Editing one expert at a time inside the routine gives byte-for-byte what editing each expert
    outside the routine gives. That is what "separable" has to mean, and it is what a mistake in
    the blocking would break: a mask chosen over the whole stack, a norm reduced across experts, a
    ragged final block quietly dropped. None of those survive this, and none of them are visible to
    a test that only compares a blocked run against an unblocked one at a tolerance.
    """
    R = _span(5, 2, seed=experts)
    source = _stack(experts, 5, 4, seed=experts)

    inside = source.clone()
    cli.orthogonalize_np_3d_(inside, R, 1.0, sparsity=sparsity, rounds=rounds,
                             restore_norms=restore_norms, block=1)

    outside = []
    for e in range(experts):
        slab = source[e:e + 1].clone()
        cli.orthogonalize_np_3d_(slab, R, 1.0, sparsity=sparsity, rounds=rounds,
                                 restore_norms=restore_norms, block=UNBLOCKED)
        outside.append(slab)

    assert torch.equal(inside, torch.cat(outside)), (
        "editing an expert inside the block loop is not the same as editing it on its own, so the "
        "loop is doing something to the stack beyond slicing it")


@pytest.mark.parametrize("store", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block", [1, 3, 8, 64])
@pytest.mark.parametrize(("sparsity", "rounds", "restore_norms"), FLAGS)
def test_blocking_changes_nothing_that_reaches_a_checkpoint(
        store, block, sparsity, rounds, restore_norms):
    """THE CLAIM THAT DECIDES WHETHER ANY OF THIS MATTERS.

    The routine computes in float32 and casts back to the weight's own dtype on the way out. Real
    checkpoints are bfloat16 or float16, and at both the blocked and unblocked rewrites produce
    identical bytes: the float32 disagreement is far below the storage precision, so it is rounded
    away by the cast that was always there.

    A float32 checkpoint is the exception and is covered separately, at a stated tolerance, because
    there the difference does survive.
    """
    experts = 13
    R = _span(5, 2, seed=experts)
    blocked = _stack(experts, 5, 4, seed=experts, dtype=store)
    whole = blocked.clone()

    cli.orthogonalize_np_3d_(blocked, R, 1.0, sparsity=sparsity, rounds=rounds,
                             restore_norms=restore_norms, block=block)
    cli.orthogonalize_np_3d_(whole, R, 1.0, sparsity=sparsity, rounds=rounds,
                             restore_norms=restore_norms, block=UNBLOCKED)

    assert torch.equal(blocked, whole), (
        f"blocking at {block} changed the edited weights at {store}, which is a dtype a real "
        f"checkpoint is stored in, so this is no longer a rounding difference below the storage "
        f"precision")


@pytest.mark.parametrize("block", [1, 3, 8, 64])
@pytest.mark.parametrize(("sparsity", "rounds", "restore_norms"), FLAGS)
def test_a_float32_checkpoint_agrees_to_within_rounding(block, sparsity, rounds, restore_norms):
    """The one dtype where the difference survives, bounded rather than waved away.

    Bounded against the magnitude of the weights, so the budget means the same thing whatever the
    seed produces. A real defect in the blocking is not a last-place difference: it would drop an
    expert, mask the wrong rows, or edit a stale copy, and every one of those moves the result by
    something comparable to the edit itself.
    """
    experts = 13
    R = _span(5, 2, seed=experts)
    blocked = _stack(experts, 5, 4, seed=experts)
    whole = blocked.clone()

    cli.orthogonalize_np_3d_(blocked, R, 1.0, sparsity=sparsity, rounds=rounds,
                             restore_norms=restore_norms, block=block)
    cli.orthogonalize_np_3d_(whole, R, 1.0, sparsity=sparsity, rounds=rounds,
                             restore_norms=restore_norms, block=UNBLOCKED)

    budget = ULP_BUDGET * torch.finfo(torch.float32).eps * whole.abs().max().item()
    worst = (blocked - whole).abs().max().item()
    assert worst <= budget, (
        f"blocking at {block} moved a float32 weight by {worst:.3e}, past the {budget:.3e} that "
        f"float32 rounding accounts for, so the difference is arithmetic rather than rounding")


def test_each_expert_still_matches_the_two_dimensional_routine_across_a_boundary():
    """The independent oracle, checked where the loop can actually get it wrong.

    `test_pure_math.py` already pins the 3-D routine to the 2-D one, at three experts, which never
    reaches the boundary at eight. This is the same claim across several boundaries and a ragged
    remainder, and it does not depend on the unblocked path being correct.
    """
    experts, out = 13, 5
    R = _span(out, 1, seed=7)
    fused = _stack(experts, out, 4, seed=7)
    before = fused.clone()

    cli.orthogonalize_np_3d_(fused, R, 1.0)

    for e in range(experts):
        slab = before[e].clone()
        cli.orthogonalize_np_(slab, R, 1.0)
        assert torch.allclose(slab, fused[e], atol=1e-6), (
            f"expert {e} does not match the 2-D routine applied to it alone")


@pytest.fixture
def einsum_sizes(monkeypatch):
    """Record the size of every intermediate the rewrite builds through `torch.einsum`.

    The intermediates are the point: `proj` at [E, K, in] and `delta` at [E, out, in] are the two
    largest tensors in the routine and both are einsum outputs, so measuring them is measuring the
    thing the change was made for.
    """
    seen = []
    real = torch.einsum

    def spy(equation, *operands, **kwargs):
        result = real(equation, *operands, **kwargs)
        seen.append(result.numel())
        return result

    monkeypatch.setattr(torch, "einsum", spy)
    return seen


@pytest.mark.parametrize(("block", "widest"), [(1, 1), (8, 8), (64, 32)])
def test_the_peak_intermediate_is_bounded_by_the_block(einsum_sizes, block, widest):
    """THE HALF THAT WOULD SURVIVE DELETING THE LOOP.

    Every equivalence test above passes just as happily with the blocking removed, because they
    are constraints on the change rather than the reason for it. This is the reason: the largest
    tensor the routine builds scales with the block and not with the stack.
    """
    experts, out, inn = 32, 5, 4
    R = _span(out, 2, seed=11)
    cli.orthogonalize_np_3d_(_stack(experts, out, inn, seed=11), R, 1.0, block=block)

    assert einsum_sizes, "no einsum ran, so this test measured nothing"
    assert max(einsum_sizes) <= widest * out * inn, (
        f"the largest intermediate covers more than {widest} experts, so the rewrite is not being "
        f"blocked and its peak is still proportional to the whole stack")


def test_the_unblocked_peak_is_what_the_blocked_one_improves_on(einsum_sizes):
    """The control, so "bounded by the block" cannot pass by the routine building nothing large.

    Without it, a rewrite that stopped computing `delta` at all would satisfy the bound above and
    read as a win. This pins the other side: unblocked, the peak really does cover every expert,
    which is the number the blocked case is measured against.
    """
    experts, out, inn = 32, 5, 4
    R = _span(out, 2, seed=11)
    cli.orthogonalize_np_3d_(_stack(experts, out, inn, seed=11), R, 1.0, block=UNBLOCKED)

    assert max(einsum_sizes) == experts * out * inn, (
        "the unblocked rewrite no longer builds a whole-stack intermediate, so the blocked test is "
        "being compared against a number that has moved")


@pytest.mark.parametrize("block", [0, -4])
def test_a_nonsense_block_still_edits_every_expert(block):
    """A block of zero or less is a caller error that must not silently skip the rewrite.

    A range with a zero step raises, and a negative one yields nothing at all: the second is the
    shape that would quietly leave the weights unedited and report success, which is the failure
    mode this project refuses everywhere else. It clamps to one instead.
    """
    R = _span(5, 1, seed=3)
    W = _stack(4, 5, 4, seed=3)
    before = W.clone()
    cli.orthogonalize_np_3d_(W, R, 1.0, block=block)
    assert not torch.equal(W, before), f"block={block} left the stack untouched"
