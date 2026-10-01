# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The streaming rewrite of a mixture-of-experts layer, in both layouts a checkpoint uses.

WHY THIS FILE EXISTS, AND WHY IT IS SYNTHETIC ON PURPOSE.

The expert path is the one part of the bake that has never run on real weights at a size the
owned hardware can hold. Every small LiquidAI model is dense; the one that is a mixture of
experts is the 8B, which does not fit the card. So the climb toward it has rungs that all
exercise the dense path and one rung at the top that exercises something nothing below it
touched, and the failure that lives in that gap is not a crash. It is a **silent partial
ablation**: a rewrite that edits some experts and misses others, writes a checkpoint that still
loads, reports success, and produces a model whose refusal behaviour is half removed for reasons
no log mentions.

Two things make the gap real rather than theoretical.

**There are two on-disk layouts and both ship.** Measured against real checkpoints on
2026-09-22: LFM2.5-8B-A1B stores one `[hidden, inner]` tensor per expert, while
Granite-3.0-1b-a400m stores the fused `[experts, hidden, inner]` stack itself. The same logical
weights therefore reach the editor as 32 separate 2-D tensors on one model and as a single 3-D
tensor on the other, down two different code paths: `orthogonalize_np_` and
`orthogonalize_np_3d_`.

**The 3-D path edits a block of experts at a time.** `cli.EXPERT_BLOCK` cuts the working set
about 16x by slicing the stack, which is sound exactly as far as the arithmetic is separable
along the expert axis. A block loop is also the obvious place for an off-by-one to leave the last
short block untouched, and nothing above would say so.

So the checks here build the SAME logical weights in both layouts and assert the two paths agree
elementwise, that every expert moved, and that the refusal span is gone from each expert
separately rather than on average. Agreement is the measurement: either layout alone can be
self-consistently wrong, and the pair cannot.

The checkpoints are synthetic because that is the point. A synthetic mixture of experts costs no
GPU and no download, so this runs on every push rather than on the one rung that can afford the
8B, which is the rung too late to find out.
"""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from senbonzakura import cli, streaming

#: Eleven experts against a block of eight, so the loop runs one full block and one short one.
#: A power of two here would let an off-by-one on the final block pass unnoticed.
EXPERTS = 11

#: Hidden size, and the inner width the expert writer contracts over. Small, because what is
#: being measured is agreement between two code paths rather than anything about scale.
HIDDEN, INNER = 6, 4

#: Two layers, so "the rewrite found the layer" and "the rewrite found every layer" are
#: different assertions.
LAYERS = 2

#: The ablation strength and the directions, fixed so every arm below edits the same way.
STRENGTH, DIRECTIONS = 1.0, 2


def _stack(seed):
    """Deterministic logical expert weights, `[experts, hidden, inner]`.

    One generator with a fixed seed rather than `torch.randn` at module scope: the two layouts
    have to be built from the same numbers for their comparison to mean anything, and a shared
    global generator makes that depend on the order the fixtures happen to run in.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(EXPERTS, HIDDEN, INNER, generator=gen, dtype=torch.float32)


def _refusal_basis(seed=7):
    """An orthonormal `[K, hidden]` basis, which is the shape the bake removes."""
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(HIDDEN, DIRECTIONS, generator=gen, dtype=torch.float32))
    return q.T.contiguous()


def _write(root, tensors, layers):
    """A model directory holding `tensors` in one shard, with the config the index reads.

    The count is passed rather than taken from `LAYERS`, because the index refuses a config whose
    declared count does not match the indices in the names, and a helper that hard-coded the
    constant would make a one-layer case fail on the fixture rather than on the rewrite.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"num_hidden_layers": layers}), encoding="utf-8")
    save_file({k: v.clone() for k, v in tensors.items()}, str(root / "model.safetensors"),
              metadata={"format": "pt"})
    return root


def _per_expert_name(layer, expert):
    """How LFM2.5-8B-A1B spells one expert's residual writer."""
    return f"model.layers.{layer}.feed_forward.experts.{expert}.w2.weight"


def _fused_name(layer):
    """How Granite-3.0-1b-a400m spells the whole stack's residual writer."""
    return f"model.layers.{layer}.block_sparse_moe.output_linear.weight"


def _per_expert_checkpoint(root, stacks):
    tensors = {"model.embed_tokens.weight": torch.zeros(HIDDEN, INNER)}
    for layer, stack in enumerate(stacks):
        for expert in range(EXPERTS):
            tensors[_per_expert_name(layer, expert)] = stack[expert]
    return _write(root, tensors, len(stacks))


def _fused_checkpoint(root, stacks):
    tensors = {"model.embed_tokens.weight": torch.zeros(HIDDEN, INNER)}
    for layer, stack in enumerate(stacks):
        tensors[_fused_name(layer)] = stack
    return _write(root, tensors, len(stacks))


def _ablate(root, basis, *, sparsity=0.0, rounds=0):
    """Drive the real streaming rewrite over `root`, editing every residual writer it finds.

    This is the bake's shape and not a shortcut around it: the layer index decides which tensors
    are writers, `load_tensor` reads one tensor at a time, the project's own ablation edits it in
    place, and `rewrite_shard` streams the shard to a sibling and renames. The dispatch on rank is
    the only thing here that is test code, and it is the same dispatch the editor makes.

    Returns the number of tensors replaced, so a caller can assert the rewrite edited the number
    of things it expected rather than trusting that it edited anything.
    """
    index = streaming.index_layers(root)
    writers = {}
    for layer in range(index.count):
        writers.update(index.writers(layer))
    assert writers, "the index found no residual writers, so nothing below measures the rewrite"

    def edit(tensor, _raw):
        if tensor.name not in writers:
            return None
        weight = streaming.load_tensor(*writers[tensor.name])
        if weight.ndim == 3:
            cli.orthogonalize_np_3d_(weight, basis, STRENGTH, sparsity=sparsity, rounds=rounds)
        else:
            cli.orthogonalize_np_(weight, basis, STRENGTH, sparsity=sparsity, rounds=rounds)
        return weight.numpy().tobytes()

    replaced = 0
    for shard in index.shards():
        replaced += streaming.rewrite_shard(shard, edit)
    return replaced


def _edited_stacks(root, fused, layers=LAYERS):
    """The edited weights read back as `[layers][experts, hidden, inner]`, from either layout."""
    held = load_file(str(root / "model.safetensors"))
    if fused:
        return [held[_fused_name(layer)] for layer in range(layers)]
    return [torch.stack([held[_per_expert_name(layer, e)] for e in range(EXPERTS)])
            for layer in range(layers)]


# ── the two layouts agree ────────────────────────────────────────────────────────────────────


#: `(sparsity, rounds)` combinations where the two layouts are required to agree, which is every
#: combination except the one the divergence check below owns. Both flags default to 0, so the
#: first row is what every shipped run has used.
AGREEING = [(0.0, 0), (0.3, 0), (0.5, 0), (0.0, 4)]


@pytest.mark.parametrize(("sparsity", "rounds"), AGREEING)
def test_the_two_stored_layouts_produce_the_same_edited_weights(tmp_path, sparsity, rounds):
    """THE CHECK THE WHOLE FILE IS FOR.

    Same logical weights, same directions, same strength, stored the two ways real checkpoints
    store them, driven through the real streaming rewrite. If the per-expert path and the fused
    block loop disagree, one of them is wrong and no single-layout test can say which.

    Run with `--sparsity` on as well as off, because the sparsity mask is the part of the fused
    path whose separability is a claim rather than an obvious fact: it takes its top-k over the
    out-row axis INSIDE each expert, so a block boundary must not be able to move which rows stay
    pristine. A mask taken across the whole stack would agree at sparsity 0 and diverge at 0.3,
    which is exactly the shape of defect a sparsity-free test reports clean.

    `--ablation-rounds` on as well as off for the same reason one layer would not have been
    enough: the alternating refinement is a loop inside the block, so it is a second place a block
    boundary could leak into the result.

    The one combination missing from `AGREEING` is both flags together, which does NOT agree. It
    has its own check below rather than being quietly dropped from this list.
    """
    stacks = [_stack(seed=100 + i) for i in range(LAYERS)]
    basis = _refusal_basis()

    per_expert = _per_expert_checkpoint(tmp_path / "lfm", stacks)
    fused = _fused_checkpoint(tmp_path / "granite", stacks)
    _ablate(per_expert, basis, sparsity=sparsity, rounds=rounds)
    _ablate(fused, basis, sparsity=sparsity, rounds=rounds)

    from_per_expert = _edited_stacks(per_expert, fused=False)
    from_fused = _edited_stacks(fused, fused=True)
    for layer, (a, b) in enumerate(zip(from_per_expert, from_fused, strict=True)):
        worst = (a - b).abs().max().item()
        assert worst < 1e-5, (
            f"layer {layer}: the per-expert and fused rewrites disagree by {worst:.3g} at worst, "
            f"so one of the two paths is editing something the other is not")


def test_sparsity_and_rounds_together_make_the_two_layouts_disagree(tmp_path):
    """A MEASURED, OPEN DEFECT, pinned here so it is loud rather than discovered mid-run.

    With `--sparsity` and `--ablation-rounds` BOTH on and two or more directions, the per-expert
    and fused paths produce materially different weights for the same configuration: measured at
    5.4e-02 at K=2 and 2.7e-01 at K=3, against weights of order 1. Turn any one of the three off
    and they agree exactly, which is what the check above pins.

    THE CAUSE, read off the two implementations rather than inferred from the size of the gap.
    Inside the refinement loop, `orthogonalize_np_` subtracts the directions ONE AT A TIME,
    recomputing each projection against the partly updated weight, while
    `_orthogonalize_np_3d_block_` subtracts all of them at once from a single batched projection.
    For an orthonormal basis those are the same operation, which is why they agree at
    `--sparsity 0`. The mask breaks that equivalence: a masked subtraction is not a projection, so
    the order it is applied in changes the result, and the gap grows with the number of
    directions. The block size is not involved, which this file also measures.

    WHY IT IS PINNED RATHER THAN FIXED HERE. Closing it means choosing which of the two orders is
    canonical, and that choice changes what a bake produces, so it belongs to the operator rather
    than to the test that found it. No shipped result is affected: both flags default to 0 and no
    run specification in the tree sets either, so the combination is reachable only by a user who
    opts into both. What is not acceptable is leaving it undetected until a mixture-of-experts run
    is compared against a dense one and the difference is read as a finding about the model.

    So this asserts the divergence is still there and still confined. If a fix lands, this check
    fails and sends its author to read the paragraph above, which is the point.
    """
    stacks = [_stack(seed=100)]
    basis = _refusal_basis()
    per_expert = _per_expert_checkpoint(tmp_path / "lfm", stacks)
    fused = _fused_checkpoint(tmp_path / "granite", stacks)
    _ablate(per_expert, basis, sparsity=0.3, rounds=4)
    _ablate(fused, basis, sparsity=0.3, rounds=4)

    a = _edited_stacks(per_expert, fused=False, layers=1)[0]
    b = _edited_stacks(fused, fused=True, layers=1)[0]
    worst = (a - b).abs().max().item()
    assert worst > 1e-3, (
        f"the two layouts now agree to {worst:.3g} under --sparsity with --ablation-rounds. If "
        f"that is a deliberate fix, say which deflation order won and replace this check; if it "
        f"is accidental, one of the two paths has stopped doing what it did")


def test_a_block_boundary_cannot_move_which_rows_sparsity_holds_pristine(tmp_path):
    """The separability claim, read off the weights rather than off a comparison.

    `EXPERT_BLOCK` is sound because no step of the fused rewrite reduces across experts. The
    sparsity mask is the step where that is least obvious, so this asserts the consequence
    directly: the mask keeps `round((1 - sparsity) * rows)` rows per expert, and that count is the
    same whichever block an expert landed in.

    The tolerance is not slack. Restoring the row lengths divides and re-multiplies every row,
    including the ones the mask held back, so a pristine row comes back changed in the last bit or
    two of float32. Counting any change at all therefore counts arithmetic noise as an edit and
    reports a different number per expert for a mask behaving perfectly: measured
    [5, 5, 4, 5, 3, 3, ...] at a tolerance of zero against a flat 3 at 1e-05.
    """
    stacks = [_stack(seed=200)]
    basis = _refusal_basis()
    before = stacks[0].clone()
    root = _fused_checkpoint(tmp_path / "granite", stacks)
    _ablate(root, basis, sparsity=0.5)
    after = _edited_stacks(root, fused=True, layers=1)[0]

    edited = [int(((before[e] - after[e]).abs().max(dim=1).values > 1e-5).sum())
              for e in range(EXPERTS)]
    expected = max(1, round(0.5 * HIDDEN))
    assert edited == [expected] * EXPERTS, (
        f"--sparsity 0.5 edited {edited} rows by expert and the mask keeps {expected} of "
        f"{HIDDEN}, so the set of pristine rows is not being chosen inside each expert and a "
        f"block boundary is deciding which rows survive")


@pytest.mark.parametrize("sparsity", [0.0, 0.3])
@pytest.mark.parametrize("rounds", [0, 4])
def test_the_block_size_does_not_change_the_fused_result(sparsity, rounds):
    """`EXPERT_BLOCK` is a memory lever and must not be a correctness one.

    The constant exists to cut the working set about 16x, which is only legitimate if a block of
    eight and a block of one compute the same thing. Asserted directly, with the expert count
    deliberately not a multiple of the block so the loop runs one full block and one short one.
    Both flags are swept because this is the check that says the divergence above is NOT about
    blocking.
    """
    assert EXPERTS % cli.EXPERT_BLOCK, (
        f"{EXPERTS} experts divides into blocks of {cli.EXPERT_BLOCK} exactly, so this check no "
        f"longer exercises a short final block")

    stack = _stack(seed=700)
    basis = _refusal_basis()
    blocked, single = stack.clone(), stack.clone()
    cli.orthogonalize_np_3d_(blocked, basis, STRENGTH, sparsity=sparsity, rounds=rounds,
                             block=cli.EXPERT_BLOCK)
    cli.orthogonalize_np_3d_(single, basis, STRENGTH, sparsity=sparsity, rounds=rounds, block=1)

    worst = (blocked - single).abs().max().item()
    assert worst == 0.0, (
        f"a block of {cli.EXPERT_BLOCK} and a block of 1 differ by {worst:.3g}, so the expert "
        f"axis is not separable the way EXPERT_BLOCK assumes and the memory lever is changing "
        f"results")


# ── every expert moved ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fused", [False, True], ids=["per-expert", "fused"])
def test_every_expert_is_edited_and_none_is_silently_skipped(tmp_path, fused):
    """The silent partial ablation, asserted per expert rather than in aggregate.

    A mean over the stack passes while a third of the experts are untouched, which is the whole
    failure. So this reads each expert separately and names the ones that did not move.
    """
    stacks = [_stack(seed=300 + i) for i in range(LAYERS)]
    basis = _refusal_basis()
    build = _fused_checkpoint if fused else _per_expert_checkpoint
    root = build(tmp_path / "m", stacks)
    _ablate(root, basis)
    after = _edited_stacks(root, fused=fused)

    for layer, (before, edited) in enumerate(zip(stacks, after, strict=True)):
        untouched = [e for e in range(EXPERTS) if torch.equal(before[e], edited[e])]
        assert not untouched, (
            f"layer {layer}: experts {untouched} came back byte for byte unchanged, so the "
            f"rewrite skipped them and reported success")


@pytest.mark.parametrize("fused", [False, True], ids=["per-expert", "fused"])
def test_the_refusal_span_is_gone_from_every_expert_separately(tmp_path, fused):
    """Correctness of the edit itself, per expert.

    The bake's contract is that the residual writer no longer writes along the refusal span. On a
    mixture of experts that is a claim about each expert, and an expert left with the direction
    intact is a route by which refusal survives a run the logs call clean.
    """
    stacks = [_stack(seed=400 + i) for i in range(LAYERS)]
    basis = _refusal_basis()
    build = _fused_checkpoint if fused else _per_expert_checkpoint
    root = build(tmp_path / "m", stacks)
    _ablate(root, basis, rounds=4)
    after = _edited_stacks(root, fused=fused)

    for layer, (before, edited) in enumerate(zip(stacks, after, strict=True)):
        for e in range(EXPERTS):
            was = (basis @ before[e]).norm().item()
            now = (basis @ edited[e]).norm().item()
            assert now < was * 1e-3, (
                f"layer {layer} expert {e}: the refusal span still carries {now:.3g} of the "
                f"{was:.3g} it started with, so this expert was not ablated")


# ── the rewrite touched nothing else ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("fused", [False, True], ids=["per-expert", "fused"])
def test_the_expert_writers_are_the_only_tensors_the_rewrite_replaces(tmp_path, fused):
    """A rewrite that edits the embedding as well as the experts also passes every check above.

    Counted rather than inspected, because the number is known in advance: one fused tensor per
    layer, or one per expert per layer, and nothing else in the checkpoint is a residual writer.
    """
    stacks = [_stack(seed=500 + i) for i in range(LAYERS)]
    root = (_fused_checkpoint if fused else _per_expert_checkpoint)(tmp_path / "m", stacks)
    before = load_file(str(root / "model.safetensors"))["model.embed_tokens.weight"]

    replaced = _ablate(root, _refusal_basis())

    expected = LAYERS if fused else LAYERS * EXPERTS
    assert replaced == expected, (
        f"the rewrite replaced {replaced} tensors and this layout holds {expected} residual "
        f"writers, so it is editing either too little or too much")
    after = load_file(str(root / "model.safetensors"))["model.embed_tokens.weight"]
    assert torch.equal(before, after), "the embedding was rewritten, and it is not a residual writer"


@pytest.mark.parametrize("fused", [False, True], ids=["per-expert", "fused"])
def test_the_shard_survives_an_edit_that_fails_part_way_through_the_experts(tmp_path, fused):
    """An interrupted expert rewrite leaves the original checkpoint, not a half-edited one.

    The expert path is where this matters most: a fused stack is one tensor, so a failure during
    it is a failure during the only write that layer gets, and the per-expert layout multiplies
    the number of chances to be interrupted by the expert count.
    """
    stacks = [_stack(seed=600 + i) for i in range(LAYERS)]
    root = (_fused_checkpoint if fused else _per_expert_checkpoint)(tmp_path / "m", stacks)
    shard = root / "model.safetensors"
    before = shard.read_bytes()

    seen = []

    def edit(tensor, raw):
        seen.append(tensor.name)
        if len(seen) > 1:
            raise RuntimeError("the card went away")
        return bytes(len(raw))

    with pytest.raises(RuntimeError, match="the card went away"):
        streaming.rewrite_shard(shard, edit)

    assert shard.read_bytes() == before, "the interrupted rewrite left a half-edited checkpoint"
    assert not list(root.glob("*.rewriting")), "the interrupted rewrite left a partial file behind"
