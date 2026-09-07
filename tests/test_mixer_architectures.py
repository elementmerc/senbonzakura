# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Hybrids are the common shape now, and the writer recogniser has to move with them.

WHY THIS FILE EXISTS

A peer probed ten current models by building each architecture on the meta device and walking it
with our writer recogniser. Nine passed. One did not: Qwen3.6-35B-A3B, Apache 2.0, where 30 of 40
layers carry no attention at all and write the residual stream through a gated delta net instead.
Refusing a mainstream permissively licensed model is a real limit on the domain claim, and the
trend across vendors is away from uniform attention stacks, so it will keep happening.

THE DESIGN QUESTION, WHICH THE PROBE COULD NOT ANSWER

The probe says what is unrecognised, not what the correct edit is. Whether a gated delta net's
output projection is dimensionally the same object as an attention o_proj is the question, and it
was settled by reading the shapes rather than reasoning about the mechanism:

    LFM2.5-8B-A1B      conv.out_proj         [2048, 2048]
    Qwen3.6-35B-A3B    linear_attn.out_proj  [2048, 4096]

Both are 2-D Linears whose OUTPUT dimension is hidden size, which is what an o_proj is. The inner
width differs and does not matter, because the projection contracts over the output axis. So one
recogniser covers both and no per-vendor edit path is needed.

WHY THIS IS SAFE RATHER THAN RECKLESS

Adding an architecture without proving the edit reaches the residual stream is how the gemma
failure happened: a bake that never landed, reported as a successful run. `validate --experiment
reach` is what makes it defensible, and on LFM2 it has already shown the mixer path landing at
mean 0.495 against attention's 0.759, with one layer at 0.076. Recognising a writer is a
precondition for editing it, not evidence that the edit works.
"""
import torch

from senbonzakura import cli


class _Block(torch.nn.Module):
    """A sequence mixer with an output projection, as the real ones present themselves.

    A real Module rather than a bare object, because the guard walks `named_children` and
    `named_parameters`: a stand-in that only satisfies the editor would let the two drift again
    without a test noticing, which is the exact defect this file guards.
    """

    def __init__(self, out_features=8, in_features=8, dim=2):
        super().__init__()
        shape = (out_features, in_features) if dim == 2 else (2, out_features, in_features)
        self.out_proj = torch.nn.Linear(1, 1)
        self.out_proj.weight = torch.nn.Parameter(torch.randn(*shape))


class _Layer(torch.nn.Module):
    """A decoder layer carrying whichever mixer the architecture puts in the attention position."""

    def __init__(self, **blocks):
        super().__init__()
        for name, block in blocks.items():
            setattr(self, name, block)


def test_a_convolution_mixer_is_recognised():
    """The first case that needed this, kept so generalising it did not lose it."""
    w = cli._conv_outproj(_Layer(conv=_Block()))
    assert w is not None and w.dim() == 2


def test_a_gated_delta_net_is_recognised():
    """THE MODEL THIS WAS ADDED FOR. 30 of Qwen3.6-35B-A3B's 40 layers write through one."""
    w = cli._conv_outproj(_Layer(linear_attn=_Block(out_features=2048, in_features=4096)))
    assert w is not None
    assert tuple(w.shape) == (2048, 4096), "a non-square mixer writer must still be accepted"


def test_a_layer_with_no_mixer_at_all_yields_nothing():
    assert cli._conv_outproj(_Layer()) is None


def test_a_mixer_whose_writer_is_not_two_dimensional_is_refused():
    """Returning it would produce a confident wrong edit instead of the loud refusal that
    `refuse_unrecognised_writers` exists to give. A rank this bake cannot handle correctly is not
    a writer it should claim.
    """
    assert cli._conv_outproj(_Layer(conv=_Block(dim=3))) is None


def test_every_recognised_block_name_is_declared_in_one_place():
    """A per-vendor branch is how this becomes unmaintainable. The list is the extension point."""
    assert "conv" in cli.MIXER_BLOCKS
    assert "linear_attn" in cli.MIXER_BLOCKS


def test_a_layer_carrying_two_mixers_takes_the_first_declared():
    """Not a shape any current model has, and worth pinning anyway: the order is the declaration
    order rather than dictionary iteration, so which one is edited is a choice and not an
    accident.
    """
    layer = _Layer(conv=_Block(out_features=4), linear_attn=_Block(out_features=16))
    assert cli._conv_outproj(layer).shape[0] == 4


def test_the_refusal_names_what_is_supported_so_the_next_gap_is_actionable():
    """A message that says "not supported" without saying what IS leaves the reader to guess which
    part of their model is the problem.
    """
    import inspect
    src = inspect.getsource(cli.layer_attn_writers)
    assert "gated delta net" in src
    assert "short convolution" in src


def test_skipping_the_mixer_still_leaves_the_control_arm_reachable():
    """`--skip-conv-ablation` is the arm that answers whether refusal travels through the mixer
    path at all. On these models it is most of the stack, so a run with it set is a partial
    abliteration by construction and the code must return empty rather than raise.
    """
    layer = _Layer(linear_attn=_Block())
    assert cli.layer_attn_writers(layer, ablate_conv=False) == []
    assert len(cli.layer_attn_writers(layer, ablate_conv=True)) == 1


# ── the guard and the editor must not drift ──────────────────────────────────────────
# They were two lists of names that had to agree, with nothing enforcing it, and they drifted the
# first time the editor learned a new block: MIXER_BLOCKS gained `linear_attn`, the guard's
# `known` did not, and Qwen3.5 was refused by the guard while the editor could edit it perfectly.
#
# That failure was the SAFE direction: a loud refusal rather than a silent partial edit. The
# opposite drift, a guard that accepts a block the editor skips, is the gemma failure exactly:
# a run completes, reports success, and a third of the layers were never touched.

def test_the_guard_recognises_every_block_the_editor_can_edit():
    """THE REGRESSION THIS SECTION IS NAMED FOR.

    Asserted as a property rather than by listing the names twice, because listing them twice is
    the defect.
    """
    layer = _Layer(**{name: _Block() for name in cli.MIXER_BLOCKS})
    _known, unrecognised = cli.residual_writers(layer, hidden_size=8, ablate_conv=True)
    assert unrecognised == [], (
        f"the editor can edit {cli.MIXER_BLOCKS} and the guard refuses {unrecognised}; the two "
        f"name lists have drifted apart again")


def test_each_mixer_block_is_recognised_by_the_guard_individually():
    """One at a time, so a block that only passes when another is present cannot hide."""
    for name in cli.MIXER_BLOCKS:
        _known, unrecognised = cli.residual_writers(
            _Layer(**{name: _Block()}), hidden_size=8, ablate_conv=True)
        assert unrecognised == [], f"the guard does not recognise {name}"


def test_the_control_arm_still_reports_a_skipped_mixer_as_unrecognised():
    """With --skip-conv-ablation the mixer is deliberately NOT edited, so the guard must still see
    it as an unedited residual writer. Silently accepting it there would turn the control arm into
    an unlabelled partial abliteration.
    """
    for name in cli.MIXER_BLOCKS:
        _known, unrecognised = cli.residual_writers(
            _Layer(**{name: _Block()}), hidden_size=8, ablate_conv=False)
        assert unrecognised, f"{name} was silently accepted while being skipped"


def test_the_guard_derives_its_names_rather_than_restating_them():
    """The structural fix, pinned. Two lists that must agree is the defect; one list is the fix."""
    import inspect
    src = inspect.getsource(cli.residual_writers)
    assert "MIXER_BLOCKS" in src, "the guard restates the block names instead of deriving them"
    assert '"linear_attn"' not in src, "a block name is hard-coded in the guard again"
