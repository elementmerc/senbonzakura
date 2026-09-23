# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Which tensor names carry a residual write. One list, for everything that needs to know.

WHY THIS IS A MODULE RATHER THAN A FEW CONSTANTS IN `cli.py`, WHERE IT USED TO LIVE.

Three separate parts of this tool have to agree about which tensors get edited:

  * the editor, which walks live modules and edits them;
  * the snapshot estimator, which sizes the reversible search before a download;
  * the streaming path, which reads a checkpoint off disk and never builds a model at all.

The names were already shared between the first two, deliberately, and the comment that sat
beside them said why: an earlier version kept two hand-written copies of the architecture names
and they drifted, so the guard refused a Qwen3.5 the editor was perfectly able to edit. That is
the defect this file exists to keep closed.

The streaming path is what forced the move. `cli.py` imports torch, optuna and transformers at
module scope, and `streaming.py` deliberately imports none of them: it reads a file format, so it
runs on any machine and its tests need no GPU and no model. Importing the names from `cli.py`
would have dragged the whole abliteration stack into a module whose whole point is that it does
not need one, and copying them into `streaming.py` would have recreated the exact two-copies
failure the comment warns about. So the names move somewhere both can reach, and nothing here
imports anything heavier than the standard library.

`cli.py` re-exports every name below, so the editor's own references are unchanged.

WHAT THIS CAN AND CANNOT DECIDE. A name match is an ESTIMATE and says so at every call site. It
cannot apply the structural conditions the editor applies with the module in hand, such as
refusing an `out_proj` on a `mixer` that is really a Mamba-2 block. It is used to size a run and
to walk a checkpoint, never to decide a close case; the real check still runs with the weights
resident, in `refuse_unrecognised_writers`.
"""

#: Child names that can hold an attention block. `mixer` is NemotronH, whose decoder layer holds
#: exactly ONE child called `mixer` that is an attention block, a Mamba-2 mixer, an MLP or a MoE
#: depending on the layer. It appears here AND in MIXER_BLOCKS AND in MLP_BLOCKS for that reason:
#: the name says where it sits, not what it is, so every position has to look at it and the
#: contents decide.
ATTN_BLOCKS = ("self_attn", "attention", "self_attention", "attn", "mixer")

#: Blocks that can sit in the attention position and write the residual stream through their own
#: `out_proj`. The field has moved away from uniform attention stacks and this is the list that
#: moves with it: LFM2 puts a short convolution there, Qwen3.5 puts a gated delta net. In every
#: case the writer is a 2-D Linear whose OUTPUT dimension is hidden size, which is dimensionally
#: the same object as an attention `o_proj`, so the row-wise norm-preserving bake applies to it
#: unchanged and no per-vendor edit path is needed.
#:
#: Every one of these was verified by building the architecture on the meta device and reading
#: the shapes, never by reasoning about the mechanism:
#:
#:     LFM2.5-8B-A1B      conv.out_proj          [2048, 2048]
#:     Qwen3.6-35B-A3B    linear_attn.out_proj   [2048, 4096]
#:     Bamba              mamba.out_proj         [hidden, 2 x hidden]
#:     FalconH1           mamba.out_proj         [hidden, 16 x hidden]
#:     Jamba              mamba.out_proj         [hidden, 2 x hidden]
#:     GraniteMoeHybrid   mamba.out_proj         [hidden, 2 x hidden]
#:
#: The inner width differs and does not matter: the projection contracts over the output axis.
MIXER_BLOCKS = ("conv", "linear_attn", "mamba", "mixer")

#: Blocks that hold the MLP-position residual writer. One list, read by the editor AND by the
#: guard, because two hand-kept copies of a set of names is exactly how the guard came to refuse
#: Qwen3.5 while the editor was perfectly able to edit it.
#:
#: `shared_mlp` is GraniteMoeHybrid's always-on expert, whose writer is `output_linear`. It sits
#: beside a routed `block_sparse_moe` and a `mamba` block in the same layer, so a layer there has
#: three residual writers rather than two.
MLP_BLOCKS = ("block_sparse_moe", "mlp", "feed_forward", "shared_mlp", "mixer")

#: Terminal projection names that carry a residual write. Every one is read from where the
#: editor reads it: `layer_attn_writers` tries `o_proj`, `out_proj` and `dense`;
#: `_block_outproj_param` takes `out_proj` on a mixer; `_mlp_downprojs` takes `down_proj`, `w2`
#: and `output_linear`, fused or per expert.
WRITER_PROJECTIONS = ("o_proj", "out_proj", "dense", "down_proj", "w2", "output_linear")


def is_writer_tensor(name, ablate_conv=True):
    """Does this tensor NAME look like a residual writer?

    NAMES, NOT ARCHITECTURE. The tempting way to size a run before a download is to compute it
    from `config.json`, and it gives the right answer: by hand it reproduces 20.13 GB for
    Qwen3-30B-A3B. It would also be a second, independent account of which tensors get edited,
    and this project has already shipped that failure once. Matching names against the SAME
    constants the editor walks keeps one list.
    """
    parts = name.split(".")
    if parts[-1] != "weight":
        return False
    blocks = set(ATTN_BLOCKS) | set(MLP_BLOCKS) | set(MIXER_BLOCKS)
    if not ablate_conv:
        # `--skip-conv-ablation` is the control arm: `layer_attn_writers` leaves the convolution
        # alone, so the snapshot is smaller and an estimate that counted it would refuse a run
        # that fits. Only blocks that are mixer-ONLY are dropped: `mixer` itself appears in all
        # three lists, because on NemotronH one child name means four things, and a name alone
        # cannot say which. Counting those is the safe direction, since over-counting a shared
        # name risks a spurious refusal only on architectures that use it.
        blocks -= set(MIXER_BLOCKS) - set(ATTN_BLOCKS) - set(MLP_BLOCKS)
    return any(p in blocks for p in parts) and any(p in WRITER_PROJECTIONS for p in parts)
