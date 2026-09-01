# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Which model layouts this tool can edit, and which it must refuse rather than half-edit.

Every case here is built from a config, on the CPU, in milliseconds. No weights are downloaded
and no GPU is touched, which is the point: the wiring these tests cover is otherwise only
exercised by models too large to run on the hardware this project is built around, and that is
exactly the wiring that produced the withdrawn gemma numbers.
"""
import pytest
import torch
from torch import nn

from senbonzakura import cli

transformers = pytest.importorskip("transformers")


def _lfm2_moe(hidden=64, layer_types=("conv", "conv", "full_attention", "full_attention")):
    cfg = transformers.Lfm2MoeConfig(
        hidden_size=hidden, num_hidden_layers=len(layer_types), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, vocab_size=256, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=32, num_dense_layers=1,
        layer_types=list(layer_types))
    return transformers.Lfm2MoeForCausalLM(cfg)


def _lfm2_dense(hidden=64, layer_types=("conv", "full_attention")):
    cfg = transformers.Lfm2Config(
        hidden_size=hidden, num_hidden_layers=len(layer_types), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, vocab_size=256,
        layer_types=list(layer_types))
    return transformers.Lfm2ForCausalLM(cfg)


def _qwen3(hidden=64, layers=2):
    cfg = transformers.Qwen3Config(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, vocab_size=256)
    return transformers.Qwen3ForCausalLM(cfg)


# ── LFM2's containers, which the walker did not know ─────────────────────────────────
def test_the_walker_finds_lfm2s_feed_forward_block():
    """`layer.feed_forward`, where every other family this tool supports says `mlp`."""
    for layer in _lfm2_moe().model.layers:
        assert cli.layer_downproj(layer), "no down-projection found on an LFM2 layer"


def test_an_lfm2_dense_layer_uses_w2_as_its_down_projection():
    """Mixtral's name in a dense position. The walker knew `w2` only as a fused expert stack."""
    kinds = [k for k, _ in cli.layer_downproj(_lfm2_dense().model.layers[0])]
    assert kinds == ["dense"]


def test_the_fused_expert_stack_is_read_as_one():
    """`Lfm2MoeExperts` exposes `down_proj` and is NOT iterable, so the unfused fallback must
    not be reached for it: `list(experts)` raises.
    """
    moe_layer = _lfm2_moe().model.layers[1]
    kinds = [k for k, _ in cli.layer_downproj(moe_layer)]
    assert kinds == ["fused3d"]


def test_the_fused_expert_tensor_is_the_layout_the_bake_assumes():
    """`[E, out, in]`. Read from the class source rather than inferred: the parameter is declared
    `empty(num_experts, hidden_dim, intermediate_dim)`. Getting this transposed would edit the
    wrong axis and produce a model whose refusal behaviour had not moved at all.
    """
    m = _lfm2_moe(hidden=64)
    (kind, w), = cli.layer_downproj(m.model.layers[1])
    assert kind == "fused3d"
    assert w.dim() == 3
    assert w.shape[0] == 4, "first axis should be the expert count"
    assert w.shape[1] == 64, "second axis should be the hidden size, which is what it writes into"


# ── the convolution blocks, which are half the residual writers on this family ───────
#
# Until 2026-08-15 a hybrid LFM2 was REFUSED here rather than edited, because the tool could not
# reach the convolution path and a partial edit reported as a whole one is the withdrawn-gemma
# failure in a new disguise. Task 36 resolved that the other way: `conv.out_proj` is a
# `[hidden, hidden]` Linear in the same position an attention `o_proj` occupies, so the same
# row-wise norm-preserving bake applies to it unchanged and the family is now editable. The
# refusal is kept, one flag away, as the control arm of the experiment that asks whether refusal
# travels through the convolution path at all.
def test_a_convolution_output_projection_is_found_as_a_residual_writer():
    m = _lfm2_dense(layer_types=("conv", "full_attention"))
    conv_layer, attn_layer = m.model.layers[0], m.model.layers[1]
    assert cli._conv_outproj(conv_layer) is not None
    assert cli._conv_outproj(attn_layer) is None, "an attention layer has no conv block"


def test_every_layer_of_a_hybrid_yields_exactly_one_attention_side_writer():
    """One or the other per layer, never neither: a layer with nothing here would be a layer the
    bake silently skips.
    """
    m = _lfm2_moe(layer_types=("conv", "conv", "full_attention", "full_attention"))
    for i, layer in enumerate(m.model.layers):
        writers = cli.layer_attn_writers(layer)
        assert len(writers) == 1, f"layer {i}"
        assert writers[0].shape == (64, 64)


def test_the_conv_writer_is_the_convolutions_own_out_proj():
    m = _lfm2_dense(layer_types=("conv",))
    layer = m.model.layers[0]
    assert cli.layer_attn_writers(layer)[0] is cli._real_tensor(layer.conv.out_proj, "weight")


def test_an_ordinary_model_is_unchanged_by_the_hybrid_path():
    """The new walker must return exactly what the old single-tensor one did on every
    architecture that has attention in every layer.
    """
    for layer in _qwen3().model.layers:
        assert cli.layer_attn_writers(layer) == [cli._attn_outproj(layer)]


def test_a_hybrid_is_no_longer_refused_now_that_the_convolutions_are_edited():
    cli.refuse_unrecognised_writers(_lfm2_moe().model.layers, 64)


def test_skipping_the_convolutions_still_refuses_unless_it_is_asked_for_in_writing():
    """The control arm has to be deliberate. A run that merely forgets the conv blocks is the
    failure this guard exists for and still gets a refusal naming the layers.
    """
    with pytest.raises(ValueError) as e:
        cli.refuse_unrecognised_writers(_lfm2_moe().model.layers, 64, ablate_conv=False)
    message = str(e.value)
    assert "2 of 4 decoder layers" in message
    assert "layer 0" in message and "Lfm2MoeShortConv" in message


def test_the_control_arm_is_allowed_but_never_quiet():
    """It warns per run, names the layers, and hands the caller the list to record."""
    said = []
    missed = cli.refuse_unrecognised_writers(
        _lfm2_moe().model.layers, 64, ablate_conv=False, accept_partial=True, log=said.append)
    assert sorted(missed) == [0, 1]
    joined = " ".join(said)
    assert "PARTIAL ABLATION" in joined and "layer 0" in joined
    assert "control arm, not a result" in joined


def test_the_control_arm_leaves_the_convolutions_out_of_the_bake():
    m = _lfm2_dense(layer_types=("conv", "full_attention"))
    conv_layer = m.model.layers[0]
    assert cli.layer_attn_writers(conv_layer, ablate_conv=False) == []


def test_a_layer_with_neither_attention_nor_a_convolution_is_refused():
    """The genuinely unsupported case, which must stay distinguishable from the deliberate skip:
    one returns an empty list, the other raises.
    """
    class _Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Linear(8, 8)

    with pytest.raises(ValueError) as e:
        cli.layer_attn_writers(_Bare())
    assert "architecture not supported" in str(e.value)


def test_a_convolution_block_without_a_two_dimensional_out_proj_is_not_claimed():
    """A `conv` container that does not hold an editable `[out, in]` matrix is not something to
    orthogonalise, and quietly treating it as one would edit the wrong tensor.
    """
    class _Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = torch.nn.Conv1d(4, 4, 3)      # a 3-D weight, not a Linear

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = _Conv()

    assert cli._conv_outproj(_Layer()) is None


def test_an_attention_layer_is_edited_either_way():
    """Only the convolution path is under test in the comparison; everything else must be held
    still, or the two arms differ by more than the thing being measured.
    """
    m = _lfm2_dense(layer_types=("conv", "full_attention"))
    attn_layer = m.model.layers[1]
    assert (cli.layer_attn_writers(attn_layer, ablate_conv=False)
            == cli.layer_attn_writers(attn_layer, ablate_conv=True))


def test_an_all_attention_lfm2_is_not_refused():
    """The guard must object to the convolution blocks specifically, not to the family."""
    m = _lfm2_moe(layer_types=("full_attention", "full_attention"))
    cli.refuse_unrecognised_writers(m.model.layers, 64)


def test_a_supported_dense_model_passes_untouched():
    cli.refuse_unrecognised_writers(_qwen3().model.layers, 64)


def test_a_norm_is_not_mistaken_for_a_residual_writer():
    """Norms sit beside the sublayers and hold a 1-D weight. Treating one as a writer would
    refuse every model this tool already supports.
    """
    _, unrecognised = cli.residual_writers(_qwen3().model.layers[0], 64)
    assert unrecognised == []


def test_a_bolted_on_residual_writer_is_caught_on_any_family():
    """The guard is dimensional rather than a list of known bad names, so an adapter or a side
    branch added to a family we do support is caught too.
    """
    layer = _qwen3().model.layers[0]
    layer.add_module("side_branch", torch.nn.Linear(32, 64))
    _, unrecognised = cli.residual_writers(layer, 64)
    assert any("side_branch" in u for u in unrecognised)


def test_a_module_that_cannot_write_the_residual_is_ignored():
    """Output width must match the hidden size; a projection into some other space is not a
    residual write and refusing on it would be a false alarm on half the model zoo.
    """
    layer = _qwen3().model.layers[0]
    layer.add_module("router", torch.nn.Linear(64, 8))
    _, unrecognised = cli.residual_writers(layer, 64)
    assert unrecognised == []


# ── the unrecognised-writer guard must see a FUSED stack, not only a 2-D matrix ──────
def test_a_hidden_fused_expert_stack_is_flagged_as_an_unrecognised_writer():
    """THE REGRESSION GUARD, and the gemma failure's newest disguise.

    `residual_writers` used to flag an unknown container only when it held a 2-D weight of the
    hidden width. A fused mixture-of-experts stack is 3-D, so the test could not see one at all,
    and the guard's whole coverage rested on the container NAME being in the known set. An
    architecture putting an equivalent stack under an unfamiliar name would have passed the guard
    and gone unablated, reporting success.
    """
    H, E, I = 64, 4, 32
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = nn.Linear(H, H)
    # A container the walker does not know, holding a fused expert stack [E, hidden, inter].
    layer.secret_experts = nn.Module()
    layer.secret_experts.down_proj = nn.Parameter(torch.randn(E, H, I))

    _, unrecognised = cli.residual_writers(layer, H)
    assert any("secret_experts" in u for u in unrecognised), (
        "a fused expert stack under an unknown name went unseen, which is exactly how a model "
        "comes out partly ablated while the run reports success")


def test_a_depthwise_convolution_kernel_is_not_read_as_an_expert_stack():
    """A Conv1d kernel is 3-D too, as [channels, in/groups, width]. Requiring the hidden size in
    the MIDDLE keeps a depthwise convolution over the residual width from tripping the guard.
    """
    H = 64
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = nn.Linear(H, H)
    layer.some_filter = nn.Module()
    layer.some_filter.weight = nn.Parameter(torch.randn(H, 1, 3))   # depthwise, hidden leading

    _, unrecognised = cli.residual_writers(layer, H)
    assert unrecognised == []


def test_a_two_dimensional_writer_is_still_caught():
    """The original behaviour, unchanged by widening the test to 3-D."""
    H = 64
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = nn.Linear(H, H)
    layer.mystery = nn.Module()
    layer.mystery.proj = nn.Linear(32, H)

    _, unrecognised = cli.residual_writers(layer, H)
    assert any("mystery" in u for u in unrecognised)


def test_the_real_lfm2_moe_has_no_unrecognised_writers():
    """The whole point of the widening is that it must not start refusing a model we support."""
    model = _lfm2_moe()
    for layer in model.model.layers:
        _, unrecognised = cli.residual_writers(layer, model.config.hidden_size)
        assert unrecognised == []
