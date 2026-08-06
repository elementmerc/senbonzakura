"""Which model layouts this tool can edit, and which it must refuse rather than half-edit.

Every case here is built from a config, on the CPU, in milliseconds. No weights are downloaded
and no GPU is touched, which is the point: the wiring these tests cover is otherwise only
exercised by models too large to run on the hardware this project is built around, and that is
exactly the wiring that produced the withdrawn gemma numbers.
"""
import pytest
import torch

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


# ── the guard, which is why the above is allowed to exist ────────────────────────────
def test_a_layer_writing_through_a_convolution_is_refused_not_half_edited():
    """Half of LFM2's layers carry no attention at all: a short convolution writes the residual
    stream through its own output projection. Ablating only what we recognise would leave that
    path live while the run reported success, which is the withdrawn-gemma failure in a new
    disguise.
    """
    with pytest.raises(ValueError) as e:
        cli.refuse_unrecognised_writers(_lfm2_moe().model.layers, 64)
    assert "conv" in str(e.value)


def test_the_refusal_names_the_layers_and_counts_them():
    """A refusal that says only "unsupported" leaves the reader nothing to act on."""
    with pytest.raises(ValueError) as e:
        cli.refuse_unrecognised_writers(_lfm2_moe().model.layers, 64)
    message = str(e.value)
    assert "2 of 4 decoder layers" in message
    assert "layer 0" in message and "Lfm2MoeShortConv" in message


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
