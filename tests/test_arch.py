"""Tests for the architecture resolvers: layer_downproj, _decoder_layers, _attn_outproj."""
import types

import pytest
import torch
import torch.nn as nn

from senbonzakura import cli


def _lin(o, i):
    return nn.Linear(i, o, bias=False)


# ── layer_downproj ──────────────────────────────────────────────────────────────────
def test_dense_layer():
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=_lin(8, 8)))
    entries = cli.layer_downproj(layer)
    assert len(entries) == 1
    assert entries[0][0] == "dense"
    assert entries[0][1].shape == (8, 8)


def test_qwen3_moe_fused():
    experts = types.SimpleNamespace(down_proj=torch.nn.Parameter(torch.randn(4, 8, 16)))
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(experts=experts))
    entries = cli.layer_downproj(layer)
    assert entries[0][0] == "fused3d"
    assert entries[0][1].shape == (4, 8, 16)


def test_mixtral_unfused_list():
    experts = [types.SimpleNamespace(w2=_lin(8, 16)) for _ in range(3)]
    layer = types.SimpleNamespace(block_sparse_moe=types.SimpleNamespace(experts=experts))
    entries = cli.layer_downproj(layer)
    assert entries[0][0] == "list"
    assert len(entries[0][1]) == 3


def test_olmoe_unfused_down_proj_list():
    experts = [types.SimpleNamespace(down_proj=_lin(8, 16)) for _ in range(2)]
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(experts=experts))
    entries = cli.layer_downproj(layer)
    assert entries[0][0] == "list"
    assert len(entries[0][1]) == 2


def test_granite_output_linear():
    mlp = types.SimpleNamespace(output_linear=torch.nn.Parameter(torch.randn(4, 8, 16)))
    layer = types.SimpleNamespace(block_sparse_moe=mlp)
    entries = cli.layer_downproj(layer)
    assert entries[0][0] == "fused3d"


def test_shared_expert_is_also_ablated():
    # Routed fused experts PLUS an always-on shared expert: both write the residual.
    experts = types.SimpleNamespace(down_proj=torch.nn.Parameter(torch.randn(4, 8, 16)))
    shared = types.SimpleNamespace(down_proj=_lin(8, 16))
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(experts=experts, shared_expert=shared))
    entries = cli.layer_downproj(layer)
    kinds = [k for k, _ in entries]
    assert "fused3d" in kinds and "dense" in kinds
    assert len(entries) == 2


def test_unknown_arch_raises_loud():
    layer = types.SimpleNamespace(something_else=1)
    with pytest.raises(ValueError, match="not supported"):
        cli.layer_downproj(layer)


# ── _decoder_layers ──────────────────────────────────────────────────────────────────
def test_decoder_layers_llama_tree():
    layers = [1, 2, 3]
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
    assert cli._decoder_layers(model) is layers


def test_decoder_layers_gptneox_tree():
    layers = [1, 2]
    model = types.SimpleNamespace(gpt_neox=types.SimpleNamespace(layers=layers))
    assert cli._decoder_layers(model) is layers


def test_decoder_layers_unknown_raises():
    with pytest.raises(ValueError, match="decoder layer stack"):
        cli._decoder_layers(types.SimpleNamespace(nope=1))


# ── _attn_outproj ────────────────────────────────────────────────────────────────────
def test_attn_outproj_o_proj():
    layer = types.SimpleNamespace(self_attn=types.SimpleNamespace(o_proj=_lin(8, 8)))
    assert cli._attn_outproj(layer).shape == (8, 8)


def test_attn_outproj_falcon_dense():
    layer = types.SimpleNamespace(self_attention=types.SimpleNamespace(dense=_lin(8, 8)))
    assert cli._attn_outproj(layer).shape == (8, 8)


def test_attn_outproj_out_proj():
    layer = types.SimpleNamespace(attn=types.SimpleNamespace(out_proj=_lin(8, 8)))
    assert cli._attn_outproj(layer).shape == (8, 8)


def test_attn_outproj_missing_raises():
    layer = types.SimpleNamespace(self_attn=types.SimpleNamespace(nope=1))
    with pytest.raises(ValueError, match="attention output projection"):
        cli._attn_outproj(layer)
