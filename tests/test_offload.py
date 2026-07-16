"""Tests for accelerate-offload-aware weight resolution.

When a model is larger than VRAM, accelerate leaves some parameters as meta tensors on the module
and keeps the resident copy in the module's hook.weights_map. These tests prove the resolver returns
the real, editable tensor in that case (and unchanged for resident weights), and that a full
snapshot -> bake -> restore round-trip works when some layers are offloaded and others are not.
"""
import pytest
import torch
import torch.nn as nn

from senbonzakura.cli import _attn_outproj, _owned_weight, _real_tensor


class _Hook:
    """Minimal stand-in for accelerate's AlignDevicesHook: just the weights_map the resolver reads."""
    def __init__(self, weights_map):
        self.weights_map = weights_map


def _meta_like(t):
    return nn.Parameter(torch.empty(t.shape, dtype=t.dtype, device="meta"), requires_grad=False)


def _offload(module, name="weight"):
    """Turn module.<name> into a meta param with the real tensor stashed in a fake offload hook,
    mimicking what accelerate does to a CPU-offloaded layer. Returns the real tensor."""
    real = getattr(module, name).detach().clone()
    setattr(module, name, _meta_like(real))
    module._hf_hook = _Hook({name: real})
    return real


# ── _real_tensor ───────────────────────────────────────────────────────────────────────
def test_resident_weight_returned_unchanged():
    lin = nn.Linear(4, 4, bias=False)
    assert _real_tensor(lin, "weight") is lin.weight


def test_meta_weight_resolved_from_weights_map():
    lin = nn.Linear(4, 4, bias=False)
    real = _offload(lin)
    got = _real_tensor(lin, "weight")
    assert got is real
    assert not got.is_meta


def test_meta_weight_no_hook_raises():
    lin = nn.Linear(4, 4, bias=False)
    lin.weight = _meta_like(lin.weight)          # meta, but no offload hook
    with pytest.raises(ValueError, match="meta device"):
        _real_tensor(lin, "weight")


def test_meta_weight_missing_key_raises():
    lin = nn.Linear(4, 4, bias=False)
    lin.weight = _meta_like(lin.weight)
    lin._hf_hook = _Hook({})                      # hook present but no 'weight' entry
    with pytest.raises(ValueError, match="meta device"):
        _real_tensor(lin, "weight")


def test_meta_weight_still_meta_in_map_raises():
    lin = nn.Linear(4, 4, bias=False)
    m = _meta_like(lin.weight)
    lin.weight = m
    lin._hf_hook = _Hook({"weight": torch.empty(m.shape, device="meta")})  # map also meta -> unusable
    with pytest.raises(ValueError, match="meta device"):
        _real_tensor(lin, "weight")


# ── _owned_weight (submodule vs raw parameter) ─────────────────────────────────────────
def test_owned_weight_submodule_path():
    parent = nn.Module()
    parent.down_proj = nn.Linear(4, 4, bias=False)      # a submodule owning .weight
    assert _owned_weight(parent, "down_proj") is parent.down_proj.weight


def test_owned_weight_raw_param_path():
    parent = nn.Module()
    parent.down_proj = nn.Parameter(torch.randn(2, 4, 4))   # fused expert stack, a raw 3D param
    assert _owned_weight(parent, "down_proj") is parent.down_proj


def test_owned_weight_raw_param_offloaded():
    parent = nn.Module()
    parent.down_proj = nn.Parameter(torch.randn(2, 4, 4))
    real = _offload(parent, "down_proj")
    got = _owned_weight(parent, "down_proj")
    assert got is real and not got.is_meta


# ── integration: mixed resident + offloaded snapshot/bake/restore ──────────────────────
def test_offload_detection_logs_low_vram(base_args, tiny_model, tiny_tok):
    # A model carrying an hf_device_map with non-int (cpu/disk) placements is partly offloaded; the
    # Abliterator should count it and announce low-VRAM mode.
    from senbonzakura import cli
    tiny_model.hf_device_map = {"model.layers.0": 0, "model.layers.1": "cpu", "model.layers.2": "disk"}
    logs = []
    a = cli.Abliterator(base_args, logs.append, model=tiny_model, tok=tiny_tok)
    assert a.offloaded == 2
    assert any("low-VRAM" in m for m in logs)


def test_attn_outproj_resolves_offloaded_layer(abl):
    layer = abl.layers[0]
    real = _offload(layer.self_attn.o_proj)         # offload just the attention out-proj
    assert _attn_outproj(layer) is real


def test_bake_restore_roundtrip_with_offloaded_layer(abl):
    # Offload layer 0's o_proj (its mlp.down_proj stays resident): the exact mixed case a >VRAM model
    # produces. snapshot -> bake -> restore must edit the offloaded weight in place and recover it.
    layer = abl.layers[0]
    real = _offload(layer.self_attn.o_proj)
    before = real.detach().clone()

    abl.snapshot_weights()
    # Peak the ablation window ON layer 0 (oP=0, D wide enough to cover it) so the offloaded o_proj
    # actually gets a non-zero strength; a window centred elsewhere would correctly skip it.
    abl.bake_pc(0, 1.0, 0.0, 3, 0, 1.0, 0.0, 3, K=1, mode="per_layer")
    # The offloaded weight was edited in place (the bake wrote through weights_map).
    assert not torch.equal(real, before)
    abl.restore_weights()
    assert torch.allclose(real, before, atol=1e-6)
