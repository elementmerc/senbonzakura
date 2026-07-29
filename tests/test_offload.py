"""Tests for accelerate-offload-aware weight resolution.

When a model is larger than VRAM, accelerate leaves some parameters as meta tensors on the module
and keeps the resident copy in the module's hook.weights_map. These tests prove the resolver returns
the real, editable tensor in that case (and unchanged for resident weights), and that a full
snapshot -> bake -> restore round-trip works when some layers are offloaded and others are not.
"""
import pytest
import torch
from torch import nn

from senbonzakura.cli import _attn_outproj, _owned_weight, _real_tensor


class _Hook:
    """Minimal stand-in for accelerate's AlignDevicesHook: just the weights_map the resolver reads."""

    def __init__(self, weights_map):
        self.weights_map = weights_map


def _meta_like(t):
    return nn.Parameter(torch.empty(t.shape, dtype=t.dtype, device="meta"), requires_grad=False)


def _offload(module, name="weight"):
    """Turn module.<name> into a meta param with the real tensor stashed in a fake offload hook,
    mimicking what accelerate does to a CPU-offloaded layer. Returns the real tensor.
    """
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


# ── the real thing: accelerate's actual dispatch_model (tranche 4, task 23) ───────────
# Everything above uses a hand-built hook. That verifies our resolution logic and cannot
# verify the assumption the logic RESTS on, which is a claim about accelerate: that
# weights_map hands back one stable tensor whose in-place edits reach the next forward.
# Measured against accelerate 1.14.0, that claim is FALSE for disk-backed offload, and a
# bake into such a layer is discarded silently. These tests pin the real behaviour so an
# accelerate upgrade that changes it is noticed here rather than in a published number.
def _dispatched(tmp_path, disk_layers=(0,), n_layers=4):
    from accelerate import dispatch_model
    from conftest import TinyModel
    model = TinyModel(H=8, NL=n_layers, V=16)
    device_map = {f"model.layers.{i}": ("disk" if i in disk_layers else "cpu")
                  for i in range(n_layers)}
    device_map["lm_head"] = "cpu"
    return dispatch_model(model, device_map=device_map,
                          offload_dir=str(tmp_path / "offload"), main_device="cpu")


def test_disk_offload_really_does_produce_a_meta_parameter(tmp_path):
    """The premise of the whole offload path, confirmed against the real library."""
    model = _dispatched(tmp_path)
    assert model.model.layers[0].self_attn.o_proj.weight.is_meta
    assert not model.model.layers[1].self_attn.o_proj.weight.is_meta


def test_a_disk_offload_map_does_not_return_a_stable_tensor(tmp_path):
    """The assumption the bake rested on, measured. Both references are held on purpose."""
    model = _dispatched(tmp_path)
    wm = model.model.layers[0].self_attn.o_proj._hf_hook.weights_map
    first = wm["weight"]
    second = wm["weight"]
    assert first is not second, "accelerate now returns a stable tensor; the guard can relax"


def test_an_in_place_edit_on_a_disk_offload_map_is_discarded(tmp_path):
    """Why the guard has to refuse: the edit reaches nothing at all."""
    model = _dispatched(tmp_path)
    wm = model.model.layers[0].self_attn.o_proj._hf_hook.weights_map
    ids = torch.tensor([[1, 2, 3]])
    before = model(input_ids=ids).logits.clone()
    wm["weight"].zero_()
    assert torch.allclose(before, model(input_ids=ids).logits)   # forward unchanged
    assert wm["weight"].abs().sum() != 0                         # and the write vanished


def test_real_tensor_refuses_a_disk_offloaded_weight_instead_of_baking_into_a_copy(tmp_path):
    """Before this, the bake ran, reported success, and changed nothing.

    That is the worst failure shape available: an unabliterated model published as an
    abliterated one, with every number in the run consistent with success.
    """
    model = _dispatched(tmp_path)
    with pytest.raises(ValueError, match="disk-offloaded"):
        _real_tensor(model.model.layers[0].self_attn.o_proj, "weight")


def test_real_tensor_still_accepts_a_resident_weight_under_real_dispatch(tmp_path):
    """The guard must not fire on the layers that are genuinely editable."""
    model = _dispatched(tmp_path)
    resident = model.model.layers[1].self_attn.o_proj
    got = _real_tensor(resident, "weight")
    assert got is resident.weight
    assert not got.is_meta


def test_the_refusal_names_the_way_out(tmp_path):
    model = _dispatched(tmp_path)
    with pytest.raises(ValueError, match="more VRAM or host-RAM headroom"):
        _real_tensor(model.model.layers[0].self_attn.o_proj, "weight")
