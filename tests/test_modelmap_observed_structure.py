# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The map against the structure of real checkpoints, read from their headers and indexes.

The fixtures in tests/fixtures/structure are tensor inventories (name, shape, dtype) taken from
checkpoints on a development machine with no weight loaded. They hold what the map is held to, and they cover the
two expert layouts this project has observed:

    inv-qwen3-1.7b.json                   dense, safetensors, shapes from headers
    inv-abl-4b.json                       dense, safetensors, an edited checkpoint
    inv-Qwen3-30B-A3B-...-Q3_K_XL.json    mixture of experts, GGUF, one stacked tensor per block
    inv-balanced-clone.json               mixture of experts, safetensors index only, one tensor
                                          per expert; shapes unknown because the shards are
                                          135-byte git-LFS pointer stubs; trimmed to blocks 0, 1, 47

No hybrid checkpoint with weights is in this list yet. A hybrid claim needs a fixture of its own.
"""
import json
from pathlib import Path

import pytest

from senbonzakura import modelmap

FIXTURES = Path(__file__).parent / "fixtures" / "structure"


def _load(stem):
    return json.loads((FIXTURES / f"inv-{stem}.json").read_text(encoding="utf-8"))


def _shapes(inventory):
    return {t["name"]: t.get("shape") for t in inventory["tensors"]}


QWEN3 = _load("qwen3-1.7b")
ABL4B = _load("abl-4b")
GGUF = _load("Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL")
INDEX = _load("balanced-clone")


def _describe(inventory, config=None):
    cfg = config if config is not None else inventory.get("config", {})
    return modelmap.describe(config=cfg, tensor_shapes=_shapes(inventory))


def test_gguf_metadata_spells_the_config_keys_the_map_reads():
    cfg = modelmap.gguf_config(GGUF["metadata"])
    assert cfg["model_type"] == "qwen3moe"
    assert cfg["num_hidden_layers"] == 48
    assert cfg["num_local_experts"] == 128
    assert cfg["num_experts_per_tok"] == 8
    assert cfg["moe_intermediate_size"] == 768
    assert cfg["hidden_size"] == 2048
    assert cfg["num_attention_heads"] == 32
    assert cfg["num_key_value_heads"] == 4


def test_gguf_metadata_gives_the_same_expert_count_as_the_safetensors_config_of_the_same_model():
    gguf_cfg = modelmap.gguf_config(GGUF["metadata"])
    assert gguf_cfg["num_local_experts"] == INDEX["config"]["num_local_experts"] == 128
    assert gguf_cfg["num_experts_per_tok"] == INDEX["config"]["num_experts_per_tok"] == 8


def test_stacked_gguf_experts_are_counted_from_the_leading_shape_axis():
    d = _describe(GGUF, modelmap.gguf_config(GGUF["metadata"]))
    assert d.observed_experts == modelmap.ObservedExperts(storage="stacked", count=128)
    assert d.experts.routed.value == 128
    assert d.experts.is_mixture is True


def test_per_expert_safetensors_experts_are_counted_by_index_within_each_block():
    d = _describe(INDEX)
    assert d.observed_experts == modelmap.ObservedExperts(storage="per_expert", count=128)
    assert d.experts.routed.value == 128
    assert d.experts.is_mixture is True


def test_the_two_expert_layouts_agree_on_the_count_and_disagree_only_on_storage():
    gguf = _describe(GGUF, modelmap.gguf_config(GGUF["metadata"])).observed_experts
    index = _describe(INDEX).observed_experts
    assert gguf.count == index.count == 128
    assert gguf.storage != index.storage


@pytest.mark.parametrize("inventory", [QWEN3, ABL4B], ids=["qwen3-1.7b", "abl-4b"])
def test_dense_checkpoints_hold_no_experts_in_either_reading(inventory):
    d = _describe(inventory)
    assert d.observed_experts.storage is None
    assert d.observed_experts.why == "no expert tensor is named in this checkpoint"
    assert d.experts.is_mixture is False


def test_a_gguf_block_prefix_is_found_from_its_names_and_is_not_the_safetensors_one():
    names = [t["name"] for t in GGUF["tensors"]]
    assert modelmap.stack_prefix_from_names(names)[0] == "blk"
    assert modelmap.block_of("blk.7.ffn_gate_exps.weight", "blk") == 7


def test_block_of_reads_the_first_integer_so_an_expert_index_is_not_a_block():
    name = "model.layers.7.mlp.experts.63.up_proj.weight"
    assert modelmap.block_of(name, "model.layers") == 7


def test_gguf_metadata_without_an_architecture_is_left_empty_not_guessed():
    assert modelmap.gguf_config({"qwen3moe.block_count": 48}) == {}


def test_stacked_experts_with_no_shape_report_no_count_with_a_reason():
    obs = modelmap.observed_experts({"blk.0.ffn_gate_exps.weight": None})
    assert obs.storage == "stacked"
    assert obs.count is None
    assert "no shape" in obs.why


def test_stacked_experts_that_disagree_on_their_count_report_no_count():
    obs = modelmap.observed_experts({
        "blk.0.ffn_gate_exps.weight": [128, 768, 2048],
        "blk.1.ffn_gate_exps.weight": [64, 768, 2048]})
    assert obs.storage == "stacked"
    assert obs.count is None
    assert "disagree" in obs.why


def test_per_expert_blocks_that_hold_different_counts_report_no_count():
    obs = modelmap.observed_experts({
        "model.layers.0.mlp.experts.7.up_proj.weight": None,
        "model.layers.1.mlp.experts.3.up_proj.weight": None})
    assert obs.storage == "per_expert"
    assert obs.count is None
    assert "disagree" in obs.why


def test_a_checkpoint_naming_both_layouts_reports_neither_count():
    obs = modelmap.observed_experts({
        "model.layers.0.mlp.experts.0.up_proj.weight": None,
        "blk.0.ffn_gate_exps.weight": [4, 8, 8]})
    assert obs.storage is None
    assert obs.count is None
    assert "both" in obs.why


def test_no_inventory_means_no_observed_experts_rather_than_an_empty_answer():
    d = modelmap.describe(config=INDEX["config"])
    assert d.observed_experts is None


def _prefix(name):
    return "blk" if name.startswith("blk.") else "model.layers"


def _role_counts(inventory, block):
    counts = {}
    for t in inventory["tensors"]:
        name = t["name"]
        if modelmap.block_of(name, _prefix(name)) != block:
            continue
        role = modelmap.tensor_role(name)
        counts[role] = counts.get(role, 0) + 1
    return counts


def test_every_block_tensor_in_every_real_fixture_has_a_role():
    for inventory in (QWEN3, ABL4B, GGUF, INDEX, LFM2):
        for t in inventory["tensors"]:
            name = t["name"]
            if modelmap.block_of(name, _prefix(name)) is not None:
                assert modelmap.tensor_role(name) != "unclassified", name


@pytest.mark.parametrize("inventory", [QWEN3, ABL4B], ids=["qwen3-1.7b", "abl-4b"])
def test_dense_block_zero_roles_are_attention_norms_mlp_and_norm(inventory):
    assert _role_counts(inventory, 0) == {
        "attention": 4, "attention_norm": 2, "mlp": 3, "norm": 2}


def test_the_two_moe_layouts_give_the_same_non_expert_roles_for_block_zero():
    """The map's spine: one architecture, two storage layouts, one answer to what a tensor is."""
    gguf = _role_counts(GGUF, 0)
    index = _role_counts(INDEX, 0)
    for role in ("attention", "attention_norm", "norm", "router"):
        assert gguf[role] == index[role], role
    assert gguf["expert_mlp"] == 3
    assert index["expert_mlp"] == 384


def test_a_dense_gated_mlp_is_an_mlp_and_a_routed_gate_is_a_router():
    assert modelmap.classify("mlp.gate_proj.weight") == "mlp"
    assert modelmap.classify("mlp.gate.weight") == "router"


def test_an_expert_path_is_an_expert_even_though_it_also_names_an_mlp():
    assert modelmap.classify("mlp.experts.12.up_proj.weight") == "expert_mlp"


def test_a_name_no_rule_covers_is_unclassified_rather_than_defaulted():
    assert modelmap.classify("frobnicator.weight") == "unclassified"


def test_a_name_outside_every_block_has_no_block_role():
    assert modelmap.tensor_role("model.embed_tokens.weight") is None
    assert modelmap.tensor_role("output_norm.weight") is None


def test_collapse_expert_index_folds_the_index_and_nothing_else():
    assert (modelmap.collapse_expert_index("mlp.experts.127.down_proj.weight")
            == "mlp.experts.{e}.down_proj.weight")


LFM2 = _load("LFM2.5-1.2B-Instruct")


def _blocks_holding(inventory, needle):
    return sorted({modelmap.block_of(t["name"], "model.layers") for t in inventory["tensors"]
                   if needle in t["name"] and modelmap.block_of(t["name"], "model.layers")
                   is not None})


def test_the_hybrid_checkpoint_has_attention_in_exactly_the_blocks_its_config_names():
    """Hybrid, observed on weights: the config's layer_types and the tensor names must agree."""
    declared = [i for i, kind in enumerate(LFM2["config"]["layer_types"]) if kind == "full_attention"]
    assert declared == [2, 5, 8, 10, 12, 14]
    assert _blocks_holding(LFM2, "self_attn.") == declared


def test_the_hybrid_conv_blocks_hold_conv_and_no_attention():
    conv_blocks = [i for i, kind in enumerate(LFM2["config"]["layer_types"]) if kind == "conv"]
    assert len(conv_blocks) == 10
    assert _blocks_holding(LFM2, "conv.") == conv_blocks
    for index in conv_blocks:
        counts = _role_counts(LFM2, index)
        assert counts.get("conv") == 3, counts
        assert "attention" not in counts, counts


def test_the_hybrid_attention_blocks_hold_attention_and_no_conv():
    for index in (2, 5, 8, 10, 12, 14):
        counts = _role_counts(LFM2, index)
        assert counts.get("attention") == 4, counts
        assert counts.get("attention_norm") == 2, counts
        assert "conv" not in counts, counts


def test_the_canonical_map_reads_the_hybrid_config_into_its_block_kinds():
    described = modelmap.describe(config=LFM2["config"])
    kinds = [k.value for k in described.mechanisms.kinds.value]
    assert kinds.count("attention") == 6
    assert kinds.count("mixer") == 10
    assert described.experts.is_mixture is False
