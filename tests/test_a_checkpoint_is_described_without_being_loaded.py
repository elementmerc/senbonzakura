# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The structure probe's pure helpers, on hand-built checkpoints rather than real ones.

WHY THESE FIXTURES AND NOT REAL MODELS. Every assertion here is about a NAME or a SHAPE, so a
checkpoint built in a temporary directory out of three tensors tests the same code paths a 35
billion parameter mixture of experts does, in milliseconds and with no cache. The real checkpoints
are measured by the probe itself and the measurements live in
`private/research/probes/2026-10-08-structure/`; this file tests the reading, not the models.

WHY THE WHOLE FILE SKIPS WHEN THE PROBE IS ABSENT. `private/` is excluded from git (see
`.git/info/exclude`), so a fresh clone and every CI job has the tests and not the probe. A skip
with a reason is the honest outcome there. It is stated loudly rather than left as a silent pass,
because a test that cannot run is not a test that passed.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

PROBE_PATH = (Path(__file__).resolve().parents[1]
              / "private" / "research" / "probes" / "structure_of_a_checkpoint.py")

if not PROBE_PATH.is_file():
    pytest.skip(
        f"{PROBE_PATH} is not present. private/ is excluded from git, so this file can only run "
        f"on a working copy that has the research tree. Nothing was verified.",
        allow_module_level=True)


def _load():
    """The probe imported by path, because `private/research/probes` is not a package."""
    spec = importlib.util.spec_from_file_location("_structure_probe_under_test", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe_mod = _load()
ProbeError = probe_mod.ProbeError


# ── building a checkpoint by hand ─────────────────────────────────────────────────

def write_safetensors(path, tensors, *, metadata=None):
    """A real safetensors file: an unsigned 64-bit little-endian header length, then JSON, then data.

    `tensors` maps a name to `(dtype_string, shape)`. The payload is zeros of the right length,
    because nothing under test ever reads a byte past the header.
    """
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        width = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1, "I64": 8}[dtype]
        size = width
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path


def llama_block(index, width=8):
    """One pre-norm dense decoder block's tensors, in the Llama naming every tool is written for."""
    return {
        f"model.layers.{index}.input_layernorm.weight": ("F32", [width]),
        f"model.layers.{index}.self_attn.q_proj.weight": ("F32", [width, width]),
        f"model.layers.{index}.self_attn.o_proj.weight": ("F32", [width, width]),
        f"model.layers.{index}.post_attention_layernorm.weight": ("F32", [width]),
        f"model.layers.{index}.mlp.gate_proj.weight": ("F32", [2 * width, width]),
        f"model.layers.{index}.mlp.down_proj.weight": ("F32", [width, 2 * width]),
    }


@pytest.fixture
def dense_snapshot(tmp_path):
    """A two-block dense checkpoint with a config and a single shard, on disk."""
    snap = tmp_path / "dense"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
        "num_key_value_heads": 2, "head_dim": 4, "torch_dtype": "float32"}))
    tensors = {"model.embed_tokens.weight": ("F32", [16, 8]), "model.norm.weight": ("F32", [8])}
    tensors.update(llama_block(0))
    tensors.update(llama_block(1))
    write_safetensors(snap / "model.safetensors", tensors)
    return snap


# ── read_safetensors_header ───────────────────────────────────────────────────────

def test_a_header_gives_every_name_and_shape_without_a_single_weight_read(tmp_path):
    path = write_safetensors(tmp_path / "m.safetensors",
                             {"a.weight": ("BF16", [3, 4]), "b.bias": ("F32", [])},
                             metadata={"format": "pt"})
    got = probe_mod.read_safetensors_header(path)
    assert got == {"a.weight": {"dtype": "BF16", "shape": [3, 4]},
                   "b.bias": {"dtype": "F32", "shape": []}}, "metadata must not become a tensor"


def test_a_file_too_short_to_hold_a_length_prefix_is_refused_by_name(tmp_path):
    path = tmp_path / "stub.safetensors"
    path.write_bytes(b"\0\0\0")
    with pytest.raises(ProbeError, match="too short to hold a safetensors length prefix"):
        probe_mod.read_safetensors_header(path)


def test_a_zero_length_header_is_refused_rather_than_read_as_an_empty_model(tmp_path):
    path = tmp_path / "empty.safetensors"
    path.write_bytes(struct.pack("<Q", 0) + b"")
    with pytest.raises(ProbeError, match="zero-length header"):
        probe_mod.read_safetensors_header(path)


def test_a_header_longer_than_the_cap_is_refused_before_any_allocation(tmp_path):
    path = write_safetensors(tmp_path / "m.safetensors", {"a.weight": ("F32", [2])})
    with pytest.raises(ProbeError, match="against a cap of 16"):
        probe_mod.read_safetensors_header(path, max_bytes=16)


def test_a_truncated_file_is_named_as_truncated_and_not_parsed(tmp_path):
    path = tmp_path / "cut.safetensors"
    path.write_bytes(struct.pack("<Q", 4096) + b'{"a": 1}')
    with pytest.raises(ProbeError, match="truncated or is not a safetensors file"):
        probe_mod.read_safetensors_header(path)


def test_a_header_that_is_not_json_says_so_with_the_path(tmp_path):
    body = b"not json at all"
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(ProbeError, match="is not valid UTF-8 JSON"):
        probe_mod.read_safetensors_header(path)


def test_a_header_that_is_not_an_object_is_refused(tmp_path):
    body = b"[1, 2, 3]"
    path = tmp_path / "list.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(ProbeError, match="not an object"):
        probe_mod.read_safetensors_header(path)


def test_an_entry_with_no_shape_is_a_format_this_probe_refuses_to_guess_at(tmp_path):
    body = json.dumps({"a.weight": {"dtype": "F32"}}).encode()
    path = tmp_path / "noshape.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(ProbeError, match="has no dtype or shape"):
        probe_mod.read_safetensors_header(path)


# ── resolve_snapshot ──────────────────────────────────────────────────────────────

def test_a_directory_holding_a_config_resolves_to_itself(dense_snapshot):
    got, how = probe_mod.resolve_snapshot(str(dense_snapshot))
    assert (got, how) == (dense_snapshot, "path")


def test_a_directory_with_no_config_is_refused_rather_than_described_as_empty(tmp_path):
    (tmp_path / "bare").mkdir()
    with pytest.raises(ProbeError, match=r"holds no config\.json"):
        probe_mod.resolve_snapshot(str(tmp_path / "bare"))


def test_a_bare_repo_id_resolves_because_a_few_real_models_predate_owners(tmp_path):
    snap = tmp_path / "models--gpt2" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    got, how = probe_mod.resolve_snapshot("gpt2", hub_root=tmp_path)
    assert got == snap
    assert "1 of 1 snapshots" in how


def test_a_spec_that_is_neither_a_path_nor_a_repo_id_says_both_shapes(tmp_path):
    with pytest.raises(ProbeError, match="nor a hub repo id"):
        probe_mod.resolve_snapshot("a/b/c", hub_root=tmp_path)


def test_a_file_where_a_snapshot_was_expected_is_refused(tmp_path):
    (tmp_path / "notadir").write_text("x")
    with pytest.raises(ProbeError, match="is not a directory"):
        probe_mod.resolve_snapshot(str(tmp_path / "notadir"))


def test_a_repo_absent_from_the_cache_says_the_probe_downloads_nothing(tmp_path):
    with pytest.raises(ProbeError, match="downloads nothing"):
        probe_mod.resolve_snapshot("owner/absent", hub_root=tmp_path)


def test_a_repo_with_no_snapshots_directory_says_it_was_never_downloaded(tmp_path):
    (tmp_path / "models--owner--name").mkdir(parents=True)
    with pytest.raises(ProbeError, match="never downloaded"):
        probe_mod.resolve_snapshot("owner/name", hub_root=tmp_path)


def test_a_cache_entry_with_only_a_licence_file_is_metadata_only(tmp_path):
    snap = tmp_path / "models--owner--name" / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "LICENSE").write_text("text")
    with pytest.raises(ProbeError, match="carries metadata only"):
        probe_mod.resolve_snapshot("owner/name", hub_root=tmp_path)


def test_shards_in_a_sibling_snapshot_are_reported_and_never_merged_in(tmp_path):
    """Two snapshots are two revisions, so gluing one's config to the other's weights would
    describe a checkpoint that does not exist. Measured in the real cache on sshleifer/tiny-gpt2.
    """
    root = tmp_path / "models--owner--name" / "snapshots"
    (root / "with-config").mkdir(parents=True)
    (root / "with-config" / "config.json").write_text("{}")
    (root / "with-weights").mkdir(parents=True)
    write_safetensors(root / "with-weights" / "model.safetensors", {"a.weight": ("F32", [2])})
    got, how = probe_mod.resolve_snapshot("owner/name", hub_root=tmp_path)
    assert got.name == "with-config"
    assert "DIFFERENT snapshot (with-weights)" in how


# ── read_config and read_index ────────────────────────────────────────────────────

def test_a_config_that_is_not_json_names_the_file(tmp_path):
    (tmp_path / "config.json").write_text("{oh dear")
    with pytest.raises(ProbeError, match="could not be read as JSON"):
        probe_mod.read_config(tmp_path)


def test_a_config_that_is_a_list_is_not_a_config(tmp_path):
    (tmp_path / "config.json").write_text("[]")
    with pytest.raises(ProbeError, match="not a config object"):
        probe_mod.read_config(tmp_path)


def test_a_single_file_model_has_no_index_and_that_is_not_an_error(dense_snapshot):
    assert probe_mod.read_index(dense_snapshot) is None


def test_an_index_gives_the_shard_map_and_the_declared_total_size(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 99}, "weight_map": {"a.weight": "s1.safetensors"}}))
    got = probe_mod.read_index(tmp_path)
    assert got == {"weight_map": {"a.weight": "s1.safetensors"}, "metadata": {"total_size": 99}}


def test_an_index_with_no_weight_map_cannot_name_the_tensors(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    with pytest.raises(ProbeError, match="no usable weight_map"):
        probe_mod.read_index(tmp_path)


def test_an_index_that_is_not_json_names_the_file(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{")
    with pytest.raises(ProbeError, match="could not be read as JSON"):
        probe_mod.read_index(tmp_path)


# ── reading the config's claims ───────────────────────────────────────────────────

def test_a_multi_tower_config_hides_its_decoder_under_text_config():
    cfg = {"model_type": "gemma4", "text_config": {"num_hidden_layers": 60},
           "vision_config": {"num_hidden_layers": 27}}
    sub, key = probe_mod.decoder_config(cfg)
    assert (sub["num_hidden_layers"], key) == (60, "text_config")


def test_a_flat_config_is_its_own_decoder_config():
    sub, key = probe_mod.decoder_config({"num_hidden_layers": 4})
    assert (sub["num_hidden_layers"], key) == (4, None)


def test_a_nested_config_with_no_block_count_is_not_mistaken_for_the_decoder():
    """A vision tower keyed `text_config` by accident must not win over a real top level."""
    sub, key = probe_mod.decoder_config({"n_layer": 12, "text_config": {"foo": 1}})
    assert key is None and sub["n_layer"] == 12


def test_a_null_valued_key_counts_as_absent_because_it_is_a_third_state():
    assert probe_mod.first_key({"num_experts": None, "n_routed_experts": 8},
                               probe_mod.EXPERT_COUNT_KEYS) == (8, "n_routed_experts")
    assert probe_mod.first_key({}, ("a", "b")) == (None, None)


def test_the_block_total_includes_blocks_stored_past_the_declared_stack():
    got = probe_mod.block_count({"num_hidden_layers": 47, "num_nextn_predict_layers": 1})
    assert got["declared"] == 47
    assert got["extra"] == 1
    assert got["total"] == 48
    assert got["extra_key"] == "num_nextn_predict_layers"


def test_gpt2s_older_spelling_of_the_block_count_is_read():
    assert probe_mod.block_count({"n_layer": 12})["total"] == 12


def test_a_config_that_declares_no_depth_falls_back_to_its_mechanism_list():
    got = probe_mod.block_count({"layers_block_type": ["mamba", "moe", "attention"]})
    assert got["total"] == 3
    assert got["declared"] is None
    assert got["inferred_from"] == "layers_block_type"
    assert "no config key of" in got["why"]


def test_a_config_that_states_no_depth_at_all_says_so_instead_of_returning_zero():
    got = probe_mod.block_count({"hidden_size": 8})
    assert got["total"] is None
    assert "does not state a depth" in got["why"]


@pytest.mark.parametrize(("cfg", "key", "expected"), [
    ({"layer_types": ["conv", "full_attention"]}, "layer_types", ["conv", "full_attention"]),
    ({"layers_block_type": ["mamba", "moe"]}, "layers_block_type", ["mamba", "moe"]),
    ({"hybrid_override_pattern": "M-*"}, "hybrid_override_pattern",
     ["mamba", "mlp", "attention"]),
])
def test_each_spelling_of_the_per_block_mechanism_list_is_read(cfg, key, expected):
    types, got_key, why = probe_mod.block_types(cfg)
    assert (types, got_key, why) == (expected, key, None)


def test_an_unknown_letter_in_the_block_pattern_is_named_rather_than_guessed_at():
    types, key, why = probe_mod.block_types({"hybrid_override_pattern": "M-Z"})
    assert types is None
    assert key == "hybrid_override_pattern"
    assert "letters ['Z']" in why


def test_the_inverted_spelling_marks_attention_and_implies_every_other_block():
    """`attn_layer_indices` names only the attention blocks. Read at face value it would turn a
    32-block model into a 3-block one, which is the single easiest misreading in this set.
    """
    types, key, why = probe_mod.block_types({"num_hidden_layers": 5, "attn_layer_indices": [1, 3]})
    assert types == ["other", "attention", "other", "attention", "other"]
    assert (key, why) == ("attn_layer_indices", None)


def test_a_uniform_stack_says_which_four_spellings_were_looked_for():
    types, key, why = probe_mod.block_types({"num_hidden_layers": 4})
    assert (types, key) == (None, None)
    for expected in ("layer_types", "layers_block_type", "hybrid_override_pattern",
                     "attn_layer_indices"):
        assert expected in why


def test_an_empty_mechanism_list_is_not_treated_as_a_list():
    assert probe_mod.block_types({"layer_types": []})[0] is None


@pytest.mark.parametrize(("cfg", "count", "count_key", "topk_key"), [
    ({"num_experts": 128, "num_experts_per_tok": 8}, 128, "num_experts", "num_experts_per_tok"),
    ({"num_local_experts": 32, "experts_per_token": 4}, 32, "num_local_experts",
     "experts_per_token"),
    ({"n_routed_experts": 256, "num_experts_per_tok": 6}, 256, "n_routed_experts",
     "num_experts_per_tok"),
])
def test_each_spelling_of_the_expert_count_is_read_and_the_key_is_recorded(
        cfg, count, count_key, topk_key):
    got = probe_mod.expert_structure(cfg)
    assert got["present"] is True
    assert got["routed_experts"] == count
    assert got["routed_experts_key"] == count_key
    assert got["experts_per_token_key"] == topk_key


def test_an_expert_key_present_but_null_is_neither_a_count_nor_an_absent_key():
    got = probe_mod.expert_structure({"num_experts": None, "enable_moe_block": False})
    assert got["present"] is False
    assert got["routed_experts"] is None
    assert got["declared_but_null"] == ["num_experts"]
    assert got["every_count_key_seen"] == ["num_experts"]


def test_a_dense_prefix_and_an_mlp_only_list_are_two_spellings_of_the_same_fact():
    assert probe_mod.expert_structure({"num_dense_layers": 2})["dense_block_prefix"] == 2
    assert probe_mod.expert_structure({"mlp_only_layers": [0, 1]})["mlp_only_blocks"] == [0, 1]
    assert probe_mod.expert_structure({})["mlp_only_blocks"] is None


def test_the_usual_head_width_identity_is_reported_rather_than_assumed():
    holds = probe_mod.head_geometry({"hidden_size": 512, "num_attention_heads": 8, "head_dim": 64})
    assert holds["quotient_is_head_dim"] is True
    fails = probe_mod.head_geometry({"hidden_size": 5120, "num_attention_heads": 64,
                                     "head_dim": 128})
    assert fails["quotient_is_head_dim"] is False
    assert fails["hidden_over_heads"] == 80.0


def test_a_config_with_no_declared_head_width_leaves_the_identity_unanswered():
    got = probe_mod.head_geometry({"hidden_size": 8, "n_head": 2})
    assert got["quotient_is_head_dim"] is None
    assert got["num_attention_heads"] == 2


def test_latent_attentions_split_head_widths_are_carried_through():
    got = probe_mod.head_geometry({"hidden_size": 2048, "num_attention_heads": 20,
                                   "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
                                   "v_head_dim": 256})
    assert got["split_head_widths"] == {"qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
                                        "v_head_dim": 256}


# ── reading the tensors ───────────────────────────────────────────────────────────

def test_the_stack_prefix_is_inferred_from_the_names_rather_than_taken_from_a_list():
    names = list(llama_block(0)) + list(llama_block(1)) + ["model.embed_tokens.weight"]
    prefix, indices, why = probe_mod.stack_prefix(names)
    assert (prefix, indices, why) == ("model.layers", [0, 1], None)


def test_a_naming_scheme_no_hard_coded_path_list_contains_still_resolves():
    """The point of inferring rather than listing: `backbone.blocks` is in nobody's path list."""
    prefix, indices, _ = probe_mod.stack_prefix(
        [f"backbone.blocks.{i}.mixer.in_proj.weight" for i in range(4)])
    assert (prefix, indices) == ("backbone.blocks", [0, 1, 2, 3])


def test_an_expert_index_does_not_win_over_the_block_index():
    names = [f"model.layers.{b}.mlp.experts.{e}.down_proj.weight"
             for b in range(3) for e in range(64)]
    prefix, indices, _ = probe_mod.stack_prefix(names)
    assert (prefix, indices) == ("model.layers", [0, 1, 2])


def test_a_checkpoint_with_no_indexed_stack_says_so_instead_of_claiming_zero_blocks():
    prefix, indices, why = probe_mod.stack_prefix(["embedding.weight", "head.weight"])
    assert (prefix, indices) == (None, [])
    assert "no indexed stack" in why


def test_the_longest_consecutive_run_wins_over_a_longer_but_scattered_prefix():
    names = ([f"model.layers.{i}.mlp.down_proj.weight" for i in range(6)]
             + [f"extra.heads.{i}.weight" for i in (0, 4, 9, 20, 33, 41, 57)])
    prefix, _, _ = probe_mod.stack_prefix(names)
    assert prefix == "model.layers"


@pytest.mark.parametrize(("indices", "expected"), [
    ({0}, [1]), ({0, 1, 2}, [3]), ({0, 1, 5, 6, 7}, [2, 3]), ({3, 9}, [1, 1]),
])
def test_consecutive_runs_are_measured_rather_than_assumed(indices, expected):
    assert probe_mod._consecutive_runs(indices) == expected


def test_a_tensor_outside_the_stack_has_no_block_index():
    assert probe_mod.block_of("model.embed_tokens.weight", "model.layers") is None
    assert probe_mod.block_of("model.layers.weight", "model.layers") is None


def test_only_the_first_index_after_the_prefix_is_the_block():
    assert probe_mod.block_of("model.layers.7.mlp.experts.63.up_proj.weight",
                              "model.layers") == 7


def test_the_part_of_a_name_below_its_block_is_what_a_viewer_would_label():
    assert probe_mod.within_block("model.layers.3.self_attn.q_proj.weight",
                                  "model.layers", 3) == "self_attn.q_proj.weight"
    assert probe_mod.within_block("lm_head.weight", "model.layers", 3) == "lm_head.weight"


@pytest.mark.parametrize(("path", "role"), [
    ("self_attn.q_proj.weight", "attention"),
    ("attn.c_attn.weight", "attention"),
    ("self_attn.q_norm.weight", "attention_norm"),
    ("mlp.gate_proj.weight", "mlp"),
    ("mlp.down_proj.weight", "mlp"),
    ("mlp.gate.weight", "router"),
    ("mlp.router.weight", "router"),
    ("mlp.experts.4.down_proj.weight", "expert_mlp"),
    ("mlp.experts.4.gate_proj.weight", "expert_mlp"),
    ("input_layernorm.weight", "norm"),
    ("ln_1.bias", "norm"),
    ("conv.conv.weight", "conv"),
    ("mamba.A_log", "ssm"),
    ("mixer.in_proj.weight", "ssm"),
    ("operator.dt_bias", "ssm"),
    ("something_nobody_has_seen.weight", "unclassified"),
])
def test_every_tensor_gets_a_role_and_the_unknown_one_is_named_not_dropped(path, role):
    assert probe_mod.classify(path) == role


def test_a_dense_gated_mlp_is_never_mistaken_for_a_router():
    """`mlp.gate.weight` is a router and `mlp.gate_proj.weight` is not, and a substring match on
    "gate" reports a router in every dense Llama ever made. That is the whole reason the
    classifier matches whole dotted segments.
    """
    assert probe_mod.classify("mlp.gate_proj.weight") == "mlp"
    assert probe_mod.classify("mlp.gate.weight") == "router"


def test_an_expert_tensor_beats_the_mlp_rule_it_also_matches():
    assert probe_mod.classify("block_sparse_moe.experts.0.w1.weight") == "expert_mlp"


@pytest.mark.parametrize(("name", "dtype", "expected"), [
    ("transformer.h.0.attn.bias", "U8", False),
    ("transformer.h.0.attn.masked_bias", "F32", False),
    ("model.layers.0.self_attn.rotary_emb.inv_freq", "F32", False),
    ("model.layers.0.mlp.down_proj.weight", "BF16", True),
    ("some.index", "I64", False),
    ("model.layers.0.self_attn.q_proj.weight", None, True),
])
def test_a_buffer_riding_along_in_the_file_is_not_counted_as_a_parameter(name, dtype, expected):
    assert probe_mod.is_parameter(name, dtype) is expected


def test_gpt2s_fused_qkv_bias_is_a_parameter_and_its_causal_mask_is_not():
    """A REGRESSION, and it was a real wrong number rather than a hypothetical one. The first
    version matched the exclusion list as a substring, so `attn.c_attn.bias` matched the causal
    mask pattern `attn.bias` and GPT-2 block 0 came back 96 parameters light.
    """
    assert probe_mod.is_parameter("transformer.h.0.attn.c_attn.bias", "F32") is True
    assert probe_mod.is_parameter("transformer.h.0.attn.bias", "U8") is False


def test_a_rank_zero_tensor_holds_one_number_and_not_zero():
    assert probe_mod.elements([]) == 1
    assert probe_mod.elements([3, 4]) == 12
    assert probe_mod.elements([0, 4]) == 0


def test_a_hundred_and_twenty_eight_experts_collapse_to_one_readable_row():
    assert (probe_mod.collapse_expert_index("mlp.experts.127.down_proj.weight")
            == "mlp.experts.{e}.down_proj.weight")
    assert probe_mod.collapse_expert_index("mlp.down_proj.weight") == "mlp.down_proj.weight"


def test_an_inventory_separates_the_stack_from_everything_outside_it():
    shapes = {name: {"dtype": dt, "shape": sh} for name, (dt, sh) in {
        **llama_block(0), "model.embed_tokens.weight": ("F32", [16, 8])}.items()}
    inv = probe_mod.inventory(shapes, "model.layers")
    assert set(inv["outside"]) == {"model.embed_tokens.weight"}
    assert inv["blocks"][0]["mlp.gate_proj.weight"]["parameters"] == 128
    assert inv["blocks"][0]["self_attn.q_proj.weight"]["shapes"] == [[8, 8]]


def test_a_name_known_from_an_index_with_no_shard_on_disk_is_unknown_and_not_zero():
    """The state a part-downloaded cache is in, and the distinction the whole record turns on."""
    inv = probe_mod.inventory({"model.layers.0.mlp.down_proj.weight": None}, "model.layers")
    entry = inv["blocks"][0]["mlp.down_proj.weight"]
    assert entry["shapes_unknown"] == 1
    assert entry["shapes"] == []
    assert entry["parameters"] == 0


def test_an_experts_shapes_are_folded_and_the_instance_count_keeps_the_arithmetic():
    shapes = {f"model.layers.0.mlp.experts.{e}.down_proj.weight":
              {"dtype": "F32", "shape": [8, 4]} for e in range(16)}
    inv = probe_mod.inventory(shapes, "model.layers")
    entry = inv["blocks"][0]["mlp.experts.{e}.down_proj.weight"]
    assert entry["instances"] == 16
    assert entry["shapes"] == [[8, 4]]
    assert entry["parameters"] == 16 * 32


def test_two_blocks_of_the_same_kind_share_a_signature_and_a_hybrid_stack_does_not():
    shapes = {}
    for name, (dt, sh) in {**llama_block(0), **llama_block(1)}.items():
        shapes[name] = {"dtype": dt, "shape": sh}
    shapes["model.layers.2.conv.conv.weight"] = {"dtype": "F32", "shape": [8, 1, 3]}
    groups = probe_mod.distinct_block_shapes(probe_mod.inventory(shapes, "model.layers")["blocks"])
    assert [g["count"] for g in groups] == [2, 1]
    assert groups[0]["blocks"] == [0, 1]
    assert groups[1]["blocks"] == [2]


def test_a_signature_names_the_role_beside_the_template():
    inv = probe_mod.inventory(
        {"model.layers.0.mlp.gate.weight": {"dtype": "F32", "shape": [4, 8]}}, "model.layers")
    assert probe_mod.block_signature(inv["blocks"][0]) == "router:mlp.gate.weight"


def test_the_shallowest_block_is_shown_in_full_and_an_empty_stack_is_an_empty_dict():
    assert probe_mod._first_block({}) == {}
    assert probe_mod._first_block({3: {"a": 1}, 7: {"b": 2}}) == {"a": 1}


# ── one model, end to end ─────────────────────────────────────────────────────────

def test_a_hand_built_dense_checkpoint_probes_to_a_complete_record(dense_snapshot):
    rec = probe_mod.probe(str(dense_snapshot))
    assert rec["evidence_tier"] == "header"
    assert rec["block_count"]["total"] == 2
    assert rec["tensors"]["stack_prefix"] == "model.layers"
    assert rec["tensors"]["names_known"] == rec["tensors"]["shapes_known"] == 14
    assert rec["tensors"]["parameters_from_shapes"] == 2 * (8 + 64 + 64 + 8 + 128 + 128) + 128 + 8
    assert rec["tensors"]["distinct_block_shapes"][0]["count"] == 2
    assert rec["consistency"] == []


def test_an_index_with_no_shards_probes_at_the_index_tier_and_says_the_shapes_are_absent(tmp_path):
    snap = tmp_path / "sharded"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "hidden_size": 8}))
    (snap / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 1234},
        "weight_map": dict.fromkeys(llama_block(0), "model-00001-of-00002.safetensors")}))
    rec = probe_mod.probe(str(snap))
    assert rec["evidence_tier"] == "index"
    assert rec["tensors"]["shapes_known"] == 0
    assert rec["tensors"]["index_total_size_bytes"] == 1234
    assert any("absent rather than zero" in note for note in rec["consistency"])


def test_one_unreadable_shard_keeps_the_others_and_records_the_reason(tmp_path):
    snap = tmp_path / "part"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "hidden_size": 8}))
    write_safetensors(snap / "model-00001-of-00002.safetensors", llama_block(0))
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"\0\0\0")
    rec = probe_mod.probe(str(snap))
    assert rec["tensors"]["shapes_known"] == 6
    assert "too short" in rec["tensors"]["shard_errors"]["model-00002-of-00002.safetensors"]


def test_no_shards_read_at_all_when_the_caller_asks_for_config_only(dense_snapshot):
    rec = probe_mod.probe(str(dense_snapshot), read_shards=False)
    assert rec["evidence_tier"] == "config"
    assert rec["tensors"]["names_known"] == 0


def test_a_block_count_the_tensors_contradict_is_a_finding_rather_than_a_crash(tmp_path):
    snap = tmp_path / "mismatch"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 9, "hidden_size": 8}))
    write_safetensors(snap / "model.safetensors", {**llama_block(0), **llama_block(1)})
    rec = probe_mod.probe(str(snap))
    assert any("tensor names run to index 1" in note for note in rec["consistency"])


def test_a_config_naming_no_architecture_is_a_note_because_nothing_can_resolve_by_class(tmp_path):
    snap = tmp_path / "anon"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "gpt2", "n_layer": 1}))
    rec = probe_mod.probe(str(snap))
    assert any("names no architectures" in note for note in rec["consistency"])


# ── the summary and the CLI ───────────────────────────────────────────────────────

def test_the_summary_names_the_keys_every_number_came_from(dense_snapshot):
    lines = probe_mod.summarise(probe_mod.probe(str(dense_snapshot)))
    joined = "\n".join(lines)
    assert "STRUCT_BLOCKS" in joined
    assert "under=num_hidden_layers" in joined
    assert "STRUCT_EXPERTS" in joined and "none declared" in joined
    assert "STRUCT_MECHANISMS" in joined


def test_the_summary_reports_a_null_valued_expert_key_as_its_own_state():
    cfg = {"architectures": ["X"], "model_type": "x", "num_hidden_layers": 1,
           "num_experts": None}
    record = {"spec": "m", "model_type": "x", "architectures": ["X"], "evidence_tier": "config",
              "block_count": probe_mod.block_count(cfg),
              "block_types": {"types": None, "key": None, "why": "uniform", "tally": None},
              "experts": probe_mod.expert_structure(cfg),
              "tensors": {"names_known": 0, "shapes_known": 0, "stack_prefix": None,
                          "stack_block_count": 0, "parameters_from_shapes": 0,
                          "dtype_histogram": {}, "distinct_block_shapes": []},
              "consistency": []}
    joined = "\n".join(probe_mod.summarise(record))
    assert "null-valued keys ['num_experts']" in joined


def test_the_cli_writes_one_json_document_and_renames_it_into_place(dense_snapshot, tmp_path):
    out = tmp_path / "out.json"
    assert probe_mod.main([str(dense_snapshot), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert [r["spec"] for r in doc["records"]] == [str(dense_snapshot)]
    assert doc["unreadable"] == {}
    assert not list(tmp_path.glob("*.part")), "a partial file must never be left behind"


def test_the_cli_refuses_an_output_directory_that_does_not_exist_before_reading_anything(
        dense_snapshot, tmp_path):
    with pytest.raises(SystemExit, match="which does not exist"):
        probe_mod.main([str(dense_snapshot), "--out", str(tmp_path / "nope" / "out.json")])


def test_an_unreadable_model_stops_the_run_by_default_and_writes_nothing(tmp_path):
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="Pass --keep-going"):
        probe_mod.main([str(tmp_path / "absent"), "--out", str(out)])
    assert not out.exists()


def test_keep_going_records_the_reason_and_carries_on_to_the_next_model(dense_snapshot, tmp_path):
    out = tmp_path / "out.json"
    assert probe_mod.main([str(tmp_path / "absent"), str(dense_snapshot),
                           "--out", str(out), "--keep-going"]) == 0
    doc = json.loads(out.read_text())
    assert len(doc["records"]) == 1
    assert list(doc["unreadable"]) == [str(tmp_path / "absent")]


def test_a_run_where_nothing_was_readable_writes_no_file_and_lists_every_reason(tmp_path):
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="no model was readable"):
        probe_mod.main([str(tmp_path / "a"), str(tmp_path / "b"),
                        "--out", str(out), "--keep-going"])
    assert not out.exists()


def test_two_runs_on_one_checkpoint_agree_byte_for_byte_apart_from_the_timestamp(
        dense_snapshot, tmp_path):
    """Reproducibility, per the baseline. Set iteration order is not stable across processes once
    the string hash seed changes, and a record that differs between runs cannot be diffed, which
    is the only thing this output is for.
    """
    first = probe_mod.probe(str(dense_snapshot))
    second = probe_mod.probe(str(dense_snapshot))
    for rec in (first, second):
        rec.pop("probed_at")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_probes_layer_count_keys_still_agree_with_the_packages_own_list():
    """A DIVERGENCE HERE IS A FINDING AND NOT A FAILURE OF EITHER SIDE. The probe keeps its own
    copy on purpose, so that it can find a spelling the package does not know. This test exists to
    make the two lists drifting apart visible at the moment it happens, rather than six months
    later when a checkpoint reads as having no layers.
    """
    from senbonzakura import streaming
    assert tuple(probe_mod.LAYER_COUNT_KEYS) == tuple(streaming.LAYER_COUNT_KEYS), (
        "the probe's layer-count spellings have drifted from senbonzakura.streaming's. One of the "
        "two has learned a spelling the other has not; decide which and update deliberately.")
