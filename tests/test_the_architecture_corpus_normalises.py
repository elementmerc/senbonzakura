# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The parsing and normalising half of the architecture corpus, with no socket in sight.

WHY THIS FILE EXISTS

`tools/research/fetch_architecture_corpus.py` turns a hundred strangers' `config.json` files
into one shape. The fetching is the boring half; the normalising is where a wrong answer
becomes a wrong map, and a wrong map reads a checkpoint's depth off a vision tower.

Everything here runs on hand-built fixtures. No test in this file touches the network, and the
cases are the ones a live run cannot be relied upon to produce on demand: a config that nests
its decoder, one that spells depth the old way, a safetensors header with a lying length, an
adapter repository with no architecture of its own.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "research"))

import fetch_architecture_corpus as fac

# ---------------------------------------------------------------------------
# What the tool refuses to fetch. This is a promise, so it gets a test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "model-00001-of-00163.bin",
        "pytorch_model.bin",
        "model.gguf",
        "weights.pth",
        "consolidated.pt",
        "flax_model.msgpack",
        "tokenizer.json",
        "https://huggingface.co/x/y/resolve/main/tokenizer.json?download=true",
    ],
)
def test_a_weight_body_or_a_tokenizer_blob_is_forbidden(name):
    assert fac.is_forbidden(name), name


@pytest.mark.parametrize(
    "name",
    ["config.json", "model.safetensors.index.json", "generation_config.json", "model.safetensors"],
)
def test_metadata_and_a_safetensors_header_are_not_forbidden(name):
    """The shard itself is allowed through `is_forbidden` on purpose.

    A range request over its header is the whole technique. The guard that stops a BODY being
    read is `Fetcher.get`, which refuses an unranged safetensors request, and that is tested
    separately below so the two cannot drift.
    """
    assert not fac.is_forbidden(name), name


def test_an_unranged_safetensors_request_is_refused_before_any_socket():
    budget = fac.Budget(deadline=float("inf"), byte_budget=1)
    fetcher = fac.Fetcher(budget, Path("/nonexistent"), lambda _m: None)
    with pytest.raises(fac.CorpusError, match="unranged safetensors"):
        fetcher.get("https://huggingface.co/x/y/resolve/main/model.safetensors")
    assert budget.requests == 0, "it must refuse without spending a request"


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://huggingface.co/x/y/resolve/main/config.json", "non-https"),
        ("https://example.invalid/config.json", "outside the allowed set"),
        ("https://huggingface.co/x/y/resolve/main/pytorch_model.bin", "forbidden"),
    ],
)
def test_check_url_refuses_the_scheme_the_host_and_the_payload(url, reason):
    with pytest.raises(fac.CorpusError, match=reason):
        fac.check_url(url)


def test_check_url_passes_the_three_hosts_the_permission_covers():
    for url in (
        "https://huggingface.co/api/models/Qwen/Qwen3-0.6B",
        "https://raw.githubusercontent.com/rasbt/llm-architecture-gallery/main/models.yml",
        "https://api.github.com/repos/rasbt/llm-architecture-gallery/commits/main",
    ):
        assert fac.check_url(url) == url


# ---------------------------------------------------------------------------
# The safetensors header. A remote file states how many bytes to ask for next,
# which is exactly the shape of input that must never be trusted.
# ---------------------------------------------------------------------------


def test_the_header_length_is_read_little_endian():
    assert fac.safetensors_header_length((1234).to_bytes(8, "little")) == 1234


@pytest.mark.parametrize(
    ("prefix", "reason"),
    [
        (b"\x01\x02", "need 8"),
        ((0).to_bytes(8, "little"), "zero"),
        ((fac.MAX_HEADER_BYTES + 1).to_bytes(8, "little"), "over the"),
        ((2**63).to_bytes(8, "little"), "over the"),
    ],
)
def test_a_lying_header_length_is_refused_rather_than_requested(prefix, reason):
    with pytest.raises(fac.CorpusError, match=reason):
        fac.safetensors_header_length(prefix)


def test_a_header_is_parsed_into_names_shapes_and_dtypes():
    blob = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [4096, 4096]},
            "model.embed_tokens.weight": {"dtype": "F32", "shape": [128000, 4096]},
        }
    ).encode("utf-8")
    parsed = fac.parse_safetensors_header(blob)
    assert parsed["metadata"] == {"format": "pt"}
    assert set(parsed["tensors"]) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
    }
    assert parsed["tensors"]["model.embed_tokens.weight"] == {"dtype": "F32", "shape": [128000, 4096]}


@pytest.mark.parametrize(
    ("blob", "reason"),
    [
        (b"not json at all", "not JSON"),
        (b"[1, 2, 3]", "expected an object"),
        (b'{"t": 7}', "spec"),
    ],
)
def test_a_malformed_header_fails_loudly(blob, reason):
    with pytest.raises(fac.CorpusError, match=reason):
        fac.parse_safetensors_header(blob)


# ---------------------------------------------------------------------------
# Tensor names. Folding is what keeps a 42,000 tensor MoE from being stored as
# 42,000 copies of the same six facts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "folded"),
    [
        ("model.layers.31.mlp.down_proj.weight", "model.layers.{i}.mlp.down_proj.weight"),
        (
            "model.layers.3.mlp.experts.17.gate_proj.weight",
            "model.layers.{i}.mlp.experts.{e}.gate_proj.weight",
        ),
        ("model.embed_tokens.weight", "model.embed_tokens.weight"),
        ("transformer.h.0.attn.c_attn.bias", "transformer.h.{i}.attn.c_attn.bias"),
        ("model.layers.5.input_layernorm.weight", "model.layers.{i}.input_layernorm.weight"),
    ],
)
def test_layer_and_expert_indices_fold_and_nothing_else_does(name, folded):
    assert fac.fold_tensor_name(name) == folded


def test_the_inventory_counts_patterns_dtypes_prefixes_and_the_depth_it_implies():
    names = {}
    for layer in range(4):
        names[f"model.layers.{layer}.self_attn.q_proj.weight"] = {"dtype": "BF16"}
        for expert in range(3):
            names[f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"] = {"dtype": "F8_E4M3"}
    names["lm_head.weight"] = {"dtype": "BF16"}
    inventory = fac.tensor_inventory(names)
    assert inventory["tensor_count"] == 4 + 12 + 1
    assert inventory["pattern_count"] == 3
    assert inventory["dtypes"] == {"BF16": 5, "F8_E4M3": 12}
    assert inventory["prefixes"] == {"lm_head": 1, "model": 16}
    assert inventory["layer_counts_by_stem"] == {"model.layers": 4}, "depth, from names alone"
    assert inventory["expert_counts_by_stem"] == {"model.layers.{i}.mlp.experts": 3}
    assert inventory["deepest_stack"] == "model.layers"
    assert inventory["deepest_stack_layers"] == 4


# ---------------------------------------------------------------------------
# Key resolution across the spellings the wild actually uses.
# ---------------------------------------------------------------------------


def test_the_modern_spelling_wins_when_a_config_carries_both():
    """A config with both keys is not a tie; candidate order is the authority."""
    config = {"n_layer": 12, "num_hidden_layers": 32}
    assert fac.resolve_key(config, fac.DEPTH_KEYS) == ("num_hidden_layers", 32)
    assert fac.observed_keys(config, fac.DEPTH_KEYS) == ["n_layer", "num_hidden_layers"]


def test_an_absent_dimension_resolves_to_nothing_rather_than_a_guess():
    assert fac.resolve_key({"hidden_size": 8}, fac.EXPERT_COUNT_KEYS) == (None, None)


@pytest.mark.parametrize(
    ("config", "depth"),
    [
        ({"num_hidden_layers": 32}, 32),
        ({"n_layer": 12}, 12),
        ({"num_layers": 48}, 48),
        ({"depth": 16}, 16),
    ],
)
def test_four_spellings_of_depth_all_read(config, depth):
    assert fac.structural_profile(config)["resolved"]["depth"]["value"] == depth


# ---------------------------------------------------------------------------
# Nested configs. This is the failure that matters: a wrapper with no depth of
# its own, whose decoder sits one or two levels down.
# ---------------------------------------------------------------------------


def test_a_wrapper_config_is_read_at_its_decoder_not_at_its_top_level():
    config = {
        "model_type": "some_vlm",
        "vision_config": {"num_hidden_layers": 27, "hidden_size": 1152},
        "text_config": {"num_hidden_layers": 48, "hidden_size": 5120, "num_attention_heads": 40},
    }
    profile = fac.structural_profile(config)
    assert profile["decoder_is_nested"] is True
    assert profile["decoder_path"] == "text_config"
    assert profile["resolved"]["depth"]["value"] == 48, "the text tower, never the vision tower"
    assert profile["nested_config_keys"] == ["text_config", "vision_config"]


def test_a_decoder_two_levels_down_is_still_found():
    config = {"model_type": "omni", "thinker_config": {"text_config": {"num_hidden_layers": 28}}}
    profile = fac.structural_profile(config)
    assert profile["decoder_path"] == "thinker_config.text_config"
    assert profile["resolved"]["depth"]["value"] == 28


def test_a_sub_config_under_an_unknown_key_is_found_by_shape():
    """The named-key list will always be behind the ecosystem, so shape has to carry it."""
    config = {"model_type": "brand_new", "backbone_cfg": {"num_hidden_layers": 11, "hidden_size": 7}}
    assert "backbone_cfg" in fac.nested_configs(config)
    assert fac.structural_profile(config)["decoder_path"] == "backbone_cfg"


def test_a_flat_config_reports_no_nesting_and_an_empty_decoder_path():
    profile = fac.structural_profile({"model_type": "llama", "num_hidden_layers": 32})
    assert profile["decoder_path"] == ""
    assert profile["decoder_is_nested"] is False
    assert profile["nested_config_keys"] == []


def test_the_top_level_wins_even_when_a_vision_tower_is_present():
    """Several checkpoints put the decoder at the top AND carry a vision tower beside it."""
    config = {"num_hidden_layers": 40, "vision_config": {"num_hidden_layers": 27}}
    profile = fac.structural_profile(config)
    assert profile["decoder_path"] == ""
    assert profile["resolved"]["depth"]["value"] == 40


# ---------------------------------------------------------------------------
# Mechanism classification, read off the evidence rather than off a model name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"num_attention_heads": 32, "num_key_value_heads": 32}, "mha"),
        ({"num_attention_heads": 32, "num_key_value_heads": 8}, "gqa"),
        ({"num_attention_heads": 32, "num_key_value_heads": 1}, "mqa"),
        ({"n_head": 16}, "mha-or-unstated"),
    ],
)
def test_the_attention_family_follows_from_the_head_counts(config, expected):
    assert expected in fac.classify_mechanisms(config)


def test_a_sparse_config_is_recognised_through_any_of_its_spellings():
    for key in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"):
        assert "moe" in fac.classify_mechanisms({key: 128}), key


def test_latent_attention_sliding_windows_and_per_layer_lists_are_each_named():
    marks = fac.classify_mechanisms(
        {
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "sliding_window": 4096,
            "layer_types": ["full_attention", "sliding_attention"],
            "use_qk_norm": True,
            "rope_theta": 1e6,
            "rope_scaling": {"rope_type": "yarn"},
        }
    )
    assert {"mla", "sliding-window", "per-layer-type-list", "qk-norm", "rope", "rope-scaled"} <= set(marks)


def test_a_state_space_hybrid_is_not_mistaken_for_a_plain_transformer():
    marks = fac.classify_mechanisms(
        {"hybrid_override_pattern": "M-M-M*-", "mamba_d_state": 128, "num_attention_heads": 32}
    )
    assert "ssm-or-linear-attention" in marks
    assert "hybrid-pattern-string" in marks


def test_a_quantised_checkpoint_says_so():
    assert "quantised-checkpoint" in fac.classify_mechanisms({"quantization_config": {"bits": 4}})


# ---------------------------------------------------------------------------
# The gallery, the tokenizer summary, and the sibling rule.
# ---------------------------------------------------------------------------


def test_the_gallery_flattens_to_facts_and_keeps_the_licence():
    entries = fac.gallery_entries(
        {
            "Zebra 9B": {
                "company": "Acme",
                "date": "2026-02-02",
                "license_name": "Apache License 2.0",
                "license_url": "https://example.com/licence",
                "summary": "prose we do not keep",
                "related_concepts": ["gqa", "rope"],
                "config": {"repo": "acme/zebra-9b", "url": "https://example.com/config.json"},
                "tech_report": {"url": "https://example.com/paper.pdf"},
            }
        }
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry["repo"] == "acme/zebra-9b"
    assert entry["license_name"] == "Apache License 2.0"
    assert entry["tech_report_url"] == "https://example.com/paper.pdf"
    assert "summary" not in entry, "the gallery author's prose is not ours to carry"


def test_an_entry_with_no_repository_is_a_record_rather_than_a_crash():
    entries = fac.gallery_entries({"Closed 7B": {"company": "Acme", "license_name": "Proprietary"}})
    assert entries[0]["repo"] is None


def test_a_gallery_entry_of_the_wrong_shape_fails_loudly():
    with pytest.raises(fac.CorpusError, match="expected a mapping"):
        fac.gallery_entries({"Broken": ["not", "a", "mapping"]})


def test_a_chat_template_is_kept_as_a_digest_and_a_prefix_not_a_copy():
    template = "{% for m in messages %}" + "x" * 5000
    summary = fac.summarise_tokenizer_config(
        {
            "chat_template": template,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": {"content": "<|begin|>", "lstrip": False},
            "eos_token": "<|end|>",
            "added_tokens_decoder": {"0": {}, "1": {}},
        }
    )
    assert summary["chat_template"]["chars"] == len(template)
    assert len(summary["chat_template"]["prefix"]) == fac.CHAT_TEMPLATE_PREFIX_CHARS
    assert summary["chat_template"]["sha256"] != ""
    assert summary["bos_token"] == "<|begin|>", "an AddedToken object reads the same as a string"
    assert summary["eos_token"] == "<|end|>"
    assert summary["added_token_count"] == 2


def test_a_named_template_list_is_recorded_by_name():
    summary = fac.summarise_tokenizer_config(
        {"chat_template": [{"name": "default", "template": "a"}, {"name": "tool_use", "template": "b"}]}
    )
    assert summary["chat_template"] == {"kind": "list", "entries": ["default", "tool_use"]}


def test_a_family_prefix_is_a_search_term_not_a_promise():
    assert fac.family_prefix("Qwen/Qwen3-235B-A22B-Instruct") == "Qwen/Qwen3"
    assert fac.family_prefix("deepseek-ai/DeepSeek-V3") == "deepseek-ai/DeepSeek"
    assert fac.family_prefix("bare-owner") == "bare-owner"


def test_a_sibling_is_taken_only_when_it_adds_an_architecture_we_do_not_hold():
    candidates = [
        {"id": "acme/known-7b", "config": {"model_type": "already_held"}},
        {"id": "acme/gated-7b", "config": {"model_type": "novel_a"}, "gated": "auto"},
        {"id": "acme/private-7b", "config": {"model_type": "novel_b"}, "private": True},
        {"id": "acme/no-config-7b"},
        {"id": "acme/novel-7b", "config": {"model_type": "novel_c"}},
        {"id": "acme/novel-again-7b", "config": {"model_type": "novel_c"}},
        {"id": "acme/novel-d-7b", "config": {"model_type": "novel_d"}},
        {"id": "acme/novel-e-7b", "config": {"model_type": "novel_e"}},
    ]
    taken = fac.choose_siblings(candidates, {"acme/in-corpus"}, {"already_held"}, per_family=2)
    assert taken == ["acme/novel-7b", "acme/novel-d-7b"], (
        "gated, private, config-less, duplicate-type and over-cap candidates are all skipped"
    )


def test_a_repository_already_in_the_corpus_is_never_taken_twice():
    assert fac.choose_siblings([{"id": "acme/x", "config": {"model_type": "t"}}], {"acme/x"}, set()) == []


# ---------------------------------------------------------------------------
# Aggregation and the budgets.
# ---------------------------------------------------------------------------


def test_the_observed_vocabulary_counts_spellings_across_records():
    records = [
        {
            "repo": "a/a",
            "name": "a",
            "status": "complete",
            "licence": {},
            "problems": [],
            "source": "gallery",
            "config": {"torch_dtype": "bfloat16", "hidden_act": "silu", "num_hidden_layers": 4},
            "profile": fac.structural_profile({"num_hidden_layers": 4, "torch_dtype": "bfloat16"}),
        },
        {
            "repo": "b/b",
            "name": "b",
            "status": "complete",
            "licence": {},
            "problems": [],
            "source": "gallery",
            "config": {"dtype": "float16", "hidden_activation": "gelu_pytorch_tanh", "n_layer": 6},
            "profile": fac.structural_profile({"n_layer": 6, "dtype": "float16"}),
        },
    ]
    vocab = fac.observed_vocabulary(records)
    assert vocab["per_dimension"]["depth"] == {"n_layer": 1, "num_hidden_layers": 1}
    assert vocab["dtype_declarations"] == {"dtype=float16": 1, "torch_dtype=bfloat16": 1}
    assert vocab["activations"]["hidden_act=silu"] == 1


def test_the_index_carries_the_revision_the_status_and_the_licence_of_every_row():
    record = {
        "repo": "a/a",
        "name": "A",
        "source": "gallery",
        "status": "partial",
        "problems": ["something was missing"],
        "hub": {"revision": "deadbeef"},
        "licence": {"gallery_license_name": "MIT"},
        "profile": fac.structural_profile({"num_hidden_layers": 4}),
        "tensors": {"tensor_count": 9, "source": "header"},
    }
    index = fac.build_index([record], {"gallery_revision": "cafe"})
    assert index["counts"] == {"partial": 1, "total": 1}
    row = index["models"][0]
    assert row["revision"] == "deadbeef"
    assert row["licence"] == "MIT"
    assert row["problems"] == ["something was missing"]
    assert index["provenance"]["gallery_revision"] == "cafe"


def test_the_byte_budget_is_a_refusal_and_not_a_warning():
    budget = fac.Budget(deadline=float("inf"), byte_budget=100)
    budget.charge(60)
    with pytest.raises(fac.BudgetExhaustedError, match="byte budget"):
        budget.charge(60)
    assert budget.requests == 2, "the request that broke the budget is still counted"


def test_the_clock_is_a_refusal_too():
    budget = fac.Budget(deadline=0.0, byte_budget=1)
    with pytest.raises(fac.BudgetExhaustedError, match="deadline"):
        budget.check_clock()


def test_a_record_is_written_atomically_and_gzipped_only_when_it_is_large(tmp_path):
    small = tmp_path / "small.json"
    fac.write_json_atomic(small, {"a": 1})
    assert small.exists() and not small.with_suffix(".json.gz").exists()
    big = tmp_path / "big.json"
    fac.write_json_atomic(big, {"a": "x" * 2_000_000})
    assert (tmp_path / "big.json.gz").exists()
    assert not big.exists(), "the uncompressed twin must not be left beside the gzip"
    assert not list(tmp_path.glob("*.part.*")), "no partial file may survive a write"


def test_two_writes_of_the_same_record_give_byte_identical_files(tmp_path):
    """Reproducibility, per the baseline: iteration order must not leak into the output."""
    payload = {"b": 2, "a": {"z": 1, "y": [3, 2, 1]}}
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    fac.write_json_atomic(first, payload)
    fac.write_json_atomic(second, dict(reversed(list(payload.items()))))
    assert first.read_bytes() == second.read_bytes()


def test_a_shard_is_chosen_predictably_and_a_repository_with_none_says_so():
    assert fac.pick_safetensors_shard(["config.json", "model.safetensors"]) == "model.safetensors"
    assert (
        fac.pick_safetensors_shard(["model-00002-of-00003.safetensors", "model-00001-of-00003.safetensors"])
        == "model-00001-of-00003.safetensors"
    )
    assert fac.pick_safetensors_shard(["config.json", "pytorch_model.bin"]) is None


def test_two_gallery_entries_sharing_one_repository_both_survive_the_write():
    """Observed in the live run: two gallery names pointed at `moonshotai/Kimi-K3`.

    Keyed by repository alone, one record silently overwrote the other and the corpus was one
    model short with nothing saying so.
    """
    records = [
        {"repo": "moonshotai/Kimi-K3", "name": "Ember-1"},
        {"repo": "moonshotai/Kimi-K3", "name": "Kimi K3"},
        {"repo": "acme/solo", "name": "Solo"},
    ]
    slugs = fac.record_slugs(records)
    assert len(slugs) == 3
    assert "acme__solo" in slugs, "a repository with no collision keeps the plain slug"
    assert sorted(k for k in slugs if k.startswith("moonshotai")) == [
        "moonshotai__Kimi-K3--Ember-1",
        "moonshotai__Kimi-K3--Kimi_K3",
    ]


def test_a_slug_cannot_escape_the_models_directory():
    slugs = fac.record_slugs(
        [{"repo": "a/b", "name": "../../etc/passwd"}, {"repo": "a/b", "name": "x"}]
    )
    assert all("/" not in slug and ".." not in slug for slug in slugs), slugs


def test_an_expert_index_is_never_counted_as_a_layer():
    """The first version of this file read a 48 layer MoE as 128 layers deep.

    `model.layers.3.mlp.experts.127.up_proj.weight` has 127 as its largest number, and a
    single maximum over every numeric component in the name picks it up as the depth.
    """
    layers, experts = fac.index_counts_by_stem("model.layers.3.mlp.experts.127.up_proj.weight")
    assert layers == {"model.layers": 4}
    assert experts == {"model.layers.{i}.mlp.experts": 128}


def test_each_stack_in_a_multimodal_checkpoint_is_counted_separately():
    """A vision tower and a decoder are two stacks, and conflating them hides both depths."""
    names = {}
    for i in range(27):
        names[f"vision_tower.encoder.layers.{i}.mlp.fc1.weight"] = {}
    for i in range(48):
        names[f"language_model.model.layers.{i}.self_attn.q_proj.weight"] = {}
    for i in range(2):
        names[f"mtp.layers.{i}.weight"] = {}
    inventory = fac.tensor_inventory(names)
    assert inventory["layer_counts_by_stem"] == {
        "language_model.model.layers": 48,
        "mtp.layers": 2,
        "vision_tower.encoder.layers": 27,
    }
    assert inventory["deepest_stack"] == "language_model.model.layers"
    assert inventory["deepest_stack_layers"] == 48


def test_a_checkpoint_with_no_indexed_stack_reports_no_depth_rather_than_zero():
    inventory = fac.tensor_inventory({"embeddings.weight": {}, "head.bias": {}})
    assert inventory["layer_counts_by_stem"] == {}
    assert inventory["deepest_stack"] is None
    assert inventory["deepest_stack_layers"] is None


def test_a_nested_module_list_does_not_become_one_stack_per_outer_layer():
    """Observed on a live checkpoint: 363 stacks reported, all of them the same inner list."""
    names = {}
    for layer in range(3):
        for inner in range(2):
            names[f"model.layers.{layer}.attn.lora.{inner}.weight"] = {}
    inventory = fac.tensor_inventory(names)
    assert inventory["layer_counts_by_stem"] == {
        "model.layers": 3,
        "model.layers.{i}.attn.lora": 2,
    }
    assert inventory["deepest_stack"] == "model.layers"


def test_the_text_tower_beats_the_audio_tower_when_both_declare_a_depth():
    """Observed on two Gemma 4 records: alphabetical order read the audio tower as the decoder.

    `audio_config` sorts before `text_config`, so a reader that walks the nested keys in sorted
    order picks the wrong stack and every number it reports afterwards is the wrong model's.
    """
    config = {
        "model_type": "gemma4",
        "audio_config": {"num_hidden_layers": 12, "hidden_size": 1536},
        "text_config": {"num_hidden_layers": 35, "hidden_size": 2048},
        "vision_config": {"num_hidden_layers": 27},
    }
    profile = fac.structural_profile(config)
    assert profile["decoder_path"] == "text_config"
    assert profile["resolved"]["depth"]["value"] == 35


def test_an_audio_only_wrapper_still_resolves_to_its_one_stack():
    """The preference is an ordering, not a requirement: with no text tower, audio is the stack."""
    config = {"model_type": "speech", "audio_config": {"num_hidden_layers": 12}}
    assert fac.structural_profile(config)["decoder_path"] == "audio_config"


def test_the_encoder_decoder_spellings_resolve():
    """A BART-shaped checkpoint names its heads and its ffn per side."""
    profile = fac.structural_profile(
        {"num_hidden_layers": 12, "decoder_attention_heads": 16, "decoder_ffn_dim": 4096, "d_model": 1024}
    )
    assert profile["resolved"]["heads"]["value"] == 16
    assert profile["resolved"]["ffn"]["value"] == 4096
    assert profile["resolved"]["width"]["value"] == 1024


# ---------------------------------------------------------------------------
# The catalogue sweep: what gets admitted, what gets skipped, and whether the
# reason is always recorded.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("repo", "fragment"),
    [
        ("bartowski/Qwen3-8B-GGUF", "repackager account"),
        ("mradermacher/something-i1-GGUF", "repackager account"),
        ("someone/Llama-3-8B-GGUF", "repackaged format"),
        ("someone/Mistral-7B-AWQ", "repackaged format"),
        ("someone/model-4bit", "repackaged format"),
        ("someone/model-mlx", "repackaged format"),
    ],
)
def test_a_repackaged_republish_is_skipped_for_shape(repo, fragment):
    assert fragment in (fac.repackaged_reason(repo) or "")


@pytest.mark.parametrize(
    "repo",
    [
        "huihui-ai/Qwen3-8B-abliterated",
        "someone/Llama-3-8B-uncensored",
        "someone/Mistral-7B-orthogonalized",
        "deepseek-ai/DeepSeek-V3",
    ],
)
def test_an_abliterated_or_ordinary_model_is_not_skipped(repo):
    """These are wanted. Measuring somebody else's refusal edit is the point of the project."""
    assert fac.repackaged_reason(repo) is None
    assert fac.harm_reason(repo) is None


@pytest.mark.parametrize(
    ("repo", "tags"),
    [
        ("someone/malware-generator-7b", None),
        ("someone/innocuous-name", ["ransomware"]),
        ("someone/prompt-injection-kit", None),
        ("someone/credential-stealer-llm", None),
    ],
)
def test_a_repository_presenting_as_harm_is_skipped_by_decision(repo, tags):
    reason = fac.harm_reason(repo, tags)
    assert reason and reason.startswith("presents as harm"), reason


def test_every_rejected_search_row_carries_a_reason():
    """A row must either enter the corpus or enter the skipped list. Never neither."""
    rows = [
        {"id": "a/public", "config": {"model_type": "llama"}},
        {"id": "b/gated", "gated": "auto", "config": {"model_type": "gemma"}},
        {"id": "c/private", "private": True, "config": {"model_type": "x"}},
        {"id": "d/no-type", "config": {}},
        {"id": "e/model-GGUF", "config": {"model_type": "llama"}},
        {},
    ]
    for row in rows:
        repo, reason = fac.discovery_candidate(row)
        assert (repo is None) != (reason is None), row


def test_novelty_is_admitted_before_a_cap_filling_repeat():
    """A new model_type in the last query beats a sibling of a known shape in the first."""
    results = {
        ("legacy", "known-family", 2, True): [
            {"id": "x/known-1", "config": {"model_type": "known", "architectures": ["KnownLM"]}},
            {"id": "x/known-2", "config": {"model_type": "known", "architectures": ["KnownLM"]}},
        ],
        ("mechanism", "rwkv", 2, True): [
            {"id": "RWKV/rwkv-7", "config": {"model_type": "rwkv7", "architectures": ["Rwkv7LM"]}},
        ],
    }
    admitted, _ = fac.choose_discoveries(results, set(), {"known"}, {"KnownLM"}, target=2)
    assert admitted[0][0] == "RWKV/rwkv-7"
    assert "new model_type rwkv7" in admitted[0][2]


def test_a_repeat_is_admitted_only_where_the_category_says_it_is_evidence():
    results = {
        ("mechanism", "based", 2, False): [
            {"id": "x/no-repeat", "config": {"model_type": "known", "architectures": ["KnownLM"]}},
        ],
        ("abliterated", "abliterated", 2, True): [
            {"id": "y/abliterated", "config": {"model_type": "known", "architectures": ["KnownLM"]}},
        ],
    }
    admitted, _ = fac.choose_discoveries(results, set(), {"known"}, {"KnownLM"}, target=10)
    assert [repo for repo, _, _ in admitted] == ["y/abliterated"]
    assert "sibling of a shape already held" in admitted[0][2]


def test_the_per_query_cap_and_the_overall_target_both_hold():
    rows = [
        {"id": f"owner/model-{i}", "config": {"model_type": f"type{i}", "architectures": [f"A{i}"]}}
        for i in range(10)
    ]
    admitted, _ = fac.choose_discoveries({("legacy", "t", 3, True): rows}, set(), set(), set(), 10)
    assert len(admitted) == 3, "the per-query cap holds"
    admitted, _ = fac.choose_discoveries({("legacy", "t", 9, True): rows}, set(), set(), set(), 4)
    assert len(admitted) == 4, "the overall target holds"


def test_a_repository_already_in_the_corpus_is_not_rediscovered():
    rows = [{"id": "a/held", "config": {"model_type": "new_type", "architectures": ["New"]}}]
    admitted, _ = fac.choose_discoveries({("legacy", "t", 3, True): rows}, {"a/held"}, set(), set(), 9)
    assert admitted == []


def test_the_gated_retry_list_is_the_fourteen_that_refused():
    """It is a list of findings, not a wishlist: each one answered gated on 2026-10-08."""
    assert len(fac.GATED_RETRY) == 14
    assert len(set(fac.GATED_RETRY)) == 14
    assert all("/" in repo for repo in fac.GATED_RETRY)


def test_every_discovery_query_is_well_formed():
    terms = [term for _, term, _, _ in fac.DISCOVERY_QUERIES]
    assert len(terms) == len(set(terms)), "a duplicated term spends a request for nothing"
    for category, term, cap, repeat in fac.DISCOVERY_QUERIES:
        assert category in {"mechanism", "legacy", "abliterated", "moe", "non-decoder"}, category
        assert term.strip() == term and term, repr(term)
        assert 1 <= cap <= 10, (term, cap)
        assert isinstance(repeat, bool)


def test_a_reserved_category_cannot_be_squeezed_out_by_novelty_elsewhere():
    """The first live sweep admitted 210 models and not one of them was abliterated."""
    novel = [
        {"id": f"owner/novel-{i}", "config": {"model_type": f"t{i}", "architectures": [f"A{i}"]}}
        for i in range(30)
    ]
    abliterated = [
        {
            "id": f"huihui-ai/Qwen3-{i}B-abliterated",
            "config": {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        }
        for i in range(5)
    ]
    results = {
        ("legacy", "lots-of-novelty", 30, True): novel,
        ("abliterated", "abliterated", 5, True): abliterated,
    }
    admitted, _ = fac.choose_discoveries(
        results, set(), {"qwen3"}, {"Qwen3ForCausalLM"}, target=12, floors={"abliterated": 4}
    )
    categories = [category for _, category, _ in admitted]
    assert categories.count("abliterated") == 4, categories
    assert len(admitted) == 12


def test_a_real_checkpoint_outranks_a_test_fixture_within_a_query():
    """A library's two-layer fixture outranks the real model on downloads, and it should not."""
    rows = [
        {"id": "optimum-intel-internal-testing/tiny-random-StableLmForCausalLM",
         "config": {"model_type": "stablelm", "architectures": ["StableLmForCausalLM"]}},
        {"id": "stabilityai/stablelm-3b-4e1t",
         "config": {"model_type": "stablelm", "architectures": ["StableLmForCausalLM"]}},
    ]
    assert fac.rank_rows(rows)[0]["id"] == "stabilityai/stablelm-3b-4e1t"
    admitted, _ = fac.choose_discoveries(
        {("legacy", "stablelm", 1, True): rows}, set(), set(), set(), 5, floors={}
    )
    assert admitted[0][0] == "stabilityai/stablelm-3b-4e1t"


@pytest.mark.parametrize(
    "repo",
    [
        "optimum-intel-internal-testing/tiny-random-jais",
        "hf-tiny-v2/tiny-random-PhimoeModel",
        "trl-internal-testing/tiny-DbrxForCausalLM",
        "explosion-testing/mpt-test",
        "ai21labs/Jamba-tiny-dev",
    ],
)
def test_a_synthetic_fixture_is_labelled_rather_than_rejected(repo):
    """For several architectures the fixture is the only ungated config on the Hub.

    Its config SHAPE is the real architecture's, so it earns a place. Its depth and width are
    two layers of noise, so it must never be counted as a published model.
    """
    assert fac.synthetic_reason(repo) is not None
    assert fac.repackaged_reason(repo) is None
    assert fac.harm_reason(repo) is None
    assert fac.discovery_candidate({"id": repo, "config": {"model_type": "x"}}) == (repo, None)


@pytest.mark.parametrize(
    "repo",
    ["stabilityai/stablelm-3b-4e1t", "EleutherAI/pythia-70m-deduped", "tiiuae/falcon-7b"],
)
def test_a_real_small_model_is_not_mistaken_for_a_fixture(repo):
    assert fac.synthetic_reason(repo) is None


def test_an_image_classifier_matched_by_an_owner_name_is_skipped():
    """A sweep for "falcon" returns `Falconsai/nsfw_image_detection`, which is a ViT."""
    row = {"id": "Falconsai/nsfw_image_detection", "pipeline_tag": "image-classification",
           "config": {"model_type": "vit"}}
    repo, reason = fac.discovery_candidate(row, "legacy")
    assert repo is None
    assert "not a language model" in reason


def test_the_non_decoder_category_is_exempt_from_that_filter():
    """A non-decoder is the point of that category, so the filter must not eat it."""
    row = {"id": "microsoft/deberta-v3-base", "pipeline_tag": "fill-mask",
           "config": {"model_type": "deberta-v2"}}
    assert fac.discovery_candidate(row, "non-decoder")[0] == "microsoft/deberta-v3-base"
    row = {"id": "x/y", "pipeline_tag": "image-classification", "config": {"model_type": "vit"}}
    assert fac.discovery_candidate(row, "non-decoder")[0] == "x/y"
    assert fac.discovery_candidate(row, "mechanism")[0] is None


@pytest.mark.parametrize(
    ("problems", "missing", "status"),
    [
        ([], [], "complete"),
        (["no safetensors shard to take a header from", "no tensor inventory was obtained"], [], "config-only"),
        (["no tensor inventory was obtained"], [], "config-only"),
        (["tensor inventory covers 1 of 39 shards: no index"], [], "partial"),
        (["no tensor inventory was obtained", "generation_config.json: HTTP 500"], [], "partial"),
        ([], ["config.json"], "partial"),
        (["no tensor inventory was obtained"], ["config.json"], "partial"),
    ],
)
def test_a_checkpoint_older_than_safetensors_is_not_a_failed_fetch(problems, missing, status):
    """Nothing was missed: the repository has no safetensors and this tool will not read a .bin.

    Calling that partial tells a reader the fetch went wrong when it went perfectly, and it
    hides the 7 records where one shard of 39 genuinely IS a hole.
    """
    assert fac.classify_status(problems, missing) == status


@pytest.mark.parametrize(
    ("config", "dimension", "value"),
    [
        # Every one of these was found by looking at what the resolver missed on a live sweep,
        # and each is named here with the family that spells it that way.
        ({"n_layer": 24, "n_embed": 1024}, "width", 1024),          # BLOOM
        ({"num_transformer_layers": 28, "model_dim": 2048}, "depth", 28),   # OpenELM
        ({"num_transformer_layers": 28, "num_query_heads": 16}, "heads", 16),  # OpenELM
        ({"num_transformer_layers": 28, "num_gqa_groups": 4}, "kv_heads", 4),  # OpenELM
        ({"num_blocks": 32, "embedding_dim": 4096}, "depth", 32),   # xLSTM
        ({"depth": 40, "hidden_size": 1024}, "depth", 40),          # Motif vision encoder
        ({"num_decoder_layers": 12, "d_model": 768}, "depth", 12),  # T5 and LongT5
        ({"n_layers": 40, "d_model": 6144}, "depth", 40),           # DBRX, MPT, OLMo, LLaDA
        ({"num_hidden_layers": 4, "dense_intermediate_size": 11008}, "ffn", 11008),
    ],
)
def test_every_alias_in_the_tables_was_observed_somewhere(config, dimension, value):
    assert fac.structural_profile(config)["resolved"][dimension]["value"] == value


def test_bloom_and_gpt2_do_not_share_a_width_spelling():
    """`n_embed` and `n_embd` differ by one letter and by one family. Both are live."""
    assert fac.structural_profile({"n_layer": 1, "n_embd": 768})["resolved"]["width"]["key"] == "n_embd"
    assert fac.structural_profile({"n_layer": 1, "n_embed": 1024})["resolved"]["width"]["key"] == "n_embed"


class _FakeHeaders:
    def __init__(self, value):
        self._value = value

    def get(self, key):
        return self._value if key == "Retry-After" else None


class _FakeHTTPError:
    """A stand-in for `HTTPError`, which holds an open file object and leaks one per test.

    `_retry_after` reads nothing but `.headers`, so the stub is the honest fixture here.
    """

    def __init__(self, code, retry_after=None):
        self.code = code
        self.headers = _FakeHeaders(retry_after)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("30", 30.0), ("0.5", 0.5), (None, None), ("Wed, 21 Oct 2026 07:28:00 GMT", None), ("", None)],
)
def test_a_retry_after_header_is_read_when_the_host_sends_one(raw, expected):
    """The host knows its own limit better than our exponential guess does."""
    assert fac.Fetcher._retry_after(_FakeHTTPError(429, raw)) == expected


def test_a_rate_limit_holds_every_worker_not_just_the_unlucky_one():
    """Measured the hard way: 120 rate limits turned 32 reachable repositories unreachable.

    A 429 is a fact about the conversation with the host, so a per-URL backoff lets the other
    three workers carry on earning more of them.
    """
    fetcher = fac.Fetcher(
        fac.Budget(deadline=float("inf"), byte_budget=1), Path("/nonexistent"), lambda _m: None
    )
    assert fetcher._cooldown_until == 0.0
    fetcher._hold_everyone(12.0, "test")
    assert fetcher._cooldown_until > time.monotonic() + 11
    assert fetcher.rate_limited == 1
    # A second, shorter hold must not shorten the first one.
    before = fetcher._cooldown_until
    fetcher._hold_everyone(1.0, "test")
    assert fetcher._cooldown_until == before


def test_a_cooldown_is_bounded_so_a_hostile_header_cannot_stall_the_run():
    fetcher = fac.Fetcher(
        fac.Budget(deadline=float("inf"), byte_budget=1), Path("/nonexistent"), lambda _m: None
    )
    fetcher._hold_everyone(10**9, "a header that says come back next year")
    assert fetcher._cooldown_until <= time.monotonic() + fac.MAX_COOLDOWN_S + 1


# ── which indices a folded pattern covers ────────────────────────────────────────────
#
# ADDED 2026-10-08, after the omission it closes cost four wrong answers to one question.
#
# Folding 62 layer names into `model.layers.{i}.eh_proj.weight` is right, and recording only HOW
# MANY folded in threw away the thing that tells a decoder layer from a trailing prediction head.
# Both look like `model.layers.{i}.something`. The head is the one covering a single index above
# the declared depth, and nothing in the record said so.
#
# Three readers then produced 2, 25, 15 and 4 for the same question. Two wrote patterns against the
# folded text and measured the storage format. One expanded the templates across every index, so a
# head at block 61 appeared on all 62, and validated a detector against ground truth computed the
# same wrong way: two instruments agreed because they shared an assumption nothing had written
# down. This field makes the question a lookup.

def test_a_decoder_stack_reads_as_a_contiguous_run_from_zero():
    got = fac.summarise_indices(set(range(61)))
    assert got == {"count": 61, "min": 0, "max": 60, "contiguous": True}


def test_a_prediction_head_reads_as_one_index_above_the_stack():
    """THE CASE THE FIELD EXISTS FOR, in DeepSeek V3's real shape."""
    got = fac.summarise_indices({61})
    assert got == {"count": 1, "min": 61, "max": 61, "contiguous": False}


def test_three_prediction_layers_read_as_three_indices_above_the_stack():
    """Step 5 and Step 3.5 carry three, so one is not the only shape to handle."""
    got = fac.summarise_indices({92, 93, 94})
    assert got["count"] == 3
    assert (got["min"], got["max"]) == (92, 94)
    assert got["contiguous"] is False, "a run that does not start at zero is not a stack"


def test_a_gap_inside_a_run_is_not_contiguous():
    """A pattern present on some blocks and absent on others is a real shape: Nemotron-H carries
    attention on 4 of 52. It must not read as a stack.
    """
    assert fac.summarise_indices({0, 1, 5, 6})["contiguous"] is False


def test_no_indices_at_all_is_recorded_rather_than_guessed():
    got = fac.summarise_indices(set())
    assert got["count"] == 0
    assert got["min"] is None and got["max"] is None
    assert got["contiguous"] is False


def test_the_inventory_separates_a_head_from_the_stack_it_sits_in():
    """End to end on the shape that caused the confusion: one pattern covering 0 to 60 and two
    covering 61 alone, all three spelled `model.layers.{i}.*`.
    """
    names = {f"model.layers.{i}.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [8, 8]}
             for i in range(61)}
    names["model.layers.61.eh_proj.weight"] = {"dtype": "BF16", "shape": [8, 8]}
    names["model.layers.61.enorm.weight"] = {"dtype": "BF16", "shape": [8]}
    idx = fac.tensor_inventory(names)["pattern_layer_indices"]

    stack = idx["model.layers.{i}.self_attn.q_proj.weight"]
    assert stack["contiguous"] is True and stack["max"] == 60
    for head in ("model.layers.{i}.eh_proj.weight", "model.layers.{i}.enorm.weight"):
        assert idx[head] == {"count": 1, "min": 61, "max": 61, "contiguous": False}


def test_an_expert_index_is_not_mistaken_for_a_layer_index():
    """The first version of the depth reader took a max over every numeric component, so expert
    127 of a 48 layer model read as 128 deep. Attribution matters here for the same reason.
    """
    names = {f"model.layers.{i}.mlp.experts.{e}.down_proj.weight": {"dtype": "BF16", "shape": [4, 4]}
             for i in range(3) for e in range(8)}
    idx = fac.tensor_inventory(names)["pattern_layer_indices"]
    folded = "model.layers.{i}.mlp.experts.{e}.down_proj.weight"
    assert idx[folded] == {"count": 3, "min": 0, "max": 2, "contiguous": True}
