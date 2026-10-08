#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Build a metadata corpus of model architectures, without downloading a single weight.

WHY THIS EXISTS

A cross-architecture reader (the "model map") has so far been written against whatever
checkpoints happen to sit in a local cache, which is a handful of families and all of them
recent. A map built that way is a map of our cache. This collects the evidence instead: the
`config.json`, the tensor inventory and the prompt-format files for about a hundred published
architectures, normalises them into one shape, and records what vocabulary the real world
actually uses for each structural dimension.

THE TRICK THAT MAKES IT CHEAP

A safetensors file opens with an 8 byte little-endian length followed by a JSON header naming
every tensor, its shape and its dtype. So one HTTP range request over the first few hundred
kilobytes of a shard yields the complete tensor inventory of a trillion-parameter model. The
weights are never touched, by any route; see `FORBIDDEN_SUFFIXES`.

WHAT IT WILL NOT DO

- No authentication. No token is read, written, passed or looked for. A gated repository
  answers 401 or 403 and is recorded as gated, which is a finding about the ecosystem rather
  than a failure of the run.
- No weight bodies, no `tokenizer.json`, no `.bin`, `.gguf` or `.pth`. The guard is a hard
  assertion on every URL, not a convention.

RESUMABILITY

Every fetched payload lands in a local cache keyed by repository, revision and filename, so a
second run is nearly free and an interrupted run resumes. A model whose fetch did not complete
is written with `status: "partial"` and the reasons, never as complete; the next run retries
exactly the missing files.

    python tools/research/fetch_architecture_corpus.py --preflight-only
    python tools/research/fetch_architecture_corpus.py --out <dir>

Exit status is 0 when the run completed within its budgets and 1 when a budget was exhausted
or the pre-flight refused.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

SCHEMA_VERSION = 1

GALLERY_REPO = "rasbt/llm-architecture-gallery"
GALLERY_YML = f"https://raw.githubusercontent.com/{GALLERY_REPO}/main/models.yml"
GALLERY_COMMIT_API = f"https://api.github.com/repos/{GALLERY_REPO}/commits/main"
HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{name}"

# Identifies the tool and nothing about the operator. Do not put a contact address here.
USER_AGENT = "senbonzakura-architecture-corpus/1.0 (metadata only; no weights)"

DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "senbon-arch-corpus" / "raw"

# Small metadata files, in descending order of value. `tokenizer.json` is deliberately absent.
WANTED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    # A LoRA-only repository has no `config.json` at all; its architecture lives in the base
    # model it names. Reading this is how such a repository becomes a recorded fact rather
    # than an unexplained hole.
    "adapter_config.json",
)
REQUIRED_FILES = ("config.json",)

# A weight body must be unreachable by accident as well as by intent, so the check is on the
# URL at the moment of the request rather than on the caller's good manners.
FORBIDDEN_SUFFIXES = (".bin", ".gguf", ".pth", ".pt", ".ckpt", ".h5", ".msgpack", ".onnx")
FORBIDDEN_NAMES = ("tokenizer.json",)

# The permission this tool runs under covers public Hugging Face repositories and the gallery,
# and nothing else. That is a sentence in a brief until it is a check, so it is a check.
ALLOWED_HOSTS = frozenset({"huggingface.co", "raw.githubusercontent.com", "api.github.com"})

# Budgets. Every one of these is a refusal rather than a warning when it is reached.
MAX_INFLIGHT = 4
POLITE_DELAY_S = 0.15
REQUEST_TIMEOUT_S = 30
DEFAULT_DEADLINE_S = 45 * 60
DEFAULT_BYTE_BUDGET = 2 * 1024**3
MAX_SMALL_FILE_BYTES = 8 * 1024**2
# A shard index for a trillion-parameter MoE is tens of megabytes of pure tensor names. It is
# still metadata, so it gets its own, larger cap rather than being truncated into a hole.
MAX_INDEX_BYTES = 64 * 1024**2
MAX_HEADER_BYTES = 64 * 1024**2
MIN_FREE_DISK_BYTES = 4 * 1024**3
RETRY_STATUSES = (408, 425, 429, 500, 502, 503, 504)
MAX_ATTEMPTS = 6
# A rate limit is a property of the whole conversation with the host, not of one URL, so one
# 429 has to slow every worker down rather than just the unlucky one. Measured the hard way:
# eight runs in an hour earned 120 of them, and a four-attempt backoff turned each into an
# "unreachable" record for a repository that was perfectly reachable.
RATE_LIMIT_COOLDOWN_S = 20.0
MAX_COOLDOWN_S = 180.0
MAX_SIBLINGS_PER_FAMILY = 2
MAX_SIBLINGS_TOTAL = 40
DEFAULT_DISCOVERY_TARGET = 210

# ---------------------------------------------------------------------------
# Discovery: the catalogue sweep.
#
# The gallery is a hundred models somebody chose because they were interesting this year. It is
# the wrong shape for the question "could one reader read anything", because the architectures a
# reader fails on are the ones nobody writes about: the 2022 decoder with `n_layer`, the recurrent
# mixer that is not attention at all, the vendor whose expert keys nobody copied.
#
# Each row is (category, search term, how many to take, whether a repeat of an architecture we
# already hold is acceptable). Novelty of `model_type` always wins; the repeat allowance exists
# for the categories where a second instance is itself the evidence (an abliterated checkpoint
# is structurally its base, which is the finding).
# ---------------------------------------------------------------------------

DiscoveryQuery = tuple[str, str, int, bool]

DISCOVERY_QUERIES: tuple[DiscoveryQuery, ...] = (
    # 1. Mixers that are not attention. The corpus had 13 records marked
    #    `ssm-or-linear-attention` and every one of them was a Nemotron or a Qwen.
    ("mechanism", "rwkv", 5, True),
    ("mechanism", "mamba", 5, True),
    ("mechanism", "mamba2", 4, True),
    ("mechanism", "falcon-mamba", 3, True),
    ("mechanism", "xlstm", 4, True),
    ("mechanism", "jamba", 4, True),
    ("mechanism", "zamba", 4, True),
    ("mechanism", "recurrentgemma", 3, True),
    ("mechanism", "griffin", 2, True),
    ("mechanism", "hyena", 3, True),
    ("mechanism", "retnet", 2, True),
    ("mechanism", "hymba", 2, True),
    ("mechanism", "based", 2, False),
    ("mechanism", "gated deltanet", 3, True),
    ("mechanism", "delta-net", 3, True),
    ("mechanism", "fla-hub", 5, True),
    ("mechanism", "linear attention", 3, True),
    ("mechanism", "rnn language model", 2, False),
    ("mechanism", "liquid foundation", 3, True),
    ("mechanism", "bitnet", 3, True),
    ("mechanism", "diffusion language model", 3, True),
    ("mechanism", "llada", 3, True),
    ("mechanism", "dream diffusion llm", 2, True),
    # 2. The older and smaller decoders, where our resolvers are least exercised. Two of the
    #    map's own paths, `gpt_neox.layers` and `model.decoder.layers`, are reached by nothing
    #    in the local cache, and `opt` and `pythia` are what reach them.
    ("legacy", "gpt-neox", 4, True),
    ("legacy", "pythia", 4, True),
    ("legacy", "facebook/opt", 4, True),
    ("legacy", "bloom", 4, True),
    ("legacy", "bloomz", 3, True),
    ("legacy", "falcon", 4, True),
    ("legacy", "mpt", 4, True),
    ("legacy", "gpt-j", 3, True),
    ("legacy", "gpt-neo", 3, True),
    ("legacy", "xglm", 3, True),
    ("legacy", "codegen", 3, True),
    ("legacy", "persimmon", 2, True),
    ("legacy", "stablelm", 4, True),
    ("legacy", "starcoder2", 3, True),
    ("legacy", "starcoder", 3, True),
    ("legacy", "santacoder", 2, True),
    ("legacy", "dbrx", 2, True),
    ("legacy", "snowflake arctic", 2, True),
    ("legacy", "command-r", 3, True),
    ("legacy", "internlm", 4, True),
    ("legacy", "baichuan", 3, True),
    ("legacy", "01-ai/yi", 4, True),
    ("legacy", "chatglm", 4, True),
    ("legacy", "minicpm", 4, True),
    ("legacy", "orion-14b", 2, True),
    ("legacy", "skywork", 3, True),
    ("legacy", "telechat", 3, True),
    ("legacy", "deci", 2, True),
    ("legacy", "openelm", 2, True),
    ("legacy", "olmo-1b", 2, True),
    ("legacy", "cerebras", 2, True),
    ("legacy", "redpajama incite", 2, True),
    ("legacy", "tinyllama", 2, True),
    ("legacy", "qwen1.5", 3, True),
    ("legacy", "aquila", 2, True),
    ("legacy", "nanbeige", 2, True),
    ("legacy", "zhinao 360", 2, True),
    ("legacy", "exaone", 3, True),
    ("legacy", "plamo", 2, True),
    ("legacy", "rakuten", 2, True),
    ("legacy", "sarashina", 2, True),
    ("legacy", "gemma-2", 3, True),
    ("legacy", "granite-3", 3, True),
    ("legacy", "phi-4", 3, True),
    ("legacy", "jais", 2, True),
    ("legacy", "openchat", 2, True),
    ("legacy", "solar-10.7b", 2, True),
    ("legacy", "h2o-danube", 2, True),
    ("legacy", "xverse", 2, True),
    ("legacy", "moss", 2, False),
    ("legacy", "codellama", 2, True),
    ("legacy", "vicuna", 2, True),
    # 3. Abliterated and uncensored checkpoints. In scope on purpose: measuring somebody
    #    else's edit is a capability this project claims, and these are the evidence for the
    #    finding that an abliterated model is structurally identical to its base.
    ("abliterated", "abliterated", 8, True),
    ("abliterated", "uncensored", 6, True),
    ("abliterated", "orthogonalized", 4, True),
    ("abliterated", "heretic", 4, True),
    ("abliterated", "abliteration", 4, True),
    ("abliterated", "decensored", 3, True),
    # 4. More expert-key combinations, from vendors the gallery skipped.
    ("moe", "mixtral", 3, True),
    ("moe", "moe instruct", 4, True),
    ("moe", "qwen2-57b-a14b", 2, True),
    ("moe", "grok", 2, True),
    ("moe", "hunyuan", 3, True),
    ("moe", "ernie 4.5", 3, True),
    ("moe", "openmoe", 2, True),
    ("moe", "jetmoe", 2, True),
    ("moe", "deepseek-moe", 2, True),
    ("moe", "phimoe", 2, True),
    ("moe", "granitemoe", 3, True),
    ("moe", "aria moe", 2, True),
    ("moe", "flex moe", 2, False),
    # 5. A few encoder and encoder-decoder shapes, so "the map must refuse this" has evidence
    #    rather than an assumption.
    ("non-decoder", "t5 base", 2, True),
    ("non-decoder", "modernbert", 2, True),
    ("non-decoder", "deberta", 2, True),
)

# The 14 repositories that answered "gated" on 2026-10-08. Re-probed unauthenticated on every
# run, because a gate can open and a corpus that never asks again would never notice.
GATED_RETRY: tuple[str, ...] = (
    "CohereLabs/command-a-reasoning-08-2025",
    "CohereLabs/tiny-aya-base",
    "CohereLabs/tiny-aya-global",
    "Soofi-Project/Soofi-S-Base",
    "fdtn-ai/antares-1b",
    "fdtn-ai/antares-350m",
    "google/gemma-3-1b-it",
    "google/gemma-3-270m",
    "google/gemma-3-27b-it",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-Prompt-Guard-2-86M",
    "meta-llama/Meta-Llama-3-8B",
)

# Repackagers and alternative runtimes. Their repositories hold no `config.json` of the shape
# this corpus reads, so admitting them spends requests to record a hole.
REPACKAGED_MARKERS = (
    "gguf", "mlx", "onnx", "openvino", "coreml", "tflite", "ggml", "awq", "gptq", "exl2",
    "exl3", "autoround", "bnb-4bit", "-i1-", "w8a8", "int4", "int8", "-4bit", "-8bit",
)
REPACKAGER_OWNERS = (
    "bartowski", "mradermacher", "quantfactory", "tensorblock", "lmstudio-community",
    "thebloke", "unsloth", "second-state", "nold", "legraphista", "featherless-ai-quants",
    "qwp4w3hyb", "dranger003", "koboldcpp", "ikawrakow", "gaianet", "richardertel",
)

# Minimum slots held back for a category, so a category the operator asked for cannot be
# squeezed out by novelty elsewhere. The first sweep admitted 210 models and not one of them
# was abliterated, because 210 brand-new model types existed and the floor did not.
CATEGORY_FLOORS: dict[str, int] = {"abliterated": 14}

# Pipeline tags that are not a language model at all. The search term often matches an owner
# name rather than an architecture, which is how a sweep for "falcon" returns an image
# classifier. Exempt for the `non-decoder` category, where a non-decoder is the point.
OFF_TARGET_PIPELINES = frozenset({
    "image-classification", "object-detection", "image-segmentation", "depth-estimation",
    "automatic-speech-recognition", "audio-classification", "audio-to-audio",
    "text-to-image", "image-to-image", "image-to-video", "text-to-video",
    "video-classification", "zero-shot-image-classification", "image-feature-extraction",
    "time-series-forecasting", "tabular-classification", "reinforcement-learning",
})

# Synthetic fixtures: two-layer random checkpoints that exist so a library's test suite has
# something to load. They are NOT rejected, because for several architectures the fixture is
# the only ungated config on the Hub, and the config's SHAPE is the real architecture's. They
# are labelled, ranked below real checkpoints within a query, and counted separately, so no
# claim about depth or width is ever drawn from one by accident.
SYNTHETIC_MARKERS = ("tiny-random", "tiny_random", "tiny-dev", "tiny-model", "-8xtiny", "8xtiny")
SYNTHETIC_OWNER_MARKERS = ("internal-testing", "hf-tiny", "explosion-testing", "tiny-v2")

# What this run refuses to fetch on grounds of content rather than shape. The execution risk is
# nil here (every byte fetched is JSON this tool parses, nothing is ever imported, executed or
# unpickled, `trust_remote_code` is never set and no weight is downloaded), so this list is not
# a safety control. It is a decision about what belongs in an artefact of ours, and it is
# written down so the decision is reviewable rather than invisible.
HARM_MARKERS = (
    "malware", "ransomware", "trojan", "backdoored", "rootkit", "keylogger", "botnet",
    "phishing", "exploit-kit", "jailbreak-payload", "prompt-injection", "injection-kit",
    "cyberweapon", "bioweapon", "csam", "child-abuse", "stalkerware", "spyware",
    "credential-stealer", "infostealer", "worm-generator", "carding", "swatting",
)
CHAT_TEMPLATE_PREFIX_CHARS = 600

# ---------------------------------------------------------------------------
# Key vocabularies. These are the point of the exercise: every alias here was
# observed in a published config, and the map has to read all of them.
# ---------------------------------------------------------------------------

DEPTH_KEYS = (
    "num_hidden_layers", "n_layer", "n_layers", "num_layers", "depth",
    "num_blocks", "n_block", "num_transformer_layers", "num_decoder_layers",
)
WIDTH_KEYS = (
    "hidden_size", "n_embd", "d_model", "dim", "model_dim", "hidden_dim", "embed_dim",
    "embedding_dim",
    "n_embed",   # BLOOM, and it is NOT the same spelling as GPT-2's `n_embd`
)
HEAD_KEYS = (
    "num_attention_heads", "n_head", "n_heads", "num_heads", "attention_heads",
    "decoder_attention_heads",
    "num_query_heads",   # OpenELM, which names the query side rather than the attention
)
KV_HEAD_KEYS = (
    "num_key_value_heads", "num_kv_heads", "n_kv_heads", "multi_query_group_num",
    "num_key_value_groups", "attention_kv_heads",
    "num_gqa_groups",        # OpenELM
    "num_attention_groups",  # Nemotron Parse and one Motif config
)
FFN_KEYS = (
    "intermediate_size", "ffn_hidden_size", "n_inner", "ffn_dim", "ffn_intermediate_size",
    "mlp_hidden_size", "decoder_ffn_dim",
    "dense_intermediate_size",   # the dense half of two sparse configs
)
VOCAB_KEYS = ("vocab_size", "padded_vocab_size", "n_vocab")
NORM_EPS_KEYS = (
    "rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps", "norm_eps", "rms_norm_epsilon",
    "layernorm_epsilon", "ln_eps",
)
EXPERT_COUNT_KEYS = (
    "num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts",
    "num_routed_experts", "num_experts_total", "n_experts",
)
EXPERT_TOPK_KEYS = (
    "num_experts_per_tok", "num_experts_per_token", "moe_top_k", "topk", "top_k_experts",
    "num_selected_experts", "moe_k",
)
SHARED_EXPERT_KEYS = (
    "n_shared_experts", "num_shared_experts", "shared_expert_intermediate_size",
    "moe_shared_expert_intermediate_size", "n_shared_expert",
)
MOE_PLACEMENT_KEYS = (
    "moe_layer_freq", "first_k_dense_replace", "decoder_sparse_step", "moe_layer_interval",
    "mlp_only_layers", "moe_every_n_layers", "num_dense_layers",
)
MOE_FFN_KEYS = ("moe_intermediate_size", "expert_intermediate_size", "moe_ffn_hidden_size")
ROPE_BASE_KEYS = (
    "rope_theta", "rotary_emb_base", "rope_base", "rotary_base", "rope_local_base_freq",
)
ROPE_SHAPE_KEYS = (
    "rope_scaling", "rope_parameters", "partial_rotary_factor", "rotary_pct", "rotary_dim",
    "rope_type", "no_rope_layers", "nope_layer_interval", "rope_interleaved",
    "original_max_position_embeddings", "rope_scaling_factor",
)
POSITION_KEYS = (
    "max_position_embeddings", "position_embedding_type", "seq_length", "max_seq_len",
    "max_sequence_length", "n_positions", "max_window_layers",
)
ATTENTION_SHAPE_KEYS = (
    "sliding_window", "sliding_window_pattern", "use_sliding_window", "layer_types",
    "attention_chunk_size", "interleaved_sliding_window", "attn_type_list",
    "hybrid_override_pattern", "layers_block_type", "full_attention_idx", "attn_implementation",
    "attention_bias", "attention_dropout", "head_dim", "qk_rope_head_dim", "qk_nope_head_dim",
    "kv_lora_rank", "q_lora_rank", "v_head_dim", "use_qk_norm", "qk_layernorm", "attn_logit_softcapping",
    "query_pre_attn_scalar", "attention_multiplier", "linear_attn_config", "attention_type",
    "window_size", "global_attn_every_n_layers", "sink_size", "attention_sink",
)
DTYPE_KEYS = ("torch_dtype", "dtype", "params_dtype", "compute_dtype")
NESTED_CONFIG_KEYS = (
    "text_config", "vision_config", "audio_config", "speech_config", "decoder", "encoder",
    "llm_config", "language_config", "thinker_config", "talker_config", "mm_config",
    "sub_configs", "perceiver_config", "projector_config", "vqgan_config", "codec_config",
)

# Which nested key holds the DECODER, when several of them declare a depth. Alphabetical order
# is not neutral: `audio_config` sorts before `text_config`, and two Gemma 4 records had their
# audio tower read as the language model because of it.
DECODER_KEY_PREFERENCE = (
    "text_config", "llm_config", "language_config", "decoder", "thinker_config",
    "talker_config", "speech_config", "audio_config", "encoder",
)
NORM_PLACEMENT_KEYS = (
    "norm_type", "normalization", "use_sandwich_norm", "sandwich_norm", "post_attention_norm",
    "pre_feedforward_layernorm", "use_qk_norm", "residual_multiplier", "embedding_multiplier",
    "logits_scaling", "mup_width_multiplier", "final_logit_softcapping", "logit_scale",
)
SSM_KEYS = (
    "mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_n_heads", "mamba_n_groups",
    "ssm_cfg", "state_size", "conv_kernel", "linear_conv_kernel_dim", "d_inner",
    "mamba_chunk_size", "use_mamba_kernels", "linear_num_value_heads", "linear_key_head_dim",
)

DIMENSIONS: dict[str, tuple[str, ...]] = {
    "depth": DEPTH_KEYS,
    "width": WIDTH_KEYS,
    "heads": HEAD_KEYS,
    "kv_heads": KV_HEAD_KEYS,
    "ffn": FFN_KEYS,
    "vocab": VOCAB_KEYS,
    "norm_eps": NORM_EPS_KEYS,
    "expert_count": EXPERT_COUNT_KEYS,
    "expert_topk": EXPERT_TOPK_KEYS,
    "shared_expert": SHARED_EXPERT_KEYS,
    "moe_placement": MOE_PLACEMENT_KEYS,
    "moe_ffn": MOE_FFN_KEYS,
    "rope_base": ROPE_BASE_KEYS,
    "rope_shape": ROPE_SHAPE_KEYS,
    "position": POSITION_KEYS,
    "attention_shape": ATTENTION_SHAPE_KEYS,
    "dtype": DTYPE_KEYS,
    "norm_placement": NORM_PLACEMENT_KEYS,
    "ssm": SSM_KEYS,
}

_LAYER_INDEX = re.compile(r"(?<=\.)\d+(?=\.)|(?<=\.)\d+$")
_EXPERT_INDEX = re.compile(r"(?<=experts\.)\d+")


class CorpusError(RuntimeError):
    """Raised for a condition the run cannot continue through."""


class BudgetExhaustedError(CorpusError):
    """Raised when the byte budget or the wall-clock deadline is reached."""


# ---------------------------------------------------------------------------
# Pure helpers. Everything below this line up to the fetch layer is testable
# without a socket, and `tests/test_the_architecture_corpus_normalises.py`
# does exactly that.
# ---------------------------------------------------------------------------


def is_forbidden(name: str) -> bool:
    """True when a filename names weights, a tokenizer blob, or anything else off limits.

    The check is on the path component only, so a query string cannot smuggle a suffix past it.
    """
    path = urllib.parse.urlsplit(name).path
    base = path.rsplit("/", 1)[-1].lower()
    if base in FORBIDDEN_NAMES:
        return True
    if base.endswith(".safetensors"):
        # A body is forbidden; a range request for the header is the whole point, and the
        # caller that wants one goes through `fetch_safetensors_header`, which says so.
        return False
    return any(base.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)


def check_url(url: str) -> str:
    """Return the URL if it is one this tool is allowed to open, or raise.

    Two separate questions, both answered here: is the scheme https on a host the permission
    covers, and does the path name something we have promised never to fetch.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise CorpusError(f"refusing a non-https URL: {url}")
    if parts.hostname not in ALLOWED_HOSTS:
        raise CorpusError(f"refusing host {parts.hostname!r}, which is outside the allowed set")
    if is_forbidden(url):
        raise CorpusError(f"refusing a forbidden URL: {url}")
    return url


def safetensors_header_length(prefix: bytes) -> int:
    """Read the 8 byte little-endian JSON header length that opens a safetensors file.

    Validates rather than trusts: a truncated prefix, a zero length or a length beyond
    `MAX_HEADER_BYTES` is a refusal, because the alternative is a range request for an
    arbitrary number of bytes chosen by a remote file.
    """
    if len(prefix) < 8:
        raise CorpusError(f"safetensors prefix is {len(prefix)} bytes, need 8")
    length = int.from_bytes(prefix[:8], "little", signed=False)
    if length == 0:
        raise CorpusError("safetensors header length is zero")
    if length > MAX_HEADER_BYTES:
        raise CorpusError(
            f"safetensors header claims {length} bytes, over the {MAX_HEADER_BYTES} cap"
        )
    return length


def parse_safetensors_header(blob: bytes) -> dict[str, Any]:
    """Parse the header JSON into `{tensor_name: {dtype, shape}}` plus the metadata block."""
    try:
        raw = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"safetensors header is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CorpusError(f"safetensors header is {type(raw).__name__}, expected an object")
    meta = raw.pop("__metadata__", None)
    tensors: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise CorpusError(f"tensor {name!r} has a {type(spec).__name__} spec")
        tensors[name] = {"dtype": spec.get("dtype"), "shape": spec.get("shape")}
    return {"tensors": tensors, "metadata": meta if isinstance(meta, dict) else None}


def fold_tensor_name(name: str) -> str:
    """Collapse layer and expert indices so 48 layers become one pattern.

    `model.layers.31.mlp.experts.7.down_proj.weight` folds to
    `model.layers.{i}.mlp.experts.{e}.down_proj.weight`. Storing 30,000 names per model is
    storing the same six facts a thousand times; the pattern set is the fact.
    """
    return _LAYER_INDEX.sub("{i}", _EXPERT_INDEX.sub("{e}", name))


def index_counts_by_stem(name: str) -> tuple[dict[str, int], dict[str, int]]:
    """Count indexed positions per stack, separating layer indices from expert indices.

    A single maximum over every numeric component in a name is the wrong instrument and the
    first version of this file used it: in `model.layers.3.mlp.experts.127.up_proj.weight` the
    largest number is the expert, so a 48 layer MoE read as 128 layers deep. The index has to
    be attributed to the stack it indexes, which is the text in front of it.

    Returns (layers by stem, experts by stem), each a count rather than a maximum index.
    """
    expert_folded = _EXPERT_INDEX.sub("{e}", name)
    layers: dict[str, int] = {}
    experts: dict[str, int] = {}
    for match in _LAYER_INDEX.finditer(expert_folded):
        # The stem is folded as well, because a nested module list indexes inside an already
        # indexed layer. Without this, one checkpoint reported 363 separate stacks: the same
        # inner list once per outer layer.
        stem = _LAYER_INDEX.sub("{i}", expert_folded[: match.start()]).rstrip(".")
        layers[stem] = max(layers.get(stem, 0), int(match.group(0)) + 1)
    for match in _EXPERT_INDEX.finditer(name):
        # The stem is folded too, or every layer contributes its own expert stack and the
        # counts come back keyed by `model.layers.0.mlp.experts`, once per layer.
        stem = _LAYER_INDEX.sub("{i}", name[: match.start()]).rstrip(".")
        experts[stem] = max(experts.get(stem, 0), int(match.group(0)) + 1)
    return layers, experts


def summarise_indices(values: set[int]) -> dict[str, Any]:
    """Which layer indices a folded pattern covers, compactly.

    WHY THIS FIELD EXISTS, AND IT IS THE MOST EXPENSIVE OMISSION IN THIS FILE SO FAR.

    Folding `model.layers.0.eh_proj` through `model.layers.61.eh_proj` into one pattern is right:
    the pattern set is the fact and storing 30,000 names is storing six facts a thousand times.
    But the first version recorded only HOW MANY names folded in, and that threw away the one
    thing a reader needs to tell a decoder layer from a trailing prediction head. Both look like
    `model.layers.{i}.something`; the head is the one covering a single index above the declared
    depth.

    The cost was not theoretical. Four separate attempts were made to count models whose
    prediction head sits inside the decoder's own stack, by three different readers, and they
    returned 2, 25, 15 and 4. Two of those wrote patterns against the folded text and measured the
    STORAGE FORMAT. One expanded the templates across every index, so a head at block 61 appeared
    on all 62 blocks, and then validated a detector against ground truth computed the same wrong
    way: two instruments agreed because they shared an assumption this file had never written
    down. A schema that records the indices makes all four of those readings unnecessary and the
    question a lookup.

    `contiguous` is the discriminator worth having precomputed: a decoder stack's pattern covers
    0 to n-1 with no gaps, and anything else wants a human looking at it.
    """
    if not values:
        return {"count": 0, "min": None, "max": None, "contiguous": False}
    lo, hi = min(values), max(values)
    return {
        "count": len(values),
        "min": lo,
        "max": hi,
        # True only for a gapless run starting at zero, which is what an ordinary stack looks like.
        "contiguous": len(values) == (hi - lo + 1) and lo == 0,
    }


def tensor_inventory(names_to_spec: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarise a tensor inventory: folded patterns, dtypes, prefixes and per-stack depths."""
    patterns: dict[str, int] = {}
    #: pattern -> the set of layer indices whose names folded into it. See `summarise_indices`.
    pattern_indices: dict[str, set[int]] = {}
    dtypes: dict[str, int] = {}
    prefixes: dict[str, int] = {}
    layer_counts: dict[str, int] = {}
    expert_counts: dict[str, int] = {}
    for name, spec in names_to_spec.items():
        folded = fold_tensor_name(name)
        patterns[folded] = patterns.get(folded, 0) + 1
        # The indices this name contributes, attributed to the pattern rather than to a stem, so a
        # reader can ask "which blocks does this pattern cover" without re-deriving it.
        for match in _LAYER_INDEX.finditer(_EXPERT_INDEX.sub("{e}", name)):
            pattern_indices.setdefault(folded, set()).add(int(match.group(0)))
        dtype = (spec or {}).get("dtype")
        if dtype:
            dtypes[str(dtype)] = dtypes.get(str(dtype), 0) + 1
        prefixes[name.split(".", 1)[0]] = prefixes.get(name.split(".", 1)[0], 0) + 1
        layers, experts = index_counts_by_stem(name)
        for stem, count in layers.items():
            layer_counts[stem] = max(layer_counts.get(stem, 0), count)
        for stem, count in experts.items():
            expert_counts[stem] = max(expert_counts.get(stem, 0), count)
    deepest = max(layer_counts.items(), key=lambda kv: (kv[1], kv[0])) if layer_counts else None
    return {
        "tensor_count": len(names_to_spec),
        "pattern_count": len(patterns),
        "patterns": dict(sorted(patterns.items())),
        # WHICH indices each pattern covers, not only how many names folded in. The field that
        # makes "is this trailing block a prediction head" a lookup rather than a fourth guess.
        "pattern_layer_indices": {
            k: summarise_indices(v) for k, v in sorted(pattern_indices.items())},
        "dtypes": dict(sorted(dtypes.items())),
        "prefixes": dict(sorted(prefixes.items())),
        "layer_counts_by_stem": dict(sorted(layer_counts.items())),
        "expert_counts_by_stem": dict(sorted(expert_counts.items())),
        "deepest_stack": deepest[0] if deepest else None,
        "deepest_stack_layers": deepest[1] if deepest else None,
    }


def resolve_key(config: dict[str, Any], candidates: tuple[str, ...]) -> tuple[str | None, Any]:
    """Return the first alias present in a config and its value, in candidate order.

    Order is the authority. Several configs carry both `num_hidden_layers` and a legacy
    `n_layer`, and when they disagree the modern spelling is the one `transformers` reads.
    """
    for key in candidates:
        if key in config:
            return key, config[key]
    return None, None


def observed_keys(config: dict[str, Any], candidates: tuple[str, ...]) -> list[str]:
    """Every alias from a vocabulary that this config actually carries."""
    return sorted(key for key in candidates if key in config)


def nested_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Find sub-configs one level down, by known key and by shape.

    Shape detection matters: a multimodal config may nest its decoder under a key nobody has
    seen before, and a map that only knows the named keys reads the wrapper's depth (absent)
    instead of the decoder's.
    """
    found: dict[str, dict[str, Any]] = {}
    for key, value in config.items():
        if not isinstance(value, dict):
            continue
        if key in NESTED_CONFIG_KEYS:
            found[key] = value
            continue
        looks_structural = any(
            alias in value for alias in DEPTH_KEYS + WIDTH_KEYS
        ) or "model_type" in value
        if looks_structural:
            found[key] = value
    return dict(sorted(found.items()))


def pick_decoder_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the path and the sub-config that actually describes the decoder stack.

    A top level that declares a depth is the decoder. Otherwise the deepest nested candidate
    that declares one wins, which is how a vision-language wrapper is read correctly.
    """
    if resolve_key(config, DEPTH_KEYS)[0] is not None:
        return "", config
    found = nested_configs(config)
    ordered = sorted(
        found,
        key=lambda key: (
            DECODER_KEY_PREFERENCE.index(key) if key in DECODER_KEY_PREFERENCE else len(DECODER_KEY_PREFERENCE),
            key,
        ),
    )
    for key in ordered:
        candidate = found[key]
        if resolve_key(candidate, DEPTH_KEYS)[0] is not None:
            return key, candidate
    for key in ordered:
        deeper_path, deeper = pick_decoder_config(found[key])
        if deeper_path and resolve_key(deeper, DEPTH_KEYS)[0] is not None:
            return f"{key}.{deeper_path}", deeper
    return "", config


def classify_mechanisms(config: dict[str, Any]) -> list[str]:
    """Name the mechanisms a config declares, from the evidence rather than from the model name."""
    marks: set[str] = set()
    heads = resolve_key(config, HEAD_KEYS)[1]
    kv_heads = resolve_key(config, KV_HEAD_KEYS)[1]
    if isinstance(heads, int) and isinstance(kv_heads, int):
        if kv_heads == 1:
            marks.add("mqa")
        elif kv_heads < heads:
            marks.add("gqa")
        else:
            marks.add("mha")
    elif isinstance(heads, int):
        marks.add("mha-or-unstated")
    if resolve_key(config, EXPERT_COUNT_KEYS)[0] is not None:
        marks.add("moe")
    if resolve_key(config, SHARED_EXPERT_KEYS)[0] is not None:
        marks.add("moe-shared-expert")
    if any(key in config for key in ("kv_lora_rank", "q_lora_rank", "qk_nope_head_dim")):
        marks.add("mla")
    if config.get("sliding_window") or config.get("use_sliding_window"):
        marks.add("sliding-window")
    if config.get("layer_types") or config.get("attn_type_list") or config.get("layers_block_type"):
        marks.add("per-layer-type-list")
    if config.get("hybrid_override_pattern"):
        marks.add("hybrid-pattern-string")
    if any(key in config for key in SSM_KEYS):
        marks.add("ssm-or-linear-attention")
    if config.get("no_rope_layers") or config.get("nope_layer_interval"):
        marks.add("nope-layers")
    if resolve_key(config, ROPE_BASE_KEYS)[0] is not None:
        marks.add("rope")
    if config.get("rope_scaling") or config.get("rope_parameters"):
        marks.add("rope-scaled")
    if config.get("use_qk_norm") or config.get("qk_layernorm"):
        marks.add("qk-norm")
    if config.get("tie_word_embeddings"):
        marks.add("tied-embeddings")
    if config.get("quantization_config"):
        marks.add("quantised-checkpoint")
    return sorted(marks)


def structural_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Normalise one config into the shape the map would have to read.

    Each dimension records the alias that was used as well as the value, because the whole
    question is whether one reader can cope with the spread of spellings.
    """
    decoder_path, decoder = pick_decoder_config(config)
    resolved: dict[str, Any] = {}
    aliases: dict[str, list[str]] = {}
    for dimension, candidates in DIMENSIONS.items():
        key, value = resolve_key(decoder, candidates)
        resolved[dimension] = {"key": key, "value": value if _is_jsonable(value) else repr(value)}
        seen = observed_keys(decoder, candidates)
        if seen:
            aliases[dimension] = seen
    nested = nested_configs(config)
    return {
        "model_type": config.get("model_type") or decoder.get("model_type"),
        "architectures": config.get("architectures") or decoder.get("architectures"),
        "transformers_version": config.get("transformers_version"),
        "decoder_path": decoder_path,
        "decoder_is_nested": bool(decoder_path),
        "resolved": resolved,
        "aliases_present": aliases,
        "mechanisms": classify_mechanisms(decoder),
        "nested_config_keys": sorted(nested),
        "unknown_top_level_keys": sorted(
            key
            for key in config
            if key not in _ALL_KNOWN_KEYS and not isinstance(config[key], (dict, list))
        ),
        "top_level_keys": sorted(config),
    }


def _is_jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, list, dict, type(None)))


_ALL_KNOWN_KEYS = {key for keys in DIMENSIONS.values() for key in keys} | {
    "model_type", "architectures", "transformers_version", "tie_word_embeddings",
    "bos_token_id", "eos_token_id", "pad_token_id", "use_cache", "initializer_range",
    "hidden_act", "hidden_activation", "activation_function", "mlp_bias", "quantization_config",
    "auto_map", "_name_or_path", "tf_legacy_loss", "architecture",
}


def summarise_tokenizer_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep what answers a prompt-format question and drop the bulk.

    A chat template can run to tens of kilobytes of Jinja. The digest plus a prefix answers
    "is this the same template" and "what shape is it" without carrying a copy of somebody
    else's model card around.
    """
    template = raw.get("chat_template")
    template_summary: Any = None
    if isinstance(template, str):
        template_summary = {
            "kind": "string",
            "chars": len(template),
            "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "prefix": template[:CHAT_TEMPLATE_PREFIX_CHARS],
        }
    elif isinstance(template, list):
        template_summary = {
            "kind": "list",
            "entries": [str(item.get("name")) for item in template if isinstance(item, dict)],
        }
    added = raw.get("added_tokens_decoder")
    return {
        "tokenizer_class": raw.get("tokenizer_class"),
        "model_max_length": raw.get("model_max_length"),
        "bos_token": _token_text(raw.get("bos_token")),
        "eos_token": _token_text(raw.get("eos_token")),
        "pad_token": _token_text(raw.get("pad_token")),
        "unk_token": _token_text(raw.get("unk_token")),
        "add_bos_token": raw.get("add_bos_token"),
        "add_eos_token": raw.get("add_eos_token"),
        "clean_up_tokenization_spaces": raw.get("clean_up_tokenization_spaces"),
        "added_token_count": len(added) if isinstance(added, dict) else None,
        "chat_template": template_summary,
        "extra_special_tokens": sorted(raw.get("extra_special_tokens", {})) or None,
        "keys": sorted(raw),
    }


def _token_text(value: Any) -> Any:
    """Special tokens appear as a bare string or as an `AddedToken` object. Read both."""
    if isinstance(value, dict):
        return value.get("content")
    return value


def gallery_entries(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the gallery YAML into one record per model, keeping only what we will use.

    The summary, image and benchmark-score fields are the gallery author's prose and are left
    behind: this corpus holds facts we can re-derive, plus the licence, which travels so that
    nothing here is ever republished by accident.
    """
    entries: list[dict[str, Any]] = []
    for name, body in sorted((document or {}).items()):
        if not isinstance(body, dict):
            raise CorpusError(f"gallery entry {name!r} is {type(body).__name__}, expected a mapping")
        config_block = body.get("config") or {}
        tech = body.get("tech_report") or {}
        repo = (config_block.get("repo") or "").strip()
        entries.append(
            {
                "name": name,
                "repo": repo or None,
                "company": body.get("company"),
                "date": body.get("date"),
                "scale": body.get("scale"),
                "context_tokens": body.get("context_tokens"),
                "vocab_size": body.get("vocab_size"),
                "license_name": body.get("license_name"),
                "license_url": body.get("license_url"),
                "decoder_type": body.get("decoder_type"),
                "attention": body.get("attention"),
                "layer_mix": body.get("layer_mix"),
                "kv_cache_per_token_bf16": body.get("kv_cache_per_token_bf16"),
                "related_concepts": body.get("related_concepts") or [],
                "config_url": config_block.get("url"),
                "tech_report_url": tech.get("url") if isinstance(tech, dict) else None,
            }
        )
    return entries


def family_prefix(repo: str) -> str:
    """The stem a family's siblings share, used to search the Hub for the ones the gallery omits.

    `Qwen/Qwen3-235B-A22B-Instruct` gives `Qwen/Qwen3`. Crude on purpose: it is a search term,
    and every candidate it produces is checked against the corpus before it is taken.
    """
    owner, _, name = repo.partition("/")
    stem = re.split(r"[-_.]", name)[0] if name else ""
    return f"{owner}/{stem}" if stem else owner


def repackaged_reason(repo: str) -> str | None:
    """Say why a repository is a repackaging of somebody else's weights, or None.

    These are skipped for shape, not for content: a GGUF or AWQ republish carries no
    `config.json` in the form this corpus reads, so fetching one spends four requests to
    record a hole.
    """
    lowered = repo.lower()
    owner = lowered.partition("/")[0]
    if owner in REPACKAGER_OWNERS:
        return f"repackager account ({owner})"
    for marker in REPACKAGED_MARKERS:
        if marker in lowered:
            return f"repackaged format ({marker})"
    return None


def harm_reason(repo: str, tags: list[str] | None = None) -> str | None:
    """Say why a repository is being skipped on content grounds, or None.

    The test is whether the repository presents itself as harm: malware, a trojaned model, a
    jailbreak payload, an injection kit. An abliterated or uncensored general model is not in
    that category and is wanted, because measuring somebody else's refusal edit is the point.
    """
    haystack = " ".join([repo.lower(), *[str(tag).lower() for tag in tags or []]])
    for marker in HARM_MARKERS:
        if marker in haystack:
            return f"presents as harm ({marker})"
    return None


def synthetic_reason(repo: str) -> str | None:
    """Say why a repository looks like a library's test fixture, or None. A label, not a veto."""
    lowered = repo.lower()
    owner = lowered.partition("/")[0]
    name = lowered.partition("/")[2]
    for marker in SYNTHETIC_OWNER_MARKERS:
        if marker in owner:
            return f"test-fixture account ({owner})"
    for marker in SYNTHETIC_MARKERS:
        if marker in lowered:
            return f"synthetic fixture name ({marker})"
    if name.startswith(("tiny-", "tiny_")):
        return "synthetic fixture name (tiny-)"
    return None


def off_target_reason(row: dict[str, Any], category: str) -> str | None:
    """Say why a search result is not a language model, or None."""
    if category == "non-decoder":
        return None
    pipeline = (row.get("pipeline_tag") or "").strip()
    if pipeline in OFF_TARGET_PIPELINES:
        return f"not a language model (pipeline_tag {pipeline})"
    return None


def discovery_candidate(row: dict[str, Any], category: str = "") -> tuple[str | None, str | None]:
    """Judge one search result. Returns (repo to take, or None) and (reason it was skipped).

    Exactly one of the two is set, so a caller cannot silently drop a row: everything either
    enters the corpus or enters the skipped list with a reason.
    """
    repo = (row.get("id") or "").strip()
    if not repo:
        return None, "search result carried no repository id"
    if row.get("private"):
        return None, "private"
    if row.get("gated"):
        return None, f"gated ({row.get('gated')})"
    reason = (
        repackaged_reason(repo)
        or harm_reason(repo, row.get("tags"))
        or off_target_reason(row, category)
    )
    if reason:
        return None, reason
    if not ((row.get("config") or {}).get("model_type") or "").strip():
        return None, "the Hub reports no model_type, so there is no transformers config to read"
    return repo, None


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order one query's results: real checkpoints first, then synthetic fixtures.

    Within each group the Hub's own order is kept, which is downloads descending. Without
    this, a library's two-layer test fixture outranks the real checkpoint it was built from,
    because CI downloads it more often than people download the model.
    """
    real = [row for row in rows if not synthetic_reason(row.get("id") or "")]
    fixtures = [row for row in rows if synthetic_reason(row.get("id") or "")]
    return real + fixtures


def choose_discoveries(
    results: dict[DiscoveryQuery, list[dict[str, Any]]],
    known_repos: set[str],
    known_types: set[str],
    known_architectures: set[str],
    target: int,
    floors: dict[str, int] | None = None,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Pick which discovered repositories enter the corpus, and record every rejection.

    The rule, stated so it can be argued with:

    1. Some slots are reserved per category, because a category the operator asked for must
       not be squeezed out by novelty found elsewhere. The first sweep admitted 210 models and
       not one was abliterated, since 210 brand-new model types existed and the floor did not.
    2. In the open slots, a candidate whose `model_type` or whose `architectures` entry is new
       to the corpus is admitted. Novelty is the point of the sweep.
    3. A candidate that adds nothing new is admitted only where its category said a repeat is
       itself evidence (an abliterated checkpoint is structurally its base, which is the
       finding), and only while that query's cap is unmet.
    4. Everything stops at `target`.

    Within a query, real checkpoints outrank synthetic test fixtures, and otherwise the Hub's
    own downloads-descending order holds, so the pick is reproducible given the same results.

    Returns (admitted as (repo, category, why), skipped as (repo, reason)).
    """
    floors = CATEGORY_FLOORS if floors is None else floors
    admitted: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    state = {
        "repos": set(known_repos),
        "types": set(known_types),
        "architectures": set(known_architectures),
    }
    per_term: dict[str, int] = {}
    ordered = {query: rank_rows(rows) for query, rows in results.items()}

    def sweep(limit: int, *, novel_only: bool, only: set[str] | None, collect_skips: bool) -> None:
        for query, rows in ordered.items():
            category, term, cap, allow_repeat = query
            if only is not None and category not in only:
                continue
            for row in rows:
                if len(admitted) >= limit or per_term.get(term, 0) >= cap:
                    break
                repo, reason = discovery_candidate(row, category)
                if repo is None:
                    if collect_skips and reason:
                        skipped.append(((row.get("id") or "?"), reason))
                    continue
                if repo in state["repos"]:
                    continue
                config = row.get("config") or {}
                model_type = str(config.get("model_type"))
                archs = [str(arch) for arch in config.get("architectures") or []]
                new_type = model_type not in state["types"]
                new_arch = any(arch not in state["architectures"] for arch in archs)
                stale = not (new_type or new_arch)
                if stale and (novel_only or not allow_repeat):
                    continue
                why = (
                    f"new model_type {model_type} [{term}]"
                    if new_type
                    else f"new architecture {','.join(archs)} [{term}]"
                    if new_arch
                    else f"{category} sibling of a shape already held [{term}]"
                )
                admitted.append((repo, category, why))
                state["repos"].add(repo)
                state["types"].add(model_type)
                state["architectures"].update(archs)
                per_term[term] = per_term.get(term, 0) + 1

    # The floors are held back from the open target, then filled from their own categories.
    reserved = min(sum(floors.values()), max(0, target // 3))
    sweep(max(0, target - reserved), novel_only=True, only=None, collect_skips=True)
    for category, floor in sorted(floors.items()):
        room = min(target, len(admitted) + floor)
        sweep(room, novel_only=True, only={category}, collect_skips=False)
        sweep(room, novel_only=False, only={category}, collect_skips=False)
    sweep(target, novel_only=True, only=None, collect_skips=False)
    sweep(target, novel_only=False, only=None, collect_skips=False)
    return admitted, _dedupe_skips(skipped)


def _dedupe_skips(skipped: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for repo, reason in skipped:
        seen.setdefault(repo, reason)
    return sorted(seen.items())


def choose_siblings(
    candidates: list[dict[str, Any]],
    known_repos: set[str],
    known_model_types: set[str],
    per_family: int = MAX_SIBLINGS_PER_FAMILY,
) -> list[str]:
    """Pick Hub siblings that would add structural vocabulary, not just another size.

    The rule, stated so it can be argued with: a sibling is taken only when its Hub-reported
    `model_type` is one the corpus does not already hold, it is public and not gated, and at
    most `per_family` are taken from any one family. A second checkpoint of an architecture we
    already have teaches the map nothing.
    """
    taken: list[str] = []
    seen_types = set(known_model_types)
    for item in candidates:
        if len(taken) >= per_family:
            break
        repo = item.get("id") or ""
        if not repo or repo in known_repos:
            continue
        if item.get("private") or item.get("gated"):
            continue
        model_type = ((item.get("config") or {}).get("model_type") or "").strip()
        if not model_type or model_type in seen_types:
            continue
        seen_types.add(model_type)
        taken.append(repo)
    return taken


# ---------------------------------------------------------------------------
# Fetch layer
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Budget:
    """Wall clock and bytes, both hard. Shared across threads, so every read takes the lock."""

    deadline: float
    byte_budget: int
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)
    bytes_used: int = 0
    requests: int = 0

    def charge(self, nbytes: int) -> None:
        with self._lock:
            self.bytes_used += nbytes
            self.requests += 1
            if self.bytes_used > self.byte_budget:
                raise BudgetExhaustedError(
                    f"byte budget exhausted: {self.bytes_used} of {self.byte_budget}"
                )

    def check_clock(self) -> None:
        if time.monotonic() > self.deadline:
            raise BudgetExhaustedError("wall-clock deadline reached")

    def remaining_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


class Fetcher:
    """HTTP with manners: bounded size, retry with backoff, a polite delay, and no credentials.

    Nothing in here reads an environment variable, a netrc or a token file. A gated repository
    is meant to answer 401 and that answer is the finding.
    """

    def __init__(self, budget: Budget, cache: pathlib.Path, log: Any) -> None:
        self.budget = budget
        self.cache = cache
        self.log = log
        self._throttle = threading.Lock()
        self._last_request = 0.0
        self._cooldown_until = 0.0
        self.rate_limited = 0

    def _pace(self) -> None:
        while True:
            with self._throttle:
                now = time.monotonic()
                wait = self._cooldown_until - now
                if wait <= 0:
                    gap = now - self._last_request
                    if gap < POLITE_DELAY_S:
                        time.sleep(POLITE_DELAY_S - gap)
                    self._last_request = time.monotonic()
                    return
            # Outside the lock, so every worker waits out the same cool-down in parallel
            # rather than queueing behind one another and each serving its own.
            time.sleep(min(wait, 5.0))

    def _hold_everyone(self, seconds: float, why: str) -> None:
        seconds = min(max(seconds, 1.0), MAX_COOLDOWN_S)
        with self._throttle:
            self.rate_limited += 1
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)
        self.log(f"  rate limited ({why}); holding every request for {seconds:.0f}s")

    @staticmethod
    def _retry_after(exc: urllib.error.HTTPError) -> float | None:
        """Read `Retry-After` when the host sends one. It knows better than our guess."""
        raw = exc.headers.get("Retry-After") if exc.headers else None
        if not raw:
            return None
        try:
            return float(str(raw).strip())
        except ValueError:
            return None

    def get(self, url: str, *, byte_range: tuple[int, int] | None = None, limit: int = MAX_SMALL_FILE_BYTES) -> bytes:
        """Fetch a URL, or raise. Never follows a path that would pull a weight body."""
        check_url(url)
        if byte_range is None and urllib.parse.urlsplit(url).path.lower().endswith(".safetensors"):
            raise CorpusError(f"refusing an unranged safetensors request: {url}")
        self.budget.check_clock()
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._pace()
            request = urllib.request.Request(   # noqa: S310 - check_url pinned the scheme and host
                url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            )
            if byte_range is not None:
                request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
            timeout = min(REQUEST_TIMEOUT_S, max(1.0, self.budget.remaining_s()))
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:   # noqa: S310
                    body = response.read(limit + 1)
                    if len(body) > limit:
                        raise CorpusError(f"{url} is over the {limit} byte cap for this kind of file")
                    self.budget.charge(len(body))
                    return body
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                    if exc.code == 429:
                        stated = self._retry_after(exc)
                        self._hold_everyone(
                            stated if stated is not None else RATE_LIMIT_COOLDOWN_S * attempt,
                            f"HTTP 429, Retry-After {stated if stated is not None else 'absent'}",
                        )
                        continue
                    delay = min(30.0, 2.0**attempt) + 0.3 * attempt
                    self.log(f"  retry {attempt}/{MAX_ATTEMPTS} after HTTP {exc.code} on {url} in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                raise CorpusError(f"HTTP {exc.code} on {url}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS:
                    delay = min(30.0, 2.0**attempt)
                    self.log(f"  retry {attempt}/{MAX_ATTEMPTS} after {exc!r} on {url} in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                raise CorpusError(f"{type(exc).__name__} on {url}: {exc}") from exc
        raise CorpusError(f"gave up on {url}: {last_error!r}")

    # -- cache ------------------------------------------------------------

    def cached_path(self, repo: str, revision: str, name: str) -> pathlib.Path:
        safe_repo = repo.replace("/", "__")
        return self.cache / safe_repo / revision[:12] / name.replace("/", "__")

    def cached_or_fetch(
        self, repo: str, revision: str, name: str, url: str, *, limit: int = MAX_SMALL_FILE_BYTES
    ) -> tuple[bytes, bool]:
        """Return the payload and whether it came from the cache.

        Written with rename-on-close, so an interrupted run never leaves a half file that a
        later run would read as complete.
        """
        path = self.cached_path(repo, revision, name)
        if path.exists():
            return path.read_bytes(), True
        body = self.get(url, limit=limit)
        write_atomic(path, body)
        return body, False


def write_atomic(path: pathlib.Path, body: bytes) -> None:
    """Write via a temporary file in the same directory and rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".part.{os.getpid()}.{threading.get_ident()}")
    try:
        tmp.write_bytes(body)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_json_atomic(path: pathlib.Path, payload: Any, *, gzip_over: int = 1024**2) -> int:
    """Serialise sorted, then write atomically, gzipping anything over a megabyte.

    Sorted keys because two runs on the same inputs have to produce byte-identical files.
    """
    blob = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if len(blob) > gzip_over:
        # mtime=0 so the same content gives the same bytes on every run.
        packed = gzip.compress(blob, mtime=0)
        write_atomic(path.with_name(path.name + ".gz"), packed)
        path.unlink(missing_ok=True)
        return len(packed)
    write_atomic(path, blob)
    path.with_name(path.name + ".gz").unlink(missing_ok=True)
    return len(blob)


# ---------------------------------------------------------------------------
# Per-model work
# ---------------------------------------------------------------------------


def hub_info(fetcher: Fetcher, repo: str) -> dict[str, Any]:
    """Ask the Hub what this repository is, through a cache that expires each day.

    The answer carries the revision sha, so it cannot be cached forever. It also costs a
    request per model per run, which is what earned 120 rate limits across a day of re-runs,
    so it is not re-asked within the same day either. The record's `fetched_at` says when.
    """
    url = f"{HF_API}/{urllib.parse.quote(repo)}"
    body, _ = fetcher.cached_or_fetch(
        repo, f"api-{time.strftime('%Y-%m-%d')}", "model-info.json", url
    )
    info = json.loads(body.decode("utf-8"))
    if not isinstance(info, dict):
        raise CorpusError(f"{url} returned {type(info).__name__}, expected an object")
    return info


def pick_safetensors_shard(siblings: list[str]) -> str | None:
    """Choose the one shard whose header is worth a range request."""
    shards = sorted(name for name in siblings if name.endswith(".safetensors") and "/" not in name)
    if not shards:
        shards = sorted(name for name in siblings if name.endswith(".safetensors"))
    if not shards:
        return None
    for name in shards:
        if name == "model.safetensors":
            return name
    return shards[0]


def fetch_safetensors_header(fetcher: Fetcher, repo: str, revision: str, shard: str) -> dict[str, Any]:
    """Two range requests: the 8 byte length, then exactly the header.

    Never a third, and never an unranged request. The length prefix is read first precisely so
    that no fixed guess is needed and no fallback to the body can exist.
    """
    url = HF_RESOLVE.format(repo=repo, rev=revision, name=urllib.parse.quote(shard))
    cache_key = f"{shard}.header.json"
    path = fetcher.cached_path(repo, revision, cache_key)
    if path.exists():
        header = parse_safetensors_header(path.read_bytes())
        header["from_cache"] = True
        header["shard"] = shard
        return header
    prefix = fetcher.get(url, byte_range=(0, 7), limit=64)
    length = safetensors_header_length(prefix)
    blob = fetcher.get(url, byte_range=(8, 8 + length - 1), limit=MAX_HEADER_BYTES)
    if len(blob) != length:
        raise CorpusError(f"header range gave {len(blob)} bytes, asked for {length}")
    write_atomic(path, blob)
    header = parse_safetensors_header(blob)
    header["from_cache"] = False
    header["shard"] = shard
    return header


# A checkpoint that predates safetensors has no header to range-request and no index to read,
# and this tool will not touch a `.bin` by any route. Nothing was missed and nothing failed, so
# calling that "partial" tells a reader the fetch went wrong when the fetch went perfectly. It
# gets its own status instead, and the record keeps `weight_formats` so the claim is checkable.
NO_TENSORS_PROBLEMS = frozenset({
    "no safetensors shard to take a header from",
    "no tensor inventory was obtained",
})


def classify_status(problems: list[str], missing_required: list[str]) -> str:
    """complete, config-only, or partial. The three are different things and were not before."""
    if missing_required:
        return "partial"
    if not problems:
        return "complete"
    if set(problems) <= NO_TENSORS_PROBLEMS:
        return "config-only"
    return "partial"


def build_record(
    fetcher: Fetcher,
    *,
    gallery: dict[str, Any] | None,
    repo: str,
    source: str,
    want_header: bool,
    log: Any,
    selection: str | None = None,
) -> dict[str, Any]:
    """Fetch and normalise one model. Returns a record; never raises for a remote refusal."""
    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "repo": repo,
        "name": (gallery or {}).get("name") or repo,
        "fetched_at": fetched_at,
        "status": "partial",
        "problems": [],
        "selection": selection,
        # A two-layer random fixture is a real config shape and a fake model. Labelled here so
        # no claim about depth, width or parameter count can be drawn from one by accident.
        "synthetic_fixture": synthetic_reason(repo),
        "gallery": gallery,
        "licence": {
            "gallery_license_name": (gallery or {}).get("license_name"),
            "gallery_license_url": (gallery or {}).get("license_url"),
            "note": "recorded so nothing here is republished by accident; metadata held for analysis only",
        },
        "files": {},
    }
    problems: list[str] = record["problems"]

    try:
        info = hub_info(fetcher, repo)
    except CorpusError as exc:
        message = str(exc)
        gated = " 401 " in f" {message} " or " 403 " in f" {message} "
        record["status"] = "gated" if gated else "unreachable"
        problems.append(f"hub api: {message}")
        record["diagnostic"] = f"{HF_API}/{repo}"
        return record

    revision = info.get("sha")
    siblings = [item.get("rfilename") for item in info.get("siblings") or [] if item.get("rfilename")]
    record["hub"] = {
        "revision": revision,
        "gated": info.get("gated"),
        "private": info.get("private"),
        "library_name": info.get("library_name"),
        "pipeline_tag": info.get("pipeline_tag"),
        "last_modified": info.get("lastModified"),
        "created_at": info.get("createdAt"),
        "api_config": info.get("config"),
        "safetensors_totals": info.get("safetensors"),
        "card_license": card.get("license") if isinstance(card := info.get("cardData"), dict) else None,
        "tags": sorted(str(tag) for tag in info.get("tags") or []),
        "file_count": len(siblings),
        "has_safetensors": any(name.endswith(".safetensors") for name in siblings),
        "weight_formats": sorted({
            pathlib.PurePosixPath(name).suffix
            for name in siblings
            if pathlib.PurePosixPath(name).suffix in {".safetensors", ".bin", ".gguf", ".pth", ".pt"}
        }),
    }
    record["licence"]["hub_card_license"] = record["hub"]["card_license"]

    if info.get("gated"):
        record["status"] = "gated"
        problems.append(f"gated on the Hub: gated={info.get('gated')!r}; not fetched, no authentication attempted")
        record["diagnostic"] = f"https://huggingface.co/{repo}"
        return record
    if not revision:
        record["status"] = "unreachable"
        problems.append("hub api returned no revision sha, so nothing here could be pinned")
        return record

    payloads: dict[str, Any] = {}
    for name in WANTED_FILES:
        if name not in siblings:
            if name in REQUIRED_FILES:
                problems.append(f"{name} is absent from the repository listing")
            continue
        url = HF_RESOLVE.format(repo=repo, rev=revision, name=urllib.parse.quote(name))
        limit = MAX_INDEX_BYTES if name.endswith("index.json") else MAX_SMALL_FILE_BYTES
        try:
            body, from_cache = fetcher.cached_or_fetch(repo, revision, name, url, limit=limit)
            payloads[name] = json.loads(body.decode("utf-8"))
            record["files"][name] = {
                "url": url,
                "revision": revision,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "from_cache": from_cache,
                "fetched_at": fetched_at,
            }
        except (CorpusError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{name}: {exc}")
            record["files"][name] = {"url": url, "revision": revision, "error": str(exc)}

    adapter = payloads.get("adapter_config.json")
    if isinstance(adapter, dict):
        record["adapter"] = {
            "base_model": adapter.get("base_model_name_or_path"),
            "peft_type": adapter.get("peft_type"),
            "r": adapter.get("r"),
            "lora_alpha": adapter.get("lora_alpha"),
            "target_modules": sorted(adapter.get("target_modules") or [])
            if isinstance(adapter.get("target_modules"), (list, set))
            else adapter.get("target_modules"),
            "keys": sorted(adapter),
        }

    config = payloads.get("config.json")
    if not isinstance(config, dict):
        record["status"] = "partial" if payloads else "unreachable"
        if isinstance(adapter, dict):
            problems.append(
                "adapter-only repository: no config.json, the architecture is the base model's "
                f"({adapter.get('base_model_name_or_path')})"
            )
        else:
            problems.append("no usable config.json, so no structural profile was derived")
        record["diagnostic"] = HF_RESOLVE.format(repo=repo, rev=revision or "main", name="config.json")
        return record

    record["config"] = config
    record["profile"] = structural_profile(config)
    if "generation_config.json" in payloads:
        record["generation_config"] = payloads["generation_config.json"]
    if isinstance(payloads.get("tokenizer_config.json"), dict):
        record["tokenizer"] = summarise_tokenizer_config(payloads["tokenizer_config.json"])
    if "special_tokens_map.json" in payloads:
        record["special_tokens_map"] = payloads["special_tokens_map.json"]

    index = payloads.get("model.safetensors.index.json")
    tensors: dict[str, dict[str, Any]] = {}
    tensor_source = "none"
    shard_count = None
    if isinstance(index, dict) and isinstance(index.get("weight_map"), dict):
        tensors = {name: {} for name in index["weight_map"]}
        tensor_source = "index"
        shard_count = len({str(value) for value in index["weight_map"].values()})

    if want_header:
        shard = pick_safetensors_shard(siblings)
        if shard is None:
            problems.append("no safetensors shard to take a header from")
        else:
            try:
                header = fetch_safetensors_header(fetcher, repo, revision, shard)
                shard_tensors = header["tensors"]
                if tensor_source == "index":
                    # The index names every tensor across every shard but carries no dtypes;
                    # the header carries dtypes for one shard. Merge, and say so.
                    for name, spec in shard_tensors.items():
                        tensors[name] = spec
                    tensor_source = "index+header"
                else:
                    tensors = shard_tensors
                    tensor_source = "header"
                record["files"][shard] = {
                    "url": HF_RESOLVE.format(repo=repo, rev=revision, name=shard),
                    "revision": revision,
                    "range": "header only",
                    "from_cache": header["from_cache"],
                    "fetched_at": fetched_at,
                }
                record["safetensors_header_metadata"] = header["metadata"]
            except CorpusError as exc:
                problems.append(f"safetensors header for {shard}: {exc}")

    shards_on_the_hub = sum(1 for name in siblings if name.endswith(".safetensors"))
    if tensors:
        # A single shard's header describes a single shard. For a repository with no index
        # that is the whole inventory we can have, and a corpus that does not say so invites
        # a reader to treat one shard of 163 as a model.
        coverage = (
            "one shard of several; no index.json, so the rest of the inventory is unknown"
            if tensor_source == "header" and shards_on_the_hub > 1
            else "every tensor in the checkpoint"
        )
        record["tensors"] = tensor_inventory(tensors) | {
            "source": tensor_source,
            "shard_count": shard_count,
            "safetensors_shards_on_the_hub": shards_on_the_hub,
            "coverage": coverage,
        }
        if tensor_source == "header" and shards_on_the_hub > 1:
            problems.append(
                f"tensor inventory covers 1 of {shards_on_the_hub} shards: the repository "
                "publishes no model.safetensors.index.json"
            )
    else:
        record["tensors"] = {"source": "none", "tensor_count": 0}
        problems.append("no tensor inventory was obtained")

    missing_required = [
        name
        for name in REQUIRED_FILES
        if name not in record["files"] or "error" in record["files"][name]
    ]
    record["status"] = classify_status(problems, missing_required)
    if record["status"] != "complete":
        record["diagnostic"] = f"https://huggingface.co/{repo}/tree/{revision}"
    log(
        f"  {record['status']:9s} {repo} "
        f"({len(record['files'])} files, {record['tensors'].get('tensor_count', 0)} tensors)"
    )
    return record


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def observed_vocabulary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count, per structural dimension, which spellings the wild actually uses."""
    vocab: dict[str, dict[str, int]] = {dimension: {} for dimension in DIMENSIONS}
    mechanisms: dict[str, int] = {}
    model_types: dict[str, int] = {}
    architectures: dict[str, int] = {}
    nested: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    activation: dict[str, int] = {}
    for record in records:
        profile = record.get("profile")
        if not profile:
            continue
        for dimension, keys in (profile.get("aliases_present") or {}).items():
            for key in keys:
                vocab.setdefault(dimension, {})
                vocab[dimension][key] = vocab[dimension].get(key, 0) + 1
        for mark in profile.get("mechanisms") or []:
            mechanisms[mark] = mechanisms.get(mark, 0) + 1
        model_type = profile.get("model_type")
        if model_type:
            model_types[str(model_type)] = model_types.get(str(model_type), 0) + 1
        for arch in profile.get("architectures") or []:
            architectures[str(arch)] = architectures.get(str(arch), 0) + 1
        for key in profile.get("nested_config_keys") or []:
            nested[key] = nested.get(key, 0) + 1
        config = record.get("config") or {}
        for key in DTYPE_KEYS:
            if key in config:
                dtypes[f"{key}={config[key]}"] = dtypes.get(f"{key}={config[key]}", 0) + 1
        for key in ("hidden_act", "hidden_activation", "activation_function"):
            if key in config:
                activation[f"{key}={config[key]}"] = activation.get(f"{key}={config[key]}", 0) + 1
    return {
        "per_dimension": {k: dict(sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))) for k, v in sorted(vocab.items())},
        "mechanisms": dict(sorted(mechanisms.items(), key=lambda kv: (-kv[1], kv[0]))),
        "model_types": dict(sorted(model_types.items(), key=lambda kv: (-kv[1], kv[0]))),
        "architectures": dict(sorted(architectures.items(), key=lambda kv: (-kv[1], kv[0]))),
        "nested_config_keys": dict(sorted(nested.items(), key=lambda kv: (-kv[1], kv[0]))),
        "dtype_declarations": dict(sorted(dtypes.items(), key=lambda kv: (-kv[1], kv[0]))),
        "activations": dict(sorted(activation.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def record_slugs(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One filename per record, with collisions resolved rather than silently overwritten.

    Two gallery entries can name the same Hub repository, which is a fact about the gallery
    and not a reason for one record to land on top of the other. When a repository is shared,
    the gallery's own name for the model joins the slug.
    """
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = (record["repo"] or record["name"]).replace("/", "__").replace(" ", "_")
        by_repo.setdefault(key, []).append(record)
    slugs: dict[str, dict[str, Any]] = {}
    for key, group in by_repo.items():
        if len(group) == 1:
            slugs[key] = group[0]
            continue
        for record in group:
            # A dot is excluded as well as a slash: a name is a stranger's string, and a slug
            # built from one has no business being able to spell a parent directory.
            suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", record["name"])
            slugs[f"{key}--{suffix}"] = record
    return slugs


def build_index(records: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for record in sorted(records, key=lambda r: (r["repo"] or "", r["name"])):
        profile = record.get("profile") or {}
        resolved = profile.get("resolved") or {}
        rows.append(
            {
                "name": record["name"],
                "repo": record["repo"],
                "source": record["source"],
                "selection": record.get("selection"),
                "synthetic_fixture": record.get("synthetic_fixture"),
                "status": record["status"],
                "revision": (record.get("hub") or {}).get("revision"),
                "model_type": profile.get("model_type"),
                "architectures": profile.get("architectures"),
                "decoder_path": profile.get("decoder_path"),
                "depth": (resolved.get("depth") or {}).get("value"),
                "depth_key": (resolved.get("depth") or {}).get("key"),
                "mechanisms": profile.get("mechanisms"),
                "tensor_count": (record.get("tensors") or {}).get("tensor_count"),
                "tensor_source": (record.get("tensors") or {}).get("source"),
                "licence": record["licence"].get("gallery_license_name") or record["licence"].get("hub_card_license"),
                "problems": record["problems"],
            }
        )
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "counts": counts | {"total": len(records)},
        "observed_vocabulary": observed_vocabulary(records),
        "models": rows,
    }


# ---------------------------------------------------------------------------
# Pre-flight and driver
# ---------------------------------------------------------------------------


def preflight(cache: pathlib.Path, out: pathlib.Path, log: Any) -> None:
    """Refuse before a long job rather than halfway through it."""
    for path, label in ((cache, "raw cache"), (out, "corpus output")):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CorpusError(f"cannot create the {label} at {path}: {exc}") from exc
        probe = path / ".preflight-write-probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise CorpusError(f"the {label} at {path} is not writable: {exc}") from exc
    free = shutil.disk_usage(cache).free
    if free < MIN_FREE_DISK_BYTES:
        raise CorpusError(
            f"only {free / 1024**3:.1f} GiB free at {cache}, want {MIN_FREE_DISK_BYTES / 1024**3:.0f} GiB"
        )
    log(f"pre-flight: {free / 1024**3:.1f} GiB free, cache {cache}, out {out}")
    for url in (GALLERY_YML, f"{HF_API}/Qwen/Qwen3-0.6B"):
        request = urllib.request.Request(   # noqa: S310 - check_url pinned the scheme and host
            check_url(url), headers={"User-Agent": USER_AGENT}, method="HEAD"
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:   # noqa: S310
                log(f"pre-flight: {url} -> HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            if exc.code >= 400 and exc.code != 405:
                raise CorpusError(f"pre-flight: {url} answered HTTP {exc.code}") from exc
            log(f"pre-flight: {url} -> HTTP {exc.code} (acceptable for HEAD)")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CorpusError(f"pre-flight: {url} is unreachable: {exc}") from exc
    for leaked in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(leaked):
            raise CorpusError(
                f"{leaked} is set in this environment. This tool must never authenticate; "
                "unset it and run again so a gated repository is recorded as gated."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=False, help="corpus output directory")
    parser.add_argument("--cache", type=pathlib.Path, default=DEFAULT_CACHE, help="raw payload cache")
    parser.add_argument("--deadline-s", type=int, default=DEFAULT_DEADLINE_S)
    parser.add_argument("--byte-budget", type=int, default=DEFAULT_BYTE_BUDGET)
    parser.add_argument("--limit", type=int, default=0, help="stop after N gallery models (0 means all)")
    parser.add_argument("--no-headers", action="store_true", help="skip safetensors header range requests")
    parser.add_argument("--no-expand", action="store_true", help="do not look for Hub siblings the gallery omits")
    parser.add_argument(
        "--no-discover", action="store_true", help="skip the catalogue sweep and the gated re-probe"
    )
    parser.add_argument(
        "--discovery-target",
        type=int,
        default=DEFAULT_DISCOVERY_TARGET,
        help="how many catalogue models to admit (0 means none)",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    started = time.monotonic()
    heartbeat = {"last": started}

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    out = args.out or (pathlib.Path.cwd() / "private" / "research" / f"architecture-corpus-{time.strftime('%Y-%m-%d')}")
    try:
        preflight(args.cache, out, log)
    except CorpusError as exc:
        print(f"pre-flight refused: {exc}", file=sys.stderr)
        return 1
    if args.preflight_only:
        log("pre-flight only, nothing fetched")
        return 0

    budget = Budget(deadline=started + args.deadline_s, byte_budget=args.byte_budget)
    fetcher = Fetcher(budget, args.cache, log)

    try:
        gallery_sha = json.loads(fetcher.get(GALLERY_COMMIT_API).decode("utf-8"))["sha"]
        yml_bytes, from_cache = fetcher.cached_or_fetch(
            GALLERY_REPO, gallery_sha, "models.yml", GALLERY_YML, limit=4 * 1024**2
        )
    except (CorpusError, KeyError, json.JSONDecodeError) as exc:
        print(f"could not read the gallery: {exc}", file=sys.stderr)
        return 1
    log(f"gallery at {gallery_sha[:12]} ({len(yml_bytes)} bytes, cached={from_cache})")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a project dependency
        print(f"PyYAML is needed to read the gallery: {exc}", file=sys.stderr)
        return 1
    entries = gallery_entries(yaml.safe_load(yml_bytes.decode("utf-8")))
    if args.limit:
        entries = entries[: args.limit]
    log(f"{len(entries)} gallery entries, {sum(1 for e in entries if e['repo'])} with a Hub repository")

    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    truncated = False

    def work(entry: dict[str, Any]) -> dict[str, Any] | None:
        if not entry["repo"]:
            return {
                "schema_version": SCHEMA_VERSION,
                "source": "gallery",
                "repo": None,
                "name": entry["name"],
                "status": "unreachable",
                "problems": ["the gallery entry names no Hub repository"],
                "gallery": entry,
                "licence": {
                    "gallery_license_name": entry.get("license_name"),
                    "gallery_license_url": entry.get("license_url"),
                },
                "files": {},
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        return build_record(
            fetcher,
            gallery=entry,
            repo=entry["repo"],
            source="gallery",
            want_header=not args.no_headers,
            log=log,
        )

    def run_pool(items: list[Any], fn: Any) -> bool:
        nonlocal truncated
        with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as pool:
            futures = {pool.submit(fn, item): item for item in items}
            for done, future in enumerate(futures, start=1):
                item = futures[future]
                try:
                    record = future.result()
                except BudgetExhaustedError as exc:
                    log(f"BUDGET: {exc}; stopping and writing what is complete")
                    truncated = True
                    for pending in futures:
                        pending.cancel()
                    return False
                except Exception as exc:  # one bad record must not lose the run
                    label = (
                        item["repo"]
                        if isinstance(item, dict)
                        else item[0]
                        if isinstance(item, tuple)
                        else item
                    )
                    log(f"  FAILED   {label}: {type(exc).__name__}: {exc}")
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "source": "gallery",
                        "repo": label if isinstance(label, str) else None,
                        "name": str(label),
                        "status": "unreachable",
                        "problems": [f"{type(exc).__name__}: {exc}"],
                        "files": {},
                        "licence": {},
                        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                if record is not None:
                    with lock:
                        records.append(record)
                if time.monotonic() - heartbeat["last"] > 30:
                    heartbeat["last"] = time.monotonic()
                    log(
                        f"heartbeat: {done}/{len(items)} done, "
                        f"{budget.bytes_used / 1024**2:.1f} MiB over {budget.requests} requests, "
                        f"{budget.remaining_s() / 60:.1f} min left"
                    )
        return True

    completed = run_pool(entries, work)

    expansions: list[str] = []
    if completed and not args.no_expand:
        known_repos = {r["repo"] for r in records if r["repo"]}
        known_types = {
            str((r.get("profile") or {}).get("model_type"))
            for r in records
            if (r.get("profile") or {}).get("model_type")
        }
        families: dict[str, None] = {}
        for repo in sorted(known_repos):
            families.setdefault(family_prefix(repo), None)
        log(f"sibling search over {len(families)} families, {len(known_types)} model types already held")
        for family in sorted(families):
            if len(expansions) >= MAX_SIBLINGS_TOTAL:
                break
            url = f"{HF_API}?search={urllib.parse.quote(family)}&limit=25&config=true&sort=downloads&direction=-1"
            try:
                candidates = json.loads(fetcher.get(url).decode("utf-8"))
            except (CorpusError, json.JSONDecodeError) as exc:
                log(f"  sibling search for {family} failed: {exc}")
                continue
            if not isinstance(candidates, list):
                continue
            for repo in choose_siblings(candidates, known_repos, known_types):
                known_repos.add(repo)
                expansions.append(repo)
        log(f"{len(expansions)} siblings chosen beyond the gallery")
        completed = run_pool(
            expansions,
            lambda repo: build_record(
                fetcher, gallery=None, repo=repo, source="hub-sibling", want_header=not args.no_headers, log=log
            ),
        )

    discovered: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    if completed and not args.no_discover:
        known_repos = {r["repo"] for r in records if r["repo"]}
        known_types = {
            str((r.get("profile") or {}).get("model_type"))
            for r in records
            if (r.get("profile") or {}).get("model_type")
        }
        known_archs = {
            str(arch)
            for r in records
            for arch in ((r.get("profile") or {}).get("architectures") or [])
        }
        log(
            f"catalogue sweep: {len(DISCOVERY_QUERIES)} queries, holding "
            f"{len(known_types)} model types and {len(known_archs)} architectures"
        )
        results: dict[DiscoveryQuery, list[dict[str, Any]]] = {}
        for query in DISCOVERY_QUERIES:
            budget.check_clock()
            _, term, _, _ = query
            url = (
                f"{HF_API}?search={urllib.parse.quote(term)}"
                "&limit=30&config=true&sort=downloads&direction=-1"
            )
            try:
                rows = json.loads(fetcher.get(url).decode("utf-8"))
            except (CorpusError, json.JSONDecodeError) as exc:
                log(f"  search {term!r} failed: {exc}")
                results[query] = []
                continue
            results[query] = rows if isinstance(rows, list) else []
        discovered, skipped = choose_discoveries(
            results, known_repos, known_types, known_archs, args.discovery_target
        )
        by_category: dict[str, int] = {}
        for _, category, _ in discovered:
            by_category[category] = by_category.get(category, 0) + 1
        log(f"{len(discovered)} discovered ({by_category}), {len(skipped)} skipped with a reason")

        retry = [repo for repo in GATED_RETRY if repo not in known_repos]
        if retry:
            log(f"re-probing {len(retry)} previously gated repositories, unauthenticated")
        todo = [(repo, "gated-retry", "gate re-probe") for repo in retry] + discovered
        completed = run_pool(
            todo,
            lambda item: build_record(
                fetcher,
                gallery=None,
                repo=item[0],
                source=f"catalogue-{item[1]}",
                want_header=not args.no_headers,
                log=log,
                selection=item[2],
            ),
        )

    provenance = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gallery_repo": GALLERY_REPO,
        "gallery_revision": gallery_sha,
        "gallery_licence": "Apache License 2.0 (LICENSE file in the repository is the authority)",
        "tool": "tools/research/fetch_architecture_corpus.py",
        "user_agent": USER_AGENT,
        "authentication": "none; no token was read, passed or sought",
        "weights_fetched": "none; safetensors headers were taken by byte range only",
        "bytes_downloaded": budget.bytes_used,
        "requests": budget.requests,
        "wall_clock_s": round(time.monotonic() - started, 1),
        "truncated_by_budget": truncated or not completed,
        "sibling_expansion": expansions,
        "catalogue_discovered": [
            {"repo": repo, "category": category, "why": why} for repo, category, why in discovered
        ],
        "catalogue_skipped": [{"repo": repo, "reason": reason} for repo, reason in skipped],
        "gated_retry_probed": list(GATED_RETRY),
        "discovery_queries": [
            {"category": c, "term": t, "cap": cap, "repeat_allowed": rep}
            for c, t, cap, rep in DISCOVERY_QUERIES
        ],
        "content_skip_policy": (
            "repackaged formats and repackager accounts are skipped for shape; repositories "
            "presenting as harm are skipped by decision. Every skip is listed with its reason"
        ),
    }

    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    slugs = record_slugs(records)
    for slug, record in sorted(slugs.items()):
        written += write_json_atomic(models_dir / f"{slug}.json", record)
    # A record left behind by an earlier run is a model the index does not list, which is the
    # one inconsistency a reader cannot detect. The tool owns this directory, so it tidies it,
    # and says which files it removed rather than doing it quietly.
    expected = {f"{slug}.json" for slug in slugs} | {f"{slug}.json.gz" for slug in slugs}
    if not truncated and completed:
        for stale in sorted(models_dir.iterdir()):
            if stale.is_file() and stale.name not in expected:
                log(f"pruning a record this run did not produce: {stale.name}")
                stale.unlink()
    elif any(path.name not in expected for path in models_dir.iterdir()):
        log("NOT pruning: this run was truncated, so an older record may be the better one")
    index_bytes = write_json_atomic(out / "index.json", build_index(records, provenance))
    log(f"wrote {len(records)} records ({written} bytes) and an index ({index_bytes} bytes) to {out}")
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    log(f"status: {counts}")
    log(f"budget: {budget.bytes_used / 1024**2:.1f} MiB over {budget.requests} requests")
    return 1 if (truncated or not completed) else 0


if __name__ == "__main__":
    sys.exit(main())
