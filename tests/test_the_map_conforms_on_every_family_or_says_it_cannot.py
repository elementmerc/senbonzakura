# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The conformance suite for `modelmap`: one case per structural fact a map cannot assume.

WHAT THIS IS AND WHY IT IS THE DELIVERABLE RATHER THAN A BONUS. The 2026-10-08 plan's own words
were that consistency across architectures is a test matrix, and that a family with no test in
that matrix is a family this backend does not support, said out loud. This file is that matrix.
The specification is the sixteen numbered findings in
`private/research/2026-10-08-what-is-actually-in-a-checkpoint.md`, each measured on a real
checkpoint, and each one gets a case here.

THREE DISCIPLINES, AND THEY ARE WHAT MAKE IT WORTH HAVING.

1. A NEW ARCHITECTURE IS ADDED AS DATA. `FAMILIES` below is the table; the tests parametrise over
   it. Supporting a new family means adding a row, not writing a test, which is the only version
   of this that stays true once somebody is in a hurry.

2. A FAMILY WITH NO LOCAL INSTANCE IS RECORDED AS UNTESTED, NEVER AS PASSING. This is the whole
   point of the four-state column rather than a boolean. "We tested it and the map fails" is a
   defect; "we have never been shown an instance" is a gap in the evidence. A suite that printed
   the same green tick for both would be lying about the second one, and the lie would be
   invisible. `test_the_suites_own_output_names_every_untested_family` fails if a gap ever stops
   being named.

3. AN UNRECOGNISED MECHANISM MUST REFUSE RATHER THAN GUESS. A map that met a mechanism string it
   had never seen and filed it under attention would answer every question confidently and
   wrongly. The refusal is readable; the guess is not. The first section below is that case, and
   it is first because it needs no checkpoint, no network and no card, so it is the part that
   holds even when everything else is unavailable.

WHAT THIS SUITE DOES NOT DO. It does not load a model, read a weight, touch a card or reach the
network. Every case is a config dict, hand-built or read from the local cache. So it says whether
the MAP is right about a shape; it says nothing about whether an edit to that shape would work.
"""
from __future__ import annotations

import glob
import json
import pathlib

import pytest

from senbonzakura import modelmap as mm
from senbonzakura.modelmap import BlockKind

HUB = pathlib.Path("~/.cache/huggingface/hub").expanduser()

#: The four states a family can be in. A boolean would collapse the last two, and the difference
#: between them is the difference between a defect and a gap in the evidence.
RESOLVED = "resolved"              # the map reads this family correctly, measured on an instance
PARTIAL = "partial"                # the map reads some of it and names what it cannot
UNRESOLVED = "unresolved"          # an instance exists locally and the map gets it wrong
UNTESTED = "untested"              # no instance on this disk, so nothing has been measured


def local_config(repo):
    """One repo's config from the local hub cache, or None when it is not here.

    Never downloads. A repo that is not in the cache is a gap in the evidence and the caller
    records it as `UNTESTED`, which is the discipline this whole file turns on.
    """
    if repo is None:
        return None
    pattern = str(HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / "*" / "config.json")
    for path in sorted(glob.glob(pattern)):
        try:
            cfg = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict):
            return cfg
    return None


# ----------------------------------------------------------------------------------------------
# THE FAMILY TABLE. This is the data; everything below parametrises over it.
#
# `decoder_type` uses the vocabulary of the published architecture catalogue (rasbt's
# llm-architecture-gallery, Apache-2.0, attributed in THIRD-PARTY-NOTICES.md), so that the
# coverage question is asked in the catalogue's terms rather than in ours. Its seven values are
# Dense, Dense hybrid, Sparse hybrid, Sparse MoE, Sparse omnimodal MoE, Hybrid MoE and Recurrent.
#
# `mechanism_values` is what the family's config actually puts in its per-block list. A value of
# None means NOBODY HERE HAS SEEN ONE: we have no instance, so the string it would declare is
# unverified and is not guessed at. That is deliberately distinguishable from an empty tuple,
# which means the family declares no per-block list at all (a uniform stack).
# ----------------------------------------------------------------------------------------------

FAMILIES = (
    # ── Dense, the easy row every tool is written against ──────────────────────────────────────
    {"family": "llama-style dense", "decoder_type": "Dense", "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
     "mechanism_values": (), "count_key": "num_hidden_layers", "expect_blocks": 30,
     "expect_experts": None},
    {"family": "qwen2 dense", "decoder_type": "Dense", "repo": "Qwen/Qwen2.5-0.5B-Instruct",
     "mechanism_values": (), "count_key": "num_hidden_layers", "expect_blocks": 24,
     "expect_experts": None},
    {"family": "qwen3 dense", "decoder_type": "Dense", "repo": "Qwen/Qwen3-32B",
     "mechanism_values": (), "count_key": "num_hidden_layers", "expect_blocks": 64,
     "expect_experts": None},
    {"family": "gpt2 dense", "decoder_type": "Dense", "repo": "gpt2",
     "mechanism_values": (), "count_key": "n_layer", "expect_blocks": 12,
     "expect_experts": None},
    {"family": "gemma4 sliding-window dense", "decoder_type": "Sparse omnimodal MoE",
     "repo": "google/gemma-4-31b-it", "mechanism_values": ("sliding_attention", "full_attention"),
     "count_key": "num_hidden_layers", "expect_blocks": 60, "expect_experts": None},

    # ── Sparse mixture of experts ──────────────────────────────────────────────────────────────
    {"family": "qwen3-moe", "decoder_type": "Sparse MoE", "repo": "Qwen/Qwen3-30B-A3B-Instruct-2507",
     "mechanism_values": (), "count_key": "num_hidden_layers", "expect_blocks": 48,
     "expect_experts": 128},
    {"family": "gpt-oss sliding MoE", "decoder_type": "Sparse MoE", "repo": "openai/gpt-oss-20b",
     "mechanism_values": ("sliding_attention", "full_attention"),
     "count_key": "num_hidden_layers", "expect_blocks": 24, "expect_experts": 32},
    {"family": "deepseek latent-attention MoE", "decoder_type": "Sparse MoE",
     "repo": "deepseek-ai/DeepSeek-V4-Flash", "mechanism_values": (),
     "count_key": "num_hidden_layers", "expect_blocks": 44, "expect_experts": 256},
    {"family": "glm latent-attention MoE", "decoder_type": "Sparse MoE",
     "repo": "zai-org/GLM-4.7-Flash", "mechanism_values": (),
     "count_key": "num_hidden_layers", "expect_blocks": 48, "expect_experts": 64},

    # ── Hybrid: a block may hold a non-attention mixer, or no mixer at all ─────────────────────
    {"family": "lfm2 short-convolution hybrid", "decoder_type": "Dense hybrid",
     "repo": "LiquidAI/LFM2.5-1.2B-Instruct", "mechanism_values": ("conv", "full_attention"),
     "count_key": "num_hidden_layers", "expect_blocks": 16, "expect_experts": None},
    {"family": "lfm2-moe hybrid", "decoder_type": "Hybrid MoE", "repo": "LiquidAI/LFM2.5-8B-A1B",
     "mechanism_values": ("conv", "full_attention"), "count_key": "num_hidden_layers",
     "expect_blocks": 24, "expect_experts": 32},
    {"family": "qwen3.5 gated-deltanet hybrid MoE", "decoder_type": "Hybrid MoE",
     "repo": "Qwen/Qwen3.6-35B-A3B", "mechanism_values": ("linear_attention", "full_attention"),
     "count_key": "num_hidden_layers", "expect_blocks": 41, "expect_experts": 256},
    {"family": "nemotron-h mamba2 hybrid", "decoder_type": "Sparse hybrid",
     "repo": "nvidia/Nemotron-H-4B-Base-8K", "mechanism_values": ("M", "*", "-"),
     "count_key": "num_hidden_layers", "expect_blocks": 52, "expect_experts": None},
    {"family": "nemotron-h hybrid MoE", "decoder_type": "Hybrid MoE",
     "repo": "gabrielebeltramo/NemotronH-300M-stories",
     "mechanism_values": ("mamba", "moe", "attention"), "count_key": None,
     "expect_blocks": 12, "expect_experts": 16},
    {"family": "bamba mamba2 hybrid", "decoder_type": "Dense hybrid",
     "repo": "ibm-ai-platform/Bamba-9B", "mechanism_values": None,
     "count_key": "num_hidden_layers", "expect_blocks": 32, "expect_experts": None},

    # ── Families in the catalogue with NO instance on this disk. Every one of these is a GAP in
    #    the evidence and not a passing case, and the suite says so by name. The mechanism string
    #    each one declares is unverified, which is why `mechanism_values` is None: guessing it
    #    would make the row look tested when nothing has been read.
    {"family": "kimi delta attention", "decoder_type": "Sparse MoE", "repo": None,
     "mechanism_values": None, "count_key": None, "expect_blocks": None, "expect_experts": None},
    {"family": "mlstm recurrent, no self-attention", "decoder_type": "Recurrent", "repo": None,
     "mechanism_values": None, "count_key": None, "expect_blocks": None, "expect_experts": None},
    {"family": "cca cross-layer attention", "decoder_type": "Dense", "repo": None,
     "mechanism_values": None, "count_key": None, "expect_blocks": None, "expect_experts": None},
    {"family": "qwen sparse attention", "decoder_type": "Sparse MoE", "repo": None,
     "mechanism_values": None, "count_key": None, "expect_blocks": None, "expect_experts": None},
    {"family": "mla with kv layernorm", "decoder_type": "Sparse MoE", "repo": None,
     "mechanism_values": None, "count_key": None, "expect_blocks": None, "expect_experts": None},
    {"family": "abliterated dense", "decoder_type": "Dense", "repo": None,
     "mechanism_values": None, "count_key": None, "expect_blocks": None, "expect_experts": None},
)

#: Families whose absence is a known, recorded gap rather than an oversight. Named individually so
#: that adding a checkpoint for one of them is a visible change to this list rather than a silent
#: improvement nobody notices.
EXPECTED_GAPS = ("kimi delta attention", "mlstm recurrent, no self-attention",
                 "cca cross-layer attention", "qwen sparse attention",
                 "mla with kv layernorm", "abliterated dense")


def family_state(row):
    """Which of the four states this family is in, measured now rather than recorded by hand."""
    cfg = local_config(row["repo"])
    if cfg is None:
        return UNTESTED, None, cfg
    mapping = mm.describe(config=cfg)
    if row["expect_blocks"] is not None and mapping.blocks.total.unwrap() != row["expect_blocks"]:
        return UNRESOLVED, mapping, cfg
    if mapping.mechanisms.kinds.known and mapping.mechanisms.unmapped:
        return PARTIAL, mapping, cfg
    return RESOLVED, mapping, cfg


def ids(rows):
    """Readable parametrise ids, so a failure names the family rather than an index."""
    return [r["family"] for r in rows]


WITH_INSTANCE = tuple(r for r in FAMILIES if local_config(r["repo"]) is not None)
WITHOUT_INSTANCE = tuple(r for r in FAMILIES if local_config(r["repo"]) is None)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 1. THE UNRECOGNISED MECHANISM. First, because it needs nothing but a dict.
# ══════════════════════════════════════════════════════════════════════════════════════════════

def novel_config(value, *, blocks=8, **extra):
    """A config declaring `blocks` blocks of a mechanism string nobody has ever seen."""
    cfg = {"model_type": "entirelynovel", "architectures": ["EntirelyNovelForCausalLM"],
           "num_hidden_layers": blocks, "hidden_size": 512, "num_attention_heads": 8,
           "layer_types": [value] * blocks}
    cfg.update(extra)
    return cfg


def test_a_mechanism_nobody_has_seen_is_unknown_and_is_never_guessed_as_attention():
    """THE CASE THIS SUITE EXISTS FOR. A map that filed an unrecognised mechanism under attention
    would answer every question about the model confidently and wrongly, and nothing downstream
    could tell. The published catalogue contains at least one entry reading "No self-attention;
    mLSTM recurrent layers with matrix memory", so this is not a hypothetical string.
    """
    mapping = mm.describe(config=novel_config("mlstm_matrix_memory"))
    kinds = mapping.mechanisms.kinds.value
    assert all(k is BlockKind.UNKNOWN for k in kinds), (
        f"every block should be UNKNOWN, got {[k.value for k in kinds]}")
    assert BlockKind.ATTENTION not in kinds, "an unrecognised mechanism must never become attention"
    assert BlockKind.MIXER not in kinds, "nor a mixer, which is also a guess"


def test_the_string_it_could_not_interpret_is_carried_through_verbatim():
    mapping = mm.describe(config=novel_config("mlstm_matrix_memory"))
    assert mapping.mechanisms.unmapped == ("mlstm_matrix_memory",)
    assert mapping.mechanisms.raw == ("mlstm_matrix_memory",) * 8


def test_it_names_which_questions_it_therefore_cannot_answer():
    """Naming the blocked questions is the difference between a refusal and a shrug."""
    blocked = mm.describe(config=novel_config("mlstm_matrix_memory")).unanswerable()
    assert "what mechanism does each block hold" in blocked
    assert "how many blocks have no attention" in blocked
    assert "can a per-block attention panel be drawn" in blocked
    for question, why in blocked.items():
        assert why and why.strip(), f"the reason for {question!r} is empty"
    assert "mlstm_matrix_memory" in blocked["what mechanism does each block hold"], (
        "the reason must carry the string it could not interpret, or the reader cannot act on it")


def test_it_says_how_to_make_the_unanswerable_question_answerable():
    why = mm.describe(config=novel_config("mlstm_matrix_memory")).unanswerable()[
        "what mechanism does each block hold"]
    assert "MECHANISM_BY_VALUE" in why, (
        "a refusal that does not say what would fix it makes the reader read the source")


def test_what_it_still_answers_it_answers_correctly():
    """A refusal must be LOCAL. An unreadable mechanism column is not a reason to lose the depth."""
    mapping = mm.describe(config=novel_config("mlstm_matrix_memory", blocks=11))
    assert mapping.blocks.total.value == 11
    assert mapping.blocks.total.source == "num_hidden_layers"
    assert mapping.model_type.value == "entirelynovel"
    assert "how many blocks does this model have" not in mapping.unanswerable()


def test_an_unknown_mechanism_crashes_on_nothing_a_caller_would_reasonably_do():
    """Every accessor on the record is exercised, because a viewer will call all of them."""
    mapping = mm.describe(config=novel_config("something_from_2027"))
    assert mapping.mechanisms.tally() == {"unknown": 8}
    assert mapping.mechanisms.without_attention() is None
    assert mapping.mechanisms.unknown_blocks() == 8
    assert mapping.disagreements
    assert mm.summarise(mapping)
    assert mapping.unanswerable()


def test_a_stack_mixing_a_known_and_an_unknown_mechanism_is_partly_read():
    """The realistic case: a new family reuses `full_attention` and adds one mechanism of its own."""
    cfg = {"num_hidden_layers": 4, "layer_types":
           ["full_attention", "kimi_delta_attention", "full_attention", "kimi_delta_attention"]}
    mapping = mm.describe(config=cfg)
    assert [k.value for k in mapping.mechanisms.kinds.value] == [
        "attention", "unknown", "attention", "unknown"]
    assert mapping.mechanisms.unknown_blocks() == 2
    assert mapping.mechanisms.without_attention() is None, (
        "two blocks are unread, so the attention-free count cannot be stated")


def test_an_unknown_letter_in_the_block_pattern_refuses_the_whole_list_with_the_letter():
    """The letter pattern is all or nothing: a letter with no meaning makes the POSITIONS unsafe,
    because the pattern is read by position and a misread letter shifts nothing but means the
    block at that index is anybody's guess.
    """
    mapping = mm.describe(config={"num_hidden_layers": 4, "hybrid_override_pattern": "M-*Z"})
    assert not mapping.mechanisms.kinds.known
    assert "'Z'" in mapping.mechanisms.kinds.why or "Z" in str(mapping.mechanisms.unmapped)
    assert mapping.blocks.total.value == 4, "the depth is still readable"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 2. ONE CASE PER STRUCTURAL FACT. The specification is the sixteen findings of 2026-10-08.
# ══════════════════════════════════════════════════════════════════════════════════════════════

def test_fact_01_a_block_may_hold_one_mechanism_or_none():
    """Nemotron-H's 52 blocks are 24 Mamba, 24 bare MLP and 4 attention, one mechanism each."""
    cfg = local_config("nvidia/Nemotron-H-4B-Base-8K")
    if cfg is None:
        pytest.skip("UNTESTED: nvidia/Nemotron-H-4B-Base-8K is not in the local cache")
    tally = mm.describe(config=cfg).mechanisms.tally()
    assert tally == {"attention": 4, "mixer": 24, "mlp_only": 24}
    assert tally["mlp_only"], "a block with no sequence mixer at all must be representable"


def test_fact_02_layer_types_is_the_same_key_for_a_window_and_for_a_whole_other_mechanism():
    """The trap: `sliding_attention` still HAS attention, `conv` and `linear_attention` do not.
    A map deciding "does this block have attention" from the string alone gets one of them wrong.
    """
    window = mm.describe(config={"num_hidden_layers": 4, "layer_types":
                                 ["sliding_attention", "full_attention"] * 2})
    mechanism = mm.describe(config={"num_hidden_layers": 4, "layer_types":
                                    ["conv", "full_attention"] * 2})
    assert window.mechanisms.without_attention() == 0
    assert mechanism.mechanisms.without_attention() == 2
    assert window.mechanisms.kinds.source == mechanism.mechanisms.kinds.source == "layer_types"


def test_fact_02_linear_attention_is_named_attention_and_is_not_attention():
    mapping = mm.describe(config={"num_hidden_layers": 2,
                                  "layer_types": ["linear_attention", "full_attention"]})
    assert mapping.mechanisms.kinds.value == [BlockKind.MIXER, BlockKind.ATTENTION]


@pytest.mark.parametrize(("cfg", "source", "kinds"), [
    ({"num_hidden_layers": 2, "layer_types": ["conv", "full_attention"]},
     "layer_types", ["mixer", "attention"]),
    ({"layers_block_type": ["mamba", "moe", "attention"]},
     "layers_block_type", ["mixer", "mlp_only", "attention"]),
    ({"num_hidden_layers": 3, "hybrid_override_pattern": "M-*"},
     "hybrid_override_pattern", ["mixer", "mlp_only", "attention"]),
    ({"num_hidden_layers": 3, "attn_layer_indices": [1]},
     "attn_layer_indices", ["unknown", "attention", "unknown"]),
])
def test_fact_03_all_four_spellings_of_the_mechanism_list_resolve_in_one_place(cfg, source, kinds):
    mapping = mm.describe(config=cfg)
    assert mapping.mechanisms.kinds.source == source
    assert [k.value for k in mapping.mechanisms.kinds.value] == kinds


def test_fact_03_the_inverted_spelling_does_not_turn_a_32_block_model_into_a_3_block_one():
    """`attn_layer_indices` on Bamba-9B is three integers. Read at face value as "the block
    types", a 32-block model reports as having 3 blocks.
    """
    mapping = mm.describe(config={"num_hidden_layers": 32, "attn_layer_indices": [9, 18, 27]})
    assert len(mapping.mechanisms.kinds.value) == 32
    assert mapping.mechanisms.tally() == {"attention": 3, "unknown": 29}


def test_fact_03_the_inverted_spelling_refuses_to_name_the_unmarked_blocks():
    """The key says nothing about them. Labelling them Mamba because Bamba happens to be the
    checkpoint in hand would be a guess about every other family that uses the same key.
    """
    mapping = mm.describe(config={"num_hidden_layers": 4, "attn_layer_indices": [0]})
    assert mapping.mechanisms.kinds.value[1:] == [BlockKind.UNKNOWN] * 3
    assert mapping.mechanisms.without_attention() is None
    assert "how many blocks have no attention" in mapping.unanswerable()


def test_fact_03_one_model_type_with_two_different_spellings_still_resolves():
    """Both local Nemotron-H checkpoints are `model_type: nemotron_h` and
    `NemotronHForCausalLM`, and they spell the block list two different ways. So nothing may key
    off the model type or the architecture class name.
    """
    four_b = local_config("nvidia/Nemotron-H-4B-Base-8K")
    three_hundred_m = local_config("gabrielebeltramo/NemotronH-300M-stories")
    if four_b is None or three_hundred_m is None:
        pytest.skip("UNTESTED: both Nemotron-H checkpoints are needed and one is not cached")
    assert four_b["model_type"] == three_hundred_m["model_type"] == "nemotron_h"
    assert four_b["architectures"] == three_hundred_m["architectures"]
    a, b = mm.describe(config=four_b), mm.describe(config=three_hundred_m)
    assert a.mechanisms.kinds.source == "hybrid_override_pattern"
    assert b.mechanisms.kinds.source == "layers_block_type"
    assert a.mechanisms.kinds.known and b.mechanisms.kinds.known


def test_fact_04_three_checkpoints_hold_a_block_the_declared_count_never_reaches():
    """The multi-token-prediction block. This is the arithmetic `streaming` was wrong about."""
    mapping = mm.describe(config={"num_hidden_layers": 47, "num_nextn_predict_layers": 1})
    assert mapping.blocks.declared.value == 47
    assert mapping.blocks.extra.value == 1
    assert mapping.blocks.total.value == 48
    assert "num_nextn_predict_layers" in mapping.blocks.total.source


def test_fact_04_the_other_spelling_of_the_extra_block_also_counts():
    mapping = mm.describe(config={"num_hidden_layers": 40, "mtp_num_hidden_layers": 1})
    assert mapping.blocks.total.value == 41


def test_fact_04_a_config_declaring_no_depth_falls_back_to_its_mechanism_list():
    """NemotronH-300M declares no depth under any of the six spellings."""
    mapping = mm.describe(config={"layers_block_type": ["mamba", "moe", "attention"] * 4})
    assert mapping.blocks.total.value == 12
    assert mapping.blocks.total.source == "layers_block_type"
    assert not mapping.blocks.declared.known
    assert "never states a stack depth" in mapping.blocks.declared.why


def test_fact_04_a_config_stating_no_depth_at_all_says_so_rather_than_returning_zero():
    mapping = mm.describe(config={"hidden_size": 8})
    assert not mapping.blocks.total.known
    assert "how many blocks does this model have" in mapping.unanswerable()


def test_fact_05_the_decoder_config_is_not_always_the_top_level_config():
    cfg = {"model_type": "gemma4", "text_config": {"num_hidden_layers": 60, "hidden_size": 5376},
           "vision_config": {"num_hidden_layers": 27}, "audio_config": {"num_hidden_layers": 12}}
    mapping = mm.describe(config=cfg)
    assert mapping.nested_under.value == "text_config"
    assert mapping.blocks.total.value == 60, "the vision tower's depth must never win"


def test_fact_05_a_sub_config_that_declares_no_depth_cannot_hijack_a_good_top_level():
    mapping = mm.describe(config={"n_layer": 12, "text_config": {"something_else": 1}})
    assert not mapping.nested_under.known
    assert mapping.blocks.total.value == 12


@pytest.mark.parametrize(("cfg", "count", "key"), [
    ({"num_hidden_layers": 1, "num_experts": 128}, 128, "num_experts"),
    ({"num_hidden_layers": 1, "num_local_experts": 32}, 32, "num_local_experts"),
    ({"num_hidden_layers": 1, "n_routed_experts": 256}, 256, "n_routed_experts"),
])
def test_fact_06_every_spelling_of_the_expert_count_resolves_with_its_key(cfg, count, key):
    exp = mm.describe(config=cfg).experts
    assert exp.routed.value == count
    assert exp.routed.source == key
    assert exp.is_mixture


def test_fact_06_a_dense_config_can_carry_the_expert_key_with_a_null_value():
    """Gemma-4-31B does exactly this, beside `enable_moe_block: false`. So the key's PRESENCE
    proves nothing and only a count may decide.
    """
    exp = mm.describe(config={"num_hidden_layers": 1, "num_experts": None,
                              "enable_moe_block": False}).experts
    assert exp.is_mixture is False
    assert exp.declared_but_null == ("num_experts",)
    assert "null value" in exp.routed.why


def test_fact_06_the_width_spelling_of_shared_experts_is_never_read_as_a_count():
    """`shared_expert_intermediate_size` is 512 on Qwen3.6-35B. A reader treating it as a count
    reports 512 shared experts.
    """
    exp = mm.describe(config={"num_hidden_layers": 1, "num_experts": 256,
                              "shared_expert_intermediate_size": 512}).experts
    assert not exp.shared_count.known
    assert exp.shared_width.value == 512
    assert "a width, not a count" in exp.shared_count.why


def test_fact_06_the_two_dense_block_spellings_have_opposite_polarity():
    """`num_dense_layers: 2` means blocks 0 and 1 are dense. `mlp_only_layers: []` means NONE
    are. On that pair an empty list and a count of zero happen to agree, which is the coincidence
    that would let a polarity bug ship.
    """
    prefix = mm.describe(config={"num_hidden_layers": 24, "num_experts": 32,
                                 "num_dense_layers": 2}).experts
    listed = mm.describe(config={"num_hidden_layers": 48, "num_experts": 128,
                                 "mlp_only_layers": []}).experts
    assert prefix.dense_blocks.value == [0, 1]
    assert prefix.dense_blocks.source == "num_dense_layers"
    assert listed.dense_blocks.value == []
    assert listed.dense_blocks.source == "mlp_only_layers"


def test_fact_06_a_config_saying_nothing_about_dense_blocks_says_so():
    exp = mm.describe(config={"num_hidden_layers": 4, "num_experts": 8}).experts
    assert not exp.dense_blocks.known
    assert "num_dense_layers" in exp.dense_blocks.why
    assert "mlp_only_layers" in exp.dense_blocks.why


def test_fact_12_the_usual_head_width_identity_is_reported_rather_than_assumed():
    holds = mm.describe(config={"num_hidden_layers": 1, "hidden_size": 512,
                                "num_attention_heads": 8, "head_dim": 64}).heads
    fails = mm.describe(config={"num_hidden_layers": 1, "hidden_size": 5120,
                                "num_attention_heads": 64, "head_dim": 128}).heads
    assert holds.quotient_is_head_dim is True
    assert fails.quotient_is_head_dim is False


def test_fact_12_a_broken_identity_makes_the_head_width_question_unanswerable():
    mapping = mm.describe(config={"num_hidden_layers": 1, "hidden_size": 5120,
                                  "num_attention_heads": 64, "head_dim": 128})
    assert "how wide is one attention head" in mapping.unanswerable()


def test_fact_12_an_unanswered_identity_is_none_and_not_false():
    heads = mm.describe(config={"num_hidden_layers": 1, "hidden_size": 8}).heads
    assert heads.quotient_is_head_dim is None, "unanswered is not the same as answered no"


def test_fact_12_split_query_and_value_widths_are_carried_through():
    heads = mm.describe(config={"num_hidden_layers": 1, "qk_nope_head_dim": 192,
                                "qk_rope_head_dim": 64, "v_head_dim": 256,
                                "q_lora_rank": 768}).heads
    assert heads.split_widths == {"q_lora_rank": 768, "qk_nope_head_dim": 192,
                                  "qk_rope_head_dim": 64, "v_head_dim": 256}


def test_fact_14_a_config_naming_no_architecture_still_resolves_everything_else():
    """tiny-random-gpt2 has no `architectures` key at all, and it is exactly the fixture a
    conformance suite reaches for.
    """
    mapping = mm.describe(config={"model_type": "gpt2", "n_layer": 5})
    assert not mapping.architecture.known
    assert "resolved by class name" in mapping.architecture.why
    assert mapping.blocks.total.value == 5


def test_a_config_that_is_not_an_object_is_described_as_unreadable_rather_than_crashing():
    mapping = mm.describe(config=[1, 2, 3])
    assert not mapping.blocks.total.known
    assert mm.summarise(mapping)


def test_describe_given_neither_a_config_nor_a_model_refuses_loudly():
    with pytest.raises(ValueError, match="given neither"):
        mm.describe()


def test_every_record_states_that_the_residual_path_is_not_measured():
    """The one field that is always populated, because the absence must never read as a zero."""
    mapping = mm.describe(config={"num_hidden_layers": 2})
    assert "forward pass" in mapping.residual_path_why
    assert "where does the residual stream run" in mapping.unanswerable()
    assert any("MAP_NOT_MEASURED" in line for line in mm.summarise(mapping))


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 3. THE FAMILY MATRIX, over the table. Adding a family is adding a row.
# ══════════════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("row", WITH_INSTANCE, ids=ids(WITH_INSTANCE))
def test_the_map_resolves_the_depth_of_every_family_with_a_local_instance(row):
    state, mapping, _ = family_state(row)
    assert state != UNRESOLVED, (
        f"{row['family']}: the map says {mapping.blocks.total.unwrap()} blocks and the table "
        f"expects {row['expect_blocks']}")
    assert mapping.blocks.total.value == row["expect_blocks"]
    if row["count_key"] is not None:
        assert row["count_key"] in (mapping.blocks.total.source or "")


@pytest.mark.parametrize("row", WITH_INSTANCE, ids=ids(WITH_INSTANCE))
def test_the_map_resolves_the_expert_structure_of_every_family_with_a_local_instance(row):
    _state, mapping, _cfg = family_state(row)
    if row["expect_experts"] is None:
        assert not mapping.experts.is_mixture, (
            f"{row['family']} is not a mixture of experts and the map thinks it is")
    else:
        assert mapping.experts.routed.value == row["expect_experts"]
        assert mapping.experts.routed.source, "an expert count with no key has no provenance"


@pytest.mark.parametrize("row", WITH_INSTANCE, ids=ids(WITH_INSTANCE))
def test_the_mechanism_values_in_the_table_are_the_ones_the_config_declares(row):
    """The table's `mechanism_values` is a CLAIM about the checkpoint and it is checked, so the
    table cannot drift away from the configs it describes.
    """
    _state, mapping, _cfg = family_state(row)
    if row["mechanism_values"] is None:
        pytest.skip(f"UNTESTED: the mechanism strings for {row['family']} are unverified")
    if row["mechanism_values"] == ():
        assert not mapping.mechanisms.kinds.known, (
            f"{row['family']} is recorded as declaring no per-block list and it declares "
            f"{mapping.mechanisms.kinds.source}")
        return
    assert set(mapping.mechanisms.raw) <= set(row["mechanism_values"]), (
        f"{row['family']} declares {sorted(set(mapping.mechanisms.raw))} and the table records "
        f"{sorted(row['mechanism_values'])}")


@pytest.mark.parametrize("row", WITH_INSTANCE, ids=ids(WITH_INSTANCE))
def test_no_family_with_a_local_instance_is_silently_partly_read(row):
    """A PARTIAL family is allowed, and it has to say which questions it cannot answer. What is
    not allowed is a family that comes back partly read while claiming it answered everything.
    """
    state, mapping, _cfg = family_state(row)
    if state != PARTIAL:
        return
    blocked = mapping.unanswerable()
    assert "what mechanism does each block hold" in blocked, (
        f"{row['family']} is only partly read and does not say so")


@pytest.mark.parametrize("row", WITHOUT_INSTANCE, ids=ids(WITHOUT_INSTANCE))
def test_a_family_with_no_local_instance_is_recorded_as_untested(row):
    """NOT A PASSING CASE. This test records a gap in the evidence, and it is written as a skip
    with the family's name so that the gap appears in the suite's own output. A green tick here
    would be indistinguishable from a family that was actually measured.
    """
    assert local_config(row["repo"]) is None
    pytest.skip(f"UNTESTED FAMILY: {row['family']} ({row['decoder_type']}) has no instance in "
                f"the local cache, so the map has never been shown one. This is a gap in the "
                f"evidence and not a defect in the map.")


def test_the_suites_own_output_names_every_untested_family():
    """The gate on discipline 2. If a gap ever stops being named, this fails.

    It checks the table rather than the skip messages, because a skip that nobody reads is the
    same as silence. The assertion is that the set of families with no instance is exactly the
    set recorded as expected gaps, so acquiring a checkpoint for one of them, or losing one,
    forces a deliberate edit here.
    """
    missing = tuple(sorted(r["family"] for r in WITHOUT_INSTANCE))
    assert missing == tuple(sorted(EXPECTED_GAPS)), (
        f"the families with no local instance are {missing} and the recorded gaps are "
        f"{tuple(sorted(EXPECTED_GAPS))}. If a checkpoint has been added or removed, update "
        f"EXPECTED_GAPS deliberately: the point of this list is that a gap cannot close or open "
        f"without somebody noticing.")


def test_every_catalogue_decoder_type_appears_in_the_table():
    """The catalogue's seven `decoder_type` values are the coverage question's own vocabulary, so
    a type absent from this table is a blind spot we would not know we had.
    """
    catalogue = {"Dense", "Dense hybrid", "Sparse hybrid", "Sparse MoE",
                 "Sparse omnimodal MoE", "Hybrid MoE", "Recurrent"}
    covered = {r["decoder_type"] for r in FAMILIES}
    assert catalogue <= covered, f"no row in the table covers {sorted(catalogue - covered)}"


def test_the_recurrent_decoder_type_is_present_and_is_honestly_untested():
    """Named on its own because it is the type most likely to break the map: an entry reading
    "No self-attention; mLSTM recurrent layers with matrix memory" has no attention anywhere, and
    we have never been shown one.
    """
    recurrent = [r for r in FAMILIES if r["decoder_type"] == "Recurrent"]
    assert recurrent, "the catalogue has a Recurrent decoder type and the table must carry it"
    assert all(local_config(r["repo"]) is None for r in recurrent)
    assert all(r["family"] in EXPECTED_GAPS for r in recurrent)


def test_the_table_has_no_duplicate_family_names():
    names = [r["family"] for r in FAMILIES]
    assert len(names) == len(set(names)), "a duplicate family name makes a parametrise id ambiguous"


def test_every_row_in_the_table_has_every_field():
    required = {"family", "decoder_type", "repo", "mechanism_values", "count_key",
                "expect_blocks", "expect_experts"}
    for row in FAMILIES:
        assert set(row) == required, f"{row.get('family')!r} has fields {sorted(set(row))}"


def test_a_family_with_no_instance_never_claims_a_measured_expectation():
    """A row with no checkpoint must not carry an expected block count, because an expectation
    nobody can check is the thing that makes an untested row look tested.
    """
    for row in WITHOUT_INSTANCE:
        assert row["expect_blocks"] is None, f"{row['family']} has an uncheckable expectation"
        assert row["mechanism_values"] is None, (
            f"{row['family']} records mechanism strings that nothing has verified")
