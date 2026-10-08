# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""One description of what a model is shaped like, for every architecture this tool meets.

WHY THIS MODULE EXISTS, AND WHY IT IS BUILT THE WAY THE MEASUREMENTS SAY RATHER THAN THE WAY THE
PLAN SAID. The 2026-10-08 plan proposed a map that answers "what are the blocks, what is in each
one, where is the residual stream read". The structure probe in `private/research/probes/` was
written first, on the operator's instruction, and reading twenty-one real checkpoints changed three
things about what this module has to be. Each is recorded beside the code that honours it:

  1. A BLOCK MAY HOLD ONE MECHANISM OR NONE. Nemotron-H's 52 blocks are 24 Mamba, 24 bare MLP and
     4 attention, one mechanism per block. So a record with an `attention` field and an `mlp` field
     cannot describe it, and `BlockKind` below is a single answer with `MLP_ONLY` and `UNKNOWN` as
     real values rather than as the absence of a value.
  2. "WHICH MECHANISM" HAS FOUR INCOMPATIBLE SPELLINGS IN THE CONFIG, one of which is inverted:
     `attn_layer_indices` names only the attention blocks and implies every other index, so read at
     face value it turns a 32-block Bamba into a 3-block model. `mechanisms` resolves all four in
     one place and nowhere else.
  3. THE SAME `model_type` CAN USE TWO DIFFERENT SPELLINGS. Both local Nemotron-H checkpoints are
     `model_type: nemotron_h` and `NemotronHForCausalLM`; the 4B spells its block list
     `hybrid_override_pattern` and the 300M spells it `layers_block_type`. So nothing here keys off
     the model type or the architecture class name. Every resolver tries every spelling and reports
     which one answered.

A fourth measurement shaped the mechanism table specifically. `layer_types` is the same key for two
completely different kinds of difference: on gpt-oss and Gemma-4 its values vary the attention
WINDOW and every block still has attention, while on LFM2 and Qwen3.6 they name a different
MECHANISM and many blocks have none. Nothing in the values is self-describing, so
`MECHANISM_BY_VALUE` is a table from the string to what it means, and a value absent from that
table is `UNKNOWN` with the string preserved rather than guessed at.

WHAT THIS MODULE IS NOT. It does not read weights, it does not import torch, transformers or
anything else heavier than the standard library, and it never builds a model. It answers two
separate questions and keeps them separate:

    from a CONFIG DICT     what the checkpoint declares about itself
    from a LIVE MODEL      what the module tree actually exposes, by getattr alone

Those two can disagree, and when they do that is a finding rather than an error, so `describe`
reports both and lists the disagreements instead of preferring one.

WHAT IT CANNOT ANSWER, said plainly so an absence is never read as a zero. **Where the residual
stream runs is not answerable from a config or a module tree.** That is a claim about the forward
pass. What is reported instead is the norm inventory per block, which is the static shadow the
pattern casts, and `residual_path_why` says in words that it is a hint to go and look rather than a
measurement. Measuring it needs a forward pass on a card.

THE DEGRADATION STYLE IS `residualleak.final_norm`'S, deliberately. An architecture this module does
not recognise comes back with a reason phrased for a person, not an exception and not an empty
list, because an empty list reads like "this model has no blocks".
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

from .writers import ATTN_BLOCKS, MIXER_BLOCKS, MLP_BLOCKS

# ----------------------------------------------------------------------------------------------
# The key spellings. EVERY spelling of EVERY key lives in this section and nowhere else in the
# package. A module that needs one of these imports it from here; a module that re-lists one has
# reintroduced the drift this file closes, and `tests/test_the_map_is_the_only_resolver.py`
# fails when that happens.
# ----------------------------------------------------------------------------------------------

#: Every spelling of "how many decoder blocks". Measured: `num_hidden_layers` on 17 of 21 local
#: checkpoints, `n_layer` on the 3 GPT-2 family members, and ABSENT on NemotronH-300M, whose only
#: statement of depth is the length of its block list.
LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "n_layers", "num_layers",
                    "num_decoder_layers", "n_block")

#: Blocks stored in the checkpoint PAST the declared stack. A multi-token-prediction head is
#: written as `model.layers.N` for N at or beyond the declared count, so a reader trusting the
#: declared number alone sees the last real block as a stray tensor with no home.
#:
#: THIS TERM IS THE DRIFT THE CONSOLIDATION FIXES. `streaming.LAYER_COUNT_KEYS` carried the six
#: spellings above and no extra term, so it was wrong by one block on GLM-4.7-Flash (47 plus 1),
#: DeepSeek-V4-Flash (43 plus 1) and Qwen3.6-35B-A3B (40 plus 1). The vendored conversion code at
#: `vendor/src/conversion/exaone.py` already computed `num_hidden_layers + num_nextn_predict_layers`
#: and was right, which means the only correct arithmetic in the repository lived in the one place
#: a consolidation would have been least likely to read.
EXTRA_BLOCK_KEYS = ("num_nextn_predict_layers", "mtp_num_hidden_layers")

#: Where a multi-tower checkpoint keeps the decoder's own config. Gemma-4-31B carries `text_config`
#: beside `vision_config` and `audio_config`; Qwen3.6-35B carries `text_config` beside
#: `vision_config`. Reading `cfg["num_hidden_layers"]` on either returns nothing, which is not the
#: same as zero blocks.
NESTED_CONFIG_KEYS = ("text_config", "language_model_config", "llm_config", "decoder_config")

#: Per-block mechanism lists, as lists of strings, one entry per block.
BLOCK_TYPE_LIST_KEYS = ("layer_types", "layers_block_type")

#: Nemotron-H's 4B spelling: ONE LETTER PER BLOCK in a single string.
BLOCK_TYPE_PATTERN_KEY = "hybrid_override_pattern"

#: Bamba's spelling, and the inverted one. It names the ATTENTION block indices and says nothing
#: about any other block, so the complement is the family's other mechanism.
BLOCK_TYPE_INDEX_KEY = "attn_layer_indices"

#: Expert-count spellings. Three of them, and a fourth state: Gemma-4-31B carries `num_experts`
#: with a NULL value beside `enable_moe_block: false`, so `"num_experts" in cfg` is true for a
#: dense model and the key's presence proves nothing.
EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")

#: How many experts fire per token. gpt-oss carries two of these three at once.
EXPERT_TOPK_KEYS = ("num_experts_per_tok", "experts_per_token", "top_k_experts", "moe_top_k")

#: Shared always-on experts. NOTE THAT ONE OF THESE IS A WIDTH AND NOT A COUNT:
#: `shared_expert_intermediate_size` is 512 on Qwen3.6-35B, so a reader treating it as "how many
#: shared experts" reports 512 of them. `shared_experts` below carries the value AND the key, and
#: `shared_expert_count` returns None for the width spelling rather than a wrong number.
SHARED_EXPERT_COUNT_KEYS = ("n_shared_experts", "num_shared_experts")
SHARED_EXPERT_WIDTH_KEYS = ("shared_expert_intermediate_size", "moe_shared_expert_intermediate_size")

#: Which blocks are NOT routed, with opposite polarity between the two spellings.
#: `num_dense_layers` is a COUNT of leading dense blocks (LFM2.5-8B-A1B says 2, so blocks 0 and 1
#: are dense); `mlp_only_layers` is a LIST of dense block indices (Qwen3-30B-A3B says `[]`, so
#: every one of its 48 blocks is routed). On that pair an empty list and a count of zero happen to
#: agree, which is exactly the coincidence that would let a polarity bug ship.
DENSE_PREFIX_KEY = "num_dense_layers"
DENSE_INDEX_KEY = "mlp_only_layers"

#: Where the decoder stack hangs, as a dotted attribute path. Was `cli._decoder_layers`' private
#: tuple. Two of these four are exercised by no checkpoint in the local cache, which is recorded
#: in the conformance suite as untested rather than left to look like coverage.
DECODER_STACK_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers")

#: The module the stack and the final norm are both children of. Was `residualleak.BASE_MODEL_PATHS`.
#:
#: THIS IS A DIFFERENT QUESTION FROM `DECODER_STACK_PATHS` AND THE STRINGS ARE NOT DERIVABLE FROM
#: EACH OTHER, which is why both lists survive the consolidation rather than one being deleted.
#: `DECODER_STACK_PATHS` answers "where are the blocks"; this answers "which module are they a child
#: of", and the final norm is a sibling of the stack rather than of the stack's entries. Keeping
#: both HERE, in one module, is what the consolidation buys: previously they sat in two files, and
#: the comment in the second one recorded that a list had already drifted once.
BASE_MODEL_PATHS = ("model", "transformer", "gpt_neox", "model.decoder")

#: Attribute names that hold the final norm. Was `residualleak.FINAL_NORM_NAMES`.
FINAL_NORM_NAMES = ("norm", "final_layernorm", "final_layer_norm", "ln_f", "final_norm")

#: Attribute names that hold a decoder stack, read off a resolved base module.
STACK_ATTRIBUTES = ("layers", "h", "blocks")


class BlockKind(enum.Enum):
    """What one block's sequence-mixing position holds. A SINGLE answer, by measurement.

    `MLP_ONLY` and `UNKNOWN` are values rather than the absence of one, which is requirement 1 in
    this module's docstring. A Nemotron-H block that holds nothing but a feed-forward network is
    `MLP_ONLY`, and that is a fact about the model, not a gap in the reading.
    """

    #: Ordinary attention, whatever its window.
    ATTENTION = "attention"
    #: A non-attention sequence mixer: a short convolution, a Mamba block, a gated delta net.
    MIXER = "mixer"
    #: Both in parallel in the same block.
    BOTH = "both"
    #: No sequence mixer at all. Nemotron-H's `-` blocks, 24 of its 52.
    MLP_ONLY = "mlp_only"
    #: The source said something, and this module has no meaning for what it said. The raw string
    #: is kept beside it so a reader can see what it was rather than a shrug.
    UNKNOWN = "unknown"


#: What each per-block mechanism STRING means. The table exists because `layer_types` is the same
#: key for two different kinds of difference, measured on the local cache:
#:
#:     gpt-oss-20b    12 sliding_attention, 12 full_attention   every block HAS attention
#:     gemma-4-31b    50 sliding_attention, 10 full_attention   every block HAS attention
#:     LFM2.5-1.2B    10 conv, 6 full_attention                 10 of 16 have NO attention
#:     Qwen3.6-35B    30 linear_attention, 10 full_attention    30 of 40 have NO attention
#:
#: `linear_attention` is the trap in that list: the name says attention and the mechanism is
#: state-space, which its own config admits by carrying `linear_conv_kernel_dim` and
#: `mamba_ssm_dtype` beside it. So the window variants map to ATTENTION and the mechanism variants
#: map to MIXER, and a string absent from this table maps to UNKNOWN rather than to a guess.
MECHANISM_BY_VALUE = {
    "attention": BlockKind.ATTENTION,
    "full_attention": BlockKind.ATTENTION,
    "sliding_attention": BlockKind.ATTENTION,
    "chunked_attention": BlockKind.ATTENTION,
    "global_attention": BlockKind.ATTENTION,
    "linear_attention": BlockKind.MIXER,
    "conv": BlockKind.MIXER,
    "short_conv": BlockKind.MIXER,
    "mamba": BlockKind.MIXER,
    "recurrent": BlockKind.MIXER,
    "mlp": BlockKind.MLP_ONLY,
    "moe": BlockKind.MLP_ONLY,
}

#: Nemotron-H's one-letter block codes, from `hybrid_override_pattern`. Measured on the 4B:
#: `M-M-M-M*-...`, 24 `M`, 24 `-` and 4 `*`.
PATTERN_LETTERS = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}

#: What the inverted spelling says about a block it does not mark. Which is nothing: the key names
#: the attention blocks and the family's other mechanism is implied by the family rather than
#: stated. Given a name so it reads as a state in the output instead of an empty string.
UNNAMED_BY_INVERTED_KEY = "not-named-by-attn_layer_indices"


# ----------------------------------------------------------------------------------------------
# Described structures. Named fields, not torch modules, so everything downstream reads one shape.
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    """One answer, the key or path that produced it, and a reason when there is no answer.

    EVERY RESOLVER IN THIS MODULE RETURNS THIS, because the measurements made the provenance part
    of the answer: "48 blocks" is a different claim from "48 blocks, from `num_hidden_layers`, plus
    1 from `num_nextn_predict_layers`", and six months later only the second one can be argued
    with. `why` is populated exactly when `value` is None, and it is written for a person.
    """

    value: object = None
    source: str | None = None
    why: str | None = None

    @property
    def known(self):
        """Whether this resolved to anything. `not known` is never the same as a zero value."""
        return self.value is not None

    def unwrap(self, default=None):
        """The value, or `default` when unknown. For a caller that genuinely has a fallback."""
        return self.value if self.known else default


@dataclass(frozen=True)
class BlockCount:
    """How many blocks there are, split into what was declared and what sits past it."""

    declared: Resolved
    extra: Resolved
    total: Resolved


@dataclass(frozen=True)
class Mechanisms:
    """One `BlockKind` per block, the key that said so, and the raw strings it said it with."""

    kinds: Resolved                   # list[BlockKind], or unknown with a reason
    raw: tuple = ()                   # the source strings, one per block, for a reader
    unmapped: tuple = ()              # sorted raw values this module has no meaning for

    def tally(self):
        """How many blocks of each kind, as a plain dict keyed by the kind's string value."""
        if not self.kinds.known:
            return {}
        out = {}
        for kind in self.kinds.value:
            out[kind.value] = out.get(kind.value, 0) + 1
        return dict(sorted(out.items()))

    def unknown_blocks(self):
        """How many blocks this module could not interpret. `None` when nothing was read at all."""
        if not self.kinds.known:
            return None
        return sum(1 for k in self.kinds.value if k is BlockKind.UNKNOWN)

    def without_attention(self):
        """How many blocks hold no attention at all, or `None` when that cannot be said.

        None rather than 0, because "this stack has no attention-free blocks" and "nobody knows
        whether it has any" are different claims and a map that returned 0 for both would have
        told the caller the first one.

        THE SECOND GUARD WAS A REAL WRONG ANSWER AND NOT A HYPOTHETICAL. Running this against
        Bamba-9B, whose spelling is the inverted `attn_layer_indices`, 3 blocks resolve to
        attention and the other 29 resolve to UNKNOWN because the key says nothing about them.
        Counting only MIXER and MLP_ONLY returned 0, which reads as "every one of its 32 blocks
        has attention" when the truth is that 29 of them are unread. So any UNKNOWN block makes
        this unanswerable, and `unknown_blocks` says how many.
        """
        if not self.kinds.known:
            return None
        if any(k is BlockKind.UNKNOWN for k in self.kinds.value):
            return None
        return sum(1 for k in self.kinds.value
                   if k in (BlockKind.MIXER, BlockKind.MLP_ONLY))


@dataclass(frozen=True)
class Experts:
    """Routed experts, shared experts, and which blocks are dense. Every field with its key."""

    routed: Resolved
    per_token: Resolved
    shared_count: Resolved
    shared_width: Resolved
    dense_blocks: Resolved
    #: Count keys present with a NULL value. Neither a count nor an absent key, and the reason
    #: `routed.known` is False on a Gemma-4 whose config does mention experts.
    declared_but_null: tuple = ()

    @property
    def is_mixture(self):
        """Whether this is a mixture of experts, decided by a COUNT and never by a key's presence."""
        return self.routed.known and int(self.routed.value) > 1


@dataclass(frozen=True)
class Heads:
    """Attention head arithmetic, including whether the usual identity holds at all."""

    hidden_size: Resolved
    attention_heads: Resolved
    key_value_heads: Resolved
    head_dim: Resolved
    #: True when `hidden_size / attention_heads == head_dim`, False when it does not, and None
    #: when one of the three is missing so the question is unanswered rather than answered no.
    #: Measured False on 6 of 21 local checkpoints.
    quotient_is_head_dim: bool | None = None
    #: Separate query and value widths, where a single head width does not exist. GLM-4.7-Flash
    #: declares `qk_nope_head_dim` 192, `qk_rope_head_dim` 64 and `v_head_dim` 256.
    split_widths: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelMap:
    """One description of one model. The shape everything downstream reads."""

    #: Where the description came from: "config", "modules", or "config+modules".
    evidence: str
    model_type: Resolved
    architecture: Resolved
    #: The key the decoder's config was nested under, or unknown when the top level is it.
    nested_under: Resolved
    blocks: BlockCount
    mechanisms: Mechanisms
    experts: Experts
    heads: Heads
    #: Live-module facts. Unknown when `describe` was given no model.
    stack_path: Resolved
    stack_length: Resolved
    base_module_path: Resolved
    final_norm_name: Resolved
    #: Per-block kinds read from the live modules rather than from the config. The two can
    #: disagree and `disagreements` says so.
    module_kinds: Resolved
    #: Why the residual stream's actual path is not in this record. Always populated.
    residual_path_why: str = ""
    #: Places the config and the modules do not agree. Each entry is a finding, not an error.
    disagreements: tuple = ()

    def unanswerable(self):
        """The questions this map cannot answer about this model, each with the reason.

        WHY A MAP HAS TO PUBLISH THIS RATHER THAN JUST RETURNING None PER FIELD. A caller reading
        one field at a time meets a None and has to guess whether it means zero, missing or
        unread. A viewer drawing a panel per block needs to know BEFORE it draws that the
        mechanism column is unreadable for this checkpoint. And the failure that matters most is
        the quiet one: a map that met an unrecognised mechanism string and filed it under
        attention would answer every question confidently and wrongly, which is strictly worse
        than one that refuses, because a refusal is readable and a guess is not.

        Returns a dict of question to reason, empty when everything asked of this record can be
        answered. The keys are phrased as the questions a caller actually has.
        """
        out = {}
        if not self.blocks.total.known:
            out["how many blocks does this model have"] = self.blocks.total.why
        if not self.mechanisms.kinds.known:
            out["what mechanism does each block hold"] = self.mechanisms.kinds.why
        elif self.mechanisms.unmapped:
            unknown_at = [i for i, k in enumerate(self.mechanisms.kinds.value)
                          if k is BlockKind.UNKNOWN]
            reason = (
                f"{len(unknown_at)} of {len(self.mechanisms.kinds.value)} blocks are declared "
                f"under {self.mechanisms.kinds.source} with values this tool has no meaning for "
                f"({list(self.mechanisms.unmapped)}), at block indices "
                f"{unknown_at[:8]}{'...' if len(unknown_at) > 8 else ''}. They are recorded as "
                f"unknown rather than guessed. Adding those strings to MECHANISM_BY_VALUE, with "
                f"a measurement behind each one, is how this becomes answerable.")
            out["what mechanism does each block hold"] = reason
            out["how many blocks have no attention"] = reason
            out["can a per-block attention panel be drawn"] = reason
        if self.mechanisms.kinds.known and self.mechanisms.without_attention() is None:
            out.setdefault("how many blocks have no attention",
                           "some blocks' mechanism is unread, so this cannot be counted")
        if not self.heads.head_dim.known:
            out["how wide is one attention head"] = self.heads.head_dim.why
        if self.heads.quotient_is_head_dim is False:
            out["how wide is one attention head"] = (
                f"hidden_size over num_attention_heads is "
                f"{self.heads.hidden_size.unwrap('?')} / {self.heads.attention_heads.unwrap('?')} "
                f"and the declared head_dim is {self.heads.head_dim.value}, so the usual identity "
                f"does not hold and the quotient is not the head width"
                + (f"; separate widths are declared under {sorted(self.heads.split_widths)}"
                   if self.heads.split_widths else ""))
        out["where does the residual stream run"] = self.residual_path_why
        return out


# ----------------------------------------------------------------------------------------------
# Reading a config. Pure functions over a dict.
# ----------------------------------------------------------------------------------------------


def _first(cfg, keys):
    """The first of `keys` present with a non-None value, and which key it was.

    None-valued counts as absent deliberately: a config can carry `num_experts: null` to mean the
    architecture supports experts and this checkpoint configures none, which is a third state. The
    caller reports it as such rather than letting it read as either a count or a missing key.
    """
    for key in keys:
        if isinstance(cfg, dict) and cfg.get(key) is not None:
            return cfg[key], key
    return None, None


def decoder_scope(cfg):
    """The part of a config that describes the decoder, and the key it was nested under.

    Returns `(scope, Resolved)`. A sub-config only wins when it actually declares a depth, so a
    key that merely ends in `_config` cannot hijack a perfectly good top level.
    """
    if not isinstance(cfg, dict):
        return {}, Resolved(why=f"the config is a {type(cfg).__name__} and not an object, so "
                                f"nothing about this model can be read from it")
    for key in NESTED_CONFIG_KEYS:
        sub = cfg.get(key)
        if isinstance(sub, dict) and any(sub.get(k) is not None for k in LAYER_COUNT_KEYS):
            return sub, Resolved(value=key, source=key)
    return cfg, Resolved(why="the top-level config is the decoder's own config; it is not nested")


def block_count(cfg):
    """The block count, split into the declared stack and anything stored past it.

    The total is what a reader of tensor names has to expect, and it is the declared depth plus the
    multi-token-prediction blocks. A config that declares no depth at all falls through to the
    length of its mechanism list rather than being refused, because NemotronH-300M is exactly that
    and this module can describe it perfectly well.
    """
    scope, _ = decoder_scope(cfg)
    declared, key = _first(scope, LAYER_COUNT_KEYS)
    extra, extra_key = _first(scope, EXTRA_BLOCK_KEYS)
    extra_res = (Resolved(value=int(extra), source=extra_key) if extra is not None
                 else Resolved(value=0, source=None,
                               why=None))
    if declared is None:
        mech = mechanisms(cfg)
        if mech.kinds.known:
            return BlockCount(
                declared=Resolved(why=f"no key of {list(LAYER_COUNT_KEYS)} is present in this "
                                      f"config, so it never states a stack depth"),
                extra=extra_res,
                total=Resolved(value=len(mech.kinds.value), source=mech.kinds.source,
                               why=None))
        why = (f"this config states no stack depth: none of {list(LAYER_COUNT_KEYS)} is present "
               f"and there is no per-block mechanism list to take a length from either. Either "
               f"the depth lives under a spelling this tool has not met, or the file is not a "
               f"decoder config.")
        return BlockCount(declared=Resolved(why=why), extra=extra_res, total=Resolved(why=why))
    declared = int(declared)
    return BlockCount(
        declared=Resolved(value=declared, source=key),
        extra=extra_res,
        total=Resolved(value=declared + int(extra or 0),
                       source=key if not extra else f"{key} + {extra_key}"))


def mechanisms(cfg):
    """One `BlockKind` per block, resolved across all four spellings of the list.

    The order of the attempts is not arbitrary. The two list spellings are unambiguous, the letter
    pattern is unambiguous once decoded, and `attn_layer_indices` is tried LAST because its sense
    is inverted and it needs a depth from elsewhere to be interpretable at all.
    """
    scope, _ = decoder_scope(cfg)
    for key in BLOCK_TYPE_LIST_KEYS:
        got = scope.get(key)
        if isinstance(got, list) and got and all(isinstance(x, str) for x in got):
            return _mechanisms_from_strings(tuple(got), key)
    pattern = scope.get(BLOCK_TYPE_PATTERN_KEY)
    if isinstance(pattern, str) and pattern:
        unknown = sorted(set(pattern) - set(PATTERN_LETTERS))
        if unknown:
            return Mechanisms(
                kinds=Resolved(source=BLOCK_TYPE_PATTERN_KEY,
                               why=f"{BLOCK_TYPE_PATTERN_KEY} is {pattern!r} and holds the "
                                   f"letters {unknown}, which this tool has no meaning for. The "
                                   f"letters it knows are {sorted(PATTERN_LETTERS)}. The block "
                                   f"kinds are unread rather than guessed."),
                raw=tuple(pattern), unmapped=tuple(unknown))
        # `raw` keeps the LETTERS and not the decoded names, because the field is the source
        # strings and a reader comparing a record against the config file has to find the same
        # characters in both. The kinds come from the decoded names. The conformance suite caught
        # this the other way round: its family table recorded the letters, the record held the
        # decoded names, and one of the two was wrong about what `raw` means.
        decoded = _mechanisms_from_strings(
            tuple(PATTERN_LETTERS[ch] for ch in pattern), BLOCK_TYPE_PATTERN_KEY)
        return Mechanisms(kinds=decoded.kinds, raw=tuple(pattern), unmapped=decoded.unmapped)
    marked = scope.get(BLOCK_TYPE_INDEX_KEY)
    depth, _ = _first(scope, LAYER_COUNT_KEYS)
    if isinstance(marked, list) and depth is not None:
        # THE INVERTED SPELLING. The key names the attention blocks and says nothing about the
        # rest, so taking the list at face value as "the block kinds" reports a 32-block Bamba as
        # a 3-block model. Every unmarked index is the family's other mechanism, and this module
        # will not pretend to know which one it is: those blocks are UNKNOWN with a reason, not
        # silently labelled Mamba because Bamba happens to be the checkpoint in hand.
        marked = {int(i) for i in marked}
        kinds = [BlockKind.ATTENTION if i in marked else BlockKind.UNKNOWN
                 for i in range(int(depth))]
        return Mechanisms(
            kinds=Resolved(value=kinds, source=BLOCK_TYPE_INDEX_KEY),
            raw=tuple("attention" if i in marked else UNNAMED_BY_INVERTED_KEY
                      for i in range(int(depth))),
            unmapped=(UNNAMED_BY_INVERTED_KEY,))
    return Mechanisms(
        kinds=Resolved(why=f"this config carries no per-block mechanism list: looked for "
                           f"{list(BLOCK_TYPE_LIST_KEYS)}, {BLOCK_TYPE_PATTERN_KEY} and "
                           f"{BLOCK_TYPE_INDEX_KEY}. The stack may be uniform, or it may be "
                           f"hybrid under a spelling this tool has not met, and those two are "
                           f"not distinguishable from the config alone."))


def _mechanisms_from_strings(raw, key):
    """Strings to kinds through `MECHANISM_BY_VALUE`, keeping whatever it could not map."""
    kinds = [MECHANISM_BY_VALUE.get(value, BlockKind.UNKNOWN) for value in raw]
    unmapped = sorted({v for v in raw if v not in MECHANISM_BY_VALUE})
    return Mechanisms(kinds=Resolved(value=kinds, source=key), raw=raw, unmapped=tuple(unmapped))


def experts(cfg):
    """Routed and shared experts, with the key behind every number.

    The width spelling of "shared experts" is kept in its own field. `shared_expert_intermediate_
    size` is 512 on Qwen3.6-35B, and a reader that treated it as a count would report 512 shared
    experts, so `shared_count` stays unknown with a reason when only the width is declared.
    """
    scope, _ = decoder_scope(cfg)
    count, count_key = _first(scope, EXPERT_COUNT_KEYS)
    topk, topk_key = _first(scope, EXPERT_TOPK_KEYS)
    shared, shared_key = _first(scope, SHARED_EXPERT_COUNT_KEYS)
    width, width_key = _first(scope, SHARED_EXPERT_WIDTH_KEYS)
    nulls = tuple(sorted(k for k in EXPERT_COUNT_KEYS
                         if isinstance(scope, dict) and k in scope and scope[k] is None))

    prefix, _ = _first(scope, (DENSE_PREFIX_KEY,))
    listed = scope.get(DENSE_INDEX_KEY) if isinstance(scope, dict) else None
    if prefix is not None:
        dense = Resolved(value=list(range(int(prefix))), source=DENSE_PREFIX_KEY)
    elif isinstance(listed, list):
        dense = Resolved(value=[int(i) for i in listed], source=DENSE_INDEX_KEY)
    else:
        dense = Resolved(why=f"this config does not say which blocks are dense: neither "
                             f"{DENSE_PREFIX_KEY} (a count of leading dense blocks) nor "
                             f"{DENSE_INDEX_KEY} (a list of dense block indices) is present")

    if count is None:
        why = (f"no expert count is declared under any of {list(EXPERT_COUNT_KEYS)}"
               + (f", though {list(nulls)} is present with a null value, which means this "
                  f"architecture supports experts and this checkpoint configures none"
                  if nulls else ""))
        routed = Resolved(why=why)
    else:
        routed = Resolved(value=int(count), source=count_key)

    return Experts(
        routed=routed,
        per_token=(Resolved(value=int(topk), source=topk_key) if topk is not None
                   else Resolved(why=f"no top-k is declared under any of "
                                     f"{list(EXPERT_TOPK_KEYS)}")),
        shared_count=(Resolved(value=int(shared), source=shared_key) if shared is not None
                      else Resolved(why=(f"no shared-expert COUNT is declared; "
                                         f"{width_key} is present but it is a width, not a count"
                                         if width is not None else
                                         f"no shared experts are declared under any of "
                                         f"{list(SHARED_EXPERT_COUNT_KEYS)}"))),
        shared_width=(Resolved(value=int(width), source=width_key) if width is not None
                      else Resolved(why="no shared-expert width is declared")),
        dense_blocks=dense,
        declared_but_null=nulls)


def heads(cfg):
    """Head arithmetic, with the usual identity reported rather than assumed."""
    scope, _ = decoder_scope(cfg)
    hidden, hidden_key = _first(scope, ("hidden_size", "n_embd", "d_model"))
    n_heads, heads_key = _first(scope, ("num_attention_heads", "n_head"))
    kv, kv_key = _first(scope, ("num_key_value_heads", "num_kv_heads"))
    dim, dim_key = _first(scope, ("head_dim",))

    holds = None
    if isinstance(hidden, int) and isinstance(n_heads, int) and n_heads and dim is not None:
        holds = abs(hidden / n_heads - float(dim)) < 1e-9
    split = {k: scope[k] for k in sorted(scope) if k in (
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "global_head_dim",
        "index_head_dim", "linear_key_head_dim", "linear_value_head_dim",
        "kv_lora_rank", "q_lora_rank")} if isinstance(scope, dict) else {}

    def res(value, key, what):
        return (Resolved(value=value, source=key) if value is not None
                else Resolved(why=f"this config does not declare {what}"))

    return Heads(
        hidden_size=res(hidden, hidden_key, "a hidden size"),
        attention_heads=res(n_heads, heads_key, "an attention head count"),
        key_value_heads=res(kv, kv_key, "a key-value head count"),
        head_dim=res(dim, dim_key, "a head width, so it cannot be compared with the quotient"),
        quotient_is_head_dim=holds,
        split_widths=split)


# ----------------------------------------------------------------------------------------------
# Reading a live model. `getattr` only, so this module still imports nothing heavy.
# ----------------------------------------------------------------------------------------------


def _walk(obj, path):
    """Follow a dotted attribute path, or None at the first missing step."""
    for attr in path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def decoder_stack(model, *, paths=None):
    """The decoder blocks and the path they were found under.

    Returns `(stack_or_None, Resolved)`. Callers that cannot continue without a stack raise on
    `not resolved.known` and put `resolved.why` in front of the user; `stack_or_raise` below does
    exactly that so no caller has to write the sentence.

    `paths` exists so a caller can pass its own re-exported copy of `DECODER_STACK_PATHS`, which
    keeps a test that monkeypatches that caller's module-level constant working against ONE
    implementation. Without it, consolidation would silently neuter those tests: the patch would
    still apply and the resolver would no longer read it, so the test would pass while measuring
    nothing. Default None means this module's own list.

    The phrase "decoder layer stack" is the one `cli._decoder_layers` has always raised and it is
    user-facing, so it is preserved verbatim even though this module otherwise says "block".
    """
    for path in (paths if paths is not None else DECODER_STACK_PATHS):
        found = _walk(model, path)
        if found is not None:
            return found, Resolved(value=path, source=path)
    tried = paths if paths is not None else DECODER_STACK_PATHS
    return None, Resolved(
        why=f"could not find the decoder layer stack on {type(model).__name__}; looked for "
            f"{', '.join(tried)}. Either this architecture arranges its modules some other way, "
            f"or the object passed in is not a causal language model.")


def stack_or_raise(model, *, paths=None):
    """The decoder stack, or a ValueError carrying the reason. The old `cli._decoder_layers`.

    Kept as its own function because the editor genuinely cannot proceed without the stack, so for
    that caller the degradation is a loud failure at load rather than a reason to carry. Everything
    that CAN carry on calls `decoder_stack` and reads the reason.
    """
    stack, resolved = decoder_stack(model, paths=paths)
    if stack is None:
        raise ValueError(resolved.why)
    return stack


def base_module(model, *, paths=None):
    """The module the stack and the final norm are both children of, and its path.

    A candidate only wins when it actually holds a stack, which is what stops a wrapper attribute
    called `model` from being mistaken for the base when the real one is further down. The walk
    keeps going past a resolving-but-empty attribute rather than stopping there, because stopping
    would report "no final norm found" on a model whose norm is one path further along.

    `paths` is injectable for the reason given on `decoder_stack`.
    """
    tried = paths if paths is not None else BASE_MODEL_PATHS
    for path in tried:
        found = _walk(model, path)
        if found is None:
            continue
        if any(getattr(found, attr, None) is not None for attr in STACK_ATTRIBUTES):
            return found, Resolved(value=path, source=path)
    return None, Resolved(
        why=f"the model tree on {type(model).__name__} does not expose a base module this tool "
            f"recognises, so its final norm could not be found. The paths looked for were "
            f"{', '.join(tried)}.")


def final_norm(model, *, names=None, paths=None):
    """The final norm module and the attribute it was found under.

    Returns `(module_or_None, Resolved)`. A name this tool does not know is a DEGRADATION and not
    a failure, in the style this module inherits: the reason comes back phrased for a person so
    the caller can show it rather than invent a sentence.

    `names` and `paths` are injectable for the reason given on `decoder_stack`: the leak metric
    re-exports `FINAL_NORM_NAMES`, a test monkeypatches that re-export, and one implementation has
    to keep honouring it or the test silently stops testing anything.
    """
    base, where = base_module(model, paths=paths)
    if base is None:
        return None, where
    looked = names if names is not None else FINAL_NORM_NAMES
    for name in looked:
        mod = getattr(base, name, None)
        if mod is not None and getattr(mod, "weight", None) is not None:
            return mod, Resolved(value=name, source=name)
    return None, Resolved(
        why=f"no final norm with a learned weight was found on {type(base).__name__}; the names "
            f"looked for were {', '.join(looked)}. Either this architecture normalises somewhere "
            f"else or it names it something new, and either way the output basis cannot be "
            f"worked out.")


def block_outproj_param(block):
    """A block's `out_proj` module when it carries a two-dimensional weight, else None.

    The structural half of the mixer test, exposed because `cli` asks it of a named child directly
    rather than over a position list. The rank is read off the parameter without materialising it,
    because the architecture probe builds models on the meta device precisely so that a question
    about shape needs no storage to answer.
    """
    if block is None:
        return None
    found = getattr(block, "out_proj", None)
    weight = getattr(found, "weight", None) if found is not None else None
    if weight is None or getattr(weight, "dim", None) is None or weight.dim() != 2:
        return None
    return found


def block_kind(layer, *, count_mixers=True):
    """What one LIVE block holds, as a `BlockKind`, read off the modules rather than the config.

    This is the module-tree counterpart of `mechanisms`, and the two exist separately on purpose:
    a config can say one thing and a tree another, and `describe` reports the disagreement. The
    position lists come from `writers.py` rather than being re-listed, because that module is
    already the single home for which child names sit where, and it says why in its own docstring.

    `count_mixers=False` mirrors the `--skip-conv-ablation` control arm, where the convolution is
    deliberately left alone, so a block whose only mixer is that convolution reads as MLP-only.
    """
    has_attention = _is_attention(layer)
    has_mixer = count_mixers and mixer_outproj(layer) is not None
    if has_attention and has_mixer:
        return BlockKind.BOTH
    if has_attention:
        return BlockKind.ATTENTION
    if has_mixer:
        return BlockKind.MIXER
    if any(getattr(layer, name, None) is not None for name in MLP_BLOCKS):
        return BlockKind.MLP_ONLY
    # Not a shrug. A block holding none of the three positions is a block this tool has never met,
    # and saying so is more useful than filing it under the position that happens to be checked
    # last. `describe` surfaces it in `disagreements` so it reaches a person.
    return BlockKind.UNKNOWN


#: The attention output projection's spellings, in the order the editor tries them.
ATTENTION_PROJECTIONS = ("o_proj", "out_proj", "dense")


def attention_block(layer):
    """The attention block on this layer and its output projection, or `(None, None)`.

    THIS IS THE SAME PREDICATE AS `cli._attn_block` AND THAT IS THE POINT, NOT A COINCIDENCE. The
    editor decides "does this layer attend" by looking for an attention-named child carrying a
    two-dimensional output projection, and if the map answered that question any other way then
    the map and the editor would disagree about the same model, which is the exact drift this
    module exists to close. So the semantics are reproduced here, in a module that imports nothing
    heavier than the standard library, and
    `test_the_map_and_the_editor_agree_about_which_layers_attend` compares the two on real trees so
    that a divergence fails a test rather than shipping.

    It is HERE rather than imported from `cli` because `cli` pulls in torch, optuna and
    transformers at module scope, and this module is meant to cost nothing to import. The
    consolidation goes the other way when `cli.py` is free: `cli._attn_block` becomes a call to
    this, and then there is one copy again.

    Three subtleties, each of which the editor's own comments record as having been paid for:

      * `ATTN_BLOCKS` is ordered and `mixer` is last, so a layer carrying both a conventional
        `self_attn` and something called `mixer` resolves to the conventional one.
      * the rank is read off the parameter directly, because this is a structural question and
        the architecture probe builds models on the meta device precisely so that it needs no
        weights. Making "does this layer attend" depend on resident storage broke that once.
      * `out_proj` on a child called `mixer` is the Mamba-2 case and belongs to the mixer path,
        so it is skipped here. NemotronH is the architecture where one child name means four
        things.
    """
    for name in ATTN_BLOCKS:
        block = getattr(layer, name, None)
        if block is None:
            continue
        for projection in ATTENTION_PROJECTIONS:
            found = getattr(block, projection, None)
            weight = getattr(found, "weight", None) if found is not None else None
            if weight is None or getattr(weight, "dim", None) is None or weight.dim() != 2:
                continue
            if name == "mixer" and projection != "o_proj":
                continue
            return block, found
    return None, None


def _is_attention(layer):
    """Whether this layer attends, by the editor's own test."""
    return attention_block(layer)[1] is not None


def mixer_outproj(layer):
    """A non-attention sequence mixer's output projection on this layer, or None.

    The counterpart of `attention_block`, and the same predicate as `cli._block_outproj_param`
    applied over `MIXER_BLOCKS`. A block whose writer is some other rank is not a mixer this tool
    can describe, and accepting it would produce a confident wrong answer rather than a refusal.
    """
    for name in MIXER_BLOCKS:
        found = block_outproj_param(getattr(layer, name, None))
        if found is not None:
            return found
    return None


def block_norms(layer):
    """The normalisation attributes this block exposes, sorted. The residual pattern's shadow.

    THIS IS NOT A MEASUREMENT OF THE RESIDUAL STREAM and the map says so in words. A block holding
    an `input_layernorm` and a `post_attention_layernorm` and nothing else is pre-norm by the usual
    convention; a block holding a third norm is doing something the convention does not cover. Both
    of those are hints to go and look with a forward pass, not conclusions.
    """
    found = []
    for name in dir(layer):
        if name.startswith("_"):
            continue
        low = name.lower()
        if "norm" not in low and not low.startswith("ln"):
            continue
        mod = getattr(layer, name, None)
        if mod is not None and getattr(mod, "weight", None) is not None:
            found.append(name)
    return sorted(found)


# ----------------------------------------------------------------------------------------------
# The whole description.
# ----------------------------------------------------------------------------------------------

#: Why the residual stream's real path is absent from every map this module produces. Stated on
#: every record rather than in documentation, so a caller reading one field at a time still meets
#: it. See the module docstring.
RESIDUAL_PATH_WHY = (
    "where the residual stream actually runs is not readable from a config or a module tree: it "
    "is a claim about the forward pass and nothing here runs one. What this record carries "
    "instead is the per-block norm inventory, which is the static shadow the pattern casts. Treat "
    "it as a hint to go and look, never as a measurement. Settling it needs a forward pass on a "
    "card.")


def _config_dict(cfg):
    """A config object as a plain dict, however it chooses to expose itself.

    THREE SHAPES, AND ONLY THE FIRST IS THE LIBRARY'S OWN. A `transformers` config has `to_dict`.
    A hand-built stand-in, a wrapper, or a config loaded from somewhere else is often just an
    object with attributes and no `to_dict` at all, and the first version of this function called
    `getattr(cfg, "to_dict", dict)()` and quietly produced an empty dict for that case. The record
    then said "this config states no stack depth" about a model whose depth was sitting right
    there on its config object, which is the exact failure mode this module is supposed to catch
    in other people's readers.
    """
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        got = to_dict()
        if isinstance(got, dict):
            return got
    attrs = getattr(cfg, "__dict__", None)
    if isinstance(attrs, dict):
        return {k: v for k, v in attrs.items() if not k.startswith("_")}
    return {}


def describe(model=None, config=None, *, count_mixers=True):
    """One `ModelMap` from a config, a live model, or both.

    Giving both is the useful case and the reason `disagreements` exists: the config's claim and
    the module tree's are two independent accounts of the same thing, and where they differ that
    is a finding about the checkpoint rather than an error in the reading. Neither is preferred.
    """
    if model is None and config is None:
        raise ValueError("describe() needs a config, a model, or both; it was given neither, and "
                         "there is nothing to describe.")
    if config is None:
        config = _config_dict(getattr(model, "config", None))
    if not isinstance(config, dict):
        # NOT COERCED TO EMPTY SILENTLY. A caller that passed something odd gets a record whose
        # config half is honestly unknown with a reason, via `decoder_scope`, rather than one that
        # looks like a model with no layers.
        config = {}

    _scope, nested = decoder_scope(config)
    counts = block_count(config)
    mech = mechanisms(config)

    stack_path = Resolved(why="no model was given, so the module tree was not read")
    stack_len = stack_path
    base_path = stack_path
    norm_name = stack_path
    module_kinds = stack_path
    disagreements = []

    if model is not None:
        stack, stack_path = decoder_stack(model)
        if stack is None:
            stack_len = Resolved(why=stack_path.why)
            module_kinds = Resolved(why=stack_path.why)
        else:
            stack_len = Resolved(value=len(stack), source=stack_path.value)
            kinds = [block_kind(layer, count_mixers=count_mixers) for layer in stack]
            module_kinds = Resolved(value=kinds, source=stack_path.value)
            unknown = [i for i, k in enumerate(kinds) if k is BlockKind.UNKNOWN]
            if unknown:
                disagreements.append(
                    f"blocks {unknown[:8]}{'...' if len(unknown) > 8 else ''} expose none of the "
                    f"attention, mixer or MLP positions this tool knows, so what they hold is "
                    f"unread rather than absent")
            if counts.total.known and len(stack) != counts.total.value:
                disagreements.append(
                    f"the config says {counts.total.value} blocks (from "
                    f"{counts.total.source}) and the module tree holds {len(stack)}")
            if mech.kinds.known and len(mech.kinds.value) == len(kinds):
                differing = [i for i, (a, b) in enumerate(zip(mech.kinds.value, kinds, strict=True))
                             if a is not b and BlockKind.UNKNOWN not in (a, b)]
                if differing:
                    disagreements.append(
                        f"the config's {mech.kinds.source} and the module tree disagree about "
                        f"blocks {differing[:8]}{'...' if len(differing) > 8 else ''}")
        _, base_path = base_module(model)
        _, norm_name = final_norm(model)

    if mech.unmapped:
        disagreements.append(
            f"the per-block list under {mech.kinds.source} holds values this tool has no meaning "
            f"for: {list(mech.unmapped)}. Those blocks are recorded as unknown rather than "
            f"guessed, and adding them to MECHANISM_BY_VALUE is how a new family is supported.")

    archs = config.get("architectures")
    return ModelMap(
        evidence=("config+modules" if model is not None and config else
                  "modules" if model is not None else "config"),
        model_type=(Resolved(value=config.get("model_type"), source="model_type")
                    if config.get("model_type") else
                    Resolved(why="this config names no model_type")),
        architecture=(Resolved(value=archs[0], source="architectures")
                      if isinstance(archs, list) and archs else
                      Resolved(why="this config names no architectures, so nothing about it can "
                                   "be resolved by class name")),
        nested_under=nested,
        blocks=counts,
        mechanisms=mech,
        experts=experts(config),
        heads=heads(config),
        stack_path=stack_path,
        stack_length=stack_len,
        base_module_path=base_path,
        final_norm_name=norm_name,
        module_kinds=module_kinds,
        residual_path_why=RESIDUAL_PATH_WHY,
        disagreements=tuple(disagreements))


def summarise(mapping):
    """The map as lines a person can read, one fact per line, every number with its key."""
    lines = []
    blocks, mech, exp = mapping.blocks, mapping.mechanisms, mapping.experts
    lines.append(f"MAP_MODEL type={mapping.model_type.unwrap('unknown')} "
                 f"arch={mapping.architecture.unwrap('unknown')} evidence={mapping.evidence}")
    if mapping.nested_under.known:
        lines.append(f"MAP_NESTED the decoder's config is under {mapping.nested_under.value!r}")
    if blocks.total.known:
        lines.append(f"MAP_BLOCKS total={blocks.total.value} from={blocks.total.source} "
                     f"declared={blocks.declared.unwrap('none')} "
                     f"extra={blocks.extra.unwrap(0)}")
    else:
        lines.append(f"MAP_BLOCKS unknown: {blocks.total.why}")
    if mech.kinds.known:
        lines.append(f"MAP_MECHANISMS from={mech.kinds.source} tally={mech.tally()} "
                     f"without_attention={mech.without_attention()}")
    else:
        lines.append(f"MAP_MECHANISMS unknown: {mech.kinds.why}")
    if exp.routed.known:
        lines.append(f"MAP_EXPERTS routed={exp.routed.value} from={exp.routed.source} "
                     f"per_token={exp.per_token.unwrap('unknown')} "
                     f"shared={exp.shared_count.unwrap('unknown')}")
    else:
        lines.append(f"MAP_EXPERTS none: {exp.routed.why}")
    if mapping.stack_length.known:
        lines.append(f"MAP_STACK path={mapping.stack_path.value} "
                     f"blocks={mapping.stack_length.value} "
                     f"base={mapping.base_module_path.unwrap('unknown')} "
                     f"final_norm={mapping.final_norm_name.unwrap('unknown')}")
    if mapping.heads.quotient_is_head_dim is False:
        lines.append(f"MAP_HEADS hidden/heads is not head_dim "
                     f"({mapping.heads.hidden_size.unwrap('?')} / "
                     f"{mapping.heads.attention_heads.unwrap('?')} against "
                     f"{mapping.heads.head_dim.unwrap('?')})")
    lines.extend(f"MAP_NOTE {why}" for why in mapping.disagreements)
    lines.append(f"MAP_NOT_MEASURED {mapping.residual_path_why}")
    return lines
