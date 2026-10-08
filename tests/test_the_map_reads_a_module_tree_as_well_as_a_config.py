# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`modelmap`'s other half: what a LIVE module tree exposes, as opposed to what a config claims.

WHY THE TWO HALVES ARE TESTED SEPARATELY AND WHY BOTH EXIST. A config is a claim a checkpoint makes
about itself and a module tree is what the library actually built. They can disagree, and when they
do that is a finding about the checkpoint rather than an error in the reading, so `describe` reports
both and prefers neither. This file drives the module half and the disagreement reporting; the
config half is `tests/test_the_map_conforms_on_every_family_or_says_it_cannot.py`.

The trees here are real `nn.Module`s built in the test, not mocks, because the thing under test is
whether a block's mechanism can be recognised from the modules in it. A mock would recognise
whatever the test told it to.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from senbonzakura import modelmap as mm
from senbonzakura.modelmap import BlockKind

# ── trees, built to be the shapes the measurements found ─────────────────────────────────────

class Attn(nn.Module):
    """An attention block, in the shape the EDITOR recognises: a 2-D `o_proj`.

    The identifying feature is the output projection and not the query projection, because that is
    what `cli._attn_block` looks for, and a map that identified attention any other way would
    disagree with the editor about the same model.
    """

    def __init__(self, width=8):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)


class Mamba(nn.Module):
    """A state-space mixer: an `out_proj` and NO query projection, which is the whole test.

    On Nemotron-H this and `Attn` live under the SAME child name, `mixer`, so a reader going by
    the child name alone cannot tell them apart and has to look at the contents.
    """

    def __init__(self, width=8):
        super().__init__()
        self.out_proj = nn.Linear(2 * width, width, bias=False)
        self.A_log = nn.Parameter(torch.zeros(4))


class MLP(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.gate_proj = nn.Linear(width, 2 * width, bias=False)
        self.down_proj = nn.Linear(2 * width, width, bias=False)


class Norm(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))


def block(kind, width=8):
    """One decoder block of the requested kind, as the real families arrange them."""
    layer = nn.Module()
    if kind in ("attention", "both"):
        layer.self_attn = Attn(width)
    if kind in ("mixer", "both"):
        layer.conv = Mamba(width)
    if kind == "nemotron_attention":
        layer.mixer = Attn(width)
    if kind == "nemotron_mamba":
        layer.mixer = Mamba(width)
    if kind != "nothing":
        layer.mlp = MLP(width)
    layer.input_layernorm = Norm(width)
    layer.post_attention_layernorm = Norm(width)
    return layer


def tree(kinds, *, stack_attr="layers", base_attr="model", norm_attr="norm", width=8):
    """A whole model: a base module holding a stack and a final norm, as the resolvers expect."""
    base = nn.Module()
    setattr(base, stack_attr, nn.ModuleList([block(k, width) for k in kinds]))
    if norm_attr:
        setattr(base, norm_attr, Norm(width))
    model = nn.Module()
    setattr(model, base_attr, base)
    return model


# ── the stack resolver ───────────────────────────────────────────────────────────────────────

def test_the_llama_style_stack_resolves_and_names_the_path_it_used():
    model = tree(["attention"] * 3)
    stack, where = mm.decoder_stack(model)
    assert len(stack) == 3
    assert where.value == "model.layers"


def test_the_gpt2_style_stack_resolves_under_its_own_path():
    model = tree(["attention"] * 2, stack_attr="h", base_attr="transformer")
    stack, where = mm.decoder_stack(model)
    assert len(stack) == 2
    assert where.value == "transformer.h"


def test_a_tree_with_no_recognised_stack_gives_a_reason_naming_every_path_tried():
    nothing = nn.Module()
    stack, where = mm.decoder_stack(nothing)
    assert stack is None
    assert not where.known
    for path in mm.DECODER_STACK_PATHS:
        assert path in where.why, f"the reason does not mention {path}"
    assert "not a causal language model" in where.why
    assert "decoder layer stack" in where.why, (
        "the phrase is the one cli._decoder_layers has always raised and it is user-facing")


def test_the_raising_form_carries_the_same_sentence_the_reason_does():
    """The editor cannot proceed without a stack, so for that one caller the degradation is a loud
    failure at load. The sentence a user sees is the same either way.
    """
    with pytest.raises(ValueError, match="could not find the decoder layer stack"):
        mm.stack_or_raise(nn.Module())


def test_the_raising_form_returns_the_stack_when_there_is_one():
    model = tree(["attention"] * 2)
    assert len(mm.stack_or_raise(model)) == 2


# ── the base module and the final norm ───────────────────────────────────────────────────────

def test_the_base_module_is_the_one_that_actually_holds_a_stack():
    model = tree(["attention"])
    base, where = mm.base_module(model)
    assert base is model.model
    assert where.value == "model"


def test_a_candidate_holding_no_stack_does_not_win():
    """A wrapper attribute called `model` must not be mistaken for the base when it holds nothing.

    This is the guard that makes `BASE_MODEL_PATHS` safe to try in order: the path matching is a
    candidate, and holding a stack is the confirmation.
    """
    model = nn.Module()
    model.model = nn.Module()          # named right, holds nothing
    base, where = mm.base_module(model)
    assert base is None
    assert not where.known
    assert "does not expose a base module" in where.why


def test_the_final_norm_resolves_and_names_the_attribute():
    model = tree(["attention"])
    norm, where = mm.final_norm(model)
    assert norm is model.model.norm
    assert where.value == "norm"


@pytest.mark.parametrize("attr", ["norm", "final_layernorm", "ln_f", "final_norm"])
def test_every_spelling_of_the_final_norm_resolves(attr):
    model = tree(["attention"], norm_attr=attr)
    _norm, where = mm.final_norm(model)
    assert where.value == attr


def test_a_final_norm_under_a_name_nobody_knows_degrades_with_a_reason_and_does_not_raise():
    """The house style this module inherits from `residualleak.final_norm`: a name this tool does
    not recognise is a degradation, because plenty of callers do not need the norm at all.
    """
    model = tree(["attention"], norm_attr="some_name_no_architecture_uses")
    norm, where = mm.final_norm(model)
    assert norm is None
    assert not where.known
    for name in mm.FINAL_NORM_NAMES:
        assert name in where.why
    assert "names it something new" in where.why


def test_a_norm_with_no_learned_weight_is_not_the_final_norm():
    """A module can be called `norm` and carry no weight, and then it is not what was wanted."""
    model = tree(["attention"], norm_attr=None)
    model.model.norm = nn.Module()
    norm, where = mm.final_norm(model)
    assert norm is None
    assert "no final norm with a learned weight" in where.why


def test_a_tree_with_no_base_module_says_the_final_norm_is_unknown_for_that_reason():
    """The wording is `residualleak.final_norm`'s, verbatim, because that sentence is on a live
    code path through `abliterate --leak-report` and reaches a user. The consolidation moved the
    resolution without rewording the output.
    """
    norm, where = mm.final_norm(nn.Module())
    assert norm is None
    assert "does not expose a base module this tool recognises" in where.why
    assert "so its final norm could not be found" in where.why


# ── what one live block holds ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("built", "expected"), [
    ("attention", BlockKind.ATTENTION),
    ("mixer", BlockKind.MIXER),
    ("both", BlockKind.BOTH),
    ("mlp_only", BlockKind.MLP_ONLY),
])
def test_a_live_block_is_classified_by_what_it_holds(built, expected):
    assert mm.block_kind(block(built)) is expected


def test_the_nemotron_case_where_one_child_name_means_two_different_things():
    """THE CASE THAT FORCES CLASSIFICATION BY CONTENTS. On Nemotron-H a decoder layer holds exactly
    one child called `mixer`, and it is attention, a Mamba block, an MLP or a mixture of experts
    depending on where the layer sits. The name says where it sits; only the contents say what it
    is, which is why `mixer` appears in all three position lists in `writers.py`.
    """
    assert mm.block_kind(block("nemotron_attention")) is BlockKind.ATTENTION
    assert mm.block_kind(block("nemotron_mamba")) is BlockKind.MIXER


def test_a_block_holding_none_of_the_three_positions_is_unknown_and_not_filed_under_one():
    bare = nn.Module()
    bare.something_nobody_has_seen = nn.Linear(4, 4)
    assert mm.block_kind(bare) is BlockKind.UNKNOWN


def test_the_control_arm_that_leaves_the_convolution_alone_reads_it_as_mlp_only():
    """Mirrors `--skip-conv-ablation`. With the mixer deliberately out of scope, a block whose only
    mixer is that convolution genuinely has no writer in the attention position.
    """
    assert mm.block_kind(block("mixer"), count_mixers=False) is BlockKind.MLP_ONLY
    assert mm.block_kind(block("both"), count_mixers=False) is BlockKind.ATTENTION


@pytest.mark.parametrize("projection", ["o_proj", "out_proj", "dense"])
def test_attention_is_recognised_through_every_output_projection_spelling(projection):
    layer = nn.Module()
    layer.self_attn = nn.Module()
    setattr(layer.self_attn, projection, nn.Linear(4, 4))
    assert mm.block_kind(layer) is BlockKind.ATTENTION


def test_a_child_in_the_attention_position_with_no_output_projection_does_not_attend():
    """Presence of the child is not enough, and the editor's own comment says why: NemotronH names
    every block `mixer`, so asking "is there a child called mixer" answers a different question
    from "does this layer attend".
    """
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.Linear(4, 4)
    layer.mlp = MLP()
    assert mm.block_kind(layer) is BlockKind.MLP_ONLY


def test_an_output_projection_of_the_wrong_rank_is_refused_rather_than_described():
    """A writer this bake could not edit correctly must not be reported as one it could."""
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = nn.Module()
    layer.self_attn.o_proj.weight = nn.Parameter(torch.zeros(2, 3, 4))
    layer.mlp = MLP()
    assert mm.block_kind(layer) is BlockKind.MLP_ONLY


def test_a_layer_with_both_a_conventional_attention_and_a_mixer_child_resolves_to_attention():
    """`ATTN_BLOCKS` is ordered with `mixer` last for exactly this case."""
    layer = nn.Module()
    layer.self_attn = Attn()
    layer.mixer = Mamba()
    layer.mlp = MLP()
    block_, projection = mm.attention_block(layer)
    assert block_ is layer.self_attn
    assert projection is layer.self_attn.o_proj


def test_out_proj_on_a_child_called_mixer_belongs_to_the_mixer_path_and_not_to_attention():
    """The Mamba-2 case. Reading it as attention would edit the wrong matrix with confidence."""
    layer = nn.Module()
    layer.mixer = Mamba()
    layer.mlp = MLP()
    assert mm.attention_block(layer) == (None, None)
    assert mm.mixer_outproj(layer) is layer.mixer.out_proj
    assert mm.block_kind(layer) is BlockKind.MIXER


def test_o_proj_on_a_child_called_mixer_is_attention_because_that_is_nemotrons_attention_layer():
    layer = nn.Module()
    layer.mixer = nn.Module()
    layer.mixer.o_proj = nn.Linear(4, 4)
    layer.mlp = MLP()
    assert mm.block_kind(layer) is BlockKind.ATTENTION


def test_a_block_whose_norms_are_listed_is_the_residual_patterns_shadow_and_says_so():
    norms = mm.block_norms(block("attention"))
    assert norms == ["input_layernorm", "post_attention_layernorm"]


def test_a_third_norm_shows_up_so_a_reader_knows_to_go_and_look():
    layer = block("attention")
    layer.pre_feedforward_layernorm = Norm()
    assert "pre_feedforward_layernorm" in mm.block_norms(layer)


def test_a_norm_shaped_attribute_with_no_weight_is_not_listed():
    layer = block("attention")
    layer.extra_norm = nn.Module()
    assert "extra_norm" not in mm.block_norms(layer)


# ── the whole description, from both sources at once ─────────────────────────────────────────

def test_a_model_and_its_config_together_report_both_and_prefer_neither():
    model = tree(["attention", "mixer", "mlp_only"])
    cfg = {"num_hidden_layers": 3, "layer_types": ["full_attention", "conv", "mlp"],
           "hidden_size": 8, "num_attention_heads": 2}
    mapping = mm.describe(model=model, config=cfg)
    assert mapping.evidence == "config+modules"
    assert mapping.stack_length.value == 3
    assert [k.value for k in mapping.module_kinds.value] == ["attention", "mixer", "mlp_only"]
    assert [k.value for k in mapping.mechanisms.kinds.value] == ["attention", "mixer", "mlp_only"]
    assert mapping.disagreements == ()


def test_a_config_and_a_tree_that_disagree_about_the_depth_report_the_disagreement():
    mapping = mm.describe(model=tree(["attention"] * 2), config={"num_hidden_layers": 9})
    assert any("the module tree holds 2" in note for note in mapping.disagreements)


def test_a_config_and_a_tree_that_disagree_about_a_mechanism_name_the_block():
    """The finding that would matter most: the config says one thing and the library built another."""
    mapping = mm.describe(model=tree(["attention", "attention"]),
                          config={"num_hidden_layers": 2, "layer_types": ["full_attention", "conv"]})
    assert any("disagree about blocks [1]" in note for note in mapping.disagreements)


def test_an_unknown_block_in_the_tree_is_surfaced_rather_than_left_in_the_record():
    bare = nn.Module()
    bare.mystery = nn.Linear(4, 4)
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([bare])
    mapping = mm.describe(model=model, config={"num_hidden_layers": 1})
    assert any("expose none of the attention, mixer or MLP positions" in n
               for n in mapping.disagreements)


def test_a_model_with_no_stack_describes_what_it_can_and_says_why_the_rest_is_missing():
    mapping = mm.describe(model=nn.Module(), config={"num_hidden_layers": 4})
    assert mapping.blocks.total.value == 4, "the config half is unaffected"
    assert not mapping.stack_length.known
    assert not mapping.module_kinds.known
    assert "could not find the decoder layer stack" in mapping.stack_length.why


def test_a_config_only_description_says_the_module_tree_was_not_read():
    mapping = mm.describe(config={"num_hidden_layers": 2})
    assert mapping.evidence == "config"
    assert "no model was given" in mapping.stack_path.why


def test_a_model_with_no_config_is_described_from_its_own_config_attribute(tiny_model):
    """The normal call: a live model carries its config, so `describe(model=...)` needs no second
    argument. Uses the suite's shared stand-in, which is a real module tree.
    """
    mapping = mm.describe(model=tiny_model)
    assert mapping.stack_length.value == 4
    assert mapping.blocks.total.value == 4
    assert all(k is BlockKind.ATTENTION for k in mapping.module_kinds.value)


def test_the_shared_stand_in_resolves_through_every_module_resolver(tiny_model):
    assert mm.decoder_stack(tiny_model)[1].value == "model.layers"
    assert mm.base_module(tiny_model)[0] is tiny_model.model
    assert len(mm.stack_or_raise(tiny_model)) == 4


def test_a_model_whose_config_is_not_a_dict_is_still_described_from_its_modules():
    model = tree(["attention"] * 2)
    model.config = object()
    mapping = mm.describe(model=model)
    assert mapping.stack_length.value == 2
    assert not mapping.blocks.total.known


def test_the_summary_of_a_full_description_names_the_stack_path(tiny_model):
    lines = mm.summarise(mm.describe(model=tiny_model))
    assert any("MAP_STACK path=model.layers" in line for line in lines)
    assert any("MAP_NOT_MEASURED" in line for line in lines)


def test_the_residual_path_is_never_claimed_even_with_a_live_model_in_hand(tiny_model):
    """The point of the field. Holding the modules does not answer where the residual stream runs,
    and a record that went quiet about that once a model was present would be the worst version.
    """
    mapping = mm.describe(model=tiny_model)
    assert "forward pass" in mapping.residual_path_why
    assert "where does the residual stream run" in mapping.unanswerable()


# ── the one-resolver discipline ──────────────────────────────────────────────────────────────

def test_the_map_is_the_only_place_that_lists_the_key_spellings():
    """The whole point of the module. Every list here is one list, and this test is what makes the
    claim checkable rather than aspirational: it asserts the map's own constants are non-empty and
    distinct, so a future edit that empties one fails rather than silently resolving nothing.
    """
    for name in ("LAYER_COUNT_KEYS", "EXTRA_BLOCK_KEYS", "NESTED_CONFIG_KEYS",
                 "BLOCK_TYPE_LIST_KEYS", "EXPERT_COUNT_KEYS", "EXPERT_TOPK_KEYS",
                 "DECODER_STACK_PATHS", "BASE_MODEL_PATHS", "FINAL_NORM_NAMES"):
        values = getattr(mm, name)
        assert values, f"{name} is empty, so whatever it resolved now resolves nothing"
        assert len(set(values)) == len(values), f"{name} has a duplicate"


def test_the_map_and_the_editor_agree_about_which_layers_attend():
    """THE CROSS-CHECK THAT MAKES THE DUPLICATION SAFE UNTIL IT CAN BE DELETED.

    `modelmap.attention_block` reproduces `cli._attn_block` in a module that imports nothing
    heavy, because the editor's copy cannot be imported here without dragging torch, optuna and
    transformers into a module whose point is that it needs none of them. Two copies of a
    predicate is the defect this whole item is about, so for as long as both exist they are
    compared on real trees and a divergence fails here.

    When `cli.py` is free, the consolidation is a deletion: `cli._attn_block` calls this, and this
    test becomes a tautology worth keeping anyway.
    """
    from senbonzakura import cli

    trees = [
        block("attention"), block("mixer"), block("both"), block("mlp_only"),
        block("nemotron_attention"), block("nemotron_mamba"), block("nothing"),
    ]
    extra = nn.Module()
    extra.self_attn = nn.Module()
    extra.self_attn.q_proj = nn.Linear(4, 4)        # child present, no output projection
    trees.append(extra)
    only_dense = nn.Module()
    only_dense.attn = nn.Module()
    only_dense.attn.dense = nn.Linear(4, 4)
    trees.append(only_dense)

    for i, layer in enumerate(trees):
        assert mm._is_attention(layer) is cli._has_attention(layer), (
            f"tree {i} ({type(layer).__name__}): the map says "
            f"{mm._is_attention(layer)} and the editor says {cli._has_attention(layer)}")
        mine = mm.attention_block(layer)[0]
        theirs = cli._attn_block(layer)[0]
        assert mine is theirs, f"tree {i}: the two resolve different blocks"


def test_the_map_and_the_editor_agree_about_the_mixer_in_every_position():
    from senbonzakura import cli

    for kind in ("attention", "mixer", "both", "mlp_only", "nemotron_mamba"):
        layer = block(kind)
        assert (mm.mixer_outproj(layer) is not None) == (
            cli._block_outproj_param(getattr(layer, "conv", None)) is not None
            or cli._block_outproj_param(getattr(layer, "mixer", None)) is not None), kind


def test_the_map_and_the_editors_composition_count_agree_on_a_hybrid_stack():
    """`cli.layer_composition` already counts attention, mixer, both and MLP-only per layer, and
    it is the number a run prints. The map must not produce a second, different tally.
    """
    from senbonzakura import cli

    kinds = ["attention", "mixer", "both", "mlp_only", "nemotron_mamba", "nemotron_attention"]
    model = tree(kinds)
    stack = list(mm.stack_or_raise(model))
    theirs = cli.layer_composition(stack)
    mine = mm.describe(model=model, config={"num_hidden_layers": len(kinds)}).module_kinds.value
    assert sum(1 for k in mine if k is BlockKind.ATTENTION) == theirs["attention"]
    assert sum(1 for k in mine if k is BlockKind.MIXER) == theirs["mixer"]
    assert sum(1 for k in mine if k is BlockKind.BOTH) == theirs["both"]
    assert sum(1 for k in mine if k is BlockKind.MLP_ONLY) == theirs["mlp_only"]


def test_the_position_lists_come_from_writers_rather_than_being_relisted():
    """`writers.py` is already the single home for which child names sit in which position, and it
    says why in its own docstring. The map importing them is what stops this module becoming the
    third copy of a list that has already drifted once in this project.
    """
    from senbonzakura import writers
    assert mm.ATTN_BLOCKS is writers.ATTN_BLOCKS
    assert mm.MIXER_BLOCKS is writers.MIXER_BLOCKS
    assert mm.MLP_BLOCKS is writers.MLP_BLOCKS


def test_the_extra_block_term_is_the_arithmetic_the_streaming_path_was_missing():
    """Recorded as a test because it is the drift this module exists to close. `streaming` took the
    declared count alone and was therefore one block short on three local checkpoints.
    """
    from senbonzakura import streaming
    assert set(mm.LAYER_COUNT_KEYS) == set(streaming.LAYER_COUNT_KEYS), (
        "the map and the streaming path must agree about the depth spellings")
    assert mm.EXTRA_BLOCK_KEYS, "the extra-block term is what streaming was missing"
    mapping = mm.describe(config={"num_hidden_layers": 47, "num_nextn_predict_layers": 1})
    assert mapping.blocks.total.value == 48
    assert mapping.blocks.declared.value == 47, (
        "both numbers are kept, because a caller walking the declared stack wants 47 and a caller "
        "reading tensor names wants 48")


def test_every_resolver_returns_a_reason_when_it_returns_nothing():
    """The invariant the whole record rests on: `value is None` always comes with a `why`."""
    empty = mm.describe(config={})
    for resolved in (empty.blocks.total, empty.blocks.declared, empty.mechanisms.kinds,
                     empty.experts.routed, empty.experts.per_token, empty.experts.shared_count,
                     empty.experts.dense_blocks, empty.heads.hidden_size, empty.heads.head_dim,
                     empty.model_type, empty.architecture, empty.stack_path):
        assert not resolved.known
        assert resolved.why and resolved.why.strip(), "a missing value with no reason is a shrug"


def test_a_resolved_value_never_carries_a_reason():
    resolved = mm.describe(config={"num_hidden_layers": 4}).blocks.total
    assert resolved.known
    assert resolved.why is None
    assert resolved.source == "num_hidden_layers"


def test_unwrap_gives_the_fallback_only_when_there_is_nothing_to_give():
    known = mm.Resolved(value=7, source="k")
    unknown = mm.Resolved(why="because")
    assert known.unwrap("fallback") == 7
    assert unknown.unwrap("fallback") == "fallback"
    assert unknown.unwrap() is None


def test_a_resolved_value_of_zero_is_known_and_is_not_treated_as_missing():
    """Zero is a value. `extra` is 0 on most checkpoints and that is a fact, not an absence."""
    zero = mm.describe(config={"num_hidden_layers": 4}).blocks.extra
    assert zero.value == 0
    assert zero.known
    assert zero.unwrap("fallback") == 0


# ── the accessors, driven directly on the states the integration tests do not reach ──────────

def test_every_mechanism_accessor_is_safe_when_nothing_was_read():
    """A caller holding an unread record must be able to call all of it. A viewer will."""
    unread = mm.Mechanisms(kinds=mm.Resolved(why="nothing was read"))
    assert unread.tally() == {}
    assert unread.unknown_blocks() is None
    assert unread.without_attention() is None


def test_a_config_that_is_not_an_object_is_refused_by_the_scope_resolver_with_a_reason():
    scope, nested = mm.decoder_scope(["not", "a", "config"])
    assert scope == {}
    assert not nested.known
    assert "not an object" in nested.why


def test_a_flat_config_says_it_is_not_nested_rather_than_leaving_the_field_blank():
    _scope, nested = mm.decoder_scope({"num_hidden_layers": 2})
    assert not nested.known
    assert "it is not nested" in nested.why


@pytest.mark.parametrize(("given", "expected"), [
    (None, {}),
    ({"num_hidden_layers": 4}, {"num_hidden_layers": 4}),
    (object(), {}),
])
def test_a_config_is_read_whatever_shape_it_arrives_in(given, expected):
    assert mm._config_dict(given) == expected


def test_a_config_object_exposing_to_dict_is_read_through_it():
    class WithToDict:
        def to_dict(self):
            return {"num_hidden_layers": 7}

    assert mm._config_dict(WithToDict()) == {"num_hidden_layers": 7}


def test_a_to_dict_that_does_not_return_a_dict_falls_through_to_the_attributes():
    """Defensive, and cheap: a wrapper with a misbehaving `to_dict` still has its attributes."""
    class Odd:
        def __init__(self):
            self.num_hidden_layers = 5
            self._private = "not carried"

        def to_dict(self):
            return "not a dict"

    assert mm._config_dict(Odd()) == {"num_hidden_layers": 5}


def test_private_attributes_on_a_config_object_are_not_carried_into_the_record():
    class WithPrivates:
        def __init__(self):
            self.num_hidden_layers = 3
            self._cache = {"something": "internal"}

    assert mm._config_dict(WithPrivates()) == {"num_hidden_layers": 3}


def test_the_summary_names_the_nesting_the_experts_and_the_broken_head_identity():
    """One rich config, so the lines a reader actually gets are exercised rather than assumed."""
    cfg = {"model_type": "richmoe", "architectures": ["RichMoeForConditionalGeneration"],
           "text_config": {"num_hidden_layers": 48, "mtp_num_hidden_layers": 1,
                           "hidden_size": 2048, "num_attention_heads": 32, "head_dim": 128,
                           "num_experts": 128, "num_experts_per_tok": 8, "n_shared_experts": 1,
                           "layer_types": ["full_attention", "conv"] * 24}}
    lines = mm.summarise(mm.describe(config=cfg))
    joined = "\n".join(lines)
    assert "MAP_NESTED the decoder's config is under 'text_config'" in joined
    assert "MAP_EXPERTS routed=128 from=num_experts" in joined
    assert "MAP_HEADS hidden/heads is not head_dim" in joined
    assert "MAP_BLOCKS total=49" in joined
    assert "without_attention=24" in joined


def test_the_summary_says_so_when_the_depth_and_the_mechanisms_are_both_unreadable():
    joined = "\n".join(mm.summarise(mm.describe(config={})))
    assert "MAP_BLOCKS unknown:" in joined
    assert "MAP_MECHANISMS unknown:" in joined
    assert "MAP_EXPERTS none:" in joined


def test_a_partly_read_mechanism_list_blocks_the_attention_count_through_the_second_guard():
    """The `without_attention() is None` path reached without any unmapped STRING: the inverted
    key produces UNKNOWN blocks from a value it never named, so the first guard does not fire.
    """
    mapping = mm.describe(config={"num_hidden_layers": 6, "attn_layer_indices": [0, 3]})
    assert mapping.mechanisms.without_attention() is None
    assert "how many blocks have no attention" in mapping.unanswerable()
