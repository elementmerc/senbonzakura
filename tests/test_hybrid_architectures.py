# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""State-space hybrids, and the architecture where one child name means four things.

WHAT THIS ADDS AND WHY IT WAS MISSING

An audit of two competing tools on 2026-09-07 found they already covered several architectures we
did not, and that our own morning's work had reached parity rather than opening a gap. Probing ten
current architectures with our real recogniser then found five it refused outright:

    Bamba              mamba.out_proj      unrecognised
    FalconH1           mamba.out_proj      unrecognised
    Jamba              mamba.out_proj      unrecognised
    GraniteMoeHybrid   mamba.out_proj      unrecognised, plus shared_mlp.output_linear
    NemotronH          every layer         0 of 4 layers recognised

Every `mamba` block puts a 2-D `out_proj` whose OUTPUT dimension is hidden size, which is
dimensionally the same object as an attention `o_proj`, so the norm-preserving bake applies to it
unchanged. That was settled by reading the shapes off real architectures, never by reasoning about
what a state-space model does.

NEMOTRONH IS THE HARD ONE AND IT IS WORTH ITS OWN PARAGRAPH

Its decoder layer holds exactly ONE child, called `mixer`, which is a Mamba-2 mixer, an attention
block, an MLP or a MoE depending on where the layer sits. So `mixer` appears in the attention
list, the sequence-mixer list AND the MLP list, and only the CONTENTS say which it is. It also
breaks an assumption every architecture before it satisfied: that each layer has a writer in both
positions. A NemotronH layer has one.

Relaxing the walkers' loud refusal is the dangerous direction, so the tolerance is stated as an
invariant instead: a layer with a writer in one position and nothing in the other is fine, a layer
with NO writer anywhere is still refused with both reasons quoted.

WHY THESE TESTS BUILD REAL ARCHITECTURES

Because a stand-in built to satisfy the walker proves the walker agrees with itself. These
construct the actual transformers classes from tiny configs, which takes milliseconds, needs no
download and no GPU, and would have caught every one of the five refusals above.
"""
from __future__ import annotations

import warnings

import pytest
import torch
from artefacts import has_model_type, needs_architecture

from senbonzakura import cli

transformers = pytest.importorskip("transformers")

#: Tiny but structurally real. Small enough to build with genuine weights on the CPU in a test.
_BASE = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, vocab_size=128)

#: model_type -> extra config. Each one is an architecture the walkers must handle, and the
#: comment says what it puts in the attention position.
ARCHITECTURES = {
    "llama": dict(num_hidden_layers=4, num_key_value_heads=4),                 # the control
    "bamba": dict(num_hidden_layers=4, num_key_value_heads=4),                 # mamba
    "jamba": dict(num_hidden_layers=4, num_key_value_heads=4),                 # mamba
    "falcon_h1": dict(num_hidden_layers=4, num_key_value_heads=4),             # attention + mamba
    "granitemoehybrid": dict(num_hidden_layers=4, num_key_value_heads=4),      # mamba + shared_mlp
    "nemotron_h": dict(num_hidden_layers=8),                                   # polymorphic mixer
    "qwen3_next": dict(num_hidden_layers=4, num_key_value_heads=2),            # gated delta net
    "glm4_moe_lite": dict(num_hidden_layers=4, num_key_value_heads=4),
}

#: Multi-head Latent Attention. Kept apart because its config needs the low-rank fields, and it is
#: here to record a NEGATIVE result: MLA needed no work, because its residual writer is still an
#: ordinary `self_attn.o_proj`. A competing tool additionally steers `q_b_proj` / `kv_b_proj`,
#: which is input-side steering rather than a residual writer we were missing.
MLA = dict(_BASE, num_hidden_layers=4, num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=16,
           qk_rope_head_dim=8, v_head_dim=8, qk_nope_head_dim=8, first_k_dense_replace=1,
           n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)


def _build(model_type, meta=True, **extra):
    """One architecture, from its config. On the meta device unless real weights are needed.

    Skips, rather than fails, when the installed transformers has never heard of this model type.
    Support for an architecture is conditional on the dependency carrying it, and several of these
    arrived after the declared `transformers>=4.56` floor, so the floors job met
    "ValueError: Unrecognized model identifier: glm4_moe_lite" and read as eleven broken tests.
    The floor is not wrong; the tests were asserting a conditional as if it were unconditional.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if not has_model_type(model_type):
        import transformers
        pytest.skip(f"transformers {getattr(transformers, '__version__', '?')} does not recognise "
                    f"the model type {model_type!r}; it arrived after the declared floor")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = AutoConfig.for_model(model_type, **{**_BASE, **extra})
        if meta:
            with torch.device("meta"):
                return AutoModelForCausalLM.from_config(cfg), cfg
        return AutoModelForCausalLM.from_config(cfg), cfg


ALL = [*ARCHITECTURES.items(), ("deepseek_v3", MLA)]


# ── the guard: nothing writes the residual stream unseen ──────────────────────────
@pytest.mark.parametrize(("model_type", "extra"), ALL, ids=[t for t, _ in ALL])
def test_every_residual_writer_is_recognised(model_type, extra):
    """THE REGRESSION, and it is the shipped guard being asked, not a copy of it.

    An unrecognised writer means an edit that misses part of the residual stream and a run that
    reports success anyway, which is how every gemma number came to be withdrawn.
    """
    model, cfg = _build(model_type, **{k: v for k, v in extra.items() if k not in _BASE})
    layers = cli._decoder_layers(model)
    assert layers, f"{model_type}: no decoder layers found"
    for i, layer in enumerate(layers):
        _rec, unrecognised = cli.residual_writers(layer, cfg.hidden_size)
        assert not unrecognised, f"{model_type} layer {i}: {unrecognised}"


@pytest.mark.parametrize(("model_type", "extra"), ALL, ids=[t for t, _ in ALL])
def test_the_whole_model_passes_the_refusal_gate(model_type, extra):
    """`refuse_unrecognised_writers` is what actually stops a run, so it is what must pass."""
    model, cfg = _build(model_type, **{k: v for k, v in extra.items() if k not in _BASE})
    missed = cli.refuse_unrecognised_writers(
        cli._decoder_layers(model), cfg.hidden_size, log=lambda _m: None)
    assert missed == {}, f"{model_type}: {missed}"


# ── the editor: recognising is not editing ────────────────────────────────────────
@pytest.mark.parametrize(("model_type", "extra"), ALL, ids=[t for t, _ in ALL])
def test_every_layer_yields_a_writer_the_editor_can_use(model_type, extra):
    """Real weights, because the editor refuses a meta tensor on purpose.

    Asserts the shape as well as the presence: a residual writer's output dimension IS the hidden
    size, and a walker that returned the wrong matrix would still return something.
    """
    model, cfg = _build(model_type, meta=False,
                        **{k: v for k, v in extra.items() if k not in _BASE})
    H = cfg.hidden_size
    for i, layer in enumerate(cli._decoder_layers(model)):
        attn, mlp = cli.layer_writers(layer)
        assert attn or mlp, f"{model_type} layer {i}: nothing to edit"
        for w in attn:
            assert w.shape[0] == H, f"{model_type} layer {i}: attention writer is not [hidden, *]"
        for kind, obj in mlp:
            for w in (obj if kind == "list" else [obj]):
                if kind == "fused3d":
                    assert w.shape[1] == H, f"{model_type} layer {i}: hidden not the middle axis"
                else:
                    assert w.shape[0] == H, f"{model_type} layer {i}: writer is not [hidden, *]"


def test_the_edit_actually_lands_on_a_state_space_hybrid():
    """Recognising a writer is not evidence the edit works. This measures it.

    Removes one direction from every writer in a real (tiny) Bamba and checks the component along
    it collapses. The residue left behind is the norm-restore leak, which is a known property of
    the edit rather than a failure of the walker, and it is compared against a plain Llama so the
    number has something to be read beside.
    """
    torch.manual_seed(0)

    @torch.no_grad()
    def leak_ratio(model_type, **extra):
        model, cfg = _build(model_type, meta=False, **extra)
        R = torch.linalg.qr(torch.randn(cfg.hidden_size, 1))[0].T.contiguous()
        before, after = [], []
        for layer in cli._decoder_layers(model):
            attn, mlp = cli.layer_writers(layer)
            dense = list(attn) + [o for k, o in mlp if k == "dense"]
            for w in dense:
                before.append(float((R @ w.float()).norm()))
                cli.orthogonalize_np_(w, R, 1.0)
                after.append(float((R @ w.float()).norm()))
        assert before, f"{model_type}: nothing was edited"
        return sum(after) / sum(before)

    hybrid = leak_ratio("bamba", num_hidden_layers=4, num_key_value_heads=4)
    control = leak_ratio("llama", num_hidden_layers=4, num_key_value_heads=4)
    assert hybrid < 0.25, f"the edit did not land on the mamba path: {hybrid}"
    # Within a factor of three of the plain architecture. The leak is a property of restoring row
    # norms, so a hybrid landing very differently would mean the walker found something else.
    assert hybrid < control * 3, f"hybrid {hybrid} against control {control}"


# ── NemotronH: one child name, four meanings ──────────────────────────────────────
@needs_architecture("Lfm2MoeConfig", "Lfm2MoeForCausalLM")
def test_a_polymorphic_mixer_is_read_by_its_contents_not_its_name():
    """Every one of the four kinds appears in an 8-layer stack, and each must resolve correctly."""
    model, cfg = _build("nemotron_h", num_hidden_layers=8)
    kinds = set()
    for layer in cli._decoder_layers(model):
        mixer = layer.mixer
        kinds.add(type(mixer).__name__)
        attends = cli._has_attention(layer)
        is_mixer = cli._block_outproj_param(mixer) is not None
        # Exactly one of the two, never both: an attention block has o_proj and a state-space
        # mixer has out_proj, and reading either as the other edits the wrong matrix confidently.
        assert not (attends and is_mixer), f"{type(mixer).__name__} read as both"
        _rec, unrecognised = cli.residual_writers(layer, cfg.hidden_size)
        assert not unrecognised, unrecognised
    assert len(kinds) >= 3, f"the fixture did not exercise the polymorphism: {kinds}"


@needs_architecture("Lfm2MoeConfig", "Lfm2MoeForCausalLM")
def test_a_layer_with_only_one_populated_position_is_allowed():
    """NemotronH's layers each hold ONE writer. Every architecture before it held two."""
    model, _cfg = _build("nemotron_h", meta=False, num_hidden_layers=8)
    only_one = 0
    for layer in cli._decoder_layers(model):
        attn, mlp = cli.layer_writers(layer)
        assert attn or mlp
        if bool(attn) != bool(mlp):
            only_one += 1
    assert only_one, "the fixture no longer exercises the single-position case"


def test_a_layer_with_no_writer_anywhere_is_still_refused():
    """The tolerance must not become an excuse. This is the case it must not swallow."""
    class _Empty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(8)

    with pytest.raises(ValueError, match="EITHER position"):
        cli.layer_writers(_Empty())


def test_the_refusal_quotes_both_underlying_reasons():
    """One reason sends the reader to look in the wrong half of the layer."""
    class _Empty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(8)

    with pytest.raises(ValueError) as e:
        cli.layer_writers(_Empty())
    msg = str(e.value)
    assert "attention position:" in msg
    assert "MLP position:" in msg


# ── the composition report, which is what auto-detection tells the operator ───────
@pytest.mark.parametrize(("model_type", "extra", "expect_warning"), [
    ("llama", dict(num_hidden_layers=4, num_key_value_heads=4), False),
    ("bamba", dict(num_hidden_layers=4, num_key_value_heads=4), True),
    ("qwen3_next", dict(num_hidden_layers=4, num_key_value_heads=2), True),
])
def test_a_run_says_what_the_stack_is_made_of(model_type, extra, expect_warning):
    """"down-proj=fused3d, 40 layers" describes a mixture-of-experts model and says nothing about
    whether attention is even present. On Qwen3.6-35B-A3B 30 of 40 layers carry none.
    """
    model, _cfg = _build(model_type, **extra)
    counts = cli.layer_composition(cli._decoder_layers(model))
    assert counts["layers"] == extra["num_hidden_layers"]
    assert sum(counts[k] for k in ("attention", "mixer", "both", "mlp_only")) == counts["layers"]
    line, warn = cli.describe_composition(counts)
    assert line
    assert bool(warn) is expect_warning, (line, warn)


def test_the_warning_points_at_the_reach_check():
    """The rule adopted the same week: recognising a writer is not evidence the edit lands."""
    model, _cfg = _build("bamba", num_hidden_layers=4, num_key_value_heads=4)
    _line, warn = cli.describe_composition(cli.layer_composition(cli._decoder_layers(model)))
    assert "validate --experiment reach" in warn


def test_a_pure_attention_stack_is_not_warned_about():
    """A warning that fires on everything is a warning nobody reads."""
    model, _cfg = _build("llama", num_hidden_layers=4, num_key_value_heads=4)
    _line, warn = cli.describe_composition(cli.layer_composition(cli._decoder_layers(model)))
    assert warn is None


# ── the skip-conv control arm must stay loud ──────────────────────────────────────
def test_skipping_the_mixer_path_is_still_reported_on_a_state_space_hybrid():
    """--skip-conv-ablation deliberately leaves sequence mixers alone, and the guard is what turns
    that into a warning the operator has to opt out of in writing. Adding `mamba` to the handled
    list must not make the control arm silent.
    """
    model, cfg = _build("bamba", num_hidden_layers=4, num_key_value_heads=4)
    for layer in cli._decoder_layers(model):
        _rec, unrecognised = cli.residual_writers(layer, cfg.hidden_size, ablate_conv=False)
        assert unrecognised, "a skipped mamba writer was silently accepted"


@needs_architecture("Lfm2MoeConfig", "Lfm2MoeForCausalLM")
def test_an_attention_mixer_is_not_reported_as_skipped():
    """The mirror image, and the trap the shared `mixer` name creates.

    On NemotronH the same child name holds attention on some layers. Dropping the name from the
    handled set when mixers are skipped would report a perfectly edited attention layer as an
    unablated one, which trains whoever meets it to ignore the warning.
    """
    model, cfg = _build("nemotron_h", num_hidden_layers=8)
    for layer in cli._decoder_layers(model):
        if not cli._has_attention(layer):
            continue
        _rec, unrecognised = cli.residual_writers(layer, cfg.hidden_size, ablate_conv=False)
        assert not unrecognised, f"an attention layer was reported as a skipped mixer: {unrecognised}"
