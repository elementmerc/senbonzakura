# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""End-to-end smoke over every architecture family this tool claims to support.

WHAT THIS IS FOR, AND WHY IT IS NOT COVERED BY THE OTHER FILES

`test_architectures.py` asks whether the walkers FIND the right tensors. This file asks whether
the whole pipeline actually MOVES them: snapshot, bake, verify every residual writer changed,
restore, save, reload from disk, and confirm the checkpoint on disk is the model that was baked
and says what it is.

That distinction is the entire history of this project's worst defect. On Gemma the walkers found
the right tensors, the bake ran, the run reported success, and the edit never reached the model's
running state, because it was applied to the weights feeding a rescaling step rather than to what
came out of it. Every Gemma number had to be withdrawn. "The tensor was found" and "the model
changed" are different claims and only the second one matters.

Everything here is built from a config on the CPU in milliseconds. No download, no GPU, no
credentials. That is deliberate: these families are otherwise only exercised by models too large
to run on the hardware this project is developed on, which is exactly the wiring where a silent
failure survives longest.

THE HYBRID CASE IS THE POINT OF THE FILE

LFM2 writes the residual stream through a short convolution on most of its layers (10 of 16 on
the 350M, which is 31% of all residual writers once the MLP down-projections are counted). The
two tests
that matter most here are that those convolutions genuinely change under a normal run, and that
under `--skip-conv-ablation` they genuinely do NOT while everything else still does. The second
is what makes the control arm of the hybrid experiment a control rather than a coincidence.
"""
import json

import pytest
import torch

from senbonzakura import cli, marker

transformers = pytest.importorskip("transformers")


# ── the families, built from configs ──────────────────────────────────────────────────
def _qwen3(hidden=32, layers=4):
    return transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128))


def _qwen3_moe(hidden=32, layers=2):
    return transformers.Qwen3MoeForCausalLM(transformers.Qwen3MoeConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        decoder_sparse_step=1, norm_topk_prob=True))


def _mixtral(hidden=32, layers=2):
    return transformers.MixtralForCausalLM(transformers.MixtralConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128,
        num_local_experts=4, num_experts_per_tok=2))


def _granite_moe(hidden=32, layers=2):
    return transformers.GraniteMoeForCausalLM(transformers.GraniteMoeConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128,
        num_local_experts=4, num_experts_per_tok=2))


def _lfm2(hidden=32, layer_types=("conv", "conv", "full_attention", "conv")):
    return transformers.Lfm2ForCausalLM(transformers.Lfm2Config(
        hidden_size=hidden, num_hidden_layers=len(layer_types), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128,
        layer_types=list(layer_types)))


def _require_lfm2_moe():
    """LFM2-MoE arrived in transformers after the declared `>=4.56` floor.

    Support for an architecture is conditional on the dependency carrying it, so the floors job
    must skip these rather than report eleven broken tests. Named here, at the construction site,
    because parametrised cases reach it through fixtures and a per-test marker misses them.
    """
    import transformers
    missing = [n for n in ("Lfm2MoeConfig", "Lfm2MoeForCausalLM") if not hasattr(transformers, n)]
    if missing:
        pytest.skip(f"transformers {getattr(transformers, '__version__', '?')} has no "
                    f"{', '.join(missing)}; this architecture arrived after the declared floor")


def _lfm2_moe(hidden=32, layer_types=("conv", "conv", "full_attention", "full_attention")):
    # Skips rather than fails where the installed transformers predates this architecture.
    # Guarded at the construction site so every parametrisation that reaches it is covered,
    # rather than the handful of test names that happened to fail on one runner.
    _require_lfm2_moe()
    return transformers.Lfm2MoeForCausalLM(transformers.Lfm2MoeConfig(
        hidden_size=hidden, num_hidden_layers=len(layer_types), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=128, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=16, num_dense_layers=1,
        layer_types=list(layer_types)))


FAMILIES = {
    "dense": _qwen3,
    "fused-moe": _qwen3_moe,
    "mixtral": _mixtral,
    "granite-moe": _granite_moe,
    "hybrid-dense": _lfm2,
    "hybrid-moe": _lfm2_moe,
}


# ── the harness ───────────────────────────────────────────────────────────────────────
def _build(model, base_args, ablate_conv=True):
    """A constructed Abliterator with real orthonormal directions, ready to bake."""
    base_args.skip_conv_ablation = not ablate_conv
    a = cli.Abliterator(base_args, lambda _m: None, model=model, tok=_FakeTok())
    dm = torch.zeros(a.NL + 1, a.KMAX, a.H)
    for li in range(a.NL + 1):
        q, _ = torch.linalg.qr(torch.randn(a.H, a.KMAX))
        dm[li] = q.T[:a.KMAX]
    a.dirs_multi = dm.to(torch.bfloat16)
    return a


class _FakeTok:
    """The abliterator only needs a tokeniser at generate/score time, which these tests skip."""

    chat_template = None
    pad_token_id = 0
    eos_token_id = 0

    def save_pretrained(self, directory):
        with open(f"{directory}/tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)


def _writers(abl):
    """Every residual-writing tensor the tool intends to edit, flattened."""
    out = []
    for layer in abl.layers:
        out += cli.layer_attn_writers(layer, ablate_conv=abl.ablate_conv)
        for kind, obj in cli.layer_downproj(layer):
            out += obj if kind == "list" else [obj]
    return out


def _bake(abl):
    """One strong bake on both components, enough that every touched tensor must move."""
    abl.bake_pc(1, 1.0, 0.5, 3, 1, 1.0, 0.5, 3, K=1, mode="per_layer")


# ── every family: the tensors actually move, and come back ────────────────────────────
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_bake_moves_every_residual_writer(family, base_args):
    """Not "the walker found it": the numbers in it are different afterwards. This is the
    assertion the withdrawn Gemma results would have failed.
    """
    abl = _build(FAMILIES[family](), base_args)
    abl.snapshot_weights()
    before = [w.detach().clone() for w in _writers(abl)]
    _bake(abl)
    after = _writers(abl)
    assert before, f"{family}: no residual writers found at all"
    unmoved = [i for i, (b, a) in enumerate(zip(before, after, strict=True)) if torch.equal(b, a)]
    assert not unmoved, f"{family}: {len(unmoved)} of {len(before)} writers were left untouched"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_restore_is_bit_identical(family, base_args):
    """The search bakes and restores once per trial, so drift here would compound silently over
    two hundred trials and leave the winner selected against a model nobody ever scored.
    """
    abl = _build(FAMILIES[family](), base_args)
    abl.snapshot_weights()
    before = [w.detach().clone() for w in _writers(abl)]
    _bake(abl)
    abl.restore_weights()
    for i, (b, a) in enumerate(zip(before, _writers(abl), strict=True)):
        assert torch.equal(b, a), f"{family}: writer {i} did not come back"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_row_norms_survive_the_bake(family, base_args):
    """The orthogonalisation is norm-preserving by construction (Heretic's row_normalization=full).
    A family whose tensor layout we read wrongly would show up here as a changed norm, because the
    normalisation would have been applied along the wrong axis.
    """
    abl = _build(FAMILIES[family](), base_args)
    abl.snapshot_weights()
    before = [w.detach().clone() for w in _writers(abl)]
    _bake(abl)
    for i, (b, a) in enumerate(zip(before, _writers(abl), strict=True)):
        # Row norms of the 2-D case, per-expert row norms of the fused 3-D case.
        assert torch.allclose(b.float().norm(dim=-1), a.float().norm(dim=-1), atol=5e-2), \
            f"{family}: writer {i} changed magnitude, so the edit was applied along the wrong axis"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_layer_is_reached_not_just_the_first(family, base_args):
    """A hybrid interleaves block types, so a first layer that resolves cleanly says nothing about
    the fourth. This asserts a writer was found in every single layer.
    """
    abl = _build(FAMILIES[family](), base_args)
    for i, layer in enumerate(abl.layers):
        attn = cli.layer_attn_writers(layer, ablate_conv=abl.ablate_conv)
        mlp = cli.layer_downproj(layer)
        assert attn, f"{family}: layer {i} has no attention-position writer"
        assert mlp, f"{family}: layer {i} has no down-projection"


# ── the hybrid, which is why this file exists ─────────────────────────────────────────
def _conv_weights(abl):
    return [c for c in (cli._conv_outproj(layer) for layer in abl.layers) if c is not None]


@pytest.mark.parametrize("build", [_lfm2, _lfm2_moe], ids=["dense", "moe"])
def test_a_hybrids_convolutions_are_edited_by_a_normal_run(build, base_args):
    abl = _build(build(), base_args)
    convs = _conv_weights(abl)
    assert len(convs) >= 2, "the fixture should carry several convolution layers"
    abl.snapshot_weights()
    before = [w.detach().clone() for w in convs]
    _bake(abl)
    for i, (b, a) in enumerate(zip(before, convs, strict=True)):
        assert not torch.equal(b, a), f"convolution {i} was not edited"


@pytest.mark.parametrize("build", [_lfm2, _lfm2_moe], ids=["dense", "moe"])
def test_the_control_arm_leaves_convolutions_alone_and_edits_everything_else(build, base_args):
    """What makes the control a control. If the convolutions moved anyway the comparison would be
    measuring nothing; if the rest did not move, the two arms would differ by more than one flag.
    """
    abl = _build(build(), base_args, ablate_conv=False)
    convs = _conv_weights(abl)
    others = _writers(abl)
    abl.snapshot_weights()
    conv_before = [w.detach().clone() for w in convs]
    other_before = [w.detach().clone() for w in others]
    _bake(abl)

    for i, (b, a) in enumerate(zip(conv_before, convs, strict=True)):
        assert torch.equal(b, a), f"convolution {i} was edited despite --skip-conv-ablation"
    unmoved = [i for i, (b, a) in enumerate(zip(other_before, others, strict=True))
               if torch.equal(b, a)]
    assert not unmoved, f"{len(unmoved)} non-convolution writers were skipped too"


# The real layer plans of the models in the ladder, read from their own `config.json` rather than
# invented. They matter because the convolution share is NOT constant across the family: it rises
# with size, from 57% at 230M to 73% at 2.6B, so a fixture built on one rung does not generalise
# to another. The 2.6B plan is verbatim; the others are their real (conv, attention) counts in the
# family's interleave, and the count is what the assertions turn on.
REAL_PLANS = {
    # LFM2.5-350M, verbatim from its config.json: 16 layers, 10 conv, 6 attention.
    "350M": ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv",
             "full_attention", "conv", "full_attention", "conv", "full_attention", "conv",
             "full_attention", "conv"],
    # LFM2.5-2.6B, verbatim: 30 layers, 22 conv, 8 attention. The model an already-abliterated
    # third-party checkpoint exists for, so the one most likely to be compared against.
    "2.6B": ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv",
             "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv",
             "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv",
             "full_attention", "conv", "conv", "full_attention", "conv", "conv"],
}


@pytest.mark.parametrize("plan", sorted(REAL_PLANS), ids=sorted(REAL_PLANS))
def test_a_real_layer_plan_is_edited_in_every_layer(plan, base_args):
    """The shape of an actual shipped model, not a four-layer toy.

    A fixture with two conv layers proves the mechanism; it does not prove the mechanism holds at
    a depth where an off-by-one in a per-layer strength profile would show up, nor at the real
    conv-to-attention ratio, which differs on every rung of the family.
    """
    types = REAL_PLANS[plan]
    abl = _build(_lfm2(layer_types=tuple(types)), base_args)
    assert len(types) == abl.NL
    convs = _conv_weights(abl)
    assert len(convs) == types.count("conv")

    abl.snapshot_weights()
    before = [w.detach().clone() for w in _writers(abl)]
    # A profile wide enough to span the model, because `layer_weight` is zero beyond its distance
    # and the four-layer fixtures hide that. Centred, with the taper reaching both ends.
    mid, span = abl.NL // 2, abl.NL
    abl.bake_pc(mid, 1.0, 0.5, span, mid, 1.0, 0.5, span, K=1, mode="per_layer")
    unmoved = [i for i, (b, a) in enumerate(zip(before, _writers(abl), strict=True))
               if torch.equal(b, a)]
    assert not unmoved, f"{plan}: {len(unmoved)} of {len(before)} writers untouched"


@pytest.mark.parametrize("plan", sorted(REAL_PLANS), ids=sorted(REAL_PLANS))
def test_a_narrow_profile_edits_a_window_and_leaves_the_rest_alone(plan, base_args):
    """The property the test above had to be written around, pinned here so nobody "fixes" it.

    `layer_weight` tapers to the minimum weight at its distance and is ZERO beyond it, so a narrow
    profile on a thirty-layer model deliberately edits a band and leaves the rest untouched. That
    is the whole point of tuning a position and a distance. It is also invisible on a four-layer
    test fixture, where every layer falls inside every profile, which is exactly how a real-depth
    assumption goes unchecked.
    """
    abl = _build(_lfm2(layer_types=tuple(REAL_PLANS[plan])), base_args)
    abl.snapshot_weights()
    before = [w.detach().clone() for w in _writers(abl)]
    abl.bake_pc(1, 1.0, 0.5, 2, 1, 1.0, 0.5, 2, K=1, mode="per_layer")
    moved = [i for i, (b, a) in enumerate(zip(before, _writers(abl), strict=True))
             if not torch.equal(b, a)]
    assert moved, "a narrow profile still has to edit its own window"
    assert len(moved) < len(before), "a narrow profile must not reach every layer of a deep model"


@pytest.mark.parametrize("plan", sorted(REAL_PLANS), ids=sorted(REAL_PLANS))
def test_the_convolutions_outnumber_the_attention_layers_in_a_real_plan(plan):
    """The claim task 36 rests on, asserted against the real configs rather than repeated from
    a model card. If a future LFM2 inverts this, the argument for the feature changes with it.

    Note the denominator. This is about the ATTENTION-POSITION writers, where the convolutions are
    10 of 16 on the 350M. Every layer also carries an MLP down-projection that any tool edits, so
    of ALL 32 residual writers the convolutions are 10, which is 31% rather than a majority.
    Earlier drafts said "most of the residual writers" and that overstates it twofold.
    """
    types = REAL_PLANS[plan]
    assert types.count("conv") > types.count("full_attention")


def test_the_control_arm_records_which_layers_it_skipped(base_args):
    abl = _build(_lfm2(layer_types=("conv", "full_attention", "conv", "conv")), base_args,
                 ablate_conv=False)
    assert sorted(abl.partial_layers) == [0, 2, 3]


def test_a_whole_hybrid_run_records_no_skipped_layers(base_args):
    abl = _build(_lfm2(), base_args)
    assert abl.partial_layers == {}


# ── the checkpoint on disk is the model that was baked ────────────────────────────────
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_saved_checkpoint_reloads_as_the_model_that_was_baked(family, base_args, tmp_path):
    """The last step, and the one with the least test coverage historically: a save that silently
    wrote the pre-bake weights would be invisible to every other assertion in this file.
    """
    abl = _build(FAMILIES[family](), base_args)
    abl.snapshot_weights()
    _bake(abl)
    baked = {k: v.detach().clone() for k, v in abl.model.state_dict().items()}

    out = tmp_path / f"saved-{family}"
    abl.model.save_pretrained(str(out), safe_serialization=True)
    marker.stamp(str(out), marker.fields(version="test", ablate_conv=abl.ablate_conv,
                                         partial_layers=abl.partial_layers),
                 log=lambda _s: None)

    reloaded = type(abl.model).from_pretrained(str(out))
    got = reloaded.state_dict()
    assert set(got) == set(baked), f"{family}: the reloaded model has different tensors"
    for k, want in baked.items():
        assert torch.equal(want.cpu(), got[k].cpu()), f"{family}: {k} differs after a round trip"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_saved_checkpoint_says_what_produced_it(family, base_args, tmp_path):
    abl = _build(FAMILIES[family](), base_args)
    out = tmp_path / f"marked-{family}"
    abl.model.save_pretrained(str(out), safe_serialization=True)
    marker.stamp(str(out), marker.fields(version="test", ablate_conv=abl.ablate_conv,
                                         partial_layers=abl.partial_layers, num_directions=1),
                 log=lambda _s: None)
    got = marker.read(str(out))
    assert got["tool"] == "senbonzakura" and got["partial"] == "false"
    # And it survives being loaded and saved again, which is how it reaches derivatives.
    again = tmp_path / f"again-{family}"
    type(abl.model).from_pretrained(str(out)).save_pretrained(str(again), safe_serialization=True)
    with open(again / "config.json", encoding="utf-8") as f:
        assert json.load(f)["senbonzakura"]["tool"] == "senbonzakura"


def test_a_partial_hybrid_checkpoint_carries_the_warning(base_args, tmp_path):
    """The whole point of the marker: this model must not be able to pass for a whole one."""
    abl = _build(_lfm2(), base_args, ablate_conv=False)
    out = tmp_path / "partial"
    abl.model.save_pretrained(str(out), safe_serialization=True)
    marker.stamp(str(out), marker.fields(version="test", ablate_conv=False,
                                         partial_layers=abl.partial_layers),
                 required=True, log=lambda _s: None)
    got = marker.read(str(out))
    assert got["partial"] == "true"
    assert "PARTIAL ABLITERATION" in got["warning"]
    assert got["unedited_layers"] == ",".join(str(i) for i in sorted(abl.partial_layers))
