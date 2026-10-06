# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The residual-leak metric, and the basis trap it must never report as a defect.

WHAT THESE TESTS ARE FOR

`residualleak.py` ships a metric read from a published project, and the metric has one trap: with
every residual writer perfectly ablated, the leak at the model's output still reads about 1% of
the residual norm, because the final norm applies a learned per-dimension weight and a diagonal
map does not preserve orthogonality to a fixed vector. The state did not change; the ruler moved.

So the tests come in two halves. The first half reproduces the measured numbers on a real model,
because a metric whose figures nobody has reproduced is an assertion. The second half drives every
path where the instrument must refuse rather than report, because the whole point of the module is
that a confident wrong number in the output basis is the worst outcome available.

THE MODELS, AND WHY EACH ONE IS HERE

    SmolLM2-135M-Instruct          the happy path, and the model the figures were measured on
    tiny-random-LlamaForCausalLM   RMSNorm with a uniform weight: fast, and every structural path
    tiny-random-gpt2               LayerNorm with a bias, which breaks the identity and must refuse
    Qwen2.5-0.5B-Instruct          final norm 1468x ill conditioned, which must also refuse

The last two are not synthetic constructions. They are shipping checkpoints whose final norms
genuinely trip the two gates, which is the only kind of evidence that the gates fire on something
real rather than on a fixture built to trip them.
"""
import math

import pytest
import torch

from senbonzakura import residualleak as rl

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"

#: Four plain requests, which is exactly `MIN_PROBE_PROMPTS`. Nothing harmful goes in this file:
#: the metric does not care what the prompt says, only where the residual lands, and the one test
#: that needs a real refusal direction reads its prompts from the cached dataset at run time.
PROBE = ["Explain how a bicycle gear works.", "Write a haiku about rain.",
         "What is the capital of Peru?", "Summarise photosynthesis in one line."]


def _load(model_id):
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
    except Exception as e:                           # pragma: no cover - no local cache
        pytest.skip(f"{model_id} is not cached here: {e}")
    return model.eval(), tok


def _templated(tok):
    """A tokenizer that renders a chat prompt, for a model whose own template is absent.

    gpt2 predates chat templates and `render_chat` is the project's one renderer, so the test
    supplies the simplest template that renders. The metric reads the last token of whatever came
    out, so the template's content is irrelevant to what is being measured; what matters is that
    gpt2 goes through the SAME renderer as everything else rather than a second path.
    """
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return tok


def _ablate(model, direction, modules):
    """Project `direction` out of every named module's output, at runtime, in float32.

    The instrument, not the product. This is the published project's runtime hook and it is used
    here because all the conditions then run against ONE checkpoint and differ in exactly one
    respect, which the weight-edit path cannot do. float32 and not float64 deliberately: that is
    what the figures being reproduced were measured in, and the residue it leaves is the 1e-8 the
    assertions below are written against.
    """
    d = direction.float()

    def project(_m, _i, out):
        y = out[0] if isinstance(out, tuple) else out
        yf = y.float()
        yf = (yf - (yf @ d).unsqueeze(-1) * d).to(y.dtype)
        return (yf, *out[1:]) if isinstance(out, tuple) else yf

    return [m.register_forward_hook(project) for m in modules]


def _residual_writers(model, *, include_embedding):
    """Every module on a Llama-shaped model whose output is added to the residual stream.

    The names are the ones `writers.WRITER_PROJECTIONS` already lists, read through it rather
    than spelled out again, because two hand-kept copies of this set is the exact defect
    `writers.py` exists to keep closed.
    """
    from senbonzakura import writers
    mods = []
    for layer in model.model.layers:
        for block in (layer.self_attn, layer.mlp):
            for name in writers.WRITER_PROJECTIONS:
                proj = getattr(block, name, None)
                if proj is not None:
                    mods.append(proj)
    if include_embedding:
        mods.append(model.model.embed_tokens)
    return mods


# ── the metric itself, with no model anywhere near it ────────────────────────────────
def test_the_leak_fraction_is_the_arithmetic_it_claims_to_be():
    """Against values computed by hand, not against a second run of the same code."""
    d = torch.tensor([1.0, 0.0, 0.0])
    # A row exactly along the direction leaks all of itself; one orthogonal to it leaks none;
    # one at 45 degrees leaks 1/sqrt(2). The mean of those three is the metric.
    rows = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]])
    expect = (1.0 + 0.0 + 1.0 / math.sqrt(2.0)) / 3.0
    assert rl.leak_fraction(rows, d) == pytest.approx(expect, rel=1e-12)
    # The SIGN is thrown away, which is the whole point of the absolute value: a component of
    # -1 is as much of a leak as one of +1.
    assert rl.leak_fraction(-rows, d) == pytest.approx(expect, rel=1e-12)


def test_a_zero_residual_row_leaks_nothing_rather_than_a_nan():
    """The clamp is on the denominator, and a NaN here would propagate silently into the mean."""
    got = rl.leak_fraction(torch.tensor([[0.0, 0.0], [1.0, 0.0]]), torch.tensor([1.0, 0.0]))
    assert got == pytest.approx(0.5)
    assert not math.isnan(got)


def test_a_direction_that_points_nowhere_is_refused_by_name():
    with pytest.raises(ValueError, match="does not point anywhere"):
        rl._unit(torch.zeros(4), "the direction")
    with pytest.raises(ValueError, match="does not point anywhere"):
        rl._unit(torch.tensor([float("nan"), 0.0]), "the direction")


def test_a_diagonal_with_a_zero_entry_is_dropped_before_anything_divides_by_it():
    """Both conventions are offered; the one that cannot be divided by is not.

    Dividing by zero gives an infinity, an infinity reaching a cosine gives a NaN, and a NaN
    compares false against every threshold, so the candidate would have read as a refusal for a
    reason that had nothing to do with the model.
    """
    both = rl.diagonal_candidates(torch.tensor([2.0, 4.0]))
    assert [label for label, _ in both] == ["the final norm's weight",
                                            "one plus the final norm's weight"]
    # weight 0 kills the first candidate; weight -1 kills the second.
    only_second = rl.diagonal_candidates(torch.tensor([0.0, 4.0]))
    assert [label for label, _ in only_second] == ["one plus the final norm's weight"]
    only_first = rl.diagonal_candidates(torch.tensor([-1.0, 4.0]))
    assert [label for label, _ in only_first] == ["the final norm's weight"]
    assert rl.diagonal_candidates(torch.tensor([0.0, -1.0])) == []


# ── the output basis: the two gates, and which diagonal gets chosen ──────────────────
def _readings(*rows):
    return [(label, leak, cos, torch.tensor(w, dtype=torch.float64))
            for label, leak, cos, w in rows]


def test_the_diagonal_is_chosen_by_measurement_and_not_by_a_class_name():
    """Gemma multiplies by `1 + weight` and everything else by `weight`.

    Guessing between those by inspecting a class name is how the gemma failure happened once
    already, so the candidate that makes `y / w` parallel to `x` wins, whatever it is called.
    """
    got = rl._output_basis(along_pre=0.01, norm_name="norm", readings=_readings(
        ("the final norm's weight", 0.5, 0.60, [1.0, 2.0]),
        ("one plus the final norm's weight", 1e-9, 1.0, [2.0, 3.0])))
    assert got.diagonal_from == "one plus the final norm's weight"
    assert got.along_post_norm_direction == 1e-9
    assert got.refused_because is None
    assert got.norm_condition_number == pytest.approx(1.5)


def test_a_norm_that_is_not_a_rescaling_refuses_the_output_figure():
    """A LayerNorm centres the residual and adds a bias, so no weight expresses the direction."""
    got = rl._output_basis(along_pre=0.012, norm_name="ln_f", readings=_readings(
        ("the final norm's weight", 0.004, 0.9985, [1.0, 1.0])))
    assert got.along_post_norm_direction is None
    assert got.diagonal_from is None
    assert "not a per-dimension rescaling" in got.refused_because
    assert "per-position figures" in got.refused_because
    # The pre-norm reading SURVIVES, labelled, because the refusal has to be attached to
    # something a reader is looking at. It is never the only thing in the box.
    assert got.along_pre_norm_direction == 0.012


def test_an_ill_conditioned_norm_refuses_rather_than_returning_a_smaller_number():
    """The algebra is exact at any conditioning. The measurement is what degrades."""
    got = rl._output_basis(along_pre=0.009, norm_name="norm", readings=_readings(
        ("the final norm's weight", 1e-9, 1.0, [0.01, 16.0])))
    assert got.along_post_norm_direction is None
    assert got.norm_condition_number == pytest.approx(1600.0)
    assert got.norm_diagonal_agreement == 1.0
    assert "too ill conditioned" in got.refused_because
    assert "per-position figures" in got.refused_because


def test_a_norm_with_no_invertible_candidate_says_the_map_has_no_inverse():
    got = rl._output_basis(along_pre=0.02, norm_name="norm", readings=[])
    assert got.along_post_norm_direction is None
    assert got.norm_condition_number is None
    assert "no inverse" in got.refused_because


def test_the_conditioning_gate_sits_clear_of_both_real_models():
    """The threshold is picked from two shipping checkpoints, so it is asserted against them.

    If somebody moves `NORM_CONDITION_CEILING` to make a run pass, this fails and names what the
    number was chosen against.
    """
    assert 3.84 < rl.NORM_CONDITION_CEILING < 1467.0, (
        "the ceiling has to sit above SmolLM2-135M-Instruct at 3.84, where the identity was "
        "measured, and below Qwen2.5-0.5B-Instruct at 1467.6, where dividing by the weight puts "
        "100% of the direction's energy into one coordinate out of 896")


# ── what reaches an artefact, and the project's own check on it ──────────────────────
def _findings(doc):
    from senbonzakura_check.registry import load_checks, run_checks
    found, _skipped = run_checks(doc, load_checks())
    return [f.check_id for f in found]


def _report(output):
    return rl.LeakReport(positions=3, leak_per_position=(1e-8, 2e-8, 1.6e-8), probe_prompts=16,
                         output=output)


def test_what_this_module_writes_passes_the_check_this_project_added_for_it():
    """The round trip, because a tool that fails its own check is the worst possible advert."""
    doc = rl.stamp_report({}, _report(rl.OutputBasis(
        along_pre_norm_direction=0.009581, along_post_norm_direction=1.2e-08,
        norm_condition_number=3.84, norm_diagonal_agreement=1.0,
        diagonal_from="the final norm's weight", refused_because=None)))
    keys = set(doc["metrics"])
    assert keys == {"residual_leak.pre-norm-residual-per-position",
                    "residual_leak.post-norm-output-basis"}
    pre = doc["metrics"]["residual_leak.pre-norm-residual-per-position"]
    assert pre["basis"] == rl.PRE_NORM_BASIS
    assert pre["measured_at"] == rl.AT_POSITION
    assert pre["value"] == pytest.approx(_report(None).mean)
    assert pre["per_position"] == [1e-8, 2e-8, 1.6e-8]
    post = doc["metrics"]["residual_leak.post-norm-output-basis"]
    assert post["basis"] == rl.POST_NORM_BASIS
    assert post["measured_at"] == rl.AT_OUTPUT
    assert "a-removed-direction-with-no-basis-named" not in _findings(doc)


def test_a_refused_output_basis_is_recorded_as_a_refusal_and_never_as_a_figure():
    """The omission is the point.

    Stamping the pre-norm reading at the output would be a basis artefact wearing the label of a
    measurement, and it is the exact combination the project's own check fires on, so writing it
    would be this tool failing its own gate. The reason goes in the artefact instead.
    """
    why = "the final norm at `norm` is too ill conditioned to express the pre-norm direction"
    doc = rl.stamp_report({}, _report(rl.OutputBasis(
        along_pre_norm_direction=0.009581, along_post_norm_direction=None,
        norm_condition_number=1467.6, norm_diagonal_agreement=1.0,
        diagonal_from="the final norm's weight", refused_because=why)))
    assert set(doc["metrics"]) == {"residual_leak.pre-norm-residual-per-position"}
    assert doc["residual_leak_output_basis_refused"] == why
    assert 0.009581 not in [e["value"] for e in doc["metrics"].values()]
    assert "a-removed-direction-with-no-basis-named" not in _findings(doc)


def test_a_degraded_run_carries_its_warnings_into_the_artefact():
    doc = rl.stamp_report({}, rl.LeakReport(
        positions=2, leak_per_position=(1e-8, 2e-8), probe_prompts=8, output=None,
        warnings=("no final norm with a learned weight was found",)))
    assert doc["residual_leak_warnings"] == ["no final norm with a learned weight was found"]
    assert "residual_leak_output_basis_refused" not in doc
    assert "a-removed-direction-with-no-basis-named" not in _findings(doc)


def test_a_leak_figure_with_no_basis_beside_it_is_caught_by_the_check():
    """The negative half: the check has to fire on the shape this module refuses to write."""
    bare = {"metrics": {"residual_leak": {"value": 0.009581,
                                          "units": "fraction-of-residual-norm"}}}
    assert "a-removed-direction-with-no-basis-named" in _findings(bare)
    wrong = {"metrics": {"residual_leak": {
        "value": 0.009581, "units": "fraction-of-residual-norm",
        "estimator": "post-norm-output-basis", "basis": rl.PRE_NORM_BASIS,
        "measured_at": rl.AT_OUTPUT}}}
    assert "a-removed-direction-with-no-basis-named" in _findings(wrong)


# ── the per-position profile, on a real model, structurally ──────────────────────────
def test_the_profile_has_one_entry_per_residual_position_and_says_which_is_which():
    model, tok = _load(TINY)
    d = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(0))
    got = rl.measure_leak(model, tok, PROBE, d, log=lambda *a: None)
    NL = len(model.model.layers)
    assert got.positions == NL + 1
    assert len(got.leak_per_position) == NL + 1
    assert got.probe_prompts == len(PROBE)
    assert got.warnings == ()
    doc = got.as_dict()
    assert doc["position_0_is"] == "the embedding output"
    assert doc["position_i_plus_1_is"] == "what decoder layer i writes into"
    assert "pre-norm" in doc["basis"]
    assert doc["mean_over_positions_pre_norm_basis"] == pytest.approx(got.mean)
    # Every figure is a fraction of a norm, so it cannot leave [0, 1].
    assert all(0.0 <= v <= 1.0 for v in got.leak_per_position), got.leak_per_position


def test_the_last_position_is_what_the_last_block_wrote_and_not_the_post_norm_output():
    """The defect this is written for, and it is invisible if you read the tuple naively.

    `hidden_states[-1]` is post-norm on every current transformers decoder, so a profile taken
    straight off the tuple reports a figure from a different space at its last entry. The tiny
    model's norm has a uniform weight, so both spaces give the same number and a naive reading
    would look fine; the test therefore CHANGES the norm and asserts that the profile does not
    move with it. A figure read before the norm cannot depend on the norm's weight, and one read
    after it must.
    """
    model, tok = _load(TINY)
    H = model.config.hidden_size
    d = torch.zeros(H)
    d[0] = 1.0

    def with_weight(value):
        with torch.no_grad():
            w = torch.ones(H)
            w[0] = value
            model.model.norm.weight.copy_(w)
        return rl.measure_leak(model, tok, PROBE, d, log=lambda *a: None)

    flat, scaled = with_weight(1.0), with_weight(8.0)
    assert scaled.leak_per_position == pytest.approx(flat.leak_per_position, rel=1e-9), (
        "the profile moved when the final norm's weight moved, so its last entry was read from "
        "`hidden_states[-1]`, which is post-norm, rather than from the last decoder block")
    assert scaled.output.along_pre_norm_direction > flat.output.along_pre_norm_direction * 2, (
        "the output figure did not move when the final norm moved, so it is not being read "
        "after the norm at all")


def test_a_model_whose_final_norm_cannot_be_found_degrades_with_a_visible_warning():
    """Never crash and never silently skip: the profile still comes out, the output does not."""
    model, tok = _load(TINY)
    import senbonzakura.residualleak as mod
    saved = mod.FINAL_NORM_NAMES
    mod.FINAL_NORM_NAMES = ("a_name_no_architecture_uses",)
    said = []
    try:
        got = rl.measure_leak(model, tok, PROBE, torch.ones(model.config.hidden_size),
                              log=said.append)
    finally:
        mod.FINAL_NORM_NAMES = saved
    # SAID OUT LOUD, not only recorded. The figure a person quotes is the one they watched arrive.
    assert any("a_name_no_architecture_uses" in line for line in said), said
    assert got.output is None
    assert len(got.warnings) == 1
    assert "a_name_no_architecture_uses" in got.warnings[0]
    assert "per-position figures are unaffected" in got.warnings[0]
    assert "the profile above is the whole measurement" in got.reading()
    assert "output" not in got.as_dict()


def test_a_model_tree_this_tool_does_not_recognise_is_named_rather_than_guessed():
    class NothingLikeAModel:
        pass

    mod, why = rl.final_norm(NothingLikeAModel())
    assert mod is None
    assert "does not expose a base module this tool recognises" in why
    assert "NothingLikeAModel" in why


def test_a_base_module_that_holds_no_decoder_stack_is_walked_past():
    """`model.model` exists on plenty of wrappers and does not always hold the layers.

    The walk keeps going rather than stopping at the first attribute that resolves, because
    stopping there would report "no final norm found" on a model whose norm is one path further
    along.
    """
    class Empty:
        pass

    class RealBase:
        def __init__(self):
            self.h = [object()]
            self.ln_f = torch.nn.LayerNorm(4)

    class Wrapper:
        def __init__(self):
            self.model = Empty()
            self.transformer = RealBase()

    wrapper = Wrapper()
    assert rl._base_model(wrapper) is wrapper.transformer
    found, name = rl.final_norm(wrapper)
    assert found is wrapper.transformer.ln_f
    assert name == "ln_f"


def test_a_final_norm_with_no_weight_is_the_same_degradation_as_no_norm_at_all():
    model, _ = _load(TINY)
    model.model.norm.weight = None
    found, why = rl.final_norm(model)
    assert found is None
    assert "no final norm with a learned weight" in why


# ── the boundaries, and the one failure that is not allowed to degrade ───────────────
def test_a_probe_too_small_to_mean_anything_is_refused_before_a_forward_pass():
    model, tok = _load(TINY)
    d = torch.ones(model.config.hidden_size)
    with pytest.raises(ValueError, match="nothing to measure the residual of"):
        rl.measure_leak(model, tok, [], d, log=lambda *a: None)
    with pytest.raises(ValueError, match=f"below the floor of {rl.MIN_PROBE_PROMPTS}"):
        rl.measure_leak(model, tok, PROBE[:1], d, log=lambda *a: None)


def test_an_empty_decoder_stack_has_no_positions_to_report():
    model, tok = _load(TINY)
    model.model.layers = torch.nn.ModuleList()
    with pytest.raises(ValueError, match="no residual-stream positions"):
        rl.measure_leak(model, tok, PROBE, torch.ones(model.config.hidden_size),
                        log=lambda *a: None)


def test_a_block_that_returns_something_other_than_a_tensor_stops_the_run():
    """The one place that does NOT degrade, and the docstring says why.

    Substituting the post-norm output for the position the last block wrote would be a figure
    from a different space wearing the profile's label, which is the exact confusion this whole
    module exists to prevent. So it raises and names the position.
    """
    with pytest.raises(ValueError, match="returned dict rather than a tensor"):
        rl._block_output({"not": "a tensor"}, "position 2")
    with pytest.raises(ValueError, match="returned NoneType rather than a tensor"):
        rl._block_output((), "position 2")
    t = torch.zeros(1, 2, 3)
    assert rl._block_output(t, "position 2") is t
    assert rl._block_output((t, "cache"), "position 2") is t


def test_a_model_whose_hidden_states_do_not_line_up_with_its_layers_is_refused():
    """A mislabelled profile is worse than no profile, so no profile is reported."""
    model, tok = _load(TINY)
    real = model.forward

    def short(*a, **kw):
        out = real(*a, **kw)
        out.hidden_states = out.hidden_states[:-1]
        return out

    model.forward = short
    with pytest.raises(ValueError, match="hidden states for 2 decoder layers"):
        rl.measure_leak(model, tok, PROBE, torch.ones(model.config.hidden_size),
                        log=lambda *a: None)


# ── the two shipping checkpoints that genuinely trip the gates ───────────────────────
def test_a_layernorm_at_the_end_refuses_the_output_basis_on_a_real_model():
    """gpt2's `ln_f` centres and biases, measured at 0.9985 parallel against an exact 1.

    Not a fixture built to fail: this is what a whole family of published checkpoints does, and
    the gate has to fire on one of them rather than on a constructed tensor.
    """
    model, tok = _load(TINY_GPT2)
    said = []
    got = rl.measure_leak(model, _templated(tok), PROBE,
                          torch.ones(model.config.hidden_size), log=said.append)
    assert any("NO OUTPUT-BASIS FIGURE" in line for line in said), said
    assert got.positions == len(model.transformer.h) + 1
    assert got.output.along_post_norm_direction is None
    assert "not a per-dimension rescaling" in got.output.refused_because
    assert got.output.norm_diagonal_agreement < 1.0 - rl.NORM_DIAGONAL_TOL
    assert "cannot be added to that profile here" in got.reading()
    doc = got.as_dict()
    assert doc["output"]["leak_along_post_norm_direction"] is None
    assert doc["output"]["leak_along_pre_norm_direction_do_not_quote_alone"] > 0.0


def test_a_shipping_qwen_trips_the_conditioning_gate_and_is_refused():
    """Qwen2.5-0.5B-Instruct's final norm runs from 0.011 to 16.75, a ratio of 1468.

    Dividing the direction by that weight puts 100% of its energy into one coordinate out of 896,
    measured on 2026-10-06, so the figure that would come out reads that coordinate rather than
    the direction. This is the test that proves the refusal is reachable on a model somebody
    actually downloads.
    """
    model, tok = _load(QWEN)
    got = rl.measure_leak(model, tok, PROBE, torch.ones(model.config.hidden_size),
                          log=lambda *a: None)
    assert got.output.along_post_norm_direction is None
    assert got.output.norm_condition_number > rl.NORM_CONDITION_CEILING
    assert got.output.norm_condition_number == pytest.approx(1467.6, rel=0.01)
    # The norm IS a clean rescaling here; it is only the division that is unusable, and the
    # refusal must say so rather than blaming the wrong thing.
    assert got.output.norm_diagonal_agreement == pytest.approx(1.0, abs=1e-6)
    assert "too ill conditioned" in got.output.refused_because
    assert got.output.diagonal_from == "the final norm's weight"


# ── reproducing the measured figures ─────────────────────────────────────────────────
def _refusal_direction(model, tok, harmful, harmless, layer):
    from senbonzakura.firsttoken import render_chat
    with torch.no_grad():
        def mean_at(prompts):
            rows = []
            for p in prompts:
                enc = tok(render_chat(tok, p), return_tensors="pt", add_special_tokens=False)
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                rows.append(hs[layer][0, -1, :].float())
            return torch.stack(rows).mean(0)
        d = mean_at(harmful) - mean_at(harmless)
    return d / d.norm()


def test_the_measured_figures_reproduce_on_the_model_they_were_measured_on():
    """The 2026-10-06 CPU experiments, re-run through the shipped module.

    THE RECIPE IS THE ONE THE FIGURES CAME FROM, which is why it is spelled out rather than
    convenient: SmolLM2-135M-Instruct in float32, the direction a difference of means at layer 18
    over 32 harmful and 32 harmless prompts, the probe the first 16 harmful. Change any of those
    and the numbers below are not the numbers being reproduced.

    WHAT THE TOLERANCES ARE AND WHY EACH ONE IS THAT WIDE. They are not uniform, because the
    three figures are known to different precisions:

      * the unablated mean, 0.0796 measured, asserted to 15%. This is the only figure that
        depends on the DIRECTION rather than on the arithmetic, so it moves with the prompt
        sample: the same recipe over 48 pairs instead of 32 gives 0.0819, a 3% move, and 15%
        covers that with room while still refusing anything near zero or near 0.5. The research
        note quotes "about 0.08" and this is the width of "about".

      * the ablated profile, asserted to be under 1e-7 at every position. Measured maximum
        1.4e-08. This is a float32 cancellation residue and nothing in it is a free parameter,
        so the bound is one order of magnitude above the measurement, which is what distinguishes
        "machine zero" from "small".

      * the basis split, 0.009581 along `d` against 1.2e-08 along `d / w`. The first is asserted
        to 1e-3 relative, because it is an exact consequence of the weights and the probe and
        reproduced to six significant figures (0.009581029 here against 0.009581041 in the note);
        the gap between the two is asserted as a RATIO above 10^5, measured at 10^6, because that
        ratio is the finding and a tolerance on it should be a tolerance on the finding.

    If these stop reproducing, the right response is to say so loudly rather than to widen a
    tolerance until it passes.
    """
    datasets = pytest.importorskip("datasets")
    try:
        harmful_ds = datasets.load_dataset("mlabonne/harmful_behaviors", split="train")
        harmless_ds = datasets.load_dataset("mlabonne/harmless_alpaca", split="train")
    except Exception as e:                           # pragma: no cover - no local cache
        pytest.skip(f"the contrast corpora are not cached here: {e}")
    model, tok = _load(SMALL)

    def column(ds):
        return "text" if "text" in ds.column_names else ds.column_names[0]

    harmful = [harmful_ds[i][column(harmful_ds)] for i in range(32)]
    harmless = [harmless_ds[i][column(harmless_ds)] for i in range(32)]
    d = _refusal_direction(model, tok, harmful, harmless, 18)
    probe = harmful[:16]

    unablated = rl.measure_leak(model, tok, probe, d, log=lambda *a: None)
    assert unablated.positions == 31
    assert unablated.mean == pytest.approx(0.0796, rel=0.15), (
        f"the unablated leak came out at {unablated.mean}, where about 0.08 was measured")

    handles = _ablate(model, d, _residual_writers(model, include_embedding=True))
    try:
        ablated = rl.measure_leak(model, tok, probe, d, log=lambda *a: None)
    finally:
        for h in handles:
            h.remove()

    worst = max(ablated.leak_per_position)
    assert worst < 1e-7, (
        f"with every residual writer ablated the worst position leaks {worst:.3e}, where the "
        f"measurement was 1.4e-08. A figure above this is not a float32 residue.")
    assert ablated.leak_per_position[-1] < 1e-7

    out = ablated.output
    assert out.along_pre_norm_direction == pytest.approx(0.009581, rel=1e-3), (
        f"the pre-norm direction read at the output came out at {out.along_pre_norm_direction}, "
        f"where 0.009581041 was measured")
    assert out.along_post_norm_direction is not None, out.refused_because
    assert out.along_post_norm_direction < 1e-7
    ratio = out.along_pre_norm_direction / out.along_post_norm_direction
    assert ratio > 1e5, (
        f"the two bases differ by a factor of {ratio:.3g}, where a factor of about 10^6 was "
        f"measured. THE FINDING IS THAT RATIO: the same state, the same forward pass, two "
        f"rulers. If it has collapsed, the metric is reporting a basis change as a defect.")
    # And the reading never offers the alarming number as a verdict.
    assert "is not evidence about the edit" in ablated.reading()


def test_leaving_the_embedding_alone_leaves_a_residue_the_published_policy_removes():
    """The disagreement between this project's bake and the published runtime project, measured.

    Their hook wraps `embed_tokens`; this project's bake deliberately does not, guarded by
    `test_position_zero_is_the_embedding_and_is_never_ablated`. Position 0 is therefore unmoved
    under our policy and at machine zero under theirs, and the difference persists up the stack
    rather than vanishing at the first block. That is a fact about mechanism and it belongs in a
    test rather than only in a research note, because the note is where it goes stale.

    This runs on the tiny model, with an arbitrary direction, because the claim being tested is
    structural rather than about refusal: it is about which positions an ablation policy reaches.
    """
    model, tok = _load(TINY)
    H = model.config.hidden_size
    d = torch.randn(H, generator=torch.Generator().manual_seed(7))
    d = d / d.norm()

    def run(include_embedding):
        handles = _ablate(model, d, _residual_writers(model,
                                                      include_embedding=include_embedding))
        try:
            return rl.measure_leak(model, tok, PROBE, d, log=lambda *a: None)
        finally:
            for h in handles:
                h.remove()

    ours, theirs = run(False), run(True)
    bare = rl.measure_leak(model, tok, PROBE, d, log=lambda *a: None)
    # Position 0 is the embedding output. Our policy leaves it exactly as it was found.
    assert ours.leak_per_position[0] == pytest.approx(bare.leak_per_position[0], rel=1e-9)
    assert theirs.leak_per_position[0] < 1e-6
    # And theirs is no worse anywhere, which is the honest form of the comparison: the residue is
    # real, and this test does not claim it costs anything behaviourally.
    assert theirs.mean <= ours.mean
