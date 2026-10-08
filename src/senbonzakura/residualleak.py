# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Did the edit land where it was supposed to land? The residual leak, with the basis named.

WHAT THIS MEASURES, AND WHY IT IS A DIFFERENT QUESTION FROM EVERY OTHER NUMBER HERE

Every figure this project publishes today is behavioural: refusal rate, harm recognition, KL,
capability. Behaviour answers "did the model change". This answers the prior question, "did the
direction actually leave the residual stream", and it answers it without a judge, without
sampling and without generating a single token, so it has no run-to-run variance worth speaking
of and nothing to validate a grader against.

The quantity is the mean, over probe prompts, of

    |component of the residual along the refusal direction| / |residual norm|

taken at the LAST TOKEN, reported once per residual-stream position. The metric is read from
`orcabonsai-27B-uncensored`, which verifies its runtime ablation this way and publishes the claim
"the residual component along the direction falls to ~1e-6 of the residual norm across depth".
Their LICENSE is Apache 2.0, so the idea and the arithmetic are ours to take with attribution,
and this is the attribution.

POSITIONS, NOT LAYERS, AND THE DIFFERENCE IS LOAD-BEARING

The vocabulary here is the one `cli.py` already fixed, and a third naming is not invented:
`leak_per_position` has `NL + 1` entries, position 0 is the embedding output, and position `i + 1`
is what decoder layer `i` writes into. That distinction cost a real investigation once already,
when one list called `dirs_per_layer` carried positions and a reader concluded layer 0 had
received no direction; it had always received one, from position 1.

So the last entry, position `NL`, is what the final decoder block writes into. It is taken from a
forward hook on that block rather than from `hidden_states[-1]`, because `hidden_states[-1]` is
NOT that: every current transformers decoder applies the final norm before appending its last
entry, so the tuple's last element is post-norm and its second-to-last is the INPUT to the final
block. Reading the profile straight off `hidden_states` therefore skips the position this metric
most wants and silently substitutes a figure from a different space.

THE TRAP, WHICH IS THE REASON THIS MODULE IS SHAPED THE WAY IT IS

With every residual writer perfectly ablated, so that `x . d == 0` to machine precision at every
position, the leak at the output still reads about 1% of the residual norm. Measured on
SmolLM2-135M-Instruct, 2026-10-06, with all 61 writers wrapped including the embedding:

    position 30 (what the last block writes into), along d : 0.000000016
    the output (post final norm),                  along d : 0.009581105

The naive reading is that the ablation failed. It did not. The final norm computes
`y = (x / rms(x)) * w` with `w` a learned per-dimension weight. The scalar `1 / rms(x)` preserves
direction; the elementwise `w` is a diagonal map, and a diagonal map does not preserve
orthogonality to a fixed vector. The state is still confined to a hyperplane, one degree of
freedom short of the full space, just a ROTATED hyperplane. For `x . d == 0`,

    y . (d / w) = sum_i (w_i x_i)(d_i / w_i) = sum_i x_i d_i = x . d = 0

so the component along `d / w` must vanish at the output for the same reason the component along
`d` vanishes before it. On the same forward pass:

    the output, along d      : 0.009581105   <- the alarming number
    the output, along d / w  : 0.000000013   <- the same state, the basis the norm maps into

Nothing came back. The ruler moved.

What the identity says, exactly, is that the two components VANISH TOGETHER. It does not say the
two normalised figures are equal, and they are not: this metric divides by the residual norm, and
the post-norm figure carries an extra factor of `sqrt(H) / (|d / w| |y|)` which is 1 only when the
norm's weight is uniform. So the post-norm figure is the right figure to read for "is the
direction gone", and the two numbers are not a pair that should agree digit for digit on a model
where the ablation is partial.

A measurement tool whose first act is to report a basis
change as a defect is worse than no tool, so the naive misreading is not reachable through this
API: there is no attribute anywhere that gives "the leak at the output" on its own.
`OutputBasis.along_pre_norm_direction` is named for what it is, it is never returned without its
post-norm sibling beside it or an explicit refusal in its place, and `reading()` never quotes it
as a verdict.

THE IDENTITY IS EXACT ALGEBRA. THE MEASUREMENT OF IT IS NOT, AND THAT IS WHY IT CAN REFUSE

`y . (d / w) = x . d` holds on every architecture, with no tolerance attached. What degrades is
the MEASUREMENT, because forming `d / w` divides by a learned weight whose smallest entries can
be tiny. Final norm weights, measured on the two instruct models cached on this machine on
2026-10-06:

    SmolLM2-135M-Instruct   min|w| 0.871094   max|w| 3.3438    max/min    3.8
    Qwen2.5-0.5B-Instruct   min|w| 0.011414   max|w| 16.7500   max/min 1467.6

and what that does to the direction, measured on a real difference-of-means direction in each:

    SmolLM2    top coordinate of d carries  5.4% of its energy,  d/w    3.9%
    Qwen2.5    top coordinate of d carries  1.4% of its energy,  d/w  100.0%

On Qwen the reported "component along `d / w`" is a reading of one coordinate out of 896, not a
reading of the direction. So past a documented conditioning threshold this REFUSES to report the
output-basis figure at all, says in plain language why, and points the reader at the per-position
profile, which needs no division, is well conditioned on every architecture, and is the basis the
residual writers actually write into. A confident wrong number here is the worst outcome
available; the per-position figures are the primary output and the output-basis figure is a
conditional extra.

WHY A MODULE OF ITS OWN

`measure.py` is explicitly not a new measurement, it runs the five existing commands and
assembles their numbers. `subspace.py` answers whether two sets of DIRECTIONS span the same span,
which is a different question with different inputs. `separation.py` must not touch the `torch`
namespace at module scope, because `parser.py` reads its choices to build a flag and that
arrangement is what keeps `--help` at 0.06s. `validate.py`'s `bake_reach` is the closest cousin
and is a relative before-and-after reduction that needs an `Abliterator`, a track and a bake;
this is an absolute figure that needs a model, a direction and some prompts, which is what lets
it be pointed at a stranger's published checkpoint. So this is a small instrument module beside
`firsttoken.py` and `writers.py`, importing `torch` and nothing else of substance.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

#: Below this many probe prompts the mean is not worth reporting. The metric has no run-to-run
#: variance worth speaking of, so this is a floor against a caller handing over one prompt and
#: quoting the result as a property of the model, not a power calculation.
MIN_PROBE_PROMPTS = 4

#: `max|w| / min|w|` on the final norm's diagonal, past which the output-basis figure is refused
#: rather than reported.
#:
#: CHOSEN FROM THE TWO REAL MODELS ON DISK, NOT FROM TASTE. SmolLM2-135M-Instruct sits at 3.8 and
#: is where the `d / w` identity was measured; Qwen2.5-0.5B-Instruct sits at 1467.6 and is where
#: forming `d / w` concentrates 100% of the direction's energy into one coordinate out of 896.
#: 100 is near the geometric mean of the two (75), so neither real model sits near the boundary:
#: there are 26x of margin below it and 15x above. It also has a meaning independent of those two
#: models. The division rescales the direction's coordinates over a range of `max|w| / min|w|`, so
#: at 100 a coordinate carrying 1% of the direction's energy can be lifted to carry as much as the
#: largest one, and past that the figure stops being a property of the direction.
NORM_CONDITION_CEILING = 100.0

#: How close `y / w` has to be to parallel with `x` before `w` is accepted as the diagonal the
#: final norm actually applies. If `y = (x / rms(x)) * w` then `y / w` is an exactly positive
#: multiple of `x`, so the cosine is 1 up to float error; anything else means the norm is not a
#: diagonal map of the normalised residual and no `w` can express `d` in its output basis.
#:
#: Measured on 2026-10-06, mean cosine over the probe:
#:
#:     SmolLM2-135M-Instruct       RMSNorm, no bias     1.000000000
#:     Qwen2.5-0.5B-Instruct       RMSNorm, no bias     1.000000000
#:     tiny-random-LlamaForCausalLM RMSNorm, no bias    1.000000000
#:     tiny-random-gpt2            LayerNorm, bias      0.998547473
#:     SmolLM2, wrong candidate (1 + w)                 0.665896296
#:
#: So the legitimate case sits within 1e-9 of 1 and the two failing cases are 1.5e-3 and 0.33
#: away. 1e-4 leaves five orders of margin on the legitimate side and still rejects a LayerNorm,
#: whose mean-centring and bias are what break the identity.
NORM_DIAGONAL_TOL = 1e-4

#: Seconds between heartbeats while walking the probe, so a long CPU run on a large model says it
#: is alive rather than looking hung.
HEARTBEAT_SECONDS = 30.0

#: Attribute names that hold the final norm on the architectures this project has met. A name this
#: list does not know is a WARNING and not an error: the per-position profile does not need the
#: norm at all, so the instrument degrades to its primary output rather than failing.
FINAL_NORM_NAMES = ("norm", "final_layernorm", "final_layer_norm", "ln_f", "final_norm")

#: Where the decoder stack hangs, in the same order `cli._decoder_layers` tries. The final norm is
#: a sibling of the stack rather than of the stack's entries, so one walk finds both.
#:
#: THIS IS A SECOND LIST OF ARCHITECTURE PATHS AND THAT IS A DELIBERATE CHOICE, because this
#: project has paid for an undeclared second copy before. `cli._decoder_layers` answers "where are
#: the blocks" and names paths ending in the stack's own attribute; this answers "which module are
#: they a child of", which is a different question and the reason the strings differ rather than
#: being derivable from each other. The stack itself is still resolved by importing that one
#: function, so there is exactly one answer to the question it owns.
#:
#: What the duplication can cost is bounded on purpose. If `cli` learns an architecture this list
#: does not know, `final_norm` returns a reason, `measure_leak` logs it and records it in
#: `warnings`, and the per-position profile comes out unaffected because it never touches the
#: norm. The failure mode is a visible degradation on a model nobody has measured yet, never a
#: wrong number on one somebody has.
BASE_MODEL_PATHS = ("model", "transformer", "gpt_neox", "model.decoder")


def leak_fraction(residual, direction):
    """Mean over rows of `|r . d| / |r|`, which is the metric itself and nothing else.

    `residual` is [N, H] and `direction` is [H]. Separated out because it is the one piece with
    no model in it, so it can be tested against hand-computed values rather than against another
    run of the same code.

    The clamp is on the DENOMINATOR only. A zero residual row has no direction to have a
    component along, and dividing by its norm would produce a NaN that propagates silently
    through the mean; clamping gives that row a leak of 0, which is the honest answer for a row
    that is identically zero.
    """
    r = residual.double()
    d = direction.double()
    return float((r @ d).abs().div(r.norm(dim=-1).clamp_min(1e-30)).mean())


@dataclass(frozen=True)
class OutputBasis:
    """The two readings of the model's output, or the reason there is only one of them.

    There is deliberately no field called `leak_at_output`. The post-norm output can be read along
    two different normals and they disagree by six orders of magnitude on a correctly ablated
    model, so a field with that name would be an invitation to quote whichever one the caller
    happened to reach for. Both are named for their basis, and the pre-norm reading never travels
    without either its post-norm sibling or `refused_because` in that sibling's place.
    """

    #: `|y . d| / |y|` at the post-norm output, with `d` the PRE-norm direction. This is the
    #: alarming number and it is an artefact of the basis, not a measurement of the ablation.
    along_pre_norm_direction: float
    #: `|y . (d / w)| / |y|` at the same output: the same state read along the normal the final
    #: norm maps the pre-norm hyperplane onto. None when the norm could not be used.
    along_post_norm_direction: float | None
    #: `max|w| / min|w|` on the diagonal the final norm applies, or None when there is none.
    norm_condition_number: float | None
    #: Mean cosine between `y / w` and `x` over the probe. 1 means `w` really is the diagonal.
    norm_diagonal_agreement: float | None
    #: Which candidate was accepted as that diagonal, phrased for a person.
    diagonal_from: str | None
    #: Plain language, set whenever `along_post_norm_direction` is None and None otherwise, so
    #: exactly one of the two is ever filled in and a reader cannot find an absence with no reason
    #: beside it.
    refused_because: str | None


@dataclass(frozen=True)
class LeakReport:
    """What the instrument found, with the basis of every number stated beside it."""

    #: `NL + 1`, the number of residual-stream positions.
    positions: int
    #: One figure per position, position 0 the embedding output and position `i + 1` what decoder
    #: layer `i` writes into. Measured in the PRE-norm residual stream throughout, which is the
    #: space the residual writers write into and the only basis in which the figures are directly
    #: comparable to each other.
    leak_per_position: tuple[float, ...]
    #: How many probe prompts the means are over.
    probe_prompts: int
    #: The five fields that decide whether this figure may be compared with another one, exactly as
    #: `stamps.pinned` returns them.
    #:
    #: CARRIED ON THE REPORT RATHER THAN DERIVED WHEN IT IS STAMPED, and that is the point. The
    #: digest, the prompt format and the precision are facts about the run that produced these
    #: numbers, so they are taken from the model, the tokeniser and the prompts that were actually
    #: measured. Deriving them later, from arguments handed separately to the writer, lets a figure
    #: and its own provenance disagree, and `baseline.comparability` reports an absent or wrong
    #: pinned field as a mismatch without being able to say which.
    pinned: dict
    #: The output, in both bases, or None when there is no final norm to speak of.
    output: OutputBasis | None = None
    #: Everything the run degraded on, each phrased for a person.
    warnings: tuple[str, ...] = ()
    #: Named so a reader of the raw record cannot mistake which space the profile lives in.
    basis: str = field(
        default="the pre-norm residual stream, the space the residual writers write into")

    @property
    def mean(self):
        """The mean over positions, which is the single number the published claim is about.

        Over the PER-POSITION figures only. The 0.000290 in this project's own research note for a
        fully ablated model is a mean that silently included the post-norm output: the per-position
        figures were all about 1e-8 and one value of 0.0090 over 31 entries is 0.00029. The mean is
        the output's basis artefact divided by the depth, which is not a property of the ablation.
        """
        return sum(self.leak_per_position) / len(self.leak_per_position)

    def reading(self):
        """One paragraph, leading with the per-position figures and never with the output."""
        lo, hi = min(self.leak_per_position), max(self.leak_per_position)
        head = (f"over {self.positions} residual-stream positions, measured in "
                f"{self.basis}, the component along the direction runs from {lo:.2e} to "
                f"{hi:.2e} of the residual norm, mean {self.mean:.2e}, on "
                f"{self.probe_prompts} probe prompts.")
        if self.output is None:
            return head + (" There is no final norm reading beside it, for the reason in the "
                           "warnings, so the profile above is the whole measurement.")
        o = self.output
        if o.along_post_norm_direction is None:
            return head + (f" The model's output is read in a different basis and cannot be "
                           f"added to that profile here: {o.refused_because}")
        return head + (
            f" At the output, after the final norm, the same state reads "
            f"{o.along_post_norm_direction:.2e} along the normal the norm maps the pre-norm "
            f"hyperplane onto ({o.diagonal_from}), and {o.along_pre_norm_direction:.2e} along the "
            f"pre-norm direction itself. The first of those is the ablation; the second is the "
            f"basis change and is not evidence about the edit.")

    def as_dict(self):
        """The record, with every key saying which space its number is in."""
        doc = {"leak_per_position_pre_norm_basis": list(self.leak_per_position),
               "positions": self.positions,
               "position_0_is": "the embedding output",
               "position_i_plus_1_is": "what decoder layer i writes into",
               "basis": self.basis,
               "mean_over_positions_pre_norm_basis": self.mean,
               "probe_prompts": self.probe_prompts,
               "warnings": list(self.warnings),
               "reading": self.reading()}
        if self.output is not None:
            o = self.output
            doc["output"] = {
                "leak_along_post_norm_direction": o.along_post_norm_direction,
                "leak_along_pre_norm_direction_do_not_quote_alone":
                    o.along_pre_norm_direction,
                "norm_condition_number": o.norm_condition_number,
                "norm_diagonal_agreement": o.norm_diagonal_agreement,
                "diagonal_from": o.diagonal_from,
                "refused_because": o.refused_because}
        return doc


#: What the two stamped entries put in their `basis` and `measured_at` fields. Short controlled
#: tokens rather than sentences, because the checker's
#: `a-removed-direction-with-no-basis-named` compares them for equality and prose is a rewording
#: away from being unreadable by a machine. The sentence a person reads is the estimator
#: description in `measurement.METRICS`, generated from the same place.
PRE_NORM_BASIS = "pre-norm-direction"
POST_NORM_BASIS = "post-norm-direction"
AT_POSITION = "residual-stream-position"
AT_OUTPUT = "model-output"


def stamp_report(doc, report):
    """Record a `LeakReport` in `doc` under the canonical metrics block, basis and all.

    Two entries where the output basis was usable and one where it was not, and the second is
    omitted rather than filled in with the pre-norm reading. That omission is the point: a figure
    at the output along the pre-norm direction would be an artefact of the final norm wearing the
    label of a measurement, and the checker's `a-removed-direction-with-no-basis-named` fires on
    exactly that combination, so writing it here would be this project failing its own check.

    `value` is the MEAN over positions for the profile, with the per-position figures beside it,
    so the entry has a scalar where every other metric has one and the profile is still in the
    artefact. The mean is over the pre-norm positions only; a mean that includes the post-norm
    output is a basis artefact divided by the depth.
    """
    from senbonzakura_check import measurement

    # THE FIVE PINNED FIELDS ARE NAMED RATHER THAN SPREAD, which the guard on this allows in those
    # words: name them explicitly where the writer knows better than the derivation does. It does
    # here, because `report.pinned` was derived from the model, tokeniser and prompts that produced
    # these very numbers, while a `stamps.pinned(...)` call at this point would re-derive them from
    # whatever a caller happened to hand the writer. The one spelling this deliberately avoids is
    # `**a_local`, which the guard skips on the strength of a dedicated test that `score` has and
    # this writer does not, so using it would be passing the gate rather than satisfying it.
    measurement.stamp(doc, "residual_leak", report.mean, "pre-norm-residual-per-position",
                      n=report.probe_prompts, by_estimator=True,
                      input_digest=report.pinned["input_digest"],
                      partition=report.pinned["partition"],
                      prompt_format=report.pinned["prompt_format"],
                      precision=report.pinned["precision"],
                      tool_version=report.pinned["tool_version"],
                      basis=PRE_NORM_BASIS, measured_at=AT_POSITION,
                      per_position=list(report.leak_per_position),
                      positions=report.positions)
    out = report.output
    if out is not None and out.along_post_norm_direction is not None:
        measurement.stamp(doc, "residual_leak", out.along_post_norm_direction,
                          "post-norm-output-basis", n=report.probe_prompts, by_estimator=True,
                          input_digest=report.pinned["input_digest"],
                          partition=report.pinned["partition"],
                          prompt_format=report.pinned["prompt_format"],
                          precision=report.pinned["precision"],
                          tool_version=report.pinned["tool_version"],
                          basis=POST_NORM_BASIS, measured_at=AT_OUTPUT,
                          norm_condition_number=out.norm_condition_number,
                          norm_diagonal_agreement=out.norm_diagonal_agreement,
                          diagonal_from=out.diagonal_from)
    elif out is not None:
        doc["residual_leak_output_basis_refused"] = out.refused_because
    if report.warnings:
        doc["residual_leak_warnings"] = list(report.warnings)
    return doc


def unmeasurable_because(mode, k):
    """Why this recipe has no single direction to report a leak for, or None when it has one.

    THE METRIC TAKES ONE VECTOR, and that is not a limitation to be papered over. The quantity is
    the component of the residual along *the* refusal direction, so it is defined against one
    direction and nothing else. A recipe that applied two directions per layer has a two
    dimensional subspace, and the leak out of a subspace is a different quantity with a different
    derivation: the figure the published claim is stated in is the single-direction one.

    So a run that cannot produce the figure says which recipe it used and why that recipe has no
    such figure. The alternative is picking one of the K directions and reporting its leak under
    the plain name, which would be a true number about something nobody asked about, published
    beside a claim it does not support. That is the failure this project keeps finding in other
    people's measurements and it is not going to ship one.

    `mode` and `k` are the winning trial's `dir_mode` and `num_directions`, taken from the same
    values the bake was given, so this cannot disagree with what was actually applied.
    """
    if int(k) != 1:
        return (f"the recipe applied {int(k)} directions per layer, and the residual leak is "
                f"defined against one. The leak out of a {int(k)} dimensional subspace is a "
                f"different quantity, so no figure is reported rather than one of the "
                f"{int(k)} being reported under the plain name.")
    if mode != "single":
        return (f"the recipe applied a different direction at every layer (dir_mode "
                f"{mode!r}), so there is no one direction for a profile across positions to be "
                f"about. A per-position figure against each position's own direction is a "
                f"different measurement and is not this one.")
    return None


#: Nobody asked for a leak report. The honest default, and not a criticism of the run.
LEAK_NOT_REQUESTED = "not_requested"
#: A report WAS asked for and no figure came back, with the reason beside it.
LEAK_REQUESTED_BUT_NOT_MEASURED = "requested_but_not_measured"
#: A profile across positions, and the output figure where the final norm allowed one.
LEAK_MEASURED = "measured"


def leak_block(report=None, *, refused_because=None, prompts=None, requested=True):
    """The leak result as a record block, ALWAYS present, whether or not it was measured.

    THE THREE STATES ARE KEPT APART, exactly as `_capability_block` keeps its own three apart and
    for the reason written there: a run nobody asked for a figure from and a run that was asked
    and could not produce one are different facts, and an artefact that spells them the same way
    hides the second one, which is the only one that needs reading.

    An absent key would be worse than either, because it reads as whoever wrote the file
    forgetting rather than as a statement.

    THE COUNT IS `probe_prompts` AND NOT `prompts`, which is not a style choice.
    `tools/ci/check_prompt_artefacts.py` refuses a committed artefact carrying a key named
    `prompts` at any depth, because that is how a harmful prompt set reaches a public repository
    wearing a measurement's name. It is the one control between such a set and a push and it is
    never widened to accommodate a field; the field is renamed. Caught by that gate's own test
    before this reached a commit, which is the gate working exactly as intended.

    When `requested` is true, exactly one of `report` and `refused_because` is given. Both, or
    neither, is a caller bug and raises here rather than writing a block that claims the figure
    exists and does not.
    """
    if not requested:
        return {"state": LEAK_NOT_REQUESTED, "measured": False, "why_not": None,
                "probe_prompts": None}
    if (report is None) == (refused_because is None):
        raise ValueError(
            "leak_block takes a report or a reason it has none, and exactly one of them: a block "
            "carrying both would say the figure exists and does not.")
    if report is None:
        return {"state": LEAK_REQUESTED_BUT_NOT_MEASURED, "measured": False,
                "why_not": refused_because,
                "probe_prompts": None if prompts is None else int(prompts)}
    out = report.output
    return {
        "state": LEAK_MEASURED,
        "measured": True,
        "basis": report.basis,
        "positions": report.positions,
        "probe_prompts": report.probe_prompts,
        "mean": report.mean,
        "per_position": list(report.leak_per_position),
        # The post-norm figure in the basis the norm maps into, and NOT the pre-norm direction read
        # at the output, which is the misreading this module's docstring exists to prevent. Absent
        # with its reason rather than filled in, which is also what the checker's
        # `a-removed-direction-with-no-basis-named` requires.
        "output_along_post_norm_direction": (
            None if out is None else out.along_post_norm_direction),
        "output_basis_refused": None if out is None else out.refused_because,
        "warnings": list(report.warnings),
    }


def _decoder_layers(model):
    """The decoder stack, resolved by the one function that already resolves it.

    Imported inside the call rather than at module scope on purpose. `cli.py` pulls in torch,
    optuna and transformers, and this module is meant to cost nothing to import; by the time
    anybody calls this they are holding a live model, so the cost is already paid. Re-listing the
    architecture paths here would be a second copy of a list this project has already drifted
    once, which is the defect `writers.py` exists to keep closed.
    """
    from .cli import _decoder_layers as resolve
    return resolve(model)


def _base_model(model):
    """The module the decoder stack and the final norm both hang off, or None."""
    for path in BASE_MODEL_PATHS:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            if getattr(obj, "layers", None) is not None or getattr(obj, "h", None) is not None:
                return obj
    return None


def final_norm(model):
    """The final norm module and the attribute it was found under, or (None, why not).

    A name this project does not recognise is a degradation and not a failure, because the
    per-position profile never touches the norm. The reason comes back phrased for a person so
    the caller can put it in front of one rather than inventing a sentence.
    """
    base = _base_model(model)
    if base is None:
        return None, (f"the model tree on {type(model).__name__} does not expose a base module "
                      f"this tool recognises, so its final norm could not be found. The "
                      f"per-position figures are unaffected.")
    for name in FINAL_NORM_NAMES:
        mod = getattr(base, name, None)
        if mod is not None and getattr(mod, "weight", None) is not None:
            return mod, name
    named = ", ".join(FINAL_NORM_NAMES)
    return None, (f"no final norm with a learned weight was found on "
                  f"{type(base).__name__}; the names looked for were {named}. Either this "
                  f"architecture normalises somewhere else or it names it something new, and "
                  f"either way the output basis cannot be worked out. The per-position figures "
                  f"are unaffected.")


def diagonal_candidates(weight):
    """The diagonals a final norm might actually be applying, as (description, tensor) pairs.

    Two, because there are two conventions in the wild and guessing between them by class name is
    how the gemma failure happened once already. Gemma's RMSNorm multiplies by `1 + weight` while
    everything else multiplies by `weight`, so both are offered and the caller picks by
    MEASURING which one makes `y / w` parallel to `x`. A candidate holding a zero entry is dropped
    here rather than later: dividing by it produces an infinity, and an infinity that reaches a
    cosine turns into a NaN, which compares false against every threshold and so would read as a
    refusal for the wrong reason.
    """
    w = weight.detach().double().flatten()
    out = []
    for label, cand in (("the final norm's weight", w),
                        ("one plus the final norm's weight", 1.0 + w)):
        if float(cand.abs().min()) > 0.0:
            out.append((label, cand))
    return out


def _unit(v, what):
    v = v.double().flatten()
    n = float(v.norm())
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"{what} has norm {n}, so it does not point anywhere and no component "
                         f"can be measured along it.")
    return v / n


def _block_output(out, where):
    """The residual tensor a decoder block handed back, whatever wrapper it came in."""
    if isinstance(out, tuple):
        out = out[0] if out else None
    if not isinstance(out, torch.Tensor):
        # Lint override, with the reason: TRY004 asks for TypeError on a type check, and every
        # failure this module raises is a ValueError about a model it cannot measure. A decoder
        # block returning something new is an architecture this tool has not met, not a caller
        # passing the wrong Python type, and `measure_leak`'s callers catch the one exception.
        raise ValueError(  # noqa: TRY004
            f"the last decoder block returned {type(out).__name__} rather than a tensor or a "
            f"tuple starting with one, so {where} could not be read. This is the position the "
            f"metric most wants and substituting the post-norm output for it would be a figure "
            f"from a different space, so the run stops here rather than reporting one.")
    return out


@torch.no_grad()
def measure_leak(model, tok, prompts, direction, *, log=print):
    """The residual leak per position, plus the output in both bases when the norm allows it.

    `prompts` are plain requests; they are rendered through `firsttoken.render_chat`, which is the
    project's one renderer, because a leak measured under a different prompt format from the
    refusal rate beside it is the defect that put the compass's read-out at a position the model
    never emits a verdict at.

    One forward pass per prompt with `output_hidden_states=True`, plus a hook on the final decoder
    block. Batching is deliberately not done: `hidden_states` materialises every position for the
    whole batch, the metric reads one token per prompt, and a padded batch would put that token at
    a pad position unless the padding side is managed, which is the defect `firsttoken.py` owns.
    One prompt at a time needs no padding and is correct.
    """
    from . import stamps
    from .firsttoken import render_chat

    if not prompts:
        raise ValueError("no probe prompts, so there is nothing to measure the residual of.")
    if len(prompts) < MIN_PROBE_PROMPTS:
        raise ValueError(
            f"{len(prompts)} probe prompt(s), below the floor of {MIN_PROBE_PROMPTS}. The mean of "
            f"fewer than that is not a property of the model and would be quoted as one.")

    layers = _decoder_layers(model)
    NL = len(layers)
    if NL == 0:
        raise ValueError("the decoder stack is empty, so there are no residual-stream positions.")
    d = _unit(direction, "the direction")

    degraded = []
    norm_mod, norm_name = final_norm(model)
    if norm_mod is None:
        # SAID OUT LOUD AS WELL AS RECORDED. A degradation that only reaches a field of the
        # returned object is invisible to somebody watching a terminal, and the figure they will
        # quote is the one they watched arrive.
        degraded.append(norm_name)
        log(f"  residual leak: {norm_name}")
        candidates = []
    else:
        # A candidate list that comes back empty is not warned about here: `_output_basis` refuses
        # the output figure and names the reason on the figure itself, which is where a reader
        # meets it, and the same sentence in two places is how they come to disagree.
        candidates = diagonal_candidates(norm_mod.weight)

    caught = {}

    def grab(_module, _inputs, out):
        caught["x"] = _block_output(out, f"position {NL}, what decoder layer {NL - 1} writes into")

    totals = [0.0] * (NL + 1)
    out_pre = 0.0
    cand_leak = [0.0] * len(candidates)
    cand_cos = [0.0] * len(candidates)
    # The normal of the hyperplane the norm maps the pre-norm one onto, per candidate diagonal.
    # `d / w` and not `d`, and normalised, because the metric divides by the residual norm and so
    # needs a unit normal for the two numbers to be on the same scale.
    post_norm_dirs = [_unit(d / w, f"the direction divided by {label}")
                      for label, w in candidates]

    handle = layers[-1].register_forward_hook(grab)
    beat = time.monotonic()
    try:
        for done, prompt in enumerate(prompts, 1):
            enc = tok(render_chat(tok, prompt), return_tensors="pt",
                      add_special_tokens=False).to(model.device)
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            if len(hs) != NL + 1:
                raise ValueError(
                    f"{type(model).__name__} returned {len(hs)} hidden states for {NL} decoder "
                    f"layers, where {NL + 1} are expected. The position vocabulary this metric "
                    f"reports in would not line up with the model's own, so no profile is "
                    f"reported rather than a mislabelled one.")
            # Positions 0 to NL-1 come straight off the tuple: entry i is what layer i READS,
            # which is the embedding output at i=0 and what layer i-1 wrote at i>0. Entry NL is
            # post-norm and is NOT position NL, so it is never used for the profile.
            for i in range(NL):
                totals[i] += leak_fraction(hs[i][0, -1, :].unsqueeze(0), d)
            x = caught["x"][0, -1, :].double()
            totals[NL] += leak_fraction(x.unsqueeze(0), d)
            y = hs[NL][0, -1, :].double()
            out_pre += leak_fraction(y.unsqueeze(0), d)
            for j, (_label, w) in enumerate(candidates):
                cand_leak[j] += leak_fraction(y.unsqueeze(0), post_norm_dirs[j])
                cand_cos[j] += float(torch.nn.functional.cosine_similarity(y / w, x, dim=0))
            if time.monotonic() - beat >= HEARTBEAT_SECONDS:      # pragma: no cover - timing
                beat = time.monotonic()
                log(f"  residual leak: {done} of {len(prompts)} probe prompts")
    finally:
        handle.remove()

    n = len(prompts)
    profile = tuple(t / n for t in totals)
    output = None
    if norm_mod is not None:
        output = _output_basis(
            along_pre=out_pre / n, norm_name=norm_name,
            readings=[(label, cand_leak[j] / n, cand_cos[j] / n, w)
                      for j, (label, w) in enumerate(candidates)])
        if output.refused_because is not None:
            log(f"  residual leak: NO OUTPUT-BASIS FIGURE. {output.refused_because}")
    return LeakReport(positions=NL + 1, leak_per_position=profile, probe_prompts=n,
                      pinned=stamps.pinned(prompts=prompts, model=model, tok=tok),
                      output=output, warnings=tuple(degraded))


def _output_basis(*, along_pre, norm_name, readings):
    """Pick the diagonal by measurement, then either report the output basis or refuse it.

    Two gates, in this order, and each refusal names the figure it is withholding rather than
    returning a smaller number with no comment.

    1. Is the norm a diagonal map of the normalised residual at all? If `y = (x / rms(x)) * w`
       then `y / w` is a positive multiple of `x` exactly, so the measured cosine settles the
       question and also settles WHICH candidate diagonal is the real one. A LayerNorm fails this
       because it mean-centres and adds a bias, and no `w` can express `d` in its output basis.
    2. Is the division well enough conditioned to mean anything? See `NORM_CONDITION_CEILING`.
    """
    if not readings:
        return OutputBasis(
            along_pre_norm_direction=along_pre, along_post_norm_direction=None,
            norm_condition_number=None, norm_diagonal_agreement=None, diagonal_from=None,
            refused_because=(
                f"the final norm at `{norm_name}` applies a weight with a zero entry, so the map "
                f"from the pre-norm residual stream to the output has no inverse and the "
                f"direction has no image in the output basis. Read the per-position figures, "
                f"which do not divide by anything."))

    label, leak, cos, w = max(readings, key=lambda r: r[2])
    cond = float(w.abs().max() / w.abs().min())
    if cos < 1.0 - NORM_DIAGONAL_TOL:
        return OutputBasis(
            along_pre_norm_direction=along_pre, along_post_norm_direction=None,
            norm_condition_number=cond, norm_diagonal_agreement=cos, diagonal_from=None,
            refused_because=(
                f"the final norm at `{norm_name}` is not a per-dimension rescaling of the "
                f"normalised residual: dividing its output by the best candidate weight leaves "
                f"something only {cos:.6f} parallel to its input, where an exact rescaling gives "
                f"1. A norm that also centres the residual or adds a bias moves the state off "
                f"every hyperplane through the origin, so there is no direction in the output "
                f"basis that corresponds to the pre-norm one, and none is invented. Read the "
                f"per-position figures, which are taken before the norm."))
    if cond > NORM_CONDITION_CEILING:
        return OutputBasis(
            along_pre_norm_direction=along_pre, along_post_norm_direction=None,
            norm_condition_number=cond, norm_diagonal_agreement=cos, diagonal_from=label,
            refused_because=(
                f"the final norm at `{norm_name}` is too ill conditioned to express the pre-norm "
                f"direction in the output basis on this model: its largest weight is {cond:.0f} "
                f"times its smallest, above the ceiling of {NORM_CONDITION_CEILING:.0f}. The "
                f"algebra is exact at any conditioning; the measurement is not, because dividing "
                f"the direction by a weight that small concentrates it into a handful of "
                f"coordinates, and the figure that comes out reads those coordinates rather than "
                f"the direction. Read the per-position figures, which need no division and are "
                f"the basis the residual writers write into."))
    return OutputBasis(
        along_pre_norm_direction=along_pre, along_post_norm_direction=leak,
        norm_condition_number=cond, norm_diagonal_agreement=cos, diagonal_from=label,
        refused_because=None)
