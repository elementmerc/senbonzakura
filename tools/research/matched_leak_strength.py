#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Match two models on the edit that ACTUALLY LANDED, not on the number you asked for.

THE PROBLEM THIS SOLVES, AND IT HAS ALREADY COST A PUBLISHED FIGURE

Two models given `--strength 1.0` have not been given the same edit. On 2026-08-05 this project
measured that the same weight edit reached 0.578 of Gemma's residual stream against Qwen3's 0.016,
because Gemma applies `post_attention_layernorm` and `post_feedforward_layernorm` to each
sublayer's output before the residual add. An edit to the output projection therefore arrives
attenuated by a norm gain that nothing in the command line knows about. Every Gemma number this
project published before that was withdrawn.

The same trap is sitting under somebody else's paper. arXiv 2607.17427 measures abliteration's
off-target effects on a task containing no refusals and reports confidence shifts whose sign
REVERSES between Gemma and Qwen, reading that as a difference between model families. It may be a
difference in how much of the edit landed. A sign reversal is exactly what a partially applied edit
can produce, and nothing in that design would have shown it.

WHAT MATCHING ON MEASURED LEAK MEANS

`senbonzakura leak` reports how much of the refusal direction is still in the residual stream. That
figure is downstream of every attenuation the architecture applies, so it is a measurement of the
edit that landed rather than the edit that was requested. So:

    for each model:  find the strength s at which the measured leak hits one agreed target
    then:            compare the two models at THOSE strengths, not at a shared number

If the reversal survives that, it is about the families. If it vanishes, it was about the norm gain,
and the correction belongs to whoever measures it.

WHY THE SEARCH IS A BISECTION AND NOT A SWEEP

A sweep spends its budget evenly over a range where almost all the interesting behaviour is in a
narrow band, and each sample costs a bake plus a set of forward passes. Leak against strength is
monotonically decreasing in every case measured so far, which is what a bisection needs, and the
monotonicity is CHECKED rather than assumed: a bracket whose ends disagree with it is reported as
such instead of being bisected into a confident wrong answer.

THE MEASUREMENT IS INJECTED, ON PURPOSE

`find_strength` takes a callable. The search logic is then testable without a card, which matters
because this file's job is to be correct about a number that will be published, and the arithmetic
is the half that can be checked for free. Driving a real model is `measure_with_senbonzakura`,
which is a thin shell around the tool's own commands.
"""
import argparse
import json
import math
import pathlib
import sys

#: How close to the target leak counts as a hit. Not tighter than the measurement's own
#: reproducibility: the leak figure needs no judge and no sampling, so it is stable to many digits,
#: but the BAKE in front of it is float arithmetic over a different BLAS path per batch size.
DEFAULT_TOLERANCE = 0.02

#: Hard cap on measurements. Each one is a bake plus a forward pass over the probe set, so this is
#: the budget a person is actually spending. A bisection halves the bracket each time, so twelve
#: steps resolve a strength of 0 to 4 to about one part in a thousand.
DEFAULT_MAX_STEPS = 12


class MatchError(Exception):
    """A refusal with a sentence that says what to do about it."""


def _finite(value, what):
    if value is None or not math.isfinite(float(value)):
        raise MatchError(
            f"{what} came back as {value!r}. A leak measurement that is not a finite number means "
            f"the measurement failed rather than that the leak is zero, and bisecting on it would "
            f"produce a strength that looks measured.")
    return float(value)


def find_strength(measure, target, *, low=0.0, high=2.0, tolerance=DEFAULT_TOLERANCE,
                  max_steps=DEFAULT_MAX_STEPS, log=print):
    """The strength at which `measure(s)` hits `target`, by bisection on a checked bracket.

    `measure` takes a strength and returns a leak figure. It is called at most `max_steps + 2`
    times: once per bracket end, then once per halving.

    Returns a dict rather than a float, and that is deliberate: a caller publishing this number
    needs the bracket, the step count and whether it actually converged, and a bare float invites
    all three to be dropped on the way to a paper.
    """
    if not 0.0 <= low < high:
        raise MatchError(f"the bracket [{low}, {high}] is not an interval with low below high.")
    target = _finite(target, "the target leak")

    at_low = _finite(measure(low), f"the leak at strength {low}")
    at_high = _finite(measure(high), f"the leak at strength {high}")
    log(f"  bracket: leak {at_low:.4g} at s={low}, {at_high:.4g} at s={high}")

    # MONOTONICITY, CHECKED. The whole method assumes more strength removes more direction. Every
    # model measured so far behaves that way and a model that does not is a finding about that
    # model, not an input to average over.
    if at_high >= at_low:
        raise MatchError(
            f"leak did not fall across the bracket: {at_low:.4g} at strength {low} and "
            f"{at_high:.4g} at {high}. Bisection assumes more strength removes more direction, so "
            f"this is a result about this model rather than a bracket to search. Widen the bracket "
            f"or report it.")
    if not (at_high <= target <= at_low):
        raise MatchError(
            f"the target leak {target:.4g} is outside what this bracket reaches: {at_high:.4g} to "
            f"{at_low:.4g}. Widen the bracket, or pick a target the model can actually hit. A "
            f"strength clamped to a bracket end is not a matched strength and nothing downstream "
            f"would say so.")

    steps = []
    lo, hi, lo_leak, hi_leak = low, high, at_low, at_high
    for step in range(max_steps):
        mid = (lo + hi) / 2.0
        leak = _finite(measure(mid), f"the leak at strength {mid}")
        steps.append({"strength": mid, "leak": leak})
        log(f"  step {step + 1}: s={mid:.4f} -> leak {leak:.4g}")
        if abs(leak - target) <= tolerance * abs(target or 1.0):
            return {"strength": mid, "leak": leak, "target": target, "converged": True,
                    "steps": steps, "bracket": [low, high],
                    "bracket_leak": [at_low, at_high]}
        if leak > target:
            lo, lo_leak = mid, leak
        else:
            hi, hi_leak = mid, leak

    # NOT CONVERGED IS AN ANSWER, and it is reported as one. A loop that fell out of its budget and
    # returned its last midpoint as though it had hit the target is how a number with no precision
    # behind it reaches a table.
    return {"strength": (lo + hi) / 2.0, "leak": steps[-1]["leak"] if steps else None,
            "target": target, "converged": False, "steps": steps,
            "bracket": [lo, hi], "bracket_leak": [lo_leak, hi_leak]}


def compare(results):
    """What two matched arms say, side by side, with the caveat the comparison needs.

    `results` maps a model name to a `find_strength` result. Returns a dict for a JSON file,
    because a figure that lives in a terminal is a figure nobody can re-read.
    """
    if len(results) < 2:
        raise MatchError("a matched comparison needs at least two models.")
    unconverged = sorted(k for k, v in results.items() if not v.get("converged"))
    rows = {k: {"strength": v["strength"], "leak": v["leak"], "converged": v["converged"]}
            for k, v in results.items()}
    strengths = [v["strength"] for v in results.values()]
    return {
        "arms": rows,
        "strength_spread": max(strengths) - min(strengths),
        "unconverged": unconverged,
        "comparable": not unconverged,
        "means": (
            "each arm was edited until its MEASURED residual leak hit one target, so the arms are "
            "matched on the edit that landed rather than on the strength that was requested. A "
            "difference between them is therefore not a difference in how much of the edit "
            "arrived. It is still not a controlled comparison of anything else: the models differ "
            "in training data, size and tokeniser, and this matches one variable."),
        "caveat_if_unconverged": (
            f"{unconverged} did not reach the target within the step budget, so those arms are NOT "
            f"matched and no difference involving them can be attributed to anything."
            if unconverged else None),
    }


def measure_with_senbonzakura(model, directions, *, probe_n=32, device="cuda", workdir,
                              log=print):
    """A `measure` callable that bakes at a strength and reads the real leak figure.

    Shells out to the tool's own commands rather than importing their internals, for the reason
    `head-to-head` gives for the same choice: a measurement that runs the published path is the one
    a reader can reproduce, and an in-process shortcut measures a path nobody else can take.
    """
    import subprocess

    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def measure(strength):
        out = workdir / f"s{strength:.6f}"
        if strength == 0.0:
            # The unedited model, which is the bracket's top end. No bake: baking at zero would
            # still rewrite every shard to produce a byte-identical copy.
            target = model
        else:
            subprocess.run(
                [sys.executable, "-m", "senbonzakura.streambake",
                 "--model", str(model), "--directions", str(directions),
                 "--out", str(out), "--strength", f"{strength:.6f}"],
                check=True)
            target = out
        done = subprocess.run(
            [sys.executable, "-m", "senbonzakura.leak", "--model", str(target),
             "--device", device, "--probe-n", str(probe_n),
             "--out", str(workdir / f"leak-s{strength:.6f}.json")],
            check=True, capture_output=True, text=True)
        log(done.stdout.strip().splitlines()[-1] if done.stdout.strip() else "")
        doc = json.loads((workdir / f"leak-s{strength:.6f}.json").read_text(encoding="utf-8"))
        block = doc.get("residual_leak") or {}
        if not block.get("measured"):
            raise MatchError(
                f"the leak at strength {strength} was not measured: "
                f"{block.get('why_not')!r}. There is nothing to bisect on.")
        return block["mean"]

    return measure


def build_parser():
    p = argparse.ArgumentParser(
        prog="matched_leak_strength",
        description="Find the strength at which each model's MEASURED leak hits one target, so "
                    "two models can be compared on the edit that landed rather than the one "
                    "that was requested.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
why this exists:
  Two models given --strength 1.0 have not been given the same edit. The same weight edit
  reached 0.578 of Gemma's residual stream against Qwen3's 0.016, because Gemma norms each
  sublayer's output before the residual add. Every Gemma figure this project published
  before measuring that was withdrawn.

  arXiv 2607.17427 reports off-target effects whose sign reverses between Gemma and Qwen and
  reads that as a difference between families. It may be a difference in how much of the
  edit landed, which is what this measures.

needs a card:
  Each measurement is a bake plus a forward pass over the probe set, so this runs where the
  model fits. The search itself is arithmetic and is unit tested without one.
""")
    p.add_argument("--model", action="append", required=True, metavar="DIR",
                   help="a model directory. Pass twice or more for a matched comparison")
    p.add_argument("--directions", action="append", required=True, metavar="FILE",
                   help="the directions file for each --model, in the same order")
    p.add_argument("--target", type=float, required=True,
                   help="the leak figure every arm is edited until it reaches")
    p.add_argument("--low", type=float, default=0.0, help="bracket low end (default: 0.0)")
    p.add_argument("--high", type=float, default=2.0, help="bracket high end (default: 2.0)")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help=f"relative closeness that counts as a hit (default: {DEFAULT_TOLERANCE})")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                   help=f"measurement budget per arm (default: {DEFAULT_MAX_STEPS})")
    p.add_argument("--probe-n", type=int, default=32, help="prompts per leak measurement")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workdir", required=True, help="where bakes and leak files are written")
    p.add_argument("--out", required=True, help="where the comparison JSON is written")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if len(a.model) != len(a.directions):
        raise SystemExit(
            f"{len(a.model)} --model and {len(a.directions)} --directions. One directions file per "
            f"model, in the same order: a direction set paired with the wrong model would produce "
            f"a strength that means nothing.")
    results = {}
    try:
        for model, directions in zip(a.model, a.directions, strict=True):
            print(f"{model}:")
            measure = measure_with_senbonzakura(
                model, directions, probe_n=a.probe_n, device=a.device,
                workdir=pathlib.Path(a.workdir) / pathlib.Path(model).name)
            results[str(model)] = find_strength(
                measure, a.target, low=a.low, high=a.high, tolerance=a.tolerance,
                max_steps=a.max_steps)
        doc = compare(results)
    except MatchError as e:
        raise SystemExit(f"matched-leak-strength: {e}") from e
    pathlib.Path(a.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\n  written to {a.out}")
    for name, row in doc["arms"].items():
        state = "" if row["converged"] else "  NOT CONVERGED, so this arm is not matched"
        print(f"  {name}: strength {row['strength']:.4f} -> leak {row['leak']:.4g}{state}")
    if not doc["comparable"]:
        print("\n  THE ARMS ARE NOT MATCHED. No difference between them can be attributed yet.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
