#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The open experiments, as one registry: what each needs, what it costs, and where it can run.

WHY A REGISTRY AND NOT FOUR SCRIPTS

Four experiments were parked across two handoffs with "needs a card" written beside them, and that
note turned out to be wrong about two of them. The cost of a parked experiment is not the compute,
it is that nobody can tell at a glance which ones are actually blocked, so they all get treated as
blocked and none of them runs. This file is the answer to "what can run tonight and what needs
money", and it is executable so the answer cannot drift from the plan.

    senbonzakura-experiments --plan            what every arm needs, and where it can run
    senbonzakura-experiments --plan --pod      the batch script for one rented session
    senbonzakura-experiments --run leak-field  one arm, here

THE RULE THAT MADE THIS NECESSARY

An arm's `where` is derived from its declared VRAM against the machine it is asked about, never
hand-written. The claim "the smallest in-stack model with a prediction head is 45 layers of 4096"
sat in a handoff and was false: a 24-layer 2048-wide one exists, is ungated and is Apache-2.0, and
believing the note would have rented a card to answer a question a laptop can answer. So the
registry holds the model and the figure, and the verdict is arithmetic.
"""
import argparse
import dataclasses
import json
import shutil
import subprocess
import sys

#: Bytes per parameter at bf16, plus the slack a forward pass needs for activations and a KV
#: cache. Deliberately generous: an arm that is called feasible and then dies at 94% of a card has
#: cost more than an arm that was called infeasible and was not.
BYTES_PER_PARAM = 2
ACTIVATION_SLACK = 1.35


@dataclasses.dataclass(frozen=True)
class Arm:
    """One experiment: the question, the cost, and the command that answers it."""

    key: str
    question: str
    #: Why the answer matters, in one sentence, so a reader can rank arms without asking.
    stake: str
    #: Billions of parameters of the largest model the arm must hold at once, or 0 when it holds
    #: none (a metadata-only or arithmetic-only arm).
    params_b: float
    #: Rough wall-clock on a card that fits it. An estimate, and labelled as one everywhere.
    hours: float
    #: What has to exist before it can run. Each entry is a sentence, not a package name.
    needs: tuple
    #: True when the code to run it exists today. False means this file specifies it and nothing
    #: implements it yet, which is a different kind of blocked from needing hardware.
    built: bool
    #: The command, as a list, or None when `built` is False.
    command: tuple = ()

    def vram_gb(self):
        """What it needs resident, in GB. Zero for an arm that loads no model."""
        if not self.params_b:
            return 0.0
        return self.params_b * 1e9 * BYTES_PER_PARAM * ACTIVATION_SLACK / 1e9

    def fits(self, card_gb):
        return self.vram_gb() <= card_gb

    def where(self, card_gb):
        """Derived, never written down. The whole reason this file is executable."""
        if not self.built:
            return "blocked: not built yet"
        if self.vram_gb() == 0:
            return "here: loads no model"
        if self.fits(card_gb):
            return f"here: {self.vram_gb():.1f} GB of {card_gb:.0f} GB"
        return f"rent: needs {self.vram_gb():.1f} GB, this card has {card_gb:.0f} GB"


#: Each arm's prerequisites, as whole sentences. Held as module constants rather than written
#: inline, because a multi-line string inside a tuple reads as two elements to anyone skimming and
#: the linter flags it for exactly that reason. One name per list also means a prerequisite can be
#: cited from a plan document without copying it.
NEEDS_LEAK_FIELD = (
    "published abliterated checkpoints and their base models on disk",
    "nothing else: `leak` extracts its own direction",
)
NEEDS_NORM_GAIN = (
    "gemma-2-2b-it and a Qwen of similar size",
    "a refusal-free task to read the off-target effect on",
    ("the strength sweep that matches the two arms on MEASURED leak rather than nominal"
     " weight, which is the part that does not exist yet"),
)
NEEDS_PREDICTION_HEAD = (
    ("a model with a prediction head small enough to hold: Mapika/decider-2b is 24 layers"
     " of 2048, Apache-2.0 and ungated"),
    "a flag that includes or excludes the head, which does not exist yet",
    ("NOTE: decider-2b declares linear attention, so it settles the behaviour and is not a"
     " representative mainstream decoder; confirm on a larger one when renting"),
)
NEEDS_DECOY = (
    ("no DDO checkpoint is published, so the decoy has to be injected here from the paper's"
     " description. That is a build rather than a download, and it is why this arm is not"
     " simply cheap"),
    "a small model to inject into",
)
NEEDS_GRID = (
    ("rented compute, and this is the one arm where that is the actual answer rather than a"
     " note nobody checked"),
    "`stream-bake`, for the models that will not fit even on a rented card",
)

ARMS = (
    Arm(
        key="leak-field",
        question="Do other people's published abliterations actually remove the direction?",
        stake="The leak metric's whole claim is that it is a ruler rather than a self-assessment."
              " Nothing has ever pointed it at a checkpoint this project did not make, so the"
              " claim is untested on the only case that distinguishes it from the flag.",
        params_b=2.0,
        hours=1.0,
        needs=NEEDS_LEAK_FIELD,
        built=True,
        command=("senbonzakura", "leak"),
    ),
    Arm(
        key="norm-gain-reversal",
        question="Is the Gemma-versus-Qwen sign reversal in arXiv 2607.17427 a norm-gain"
                 " artefact rather than a fact about model families?",
        stake="The strongest publishable result available here. That paper reads a sign reversal"
              " as a difference between families; we measured in August that the edit never"
              " properly reached Gemma's residual stream, so the two arms were never given the"
              " same effective edit. If the reversal vanishes once the strengths are matched on"
              " measured leak, the finding is about an implementation detail and the correction"
              " is ours.",
        params_b=2.6,
        hours=3.0,
        needs=NEEDS_NORM_GAIN,
        built=False,
    ),
    Arm(
        key="prediction-head",
        question="Should an abliteration edit a multi-token prediction head, or skip it?",
        stake="An open design question with two defensible answers and no measurement. The head"
              " writes to the residual stream like any other block, so the editor currently"
              " treats it as one; whether that helps or hurts has never been read.",
        params_b=2.0,
        hours=2.0,
        needs=NEEDS_PREDICTION_HEAD,
        built=False,
    ),
    Arm(
        key="decoy-exposure",
        question="Does `validate` expose a decoy direction, or does it pass one?",
        stake="Decoy injection makes a low leak figure the signature of a SUCCESSFUL defence,"
              " which is the one input that inverts the metric shipped this release. `validate`"
              " should catch it: a decoy is high-magnitude and orthogonal to refusal, so it"
              " should fail leave-one-cluster-out generalisation while scoring well on"
              " magnitude, which is the exact contrast that command was built for. That is a"
              " hypothesis stated in the docs and nothing has tested it.",
        params_b=1.5,
        hours=4.0,
        needs=NEEDS_DECOY,
        built=False,
    ),
    Arm(
        key="grid-at-scale",
        question="Does the multi-direction advantage hold at 7B and above?",
        stake="The honest limit of `head-to-head`. Every arm so far ran on a 6 GB laptop card,"
              " so the published grid stops at 1.7B while the nearest rival study runs sixteen"
              " models at 7B to 14B. Their range is better evidence about large models than ours"
              " and the CONTRACT now says so. This is the arm that changes that.",
        params_b=14.0,
        hours=40.0,
        needs=NEEDS_GRID,
        built=True,
        command=("senbonzakura", "head-to-head"),
    ),
)


def card_gb():
    """The largest card visible here, in GB, or 0.0 when there is none.

    Read rather than assumed, because the whole point of the `where` column is that it is about
    the machine asking. A machine with no card is not an error: it gives every model-loading arm
    the honest answer "rent".
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return 0.0
    try:
        done = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return 0.0
    sizes = [float(line.strip()) / 1024 for line in done.stdout.splitlines() if line.strip()]
    return max(sizes, default=0.0)


def plan(arms=ARMS, *, gb=None):
    """The table, as rows. Separated from printing so a test can read it."""
    gb = card_gb() if gb is None else gb
    rows = [{"key": arm.key, "hours": arm.hours, "vram_gb": round(arm.vram_gb(), 1),
             "built": arm.built, "where": arm.where(gb), "question": arm.question}
            for arm in arms]
    return {"card_gb": gb, "arms": rows,
            "runnable_here": [r["key"] for r in rows if r["where"].startswith("here")],
            "needs_renting": [r["key"] for r in rows if r["where"].startswith("rent")],
            "needs_building": [r["key"] for r in rows if r["where"].startswith("blocked")]}


def pod_script(arms=ARMS, *, gb=None):
    """A batch for one rented session: every arm that will not run on this machine.

    ONE SESSION RATHER THAN ONE PER ARM, which is the only part of this that saves money. A rented
    card is billed by the hour from the moment it boots, so four arms launched separately pay four
    setup costs and four idle tails. This orders them longest first, so the session's total is the
    longest arm rather than the sum when a pod has more than one card.
    """
    gb = card_gb() if gb is None else gb
    chosen = sorted((a for a in arms if not a.fits(gb) and a.built),
                    key=lambda a: -a.hours)
    lines = ["#!/usr/bin/env bash",
             "# Generated by tools/research/experiments.py --plan --pod. Do not edit here.",
             "set -euo pipefail",
             "",
             "# Longest arm first: a pod is billed from boot, so the tail of a short arm waiting",
             "# on a long one is the cost this ordering avoids.",
             f"# Estimated total if run in series: {sum(a.hours for a in chosen):.0f} hours.",
             ""]
    if not chosen:
        lines.append("echo 'Nothing needs renting: every built arm fits this machine.'")
        return "\n".join(lines) + "\n"
    for arm in chosen:
        lines += [f"# ── {arm.key}: ~{arm.hours:.0f}h, needs {arm.vram_gb():.1f} GB ──",
                  f"#    {arm.question}",
                  f"echo '>>> {arm.key}'",
                  "#    " + " ".join(arm.command) + "   # fill in the arm's own flags",
                  ""]
    unbuilt = [a.key for a in arms if not a.built]
    if unbuilt:
        lines += ["# NOT IN THIS BATCH, because the code does not exist yet rather than because",
                  "# the hardware is missing. Building these is cheaper than renting:",
                  *(f"#   {k}" for k in unbuilt), ""]
    return "\n".join(lines) + "\n"


def build_parser():
    p = argparse.ArgumentParser(
        prog="experiments",
        description="The open experiments: what each needs, what it costs, where it can run.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true", help="print the table and exit")
    p.add_argument("--pod", action="store_true",
                   help="with --plan, print a batch script for one rented session")
    p.add_argument("--json", action="store_true", help="with --plan, print the table as JSON")
    p.add_argument("--card-gb", dest="gb", type=float, default=None,
                   help="pretend this machine has a card of this size, for planning")
    p.add_argument("--run", default=None, metavar="ARM",
                   help=f"run one arm here. One of: {', '.join(a.key for a in ARMS)}")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.run:
        arm = next((x for x in ARMS if x.key == a.run), None)
        if arm is None:
            raise SystemExit(f"no arm called {a.run!r}. Try --plan.")
        if not arm.built:
            raise SystemExit(
                f"{arm.key} is specified here and not implemented. What it still needs:\n"
                + "\n".join(f"  - {n}" for n in arm.needs))
        raise SystemExit(
            f"{arm.key} runs through the tool's own command rather than this script, which only "
            f"plans:\n    {' '.join(arm.command)}\nSee --plan for what it needs.")

    table = plan(gb=a.gb)
    if a.json:
        print(json.dumps(table, indent=2))
        return 0
    if a.pod:
        print(pod_script(gb=a.gb), end="")
        return 0
    print(f"This machine's largest card: {table['card_gb']:.0f} GB\n")
    print(f"{'arm':<22} {'hours':>6} {'VRAM':>7}  where")
    print("-" * 78)
    for row in table["arms"]:
        print(f"{row['key']:<22} {row['hours']:>6.0f} {row['vram_gb']:>6.1f}G  {row['where']}")
    print()
    for label, keys in (("runnable here", table["runnable_here"]),
                        ("needs renting", table["needs_renting"]),
                        ("needs building first", table["needs_building"])):
        print(f"{label}: {', '.join(keys) if keys else 'none'}")
    print("\nThe hours are estimates and are labelled as such wherever they appear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
