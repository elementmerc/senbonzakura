# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Named ablation recipes, so a run can say which method it was rather than list its knobs.

WHY THIS EXISTS

The published methods in this field differ along three axes and are usually discussed as though
they differ along one. Separating them is most of the value here:

    the EDIT          does the surgery change the length of each weight row, or only its direction?
    the SEARCH        is the strength fixed, or optimised over many trials?
    the DIRECTIONS    one difference-of-means axis, or a span of several?

Every combination below is reachable today by hand, from flags that already exist. What was
missing is a NAME, and the name is what makes a comparison a comparison: two runs whose settings a
reader has to diff by eye are not two arms of an experiment, they are two runs.

WHY THE NAMES DESCRIBE BEHAVIOUR RATHER THAN CITING PROJECTS

None of these is somebody else's tool. `single-pass` is not DECCP, it is this codebase configured
the way DECCP is described as working, which is a different claim and a much weaker one. Calling
an arm `deccp` would say we had reproduced their implementation, and what we would actually have
measured is ours. Each recipe records what it RESEMBLES separately from what it IS.

WHY ONE OF THESE IS A CELL RATHER THAN A TOOL

`single-pass` differs from `searched` in TWO ways at once: it does not search, and it removes one
direction. A result comparing them cannot say which half did the work, and this project's whole
argument is about direction count. `searched-one-direction` is the missing cell:

                        one direction        as many as separate
    no search           single-pass          (not built; nothing does this)
    searched            searched-one-        searched
                        direction

Reading down a column isolates the search. Reading across a row isolates the direction count.
Neither is readable from the diagonal alone, and the diagonal is what the field publishes.

WHAT THIS IS FOR

The field ships on a claimed one to three percent degradation with no benchmarks behind it, and a
comparative study reports one family at eight points on grade-school arithmetic. Nobody can
currently say which method to use for a given base model, because nobody measures. A tool that
holds several methods and one ruler can answer that, and it does not have to bet on a method to
do it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Method:
    """One named recipe: the flags it fixes, and what it does and does not claim."""

    name: str
    summary: str
    #: Argument values this recipe pins. Anything absent is left to the command line.
    settings: dict = field(default_factory=dict)
    #: The published approach this configuration resembles. NOT a claim to reproduce it.
    resembles: str = ""
    #: Where it sits on the three axes above, so a reader can see what is being varied.
    edit: str = ""
    search: str = ""
    directions: str = ""


#: A flat, full-strength profile: the same ablation at every layer in the window. `bake_pc` reads
#: (position, max_weight, min_weight, distance) per component, so equal max and min is uniform.
FLAT = [0, 1.0, 1.0, 0]

METHODS = {
    "searched": Method(
        name="searched",
        summary="The default. Optimises how much to ablate, where, and across how many "
                "directions, against refusal, the keyword rate, drift and brokenness.",
        settings={},
        resembles="the Heretic family, which is what the comparative study benchmarked",
        edit="direction only; row norms are captured and restored",
        search="Optuna over many trials",
        directions="up to --max-directions, chosen by measured separation",
    ),
    "single-pass": Method(
        name="single-pass",
        summary="One direction, full strength, every layer in the window, no search at all. "
                "Minutes rather than an hour.",
        settings={"bake_profile": {"o_profile": FLAT, "d_profile": FLAT,
                                   "num_directions": 1, "dir_mode": "per_layer"}},
        resembles="DECCP and the other single-pass tools, which the same study ranks best on "
                  "capability retention",
        edit="direction only; identical surgery to the default",
        search="none; the strength is fixed rather than optimised",
        directions="one, the difference of means",
    ),
    "searched-one-direction": Method(
        name="searched-one-direction",
        summary="Our search, their directions: the strength and placement are optimised exactly "
                "as the default does it, but only the global difference of means is ever "
                "removed. No clustering, no hedging contrast.",
        settings={"args": {"max_directions": 1}},
        edit="direction only; identical surgery to the default",
        search="Optuna over many trials, identical to the default",
        directions="one, the difference of means",
    ),
    "single-pass-raw": Method(
        name="single-pass-raw",
        summary="A single pass WITHOUT restoring the original row norms. Present as a control, "
                "not as a recommendation.",
        settings={"bake_profile": {"o_profile": FLAT, "d_profile": FLAT,
                                   "num_directions": 1, "dir_mode": "per_layer"},
                  "no_good_orth": True},
        resembles="the naive formulation most tutorials describe",
        edit="length and direction both change",
        search="none",
        directions="one, the difference of means",
    ),
}

DEFAULT_METHOD = "searched"
CHOICES = sorted(METHODS)


def get(name):
    """The recipe by name, or a failure that lists what exists."""
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(
            f"unknown method {name!r}. Available: {', '.join(CHOICES)}.") from None


def describe(name):
    """The recipe in the words a reader of a result needs, rather than a settings dump."""
    m = get(name)
    lines = [f"method: {m.name}", f"  {m.summary}",
             f"  edit:       {m.edit}",
             f"  search:     {m.search}",
             f"  directions: {m.directions}"]
    if m.resembles:
        lines.append(f"  resembles:  {m.resembles}")
        lines.append("  This is this codebase configured that way, NOT a reimplementation of "
                     "that tool, and a result from it is a statement about senbonzakura.")
    return lines
