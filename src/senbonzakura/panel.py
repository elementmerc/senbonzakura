# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Several judges, and what it means when they disagree.

WHY THIS EXISTS

Every interval this project publishes comes from resampling PROMPTS. Judge choice sits silently
inside all of them as an uncontrolled term, and it is not a small one: the whole comparison
between this tool and another is decided by a single ruler that this project wrote. A tool whose
pitch is "here is how your number could be wrong" cannot ship a headline decided by one instrument
of its own making.

The v0.7 gate is specific, and read out of Opik's LLM Juries metric on 2026-08-06: each arm is
scored by at least three judges that differ in a way that matters, each judge's own verdict is
kept, and the reported interval carries the spread BETWEEN judges as well as the spread across
prompts. The three that differ in a way that matters here are the keyword ruler, the compass, and
a held-out instruct model prompted to classify: one reads surface strings, one reads a learned
direction, one reads meaning.

THE RULE THAT MAKES IT WORTH HAVING

**Where the judges disagree about which tool won, there is no winner.** Exactly as a gap inside
the seed spread is a tie. That is the whole point: a panel whose disagreements get averaged away
into a single number is a more expensive version of one judge, and it would hide the finding
rather than produce it.

Judge disagreement is a publishable result in its own right. It is the honest measurement of how
much any single-ruler abliteration comparison can be trusted, ours or anybody else's, and this
project is better placed to make it than most because it has already withdrawn results over
judges.

HOW THIS COMPOSES WITH `power`

A judge that reports a gap smaller than its own comparison can resolve has not found a winner, it
has found noise, and counting it as a vote would let an underpowered judge break a tie. So each
judge's verdict runs through `power.can_resolve` when a spread is available, and an unresolvable
gap is a TIE for that judge rather than a narrow win.
"""
from __future__ import annotations

from . import power

#: The verdicts a single judge, or the panel, can return.
A, B, TIE, NO_WINNER = "a", "b", "tie", "no-winner"

#: The gate's floor. Fewer than this is not a panel, and saying so is the difference between a
#: requirement and a suggestion.
MIN_JUDGES = 3

#: How many judges must actually REACH a verdict before the panel reports a winner.
#:
#: THE HOLE THIS CLOSES, found by two reviewers independently on 2026-09-21. A judge that returns
#: TIE drops out of the `decided` set, including a judge forced to TIE because the gap was below
#: what it could resolve. So one decisive judge and two ties satisfied "nobody disagreed" and the
#: panel reported that judge's verdict with `agreed: True`, over a report line that read "all 3
#: judges that reached a verdict agree".
#:
#: That is the single-ruler headline `MIN_JUDGES` exists to prevent, arrived at through the back
#: door: unanimity among one is not unanimity. Two is the floor rather than a majority of the
#: roster, because the claim being made is "more than one instrument saw this", which is the
#: weakest claim that is worth more than one judge's word.
MIN_DECIDING = 2


class PanelError(ValueError):
    """The panel cannot return a verdict, as opposed to returning an inconclusive one."""


def judge_verdict(name, score_a, score_b, *, higher_is_better, sd=None, n_per_arm=None):
    """One judge's reading: which arm it prefers, and whether it can see the difference at all.

    `sd` and `n_per_arm` are optional because not every judge produces a spread. When they are
    absent the verdict is reported with `resolvable: None`, meaning UNKNOWN rather than yes: a
    judge that cannot say whether it could see the gap has not established that it did.
    """
    gap = score_a - score_b
    better = A if (gap > 0) == bool(higher_is_better) else B
    out = {
        "judge": name,
        "score_a": score_a,
        "score_b": score_b,
        "gap": abs(gap),
        "higher_is_better": bool(higher_is_better),
        "verdict": TIE if gap == 0 else better,
        "resolvable": None,
    }
    if sd is not None and n_per_arm is not None:
        check = power.can_resolve(abs(gap), sd, n_per_arm)
        out["resolvable"] = check["resolvable"]
        out["detectable_gap"] = check["detectable_gap"]
        if not check["resolvable"]:
            # NOT a narrow win. A gap this judge cannot resolve is noise, and letting it vote
            # would allow an underpowered judge to break a tie between two that can see.
            out["verdict"] = TIE
    return out


def panel_verdict(judges):
    """The panel's answer, which is NO WINNER whenever its members disagree about one.

    Takes the output of `judge_verdict`. Returns the verdict, every judge's own reading kept
    intact, and the between-judge spread, because the spread is the measurement the panel exists
    to produce and folding it into a mean would discard the finding.
    """
    if len(judges) < MIN_JUDGES:
        raise PanelError(
            f"{len(judges)} judge(s) is not a panel; the gate asks for at least {MIN_JUDGES} that "
            f"differ in a way that matters. Scoring one arm with one ruler twice is not two "
            f"judges, and averaging two rulers is not three.")

    names = [j["judge"] for j in judges]
    if len(set(names)) != len(names):
        raise PanelError(
            f"two judges share a name: {sorted(names)}. A panel's whole value is that its members "
            f"are different instruments, and a duplicate is either a mistake or the same ruler "
            f"counted twice.")

    deciders = [j for j in judges if j["verdict"] in (A, B)]
    decided = {j["verdict"] for j in deciders}
    ties = [j["judge"] for j in judges if j["verdict"] == TIE]

    if len(decided) > 1:
        verdict, why = NO_WINNER, "disagree"
    elif not decided:
        verdict, why = TIE, None
    elif len(deciders) < MIN_DECIDING:
        # NOT a win. One judge saw a difference and the rest could not, which is a weaker claim
        # than one judge alone would make, because the others looked and failed to find it.
        verdict, why = NO_WINNER, "under-supported"
    else:
        verdict, why = decided.pop(), None

    # THE SPREAD IS OVER THE JUDGES THAT DECIDED, not over all of them. A tie-voter's gap is the
    # number this module has just ruled to be noise, and folding it into "the size of the win"
    # reports a range whose lower end the panel itself does not believe. With no deciders there
    # is no win to size, and the fields are None rather than a range over noise.
    gaps = [j["gap"] for j in deciders]
    unknown = [j["judge"] for j in judges if j.get("resolvable") is None]
    return {
        "verdict": verdict,
        "judges": list(judges),
        "n_judges": len(judges),
        # How many reached a verdict, as opposed to how many were asked. `report` said "all
        # {n_judges} judges that reached a verdict agree" using the roster size, which is a false
        # sentence on any panel where somebody tied.
        "n_decided": len(deciders),
        "agreed": verdict in (A, B),
        "no_winner_because": why,
        "tie_votes": ties,
        # The between-judge spread, reported rather than averaged away. A panel that agrees on a
        # winner and disagrees wildly on the size of the win has still found something.
        "gap_min": min(gaps) if gaps else None,
        "gap_max": max(gaps) if gaps else None,
        "gap_spread": (max(gaps) - min(gaps)) if gaps else None,
        # Judges whose own resolving power is unknown. Not counted as agreement: see `report`.
        "resolving_power_unknown": unknown,
    }


def report(p):
    """The panel's answer as lines a person reads, including what it did not establish."""
    lines = []
    v = p["verdict"]
    if v == NO_WINNER and p.get("no_winner_because") == "under-supported":
        decider = next(j["judge"] for j in p["judges"] if j["verdict"] in (A, B))
        lines.append(
            f"NO WINNER: only {p['n_decided']} of {p['n_judges']} judges reached a verdict "
            f"({decider}), and {MIN_DECIDING} are required. The others did not disagree; they "
            f"could not see a difference at all.")
        lines.append(
            "  One instrument finding a gap that the others looked for and did not find is "
            "weaker evidence than that instrument on its own, not stronger. Reporting it as a "
            "win is the single-ruler headline this panel exists to prevent.")
    elif v == NO_WINNER:
        split = {}
        for j in p["judges"]:
            split.setdefault(j["verdict"], []).append(j["judge"])
        lines.append(
            "NO WINNER: the judges disagree about which arm won, so there is no result to "
            "report. " + "; ".join(f"{k} -> {', '.join(sorted(names))}"
                                   for k, names in sorted(split.items())))
        lines.append(
            "  This is a finding rather than a failure. It measures how far a comparison of this "
            "kind depends on which ruler is used, which is the thing a single-judge headline "
            "cannot tell anybody, ours or anyone else's.")
    elif v == TIE:
        lines.append(
            f"TIE: no judge found a difference it could resolve. Judges that reported a tie: "
            f"{', '.join(sorted(p['tie_votes']))}.")
    else:
        lines.append(
            f"{v.upper()} wins: {p['n_decided']} of {p['n_judges']} judges reached a verdict and "
            f"they agree."
            + (f" {len(p['tie_votes'])} could not resolve a difference: "
               f"{', '.join(sorted(p['tie_votes']))}." if p["tie_votes"] else ""))
        lines.append(
            f"  The size of the win is NOT agreed: across the judges that decided, the gap runs "
            f"{p['gap_min']:.4g} to {p['gap_max']:.4g}, a spread of {p['gap_spread']:.4g}. Quote "
            f"the range rather than one judge's figure, and only where those judges share units.")
    if p["resolving_power_unknown"]:
        lines.append(
            f"  NOT CHECKED: {', '.join(sorted(p['resolving_power_unknown']))} reported no "
            f"spread, so whether that judge could resolve the gap it reported is unknown. "
            f"Unknown is not the same as yes.")
    return lines
