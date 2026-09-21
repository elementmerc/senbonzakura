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

    decided = {j["verdict"] for j in judges if j["verdict"] in (A, B)}
    ties = [j["judge"] for j in judges if j["verdict"] == TIE]

    if len(decided) > 1:
        verdict = NO_WINNER
    elif not decided:
        verdict = TIE
    else:
        verdict = decided.pop()

    gaps = [j["gap"] for j in judges]
    unknown = [j["judge"] for j in judges if j.get("resolvable") is None]
    return {
        "verdict": verdict,
        "judges": list(judges),
        "n_judges": len(judges),
        "agreed": verdict in (A, B),
        "tie_votes": ties,
        # The between-judge spread, reported rather than averaged away. A panel that agrees on a
        # winner and disagrees wildly on the size of the win has still found something.
        "gap_min": min(gaps),
        "gap_max": max(gaps),
        "gap_spread": max(gaps) - min(gaps),
        # Judges whose own resolving power is unknown. Not counted as agreement: see `report`.
        "resolving_power_unknown": unknown,
    }


def report(p):
    """The panel's answer as lines a person reads, including what it did not establish."""
    lines = []
    v = p["verdict"]
    if v == NO_WINNER:
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
            f"{v.upper()} wins, and all {p['n_judges']} judges that reached a verdict agree.")
        lines.append(
            f"  The size of the win is NOT agreed: the gap runs {p['gap_min']:.4g} to "
            f"{p['gap_max']:.4g} across judges, a spread of {p['gap_spread']:.4g}. Quote the "
            f"range rather than one judge's figure.")
    if p["resolving_power_unknown"]:
        lines.append(
            f"  NOT CHECKED: {', '.join(sorted(p['resolving_power_unknown']))} reported no "
            f"spread, so whether that judge could resolve the gap it reported is unknown. "
            f"Unknown is not the same as yes.")
    return lines
