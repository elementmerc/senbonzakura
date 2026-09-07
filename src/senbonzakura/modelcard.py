# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Turn a run's artefacts into the card that should go beside the weights.

WHY THIS EXISTS

An abliterated model gets published with a sentence like "minimal degradation" and nothing behind
it. The field ships on a claimed one to three percent with no benchmarks, and a comparative study
puts one family at eight points on grade-school arithmetic. Neither the claim nor the counter-claim
comes with an interval, and nobody can check either.

Everything needed to do better is already written down by the time a run finishes. `abliteration.json`
records the method, the corpus, the statistic and the separation numbers; a `capability` run records
what the edit cost with a paired interval. This assembles them into one page, and refuses to state
anything the artefacts do not support.

THE RULES IT ENFORCES, WHICH ARE THE POINT

A number that is not in an artefact does not appear. A rate the sample cannot carry is printed as
counts. An interval that spans zero is written as "not distinguishable from no change" rather than
as a delta with a caveat somewhere below it. And a section with no evidence says NOT MEASURED, in
those words, rather than being quietly omitted: an absent section reads as nothing to report, and
"nobody measured this" is a different statement that a reader deserves.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Sections a complete card carries. Missing ones are declared rather than dropped, because the
#: gap is information: a reader can tell "measured and fine" from "never looked at".
SECTIONS = ("what was done", "refusal", "capability", "corpus", "reproducing it")

NOT_MEASURED = "**NOT MEASURED.** Nothing in the supplied artefacts covers this."


def load(path):
    """One artefact, or None when it was not supplied. A missing file is a gap, not an error."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _rate_line(count, n, label):
    """A rate with its counts, or the counts alone when the sample cannot carry a rate."""
    from .metrics import reportable_rate

    r = reportable_rate(count, n)
    lo, hi = r["ci"] or (None, None)
    if r["rate"] is None:
        return (f"- {label}: **{count}/{n}**, not stated as a rate ({r['why_not']})")
    return (f"- {label}: **{count}/{n} = {r['rate']:.1%}**, 95% CI "
            f"[{lo:.1%}, {hi:.1%}]")


def capability_section(cap):
    """What the edit cost, or a plain statement that nobody looked."""
    if not cap:
        return [
            NOT_MEASURED,
            "",
            ("Refusal rates and drift cannot see reasoning loss. A model can hold a low KL with "
             "nothing broken and still have lost multi-step arithmetic, because neither of those "
             "asks it to reason. Until this is filled in, no claim about capability is supported "
             "by anything here."),
        ]
    s = cap.get("summary", {})
    lines = [(f"Task: `{cap.get('task', 'unknown')}` on `{cap.get('eval', 'unknown')}`, "
              f"{s.get('n', 0)} items.")]
    lines.append(_rate_line(s.get("correct", 0), s.get("graded", 0), "accuracy"))
    if s.get("indeterminate"):
        lines.append(
            f"- **{s['indeterminate']} of {s['n']} answers could not be graded.** Usually the "
            f"token budget rather than the model. Read the accuracy above with that in mind.")
    change = cap.get("change")
    if not change:
        lines += [
            "",
            ("No paired comparison against a reference model was supplied, so the figure above "
             "is an absolute score and not a statement about what the edit cost."),
        ]
        return lines
    lo, hi = change["delta_ci"]
    lines += ["", f"Compared against the reference on {change['compared_on']} shared items:"]
    if change["distinguishable_from_zero"]:
        lines.append(f"- change in accuracy: **{change['delta_accuracy']:+.1%}**, "
                     f"95% CI [{lo:+.1%}, {hi:+.1%}]")
    else:
        lines.append(f"- change in accuracy: **not distinguishable from no change**. The point "
                     f"estimate is {change['delta_accuracy']:+.1%} and the interval "
                     f"[{lo:+.1%}, {hi:+.1%}] spans zero.")
    lines.append(f"- {change['items_broken']} items it used to get right and now does not; "
                 f"{change['items_fixed']} the other way.")
    if change["items_fixed"] and change["items_broken"]:
        lines.append("  Items moved in both directions, so this is a model that answers "
                     "differently rather than one that is uniformly worse.")
    return lines


def refusal_section(abl):
    if not abl:
        return [NOT_MEASURED]
    lines = []
    for key, label in (("baseline_refusal", "refusal before"), ("post_bake_refusal", "after"),
                       ("post_bake_kl", "KL drift"), ("post_bake_broken", "broken output")):
        if abl.get(key) is not None:
            lines.append(f"- {label}: **{abl[key]}**")
    if not lines:
        return [NOT_MEASURED]
    lines.append("")
    lines.append("These are refusal and coherence rulers. None of them measures capability; see "
                 "that section.")
    return lines


def method_section(abl):
    if not abl:
        return [NOT_MEASURED]
    lines = [f"- method: **{abl.get('method', 'searched')}**"]
    if abl.get("separation_statistic"):
        lines.append(f"- separation statistic: `{abl['separation_statistic']}`")
    lines.append(f"- matched scoring: {'yes' if abl.get('matched_scoring') else 'no'}")
    if abl.get("matched_source"):
        lines.append(f"- matched control pool: `{abl['matched_source']}`")
    if abl.get("matching_quality") is not None:
        lines.append(f"- matching quality alarm: {abl['matching_quality']} "
                     f"(near 1.0 means the matching achieved nothing)")
    if abl.get("match_closeness") is not None:
        lines.append(f"- match closeness: {abl['match_closeness']} "
                     f"(about 1.0 is as near as the space allows)")
    return lines


def corpus_section(abl):
    if not abl:
        return [NOT_MEASURED]
    lines = []
    for key, label in (("track", "corpus"), ("corpus_sha256", "corpus digest"),
                       ("dir_prompts", "prompts used to find directions")):
        if abl.get(key):
            lines.append(f"- {label}: `{abl[key]}`")
    return lines or [NOT_MEASURED]


def build(abl=None, cap=None, command=None):
    """The card, as markdown lines."""
    out = ["# Abliteration report", ""]
    if abl and abl.get("model"):
        out += [f"Base model: `{abl['model']}`", ""]
    out += [
        ("Every number here comes from an artefact this run produced. A section with no evidence "
         "says so rather than being left out, because an absent section reads as nothing to "
         "report and that is a different claim."),
        "",
    ]

    out += ["## What was done", "", *method_section(abl), ""]
    out += ["## Refusal and coherence", "", *refusal_section(abl), ""]
    out += ["## Capability, which is what the edit cost", "", *capability_section(cap), ""]
    out += ["## Corpus", "", *corpus_section(abl), ""]
    out += ["## Reproducing it", ""]
    if command:
        out += ["```sh", command, "```", ""]
    else:
        out += [
            ("**NOT RECORDED.** No command was supplied, so this run cannot be reproduced from "
             "this page."),
            "",
        ]
    return out


def build_parser():
    import argparse

    ap = argparse.ArgumentParser(
        prog="senbonzakura report",
        description="Assemble a run's artefacts into a model card that states only what they "
                    "support.")
    ap.add_argument("--abliteration", default="", help="an abliteration.json from a run")
    ap.add_argument("--capability", default="",
                    help="a capability run's output, ideally one with --compare-to so the card "
                         "can state what the edit cost rather than only an absolute score")
    ap.add_argument("--command", default="", help="the command that produced the model")
    ap.add_argument("--out", default="", help="write here instead of standard output")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if not a.abliteration and not a.capability:
        raise SystemExit(
            "nothing to report on: pass --abliteration, --capability, or both. A card with no "
            "artefacts behind it would be a template, and this exists to stop those being "
            "published.")
    lines = build(load(a.abliteration), load(a.capability), a.command or None)
    text = "\n".join(lines)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0
