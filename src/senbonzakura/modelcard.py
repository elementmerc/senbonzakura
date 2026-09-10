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
import re
from pathlib import Path

from ._version import __version__

#: Sections a complete card carries. Missing ones are declared rather than dropped, because the
#: gap is information: a reader can tell "measured and fine" from "never looked at".
SECTIONS = ("what was done", "refusal", "capability", "corpus", "licence and use",
            "reproducing it")

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


def build(abl=None, cap=None, command=None, licence=None, licence_link=None):
    """The card, as markdown lines."""
    out = front_matter(abl, licence or "other", licence_link)
    out += ["# Abliteration report", ""]
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
    out += ["## Licence, and what this model is", "",
            *licence_section(abl, licence, licence_link), ""]
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


#: Identifiers the HuggingFace Hub actually renders a licence badge for. Not exhaustive of the
#: Hub's list, and deliberately permissive about the shape rather than the membership: what is
#: refused is a value the Hub will silently decline, such as `Apache 2.0` where it wants
#: `apache-2.0`. The flag's help text already names the right forms; this stops a near-miss being
#: written straight through into the YAML and publishing weights the Hub reports as unlicensed.
_LICENCE_ID = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")


def licence_complaint(value):
    """Why the Hub would not render this identifier, or None."""
    if not value or _LICENCE_ID.match(value):
        return None
    return (f"--base-licence {value!r} is not a shape the HuggingFace Hub accepts: it wants a "
            f"lowercase SPDX-style identifier such as `apache-2.0`, `mit`, `gemma` or "
            f"`llama3.2`, and renders no licence at all for anything else. Writing it through "
            f"would publish weights the Hub reports as unlicensed, which is the fault this card "
            f"exists to prevent.")


def front_matter(abl, licence, licence_link):
    """The YAML block HuggingFace reads to render licence and lineage metadata.

    WITHOUT THIS THE PAGE HAS NO LICENCE AT ALL, whatever the prose below says. HuggingFace renders
    the licence badge, the base-model lineage and the model's tags from this block and from nothing
    else, so a card that discusses the licence in three careful paragraphs and omits the block
    publishes weights that the Hub itself reports as unlicensed.

    `base_model` is the lineage. It is not decoration: it is how a reader of the derived weights
    finds the terms they are actually bound by, and it is the one fact this tool always knows,
    because the run recorded it.
    """
    out = ["---", f"license: {licence}"]
    # The Hub requires `license_name` alongside `license: other` and renders nothing without it,
    # so `--licence-unknown` produced a card whose prose said UNRESOLVED loudly and whose metadata
    # said nothing at all. A human reading the page was warned; the Hub was not.
    if licence == "other":
        out.append("license_name: unresolved-see-card")
    if licence_link:
        out.append(f"license_link: {licence_link}")
    if abl and abl.get("model"):
        out += ["base_model:", f"  - {abl['model']}"]
    out += ["tags:", "  - abliterated", "  - senbonzakura", "---", ""]
    return out


def licence_section(abl=None, licence=None, licence_link=None):
    """What travels with the weights, said on the page that travels with the weights.

    THE GAP THIS CLOSES. The repository says all of this carefully: that a base model's licence
    is not ours to loosen, that abliteration removes safety guardrails wholesale and that is
    both the point and the danger. Every word of it reaches a reader of the repository, and none
    of it reached the HuggingFace page of a checkpoint somebody abliterated with this tool,
    which is the only artefact a downstream user of those weights will ever see.

    WHAT THIS USED TO OMIT, and why it mattered. The paragraph below asserted that the base
    model's licence applies and then never said WHICH licence, never linked it, and reproduced not
    one term of it. A reader of the published weights was told they were bound by something
    unnamed. Two reviewers reached that independently, and the fix is not more prose: it is that
    the licence is now DATA the publisher supplies, refused rather than guessed, the same way
    every bundled corpus in `corpora.py` carries its licence and its attribution as fields.

    The tool does not infer it. A model's terms are not derivable from its weights, a wrong guess
    is worse than a blank, and this project does not publish figures it did not measure.
    """
    unresolved = [] if licence else [
        ("> **LICENCE UNRESOLVED.** This card was generated with `--licence-unknown`, so the "
         "licence governing these weights is not stated here and the metadata above says "
         "`other`. Publishing weights beside this card publishes them effectively unlicensed. "
         "Resolve it and regenerate with `--base-licence` before you do."),
        "",
    ]
    named = (f"**This model is under `{licence}`, its base model's licence, unchanged.**"
             if licence else
             "**This model is under its base model's licence, unchanged, and this card does not "
             "say which licence that is.**")
    link = f" The terms are at {licence_link}." if licence_link else ""
    base = f" The base model is `{abl['model']}`." if abl and abl.get("model") else ""
    return [
        *unresolved,
        (named + base + link + " Abliteration is an edit "
         "to existing weights. It does not create a new work with a new licence, and nothing "
         "this tool does can loosen the terms the base model came with. Whatever they permit "
         "and forbid, they still permit and forbid here. Check them before redistributing this."),
        "",
        ("**These weights have been modified from the base model.** Several licences require a "
         "derived work to say so prominently, Apache-2.0 among them (section 4(b), which asks "
         "that modified files carry notices stating that you changed them). Some go further: "
         "families exist whose terms require a naming prefix, an attribution string, or that the "
         "original terms and use policy are passed on to whoever you give the weights to. This "
         "tool does not know which of those apply to your base model and does not guess. Read the "
         "licence named above and satisfy it before you publish."),
        "",
        ("**Its refusal behaviour has been removed on purpose.** That is what the numbers above "
         "measure. It will answer requests the original declined, including harmful ones, and "
         "the capability section is the honest account of what that cost. Deploy it "
         "accordingly, and do not put it somewhere the original's refusals were load-bearing."),
        "",
        ("Card generated by [senbonzakura](https://github.com/elementmerc/senbonzakura) "
         f"{__version__}."),
    ]


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
    ap.add_argument("--base-licence", dest="base_licence", default="",
                    help="the base model's licence, as an SPDX identifier where one exists "
                         "(apache-2.0, mit, gemma, llama3.2, other). REQUIRED, because the card "
                         "states that this licence governs the weights and a card that says so "
                         "without naming it tells a reader they are bound by something unnamed. "
                         "It is not inferred: a model's terms are not derivable from its weights.")
    ap.add_argument("--base-licence-link", dest="base_licence_link", default="",
                    help="URL of the base model's licence text, rendered on the Hub beside it")
    ap.add_argument("--licence-unknown", dest="licence_unknown", action="store_true",
                    help="generate the card without a licence, marking it UNRESOLVED on the page "
                         "and in the metadata. For a card you are reading yourself; publishing "
                         "weights with it is publishing them unlicensed")
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
    if not a.base_licence and not a.licence_unknown:
        raise SystemExit(
            "--base-licence is required. This card states that the base model's licence governs "
            "the weights, and it used to say that without naming the licence, linking it, or "
            "reproducing a term of it, which tells a reader they are bound by something unnamed.\n"
            "  * Pass --base-licence with the base model's licence (apache-2.0, mit, gemma, "
            "llama3.2, other), and --base-licence-link where there is a URL for the text.\n"
            "  * Or pass --licence-unknown to generate a card marked UNRESOLVED, which is for "
            "reading rather than for publishing weights beside.\n"
            "It is not inferred from the model, deliberately: a model's terms are not derivable "
            "from its weights and a wrong guess is worse than a blank one.")
    bad = licence_complaint(a.base_licence)
    if bad:
        raise SystemExit(bad)
    lines = build(load(a.abliteration), load(a.capability), a.command or None,
                  licence=a.base_licence or None, licence_link=a.base_licence_link or None)
    text = "\n".join(lines)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(main)

