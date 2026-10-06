# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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

from . import argresolve, say
from ._version import __version__
from .measure import REDACTED, SECRET_FLAGS

#: Sections a complete card carries. Missing ones are declared rather than dropped, because the
#: gap is information: a reader can tell "measured and fine" from "never looked at".
SECTIONS = ("what was done", "what was measured", "refusal", "capability", "corpus",
            "what this does not cover", "licence and use", "reproducing it")

NOT_MEASURED = "**NOT MEASURED.** Nothing in the supplied artefacts covers this."

#: What a field says about itself when the artefact holding it says nothing. Distinct from
#: `NOT_MEASURED`, which is about a whole section: a figure can be measured perfectly well and
#: still arrive with no record of the conditions it was measured under, and those are different
#: problems for a reader deciding whether to trust it.
NOT_RECORDED = "**not recorded in this artefact**"

#: The fields `baseline.PINNED` requires before two measurements may be compared at all, each with
#: the label a reader of the published page meets instead of the field name.
#:
#: `stamps.py` says why they exist and what goes wrong without them. This is the only surface a
#: customer's risk function ever reads them on, and until 2026-10-06 not one of them reached it:
#: the card published a before and an after with no statement of the corpus, the rows, the prompt
#: format, the precision or the version of the tool, which is five of the five things that decide
#: whether those two numbers describe the same experiment.
PINNED_LABELS = (
    ("input_digest", "corpus the figures were taken on"),
    ("partition", "which rows of it"),
    ("prompt_format", "prompt format"),
    ("precision", "numerical precision"),
    ("tool_version", "tool version"),
)

#: Above this sample size the count behind a recorded rate stops being recoverable from the rate
#: and the denominator, because the artefacts record a rate to four decimal places and the rounding
#: error then exceeds half a reply. Nothing this project runs comes near it; the constant exists so
#: that a run which does gets a stated limit rather than a quietly wrong integer.
COUNT_RECOVERABLE_N = 10_000

#: Decimal places for a divergence figure. Four, because the measurement behind it rests on a few
#: hundred prompts and anything past the fourth decimal is noise being published as precision.
KL_PLACES = 4


#: MOVED TO `say.refusal_text`, 2026-09-27, rather than kept here. This function existed in this
#: module for about an hour before the same shape was needed for `--device` and for the entry
#: point's notices, which is exactly how two hand-kept copies of a fact begin.
_refusal = say.refusal_text


def _number(value, kind):
    """One figure, rendered for a reader of the published card rather than for a debugger.

    `rate` is a proportion in 0..1 and comes out as a percentage, because every other surface in
    this tool calls that quantity a proportion and prints a percentage. `kl` keeps its own units and
    is rounded. Anything that is not a number at all is passed through untouched, so a string
    already formatted upstream is not mangled here.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if kind == "rate":
        return f"{value * 100:.1f}%" if 0.0 <= float(value) <= 1.0 else value
    if kind == "kl":
        return f"{round(float(value), KL_PLACES):g}"
    return value


def load(path):
    """One artefact, or None when it was NOT SUPPLIED. A supplied one that cannot be read refuses.

    THE THREE CASES WERE ONE, and that is what a surface audit caught on 2026-09-27. This returned
    None for "no path given", "the path is not a file" and "the bytes are not JSON" alike, and the
    caller renders None as "NOT MEASURED". So `--abliteration notjson.txt` produced a complete,
    publishable model card, said the evidence was absent, and exited 0. The user concludes their
    run wrote an empty artefact; the truth is the card never read it.

    Not supplied is genuinely a gap and still returns None. Supplied and unreadable is the user
    naming a file they believe in, and the only honest answer is to stop.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise SystemExit(_refusal(
            "senbonzakura report: that artefact is not a file.",
            f"{path}",
            "A card reports what the artefacts say. Naming one that is not there and writing the "
            "card anyway would publish 'NOT MEASURED' for evidence that exists somewhere else."))
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise SystemExit(_refusal(
            "senbonzakura report: that artefact is not readable JSON.",
            f"{path}: {e}",
            "It was given as an artefact to report on, so the card is not written. Reporting "
            "'NOT MEASURED' here would say the run measured nothing, when what happened is that "
            "this file could not be read.")) from e


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


def _reading(abl, *keys):
    """The first of these names the artefact actually carries.

    THE DEFECT THIS EXISTS FOR, found by a hostile outside review on 2026-09-17. The abliterate
    run writes `baseline_refusals` and `post_bake_refusals`; this file read `baseline_refusal`
    and `post_bake_refusal`. The neighbouring `post_bake_kl` and `post_bake_broken` matched, so
    the card rendered KL and broken output and silently dropped the refusal delta: the two
    numbers the card exists to carry, in a tool whose tagline is "with receipts". The corpus
    section failed the same way, reading `corpus_sha256` against a written `track_digest`, and
    printed NOT MEASURED, which its own prose defines as "nothing in the supplied artefacts
    covers this". That was a false statement about an artefact that did cover it.

    Both spellings are accepted rather than one being renamed, because cards are generated from
    artefacts that already exist on disk and a rename would silently blank the older ones in
    exactly the way this is fixing.
    """
    for key in keys:
        value = abl.get(key)
        if value is not None:
            return value
    return None


def eval_provenance(abl):
    """What the artefact says its refusal figures were taken on, as a dict, possibly empty.

    `cli.Run.eval_provenance` writes four facts a reader needs and one sentence that can disqualify
    the whole figure, and this card read none of them for as long as both existed.
    """
    value = (abl or {}).get("refusal_eval")
    return value if isinstance(value, dict) else {}


def stamped_readings(doc, metric):
    """Every stamped reading of one metric in an artefact, in sorted slot order.

    Sorted rather than in insertion order, because a published page has to come out byte-identical
    on a second run over the same file; that is the reproducibility rule applied where this module
    serialises.

    `measurement.stamp` is what writes these, and what it records is exactly what a card of this
    kind needs: the estimator, the account of how the estimator works, the sample size, the
    interval and the method behind the interval. A `senbonzakura score` artefact carries all of it
    and this module ignored every field.
    """
    block = (doc or {}).get("metrics")
    if not isinstance(block, dict):
        return []
    return sorted((slot, b) for slot, b in block.items()
                  if isinstance(b, dict) and b.get("metric") == metric)


def _stamped_field(doc, field):
    """The first value any stamped metric in this artefact carries for one field, or None."""
    block = (doc or {}).get("metrics")
    if not isinstance(block, dict):
        return None
    for _slot, b in sorted(block.items()):
        if isinstance(b, dict) and b.get(field) is not None:
            return b[field]
    return None


def _format_name(value):
    """The prompt format as a comparable name, from either shape the artefacts write it in."""
    if isinstance(value, dict):
        return value.get("name") or value.get("source")
    return value


def refusal_n(abl=None, ref=None):
    """How many replies the refusal figures rest on, or None when nothing recorded it.

    A rate with no denominator is the defect `metrics.reportable_rate` exists to refuse, and this
    card published four of them on every page it has ever written. The count was never missing:
    the abliteration record carries it under `refusal_eval.rows_scored` and a scored artefact
    carries it as `n`. Neither reached the reader, so a rate over nine replies and a rate over
    three thousand came out looking the same, and that difference is the difference between
    evidence and an anecdote.
    """
    if ref and ref.get("n") is not None:
        return int(ref["n"])
    rows = eval_provenance(abl).get("rows_scored")
    return None if rows is None else int(rows)


def pinned_fields(abl=None, ref=None):
    """The five comparability fields, preferring the artefact that actually took the measurement.

    PER FIGURE RATHER THAN PER RUN, for `partition`, and that is the one subtlety here. An
    abliteration record stamps the separation statistic, whose partition is `search` because
    candidate directions are fitted on one half of the rows and scored on the other. That field is
    true, and it is true about a different number. The refusal figures' own partition is in
    `refusal_eval`, so taking the separation's and printing it under a refusal rate would be the
    shape this project keeps finding: a correct value under a label it does not belong to.

    The other four are properties of the run rather than of one figure, so a stamp anywhere in the
    artefact answers for them, with the record's own top-level spellings as the fallback for a
    record that stamps only the one statistic.
    """
    abl = abl or {}
    return {
        "input_digest": (_stamped_field(ref, "input_digest") or _stamped_field(abl, "input_digest")
                         or _reading(abl, "corpus_sha256", "track_digest")),
        "partition": (_stamped_field(ref, "partition") if ref
                      else eval_provenance(abl).get("partition")
                      or _stamped_field(abl, "partition")),
        "prompt_format": (_stamped_field(ref, "prompt_format")
                          or _stamped_field(abl, "prompt_format")
                          or _format_name((ref or {}).get("chat_template"))
                          or _format_name(abl.get("chat_template"))),
        "precision": _stamped_field(ref, "precision") or _stamped_field(abl, "precision"),
        "tool_version": (_stamped_field(ref, "tool_version") or _stamped_field(abl, "tool_version")
                         or abl.get("tool_version")),
    }


def _value_text(value):
    """One pinned field's value, rendered. A digest of several splits is several facts, not a dict.

    Same rule `corpus_section` already applies, for the same reason: a dict repr in a published
    artefact is a debugger's output that escaped, and a reader cannot copy one split out of it.
    """
    if isinstance(value, dict):
        return ", ".join(f"`{k}` `{v}`" for k, v in sorted(value.items()))
    return f"`{value}`"


def _count_from(value, n):
    """The count behind a recorded rate, or None when the sample is too large to recover it.

    RECOVERED RATHER THAN READ. The abliteration record writes the rate and the row count and
    never the count itself, so a card that wants to print `12/128` has to work the integer back
    out. Rates are recorded to four decimal places, so the error in `rate * n` stays under half a
    reply up to `COUNT_RECOVERABLE_N` and the integer comes back as it went in. Above that the
    honest answer is that the artefact should have written the count down, and this returns None so
    the caller states the rate without inventing a numerator.
    """
    if n is None or n <= 0 or n > COUNT_RECOVERABLE_N:
        return None
    return round(float(value) * n)


def _rate_reading(value, n, label):
    """One rate as a reader of a published page has to meet it: with its counts and its doubt.

    The bare percentage survives only where nothing recorded the denominator, and then it says so
    on its own line rather than below, because a reader who stops at the figure has to stop at the
    caveat too. That placement is the whole requirement: a rate the sample cannot carry, announced
    in a footnote, is a rate that gets quoted without the footnote.
    """
    count = _count_from(value, n)
    if count is None:
        return [(f"- {label}: **{_number(value, 'rate')}**, over a sample this artefact does not "
                 f"record, so no interval can be put on it")]
    return [_rate_line(count, n, label)]


def _stamped_reading(block, label):
    """One rate from a stamped metric block, with the interval the writer computed, not a new one.

    Recomputing it here would be a second implementation of the same interval, which is how two
    numbers under one name begin. The writer already ran `metrics.reportable_rate` over the actual
    counts and recorded what came out, including its refusal to state a rate at all, so this reads
    that decision rather than making it again.
    """
    value, n = block.get("value"), block.get("n")
    interval = block.get("interval")
    # `n is not None` as well as the flag, because the sentence names the sample: an artefact that
    # recorded the refusal and not the count would otherwise publish "None replies is below the
    # floor", and a card that cannot say how small the sample was says that instead.
    if block.get("reportable") is False and n is not None:
        return [(f"- {label}: **{n} replies is below the floor this project will state a rate "
                 f"over**, so the figure is not published as a rate here. The floor and the reason "
                 f"for it are in `metrics.reportable_rate`.")]
    if not (isinstance(interval, (list, tuple)) and len(interval) == 2):
        return _rate_reading(value, n, label)
    count = _count_from(value, n)
    counts = f"{count}/{n} = " if count is not None else ""
    return [(f"- {label}: **{counts}{_number(value, 'rate')}**, 95% CI "
             f"[{interval[0] * 100:.1f}%, {interval[1] * 100:.1f}%]")]


# THE ARTEFACT'S OWN REASONS FOR NOT TRUSTING ITS OWN FIGURE. Each is a field a writer fills in
# when it already knows the number is compromised, and the card dropped every one of them.
#
# THE WORST OF THE THREE IS THE LAST. `cli.Run.eval_provenance` writes a sentence saying the
# refusal figures are selection set figures, that the search chose its winner by scoring these same
# rows, and that they are therefore the best of however many trials rather than a measurement. A
# fixture in this repository is named for a run whose budget warning said its refusal rate described
# the budget rather than the model. Both sentences were sitting in the artefact while the card
# printed the rate they disqualify and said nothing at all.
def artefact_warnings(abl=None, ref=None):
    """Every self-withdrawal the supplied artefacts carry, each paired with whose figure it is about.

    Returned as text rather than rendered, so the caller can put them where the reader meets the
    figure instead of at the foot of the page. The source travels with the sentence because a page
    can carry two sets of refusal figures taken under different conditions, and a caveat that does
    not say which set it disqualifies is one a reader applies to the wrong one.
    """
    out, seen = [], set()
    for doc, whose in ((ref, "the scored figures"), (abl, "the search's own figures")):
        if not doc:
            continue
        settings = doc.get("generation_settings") or doc.get("reply_budget") or {}
        for warning in (doc.get("budget_warning"),
                        settings.get("budget_warning") if isinstance(settings, dict) else None,
                        doc.get("self_invalidated")):
            if warning and str(warning) not in seen:
                seen.add(str(warning))
                out.append((whose, str(warning)))
    note = eval_provenance(abl).get("note")
    # Only when the figures below ARE the record's own. A held out rate supplied through
    # `--refusal` is the number that note tells the reader to go and get, so repeating the note
    # beside it would warn about a problem that has been fixed.
    if note and not ref:
        out.append(("the search's own figures", str(note)))
    return out


def refusal_section(abl, ref=None):
    """The refusal figures, with the sample, the interval and the artefact's own caveats beside them.

    `ref` is a `senbonzakura score` artefact and it is the publishable figure. The abliteration
    record's rates come from the rows the search scored its own trials against, and the record says
    so in a sentence it writes itself; the rate a customer may quote is the one taken afterwards on
    the held out rows. Until this flag existed the card could not accept that artefact at all, so
    the only number it could ever publish was the one the tool's own output tells you not to
    publish.
    """
    if not abl and not ref:
        return [NOT_MEASURED]
    lines = []
    if ref:
        lines += _held_out_lines(abl, ref)
    abl = abl or {}
    # RENDERED FOR A READER, which is what a model card is for. A surface audit on 2026-09-27 found
    # this publishing `refusal before: **0.5625**` and `KL drift: **0.4076494872570038**`: a
    # proportion with no unit beside a number carrying sixteen significant figures, in an artefact
    # meant for strangers on the Hub.
    #
    # Sixteen figures is not precision, it is a float's repr. The measurement behind it comes from a
    # few hundred prompts, so the fourth decimal is already noise, and quoting all of them invites a
    # reader to compare two cards on digits that mean nothing. Rates become percentages because the
    # rest of this tool calls them proportions and prints them as percentages, and a card that
    # spells the same quantity differently is one more thing for a reader to reconcile.
    n = refusal_n(abl)
    own = []
    for keys, label, kind in (
            (("baseline_refusals", "baseline_refusal"), "refusal before", "rate"),
            (("post_bake_refusals", "post_bake_refusal"), "after", "rate"),
            (("post_bake_heretic",), "after, by Heretic's keyword metric", "rate"),
            (("post_bake_kl",), "KL drift", "kl"),
            (("post_bake_broken",), "broken output", "rate")):
        value = _reading(abl, *keys)
        if value is None:
            continue
        # The two rates whose denominator the record knows get counts and an interval; KL is not a
        # proportion and has no count behind it, so it keeps its own units.
        if kind == "rate":
            own += _rate_reading(value, n, label)
        else:
            own.append(f"- {label}: **{_number(value, kind)}**")
    if own and ref:
        # Kept rather than dropped, and labelled rather than mixed in. They are what the edit was
        # steered by, which is worth a reader's time, and they are not the figure to quote.
        lines += ["", ("The search's own figures, on the rows it chose its winner by scoring, kept "
                       "here because they are what the edit was steered by and not because they "
                       "are publishable:"), ""]
    lines += own
    if not lines:
        return [NOT_MEASURED]
    # WHERE THE READER MEETS THE RATE, which is the requirement and not a preference. Each of these
    # is the artefact saying its own figure should not be quoted, and each of them used to be
    # dropped on the floor by this module while the figure it disqualifies was published in bold.
    for whose, warning in artefact_warnings(abl, ref):
        lines += ["", f"> **Read {whose} with this.** {warning}"]
    lines.append("")
    lines.append("These are refusal and coherence rulers. None of them measures capability; see "
                 "that section.")
    return lines


def _held_out_lines(abl, ref):
    """The publishable refusal figures: the ones taken after the edit, on rows nothing was fitted on.

    Every rate comes out of the stamp the scoring command wrote, so the estimator, the sample and
    the interval on this page are the ones in the artefact rather than a second set computed here.
    """
    lines = []
    for _slot, block in stamped_readings(ref, "refusal_rate"):
        estimator = block.get("estimator", "unnamed estimator")
        lines += _stamped_reading(block, f"refusal after the edit, by `{estimator}`")
    # The unstamped rates the same artefact carries, which the stamp does not cover and which a
    # risk function asks about immediately: an answer that lectures instead of helping, and output
    # that has fallen apart.
    n = refusal_n(abl, ref)
    for key, label in (("soft_refusal", "replies that lectured rather than helped"),
                       ("broken", "replies that came out broken")):
        if ref.get(key) is not None:
            lines += _rate_reading(ref[key], n, label)
    if not lines:
        lines.append("**The supplied scored artefact carries no refusal figures.**")
    return lines


def measured_section(abl=None, cap=None, ref=None):
    """What the figures on this page are figures OF: the instrument, the sample, the conditions.

    THE GAP THIS CLOSES, and it is the first one a customer's risk function meets. The card used to
    print `refusal before: 39.1%` and `after: 1.2%` and nothing else. No sample size, so a rate
    over nine replies read the same as a rate over three thousand. No interval, so the page made no
    statement about its own precision. No estimator, in a project whose `measurement` module exists
    because two procedures under one metric name cost four published claims. And none of the five
    fields that decide whether two of these numbers may be compared at all, every one of which was
    already written in the artefact the card was reading.
    """
    if not (abl or cap or ref):
        return [NOT_MEASURED]
    lines = []
    for doc, metric in ((ref or abl, "refusal_rate"), (abl, "kl"), (cap, "capability")):
        for _slot, block in stamped_readings(doc, metric):
            lines.append(f"- {block.get('measures', metric)}, by `{block.get('estimator')}`: "
                         f"{block.get('estimator_description', NOT_RECORDED)}")
            if block.get("interval_method"):
                lines.append(f"  - the doubt beside it: {block['interval_method']}")
    if not lines:
        # NAMED RATHER THAN GUESSED. The card could assert that an abliteration run's refusal
        # figure came from this project's own ruler, because today it does, and that assertion
        # would be a second place the fact lives: the day a writer changes estimator the card goes
        # on saying the old one with complete confidence. So the name comes from the artefact's own
        # stamp, and an artefact with no stamp gets this sentence instead of a plausible label.
        lines.append("- **the supplied artefacts do not declare which estimator produced their "
                     "figures.** Run `senbonzakura score` and pass its output to `--refusal` for a "
                     "page that names the instrument, the sample and the interval.")
    # ONLY WHERE THERE ARE REFUSAL FIGURES TO DESCRIBE. Every label below says "the figures", and
    # on a card built from a capability artefact alone there are none, so printing six fields of
    # `not recorded` would describe a measurement nobody asked this card to report.
    if abl or ref:
        n = refusal_n(abl, ref)
        lines.append(f"- replies the refusal figures rest on: **{n}**" if n is not None else
                     f"- replies the refusal figures rest on: {NOT_RECORDED}")
        pinned = pinned_fields(abl, ref)
        for field, label in PINNED_LABELS:
            value = pinned.get(field)
            lines.append(f"- {label}: {_value_text(value) if value is not None else NOT_RECORDED}")
    settings = (ref or {}).get("generation_settings") or (abl or {}).get("generation_settings") \
        or (abl or {}).get("reply_budget") or {}
    if isinstance(settings, dict) and settings.get("max_new_tokens") is not None:
        lines.append(f"- tokens each reply was allowed: **{settings['max_new_tokens']}**"
                     + (", decoded greedily, so there is no sampling noise in these figures"
                        if settings.get("greedy") else ""))
    return [
        *lines,
        "",
        # THE REGISTER IS THE POINT, per the project's own explanation rule: a risk officer is
        # clever and is not a statistician, and a page that prints an interval without saying what
        # one is has handed them a decoration.
        ("**How to read the intervals.** A rate here is what happened on the replies that were "
         "scored. The interval beside it is the range the true rate could sit in if the same "
         "prompts were run again on a fresh sample of the same size. Narrow means the sample was "
         "big enough to pin the number down. Wide means it was not, however confident the headline "
         "figure looks: two models whose intervals overlap have not been shown to differ."),
        # Paired with the field it explains, so a card that does not carry that field does not
        # carry a paragraph about it either.
        *(["",
           ("**Why the rows matter.** `which rows of it` says whether the figures come from "
            "prompts the edit was chosen by. A search tries many configurations, scores each one, "
            "and keeps the best; score the winner on those same prompts afterwards and you "
            "measure the best of all those attempts rather than the model, and it flatters the "
            "result by an amount nobody can work out after the fact. A figure a customer may "
            "quote is one taken on rows that were held back.")] if (abl or ref) else []),
    ]


def limits_section(abl=None, cap=None, ref=None):
    """What this page does not cover, which is the section that makes the rest of it credible.

    NOT SOFTENED AND NOT BURIED, and the order is deliberate: the gaps that apply to THIS run come
    first, because they are the ones a reader can do something about, and the standing limits of
    the instruments follow. Nothing here is conditional on the figures looking good.

    Assembled from what the evidence actually resolved to rather than written as fixed prose. The
    same mistake has been made twice in this project by a static paragraph that did not know what
    the sections above it had found: a dual-use disclosure that inverted at a tag, and a licence
    paragraph that said "that is what the numbers above measure" on a card whose every figure read
    NOT MEASURED.
    """
    gaps = []
    if not cap:
        gaps.append("**What the edit cost.** No capability artefact was supplied, so nothing here "
                    "says whether the model still reasons, still does arithmetic, or still calls "
                    "a tool correctly. A low divergence does not answer that; none of these "
                    "instruments asks the model to reason.")
    held_out = eval_provenance(abl).get("held_out")
    if not ref and held_out is False:
        gaps.append("**Whether the refusal figures are a measurement.** They were taken on the "
                    "rows the search scored its own trials against, so they are the best of those "
                    "trials rather than a reading of the model. Score the held back rows with "
                    "`senbonzakura score` and pass the result to `senbonzakura report --refusal`.")
    elif not ref and held_out is None:
        gaps.append("**Whether the refusal figures were held back.** The artefact could not "
                    "establish it, which happens when the prompts came from a bare file with no "
                    "manifest, and a figure whose partition nobody can establish is not one to "
                    "publish.")
    n = refusal_n(abl, ref)
    if n is None:
        gaps.append("**How much evidence is behind the refusal figures.** The number of replies "
                    "they were computed over is not in the artefacts, so no interval can be put "
                    "on them and they cannot be compared with anybody else's.")
    else:
        from .metrics import MIN_REPORTABLE_N
        if n < MIN_REPORTABLE_N:
            gaps.append(f"**Enough replies to state a rate.** There were {n}, and this project "
                        f"will not state a rate over fewer than {MIN_REPORTABLE_N}: under that, a "
                        f"rate cannot support a claim in any framing, so the counts are given "
                        f"instead of a percentage.")
    gaps += [
        ("**Whether the model is safe to deploy.** This page says what one edit changed on one "
         "corpus. It is not a safety assessment, it is not an audit against any standard, and "
         "nothing in it is a statement about how the model behaves on traffic that does not look "
         "like the corpus named above."),
        ("**What a refusal is.** The refusal figures come from string matchers over the reply: "
         "lists of the phrases a refusal usually contains. They are fast, they are the same rule "
         "everywhere in this tool, and they are not a reader. A model that declines in words the "
         "list does not hold is scored as compliant, and a complete answer that mentions the law "
         "can be scored as a refusal. Measured on 64 replies from a model that delays its refusal "
         "on purpose, the matcher agreed with a validated judge about 88 times in 100."),
        ("**One corpus, one language, one reply per prompt.** The rates describe the prompts in "
         "the corpus named above, in English, with one deterministic reply each. A deployment that "
         "samples at a temperature will not reproduce them exactly, and no figure here covers "
         "another language or another set of requests."),
        ("**A conversation that continues.** Every figure here comes from a single request. A "
         "refusal that holds on the first ask and gives way on the third is invisible to all of "
         "them, and so is a refusal that a finetune brings back. `senbonzakura multiturn` and "
         "`senbonzakura tamper` measure those, and their artefacts are not read by this page."),
        ("**Whether the writing is any good.** Nothing here reads the prose. A model can hold "
         "every number on this page and still write badly."),
    ]
    return [f"- {g}" for g in gaps]


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
    for keys, label in ((("track",), "corpus"),
                        (("corpus_sha256", "track_digest"), "corpus digest"),
                        (("dir_prompts",), "prompts used to find directions")):
        value = _reading(abl, *keys)
        if not value:
            continue
        # A DIGEST IS THREE FACTS, NOT A PYTHON LITERAL. This published
        # `corpus digest: {'bad_ds': '39cc...', 'good_ds': '6b2b...', 'bad_eval_ds': 'c5db...'}`,
        # single quotes and all, into Markdown for the Hub. A dict repr in a published artefact is a
        # debugger's output that escaped, and a reader cannot copy one split out of it.
        if isinstance(value, dict):
            lines.append(f"- {label}:")
            for split, digest in sorted(value.items()):
                lines.append(f"  - `{split}`: `{digest}`")
        else:
            lines.append(f"- {label}: `{value}`")
    return lines or [NOT_MEASURED]


#: A flag's value as a command line carries it: bare, or quoted either way.
_FLAG_VALUE = r"""(?:"[^"]*"|'[^']*'|\S+)"""


def redact_command(command, secrets=()):
    """The command line with every secret taken out, for a page that is meant to be published.

    AT THE SINK RATHER THAN AT A CALLER. The card had two sources of a command, the run's own
    `sys.argv` and `senbonzakura report --command`, which is text somebody pasted, and the run's
    copy put `--hf-token <value>` into the README beside the weights from 0.4.0 to 0.4.1. Redacting
    here covers both, and any third that arrives later.

    Two rules covering each other, as `measure._shown` does for its own record: the value after a
    secret flag goes whatever it is, and a secret the caller knows goes wherever it sits on the
    line. The flag itself stays, because a reader reproducing the run needs to know one was used.
    """
    for flag in SECRET_FLAGS:
        command = re.sub(rf"({re.escape(flag)})(=|\s+){_FLAG_VALUE}", rf"\1\2{REDACTED}", command)
    for secret in secrets:
        # Skipping an empty one is not tidiness: `replace("", ...)` inserts between every
        # character of the line.
        if secret:
            command = command.replace(secret, REDACTED)
    return command


def build(abl=None, cap=None, command=None, licence=None, licence_link=None, secrets=(),
          ref=None):
    """The card, as markdown lines. `secrets` are values to keep off the page wherever they
    appear in `command`, on top of the flags `redact_command` always strips.

    `ref` is a `senbonzakura score` artefact holding the refusal figures taken after the edit on
    rows nothing was fitted on. It is optional because a card is better than no card, and it is
    the only input that makes the refusal section publishable.
    """
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
    out += ["## What was measured", "", *measured_section(abl, cap, ref), ""]
    out += ["## Refusal and coherence", "", *refusal_section(abl, ref), ""]
    out += ["## Capability, which is what the edit cost", "", *capability_section(cap), ""]
    out += ["## Corpus", "", *corpus_section(abl), ""]
    out += ["## What this does not cover", "", *limits_section(abl, cap, ref), ""]
    out += ["## Licence, and what this model is", "",
            # `measured` is what the evidence sections actually resolved to, not what was asked
            # for: a capability artefact that exists but says nothing still leaves the dual-use
            # paragraph unable to point at numbers.
            *licence_section(abl, licence, licence_link, measured=bool(cap)), ""]
    out += ["## Reproducing it", ""]
    if command:
        out += ["```sh", redact_command(command, secrets), "```", ""]
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

#: The identifiers the HuggingFace Hub actually renders. Not exhaustive of SPDX, deliberately: this
#: is the list the Hub's own licence picker offers, and an identifier outside it renders as no
#: licence at all.
#:
#: WHY MEMBERSHIP AND NOT SHAPE, 2026-09-27. This check used to be `^[a-z0-9][a-z0-9.-]*$`, which
#: asks whether a string LOOKS like an identifier. `--base-licence banana` looks exactly like one,
#: so a surface audit passed it straight into the YAML front matter, where `license: banana` is what
#: the Hub reads to decide what these weights may be used for. A shape check on a field whose whole
#: job is to be one of a known set answers a narrower question than the one being asked.
KNOWN_LICENCES = frozenset({
    "apache-2.0", "mit", "openrail", "bigscience-openrail-m", "creativeml-openrail-m",
    "bigscience-bloom-rail-1.0", "bigcode-openrail-m", "afl-3.0", "artistic-2.0", "bsl-1.0",
    "bsd", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear", "c-uda", "cc", "cc0-1.0",
    "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0",
    "cc-by-nc-2.0", "cc-by-nc-3.0", "cc-by-nc-4.0", "cc-by-nd-4.0", "cc-by-nc-nd-3.0",
    "cc-by-nc-nd-4.0", "cc-by-nc-sa-2.0", "cc-by-nc-sa-3.0", "cc-by-nc-sa-4.0", "cdla-sharing-1.0",
    "cdla-permissive-1.0", "cdla-permissive-2.0", "epl-1.0", "epl-2.0", "etalab-2.0", "eupl-1.1",
    "agpl-3.0", "gfdl", "gpl", "gpl-2.0", "gpl-3.0", "lgpl", "lgpl-2.1", "lgpl-3.0", "isc",
    "lppl-1.3c", "ms-pl", "mpl-2.0", "odc-by", "odbl", "openrail++", "osl-3.0", "postgresql",
    "ofl-1.1", "ncsa", "unlicense", "zlib", "pddl", "wtfpl", "ecl-2.0", "gemma", "llama2",
    "llama3", "llama3.1", "llama3.2", "llama3.3", "llama4", "deepfloyd-if-license",
    "intel-research", "apple-ascl", "apple-amlr", "fair-noncommercial-research-license",
    "other", "unknown",
})


def licence_complaint(value):
    """Why the Hub would not render this identifier, or None."""
    if not value or value in KNOWN_LICENCES:
        return None
    shaped = bool(_LICENCE_ID.match(value))
    near = sorted(k for k in KNOWN_LICENCES if k.startswith(value[:3].lower()))[:4]
    return _refusal(
        f"--base-licence {value!r} is not an identifier the HuggingFace Hub renders"
        + (", even though it is shaped like one." if shaped else "."),
        "That string goes into the card's YAML front matter as `license:`, which is what the Hub "
        "reads to decide what these weights may be used for, and it renders nothing at all for an "
        "identifier it does not know. Publishing that is weights the Hub reports as unlicensed, "
        "which is the fault this card exists to prevent.",
        *([f"Did you mean: {', '.join(near)}?"] if near else []),
        "Common ones here: apache-2.0, mit, gemma, llama3.2, agpl-3.0, cc-by-nc-4.0. Use `other` "
        "when the terms are bespoke, and `--licence-unknown` for a card marked UNRESOLVED rather "
        "than guessing.")


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


def licence_section(abl=None, licence=None, licence_link=None, *, measured=True):
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
        # CONDITIONAL, because the unmeasured version of this paragraph was a lie the card told
        # about itself. It said "That is what the numbers above measure" and "the capability section
        # is the honest account of what that cost" on cards whose capability section read NOT
        # MEASURED and whose refusal section carried no figures at all. Same failure as the
        # evaluation track card whose dual-use disclosure inverted at the tag: static prose that
        # does not know what the evidence sections resolved to.
        #
        # The warning is not weakened in the unmeasured case. It is strengthened: nobody knows what
        # this edit cost, which is a worse position to deploy from than knowing it was expensive.
        ("**Its refusal behaviour has been removed on purpose.** That is what the numbers above "
         "measure. It will answer requests the original declined, including harmful ones, and "
         "the capability section is the honest account of what that cost. Deploy it "
         "accordingly, and do not put it somewhere the original's refusals were load-bearing."
         if measured else
         "**Its refusal behaviour has been removed on purpose.** It will answer requests the "
         "original declined, including harmful ones. **What that cost has not been measured**, so "
         "this card cannot tell you what else changed: whether it still reasons as well, whether "
         "it still recognises harm it now complies with, or whether anything else went with the "
         "refusals. Deploy it accordingly, and do not put it somewhere the original's refusals "
         "were load-bearing."),
        "",
        ("Card generated by [senbonzakura](https://github.com/elementmerc/senbonzakura) "
         f"{__version__}."),
    ]


def build_parser():

    ap = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog="senbonzakura report",
        description="Assemble a run's artefacts into a model card that states only what they "
                    "support.")
    ap.add_argument("--abliteration", default="", help="an abliteration.json from a run")
    ap.add_argument("--refusal", default="",
                    help="a `senbonzakura score` artefact holding the refusal figures taken AFTER "
                         "the edit on rows the search never scored. This is the publishable "
                         "figure: an abliteration record's own rates come from the rows the "
                         "search chose its winner by scoring, and the record says so itself")
    ap.add_argument("--capability", default="",
                    help="a capability run's output, ideally one with --compare-to so the card "
                         "can state what the edit cost rather than only an absolute score")
    ap.add_argument("--base-licence", dest="base_licence", default="",
                    help="the base model's licence, as an SPDX identifier where one exists "
                         "(apache-2.0, mit, gemma, llama3.2, other). Required, and not inferred: a "
                         "model's terms are not derivable from its weights.")
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
    if not a.abliteration and not a.capability and not a.refusal:
        # SHAPE, NOT REASONING. This said "A card with no artefacts behind it would be a template,
        # and this exists to stop those being published", which explains the refusal to somebody
        # who has not asked why and never shows them what to type.
        raise SystemExit(
            "nothing to report on: a card needs at least one artefact behind it.\n"
            "  --abliteration FILE   an abliteration.json from a run\n"
            "  --refusal FILE        a `senbonzakura score` run's output, scored on held back rows\n"
            "  --capability FILE     a capability run's output\n"
            "  --base-licence NAME   the base model's licence, which is not inferred\n"
            "\n"
            "  senbonzakura report --abliteration edited/abliteration.json \\\n"
            "      --base-licence apache-2.0 --out edited/README.md")
    if not a.base_licence and not a.licence_unknown:
        raise SystemExit(
            "--base-licence is required. The card states that the base model's licence\n"
            "governs these weights, so it has to name which licence that is.\n"
            "  * Pass --base-licence apache-2.0 (or mit, gemma, llama3.2, other), and\n"
            "    --base-licence-link where there is a URL for the text.\n"
            "  * Or pass --licence-unknown for a card marked UNRESOLVED, which is for\n"
            "    reading rather than for publishing weights beside.\n"
            "It is not inferred: a model's terms are not derivable from its weights.")
    bad = licence_complaint(a.base_licence)
    if bad:
        raise SystemExit(bad)
    lines = build(load(a.abliteration), load(a.capability), a.command or None,
                  licence=a.base_licence or None, licence_link=a.base_licence_link or None,
                  ref=load(a.refusal))
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

