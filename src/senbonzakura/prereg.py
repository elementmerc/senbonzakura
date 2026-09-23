# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The pre-registration format: what was promised, in a shape a machine can check.

WHY THIS EXISTS

This project has written pre-registrations by hand since 2026-07-30, and they have already
earned their keep: one caught a confound before the run, and a second was amended mid-design
specifically so a late addition could not later be read as a result. That habit is the strongest
part of the method, and it lives in four Markdown files with four different heading structures,
which means it is a habit rather than a practice. Nobody outside this project could adopt it and
nothing verifies that a run did what its pre-registration said.

WHY MARKDOWN WITH A BLOCK, RATHER THAN A JSON OR YAML DOCUMENT

A pre-registration is an argument. The four written so far spend most of their length on why the
question is worth asking and on what would make the answer worthless, and that prose is the part
that does the work. Replacing it with fields would keep the parts a machine reads and throw away
the parts a reviewer reads, which is the wrong half to keep.

So the file stays Markdown and carries ONE fenced block, tagged `prereg`, holding the claims that
can be checked mechanically. The prose is the pre-registration; the block is the handle. This also
satisfies the rung's own test: *if the format cannot express what was already written by hand, the
format is wrong*. It can, because it does not ask the prose to change.

JSON inside the block rather than TOML or YAML, for the same reason the check registry chose it:
`tomllib` is 3.11+ and this project supports 3.10, YAML is a dependency, and JSON costs nothing
on any supported interpreter.

WHAT THE VALIDATOR REFUSES, AND WHY EACH RULE IS THERE

Every rule below is a defect this project actually shipped. None of them is schema hygiene.

  - **Exactly one primary.** Decision Q-12 locks primary-comparison discipline. A study with two
    primaries has two chances to succeed and reports whichever worked, and a study with none
    promotes a secondary after the fact. Both have happened in this field and one nearly happened
    here.
  - **Every comparison names its instrument and its partition.** On 2026-08-16 a run's 0.0%
    refusal rate travelled as a measured figure when it had been scored on the SELECTION
    partition, the rows the search had run 200 trials against.
  - **The hypothesis must state what would falsify it.** Three of the four hand-written
    pre-registrations have a heading close to *the hypothesis, stated so it can fail*, and the one
    that does not is the oldest. A hypothesis with no rejection condition cannot be wrong, so it
    cannot be evidence.
  - **Threats must be non-empty.** All four carry a section on what would make the measurement
    worthless. It is the most universal feature of the practice and the one most obviously worth
    keeping.
  - **Amendments carry a date and say whether any result existed yet.** This is the rule the
    format exists to make checkable. An amendment written before any number is ordinary practice;
    the same words written after are a post-hoc change wearing a pre-registration's clothes, and
    nothing but an honest flag distinguishes them. `lfm2-conv` records a third arm "added
    2026-08-15 before it could be read as a result", which is exactly the declaration this field
    makes mandatory instead of optional.

The validator returns FINDINGS rather than raising, because a pre-registration with a problem is
still a document somebody needs to read, and refusing to load it would hide the very fields that
say what is wrong with it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

#: The fenced block the machine reads. One per file: two would mean two documents disagreeing
#: about what was promised, which is the ambiguity this whole format exists to remove.
BLOCK = re.compile(r"^```prereg[ \t]*\r?\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)

#: Fields every pre-registration must carry, each with the reason it is mandatory.
REQUIRED = {
    "title": "what this pre-registration is about, in a line a reader can scan",
    "date": "when it was registered, ISO 8601. A pre-registration with no date cannot be shown "
            "to predate the result it claims to predate, which is its entire function",
    "question": "the question in one sentence. Every one of the four written by hand opens with "
                "exactly this and it is the discipline that stops a study drifting",
    "hypothesis": "what is expected, stated so it can fail",
    "falsified_by": "what result would REJECT the hypothesis. A claim with no rejection "
                    "condition cannot be evidence",
    "primary": "the ONE comparison this study is about (Q-12)",
    "threats": "what would make this measurement worthless, listed before it is taken",
}

#: Fields a comparison must carry. `partition` is here because of 2026-08-16.
COMPARISON_FIELDS = {
    "name": "what this comparison is called, so a result can be matched back to it",
    "metric": "what is measured",
    "instrument": "what measures it. Two numbers from two instruments are not a comparison",
    "partition": "which rows it is scored on. A rate with no partition is the 2026-08-16 defect",
    "decision_rule": "what counts as a pass, fixed now rather than after the number arrives",
}

#: Severities. REFUSED means the document cannot be treated as a pre-registration at all.
REFUSED, FINDING, NOTE = "refused", "finding", "note"


class PreregError(ValueError):
    """The file is not a pre-registration, as opposed to being a flawed one."""


def extract(text):
    """The machine-readable block from a pre-registration's Markdown, as a dict.

    Raises rather than returning findings, because "there is no block" and "the block says
    something questionable" are different kinds of answer and a caller wanting the second should
    not have to sift it out of the first.
    """
    found = BLOCK.findall(text)
    if not found:
        raise PreregError(
            "no ```prereg block in this file, so nothing here can be checked. A pre-registration "
            "is Markdown prose plus one fenced block tagged `prereg` holding the claims a machine "
            "reads. The prose is the pre-registration; the block is the handle on it.")
    if len(found) > 1:
        raise PreregError(
            f"{len(found)} ```prereg blocks in one file. Two blocks are two documents disagreeing "
            f"about what was promised, which is the ambiguity this format exists to remove.")
    try:
        loaded = json.loads(found[0])
    except ValueError as e:
        raise PreregError(f"the ```prereg block is not valid JSON: {e}") from e
    if not isinstance(loaded, dict):
        raise PreregError(
            f"the ```prereg block is a {type(loaded).__name__}, and it has to be an object.")
    return loaded


def load(path):
    """A pre-registration from disk. The prose stays where it is; this reads the block."""
    return extract(Path(path).read_text(encoding="utf-8"))


def _comparison_findings(where, comp):
    out = []
    if not isinstance(comp, dict):
        return [(REFUSED, f"{where} is a {type(comp).__name__}, and it has to be an object")]
    for field, why in COMPARISON_FIELDS.items():
        if not str(comp.get(field) or "").strip():
            out.append((FINDING, f"{where} has no `{field}`: {why}"))
    return out


def validate(doc):
    """Every problem with a pre-registration, as (severity, sentence) pairs. Empty means clean.

    Findings rather than exceptions, deliberately. A pre-registration with a problem is still a
    document somebody has to read, and refusing to load it would hide the very fields that say
    what is wrong with it.
    """
    findings = []

    for field, why in REQUIRED.items():
        value = doc.get(field)
        empty = not value if isinstance(value, list) else not str(value or "").strip()
        if empty:
            findings.append((REFUSED, f"no `{field}`: {why}"))

    if doc.get("date") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(doc["date"])):
        findings.append((FINDING, (
            f"`date` is {doc['date']!r}, which is not an ISO 8601 date. The whole point of the "
            f"field is that it can be ordered against the date a result exists.")))

    primary = doc.get("primary")
    if isinstance(primary, list):
        findings.append((REFUSED, (
            f"`primary` is a list of {len(primary)}. Exactly one primary, per decision Q-12: a "
            f"study with two primaries has two chances to succeed and reports whichever worked. "
            f"Everything else belongs under `secondaries`, where it is explicitly not the claim.")))
    elif primary is not None:
        findings += _comparison_findings("`primary`", primary)

    for i, sec in enumerate(doc.get("secondaries") or []):
        findings += [(s, m) for s, m in _comparison_findings(f"`secondaries[{i}]`", sec)
                     # A secondary missing a decision rule is a note rather than a finding: it is
                     # already declared as not the claim, so no verdict rests on it.
                     if not (m.endswith("arrives") and s == FINDING)]

    threats = doc.get("threats")
    if isinstance(threats, list) and threats and all(isinstance(t, str) for t in threats):
        if len(threats) == 1 and len(threats[0]) < 40:
            findings.append((NOTE, (
                "one short `threats` entry. Every pre-registration this project has written by "
                "hand carries several, and the section is the most useful part of the document "
                "to a sceptical reader.")))
    elif threats is not None and not isinstance(threats, list):
        findings.append((FINDING, "`threats` should be a list of sentences."))

    findings += _amendment_findings(doc)
    return findings


def _amendment_findings(doc):
    """THE RULE THE FORMAT EXISTS TO MAKE CHECKABLE.

    An amendment written before any number exists is ordinary practice and a sign of a careful
    study. The same words written after a number exists are a post-hoc change, and nothing but an
    honest declaration tells them apart. So the declaration is mandatory and its absence is a
    finding rather than a silence.
    """
    out = []
    amendments = doc.get("amendments") or []
    if not isinstance(amendments, list):
        return [(FINDING, "`amendments` should be a list of objects.")]
    for i, am in enumerate(amendments):
        where = f"`amendments[{i}]`"
        if not isinstance(am, dict):
            out.append((FINDING, f"{where} is a {type(am).__name__}, and it has to be an object"))
            continue
        if not str(am.get("date") or "").strip():
            out.append((FINDING, (
                f"{where} has no `date`, so it cannot be placed relative to the result it may or "
                f"may not have followed.")))
        if not str(am.get("what") or "").strip():
            out.append((FINDING, f"{where} does not say what changed"))
        declared = am.get("before_any_result")
        if declared is None:
            out.append((FINDING, (
                f"{where} does not declare `before_any_result`. An amendment made before any "
                f"number exists is ordinary practice; the same words after a number exists are a "
                f"post-hoc change wearing a pre-registration's clothes, and only this field "
                f"tells a reader which happened.")))
        elif declared is False:
            out.append((NOTE, (
                f"{where} was made AFTER a result existed, and says so. That is honest, and it "
                f"is not a pre-registration of that change: anything resting on it is "
                f"exploratory and has to be reported as exploratory.")))
    return out


def worst(findings):
    """The severity a caller should act on, or None when there is nothing to act on."""
    for level in (REFUSED, FINDING, NOTE):
        if any(s == level for s, _ in findings):
            return level
    return None


# ── checking a pre-registration against the run that claims to satisfy it ────────

#: How a run artefact's fields map onto what the pre-registration promised. Kept here rather than
#: inferred, because a mapping guessed from field names is how a comparison ends up describing
#: something other than what it claims.
RUN_FIELDS = {
    # A one-element path, and the first version wrote two. `separation_statistic` is a plain
    # string in the artefact, so digging for a nested `estimator` returned None and the
    # comparison reported the promise as unchecked. It failed loudly rather than passing, which
    # is the whole reason the absent case is a NOTE saying "not a pass" instead of a silence.
    "instrument": ("separation_statistic",),
    "partition": ("refusal_eval", "partition"),
    "tool_version": ("provenance", "senbonzakura", "version"),
}


def _dig(doc, path):
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def compare_to_run(doc, run):
    """Did the run do what the pre-registration said? Findings, same shape as `validate`.

    "We did what we said" becomes verifiable rather than asserted, which is the exit-gate line
    this function exists for. It reports what it could NOT check as loudly as what failed,
    because a comparison that silently skips a promise it could not find is the same defect as a
    gate that compares incomparable numbers: it returns clean and means nothing.
    """
    findings = []
    primary = doc.get("primary")
    if not isinstance(primary, dict):
        return [(REFUSED, (
            "this pre-registration has no usable `primary`, so there is nothing to hold the run "
            "against."))]

    for field, path in RUN_FIELDS.items():
        promised = primary.get(field) if field != "tool_version" else doc.get("tool_version")
        actual = _dig(run, path)
        if promised is None:
            continue
        if actual is None:
            findings.append((NOTE, (
                f"the pre-registration promised {field} {promised!r} and the run artefact "
                f"carries no {'.'.join(path)}, so this promise was NOT checked. Not a pass.")))
            continue
        if str(actual) != str(promised):
            findings.append((FINDING, f"{field}: promised {promised!r}, ran {actual!r}."))

    amended = [a for a in (doc.get("amendments") or [])
               if isinstance(a, dict) and a.get("before_any_result") is False]
    if amended:
        findings.append((NOTE, (
            f"{len(amended)} amendment(s) were made after a result existed. Whatever rests on "
            f"them is exploratory.")))
    return findings
