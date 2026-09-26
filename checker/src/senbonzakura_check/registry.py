# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The check registry: checks are data, and this module is the engine that reads them.

THE RULE THIS FILE EXISTS TO ENFORCE

**The engine never hardcodes a check.** Every check is a file in `checks/`, and adding one means
adding a file, not editing code. That is the Nuclei pattern, and `strategy-2026-08-02.md` D4a is
blunt about why it matters: Nuclei took its domain in eighteen months *because of* the
contribution format, not despite deferring it. A domain nobody else can extend is a tool, not a
domain.

WHY JSON AND NOT TOML OR YAML

TOML reads better for a human writing one, and it was the first choice. It costs a dependency:
`tomllib` is 3.11+, this project supports 3.10, and the checker's entire pitch is that it
installs in seconds with nothing. YAML costs a bigger one. JSON costs nothing on any supported
Python, and a contributor writing a check is already reading JSON, because JSON is what the
artefacts being checked are written in. Revisit if contributors find it painful; the loader is
the only thing that would change.

WHAT EVERY CHECK MUST CARRY, AND WHY EACH FIELD IS MANDATORY

    id            stable, kebab-case. A report line has to be traceable to the file that produced
                  it, and a renamed id silently orphans every reference to it.
    title         one line, what fired.
    detects       what the check looks for, in plain language.
    incident      THE CITATION. What went wrong in the real world to motivate this. A finding
                  with no citation is an opinion, and a checker that reports opinions gets
                  uninstalled once and never again.
    remedy        what to do about it.
    confidence    high / medium / low: how much to believe the finding.
    severity      withdraws / qualifies / notes: how far it could move the number if you do.
                  OPTIONAL with a default, unlike everything else here, because making it
                  mandatory would refuse every check file written before it existed, and a
                  contribution format that invalidates existing contributions is not one.
    false_positive  WHAT WOULD MAKE THIS CHECK WRONG. Loophole 5 of the v0.8 plan, and the
                  requirement is absolute: "a check that cannot say what would make it wrong
                  does not ship". It goes in the REPORT, not in the documentation, because the
                  person deciding whether to believe a finding is reading the report.
    applies_to    a rule. When it is false the artefact is not the kind this check knows about,
                  and the check is SKIPPED rather than passed. Those are different outcomes: a
                  silent pass on a document we could not understand is the worst possible output,
                  because it is indistinguishable from a clean bill of health.
    rule          a rule. When it is true, the check FIRES.
    control       `fires_on` and `passes_on`, each a list of documents. Every check ships proof
                  that it can fail. Read out of soup on 2026-09-02: a check nobody has watched
                  fail is a check nobody has tested. `tests/` runs all of them.

    arity        `document` (the default) or `pair`. OPTIONAL with a default, for the same
                  reason severity is: a field that invalidates every check file written before
                  it existed is a breaking change to a contribution format.

A CHECK THAT NEEDS TWO DOCUMENTS, AND WHY THE ENGINE GREW AN ARITY RATHER THAN A SECOND ENGINE

Two defects this project has actually shipped cannot be seen in one artefact. An arm that differs
from its counterpart in more than the variable the comparison names is a fact about two arms; a
figure compared across a change of instrument is a fact about two figures. Both were skipped when
the checks were first written, purely because `run_checks` read one document.

The design is three decisions, and each one is a refusal to do the convenient thing.

**A pair is a document.** `run_pair_checks` builds `{"arm_a": ..., "arm_b": ...}` and evaluates
the ordinary rule language against it, so a pair check is written with the same operators, the
same dotted paths and the same controls as any other. The alternative, a second rule vocabulary
that takes two documents, would mean every operator existing twice and drifting apart, which is
the failure this codebase has already had with two hand-maintained architecture lists.

**A pair check is SKIPPED on a single document, never passed.** `run_checks` puts every
`arity: pair` check into `skipped` without evaluating it. Skipped and passed are the central
distinction in this whole design: a pair check reported as passing on one file would be claiming
a comparison was sound when no comparison was examined.

**The caller names the pair; nothing infers it.** `senbonzakura check --pair a.json b.json` takes
exactly two result files and refuses anything else. Sweeping a directory and pairing everything in
it was the other candidate and it manufactures comparisons nobody ran: `head-to-head/results/`
holds thirty arms across several models and tools, and 435 of those pairs are not comparisons at
all. Which two artefacts are arms of one experiment is a claim only the person who ran them can
make, and a checker that guesses it reports findings about experiments that never existed.

THE RULE LANGUAGE IS DELIBERATELY SMALL

It is a handful of operators over dotted paths, not an expression language. Two reasons. A rule
that can do anything is code in a JSON costume, and it would have to be reviewed like code by
whoever runs the checker on their own artefacts. And the checks that are actually wanted turn
out to need very little: is this field missing, does this field disagree with that one, do these
two row sets overlap. Extend it when a real check cannot be written, not in anticipation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKS_DIR = Path(__file__).resolve().parent / "checks"

#: The largest artefact this tool will read, in bytes.
#:
#: SECURITY.md puts crafted result files in scope in so many words: "This tool loads other people's
#: files, so this is the category that matters most." Both ingestion points read an entire
#: caller-supplied path into memory, and REPRODUCING.md invites exactly that, telling people to
#: point the checker at somebody else's lm-evaluation-harness and Inspect output.
#:
#: 64 MB is generous rather than tight. These artefacts are aggregate JSON: the largest this
#: project publishes is under 100 KB, and a harness result file carrying per-sample records runs to
#: a few megabytes. A file past this is not a result file that got big, it is a different kind of
#: object, and refusing it names the limit rather than letting the machine decide by running out of
#: memory.
MAX_ARTEFACT_BYTES = 64 * 1024 * 1024

#: How deep a JSON document may nest before this refuses it.
#:
#: The separate hazard, and the one a size cap does not cover: `[[[[...]]]]` is small on disk and
#: raises RecursionError in the parser. Neither ingestion point caught RecursionError, which is not
#: an OSError and not a JSONDecodeError, so a few kilobytes produced a traceback rather than a
#: refusal. Python's own limit is around 1000 frames and the parser burns several per level; 200 is
#: far past any real artefact and far short of the interpreter's ceiling.
MAX_ARTEFACT_DEPTH = 200


class ArtefactTooLargeError(Exception):
    """A file this tool declines to read, with the limit named in the message."""


def too_deep(obj, limit=None):
    """Does a parsed document nest past `limit`? PUBLIC, because two packages need this answer.

    `senbonzakura.probe` reads probe manifests and item files that a stranger wrote, which is the
    same threat this module's bound was added for, and it had grown a near-copy of `_depth` rather
    than reach into another distribution's private name. Copying a limit is the drift that matters
    and it had already imported this module's constant; copying the walker is how two guards start
    disagreeing about what "too deep" means. So the walker is part of the interface now.
    """
    return _depth(obj, MAX_ARTEFACT_DEPTH if limit is None else limit) > (
        MAX_ARTEFACT_DEPTH if limit is None else limit)


def _depth(obj, limit, _at=0):
    """Deepest nesting in a parsed document, giving up as soon as it passes `limit`.

    Iterative in the sense that matters: it stops at the limit rather than walking a document that
    is already known to be too deep, so the guard cannot itself be the thing that runs out of
    stack.
    """
    if _at > limit:
        return _at
    if isinstance(obj, dict):
        return max((_depth(v, limit, _at + 1) for v in obj.values()), default=_at)
    if isinstance(obj, list):
        return max((_depth(v, limit, _at + 1) for v in obj), default=_at)
    return _at


def read_json_bounded(path, *, max_bytes=MAX_ARTEFACT_BYTES, max_depth=MAX_ARTEFACT_DEPTH):
    """Parse a JSON artefact, bounded in size and in nesting depth.

    Raises OSError, json.JSONDecodeError or ArtefactTooLargeError. Callers already distinguish the first
    two and turn them into their own refusal sentences, so this adds one more of the same shape
    rather than a new failure mode to handle.

    The size is checked with `stat` BEFORE the read, so an over-large file is never held in memory
    even briefly. The depth can only be checked after parsing, which is why the parse itself is
    guarded for RecursionError.
    """
    p = Path(path)
    size = p.stat().st_size
    if size > max_bytes:
        raise ArtefactTooLargeError(
            f"it is {size:,} bytes and this reads at most {max_bytes:,}. Result artefacts are "
            f"aggregate JSON and are thousands of times smaller than this; a file this large is "
            f"something else.")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except RecursionError:
        raise ArtefactTooLargeError(
            f"its JSON nests deeper than the parser will go. This reads at most {max_depth} "
            f"levels.") from None
    if _depth(doc, max_depth) > max_depth:
        raise ArtefactTooLargeError(
            f"its JSON nests deeper than {max_depth} levels. A result artefact is a few levels "
            f"deep; this is not one.")
    return doc

#: Every field a check file must carry. Absence of any one is a refusal, not a default.
REQUIRED_FIELDS = (
    "id", "title", "detects", "incident", "remedy",
    "confidence", "false_positive", "applies_to", "rule", "control",
)

CONFIDENCES = ("high", "medium", "low")

#: How far a finding could move the number, worst first. NOT the same axis as `confidence`, and
#: keeping them apart is the whole reason this exists: confidence is how much to believe the
#: finding, severity is what it costs if you do. A high-confidence cosmetic note and a
#: medium-confidence withdrawal are not the same news, and a report that sorts them together
#: invites a reader to treat them as the same weight.
#:
#:     withdraws   the figure cannot be quoted as what it claims to be at all
#:     qualifies   the figure stands with a caveat attached, and not without one
#:     notes       two numbers in the artefact could be confused for each other
#:
#: OPTIONAL, WITH A DOCUMENTED DEFAULT, and that is a deliberate compromise rather than laziness.
#: Making it mandatory would refuse every check file written before today, including anybody
#: else's: the engine's own promise is that a check is a file, and a field that invalidates
#: existing files is a breaking change to a contribution format that is trying to attract
#: contributions. `tests/test_check_severity.py` asserts that every check WE ship declares one,
#: which is where the discipline actually lives.
SEVERITIES = ("withdraws", "qualifies", "notes")
DEFAULT_SEVERITY = "qualifies"

#: How many documents a check needs to answer its question.
#:
#: OPTIONAL WITH A DEFAULT, for the same reason `severity` is: making it mandatory would refuse
#: every check file written before it existed, including anybody else's, and a contribution
#: format that invalidates existing contributions is not one.
ARITIES = ("document", "pair")
DEFAULT_ARITY = "document"

#: The keys a pair is assembled under. Named here rather than spelled at each call site, because
#: a check file addresses them by these exact strings and a second spelling of one of them is how
#: a check starts reading a field nobody writes.
PAIR_KEYS = ("arm_a", "arm_b")

#: A sentinel distinct from None, because a JSON document may legitimately hold a null and
#: "the field is absent" and "the field is present and null" are different claims about it.
MISSING = object()


class CheckError(Exception):
    """A check file is malformed, or a rule cannot be evaluated.

    Raised loudly rather than skipped. A registry that silently drops a check it could not read
    reports a clean run over a set of checks nobody can enumerate.
    """


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    detects: str
    incident: str
    remedy: str
    confidence: str
    false_positive: str
    applies_to: dict
    rule: dict
    control: dict
    severity: str = DEFAULT_SEVERITY
    arity: str = DEFAULT_ARITY
    source: Path | None = None


@dataclass(frozen=True)
class Finding:
    """One check firing on one artefact.

    Carries the check's own account of what would make it wrong, so the report can print it
    beside the finding rather than expecting the reader to go and look it up.
    """

    check_id: str
    title: str
    detects: str
    incident: str
    remedy: str
    confidence: str
    false_positive: str
    severity: str = DEFAULT_SEVERITY
    artefact: str | None = None


def dotted(doc: Any, path: str) -> Any:
    """The value at a dotted path, or MISSING.

    List indices are written as integers: `results.0.name`. Anything that is not a mapping or a
    sequence at a step means the path does not exist here, which is MISSING rather than an error:
    the artefact is somebody else's and being shaped differently from what a check expected is
    the normal case, not a fault.
    """
    cur = doc
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return MISSING
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return MISSING
        else:
            return MISSING
    return cur


def _as_set(value):
    if isinstance(value, (list, tuple, set)):
        return {v if isinstance(v, (str, int, float, bool)) else json.dumps(v, sort_keys=True)
                for v in value}
    return set()


#: The per-entry predicates `any_entry` understands. Total, like every rule operator: each
#: answers true or false for any value, and none raises on a shape it did not expect.
_ENTRY_PREDICATES = {
    "present": lambda v: v is not MISSING,
    "absent": lambda v: v is MISSING,
    "truthy": lambda v: v is not MISSING and bool(v),
    # ONE WORD, ONE MEANING, ALIGNED 2026-09-21. `falsy` used to mean "missing or falsy" here
    # while the fixed-path operator of the same name meant "present and falsy", so the two
    # vocabularies disagreed on the only case that matters, and the comment below claimed they
    # were one. A reader of a check file met a second grammar without being told. The two
    # meanings are both wanted, so they now have two words rather than one word and a surprise.
    "falsy": lambda v: v is not MISSING and not v,
    "unset_or_falsy": lambda v: v is MISSING or not v,
    # "IT CARRIES A MEASUREMENT". Needed by applicability rather than by any rule: a check whose
    # rule reads a denominator has nothing to say about an entry that has none, and `any_outside`
    # steps over such an entry silently, so without this the check applies and reports clean.
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    # ZERO IS NOT FALSY HERE, and the distinction is the entire point of the check that needed
    # this. `null` and `0` are both falsy in Python and they are opposite claims about a
    # measurement: one says it was never taken, the other says it was taken and came out zero.
    # A bool is excluded because a bool is an int in Python and is never a measurement.
    "zero": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0,
}


def _entry_matches(entry, conditions) -> bool:
    """Whether one mapping entry satisfies every condition in `conditions`."""
    if not isinstance(entry, dict):
        return False
    for cond in conditions:
        if not isinstance(cond, dict) or not isinstance(cond.get("field"), str):
            raise CheckError(f"an `any_entry` condition needs a string `field`, got {cond!r}")
        value = entry.get(cond["field"], MISSING)
        if "equals" in cond:
            if value is MISSING or value != cond["equals"]:
                return False
            continue
        predicate = _ENTRY_PREDICATES.get(cond.get("is"))
        if predicate is None:
            raise CheckError(
                f"an `any_entry` condition needs `equals` or an `is` from "
                f"{', '.join(sorted(_ENTRY_PREDICATES))}, got {cond!r}")
        if not predicate(value):
            return False
    return True


def evaluate(rule: dict, doc: Any) -> bool:
    """Whether `rule` holds for `doc`.

    Every operator is total: it answers true or false for any document, and never raises on a
    shape it did not expect. A rule that could raise would turn an unfamiliar artefact into a
    crash, when the honest answer is "this check does not apply here".
    """
    if not isinstance(rule, dict) or "op" not in rule:
        raise CheckError(f"a rule must be an object with an `op`, got {rule!r}")
    op = rule["op"]

    if op == "always":
        return True
    if op == "never":
        return False

    if op in ("all_of", "any_of"):
        subs = rule.get("rules")
        if not isinstance(subs, list) or not subs:
            raise CheckError(f"`{op}` needs a non-empty `rules` list")
        results = (evaluate(s, doc) for s in subs)
        return all(results) if op == "all_of" else any(results)

    if op == "not":
        sub = rule.get("rule")
        if not isinstance(sub, dict):
            raise CheckError("`not` needs a `rule` object")
        return not evaluate(sub, doc)

    if op == "any_missing":
        # THE ONE OPERATOR ADDED AFTER THE FACT, and the rule for extending the vocabulary said
        # to do it when a real check cannot be written otherwise and to name that check. This is
        # it: "does any metric in this artefact lack an estimator" cannot be asked with fixed
        # paths, because the metric keys differ per harness (`gsm8k.acc,none` under lm-eval,
        # `choice.accuracy` under Inspect, `refusal_rate.heretic-keyword` under ours). Without a
        # quantifier the estimator check would have to be rewritten once per harness, which is
        # the per-harness maintenance the adapters exist to avoid.
        mapping = dotted(doc, rule.get("path", ""))
        field = rule.get("field")
        if not isinstance(field, str):
            raise CheckError("`any_missing` needs a string `field`")
        if not isinstance(mapping, dict):
            return False
        return any(
            not isinstance(v, dict) or not v.get(field) for v in mapping.values())

    if op == "any_outside":
        # THE SECOND OPERATOR ADDED AFTER THE FACT, and the rule says to name the check that
        # needed it. Three did: `impossible-proportion-reported`, `rate-reported-on-a-sample-too-
        # small-to-carry-it`, and the sample-size half of the budget check. All three ask "is any
        # metric's number outside the range its own units allow", which needs a numeric comparison
        # (the vocabulary had none) and a quantifier over metric keys that differ per harness, for
        # the same reason `any_missing` exists.
        #
        # FOUND BY A HOSTILE OUTSIDE REVIEW, 2026-09-17: `senbonzakura check` reported
        # "0 finding(s)" on a refusal rate of MINUS 0.5 and a keyword rate of 1.5. It was not
        # wrong, it was empty, and zero findings over zero applicable checks is arithmetically
        # identical to a clean sweep.
        mapping = dotted(doc, rule.get("path", ""))
        field = rule.get("field")
        if not isinstance(field, str):
            raise CheckError("`any_outside` needs a string `field`")
        low, high = rule.get("min"), rule.get("max")
        if low is None and high is None:
            raise CheckError("`any_outside` needs at least one of `min` or `max`")
        when = rule.get("when") or {}
        if not isinstance(mapping, dict):
            return False
        for entry in mapping.values():
            if not isinstance(entry, dict):
                continue
            if when and entry.get(when.get("field")) != when.get("equals"):
                continue
            value = entry.get(field)
            # A bool is an int in Python and is never a measurement. Excluded explicitly so a
            # `higher_is_better: false` can never be read as the number zero.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if low is not None and value < low:
                return True
            if high is not None and value > high:
                return True
        return False

    if op == "any_entry":
        # THE THIRD OPERATOR ADDED AFTER THE FACT, and the rule says to name the checks that
        # needed it. Two did: `null-reported-as-zero` and `a-rate-with-no-partition-beside-it`.
        #
        # Both ask a question about ONE metric entry satisfying SEVERAL conditions at once, which
        # `any_missing` and `any_outside` cannot express: each of those quantifies one predicate
        # over the mapping, so writing "a metric whose value is zero AND which has no denominator"
        # as a conjunction of two of them asks instead for "some metric has a zero value, and some
        # metric, possibly a different one, has no denominator". On a document with two metrics
        # those are different questions and the second one fires wrongly.
        #
        # The predicate vocabulary is deliberately the same five words the fixed-path operators
        # use, plus `equals`, so a reader of a check file does not meet a second grammar.
        mapping = dotted(doc, rule.get("path", ""))
        conditions = rule.get("all")
        if not isinstance(conditions, list) or not conditions:
            raise CheckError("`any_entry` needs a non-empty `all` list of per-entry conditions")
        if not isinstance(mapping, dict):
            return False
        return any(_entry_matches(entry, conditions) for entry in mapping.values())

    if op == "less_than":
        # THE FOURTH, needed by `quoted-at-a-budget-below-the-visibility-floor`: the generation
        # budget lives at a FIXED path (`raw.generation.max_new_tokens`) rather than inside the
        # per-harness metrics mapping, so `any_outside` cannot reach it, and the vocabulary had no
        # numeric comparison outside that mapping at all.
        #
        # A non-number is NOT less than anything here. A missing budget is a different finding
        # from a short one, and silently reading absence as zero would make this check fire on
        # every artefact that does not record a budget, which is most of them.
        limit = rule.get("value")
        if not isinstance(rule.get("path"), str):
            raise CheckError("`less_than` needs a string `path`")
        if not isinstance(limit, (int, float)) or isinstance(limit, bool):
            raise CheckError("`less_than` needs a numeric `value`")
        value = dotted(doc, rule["path"])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return value < limit

    if op == "disagrees":
        # THE FIFTH, needed by `an-arm-labelled-by-a-setting-it-did-not-apply` and by
        # `a-count-that-is-not-the-count-that-was-applied`. Both ask whether two fields of one
        # artefact that are supposed to describe the same quantity actually do.
        #
        # A LIST ON ONE SIDE IS COMPARED ELEMENTWISE, which is not a convenience: the second check
        # compares `directions_per_layer`, a per-layer list, against `num_directions`, a scalar,
        # and the finding is that any layer disagrees. Collapsing the list first would need a
        # choice of summary, and every summary of that list has already been the wrong one at
        # least once here.
        #
        # Missing on either side is FALSE. "These two fields disagree" is a claim about two
        # recorded values, and an artefact that records only one has not made it.
        paths = rule.get("paths")
        if not isinstance(paths, list) or len(paths) != 2:
            raise CheckError("`disagrees` needs exactly two `paths`")
        a, b = (dotted(doc, p) for p in paths)
        if a is MISSING or b is MISSING or a is None or b is None:
            return False
        if isinstance(a, (list, tuple)) and not isinstance(b, (list, tuple)):
            return any(item != b for item in a)
        if isinstance(b, (list, tuple)) and not isinstance(a, (list, tuple)):
            return any(item != a for item in b)
        return a != b

    if op == "differs_in_more_than":
        # THE SIXTH OPERATOR ADDED AFTER THE FACT, and the rule says to name the check that
        # needed it. `arms-that-differ-in-more-than-the-named-variable` did, and nothing in the
        # existing vocabulary can express it: `disagrees` answers "do these two fields differ"
        # one pair at a time, and an `any_of` over several of those answers "at least one
        # differs", which is true of every honest comparison ever run. The finding is about the
        # COUNT, because a comparison that moves one setting attributes its result to that
        # setting and a comparison that moves three attributes it to nothing.
        #
        # ONLY KEYS RECORDED ON BOTH SIDES ARE COMPARED, and that is a deliberate narrowing
        # rather than an oversight. A key one arm writes and the other does not is a difference
        # in what was RECORDED, which is usually two versions of a producer rather than two
        # settings, and counting it would fire on every pair of artefacts written months apart.
        # That question belongs to `a-figure-compared-across-an-instrument-change`, which asks it
        # directly instead of inferring it from a field count.
        paths = rule.get("paths")
        if not isinstance(paths, list) or len(paths) != 2:
            raise CheckError("`differs_in_more_than` needs exactly two `paths`")
        limit = rule.get("value")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise CheckError("`differs_in_more_than` needs a non-negative integer `value`")
        a, b = (dotted(doc, p) for p in paths)
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        differing = sum(1 for k in a.keys() & b.keys() if a[k] != b[k])
        return differing > limit

    if op == "intersects":
        paths = rule.get("paths")
        if not isinstance(paths, list) or len(paths) != 2:
            raise CheckError("`intersects` needs exactly two `paths`")
        a, b = (dotted(doc, p) for p in paths)
        if a is MISSING or b is MISSING:
            return False
        return bool(_as_set(a) & _as_set(b))

    path = rule.get("path")
    if not isinstance(path, str):
        raise CheckError(f"`{op}` needs a string `path`")
    value = dotted(doc, path)

    if op == "absent":
        return value is MISSING
    if op == "present":
        return value is not MISSING
    if op == "truthy":
        return value is not MISSING and bool(value)
    if op == "falsy":
        return value is not MISSING and not bool(value)
    if op == "equals":
        return value is not MISSING and value == rule.get("value")
    if op == "not_equals":
        return value is not MISSING and value != rule.get("value")
    if op == "in":
        values = rule.get("values")
        if not isinstance(values, list):
            raise CheckError("`in` needs a `values` list")
        return value is not MISSING and value in values
    if op == "not_in":
        values = rule.get("values")
        if not isinstance(values, list):
            raise CheckError("`not_in` needs a `values` list")
        return value is not MISSING and value not in values

    raise CheckError(
        f"unknown rule operator {op!r}. The vocabulary is deliberately small; if a real check "
        f"cannot be written with it, extend the engine and say in the commit which check needed "
        f"it.")


def _validate(raw: dict, source: Path | None) -> Check:
    where = f" in {source}" if source else ""
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise CheckError(
            f"check{where} is missing {', '.join(missing)}. Every field is mandatory: a check "
            f"with no incident is an opinion, and one that cannot say what would make it wrong "
            f"cannot be judged by the person reading its finding.")

    severity = raw.get("severity", DEFAULT_SEVERITY)
    if severity not in SEVERITIES:
        raise CheckError(
            f"check {raw['id']!r}{where} has severity {severity!r}; expected one of "
            f"{', '.join(SEVERITIES)}. Severity is how far the finding could move the number and "
            f"is not the same axis as confidence, which is how much to believe it.")

    if raw["confidence"] not in CONFIDENCES:
        raise CheckError(
            f"check {raw['id']!r}{where} has confidence {raw['confidence']!r}; "
            f"expected one of {', '.join(CONFIDENCES)}")

    arity = raw.get("arity", DEFAULT_ARITY)
    if arity not in ARITIES:
        raise CheckError(
            f"check {raw['id']!r}{where} has arity {arity!r}; expected one of "
            f"{', '.join(ARITIES)}. A check reads one artefact or compares two, and the engine "
            f"has to know which before it can decide whether a single file SKIPS it or passes "
            f"it.")

    control = raw["control"]
    if not isinstance(control, dict):
        raise CheckError(f"check {raw['id']!r}{where}: `control` must be an object")
    for key in ("fires_on", "passes_on"):
        got = control.get(key)
        if not isinstance(got, list) or not got:
            raise CheckError(
                f"check {raw['id']!r}{where} has no `control.{key}`. Every check ships a document "
                f"that makes it fire and one that does not, because a check nobody has watched "
                f"fail is a check nobody has tested.")
        if arity == "pair":
            # A PAIR CHECK'S CONTROL IS A PAIR, and it is refused rather than coerced. A control
            # written as a single document would be evaluated against a pair document whose two
            # arms are both missing, where every rule here answers false: the check would pass
            # its negative control for the reason that it examined nothing, which is the exact
            # thing the controls exist to rule out.
            for i, entry in enumerate(got):
                if not isinstance(entry, list) or len(entry) != 2:
                    raise CheckError(
                        f"check {raw['id']!r}{where}: `control.{key}[{i}]` must be a two-element "
                        f"list, because this check compares two artefacts and a control that is "
                        f"not a pair proves nothing about it.")

    return Check(**{f: raw[f] for f in REQUIRED_FIELDS},
                 severity=severity, arity=arity, source=source)


def load_checks(directory: Path | str | None = None) -> list[Check]:
    """Every check in `directory`, sorted by id.

    Sorted so a report's order does not depend on the filesystem's, per baseline section 2.1 on
    reproducibility: two runs on the same input produce the same output.
    """
    root = Path(directory) if directory is not None else CHECKS_DIR
    out = []
    for path in sorted(root.glob("*.json")):
        try:
            raw = read_json_bounded(path)
        except (OSError, json.JSONDecodeError, ArtefactTooLargeError) as e:
            raise CheckError(f"could not read the check at {path}: {e}") from e
        out.append(_validate(raw, path))

    ids = [c.id for c in out]
    duplicated = {i for i in ids if ids.count(i) > 1}
    if duplicated:
        raise CheckError(
            f"two check files share an id: {', '.join(sorted(duplicated))}. An id is how a "
            f"report line is traced back to the check that produced it.")
    return sorted(out, key=lambda c: c.id)


def run_checks(doc: Any, checks=None, *, artefact: str | None = None):
    """Run `checks` over one artefact. Returns (findings, skipped_ids).

    SKIPPED IS RETURNED SEPARATELY AND IS NOT A PASS. A check whose `applies_to` is false did not
    examine this document, and reporting that as "nothing found" is the worst output the v0.8
    plan names: indistinguishable from a clean bill of health. The caller is expected to say how
    many checks did not apply.
    """
    checks = load_checks() if checks is None else checks
    findings, skipped = [], []
    for check in checks:
        # A PAIR CHECK IS SKIPPED HERE, NEVER PASSED. It asks a question about two artefacts and
        # one was supplied, so it did not examine anything; reporting that as a pass would be
        # this command claiming a comparison is sound when no comparison was read.
        if check.arity == "pair":
            skipped.append(check.id)
            continue
        if not evaluate(check.applies_to, doc):
            skipped.append(check.id)
            continue
        if evaluate(check.rule, doc):
            findings.append(Finding(
                check_id=check.id, title=check.title, detects=check.detects,
                incident=check.incident, remedy=check.remedy,
                confidence=check.confidence, false_positive=check.false_positive,
                severity=check.severity, artefact=artefact))
    return sorted(findings, key=weight), skipped


def as_pair(doc_a, doc_b) -> dict:
    """The one document a pair check is evaluated against.

    The keys are `arm_a` and `arm_b`, and a check file reaches into either side with an ordinary
    dotted path (`arm_a.metrics`, `arm_b.settings`). That is the whole trick: a pair check needs
    no new rule vocabulary, only a document with two halves.
    """
    return dict(zip(PAIR_KEYS, (doc_a, doc_b), strict=True))


def run_pair_checks(doc_a, doc_b, checks=None, *, artefact: str | None = None):
    """Run the PAIR checks over two artefacts. Returns (findings, skipped_ids).

    Single-document checks are not run here and are not reported as skipped: the caller has
    already run them over each file on its own, and listing them again would inflate the
    "did not apply" count with checks that did apply, somewhere else.
    """
    checks = load_checks() if checks is None else checks
    pair = as_pair(doc_a, doc_b)
    findings, skipped = [], []
    for check in checks:
        if check.arity != "pair":
            continue
        if not evaluate(check.applies_to, pair):
            skipped.append(check.id)
            continue
        if evaluate(check.rule, pair):
            findings.append(Finding(
                check_id=check.id, title=check.title, detects=check.detects,
                incident=check.incident, remedy=check.remedy,
                confidence=check.confidence, false_positive=check.false_positive,
                severity=check.severity, artefact=artefact))
    return sorted(findings, key=weight), skipped


def weight(finding) -> tuple:
    """The order a report lists findings in: worst first, then most believable, then by id.

    THE RUNG ASKS FOR FINDINGS RANKED BY HOW BADLY THEY COULD MOVE THE NUMBER, and until this
    existed the report listed them in whatever order the loader happened to produce, which is
    alphabetical by check id. Alphabetical is stable and reproducible, which is why nobody
    noticed it was also meaningless: a reader skimming the first finding was reading the one
    whose name sorted earliest.

    Ties break on confidence and then on id, so the order stays total and two runs over the same
    input still produce the same report, per baseline section 2.1.
    """
    return (SEVERITIES.index(finding.severity) if finding.severity in SEVERITIES else len(SEVERITIES),
            CONFIDENCES.index(finding.confidence) if finding.confidence in CONFIDENCES else len(CONFIDENCES),
            finding.check_id)
