# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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
    confidence    high / medium / low.
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

#: Every field a check file must carry. Absence of any one is a refusal, not a default.
REQUIRED_FIELDS = (
    "id", "title", "detects", "incident", "remedy",
    "confidence", "false_positive", "applies_to", "rule", "control",
)

CONFIDENCES = ("high", "medium", "low")

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

    if raw["confidence"] not in CONFIDENCES:
        raise CheckError(
            f"check {raw['id']!r}{where} has confidence {raw['confidence']!r}; "
            f"expected one of {', '.join(CONFIDENCES)}")

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

    return Check(**{f: raw[f] for f in REQUIRED_FIELDS}, source=source)


def load_checks(directory: Path | str | None = None) -> list[Check]:
    """Every check in `directory`, sorted by id.

    Sorted so a report's order does not depend on the filesystem's, per baseline section 2.1 on
    reproducibility: two runs on the same input produce the same output.
    """
    root = Path(directory) if directory is not None else CHECKS_DIR
    out = []
    for path in sorted(root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
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
        if not evaluate(check.applies_to, doc):
            skipped.append(check.id)
            continue
        if evaluate(check.rule, doc):
            findings.append(Finding(
                check_id=check.id, title=check.title, detects=check.detects,
                incident=check.incident, remedy=check.remedy,
                confidence=check.confidence, false_positive=check.false_positive,
                artefact=artefact))
    return findings, skipped
