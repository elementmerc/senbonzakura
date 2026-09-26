# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The check registry: the engine, and every shipped check's own control.

TWO KINDS OF TEST IN HERE, and the second kind is the point.

The first kind tests the engine: the rule operators, the loader's refusals, the difference
between a skip and a pass. Ordinary unit tests.

The second kind is generated from the checks themselves. Every check file ships
`control.fires_on` and `control.passes_on`, and this module turns each one into a test case. So
adding a check adds its own tests, and a check that cannot demonstrate firing cannot be merged.

That requirement was read out of soup on 2026-09-02: their parity harness ships a flag that
breaks the comparison deliberately and requires a non-zero difference. The principle is the one
this project arrived at the hard way, after a dead-flag audit passed a tree with the bug
reinstated: **a check nobody has watched fail is a check nobody has tested.**
"""
import json

import pytest
from senbonzakura_check import registry
from senbonzakura_check.registry import (
    MISSING,
    Check,
    CheckError,
    dotted,
    evaluate,
    load_checks,
    run_checks,
)

CHECKS = load_checks()


# ── the shipped checks, and their own controls ───────────────────────────────────────────────

def test_some_checks_actually_ship():
    """Zero checks would make every generated case below vacuously absent."""
    assert CHECKS, "no checks found; has the directory moved?"


def _as_document(check, control):
    """The document a control is evaluated against, which for a pair check is the pair.

    A pair check's controls are two-element lists, and the registry refuses any other shape. They
    are assembled here exactly as `run_pair_checks` assembles a real pair, so a control that
    passes here is a control the engine would evaluate identically.
    """
    return registry.as_pair(*control) if check.arity == "pair" else control


def _control_cases(kind):
    return [(c.id, i, _as_document(c, doc))
            for c in CHECKS for i, doc in enumerate(c.control[kind])]


@pytest.mark.parametrize(("check_id", "i", "doc"),
                         _control_cases("fires_on"),
                         ids=[f"{cid}[{i}]" for cid, i, _ in _control_cases("fires_on")])
def test_every_check_fires_on_its_own_positive_control(check_id, i, doc):
    """THE REQUIREMENT THAT MAKES A CHECK MERGEABLE.

    A check that has never been watched failing is a check nobody has tested, and a linter full
    of those is one that reports a clean run over checks that could not fire if they tried.
    """
    check = next(c for c in CHECKS if c.id == check_id)
    assert evaluate(check.applies_to, doc), (
        f"{check_id}: positive control {i} does not even reach the check, because `applies_to` "
        f"is false for it. A control that is skipped proves nothing.")
    assert evaluate(check.rule, doc), f"{check_id}: positive control {i} did not fire"


@pytest.mark.parametrize(("check_id", "i", "doc"),
                         _control_cases("passes_on"),
                         ids=[f"{cid}[{i}]" for cid, i, _ in _control_cases("passes_on")])
def test_every_check_stays_quiet_on_its_own_negative_control(check_id, i, doc):
    """The other half. A check that fires on everything is worse than no check.

    A negative control is allowed to be SKIPPED rather than passed: a document the check knows
    nothing about is a legitimate quiet outcome, and saying so is more honest than pretending
    the check examined it.
    """
    check = next(c for c in CHECKS if c.id == check_id)
    if not evaluate(check.applies_to, doc):
        return
    assert not evaluate(check.rule, doc), (
        f"{check_id}: negative control {i} fired, so the check reports a defect on a document "
        f"that is fine")


@pytest.mark.parametrize("check", CHECKS, ids=[c.id for c in CHECKS])
def test_every_check_says_what_would_make_it_wrong(check):
    """Loophole 5 of the v0.8 plan, and it is absolute: a check that cannot say what would make
    it wrong does not ship.

    A linter with a bad false-positive rate is uninstalled once and never again, so the person
    reading a finding has to be able to judge it without going and reading the source. That
    means the sentence lives in the check, travels into the report, and is long enough to say
    something.
    """
    assert len(check.false_positive) > 80, (
        f"{check.id}: `false_positive` is too short to be a real account of when this check is "
        f"wrong. It is printed beside the finding and is what lets a reader judge it.")
    assert len(check.incident) > 60, (
        f"{check.id}: `incident` is too short. A finding with no citation is an opinion.")


@pytest.mark.parametrize("check", CHECKS, ids=[c.id for c in CHECKS])
def test_every_check_has_a_kebab_case_id_matching_its_filename(check):
    """The id is how a report line is traced back to the check that produced it, so it has to be
    findable. A file whose name and id disagree sends the reader to the wrong place.
    """
    assert check.source is not None
    assert check.source.stem == check.id
    assert check.id == check.id.lower().strip()
    assert " " not in check.id and "_" not in check.id


# ── the `applies_to` defect class, and the guard that stops it coming back ───────────────────

#: An artefact that records a measurement and nothing else. Every normalised field that comes out
#: of this as None is one the ADAPTER wrote rather than the producer, which is the whole trap: a
#: check testing `present` on one of them examines every artefact in existence and reports each
#: one clean on a question it never asked.
_RECORDS_NOTHING_ELSE = {"model": "m", "label": "arm", "instrument": "senbonzakura.refusal_rate",
                         "refusal": 0.31, "n": 200, "eval": "held-out"}


def _keys_the_adapter_always_writes():
    from senbonzakura_check import normalise
    doc = normalise(json.loads(json.dumps(_RECORDS_NOTHING_ELSE)))
    return {k for k, v in doc.items() if v is None}


def _unguarded_present(rule, guarded=frozenset()):
    """Every path a `present` test reads without a `not_equals null` beside it.

    `guarded` carries the paths an enclosing `all_of` has already pinned to a non-null value,
    because `present` AND `not_equals null` together is the correct way to ask this question and
    the pair has to read as one test rather than as an offence plus an unrelated rule.
    """
    if not isinstance(rule, dict):
        return set()
    op = rule.get("op")
    if op == "all_of":
        subs = rule.get("rules") or []
        here = guarded | {s.get("path") for s in subs
                          if isinstance(s, dict) and s.get("op") == "not_equals"
                          and s.get("value") is None}
        return set().union(*(_unguarded_present(s, here) for s in subs)) if subs else set()
    if op in ("any_of", "not"):
        subs = rule.get("rules") or ([rule["rule"]] if isinstance(rule.get("rule"), dict) else [])
        return set().union(*(_unguarded_present(s, guarded) for s in subs)) if subs else set()
    if op == "present" and rule.get("path") not in guarded:
        return {rule.get("path")}
    return set()


@pytest.mark.parametrize("check", CHECKS, ids=[c.id for c in CHECKS])
def test_no_check_tests_present_on_a_field_the_adapter_always_writes(check):
    """THE DEFECT CLASS, GATED SO IT CANNOT COME BACK.

    Found on 2026-09-21 in `quoted-at-a-budget-below-the-visibility-floor`, which asked
    `present` on `generation_budget`. The adapter writes that key on every artefact it
    normalises, with None in it when the producer recorded no budget, so the check applied to
    every artefact in the corpus and reported each one clean on a question it never asked. Two
    more checks had the identical shape on `budget_warning` and on `chat_template`.

    `present` and `absent` are the right questions to ask of a RAW path, where absence is the
    producer's own silence. On a normalised path they are answered by the adapter rather than by
    the artefact, and the honest test is `present` together with `not_equals null`, which is what
    this allows.
    """
    always = _keys_the_adapter_always_writes()
    offending = sorted(p for p in _unguarded_present(check.applies_to) if p in always)
    assert not offending, (
        f"{check.id}: `applies_to` tests `present` on {offending}, which the adapter writes on "
        f"every artefact whether or not the producer recorded anything. The check would examine "
        f"every file and report each one clean on a question it never asked. Pair it with "
        f"`not_equals <path> null`.")


def test_that_guard_can_actually_fail():
    """A guard nobody has watched fail is a guard nobody has tested, which is the same rule the
    per-check controls exist for. Rebuilt here as the exact shape the three real checks had.
    """
    always = _keys_the_adapter_always_writes()
    assert always, "the adapter writes no unconditional fields, so the guard above tests nothing"
    victim = min(always)
    assert _unguarded_present({"op": "present", "path": victim}) == {victim}
    assert _unguarded_present({"op": "any_of", "rules": [{"op": "present", "path": victim}]})
    assert not _unguarded_present({"op": "all_of", "rules": [
        {"op": "present", "path": victim},
        {"op": "not_equals", "path": victim, "value": None}]})


@pytest.mark.parametrize(("check_id", "field"), [
    ("quoted-at-a-budget-below-the-visibility-floor", "generation_budget"),
    ("artefact-carries-its-own-warning", "budget_warning"),
    # THE SECOND FIELD OF THE SAME KIND, added 2026-09-26. The sibling above exists because a
    # reviewer fed this checker a file whose own `budget_warning` said the number might be wrong and
    # was told `nothing found`. That was fixed for one field, and a first-time reader met the same
    # failure through a new one: the compass told them THE AUC ABOVE IS NOT A MEASUREMENT, in those
    # words, and the checker called the file clean.
    ("a-figure-its-own-file-calls-invalid", "self_invalidated"),
    ("chat-template-never-applied", "chat_template"),
])
def test_an_artefact_recording_nothing_for_a_field_skips_the_check_that_reads_it(check_id, field):
    """The behavioural half of the same property, per check rather than over the rule tree.

    Asserted through `check_document`, so it goes through detection and normalisation the way a
    real file does: the trap was created by the adapter, and a test that fed the rule evaluator a
    handwritten document would never have met it.
    """
    from senbonzakura_check import check_document, normalise

    doc = json.loads(json.dumps(_RECORDS_NOTHING_ELSE))
    assert normalise(doc)[field] is None, (
        f"this fixture was supposed to record nothing under {field}")
    findings, skipped = check_document(doc, CHECKS)
    assert check_id not in {f.check_id for f in findings}
    assert check_id in skipped, (
        f"{check_id} examined an artefact that records nothing under {field}. Skipped and passed "
        f"are different outcomes and this one is a skip.")


# ── the loader's refusals ────────────────────────────────────────────────────────────────────

def _minimal(**over):
    doc = {
        "id": "x", "title": "t", "detects": "d",
        "incident": "i", "remedy": "r", "confidence": "high",
        "false_positive": "fp",
        "applies_to": {"op": "always"}, "rule": {"op": "never"},
        "control": {"fires_on": [{}], "passes_on": [{}]},
    }
    doc.update(over)
    return doc


def _write(tmp_path, name, doc):
    (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("field", registry.REQUIRED_FIELDS)
def test_a_check_missing_any_mandatory_field_is_refused(field, tmp_path):
    """Every field is mandatory and none has a default. A check with no incident is an opinion,
    and one with no `false_positive` cannot be judged by whoever reads its finding.
    """
    doc = _minimal()
    del doc[field]
    _write(tmp_path, "x.json", doc)
    with pytest.raises(CheckError, match=field):
        load_checks(tmp_path)


def test_an_unknown_confidence_is_refused(tmp_path):
    _write(tmp_path, "x.json", _minimal(confidence="pretty sure"))
    with pytest.raises(CheckError, match="confidence"):
        load_checks(tmp_path)


@pytest.mark.parametrize("control", [
    {"fires_on": [], "passes_on": [{}]},
    {"fires_on": [{}], "passes_on": []},
    {"fires_on": [{}]},
    "not an object",
])
def test_a_check_without_both_controls_is_refused(control, tmp_path):
    """No exceptions, including for a check whose author is certain it works."""
    _write(tmp_path, "x.json", _minimal(control=control))
    with pytest.raises(CheckError):
        load_checks(tmp_path)


def test_two_checks_sharing_an_id_are_refused(tmp_path):
    _write(tmp_path, "a.json", _minimal(id="same"))
    _write(tmp_path, "b.json", _minimal(id="same"))
    with pytest.raises(CheckError, match="share an id"):
        load_checks(tmp_path)


def test_an_unreadable_check_stops_the_run_rather_than_being_skipped(tmp_path):
    """A REGISTRY THAT SILENTLY DROPS WHAT IT COULD NOT READ reports a clean run over a set of
    checks nobody can enumerate. That is the same defect the checker exists to find.
    """
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckError, match="could not read"):
        load_checks(tmp_path)


def test_an_empty_directory_loads_to_nothing_without_complaining(tmp_path):
    """Not an error: a caller may legitimately point this at their own check set."""
    assert load_checks(tmp_path) == []


# ── dotted paths ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("path", "expected"), [
    ("a", 1),
    ("b.c", 2),
    ("b.d.0", "first"),
    ("b.d.1", "second"),
    ("nothing", MISSING),
    ("b.nothing", MISSING),
    ("a.b", MISSING),
    ("b.d.9", MISSING),
    ("b.d.x", MISSING),
])
def test_dotted_reads_a_path_or_says_it_is_missing(path, expected):
    doc = {"a": 1, "b": {"c": 2, "d": ["first", "second"]}}
    assert dotted(doc, path) is expected or dotted(doc, path) == expected


def test_a_present_null_is_not_the_same_as_an_absent_field():
    """MISSING is a sentinel rather than None, because those are different claims.

    A harness that writes `"chat_template": null` has said something ("there isn't one"); one
    that omits the key has said nothing. A check that conflates them fires on the wrong artefacts.
    """
    doc = {"here": None}
    assert dotted(doc, "here") is None
    assert dotted(doc, "absent") is MISSING
    assert evaluate({"op": "present", "path": "here"}, doc) is True
    assert evaluate({"op": "absent", "path": "here"}, doc) is False
    assert evaluate({"op": "absent", "path": "gone"}, doc) is True


# ── the rule operators ───────────────────────────────────────────────────────────────────────

DOC = {"n": 3, "zero": 0, "s": "x", "t": True, "f": False,
       "list": [1, 2, 3], "other": [3, 4], "disjoint": [9]}


@pytest.mark.parametrize(("rule", "expected"), [
    ({"op": "always"}, True),
    ({"op": "never"}, False),
    ({"op": "present", "path": "n"}, True),
    ({"op": "present", "path": "gone"}, False),
    ({"op": "absent", "path": "gone"}, True),
    ({"op": "truthy", "path": "n"}, True),
    ({"op": "truthy", "path": "zero"}, False),
    ({"op": "truthy", "path": "gone"}, False),
    ({"op": "falsy", "path": "zero"}, True),
    ({"op": "falsy", "path": "gone"}, False),
    ({"op": "equals", "path": "s", "value": "x"}, True),
    ({"op": "equals", "path": "s", "value": "y"}, False),
    ({"op": "equals", "path": "gone", "value": None}, False),
    ({"op": "not_equals", "path": "s", "value": "y"}, True),
    ({"op": "in", "path": "n", "values": [1, 3]}, True),
    ({"op": "in", "path": "n", "values": [1, 2]}, False),
    ({"op": "not_in", "path": "n", "values": [1, 2]}, True),
    ({"op": "intersects", "paths": ["list", "other"]}, True),
    ({"op": "intersects", "paths": ["list", "disjoint"]}, False),
    ({"op": "intersects", "paths": ["list", "gone"]}, False),
    ({"op": "not", "rule": {"op": "never"}}, True),
    ({"op": "all_of", "rules": [{"op": "always"}, {"op": "never"}]}, False),
    ({"op": "any_of", "rules": [{"op": "always"}, {"op": "never"}]}, True),
])
def test_each_operator_answers_the_question_it_claims_to(rule, expected):
    assert evaluate(rule, DOC) is expected


def test_a_truthy_on_a_missing_field_is_false_not_an_error():
    """EVERY OPERATOR IS TOTAL, and that is a design requirement rather than a convenience.

    The input is somebody else's artefact, possibly from a version nobody here has seen. A rule
    that raised on an unfamiliar shape would turn "this check does not apply" into a crash.
    """
    assert evaluate({"op": "truthy", "path": "deeply.nested.absent"}, {}) is False
    assert evaluate({"op": "intersects", "paths": ["a", "b"]}, {"a": "not a list"}) is False


@pytest.mark.parametrize("rule", [
    {"op": "no-such-operator", "path": "a"},
    {"op": "all_of", "rules": []},
    {"op": "all_of"},
    {"op": "not"},
    {"op": "intersects", "paths": ["only-one"]},
    {"op": "equals"},
    {"op": "in", "path": "a"},
    {"op": "not_in", "path": "a"},
    "not a rule at all",
    {"no-op-key": 1},
])
def test_a_malformed_rule_is_refused_loudly(rule):
    """A rule that cannot be evaluated is an authoring error and has to stop the run. Treating it
    as false would let a broken check sit in the registry reporting nothing forever.
    """
    with pytest.raises(CheckError):
        evaluate(rule, DOC)


# ── running checks over an artefact ──────────────────────────────────────────────────────────

def _check(**over):
    return Check(**{
        "id": "c", "title": "t", "detects": "d", "incident": "i", "remedy": "r",
        "confidence": "high", "false_positive": "fp",
        "applies_to": {"op": "always"}, "rule": {"op": "always"},
        "control": {"fires_on": [{}], "passes_on": [{}]}
    , **over})


def test_a_firing_check_produces_a_finding_carrying_its_own_caveat():
    """The report prints `false_positive` beside the finding, so it has to travel with it."""
    findings, skipped = run_checks({}, [_check()], artefact="run.json")
    assert skipped == []
    assert len(findings) == 1
    f = findings[0]
    assert (f.check_id, f.artefact, f.confidence) == ("c", "run.json", "high")
    assert f.false_positive == "fp" and f.incident == "i" and f.remedy == "r"


def test_a_check_that_does_not_apply_is_skipped_and_not_counted_as_a_pass():
    """THE WORST POSSIBLE OUTPUT, per the v0.8 plan, is a silent pass on a document the checker
    could not understand, because it is indistinguishable from a clean bill of health.

    So `run_checks` returns the skips separately and the caller is expected to report them.
    """
    findings, skipped = run_checks({}, [_check(applies_to={"op": "never"})])
    assert findings == []
    assert skipped == ["c"]


def test_a_check_that_applies_and_does_not_fire_is_neither_a_finding_nor_a_skip():
    findings, skipped = run_checks({}, [_check(rule={"op": "never"})])
    assert findings == [] and skipped == []


# ── a check that needs two documents ─────────────────────────────────────────────────────────

def test_a_pair_check_is_skipped_on_a_single_document_and_never_passed():
    """THE CENTRAL DISTINCTION OF THE PAIR DESIGN, and the reason the engine grew an arity at all.

    A pair check asks a question about two artefacts. Handed one, it examined nothing, and
    reporting that as a pass would be the checker claiming a comparison is sound when no
    comparison was read. That is the same defect as a silent pass on an unparsed file, one level
    up.
    """
    check = _check(arity="pair", rule={"op": "always"}, applies_to={"op": "always"})
    findings, skipped = run_checks({"anything": 1}, [check])
    assert findings == [], "a pair check fired on a single document"
    assert skipped == ["c"], "a pair check on one document must be SKIPPED, not passed"


def test_a_single_document_check_does_not_run_on_a_pair():
    """The mirror. `run_pair_checks` runs the pair checks and leaves the rest alone: each arm has
    already been checked on its own, and listing the single-document checks as skipped here would
    inflate the "did not apply" count with checks that did apply, somewhere else.
    """
    findings, skipped = registry.run_pair_checks({}, {}, [_check(rule={"op": "always"})])
    assert findings == [] and skipped == []


def test_a_pair_check_reads_both_arms_through_ordinary_dotted_paths():
    check = _check(arity="pair", applies_to={"op": "always"},
                   rule={"op": "disagrees", "paths": ["arm_a.k", "arm_b.k"]})
    fired, _ = registry.run_pair_checks({"k": 1}, {"k": 3}, [check], artefact="a vs b")
    quiet, _ = registry.run_pair_checks({"k": 1}, {"k": 1}, [check])
    assert [f.check_id for f in fired] == ["c"] and fired[0].artefact == "a vs b"
    assert quiet == []


def test_a_pair_check_whose_applies_to_is_false_is_skipped():
    check = _check(arity="pair", applies_to={"op": "never"})
    findings, skipped = registry.run_pair_checks({}, {}, [check])
    assert findings == [] and skipped == ["c"]


def test_an_unknown_arity_is_refused(tmp_path):
    _write(tmp_path, "x.json", _minimal(arity="triple"))
    with pytest.raises(CheckError, match="arity"):
        load_checks(tmp_path)


@pytest.mark.parametrize("control", [
    {"fires_on": [{"k": 1}], "passes_on": [[{}, {}]]},
    {"fires_on": [[{}, {}]], "passes_on": [[{}]]},
    {"fires_on": [[{}, {}, {}]], "passes_on": [[{}, {}]]},
])
def test_a_pair_check_whose_control_is_not_a_pair_is_refused(control, tmp_path):
    """A control written as a single document would be evaluated against a pair whose two arms
    are both missing, where every rule answers false. The check would pass its negative control
    because it examined nothing, which is exactly what the controls exist to rule out.
    """
    _write(tmp_path, "x.json", _minimal(arity="pair", control=control))
    with pytest.raises(CheckError, match="two-element"):
        load_checks(tmp_path)


def test_differs_in_more_than_counts_only_keys_both_sides_record():
    """A key one arm writes and the other does not is a difference in what was RECORDED, which is
    usually two versions of a producer rather than two settings. Counting it would fire on every
    pair of artefacts written months apart.
    """
    rule = {"op": "differs_in_more_than", "paths": ["arm_a.s", "arm_b.s"], "value": 1}
    two_differ = registry.as_pair({"s": {"k": 1, "search": "tpe"}},
                                  {"s": {"k": 3, "search": "random"}})
    one_differs = registry.as_pair({"s": {"k": 1, "search": "tpe"}},
                                   {"s": {"k": 3, "search": "tpe"}})
    only_one_side = registry.as_pair({"s": {"k": 1, "search": "tpe", "seed": 7}},
                                     {"s": {"k": 3}})
    assert evaluate(rule, two_differ)
    assert not evaluate(rule, one_differs)
    assert not evaluate(rule, only_one_side)


@pytest.mark.parametrize("doc", [
    {"arm_a": {"s": "not a mapping"}, "arm_b": {"s": {"k": 1}}},
    {"arm_a": {}, "arm_b": {}},
    {},
])
def test_differs_in_more_than_is_total_like_every_other_operator(doc):
    """No operator may raise on a shape it did not expect: an unfamiliar artefact must become
    "this check does not apply here" rather than a crash.
    """
    assert evaluate({"op": "differs_in_more_than", "paths": ["arm_a.s", "arm_b.s"],
                     "value": 0}, doc) is False


@pytest.mark.parametrize("rule", [
    {"op": "differs_in_more_than", "paths": ["arm_a.s"], "value": 1},
    {"op": "differs_in_more_than", "paths": ["arm_a.s", "arm_b.s"]},
    {"op": "differs_in_more_than", "paths": ["arm_a.s", "arm_b.s"], "value": "one"},
    {"op": "differs_in_more_than", "paths": ["arm_a.s", "arm_b.s"], "value": -1},
])
def test_a_malformed_differs_in_more_than_is_a_loud_refusal(rule):
    with pytest.raises(CheckError):
        evaluate(rule, registry.as_pair({"s": {}}, {"s": {}}))


# ── the split: the ordinary ceiling case, and the serious labelling case ─────────────────────

#: One arm of the 2026-08-03 comparison, in the shape the bake writes. Ceiling 3, applied 1,
#: which is the ORDINARY outcome of a search that was free to use fewer directions.
_CEILING_ABOVE_APPLIED = {
    "model": "Qwen/Qwen3-1.7B", "label": "k3", "max_directions": 3, "num_directions": 1,
    "dir_mode": "per_layer", "post_bake_refusals": 0.04, "refusal_eval": "measure rows 132-331",
}
_CEILING_EQUALS_APPLIED = {**_CEILING_ABOVE_APPLIED, "label": "k1",
                           "max_directions": 1, "num_directions": 1}


def test_the_ordinary_ceiling_case_fires_only_the_notes_half():
    """A search ceiling above the applied count is what a search DOES. Firing a withdrawal on it
    would shout on every normal run, and a check people learn to ignore is worse than no check.
    """
    from senbonzakura_check import check_document

    findings, skipped = check_document(json.loads(json.dumps(_CEILING_ABOVE_APPLIED)), CHECKS)
    fired = {f.check_id: f.severity for f in findings}
    assert fired.get("a-search-ceiling-above-the-count-it-applied") == "notes"
    assert "an-arm-labelled-by-a-setting-it-did-not-apply" not in fired
    assert "an-arm-labelled-by-a-setting-it-did-not-apply" in skipped


def test_the_serious_case_needs_two_arms_and_withdraws_when_it_has_them():
    """THE WITHDRAWN INCIDENT ITSELF. Two arms named by ceilings of 1 and 3 that both applied
    one direction: the same edit under two labels, and the significant difference between them
    came from something nobody had meant to compare.
    """
    from senbonzakura_check import check_pair

    findings, _ = check_pair(json.loads(json.dumps(_CEILING_EQUALS_APPLIED)),
                             json.loads(json.dumps(_CEILING_ABOVE_APPLIED)), CHECKS)
    fired = {f.check_id: f.severity for f in findings}
    assert fired.get("an-arm-labelled-by-a-setting-it-did-not-apply") == "withdraws"


def test_a_comparison_whose_arms_really_did_differ_is_quiet():
    """The half that makes the split worth having. If the ordinary case tripped the withdrawal,
    or an honest comparison did, the split would have bought nothing.
    """
    from senbonzakura_check import check_pair

    honest = {**_CEILING_ABOVE_APPLIED, "max_directions": 3, "num_directions": 3}
    findings, _ = check_pair(json.loads(json.dumps(_CEILING_EQUALS_APPLIED)),
                             json.loads(json.dumps(honest)), CHECKS)
    assert "an-arm-labelled-by-a-setting-it-did-not-apply" not in {f.check_id for f in findings}

    same = json.loads(json.dumps(_CEILING_ABOVE_APPLIED))
    findings, _ = check_pair(same, json.loads(json.dumps(_CEILING_ABOVE_APPLIED)), CHECKS)
    assert "an-arm-labelled-by-a-setting-it-did-not-apply" not in {f.check_id for f in findings}, (
        "two copies of the ordinary case tripped the withdrawal, so the split bought nothing")


def test_checks_run_in_a_stable_order():
    """Two runs on the same input produce the same output, per baseline section 2.1. The loader
    sorts by id so a report's order does not depend on the filesystem's.
    """
    assert [c.id for c in load_checks()] == sorted(c.id for c in load_checks())


def test_the_checker_never_writes_to_what_it_reads(tmp_path):
    """The one unforgivable behaviour: it is pointed at other people's evidence.

    Asserted by mtime and bytes rather than by reading the source, because the thing that would
    break this is a future check needing somewhere to put state.
    """
    artefact = tmp_path / "theirs.json"
    artefact.write_text(json.dumps({"kl": 0.1}), encoding="utf-8")
    before = (artefact.read_bytes(), artefact.stat().st_mtime_ns)

    doc = json.loads(artefact.read_text(encoding="utf-8"))
    run_checks(doc, CHECKS, artefact=str(artefact))

    assert (artefact.read_bytes(), artefact.stat().st_mtime_ns) == before
    assert list(tmp_path.iterdir()) == [artefact], "the checker left something behind"


# ── the checks have to reach an installed wheel ──────────────────────────────────────────────

def test_the_check_files_are_declared_as_package_data():
    """A WHEEL WITHOUT THEM LOADS ZERO CHECKS AND REPORTS EVERY ARTEFACT CLEAN.

    The engine globs `checks/*.json` at runtime, so these are data the code reads and not
    examples sitting beside it. That is exactly the shape of the 0.3.0 defect: a wheel that
    installed, imported and answered `--help` perfectly while `--track default` failed for every
    user, because a generated artefact was missing and nothing checked the glob.

    Declared in the CHECKER's own pyproject since decision Q-29 moved it to its own
    distribution, so this reads that file rather than the abliterator's.
    """
    from pathlib import Path

    from tomlread import tomllib

    root = Path(__file__).resolve().parents[1]
    with (root / "checker" / "pyproject.toml").open("rb") as f:
        cfg = tomllib.load(f)
    globs = cfg["tool"]["setuptools"]["package-data"]["senbonzakura_check"]
    assert "checks/*.json" in globs, (
        "the check registry is not declared as package data, so a built wheel would carry no "
        "checks and would report every artefact clean")


def test_the_check_subpackage_imports_nothing_heavy():
    """The torch-free boundary, asserted at the import graph rather than at install time.

    `roadmap.md` calls a torch-free checker the largest single adoption lever this project has,
    and Q-29 settled that it ships as its own distribution. Nothing under `check/` may reach the
    deep-learning stack, and the failure mode is silent: importing something convenient from the
    abliterator side breaks nothing on any developer machine and only shows up for a user.

    This is the cheap half. The expensive half, resolving the checker's dependencies in a clean
    environment and asserting torch is absent, belongs with the second distribution when it
    exists.
    """
    import ast
    from pathlib import Path

    heavy = {"torch", "transformers", "accelerate", "optuna", "datasets",
             "bitsandbytes", "sentencepiece", "gguf", "numpy", "pyarrow"}
    pkg = Path(registry.__file__).resolve().parent
    offenders = []
    for py in sorted(pkg.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            offenders += [f"{py.name}:{node.lineno} imports {n}" for n in names if n in heavy]
    assert not offenders, (
        "the checker must import with no deep-learning stack at all:\n  " + "\n  ".join(offenders))


# ── dogfooding: the checker, pointed at this project's own published evidence ────────────────

def _published_arms():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "head-to-head" / "results"
    return sorted(root.rglob("*.json"))


@pytest.mark.parametrize("path", _published_arms(), ids=lambda p: p.name)
def test_the_checker_finds_nothing_wrong_with_our_own_published_arms(path):
    """POINTED AT OUR OWN EVIDENCE, which is the only honest way to calibrate a checker.

    These thirty-one artefacts are the 2026-09-10 head-to-head, committed to this repository and
    recomputed by `test_published_head_to_head.py`. If a check fires on one, either the artefact
    has the defect, in which case the published figure needs re-examining, or the check is wrong,
    in which case it would have fired on a stranger's artefact too.

    It found one on the first run: `metric-reported-without-its-estimator` fired on every drift
    arm, because the estimator is recorded in a prose `instrument` sentence under a field name
    the check did not know. That was the check being incomplete rather than the artefacts being
    wrong, and it is why `instrument` is now one of the places it looks.
    """
    from senbonzakura_check import UnknownArtefactError, check_document

    doc = json.loads(path.read_text(encoding="utf-8"))
    try:
        findings, _ = check_document(doc, CHECKS)
    except UnknownArtefactError:
        # The run summary is an index of which arms ran, not a measurement, and refusing it is
        # the correct answer rather than a gap. `test_check_adapters.py` asserts that directly.
        pytest.skip(f"{path.name} is not a measurement artefact")
    assert not findings, (
        f"{path.name}: " + "; ".join(f"{f.check_id} ({f.title})" for f in findings))


def test_the_dogfooding_above_is_not_vacuous():
    """A checker that cannot fire reports everything clean, and that is what "0 findings" looks
    like from the outside.

    THIS IS NOT HYPOTHETICAL. The first version of the estimator check used `falsy` on
    `instrument`, which is true only when the field is PRESENT and empty; on an artefact with no
    `instrument` at all it answered false, the enclosing `all_of` could never hold, and the check
    reported thirty-one published artefacts clean. That looked exactly like success.
    """
    findings, _ = run_checks({"metrics": {"kl": {"value": 0.1}}}, CHECKS)
    assert [f.check_id for f in findings] == ["metric-reported-without-its-estimator"], (
        "a bare metric with no provenance anywhere must still be caught")


# ── applicability: a check that examines a document must be able to speak about it ───────────
#
# THE CLASS THIS GUARDS, and it has now been found in four separate spellings. A check reports
# one of three outcomes: it fires, it passes, or it skips. `skipped` and `passed` are the whole
# point of the design, because a check that examined an artefact and stayed quiet is evidence and
# a check that could never have spoken is not. When `applies_to` claims a document the rule cannot
# possibly speak about, the report says PASSED and a reader counting examined checks is being told
# a question was asked. Three checks were fixed for it on the morning of 2026-09-21 and four more
# were found that afternoon, by running the checker over three real coherence artefacts recovered
# from the ROG: six checks examined them and only two could have fired.
#
# WHAT THESE TWO DO NOT CATCH, said here rather than discovered as a fifth spelling. They cover
# the two shapes actually found and no more. A rule gated by an `all_of` over a field, rather
# than by the `when` key, is invisible to the second test. A check whose rule simply reads a path
# no artefact in the wild carries is invisible to both, because nothing declares that intent. And
# `absent` on an always-written key is the mirror defect, a check that can never apply at all
# rather than one that always does, which neither test looks for. The general property, "being
# examined implies the check could have fired", is not mechanically decidable from the rule
# alone; the honest move is a corpus of real artefacts and a look at which checks spoke, which is
# how both of these were found in the first place.

def _always_written_keys():
    """The top-level keys the senbonzakura adapter writes on EVERY record, whatever the input.

    Computed from the adapter rather than listed here, so a key added to `normalise` is covered
    the day it is added rather than the day somebody remembers this test exists.
    """
    from senbonzakura_check.adapters import normalise
    return set(normalise({"label": "a", "model": "m", "nll": 1.0}))


def _nodes(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _nodes(value)


def _paired_with_a_value_test(applies_to):
    """Paths whose `present` sits in an `all_of` beside a test of the value itself.

    `present` is true of a key written as null, so on its own it is not a test of anything for a
    key the adapter always writes. Paired with `not_equals null` or `truthy` it is exactly right,
    and that pairing is the convention the checks fixed earlier already use.
    """
    out = set()
    for node in _nodes(applies_to):
        if node.get("op") != "all_of":
            continue
        rules = [r for r in (node.get("rules") or []) if isinstance(r, dict)]
        present = {r.get("path") for r in rules if r.get("op") == "present"}
        # `falsy` AND `equals` ARE NOT EVIDENCE OF A GUARD, removed 2026-09-21 after a reviewer
        # pointed out that this test accepted them. On a key the adapter always writes as null,
        # `present` is true and fixed-path `falsy` is FALSE, so `present AND falsy` is not the
        # vacuous pair; but `present AND equals null` is exactly the vacuous claim this test was
        # written to catch, and it would have been waved through. Only an operator that requires
        # the key to carry something counts.
        valued = {r.get("path") for r in rules
                  if r.get("op") == "truthy"
                  or (r.get("op") == "not_equals" and r.get("value") is None)}
        out |= present & valued
    return out


@pytest.mark.parametrize("check", CHECKS, ids=[c.id for c in CHECKS])
def test_applicability_never_rests_on_a_key_the_adapter_always_writes(check):
    """`present` on such a key is true of every senbonzakura artefact ever produced."""
    always = _always_written_keys()
    guarded = _paired_with_a_value_test(check.applies_to)
    unguarded = sorted({node.get("path") for node in _nodes(check.applies_to)
                        if node.get("op") == "present"
                        and node.get("path") in always
                        and node.get("path") not in guarded})
    assert not unguarded, (
        f"{check.id}: `applies_to` tests {unguarded} with `present`, and the senbonzakura "
        f"adapter writes those keys on every record, as null when the artefact says nothing. So "
        f"this check claims every artefact this project has ever produced and then has nothing "
        f"to read, which the report prints as PASSED. Pair it with `not_equals` null, or test "
        f"the value with `truthy`.")


@pytest.mark.parametrize("check", CHECKS, ids=[c.id for c in CHECKS])
def test_a_rule_gated_on_a_condition_is_only_applied_where_that_condition_can_hold(check):
    """A `when` gate inside the rule has to be mirrored in `applies_to`.

    `impossible-proportion-reported` scores values against 0 and 1 `when` a metric's units say
    `proportion`, and applied to any document carrying a metrics block at all. Handed a coherence
    figure in nats-per-token it examined it, could not fire by construction, and reported clean.
    Its own `false_positive` sentence said the rule could not fire on such an artefact; the
    applicability did not agree, and the applicability is what the report counts.
    """
    gates = [node["when"] for node in _nodes(check.rule)
             if isinstance(node.get("when"), dict)]
    if not gates:
        return
    applies = list(_nodes(check.applies_to))
    for gate in gates:
        mirrored = any(gate == sub for node in applies for sub in (node.get("all") or []))
        assert mirrored, (
            f"{check.id}: the rule only speaks when {gate}, but `applies_to` does not require "
            f"it, so every artefact where that condition is false is examined and reported "
            f"clean on a question this check cannot ask. Mirror the gate in `applies_to`.")


def test_a_rate_with_no_denominator_is_skipped_by_the_sample_size_check():
    """The fifth spelling of examined-but-unable-to-speak, and the one the earlier sweep missed.

    `any_outside` steps over an entry whose field is not a number, so a proportion carrying
    `n: null` made the sample-size check APPLY and then come back clean. That is not a corner
    case: the senbonzakura adapter writes `n: null` for every figure it lifts out of an
    abliteration record, and the check's own `false_positive` text called it "a coverage gap
    rather than a pass" while it was a pass.

    Asserted as a skip rather than through `control.passes_on`, because the negative-control test
    deliberately tolerates a skip, so moving the null-denominator case there would have proved
    nothing.
    """
    from senbonzakura_check import check_document

    check = next(c for c in CHECKS if c.id == "rate-reported-on-a-sample-too-small-to-carry-it")
    doc = {"label": "arm", "model": "m",
           "metrics": {"refusal_rate": {"metric": "refusal_rate", "value": 0.5,
                                        "units": "proportion", "n": None}}}
    _, skipped = check_document(doc, [check])
    assert check.id in skipped, (
        "a proportion with no denominator was examined by the sample-size check and reported "
        "clean, on a question it has no way to ask")

    with_n = {"label": "arm", "model": "m",
              "metrics": {"refusal_rate": {"metric": "refusal_rate", "value": 0.5,
                                           "units": "proportion", "n": 4}}}
    findings, skipped = check_document(with_n, [check])
    assert check.id not in skipped and [f.check_id for f in findings] == [check.id], (
        "narrowing the applicability has stopped the check firing where it should")


def test_a_present_paired_with_equals_null_is_not_treated_as_a_guarded_test():
    """A guard on the guard above.

    `_paired_with_a_value_test` decides which `present` tests are excused, and it accepted any of
    `truthy`, `falsy`, `equals` and `not_equals` as evidence that the presence test was real. On
    a key the adapter always writes as null, `present AND equals null` is exactly the vacuous
    claim the guard exists to catch, and it would have been waved through. Only an operator that
    requires the key to CARRY something counts.
    """
    vacuous = {"op": "all_of", "rules": [
        {"op": "present", "path": "eval_split"},
        {"op": "equals", "path": "eval_split", "value": None}]}
    assert _paired_with_a_value_test(vacuous) == set(), (
        "`equals null` was accepted as evidence that a `present` test is guarded")

    real = {"op": "all_of", "rules": [
        {"op": "present", "path": "eval_split"},
        {"op": "not_equals", "path": "eval_split", "value": None}]}
    assert _paired_with_a_value_test(real) == {"eval_split"}


# ── the rule that reads a file's own verdict on itself ────────────────────────────────

def test_a_file_that_calls_its_own_figure_invalid_is_refused():
    """FOUND BY THE NOVICE READER, 2026-09-26, and it was their first real result.

    The compass sets `self_invalidated` when the model's verdict was not at the position being
    scored, so the AUC describes whatever token happened to be there. The run said so in capitals.
    The checker, on the file that run had just written, said `nothing found (12 checks did not
    apply)`. Their words: two commands in the same tool looked at one run and disagreed, and the one
    that reads like a verdict was the one that was wrong.
    """
    from senbonzakura_check import check_document

    doc = json.loads(json.dumps(_RECORDS_NOTHING_ELSE))
    doc["self_invalidated"] = True
    findings, _skipped = check_document(doc, CHECKS)
    fired = {f.check_id for f in findings}
    assert "a-figure-its-own-file-calls-invalid" in fired, (
        "an artefact recording that its own headline figure is not a measurement is still reported "
        "clean, which is the sibling defect one field along")


def test_a_file_that_does_not_invalidate_itself_is_not_flagged():
    from senbonzakura_check import check_document

    doc = json.loads(json.dumps(_RECORDS_NOTHING_ELSE))
    doc["self_invalidated"] = False
    findings, _skipped = check_document(doc, CHECKS)
    assert "a-figure-its-own-file-calls-invalid" not in {f.check_id for f in findings}
