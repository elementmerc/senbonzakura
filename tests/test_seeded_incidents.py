# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The seeded corpus of known-bad setups, run through the checker end to end.

WHY THIS IS NOT THE SAME TEST AS THE PER-CHECK CONTROLS

Every check ships `control.fires_on`, and `test_check_registry.py` turns each one into a case.
Those documents are written in the NORMALISED vocabulary and are handed straight to the rule
evaluator, which is the right way to test a rule and tests nothing about the path an artefact
actually takes. A check can pass every one of its own controls and never fire in the field,
because the adapter did not carry the field it reads, or refused the document outright. That is
not hypothetical: a hostile reviewer on 2026-09-17 fed the checker an artefact carrying its own
short-budget warning and got `nothing found`, and the reason was that the adapter was not
carrying the field into the normalised document, so no check could have read it.

So these fixtures are whole artefacts, in the shape a producer writes them, and they go through
`check_document`, which means through detection, adaptation and normalisation first.

TWO HALVES, AND THE SECOND ONE IS THE MEASUREMENT

`fixtures/ours/` reconstructs this project's own incidents. The checks were written against them,
so a catch here proves a check works and says nothing about whether the check set is any good.

`fixtures/held-out/` holds incidents from outside this project, chosen after the checks were
frozen. A MISS THERE IS A RESULT AND IS REPORTED AS ONE. Nothing in that directory may cause a
check to be written or widened, because fitting the checks to it converts the only honest
measurement of the set into a self-assessment. That is loophole 2 of the v0.8 rung, and it is
why the held-out half asserts only that the checker ran and said something legible, never that
it caught anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from senbonzakura_check import UnknownArtefactError, check_document
from senbonzakura_check.registry import load_checks

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CHECKS = load_checks()

#: The provenance block every fixture carries. Ignored by the checker; read by this file, because
#: an artefact taken from a record and one somebody invented to exercise a rule are different
#: kinds of evidence and a corpus that cannot say which is which will be quoted as though it were
#: all the first kind.
REQUIRED_PROVENANCE = ("incident", "documented_at", "kind", "inferred", "not_reproduced")
KINDS = ("reconstructed", "invented")


def _fixtures(half):
    return sorted((FIXTURES / half).glob("*.json"))


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", _fixtures("ours") + _fixtures("held-out"),
                         ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_fixture_says_where_it_came_from(path):
    """A fixture with no provenance is a claim about the world with nothing behind it.

    This project has published a p-value that came out of a fixture built to exercise a report,
    which is what that failure looks like when nobody has to write down which kind of artefact
    they are holding.
    """
    prov = _load(path).get("_provenance")
    assert isinstance(prov, dict), f"{path.name} carries no `_provenance` block"
    missing = [f for f in REQUIRED_PROVENANCE if not prov.get(f)]
    assert not missing, f"{path.name}: `_provenance` is missing {missing}"
    assert prov["kind"] in KINDS, (
        f"{path.name}: `kind` is {prov['kind']!r}; it has to be one of {KINDS}, because an "
        f"artefact taken from the record and one built to exercise a rule are different evidence")


@pytest.mark.parametrize("path", _fixtures("ours"), ids=lambda p: p.stem)
def test_the_checker_catches_each_incident_it_was_written_against(path):
    """THE EASY HALF, and it is still necessary: a check nobody has watched fire in the field is
    a check nobody has tested in the field.

    The fixture names which checks are supposed to fire. Asserted as a subset rather than as
    equality, because an artefact reconstructed faithfully may carry a second real defect, and
    forbidding that would push the corpus towards artefacts that are wrong in exactly one way,
    which no real artefact is.
    """
    doc = _load(path)
    expect = set(doc["_provenance"].get("expect") or ())
    assert expect, f"{path.name}: `_provenance.expect` names no check, so this asserts nothing"
    findings, skipped = check_document(doc, CHECKS)
    fired = {f.check_id for f in findings}
    assert expect <= fired, (
        f"{path.name}: expected {sorted(expect - fired)} to fire and they did not. Fired: "
        f"{sorted(fired)}. Skipped as not applicable: {sorted(skipped)}. A check that passes its "
        f"own control and misses the artefact it was written for is reading a field the adapter "
        f"does not carry.")


@pytest.mark.parametrize("path", _fixtures("ours"), ids=lambda p: p.stem)
def test_every_finding_on_a_seeded_incident_carries_its_citation(path):
    """The report prints the incident and the caveat beside the finding, so both have to travel
    with it. A finding with no citation is an opinion, and a checker that reports opinions is
    uninstalled once and never again.
    """
    findings, _ = check_document(_load(path), CHECKS)
    for f in findings:
        assert len(f.incident) > 60, f"{f.check_id} arrived with no usable incident"
        assert len(f.false_positive) > 80, f"{f.check_id} arrived with no usable caveat"


@pytest.mark.parametrize("path", _fixtures("held-out"), ids=lambda p: p.stem)
def test_a_held_out_incident_is_at_least_read_rather_than_refused(path):
    """THE HONEST HALF, and note what it does NOT assert.

    It does not assert that anything fires. The point of a held-out incident is to measure the
    check set against a defect nobody wrote a check for, and a test that demanded a catch would
    force the next person to widen a rule until it passed, which is the fitting this half exists
    to avoid.

    What it does assert is that the artefact is recognised and checked rather than refused, so a
    reported miss is a genuine miss and not an adapter shrugging. Whether anything fired is
    recorded in the fixture's own `_provenance.result` and in the review artefact, as a finding
    about the checker rather than as a passing test.
    """
    doc = _load(path)
    try:
        findings, skipped = check_document(doc, CHECKS)
    except UnknownArtefactError as e:
        pytest.fail(
            f"{path.name} was not recognised by any adapter, so this fixture measures nothing "
            f"about the checks: {e}")
    assert len(skipped) + len(findings) >= 0
    assert doc["_provenance"].get("result"), (
        f"{path.name}: `_provenance.result` has to record what the checker actually did with "
        f"this artefact, including 'nothing fired'. A miss written down is the result; a miss "
        f"left unwritten becomes a catch by the time anybody reads the corpus.")
