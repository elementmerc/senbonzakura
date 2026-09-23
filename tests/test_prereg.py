# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The pre-registration format, and the four this project wrote before the format existed.

The rung sets its own acceptance test and it is the right one: *"the pre-registrations this
project has already written validate against it or are amended in the same commit"*, and *"if the
format cannot express what was already written by hand, the format is wrong."* So the last test
in this file is not a formality. It is the design being judged against four documents that were
written with no schema in mind, by somebody who did not know a schema was coming.

Every validation rule has a test that makes it fire AND a test that it stays quiet on a clean
document, because a rule that cannot fire is decoration and a rule that always fires gets removed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from senbonzakura import prereg

ROOT = Path(__file__).resolve().parent.parent
PREREG_DIR = ROOT / "private" / "plans"


def _clean(**over):
    """A pre-registration with nothing wrong with it, as the baseline every test perturbs."""
    doc = {
        "title": "Does the projection help the search?",
        "date": "2026-09-21",
        "question": "Does good-orthogonalisation change what the search converges on?",
        "hypothesis": "The projected form reaches a lower refusal rate at equal KL.",
        "falsified_by": "The raw arm reaches an equal or lower refusal rate at equal or lower KL.",
        "primary": {
            "name": "projected against raw",
            "metric": "refusal_rate",
            "instrument": "senbonzakura score, keyword ruler",
            "partition": "measurement",
            "decision_rule": "a gap larger than the seed spread, on five seeds",
        },
        "threats": [
            "Both arms could land on different dir_mode, making this a comparison of searches.",
            "A budget below the visibility floor reads every refusal rate low.",
        ],
    }
    doc.update(over)
    return doc


def _markdown(doc, prose="# A pre-registration\n\nSome argument.\n"):
    return prose + "\n```prereg\n" + json.dumps(doc, indent=2) + "\n```\n"


# ── extracting the block ─────────────────────────────────────────────────────────
def test_a_file_with_no_block_is_not_a_pre_registration():
    with pytest.raises(prereg.PreregError, match="no ```prereg block"):
        prereg.extract("# Just prose\n\nNothing machine readable here.\n")


def test_two_blocks_are_refused_rather_than_merged():
    """Two blocks are two documents disagreeing about what was promised."""
    text = _markdown(_clean()) + _markdown(_clean(title="Something else"))
    with pytest.raises(prereg.PreregError, match="2 ```prereg blocks"):
        prereg.extract(text)


def test_a_block_that_is_not_json_says_so():
    with pytest.raises(prereg.PreregError, match="not valid JSON"):
        prereg.extract("```prereg\n{ not json\n```\n")


def test_a_block_that_is_not_an_object_is_refused():
    with pytest.raises(prereg.PreregError, match="list"):
        prereg.extract("```prereg\n[1, 2]\n```\n")


def test_the_prose_around_the_block_is_left_alone():
    """The prose IS the pre-registration. If this format ever starts rewriting it, it has
    stopped being a handle and become a replacement.
    """
    doc = prereg.extract(_markdown(_clean(), prose="# Title\n\nA long argument.\n"))
    assert doc["title"] == "Does the projection help the search?"


# ── the rules, each fired and each left quiet ────────────────────────────────────
def test_a_complete_pre_registration_has_nothing_to_report():
    assert prereg.validate(_clean()) == []


@pytest.mark.parametrize("field", sorted(prereg.REQUIRED))
def test_every_required_field_is_actually_required(field):
    """Parametrised over the declared set, so adding a field to REQUIRED without meaning it
    fails here rather than quietly widening what the format demands.
    """
    doc = _clean()
    del doc[field]
    findings = prereg.validate(doc)
    assert any(s == prereg.REFUSED and f"`{field}`" in m for s, m in findings), findings


def test_two_primaries_are_refused_because_two_chances_is_not_one_claim():
    """Q-12. A study with two primaries reports whichever worked."""
    doc = _clean(primary=[_clean()["primary"], _clean()["primary"]])
    findings = prereg.validate(doc)
    assert any(s == prereg.REFUSED and "Q-12" in m for s, m in findings), findings


@pytest.mark.parametrize("field", sorted(prereg.COMPARISON_FIELDS))
def test_a_comparison_missing_any_of_its_fields_is_reported(field):
    primary = _clean()["primary"]
    del primary[field]
    findings = prereg.validate(_clean(primary=primary))
    assert any(f"`{field}`" in m for _, m in findings), findings


def test_a_rate_with_no_partition_is_the_two_thousand_and_sixteen_defect():
    """Named separately from the parametrised test above because it is the one with an incident
    behind it: on 2026-08-16 a 0.0% refusal rate travelled as measured when it had been scored
    on the selection partition.
    """
    primary = _clean()["primary"]
    primary["partition"] = ""
    findings = prereg.validate(_clean(primary=primary))
    assert any("2026-08-16" in m for _, m in findings), findings


def test_a_date_that_is_not_a_date_is_reported():
    findings = prereg.validate(_clean(date="last Tuesday"))
    assert any(s == prereg.FINDING and "ISO 8601" in m for s, m in findings), findings


def test_secondaries_are_allowed_and_are_not_treated_as_claims():
    """A secondary with no decision rule is fine: it is already declared as not the claim."""
    sec = {"name": "coherence", "metric": "nll", "instrument": "senbonzakura coherence",
           "partition": "fixed-passage"}
    assert prereg.validate(_clean(secondaries=[sec])) == []


# ── amendments, which are the whole reason the format is machine-checkable ───────
def test_an_amendment_that_does_not_say_when_it_was_made_is_reported():
    doc = _clean(amendments=[{"what": "added a third arm", "before_any_result": True}])
    findings = prereg.validate(doc)
    assert any("`date`" in m for _, m in findings), findings


def test_an_amendment_that_does_not_declare_whether_a_result_existed_is_reported():
    """THE RULE THE FORMAT EXISTS FOR. The same words before and after a number are different
    documents, and only this declaration separates them.
    """
    doc = _clean(amendments=[{"date": "2026-08-15", "what": "added a third arm"}])
    findings = prereg.validate(doc)
    assert any("before_any_result" in m for _, m in findings), findings


def test_an_amendment_made_before_any_result_is_clean():
    doc = _clean(amendments=[
        {"date": "2026-08-15", "what": "added a third arm", "before_any_result": True}])
    assert prereg.validate(doc) == []


def test_an_amendment_made_after_a_result_is_noted_rather_than_hidden():
    """Declaring it is honest and it is still not a pre-registration of that change. The finding
    is a NOTE, not a refusal: the document is fine, the CLAIM resting on the amendment is
    exploratory and has to be reported that way.
    """
    doc = _clean(amendments=[
        {"date": "2026-09-01", "what": "dropped an arm", "before_any_result": False}])
    findings = prereg.validate(doc)
    assert [s for s, _ in findings] == [prereg.NOTE]
    assert "exploratory" in findings[0][1]


def test_worst_ranks_a_refusal_above_a_finding_above_a_note():
    assert prereg.worst([]) is None
    assert prereg.worst([(prereg.NOTE, "x")]) == prereg.NOTE
    assert prereg.worst([(prereg.NOTE, "x"), (prereg.FINDING, "y")]) == prereg.FINDING
    assert prereg.worst([(prereg.FINDING, "y"), (prereg.REFUSED, "z")]) == prereg.REFUSED


# ── holding a run against what was promised ──────────────────────────────────────
def _run(**over):
    run = {
        "separation_statistic": "cohens-d",
        "refusal_eval": {"partition": "measurement"},
        "provenance": {"senbonzakura": {"version": "0.4.0"}},
    }
    run.update(over)
    return run


def test_a_run_that_did_what_was_promised_reports_nothing():
    doc = _clean()
    doc["primary"]["instrument"] = "cohens-d"
    assert prereg.compare_to_run(doc, _run()) == []


def test_a_run_that_used_a_different_instrument_is_reported():
    doc = _clean()
    doc["primary"]["instrument"] = "welch-ratio"
    findings = prereg.compare_to_run(doc, _run())
    assert any("instrument" in m and "welch-ratio" in m for _, m in findings), findings


def test_a_run_scored_on_a_different_partition_is_reported():
    findings = prereg.compare_to_run(
        _clean(), _run(refusal_eval={"partition": "selection"}))
    assert any("partition" in m and "selection" in m for _, m in findings), findings


def test_a_promise_the_artefact_cannot_answer_is_not_a_pass():
    """THE SILENT-PASS TRAP, and it is the same shape as a gate comparing incomparable numbers.

    A comparison that skips a promise it could not find returns clean and means nothing. So an
    absent field is reported, in the words "was NOT checked", rather than omitted.
    """
    findings = prereg.compare_to_run(_clean(), _run(refusal_eval=None))
    assert any("NOT checked" in m for _, m in findings), findings


def test_a_pre_registration_with_no_usable_primary_cannot_hold_a_run_to_anything():
    findings = prereg.compare_to_run({"primary": None}, _run())
    assert findings and findings[0][0] == prereg.REFUSED


# ── the rung's own acceptance test ───────────────────────────────────────────────
def _written_by_hand():
    return sorted(PREREG_DIR.glob("pre-registration-*.md"))


def test_there_are_pre_registrations_to_check():
    """Without this the parametrised test below silently becomes zero tests.

    `private/` is git-excluded, so a clean checkout legitimately has none. That is a SKIP and not
    a pass: the difference matters, because a suite that reports green having validated nothing is
    the failure this project keeps finding.
    """
    if not PREREG_DIR.is_dir():
        pytest.skip("no private/plans in this checkout, so there is nothing to validate")
    assert _written_by_hand(), (
        "private/plans exists and holds no pre-registration-*.md. The acceptance test below "
        "would then run zero cases and report green.")


# `ids` MUST SURVIVE AN EMPTY PARAMETER SET, and that is not a style point. `private/` is
# git-excluded, so on any clean checkout `_written_by_hand()` is empty; pytest then generates one
# placeholder case holding its `NOTSET` sentinel and calls this function on it. `NOTSET.stem`
# raises, and the raise happens at COLLECTION, which does not skip the test, it kills the whole
# session: 4,578 selected tests never ran and the job reported an error. The sibling test above
# handles the same emptiness by skipping, which is why nothing here looked wrong locally, where
# `private/plans` exists and this path is never taken. One spelling of the condition was covered
# and the other took down the run.
def _prereg_id(path):
    return getattr(path, "stem", "none")


@pytest.mark.parametrize("path", _written_by_hand(), ids=_prereg_id)
def test_a_pre_registration_written_before_the_format_existed_still_validates(path):
    """THE DESIGN UNDER TEST, not the documents.

    These four were written between 2026-07-30 and 2026-09-13 with no schema in mind. If the
    format cannot express what they already say, the rung's own words are that the format is
    wrong, and the fix is here rather than in them.
    """
    doc = prereg.load(path)
    findings = prereg.validate(doc)
    refused = [m for s, m in findings if s == prereg.REFUSED]
    assert not refused, (
        f"{path.name} cannot be expressed in this format: {refused}. Per the v0.9 rung, that "
        f"means the FORMAT is wrong, not the pre-registration.")


def test_the_case_id_survives_the_empty_parameter_set_a_clean_checkout_produces():
    """The regression that took a whole CI job down, asserted directly.

    `private/` is git-excluded, so on a clean checkout the parameter list above is empty. pytest
    then builds one placeholder case holding `NOTSET` and passes it to the id function. A
    `p.stem` there raises during COLLECTION, which is not a skipped test: it ends the session, so
    4,578 selected tests never ran and the job reported an error rather than a failure.

    Asserted against pytest's own sentinel rather than against `None`, because the defect was
    specifically that the sentinel is not path-shaped and not falsy in the way a hand-written
    stand-in would be.
    """
    from _pytest.mark.structures import NOTSET

    assert _prereg_id(NOTSET) == "none"
    assert _prereg_id(PREREG_DIR / "pre-registration-example.md") == "pre-registration-example"


# ── the paths CI had never executed ──────────────────────────────────────────────
#
# WHY THESE ARE GROUPED AND NAMED THAT WAY. Everything below was uncovered on every runner, and
# for two different reasons that look the same in a coverage report. `load` was exercised only by
# the parametrised acceptance test above, which reads `private/plans`; `private/` is git-excluded,
# so on a runner those cases SKIP and the loader this format depends on had never once been
# called outside this developer's machine. The rest are refusal paths for malformed documents,
# which nothing had ever handed it. A validator whose error branches are untested is a validator
# that has only ever been shown documents that were already fine.

def test_a_pre_registration_is_read_from_a_path(tmp_path):
    """The loader, on a file, with no dependency on a git-excluded directory existing."""
    path = tmp_path / "pre-registration-example.md"
    path.write_text(_markdown(_clean()), encoding="utf-8")
    doc = prereg.load(path)
    assert doc["title"] == _clean()["title"]
    assert prereg.validate(doc) == []


def test_a_primary_that_is_not_an_object_is_refused_rather_than_read_field_by_field():
    findings = prereg.validate(_clean(primary="refusal goes down"))
    assert any(s == prereg.REFUSED and "has to be an object" in m for s, m in findings), findings


def test_threats_written_as_prose_rather_than_a_list_is_reported():
    """A single string is the natural mistake and it silently satisfies "carries threats" for
    anything doing a truthiness check, which is why it is called out rather than accepted.
    """
    findings = prereg.validate(_clean(threats="the budget could be too low"))
    assert any(s == prereg.FINDING and "`threats`" in m for s, m in findings), findings


def test_one_short_threat_is_noted_because_the_section_is_the_useful_one():
    findings = prereg.validate(_clean(threats=["seeds"]))
    assert any(s == prereg.NOTE and "`threats`" in m for s, m in findings), findings


def test_amendments_written_as_something_other_than_a_list_is_reported():
    findings = prereg.validate(_clean(amendments={"date": "2026-08-15"}))
    assert any(s == prereg.FINDING and "list of objects" in m for s, m in findings), findings


def test_an_amendment_that_is_not_an_object_is_reported_and_the_rest_still_checked():
    """The `continue` matters: one malformed entry must not stop the entries after it being
    read, or a bad first amendment hides every later one.
    """
    doc = _clean(amendments=["added an arm",
                             {"date": "2026-08-15", "what": "dropped an arm"}])
    findings = prereg.validate(doc)
    assert any("has to be an object" in m for _, m in findings), findings
    assert any("before_any_result" in m for _, m in findings), (
        "the amendment after the malformed one was never examined")


def test_an_amendment_that_does_not_say_what_changed_is_reported():
    doc = _clean(amendments=[{"date": "2026-08-15", "before_any_result": True}])
    findings = prereg.validate(doc)
    assert any("does not say what changed" in m for _, m in findings), findings
