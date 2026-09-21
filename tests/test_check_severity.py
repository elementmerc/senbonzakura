# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Findings are ranked by how far they could move the number, and severity is where that lives.

THE RUNG ASKS FOR IT AND THE REPORT DID NOT DO IT. Until 2026-09-21 the command printed findings
in whatever order the loader produced, which is alphabetical by check id. Alphabetical is stable
and reproducible, which is exactly why nobody noticed it was also meaningless: a reader skimming
the first finding was reading the one whose name sorted earliest, not the one that mattered most.

SEVERITY IS NOT CONFIDENCE, and separating them is the point rather than a nicety. Confidence is
how likely the finding is to be right; severity is what it costs if it is. A high-confidence note
about two fields that could be read for each other and a medium-confidence finding that withdraws
a published figure are not the same news, and one number cannot carry both claims.

The field is OPTIONAL in the engine, because making it mandatory would refuse every check file
written before it existed, including a contributor's. The discipline lives here instead: every
check THIS PROJECT ships declares one explicitly, and the test below fails if one does not.
"""
import pytest
from senbonzakura_check.registry import (
    CONFIDENCES,
    SEVERITIES,
    CheckError,
    Finding,
    load_checks,
    run_checks,
    weight,
)
from senbonzakura_check.registry import Check as _Check

CHECKS = load_checks()


@pytest.mark.parametrize("check", CHECKS, ids=[c.id for c in CHECKS])
def test_every_shipped_check_declares_its_severity_explicitly(check):
    """The engine has a default so a foreign check file still loads. Ours do not lean on it:
    a default severity on our own check is a judgement nobody made, sitting in a report line
    that reads as though somebody had.
    """
    import json

    assert check.source is not None
    raw = json.loads(check.source.read_text(encoding="utf-8"))
    assert "severity" in raw, (
        f"{check.id} does not declare a severity, so it inherits the default and the report "
        f"ranks it on a judgement nobody made. Add one of {', '.join(SEVERITIES)}.")
    assert raw["severity"] in SEVERITIES


def test_the_two_axes_are_not_the_same_axis():
    """A check set where severity is a relabelling of confidence has added a column and no
    information, and the ranking it produces is the one it already had.
    """
    pairs = {(c.confidence, c.severity) for c in CHECKS}
    by_confidence = {}
    for confidence, severity in pairs:
        by_confidence.setdefault(confidence, set()).add(severity)
    assert any(len(s) > 1 for s in by_confidence.values()), (
        f"every confidence level maps to exactly one severity across the shipped checks "
        f"({sorted(pairs)}), so severity carries nothing confidence did not already say")


def _finding(**over):
    base = {"check_id": "c", "title": "t", "detects": "d", "incident": "i", "remedy": "r",
            "confidence": "high", "false_positive": "fp", "severity": "qualifies"}
    base.update(over)
    return Finding(**base)


def test_findings_come_back_worst_first():
    """The ordering contract, asserted on the function that produces it rather than on a
    rendered string, so a change to the report's wording cannot quietly change the ranking.
    """
    findings = [
        _finding(check_id="a-note", severity="notes", confidence="high"),
        _finding(check_id="a-withdrawal", severity="withdraws", confidence="medium"),
        _finding(check_id="a-qualifier", severity="qualifies", confidence="high"),
    ]
    assert [f.check_id for f in sorted(findings, key=weight)] == [
        "a-withdrawal", "a-qualifier", "a-note"], (
        "a finding that withdraws a figure has to arrive before one that only qualifies it, "
        "even when the qualifier is the more confident of the two")


def test_confidence_breaks_a_tie_and_the_id_breaks_that():
    """Total and deterministic, so two runs over the same input produce the same report."""
    findings = [
        _finding(check_id="z", severity="withdraws", confidence="high"),
        _finding(check_id="a", severity="withdraws", confidence="high"),
        _finding(check_id="m", severity="withdraws", confidence="medium"),
    ]
    assert [f.check_id for f in sorted(findings, key=weight)] == ["a", "z", "m"]


def test_an_unknown_severity_sorts_last_rather_than_raising():
    """A Finding can be built by hand, by a test or by somebody embedding the engine, and a
    ranking function that raised would take down a report over a label it could have ignored.
    The unknown one goes last, which is the honest place for a claim nobody here can weigh.
    """
    odd = _finding(check_id="odd", severity="whatever", confidence="unheard-of")
    worst = _finding(check_id="worst", severity="withdraws", confidence="high")
    assert [f.check_id for f in sorted([odd, worst], key=weight)] == ["worst", "odd"]


def test_run_checks_returns_them_already_ranked():
    """Ranked by the engine rather than by each caller, because the JSON output and the human
    output are two callers and the one that was not updated is the one somebody automates.
    """
    def _c(cid, severity):
        return _Check(id=cid, title="t", detects="d", incident="i", remedy="r",
                      confidence="high", false_positive="fp", severity=severity,
                      applies_to={"op": "always"}, rule={"op": "always"},
                      control={"fires_on": [{}], "passes_on": [{}]})

    findings, _ = run_checks({}, [_c("a-note", "notes"), _c("b-withdrawal", "withdraws")])
    assert [f.check_id for f in findings] == ["b-withdrawal", "a-note"]


def test_an_unknown_severity_in_a_check_file_is_refused(tmp_path):
    """A typo in a check file has to stop the run. Silently defaulting it would rank the check
    on a judgement its author did not make, in a report that shows the ranking as though
    somebody had.
    """
    import json

    doc = {"id": "x", "title": "t", "detects": "d", "incident": "i", "remedy": "r",
           "confidence": "high", "severity": "critical", "false_positive": "fp",
           "applies_to": {"op": "always"}, "rule": {"op": "never"},
           "control": {"fires_on": [{}], "passes_on": [{}]}}
    (tmp_path / "x.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(CheckError, match="severity"):
        load_checks(tmp_path)


def test_a_check_file_without_a_severity_still_loads(tmp_path):
    """THE COMPATIBILITY PROMISE, asserted rather than described. A contribution format that
    invalidates existing contributions is not a contribution format, and a contributor's check
    written last month has to keep working.
    """
    import json

    doc = {"id": "x", "title": "t", "detects": "d", "incident": "i", "remedy": "r",
           "confidence": "high", "false_positive": "fp",
           "applies_to": {"op": "always"}, "rule": {"op": "never"},
           "control": {"fires_on": [{}], "passes_on": [{}]}}
    (tmp_path / "x.json").write_text(json.dumps(doc), encoding="utf-8")
    loaded = load_checks(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].severity in SEVERITIES


@pytest.mark.parametrize("severity", SEVERITIES)
def test_the_report_has_a_sentence_for_every_severity(severity):
    """The label alone tells a reader who has not read the documentation nothing, and the
    person reading a finding is exactly the person who has not read the documentation.
    """
    from senbonzakura_check.cli import _SEVERITY_SENTENCE

    assert len(_SEVERITY_SENTENCE[severity]) > 25


@pytest.mark.parametrize("confidence", CONFIDENCES)
def test_the_report_still_says_how_much_to_believe_it(confidence):
    """Severity was added beside confidence, not instead of it. Dropping confidence from the
    line would leave a reader with the cost and no idea of the odds.
    """
    import io

    from senbonzakura_check import cli

    out = io.StringIO()
    cli._render("run.json", [_finding(confidence=confidence, severity="withdraws")], [], None, out)
    rendered = out.getvalue()
    assert f"{confidence} confidence" in rendered
    assert "CANNOT BE QUOTED" in rendered
