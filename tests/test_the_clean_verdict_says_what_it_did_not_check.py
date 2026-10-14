# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A clean verdict must mean that the checks which applied found nothing, and say what did not apply.

THE FAILURE THIS EXISTS TO CATCH: a file that six of twenty-one checks examined printed
"nothing found" beside a clean-looking summary. "Nothing found" and "nothing looked" render the same
way on the screen, and a reader who hopes for the first will take the second. Loophole 7 of the v0.8
plan names this as the most damaging output a checker can give, because it is a clean bill of health
on a file nobody read.

WHAT THIS PINS: the floor between "clean" and "inconclusive", the wording that carries it, the list
of checks that did not examine a file, and the JSON fields a CI consumer branches on. It also pins
the silence that a genuinely clean verdict is allowed, because a test that only ever sees the loud
path cannot tell a working quiet mode from a broken one.
"""
from __future__ import annotations

import io
import json

import pytest
from senbonzakura_check import cli, normalise
from senbonzakura_check.registry import Finding, load_checks, run_checks

CHECKS = load_checks()
SINGLE_FILE = [c for c in CHECKS if c.arity != "pair"]
PAIR_COUNT = len(CHECKS) - len(SINGLE_FILE)

#: The lm-evaluation-harness result that the existing CLI tests use. It is read through the real
#: adapter, so the applied count below is what a user would actually see.
GOOD = {"version": 2, "status": "success",
        "eval": {"task": "t", "model": "m"},
        "results": {"total_samples": 200, "completed_samples": 200,
                    "scores": [{"name": "s", "scorer": "choice", "scored_samples": 200,
                                "metrics": {"accuracy": {"name": "accuracy", "value": 0.5}}}]}}


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _run(argv):
    out = io.StringIO()
    code = cli.main(argv, out=out)
    return code, out.getvalue()


def _flat(text):
    """The words of the output, with the terminal's line breaks removed."""
    return " ".join(text.split())


def _finding():
    return Finding(check_id="x", title="t", detects="d", incident="i", remedy="r",
                   confidence="high", false_positive="f", severity="qualifies", artefact="a")


# ── the verdict, as a pure function ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(("applied", "expected"), [
    (0, "inconclusive"),
    (10, "inconclusive"),   # 10 of 21 is under half, so 10.5 is the line
    (11, "clean"),
    (21, "clean"),
])
def test_the_floor_is_half_the_single_file_checks(applied, expected):
    assert cli._verdict([], None, True, applied, 21) == expected


def test_no_applicable_checks_is_never_clean():
    """A zero denominator means nothing could have looked, so silence is not evidence of anything."""
    assert cli._verdict([], None, True, 0, 0) == "inconclusive"


def test_a_finding_outranks_the_floor():
    assert cli._verdict([_finding()], None, True, 1, 21) == "finding"


@pytest.mark.parametrize(("named", "expected"), [(True, "unchecked"), (False, "not_a_result")])
def test_a_file_that_was_never_read_is_not_a_verdict_on_its_contents(named, expected):
    assert cli._verdict([], "could not read it", named, 0, 21) == expected


# ── the wording a reader sees ─────────────────────────────────────────────────────────────────

def test_a_thin_file_is_inconclusive_and_says_so_in_the_output(tmp_path):
    code, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    flat = _flat(text)
    assert "INCONCLUSIVE, NOT CLEAN" in flat
    assert "✓" not in text, "a file that was mostly unexamined must not get the clean tick"
    assert "Nothing here says the file is right" in flat
    assert code == 0, "the exit status is unchanged by design; see the decision note in the report"


def test_the_not_examined_list_names_exactly_the_checks_that_did_not_apply(tmp_path):
    """The reader is shown which checks said nothing, so the list has to be the true one.

    The expected list is computed from the registry against the same document, not typed in, so a
    new check that applies to this file moves the expectation with it rather than silently passing.
    """
    _, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    flat = _flat(text)
    findings, skipped = run_checks(normalise(GOOD), CHECKS)
    expected = sorted(c.id for c in SINGLE_FILE if c.id in set(skipped))
    assert expected, "this fixture should leave some single-file checks unexamined"
    assert f"Did not examine ({len(expected)}):" in flat
    listed = flat.split("Did not examine (")[1].split("):", 1)[1].split("Nothing here")[0]
    shown = sorted(name.strip() for name in listed.split(",") if name.strip())
    assert shown == expected


def test_pair_checks_are_not_listed_as_unexamined_because_they_never_apply_to_one_file(tmp_path):
    _, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    pair_ids = [c.id for c in CHECKS if c.arity == "pair"]
    assert pair_ids
    for pid in pair_ids:
        assert pid not in _flat(text).split("Did not examine (")[-1].split("Nothing here")[0]


def test_the_count_line_reconciles_with_the_footer(tmp_path):
    _, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    flat = _flat(text)
    assert f"single-file checks applied, {PAIR_COUNT} pair checks not run" in flat
    assert f"{len(SINGLE_FILE)} single-file" in flat
    assert f"{len(CHECKS)} checks available" in flat


def test_a_clean_verdict_prints_the_tick_and_the_count(tmp_path):
    """The clean path, rendered directly with every single-file check applied."""
    out = io.StringIO()
    cli._render("a.json", [], None, out, applied=len(SINGLE_FILE), denominator=len(SINGLE_FILE),
                not_examined=[], pairs=PAIR_COUNT)
    text = out.getvalue()
    assert "✓  a.json" in text
    expected = (f"nothing found ({len(SINGLE_FILE)} of {len(SINGLE_FILE)} single-file checks "
                f"applied, {PAIR_COUNT} pair checks not run)")
    assert expected in _flat(text)
    assert "INCONCLUSIVE" not in text
    assert "Did not examine" not in text


def test_a_clean_verdict_is_silent_in_quiet_mode():
    """Quiet mode drops clean files. The silence is only allowed where the verdict is clean."""
    out = io.StringIO()
    cli._render("a.json", [], None, out, quiet=True, applied=len(SINGLE_FILE),
                denominator=len(SINGLE_FILE), not_examined=[], pairs=PAIR_COUNT)
    assert out.getvalue() == ""


def test_an_inconclusive_verdict_is_not_silent_in_quiet_mode():
    out = io.StringIO()
    cli._render("a.json", [], None, out, quiet=True, applied=3, denominator=21,
                not_examined=["x"], pairs=PAIR_COUNT)
    assert "~  a.json" in out.getvalue()
    assert "INCONCLUSIVE" in out.getvalue()


def test_the_footer_counts_inconclusive_files_only_when_there_are_some(tmp_path):
    _, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    assert ", 1 inconclusive," in _flat(text)


# ── the machine-readable shape a CI consumer branches on ──────────────────────────────────────

def test_json_carries_the_verdict_in_one_word(tmp_path):
    _, text = _run(["--json", str(_write(tmp_path, "ok.json", GOOD))])
    entry = json.loads(text)[0]
    assert entry["verdict"] == "inconclusive"
    assert entry["applicable"] == len(SINGLE_FILE)
    assert entry["applied"] < entry["applicable"] / 2
    assert entry["not_examined"]
    assert set(entry["not_examined"]) <= {c.id for c in SINGLE_FILE}


def test_json_still_carries_the_old_keys_so_existing_consumers_do_not_break(tmp_path):
    _, text = _run(["--json", str(_write(tmp_path, "ok.json", GOOD))])
    entry = json.loads(text)[0]
    for key in ("artefact", "unchecked", "not_a_result", "applied", "skipped", "findings"):
        assert key in entry, f"the CI action reads `{key}`, and it has to survive this change"


def test_json_reports_a_finding_as_a_finding(tmp_path):
    bad = {"version": 2, "status": "success", "eval": {"task": "t", "model": "m"},
           "results": {"total_samples": 200, "completed_samples": 200,
                       "scores": [{"name": "s", "scorer": None, "scored_samples": 200,
                                   "metrics": {"accuracy": {"name": "accuracy", "value": 0.5}}}]}}
    _, text = _run(["--json", str(_write(tmp_path, "bad.json", bad))])
    assert json.loads(text)[0]["verdict"] == "finding"
