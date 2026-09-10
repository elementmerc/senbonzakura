# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The summary says where each KL figure came from by reading the artefact, not by asserting it.

WHAT WAS WRONG

`_heretic_report` returned `"kl_estimator": "Heretic, its own evaluation"` beside a number that
was ours. Panel finding S1 moved the measurement: the equal-budget selection pass now computes KL
for BOTH tools with one estimator and stamps every row it measured with `kl_source`. The pass was
changed; the label describing the pass was not, and nothing compared them, so the run's own summary
attributed our measurement to the other tool.

That label is the kind of thing a reader trusts precisely because it looks like bookkeeping. On
2026-08-12 the two tools' KL figures came out a hundredfold apart with no honest sentence available
across them, and the whole apparatus of a shared estimator exists to fix that. A provenance line
that is wrong undoes it silently.

THE SECOND HALF, WHICH IS EASIER TO MISS

Fixing Heretic's label leaves the two `kl` columns still measured on DIFFERENT slices: ours from
the abliterator's own coherence slice, cut from the track by the run; theirs from the staged shared
slice. Both run through `senbonzakura.firsttoken`, which makes them look like one instrument. They
are not one exam, and they sit in adjacent columns of the same table. The comparable coherence axis
is `drift`, measured after the fact against one base on rows neither tool has seen, so the columns
that are not comparable have to say so where the reader meets them.
"""
import ast
import json
from pathlib import Path

from senbonzakura import headtohead, headtohead_report

SCRIPT = Path(__file__).resolve().parents[1] / "head-to-head" / "best_of_n_heretic.py"


def _arm(tmp_path, **winner):
    (tmp_path / "best_of_n.json").write_text(
        json.dumps({"trials_ran": 200, "winner": {"refusals": 0.0, "kl": 0.2439, **winner}}),
        encoding="utf-8")
    return tmp_path


# ── the constant, held against the script that actually writes it ────────────────────

def test_the_stamp_matches_the_script_that_writes_it():
    """A duplicated constant is only safe while something compares the copies.

    `best_of_n_heretic.py` imports `heretic`, which exists only inside the sealed image, so it
    cannot be imported here and the value has to be duplicated. Read from the syntax tree rather
    than matched with a pattern, for the reason the sibling argv test gives.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    found[t.id] = node.value.value
    assert "KL_SOURCE_MEASURED" in found, "the script no longer defines the stamp"
    assert found["KL_SOURCE_MEASURED"] == headtohead.KL_SOURCE_MEASURED, (
        f"the script stamps {found['KL_SOURCE_MEASURED']!r} and the reporter looks for "
        f"{headtohead.KL_SOURCE_MEASURED!r}, so every measured figure would be reported as "
        f"unlabelled")


# ── reading the provenance rather than asserting it ──────────────────────────────────

def test_a_figure_the_pass_measured_is_attributed_to_our_estimator(tmp_path):
    """THE DEFECT. The number is ours and the label said it was theirs."""
    got = headtohead._heretic_report(_arm(tmp_path, kl_source=headtohead.KL_SOURCE_MEASURED))
    assert got["kl"] == 0.2439
    assert "senbonzakura.firsttoken" in got["kl_estimator"]
    assert "Heretic, its own evaluation" not in got["kl_estimator"]


def test_a_figure_stamped_by_something_else_is_reported_in_its_own_words(tmp_path):
    """If the pass ever borrows a figure again, the label follows the data rather than the code."""
    got = headtohead._heretic_report(_arm(tmp_path, kl_source="taken from the tool's own study"))
    assert got["kl_estimator"] == "taken from the tool's own study"


def test_an_unstamped_figure_is_called_unlabelled_rather_than_given_an_owner(tmp_path):
    """An unlabelled number given a plausible owner is exactly how this went wrong."""
    got = headtohead._heretic_report(_arm(tmp_path))
    assert "UNRECORDED" in got["kl_estimator"]
    assert "senbonzakura" not in got["kl_estimator"]
    assert "Heretic, its own evaluation" not in got["kl_estimator"]


def test_a_missing_artefact_does_not_invent_a_provenance(tmp_path):
    got = headtohead._heretic_report(tmp_path)
    assert got["kl"] is None
    assert "UNRECORDED" in got["kl_estimator"]


# ── the two columns that are not comparable ──────────────────────────────────────────

def test_our_own_kl_column_says_it_is_not_the_other_tool_s_exam(tmp_path):
    """Both figures go through one estimator on two slices, which reads as one measurement."""
    (tmp_path / "abliteration.json").write_text(
        json.dumps({"post_bake_refusals": 0.0, "post_bake_kl": 0.1609, "trials_ran": 200}),
        encoding="utf-8")
    got = headtohead._senbon_report(tmp_path)
    assert got["kl"] == 0.1609
    assert "NOT comparable" in got["kl_estimator"]
    assert "drift" in got["kl_estimator"], (
        "a caveat that does not name the axis that IS comparable leaves the reader nowhere")


# ── THE REPORTER THAT WAS LEFT BEHIND ────────────────────────────────────────────────
#
# This file's own docstring says it covers "the run's own summary", and until the
# 2026-09-10 panel it did not: the fix landed in `headtohead._heretic_report` and
# `headtohead_report.own_numbers` went on hardcoding the old provenance. That is the
# reporter `head-to-head run` prints at the end and `head-to-head report` prints on
# demand, so the fixed one fed the lesser summary while the reader met the stale one.

def _run_dir(tmp_path, **winner):
    arm = tmp_path / "heretic-seed42"
    arm.mkdir()
    (arm / "best_of_n.json").write_text(
        json.dumps({"trials_ran": 200, "winner": {"refusals": 0.0, "kl": 0.2439, **winner}}),
        encoding="utf-8")
    return tmp_path


def test_the_run_summary_reads_the_stamp_too(tmp_path):
    got = headtohead_report.own_numbers(
        str(_run_dir(tmp_path, kl_source=headtohead.KL_SOURCE_MEASURED)), "heretic", 42)
    assert got["own_kl"] == 0.2439
    assert "senbonzakura.firsttoken" in got["own_kl_estimator"]
    assert "Heretic, its own evaluation" not in got["own_kl_estimator"], (
        "this is the label the reader actually meets at the end of every run")
    assert got["own_kl_is_ours"] is True


def test_the_run_summary_names_an_unstamped_figure_as_unlabelled(tmp_path):
    got = headtohead_report.own_numbers(str(_run_dir(tmp_path)), "heretic", 42)
    assert "UNRECORDED" in got["own_kl_estimator"]
    assert got["own_kl_is_ours"] is False


def test_both_reporters_give_the_same_answer_to_the_same_artefact(tmp_path):
    """The point of the shared helper: this is what would have caught the original miss.

    Neither reporter is asked what it believes. Both are handed the same winner document and the
    two labels are compared with each other, so a fix applied to one and not the other fails here
    whichever one was left behind.
    """
    for i, stamp in enumerate((headtohead.KL_SOURCE_MEASURED, "somewhere else", None)):
        winner = {"kl_source": stamp} if stamp else {}
        a_dir = tmp_path / f"a{i}"
        a_dir.mkdir()
        b_dir = tmp_path / f"b{i}"
        b_dir.mkdir()
        a = headtohead._heretic_report(_arm(a_dir, **winner))
        b = headtohead_report.own_numbers(str(_run_dir(b_dir, **winner)), "heretic", 42)
        assert a["kl_estimator"] == b["own_kl_estimator"], (
            f"the two reporters disagree about a {stamp!r} artefact, which is exactly the drift "
            f"that let one be fixed and the other left")


def test_the_harder_ground_paragraph_is_not_printed_over_our_own_measurement(tmp_path):
    """The prose above the table, not just the per-row label.

    It told the reader that Heretic's figure came from "the harmless prompts it was given, its
    directions included" and that ours was "therefore the harder ground". Once the selection pass
    measures both with one estimator on one slice, every clause of that is false about the number
    printed beneath it. So this drives the real report over a real run directory and reads what it
    printed, rather than asserting anything about a fixture of my own making.
    """
    run = _run_dir(tmp_path, kl_source=headtohead.KL_SOURCE_MEASURED)
    (run / "scored-heretic-seed42.json").write_text(
        json.dumps({"auc": 0.98, "controls": {"length_only_auc": 0.5}}), encoding="utf-8")
    text = headtohead_report.render(headtohead_report.collect(str(run)))[0]
    assert "the harder ground" not in text, (
        "printed over a figure our own estimator produced, this claim is false in every clause")
    assert "measured by this pass" in text or "OUR" in text


def test_the_harder_ground_paragraph_survives_where_it_is_still_true(tmp_path):
    """An unstamped artefact is a genuine self-report, and the old sentence describes it."""
    run = _run_dir(tmp_path)
    (run / "scored-heretic-seed42.json").write_text(
        json.dumps({"auc": 0.98, "controls": {"length_only_auc": 0.5}}), encoding="utf-8")
    text = headtohead_report.render(headtohead_report.collect(str(run)))[0]
    assert "the harder ground" in text
