# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`measure` can report an instrument that answers in a judgement rather than a number.

Every stage until 2026-10-15 stamped a figure with units, and the table was built around that.
`validate` does not: it asks whether a direction set carries refusal or carries topic, and the
honest answer has three values and no units. Rather than invent a threshold and print a number,
the table grew a second kind of row.

**The guardrail is the reason this file is long.** A row that prints whatever word an instrument
put in its result file is a row that can be made to print anything, and the one thing this project
most carefully never prints is a verdict on whether a model is safe. `diff` must never grow a
safety verdict; `localisation_claim: false` makes marks structurally unreachable on the card. This
is the same move one layer along: the category must come from a closed set the stage declared, and
anything else becomes a refusal that names what it refused.

So these tests care much less about the happy path than about:

- a word outside the vocabulary never reaching the table;
- the absence of a stamp being reported as designed rather than as a fault;
- the new field not disturbing a reader written before it existed.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura import measure  # noqa: E402

GOOD = {"verdict_kind": "does not generalise",
        "verdict": "on held-out clusters they score no better than random directions."}


def _row(result, stage="validate"):
    rows = measure.verdict_rows({stage: result})
    assert len(rows) == 1, "one stage in, one row out"
    return rows[0]


# --------------------------------------------------------------------------------------------
# The kinds exist and are distinguishable.
# --------------------------------------------------------------------------------------------

def test_every_reading_declares_a_kind():
    for name, reading in measure.READINGS.items():
        assert reading.kind in (measure.KIND_FIGURE, measure.KIND_VERDICT), name


def test_a_verdict_stage_declares_its_vocabulary():
    """A verdict stage with no declared vocabulary could say anything, which is the whole risk."""
    for name, reading in measure.READINGS.items():
        if reading.kind != measure.KIND_VERDICT:
            continue
        assert measure.VERDICT_VOCABULARY.get(name), (
            f"{name} answers in verdicts and declares no vocabulary, so nothing constrains what "
            f"it can make this table print")


def test_the_vocabulary_names_only_verdict_stages():
    """A vocabulary for a figure stage is dead configuration, and dead configuration misleads."""
    for name in measure.VERDICT_VOCABULARY:
        assert name in measure.READINGS, f"{name} declares verdicts and is not a stage"
        assert measure.READINGS[name].kind == measure.KIND_VERDICT, name


# --------------------------------------------------------------------------------------------
# The guardrail.
# --------------------------------------------------------------------------------------------

def test_a_verdict_outside_the_vocabulary_is_refused():
    row = _row({"verdict_kind": "this model is safe"})
    assert row.figure == "not reported"
    assert "this model is safe" in row.note, (
        "the refusal names the value it refused, because 'not reported' with no value is "
        "indistinguishable from the stage never running, and the two need different fixes")


@pytest.mark.parametrize("claim", [
    "safe",
    "this model is safe to deploy",
    "no harmful content",
    "passed",
    "SAFE",
])
def test_no_safety_claim_can_reach_the_table(claim):
    """The property, stated as the property rather than as one example.

    If somebody later widens `VERDICT_VOCABULARY` to include a word like this, this test is the
    thing that should stop them, which is why it tests the vocabulary and not just the row.
    """
    assert claim not in measure.VERDICT_VOCABULARY.get("validate", ()), (
        f"{claim!r} is a safety claim and must never be a verdict this table can print")
    assert _row({"verdict_kind": claim}).figure == "not reported"


def test_the_declared_vocabulary_contains_no_safety_language():
    """Every stage, every word, checked for the vocabulary this project does not use."""
    banned = ("safe", "unsafe", "harmful", "harmless", "secure", "dangerous", "clean", "passed")
    for stage, words in measure.VERDICT_VOCABULARY.items():
        for word in words:
            low = word.lower()
            for b in banned:
                assert b not in low, (
                    f"{stage} declares the verdict {word!r}, which carries {b!r}. This table "
                    f"reports what an instrument measured, never whether a model is acceptable")


def test_a_missing_verdict_field_is_reported_as_missing():
    row = _row({"summary": {}})
    assert row.figure == "not reported"
    assert "verdict_kind" in row.note, "the note names the field that was absent"


def test_a_result_that_is_not_a_document_does_not_crash_the_table():
    row = _row("the stage failed and this is its message")
    assert row.figure == "not measured"


# --------------------------------------------------------------------------------------------
# The happy path, and the reason beside the category.
# --------------------------------------------------------------------------------------------

def test_a_declared_verdict_is_reported_with_the_stage_sentence_as_its_reason():
    row = _row(GOOD)
    assert row.figure == "does not generalise"
    assert row.units == "", "a judgement has no units, and inventing one would be a lie"
    assert "no better than random directions" in row.note, (
        "the instrument's own sentence is the reason, which is where free text is harmless: it "
        "explains a category that has already been constrained")
    assert row.kind == measure.KIND_VERDICT


def test_every_declared_verdict_round_trips():
    for word in measure.VERDICT_VOCABULARY["validate"]:
        assert _row({"verdict_kind": word}).figure == word


def test_validate_and_measure_agree_on_the_vocabulary():
    """The two halves of one contract, checked against each other rather than against a comment.

    `validate` sets the category beside its sentence; `measure` declares which categories it will
    print. If either side gains a word the other does not know, the row silently becomes
    "not reported" on a real run, and nothing else would catch it.
    """
    import re

    source = (ROOT / "src" / "senbonzakura" / "validate.py").read_text(encoding="utf-8")
    emitted = set(re.findall(r'verdict_kind = "([^"]+)"', source))
    assert emitted, "no verdict_kind assignments found; this test has stopped measuring anything"
    declared = set(measure.VERDICT_VOCABULARY["validate"])
    assert emitted <= declared, (
        f"validate can emit {sorted(emitted - declared)}, which measure would refuse to print")


# --------------------------------------------------------------------------------------------
# It is a stage, and it is opt-in.
# --------------------------------------------------------------------------------------------

def test_validate_is_a_stage_and_is_not_in_the_default_run():
    assert "validate" in measure.STAGE_ORDER
    assert "validate" in measure.OPT_IN_STAGES, (
        "it fits directions and bakes the model across a strength grid, which is a longer job "
        "than the rest of this command put together")


def test_validate_is_not_also_named_as_an_instrument_measure_never_runs():
    """`OPT_IN_STAGES` and `NOT_IN_MEASURE` are different claims and must not both be made."""
    assert "validate" not in measure.NOT_IN_MEASURE
    assert not (set(measure.OPT_IN_STAGES) & set(measure.NOT_IN_MEASURE)), (
        "a stage cannot both run on request and never run")


def test_the_default_run_excludes_every_opt_in_stage():
    chosen = measure.wanted_stages(SimpleNamespace(only=None, baseline=None))
    assert not (set(chosen) & measure.OPT_IN_STAGES)
    assert "score" in chosen, "the default run still happens"


def test_asking_for_validate_by_name_runs_it():
    args = SimpleNamespace(only=["validate"], baseline=None)
    assert measure.wanted_stages(args) == ["validate"]


def test_the_stage_runs_one_experiment_rather_than_the_whole_grid():
    """`validate`'s own default is `all`, which bakes across a strength grid. Not as a stage."""
    args = SimpleNamespace(model="m", device="cpu", track="default", hf_token=None,
                           baseline=None, capability_n=4, trust_remote_code=False, n=60)
    argv = measure.stage_argv("validate", args, Path("out"))
    assert "--experiment" in argv
    assert argv[argv.index("--experiment") + 1] == "e1"
    assert "all" not in argv


# --------------------------------------------------------------------------------------------
# The old shape still means what it meant.
# --------------------------------------------------------------------------------------------

def test_a_figure_row_is_unchanged_and_defaults_to_the_figure_kind():
    row = measure.Row("score", "0.12", "fraction", "a note")
    assert row.kind == measure.KIND_FIGURE
    assert (row.stage, row.figure, row.units, row.note) == ("score", "0.12", "fraction", "a note")


def test_the_note_stays_at_index_three():
    """The class docstring calls this ordering load-bearing, and it is: readers index positionally."""
    row = measure.Row("score", "0.12", "fraction", "a note", "±0.01", measure.KIND_FIGURE)
    assert row[3] == "a note"
    assert row[4] == "±0.01"
