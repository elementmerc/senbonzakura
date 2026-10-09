# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Q-116, option E of Q-54: `measure.json` is a declared summary, not a measurement.

THE PROBLEM IN ONE PICTURE. `measure` runs several instruments and writes one file. Every
measurement file must stamp the fields saying under what conditions its number was taken, and the
instruments legitimately differ on several of them. A school report card has one row per subject
and each subject's exam had its own paper, date and marker; writing one date on the card lies
about every other subject.

So the card is not an exam result. It declares itself a summary, carries no stamp of its own,
holds each instrument's stamp verbatim, and NAMES the fields they disagree on rather than
omitting them, because an aggregate with no stamp invites the reading "nobody stamped this" while
one that says which fields differ is making a claim a reader can check.
"""
import json

import pytest

from senbonzakura import baseline, measure


def _stamped(slot, **pinned):
    """A stage artefact with one stamped metric. `slot` is the key, `pinned` the stamped fields,
    so a test can give the slot and the `metric` identity different values on purpose.
    """
    return {"metrics": {slot: {"value": 1.0, **pinned}}}


def _write(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# ── the two modules agree about the word ──────────────────────────────────────────────

def test_the_writer_and_the_reader_use_the_same_word_for_a_summary():
    """`baseline` holds the literal rather than importing it, so that reading a baseline does not
    pay for `measure`'s stage machinery. That saving is only safe while a test holds the two
    together, because nothing else would notice them drifting apart.
    """
    assert measure.SUMMARY_KIND == baseline._SUMMARY_KIND


# ── what the summary carries ──────────────────────────────────────────────────────────

def test_the_stage_stamps_are_read_back_from_the_files_the_stages_wrote(tmp_path):
    """VERBATIM, and read back rather than remembered. What a reader needs is what each instrument
    actually recorded, which is not the same document as what the summary believes it asked for.
    """
    _write(tmp_path / "score.json",
           _stamped("refusal", model="M", metric="refusal", partition="measure",
                    input_digest="abc", precision="bf16", ignored="not pinned"))
    _write(tmp_path / "coherence.json",
           _stamped("coherence", model="M", metric="coherence", partition="fixed-passage",
                    input_digest="def", precision="bf16"))
    got = measure.stage_stamps(tmp_path, {"score": "score.json", "coherence": "coherence.json"})
    assert got["score"]["refusal"]["partition"] == "measure"
    assert got["coherence"]["coherence"]["partition"] == "fixed-passage"
    assert "ignored" not in got["score"]["refusal"], (
        "only the pinned fields are carried; copying the whole block would make the summary a "
        "second copy of every artefact beside it")
    assert "value" not in got["score"]["refusal"], (
        "the FIGURE must not be copied into the summary. Two homes for one number is how a "
        "stale copy gets quoted")


def test_a_stage_whose_file_cannot_be_read_says_so_rather_than_being_dropped(tmp_path):
    """An absent key and a key saying "this could not be read" are different claims, and the
    second is the true one. Dropping it would make a summary describe three instruments as four.
    """
    (tmp_path / "score.json").write_text("{ not json", encoding="utf-8")
    got = measure.stage_stamps(tmp_path, {"score": "score.json"})
    assert "score" in got
    assert "unreadable" in got["score"]
    assert "JSONDecodeError" in got["score"]["unreadable"] or "ValueError" in got["score"]["unreadable"]


def test_a_stage_file_that_is_missing_entirely_says_so(tmp_path):
    got = measure.stage_stamps(tmp_path, {"score": "nothing-here.json"})
    assert "unreadable" in got["score"]


def test_a_stage_file_with_no_metrics_block_is_named_as_unstamped(tmp_path):
    _write(tmp_path / "score.json", {"model": "M"})
    got = measure.stage_stamps(tmp_path, {"score": "score.json"})
    assert got["score"]["unstamped"].startswith("score.json carries no metrics")


# ── which fields differ, computed rather than typed ───────────────────────────────────

def test_the_differing_fields_are_computed_from_the_stamps_actually_written():
    stamps = {
        "score": {"refusal": {"model": "M", "metric": "refusal", "partition": "measure"}},
        "coherence": {"coherence": {"model": "M", "metric": "coherence",
                                    "partition": "fixed-passage"}},
    }
    assert measure.pinned_fields_that_differ(stamps) == ["metric", "partition"]


def test_a_field_every_instrument_agrees_on_is_not_listed():
    stamps = {
        "a": {"x": {"model": "M", "precision": "bf16"}},
        "b": {"y": {"model": "M", "precision": "bf16"}},
    }
    assert measure.pinned_fields_that_differ(stamps) == []


def test_a_fifth_instrument_changes_the_answer_without_anybody_editing_a_tuple():
    """The reason this is computed. A typed list would need somebody to remember, and the thing
    nobody remembers is the thing that silently stops being true.
    """
    four = {f"s{i}": {"m": {"model": "M", "partition": "measure"}} for i in range(4)}
    assert measure.pinned_fields_that_differ(four) == []
    four["s5"] = {"m": {"model": "M", "partition": "search"}}
    assert measure.pinned_fields_that_differ(four) == ["partition"]


def test_the_unreadable_and_unstamped_markers_are_not_mistaken_for_pinned_fields():
    stamps = {"a": {"unreadable": "OSError: gone"},
              "b": {"m": {"model": "M"}}}
    assert measure.pinned_fields_that_differ(stamps) == []


def test_the_differing_fields_are_sorted_so_two_runs_write_the_same_bytes():
    """Iteration order of a set must not reach a result file: two runs on the same input produce
    byte-identical output, which is the reproducibility rule this project holds everywhere.
    """
    stamps = {"a": {"m": {"partition": "x", "metric": "p", "model": "M", "precision": "bf16"}},
              "b": {"m": {"partition": "y", "metric": "q", "model": "M", "precision": "f16"}}}
    got = measure.pinned_fields_that_differ(stamps)
    assert got == sorted(got)
    assert got == ["metric", "partition", "precision"]


def test_a_nested_pinned_value_is_compared_by_content_not_by_identity():
    """Some pinned values are dicts (a corpus pin is `{"id", "revision", "kind"}`). Two equal
    dicts must not read as two values, or every run would report a field as differing.
    """
    pin = {"id": "c", "revision": "r", "kind": "hf"}
    stamps = {"a": {"m": {"input_digest": dict(pin)}},
              "b": {"m": {"input_digest": dict(pin)}}}
    assert measure.pinned_fields_that_differ(stamps) == []
    stamps["b"]["m"]["input_digest"] = {**pin, "revision": "other"}
    assert measure.pinned_fields_that_differ(stamps) == ["input_digest"]


# ── the reader prefers the declaration over the shape ─────────────────────────────────

def test_a_declared_summary_is_recognised_by_its_kind():
    doc = {"kind": "summary", "stages": {"score": "score.json", "compass": "compass.json"}}
    key, files = baseline._is_measure_summary(doc)
    assert key == "kind"
    assert files == ["compass.json", "score.json"]


def test_a_summary_that_declares_itself_and_lists_nothing_is_still_a_summary():
    """Falling through to the shape check here would report it as an old unstamped artefact, which
    is the false advice this whole branch exists to have stopped giving.
    """
    assert baseline._is_measure_summary({"kind": "summary"}) == ("kind", [])


def test_a_summary_written_before_the_declaration_existed_is_still_recognised():
    """They are still on people's disks and still deserve the refusal that points at the per-stage
    files rather than one saying the file predates a stamp introduced in September.
    """
    key, files = baseline._is_measure_summary({"stages": {"score": "score.json"}})
    assert key == "stages"
    assert files == ["score.json"]


def test_an_ordinary_measurement_is_not_mistaken_for_a_summary():
    assert baseline._is_measure_summary(_stamped("refusal", model="M")) is None


def test_the_refusal_names_the_per_stage_files_to_record_instead():
    doc = {"kind": "summary", "stages": {"score": "score.json"}}
    with pytest.raises(baseline.BaselineError) as e:
        baseline.from_artefact(doc, "refusal", seeds=1)
    msg = str(e.value)
    assert "measure` summary" in msg
    assert "score.json" in msg
    assert "That is correct for what it is" in msg


# ── end to end: the file measure actually writes ──────────────────────────────────────

def test_the_summary_written_by_a_run_declares_itself_and_carries_no_metrics(tmp_path,
                                                                             monkeypatch):
    """The whole point, asserted on the real output rather than on the pieces."""
    out = tmp_path / "out"
    out.mkdir()
    _write(out / "score.json",
           _stamped("refusal", model="M", metric="refusal", partition="measure"))
    _write(out / "coherence.json",
           _stamped("coherence", model="M", metric="coherence", partition="fixed-passage"))

    monkeypatch.setattr(measure, "run", lambda args, log=print: ({"score": None,
                                                                 "coherence": None}, []))
    monkeypatch.setattr(measure, "verdict_rows", lambda results: [])
    rc = measure.main(["M", "--out", str(out), "--track", "default"])
    assert rc == 0

    doc = json.loads((out / "measure.json").read_text(encoding="utf-8"))
    assert doc["kind"] == measure.SUMMARY_KIND
    assert "metrics" not in doc, (
        "a summary must carry no aggregate metrics block: one value for a field its instruments "
        "measured differently would be false about all but one of them")
    assert doc["stage_stamps"]["score"]["refusal"]["partition"] == "measure"
    assert doc["pinned_fields_that_differ"] == ["metric", "partition"]
    assert "legitimately differ" in doc["pinned_fields_that_differ_note"]


def test_the_summary_a_run_writes_is_refused_by_baseline_with_the_useful_advice(tmp_path,
                                                                               monkeypatch):
    """The two halves meet: the file this command writes is the file the next command refuses, and
    the refusal has to point somewhere a reader can go.
    """
    out = tmp_path / "out"
    out.mkdir()
    _write(out / "score.json", _stamped("refusal", model="M", metric="refusal"))
    monkeypatch.setattr(measure, "run", lambda args, log=print: ({"score": None}, []))
    monkeypatch.setattr(measure, "verdict_rows", lambda results: [])
    measure.main(["M", "--out", str(out), "--track", "default"])

    doc = json.loads((out / "measure.json").read_text(encoding="utf-8"))
    with pytest.raises(baseline.BaselineError) as e:
        baseline.from_artefact(doc, "refusal", seeds=1)
    assert "score.json" in str(e.value)
