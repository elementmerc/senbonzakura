# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A second behaviour joins a score file without destroying the first one.

The defect these cover is the one that blocked the four-behaviour card: `score` replaced `--out`
wholesale, so writing behaviour two deleted behaviour one and nothing anywhere said a measurement
had been lost. Every refusal below is checked by its message as well as its type, because a writer
that refuses without naming the fix sends an operator to re-run a pass they already paid for.
"""
import json
from pathlib import Path

import pytest
from senbonzakura_check.measurement import METRICS_KEY

from senbonzakura import behaviourset


def result(*, model="org/model", slot="refusal_rate.senbonzakura-ruler", value=0.4, n=120,
           precision="bfloat16", prompt_format="tokenizer:abc123", tool_version="0.5.1",
           input_digest="1111111111111111", partition="measure", extra=None):
    """A stamped single-behaviour result of the shape `score` writes."""
    doc = {
        "label": "arm", "model": model, "eval": "corpus", "n": n,
        "refusal": value,
        METRICS_KEY: {
            slot: {
                "metric": slot.split(".")[0], "estimator": slot.split(".")[1], "value": value,
                "n": n, "precision": precision, "prompt_format": prompt_format,
                "tool_version": tool_version, "input_digest": input_digest,
                "partition": partition, "seeds": "no-seed",
            },
        },
    }
    doc.update(extra or {})
    return doc


# ── the shape a set takes ─────────────────────────────────────────────────────────────────────────

def test_a_fresh_set_holds_one_behaviour_and_declares_itself():
    merged = behaviourset.merge(None, result(), behaviour="refusal")
    assert merged[behaviourset.SCHEMA_FIELD] == behaviourset.SCHEMA
    assert merged[behaviourset.ROSTER_KEY] == ["refusal"]
    assert merged["model"] == "org/model"
    assert list(merged[METRICS_KEY]) == ["refusal_rate.senbonzakura-ruler"]
    # The entry is the run's own document, verbatim, so nothing a pass recorded is dropped.
    assert merged[behaviourset.BEHAVIOURS_KEY]["refusal"] == result()


def test_a_second_behaviour_does_not_destroy_the_first():
    one = behaviourset.merge(None, result(), behaviour="refusal")
    two = behaviourset.merge(one, result(slot="capability.code-graded", value=0.8),
                             behaviour="sycophancy")
    assert two[behaviourset.ROSTER_KEY] == ["refusal", "sycophancy"]
    assert set(two[METRICS_KEY]) == {"refusal_rate.senbonzakura-ruler", "capability.code-graded"}
    assert two[METRICS_KEY]["refusal_rate.senbonzakura-ruler"]["value"] == 0.4
    assert two[METRICS_KEY]["capability.code-graded"]["value"] == 0.8


def test_the_shared_identity_is_recorded_and_the_varying_fields_are_named():
    merged = behaviourset.merge(None, result(), behaviour="refusal")
    assert merged[behaviourset.SHARED_KEY] == {
        "precision": "bfloat16", "prompt_format": "tokenizer:abc123", "tool_version": "0.5.1"}
    assert merged["pinned_may_differ"] == list(behaviourset.MAY_DIFFER)
    assert "own corpus" in merged["pinned_may_differ_doc"]


def test_two_behaviours_may_have_been_scored_on_different_corpora():
    """The point of the whole design: behaviours do not share a generation pass or a corpus."""
    one = behaviourset.merge(None, result(input_digest="aaaaaaaaaaaaaaaa"), behaviour="refusal")
    two = behaviourset.merge(one, result(slot="capability.code-graded",
                                         input_digest="bbbbbbbbbbbbbbbb", partition="first-40-rows"),
                             behaviour="sycophancy")
    digests = {slot: block["input_digest"] for slot, block in two[METRICS_KEY].items()}
    assert digests == {"refusal_rate.senbonzakura-ruler": "aaaaaaaaaaaaaaaa",
                       "capability.code-graded": "bbbbbbbbbbbbbbbb"}


# ── idempotence and reproducibility ───────────────────────────────────────────────────────────────

def test_re_running_a_behaviour_replaces_its_own_entry_and_leaves_the_others():
    one = behaviourset.merge(None, result(), behaviour="refusal")
    two = behaviourset.merge(one, result(slot="capability.code-graded"), behaviour="sycophancy")
    again = behaviourset.merge(two, result(value=0.1, n=200), behaviour="refusal")
    assert again[behaviourset.ROSTER_KEY] == ["refusal", "sycophancy"]
    assert again[METRICS_KEY]["refusal_rate.senbonzakura-ruler"]["value"] == 0.1
    assert again[METRICS_KEY]["refusal_rate.senbonzakura-ruler"]["n"] == 200
    assert "capability.code-graded" in again[METRICS_KEY]


def test_a_re_run_under_a_different_estimator_leaves_no_stale_slot_behind():
    """The block is rebuilt from the entries, so a renamed estimator cannot leave its old slot."""
    one = behaviourset.merge(None, result(), behaviour="refusal")
    again = behaviourset.merge(one, result(slot="refusal_rate.heretic-keyword"), behaviour="refusal")
    assert list(again[METRICS_KEY]) == ["refusal_rate.heretic-keyword"]


def test_the_same_behaviours_serialise_to_the_same_bytes_whatever_order_they_arrived_in():
    a, b = result(), result(slot="capability.code-graded")
    first = behaviourset.merge(behaviourset.merge(None, a, behaviour="refusal"), b,
                               behaviour="sycophancy")
    second = behaviourset.merge(behaviourset.merge(None, b, behaviour="sycophancy"), a,
                                behaviour="refusal")
    assert json.dumps(first, indent=2) == json.dumps(second, indent=2)


# ── the refusals ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", ["", "Refusal", "9lives", "has space", "a" * 41, None, 7])
def test_a_key_that_could_not_travel_in_an_address_is_refused(key):
    with pytest.raises(behaviourset.BehaviourSetError, match="not a usable behaviour key"):
        behaviourset.merge(None, result(), behaviour=key)


def test_a_result_with_no_stamped_metric_is_refused_rather_than_stored():
    bare = {"model": "org/model", "n": 30, "refusal": 0.4}
    with pytest.raises(behaviourset.BehaviourSetError, match="no stamped metrics block"):
        behaviourset.merge(None, bare, behaviour="refusal")


def test_a_metric_slot_holding_something_other_than_a_stamp_is_refused():
    broken = result()
    broken[METRICS_KEY]["refusal_rate.senbonzakura-ruler"] = 0.4
    with pytest.raises(behaviourset.BehaviourSetError, match="other than a stamp block"):
        behaviourset.merge(None, broken, behaviour="refusal")


def test_a_result_that_names_no_model_is_refused():
    with pytest.raises(behaviourset.BehaviourSetError, match="names no model"):
        behaviourset.merge(None, result(model="   "), behaviour="refusal")


def test_a_result_that_disowned_its_own_measurement_may_not_join_a_set():
    """A set is read at the document level, so one invalidation would withdraw all four figures."""
    disowned = result(extra={"self_invalidated": "the replies carried no verdict."})
    with pytest.raises(behaviourset.BehaviourSetError, match="disowned its own measurement"):
        behaviourset.merge(None, disowned, behaviour="refusal")



def test_a_different_model_is_refused_rather_than_merged():
    one = behaviourset.merge(None, result(model="org/one"), behaviour="refusal")
    with pytest.raises(behaviourset.BehaviourSetError, match="One card names one model"):
        behaviourset.merge(one, result(model="org/two", slot="capability.code-graded"),
                           behaviour="sycophancy")


@pytest.mark.parametrize(("field", "other"), [("precision", {"precision": "nf4"}),
                                              ("prompt_format", {"prompt_format": "raw"}),
                                              ("tool_version", {"tool_version": "0.6.0"})])
def test_a_different_instrument_is_refused_rather_than_merged(field, other):
    one = behaviourset.merge(None, result(), behaviour="refusal")
    with pytest.raises(behaviourset.BehaviourSetError, match="two instruments"):
        behaviourset.merge(one, result(slot="capability.code-graded", **other),
                           behaviour="sycophancy")


def test_two_behaviours_claiming_one_metric_slot_are_refused():
    one = behaviourset.merge(None, result(), behaviour="refusal")
    with pytest.raises(behaviourset.BehaviourSetError, match="both stamp the metric slot"):
        behaviourset.merge(one, result(), behaviour="sycophancy")


def test_a_stamp_missing_a_pinned_field_cannot_be_shown_to_agree():
    missing = result()
    del missing[METRICS_KEY]["refusal_rate.senbonzakura-ruler"]["precision"]
    with pytest.raises(behaviourset.BehaviourSetError, match="records no precision"):
        behaviourset.merge(None, missing, behaviour="refusal")


def test_a_result_whose_own_slots_disagree_has_no_single_value_to_agree_with():
    two = result()
    two[METRICS_KEY]["refusal_rate.heretic-keyword"] = dict(
        two[METRICS_KEY]["refusal_rate.senbonzakura-ruler"], precision="nf4")
    with pytest.raises(behaviourset.BehaviourSetError, match="more than one precision"):
        behaviourset.merge(None, two, behaviour="refusal")


def test_a_single_behaviour_result_file_is_not_adopted_by_guessing_what_it_holds():
    with pytest.raises(behaviourset.BehaviourSetError, match="not a behaviour set"):
        behaviourset.merge(result(), result(), behaviour="refusal")


def test_a_set_that_disowned_itself_is_not_added_to():
    one = behaviourset.merge(None, result(), behaviour="refusal")
    one["self_invalidated"] = "the whole run is withdrawn."
    with pytest.raises(behaviourset.BehaviourSetError, match="disowned its own measurement"):
        behaviourset.merge(one, result(slot="capability.code-graded"), behaviour="sycophancy")


def test_a_set_whose_behaviours_block_is_not_an_object_is_refused():
    one = behaviourset.merge(None, result(), behaviour="refusal")
    one[behaviourset.BEHAVIOURS_KEY] = ["refusal"]
    with pytest.raises(behaviourset.BehaviourSetError, match="behaviours block is list"):
        behaviourset.merge(one, result(slot="capability.code-graded"), behaviour="sycophancy")


def test_a_set_entry_that_is_not_an_object_is_refused():
    one = behaviourset.merge(None, result(), behaviour="refusal")
    one[behaviourset.BEHAVIOURS_KEY]["honesty"] = 1
    with pytest.raises(behaviourset.BehaviourSetError, match="'honesty' entry is int"):
        behaviourset.merge(one, result(slot="capability.code-graded"), behaviour="sycophancy")


def test_a_set_entry_under_an_unusable_key_is_refused():
    one = behaviourset.merge(None, result(), behaviour="refusal")
    one[behaviourset.BEHAVIOURS_KEY]["NOT A KEY"] = result()
    with pytest.raises(behaviourset.BehaviourSetError, match="not a usable behaviour key"):
        behaviourset.merge(one, result(slot="capability.code-graded"), behaviour="sycophancy")


def test_a_set_with_entries_and_no_shared_identity_cannot_be_shown_to_agree():
    one = behaviourset.merge(None, result(), behaviour="refusal")
    del one[behaviourset.SHARED_KEY]
    with pytest.raises(behaviourset.BehaviourSetError, match="records no shared identity"):
        behaviourset.merge(one, result(slot="capability.code-graded"), behaviour="sycophancy")


def test_a_set_with_entries_and_no_model_is_refused():
    one = behaviourset.merge(None, result(), behaviour="refusal")
    del one["model"]
    with pytest.raises(behaviourset.BehaviourSetError, match="names no model"):
        behaviourset.merge(one, result(slot="capability.code-graded"), behaviour="sycophancy")


def test_an_empty_set_file_is_treated_as_a_fresh_one():
    merged = behaviourset.merge({}, result(), behaviour="refusal")
    assert merged[behaviourset.ROSTER_KEY] == ["refusal"]


def test_a_set_declaring_itself_with_no_entries_accepts_the_first_behaviour():
    empty = {behaviourset.SCHEMA_FIELD: behaviourset.SCHEMA, behaviourset.BEHAVIOURS_KEY: {}}
    merged = behaviourset.merge(empty, result(), behaviour="refusal")
    assert merged[behaviourset.ROSTER_KEY] == ["refusal"]


def test_a_file_that_is_not_an_object_at_all_is_refused():
    with pytest.raises(behaviourset.BehaviourSetError, match="already at this path is list"):
        behaviourset.merge(["nope"], result(), behaviour="refusal")


def test_a_result_that_is_not_an_object_is_refused():
    with pytest.raises(behaviourset.BehaviourSetError, match="'refusal' result is str"):
        behaviourset.merge(None, "nope", behaviour="refusal")


# ── the disk half ─────────────────────────────────────────────────────────────────────────────────

def test_write_then_write_leaves_both_behaviours_on_disk(tmp_path):
    out = tmp_path / "score.json"
    behaviourset.write(out, result(), behaviour="refusal")
    behaviourset.write(out, result(slot="capability.code-graded"), behaviour="sycophancy")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc[behaviourset.ROSTER_KEY] == ["refusal", "sycophancy"]
    assert len(doc[METRICS_KEY]) == 2


def test_write_leaves_no_part_file_behind(tmp_path):
    out = tmp_path / "score.json"
    behaviourset.write(out, result(), behaviour="refusal")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["score.json"]


def test_a_refused_write_leaves_the_existing_set_untouched(tmp_path):
    out = tmp_path / "score.json"
    behaviourset.write(out, result(), behaviour="refusal")
    before = out.read_bytes()
    with pytest.raises(behaviourset.BehaviourSetError):
        behaviourset.write(out, result(model="org/other", slot="capability.code-graded"),
                           behaviour="sycophancy")
    assert out.read_bytes() == before


def test_read_of_a_missing_file_is_a_fresh_set(tmp_path):
    assert behaviourset.read(tmp_path / "nothing.json") is None


def test_read_of_an_empty_file_is_a_fresh_set(tmp_path):
    target = tmp_path / "score.json"
    target.write_text("   \n", encoding="utf-8")
    assert behaviourset.read(target) is None


def test_a_half_written_file_is_refused_rather_than_read_as_empty(tmp_path):
    """Reading it as empty is the destructive write this module exists to prevent."""
    target = tmp_path / "score.json"
    target.write_text('{"schema": "senbonzakura-behaviour-set/1", "behaviou', encoding="utf-8")
    with pytest.raises(behaviourset.BehaviourSetError, match="not valid JSON"):
        behaviourset.read(target)


def test_a_file_that_is_not_utf8_is_refused(tmp_path):
    target = tmp_path / "score.json"
    target.write_bytes(b"\xff\xfe\x00nonsense")
    with pytest.raises(behaviourset.BehaviourSetError, match="could not be read as UTF-8"):
        behaviourset.read(target)


def test_a_file_too_large_to_be_a_set_is_refused_before_parsing(tmp_path, monkeypatch):
    target = tmp_path / "score.json"
    target.write_text("{}" + " " * 100, encoding="utf-8")
    monkeypatch.setattr(behaviourset, "MAX_SET_BYTES", 8)
    with pytest.raises(behaviourset.BehaviourSetError, match="read up to"):
        behaviourset.read(target)


def test_a_path_that_cannot_even_be_sized_is_refused_rather_than_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "score.json"
    target.write_text("{}", encoding="utf-8")

    def refuse(self, *a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "stat", refuse)
    with pytest.raises(behaviourset.BehaviourSetError, match="cannot be examined"):
        behaviourset.read(target)


def test_a_path_that_cannot_be_examined_is_refused_rather_than_overwritten(tmp_path):
    """A directory where a file should be: there is something there, and it is not a set."""
    target = tmp_path / "score.json"
    target.mkdir()
    with pytest.raises(behaviourset.BehaviourSetError, match="could not be read as UTF-8"):
        behaviourset.read(target)
