# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The adapters answer "not recorded" for malformed fields, and never invent a figure to fill the gap.

THE FAILURE THIS EXISTS TO CATCH: an adapter reads a file somebody else wrote. If a field has the
wrong type, the adapter must leave the figure out, so the check that reads it sees "not recorded"
and says so. The two ways to get this wrong are a crash, which aborts a sweep, and a guess, which
produces a number nobody measured. Both are silent failures from the reader's side.

Each malformed case here is paired with a well-formed case beside it. A guard that drops everything
would pass the malformed half, so the positive half is what shows the guard is the thing doing the
work.
"""
from __future__ import annotations

from senbonzakura_check.adapters import inspect_log, lm_eval_log, senbonzakura_log

# ── lm-evaluation-harness ─────────────────────────────────────────────────────────────────────

def test_a_splits_block_that_is_not_a_mapping_records_no_split():
    assert lm_eval_log._splits("not a mapping") is None
    assert lm_eval_log._splits(None) is None


def test_a_task_config_that_is_not_a_mapping_is_skipped_not_guessed_at():
    assert lm_eval_log._splits({"gsm8k": "test"}) is None


def test_an_empty_split_name_is_not_recorded_as_a_split():
    assert lm_eval_log._splits({"gsm8k": {"test_split": ""}}) is None


def test_a_well_formed_split_is_recorded_as_the_split_it_names():
    assert lm_eval_log._splits({"gsm8k": {"test_split": "test"}}) == {"gsm8k": "test:test"}


def test_a_task_whose_results_are_not_a_mapping_contributes_no_metrics_and_the_rest_still_do():
    doc = {"results": {"broken": "a string, not a mapping",
                       "gsm8k": {"exact_match,strict-match": 0.5, "alias": "gsm8k"}}}
    metrics = lm_eval_log.LmEvalAdapter.normalise(doc)["metrics"]
    assert not any(name.startswith("broken.") for name in metrics), metrics
    assert "gsm8k.exact_match,strict-match" in metrics, (
        "the well-formed task beside the broken one must still be read")


def test_a_non_numeric_metric_value_is_not_read_as_a_figure():
    doc = {"results": {"gsm8k": {"exact_match,none": "0.5", "acc,none": 0.25}}}
    metrics = lm_eval_log.LmEvalAdapter.normalise(doc)["metrics"]
    assert "gsm8k.exact_match,none" not in metrics, "a string is not a measurement"
    assert "gsm8k.acc,none" in metrics


# ── Inspect ───────────────────────────────────────────────────────────────────────────────────

def _inspect_doc(scores):
    return {"eval": {"task": "t", "model": "m"},
            "results": {"total_samples": 4, "completed_samples": 4, "scores": scores}}


def test_a_score_that_is_not_an_object_is_skipped_and_the_others_read():
    doc = _inspect_doc(["not a score", {"name": "choice", "metrics": {
        "accuracy": {"name": "accuracy", "value": 0.5}}}])
    metrics = inspect_log.InspectAdapter.normalise(doc)["metrics"]
    assert "choice.accuracy" in metrics


def test_a_metric_with_a_non_numeric_value_is_not_read_as_a_figure():
    doc = _inspect_doc([{"name": "choice", "metrics": {
        "accuracy": {"name": "accuracy", "value": "high"},
        "stderr": {"name": "stderr", "value": 0.1}}}])
    metrics = inspect_log.InspectAdapter.normalise(doc)["metrics"]
    assert "choice.accuracy" not in metrics
    assert "choice.stderr" in metrics


def test_a_score_with_no_name_is_keyed_by_its_metric_alone():
    doc = _inspect_doc([{"metrics": {"accuracy": {"name": "accuracy", "value": 0.5}}}])
    metrics = inspect_log.InspectAdapter.normalise(doc)["metrics"]
    assert "accuracy" in metrics


# ── senbonzakura: conversion receipts and the compass ─────────────────────────────────────────

def test_a_conversion_receipt_whose_source_is_not_a_mapping_names_no_model():
    doc = {"record": senbonzakura_log._CONVERSION_KIND, "source": "a path, not a mapping",
           "target": {"path": "out.gguf"}}
    assert senbonzakura_log.SenbonzakuraAdapter.normalise(doc)["model"] is None


def test_a_conversion_receipt_with_a_mapping_source_names_its_model():
    doc = {"record": senbonzakura_log._CONVERSION_KIND,
           "source": {"path": "org/model"}, "target": {"path": "out.gguf"}}
    assert senbonzakura_log.SenbonzakuraAdapter.normalise(doc)["model"] == "org/model"


def _compass(**over):
    doc = {"model": "m", "mode": "logit_margin", "auc": 0.9, "auc_ci": [0.85, 0.95],
           "n_harmful": 100, "n_harmless": 100,
           "controls": {"length_only_auc": 0.6}}
    doc.update(over)
    return doc


def test_a_compass_with_integer_counts_reports_its_total_sample_size():
    metrics = senbonzakura_log.SenbonzakuraAdapter.normalise(_compass())["metrics"]
    assert metrics["compass_auc.margin-past-preamble"]["n"] == 200


def test_a_compass_whose_counts_are_not_integers_reports_no_sample_size_rather_than_a_guess():
    doc = _compass(n_harmful="100", n_harmless=100)
    metrics = senbonzakura_log.SenbonzakuraAdapter.normalise(doc)["metrics"]
    assert metrics["compass_auc.margin-past-preamble"]["n"] is None


def test_a_compass_whose_length_control_is_not_a_number_has_no_control_entry():
    doc = _compass(controls={"length_only_auc": "0.6"})
    metrics = senbonzakura_log.SenbonzakuraAdapter.normalise(doc)["metrics"]
    assert "compass_auc.length-only-control" not in metrics
    assert "compass_auc.margin-past-preamble" in metrics


def test_a_compass_with_a_well_formed_length_control_has_one():
    metrics = senbonzakura_log.SenbonzakuraAdapter.normalise(_compass())["metrics"]
    assert metrics["compass_auc.length-only-control"]["value"] == 0.6


def test_a_topic_matched_arm_that_is_not_a_mapping_is_ignored_and_the_main_arm_still_counts():
    """Two halves, so the arm is actually reached: a main arm that does not fire, with a broken
    topic-matched arm beside it, must come back clean rather than crash; and a main arm that does
    fire must still be read.
    """
    quiet = _compass(topic_matched="not a mapping", controls={"length_only_auc": 0.6})
    assert senbonzakura_log.SenbonzakuraAdapter.normalise(quiet)["self_invalidated"] is None
    loud = _compass(topic_matched="not a mapping", controls={"length_only_auc": 0.95})
    assert senbonzakura_log.SenbonzakuraAdapter.normalise(loud)["self_invalidated"]


def test_a_compass_with_no_null_rulers_recorded_claims_no_null_beat_it_cannot_support():
    doc = _compass(controls={})
    assert senbonzakura_log.SenbonzakuraAdapter.normalise(doc)["self_invalidated"] is None


def test_a_null_ruler_that_is_not_a_number_or_is_the_strongest_summary_is_not_a_candidate():
    doc = _compass(controls={"nulls": {"strongest_auc": 0.99, "word_count_auc": "0.99"}})
    assert senbonzakura_log.SenbonzakuraAdapter.normalise(doc)["self_invalidated"] is None


def test_a_null_ruler_that_beats_the_compass_is_read_when_it_is_a_number():
    doc = _compass(controls={"nulls": {"word_count_auc": 0.99}})
    assert senbonzakura_log.SenbonzakuraAdapter.normalise(doc)["self_invalidated"]
