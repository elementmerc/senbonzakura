# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Arguments a command can work out for itself, and the ones it must never guess.

WHY THE LINE IS WHERE IT IS

`--out` is a place to put a file, and a wrong guess is visible the moment the command says where
it wrote. `--metric` decides which property a later run is gated on, and a wrong guess produces a
gate that passes while the thing it was meant to protect moves. So an output path gets a default
and a metric gets one only when the artefact stamped exactly one, with every other case refused
by name.

`--seeds` keeps no default at all, on purpose. Its help says the figure's spread is stated rather
than inferred, because one artefact is one run and a baseline claiming a spread it does not have
is worse than none. That reasoning did not change because the flag beside it became optional.
"""
from __future__ import annotations

import json

import pytest

from senbonzakura import argresolve, baseline


# ── the model, in two spellings ──────────────────────────────────────────────────

def test_the_positional_is_accepted_where_the_flag_would_be():
    assert argresolve.pick_model("Qwen/Qwen3-1.7B", None) == "Qwen/Qwen3-1.7B"


def test_two_different_models_are_refused_rather_than_ranked():
    """Silently preferring one is how a run measures a model nobody asked for, with the artefact
    recording the winner and nothing saying the other was ever mentioned.
    """
    with pytest.raises(SystemExit) as e:
        argresolve.pick_model("c/d", "a/b")
    assert "a/b" in str(e.value) and "c/d" in str(e.value), "both have to be named to be acted on"


def test_the_same_model_twice_is_not_a_conflict():
    assert argresolve.pick_model("a/b", "a/b") == "a/b"


def test_the_refusal_names_the_command_the_person_typed():
    with pytest.raises(SystemExit, match="senbonzakura capability"):
        argresolve.pick_model(None, None, command="senbonzakura capability")


def test_capability_takes_the_model_without_a_flag():
    from senbonzakura import capability

    a = capability.build_parser().parse_args(["Qwen/Qwen3-1.7B"])
    assert argresolve.pick_model(a.model_positional, a.model) == "Qwen/Qwen3-1.7B"


def test_capability_still_takes_the_flag_because_every_run_spec_passes_it():
    from senbonzakura import capability

    a = capability.build_parser().parse_args(["--model", "Qwen/Qwen3-1.7B", "--out", "x.json"])
    assert argresolve.pick_model(a.model_positional, a.model) == "Qwen/Qwen3-1.7B"


# ── the default output, and what it refuses to do ────────────────────────────────

def test_a_default_output_that_already_exists_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "capability.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        argresolve.refuse_to_overwrite("capability.json")
    assert "--out" in str(e.value), "the refusal must say how to get past it"


def test_a_free_default_output_is_returned_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert argresolve.refuse_to_overwrite("capability.json") == "capability.json"


def test_the_track_builder_no_longer_demands_an_output_directory():
    from senbonzakura import track

    a = track.build_parser().parse_args([])
    assert a.out == "track", (
        "the default has to be the directory the abliterator picks up when --track is left out, "
        "or the two conveniences point at different places")


# ── the metric, which is only inferred when there is nothing to choose between ───

def test_one_stamped_metric_is_read_rather_than_demanded():
    assert baseline.only_metric({"metrics": {"coherence": {}}}) == "coherence"


def test_several_stamped_metrics_are_refused_and_all_of_them_named():
    with pytest.raises(baseline.BaselineError) as e:
        baseline.only_metric({"metrics": {"coherence": {}, "refusal_rate": {}}})
    msg = str(e.value)
    assert "coherence" in msg and "refusal_rate" in msg, (
        "a refusal that does not list the candidates leaves the reader to open the file")


def test_an_artefact_with_no_stamp_says_that_rather_than_naming_a_flag():
    with pytest.raises(baseline.BaselineError, match="no `metrics` block"):
        baseline.only_metric({})


def test_the_seeds_are_still_required():
    """A baseline claiming a spread it does not have is worse than no baseline, and nothing about
    the flags beside it changed that.
    """
    from senbonzakura.baseline import build_parser

    required = {a.dest for a in build_parser()._actions if a.required}
    assert "seeds" in required
    assert "out" not in required and "metric" not in required and "measurement" not in required


def test_the_default_baseline_path_is_one_file_per_metric():
    assert baseline.default_out("coherence") == "baselines/coherence.json"
    assert "/" not in baseline.default_out("a/b").split("baselines/")[1], (
        "a metric key with a slash in it would write outside the baselines directory")


def test_two_different_artefacts_are_refused_rather_than_ranked(capsys, tmp_path):
    assert baseline.main([
        str(tmp_path / "one.json"), "--measurement", str(tmp_path / "two.json"),
        "--seeds", "0"]) == 2
    assert "Pass one" in capsys.readouterr().err


def test_no_artefact_at_all_is_refused_with_both_ways_to_give_one(capsys):
    assert baseline.main(["--seeds", "0"]) == 2
    err = capsys.readouterr().err
    assert "--measurement" in err and "first argument" in err


def test_the_metric_and_the_path_are_both_worked_out_end_to_end(tmp_path, monkeypatch, capsys):
    """The whole point: one artefact, one seed list, nothing else typed."""
    monkeypatch.chdir(tmp_path)
    art = tmp_path / "coherence.json"
    # Every pinned field, because a baseline with a hole in it is refused and that refusal is
    # correct: each of these decides whether a later measurement may be compared with this one.
    # `model` sits at the top of the artefact and everything else inside the metric's block,
    # because the first describes the file and the rest describe that one measurement.
    art.write_text(json.dumps({"model": "Qwen/Qwen3-1.7B", "metrics": {"coherence": {
        "value": 11.2, "interval": [11.0, 11.4], "n": 200, "higher_is_better": False,
        "estimator": "senbonzakura-ruler", "units": "perplexity",
        "input_digest": "0" * 64, "partition": "measure",
        "prompt_format": "chat", "tool_version": "0.4.0", "precision": "bfloat16"}}}))
    rc = baseline.main([str(art), "--seeds", "0"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert (tmp_path / "baselines" / "coherence.json").is_file(), out.out
    assert "stamped only 'coherence'" in out.out, (
        "an inferred metric has to be announced, or a reader cannot tell it was inferred")
