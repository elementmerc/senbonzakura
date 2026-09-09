# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for the record a run writes about its own inputs before it starts searching.

It exists because `--resume` continues the persisted study and takes everything else from the
command line, so the two surfaces failed in opposite ways: the guided mode printed a resume
command with no `--model` at all, which argparse rejects, and the non-interactive path accepted a
resume with a different `--track`, which silently carries on one corpus's trials while scoring new
ones against another.
"""
import json

import pytest

from senbonzakura import runrecord


def test_a_record_round_trips(tmp_path):
    runrecord.write(tmp_path, model="Qwen/Qwen3-1.7B", track="default", trials=200)
    got = runrecord.read(tmp_path)
    assert got["model"] == "Qwen/Qwen3-1.7B"
    assert got["track"] == "default"
    assert got["trials"] == 200


def test_a_directory_with_no_record_reads_as_none(tmp_path):
    assert runrecord.read(tmp_path) is None


@pytest.mark.parametrize("text", ["", "{", "not json", "[]", '"a string"'])
def test_a_damaged_record_is_distinguishable_from_an_absent_one(tmp_path, text):
    """THE DEFECT THE PANEL FOUND, in code written that same morning.

    `read` used to return None for four conditions: no file, no permission, truncated JSON, and
    JSON that is not an object. The guard returned silently on None, so "there is no record, which
    is fine" and "the record is damaged, which is not" were one value with one behaviour. The
    guard switched itself off on the only input it could not vouch for.
    """
    (tmp_path / runrecord.NAME).write_text(text, encoding="utf-8")
    with pytest.raises(runrecord.UnreadableError):
        runrecord.read(tmp_path)
    # And the caller that only decorates a menu still gets its tolerant answer.
    assert runrecord.read_quiet(tmp_path) is None


def test_a_damaged_record_refuses_the_resume_instead_of_skipping_the_check(tmp_path):
    """And it must refuse BEFORE the caller overwrites it.

    `cli` calls the guard on one line and `runrecord.write` on the next. Declining to check meant
    the damaged file was replaced with the new run's inputs, so the only evidence of what the
    completed trials were scored on was destroyed by the act of not checking it.
    """
    (tmp_path / runrecord.NAME).write_text("{truncated", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        runrecord.refuse_across_inputs(tmp_path, {"model": "M", "track": "T"})
    message = str(e.value)
    assert "cannot be read" in message
    # It names both ways forward, because "try again" is not one of them.
    assert "--model" in message and "fresh run" in message


def test_a_second_run_overwrites_the_record(tmp_path):
    runrecord.write(tmp_path, model="A", track="t1")
    runrecord.write(tmp_path, model="B", track="t2")
    assert runrecord.read(tmp_path)["model"] == "B"


# ── the guard ────────────────────────────────────────────────────────────────────
def test_matching_inputs_pass_and_return_the_record(tmp_path):
    runrecord.write(tmp_path, model="M", track="T")
    got = runrecord.refuse_across_inputs(tmp_path, {"model": "M", "track": "T"})
    assert got["model"] == "M"


def test_no_record_is_not_a_refusal(tmp_path):
    """An older run wrote none. Refusing on its absence turns every resume of an existing
    directory into a failure at exactly the moment resuming matters.
    """
    assert runrecord.refuse_across_inputs(tmp_path, {"model": "M", "track": "T"}) is None


def test_a_different_track_is_refused_loudly(tmp_path):
    """The quiet half. `--track` has a default, so a resume that omits it scores new trials on a
    corpus the completed ones never saw, and reports the two as one number.
    """
    runrecord.write(tmp_path, model="M", track="mytrack")
    with pytest.raises(SystemExit) as e:
        runrecord.refuse_across_inputs(tmp_path, {"model": "M", "track": "default"})
    message = str(e.value)
    assert "mytrack" in message and "default" in message
    # And it names both ways out, because "try again" is not one of them.
    assert "--track mytrack" in message
    assert "fresh run" in message


def test_a_different_model_is_refused_loudly(tmp_path):
    runrecord.write(tmp_path, model="Qwen/Qwen3-1.7B", track="T")
    with pytest.raises(SystemExit) as e:
        runrecord.refuse_across_inputs(tmp_path, {"model": "google/gemma-3-270m", "track": "T"})
    assert "Qwen/Qwen3-1.7B" in str(e.value)


def test_a_field_the_record_does_not_carry_is_not_a_change(tmp_path):
    (tmp_path / runrecord.NAME).write_text(json.dumps({"model": "M"}), encoding="utf-8")
    assert runrecord.mismatches(runrecord.read(tmp_path), {"model": "M", "track": "T"}) == []


def test_every_pinned_field_carries_the_reason_it_is_pinned():
    """The refusal quotes these, so an empty one would print a blank parenthesis at the person."""
    for field, why in runrecord.PINNED.items():
        assert why and not why.endswith("."), field


# ── what counts as an occupied directory ─────────────────────────────────────────
def test_the_run_record_does_not_make_a_directory_look_occupied(tmp_path):
    """It is written by the run itself before the search starts. Counting it would make every
    run report its own output directory as already used by a previous one.
    """
    runrecord.write(tmp_path, model="M", track="T")
    assert runrecord.occupied_by(tmp_path) == []
    assert runrecord.NAME not in runrecord.RUN_ARTEFACTS


def test_occupied_by_is_reachable_without_importing_torch():
    """The guided mode calls it, and runs on a base install where torch is not present.

    Reaching it through `cli` imported torch, optuna and transformers to answer a question about
    which files are in a directory, and the guided mode died on an ImportError four questions in.
    """
    import subprocess
    import sys
    probe = ("import sys; from senbonzakura import runrecord; "
             "print(sorted(m for m in ('torch', 'optuna', 'transformers') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         check=True, timeout=120)
    assert out.stdout.strip() == "[]"
