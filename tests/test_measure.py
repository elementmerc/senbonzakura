# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura measure` builds five command lines, and every one of them has to parse.

WHY THIS IS THE CENTRAL ASSERTION

`measure` does not measure anything itself. It constructs the command line each instrument would
have been given by hand and hands it to that instrument's own `main`. That is the whole design:
one implementation of each number, so a figure from `measure` and a figure from the command it
wrapped cannot disagree.

The failure it creates is a flag that no longer exists, or never did. Built by hand and never
parsed, `measure` would load a model, spend an hour, and die on the third stage with
`unrecognized arguments`. So every constructed line is parsed here, by the real parser, against
the real command, with no model and no GPU.

That is not the same claim as "the run succeeds", which needs hardware, and nothing in this file
pretends otherwise.
"""
from __future__ import annotations

import importlib
import json
import types
from pathlib import Path

import pytest

from senbonzakura import measure
from senbonzakura.entry import DELEGATED


def _args(**over):
    a = dict(model="Qwen/Qwen3-1.7B", device="cuda", hf_token=None, trust_remote_code=False,
             track="default", out="measurements", baseline=None, n=200, capability_n=40,
             only=None, force=False, dry_run=False)
    a.update(over)
    return types.SimpleNamespace(**a)


# ── the command lines ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", measure.STAGE_ORDER)
def test_every_stage_line_parses_against_the_command_that_owns_it(name):
    """THE ASSERTION. A flag that does not exist fails here rather than an hour into a run."""
    argv = measure.stage_argv(name, _args(baseline="Qwen/Qwen3-1.7B"), Path("out"))
    module = importlib.import_module(f"senbonzakura.{DELEGATED[name][0]}")
    module.build_parser().parse_args(argv)          # SystemExit on an unknown flag


@pytest.mark.parametrize("name", measure.STAGE_ORDER)
def test_every_stage_writes_into_the_output_directory_it_was_given(name):
    argv = measure.stage_argv(name, _args(baseline="b"), Path("somewhere"))
    out = argv[argv.index("--out") + 1]
    assert out == str(Path("somewhere") / measure.OUTPUTS[name]), (
        f"{name} would write to {out}, outside the directory the user named")


def test_every_stage_is_dispatched_through_the_command_registry():
    """Named through `entry.DELEGATED` rather than hardcoded, so a command that moves to another
    module keeps being run instead of falling out of this quietly.
    """
    for name in measure.STAGE_ORDER:
        assert name in DELEGATED, f"{name} is not a command, so nothing can run it"


def test_every_stage_has_a_reading_and_an_output_file():
    """A stage nobody can read a number out of contributes a blank row to the table."""
    for name in measure.STAGE_ORDER:
        assert name in measure.READINGS, f"{name} produces a file and no reading of it"
        assert name in measure.OUTPUTS, f"{name} has no output filename"


# ── the harmful prompts this command does NOT leave on disk ──────────────────────

def test_the_compass_stage_does_not_keep_per_prompt_rows():
    """`compass` writes the prompt text and the reply beside each margin, and those prompts are
    harmful. Somebody running one command to find out whether their model is broken has not asked
    for that file. `senbonzakura compass` still writes it when they do ask.
    """
    argv = measure.stage_argv("compass", _args(), Path("out"))
    assert "--no-margins" in argv


def test_the_token_never_reaches_the_printed_line_or_the_summary():
    argv = measure.stage_argv("score", _args(hf_token="hf_secret"), Path("out"))
    assert "hf_secret" in argv, "the stage itself needs the token, or a gated model fails"
    assert "hf_secret" not in measure._shown(argv, "hf_secret"), (
        "the token survives into what is printed and into measure.json")


# ── which stages run ─────────────────────────────────────────────────────────────

def test_drift_is_left_out_when_there_is_nothing_to_compare_against():
    assert "drift" not in measure.wanted_stages(_args())


def test_drift_is_added_when_a_baseline_is_named():
    assert "drift" in measure.wanted_stages(_args(baseline="Qwen/Qwen3-1.7B"))


def test_asking_for_drift_without_a_baseline_is_refused_rather_than_ignored():
    """Somebody who typed `--only drift` has said what they want. Silently running nothing would
    leave them reading an empty table for a reason the tool knew and did not say.
    """
    with pytest.raises(SystemExit, match="--baseline"):
        measure.wanted_stages(_args(only=["drift"]))


def test_only_keeps_the_declared_order():
    chosen = measure.wanted_stages(_args(only=["capability", "score"]))
    assert chosen == ["score", "capability"]


# ── the table ────────────────────────────────────────────────────────────────────

def test_a_failed_stage_is_named_in_the_table_rather_than_dropped():
    rows = measure.verdict_rows({"score": "CUDA out of memory"})
    assert rows and rows[0][1] == "not measured"
    assert "out of memory" in rows[0][2]


def test_a_stage_that_wrote_a_file_without_the_figure_says_so():
    rows = measure.verdict_rows({"score": {"label": "x"}})
    assert rows[0][1] == "not reported"


def test_the_figure_is_read_from_the_key_the_command_actually_writes():
    rows = measure.verdict_rows({"score": {"refusal": 0.031}})
    assert rows[0][1] == "0.031"


def test_the_table_never_prints_a_verdict():
    """This project has withdrawn published numbers. A green tick over four instruments is the
    artefact that invites somebody to quote a result they have not read.
    """
    rendered = " ".join(measure.format_table(
        measure.verdict_rows({"score": {"refusal": 0.0}, "coherence": {"ppl": 11.2}})))
    for word in ("PASS", "FAIL", "OK", "GOOD", "SAFE", "✓"):
        assert word not in rendered, f"the table renders a verdict ({word}) rather than a figure"


# ── resuming ─────────────────────────────────────────────────────────────────────

def test_a_stage_whose_file_is_already_there_is_not_re_run(tmp_path, monkeypatch):
    (tmp_path / measure.OUTPUTS["score"]).write_text(json.dumps({"refusal": 0.5}))
    ran = []
    monkeypatch.setattr(measure, "run_stage", lambda *a, **k: ran.append(a[0]))
    results, failures = measure.run(_args(out=str(tmp_path), only=["score"]), log=lambda _m: None)
    assert not ran, "an hour of generation was repeated over a result that was already there"
    assert results["score"]["refusal"] == 0.5
    assert not failures


def test_force_re_runs_it(tmp_path, monkeypatch):
    (tmp_path / measure.OUTPUTS["score"]).write_text(json.dumps({"refusal": 0.5}))
    ran = []
    monkeypatch.setattr(measure, "run_stage", lambda *a, **k: ran.append(a[0]))
    measure.run(_args(out=str(tmp_path), only=["score"], force=True), log=lambda _m: None)
    assert ran == ["score"]


def test_one_stage_failing_does_not_stop_the_others(tmp_path, monkeypatch):
    """A model too large for the card fails `capability` and still has a refusal rate. Reporting
    three numbers and a named failure is more use than reporting nothing.
    """
    def _fake(name, _argv, **_k):
        if name == "score":
            raise measure.StageFailed("CUDA out of memory")
        (tmp_path / measure.OUTPUTS[name]).write_text(json.dumps({"ppl": 11.2}))

    monkeypatch.setattr(measure, "run_stage", _fake)
    results, failures = measure.run(
        _args(out=str(tmp_path), only=["score", "coherence"]), log=lambda _m: None)
    assert failures == ["score"]
    assert results["coherence"]["ppl"] == 11.2


def test_a_stage_that_claims_success_and_writes_nothing_is_a_failure(tmp_path, monkeypatch):
    """The shape this project has been caught by twice: a job reporting success having measured
    nothing, because the only thing checked was that it did not raise.
    """
    monkeypatch.setattr(measure, "run_stage", lambda *a, **k: 0)
    results, failures = measure.run(_args(out=str(tmp_path), only=["score"]), log=lambda _m: None)
    assert failures == ["score"]
    assert "wrote no file" in results["score"]


# ── the dry run ──────────────────────────────────────────────────────────────────

def test_the_dry_run_prints_the_lines_and_loads_nothing(capsys, tmp_path):
    assert measure.main([
        "Qwen/Qwen3-1.7B", "--dry-run", "--out", str(tmp_path), "--hf-token", "hf_secret"]) == 0
    out = capsys.readouterr().out
    assert "senbonzakura score" in out and "senbonzakura capability" in out
    assert "hf_secret" not in out
    assert not list(tmp_path.iterdir()), "a dry run wrote something"


# ── the readings name keys the commands actually write ───────────────────────────

def test_every_reading_names_a_key_its_command_writes():
    """THE MISTAKE THIS CAUGHT, on the day it was written.

    `drift` was read for `delta_ppl`, which it has never written: it reports a KL divergence, and
    the table would have carried a blank row for it forever while every other stage looked fine.
    Nothing else would have failed, because a missing key reads as "not reported" by design.

    Checked against the source rather than against a run, because a run needs a GPU and this
    class of error is a typo. It is a floor, not a proof: it says the key exists in the module
    that writes the file, not that it is the right one to quote.
    """
    import ast
    import pathlib

    pkg = pathlib.Path(measure.__file__).parent
    for name, (key, _note) in measure.READINGS.items():
        source = (pkg / f"{DELEGATED[name][0]}.py").read_text(encoding="utf-8")
        written = {n.value for n in ast.walk(ast.parse(source))
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert key in written, (
            f"`measure` reads {key!r} out of {name}'s result file and that module never writes "
            f"the string. The table would carry a blank row and nothing would fail")
