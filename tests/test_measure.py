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
    # Not a credential: a string this test watches for, to prove it never reaches the log.
    argv = measure.stage_argv("score", _args(hf_token="hf_secret"), Path("out"))  # noqa: S106
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
    assert "out of memory" in rows[0][3]


def test_the_figure_is_read_from_the_stamp_the_command_writes():
    rows = measure.verdict_rows({
        "score": {"metrics": {"refusal_rate.senbonzakura-ruler": {"value": 0.031}}}})
    assert rows[0][1] == "0.031"


def test_the_table_never_prints_a_verdict():
    """Withdrawn numbers are this project's history. A green tick over four instruments is the
    artefact that invites somebody to quote a result they have not read.
    """
    rendered = " ".join(measure.format_table(measure.verdict_rows({
        "score": {"metrics": {"refusal_rate.senbonzakura-ruler": {"value": 0.0}}},
        "coherence": {"metrics": {"coherence": {"value": 11.2}}}})))
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
            raise measure.StageError("CUDA out of memory")
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

def test_every_reading_names_a_metric_its_command_actually_stamps():
    """THE GUARD THAT WAS NOT STRONG ENOUGH, twice, and this is the second version.

    The first read a plain top-level key and asserted only that the string appeared SOMEWHERE in
    the module that writes the file. It caught `drift` being read for a `delta_ppl` that module
    never mentions, and then passed `capability`'s `accuracy`, which that module does write, one
    level down under `summary`. So a real run on the ROG printed "not reported" for a model that
    had scored 29 of 39 in the log two lines above.

    This asserts against the stamp instead: `measurement.stamp(result, "<name>", ...)` is the one
    call that puts a figure in the `metrics` block, and every reading here has to name something
    a stage stamps. The names with a dot in them are estimator-qualified and are built rather
    than written literally, so the prefix is what can be checked; that is stated here rather than
    left as a hole somebody finds later.
    """
    import ast
    import pathlib

    pkg = pathlib.Path(measure.__file__).parent
    for name, (metric, _note) in measure.READINGS.items():
        source = (pkg / f"{DELEGATED[name][0]}.py").read_text(encoding="utf-8")
        stamped = set()
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "stamp" and len(node.args) >= 2):
                continue
            key = node.args[1]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                stamped.add(key.value)
        assert stamped, f"{name} stamps nothing, so `measure` can read no figure out of it"
        assert metric.split(".")[0] in stamped, (
            f"`measure` reads {metric!r} out of {name}, and that module stamps "
            f"{sorted(stamped)}. The table would carry a blank row and nothing would fail.")


def test_a_reading_that_finds_nothing_says_what_the_file_does_carry():
    """A row saying only "not reported" beside a stage that printed its number is useless. The
    first version did exactly that for `capability` and nothing in the output pointed anywhere.
    """
    rows = measure.verdict_rows({"score": {"metrics": {"something_else": {"value": 1}}}})
    assert rows[0][1] == "not reported"
    assert "something_else" in rows[0][3], (
        "the row has to name what the file does carry, or the reader is left opening it by hand")


def test_a_result_with_no_metrics_block_is_named_as_such():
    rows = measure.verdict_rows({"score": {"refusal": 0.2}})
    assert rows[0][1] == "not reported"
    assert "metrics" in rows[0][3]


def test_a_stamped_figure_is_read_from_its_value():
    rows = measure.verdict_rows({
        "coherence": {"metrics": {"coherence": {"value": 13.6137, "n": 268}}}})
    assert rows[0][1] == "13.6137"


# ── the table's numbers fit in the table ─────────────────────────────────────────

def test_a_long_float_is_cut_to_something_a_column_can_hold():
    """FOUND BY RUNNING IT on the ROG, 2026-09-23. `coherence` reported
    `13.613728595914115` beside a refusal rate of `0.2083`, which is fifteen decimal places of a
    perplexity nobody can use and a column that no longer lines up.
    """
    rows = measure.verdict_rows({
        "coherence": {"metrics": {"coherence": {"value": 13.613728595914115}}}})
    assert rows[0][1] == "13.6137"


def test_a_round_number_does_not_grow_a_tail_of_zeros():
    def _row(v):
        return measure.verdict_rows(
            {"score": {"metrics": {"refusal_rate.senbonzakura-ruler": {"value": v}}}})[0][1]

    assert _row(0.0) == "0"
    assert _row(0.5) == "0.5"


def test_a_non_number_is_left_alone():
    """Some stages report a string where a figure would be, and inventing a format for it would
    be this module deciding what another command meant.
    """
    assert measure.verdict_rows({
        "score": {"metrics": {"refusal_rate.senbonzakura-ruler": {"value": "withheld"}}}
    })[0][1] == "withheld"


def test_the_column_lines_up_across_mixed_magnitudes():
    lines = measure.format_table(measure.verdict_rows({
        "score": {"metrics": {"refusal_rate.senbonzakura-ruler": {"value": 0.2083}}},
        "coherence": {"metrics": {"coherence": {"value": 13.613728595914115}}}}))
    ends = {line.index("   ", line.index(line.split()[1])) for line in lines}
    assert len(ends) == 1, "the figures column is ragged:\n" + "\n".join(lines)


def test_the_units_come_out_of_the_stamp_and_not_out_of_this_module():
    """A row that names its own units can be wrong about them, and one was: the coherence row
    said "perplexity" over a value that is the log likelihood. The stamp declares
    `nats-per-token`, and that is what the table now shows beside the number.
    """
    rows = measure.verdict_rows({"coherence": {"metrics": {"coherence": {
        "value": 2.6111, "units": "nats-per-token"}}}})
    assert rows[0][2] == "nats-per-token"
    assert "perplexity" not in rows[0][3].split(".")[0], (
        "the first clause of the note calls the figure a perplexity, and the file says otherwise")


def test_a_stamp_without_units_leaves_the_column_empty_rather_than_guessing():
    rows = measure.verdict_rows({"coherence": {"metrics": {"coherence": {"value": 2.6}}}})
    assert rows[0][2] == ""


def test_the_rendered_table_puts_the_units_beside_the_number():
    line, = measure.format_table(measure.verdict_rows({
        "coherence": {"metrics": {"coherence": {"value": 2.6111,
                                                "units": "nats-per-token"}}}}))
    assert "2.6111  nats-per-token" in line


def test_every_row_has_the_same_shape_whatever_happened_to_the_stage():
    """THE DEFECT THIS CAUGHT. When the rows grew a units column, the failure branch kept
    appending three fields, so `format_table` unpacking four would raise on any run where an
    instrument failed: the exact run where the table matters most.

    Asserted on the shape rather than on one branch, so the next column added cannot reintroduce
    it in whichever branch is forgotten next.
    """
    rows = measure.verdict_rows({
        "score": "CUDA out of memory",                                      # failed outright
        "compass": {"metrics": {}},                                         # wrote nothing usable
        "coherence": {"metrics": {"coherence": {"value": 2.6, "units": "nats-per-token"}}},
    })
    assert len(rows) == 3
    # AGAINST THE ROW TYPE, not against a number typed here. This asserted `== {4}` and
    # broke when the interval column was added, which is the same fragility the rows
    # themselves had: a shape written down in two places disagrees the moment one moves.
    assert {len(r) for r in rows} == {len(measure.Row._fields)}, (
        f"ragged rows: {[len(r) for r in rows]}")
    measure.format_table(rows)      # raises if any row disagrees with the rest


# ── the dispatch, and what it does with a stage that raises ──────────────────────

def test_a_stage_is_run_through_the_module_that_owns_the_command(monkeypatch):
    """In process, through `entry.DELEGATED`, so a command that moves keeps being run."""
    import sys
    import types as _types

    seen = {}
    fake = _types.ModuleType("senbonzakura.score")
    fake.main = lambda argv: seen.setdefault("argv", argv) or 0
    monkeypatch.setitem(sys.modules, "senbonzakura.score", fake)
    measure.run_stage("score", ["--out", "x.json"], log=lambda _m: None)
    assert seen["argv"] == ["--out", "x.json"]


def test_a_refusal_from_a_stage_becomes_one_failed_stage(monkeypatch):
    """Several commands say no with `SystemExit`. Left uncaught it would end the whole run, so
    the other four instruments would report nothing because one of them declined.
    """
    import sys
    import types as _types

    fake = _types.ModuleType("senbonzakura.score")

    def _refuse(_argv):
        raise SystemExit("that track does not exist")

    fake.main = _refuse
    monkeypatch.setitem(sys.modules, "senbonzakura.score", fake)
    with pytest.raises(measure.StageError, match="that track does not exist"):
        measure.run_stage("score", [], log=lambda _m: None)


def test_a_bare_exit_status_still_produces_a_readable_reason(monkeypatch):
    import sys
    import types as _types

    fake = _types.ModuleType("senbonzakura.score")

    def _exit(_argv):
        raise SystemExit(3)

    fake.main = _exit
    monkeypatch.setitem(sys.modules, "senbonzakura.score", fake)
    with pytest.raises(measure.StageError, match="exited 3"):
        measure.run_stage("score", [], log=lambda _m: None)


def test_any_other_exception_is_named_by_its_type(monkeypatch):
    """Deliberately broad: whatever one instrument does wrong, the other four still have numbers.
    The type is kept because "RuntimeError: CUDA out of memory" is actionable and "failed" is not.
    """
    import sys
    import types as _types

    fake = _types.ModuleType("senbonzakura.score")

    def _boom(_argv):
        raise RuntimeError("CUDA out of memory")

    fake.main = _boom
    monkeypatch.setitem(sys.modules, "senbonzakura.score", fake)
    with pytest.raises(measure.StageError, match="RuntimeError: CUDA out of memory"):
        measure.run_stage("score", [], log=lambda _m: None)


# ── the command end to end, with nothing loaded ──────────────────────────────────

def test_main_writes_the_summary_and_reports_success(tmp_path, monkeypatch, capsys):
    def _fake(name, _argv, **_k):
        (tmp_path / measure.OUTPUTS[name]).write_text(json.dumps(
            {"metrics": {measure.READINGS[name][0]: {"value": 0.5, "units": "proportion"}}}))
        return 0

    monkeypatch.setattr(measure, "run_stage", _fake)
    rc = measure.main(["Qwen/Qwen3-1.7B", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0, out
    summary = json.loads((tmp_path / "measure.json").read_text())
    assert summary["model"] == "Qwen/Qwen3-1.7B"
    assert summary["failed"] == []
    assert "commands" in summary, "a reader has to be able to re-run any single stage"
    assert "None of this is a pass or a fail" in out, (
        "the closing caveat is the thing that stops a reader quoting this as a verdict")


def test_main_reports_non_zero_when_an_instrument_produced_nothing(tmp_path, monkeypatch, capsys):
    """A script gating on this has to be able to tell. The table still prints the rest."""
    def _fake(name, _argv, **_k):
        if name == "score":
            raise measure.StageError("CUDA out of memory")
        (tmp_path / measure.OUTPUTS[name]).write_text(json.dumps(
            {"metrics": {measure.READINGS[name][0]: {"value": 0.5}}}))
        return 0

    monkeypatch.setattr(measure, "run_stage", _fake)
    rc = measure.main(["Qwen/Qwen3-1.7B", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "score" in json.loads((tmp_path / "measure.json").read_text())["failed"]
    assert "compass" in out, "one failure must not hide the instruments that worked"


def test_the_token_is_not_written_into_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(measure, "run_stage", lambda name, _a, **_k: (
        (tmp_path / measure.OUTPUTS[name]).write_text("{}") or 0))
    measure.main(["m", "--out", str(tmp_path), "--hf-token", "hf_secret"])
    assert "hf_secret" not in (tmp_path / "measure.json").read_text()


def test_a_stage_that_invalidated_its_own_figure_is_not_reported_as_a_number():
    """THE WORSE INSTANCE OF THE SAME DEFECT, because this table exists to be quoted.

    A first-time reader on 2026-09-26 got `THE AUC ABOVE IS NOT A MEASUREMENT OF HARM
    DISCRIMINATION` from the compass and exit 0. That was fixed at the shell, through
    `entry.exit_status`. `measure` calls each stage in process and returns what its `main` returns,
    so it never passes through `exit_status` and would have printed that AUC into the one artefact
    the documentation tells a reader to quote from, with a partition caption beside it and no hint
    that the producing stage had disowned the number.
    """
    res = {"compass": {"self_invalidated": True,
                       "metrics": {"compass_auc.margin-past-preamble": {"value": 0.7109,
                                                                       "units": ""}}}}
    row = measure.verdict_rows(res)[0]
    assert row.figure == "not a measurement", (
        f"the table reports {row.figure!r} for a stage that recorded its own figure is not a "
        f"measurement")
    assert "0.7109" not in row.figure
    assert "not a measurement of" in row.note


def test_an_ordinary_stage_is_still_reported_as_a_number():
    """The guard must not swallow every figure: `self_invalidated` absent or false is the normal
    case and is what every good run writes.
    """
    for marker in ({}, {"self_invalidated": False}, {"self_invalidated": None}):
        res = {"compass": {**marker,
                           "metrics": {"compass_auc.margin-past-preamble": {"value": 0.9887,
                                                                           "units": ""}}}}
        row = measure.verdict_rows(res)[0]
        assert row.figure == "0.9887", f"a good run reported {row.figure!r} with marker {marker}"
