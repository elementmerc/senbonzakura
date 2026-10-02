# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Three commands that could tell their own figure was invalid, and told nobody who could act.

The mechanism was already here and already consumed: `entry.exit_status` turns `self_invalidated`
into a non-zero exit for every entry point at once, `measure`'s table prints "not a measurement"
for a stage that set it, and the checker's adapter reads it off the artefact. Only `compass` ever
set it. So `capability`'s budget verdict, `drift`'s arithmetic and `score --harm-recognition`'s
unreadable replies each printed a correct diagnostic in capitals, wrote a clean looking file and
exited 0, which is the shape of the 2026-08-05 run that came back with five jobs done having
measured nothing.

What is checked here is both halves of every condition, because a refusal that always fires is as
useless as one that never does:

- the planted condition, where the field must be set, the reason must be readable and the exit
  must be non-zero;
- the healthy run beside it, where the field must be absent.

And the line between the two kinds of bad news. A caveat is not a refusal: a drift below the
bfloat16 floor, an accuracy withheld for too few graded items and a comparison whose pairing could
not be verified are all reported beside their number and left usable, because the number is still
a true statement about the model. Each of those has a test here saying so, so that a later change
which escalates one of them to a refusal has to argue with a test rather than with a comment.
"""
import json

import pytest

from senbonzakura import capability, drift, score
from senbonzakura.entry import exit_status
from senbonzakura.metrics import MIN_REPORTABLE_N


# ── capability: the budget verdict, which was printed and never recorded ──────────────
def _summary(verdicts):
    return capability.summarise(verdicts)


def test_a_capability_run_that_graded_nothing_says_so_in_the_field_a_machine_reads():
    reason = capability.not_a_measurement(_summary(["indeterminate"] * 40))
    assert reason
    assert "nothing in this run could be graded" in reason
    assert "--max-new" in reason, "a refusal with no way out is a dead end"


def test_a_capability_run_past_the_budget_ceiling_is_not_a_measurement():
    """41 of 200 ungraded on every arm including the unedited reference is the case that forced
    the ceiling, and a 2.5 point drop was about to be read as a capability cost.
    """
    verdicts = ["correct"] * 150 + ["wrong"] * 9 + ["indeterminate"] * 41
    summary = _summary(verdicts)
    assert summary["budget_suspect"], "the planted condition has to be the one being tested"
    reason = capability.not_a_measurement(summary)
    assert reason and "41 of 200" in reason
    assert "10%" in reason, "the ceiling it crossed, so the reader can see the rule"


def test_a_healthy_capability_run_sets_nothing():
    summary = _summary(["correct"] * 30 + ["wrong"] * 10)
    assert summary["accuracy"] is not None
    assert capability.not_a_measurement(summary) is None


def test_one_ungraded_answer_in_forty_is_a_caveat_and_not_a_refusal():
    """2.5% is under the ceiling, `report` says the accuracy stands, and so does this."""
    summary = _summary(["correct"] * 30 + ["wrong"] * 9 + ["indeterminate"])
    assert summary["indeterminate"] == 1
    assert capability.not_a_measurement(summary) is None


def test_a_rate_withheld_for_too_few_graded_items_is_a_caveat_and_not_a_refusal():
    """`summarise` already withholds the rate and records why, so there is no figure to misquote:
    the counts go out as counts. A run that grades 12 of 12 has measured 12 items, and saying so
    is honest.
    """
    summary = _summary(["correct"] * 8 + ["wrong"] * 4)
    assert summary["accuracy"] is None and summary["accuracy_withheld_because"]
    assert summary["graded"] == 12
    assert capability.not_a_measurement(summary) is None


def test_a_comparison_against_a_reference_that_could_not_grade_itself_is_not_quotable():
    """The pairs that survive are dominated by the reference's easy items, so the change
    understates the cost in the flattering direction, and this exited 0.
    """
    summary = _summary(["correct"] * 30 + ["wrong"] * 10)
    change = {"reference_indeterminate_rate": 0.35, "delta_accuracy": -0.02}
    reason = capability.not_a_measurement(summary, change)
    assert reason and "35.0%" in reason
    assert "accuracy stands" in reason, "the run's own figure is fine and must be said to be"


def test_a_reference_inside_the_ceiling_is_not_refused():
    summary = _summary(["correct"] * 30 + ["wrong"] * 10)
    assert capability.not_a_measurement(
        summary, {"reference_indeterminate_rate": 0.02}) is None


def test_a_reference_that_predates_the_indeterminate_rate_is_not_treated_as_bad():
    """Absent is unknown, which is a different thing from high. Refusing on absent would fire on
    every reference artefact written before the field existed, including this project's own
    published arms, and a refusal that fires on correct historical runs gets switched off.
    """
    summary = _summary(["correct"] * 30 + ["wrong"] * 10)
    assert capability.not_a_measurement(
        summary, {"reference_indeterminate_rate": None, "items_verified": False}) is None


def test_a_change_that_spans_zero_is_a_result_rather_than_an_invalidation():
    summary = _summary(["correct"] * 30 + ["wrong"] * 10)
    assert capability.not_a_measurement(
        summary, {"distinguishable_from_zero": False,
                  "reference_indeterminate_rate": 0.0}) is None


def test_the_capability_reason_is_the_one_the_exit_status_uses():
    """Read off one place, so the status, the artefact and the printed verdict cannot disagree.
    `main` is the integration; this is the contract it rests on.
    """
    bad = _summary(["indeterminate"] * 40)
    assert exit_status({"self_invalidated": capability.not_a_measurement(bad)}) == 1
    good = _summary(["correct"] * 40)
    assert exit_status({"self_invalidated": capability.not_a_measurement(good)}) == 0


# ── score --harm-recognition: COMPASS_UNREADABLE, wired to nothing ────────────────────
def test_unreadable_replies_invalidate_the_recognition_rate():
    reason = score.harm_recognition_validity(200, 0.42)
    assert reason and "42.0%" in reason
    assert "--max-new" in reason


def test_the_recognition_reason_says_why_an_unreadable_reply_is_not_neutral():
    """`harm_recognition_rate` divides by every reply, so an unread reply is counted as a model
    that did not recognise harm: the indeterminate share comes straight off the headline.
    """
    reason = score.harm_recognition_validity(200, 0.42)
    assert "counts against the recognition rate" in reason


def test_a_rate_over_too_few_replies_is_refused_rather_than_caveated():
    """The project's own floor, applied to the one rate that had escaped it: `--harm-recognition
    --n 4` printed a confident percentage over four replies.
    """
    reason = score.harm_recognition_validity(4, 0.0)
    assert reason and str(MIN_REPORTABLE_N) in reason
    assert "--n" in reason


def test_a_healthy_compass_pass_sets_nothing():
    assert score.harm_recognition_validity(200, 0.0) is None
    assert score.harm_recognition_validity(MIN_REPORTABLE_N, 0.05) is None


def test_the_boundaries_are_the_documented_ones_and_not_one_step_either_side():
    """Exactly at the ceiling is reported through, exactly at the floor is reported. Both are
    stated rather than left to whichever way a comparison operator happened to be written.
    """
    assert score.harm_recognition_validity(200, capability.MAX_INDETERMINATE) is None
    assert score.harm_recognition_validity(200, capability.MAX_INDETERMINATE + 1e-9)
    assert score.harm_recognition_validity(MIN_REPORTABLE_N, 0.0) is None
    assert score.harm_recognition_validity(MIN_REPORTABLE_N - 1, 0.0)


def test_both_conditions_are_reported_rather_than_the_first_one_found():
    """A run that is too small AND unreadable has two things wrong with it, and fixing only the
    one it was told about wastes the next run.
    """
    reason = score.harm_recognition_validity(4, 0.9)
    assert "carried no verdict" in reason and "below the floor" in reason


def test_the_compass_ceiling_has_one_home():
    """Two copies of one threshold is how two commands come to disagree about one rule."""
    assert score.MAX_INDETERMINATE is capability.MAX_INDETERMINATE


# ── drift: the arithmetic, which could produce nan and print it as a result ───────────
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_divergence_that_is_not_a_number_is_not_a_measurement(value):
    reason = drift.validity_verdict(value, 64)
    assert reason and "rather than as a number" in reason
    assert "float32" in reason, "the route to a real figure"


def test_a_materially_negative_divergence_means_the_rows_were_not_lined_up():
    """A KL cannot be negative, so a value this far below zero is not resolution noise: it is each
    prompt having been compared against a different prompt, which a stale base cache does.
    """
    reason = drift.validity_verdict(-0.5, 64)
    assert reason and "cannot be negative" in reason
    assert "--base-cache" in reason


def test_a_negative_smaller_than_the_resolution_is_left_alone():
    """Within the floor the arithmetic already states, a tiny negative IS zero, and refusing there
    would fire on the most important legitimate run there is: an edit that changed nothing.
    """
    assert drift.validity_verdict(-1e-9, 64) is None
    assert drift.validity_verdict(-drift.BF16_KL_FLOOR, 64) is None
    assert drift.validity_verdict(0.0, 64) is None


def test_a_drift_over_one_prompt_is_not_a_measurement():
    """No interval can be resampled from one observation, and the interval is the part that says
    how much of a figure is noise. Every headline comparison this project makes turns on drift.
    """
    reason = drift.validity_verdict(0.06, 1)
    assert reason and "--prompts" in reason


def test_two_prompts_is_thin_and_still_a_measurement():
    assert drift.validity_verdict(0.06, 2) is None


@pytest.mark.parametrize("value", [0.0614, 0.065, 0.34, 1.0, 5e-4, 1e-9])
def test_a_healthy_drift_run_sets_nothing(value):
    assert drift.validity_verdict(value, 64) is None


def test_below_the_precision_floor_is_a_caveat_and_not_a_refusal():
    """The two verdicts are deliberately separate. The same edit measured in float32 and bfloat16
    diverges by 0.7% at 0.0007 and by under 0.1% at 0.01, so a figure under the floor is still a
    true statement about the model: it just cannot carry four decimal places. `precision_verdict`
    reports that beside the number; escalating it to a refusal would withhold every one of the
    smallest and best drift figures a run can produce.
    """
    assert drift.precision_verdict(5e-4, "bfloat16")[0] is False
    assert drift.validity_verdict(5e-4, 64) is None


# ── and the same three, through main(), where the artefact is what gets quoted ─────────
class _FakeTok:
    chat_template = "tpl"


def _drift_main(tmp_path, monkeypatch, kl_value, n_prompts=4):
    import torch

    prompts = tmp_path / "p.txt"
    prompts.write_text("\n".join(f"prompt {i}" for i in range(n_prompts)) + "\n",
                       encoding="utf-8")
    monkeypatch.setattr(drift, "load_model_and_tokenizer", lambda *a, **k: (object(), _FakeTok()))
    monkeypatch.setattr(drift, "load_tokenizer", lambda *a, **k: _FakeTok())
    monkeypatch.setattr(drift, "first_token_logprobs",
                        lambda m, t, ps, batch=16, log=None:
                        torch.log(torch.full((len(ps), 2), 0.5)))
    monkeypatch.setattr(drift, "kl", lambda *a: kl_value)
    out = tmp_path / "drift.json"
    res = drift.main(["--model", "cand", "--base", "base", "--prompts", str(prompts),
                      "--out", str(out), "--label", "planted"])
    return res, json.loads(out.read_text(encoding="utf-8"))


def test_a_planted_nan_reaches_the_file_the_number_gets_quoted_out_of(tmp_path, monkeypatch,
                                                                     capsys):
    res, doc = _drift_main(tmp_path, monkeypatch, float("nan"))
    assert doc["self_invalidated"], "the terminal is not where a verdict is kept"
    assert doc["self_invalidated"] == res["self_invalidated"]
    assert exit_status(res) == 1
    said = capsys.readouterr().out
    assert "DRIFT_NOT_A_MEASUREMENT" in said
    assert str(tmp_path / "drift.json") in said, "a diagnostic with no path is not a diagnostic"


def test_a_healthy_drift_run_writes_no_such_claim(tmp_path, monkeypatch):
    res, doc = _drift_main(tmp_path, monkeypatch, 0.0614)
    assert "self_invalidated" not in doc
    assert exit_status(res) == 0


def test_a_compass_pass_of_unreadable_replies_records_it(monkeypatch, tiny_model, tiny_tok,
                                                         tmp_path, capsys):
    from datasets import Dataset

    ev = str(tmp_path / "eval")
    Dataset.from_dict({"text": [f"bad thing {i}" for i in range(4)]}).save_to_disk(ev)
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    tiny_tok._decode_text = "let me think about this at length and never reach a conclusion"
    out = tmp_path / "r.json"
    res = score.main(["--model", "x", "--eval", ev, "--out", str(out), "--harm-recognition",
                      "--max-new", "3", "--device", "cpu"])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["self_invalidated"] == res["self_invalidated"]
    assert exit_status(res) == 1
    said = capsys.readouterr().out
    assert "COMPASS_NOT_A_MEASUREMENT" in said and str(out) in said


def test_a_compass_pass_over_enough_readable_replies_records_nothing(monkeypatch, tiny_model,
                                                                     tiny_tok, tmp_path):
    from datasets import Dataset

    ev = str(tmp_path / "eval")
    Dataset.from_dict({"text": [f"bad thing {i}" for i in range(MIN_REPORTABLE_N)]}).save_to_disk(ev)
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    tiny_tok._decode_text = "HARMFUL"
    out = tmp_path / "r.json"
    res = score.main(["--model", "x", "--eval", ev, "--out", str(out), "--harm-recognition",
                      "--max-new", "3", "--device", "cpu", "--batch", "8"])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert res["harm_recognition"] == 1.0
    assert "self_invalidated" not in doc
    assert exit_status(res) == 0


def test_a_capability_run_that_graded_nothing_carries_the_reason_into_its_artefact(
        tmp_path, monkeypatch, tiny_model, tiny_tok, capsys):
    import senbonzakura.cli as _cli

    monkeypatch.setattr(_cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    monkeypatch.setattr(_cli, "render_chat", lambda _tok, p: p)
    bench = tmp_path / "bench.jsonl"
    bench.write_text("".join(
        json.dumps({"question": f"q{i}", "answer": f"#### {i}"}) + "\n" for i in range(3)),
        encoding="utf-8")
    out = tmp_path / "cap.json"
    code = capability.main(["--model", "m", "--device", "cpu", "--eval", str(bench), "--n", "3",
                            "--max-new", "4", "--out", str(out)])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1
    assert doc["self_invalidated"], "the exit status was right and the file said nothing"
    assert "nothing in this run could be graded" in doc["self_invalidated"]
    assert "CAPABILITY_NOT_A_MEASUREMENT" in capsys.readouterr().out
