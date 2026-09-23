# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Verdicts that were computed correctly and then acted on by nothing.

THE SHAPE

The 2026-09-09 panel had four personas independently reach the same failure: a check that is
written, correct, runs and passes, with nothing wired between its answer and any consequence. It is
invisible to a reviewer asking "is this correct?", because the answer is always yes, and invisible
to a suite where every test asks whether a detector computes the right answer and none asks whether
firing one changes an outcome.

Three of them are covered here, and they are the same defect in three commands:

  * `validate` computed `reach.architectures_failing` and `e4.degenerate_reason` and returned a
    literal 0 for every outcome it could reach. `reach` is the check that answers "did the edit
    touch the residual stream at all", the question that withdrew a family of numbers when it was
    answered wrongly on Gemma.
  * The compass printed "THE AUC ABOVE IS NOT A MEASUREMENT OF HARM DISCRIMINATION on this run"
    and exited 0. Its `suspect` key was read nowhere outside its own module.
  * `headtohead.score_arms` decided an arm was scored on `exit == 0 and the file exists`, so the
    suspect arm was recorded as fine and its AUC went into the table under the heading "the axis
    where a comparison means something".

WHAT THESE TESTS ARE FOR, WHICH IS NOT THE SAME AS WHAT THE VERDICTS ARE FOR

None of these asserts that a verdict is computed correctly; other files do that, and they always
did, which is exactly why the gap survived. Every test here asserts that reaching a verdict CHANGES
something a caller can observe.
"""
import json

import pytest

from senbonzakura import headtohead, margin, validate


# ── validate: a bad instrument is not the same thing as an unwelcome result ───────────
def test_a_record_with_a_clean_reach_and_a_readable_grid_is_usable():
    assert validate.unusable_reasons(
        {"reach": {"architectures_failing": []}, "e4": {"degenerate_reason": None}}) == []


def test_an_edit_that_never_reached_the_stream_makes_the_record_unusable():
    reasons = validate.unusable_reasons({"reach": {"architectures_failing": ["conv", "attn"]}})
    assert len(reasons) == 1
    assert "conv, attn" in reasons[0]
    assert "did not reach" in reasons[0]


def test_a_grid_with_no_spread_makes_the_record_unusable():
    reasons = validate.unusable_reasons({"e4": {"degenerate_reason": "no fitted arms"}})
    assert reasons and "no fitted arms" in reasons[0]


def test_both_faults_are_reported_together_rather_than_the_first_one_only():
    reasons = validate.unusable_reasons(
        {"reach": {"architectures_failing": ["conv"]}, "e4": {"degenerate_reason": "flat"}})
    assert len(reasons) == 2, "a caller fixing one should already know about the other"


@pytest.mark.parametrize("verdict", [
    "the extra directions do NOT generalise: on held-out clusters they score no better than random",
    "mixed: the extra directions beat random on some held-out clusters and not others",
])
def test_an_experiment_answering_no_is_a_result_and_not_a_failure(verdict):
    """THE DISTINCTION THE OBVIOUS FIX GETS BACKWARDS.

    A command that reports failure whenever an experiment disagrees with us is one nobody can use
    to find anything out. e1 is allowed to say the directions do not generalise; that is the
    experiment working. Only an instrument that could not take a reading earns a non-zero status.
    """
    assert validate.unusable_reasons({"e1": {"verdict": verdict}}) == []


def test_a_record_missing_an_experiment_entirely_is_not_called_unusable():
    """`--experiment e4` writes no `reach` key at all, and absent is not the same as failing."""
    assert validate.unusable_reasons({"experiment": "e4", "e4": {"degenerate_reason": None}}) == []
    assert validate.unusable_reasons({}) == []


# ── the compass: one predicate, because two things have to act on it ──────────────────
def _compass(harmful_suspect=False, harmless_suspect=False):
    return {"readout": {"harmful": {"suspect": harmful_suspect, "verdict_prob_mass_mean": 0.4,
                                    "argmax_is_verdict": 0.9},
                        "harmless": {"suspect": harmless_suspect, "verdict_prob_mass_mean": 0.4,
                                     "argmax_is_verdict": 0.9},
                        "position": "last"}}


def test_a_sound_readout_names_no_suspect_arms():
    assert margin.suspect_readout_arms(_compass()) == []


@pytest.mark.parametrize(("h", "l", "want"), [
    (True, False, ["harmful"]),
    (False, True, ["harmless"]),
    (True, True, ["harmful", "harmless"]),
])
def test_each_arm_is_reported_on_its_own(h, l, want):
    assert margin.suspect_readout_arms(_compass(h, l)) == want


def test_a_record_with_no_readout_at_all_is_not_treated_as_suspect():
    """Takes a parsed artefact as well as an in-process result, so it meets partial records."""
    assert margin.suspect_readout_arms({}) == []
    assert margin.suspect_readout_arms(None) == []
    assert margin.suspect_readout_arms({"readout": {"harmful": None, "harmless": None}}) == []


# ── the benchmark: the consequence that was missing ───────────────────────────────────
class _Arm:
    def __init__(self, tool="senbon", seed=1):
        self.tool, self.seed = tool, seed


def _scored(tmp_path, artefact, *, code=0):
    """Run score_arms against a fake runner that writes `artefact` as the compass output."""
    arm = tmp_path / "senbon-seed1"
    (arm / "model").mkdir(parents=True)
    (arm / "model" / "config.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "scores"
    out.mkdir()

    def runner(argv, log=None):
        target = out / "scored-senbon-seed1.json"
        target.write_text(json.dumps(artefact), encoding="utf-8")
        return code

    r = _Arm()
    r.arm = arm
    return headtohead.score_arms([r], harmful=tmp_path / "h", harmless=tmp_path / "l",
                                 out=out, runner=runner, log=lambda _m: None)


def test_a_suspect_compass_arm_is_not_recorded_as_scored(tmp_path, monkeypatch):
    """THE DEFECT ITSELF: exit 0 plus a file on disk used to be enough.

    The compass wrote its own verdict that the number is not a measurement of harm discrimination,
    into this very file, and the benchmark read the exit code instead.
    """
    monkeypatch.setattr(headtohead, "arm_model_dir", lambda r, a: r.arm / "model")
    scored = _scored(tmp_path, _compass(harmful_suspect=True))
    assert scored[0]["ok"] is False
    assert "not a measurement of harm discrimination" in scored[0]["reason"]
    assert scored[0]["path"] is None, "an unusable score must not be handed on as a path to read"


def test_a_sound_compass_arm_is_still_recorded_as_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(headtohead, "arm_model_dir", lambda r, a: r.arm / "model")
    scored = _scored(tmp_path, _compass())
    assert scored[0]["ok"] is True
    assert scored[0]["path"] is not None


def test_a_score_file_that_cannot_be_read_fails_only_its_own_arm(tmp_path, monkeypatch):
    """A damaged artefact is this arm's failure, not the sweep's.

    `_read_json` raises on a damaged file, which is right, and letting that escape here would take
    down the scoring of every arm behind it over one bad file.
    """
    monkeypatch.setattr(headtohead, "arm_model_dir", lambda r, a: r.arm / "model")
    arm = tmp_path / "senbon-seed1"
    (arm / "model").mkdir(parents=True)
    (arm / "model" / "config.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "scores"
    out.mkdir()

    def runner(argv, log=None):
        (out / "scored-senbon-seed1.json").write_text("{not json", encoding="utf-8")
        return 0

    r = _Arm()
    r.arm = arm
    scored = headtohead.score_arms([r], harmful=tmp_path / "h", harmless=tmp_path / "l",
                                   out=out, runner=runner, log=lambda _m: None)
    assert scored[0]["ok"] is False


# ── S4: a probe that could not reach the decision it exists to inform ─────────────────
def test_the_capability_probe_no_longer_depends_on_the_rescore(base_args, tiny_model,
                                                               tiny_tok, track,
                                                               monkeypatch):
    """The old assertion here was the exact opposite, and the inversion is the point.

    It asserted that `--capability-eval` without `--eval-refusal-final` was REFUSED, because the
    probe measured what each finalist cost and the finalists only existed inside the best-of-N
    re-score. That was a true description of where the probe ran, and the refusal was the right
    response to it: the run would otherwise have paid for a baseline in generations and been
    unable to move the shipped model by a single trial.

    On 2026-09-22 the probe moved. It now runs in `_bake_and_save`, on the weights that ship, for
    every method including the ones that pin a profile and never search at all, so it depends on
    nothing this flag controls. Leaving the guard in place would have been actively harmful once
    the probe became the default: every run that had not set `--eval-refusal-final` would have
    refused to start over a coupling that no longer exists.

    So what is asserted now is that the run gets PAST that point. It still fails, because the
    benchmark named here cannot be fetched, and that is the correct refusal: a probe whose
    benchmark cannot be read would otherwise score every candidate identically.
    """
    from senbonzakura import cli

    base_args.track = track
    base_args.capability_eval = "some/benchmark::test"
    base_args.eval_refusal_final = 0
    real = cli.Abliterator
    monkeypatch.setattr(cli, "Abliterator",
                        lambda args, log: real(args, log, model=tiny_model, tok=tiny_tok))
    with pytest.raises(SystemExit) as e:
        cli.Abliterator(base_args, lambda _m: None).run()
    message = str(e.value)
    assert "--eval-refusal-final" not in message, (
        "the run was refused over the old coupling between the probe and the re-score pass. That "
        "coupling is gone, and with the probe on by default this refusal would fire on almost "
        "every run")
    assert "cannot be read" in message, (
        f"expected the run to reach the probe and fail on the unfetchable benchmark, and it said: "
        f"{message}")


def test_a_probe_whose_benchmark_cannot_be_read_refuses_and_says_which(base_args, tiny_model,
                                                                       tiny_tok, track,
                                                                       monkeypatch):
    """A probe that cannot load its benchmark must refuse rather than measure nothing.

    THE PAIR THIS BELONGED TO IS GONE. It asserted that the S4 gate did not fire when the re-score
    pass could reach the probe, and that gate no longer exists, so the assertion had nothing left
    to be about. What survives is the behaviour underneath it: an unreadable benchmark is refused
    loudly, because a probe with nothing behind it would score every candidate identically and the
    run would report that as no capability cost.

    Its old body also carried a latent bug that only surfaced when this path started raising.
    `pytest.raises` hands back an ExceptionInfo, which has `.value`; a caught exception does not,
    and `str(e.value)` on a SystemExit raises AttributeError. The branch had never executed. The
    comment above it describes fixing an earlier version that made every failure read as a pass,
    which is worth keeping in view: the repair was real and the repaired branch was still wrong.
    """
    from senbonzakura import cli

    base_args.track = track
    base_args.capability_eval = "some/benchmark::test"
    base_args.eval_refusal_final = 128
    real = cli.Abliterator
    monkeypatch.setattr(cli, "Abliterator",
                        lambda args, log: real(args, log, model=tiny_model, tok=tiny_tok))
    # NOT `except Exception: refusal = None`. That is what this used to do, and it made any
    # failure at all read as a pass: mutating `Abliterator.run` to raise immediately left this
    # test green, so it could not distinguish "the gate correctly let the run through" from "the
    # run never reached the gate". Other failures are still this fixture's business rather than
    # this test's subject, so they are allowed, but they are allowed EXPLICITLY and the run has to
    # be shown to have got past the gate under test.
    # The gate under test raises SystemExit. So the ONLY two outcomes that show it was reached
    # and did not fire are a clean run and a SystemExit saying something else. Anything else means
    # the run died before the gate could have run, and this test proved nothing.
    #
    # It used to catch bare `Exception` and set `refusal = None`, which made every failure read as
    # a pass: mutating `Abliterator.run` to raise on its first line left it green.
    with pytest.raises(SystemExit) as e:
        cli.Abliterator(base_args, lambda _m: None).run()
    refusal = str(e.value)
    assert "some/benchmark::test" in refusal, (
        f"the refusal has to name the benchmark that could not be read, or the operator cannot "
        f"tell which of their flags is wrong: {refusal}")
    assert "score every candidate the same" in refusal, (
        f"and it has to say why an unreadable probe is worse than no probe: {refusal}")
