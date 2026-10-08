# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The decoy, and the two ways an experiment about it could quietly answer the wrong question.

WHY THIS FILE EXISTS

A published defence makes a LOW leak figure the signature of a model whose refusal was never
touched: give a contrastive estimator a high-magnitude feature orthogonal to refusal and it finds
that instead, so the edit lands on nothing. `residualleak.py` tells a reader that `validate` is the
command for this question, because a decoy should fail leave-one-cluster-out generalisation while
still scoring well on magnitude. **That is a hypothesis in a docstring.** Testing it needs a
defended model, and none is published, so the decoy has to be built.

THE TWO WAYS THE EXPERIMENT COULD ANSWER THE WRONG QUESTION, which is what these tests guard:

1. **The decoy is not actually a decoy.** A direction orthogonal to one refusal direction but
   inside the span of the others is partly the real thing, and the experiment would be measuring
   whether `validate` catches a half-decoy. A null result would mean nothing.
2. **The injection did not work, and the sweep is read as a clean bill of health.** An alpha too
   small leaves the estimator preferring refusal, so the arm is an undefended model; an alpha too
   large breaks the model, and a broken model defeats the estimator too. Every arm failing is a
   finding about the script, not about the detector, and that is the one conclusion the data cannot
   support.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "research"))
import decoy_injection as di

H = 32


def _refusal(k=3, h=H, seed=1):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((k, h))
    q, _ = np.linalg.qr(R.T)
    return q.T[:k]


# ── the decoy has to be a decoy ─────────────────────────────────────────────────────

def test_the_decoy_is_orthogonal_to_the_whole_refusal_span():
    """GUARD 1. Orthogonal to one direction and inside the span of the others is a half-decoy."""
    R = _refusal(k=4)
    d = di.orthogonal_decoy(R)
    assert np.abs(R @ d).max() < 1e-8
    assert np.linalg.norm(d) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [1, 2, 8, 16])
def test_it_stays_orthogonal_across_span_sizes(k):
    R = _refusal(k=k)
    assert np.abs(R @ di.orthogonal_decoy(R)).max() < 1e-8


def test_a_span_that_fills_the_space_leaves_no_room_and_says_so():
    """Rather than returning a vector that is 'nearly' orthogonal, which is a half-decoy."""
    with pytest.raises(di.DecoyError, match="no room"):
        di.orthogonal_decoy(_refusal(k=H, h=H))


def test_a_one_dimensional_refusal_still_gets_a_decoy():
    d = di.orthogonal_decoy(_refusal(k=1))
    assert np.linalg.norm(d) == pytest.approx(1.0)


def test_different_seeds_give_different_decoys():
    R = _refusal()
    assert not np.allclose(di.orthogonal_decoy(R, seed=0), di.orthogonal_decoy(R, seed=1))


# ── the rank-one term ───────────────────────────────────────────────────────────────

def test_the_update_fires_in_proportion_to_the_refusal_signal():
    """What makes a difference of means find the decoy: the added term contributes `alpha * decoy`
    exactly in proportion to how much refusal is present at that layer's input.
    """
    R = _refusal(k=1)
    d = di.orthogonal_decoy(R)
    U = di.rank_one_update(d, R[0], alpha=4.0)
    # An input along the refusal direction produces 4 * decoy out.
    out = U @ R[0]
    assert np.allclose(out, 4.0 * d, atol=1e-9)


def test_an_input_orthogonal_to_refusal_produces_nothing():
    """The other half: the decoy must not fire on harmless inputs, or it is not reading refusal."""
    R = _refusal(k=1)
    d = di.orthogonal_decoy(R)
    U = di.rank_one_update(d, R[0], alpha=4.0)
    assert np.allclose(U @ d, 0.0, atol=1e-9)


@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_a_decoy_with_no_magnitude_is_refused(alpha):
    R = _refusal(k=1)
    with pytest.raises(di.DecoyError, match="not a decoy"):
        di.rank_one_update(di.orthogonal_decoy(R), R[0], alpha=alpha)


def test_the_update_scales_with_alpha():
    R = _refusal(k=1)
    d = di.orthogonal_decoy(R)
    small = di.rank_one_update(d, R[0], alpha=1.0)
    large = di.rank_one_update(d, R[0], alpha=10.0)
    assert np.allclose(large, 10.0 * small)


# ── whether the estimator took the bait, measured rather than assumed ───────────────

def test_a_large_decoy_makes_the_estimator_prefer_it():
    """THE PROPERTY THE INJECTION EXISTS TO PRODUCE. Measured, because if it is false then any
    conclusion about `validate` is a conclusion about an undefended model.
    """
    R = _refusal(k=1)
    d = di.orthogonal_decoy(R)
    harmless = np.zeros(H)
    harmful = R[0] + 20.0 * d          # refusal present, plus a big decoy write
    got = di.estimator_prefers_decoy(harmful, harmless, d, R)
    assert got["prefers_decoy"] is True
    assert got["margin"] > 0


def test_a_small_decoy_leaves_the_estimator_on_refusal():
    """Which is an arm that says nothing, and has to be detectable as such."""
    R = _refusal(k=1)
    d = di.orthogonal_decoy(R)
    got = di.estimator_prefers_decoy(R[0] + 0.01 * d, np.zeros(H), d, R)
    assert got["prefers_decoy"] is False


def test_identical_means_are_refused_rather_than_scored():
    R = _refusal(k=1)
    with pytest.raises(di.DecoyError, match="no difference of means"):
        di.estimator_prefers_decoy(np.zeros(H), np.zeros(H), di.orthogonal_decoy(R), R)


def test_the_alignments_are_reported_and_not_only_the_verdict():
    """"Preferred the decoy by a nose" and "overwhelmingly" are different experiments."""
    R = _refusal(k=1)
    d = di.orthogonal_decoy(R)
    got = di.estimator_prefers_decoy(R[0] + 20.0 * d, np.zeros(H), d, R)
    assert 0.0 <= got["alignment_decoy"] <= 1.0
    assert 0.0 <= got["alignment_refusal"] <= 1.0


# ── an arm, and the two ways it is uninformative ────────────────────────────────────

def _estimator(prefers, decoy=0.9, refusal=0.1):
    return {"prefers_decoy": prefers, "alignment_decoy": decoy,
            "alignment_refusal": refusal, "margin": decoy - refusal}


def test_an_arm_where_the_estimator_stayed_on_refusal_is_uninformative():
    arm = di.report(alpha=1.0, estimator=_estimator(False), refusal_before=0.6,
                    refusal_after=0.6, validate_flagged=False)
    assert arm["injection_worked"] is False
    assert arm["verdict"] == "uninformative"
    assert "too small" in arm["why_not"]
    assert arm["validate_flagged"] is None, "a verdict was recorded for an arm that said nothing"


def test_an_arm_that_broke_the_model_is_uninformative_and_says_why():
    """A broken model defeats the estimator too, so it would look like a successful defence."""
    arm = di.report(alpha=64.0, estimator=_estimator(True), refusal_before=0.6,
                    refusal_after=0.1, validate_flagged=False)
    assert arm["injection_worked"] is False
    assert "broke the model" in arm["why_not"]
    assert arm["validate_flagged"] is None


def test_a_working_injection_that_validate_caught_is_the_hypothesis_holding():
    arm = di.report(alpha=16.0, estimator=_estimator(True), refusal_before=0.6,
                    refusal_after=0.61, validate_flagged=True)
    assert arm["injection_worked"] is True
    assert arm["verdict"] == "validate exposed the decoy"


def test_a_working_injection_that_validate_missed_is_named_a_defect():
    """The result that would matter most, and the docs would have to change for it."""
    arm = di.report(alpha=16.0, estimator=_estimator(True), refusal_before=0.6,
                    refusal_after=0.6, validate_flagged=False)
    assert arm["injection_worked"] is True
    assert "defect in the detector" in arm["verdict"]


def test_the_refusal_drift_allowance_is_small_and_stated():
    assert 0 < di.MAX_REFUSAL_DRIFT <= 0.1


# ── the conclusion, which must refuse to be drawn from arms that said nothing ───────

def test_a_sweep_where_no_injection_worked_draws_no_conclusion():
    """GUARD 2, AND THE EXPENSIVE ONE. Reading a sweep in which nothing worked as a clean bill of
    health for the detector is the single conclusion this data cannot support.
    """
    arms = [di.report(alpha=a, estimator=_estimator(False), refusal_before=0.6,
                      refusal_after=0.6, validate_flagged=False) for a in (1.0, 2.0)]
    got = di.conclusion(arms)
    assert got["answer"] is None
    assert got["informative_arms"] == 0
    assert "says nothing about" in got["why"]
    assert "cannot support" in got["why"]


def test_a_sweep_where_validate_caught_every_working_arm_confirms_the_hypothesis():
    arms = [di.report(alpha=1.0, estimator=_estimator(False), refusal_before=0.6,
                      refusal_after=0.6, validate_flagged=False),
            di.report(alpha=16.0, estimator=_estimator(True), refusal_before=0.6,
                      refusal_after=0.6, validate_flagged=True)]
    got = di.conclusion(arms)
    assert got["answer"] == "validate exposes a decoy"
    assert got["informative_arms"] == 1
    assert got["alphas_that_worked"] == [16.0]


def test_one_miss_on_a_working_arm_is_enough_to_say_validate_passes_a_decoy():
    """A detector that catches a decoy at three magnitudes and misses it at a fourth has not
    caught decoys. The docs point readers at `validate` for this and would have to change.
    """
    arms = [di.report(alpha=16.0, estimator=_estimator(True), refusal_before=0.6,
                      refusal_after=0.6, validate_flagged=True),
            di.report(alpha=64.0, estimator=_estimator(True), refusal_before=0.6,
                      refusal_after=0.62, validate_flagged=False)]
    got = di.conclusion(arms)
    assert got["answer"] == "validate PASSES a decoy"
    assert got["alphas_validate_missed"] == [64.0]
    assert "wrong as written" in got["why"]


# ── the surface ─────────────────────────────────────────────────────────────────────

def test_the_default_sweep_spans_more_than_one_order_of_magnitude():
    """Both failure modes are at the ends, so a sweep that does not reach them finds neither."""
    assert max(di.DEFAULT_ALPHAS) / min(di.DEFAULT_ALPHAS) >= 16


def test_the_parser_declares_what_the_help_describes():
    dests = {a.dest for a in di.build_parser()._actions}
    for flag in ("model", "directions", "layer", "alpha", "seed", "workdir", "out"):
        assert flag in dests, f"--{flag} is described and not declared"


def test_the_command_reports_no_measurement_it_did_not_take(tmp_path, capsys):
    """The driver needs a card and is not written. What must never happen is this file emitting a
    document that reads like a result.
    """
    import json
    out = tmp_path / "sweep.json"
    assert di.main(["--model", "m", "--directions", "d", "--layer", "5",
                    "--workdir", str(tmp_path), "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["ran"] is False
    assert doc["specified"] is True
    assert "needs a card" in doc["why"]


def test_the_module_says_it_is_a_defence_and_ships_no_weights():
    """A measurement project building a defence has to say why, where a reader will meet it."""
    text = pathlib.Path(di.__file__).read_text(encoding="utf-8")
    head = text.split('"""')[1]
    assert "HARDER to uncensor" in head
    assert "No defended weights are published" in head or "no defended weights" in head.lower()


def test_a_seed_shared_with_the_refusal_set_is_a_retry_and_not_a_refusal():
    """THE DEFECT THE FIRST VERSION SHIPPED, and the reason it was wrong twice.

    A caller who seeds the refusal set and the decoy with the same number draws the set's own first
    row as its "random" vector, which lies in the span by construction. The first version refused,
    and blamed the span for filling the space: a confident wrong reason, which costs more than a
    crash. A retry distinguishes the two, because a span that really does fill the space collapses
    every draw and a correlated seed collapses one.
    """
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((3, H))
    q, _ = np.linalg.qr(raw.T)
    R = q.T[:3]
    d = di.orthogonal_decoy(R, seed=7)          # the same seed the set was built from
    assert np.abs(R @ d).max() < 1e-8
    assert np.linalg.norm(d) == pytest.approx(1.0)


def test_a_span_that_truly_fills_the_space_still_refuses_and_says_which_it_is():
    """The retry must not turn a real refusal into an endless loop or a wrong vector."""
    with pytest.raises(di.DecoyError, match="no room"):
        di.orthogonal_decoy(_refusal(k=H, h=H))
