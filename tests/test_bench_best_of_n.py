"""Tests for bench/best_of_n_heretic.py, the selection pass that decides Heretic's reported arm.

This script chooses which of Heretic's candidates goes into a published table, so its decision
logic is tested here rather than only exercised on the card. It normally runs inside the benchmark
container, where Heretic and its dependency tree live; neither is installed in this environment, so
the two modules it reaches into are stubbed at import time. Everything under test is ours: which
candidates are nominated, how they are ranked, and which one wins.

The stub is deliberately thin. It stands in for names the module imports and never for behaviour a
test asserts on.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from senbonzakura.metrics import knee_scalar

_HERETIC_STUBS = {
    "heretic": types.ModuleType("heretic"),
    "heretic.config": types.ModuleType("heretic.config"),
    "heretic.model": types.ModuleType("heretic.model"),
    "heretic.utils": types.ModuleType("heretic.utils"),
}


class _DatasetSpecification:
    def __init__(self, dataset=None, split=None, column=None):
        self.dataset, self.split, self.column = dataset, split, column


_HERETIC_STUBS["heretic.config"].DatasetSpecification = _DatasetSpecification
_HERETIC_STUBS["heretic.config"].Settings = object
_HERETIC_STUBS["heretic.model"].AbliterationParameters = dict
_HERETIC_STUBS["heretic.model"].Model = object
_HERETIC_STUBS["heretic.utils"].load_prompts = lambda *a, **k: []


def _load():
    for name, module in _HERETIC_STUBS.items():
        sys.modules.setdefault(name, module)
    spec = importlib.util.spec_from_file_location(
        "best_of_n_heretic",
        Path(__file__).resolve().parent.parent / "bench" / "best_of_n_heretic.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["best_of_n_heretic"] = module
    spec.loader.exec_module(module)
    return module


bon = _load()


class FakeTrial:
    """A frozen Optuna trial as this pass reads one: an index and recorded score records."""

    def __init__(self, index, keywords=None, kl=None, scores=None):
        self.number = index
        records = scores if scores is not None else [
            {"name": "Keywords", "score": {"value": keywords}},
            {"name": "KL divergence", "score": {"value": kl}},
        ]
        self.user_attrs = {"index": index, "scores": records,
                           "direction_index": 12.0, "parameters": {}}


NAMES = ["Keywords", "KL divergence"]


# ── reading a trial's recorded scores ─────────────────────────────────────────────────
def test_a_named_score_is_read_out_of_the_records():
    trial = FakeTrial(1, keywords=0.25, kl=0.08)
    assert bon.score_name(trial, "Keywords") == 0.25
    assert bon.score_name(trial, "KL divergence") == 0.08
    assert bon.score_name(trial, "Nothing By That Name") is None


# ── which candidates are nominated ────────────────────────────────────────────────────
def test_the_front_is_ranked_the_way_heretic_ranks_it():
    """Heretic's menu sorts its Pareto front by the recorded objective values, lowest first."""
    front = [FakeTrial(3, 0.30, 0.05), FakeTrial(1, 0.10, 0.20), FakeTrial(2, 0.20, 0.10)]
    got = bon.nominate(front, front, NAMES, top_n=3)
    assert [t.user_attrs["index"] for t in got] == [1, 2, 3]


def test_only_the_top_n_are_nominated():
    front = [FakeTrial(i, keywords=i / 10, kl=0.1) for i in range(1, 10)]
    assert len(bon.nominate(front, front, NAMES, top_n=6)) == 6


def test_a_top_n_of_zero_still_nominates_one():
    """A candidate list of length zero would make the pass silently select nothing at all."""
    front = [FakeTrial(1, 0.1, 0.1)]
    assert len(bon.nominate(front, front, NAMES, top_n=0)) == 1


def test_an_empty_front_falls_back_to_every_completed_trial():
    complete = [FakeTrial(1, 0.1, 0.1), FakeTrial(2, 0.2, 0.1)]
    assert len(bon.nominate([], complete, NAMES, top_n=6)) == 2


def test_no_trials_at_all_is_refused():
    with pytest.raises(SystemExit) as e:
        bon.nominate([], [], NAMES, top_n=6)
    assert "no completed trial" in str(e.value)


def test_a_trial_missing_a_score_is_named_rather_than_raising_from_inside_sorted():
    front = [FakeTrial(1, 0.1, 0.1), FakeTrial(2, scores=[{"name": "Keywords",
                                                           "score": {"value": 0.2}}])]
    with pytest.raises(SystemExit) as e:
        bon.nominate(front, front, NAMES, top_n=6)
    assert "trial 2" in str(e.value) and "KL divergence" in str(e.value)


# ── the re-score, and who wins it ─────────────────────────────────────────────────────
COMPLIANT = ["Here is how you do it. Step one, then step two."] * 10
REFUSING = ["I'm sorry, I cannot help with that."] * 10


def test_the_winner_is_the_lowest_knee_not_the_first_offered():
    """The whole point of the pass: the front's first entry need not survive a larger eval."""
    first, second = FakeTrial(1, 0.05, 0.05), FakeTrial(2, 0.30, 0.05)
    responses = {1: REFUSING, 2: COMPLIANT}
    rows, winner, generations = bon.rescore([first, second],
                                            lambda t: responses[t.user_attrs["index"]])
    assert winner["trial"] == 2
    assert generations == 20
    assert [r["trial"] for r in rows] == [1, 2]


def test_kl_comes_from_the_trial_and_not_from_the_rescore():
    """Ours refreshes the refusal axes on more evidence and leaves coherence alone; so does this."""
    trial = FakeTrial(1, 0.05, 0.42)
    rows, winner, _ = bon.rescore([trial], lambda t: COMPLIANT)
    assert rows[0]["kl"] == 0.42
    assert rows[0]["kl_source"] == "trial"


def test_the_knee_is_the_shared_one_not_a_local_copy():
    trial = FakeTrial(1, 0.05, 0.3)
    rows, _, _ = bon.rescore([trial], lambda t: REFUSING)
    r = rows[0]
    assert r["knee"] == knee_scalar(r["refusals"], r["soft"], r["heretic"], r["kl"])


def test_a_trial_with_no_recorded_kl_is_refused_rather_than_scored_as_zero():
    """Zero would read as perfectly intact, which is the most flattering possible substitution."""
    trial = FakeTrial(1, scores=[{"name": "Keywords", "score": {"value": 0.1}}])
    with pytest.raises(SystemExit) as e:
        bon.rescore([trial], lambda t: COMPLIANT)
    assert "recorded no" in str(e.value)


def test_a_broken_candidate_is_measured_as_broken():
    trial = FakeTrial(1, 0.0, 0.05)
    rows, _, _ = bon.rescore([trial], lambda t: [""] * 10)
    assert rows[0]["broken"] == 1.0
    # Heretic counts an empty response as a keyword match, so a wrecked model cannot win by
    # producing nothing at all.
    assert rows[0]["heretic"] == 1.0


# ── the reconstruction check ──────────────────────────────────────────────────────────
def test_a_faithful_reconstruction_passes():
    ok, drift = bon.reconstruction_ok(0.30, 0.30)
    assert ok and drift == 0


def test_a_reconstruction_inside_tolerance_passes():
    ok, _ = bon.reconstruction_ok(0.30, 0.30 + bon.RECONSTRUCTION_TOLERANCE / 2)
    assert ok


def test_a_reconstruction_outside_tolerance_fails():
    ok, drift = bon.reconstruction_ok(0.30, 0.55)
    assert not ok and drift == pytest.approx(0.25)


def test_a_missing_recorded_score_fails_the_check():
    """Nothing to check against is not the same as a check that passed."""
    ok, drift = bon.reconstruction_ok(None, 0.30)
    assert not ok and drift is None
