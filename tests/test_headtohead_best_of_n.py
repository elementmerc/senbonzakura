# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for head-to-head/best_of_n_heretic.py, the selection pass that decides Heretic's reported arm.

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
        Path(__file__).resolve().parent.parent / "head-to-head" / "best_of_n_heretic.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["best_of_n_heretic"] = module
    spec.loader.exec_module(module)
    return module


bon = _load()


class FakeTrial:
    """A frozen Optuna trial as v1.4.0 records one.

    `refusals` is a COUNT out of `n_bad_prompts`, not a rate; reading it as a rate would make every
    candidate look far worse than it is and would flip the ranking.
    """

    def __init__(self, index, refusals=None, kl=None, n_eval=64, attrs=None):
        self.number = index
        self.user_attrs = {"index": index, "direction_index": 12.0, "parameters": {}}
        if attrs is None:
            self.user_attrs.update({"refusals": refusals, "kl_divergence": kl,
                                    "n_bad_prompts": n_eval})
        else:
            self.user_attrs.update(attrs)


# ── reading what the trial recorded ───────────────────────────────────────────────────
def test_the_refusal_count_is_read_as_a_fraction_of_the_eval_set():
    """v1.4.0 records a COUNT. Reading it as a rate would make 32/64 look like 3200%."""
    assert bon.refusal_fraction(FakeTrial(1, refusals=32, kl=0.08, n_eval=64)) == 0.5


def test_a_trial_with_no_recorded_count_has_no_fraction():
    assert bon.refusal_fraction(FakeTrial(1, attrs={"kl_divergence": 0.1})) is None


def test_a_zero_eval_set_does_not_divide_by_zero():
    assert bon.refusal_fraction(FakeTrial(1, refusals=0, kl=0.1, n_eval=0)) is None


# ── which candidates are nominated ────────────────────────────────────────────────────
def test_the_front_is_built_the_way_heretic_builds_it():
    """Sort by (refusal count, KL), then keep each trial that improves on the lowest KL so far.

    Reproduced from v1.4.0's own trial loop rather than taken from Optuna's `best_trials`, because
    the objective Heretic hands Optuna is a scalarised score rather than these two quantities. A
    different front would nominate a different six and publish a comparison against a Heretic
    nobody runs.
    """
    trials = [
        FakeTrial(1, refusals=2, kl=0.30),   # fewest refusals, but the worst KL
        FakeTrial(2, refusals=5, kl=0.20),   # more refusals, better KL: on the front
        FakeTrial(3, refusals=8, kl=0.25),   # dominated by trial 2, so not on the front
        FakeTrial(4, refusals=9, kl=0.05),   # most refusals, best KL: on the front
    ]
    assert [t.user_attrs["index"] for t in bon.heretic_front(trials)] == [1, 2, 4]


def test_a_dominated_trial_never_reaches_the_candidates():
    trials = [FakeTrial(1, refusals=2, kl=0.10), FakeTrial(2, refusals=6, kl=0.40)]
    assert [t.user_attrs["index"] for t in bon.nominate(trials, top_n=6)] == [1]


def test_ties_on_refusals_are_broken_by_kl():
    trials = [FakeTrial(1, refusals=4, kl=0.30), FakeTrial(2, refusals=4, kl=0.10)]
    assert [t.user_attrs["index"] for t in bon.heretic_front(trials)] == [2]


def test_only_the_top_n_are_nominated():
    trials = [FakeTrial(i, refusals=i, kl=1.0 / i) for i in range(1, 10)]
    assert len(bon.nominate(trials, top_n=6)) == 6


def test_a_top_n_of_zero_still_nominates_one():
    """A candidate list of length zero would make the pass silently select nothing at all."""
    assert len(bon.nominate([FakeTrial(1, refusals=1, kl=0.1)], top_n=0)) == 1


def test_no_trials_at_all_is_refused():
    with pytest.raises(SystemExit) as e:
        bon.nominate([], top_n=6)
    assert "no completed trial" in str(e.value)


def test_a_trial_missing_an_attribute_is_named_rather_than_raising_from_inside_sorted():
    trials = [FakeTrial(1, refusals=1, kl=0.1),
              FakeTrial(2, attrs={"refusals": 3})]
    with pytest.raises(SystemExit) as e:
        bon.nominate(trials, top_n=6)
    assert "trial 2" in str(e.value) and "kl_divergence" in str(e.value)


# ── the re-score, and who wins it ─────────────────────────────────────────────────────
COMPLIANT = ["Here is how you do it. Step one, then step two."] * 10
REFUSING = ["I'm sorry, I cannot help with that."] * 10


def test_the_winner_is_the_lowest_knee_not_the_first_offered():
    """The whole point of the pass: the front's first entry need not survive a larger eval."""
    first, second = FakeTrial(1, refusals=3, kl=0.05), FakeTrial(2, refusals=19, kl=0.05)
    responses = {1: REFUSING, 2: COMPLIANT}
    rows, winner, generations = bon.rescore([first, second],
                                            lambda t: responses[t.user_attrs["index"]])
    assert winner["trial"] == 2
    assert generations == 20
    assert [r["trial"] for r in rows] == [1, 2]


def test_kl_comes_from_the_trial_and_not_from_the_rescore():
    """Ours refreshes the refusal axes on more evidence and leaves coherence alone; so does this."""
    rows, _, _ = bon.rescore([FakeTrial(1, refusals=3, kl=0.42)], lambda t: COMPLIANT)
    assert rows[0]["kl"] == 0.42
    assert rows[0]["kl_source"] == "trial"


def test_the_knee_is_the_shared_one_not_a_local_copy():
    rows, _, _ = bon.rescore([FakeTrial(1, refusals=3, kl=0.3)], lambda t: REFUSING)
    r = rows[0]
    assert r["knee"] == knee_scalar(r["refusals"], r["soft"], r["heretic"], r["kl"],
                                    broken=r["broken"])


def test_a_trial_with_no_recorded_kl_is_refused_rather_than_scored_as_zero():
    """Zero would read as perfectly intact, which is the most flattering possible substitution."""
    with pytest.raises(SystemExit) as e:
        bon.rescore([FakeTrial(1, attrs={"refusals": 3})], lambda t: COMPLIANT)
    assert "recorded no" in str(e.value)


def test_a_broken_candidate_is_measured_as_broken():
    rows, _, _ = bon.rescore([FakeTrial(1, refusals=0, kl=0.05)], lambda t: [""] * 10)
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


# ── the other tool's own argument parsing ─────────────────────────────────────────────
def test_the_command_line_is_blanked_before_heretics_settings_are_built():
    """Heretic's Settings parses sys.argv the moment one is constructed.

    Found by a dry run: it met this pass's own flags, did not recognise them, printed its usage and
    exited 2, several minutes into a GPU job and naming nothing about the real cause.
    """
    source = (Path(__file__).resolve().parent.parent / "head-to-head" / "best_of_n_heretic.py").read_text()
    body = source.split("def main(")[1]
    blank = body.index("sys.argv = sys.argv[:1]")
    assert blank < body.index("open_study("), \
        "argv must be blanked before anything constructs Heretic's Settings"
    assert blank > body.index("ap.parse_args("), \
        "argv must survive until this pass has parsed its own flags"
