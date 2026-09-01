# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A search that measured nothing must say so, not die in `min()`.

`_select_knee` runs after the whole search. Every way it can fail arrives hours into a run that
has already spent the GPU time, so each failure has to name what went wrong, what state the run
is in, and how to recover without searching again. `min() arg is an empty sequence` does none
of those things.

The study is faked rather than searched: what is under test is the selection, and driving a real
Optuna study to an all-failed state would test Optuna.
"""
import types

import pytest

from senbonzakura import cli


class _Trial:
    """The parts of an Optuna trial that _select_knee reads."""

    def __init__(self, user_attrs=None, params=None, state="TrialState.COMPLETE"):
        self.user_attrs = user_attrs or {}
        self.params = params if params is not None else _params()
        self.state = state


def _params(**over):
    p = {"max_weight_position": 8, "max_weight": 1.0, "min_weight": 0.2,
         "min_weight_distance": 4, "d_max_weight_position": 8, "d_max_weight": 0.0,
         "d_min_weight": 0.0, "d_min_weight_distance": 4,
         "num_directions": 1, "dir_mode": "per_layer"}
    p.update(over)
    return p


def _attrs(refusals=0.0, kl=0.01, broken=0.0, heretic=0.0):
    return {"refusals": refusals, "kl": kl, "broken": broken, "heretic": heretic}


def _selector(tmp_path, max_kl=None):
    obj = cli.Abliterator.__new__(cli.Abliterator)
    obj.args = types.SimpleNamespace(out=str(tmp_path), max_kl=max_kl, seed=42,
                                     search="pareto")
    obj.log = lambda _m: None
    return obj


class _Study:
    def __init__(self, trials):
        self.trials = trials
        self.best_trials = [t for t in trials if t.user_attrs]


# ── the empty case: every trial raised before it could be scored ──────────────────
def test_no_scored_trial_exits_loudly_rather_than_dying_in_min(tmp_path):
    study = _Study([_Trial(state="TrialState.FAIL") for _ in range(5)])
    with pytest.raises(SystemExit) as e:
        _selector(tmp_path)._select_knee(study, None, None)
    msg = str(e.value)
    assert "min()" not in msg                       # not the bare Python error
    assert "no usable trial" in msg


def test_the_message_counts_what_ran_and_what_failed(tmp_path):
    # "5 ran, 5 failed" is the difference between a broken model and a broken eval set,
    # and it is the first thing an operator needs at 3am.
    study = _Study([_Trial(state="TrialState.FAIL") for _ in range(5)])
    with pytest.raises(SystemExit) as e:
        _selector(tmp_path)._select_knee(study, None, None)
    msg = str(e.value)
    assert "5 ran" in msg
    assert "5 failed" in msg


def test_the_message_names_the_likely_causes_and_both_recoveries(tmp_path):
    study = _Study([_Trial(state="TrialState.FAIL")])
    with pytest.raises(SystemExit) as e:
        _selector(tmp_path)._select_knee(study, None, None)
    msg = str(e.value)
    assert "VRAM" in msg                    # the usual cause
    assert "--resume" in msg                # keeps the completed work
    assert "--bake-config" in msg           # saves without searching


def test_the_message_names_the_study_when_there_is_one(tmp_path):
    # Without the path, "re-run with --resume" is advice the operator cannot follow.
    study = _Study([_Trial(state="TrialState.FAIL")])
    with pytest.raises(SystemExit) as e:
        _selector(tmp_path)._select_knee(study, "/runs/senbon-study.db", None)
    assert "/runs/senbon-study.db" in str(e.value)


def test_a_trial_that_ran_but_recorded_nothing_counts_as_unusable(tmp_path):
    # State COMPLETE with empty user_attrs is the subtle case: Optuna is content and there
    # is still nothing to select on. Counting it as usable is what reaches min() empty.
    study = _Study([_Trial(user_attrs={}, state="TrialState.COMPLETE")])
    with pytest.raises(SystemExit, match="no usable trial"):
        _selector(tmp_path)._select_knee(study, None, None)


# ── the ceiling case: trials exist but none met the limit that was asked for ──────
def test_max_kl_unmet_refuses_to_bake_something_else(tmp_path):
    # A run told to stay under a drift and then handed a model above it has answered a
    # different question, so this exits rather than quietly returning the best available.
    study = _Study([_Trial(_attrs(kl=0.9)), _Trial(_attrs(kl=0.7))])
    with pytest.raises(SystemExit) as e:
        _selector(tmp_path, max_kl=0.05)._select_knee(study, None, None)
    msg = str(e.value)
    assert "no configuration met --max-kl 0.05" in msg
    assert "0.7" in msg                    # the least drift actually achieved
    assert "--resume" in msg


def test_max_kl_unmet_message_suggests_a_reachable_value(tmp_path):
    # Telling an operator to raise a limit without saying how far is advice they have to
    # guess at, and each guess is another search.
    study = _Study([_Trial(_attrs(kl=0.31))])
    with pytest.raises(SystemExit) as e:
        _selector(tmp_path, max_kl=0.05)._select_knee(study, None, None)
    assert "0.310" in str(e.value)


def test_an_incoherent_trial_under_the_ceiling_does_not_count_as_intact(tmp_path):
    # Under the KL limit but wrecked output. Selecting it would produce a model that is
    # technically within drift and generates garbage.
    study = _Study([_Trial(_attrs(kl=0.01, broken=0.9))])
    with pytest.raises(SystemExit, match="every trial under the limit was incoherent"):
        _selector(tmp_path, max_kl=0.05)._select_knee(study, None, None)
