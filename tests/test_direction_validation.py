"""Tests for tools/direction_validation.py, the experiment that judges the directions.

This is the instrument that decides whether the multi-direction claim survives, so the tests
plant data whose answer is known and demand that the experiment reach it. An experiment that
cannot tell a topic direction from a refusal direction on data built to contain one or the other
is not evidence about either, and this project has already shipped one of those.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from senbonzakura import cli

_SPEC = importlib.util.spec_from_file_location(
    "direction_validation", Path(__file__).resolve().parent.parent / "tools" / "direction_validation.py")
dv = importlib.util.module_from_spec(_SPEC)
sys.modules["direction_validation"] = dv
_SPEC.loader.exec_module(dv)


class _Stub:
    """Enough of an Abliterator for experiment_1: the residuals and the loader it reads."""

    def __init__(self, Rb, Rg, seed=42, clusters=4):
        self.NL = Rb.shape[0] - 1
        self.H = Rb.shape[2]
        self._Rb, self._Rg = Rb, Rg
        self.args = SimpleNamespace(
            track="t", good_ds=None, dir_prompts=Rb.shape[1], seed=seed,
            direction_clusters=clusters)

    def load(self, path, n):
        # The prompts carry which side they came from. Keying on row count instead once made
        # both clouds identical, and every separation came back as exactly zero: the same
        # signature as the real defect, produced by the test rather than by the code.
        tag = "GOOD" if "good" in str(path) else "BAD"
        rows = self._Rg.shape[1] if tag == "GOOD" else self._Rb.shape[1]
        return [f"{tag} {i}" for i in range(rows)]

    def collect_resid(self, prompts):
        return self._Rg if prompts and prompts[0].startswith("GOOD") else self._Rb


def _stack(rows, NL1):
    return rows.unsqueeze(0).repeat(NL1, 1, 1)


def _multi_component_refusal(n_per=40, H=32, NL1=5, seed=0):
    """Refusal made of TWO shared components, mixed in different proportions per cluster.

    This is the multi-direction hypothesis stated as data. Both components appear in every
    cluster, including one the fit never saw, but the mix differs, so the global mean difference
    cannot represent them both. What is left over after d0 is real, shared and transferable, and
    a direction fitted on three clusters should still separate the fourth.
    """
    torch.manual_seed(seed)
    mixes = [(1.0, 0.15), (0.85, 0.4), (0.4, 0.85), (0.15, 1.0)]
    groups = []
    for t, (w0, w1) in enumerate(mixes):
        g = torch.randn(n_per, H) * 0.3
        g[:, 0] += 6.0 * w0        # refusal component one
        g[:, 2] += 6.0 * w1        # refusal component two, same in every cluster, different mix
        g[:, 4 + t] += 1.0         # a small topic offset, unique per cluster
        groups.append(g)
    Rb = torch.cat(groups)
    Rg = torch.randn(4 * n_per, H) * 0.3
    Rg[:, 1] += 5.0
    return _stack(Rb, NL1), _stack(Rg, NL1)


def _single_component_refusal(n_per=40, H=32, NL1=5, seed=0):
    """ONE shared refusal direction plus per-cluster topics.

    The subtle case, and the reason this fixture is here rather than the obvious one. When
    refusal is a single direction, d0 absorbs it completely and the extra directions can only
    ever carry cluster-specific structure. "The extras do not generalise" is then the CORRECT
    answer, and it means refusal is one direction rather than that the method is broken.
    """
    torch.manual_seed(seed)
    groups = []
    for t in range(4):
        g = torch.randn(n_per, H) * 0.3
        g[:, 0] += 5.0            # the one shared refusal direction
        g[:, 3 + t] += 5.0        # this cluster's own topic
        groups.append(g)
    Rb = torch.cat(groups)
    Rg = torch.randn(4 * n_per, H) * 0.3
    Rg[:, 1] += 5.0
    return _stack(Rb, NL1), _stack(Rg, NL1)


def _topic_only(n_per=40, H=32, NL1=5, seed=0):
    """Each cluster displaced along its OWN direction, nothing shared at all.

    A pile of topic directions. The experiment must not report this as refusal.
    """
    torch.manual_seed(seed)
    groups = []
    for t in range(4):
        g = torch.randn(n_per, H) * 0.3
        g[:, 3 + t] += 6.0
        groups.append(g)
    Rb = torch.cat(groups)
    Rg = torch.randn(4 * n_per, H) * 0.3
    Rg[:, 1] += 5.0
    return _stack(Rb, NL1), _stack(Rg, NL1)


def _run(Rb, Rg, clusters=4):
    a = _Stub(Rb, Rg, clusters=clusters)
    return dv.experiment_1(a, lambda m: None)


# ── the experiment must reach the right answer on data whose answer is known ───────────
def test_a_multi_component_refusal_subspace_is_recognised_as_generalising():
    """The multi-direction hypothesis, planted in data, must be detected."""
    r = _run(*_multi_component_refusal())
    assert r["summary"], "no usable folds, so the fixture is not exercising the experiment"
    assert "do NOT generalise" not in r["verdict"]
    assert r["summary"]["mean_extra"] > r["summary"]["mean_random"]


def test_a_single_refusal_direction_correctly_reports_no_generalising_extras():
    """The subtle case, and the one most likely to be misread as a broken experiment.

    With one shared refusal direction, d0 absorbs it and the extras can only carry topic. The
    honest answer is "the extras do not generalise", and it means refusal is one direction.
    """
    r = _run(*_single_component_refusal())
    assert r["summary"]
    assert "do NOT generalise" in r["verdict"]


def test_topic_only_directions_are_caught():
    """The failure mode the experiment exists to detect."""
    r = _run(*_topic_only())
    assert r["summary"], "no usable folds"
    assert r["summary"]["folds_where_extra_beats_random_max"] <= r["summary"]["folds"] * 0.25, (
        f"topic-only directions were not caught: {r['summary']}")
    assert "do NOT generalise" in r["verdict"]


def test_the_two_fixtures_actually_differ():
    """Guards against a verdict that is constant regardless of the data."""
    multi = _run(*_multi_component_refusal())["verdict"]
    topic = _run(*_topic_only())["verdict"]
    assert multi != topic


def test_every_fold_reports_its_random_floor():
    """A fitted score without the floor beside it cannot be read."""
    r = _run(*_multi_component_refusal())
    for fold in r["folds"]:
        if fold["best_extra_on_held_out"] is None:
            continue
        assert fold["random_mean"] is not None and fold["random_max"] is not None
        assert fold["held_out_rows"] >= cli.MIN_CLUSTER_ROWS


def test_the_held_out_cluster_is_never_fitted_on():
    """The whole design rests on this, so it is asserted rather than assumed.

    `extra_on_train` is the score on the rows the direction was fitted to and
    `best_extra_on_held_out` is the score on rows it never saw. If the fit leaked, the two would
    be suspiciously equal on topic-only data, where the held-out score should collapse.
    """
    r = _run(*_topic_only())
    pairs = [(f["extra_on_train"], f["best_extra_on_held_out"]) for f in r["folds"]
             if f["extra_on_train"] is not None and f["best_extra_on_held_out"] is not None]
    assert pairs
    assert any(train > held * 2 for train, held in pairs), (
        "held-out scores match training scores on topic-only data, which suggests a leak")


def test_no_usable_folds_is_reported_rather_than_crashing():
    """A corpus too small to hold out from must say so, not divide by zero."""
    torch.manual_seed(0)
    Rb = _stack(torch.randn(4, 16), 3)
    Rg = _stack(torch.randn(4, 16), 3)
    r = dv.experiment_1(_Stub(Rb, Rg, clusters=2), lambda m: None)
    assert r["summary"] == {} and r["verdict"] == "no usable folds"


# ── argument plumbing ─────────────────────────────────────────────────────────────────
def test_the_args_come_from_the_real_parser():
    _, args = dv.build_args(["--model", "m", "--out", "/tmp/x.json"])
    reference = cli.build_parser().parse_args(["--model", "m"])
    ignore = {"model", "track", "device", "max_directions", "direction_clusters",
              "dir_prompts", "eval_refusal", "eval_kl", "seed"}
    for key, value in vars(reference).items():
        if key not in ignore:
            assert getattr(args, key) == value, f"{key} drifted from the parser default"


def test_the_experiment_choice_reaches_the_record():
    own, _ = dv.build_args(["--model", "m", "--out", "/tmp/x.json", "--experiment", "e1"])
    assert own.experiment == "e1"
    with pytest.raises(SystemExit):
        dv.build_args(["--model", "m", "--out", "/tmp/x.json", "--experiment", "nonsense"])


# ── regression guards: the mistakes of 2026-08-03, made permanent ─────────────────────
# Every one of these encodes a specific error that produced a confident, wrong result. They
# matter MOST when the numbers start looking good, which is exactly when a guard gets dropped
# as noise. Each names the failure it prevents so nobody deletes it as redundant.

def test_a_grid_with_no_spread_refuses_to_be_read():
    """The floor effect that made the first K sweep meaningless.

    Eleven arms all landed at exactly 0.0% refusal, because the default window removed
    everything at K=1. The table looked like data and contained none. A sweep whose arms all
    share one value says nothing about the thing it varied and must say so.
    """
    flat = [{"arm": f"fitted-K{k}", "K": k, "strength": 1.0, "random_extras": False,
             "harmful_refusal": 0.0, "harmless_refusal": 0.0, "kl": 0.3 * k} for k in (1, 2, 3)]
    reason = dv.degenerate_reason(flat)
    assert reason and "no dynamic range" in reason
    assert "floor" in reason


def test_a_ceiling_is_caught_as_well_as_a_floor():
    """The mirror case: nothing was removed, so K is equally unmeasurable."""
    flat = [{"arm": f"fitted-K{k}", "K": k, "strength": 0.1, "random_extras": False,
             "harmful_refusal": 1.0, "harmless_refusal": 0.0, "kl": 0.01} for k in (1, 2, 3)]
    reason = dv.degenerate_reason(flat)
    assert reason and "ceiling" in reason


def test_a_grid_with_spread_is_allowed_through():
    """The guard must not fire on a readable grid, or it becomes noise nobody reads."""
    rows = [{"arm": "fitted-K1", "K": 1, "strength": 0.5, "random_extras": False,
             "harmful_refusal": 0.40, "harmless_refusal": 0.0, "kl": 0.20},
            {"arm": "fitted-K3", "K": 3, "strength": 0.5, "random_extras": False,
             "harmful_refusal": 0.15, "harmless_refusal": 0.0, "kl": 0.35}]
    assert dv.degenerate_reason(rows) is None


def test_kl_is_only_compared_at_matched_refusal_removal():
    """Comparing KL across arms that removed different amounts of refusal compares nothing.

    More ablation always costs more KL, so the arm that cut harder "loses" whatever the quality
    of its directions. This is the comparison Wollschlaeger and Piras both make at matched
    attack success, and the one the withdrawn work never made.
    """
    rows = [
        # K=1 needs full strength to reach 10% refusal, and pays 0.9 KL for it.
        {"arm": "fitted-K1-s1.0", "K": 1, "strength": 1.0, "random_extras": False,
         "harmful_refusal": 0.10, "harmless_refusal": 0.0, "kl": 0.90},
        {"arm": "fitted-K1-s0.3", "K": 1, "strength": 0.3, "random_extras": False,
         "harmful_refusal": 0.80, "harmless_refusal": 0.0, "kl": 0.10},
        # K=3 reaches the same 10% at lower strength and lower KL: the multi-direction win.
        {"arm": "fitted-K3-s0.5", "K": 3, "strength": 0.5, "random_extras": False,
         "harmful_refusal": 0.10, "harmless_refusal": 0.0, "kl": 0.40},
    ]
    table = dv.matched_refusal_table(rows)
    assert table["baseline_refusal"] == pytest.approx(0.80)
    band = (table["targets"].get("87%_removed") or table["targets"].get("90%_removed")
            or table["targets"].get("75%_removed"))
    assert band, f"no matched band was produced: {table}"
    # Both K reached the band; the cheaper one must be reported as cheaper.
    assert band["3"]["kl"] < band["1"]["kl"]


def test_a_k_that_never_reaches_the_target_gets_no_entry():
    """A K that cannot reach the target must be absent, not credited with its best near-miss."""
    rows = [
        {"arm": "fitted-K1", "K": 1, "strength": 1.0, "random_extras": False,
         "harmful_refusal": 0.05, "harmless_refusal": 0.0, "kl": 0.5},
        {"arm": "fitted-K3", "K": 3, "strength": 1.0, "random_extras": False,
         "harmful_refusal": 0.60, "harmless_refusal": 0.0, "kl": 0.1},
    ]
    table = dv.matched_refusal_table(rows)
    for band in table["targets"].values():
        if "1" in band and "3" not in band:
            break
    else:
        pytest.fail(f"K=3 should be missing from at least one band: {table['targets']}")


def test_the_random_control_is_carried_into_the_grid():
    """E2's floor has to exist at every strength, or "fitted beats random" is untested."""
    rows = [{"arm": "fitted-K2-s0.5", "K": 2, "strength": 0.5, "random_extras": False,
             "harmful_refusal": 0.2, "harmless_refusal": 0.0, "kl": 0.3},
            {"arm": "random-K2-s0.5", "K": 2, "strength": 0.5, "random_extras": True,
             "harmful_refusal": 0.5, "harmless_refusal": 0.0, "kl": 0.3}]
    # The matched table considers fitted arms only; the random arms must not leak into it.
    table = dv.matched_refusal_table(rows)
    for band in table["targets"].values():
        for entry in band.values():
            assert entry["harmful_refusal"] != 0.5, "a random-extras arm leaked into the comparison"
