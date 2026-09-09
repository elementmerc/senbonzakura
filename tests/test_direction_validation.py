# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for tools/direction_validation.py, the experiment that judges the directions.

This is the instrument that decides whether the multi-direction claim survives, so the tests
plant data whose answer is known and demand that the experiment reach it. An experiment that
cannot tell a topic direction from a refusal direction on data built to contain one or the other
is not evidence about either, and this project has already shipped one of those.
"""
import json
from types import SimpleNamespace

import pytest
import torch

from senbonzakura import cli
from senbonzakura import validate as dv


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


def test_a_random_extras_arm_never_enters_the_matched_comparison():
    """The matched table ranks FITTED arms; a random arm in it would be a floor scoring as a result.

    The previous version of this test could not fail. Its rows produced no bands at all, so the
    assertion loop body never ran, and even with the `random_extras` filter removed the surviving
    assertion still held. A leak test whose leak changes nothing tests nothing, so this one names
    the random arm's KL and demands it be absent by identity rather than by a property it might
    share with a fitted arm.
    """
    anchor = {"arm": "unablated-K1-s0", "K": 1, "strength": 0.0, "random_extras": False,
              "harmful_refusal": 0.80, "harmless_refusal": 0.0, "kl": 0.0}
    # Both land inside the 50% band (target 0.40), so both are genuinely eligible for selection
    # and only the filter keeps the random one out. The random arm is CHEAPER, so a leak would
    # win the band outright rather than sit in it unnoticed.
    fitted = {"arm": "fitted-K2-s0.5", "K": 2, "strength": 0.5, "random_extras": False,
              "harmful_refusal": 0.40, "harmless_refusal": 0.0, "kl": 0.30}
    random_arm = {"arm": "random-K2-s0.5", "K": 2, "strength": 0.5, "random_extras": True,
                  "harmful_refusal": 0.40, "harmless_refusal": 0.0, "kl": 0.01}

    table = dv.matched_refusal_table([anchor, fitted, random_arm])
    band = table["targets"]["50%_removed"]
    assert band, "the fixture produced no band, so this test would assert nothing"
    assert band["2"]["kl"] == pytest.approx(0.30), (
        "the random-extras arm won the band, so the floor is being scored as a result")
    assert all(e["kl"] != pytest.approx(0.01) for e in band.values())


# ── loading an externally optimised direction set ─────────────────────────────────────
class _LoadStub:
    def __init__(self, NL=4, H=8):
        self.NL, self.H, self.KMAX = NL, H, 3
        self.dirs_multi = None
        self.dirs_per_layer = None


def test_a_loaded_direction_set_installs_and_counts_itself(tmp_path):
    a = _LoadStub()
    dirs = torch.zeros(a.NL + 1, 2, a.H)
    for li in range(2, a.NL + 1):
        dirs[li] = torch.eye(a.H)[:2]
    path = tmp_path / "d.pt"
    torch.save({"dirs_multi": dirs}, path)

    dv.load_directions(a, str(path), lambda m: None)
    assert a.dirs_per_layer[:2] == [0, 0]
    assert a.dirs_per_layer[2:] == [2] * (a.NL - 1)
    assert a.KMAX == 2


def test_a_direction_set_from_another_model_is_refused(tmp_path):
    """Shapes that do not fit would otherwise be scored as though they did."""
    a = _LoadStub(NL=4, H=8)
    path = tmp_path / "d.pt"
    torch.save({"dirs_multi": torch.zeros(9, 2, 64)}, path)   # a different model entirely
    with pytest.raises(SystemExit, match="does not fit this model"):
        dv.load_directions(a, str(path), lambda m: None)


def test_a_non_orthonormal_direction_set_is_refused(tmp_path):
    """The guard that stops correlated rows being read as extra directions.

    The bake computes R^T(RW), a projection only for an orthonormal R. Correlated rows
    over-subtract along whatever they share, which shows up as a stronger ablation and would be
    credited to "more directions".
    """
    a = _LoadStub()
    dirs = torch.zeros(a.NL + 1, 2, a.H)
    d0 = torch.zeros(a.H); d0[0] = 1.0
    near = torch.zeros(a.H); near[0] = 0.95; near[1] = 0.312
    for li in range(2, a.NL + 1):
        dirs[li, 0], dirs[li, 1] = d0, near / near.norm()
    path = tmp_path / "d.pt"
    torch.save({"dirs_multi": dirs}, path)

    with pytest.raises(SystemExit, match="not orthonormal"):
        dv.load_directions(a, str(path), lambda m: None)


def test_a_single_direction_per_layer_is_not_called_non_orthonormal(tmp_path):
    """One row is trivially orthonormal; the guard must not fire on a K=1 set."""
    a = _LoadStub()
    dirs = torch.zeros(a.NL + 1, 1, a.H)
    for li in range(1, a.NL + 1):
        dirs[li, 0, 0] = 1.0
    path = tmp_path / "d.pt"
    torch.save({"dirs_multi": dirs}, path)
    dv.load_directions(a, str(path), lambda m: None)
    assert a.KMAX == 1


# ── the unablated anchor ──────────────────────────────────────────────────────────────
def test_the_unablated_anchor_is_preferred_as_the_baseline():
    """"Percent removed" must be relative to the untouched model, not to the gentlest arm.

    The first grid's weakest arm had already stripped most of the refusal, so "90% removed" meant
    90% below an already-gutted 21%. The K ranking was unharmed (every arm shares one anchor) but
    every percentage in the report read stronger than it was.
    """
    rows = [
        {"arm": "unablated", "K": 1, "strength": 0.0, "random_extras": False,
         "harmful_refusal": 0.72, "harmless_refusal": 0.0, "kl": 0.0},
        {"arm": "fitted-K1", "K": 1, "strength": 0.5, "random_extras": False,
         "harmful_refusal": 0.20, "harmless_refusal": 0.0, "kl": 0.10},
    ]
    table = dv.matched_refusal_table(rows)
    assert table["baseline_refusal"] == pytest.approx(0.72)
    assert table["baseline_is_unablated"] is True


def test_without_an_anchor_the_baseline_falls_back_and_says_so():
    rows = [{"arm": "fitted-K1", "K": 1, "strength": 0.5, "random_extras": False,
             "harmful_refusal": 0.20, "harmless_refusal": 0.0, "kl": 0.10},
            {"arm": "fitted-K3", "K": 3, "strength": 0.5, "random_extras": False,
             "harmful_refusal": 0.05, "harmless_refusal": 0.0, "kl": 0.30}]
    table = dv.matched_refusal_table(rows)
    assert table["baseline_refusal"] == pytest.approx(0.20)
    assert table["baseline_is_unablated"] is False


def test_the_anchor_is_excluded_from_the_spread_check():
    """The anchor is unablated by definition, so it must not make a flat grid look varied."""
    flat = [{"arm": "unablated", "K": 1, "strength": 0.0, "random_extras": False,
             "harmful_refusal": 0.72, "harmless_refusal": 0.0, "kl": 0.0}]
    flat += [{"arm": f"fitted-K{k}", "K": k, "strength": 1.0, "random_extras": False,
              "harmful_refusal": 0.0, "harmless_refusal": 0.0, "kl": 0.3 * k} for k in (1, 2, 3)]
    reason = dv.degenerate_reason(flat)
    assert reason and "no dynamic range" in reason, (
        "the anchor disguised a floored grid as a grid with spread")


# ── the bake-and-measure arm, against the tiny model ──────────────────────────────────
def _ready_abl(base_args, tiny_model, tiny_tok, track):
    """An Abliterator with directions, eval sets and a snapshot: what `_measure` expects."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    a.bad_eval = a.load(f"{track}/bad_eval_ds", 4)
    a.kl_eval = a.load(f"{track}/good_ds", 4)
    a.orig_lp = a.first_token_logprobs(a.kl_eval)
    a.snapshot_weights()
    return a


def test_measure_reports_both_arms_and_the_kl(base_args, tiny_model, tiny_tok, track):
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)
    row = dv._measure(a, "fitted-K1-s0.5", 1, lambda m: None, strength=0.5)

    assert row["arm"] == "fitted-K1-s0.5" and row["K"] == 1
    assert row["strength"] == pytest.approx(0.5)
    assert row["random_extras"] is False
    for key in ("harmful_refusal", "harmless_refusal", "kl"):
        assert isinstance(row[key], float)
    assert 0.0 <= row["harmful_refusal"] <= 1.0


def test_measure_restores_the_weights_it_baked(base_args, tiny_model, tiny_tok, track):
    """Every arm must start from the pristine model, or arm N measures arms 1..N-1 as well."""
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)
    before = [p.detach().clone() for p in a.model.parameters()]
    dv._measure(a, "fitted-K1", 1, lambda m: None, strength=1.0)
    for b, p in zip(before, a.model.parameters(), strict=True):
        assert torch.equal(b, p), "a measured arm left the weights changed"


def test_measure_restores_the_directions_after_randomising_them(
        base_args, tiny_model, tiny_tok, track):
    """The random control must not leak into the fitted arms that follow it."""
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)
    before = a.dirs_multi.clone()
    dv._measure(a, "random-K2", 2, lambda m: None, randomise_extras=True, seed=1, strength=1.0)
    assert torch.equal(before, a.dirs_multi), "the randomised directions were not put back"


def test_the_randomised_extras_keep_the_primary_direction(base_args, tiny_model, tiny_tok, track):
    """The control varies only the EXTRAS. Replacing direction 0 would change what is compared."""
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)
    primary = a.dirs_multi[:, 0].clone()
    seen = {}

    real_bake = a.bake

    def spy(*args, **kw):
        seen["primary"] = a.dirs_multi[:, 0].clone()
        return real_bake(*args, **kw)

    a.bake = spy
    dv._measure(a, "random-K2", 2, lambda m: None, randomise_extras=True, seed=1, strength=1.0)
    assert torch.equal(primary, seen["primary"]), "the control replaced the primary direction"


def test_measure_at_zero_strength_is_the_unablated_anchor(base_args, tiny_model, tiny_tok, track):
    """Strength 0 must leave the model alone, or the anchor is not an anchor."""
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)
    before = [p.detach().clone() for p in a.model.parameters()]
    row = dv._measure(a, "unablated", 1, lambda m: None, strength=0.0)
    assert row["strength"] == 0.0
    for b, p in zip(before, a.model.parameters(), strict=True):
        assert torch.equal(b, p)


# ── the grid and the entry point ──────────────────────────────────────────────────────
def test_experiment_4_runs_the_anchor_the_grid_and_the_control(
        base_args, tiny_model, tiny_tok, track):
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)
    out = dv.experiment_4(a, lambda m: None, [0.5, 1.0])

    arms = [r["arm"] for r in out["rows"]]
    assert any(r["strength"] == 0.0 for r in out["rows"]), "no unablated anchor was measured"
    assert any(r["random_extras"] for r in out["rows"]) or max(a.dirs_per_layer) < 2
    assert out["strengths"] == [0.5, 1.0]
    assert "matched_refusal" in out and "degenerate_reason" in out
    assert len(arms) == len(set(arms)), "two arms share a label, so one overwrites the other"


def test_experiment_1_runs_against_the_tiny_model(base_args, tiny_model, tiny_tok, track):
    """The leave-one-cluster-out path, end to end, on a corpus too small for usable folds.

    The toy track cannot fill two clusters of MIN_CLUSTER_ROWS, so the honest answer is "no
    usable folds" and this asserts the experiment says that rather than dividing by zero.
    """
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.args.track = track
    out = dv.experiment_1(a, lambda m: None)
    assert "verdict" in out and "folds" in out


def test_the_subcommand_is_reachable_through_the_cli():
    """`senbonzakura validate` must dispatch, or the promotion from tools/ did nothing."""
    assert "validate" in cli.DELEGATED
    assert cli._delegate("validate") is dv.main


def test_the_subcommand_is_documented_in_the_help():
    """A dispatched command nobody can find is not a command."""
    text = cli.build_parser().format_help()
    assert "validate" in text
    assert "matched refusal" in text


def test_main_runs_e1_and_writes_a_record(base_args, tiny_model, tiny_tok, track,
                                          tmp_path, monkeypatch):
    out = tmp_path / "v.json"
    real = cli.Abliterator
    monkeypatch.setattr(cli, "Abliterator",
                        lambda args, log: real(args, log, model=tiny_model, tok=tiny_tok))
    rc = dv.main(["--model", "fixture", "--track", track, "--device", "cpu",
                  "--experiment", "e1", "--dir-prompts", "8", "--out", str(out)])
    assert rc == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["experiment"] == "e1" and "e1" in record
    assert record["directions_from"] is None


def test_main_record_is_written_atomically(base_args, tiny_model, tiny_tok, track,
                                           tmp_path, monkeypatch):
    out = tmp_path / "v.json"
    real = cli.Abliterator
    monkeypatch.setattr(cli, "Abliterator",
                        lambda args, log: real(args, log, model=tiny_model, tok=tiny_tok))
    monkeypatch.setattr(dv.json, "dump", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        dv.main(["--model", "fixture", "--track", track, "--device", "cpu",
                 "--experiment", "e1", "--dir-prompts", "8", "--out", str(out)])
    assert not out.exists() and not list(tmp_path.glob("*.part"))


# ── the shared preparation, which was two copies for an afternoon ─────────────────────
def test_prepare_installs_directions_eval_sets_and_a_snapshot(base_args, tiny_model, tiny_tok,
                                                              track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.args.track = track
    dv.prepare_for_bakes(a, lambda m: None)

    assert a.dirs_multi is not None and a.dirs_per_layer
    assert a.bad_eval and a.kl_eval
    assert a.orig_lp is not None
    assert a._pristine, "no pristine snapshot, so an arm could not be restored"


def test_prepare_says_so_when_the_kl_set_is_not_disjoint(base_args, tiny_model, tiny_tok, track):
    """Measuring coherence on the prompts the directions were fitted on flatters it.

    The fallback is legitimate on a small corpus and must not be silent, because the number it
    produces is not comparable to one measured on a held-out slice.
    """
    lines = []
    base_args.dir_prompts = 64          # more than the toy harmless set holds
    a = cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok)
    a.args.track = track
    dv.prepare_for_bakes(a, lines.append)

    joined = "\n".join(lines)
    assert "too few to spare a slice" in joined
    assert "flatters it" in joined


def test_prepare_uses_a_disjoint_kl_slice_when_it_can(base_args, tiny_model, tiny_tok, track,
                                                      monkeypatch):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.args.track = track
    a.args.dir_prompts = 2
    a.args.eval_kl = 2

    lines = []
    dv.prepare_for_bakes(a, lines.append)
    fitted = a.load(f"{track}/good_ds", a.args.dir_prompts)
    assert not (set(a.kl_eval) & set(fitted)), "the coherence set overlaps the fitted prompts"
    assert "too few to spare" not in "\n".join(lines)


def test_prepare_can_install_an_external_direction_set(base_args, tiny_model, tiny_tok, track,
                                                       tmp_path):
    """The RDO path: directions come from a file, everything else is identical."""
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.args.track = track
    dirs = torch.zeros(a.NL + 1, 1, a.H)
    for li in range(1, a.NL + 1):
        dirs[li, 0, 0] = 1.0
    path = tmp_path / "d.pt"
    torch.save({"dirs_multi": dirs}, path)

    dv.prepare_for_bakes(a, lambda m: None, str(path))
    assert a.KMAX == 1
    assert a.dirs_per_layer[0] == 0 and a.dirs_per_layer[1] == 1
    assert a.bad_eval and a.orig_lp is not None


def test_experiments_2_and_3_runs_both_halves(base_args, tiny_model, tiny_tok, track):
    a = cli.Abliterator(base_args, lambda m: None, model=tiny_model, tok=tiny_tok)
    a.args.track = track
    out = dv.experiments_2_and_3(a, lambda m: None, "all")

    assert out["rows"], "no arms were measured"
    assert "max_k_available" in out
    labels = [r["arm"] for r in out["rows"]]
    assert any(x.startswith("fitted-K") for x in labels)


def test_main_runs_the_e4_grid_and_records_it(base_args, tiny_model, tiny_tok, track,
                                              tmp_path, monkeypatch):
    out = tmp_path / "v.json"
    real = cli.Abliterator
    monkeypatch.setattr(cli, "Abliterator",
                        lambda args, log: real(args, log, model=tiny_model, tok=tiny_tok))
    rc = dv.main(["--model", "fixture", "--track", track, "--device", "cpu",
                  "--experiment", "e4", "--strengths", "0.5,1.0",
                  "--dir-prompts", "4", "--eval-refusal", "2", "--eval-kl", "2",
                  "--out", str(out)])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["experiment"] == "e4"
    assert record["e4"]["strengths"] == [0.5, 1.0]
    assert "matched_refusal" in record["e4"]
    # The status has to FOLLOW the verdict in the record rather than be asserted beside it. This
    # used to read `assert rc == 0`, which passed because the command returned a literal zero for
    # every outcome it could reach. On this fixture the grid is legitimately degenerate: a toy
    # model with random weights has no spread to read, so 1 is the correct answer here and 0 would
    # be the bug. Tying the two together is what makes this test able to fail.
    degenerate = record["e4"]["degenerate_reason"]
    assert rc == (1 if degenerate else 0), (
        f"the record says degenerate_reason={degenerate!r} and the command exited {rc}")


def test_all_prepares_exactly_once(base_args, tiny_model, tiny_tok, track, tmp_path, monkeypatch):
    """`--experiment all` runs e2/e3 and then e4, and both need the same setup.

    Preparing twice would re-snapshot weights that an arm had already baked and restore to the
    wrong pristine state, so every arm after the second preparation would be measured against a
    model that is not the original.
    """
    calls = []
    real = dv.prepare_for_bakes
    monkeypatch.setattr(dv, "prepare_for_bakes",
                        lambda a, log, d=None: (calls.append(1), real(a, log, d))[1])
    real_abl = cli.Abliterator
    monkeypatch.setattr(cli, "Abliterator",
                        lambda args, log: real_abl(args, log, model=tiny_model, tok=tiny_tok))

    dv.main(["--model", "fixture", "--track", track, "--device", "cpu", "--experiment", "all",
             "--strengths", "1.0", "--dir-prompts", "4", "--eval-refusal", "2", "--eval-kl", "2",
             "--out", str(tmp_path / "v.json")])
    assert len(calls) == 1, f"prepared {len(calls)} times; a second snapshot loses the pristine weights"


def test_the_matched_band_picks_the_arm_closest_to_the_target_not_the_cheapest():
    """The selection that decided a verdict wrongly on 2026-08-04.

    K=1 reached 14.8% refusal for KL 0.019; K=2 reached 17.2% for KL 0.018. Picking the cheapest
    arm that CLEARED the target crowned K=2, which had simply done less work. Two arms that
    overshoot by different amounts are not at the same refusal level, and comparing their KL is
    the mistake matching exists to prevent.
    """
    base = 0.766
    rows = [
        {"arm": "anchor", "K": 1, "strength": 0.0, "random_extras": False,
         "harmful_refusal": base, "harmless_refusal": 0.0, "kl": 0.0},
        # K=1 overshoots hard and is cheap for how far it went.
        {"arm": "fitted-K1", "K": 1, "strength": 1.0, "random_extras": False,
         "harmful_refusal": 0.148, "harmless_refusal": 0.0, "kl": 0.019},
        # K=1 also has an arm that lands ON the target.
        {"arm": "fitted-K1-near", "K": 1, "strength": 0.6, "random_extras": False,
         "harmful_refusal": 0.380, "harmless_refusal": 0.0, "kl": 0.009},
        {"arm": "fitted-K2", "K": 2, "strength": 1.0, "random_extras": False,
         "harmful_refusal": 0.172, "harmless_refusal": 0.0, "kl": 0.018},
    ]
    t = dv.matched_refusal_table(rows)
    band = t["targets"]["50%_removed"]
    # The K=1 entry must be the arm that landed near 0.383, not the one that blew past it.
    assert band["1"]["harmful_refusal"] == pytest.approx(0.380)
    assert band["1"]["kl"] == pytest.approx(0.009)
    # And K=2, which overshot to 0.172, is not in this band AT ALL. Membership is two-sided, so an
    # arm that blew past the target belongs to a deeper band rather than sitting in this one with
    # a large overshoot for a reader to notice. That is the stronger form of the same fix.
    assert "2" not in band, "an arm that overshot by 0.21 is not at this band's refusal level"
    assert band["1"]["overshoot"] == pytest.approx(0.003, abs=0.01)
    # It does land in a deeper band, so the arm is not lost, only filed correctly.
    assert any("2" in b for name, b in t["targets"].items() if name != "50%_removed")


def test_every_matched_entry_reports_its_overshoot():
    """A band entry without its distance from the target cannot be read as matched or not."""
    rows = [{"arm": "anchor", "K": 1, "strength": 0.0, "random_extras": False,
             "harmful_refusal": 0.8, "harmless_refusal": 0.0, "kl": 0.0},
            {"arm": "fitted-K1", "K": 1, "strength": 1.0, "random_extras": False,
             "harmful_refusal": 0.2, "harmless_refusal": 0.0, "kl": 0.05}]
    table = dv.matched_refusal_table(rows)
    assert table["target_tolerance"] > 0
    for band in table["targets"].values():
        for entry in band.values():
            assert "overshoot" in entry


# ── the diagnostic that explains a cluster failure ────────────────────────────────────
def _aligned_clusters(n_per=40, H=32, NL1=5, seed=0):
    """Clusters that differ from harmless only in DEGREE, all along one axis.

    What "amounts of refusal" looks like: every group sits further along the same direction. The
    published methods cluster prompt text into semantic categories instead, which would give
    groups differing in direction rather than magnitude.
    """
    torch.manual_seed(seed)
    groups = []
    for t in range(4):
        g = torch.randn(n_per, H) * 0.3
        g[:, 0] += 3.0 + 2.0 * t          # same direction, different amount
        groups.append(g)
    Rb = torch.cat(groups)
    Rg = torch.randn(4 * n_per, H) * 0.3
    Rg[:, 1] += 5.0
    return _stack(Rb, NL1), _stack(Rg, NL1)


def test_clusters_that_differ_only_in_degree_are_named_as_such():
    r = _run(*_aligned_clusters())
    al = r["cluster_alignment"]
    assert al, "no alignment was measured"
    assert al["median_residual_fraction"] < 0.1
    assert "amounts of refusal rather than kinds" in al["reading"]


def test_clusters_pointing_different_ways_are_not_blamed_on_alignment():
    """The multi-component fixture has genuinely different directions; the diagnostic must say so."""
    r = _run(*_multi_component_refusal())
    al = r["cluster_alignment"]
    assert al["median_residual_fraction"] >= 0.1
    assert "genuinely different directions" in al["reading"]


def test_the_alignment_diagnostic_reports_its_sample_size():
    r = _run(*_topic_only())
    al = r["cluster_alignment"]
    assert al["clusters_measured"] > 0
    assert 0.0 <= al["fraction_above_0_99"] <= 1.0


# ── the transfer gap ──────────────────────────────────────────────────────────────────
def test_the_hook_uses_the_bakes_own_layer_profile(base_args, model_factory, tiny_tok, track):
    """Give the hook its own profile and the comparison measures the profile, not the operation.

    The registered hooks are invoked directly rather than through a forward pass: the shared
    fixture's layer modules have no `forward` of their own, which is the same reason a hook on
    them never fires, and is exactly why the transfer question needs asking on a real model.
    """
    a = _ready_abl(base_args, model_factory(H=8, NL=12, V=16), tiny_tok, track)
    NL = a.NL
    P, D = int(NL * 0.6), max(2, NL // 4)
    outside = [i for i in range(NL) if abs(i - P) > D]
    inside = [i for i in range(NL) if cli.layer_weight(i, P, 1.0, 0.0, D) > 0]
    assert outside and inside, "the fixture needs layers on both sides of the window"

    x = torch.randn(1, 2, a.H)
    with dv.residual_ablation(a, 1, P, 1.0, 0.0, D):
        hooks = {i: list(layer._forward_hooks.values())[-1] for i, layer in enumerate(a.layers)}
        for i in outside:
            assert cli.layer_weight(i, P, 1.0, 0.0, D) == 0.0
            assert torch.equal(hooks[i](a.layers[i], None, x), x), (
                f"layer {i} is outside the bake window and was ablated anyway")
        touched = [i for i in inside
                   if not torch.allclose(hooks[i](a.layers[i], None, x), x, atol=1e-6)]
        assert touched, "no layer inside the window was ablated, so this proves nothing"


def test_the_hook_scales_with_the_layer_weight(base_args, model_factory, tiny_tok, track):
    """Strength must taper with distance from the window centre, as the bake's does."""
    a = _ready_abl(base_args, model_factory(H=8, NL=12, V=16), tiny_tok, track)
    NL = a.NL
    P, D = int(NL * 0.6), max(2, NL // 4)
    x = torch.randn(1, 2, a.H)

    with dv.residual_ablation(a, 1, P, 1.0, 0.0, D):
        hooks = {i: list(layer._forward_hooks.values())[-1] for i, layer in enumerate(a.layers)}
        removed = {}
        for i in range(NL):
            if cli.layer_weight(i, P, 1.0, 0.0, D) > 0:
                removed[i] = float((x - hooks[i](a.layers[i], None, x)).norm())

    # The centre of the window must remove at least as much as its edge.
    edge = min(removed, key=lambda i: cli.layer_weight(i, P, 1.0, 0.0, D))
    assert removed[P] >= removed[edge], (
        f"the window centre removed {removed[P]:.4f} and its edge {removed[edge]:.4f}")


def test_the_hooks_are_removed_afterwards(base_args, tiny_model, tiny_tok, track):
    """A leaked hook would silently ablate the bake measurement that follows it."""
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)
    before = [len(layer._forward_hooks) for layer in a.layers]
    with dv.residual_ablation(a, 1, int(a.NL * 0.6), 1.0, 0.0, 2):
        during = [len(layer._forward_hooks) for layer in a.layers]
    after = [len(layer._forward_hooks) for layer in a.layers]
    assert during != before, "no hook was installed"
    assert after == before, "a hook outlived its context"


def test_transfer_gap_restores_the_weights(base_args, tiny_model, tiny_tok, track):
    """It bakes to measure the bake arm, so it must put the model back."""
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)
    before = [p.detach().clone() for p in a.model.parameters()]
    dv.transfer_gap(a, lambda m: None, K=1, strength=1.0)
    for b, p in zip(before, a.model.parameters(), strict=True):
        assert torch.equal(b, p), "the transfer measurement left the model baked"


def test_transfer_gap_reports_both_arms_and_their_difference(base_args, tiny_model, tiny_tok,
                                                             track):
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)
    row = dv.transfer_gap(a, lambda m: None, K=1, strength=1.0)
    assert set(row) == {"K", "strength", "hook", "bake", "refusal_delta_bake_minus_hook"}
    for arm in ("hook", "bake"):
        assert {"harmful_refusal", "harmless_refusal", "kl"} <= set(row[arm])
    assert row["refusal_delta_bake_minus_hook"] == pytest.approx(
        row["bake"]["harmful_refusal"] - row["hook"]["harmful_refusal"], abs=1e-4)


def test_agreement_and_disagreement_get_different_readings(base_args, tiny_model, tiny_tok, track,
                                                           monkeypatch):
    """The verdict must depend on the numbers, not be constant."""
    a = _ready_abl(base_args, tiny_model, tiny_tok, track)

    monkeypatch.setattr(dv, "transfer_gap",
                        lambda *args, **kw: {"K": 1, "strength": 1.0, "hook": {}, "bake": {},
                                             "refusal_delta_bake_minus_hook": 0.001})
    assert "not the problem" in dv.experiment_transfer(a, lambda m: None, [1], [1.0])["reading"]

    monkeypatch.setattr(dv, "transfer_gap",
                        lambda *args, **kw: {"K": 1, "strength": 1.0, "hook": {}, "bake": {},
                                             "refusal_delta_bake_minus_hook": 0.42})
    r = dv.experiment_transfer(a, lambda m: None, [1], [1.0])
    assert "optimising an operation the evaluation does not perform" in r["reading"]
    assert r["max_abs_delta"] == pytest.approx(0.42)


# ── projection magnitude: purity is not effect ────────────────────────────────────────
class _MagStub:
    """An Abliterator shaped just enough for `projection_magnitude`."""

    def __init__(self, dirs_multi, Rb, NL, H):
        self.dirs_multi, self._Rb, self.NL, self.H = dirs_multi, Rb, NL, H
        self.args = type("A", (), {"track": "t", "good_ds": None, "dir_prompts": 8})()

    def load(self, path, n):
        return ["p"] * n

    def collect_resid(self, prompts):
        return self._Rb


def _mag_case(second_scale):
    """Two directions; the second occupies `second_scale` of the activation the first does."""
    NL, H, N = 12, 8, 32
    torch.manual_seed(0)
    dirs = torch.zeros(NL + 1, 2, H)
    for li in range(NL + 1):
        dirs[li, 0, 0] = 1.0
        dirs[li, 1, 1] = 1.0
    Rb = torch.zeros(NL + 1, N, H)
    Rb[:, :, 0] = 10.0                       # lots of activation along direction 0
    Rb[:, :, 1] = 10.0 * second_scale        # and this much along direction 1
    return _MagStub(dirs, Rb, NL, H)


def test_a_direction_the_activations_barely_occupy_is_named_as_such():
    """The failure arXiv:2603.22061 describes: geometrically fine, functionally too small."""
    r = dv.projection_magnitude(_mag_case(0.05), lambda m: None, K=2)
    assert r["weakest_fraction_of_primary"] == pytest.approx(0.05, abs=0.01)
    assert "cannot do much whatever its separation score says" in r["reading"]


def test_directions_of_comparable_magnitude_are_not_blamed_on_size():
    r = dv.projection_magnitude(_mag_case(0.8), lambda m: None, K=2)
    assert r["weakest_fraction_of_primary"] > 0.25
    assert "not explained by magnitude" in r["reading"]


def test_the_primary_direction_is_its_own_reference():
    r = dv.projection_magnitude(_mag_case(0.5), lambda m: None, K=2)
    assert r["per_direction"]["0"]["fraction_of_primary"] == pytest.approx(1.0, abs=1e-3)


def test_the_random_floor_always_covers_the_largest_direction_count():
    """The arm a headline quotes is the highest K, so it is the one that most needs a floor.

    It used to be the first two K above 1 and nothing else. On 2026-08-04 that left the K=4
    headline compared against random controls at K=2 and K=3, a different direction count.
    """
    for ks, want in (([1, 2, 3, 4], {2, 3, 4}),
                     ([1, 2, 3, 5, 8], {2, 3, 8}),
                     ([1, 2], {2}),
                     ([1], set())):
        floor_ks = dv.floor_direction_counts(ks)
        assert set(floor_ks) == want, (ks, floor_ks)
        multi = [k for k in ks if k > 1]
        if multi:
            assert max(multi) in floor_ks, f"the largest K went unfloored for {ks}"
        assert len(floor_ks) <= 3, "the floor must stay bounded in cost"


# ── the comparison gate (added 2026-08-05) ────────────────────────────────────────────
def _band(*pairs):
    """A band of (K, refusal, kl) triples in the shape matched_refusal_table emits."""
    return {str(K): {"kl": kl, "strength": 1.0, "harmful_refusal": ref, "overshoot": 0.0}
            for K, ref, kl in pairs}


def test_arms_at_different_refusal_levels_are_refused_a_ranking():
    """More ablation always costs more KL, so ranking unmatched arms compares strengths."""
    # Baseline 0.8, tolerance 0.05 -> arms may span at most 0.04 in refusal. These span 0.10.
    v = dv.rank_band(_band((1, 0.10, 0.90), (2, 0.20, 0.10)), baseline=0.8, tolerance=0.05, n_eval=1000)
    assert v["comparable"] is False
    assert "not at the same level" in v["reason"]
    assert "cheapest" not in v, "an unrankable band must not name a winner"


def test_a_one_percent_kl_difference_is_a_tie_not_a_win():
    """The exact shape of the 2026-08-04 report: 0.0088 against 0.0089, crowned MULTI WINS."""
    v = dv.rank_band(_band((1, 0.15, 0.0089), (2, 0.15, 0.0088)), baseline=0.8, tolerance=0.05, n_eval=1000)
    assert v["comparable"] is True
    assert v["cheapest"] is None, "a 1% difference was reported as a winner"
    assert "tie" in v["reason"]


def test_a_real_advantage_is_ranked_and_its_margin_reported():
    v = dv.rank_band(_band((1, 0.15, 0.80), (4, 0.16, 0.10)), baseline=0.8, tolerance=0.05, n_eval=1000)
    assert v["comparable"] is True and v["cheapest"] == 4
    assert v["margin"] == pytest.approx(8.0, abs=0.01)


def test_a_band_only_one_arm_reached_is_not_a_comparison():
    v = dv.rank_band(_band((4, 0.15, 0.10)), baseline=0.8, tolerance=0.05, n_eval=1000)
    assert v["comparable"] is False and "nothing to compare" in v["reason"]


def test_a_zero_kl_arm_cannot_win_by_division():
    """A degenerate 0.0 KL would otherwise divide by zero or claim an infinite margin.

    It is refused a ranking outright rather than called a tie: a reason string saying that 0.0 and
    0.5 are "within 5% of each other" was factually false, and the same wording appeared for
    negative KL.
    """
    for bad in (0.0, -0.001):
        v = dv.rank_band(_band((1, 0.15, bad), (2, 0.15, 0.5)), baseline=0.8, tolerance=0.05, n_eval=1000)
        assert v["comparable"] is False, bad
        assert "cheapest" not in v, "a non-positive KL must not produce a winner"
        assert "not a positive divergence" in v["reason"]
        assert "within" not in v["reason"], "the tie wording must not describe a non-positive KL"


def test_the_ranking_travels_beside_the_table_not_inside_the_bands():
    """Verdict keys mixed in with the arms break every consumer that iterates a band."""
    rows = [{"arm": "unablated-K1-s0", "K": 1, "strength": 0.0, "random_extras": False,
             "harmful_refusal": 0.8, "harmless_refusal": 0.0, "kl": 0.0}]
    for K, ref, kl in ((1, 0.20, 0.9), (2, 0.21, 0.1)):
        rows.append({"arm": f"fitted-K{K}", "K": K, "strength": 1.0, "random_extras": False,
                     "harmful_refusal": ref, "harmless_refusal": 0.0, "kl": kl})
    t = dv.matched_refusal_table(rows)
    assert "ranking" in t
    for band, entries in t["targets"].items():
        for key, entry in entries.items():
            assert key.isdigit(), f"{key!r} is not a direction count"
            assert set(entry) == {"kl", "strength", "harmful_refusal", "overshoot"}
        assert band in t["ranking"]


# ── provenance and the statistical matching tolerance (added 2026-08-05) ──────────────
def test_the_code_version_is_recorded_or_honestly_unknown():
    """Records with no build stamp cannot be interpreted after the library changes underneath."""
    v = cli.code_version()
    assert isinstance(v, str) and v
    assert v.startswith("unknown") or any(ch.isalnum() for ch in v)


def test_a_direction_sidecar_is_folded_into_the_result(tmp_path):
    """Two ladder arms differ by what is INSIDE the direction file, not by its name."""
    pt = tmp_path / "dirs.pt"
    (tmp_path / "dirs.pt.json").write_text(
        json.dumps({"init": "mean-diff", "induce": 0.2, "score": "per-token",
                    "history": [{"step": 0}]}), encoding="utf-8")
    meta = dv._directions_meta(str(pt))
    assert meta["init"] == "mean-diff" and meta["induce"] == 0.2
    assert "history" not in meta, "the optimisation history bloats every record it lands in"


def test_a_missing_sidecar_is_reported_rather_than_silently_absent():
    meta = dv._directions_meta("/definitely/not/here.pt")
    assert meta and "error" in meta


def test_no_directions_file_means_no_sidecar():
    assert dv._directions_meta(None) is None


def test_more_evaluation_prompts_make_equivalence_easier_not_harder():
    """THE INVERSION THIS REPLACES, and it is the one that matters.

    The old rule compared the observed spread to two standard errors of the difference and
    ranked the arms whenever the spread was smaller. That treats failure to detect a difference
    as evidence of equivalence, and it produces a gate that gets EASIER the less data you
    collect: at p=0.15 the width is 0.128 on 64 prompts and 0.032 on a thousand, so two arms
    differing by twelve points were certified "at the same level" on a small evaluation and
    refused on a large one. The guide calls matched refusal "the whole ball game" and it was
    enforced by a rule that rewarded collecting less.

    Two one-sided tests instead: the WHOLE interval for the difference must sit inside a margin
    declared in advance. A small evaluation now certifies nothing.
    """
    small, _s, _u, why_small = dv.refusal_equivalent([0.15, 0.16], n_eval=64)
    large, _s2, _u2, _w2 = dv.refusal_equivalent([0.15, 0.16], n_eval=1000)
    assert large is True, "a one-point gap on a thousand prompts is equivalence by any reading"
    assert small is False, "sixty-four prompts cannot pin a difference tightly enough to certify"
    assert "not detecting a difference is not the same" in why_small.lower()


def test_a_gap_past_the_margin_is_refused_at_any_evaluation_size():
    """The margin is a judgement about what matters, so no amount of data makes 12 points fine."""
    for n in (64, 1000, 100000):
        ok, spread, _upper, why = dv.refusal_equivalent([0.15, 0.27], n_eval=n)
        assert ok is False
        assert spread == pytest.approx(0.12)
        assert "past the" in why


def test_without_an_evaluation_size_nothing_can_be_certified():
    """The honest answer to "how precise is this" when nothing recorded the sample size."""
    ok, _spread, upper, why = dv.refusal_equivalent([0.15, 0.16], n_eval=None)
    assert ok is False
    assert upper is None
    assert "was not recorded" in why


def test_a_zero_refusal_band_does_not_collapse_its_own_interval():
    """At p=0 the binomial variance is zero, which would certify every pair as equivalent."""
    ok, _spread, upper, _why = dv.refusal_equivalent([0.0, 0.0], n_eval=128)
    assert upper > 0, "a zero-variance interval would make any pair look identical"
    assert ok is True, "two arms both at zero refusal genuinely are at the same level"


def test_the_margin_is_a_declared_number_rather_than_one_derived_from_the_data():
    """An equivalence claim rests on this, so it is in the open with a reason beside it."""
    assert pytest.approx(0.05) == dv.EQUIVALENCE_MARGIN



def test_a_grid_with_no_dynamic_range_is_refused_a_ranking():
    """The 2026-08-04 failure in its purest form: every arm identical, so spread is trivially 0.

    A zero spread passes the matched-level test by construction, so without the grid's own
    degeneracy verdict a floored grid produces a confident winner in every band.
    """
    rows = [{"arm": "unablated-K1-s0", "K": 1, "strength": 0.0, "random_extras": False,
             "harmful_refusal": 0.72, "harmless_refusal": 0.0, "kl": 0.0}]
    rows.extend({"arm": f"fitted-K{K}", "K": K, "strength": 1.0, "random_extras": False,
                 "harmful_refusal": 0.0, "harmless_refusal": 0.0, "kl": 0.1 * K}
                for K in (1, 2, 3, 4))
    t = dv.matched_refusal_table(rows, n_eval=128)
    assert t["degenerate_reason"], "this grid has no dynamic range and the table should say so"
    for band, verdict in t["ranking"].items():
        assert verdict["comparable"] is False, band
        assert "cheapest" not in verdict, f"{band} named a winner from a floored grid"


def test_an_empty_table_keeps_the_same_key_set_as_a_full_one():
    """A reader of the normal shape must not KeyError on the empty one."""
    only_random = [{"arm": "random-K2", "K": 2, "strength": 1.0, "random_extras": True,
                    "harmful_refusal": 0.2, "harmless_refusal": 0.0, "kl": 0.1}]
    empty = dv.matched_refusal_table(only_random)
    full = dv.matched_refusal_table([
        *only_random,
        {"arm": "unablated-K1-s0", "K": 1, "strength": 0.0, "random_extras": False,
         "harmful_refusal": 0.8, "harmless_refusal": 0.0, "kl": 0.0}])
    assert set(full) - set(empty) == set(), f"missing from the empty shape: {set(full) - set(empty)}"


def test_a_band_with_no_arms_says_so_rather_than_claiming_one_arm():
    assert "no arm reached" in dv.rank_band({}, baseline=0.8, tolerance=0.05)["reason"]


def test_a_stamped_sync_carries_the_version_where_git_cannot(tmp_path, monkeypatch):
    """The GPU box is a file copy, not a checkout, so `git describe` returns nothing there.

    That is the one machine whose build actually needs identifying: a run once executed against a
    checkout predating the changes it existed to measure. A sync stamps CODE_VERSION beside the
    package and it is the authority when git is absent.
    """
    import subprocess as sp
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    monkeypatch.setattr(cli, "__file__", str(pkg / "cli.py"))
    monkeypatch.setattr(sp, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git here")))

    assert "neither a git checkout nor a stamped sync" in cli.code_version()
    (pkg / "CODE_VERSION").write_text("v0.3.0-124-gabcdef\n", encoding="utf-8")
    v = cli.code_version()
    assert "v0.3.0-124-gabcdef" in v and "stamped at sync" in v
