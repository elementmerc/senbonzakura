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
    assert rc == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["experiment"] == "e4"
    assert record["e4"]["strengths"] == [0.5, 1.0]
    assert "matched_refusal" in record["e4"]


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
    band = dv.matched_refusal_table(rows)["targets"]["50%_removed"]
    # The K=1 entry must be the arm that landed near 0.383, not the one that blew past it.
    assert band["1"]["harmful_refusal"] == pytest.approx(0.380)
    assert band["1"]["kl"] == pytest.approx(0.009)
    assert abs(band["1"]["overshoot"]) < abs(band["2"]["overshoot"])


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
