# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Per-architecture bake validation: does the weight edit reach the residual stream, per layer type?

The bake edits a convolution's `out_proj` exactly as it edits an attention `o_proj`, because they
are dimensionally identical `[hidden, hidden]` Linears in the same residual position. That is
correct linear algebra and, until this measurement existed, an untested claim about behaviour. It
is not academic: on LFM2.5-8B-A1B, 18 of 24 layers write through a convolution, so it governs 75%
of what the tool edits in that position.

The failure it guards against has happened. On Gemma the edit never reached the residual stream at
all, because a learned-gain normalisation sat between the edited weight and the stream; the
whole-model disagreement was 0.578 against Qwen3's 0.016 and every Gemma number was withdrawn.
`transfer_gap` detects that shape when it applies to a whole model. It cannot see an edit that
lands on some layers and not others, which is the shape a hybrid can fail in.
"""
import pytest
import torch

from senbonzakura import cli, validate


def _prepare(abl):
    """Give the abliterator a real direction set and harmful prompts it can load."""
    abl.args.dir_prompts = 4
    return abl


def test_a_landing_edit_reports_a_reduction(abl, monkeypatch, tmp_path):
    """The healthy case: after the bake, the residual's component along the direction is smaller."""
    _prepare(abl)
    monkeypatch.setattr(abl, "load", lambda *a, **k: ["prompt one", "prompt two"])

    calls = {"n": 0}
    NL, H = abl.NL, abl.H

    def fake_collect(prompts):
        calls["n"] += 1
        # Second call is post-bake: the component along every direction is halved.
        scale = 1.0 if calls["n"] == 1 else 0.5
        return torch.ones(NL + 1, len(prompts), H) * scale

    monkeypatch.setattr(abl, "collect_resid", fake_collect)
    monkeypatch.setattr(abl, "bake", lambda *a, **k: None)
    monkeypatch.setattr(abl, "restore_weights", lambda: None)

    got = validate.bake_reach(abl, lambda _m: None, K=1, strength=1.0)
    assert got["per_layer"], "no edited layers were measured"
    for row in got["per_layer"]:
        assert row["reduction"] == pytest.approx(0.5, abs=1e-3)
    assert not got["architectures_failing"]


def test_an_edit_that_changes_nothing_is_reported_as_not_reaching(abl, monkeypatch):
    """THE gemma shape. The bake runs, reports success, and the stream is untouched."""
    _prepare(abl)
    monkeypatch.setattr(abl, "load", lambda *a, **k: ["p1", "p2"])
    NL, H = abl.NL, abl.H
    monkeypatch.setattr(abl, "collect_resid", lambda p: torch.ones(NL + 1, len(p), H))
    monkeypatch.setattr(abl, "bake", lambda *a, **k: None)
    monkeypatch.setattr(abl, "restore_weights", lambda: None)

    got = validate.bake_reach(abl, lambda _m: None, K=1, strength=1.0)
    assert got["architectures_failing"], "an edit that changed nothing was not flagged"
    assert "does NOT reach" in got["reading"]
    assert "gemma" in got["reading"]


def test_the_reading_names_publication_when_the_edit_does_not_land(abl, monkeypatch):
    _prepare(abl)
    monkeypatch.setattr(abl, "load", lambda *a, **k: ["p"])
    NL, H = abl.NL, abl.H
    monkeypatch.setattr(abl, "collect_resid", lambda p: torch.ones(NL + 1, len(p), H))
    monkeypatch.setattr(abl, "bake", lambda *a, **k: None)
    monkeypatch.setattr(abl, "restore_weights", lambda: None)
    got = validate.bake_reach(abl, lambda _m: None)
    assert "publish" in got["reading"].lower()


def test_no_harmful_prompts_is_refused_rather_than_averaged_over_nothing(abl, monkeypatch):
    _prepare(abl)
    monkeypatch.setattr(abl, "load", lambda *a, **k: [])
    with pytest.raises(ValueError, match="nothing to project"):
        validate.bake_reach(abl, lambda _m: None)


def test_only_layers_the_profile_edits_are_measured(abl, monkeypatch):
    """A layer the weight profile skipped is not evidence about whether the edit lands, and
    including it would dilute the fraction being reported.
    """
    _prepare(abl)
    monkeypatch.setattr(abl, "load", lambda *a, **k: ["p"])
    NL, H = abl.NL, abl.H
    monkeypatch.setattr(abl, "collect_resid", lambda p: torch.ones(NL + 1, len(p), H))
    monkeypatch.setattr(abl, "bake", lambda *a, **k: None)
    monkeypatch.setattr(abl, "restore_weights", lambda: None)
    monkeypatch.setattr(cli, "layer_weight",
                        lambda idx, *a, **k: 1.0 if idx == 0 else 0.0)
    got = validate.bake_reach(abl, lambda _m: None)
    assert [r["layer"] for r in got["per_layer"]] == [0]


def test_a_dense_model_says_there_is_no_cross_type_comparison(abl, monkeypatch):
    """The tiny fixture is a plain decoder: one residual-writing position, so a ratio between
    layer types would be a number about nothing.
    """
    _prepare(abl)
    monkeypatch.setattr(abl, "load", lambda *a, **k: ["p"])
    NL, H = abl.NL, abl.H
    calls = {"n": 0}

    def fake(p):
        calls["n"] += 1
        return torch.ones(NL + 1, len(p), H) * (1.0 if calls["n"] == 1 else 0.2)
    monkeypatch.setattr(abl, "collect_resid", fake)
    monkeypatch.setattr(abl, "bake", lambda *a, **k: None)
    monkeypatch.setattr(abl, "restore_weights", lambda: None)

    got = validate.bake_reach(abl, lambda _m: None)
    assert set(got["by_kind"]) == {"attention"}
    assert "one residual-writing position" in got["reading"]


def test_the_weights_are_restored_even_though_the_measurement_bakes(abl, monkeypatch):
    """It edits the model to measure it. Leaving it edited would poison whatever ran next."""
    _prepare(abl)
    monkeypatch.setattr(abl, "load", lambda *a, **k: ["p"])
    NL, H = abl.NL, abl.H
    monkeypatch.setattr(abl, "collect_resid", lambda p: torch.ones(NL + 1, len(p), H))
    seen = {"baked": False, "restored": False}
    monkeypatch.setattr(abl, "bake", lambda *a, **k: seen.update(baked=True))
    monkeypatch.setattr(abl, "restore_weights", lambda: seen.update(restored=True))
    validate.bake_reach(abl, lambda _m: None)
    assert seen["baked"] and seen["restored"]


def test_reach_is_a_selectable_experiment():
    got = validate.build_args(["--model", "m", "--out", "o.json", "--experiment", "reach"])
    own = got[0] if isinstance(got, tuple) else got
    assert own.experiment == "reach"


def test_reach_also_runs_as_part_of_all():
    """It is cheap next to the grids and it invalidates them when it fails, so it should not be a
    thing somebody has to remember to ask for.
    """
    import inspect
    src = inspect.getsource(validate.main)
    assert '("reach", "all")' in src, "reach does not run under --experiment all"


def test_the_floor_is_a_floor_not_a_target():
    """MIN_REACH exists to catch "did not land at all", not to certify a good edit."""
    assert 0 < validate.MIN_REACH < 0.5


def test_one_dead_layer_is_not_hidden_by_a_healthy_average(abl, monkeypatch):
    """MEASURED, not invented. On LFM2.5-350M the conv layers averaged 0.495 against attention's
    0.759, which reads as healthy, while one conv layer had moved by 0.076. The first version of
    this check compared only means and called that "comparable, so the bake treats both positions
    alike". A summary that hides a per-item failure is the thing this command exists to catch.
    """
    _prepare(abl)
    monkeypatch.setattr(abl, "load", lambda *a, **k: ["p"])
    NL, H = abl.NL, abl.H
    calls = {"n": 0}

    def fake(p):
        calls["n"] += 1
        if calls["n"] == 1:
            return torch.ones(NL + 1, len(p), H)
        # Every layer halves except layer 1, which barely moves.
        out = torch.ones(NL + 1, len(p), H) * 0.5
        out[2] = 0.97
        return out

    monkeypatch.setattr(abl, "collect_resid", fake)
    monkeypatch.setattr(abl, "bake", lambda *a, **k: None)
    monkeypatch.setattr(abl, "restore_weights", lambda: None)

    got = validate.bake_reach(abl, lambda _m: None)
    assert 1 in got["layers_below_floor"], got["per_layer"]
    assert "not everywhere" in got["reading"].lower()
    assert str(1) in got["reading"], "the reading has to name the layer, not just count it"
    # And it must NOT read as the whole-architecture gemma failure, which is a different thing.
    assert "does NOT reach" not in got["reading"]
    # CHANGED 2026-09-07, and the change is the point. A below-floor layer used to be rolled up
    # into `architectures_failing`, which turned one shallow layer into a verdict on its whole
    # layer type. It is a fact about that layer, `layers_below_floor` names it, and the reading
    # says it. Attributing it to an architecture is what the depth-overlap work removed.
    assert got["architectures_failing"] == [], (
        "one layer below the floor must not be reported as its architecture failing")


def test_a_whole_type_failing_still_reads_as_the_gemma_shape(abl, monkeypatch):
    """The two failures want different words: one layer lagging is worth understanding, a whole
    layer type not moving means the numbers are not publishable.
    """
    _prepare(abl)
    monkeypatch.setattr(abl, "load", lambda *a, **k: ["p"])
    NL, H = abl.NL, abl.H
    monkeypatch.setattr(abl, "collect_resid", lambda p: torch.ones(NL + 1, len(p), H))
    monkeypatch.setattr(abl, "bake", lambda *a, **k: None)
    monkeypatch.setattr(abl, "restore_weights", lambda: None)
    got = validate.bake_reach(abl, lambda _m: None)
    assert "does NOT reach" in got["reading"]
    assert "publish" in got["reading"].lower()


def test_experiment_reach_picks_the_lowest_k_and_the_strongest_setting(abl, monkeypatch):
    """The reach question is "does the edit land at all", so it wants the clearest signal: the
    simplest direction budget and the hardest push. Measuring it at a weak strength would report
    a small reduction that means nothing about whether the machinery works.
    """
    seen = {}
    monkeypatch.setattr(validate, "bake_reach",
                        lambda a, log, K=1, strength=1.0: seen.update(K=K, strength=strength) or {})
    validate.experiment_reach(abl, lambda _m: None, [2, 1, 4], [0.3, 1.0, 0.7])
    assert seen["K"] == 1, "reach used a larger direction budget than it needed"
    assert seen["strength"] == 1.0, "reach measured at less than full strength"


def test_experiment_reach_survives_empty_inputs(abl, monkeypatch):
    monkeypatch.setattr(validate, "bake_reach",
                        lambda a, log, K=1, strength=1.0: {"K": K, "strength": strength})
    got = validate.experiment_reach(abl, lambda _m: None, [], [])
    assert got["K"] == 1 and got["strength"] == 1.0
