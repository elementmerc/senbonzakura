# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The instrument against the planted models, and against deliberately broken copies of itself.

A fixture that has never failed has not been shown to work, so every gate in `classify` and the two
places the measurement could be quietly wrong (which block is ablated, whether anything is removed)
get a mutant here, and each mutant must be caught by the fixture built for it.
"""
import dataclasses

import pytest
import torch
from torch import nn

from senbonzakura import attribution as A
from senbonzakura import attribution_fixtures as fx

FAST = dataclasses.replace(A.DEFAULT, resamples=300, shuffled_label_replicates=12,
                           random_direction_replicates=12, split_half_repeats=60)


@pytest.fixture(scope="module")
def runs():
    out = {}
    for name in fx.EXPECTED:
        subject = fx.ToySubject(name)
        out[name] = (subject, fx.partitions(name), A.attribute(subject, fx.partitions(name), seed=0))
    return out


# ---------------------------------------------------------------- the answer key, at the defaults

def test_every_planted_model_gets_its_planted_status_at_the_default_conventions():
    # Run at the DEFAULT conventions and not the reduced ones, because the defaults are what the
    # artefact records and what a reader would reproduce.
    for name, expected in fx.EXPECTED.items():
        subject = fx.ToySubject(name)
        result = A.attribute(subject, fx.partitions(name), seed=0)
        assert result["status"] == expected, (name, result["status_reason"])


def test_localised_names_exactly_the_planted_block(runs):
    result = runs["one_block"][2]
    assert {b["index"] for b in result["blocks"] if b["exceeds_null"]} == {5}


def test_spread_does_not_single_out_blocks_as_if_one_mattered_more(runs):
    result = runs["spread"][2]
    assert result["concentration"]["effective_blocks"] > 0.9 * fx.N_BLOCKS


def test_redundant_has_single_effects_near_zero_and_a_large_total(runs):
    result = runs["redundant"][2]
    assert max(abs(b["effect"]) for b in result["blocks"]) < 0.2 * result["totals"]["effect"]
    assert result["totals"]["effect"] > 0.5


def test_a_negative_result_carries_a_reason_a_person_can_read(runs):
    for name in ("spread", "redundant", "absent", "orthogonal", "bad_readout", "underpowered"):
        reason = runs[name][2]["status_reason"]
        assert len(reason) > 40 and "nan" not in reason.lower(), name


def test_the_orthogonal_plant_is_not_reported_as_spread_out(runs):
    assert runs["orthogonal"][2]["status"] == A.DIRECTION_INERT != A.DIFFUSE


def test_no_blocks_travel_with_a_status_that_gave_no_per_block_answer(runs):
    for name in ("underpowered",):
        assert "blocks" not in runs[name][2]


def test_the_same_seed_gives_the_same_artefact():
    a = A.attribute(fx.ToySubject("one_block"), fx.partitions("one_block"), seed=3, conv=FAST)
    b = A.attribute(fx.ToySubject("one_block"), fx.partitions("one_block"), seed=3, conv=FAST)
    assert a == b


# ------------------------------------------------------------------------------- tint_allowed

def _supported(subject, parts, blocks, direction, ranking):
    return {"ran": True, "supported": True, "blocks": list(blocks)}


def test_tint_is_allowed_only_for_a_localised_result_whose_check_was_supported():
    subject, parts = fx.ToySubject("one_block"), fx.partitions("one_block")
    unchecked = A.attribute(subject, parts, seed=0, conv=FAST)
    checked = A.attribute(subject, parts, seed=0, conv=FAST, predictive=_supported)
    refuted = A.attribute(subject, parts, seed=0, conv=FAST,
                          predictive=lambda *_: {"ran": True, "supported": False})
    assert unchecked["status"] == checked["status"] == A.LOCALISED
    assert unchecked["tint_allowed"] is False
    assert unchecked["predictive_check"]["ran"] is False
    assert checked["tint_allowed"] is True
    assert refuted["tint_allowed"] is False


def test_tint_is_never_allowed_for_any_other_status(runs):
    for name, (_, _, result) in runs.items():
        if result["status"] != A.LOCALISED:
            assert A.tint_allowed({**result, "predictive_check": {"ran": True, "supported": True}}) \
                is False, name
            assert result["tint_allowed"] is False


# ----------------------------------------------------------------- mutants of the classification

def _evidence(runs, name):
    return runs[name][2]["evidence"]


def _status_with(evidence, **changes):
    return A.classify({**evidence, **changes}, A.DEFAULT, fx.N_BLOCKS)[0]


def test_dropping_the_direction_inert_gate_would_misreport_the_orthogonal_plant(runs):
    ev = _evidence(runs, "orthogonal")
    assert _status_with(ev) == A.DIRECTION_INERT
    assert _status_with(ev, total_lo=1.0, t_null=-1.0) != fx.EXPECTED["orthogonal"]


def test_dropping_the_redundancy_check_would_misreport_the_redundant_plant(runs):
    ev = _evidence(runs, "redundant")
    assert _status_with(ev, ratio_hi=1.0) != fx.EXPECTED["redundant"]


def test_dropping_the_readout_gate_would_misreport_the_bad_readout_plant(runs):
    # The bad readout stops before any evidence exists (it is checked before anything is ranked),
    # so its evidence is the localised plant's with the agreement the bad readout reports.
    ev = {**_evidence(runs, "one_block"), "agreement": 0.4}
    assert A.classify(ev, A.DEFAULT, fx.N_BLOCKS)[0] == fx.EXPECTED["bad_readout"]
    assert _status_with(ev, agreement=1.0) != fx.EXPECTED["bad_readout"]


def test_an_unvalidated_readout_stops_the_run_before_any_block_is_ranked(runs):
    result = runs["bad_readout"][2]
    assert result["status"] == A.READOUT_INVALID and "blocks" not in result
    assert result["controls"]["readout_agreement"] == pytest.approx(0.4)


def test_a_readout_that_could_not_be_validated_at_all_is_readout_invalid(runs):
    ev = {**_evidence(runs, "one_block"), "agreement": None}
    status, reason = A.classify(ev, A.DEFAULT, fx.N_BLOCKS)
    assert status == A.READOUT_INVALID and "could not be validated" in reason


def test_dropping_the_absence_gate_would_misreport_the_absent_plant(runs):
    ev = _evidence(runs, "absent")
    assert _status_with(ev, baseline_auc_lo=1.0) != fx.EXPECTED["absent"]


def test_dropping_the_sample_floor_would_misreport_the_underpowered_plant(runs):
    # The underpowered plant stops before any evidence exists, so its evidence is the localised
    # plant's with the sample size it would have had.
    ev = {**_evidence(runs, "one_block"), "n_smaller_arm": 24}
    assert A.classify(ev, A.DEFAULT, fx.N_BLOCKS)[0] == fx.EXPECTED["underpowered"]
    assert _status_with(ev, n_smaller_arm=10**6) != fx.EXPECTED["underpowered"]


def test_dropping_the_concentration_check_would_call_the_spread_plant_localised(runs):
    ev = _evidence(runs, "spread")
    assert _status_with(ev) == A.DIFFUSE
    assert _status_with(ev, n_eff=1.0, reliability=1.0) == A.LOCALISED


def test_a_localised_set_that_halves_do_not_reproduce_is_unresolved_not_localised(runs):
    ev = _evidence(runs, "one_block")
    assert _status_with(ev, reliability=0.1) == A.UNRESOLVED


def test_no_block_clearing_the_null_is_diffuse_even_with_a_real_total(runs):
    ev = _evidence(runs, "one_block")
    assert _status_with(ev, exceeding=[]) == A.DIFFUSE


# ------------------------------------------------------------- mutants of the measurement itself

def _run(name, **patches):
    subject, parts = fx.ToySubject(name), fx.partitions(name)
    return A.attribute(subject, parts, seed=0, conv=FAST)


def test_ablating_the_wrong_block_is_caught_by_the_one_block_plant(monkeypatch):
    real = A.ablating

    def off_by_one(layers, direction, blocks):
        return real(layers, direction, tuple(min(k + 1, len(layers) - 1) for k in blocks))
    monkeypatch.setattr(A, "ablating", off_by_one)
    # fx.ToySubject binds the module, not the name, so the patch reaches the readout.
    result = _run("one_block")
    assert {b["index"] for b in result["blocks"] if b["exceeds_null"]} != {5} \
        or result["status"] != A.LOCALISED


def test_an_ablation_that_removes_nothing_is_caught_by_the_one_block_plant(monkeypatch):
    @A.contextlib.contextmanager
    def nothing(layers, direction, blocks):
        yield
    monkeypatch.setattr(A, "ablating", nothing)
    assert _run("one_block")["status"] != A.LOCALISED


def test_the_family_wise_threshold_is_what_keeps_noise_blocks_out_across_seeds(monkeypatch):
    # One seed cannot show this. A single run clears a noise block only about one time in three,
    # because each block also needs an interval clear of zero, so a mutant with no threshold at all
    # SURVIVED a single seeded run when this was first written. Across sixteen seeds the mutant
    # admits noise blocks and the real threshold never does.
    seeds = range(16)

    def cleared(seed):
        result = A.attribute(fx.ToySubject("one_block", seed=seed),
                             fx.partitions("one_block", seed=seed), seed=seed, conv=FAST)
        return {b["index"] for b in result["blocks"] if b["exceeds_null"]}

    assert all(cleared(s) == {5} for s in seeds)
    real = A._null_replicates

    def per_block(*args, **kwargs):
        out = real(*args, **kwargs)
        out["max_block"] = torch.zeros_like(out["max_block"])
        return out
    monkeypatch.setattr(A, "_null_replicates", per_block)
    assert any(cleared(s) != {5} for s in seeds)


# ---------------------------------------------------------------------------- the hook, directly

class _Plain(nn.Module):
    """A block returning a bare tensor, and reading its input from a keyword."""

    def forward(self, hidden_states):
        return hidden_states + 1.0


class _Tuple(nn.Module):
    def forward(self, hidden_states):
        return (hidden_states + 1.0, "cache")


def test_ablating_removes_only_the_component_along_the_direction():
    layers = nn.ModuleList([_Plain()])
    x = torch.zeros(1, 2, 3)
    d = torch.tensor([1.0, 0.0, 0.0])
    with A.ablating(layers, d, [0]):
        out = layers[0](hidden_states=x)
    # The block wrote (1, 1, 1); its component along x is removed, the rest is kept.
    assert torch.allclose(out, torch.tensor([0.0, 1.0, 1.0]).expand(1, 2, 3))
    assert torch.allclose(layers[0](hidden_states=x), torch.ones(1, 2, 3))


def test_ablating_keeps_the_rest_of_a_tuple_output_and_normalises_the_direction():
    layers = nn.ModuleList([_Tuple()])
    x = torch.zeros(1, 1, 3)
    with A.ablating(layers, torch.tensor([5.0, 0.0, 0.0]), [0]):
        out = layers[0](x)
    assert out[1] == "cache" and torch.allclose(out[0], torch.tensor([0.0, 1.0, 1.0]).expand(1, 1, 3))


def test_ablating_an_empty_list_is_the_unedited_model_and_removes_its_hooks():
    layers = nn.ModuleList([_Plain()])
    with A.ablating(layers, None, []):
        assert not layers[0]._forward_hooks
    with A.ablating(layers, torch.ones(3), [0]):
        assert layers[0]._forward_hooks
    assert not layers[0]._forward_hooks


def test_ablating_removes_its_hooks_even_when_the_forward_pass_fails():
    layers = nn.ModuleList([_Plain()])
    with pytest.raises(RuntimeError), A.ablating(layers, torch.ones(3), [0]):
        raise RuntimeError("boom")
    assert not layers[0]._forward_hooks


@pytest.mark.parametrize("blocks", [[1], [-1]])
def test_ablating_a_block_that_does_not_exist_fails_loudly(blocks):
    with pytest.raises(A.AttributionError, match="do not exist"), \
            A.ablating(nn.ModuleList([_Plain()]), torch.ones(3), blocks):
        pass


@pytest.mark.parametrize("bad", [torch.zeros(3), torch.tensor([float("nan"), 0.0, 0.0])])
def test_ablating_with_a_direction_that_points_nowhere_fails_loudly(bad):
    with pytest.raises(A.AttributionError, match="points nowhere"), \
            A.ablating(nn.ModuleList([_Plain()]), bad, [0]):
        pass


def test_a_block_whose_input_cannot_be_found_fails_loudly():
    class Hidden(nn.Module):
        def forward(self, other):
            return other + 1.0
    layers = nn.ModuleList([Hidden()])
    with A.ablating(layers, torch.ones(3), [0]), \
            pytest.raises(A.AttributionError, match="could not find the hidden states"):
        layers[0](other=torch.zeros(1, 1, 3))


def test_a_block_returning_something_unknown_fails_loudly():
    class Odd(nn.Module):
        def forward(self, hidden_states):
            return "text"
    layers = nn.ModuleList([Odd()])
    with A.ablating(layers, torch.ones(3), [0]), \
            pytest.raises(A.AttributionError, match="has not been met"):
        layers[0](torch.zeros(1, 1, 3))


def test_a_block_returning_an_empty_tuple_fails_loudly():
    with pytest.raises(A.AttributionError):
        A.unwrap(())


# ---------------------------------------------------------------------------- the statistics

def test_auc_matches_the_naive_pairwise_definition_including_ties():
    pos, neg = [3, 2, 2, 5], [1, 2, 4]
    naive = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert A.auc(pos, neg) == pytest.approx(naive)


def test_auc_agrees_with_the_compass_implementation():
    from senbonzakura import margin
    pos, neg = [0.3, 0.9, 0.9, 1.5], [0.1, 0.9, 1.0]
    assert A.auc(pos, neg) == pytest.approx(margin.auc(pos, neg))


def test_auc_refuses_an_empty_side():
    with pytest.raises(A.AttributionError, match="at least one value"):
        A.auc([], [1.0])


def test_effective_blocks_is_one_for_a_single_block_and_n_for_an_even_spread():
    assert A.effective_blocks(torch.tensor([0.0, 4.0, 0.0])) == pytest.approx(1.0)
    assert A.effective_blocks(torch.ones(8)) == pytest.approx(8.0)
    assert A.effective_blocks(torch.tensor([-1.0, -2.0])) == 2.0


def test_arm_refuses_prompts_without_matching_group_labels():
    with pytest.raises(A.AttributionError, match="needs the request"):
        A.Arm((1, 2), (1,))


def test_resampling_whole_requests_widens_the_interval_compared_with_resampling_rows():
    # Seven variants per request is the shipped track's shape. The same drops resampled as rows
    # would claim seven times the sample and an interval that is too narrow.
    gen = torch.Generator().manual_seed(0)
    drops = torch.randn(30, generator=gen).repeat_interleave(7).unsqueeze(0).double()
    zeros = torch.zeros_like(drops)
    grouped = tuple(i // 7 for i in range(210))
    _, by_request = A.bootstrap_effects(drops, zeros, grouped, grouped, 400,
                                         torch.Generator().manual_seed(1))
    _, by_row = A.bootstrap_effects(drops, zeros, tuple(range(210)), tuple(range(210)), 400,
                                     torch.Generator().manual_seed(1))
    assert float(by_request.std()) > 1.8 * float(by_row.std())


# ------------------------------------------------------------------------ boundary failures

def test_a_partition_with_no_prompts_is_refused_with_its_name():
    parts = fx.partitions("one_block")
    parts["search"] = {"harmful": A.arm([]), "harmless": parts["search"]["harmless"]}
    with pytest.raises(A.AttributionError, match="search partition has no harmful"):
        A.attribute(fx.ToySubject("one_block"), parts, conv=FAST)


def test_a_missing_partition_is_refused():
    parts = fx.partitions("one_block")
    del parts["fit"]
    with pytest.raises(A.AttributionError, match="fit partition is missing"):
        A.attribute(fx.ToySubject("one_block"), parts, conv=FAST)


def test_identical_class_means_leave_no_direction_to_extract():
    same = torch.ones(3, 5, 4)
    with pytest.raises(A.AttributionError, match="identical mean residuals"):
        A.fit_direction(same, same, same, same)


def test_fit_and_search_residuals_from_different_models_are_refused():
    with pytest.raises(A.AttributionError, match="not taken from the same model"):
        A.fit_direction(torch.ones(3, 2, 4), torch.zeros(3, 2, 4),
                        torch.ones(4, 2, 4), torch.zeros(4, 2, 4))


def test_the_direction_is_never_chosen_at_the_embedding_position():
    fit_h, fit_l = torch.zeros(3, 6, 2), torch.zeros(3, 6, 2)
    fit_h[0, :, 0] = 100.0           # a huge contrast, but at position 0
    fit_h[2, :, 1] = 1.0
    search_h, search_l = fit_h.clone(), fit_l.clone()
    _, position, _ = A.fit_direction(fit_h, fit_l, search_h, search_l)
    assert position == 2


def test_fit_direction_with_only_the_embedding_live_falls_back_to_it():
    fit_h, fit_l = torch.zeros(2, 4, 2), torch.zeros(2, 4, 2)
    fit_h[0, :, 0] = 1.0
    _, position, _ = A.fit_direction(fit_h, fit_l, fit_h.clone(), fit_l.clone())
    assert position == 0


def test_a_readout_returning_the_wrong_shape_or_non_finite_values_stops_the_run():
    class Broken(fx.ToySubject):
        def readout(self, prompts, direction=None, blocks=()):
            return torch.full((len(prompts),), float("nan"))
    with pytest.raises(A.AttributionError, match="non-finite"):
        A.attribute(Broken("one_block"), fx.partitions("one_block"), conv=FAST)


def test_a_status_the_instrument_cannot_return_is_refused():
    with pytest.raises(A.AttributionError, match="not a status"):
        A._finish("probably", "", A.DEFAULT, 3, 0)


def test_the_artefact_states_what_it_does_not_claim(runs):
    limits = " ".join(runs["one_block"][2]["limits"])
    assert "Whole decoder blocks only" in limits
    assert "direction_inert" in limits
    assert "3B parameters" in limits


def test_a_localised_result_with_a_failing_predictive_check_reports_it_as_a_finding():
    result = A.attribute(fx.ToySubject("one_block"), fx.partitions("one_block"), seed=0, conv=FAST,
                         predictive=lambda *_: {"ran": True, "supported": False, "why": "x"})
    assert result["status"] == A.LOCALISED
    assert result["predictive_check"]["supported"] is False
    assert result["tint_allowed"] is False


# ------------------------------------------------------------------------------- the early stop

FULL = dataclasses.replace(FAST, null_early_stop=False)


@pytest.fixture(scope="module")
def stopped_and_full():
    out = {}
    for name in fx.EXPECTED:
        parts = fx.partitions(name)
        out[name] = (A.attribute(fx.ToySubject(name), parts, seed=0, conv=FAST),
                     A.attribute(fx.ToySubject(name), fx.partitions(name), seed=0, conv=FULL))
    return out


@pytest.mark.parametrize("name", list(fx.EXPECTED))
def test_stopping_the_null_early_gives_the_full_runs_answer_on_every_plant(stopped_and_full, name):
    early, full = stopped_and_full[name]
    assert early["status"] == full["status"]
    assert early["status_reason"] == full["status_reason"]
    assert [b["effect"] for b in early.get("blocks", [])] == [b["effect"] for b in full.get("blocks", [])]
    assert [b["exceeds_null"] for b in early.get("blocks", [])] == [b["exceeds_null"] for b in full.get("blocks", [])]
    if "controls" in full and "total_threshold" in full["controls"]:
        assert early["controls"]["total_threshold"] == full["controls"]["total_threshold"]
        if not early["controls"]["family_wise_threshold_is_lower_bound"]:
            assert early["controls"]["family_wise_threshold"] == full["controls"]["family_wise_threshold"]
        else:
            assert early["controls"]["family_wise_threshold"] <= full["controls"]["family_wise_threshold"]


def test_the_early_stop_actually_stops_on_a_plant_with_no_strong_block(stopped_and_full):
    # Without this the equivalence above could pass because nothing ever stopped.
    stopped = [n for n, (early, _) in stopped_and_full.items()
               if (early.get("controls") or {}).get("family_wise_threshold_is_lower_bound")]
    assert stopped, "no plant stopped early, so the early stop was never exercised"
    for name in stopped:
        early = stopped_and_full[name][0]
        assert early["controls"]["null_replicates_measured_in_full"] < (
            FAST.shuffled_label_replicates + FAST.random_direction_replicates)


def test_a_localised_plant_measures_every_replicate_in_full(stopped_and_full):
    early, full = stopped_and_full["one_block"]
    assert early["status"] == "localised"
    assert not early["controls"]["family_wise_threshold_is_lower_bound"]
    assert early["controls"]["null_replicates_measured_in_full"] == (
        FAST.shuffled_label_replicates + FAST.random_direction_replicates)
    assert early["controls"]["family_wise_threshold"] == full["controls"]["family_wise_threshold"]


def test_the_floor_is_the_full_quantile_once_every_replicate_is_measured():
    maxima = [0.3, 0.9, 0.1, 1.2, 0.5, 0.7]
    assert A.threshold_floor(maxima, 6, 0.95) == float(torch.quantile(torch.tensor(maxima, dtype=torch.float64), 0.95))


def test_the_floor_cannot_stop_a_run_before_enough_replicates_are_in():
    assert A.threshold_floor([5.0, 5.0], 40, 0.95) < -1e200


@pytest.mark.parametrize("seed", range(20))
def test_the_floor_never_exceeds_the_threshold_the_full_run_reaches(seed):
    gen = torch.Generator().manual_seed(seed)
    maxima = (torch.randn(40, generator=gen, dtype=torch.float64) * 0.4 + 0.5).tolist()
    final = A.threshold_floor(maxima, 40, 0.95)
    for r in range(1, 41):
        assert A.threshold_floor(maxima[:r], 40, 0.95) <= final


def test_switching_the_early_stop_off_measures_every_replicate_in_full(stopped_and_full):
    for _, full in stopped_and_full.values():
        controls = full.get("controls") or {}
        if "null_replicates_measured_in_full" in controls:
            assert not controls["family_wise_threshold_is_lower_bound"]
            assert controls["null_replicates_measured_in_full"] == (
                FULL.shuffled_label_replicates + FULL.random_direction_replicates)


# ------------------------------------------------------------------------------- the raw drops

def test_keeping_the_raw_drops_changes_nothing_the_result_says():
    raw = {}
    kept = A.attribute(fx.ToySubject("spread"), fx.partitions("spread"), seed=0, conv=FULL, raw=raw)
    plain = A.attribute(fx.ToySubject("spread"), fx.partitions("spread"), seed=0, conv=FULL)
    assert kept == plain


def test_the_raw_drops_hold_every_condition_and_every_replicate_with_the_full_null():
    raw = {}
    A.attribute(fx.ToySubject("spread"), fx.partitions("spread"), seed=0, conv=FULL, raw=raw)
    n_h = len(fx.partitions("spread")["measure"]["harmful"].prompts)
    assert raw["drop_h"].shape == (len(raw["conditions"]), n_h)
    assert len(raw["groups_h"]) == n_h
    assert raw["base_h"].shape == (n_h,)
    assert len(raw["replicates"]) == FULL.shuffled_label_replicates + FULL.random_direction_replicates
    nb = fx.ToySubject("spread").n_blocks
    dh, dl = raw["replicates"][0]
    assert dh.shape == (nb + 1, n_h) and dl.shape[0] == nb + 1
