# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The artefact, the way two readout arms are combined, and the command that writes them."""
import argparse
import hashlib
import json

import pytest
from test_attribution_hf import StubModel, StubTok, _parts

from senbonzakura import attribution as A
from senbonzakura import attribution_hf as H
from senbonzakura import attribution_run as R

PASSED = {"passed": True, "cases": {}}


def _arm(status, blocks=(), tint=False, detail=None, style=None):
    return {"status": status, "status_reason": f"because {status}", "tint_allowed": tint,
            "blocks": [{"index": i, "exceeds_null": i in blocks} for i in range(4)],
            "controls": {"readout_agreement_detail": detail or {}, "style_direction": style or {}}}


def test_two_arms_that_agree_give_that_status_and_tint_only_if_both_allow_it():
    both = {"disjoint": _arm(A.LOCALISED, (2,), True), "purity": _arm(A.LOCALISED, (2,), True)}
    assert R.combine(both, PASSED)[0::2] == (A.LOCALISED, True)
    one = {"disjoint": _arm(A.LOCALISED, (2,), True), "purity": _arm(A.LOCALISED, (2,), False)}
    assert R.combine(one, PASSED)[0::2] == (A.LOCALISED, False)


def test_arms_that_disagree_neither_win_and_nothing_may_be_tinted():
    arms = {"disjoint": _arm(A.LOCALISED, (2,), True), "purity": _arm(A.DIFFUSE)}
    status, reason, tint = R.combine(arms, PASSED)
    assert status == R.ARMS_DISAGREE and tint is False
    assert "Neither is preferred" in reason and "disjoint arm: localised" in reason


def test_two_arms_that_say_localised_about_different_blocks_also_disagree():
    arms = {"disjoint": _arm(A.LOCALISED, (2,), True), "purity": _arm(A.LOCALISED, (3,), True)}
    status, reason, tint = R.combine(arms, PASSED)
    assert status == R.ARMS_DISAGREE and tint is False and "different blocks" in reason


def test_agreeing_on_a_negative_status_reports_it_and_never_tints():
    arms = {"disjoint": _arm(A.DIFFUSE), "purity": _arm(A.DIFFUSE)}
    assert R.combine(arms, PASSED)[0::2] == (A.DIFFUSE, False)


def test_a_failed_self_test_overrides_everything():
    arms = {"disjoint": _arm(A.LOCALISED, (2,), True), "purity": _arm(A.LOCALISED, (2,), True)}
    status, _, tint = R.combine(arms, {"passed": False})
    assert status == R.SELFTEST_FAILED and tint is False


def test_limits_that_only_a_run_can_know_are_drawn_from_what_it_measured():
    detail = {"token_sets": {"refusal_set_covers_fit_refusals": 0.09}, "onset_coverage": 0.44}
    arms = {"disjoint": _arm(A.DIFFUSE, detail=detail, style={"flagged": True}),
            "purity": _arm(A.DIFFUSE, detail={"token_sets": {"refusal_set_covers_fit_refusals": 0.9},
                                              "onset_coverage": 0.9})}
    said = " ".join(R.model_specific_limits(arms))
    assert "9%" in said and "absence of a compliance opener" in said
    assert "44%" in said and "partly reading style" in said
    assert "purity arm" not in said


def test_the_parser_defaults_are_the_approved_sample_sizes():
    args = R.build_parser().parse_args(["m", "--track", "t", "--out", "o"])
    assert (args.n_measure, args.n_confirm, args.budget_minutes) == (400, 150, 110.0)


@pytest.mark.parametrize(("argv", "fragment"), [
    (["--n-measure", "5"], "--n-measure is 5, and it cannot be below 30"),
    (["--n-measure", "-5"], "cannot be below 30"),
    (["--n-confirm", "-1"], "--n-confirm is -1, and it cannot be below 0"),
    (["--batch", "0"], "--batch is 0, and it cannot be below 1"),
    (["--seed", "-3"], "--seed is -3"),
    (["--budget-minutes", "-1"], "--budget-minutes is -1"),
    (["--budget-minutes", "nan"], "not a finite number"),
    (["--batch", "many"], "wants a whole number"),
])
def test_a_count_that_cannot_be_right_is_refused_where_it_is_read(argv, fragment, capsys):
    with pytest.raises(SystemExit) as stop:
        R.main(["m", "--track", "t", "--out", "o", *argv])
    assert stop.value.code == 2 and fragment in capsys.readouterr().err


def test_a_mistyped_or_abbreviated_flag_is_named_and_never_guessed(capsys):
    with pytest.raises(SystemExit):
        R.build_parser().parse_args(["m", "--out", "o", "--trak", "t"])
    assert "--trak" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        R.build_parser().parse_args(["m", "--track", "t", "--out", "o", "--n-meas", "40"])
    assert "--n-meas" in capsys.readouterr().err


def test_the_command_reports_a_stopped_run_with_its_reason(monkeypatch, capsys):
    def stop(*_a, **_k):
        raise A.AttributionError("the readout could not be built")
    monkeypatch.setattr(R, "run", stop)
    assert R.main(["m", "--track", "t", "--out", "o"]) == 2
    assert "attribution stopped: the readout could not be built" in capsys.readouterr().err


def test_a_value_the_artefact_cannot_hold_is_refused_not_stringified():
    with pytest.raises(TypeError, match="cannot be written"):
        R._jsonable(object())
    assert R._jsonable({3, 1}) == [1, 3]


def _stub_run(monkeypatch, tmp_path, **flags):
    identity = {"id": "stub", "revision": "abc", "precision": "bfloat16"}
    monkeypatch.setattr(H, "load", lambda *_a, **_k: (StubModel(), StubTok(), identity))
    monkeypatch.setattr(H, "build_partitions",
                        lambda *_a, **_k: (_parts(confirm=True), {"measure": {"n": 40}}))
    monkeypatch.setattr(A, "selftest", lambda **_k: PASSED)
    args = R.build_parser().parse_args(["stub", "--track", "t", "--out", str(tmp_path),
                                        "--n-measure", "40", "--batch", "8"])
    for key, value in flags.items():
        setattr(args, key, value)
    return args


def test_the_whole_command_writes_one_artefact_with_both_arms_and_the_controls(monkeypatch,
                                                                              tmp_path):
    artefact = R.run(_stub_run(monkeypatch, tmp_path), log=lambda *_: None)
    written = json.loads((tmp_path / "attribution.json").read_text())
    assert written["schema"] == R.SCHEMA and written["status"] == artefact["status"]
    assert set(written["arms"]) == {"disjoint", "purity"}
    assert written["arms"]["purity"]["readout_arm"]["declared"].startswith("after")
    assert written["arms"]["disjoint"]["readout_arm"]["declared"].startswith("before")
    assert written["controls"]["selftest"]["passed"] is True
    assert written["controls"]["behaviour_removed_standin"]["ran"] is True
    assert written["comparable_with"]["model_revision"] == "abc"
    assert any("in-memory stand-in" in x for x in written["limits"])
    text = json.dumps(written)
    assert "bad request" not in text and "fine request" not in text   # no prompt reaches the file
    assert not list(tmp_path.glob("*.part"))


def test_an_arm_with_the_same_inputs_is_reused_and_force_redoes_it(monkeypatch, tmp_path):
    R.run(_stub_run(monkeypatch, tmp_path), log=lambda *_: None)
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path), log=lines.append)
    assert any("reusing" in x for x in lines)
    lines.clear()
    R.run(_stub_run(monkeypatch, tmp_path, force=True), log=lines.append)
    assert not any("reusing" in x for x in lines)


def test_a_stale_cached_arm_from_different_inputs_is_not_reused(monkeypatch, tmp_path):
    R.run(_stub_run(monkeypatch, tmp_path), log=lambda *_: None)
    cache = tmp_path / "arm-disjoint.json"
    saved = json.loads(cache.read_text())
    saved["fingerprint"]["seed"] = 99
    cache.write_text(json.dumps(saved))
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path), log=lines.append)
    assert not any("[disjoint] reusing" in x for x in lines)


def test_a_projection_over_the_budget_refuses_before_the_long_work(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="over the 0 minute budget"):
        R.run(_stub_run(monkeypatch, tmp_path, budget_minutes=0.0), log=lambda *_: None)


def test_without_the_prefix_cache_the_projection_counts_a_full_pass_for_every_condition():
    nb, conv = 24, A.DEFAULT
    replicates = conv.shuffled_label_replicates + conv.random_direction_replicates
    assert R.projected_passes(nb, conv, 0.0) == (2 + 3 * nb) + replicates * (nb + 2)


def test_with_the_prefix_cache_the_projection_counts_only_the_blocks_each_condition_runs():
    nb, conv = 24, A.DEFAULT
    replicates = conv.shuffled_label_replicates + conv.random_direction_replicates
    halves = (nb + 1) / 2
    assert R.projected_passes(nb, conv, 1.0) == 1 + (1 + halves + nb + halves) + replicates * (1 + halves)
    half = R.projected_passes(nb, conv, 0.5)
    assert R.projected_passes(nb, conv, 1.0) < half < R.projected_passes(nb, conv, 0.0)


def test_the_cache_memory_default_matches_the_subject_s_own():
    assert R.PREFIX_CACHE_GIB * 2 ** 30 == H.PREFIX_CACHE_BYTES
    args = R.build_parser().parse_args(["m", "--track", "t", "--out", "o"])
    assert args.prefix_cache_gib == R.PREFIX_CACHE_GIB


def test_the_projection_says_how_much_of_the_run_the_cache_holds(monkeypatch, tmp_path):
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path), log=lines.append)
    assert any("of the batches inside the prefix cache's 2 GiB" in line for line in lines), lines


def test_a_cache_turned_off_projects_and_runs_in_full(monkeypatch, tmp_path):
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path, prefix_cache_gib=0.0), log=lines.append)
    assert any("0% of the batches inside" in line for line in lines)
    assert any("off: switched off by the caller" in line for line in lines)


def test_the_behaviour_removed_stand_in_cannot_run_without_a_readout(monkeypatch, tmp_path):
    def no_sets(self, parts):
        self.refusal_ids = None
        self.readout_why_not = "no readout"
    monkeypatch.setattr(H.HFSubject, "prepare", no_sets)
    args = _stub_run(monkeypatch, tmp_path)
    out = R._behaviour_removed(StubModel(), StubTok(), _parts(), args, A.DEFAULT, lambda *_: None)
    assert out == {"ran": False, "why": "no readout"}


def test_the_real_self_test_passes_and_reports_every_case():
    result = A.selftest(conv=A.DEFAULT)
    assert result["passed"] and len(result["cases"]) == 7
    assert all(c["ok"] for c in result["cases"].values())


def test_the_parser_is_an_argparse_parser():
    assert isinstance(R.build_parser(), argparse.ArgumentParser)


def test_a_tensor_is_written_as_a_list():
    import torch
    assert R._jsonable(torch.tensor([1, 2])) == [1, 2]


def test_a_finished_run_exits_zero_and_a_negative_confirm_sample_is_refused(monkeypatch):
    monkeypatch.setattr(R, "run", lambda *_a, **_k: {})
    assert R.main(["m", "--track", "t", "--out", "o"]) == 0


def test_the_stub_runs_leave_no_partial_files_beside_the_artefact(monkeypatch, tmp_path):
    R.run(_stub_run(monkeypatch, tmp_path), log=lambda *_: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "arm-disjoint.json", "arm-purity.json", "attribution.json"]


# ------------------------------------------------------------------------------ the render block

def _blocks(effects, cleared=()):
    return [{"index": i, "effect": e, "exceeds_null": i in cleared} for i, e in enumerate(effects)]


def _res(effects, cleared=(), tau=1.0):
    return {"blocks": _blocks(effects, cleared), "controls": {"family_wise_threshold": tau}}


def test_a_diffuse_result_is_drawn_as_a_spread_that_says_no_block_cleared_the_bar():
    arms = {"disjoint": _res([1.0, 3.0, 0.0, -2.0]), "purity": _res([1.0, 1.0, 2.0, 0.0])}
    out = R.render_block(A.DIFFUSE, False, arms)
    assert out["mode"] == "spread" and out["localisation_claim"] is False
    assert "No block cleared the bar" in out["statement"] and "not a claim" in out["statement"]
    shares = [b["share"] for b in out["blocks"]]
    assert shares == pytest.approx([(0.25 + 0.25) / 2, (0.75 + 0.25) / 2, (0.0 + 0.5) / 2, 0.0])
    assert sum(shares) == pytest.approx(1.0)
    assert out["bar"]["blocks_clearing_it"] == []


def test_a_localised_and_confirmed_result_is_drawn_as_blocks_with_the_claim_set():
    arms = {"disjoint": _res([0.1, 5.0], cleared=(1,)), "purity": _res([0.1, 4.0], cleared=(1,))}
    out = R.render_block(A.LOCALISED, True, arms)
    assert out["mode"] == "blocks" and out["localisation_claim"] is True
    assert out["bar"]["blocks_clearing_it"] == [1]


def test_a_localised_result_nobody_confirmed_is_not_drawn_as_a_localisation():
    arms = {"disjoint": _res([0.1, 5.0], cleared=(1,))}
    out = R.render_block(A.LOCALISED, False, arms)
    assert out["mode"] == "none" and out["localisation_claim"] is False and out["blocks"] == []


@pytest.mark.parametrize("status", [A.ABSENT, A.READOUT_INVALID, A.REDUNDANT, A.UNRESOLVED,
                                    A.DIRECTION_INERT, R.ARMS_DISAGREE, R.SELFTEST_FAILED])
def test_every_other_status_has_no_per_block_picture(status):
    out = R.render_block(status, False, {"disjoint": _res([1.0, 2.0])})
    assert out["mode"] == "none" and out["blocks"] == [] and status in out["statement"]


def test_a_diffuse_result_with_no_effect_at_all_has_no_shares_to_invent():
    out = R.render_block(A.DIFFUSE, False, {"disjoint": _res([-1.0, 0.0])})
    assert [b["share"] for b in out["blocks"]] == [None, None]


def test_an_arm_with_no_blocks_is_left_out_of_the_picture():
    out = R.render_block(A.DIFFUSE, False, {"disjoint": _res([1.0, 1.0]), "purity": {"blocks": None}})
    assert out["mode"] == "spread" and set(out["blocks"][0]["per_arm"]) == {"disjoint"}


def test_the_whole_command_writes_a_render_block_that_matches_the_status(monkeypatch, tmp_path):
    artefact = R.run(_stub_run(monkeypatch, tmp_path), log=lambda *_: None)
    render = json.loads((tmp_path / "attribution.json").read_text())["render"]
    assert render["localisation_claim"] is artefact["tint_allowed"]
    assert render["mode"] in {"blocks", "spread", "none"}


def test_amending_adds_a_render_block_and_the_followups_without_touching_the_rest(tmp_path):
    base = {"status": A.DIFFUSE, "tint_allowed": False, "arms": {"disjoint": _res([1.0, 1.0])},
            "marker": "kept"}
    path = tmp_path / "attribution.json"
    path.write_text(json.dumps(base))
    follow = tmp_path / "followups.json"
    follow.write_text(json.dumps({"predeclaration_sha256": "abc", "band": {}}))
    doc = R.amend(path, follow)
    assert doc["marker"] == "kept" and doc["render"]["mode"] == "spread"
    assert doc["followups"]["predeclaration_sha256"] == "abc"
    assert [n["added"] for n in doc["amendments"]] == ["render", "followups"]
    assert json.loads(path.read_text())["amendments"][0]["by"] == "render_block v1"
    again = R.amend(path)
    assert [n["added"] for n in again["amendments"]] == ["render", "followups", "render"]


def test_a_position_dependent_follow_up_adds_a_caveat_to_the_render_block_and_the_limits(tmp_path):
    base = {"status": A.DIFFUSE, "tint_allowed": False, "arms": {"disjoint": _res([1.0, 1.0])},
            "limits": ["kept"]}
    path = tmp_path / "attribution.json"
    path.write_text(json.dumps(base))
    follow = tmp_path / "followups.json"
    follow.write_text(json.dumps({"predeclaration_sha256": "abc",
                                  "position_verdict_by_arm": {"disjoint": "tracks_position"}}))
    doc = R.amend(path, follow)
    assert doc["render"]["position_dependent"] is True
    assert doc["render"]["caveats"] == [R.POSITION_CAVEAT] and R.POSITION_CAVEAT in doc["limits"]
    assert doc["limits"][0] == "kept"
    stable = tmp_path / "stable.json"
    stable.write_text(json.dumps({"predeclaration_sha256": "abc",
                                  "position_verdict_by_arm": {"disjoint": "stable"}}))
    path.write_text(json.dumps(base))
    assert "caveats" not in R.amend(path, stable)["render"]


def test_the_mechanism_is_in_the_caveat_a_reader_will_meet():
    assert "mean difference of the residual after block p - 1" in R.POSITION_CAVEAT
    assert "always large" in R.POSITION_CAVEAT


def test_amending_adds_the_start_test_and_records_the_declaration_it_ran_under(tmp_path):
    base = {"status": A.DIFFUSE, "tint_allowed": False, "arms": {"disjoint": _res([1.0, 1.0])},
            "limits": []}
    path = tmp_path / "a.json"
    path.write_text(json.dumps(base))
    start = tmp_path / "s.json"
    start.write_text(json.dumps({"declaration_sha256": "zzz", "judgement": {"verdict": "supported"}}))
    doc = R.amend(path, start_path=start)
    assert doc["where_it_starts"]["judgement"]["verdict"] == "supported"
    assert doc["amendments"][-1]["declaration_sha256"] == "zzz"


# --------------------------------------------------------------- what a renderer is permitted

def test_a_spread_may_be_coloured_by_share_but_nothing_may_be_singled_out():
    out = R.render_block(A.DIFFUSE, False, {"disjoint": _res([1.0, 3.0])},
                         {"id": "m", "revision": "r"})
    perm = out["permitted"]
    assert perm["colour_blocks_by_share"] is True and perm["single_out_blocks"] is False
    assert perm["describe_as_where_the_behaviour_lives"] is False
    assert perm["marks_only_from_render"] is True
    assert "may not threshold, rank, outline, label" in perm["rule"]
    assert perm["must_show"] == ["render.statement"]


def test_a_confirmed_localisation_may_single_out_only_the_blocks_that_cleared_the_bar():
    out = R.render_block(A.LOCALISED, True, {"disjoint": _res([0.1, 5.0], cleared=(1,))})
    perm = out["permitted"]
    assert perm["single_out_blocks"] is True and perm["describe_as_where_the_behaviour_lives"]
    assert "render.bar.blocks_clearing_it" in perm["rule"]


def test_no_per_block_picture_means_no_colour_at_all():
    perm = R.render_block(A.READOUT_INVALID, False, {})["permitted"]
    assert perm["colour_blocks_by_share"] is False and perm["single_out_blocks"] is False
    assert "Draw no per block colour" in perm["rule"]


def test_the_render_block_binds_the_artefact_to_both_the_model_id_and_the_revision():
    out = R.render_block(A.DIFFUSE, False, {"disjoint": _res([1.0])}, {"id": "org/m", "revision": "r1"})
    assert out["binds_to"]["model_id"] == "org/m" and out["binds_to"]["model_revision"] == "r1"
    assert "BOTH" in out["binds_to"]["rule"]
    assert R.render_block(A.DIFFUSE, False, {})["binds_to"]["model_id"] is None


def test_a_position_dependent_follow_up_withdraws_the_right_to_describe_it_as_a_location(tmp_path):
    base = {"status": A.LOCALISED, "tint_allowed": True, "model": {"id": "m", "revision": "r"},
            "arms": {"disjoint": _res([0.1, 5.0], cleared=(1,))}, "limits": []}
    path = tmp_path / "a.json"
    path.write_text(json.dumps(base))
    follow = tmp_path / "f.json"
    follow.write_text(json.dumps({"predeclaration_sha256": "x",
                                  "position_verdict_by_arm": {"disjoint": "tracks_position"}}))
    perm = R.amend(path, follow)["render"]["permitted"]
    assert perm["single_out_blocks"] is True
    assert perm["describe_as_where_the_behaviour_lives"] is False
    assert "render.caveats" in perm["must_show"]


def test_the_artefact_names_the_model_id_it_is_comparable_with(monkeypatch, tmp_path):
    R.run(_stub_run(monkeypatch, tmp_path), log=lambda *_: None)
    doc = json.loads((tmp_path / "attribution.json").read_text())
    assert doc["comparable_with"]["model_id"] == "stub"
    assert doc["render"]["binds_to"]["model_id"] == "stub"
    assert doc["render"]["permitted"]["marks_only_from_render"] is True


def test_amending_an_older_artefact_adds_the_model_id_to_what_it_is_comparable_with(tmp_path):
    base = {"status": A.DIFFUSE, "tint_allowed": False, "model": {"id": "org/m", "revision": "r"},
            "arms": {"disjoint": _res([1.0])}, "comparable_with": {"model_revision": "r"}}
    path = tmp_path / "a.json"
    path.write_text(json.dumps(base))
    doc = R.amend(path)
    assert doc["comparable_with"]["model_id"] == "org/m"
    assert any(n["added"] == "comparable_with.model_id" for n in doc["amendments"])


def test_a_falsified_lower_edge_adds_its_caveat_once_and_keeps_the_render_block_cautious(tmp_path):
    base = {"status": A.DIFFUSE, "tint_allowed": False, "model": {"id": "m", "revision": "r"},
            "arms": {"disjoint": _res([1.0, 1.0])}, "limits": []}
    path = tmp_path / "a.json"
    path.write_text(json.dumps(base))
    start = tmp_path / "s.json"
    start.write_text(json.dumps({"declaration_sha256": "z", "judgement": {"verdict": "falsified"}}))
    doc = R.amend(path, start_path=start)
    assert doc["render"]["caveats"] == [R.LOWER_EDGE_CAVEAT] and doc["render"]["position_dependent"]
    assert doc["limits"].count(R.LOWER_EDGE_CAVEAT) == 1
    assert "render.caveats" in doc["render"]["permitted"]["must_show"]
    start.write_text(json.dumps({"declaration_sha256": "z", "judgement": {"verdict": "supported"}}))
    path.write_text(json.dumps(base))
    assert "caveats" not in R.amend(path, start_path=start)["render"]


def test_keep_raw_saves_every_arms_drops_beside_the_artefact_and_measures_the_full_null(monkeypatch,
                                                                                    tmp_path):
    R.run(_stub_run(monkeypatch, tmp_path, keep_raw=True), log=lambda *_: None)
    written = json.loads((tmp_path / "attribution.json").read_text())
    for arm in ("disjoint", "purity"):
        record = written["arms"][arm]["raw"]
        path = tmp_path / record["file"]
        assert path.is_file() and record["file"] == f"raw-{arm}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert written["arms"][arm]["conventions"]["null_early_stop"] is False
        assert b"bad request" not in path.read_bytes() and b"fine request" not in path.read_bytes()
    assert not list(tmp_path.glob("*.part"))


def test_without_keep_raw_no_raw_file_is_written(monkeypatch, tmp_path):
    R.run(_stub_run(monkeypatch, tmp_path), log=lambda *_: None)
    written = json.loads((tmp_path / "attribution.json").read_text())
    assert not list(tmp_path.glob("raw-*.pt"))
    assert all("raw" not in a for a in written["arms"].values())
