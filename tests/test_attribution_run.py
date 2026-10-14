# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The artefact, the way two readout arms are combined, and the command that writes them."""
import argparse
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


def test_a_sample_below_the_reporting_floor_is_refused_before_anything_loads(capsys):
    with pytest.raises(SystemExit, match="at least 30"):
        R.main(["m", "--track", "t", "--out", "o", "--n-measure", "5"])


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
    with pytest.raises(SystemExit, match="at least 30"):
        R.main(["m", "--track", "t", "--out", "o", "--n-confirm", "-1"])


def test_the_stub_runs_leave_no_partial_files_beside_the_artefact(monkeypatch, tmp_path):
    R.run(_stub_run(monkeypatch, tmp_path), log=lambda *_: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "arm-disjoint.json", "arm-purity.json", "attribution.json"]
