# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The declared criteria, judged on profiles whose answer is known by construction."""
import hashlib

import pytest

from senbonzakura import attribution_followups as F


def profile(peak_block, n=24, height=1.0, floor=0.05):
    out = [floor] * n
    out[peak_block] = height
    return out


def test_the_declaration_is_fixed_text_with_a_checksum_that_would_catch_an_edit():
    assert hashlib.sha256(F.PREDECLARATION.encode()).hexdigest() == F.PREDECLARATION_SHA256
    assert "Falsified if" in F.PREDECLARATION and "MIXED" in F.PREDECLARATION
    assert tuple(range(12, 22)) == F.BAND and F.POSITIONS == (15, 18, 24)


def test_a_peak_that_stays_at_the_reference_block_everywhere_is_stable():
    reference = profile(20)
    result = F.judge_positions({15: profile(20), 18: profile(19), 24: profile(21)}, reference)
    assert result["verdict"] == "stable" and result["peaks_staying_put"] == 3


def test_a_peak_that_follows_the_fitting_position_is_flagged_as_tracking_it():
    reference = profile(20)
    result = F.judge_positions({15: profile(14), 18: profile(17), 24: profile(23)}, reference)
    assert result["verdict"] == "tracks_position"
    assert result["peaks_at_fitting_position"] == 3 and result["peak_moves_with_position"]


def test_a_peak_near_the_reference_but_a_band_share_below_the_floor_is_not_stable():
    spread_out = [0.0] * 24
    spread_out[20] = 1.0
    for k in (0, 1, 2, 3, 4, 5, 22, 23):
        spread_out[k] = 0.6
    result = F.judge_positions({15: spread_out, 18: spread_out, 24: spread_out}, profile(20))
    assert result["band_share_ok_everywhere"] is False and result["verdict"] == "mixed"


def test_anything_that_is_neither_is_reported_as_mixed():
    reference = profile(20)
    result = F.judge_positions({15: profile(20), 18: profile(5), 24: profile(10)}, reference)
    assert result["verdict"] == "mixed"


def test_the_two_criteria_cannot_both_hold_for_the_same_peak_at_the_declared_positions():
    for position in F.POSITIONS:
        assert abs((position - 1) - 20) > 2 * F.PEAK_TOLERANCE, position


def test_a_missing_position_cannot_be_judged():
    with pytest.raises(ValueError, match="no profile"):
        F.judge_positions({15: profile(20)}, profile(20))


def test_band_share_counts_only_the_positive_effect_and_handles_an_all_zero_profile():
    effects = [0.0] * 24
    effects[13], effects[2], effects[5] = 2.0, 1.0, -9.0
    assert F.positive_share(effects, F.BAND) == pytest.approx(2.0 / 3.0)
    assert F.positive_share([0.0] * 24, F.BAND) is None


def test_the_peak_prefers_the_earlier_block_on_a_tie():
    assert F.peak([1.0, 3.0, 3.0]) == 1


def test_the_band_verdict_follows_the_lower_end_of_the_interval_and_keeps_a_negative():
    yes = F.judge_band({"ran": True, "contrast": {"ci": [0.1, 0.4]}})
    no = F.judge_band({"ran": True, "contrast": {"ci": [0.0, 0.4]}})
    assert yes["verdict"] == "supported" and "exploratory" in yes["framing"]
    assert no["verdict"] == "not_supported" and "negative result is kept" in no["framing"]
    assert F.judge_band({"ran": False, "why": "too few"})["verdict"] == "not_run"
    assert F.judge_band({"ran": False})["why"]


# ----------------------------------------------------------------------------- the run, on a stub

def _original_artefact(tmp_path, monkeypatch):
    from test_attribution_hf import StubModel, StubTok, _parts

    from senbonzakura import attribution as A
    from senbonzakura import attribution_hf as H
    from senbonzakura import attribution_run as R
    identity = {"id": "stub", "revision": "abc", "precision": "bfloat16"}
    digests = {"measure": {"harmful": {"n": 40}}}
    monkeypatch.setattr(H, "load", lambda *_a, **_k: (StubModel(), StubTok(), identity))
    monkeypatch.setattr(H, "build_partitions", lambda *_a, **_k: (_parts(confirm=True), digests))
    monkeypatch.setattr(A, "selftest", lambda **_k: {"passed": True, "cases": {}})
    args = R.build_parser().parse_args(["stub", "--track", "t", "--out", str(tmp_path),
                                        "--n-measure", "40", "--batch", "8"])
    R.run(args, log=lambda *_: None)
    return tmp_path / "attribution.json", identity, digests


def test_the_follow_ups_run_end_to_end_and_reproduce_the_original_profile(tmp_path, monkeypatch):
    import json

    original, _, _ = _original_artefact(tmp_path / "orig", monkeypatch)
    position = json.loads(original.read_text())["arms"]["disjoint"]["direction"]["position"]
    monkeypatch.setattr(F, "BAND", (1,))
    monkeypatch.setattr(F, "POSITIONS", (1, 2, 3))
    monkeypatch.setattr(F, "REFERENCE_POSITION", position)
    args = F.build_parser().parse_args(["stub", "--track", "t", "--artefact", str(original),
                                        "--out", str(tmp_path / "f"), "--batch", "8"])
    out = F.run(args, log=lambda *_: None)
    written = json.loads((tmp_path / "f" / "followups.json").read_text())
    assert written["predeclaration_sha256"] == F.PREDECLARATION_SHA256 and written["exploratory"]
    assert set(written["arms"]) == {"disjoint", "purity"}
    for arm in written["arms"].values():
        assert arm["reference_reproduces_original_to"] < 1e-9
        assert arm["judgement"]["verdict"] in {"stable", "tracks_position", "mixed"}
        assert set(arm["positions"]) == {str(position), "1", "2", "3"}
    assert written["band"]["judgement"]["verdict"] in {"supported", "not_supported"}
    assert out["position_arms_agree"] in (True, False)


def test_inputs_that_are_not_the_original_runs_are_refused(tmp_path, monkeypatch):
    import json

    from senbonzakura import attribution as A
    from senbonzakura import attribution_hf as H
    original, _, _ = _original_artefact(tmp_path / "orig", monkeypatch)
    monkeypatch.setattr(H, "build_partitions", lambda *_a, **_k: (_parts_unused(), {"measure": {}}))
    args = F.build_parser().parse_args(["stub", "--track", "t", "--artefact", str(original),
                                        "--out", str(tmp_path / "f")])
    with pytest.raises(A.AttributionError, match="not the inputs the original run used"):
        F.run(args, log=lambda *_: None)
    assert json.loads(original.read_text())["status"]


def _parts_unused():
    from test_attribution_hf import _parts
    return _parts(confirm=True)


def test_a_direction_that_does_not_reproduce_stops_the_follow_up(tmp_path, monkeypatch):
    import json

    from senbonzakura import attribution as A
    original, _, _ = _original_artefact(tmp_path / "orig", monkeypatch)
    doc = json.loads(original.read_text())
    doc["arms"]["disjoint"]["direction"]["held_out_auc"] += 0.1
    original.write_text(json.dumps(doc))
    args = F.build_parser().parse_args(["stub", "--track", "t", "--artefact", str(original),
                                        "--out", str(tmp_path / "f")])
    with pytest.raises(A.AttributionError, match="not reproducing"):
        F.run(args, log=lambda *_: None)


def test_main_reports_a_stopped_follow_up_with_its_reason(monkeypatch, capsys):
    from senbonzakura import attribution as A

    def stop(*_a, **_k):
        raise A.AttributionError("nope")
    monkeypatch.setattr(F, "run", stop)
    assert F.main(["m", "--track", "t", "--artefact", "a", "--out", "o"]) == 2
    assert "follow-up stopped: nope" in capsys.readouterr().err
    monkeypatch.setattr(F, "run", lambda *_a, **_k: {})
    assert F.main(["m", "--track", "t", "--artefact", "a", "--out", "o"]) == 0


def test_fitting_at_a_chosen_position_is_refused_when_the_position_does_not_exist():
    import torch

    from senbonzakura import attribution as A
    x = torch.ones(3, 2, 4)
    with pytest.raises(A.AttributionError, match="does not exist"):
        A.fit_direction_at(x, x * 0, x, x * 0, 9)
    with pytest.raises(A.AttributionError, match="identical mean residuals"):
        A.fit_direction_at(x, x, x, x, 1)


# --------------------------------------------------------------------------- where it starts

def _rows(early_hi=0.05, contrast_lo=0.5, override=None):
    base = {p: {"early_drop_ci": [-0.02, early_hi], "contrast_ci": [contrast_lo, 0.9]}
            for p in F.START_POSITIONS}
    base.update(override or {})
    return base


def test_the_start_declaration_states_where_the_idea_came_from_and_is_checksummed():
    import hashlib
    text = F.START_DECLARATION
    assert hashlib.sha256(text.encode()).hexdigest() == F.START_SHA256
    assert "READING" in text and "NOT declared before those tables existed" in text
    assert "FALSIFIED if criterion 1 fails at ANY" in text and "UNINFORMATIVE" in text
    assert F.START_POSITIONS == (15, 18, 21, 24) and tuple(range(10)) == F.EARLY
    assert (F.EARLY_DROP_CEILING, F.CONTRAST_FLOOR) == (0.15, 0.15)


def test_an_early_set_that_does_nothing_with_a_visible_contrast_everywhere_is_supported():
    out = F.judge_start(_rows())
    assert out["verdict"] == "supported" and "exploratory" in out["framing"]


def test_one_position_where_the_early_blocks_matter_falsifies_the_lower_edge():
    out = F.judge_start(_rows(override={15: {"early_drop_ci": [0.1, 0.3], "contrast_ci": [0.5, 0.9]}}))
    assert out["verdict"] == "falsified" and out["positions_where_early_blocks_matter"] == [15]


def test_a_test_that_could_not_see_the_later_set_matter_is_uninformative_not_supported():
    out = F.judge_start(_rows(override={24: {"early_drop_ci": [0.0, 0.02], "contrast_ci": [0.05, 0.4]}}))
    assert out["verdict"] == "uninformative"
    assert out["positions_where_the_test_could_not_see_a_difference"] == [24]


def test_falsified_takes_precedence_over_uninformative():
    rows = _rows(override={15: {"early_drop_ci": [0.1, 0.3], "contrast_ci": [0.0, 0.1]}})
    assert F.judge_start(rows)["verdict"] == "falsified"


def test_the_ceiling_is_inclusive_and_the_floor_is_exclusive():
    assert F.judge_start(_rows(early_hi=0.15))["verdict"] == "supported"
    assert F.judge_start(_rows(early_hi=0.1500001))["verdict"] == "falsified"
    assert F.judge_start(_rows(contrast_lo=0.15))["verdict"] == "uninformative"


def test_a_missing_position_cannot_be_judged_for_the_start_test():
    with pytest.raises(ValueError, match="no result"):
        F.judge_start({15: {"early_drop_ci": [0, 0], "contrast_ci": [1, 1]}})


def _start_args(tmp_path, original):
    return F.build_parser().parse_args(["stub", "--track", "t", "--artefact", str(original),
                                        "--out", str(tmp_path / "s"), "--batch", "8",
                                        "--n-fresh", "20"])


def _patch_start(monkeypatch):
    from test_attribution_hf import _parts

    from senbonzakura import attribution as A
    from senbonzakura import attribution_hf as H
    fresh = A.arm([f"bad fresh {i}" for i in range(20)], [f"fresh{i}" for i in range(20)])
    monkeypatch.setattr(H, "fresh_requests",
                        lambda *_a, **_k: (fresh, {"n": 20, "digest": "d"}))
    monkeypatch.setattr(F, "START_POSITIONS", (1, 2, 3))
    monkeypatch.setattr(F, "EARLY", (0,))
    monkeypatch.setattr(F, "LATE_FROM", 1)
    return _parts


def test_the_start_test_runs_end_to_end_on_a_stub_and_records_what_it_was_declared_as(
        tmp_path, monkeypatch):
    import json
    original, _, _ = _original_artefact(tmp_path / "orig", monkeypatch)
    _patch_start(monkeypatch)
    out = F.run_start(_start_args(tmp_path, original), log=lambda *_: None)
    written = json.loads((tmp_path / "s" / "start.json").read_text())
    assert written["declaration_sha256"] == F.START_SHA256 and written["exploratory"] is True
    assert set(written["positions"]) == {"1", "2", "3"}
    assert out["judgement"]["verdict"] in {"supported", "falsified", "uninformative"}
    assert written["inputs"]["fresh_requests"]["digest"] == "d"
    for row in written["positions"].values():
        assert set(row["harmful_refusal_rate"]) == {"none", "early", "later"}
        assert len(row["early_drop_ci"]) == 2 and len(row["contrast_ci"]) == 2
    assert "bad fresh" not in json.dumps(written)


def test_the_start_test_refuses_inputs_that_are_not_the_original_runs(tmp_path, monkeypatch):
    from senbonzakura import attribution as A
    from senbonzakura import attribution_hf as H
    original, _, _ = _original_artefact(tmp_path / "orig", monkeypatch)
    monkeypatch.setattr(H, "build_partitions",
                        lambda *_a, **_k: (_parts_unused(), {"measure": {}}))
    with pytest.raises(A.AttributionError, match="not the inputs the original run used"):
        F.run_start(_start_args(tmp_path, original), log=lambda *_: None)


def test_main_dispatches_start_to_the_start_runner(monkeypatch):
    seen = []
    monkeypatch.setattr(F, "run_start", lambda args, **_k: seen.append(args.n_fresh))
    monkeypatch.setattr(F, "run", lambda *_a, **_k: seen.append("followups"))
    argv = ["m", "--track", "t", "--artefact", "a", "--out", "o"]
    assert F.main(["start", *argv, "--n-fresh", "7"]) == 0
    assert F.main(argv) == 0
    assert seen == [7, "followups"]


def test_fresh_requests_exclude_what_was_used_and_leave_other_draws_alone(monkeypatch, tmp_path):
    from senbonzakura import attribution as A
    from senbonzakura import attribution_hf as H
    from senbonzakura import track
    rows = [f"Explain how to thing{i}" for i in range(30)]
    monkeypatch.setattr(track, "load_partitions",
                        lambda _p: ({"fit": [], "search": [], "measure": rows}, {}))
    groups = sorted(track.group_by_request(rows))
    used = set(groups[:10])
    arm, info = H.fresh_requests(tmp_path, used, n=8, seed=0)
    assert len(arm.prompts) == 8 and not set(arm.groups) & used
    assert info["unused_requests_available"] == 20
    again, _ = H.fresh_requests(tmp_path, used, n=8, seed=0)
    assert again.groups == arm.groups
    with pytest.raises(A.AttributionError, match="fewer than the 30 asked for"):
        H.fresh_requests(tmp_path, used, n=30, seed=0)


def test_the_follow_up_parser_applies_the_same_bounds_and_names_a_mistyped_flag(capsys):
    argv = ["m", "--track", "t", "--artefact", "a", "--out", "o"]
    for bad, fragment in ((["--batch", "0"], "cannot be below 1"),
                          (["--n-fresh", "0"], "cannot be below 1"),
                          (["--n-measure", "3"], "cannot be below 30"),
                          (["--seed", "-1"], "--seed is -1"),
                          (["--n-confirm", "-2"], "--n-confirm is -2")):
        with pytest.raises(SystemExit):
            F.build_parser().parse_args([*argv, *bad])
        assert fragment in capsys.readouterr().err
    with pytest.raises(SystemExit):
        F.build_parser().parse_args([*argv, "--n-fres", "5"])
    assert "--n-fres" in capsys.readouterr().err
