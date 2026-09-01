# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for tools/report_bands.py, the thing that actually prints a verdict.

Every guard in `matched_refusal_table` was bypassed for weeks because the reports were Python
heredocs inside run specs that recomputed `min(kls)` themselves. These tests are written around
that: the report must never name a winner the harness refused to name.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "report_bands", Path(__file__).resolve().parent.parent / "tools" / "report_bands.py")
rb = importlib.util.module_from_spec(_SPEC)
sys.modules["report_bands"] = rb
_SPEC.loader.exec_module(rb)


def _doc(ranking, targets=None, **over):
    d = {"model": "m", "seed": 42, "code_version": "vX",
         "e4": {"degenerate_reason": None,
                "matched_refusal": {
                    "baseline_is_unablated": True, "baseline_refusal": 0.8, "n_eval": 128,
                    "targets": targets or {"50%_removed": {
                        "1": {"kl": 0.9, "strength": 1.0, "harmful_refusal": 0.40,
                              "overshoot": 0.0},
                        "2": {"kl": 0.1, "strength": 1.0, "harmful_refusal": 0.40,
                              "overshoot": 0.0}}},
                    "ranking": ranking}}}
    d.update(over)
    return d


@pytest.fixture
def arm(tmp_path):
    def write(label, doc):
        p = tmp_path / f"{label}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        return f"{label}={p}"
    return write


# ── the failure this file exists to prevent ───────────────────────────────────────────
def test_a_band_the_harness_refused_to_rank_never_names_a_winner(arm, capsys):
    doc = _doc({"50%_removed": {"comparable": False, "spread": 0.2,
                                "reason": "the arms are not at the same level"}})
    assert rb.main([arm("A", doc)]) == 0
    out = capsys.readouterr().out
    assert "NO RANKING" in out and "not at the same level" in out
    assert "cheapest" not in out, "a refused band named a winner anyway"


def test_a_tie_is_not_reported_as_a_win(arm, capsys):
    doc = _doc({"50%_removed": {"comparable": True, "cheapest": None,
                                "reason": "K=1 and K=2 are within 5% on KL"}})
    assert rb.main([arm("A", doc)]) == 0
    out = capsys.readouterr().out
    assert "NO RANKING" in out and "within 5%" in out
    assert "cheapest K=" not in out


def test_a_real_winner_is_reported_with_its_margin(arm, capsys):
    doc = _doc({"50%_removed": {"comparable": True, "cheapest": 2, "margin": 9.0,
                                "reason": "K=2 is cheapest"}})
    assert rb.main([arm("A", doc)]) == 0
    out = capsys.readouterr().out
    assert "cheapest K=2" in out and "9.0x" in out


def test_the_verdict_is_read_not_recomputed(arm, capsys):
    """The arms plainly favour K=2 on KL; the harness said no. The harness wins."""
    doc = _doc({"50%_removed": {"comparable": False, "reason": "the grid says nothing about K"}})
    assert rb.main([arm("A", doc)]) == 0
    out = capsys.readouterr().out
    assert "cheapest" not in out, "the report recomputed a winner from the raw KL values"


# ── a report that found nothing must not look like one that worked ────────────────────
def test_a_missing_file_makes_the_report_fail(tmp_path, capsys):
    assert rb.main([f"A={tmp_path / 'nope.json'}"]) == 1
    assert "INCOMPLETE" in capsys.readouterr().out


def test_a_file_without_a_grid_makes_the_report_fail(arm):
    assert rb.main([arm("A", {"model": "m"})]) == 1


def test_allow_missing_is_opt_in(tmp_path, capsys):
    assert rb.main([f"A={tmp_path / 'nope.json'}", "--allow-missing"]) == 0
    assert "INCOMPLETE" in capsys.readouterr().out


def test_a_truncated_file_is_reported_not_raised(tmp_path, capsys):
    p = tmp_path / "half.json"
    p.write_text('{"e4": ', encoding="utf-8")
    assert rb.main([f"A={p}"]) == 1
    assert "unreadable" in capsys.readouterr().out


# ── the things that make two arms distinguishable ─────────────────────────────────────
def test_provenance_is_printed_above_the_numbers(arm, capsys):
    doc = _doc({"50%_removed": {"comparable": True, "cheapest": 2, "margin": 2.0, "reason": "x"}},
               directions_meta={"init": "mean-diff", "score": "per-token", "induce": 0.2,
                                "start_layer": 9, "steps": 200})
    assert rb.main([arm("A3", doc)]) == 0
    out = capsys.readouterr().out
    for token in ("init=mean-diff", "score=per-token", "induce=0.2", "code_version=vX", "seed=42"):
        assert token in out, token


def test_a_grid_with_no_anchor_refuses_to_show_bands(arm, capsys):
    doc = _doc({})
    doc["e4"]["matched_refusal"]["baseline_is_unablated"] = False
    assert rb.main([arm("A", doc)]) == 0
    out = capsys.readouterr().out
    assert "NO UNABLATED ANCHOR" in out
    assert "50%_removed" not in out


def test_a_degenerate_grid_shows_its_reason_instead_of_bands(arm, capsys):
    doc = _doc({})
    doc["e4"]["degenerate_reason"] = "every fitted arm landed on floor"
    assert rb.main([arm("A", doc)]) == 0
    out = capsys.readouterr().out
    assert "UNREADABLE GRID" in out and "landed on floor" in out
    assert "50%_removed" not in out


def test_several_arms_are_all_reported(arm, capsys):
    good = _doc({"50%_removed": {"comparable": True, "cheapest": 1, "margin": 2.0, "reason": "x"}})
    assert rb.main([arm("A0", good), arm("A1", good), arm("A2", good)]) == 0
    out = capsys.readouterr().out
    for label in ("A0", "A1", "A2"):
        assert f"--- {label}" in out


def test_a_malformed_argument_is_a_usage_error(capsys):
    assert rb.main(["justapath.json"]) == 2
    assert "expected LABEL=path" in capsys.readouterr().err


def test_a_band_with_no_recorded_verdict_says_so_rather_than_guessing(arm, capsys):
    assert rb.main([arm("A", _doc({}))]) == 0
    assert "no verdict recorded" in capsys.readouterr().out
