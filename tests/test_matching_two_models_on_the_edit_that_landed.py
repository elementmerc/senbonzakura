# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Matching two models on measured leak, and the four ways the search could lie about it.

WHY THIS FILE EXISTS

Two models given `--strength 1.0` have not been given the same edit. The same weight edit reached
0.578 of Gemma's residual stream against Qwen3's 0.016, because Gemma norms each sublayer's output
before the residual add, and every Gemma figure this project published before measuring that was
withdrawn. The same trap sits under arXiv 2607.17427, which reads a sign reversal between the two
families as a fact about families.

So the search exists to produce a number that will be published, and the arithmetic is the half
that can be checked without a card. The measurement is injected for exactly that reason.

THE FOUR WAYS IT COULD LIE, each with a test below:

1. Return its last midpoint as though it had converged, when the budget ran out.
2. Bisect a bracket whose ends do not straddle the target, clamping to an end.
3. Bisect on a model where leak does not fall with strength, where the method does not apply.
4. Treat a failed measurement as a leak of zero, which is the strongest possible result.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "research"))
import matched_leak_strength as m


def falling(at_zero=1.0, rate=1.0):
    """A plausible leak curve: falls smoothly with strength, never negative."""
    def measure(s):
        return at_zero * (2.718281828 ** (-rate * s))
    return measure


def test_it_finds_a_strength_that_hits_the_target():
    curve = falling()
    got = m.find_strength(curve, 0.5, low=0.0, high=2.0, log=lambda _x: None)
    assert got["converged"] is True
    assert abs(got["leak"] - 0.5) <= m.DEFAULT_TOLERANCE * 0.5
    assert 0.0 < got["strength"] < 2.0


def test_two_models_with_different_responses_get_different_strengths():
    """THE POINT OF THE WHOLE FILE. A model whose edit lands at a third of the rate needs more
    strength to reach the same place, and a shared `--strength 1.0` would have compared them at
    two different effective edits while reporting one number.
    """
    gentle = m.find_strength(falling(rate=0.3), 0.5, high=8.0, log=lambda _x: None)
    sharp = m.find_strength(falling(rate=3.0), 0.5, high=8.0, log=lambda _x: None)
    assert gentle["strength"] > sharp["strength"] * 2, (
        "the two arms came out at similar strengths, which cannot be right for curves this "
        "different and would mean the search is not following the measurement")


def test_running_out_of_budget_reports_not_converged_rather_than_its_last_guess():
    """LIE 1. A loop that fell out of its budget and returned its midpoint as an answer is how a
    number with no precision behind it reaches a table.
    """
    got = m.find_strength(falling(), 0.5, max_steps=1, tolerance=1e-12, log=lambda _x: None)
    assert got["converged"] is False
    assert got["strength"] is not None, "it still reports where it got to, labelled"


def test_a_target_outside_the_bracket_is_refused_rather_than_clamped():
    """LIE 2. A strength clamped to a bracket end is not a matched strength, and nothing
    downstream would say so.
    """
    with pytest.raises(m.MatchError, match="outside what this bracket reaches"):
        m.find_strength(falling(), 0.001, low=0.0, high=1.0, log=lambda _x: None)


def test_a_target_above_the_unedited_leak_is_refused_too():
    with pytest.raises(m.MatchError, match="outside what this bracket reaches"):
        m.find_strength(falling(at_zero=1.0), 5.0, log=lambda _x: None)


def test_a_model_whose_leak_does_not_fall_is_a_finding_and_not_a_bracket():
    """LIE 3. The method assumes more strength removes more direction. A model that does not
    behave that way is a result about that model, not an input to average over.
    """
    with pytest.raises(m.MatchError, match="did not fall across the bracket"):
        m.find_strength(lambda _s: 1.0, 0.5, log=lambda _x: None)


def test_a_rising_curve_is_refused_with_both_ends_named():
    with pytest.raises(m.MatchError, match="did not fall"):
        m.find_strength(lambda s: 0.1 + s, 0.5, log=lambda _x: None)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_a_failed_measurement_is_never_treated_as_a_leak_of_zero(bad):
    """LIE 4, AND THE WORST. Zero leak is the strongest result the metric can report, so a failed
    measurement read as zero does not look like a failure. It looks like success.
    """
    with pytest.raises(m.MatchError, match="measurement failed"):
        m.find_strength(lambda _s: bad, 0.5, log=lambda _x: None)


def test_a_measurement_that_fails_mid_search_is_not_swallowed():
    calls = []

    def flaky(s):
        calls.append(s)
        return float("nan") if len(calls) > 2 else falling()(s)

    with pytest.raises(m.MatchError, match="measurement failed"):
        m.find_strength(flaky, 0.5, log=lambda _x: None)


def test_an_inverted_bracket_is_refused():
    with pytest.raises(m.MatchError, match="not an interval"):
        m.find_strength(falling(), 0.5, low=2.0, high=1.0, log=lambda _x: None)


def test_the_result_carries_the_bracket_and_the_steps():
    """A caller publishing this number needs the bracket, the step count and whether it
    converged. A bare float invites all three to be dropped on the way to a paper.
    """
    got = m.find_strength(falling(), 0.5, log=lambda _x: None)
    assert got["bracket"] == [0.0, 2.0]
    assert len(got["bracket_leak"]) == 2
    assert got["steps"] and all({"strength", "leak"} <= set(s) for s in got["steps"])


def test_the_budget_is_honoured():
    seen = []
    m.find_strength(lambda s: (seen.append(s), falling()(s))[1], 0.5, max_steps=3,
                    tolerance=1e-12, log=lambda _x: None)
    assert len(seen) <= 3 + 2, "two bracket ends plus the step budget"


# ── the comparison ──────────────────────────────────────────────────────────────────

def _result(strength, leak, converged=True):
    return {"strength": strength, "leak": leak, "converged": converged, "steps": [],
            "target": 0.5, "bracket": [0.0, 2.0], "bracket_leak": [1.0, 0.1]}


def test_the_comparison_states_what_matching_does_and_does_not_control():
    doc = m.compare({"gemma": _result(1.4, 0.5), "qwen": _result(0.3, 0.5)})
    assert doc["comparable"] is True
    assert "edit that landed" in doc["means"]
    assert "training data, size and tokeniser" in doc["means"], (
        "the comparison claims more than it controls if it does not say what it leaves alone")


def test_an_unconverged_arm_makes_the_comparison_incomparable():
    """The failure this prevents: a table of two strengths where one of them never reached the
    target, and a difference between the arms attributed to the models.
    """
    doc = m.compare({"gemma": _result(1.4, 0.5), "qwen": _result(2.0, 0.9, converged=False)})
    assert doc["comparable"] is False
    assert doc["unconverged"] == ["qwen"]
    assert "qwen" in doc["caveat_if_unconverged"]


def test_one_model_is_not_a_matched_comparison():
    with pytest.raises(m.MatchError, match="at least two"):
        m.compare({"only": _result(1.0, 0.5)})


def test_the_spread_is_reported_because_it_is_the_finding():
    """How far apart the two matched strengths are IS the norm-gain effect, measured."""
    doc = m.compare({"gemma": _result(1.4, 0.5), "qwen": _result(0.3, 0.5)})
    assert doc["strength_spread"] == pytest.approx(1.1)


# ── the command line ────────────────────────────────────────────────────────────────

def test_mismatched_model_and_directions_counts_are_refused():
    with pytest.raises(SystemExit, match="One directions file per model"):
        m.main(["--model", "a", "--model", "b", "--directions", "d", "--target", "0.5",
                "--workdir", "w", "--out", "o"])


def test_the_parser_declares_what_the_help_describes():
    dests = {a.dest for a in m.build_parser()._actions}
    for flag in ("model", "directions", "target", "low", "high", "tolerance", "max_steps",
                 "probe_n", "device", "workdir", "out"):
        assert flag in dests, f"--{flag} is described and not declared"


def test_an_unmatched_run_exits_non_zero(tmp_path, monkeypatch):
    """The exit code is the interface for a batch script, and a run whose arms are not matched has
    not produced a comparison.
    """
    monkeypatch.setattr(m, "measure_with_senbonzakura",
                        lambda *a, **k: falling())
    monkeypatch.setattr(m, "find_strength",
                        lambda *a, **k: _result(1.0, 0.9, converged=False))
    out = tmp_path / "cmp.json"
    assert m.main(["--model", "a", "--model", "b", "--directions", "d", "--directions", "e",
                   "--target", "0.5", "--workdir", str(tmp_path), "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["comparable"] is False


def test_a_matched_run_exits_zero_and_writes_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "measure_with_senbonzakura", lambda *a, **k: falling())
    out = tmp_path / "cmp.json"
    assert m.main(["--model", "a", "--model", "b", "--directions", "d", "--directions", "e",
                   "--target", "0.5", "--workdir", str(tmp_path), "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert set(doc["arms"]) == {"a", "b"}
    assert doc["comparable"] is True
