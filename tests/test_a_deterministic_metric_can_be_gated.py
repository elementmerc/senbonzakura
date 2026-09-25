# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`baseline` accepted nothing this repository could produce, and this is the guard on the repair.

THE DEFECT. `from_artefact` demanded an interval from every metric. `coherence` is one
deterministic forward pass over one fixed passage and has no run-to-run spread for an interval to
describe, so the one writer carrying all eight pinned fields was refused on the field it could not
honestly supply, and the other four were refused on the pinned fields. Nothing passed, which is the
exact condition `baseline`'s own header says the module was written to end. A gate that accepts no
input is indistinguishable from a gate that finds nothing wrong, and it had been registered in the
dispatch table and documented in the CLI reference for a fortnight.

WHAT THIS OWNS. Not "the deterministic branch works" alone: the shape of the failure was that
nobody had ever fed the gate a real artefact, so every test here builds one the way a writer builds
it, through `measurement.stamp`, and puts it through `from_artefact` and `verdict` rather than
constructing a baseline dict by hand. A test that hand-writes the artefact is a test that would
have passed while the tool refused everything.
"""
import pytest
from senbonzakura_check import measurement

from senbonzakura import baseline, stamps

MODEL = "qwen3-1.7b"


def _stamped(metric, value, estimator, **fields):
    """One artefact, built the way a measuring command builds it."""
    doc = {"model": MODEL}
    measurement.stamp(doc, metric, value, estimator, **fields)
    return doc


def _pinned_fields(**over):
    out = {"input_digest": "abc123def4567890", "partition": "measure", "prompt_format": "raw",
           "tool_version": "0.4.0", "precision": "bfloat16"}
    out.update(over)
    return out


def _coherence_artefact(value=2.4131):
    return _stamped("coherence", value, "neutral-passage-nll", n=1, deterministic=True,
                    **_pinned_fields(partition=baseline.FIXED_PASSAGE))


def _refusal_artefact(value=0.094, interval=(0.06, 0.13)):
    return _stamped("refusal_rate", value, "senbonzakura-ruler", n=200, by_estimator=True,
                    interval=list(interval), **_pinned_fields())


def test_a_deterministic_artefact_becomes_a_baseline():
    b = baseline.from_artefact(_coherence_artefact(), "coherence", seeds=[0])
    assert b["deterministic"] is True
    assert b["interval"] is None
    assert b["point"] == pytest.approx(2.4131)
    # The pinned fields survive the adapter: without them the gate refuses to compare, which is
    # how this metric was excluded from the gate entirely before the partition was added.
    assert b["partition"] == baseline.FIXED_PASSAGE
    assert b["precision"] == "bfloat16"


def test_an_interval_artefact_still_becomes_a_baseline():
    b = baseline.from_artefact(_refusal_artefact(), "refusal_rate.senbonzakura-ruler", seeds=[0])
    assert b["deterministic"] is False
    assert b["interval"] == [0.06, 0.13]


def test_an_unchanged_deterministic_reading_passes():
    b = baseline.from_artefact(_coherence_artefact(), "coherence", seeds=[0])
    ok, headline, detail = baseline.verdict(b, 2.4131, None)
    assert ok
    assert "unchanged" in headline
    # The detail says WHY an exact match is the pass condition, because a reader meeting "unchanged"
    # on a metric with no interval will otherwise assume the gate simply had nothing to compare.
    assert "same number" in detail


def test_a_deterministic_reading_that_moved_the_wrong_way_regresses():
    b = baseline.from_artefact(_coherence_artefact(), "coherence", seeds=[0])
    # Coherence is a negative log likelihood: lower is better, so a rise is the worse direction.
    assert b["direction"] == baseline.LOWER_IS_BETTER
    ok, headline, detail = baseline.verdict(b, 2.5000, None)
    assert not ok
    assert "REGRESSED" in headline
    assert "no run-to-run noise" in detail


def test_a_deterministic_reading_that_improved_passes_and_says_so():
    b = baseline.from_artefact(_coherence_artefact(), "coherence", seeds=[0])
    ok, headline, _ = baseline.verdict(b, 2.3000, None)
    assert ok
    assert "improved" in headline


def test_the_smallest_possible_move_is_not_swallowed():
    """No tolerance, deliberately. A tolerance here would be an interval nobody measured."""
    b = baseline.from_artefact(_coherence_artefact(2.4131), "coherence", seeds=[0])
    ok, headline, _ = baseline.verdict(b, 2.4131 + 1e-9, None)
    assert not ok, "a deterministic metric that moved at all has either changed or was not deterministic"
    assert "REGRESSED" in headline


def test_the_bluntness_guard_is_skipped_for_a_deterministic_baseline():
    """It asks whether this run could have SEEN a move. A number with no width always could."""
    b = baseline.from_artefact(_coherence_artefact(), "coherence", seeds=[0])
    baseline.refuse_if_too_blunt(b, None)


def test_the_bluntness_guard_still_fires_for_an_interval_baseline():
    b = baseline.from_artefact(_refusal_artefact(), "refusal_rate.senbonzakura-ruler", seeds=[0])
    with pytest.raises(baseline.BaselineError, match="times wider"):
        baseline.refuse_if_too_blunt(b, (0.05, 0.95))


def test_a_metric_with_neither_an_interval_nor_the_flag_is_still_refused():
    """The original refusal has to survive, or the repair has opened the hole it closed."""
    doc = _stamped("coherence", 2.41, "neutral-passage-nll", n=1, **_pinned_fields())
    with pytest.raises(baseline.BaselineError, match="has no interval"):
        baseline.from_artefact(doc, "coherence", seeds=[0])


def test_a_metric_claiming_both_is_refused_rather_than_resolved():
    """Two contradicting claims, and nothing here can tell which one is the mistake."""
    doc = _stamped("coherence", 2.41, "neutral-passage-nll", n=1, deterministic=True,
                   interval=[2.4, 2.5], **_pinned_fields())
    with pytest.raises(baseline.BaselineError, match="deterministic and also carries an interval"):
        baseline.from_artefact(doc, "coherence", seeds=[0])


def test_record_refuses_the_same_contradiction():
    with pytest.raises(baseline.BaselineError, match="deterministic and also carries an interval"):
        baseline.record(model=MODEL, metric="coherence", direction=baseline.LOWER_IS_BETTER,
                        point=2.41, interval=(2.4, 2.5), deterministic=True, seeds=[0], n=1,
                        estimator="neutral-passage-nll", **_pinned_fields())


def test_record_refuses_a_hole_that_claims_nothing():
    with pytest.raises(baseline.BaselineError, match="not declared deterministic"):
        baseline.record(model=MODEL, metric="coherence", direction=baseline.LOWER_IS_BETTER,
                        point=2.41, interval=None, seeds=[0], n=1,
                        estimator="neutral-passage-nll", **_pinned_fields())


def test_two_instruments_are_not_compared_against_each_other():
    """A baseline with a spread and a run declaring determinism changed instrument between them."""
    det = baseline.from_artefact(_coherence_artefact(), "coherence", seeds=[0])
    iv = baseline.from_artefact(_refusal_artefact(), "refusal_rate.senbonzakura-ruler", seeds=[0])
    with pytest.raises(baseline.BaselineError, match="not produced by the same"):
        baseline.verdict(det, 2.41, (2.4, 2.5))
    with pytest.raises(baseline.BaselineError, match="nothing to compare it against"):
        baseline.verdict(iv, 0.094, None)


def test_interval_of_reads_both_shapes_without_the_caller_knowing():
    assert baseline.interval_of({"deterministic": True, "interval": None}) is None
    assert baseline.interval_of({"deterministic": False, "interval": [0.1, 0.2]}) == (0.1, 0.2)
    # The shape that used to reach `tuple(None)` and be reported as an unreadable measurement.
    assert baseline.interval_of({"interval": None}) is None


def test_the_gate_runs_end_to_end_over_a_deterministic_pair(tmp_path):
    """Through the command, not through the library. A gate verified only through its parts is a
    gate whose plumbing has never been run.
    """
    from senbonzakura import gate
    b = baseline.from_artefact(_coherence_artefact(), "coherence", seeds=[0])
    recorded = tmp_path / "baseline.json"
    baseline.write(recorded, b)

    for now, expected in ((2.4131, gate.OK), (2.5000, gate.REGRESSED)):
        current = tmp_path / f"now-{now}.json"
        baseline.write(current, baseline.from_artefact(
            _coherence_artefact(now), "coherence", seeds=[0]))
        lines = []
        argv = ["--baseline", str(recorded), "--measurement", str(current)]
        assert gate.run(argv, log=lines.append) == expected, "\n".join(lines)


def test_pinned_derives_every_field_a_gate_needs():
    class Model:
        dtype = "torch.bfloat16"

    class Tok:
        senbon_chat_template = "qwen"

    got = stamps.pinned(prompts=["a", "b"], model=Model(), tok=Tok(), skip=4, recorded_skip=4)
    assert set(got) >= set(baseline.PINNED) - {"model", "metric", "estimator"}
    assert got["partition"] == stamps.MEASURE
    assert got["prompt_format"] == "qwen"
    assert got["precision"] == "bfloat16"
