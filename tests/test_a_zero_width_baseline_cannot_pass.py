# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A baseline with a zero-width interval cannot make an imprecise run pass the gate.

THE FAILURE THIS EXISTS TO CATCH: the too-blunt guard exists so that a run which could not have seen a
move is refused rather than passed. It took its ratio against the baseline's width, and a zero-width
baseline has no ratio, so the guard returned without checking. A refusal rate that went from 0.094 to
0.55, measured on a run whose interval spans 0.05 to 0.95, then printed "no evidence the property
moved". The reassurance was the defect.

The defect was confirmed against the real gate command on 2026-10-14, before this change.

Also pinned here: the gate compares the seed set a baseline was measured over, so a run over different
seeds is refused rather than passed with the baseline's seeds printed beside it. That was a strict
expected failure until the writers stated their seeds, and the marker came off in the same commit.
"""
from __future__ import annotations

import json

import pytest

from senbonzakura import baseline as b
from senbonzakura import gate

COMMON = dict(model="Qwen/Qwen3-0.6B", metric="refusal_rate", direction=b.LOWER_IS_BETTER,
              input_digest="sha256:input", partition="measure", prompt_format="renderer:v3",
              tool_version="0.9.0", estimator="senbonzakura-ruler", precision="bfloat16",
              seeds=[42], n=200)


def _addressed(doc):
    doc[b.ADDRESS_FIELD] = b.address(doc)
    return doc


def _write(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_record_refuses_a_zero_width_interval_on_a_metric_that_varies():
    """A zero-width interval is a claim of perfect precision, and no bootstrap produces one."""
    with pytest.raises(b.BaselineError, match="zero width"):
        b.record(point=0.094, interval=[0.094, 0.094], **COMMON)


def test_record_still_accepts_a_real_interval():
    doc = b.record(point=0.094, interval=[0.061, 0.133], **COMMON)
    assert doc["interval"] == [0.061, 0.133]


def test_a_wide_run_against_a_zero_width_baseline_is_refused_as_too_blunt():
    """The guard itself, given a baseline that record would have refused. A file written before the
    record check existed reaches the guard this way, so the guard refuses on its own.
    """
    baseline = {"deterministic": False, "interval": [0.094, 0.094], "n": 200, "seeds": [42]}
    with pytest.raises(b.BaselineError, match="zero width"):
        b.refuse_if_too_blunt(baseline, [0.05, 0.95])


def test_a_zero_width_run_against_a_zero_width_baseline_is_not_blunt():
    """Nothing wider than nothing: the guard must still answer when neither side has any width."""
    baseline = {"deterministic": False, "interval": [0.094, 0.094], "n": 200, "seeds": [42]}
    b.refuse_if_too_blunt(baseline, [0.094, 0.094])


def test_the_gate_refuses_a_wide_run_against_a_zero_width_baseline_end_to_end(tmp_path):
    """Through the gate's own entry point, the way a CI job reaches it, not only the function."""
    base = _addressed(dict(b.record(point=0.094, interval=[0.061, 0.133], **COMMON),
                           interval=[0.094, 0.094]))
    current = b.record(point=0.55, interval=[0.05, 0.95], **COMMON)
    lines: list[str] = []
    status = gate.run(["--baseline", str(_write(tmp_path / "base.json", base)),
                       "--current", str(_write(tmp_path / "now.json", current))],
                      log=lines.append)
    assert status == gate.REFUSED, "a pass here is the reassurance the guard exists to prevent"
    assert any("zero width" in line for line in lines), lines


def test_a_run_over_a_different_seed_set_is_refused(tmp_path):
    base = _addressed(b.record(point=0.094, interval=[0.061, 0.133], **COMMON))
    other = dict(COMMON, seeds=[1, 2, 3])
    current = b.record(point=0.094, interval=[0.061, 0.133], **other)
    lines: list[str] = []
    status = gate.run(["--baseline", str(_write(tmp_path / "base.json", base)),
                       "--current", str(_write(tmp_path / "now.json", current))],
                      log=lines.append)
    assert status == gate.REFUSED, "a different seed set builds a different interval"
