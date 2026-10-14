# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every measurement states the seeds its interval was drawn over, and the gate refuses on a mismatch.

THE FAILURE THIS EXISTS TO CATCH: a baseline and a run over different seed sets built different
intervals, and the gate compared them silently, printing the baseline's seeds beside a result from
another set. The seed set was recorded and printed but never compared.

Two things must hold for the fix to stay fixed. A writer must say its seeds, and it must not be able
to leave them out without a failure at the call. And a method that draws no samples must say so
(`NO_SEED`), which is a different claim from a missing field: a missing field is unknown and the gate
refuses on it, so the two must never look alike.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from senbonzakura_check import measurement
from senbonzakura_check.measurement import NO_SEED

from senbonzakura import baseline as b
from senbonzakura import gate, stamps

SRC = Path(__file__).resolve().parent.parent / "src" / "senbonzakura"

COMMON = dict(model="Qwen/Qwen3-0.6B", metric="refusal_rate", direction=b.LOWER_IS_BETTER,
              input_digest="sha256:input", partition="measure", prompt_format="renderer:v3",
              tool_version="0.9.0", estimator="senbonzakura-ruler", precision="bfloat16", n=200)


def _run(tmp_path, baseline_doc, current_doc):
    base = tmp_path / "base.json"
    now = tmp_path / "now.json"
    b.write(base, baseline_doc)
    b.write(now, current_doc)
    lines: list[str] = []
    status = gate.run(["--baseline", str(base), "--current", str(now)], log=lines.append)
    return status, lines


# ── the writer's side: the seeds are required and cannot be left out ──────────────────────────

def test_stamps_pinned_refuses_to_run_without_seeds():
    """No default. A writer that never thought about seeds must fail at the call, not stamp."""
    with pytest.raises(TypeError, match="seeds"):
        stamps.pinned(prompts=["a"], model=None, tok=None)


@pytest.mark.parametrize("seeds", [None, [], "maybe", ["a"], ""])
def test_a_seed_that_is_not_a_seed_is_refused_at_the_pinned_call(seeds):
    with pytest.raises(measurement.MeasurementError):
        stamps.pinned(prompts=["a"], model=None, tok=None, seeds=seeds)


def test_none_is_refused_for_the_reason_a_writer_that_did_not_decide_would_need_to_hear():
    with pytest.raises(measurement.MeasurementError, match="seeds=None"):
        stamps.pinned(prompts=["a"], model=None, tok=None, seeds=None)


def test_a_sampled_writer_records_its_seeds_sorted():
    got = stamps.pinned(prompts=["a"], model=None, tok=None, seeds=[44, 42, 43])
    assert got["seeds"] == [42, 43, 44]


def test_a_method_with_no_sampling_records_the_statement_not_an_absence():
    got = stamps.pinned(prompts=["a"], model=None, tok=None, seeds=NO_SEED)
    assert got["seeds"] == NO_SEED
    assert "seeds" in got, "absence is unknown, and the gate refuses on unknown"


def test_every_call_to_the_pinned_builder_in_the_source_states_its_seeds():
    """A structural guard, read from the source, so a new caller cannot be added without a seed.

    The builder would refuse at run time anyway, but a caller on a rarely-run path could ship and
    fail only when somebody ran that command. This finds the caller at review time.
    """
    missing = [
        f"{path.name}:{node.lineno}"
        for path in sorted(SRC.glob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pinned"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "stamps"
            and not any(k.arg == "seeds" for k in node.keywords))
    ]
    assert not missing, f"stamps.pinned called without seeds at: {missing}"


def test_every_direct_stamp_states_its_seeds_or_spreads_a_block_that_does():
    """A direct `measurement.stamp` either names `seeds=`, or spreads a `**` mapping that carries
    them. The spread is the case the static reading cannot follow, so the test names the spread
    rather than pretending to verify it; the builder test above is what proves the spread carries
    seeds.
    """
    unnamed = []
    for path in sorted(SRC.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "stamp"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "measurement"):
                named = any(k.arg == "seeds" for k in node.keywords)
                spread = any(k.arg is None for k in node.keywords)
                if not (named or spread):
                    unnamed.append(f"{path.name}:{node.lineno}")
    assert not unnamed, f"measurement.stamp without seeds or a spread at: {unnamed}"


# ── the gate's side: a mismatch refuses, and an agreement is still comparable ─────────────────

def _baseline(seeds, point=0.094, interval=(0.061, 0.133)):
    return b.record(point=point, interval=list(interval), seeds=seeds, **COMMON)


def test_a_different_seed_set_is_refused_rather_than_compared(tmp_path):
    status, lines = _run(tmp_path, _baseline([42, 1337]), _baseline([1, 2, 3], point=0.102,
                                                                   interval=(0.068, 0.142)))
    assert status == gate.REFUSED
    assert any("seeds: baseline [42, 1337], this run [1, 2, 3]" in line for line in lines), lines


def test_the_same_seed_set_is_comparable_and_says_so(tmp_path):
    status, lines = _run(tmp_path, _baseline([42, 1337]), _baseline([42, 1337], point=0.102,
                                                                   interval=(0.068, 0.142)))
    assert status == gate.OK
    assert any("seeds 42, 1337" in line for line in lines), lines


def test_a_no_seed_baseline_and_a_no_seed_run_are_comparable(tmp_path):
    status, _ = _run(tmp_path, _baseline(NO_SEED), _baseline(NO_SEED, point=0.102,
                                                            interval=(0.068, 0.142)))
    assert status == gate.OK


def test_a_no_seed_baseline_against_a_sampled_run_is_refused(tmp_path):
    """The two claims are different, so they are not compared, whichever way round they are given."""
    status, _ = _run(tmp_path, _baseline(NO_SEED), _baseline([42], point=0.102,
                                                            interval=(0.068, 0.142)))
    assert status == gate.REFUSED


def test_the_no_seed_statement_is_printed_as_what_it_means(tmp_path):
    status, lines = _run(tmp_path, _baseline(NO_SEED), _baseline(NO_SEED, point=0.102,
                                                                 interval=(0.068, 0.142)))
    assert status == gate.OK
    joined = " ".join(lines)
    assert "no seed, because the method draws no samples" in joined
    assert "no-seed" not in joined, "the raw token should not reach a reader"


# ── building a baseline: the stamp and the stated seeds must agree ────────────────────────────

def _stamped_with(seeds):
    doc = {"model": "m"}
    measurement.stamp(doc, "refusal_rate", 0.094, "senbonzakura-ruler", n=200, by_estimator=True,
                      interval=[0.061, 0.133], input_digest="sha256:input", partition="measure",
                      prompt_format="renderer:v3", tool_version="0.9.0", precision="bfloat16",
                      seeds=seeds)
    return doc


def test_a_baseline_built_with_the_seeds_its_stamp_states_is_accepted():
    made = b.from_artefact(_stamped_with([42, 1337]), "refusal_rate.senbonzakura-ruler",
                           seeds=[42, 1337])
    assert made["seeds"] == [42, 1337]


def test_a_baseline_that_states_different_seeds_from_its_stamp_is_refused():
    with pytest.raises(b.BaselineError, match="stamped as resting on"):
        b.from_artefact(_stamped_with([42, 1337]), "refusal_rate.senbonzakura-ruler",
                        seeds=[1, 2, 3])


def test_a_baseline_stating_seeds_for_a_stamp_with_none_is_refused():
    with pytest.raises(b.BaselineError, match="no seed"):
        b.from_artefact(_stamped_with(NO_SEED), "refusal_rate.senbonzakura-ruler", seeds=[42])
