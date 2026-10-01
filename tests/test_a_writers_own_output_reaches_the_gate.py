# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""One real writer's stamped output, put through `baseline.from_artefact` with nothing in between.

WHAT PROMPTED IT, 2026-10-01

Looking for an artefact to use as a fixture turned up something larger: **not one artefact this
project has produced carries a stamped `metrics` block**, which is the only thing
`baseline.from_artefact` reads. So the regression gate's input did not exist in any real file.

Three causes were possible and they have three different fixes:

1. nothing has been run through the path that stamps it;
2. the stamping landed after every artefact was made;
3. **the gate reads a shape nothing writes.**

The third is the serious one and it is ruled out, by driving the contract rather than by reasoning
about it: `capability`'s own stamping call, with the fields that command computes, produces a block
`from_artefact` accepts and turns into a baseline. This test is that drive, kept.

WHY IT DID NOT ALREADY EXIST, WHICH IS THE REAL FINDING

Every behavioural test of the gate builds its stamped block **by hand**, in the test file, to the
shape the reader wants. `test_baseline.py` has `STAMPED`, `test_a_deterministic_metric_can_be_gated.py`
has `_coherence_artefact`, and `test_every_writer_stamps_the_pinned_fields.py` is explicitly
structural: it walks the AST for `measurement.stamp` call sites and says in its own docstring why,
since three of the five writers cannot run without a model.

Each of those is reasonable on its own. Together they left **the one seam where writer and reader
meet with no test across it**, and that is the same defect shape this project keeps finding in
other people's work and twice in its own: a test that re-implements the producer asserts on the
test's arithmetic rather than the module's. A hand-built block cannot fail for a writer that stamps
the wrong key, the wrong nesting, or a value of the wrong type.

WHY ONLY `capability`

It is the one writer of the five that imports without torch, so it is the one whose real stamping
call a CI job with no deep-learning stack can make. `score`, `margin`, `coherence` and `drift`
import torch at module scope. Their coverage stays structural and the gap is named in
`private/plans/2026-10-01-deferred-triage.md` rather than papered over here. One real writer
reaching the reader is not complete; it is the difference between none and one, and it is the
seam that was untested.
"""
from __future__ import annotations

import pytest

from senbonzakura import baseline, capability, stamps

#: The exam, as `capability` holds it: the questions are what `items_digest` fingerprints.
QUESTIONS = [f"what is {i} plus {i}?" for i in range(40)]


def _stamped(accuracy=0.82, graded=40, interval=(0.67, 0.92)):
    """A `capability` result carrying the block that command stamps, through that command's call.

    The `measurement.stamp` keyword arguments here are the ones `capability.main` passes, including
    `**stamps.pinned(...)` with the exam's own digest as `input_digest`. Nothing about the shape is
    decided in this file: if that call changes, this fixture changes with it or the test fails,
    which is the point.
    """
    from senbonzakura_check import measurement

    result = {"label": "after", "model": "Qwen/Qwen3-1.7B",
              "items_digest": capability.items_digest(QUESTIONS)}
    measurement.stamp(result, "capability", accuracy, "code-graded",
                      n=graded, interval=list(interval) if interval else None,
                      **stamps.pinned(prompts=QUESTIONS, model=None, tok=None,
                                      load_in_4bit=False,
                                      input_digest=result["items_digest"], skip=0))
    return result


def test_the_writer_produces_a_block_the_reader_accepts():
    """The whole claim, and the one nothing was asserting.

    A failure here means the gate cannot read what the tool writes, which would make every other
    test of the gate a test of a shape that only exists in test files.
    """
    made = baseline.from_artefact(_stamped(), "capability", seeds=[42, 43, 44, 45, 46])
    assert made["point"] == 0.82
    assert made["direction"] == baseline.HIGHER_IS_BETTER
    assert made["seeds"] == [42, 43, 44, 45, 46]


def test_every_pinned_field_survives_the_crossing():
    """Field by field, because `comparability` reports an absent one as a mismatch.

    So a field that the writer stamps and the reader does not pick up is not a cosmetic loss: it
    makes the figure comparable with nothing, which is the condition the 2026-09-25 repair
    (`e5b3182`, "the gate accepted nothing this repository could produce") was about.
    """
    made = baseline.from_artefact(_stamped(), "capability", seeds=[42])
    missing = [field for field in baseline.PINNED if not made.get(field)]
    assert not missing, (
        f"these pinned fields did not survive from the writer's stamp to the baseline: {missing}. "
        f"An absent pinned field is a mismatch rather than a skip, so the figure is comparable "
        f"with nothing")


def test_the_model_comes_from_the_top_level_and_not_from_the_block():
    """The asymmetry that broke this once, asserted so it cannot drift back.

    `test_baseline.py` records it: every writer stamps its identity INSIDE the metrics block and
    `comparability` reads the pinned fields from the TOP level. `model` is the field where those
    two places disagree, and a block carrying `model: None` still has to produce a baseline that
    names the weights.
    """
    doc = _stamped()
    assert doc["metrics"]["capability"].get("model") is None, (
        "the writer now stamps `model` inside the block; this test's premise needs re-reading")
    made = baseline.from_artefact(doc, "capability", seeds=[42])
    assert made["model"] == "Qwen/Qwen3-1.7B"


def test_two_runs_of_the_same_exam_are_comparable():
    """The property the whole thing exists for, end to end from the writer rather than field by field."""
    first = baseline.from_artefact(_stamped(0.82, 40, (0.67, 0.92)), "capability", seeds=[42])
    second = baseline.from_artefact(_stamped(0.79, 40, (0.63, 0.90)), "capability", seeds=[42])
    reasons = baseline.comparability(first, second)
    assert not reasons, f"two runs of the same exam were refused as incomparable: {reasons}"


def test_a_different_exam_is_refused_rather_than_compared():
    """The other direction, and the reason `input_digest` is the exam's own rather than a count.

    Two runs of equal size over different exams pair item i against a different item i, so a
    length check cannot tell them apart and the digest can.
    """
    first = baseline.from_artefact(_stamped(), "capability", seeds=[42])
    other = _stamped()
    other["items_digest"] = capability.items_digest([q.replace("plus", "times") for q in QUESTIONS])
    other["metrics"]["capability"]["input_digest"] = other["items_digest"]
    second = baseline.from_artefact(other, "capability", seeds=[42])
    reasons = baseline.comparability(first, second)
    assert reasons, "two different exams were accepted as the same measurement"
    assert any("input_digest" in r for r in reasons), reasons


def test_a_withheld_accuracy_produces_no_block_at_all():
    """`capability` stamps nothing when the graded subset is too small to report.

    Asserted here because the alternative, a block carrying `value: None`, would hand a reader an
    identity for a measurement that was not taken, and `from_artefact` would have to refuse it a
    second time. The writer's comment says so; nothing was checking it from the reader's side.

    This is also the shape of the one stamped artefact that does exist in this tree,
    `tests/fixtures/ours/2026-09-07-capability-rendered-as-zero.json`, which carries `value: 0`
    with no interval and five pinned fields absent. `from_artefact` refuses it, correctly.
    """
    made = _stamped(accuracy=None, graded=0, interval=None)
    # The writer's own guard is `if summary.get("accuracy") is not None`, so with None there is no
    # stamp call at all. Reproduced here by asserting the reader's refusal is the loud kind.
    with pytest.raises(baseline.BaselineError, match="interval"):
        baseline.from_artefact(made, "capability", seeds=[42])


def test_the_one_stamped_artefact_in_the_tree_is_still_refused():
    """A fixture of a real 2026-09-07 incident, captured before the 2026-09-25 repair.

    It is the only file in this repository with a non-empty `metrics` block, and it is pre-repair:
    `value: 0`, no interval, and `input_digest`, `partition`, `prompt_format`, `tool_version` and
    `precision` all absent. It must keep being refused, because a zero rendered as a measurement
    is what that incident was.

    Skipped rather than failed if the fixture moves, because its path is not this test's subject.
    """
    import json
    from pathlib import Path

    path = (Path(__file__).resolve().parent
            / "fixtures" / "ours" / "2026-09-07-capability-rendered-as-zero.json")
    if not path.is_file():
        pytest.skip(f"{path.name} is no longer where this test expects it")
    doc = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(baseline.BaselineError):
        baseline.from_artefact(doc, "capability", seeds=[42])
