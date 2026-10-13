# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A point estimate that is not a finite number is a refusal, on both sides of the gate.

Exit 0 from `senbonzakura gate` is a promise that a measured property did not move. A NaN point
estimate means the measurement did not produce a number at all, which is the one thing that must
never be read as agreement.

**It was.** Found by the integrator journey of 2026-10-11, and reproduced here before it was fixed:
a baseline-shaped file whose `point` was NaN, infinity, or the string `"0.9"` reached
`baseline.verdict`, compared cleanly against the baseline's interval, and the gate printed

    gate OK: auc: within interval
      baseline 0.9872 [0.9722, 0.9972] on n=60, this run nan [0.9722, 0.9972]. The intervals
      overlap, so there is no evidence the property moved. Moved +nan on the point estimate,
      which on its own is not a finding.

and exited 0.

**The two sides disagreed, which is how it survived.** `baseline.record` did refuse a NaN, but only
by accident: the interval test `lo <= point <= hi` is False for NaN, so the writer reported "point
estimate nan lies outside its own interval", a true sentence about the wrong problem. The reader had
no check whatsoever. `baseline.numeric_point` is now the single rule both call, which is why the
writer's message and the gate's message are the same sentence.

The content address does not help here and is not supposed to: it is self-computed, so it detects an
accidental edit and not a deliberate one. The journey re-addressed its files, as anything producing
these files legitimately would.
"""

import json
import math
import subprocess
import sys

import pytest

from senbonzakura import baseline

COMMON = dict(model="Qwen/Qwen3-1.7B", metric="auc", direction="higher_is_better",
              input_digest="a" * 64, partition="measure", prompt_format="chat-v1",
              tool_version="0.4.1", estimator="bootstrap-percentile", precision="bfloat16",
              seeds=[1, 2, 3, 4, 5], n=60)

NOT_NUMBERS = [float("nan"), float("inf"), float("-inf"), "0.9", None, [0.9], {"v": 0.9}, True]


# ── the rule itself ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("point", NOT_NUMBERS)
def test_a_point_that_is_not_a_finite_number_is_refused(point):
    with pytest.raises(baseline.BaselineError):
        baseline.numeric_point(point)


@pytest.mark.parametrize("point", [0.0, 1.0, 0.9872, -3.5, 0, 42])
def test_a_real_number_passes_and_comes_back_as_a_float(point):
    """The other direction. A rule that refused everything would satisfy every test above."""
    out = baseline.numeric_point(point)
    assert isinstance(out, float)
    assert out == float(point)
    assert math.isfinite(out)


def test_true_is_not_a_measurement():
    """`bool` is a subclass of `int`, so a naive isinstance check accepts `True` as 1.0.

    Nothing produces this deliberately; it is here because the obvious implementation of this
    function has the bug and the obvious test does not catch it.
    """
    with pytest.raises(baseline.BaselineError):
        baseline.numeric_point(True)


def test_the_refusal_says_the_measurement_produced_no_number():
    """The message has to say what to do, because a CI job's operator reads only this line."""
    with pytest.raises(baseline.BaselineError) as e:
        baseline.numeric_point(float("nan"))
    text = str(e.value)
    assert "not a finite number" in text
    assert "did not produce" in text
    assert "Re-run the measurement" in text


def test_the_side_is_named_so_a_reader_knows_which_file_is_wrong():
    """A gate reads two files. "a point estimate of nan" alone does not say which one."""
    with pytest.raises(baseline.BaselineError) as e:
        baseline.numeric_point(float("nan"), side="this run")
    assert "this run" in str(e.value)


# ── both sides call it ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("point", [float("nan"), float("inf"), "0.9"])
def test_the_writer_refuses_to_record_one(point):
    """And with the same sentence as the reader, which is the point of sharing the rule.

    Before this, the writer refused a NaN as "lies outside its own interval" and the reader did not
    refuse it at all.
    """
    with pytest.raises(baseline.BaselineError) as e:
        baseline.record(point=point, interval=[0.9, 1.0], **COMMON)
    assert "not a finite number" in str(e.value) or "not a number" in str(e.value)


def _write(tmp_path, name, point):
    """A baseline-shaped file built by the real writer, with only the point replaced.

    Re-addressed, because a producer of these files computes the address the same way and the
    address is not a signature.
    """
    good = baseline.record(point=0.9872, interval=[0.9722, 0.9972], **COMMON)
    doc = dict(good)
    doc["point"] = point
    doc.pop(baseline.ADDRESS_FIELD, None)
    doc[baseline.ADDRESS_FIELD] = baseline.address(doc)
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return good, path


def _gate(baseline_path, current_path):
    return subprocess.run(
        [sys.executable, "-m", "senbonzakura", "gate",
         "--baseline", str(baseline_path), "--current", str(current_path)],
        capture_output=True, text=True, timeout=300, check=False)


@pytest.mark.parametrize("point", [float("nan"), float("inf"), "0.9"])
def test_the_gate_refuses_rather_than_passing(tmp_path, point):
    """Driven as a subprocess, because the exit code IS the contract an integrator wires to.

    REFUSED is 2 and not 1 deliberately: 1 is this command's code for REGRESSED, and a file that
    never produced a number did not regress. `gate`'s own docstring makes that argument about a
    missing `point`; this is the same argument about a `point` that is present and useless.
    """
    good = baseline.record(point=0.9872, interval=[0.9722, 0.9972], **COMMON)
    base = tmp_path / "b.json"
    base.write_text(json.dumps(good, indent=1), encoding="utf-8")
    _, current = _write(tmp_path, "c.json", point)

    r = _gate(base, current)
    assert r.returncode == 2, (
        f"a measurement with point {point!r} got exit {r.returncode}, and 0 would be the gate "
        f"promising a property did not move on the strength of a number that does not exist.\n"
        f"{r.stdout}\n{r.stderr}")
    assert "REFUSED" in r.stdout + r.stderr
    # The VERDICT line, not the phrase: the refusal quotes "within interval" while explaining what
    # it is declining to say, which a naive substring check reads as the verdict it is refusing.
    assert "gate OK" not in r.stdout


def test_a_real_comparison_still_passes(tmp_path):
    """The control, as a subprocess too.

    Without it every assertion above is satisfied by a gate that refuses everything, which would be
    a worse defect than the one being fixed.
    """
    good = baseline.record(point=0.9872, interval=[0.9722, 0.9972], **COMMON)
    base = tmp_path / "b.json"
    base.write_text(json.dumps(good, indent=1), encoding="utf-8")

    r = _gate(base, base)
    assert r.returncode == 0, f"a legitimate comparison was refused:\n{r.stdout}\n{r.stderr}"
    assert "gate OK" in r.stdout
