# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A rate over too few observations must not be printed as a rate.

WHY THIS FILE EXISTS

Three separate wrong numbers in one night, 2026-09-06, all the same shape.

  * "7.0% against 3.0%" of positions were near-ties. That was 5 observations against 2, and the
    direction REVERSED when the sample grew to 1092.
  * Five generations hashed identical and were read as determinism. All five were the empty
    string.
  * A 280-row matched corpus turned out to be 40 distinct requests, which partitioned three ways
    would have produced an AUC over about 13 items.

None of the surfaces refused. Every rate in `metrics` divides by `max(1, len(texts))` and returns
a clean float for a single row, which is a number that looks exactly like a measurement.

WHAT THE FLOOR IS AND IS NOT

It is a floor against absurdity, not a guarantee of power: 30 is still weak for comparing two
rates. The interval is what tells a reader how weak, which is why it is reported whether or not
the rate is.
"""
import pytest

from senbonzakura import capability as cap
from senbonzakura import metrics


def test_a_tiny_sample_gets_no_rate():
    """THE REGRESSION THIS FILE IS NAMED FOR."""
    r = metrics.reportable_rate(2, 5)
    assert r["rate"] is None
    assert r["reportable"] is False
    assert "below the floor" in r["why_not"]


def test_the_counts_survive_even_when_the_rate_does_not():
    """`62/120` says what `51.7%` hides, and counts cannot be quoted as something they are not."""
    r = metrics.reportable_rate(2, 5)
    assert r["count"] == 2 and r["n"] == 5


def test_a_large_enough_sample_gets_its_rate():
    r = metrics.reportable_rate(62, 120)
    assert r["reportable"] is True
    assert r["rate"] == pytest.approx(62 / 120)


def test_the_rog_result_that_reversed_would_have_shown_overlapping_intervals():
    """The specific number this was built after.

    At n=66 and n=71 the near-tie rates were 3.0% and 7.0%, which read as a clean confirmation.
    The intervals overlap heavily, so it never was one, and the direction reversed at n=1092.
    """
    stock = metrics.reportable_rate(2, 66)
    abliterated = metrics.reportable_rate(5, 71)
    lo_a, hi_a = stock["ci"]
    lo_b, hi_b = abliterated["ci"]
    assert lo_b < hi_a, "the intervals must overlap; if they do not, this example is wrong"


# ── the interval has to behave where it matters ──────────────────────────────────────

@pytest.mark.parametrize(("count", "n"), [(0, 1), (1, 1), (0, 5), (5, 5), (1, 3), (30, 30)])
def test_the_interval_stays_inside_zero_and_one(count, n):
    """The normal approximation runs below zero and above one at small n, which is exactly where
    this is used. Wilson's does not, and that is the whole reason for choosing it.
    """
    lo, hi = metrics.wilson_interval(count, n)
    assert 0.0 <= lo <= hi <= 1.0


def test_an_empty_sample_says_it_knows_nothing():
    assert metrics.wilson_interval(0, 0) == (0.0, 1.0)


def test_the_interval_narrows_as_the_sample_grows():
    """Otherwise it is decoration rather than information."""
    widths = []
    for n in (10, 100, 1000, 10000):
        lo, hi = metrics.wilson_interval(n // 2, n)
        widths.append(hi - lo)
    assert widths == sorted(widths, reverse=True)


def test_the_interval_contains_the_point_estimate():
    lo, hi = metrics.wilson_interval(30, 100)
    assert lo <= 0.30 <= hi


def test_a_unanimous_result_does_not_claim_certainty():
    """5 out of 5 is not 100% with no doubt attached, and reporting it that way is how a small
    sample becomes a confident claim.
    """
    lo, hi = metrics.wilson_interval(5, 5)
    assert lo < 1.0, "an interval that excludes doubt at n=5 is the defect, not the fix"


def test_metrics_still_imports_nothing():
    """The floor had to be written without imports because this module has none by design: the
    head of the file explains why, and the head-to-head runs it inside a container holding a
    competitor's dependency tree.
    """
    import pathlib
    src = pathlib.Path(metrics.__file__).read_text(encoding="utf-8")
    imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert imports == [], f"metrics.py must stay import-free; found {imports}"


# ── the capability summary honours it ────────────────────────────────────────────────

def test_a_small_capability_run_reports_counts_and_refuses_the_rate():
    summary = cap.summarise(["correct"] * 3 + ["wrong"] * 2)
    assert summary["accuracy"] is None
    assert summary["accuracy_reportable"] is False
    text = "\n".join(cap.report(summary))
    assert "3/5" in text
    assert "NOT REPORTED as a rate" in text


def test_a_large_capability_run_reports_the_rate_with_its_interval():
    summary = cap.summarise(["correct"] * 80 + ["wrong"] * 40)
    assert summary["accuracy"] == pytest.approx(80 / 120, abs=1e-4)
    text = "\n".join(cap.report(summary))
    assert "80/120" in text, "the counts go beside the rate, not instead of it"
    assert "95% CI" in text


def test_the_interval_is_reported_even_when_the_rate_is_withheld():
    """A reader who cannot have the rate can still have the width, which is the more honest half."""
    summary = cap.summarise(["correct"] * 3 + ["wrong"] * 2)
    assert summary["accuracy_ci"] is not None
