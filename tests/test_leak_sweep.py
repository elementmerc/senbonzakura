# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The measurement behind a published figure, checked against the figure.

WHY THIS FILE EXISTS

`tools/leak_sweep.py` produces the numbers for a chart that goes out in public. A chart drawn from
a handoff paragraph is a picture of somebody's memory; this is what makes it a rendering of a
measurement instead. That only helps if the measurement itself is pinned, so these assert the
shape of the curve AND the value at the point the published range is quoted from.

If the published range ever stops matching what the script produces, one of the two is wrong and
this is where that gets noticed, rather than in a reply from somebody who reran it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import leak_sweep


def test_the_leak_grows_with_how_uneven_the_rows_are():
    """The claim is not "there is a leak", it is "how big depends on the model"."""
    got = [leak_sweep.measure(s, 2, 0, 0)[0] for s in (0.2, 2.0, 10.0)]
    assert got[0] < got[1] < got[2], got
    assert got[0] < 0.15, "an almost-even matrix should barely leak"
    assert got[2] > 0.30, "a very uneven one should leak a lot"


@pytest.mark.parametrize("k", [1, 2, 4])
def test_four_rounds_removes_the_leak_and_keeps_the_lengths(k):
    """Both properties, or it is not a fix.

    Restoring the lengths is not a decoration: the model needs them. A cleaner cut bought by
    abandoning them would be a different edit, not a better one.
    """
    from senbonzakura.cli import ABLATION_ROUNDS

    leak, err = leak_sweep.measure(4.0, k, 0, ABLATION_ROUNDS)
    assert leak < 1e-4, f"the direction is still there at K={k}: {leak}"
    assert err < 1e-3, f"the row lengths did not survive at K={k}: {err}"


def test_the_published_range_matches_what_the_script_measures():
    """THE GUARD ON THE PUBLIC CLAIM.

    We say the standard edit puts back roughly 20 to 45% on realistic weights, quoting real
    models measured at 6 to 7.5x row-length spread. If the script stops producing that, the
    sentence is wrong and this fails before a reader reruns it.
    """
    at = {s: leak_sweep.measure(s, 2, 0, 0)[0] for s in (6.0, 8.0)}
    assert 0.20 <= at[6.0] <= 0.45, at
    assert 0.20 <= at[8.0] <= 0.45, at


def test_the_sweep_writes_both_arms_at_every_point():
    """A curve with only the bad arm on it is an argument, not a measurement."""
    from senbonzakura.cli import ABLATION_ROUNDS

    rows = leak_sweep.sweep(spreads=(1.0, 4.0), seeds=1, directions=(2,))
    assert {r["rounds"] for r in rows} == {0, ABLATION_ROUNDS}
    for r in rows:
        assert r["leak_fraction_min"] <= r["leak_fraction_median"] <= r["leak_fraction_max"]


def test_a_model_directory_with_no_weights_is_refused_plainly(tmp_path):
    with pytest.raises(SystemExit, match=r"no \.safetensors"):
        leak_sweep.model_spreads(tmp_path)


def test_row_norm_stats_declines_a_matrix_too_small_to_describe():
    """A percentile over a handful of rows is not a percentile."""
    import torch

    assert leak_sweep._row_norm_stats(torch.randn(10, 4)) is None
    assert leak_sweep._row_norm_stats(torch.randn(200, 4)) is not None


def test_zero_seeds_is_refused():
    with pytest.raises(SystemExit, match="at least 1"):
        leak_sweep.main(["--seeds", "0"])
