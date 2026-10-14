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
