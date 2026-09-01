# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The compass must skip the rows the search saw, not a guess at how many there were.

`margin.py` carried `--skip-harmful 128`, reasoned from "the largest --eval-refusal-final any auto
preset uses". That is a statement about this tool's flags. The track that shipped holds 132 harmful
rows for selection, and `flag_violations` permits a run to use all 132 and recommends exactly that.
A run that took the recommendation had rows 128 to 131 scored as though they were held out, with
nothing saying so.

`track.json` records `skip_harmful` for precisely this purpose. These tests pin that the recorded
boundary wins, that an explicit override still works, and that a contradiction is refused rather
than silently resolved.
"""
import json

import pytest

from senbonzakura.margin import (
    LEGACY_SKIP_HARMFUL,
    LEGACY_SKIP_HARMLESS,
    resolve_skips,
)


def _track(tmp_path, *, skip_harmful=132, skip_harmless=385, schema="senbonzakura-track/1"):
    d = tmp_path / "trk"
    d.mkdir(exist_ok=True)
    (d / "track.json").write_text(json.dumps({
        "schema": schema,
        "counts": {"harmful": {"fit": 259, "search": skip_harmful, "measure": 4504},
                   "harmless": {"fit": 257, "search": 128, "measure": 4597}},
        "skip_harmful": skip_harmful,
        "skip_harmless": skip_harmless,
    }), encoding="utf-8")
    return d


# ── the manifest is the authority ──────────────────────────────────────────────────
def test_the_recorded_boundary_beats_the_default(tmp_path):
    # THE REGRESSION GUARD. 132, not 128: the four-row contamination this file exists for.
    h, g = resolve_skips(_track(tmp_path), None, None)
    assert h == 132
    assert h != LEGACY_SKIP_HARMFUL
    assert g == 385


def test_the_boundary_is_read_for_both_arms(tmp_path):
    h, g = resolve_skips(_track(tmp_path, skip_harmful=200, skip_harmless=500), None, None)
    assert (h, g) == (200, 500)


def test_reading_the_boundary_is_announced(tmp_path):
    logged = []
    resolve_skips(_track(tmp_path), None, None, log=logged.append)
    assert any("132" in m and "recorded boundary" in m for m in logged)


# ── an explicit value still wins, because scoring the selection set is a real thing to want ──
def test_an_explicit_skip_is_honoured_when_no_track_is_given(tmp_path):
    assert resolve_skips(None, 0, 0) == (0, 0)


def test_an_explicit_skip_matching_the_track_is_fine(tmp_path):
    assert resolve_skips(_track(tmp_path), 132, 385) == (132, 385)


def test_zero_is_not_confused_with_unset(tmp_path):
    """`--skip-harmful 0` scores the selection set deliberately; it must not read as absent."""
    h, _ = resolve_skips(None, 0, None)
    assert h == 0


# ── a contradiction is refused, not resolved ───────────────────────────────────────
def test_a_skip_that_contradicts_the_track_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        resolve_skips(_track(tmp_path), 128, None)
    msg = str(e.value)
    assert "128" in msg and "132" in msg
    assert "--skip-harmful" in msg


def test_the_refusal_says_which_to_drop(tmp_path):
    with pytest.raises(SystemExit) as e:
        resolve_skips(_track(tmp_path), 128, None)
    msg = str(e.value)
    assert "--track" in msg          # names both ways out
    assert "manifest" in msg


def test_the_harmless_arm_is_checked_too(tmp_path):
    with pytest.raises(SystemExit, match="--skip-harmless"):
        resolve_skips(_track(tmp_path), None, 320)


# ── no manifest: the old behaviour, but it says it is guessing ─────────────────────
def test_without_a_track_the_legacy_defaults_apply(tmp_path):
    assert resolve_skips(None, None, None) == (LEGACY_SKIP_HARMFUL, LEGACY_SKIP_HARMLESS)


def test_falling_back_to_a_default_warns_that_it_is_a_guess(tmp_path):
    logged = []
    resolve_skips(None, None, None, log=logged.append)
    joined = " ".join(logged)
    assert "WARNING" in joined
    assert "guess" in joined
    assert "--track" in joined       # says how to fix it


def test_a_track_with_no_recorded_boundary_falls_back_and_says_so(tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    assert resolve_skips(d, None, None) == (LEGACY_SKIP_HARMFUL, LEGACY_SKIP_HARMLESS)


def test_an_explicit_value_against_a_boundaryless_track_is_flagged_unverified(tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    logged = []
    assert resolve_skips(d, 64, 64, log=logged.append) == (64, 64)
    assert any("unverified" in m for m in logged)
