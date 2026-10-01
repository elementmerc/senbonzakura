# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`validate` was the one entry point that could read past a recorded partition boundary.

`track.flag_violations` has existed since the partition rebuild and refuses exactly this. It was
wired into the abliterate path (`cli.py:3569`) and into the head-to-head stager
(`headtohead_stage.py:179`), and not into `validate`, which is the entry point where it matters
most: every control in that module depends on measuring on rows the directions were not fitted
on, and `prepare_for_bakes` reads `--dir-prompts` plus `--eval-kl` rows out of the concatenated
`good_ds` and slices by index.

The defect was recorded in the ledger as "`prepare_for_bakes` cannot see a partition boundary",
which was true when written and had become true of one caller rather than all of them. These
tests pin the wiring, not the arithmetic: `test_track.py` already owns whether
`flag_violations` computes the right answer, and duplicating that here would give two tests one
fact to disagree about.
"""
import json

import pytest

from senbonzakura import validate


def _manifest(tmp_path, *, harmless_fit=257, harmless_search=128, harmful_search=200):
    track = tmp_path / "track"
    track.mkdir()
    (track / "track.json").write_text(json.dumps({
        "counts": {
            "harmful": {"fit": 259, "search": harmful_search, "measure": 4436},
            "harmless": {"fit": harmless_fit, "search": harmless_search, "measure": 4597},
        },
        "sha256_of_tar": "0" * 64,
    }))
    return track


class _Args:
    def __init__(self, track, **kw):
        self.track = str(track)
        self.eval_refusal = 0
        self.eval_refusal_final = 0
        self.dir_prompts = 0
        self.eval_kl = 0
        for k, v in kw.items():
            setattr(self, k, v)


def test_a_harmless_read_past_fit_and_search_is_refused(tmp_path):
    """The KL reference must not land on the rows the compass reports on."""
    track = _manifest(tmp_path)          # harmless fit + search = 385
    args = _Args(track, dir_prompts=300, eval_kl=200)   # 500 > 385
    with pytest.raises(SystemExit) as e:
        validate._refuse_flags_past_the_track(args, lambda _m: None)
    msg = str(e.value)
    assert "--dir-prompts" in msg and "--eval-kl" in msg
    # The boundary is named, so the message tells the user what to lower it to rather than only
    # that something is wrong.
    assert "385" in msg


def test_a_harmful_selection_read_past_search_is_refused(tmp_path):
    track = _manifest(tmp_path)
    args = _Args(track, eval_refusal_final=256)          # 256 > 200
    with pytest.raises(SystemExit) as e:
        validate._refuse_flags_past_the_track(args, lambda _m: None)
    assert "--eval-refusal-final" in str(e.value)


def test_a_configuration_inside_the_boundaries_is_allowed_and_logs_them(tmp_path):
    track = _manifest(tmp_path)
    args = _Args(track, dir_prompts=257, eval_kl=128)    # exactly 385
    said = []
    validate._refuse_flags_past_the_track(args, said.append)
    assert any("track boundaries" in s for s in said), (
        "a permitted run still records the boundaries it ran inside, because a number whose "
        "partition is not in the log cannot be audited later")


def test_a_track_with_no_manifest_is_not_refused(tmp_path):
    """Deliberately permissive, and it matches `cli.py:3567` rather than being stricter.

    A track built before manifests existed has no boundaries to check against. Refusing it here
    would reject every such track at once, which is a different decision from this fix and one
    that belongs to the operator.
    """
    track = tmp_path / "bare"
    track.mkdir()
    args = _Args(track, dir_prompts=10_000, eval_kl=10_000)
    said = []
    validate._refuse_flags_past_the_track(args, said.append)   # must not raise
    assert not said


def test_the_guard_runs_before_anything_is_loaded(tmp_path):
    """Placement is the point: the refusal must not cost a model download.

    The abliterate path checks after the model loads, because `kageyoshi` overwrites
    `eval_refusal_final` from the loaded model's size and a check before that would inspect the
    parser default. Nothing in `validate` rewrites these flags, so the earliest moment is also
    the correct one. This test pins that by asserting the guard is reached before the
    `Abliterator` is constructed.
    """
    import inspect

    src = inspect.getsource(validate.main)
    guard = src.index("_refuse_flags_past_the_track")
    built = src.index("cli.Abliterator")
    assert guard < built, (
        "the boundary check moved after the Abliterator was constructed, so a run with "
        "impossible flags now pays for a model load before being refused")
