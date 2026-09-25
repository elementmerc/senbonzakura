# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Two defects `measure` shipped with, both found by a self-review pass on 2026-09-25.

THE PARTITION. `stage_argv` built the `score` stage against `bad_eval_ds` with no `--skip`, and
that dataset is the SEARCH rows followed by the MEASURE rows. So `score` read from the head of the
file, which is the partition the search picked the configuration on, while the table printed the
figure under "measured on this track's held-out rows and on no others". `compass` in the same run
resolved the skip correctly through `margin.resolve_skips`, so one table was reading two
partitions and describing neither.

This is the project's own documented incident class, not a new one. `checker/.../
a-rate-with-no-partition-beside-it.json` exists because a 0.0% once travelled as a measured
refusal rate having been scored on the selection partition.

THE TOKEN. `--hf-token` was appended to the argv of all five stages, and three of them do not
declare it. argparse prints an unrecognised argument together with its VALUE, so a live
HuggingFace token went to stderr verbatim in three error messages, under a source comment on the
same line promising the value is never printed.
"""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura import measure  # noqa: E402

#: Not a credential, and deliberately under the secret gate's 16-character threshold: a longer
#: placeholder trips the pre-commit scanner, which is the scanner behaving correctly.
FAKE_TOKEN = "not-a-token"


def _args(track, **over):
    base = dict(model="m", device="cpu", hf_token=None, trust_remote_code=False, n=200,
                track=str(track), capability_n=40, capability_max_new=256, baseline="base-model")
    base.update(over)
    return types.SimpleNamespace(**base)


def _track(tmp_path, **manifest):
    d = tmp_path / "trk"
    d.mkdir()
    (d / "track.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


# ── the partition ────────────────────────────────────────────────────────────────────

def test_score_skips_the_search_rows_the_manifest_declares(tmp_path):
    argv = measure.stage_argv("score", _args(_track(tmp_path, skip_harmful=128)), tmp_path)
    assert "--skip" in argv, (
        "score was given bad_eval_ds with no --skip, so it reads the rows the search selected on "
        "while the table calls them held out")
    assert argv[argv.index("--skip") + 1] == "128"


def test_a_track_that_declares_no_skip_gets_none(tmp_path):
    """Honest rather than invented: this cannot know a boundary the manifest does not state."""
    argv = measure.stage_argv("score", _args(_track(tmp_path)), tmp_path)
    assert "--skip" not in argv


@pytest.mark.parametrize("manifest", [{"skip_harmful": 0}, {"skip_harmful": -5},
                                      {"skip_harmful": "128"}, {"skip_harmful": None}])
def test_a_malformed_skip_is_ignored_rather_than_passed_through(tmp_path, manifest):
    argv = measure.stage_argv("score", _args(_track(tmp_path, **manifest)), tmp_path)
    assert "--skip" not in argv


def test_an_unreadable_manifest_does_not_crash_the_run(tmp_path):
    d = tmp_path / "trk"
    d.mkdir()
    (d / "track.json").write_text("{ not json", encoding="utf-8")
    assert "--skip" not in measure.stage_argv("score", _args(d), tmp_path)


# ── the token ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stage", ["compass", "coherence", "drift"])
def test_the_token_never_reaches_a_stage_that_cannot_parse_it(tmp_path, stage):
    argv = measure.stage_argv(stage, _args(_track(tmp_path), hf_token=FAKE_TOKEN), tmp_path)
    assert "--hf-token" not in argv, (
        f"{stage} does not declare --hf-token, and argparse prints an unrecognised argument "
        f"together with its value")
    assert FAKE_TOKEN not in argv


@pytest.mark.parametrize("stage", ["score", "capability"])
def test_the_token_still_reaches_the_stages_that_need_it(tmp_path, stage):
    argv = measure.stage_argv(stage, _args(_track(tmp_path), hf_token=FAKE_TOKEN), tmp_path)
    assert "--hf-token" in argv and FAKE_TOKEN in argv


def test_the_token_list_matches_the_parsers_that_actually_declare_the_flag():
    """The list is data, so this is what stops it drifting from the parsers it describes."""
    import importlib

    for stage, module in (("score", "score"), ("capability", "capability"),
                          ("compass", "margin"), ("coherence", "coherence"), ("drift", "drift")):
        src = Path(importlib.import_module(f"senbonzakura.{module}").__file__).read_text(
            encoding="utf-8")
        declares = "--hf-token" in src
        listed = stage in measure.TAKES_HF_TOKEN
        assert declares == listed, (
            f"{stage} {'declares' if declares else 'does not declare'} --hf-token but is "
            f"{'listed' if listed else 'not listed'} in TAKES_HF_TOKEN")
