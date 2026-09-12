# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A missing dataset must cost nothing to discover, not a model download.

WHY THIS FILE EXISTS

`--track` defaults to the relative directory `track`. A run started anywhere that directory does
not exist, or with `HF_HUB_OFFLINE=1` and no flag, used to die inside `extract_directions` on
`could not fetch the Hub dataset 'track/bad_ds'`. That is AFTER `Abliterator.__init__` has pulled
and loaded the weights, so on a rented pod it is a 61 GB download and a full load thrown away for
a mistake that was visible from the command line before anything started.

Reported by hephaestus-67 on 2026-09-05 after hitting it on real hardware, and it sat in the
ledger for a day with the diagnosis written and no fix, which is the failure the ledger exists to
prevent rather than to record.

WHAT IS ACTUALLY UNDER TEST

Not that a missing dataset raises: it always did. That it raises EARLY, that it names every fault
rather than the first, and that it points at the bundled track, which is the fix for the common
case and needs no network.
"""
import types

import pytest
from artefacts import needs_track

from senbonzakura import cli, lengthsweep


def _args(**over):
    a = dict(track="track", good_ds=None, hedge_ds="", clean_ds="", harmless_matched="",
             # Sound by default: these tests are about the ORDER pre-flights run in, so
             # none should trip the generation-budget gate before reaching its subject.
             gen_tokens=lengthsweep.DEFAULT_BUDGET, short_budget_ok=False,
             text_column=None, hf_token=None)
    a.update(over)
    return types.SimpleNamespace(**a)


def test_a_missing_track_is_refused():
    with pytest.raises(SystemExit, match="cannot be read"):
        cli._preflight_datasets(_args(track="definitely-not-a-track"))


def test_every_fault_is_reported_not_just_the_first():
    """THE POINT OF A PRE-FLIGHT ON RENTED HARDWARE.

    Stopping at the first fault turns one wasted start into three. All three required datasets are
    missing here and all three must be named.
    """
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(_args(track="definitely-not-a-track"))
    msg = str(e.value)
    for part in ("bad_ds", "good_ds", "bad_eval_ds"):
        assert f"definitely-not-a-track/{part}" in msg, f"{part} was not named"
    assert msg.startswith("3 of the datasets")


def test_each_fault_says_what_the_dataset_is_for():
    """A path alone does not tell a reader which of their flags was wrong."""
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(_args(track="definitely-not-a-track"))
    msg = str(e.value)
    assert "directions are extracted from" in msg
    assert "refusal is scored on" in msg


def test_the_bundled_track_is_offered_when_the_track_is_the_problem():
    """The fix for the common case is one flag, and it needs neither network nor download."""
    with pytest.raises(SystemExit, match=r"--track default"):
        cli._preflight_datasets(_args(track="definitely-not-a-track"))


@needs_track
def test_the_bundled_track_hint_is_absent_when_the_track_was_fine():
    """Advice that does not apply is noise, and noise in an error is how errors stop being read.

    Here the track resolves and only an explicitly supplied flag is broken, so pointing at
    `--track default` would send the reader to fix the one thing that was not wrong.
    """
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(_args(track="default", hedge_ds="definitely-not-a-dataset"))
    msg = str(e.value)
    assert "--hedge-ds" in msg
    assert "--track default" not in msg


@needs_track
def test_the_bundled_track_passes():
    """The happy path, and it is load-bearing.

    A pre-flight that rejected the shipped corpus would make the wheel unusable offline, which is
    the situation it was written for.
    """
    cli._preflight_datasets(_args(track="default"))


@needs_track
def test_an_optional_dataset_is_only_checked_when_it_was_given():
    """Empty means "not asked for", and must not be reported as a missing dataset."""
    cli._preflight_datasets(_args(track="default", hedge_ds="", clean_ds="",
                                  harmless_matched=""))


def test_a_supplied_optional_dataset_is_checked():
    with pytest.raises(SystemExit, match="definitely-not-a-dataset"):
        cli._preflight_datasets(_args(track="default",
                                      harmless_matched="definitely-not-a-dataset"))


def test_an_overridden_good_ds_is_checked_instead_of_the_track_one():
    """--good-ds replaces the track's harmless set, so it is the one that has to exist."""
    with pytest.raises(SystemExit) as e:
        cli._preflight_datasets(_args(track="default", good_ds="definitely-not-a-dataset"))
    msg = str(e.value)
    assert "definitely-not-a-dataset" in msg
    assert "default/good_ds" not in msg, "the replaced dataset must not also be demanded"


# ── the ordering, which is the whole defect ──────────────────────────────────────────

def test_the_preflight_runs_before_the_model_is_constructed(monkeypatch):
    """THE REGRESSION THIS FILE IS NAMED FOR.

    A check that runs in the right place and a check that runs in the wrong place both raise, and
    only one of them saves a download. This asserts the ORDER by making the model constructor
    explode: if the pre-flight moved back below it, this test would see that explosion instead of
    the dataset error and fail.
    """
    built = []

    def _never(*_a, **_k):
        built.append(True)
        raise AssertionError("the model was constructed before the datasets were checked")

    monkeypatch.setattr(cli, "Abliterator", _never)
    args = _args(track="definitely-not-a-track")
    args.load_in_4bit = False
    with pytest.raises(SystemExit, match="cannot be read"):
        cli.run_parsed(args, None, [])
    assert not built, "the pre-flight did not run before the model"


def test_an_unusable_torch_is_reported_before_a_missing_dataset(monkeypatch):
    """Both are pre-flights, and the order between them is a deliberate choice.

    The torch check is instant and touches nothing; this one may reach the network for a Hub
    track. More importantly an interpreter that cannot run the model makes every other fault
    moot, so sending the operator to rebuild a corpus first would be sending them to fix the
    thing that was not going to help. Both still land long before the weights.
    """
    monkeypatch.setattr(cli, "torch_version_ok", lambda _v: False)
    args = _args(track="definitely-not-a-track")
    args.load_in_4bit = False
    with pytest.raises(SystemExit, match="senbonzakura needs torch"):
        cli.run_parsed(args, None, [])
