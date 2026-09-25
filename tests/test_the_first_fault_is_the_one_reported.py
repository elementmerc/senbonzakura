# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A command line with two faults is told about the one it can act on first.

WHY THIS FILE EXISTS, AND IT HAS NOW HAPPENED TWICE IN THE SAME FUNCTION

`run_parsed` runs its pre-flights in a deliberate order: everything decidable from the command
line alone comes before anything that reads a track, touches the Hub or loads a model. A check
that jumps that queue reports the second thing wrong with somebody's command instead of the
first, and the first is usually the one they typed.

It happened to the slow-probe guard on 2026-09-22: it announced a four-hour capability probe to
somebody whose track did not exist. It happened to `resolve_track` on 2026-09-23, and that one
reached CI: eight tests failed at once, each asserting some other refusal and receiving "no
evaluation track to measure against", because a runner has neither a `./track` directory nor a
bundled track while both development machines have both. The local suite was green.

So the order is asserted here rather than left to a comment. Each test gives a command line with
TWO faults, one of them track-shaped, and requires the track-independent one to be the one
reported.
"""
from __future__ import annotations

import types

import pytest

from senbonzakura import cli, lengthsweep


def _args(**over):
    """A namespace with NO track available, which is the state a fresh runner is in."""
    a = dict(model="x", model_positional=None, track=None,
             method="searched", max_directions=1,
             good_ds=None, hedge_ds="", clean_ds="", harmless_matched="", text_column=None,
             hf_token=None, load_in_4bit=False, matched_scoring=False,
             gen_tokens=lengthsweep.DEFAULT_BUDGET, short_budget_ok=False,
             out="abliterated", capability_n=0, capability_eval="", device="cuda")
    a.update(over)
    return types.SimpleNamespace(**a)


# ── resolution, with the branches actually driven ────────────────────────────────
#
# THE FIRST VERSION OF THESE PASSED FOR THE WRONG REASON. They set `track=None`, and
# `resolve_track` returns immediately for anything that is not the `auto` sentinel, so the
# resolution logic was never reached and the assertion held on a machine that does have a
# bundled track. Every branch below is now driven with the sentinel and with the two sources
# controlled, so this file can reproduce a runner's state on a machine that is not one.

def _auto(**over):
    from senbonzakura.parser import TRACK_AUTO

    return _args(track=TRACK_AUTO, **over)


def _no_sources(monkeypatch, tmp_path):
    """A machine with neither a local track nor a bundled one, which is every CI runner."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("senbonzakura.bundled.is_available", lambda: False)


def test_an_explicit_track_is_returned_untouched(tmp_path, monkeypatch):
    _no_sources(monkeypatch, tmp_path)
    args = _args(track="mytrack")
    assert cli.resolve_track(args, log=lambda _m: None) == "mytrack"


def test_a_local_track_directory_wins(tmp_path, monkeypatch):
    """`./track` is what `senbonzakura track` builds into, so somebody who built one means it."""
    _no_sources(monkeypatch, tmp_path)
    (tmp_path / "track").mkdir()
    said = []
    args = _auto()
    assert cli.resolve_track(args, log=said.append) == "track"
    assert any("./track" in line for line in said), "the choice has to be in the log"


def test_the_bundled_track_is_the_fallback(tmp_path, monkeypatch):
    _no_sources(monkeypatch, tmp_path)
    monkeypatch.setattr("senbonzakura.bundled.is_available", lambda: True)
    said = []
    args = _auto()
    assert cli.resolve_track(args, log=said.append)
    assert any("bundled" in line for line in said), (
        "a run measured on the bundled track must say so, with its licence position")


def test_resolution_returns_none_when_the_machine_has_neither(tmp_path, monkeypatch):
    """A runner's state, reproduced on a machine that is not a runner. This is the case the
    whole fix is about: resolution reports that it found nothing, and does not refuse.
    """
    _no_sources(monkeypatch, tmp_path)
    args = _auto()
    assert cli.resolve_track(args, log=lambda _m: None) is None
    assert args.track is None


def test_the_refusal_still_exists_and_still_says_what_to_do():
    """Deferring it must not lose it. A run with no track cannot proceed, and the message names
    both routes to getting one.
    """
    with pytest.raises(SystemExit) as e:
        cli.refuse_without_a_track(_args())
    msg = str(e.value)
    assert "senbonzakura track build" in msg, "the message must name the command that builds one"
    assert "--track" in msg, "and the flag for a track you already have"


def test_a_resolved_track_is_not_refused():
    cli.refuse_without_a_track(_args(track="default"))


def test_the_four_bit_rejection_beats_the_missing_track():
    """`--load-in-4bit` is wrong whatever track you have, and it is what the person typed."""
    with pytest.raises(SystemExit, match="full precision"):
        cli.run_parsed(_args(load_in_4bit=True), None, [])


def test_the_unused_matched_corpus_beats_the_missing_track():
    with pytest.raises(SystemExit, match="--matched-scoring"):
        cli.run_parsed(_args(harmless_matched="some/corpus"), None, [])


def test_the_missing_model_beats_the_missing_track():
    """Two things absent, and the model is the one the short form is about."""
    with pytest.raises(SystemExit, match="no model given"):
        cli.run_parsed(_args(model=None), None, [])


def test_the_refusal_is_ordered_after_the_output_check_in_the_source():
    """The occupied-output check needs no track either, and its own test asserts its message.

    Read off the source rather than driven, because driving it needs a directory holding a
    previous run and this is an ordering claim, not a behavioural one.
    """
    import inspect

    body = inspect.getsource(cli.run_parsed)
    assert body.index("_preflight_output(args)") < body.index("refuse_without_a_track(args)"), (
        "the track refusal runs before the occupied-output check, so somebody whose output "
        "directory already holds a run is told about tracks instead")
