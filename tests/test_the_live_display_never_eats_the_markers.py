# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The live panel is drawn beside the log. It never captures, wraps or replaces a line of it.

WHY THIS FILE IS THE MOST IMPORTANT PART OF THE PANEL

`tools/ci/smoke_end_to_end.py` decides whether a build is fit to ship by running the real commands
and grepping their output for specific needles: `DONE`, `MARGIN_DONE`, `SCORE_DONE`, `DRIFT_DONE`
and five phrases. Its own header says why it works that way, because a stage once printed a success
line unconditionally.

A live display is the classic way to break exactly that. Take stdout, redraw it, and every one of
those greps stops finding its needle. The job then passes, because an absent marker in a job that
greps for markers is indistinguishable from a marker that was never required. A guard that cannot
see the thing it guards reports clean, and this project has hit that shape often enough to name it.

So the panel subscribes to the structured event stream and writes to the terminal. It is off in
every non-interactive context, which is every CI run, every pipe and every redirect, and this file
holds the assertions that keep it that way.

THE NEEDLES ARE READ OUT OF THE SMOKE HARNESS, not copied here. A copied list is a second source of
truth that goes stale on the day somebody adds a tenth check, and the copy would still pass.
"""
from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pytest

from senbonzakura import events, livedisplay

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "tools" / "ci" / "smoke_end_to_end.py"


def _needles():
    """Every marker and phrase the smoke harness requires, read from the harness itself."""
    if not SMOKE.is_file():
        pytest.skip(f"{SMOKE} is not in this checkout")
    text = SMOKE.read_text(encoding="utf-8")
    found = set(re.findall(r'expect_(?:marker|text)=["\']([^"\']+)["\']', text))
    assert found, "no expectations found in the smoke harness, so this test would assert nothing"
    return sorted(found)


def _args(**kw):
    kw.setdefault("no_panel", False)
    return argparse.Namespace(**kw)


class _NotATerminal(io.StringIO):
    def isatty(self):
        return False


class _ATerminal(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        raise OSError("a StringIO has no file descriptor")


# ── the panel is off wherever output is not a terminal ───────────────────────────

def test_the_panel_is_off_when_output_is_piped():
    """The CI case, the `| tee` case and the `> log.txt` case, which are the same case."""
    reason = livedisplay.why_not(_args(), stream=_NotATerminal())
    assert reason is not None
    assert "terminal" in reason


def test_a_piped_run_gets_a_null_panel_that_does_nothing():
    log = events.EventLog(None)
    p = livedisplay.attach(log, _args(), total_trials=10, stream=_NotATerminal())
    assert isinstance(p, livedisplay.NullPanel)
    assert p.active is False
    with p:
        p.event({"kind": "trial", "number": 1, "objective": 0.5})
        p.note("something")


def test_no_panel_wins_even_on_a_terminal():
    assert "--no-panel" in livedisplay.why_not(_args(no_panel=True), stream=_ATerminal())


def test_a_terminal_of_unknown_width_still_draws(monkeypatch):
    """An unknown width must not veto the livedisplay. `isatty` is the guard, not the size.

    WHAT PROMPTED THIS, and it is the reason the panel was driven through a pty before being
    believed: the first version refused whenever `os.get_terminal_size` failed or returned 0. A pty
    with no window size set reports 0 columns, as do some multiplexers and some IDE consoles, so the
    panel was invisible on real terminals while every unit test passed. Width decides whether a
    panel would look right; `isatty` decides whether drawing is safe at all, and only the second one
    protects the markers.
    """
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("COLUMNS", "100")
    assert livedisplay.why_not(_args(), stream=_ATerminal()) is None


def test_a_terminal_too_narrow_to_draw_in_is_declined(monkeypatch):
    """Below the minimum a panel is worse than the log it would sit beside."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("COLUMNS", str(livedisplay.MIN_COLUMNS - 10))
    assert livedisplay.why_not(_args(), stream=_ATerminal()) is not None


@pytest.mark.parametrize("env", ["CI", "NO_COLOR"])
def test_a_declared_non_interactive_environment_turns_it_off(monkeypatch, env):
    monkeypatch.setenv(env, "1")
    monkeypatch.setenv("TERM", "xterm")
    assert livedisplay.why_not(_args(), stream=_ATerminal()) is not None


def test_a_dumb_terminal_turns_it_off(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert livedisplay.why_not(_args(), stream=_ATerminal()) is not None


# ── the markers themselves ──────────────────────────────────────────────────────

@pytest.mark.parametrize("needle", _needles())
def test_every_smoke_needle_still_reaches_a_piped_stream(needle, capsys):
    """A log line carrying a needle is printed unchanged while a null panel is held open.

    This is the property the smoke harness depends on, asserted against the real needle list.
    """
    log = events.EventLog(None)
    with livedisplay.attach(log, _args(), stream=_NotATerminal()):
        print(f"some output with {needle} in it")
    assert needle in capsys.readouterr().out


def test_the_panel_module_does_not_touch_stdout_at_import():
    """Importing it must not reconfigure, wrap or redirect anything.

    A module that installs a display at import time is a module that changes every program that
    imports it, including the ones that only wanted to read `why_not`.
    """
    source = (ROOT / "src" / "senbonzakura" / "livedisplay.py").read_text(encoding="utf-8")
    body = [ln for ln in source.splitlines()
            if ln and not ln.startswith((" ", "\t", "#", '"', "'"))]
    for forbidden in ("sys.stdout =", "sys.stderr =", "logging.basicConfig",
                      "install(", "reconfigure("):
        assert not any(forbidden in ln for ln in body), (
            f"{forbidden!r} at module level would change every importer's output")


# ── the panel cannot take a run down ────────────────────────────────────────────

def test_an_observer_that_raises_is_dropped_and_the_run_continues():
    """The rule `EventLog` already applies to its file sink, now applied to views of the stream.

    A panel that throws on a resized terminal must not end an abliteration that is twenty minutes in
    and holding a model.
    """
    said = []
    log = events.EventLog(None, log=said.append)

    def hostile(_rec):
        raise RuntimeError("the terminal went away")

    log.observe(hostile)
    log.emit("trial", number=1)          # must not raise
    log.emit("trial", number=2)          # and must not raise a second time
    assert any("dropped" in m for m in said), "dropping an observer has to be reported once"
    assert len(said) == 1, "it must be reported once, not once per event for the rest of the run"


def test_observing_works_without_any_json_sink():
    """A panel is wanted on ordinary runs, which write no JSONL at all.

    `enabled` is about the file sink. Reading it as "is anybody watching" would mean the panel only
    worked on runs that also passed --json-events, which is nearly none of them.
    """
    seen = []
    log = events.EventLog(None)
    assert not log.enabled
    log.observe(seen.append)
    log.emit("trial", number=7, objective=0.25)
    assert [r["number"] for r in seen] == [7]


def test_unobserving_stops_delivery():
    seen = []
    log = events.EventLog(None)
    stop = log.observe(seen.append)
    log.emit("trial", number=1)
    stop()
    log.emit("trial", number=2)
    assert [r["number"] for r in seen] == [1]


def test_the_envelope_still_cannot_be_overwritten_by_a_call_site():
    """The existing guarantee, re-asserted now that observers see the record too."""
    seen = []
    log = events.EventLog(None)
    log.observe(seen.append)
    log.emit("trial", kind="lies", seq=999, number=3)
    assert seen[0]["kind"] == "trial"
    assert seen[0]["seq"] == 1


# ── the panel itself, driven without a terminal ───────────────────────────────────────────────────
#
# WHY THIS SECTION EXISTS, 2026-09-27
#
# Everything above checks that the display stays OUT of the way, which is the property that protects
# CI. None of it exercised the panel, because the panel needs a terminal, so `livedisplay.py` sat at
# 43% coverage: 74 of 131 statements, the whole `_RichPanel` class. That took the project's total
# from 94.06% to 93.98% against a 94% floor, and CI went red on a two-hundredth of a percent with
# every one of 5,802 tests passing.
#
# The floor was not the problem. Shipping a class nobody had automated a single line of was, and the
# only reason it was known to work at all is that it had been driven by hand through a pty once.
#
# `rich` will render into any file object if told to force terminal mode, so none of this needs a
# pty: the point is to exercise the rendering, the event handling and the teardown.

class _Recording(io.StringIO):
    """A stream rich will draw into, and which reports itself as a terminal."""

    def isatty(self):
        return True


def _panel(total=3, stream=None):
    """A panel rendering into a string, with a forced width so the output is machine-independent."""
    from rich.console import Console

    from senbonzakura.livedisplay import _RichPanel
    log = events.EventLog(None)
    sink = stream or _Recording()
    console = Console(file=sink, force_terminal=True, width=100, legacy_windows=False)
    p = _RichPanel(log, total_trials=total, log=lambda _m: None, console=console)
    p._sink = sink
    return log, p


def _trial(number, refusals, kl, objective=None):
    return {"kind": "trial", "number": number, "refusals": refusals, "soft": 0.05,
            "broken": 0.0, "kl": kl, "objective": objective if objective is not None else refusals + kl}


def test_the_panel_draws_a_box_with_its_title():
    log, p = _panel()
    with p:
        log.emit("trial", **{k: v for k, v in _trial(0, 0.58, 0.02).items() if k != "kind"})
    drawn = p._sink.getvalue()
    assert "senbonzakura" in drawn
    assert any(glyph in drawn for glyph in "─━╭│"), "no box was drawn at all"


def test_the_panel_shows_the_trial_count_against_the_budget():
    log, p = _panel(total=7)
    with p:
        for i in range(3):
            log.emit("trial", **{k: v for k, v in _trial(i, 0.5, 0.1).items() if k != "kind"})
    assert "3 of 7" in p._sink.getvalue()


def test_the_best_row_tracks_the_lowest_objective_not_the_latest():
    """Read from the event rather than recomputed, so the panel cannot disagree with the search."""
    log, p = _panel()
    with p:
        log.emit("trial", number=0, refusals=0.58, soft=0.05, broken=0.0, kl=0.02, objective=0.60)
        log.emit("trial", number=1, refusals=0.31, soft=0.05, broken=0.0, kl=0.11, objective=0.42)
        log.emit("trial", number=2, refusals=0.28, soft=0.05, broken=0.0, kl=0.20, objective=0.48)
    assert p._best["number"] == 1, "the best row is not the lowest objective"
    assert p._last["number"] == 2, "the latest row is not the most recent trial"


def test_a_trial_with_an_unusable_objective_does_not_become_the_best():
    """A malformed event must not win, and must not raise either."""
    log, p = _panel()
    with p:
        log.emit("trial", number=0, refusals=0.5, soft=0.0, broken=0.0, kl=0.1, objective=0.6)
        log.emit("trial", number=1, refusals=0.1, soft=0.0, broken=0.0, kl=0.1, objective=None)
    assert p._best["number"] == 0


def test_rates_are_shown_as_percentages_and_kl_as_itself():
    log, p = _panel()
    with p:
        log.emit("trial", number=4, refusals=0.3125, soft=0.05, broken=0.0, kl=0.1234, objective=0.4)
    drawn = " ".join(p._sink.getvalue().split())
    assert "31.2%" in drawn, "a refusal rate is not shown as a percentage"
    assert "0.1234" in drawn, "KL is not shown in its own units"


def test_a_field_that_is_missing_or_unparseable_renders_as_a_question_mark():
    """A panel must never be the reason a twenty-minute run dies."""
    log, p = _panel()
    with p:
        log.emit("trial", number=1, refusals="not a number", kl=None)
    drawn = p._sink.getvalue()
    assert "?" in drawn


def test_events_that_are_not_trials_are_ignored():
    log, p = _panel()
    with p:
        log.emit("stage", name="bake")
        log.emit("saved", path="x")
    assert p._trials == 0
    assert "waiting for the first trial" in p._sink.getvalue()


def test_a_note_is_shown_and_only_the_last_few_are_kept():
    log, p = _panel()
    with p:
        for i in range(6):
            p.note(f"note number {i}")
    drawn = p._sink.getvalue()
    assert "note number 5" in drawn, "the most recent note is not shown"
    assert len(p._notes) <= 3, "notes accumulate without bound, so the panel grows all run"


def test_the_panel_unsubscribes_on_exit_so_a_later_event_cannot_reach_a_closed_display():
    """Unsubscribing BEFORE teardown is deliberate: see the comment in `__exit__`."""
    log, p = _panel()
    with p:
        log.emit("trial", number=0, refusals=0.5, soft=0.0, broken=0.0, kl=0.1, objective=0.6)
    before = p._trials
    log.emit("trial", number=1, refusals=0.4, soft=0.0, broken=0.0, kl=0.1, objective=0.5)
    assert p._trials == before, "the panel was still receiving events after it closed"


def test_attach_returns_a_real_panel_when_the_stream_looks_like_a_terminal(monkeypatch):
    """The success path of `attach`, which nothing reached before."""
    from senbonzakura import livedisplay
    monkeypatch.setattr(livedisplay, "_a_terminal_we_can_draw_in", lambda _s: True)
    log = events.EventLog(None)
    p = livedisplay.attach(log, _args(), total_trials=2, stream=_Recording())
    assert p.active is True
    with p:
        log.emit("trial", number=0, refusals=0.5, soft=0.0, broken=0.0, kl=0.1, objective=0.6)


def test_attach_degrades_to_the_null_panel_when_construction_fails(monkeypatch):
    """A panel that cannot be built is not a failed run, and the reason is said once."""
    from senbonzakura import livedisplay
    monkeypatch.setattr(livedisplay, "_a_terminal_we_can_draw_in", lambda _s: True)

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no terminal after all")

    monkeypatch.setattr(livedisplay, "_RichPanel", _Boom)
    said = []
    p = livedisplay.attach(events.EventLog(None), _args(), log=said.append, stream=_Recording())
    assert isinstance(p, livedisplay.NullPanel)
    assert any("could not start" in m for m in said), "the fallback happened silently"
