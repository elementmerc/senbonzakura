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
