# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The live display: a view of a search that is already happening, and never the search itself.

NAMED `livedisplay` AND NOT `panel`, deliberately. In this project a "panel" is a panel of judges:
`panel.py` holds the multi-instrument verdict logic, and the review panel in `CLAUDE.md` is a panel
of reviewers. A live terminal display sharing that word cost one module its contents for a few
minutes before the tests caught it. The user-facing flag stays `--no-panel`, because that is what
somebody looking at a box on their terminal will type.

WHAT THIS IS FOR

A default `senbonzakura <model>` run prints one line per trial and takes between twenty minutes and
two hours. Those lines are dense and correct and they are not a picture. Somebody watching a search
wants to know the best refusal rate so far, whether KL is climbing, how many trials are left and
whether the thing is still moving; reconstructing that from a scrolling log is work the tool should
be doing.

THE RULE THAT MATTERS MORE THAN THE PANEL

**Every marker the plain log emits is still emitted, unchanged, whether the panel is running or
not.** The panel is drawn beside the log, never instead of it. This is not tidiness. CI decides
whether a release is fit to ship by grepping this tool's output for nine specific needles, `DONE`
and `MARGIN_DONE` among them, and a panel that captured stdout and redrew it would turn every one
of those checks green-by-absence. A guard that cannot see the thing it guards reports clean.

So the panel subscribes to the structured event stream, which is a view of the work by the module's
own description, and touches the log not at all.

WHEN IT DOES NOT RUN, which is most of the time

  * `--no-panel` was given.
  * stdout is not a terminal, which covers every CI run, every pipe and every redirect to a file.
  * `rich` cannot be imported.
  * the terminal is too narrow to draw in.

In all four cases this module returns an object that does nothing at all, and the run is
byte-identical to a run from before the panel existed. The first two are decisions, the second two
are conditions, and none of them is an error.

ON THE DEPENDENCY

`rich` is declared, and it is already an install-time dependency of both `transformers` and
`accelerate`, so it adds nothing to the 5.9 GB. It is still imported defensively, because a floor
old enough to satisfy the cooldown rule is old enough that somebody's environment may not have it.
"""
from __future__ import annotations

import os
import sys

#: Below this many columns a two-panel layout is worse than the log it sits beside.
MIN_COLUMNS = 60


class NullPanel:
    """What every disabled path returns. Every method is a no-op that costs a call.

    A null object rather than `None` so the call sites carry no `if panel is not None` branches.
    Those branches are where a live-display feature usually breaks the thing it decorates.
    """

    active = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def event(self, _rec):
        pass

    def note(self, _text):
        pass


def _a_terminal_we_can_draw_in(stream):
    """Whether this stream is an interactive terminal wide enough for a panel.

    `isatty` is asked of the actual stream rather than inferred, and a missing or lying `isatty` is
    treated as "not a terminal", which is the safe direction: the cost of not drawing is a plainer
    log, and the cost of drawing into a pipe is corrupted output that CI greps.
    """
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    # A dumb terminal announces that it cannot do cursor movement. Honour it.
    if os.environ.get("TERM", "").lower() in ("dumb", ""):
        return False
    # Respected because a lot of tooling sets it and a live panel is exactly what it is about.
    if os.environ.get("NO_COLOR") or os.environ.get("CI"):
        return False
    # WIDTH IS ASKED OF THE STREAM FIRST AND THE ENVIRONMENT SECOND, and an unknown width does NOT
    # veto the panel. `isatty` is the guard that keeps a display out of a pipe; width only decides
    # whether a panel would look right. A terminal that reports no size at all is common enough
    # (a pty with no winsize set, some terminal multiplexers, some IDE consoles) that refusing on it
    # made the panel invisible on real terminals. Found by driving it through a pty, where the first
    # version of this function reported 0 columns and silently declined to draw.
    #
    # `shutil.get_terminal_size` reads COLUMNS, then asks the OS, then falls back to 80, which is
    # the same sequence `rich` itself would use.
    columns = 0
    try:
        columns = os.get_terminal_size(stream.fileno()).columns
    except (OSError, ValueError, AttributeError):
        pass
    if columns <= 0:
        import shutil
        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    return columns >= MIN_COLUMNS


def why_not(args, stream=None):
    """The reason a panel will not be drawn, or None if it will be. Separated so it can be tested.

    Returned as a sentence rather than a flag, because "the panel did not appear" is otherwise the
    kind of thing somebody files a bug about.
    """
    stream = stream or sys.stdout
    if getattr(args, "no_panel", False):
        return "--no-panel was given"
    if not _a_terminal_we_can_draw_in(stream):
        return "output is not an interactive terminal, so the plain log is used"
    try:
        import rich  # noqa: F401
    except ImportError:
        return "rich is not installed, so the plain log is used"
    return None


def attach(events, args, *, total_trials=None, log=None, stream=None):
    """A live panel over this run's event stream, or a `NullPanel`.

    `events` is an `EventLog`. The panel registers as an observer and unregisters on exit. Nothing
    it does can reach the search: `EventLog.emit` drops an observer that raises.
    """
    reason = why_not(args, stream=stream)
    if reason is not None:
        return NullPanel()
    try:
        return _RichPanel(events, total_trials=total_trials, log=log, stream=stream or sys.stdout)
    except Exception as e:
        # A panel that cannot be built is not a failed run. Said once, then forgotten.
        if log:
            log(f"  the live panel could not start ({type(e).__name__}: {e}); using the plain log")
        return NullPanel()


class _RichPanel:
    """The real thing. Constructed only when `why_not` returned None."""

    active = True

    def __init__(self, events, *, total_trials=None, log=None, stream=None):
        from rich.console import Console
        from rich.live import Live

        self._events = events
        self._log = log or (lambda _m: None)
        self._total = int(total_trials) if total_trials else None
        self._console = Console(file=stream or sys.stdout)
        # The best row so far, which is the only thing a watcher actually wants.
        self._best = None
        self._last = None
        self._trials = 0
        self._notes = []
        self._stop = None
        self._live = Live(self._render(), console=self._console,
                          refresh_per_second=4, transient=False)

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def __enter__(self):
        self._stop = self._events.observe(self.event)
        self._live.__enter__()
        return self

    def __exit__(self, *exc):
        # UNSUBSCRIBE FIRST. If the display teardown raises, an observer still holding a reference
        # to a closed display would then raise on every subsequent event for the rest of the run.
        if self._stop:
            self._stop()
            self._stop = None
        try:
            self._live.__exit__(*exc)
        except Exception:
            pass
        return False

    # ── input ───────────────────────────────────────────────────────────────────

    def event(self, rec):
        """One record from the stream. Only the kinds this panel draws are read."""
        if rec.get("kind") != "trial":
            return
        self._trials += 1
        self._last = rec
        if self._best is None or self._is_better(rec, self._best):
            self._best = rec
        self._live.update(self._render())

    def note(self, text):
        """A line worth keeping on the panel, e.g. a stage boundary. The log gets it too, elsewhere."""
        self._notes.append(str(text))
        del self._notes[:-3]
        self._live.update(self._render())

    @staticmethod
    def _is_better(a, b):
        """Lower objective wins, which is what the search itself minimises.

        Read from the event rather than recomputed. A panel that ranked trials by its own rule would
        eventually disagree with the run's own answer, and a reader would believe the panel.
        """
        try:
            return float(a.get("objective")) < float(b.get("objective"))
        except (TypeError, ValueError):
            return False

    # ── output ──────────────────────────────────────────────────────────────────

    def _render(self):
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column()

        done = f"{self._trials}" + (f" of {self._total}" if self._total else "")
        table.add_row("trials", done)
        for label, rec in (("best so far", self._best), ("latest", self._last)):
            if rec is None:
                table.add_row(label, "waiting for the first trial")
                continue
            table.add_row(label, self._describe(rec))
        for note in self._notes:
            table.add_row("", note)
        table.add_row("", "[dim]the full log is printing beside this; --no-panel turns this off[/dim]")
        return Panel(table, title="senbonzakura", border_style="cyan")

    @staticmethod
    def _describe(rec):
        """One trial as a sentence, with every rate as a percentage and KL as itself."""
        def pct(key):
            try:
                return f"{float(rec.get(key)) * 100:.1f}%"
            except (TypeError, ValueError):
                return "?"

        try:
            kl = f"{float(rec.get('kl')):.4f}"
        except (TypeError, ValueError):
            kl = "?"
        return (f"trial {rec.get('number', '?')}  refusals {pct('refusals')}  "
                f"soft {pct('soft')}  broken {pct('broken')}  KL {kl}")


def add_argument(parser):
    """`--no-panel`, defined here so the flag and the behaviour live together."""
    parser.add_argument("--no-panel", dest="no_panel", action="store_true",
                        help="never draw the live panel, even on a terminal. The panel is a view "
                             "beside the log and turning it off changes nothing else: the same "
                             "lines, and the same markers, are printed either way.")
