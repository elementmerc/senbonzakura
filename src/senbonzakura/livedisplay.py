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
  * the terminal is narrower than `MIN_COLUMNS`, which is a floor and not a preference: below it
    there is no room for a label and its value on one row. A merely narrow terminal still gets a
    panel; it gets the one-column form instead.

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
import time

#: THE FLOOR, not the breakpoint. Four columns go to the frame and its padding, and a label and
#: its value need the rest, so below this there is genuinely nowhere to put a row and the plain log
#: is the better answer. It used to be 60, which is the design's FIRST BREAKPOINT rather than a
#: floor, so every terminal under 60 columns got no panel at all instead of the one-column form the
#: design asks for. The two are separate numbers because they answer separate questions: "can this
#: be drawn" and "how should it be drawn".
MIN_COLUMNS = 24

#: The design's first breakpoint: under this many columns the panel drops to one column and no
#: chart. A stat grid of four side-by-side cells needs about sixty columns before the values start
#: wrapping into each other, and a frontier plot narrower than that is a smudge with a scale on it.
ONE_COLUMN_BELOW = 60

#: The kit's own ink, read from `assets/brand/mark.svg` rather than chosen here. The mark runs pink
#: through purple on a navy ground, and these are three of its five stops.
#:
#: IT USED TO BE `border_style="cyan"`, a colour that appears in no palette this project owns: not
#: in the mark, not in `banner.py`'s 256-colour table, not in the docs site's CSS. A run on a
#: terminal draws a banner and then this panel, so two unrelated colour systems were on screen at
#: once. A terminal picking its own pink would be a third identity, which is the mistake the first
#: brand build already made with a DejaVu tagline.
#:
#: Hex rather than a 256-colour index because rich degrades hex to the nearest available colour on
#: a terminal that cannot show it, and picking the index by hand is us doing that job worse.
BRAND_PINK = "#F27FA6"
BRAND_DEEP_PINK = "#E06A9C"
BRAND_PURPLE = "#8B4791"

#: BRAILLE, CHOSEN BY THE OPERATOR ON 2026-09-28 FROM NINE RENDERED OPTIONS, and the choice turned
#: on width rather than taste. Every one of these cells is East Asian width "N", so each frame is
#: exactly one column and the thing beside it never moves. Two candidates that looked fine in a
#: listing, the block bar and the growing dots, are width "A" (ambiguous): a terminal may draw those
#: double, and a spinner that changes width shifts the border a column every frame, which reads as
#: the terminal misbehaving rather than as the run being alive.
#:
#: Ten frames rather than four, so it reads as turning rather than flicking between states.
#: `test_the_dashboard_says_what_it_measured` asserts the single-column property and that the
#: header's length does not change as it turns, so a later edit cannot quietly bring back one that
#: jitters.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: How many trial marks the frontier keeps. A long search is thousands of trials and the plot has a
#: few hundred cells, so past this the oldest are dropped: the recent marks are the ones that say
#: whether the search is still finding anything, and they are the ones a reader is looking at.
MAX_POINTS = 400


def _about(seconds):
    """A hedged estimate, rounded to the unit it deserves.

    `about 16m 45s` is the hedge without the honesty: the seconds are four significant figures on
    an extrapolation from a per-trial mean, over a search whose trials genuinely differ in cost. So
    anything over a minute is said to the minute, which is the precision the estimate has.
    """
    if seconds is None:
        return "not yet"
    if seconds < 60:
        return "under a minute"
    # FORMATTED HERE RATHER THAN THROUGH `_duration`, which always prints its seconds: rounding to
    # the minute and then rendering "17m 00s" puts the false precision straight back, with a zero
    # that looks measured.
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes}m"
    return f"about {minutes // 60}h {minutes % 60:02d}m"


def _duration(seconds):
    """A duration the way the rest of the tool already spells it.

    Delegated rather than reimplemented: `resources.fmt_duration` is the project's one answer to
    "how long is that in words", and a panel with its own would be a second spelling of the same
    number on the same screen as the log. Imported lazily because this module is imported on paths
    that have no reason to pull `resources` in.
    """
    from .resources import fmt_duration
    return fmt_duration(seconds)


# ── what the card is doing, measured or left blank ───────────────────────────────

#: What the panel says where temperature and power would be, when the bindings will not import.
#: The DISTRIBUTION is named, not the module: `pip install pynvml` fetches a third-party wrapper
#: rather than the bindings `pyproject.toml` declares, so naming what the code imports would send
#: somebody to the wrong package. A declared dependency can still be missing, on an environment
#: where the wheel never landed, so this sentence is reachable on a normal install.
NO_TELEMETRY = "temp and power need nvidia-ml-py, which did not import: pip install nvidia-ml-py"


def card_telemetry(device):
    """(used_bytes, total_bytes, temperature_c, power_w, reason) for `device`.

    Any of the four figures may be None, and `reason` is a sentence saying why the last two are
    missing, or None when they are not missing.

    NOTHING HERE IS ESTIMATED. Scene 8 draws a VRAM bar, a temperature and a power figure, and the
    first two thirds of that are already measurable: `resources.cuda_free_total` reads the card.
    Temperature and power need `pynvml`, and where it will not import they come back None and the
    panel leaves the space empty rather than filling it with a plausible number, because a
    dashboard that guesses is the exact thing this project keeps having to withdraw figures over.

    THE BINDINGS ARE A BASE DEPENDENCY, so the empty case is now the unusual one. They were an
    undeclared import until 2026-09-28: read here, named in no packaging file, and therefore blank
    on every install that had ever shipped. The operator's fix was to declare them in the base
    install rather than behind an extra, on the grounds that the install story is one `pip install`
    and one `senbonzakura setup`, and that fifty kilobytes is not worth a decision a user has to
    make. This reverses decision Q-42 D3.

    THE REASON IS RETURNED BECAUSE A BLANK ON ITS OWN IS NOT HONEST, IT IS ONLY QUIET, and declared
    is not importable: an old driver or an environment the wheel never reached still lands here.
    Fail loud, never silent (baseline Section 2.1): the figure is still withheld, and the
    withholding now says what it would take to have it.

    `pynvml` is imported inside the call and a read that fails costs the reading, never the run. A
    telemetry read is decoration; it may not be the reason an abliteration stops.
    """
    used = total = temp = power = None
    try:
        from .resources import cuda_free_total
        pair = cuda_free_total(str(device))
        if pair:
            free, total = pair
            used = total - free
    except Exception:
        pass

    try:
        import pynvml
    except Exception:
        return used, total, temp, power, NO_TELEMETRY

    try:
        pynvml.nvmlInit()
        try:
            index = 0
            text = str(device)
            if ":" in text:
                index = int(text.split(":", 1)[1])
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        # NAMED, NOT SWALLOWED. The bindings are present and the card still would not answer, which
        # is a different situation from not having them and wants a different sentence.
        return used, total, None, None, f"the card would not report temp and power ({type(e).__name__})"
    return used, total, temp, power, None


def bar(used, total, width=18):
    """A proportion as a bar, or an empty frame when there is nothing to measure.

    An empty frame rather than no row at all, because a row that appears and disappears between
    redraws makes the panel jump, and a reader reads the jump as something happening.
    """
    if not total or used is None or used < 0:
        return "░" * width
    filled = max(0, min(width, round(width * (used / total))))
    return "▓" * filled + "░" * (width - filled)


# ── the frontier ─────────────────────────────────────────────────────────────────

def frontier(points, best=None, best_label=None, width=72, height=5):
    """The search as a plot with axes: drift across, refusal up, one mark per trial.

    THE ONE PICTURE THAT SAYS WHETHER THE SEARCH IS WORKING, and until this existed it was only
    inferable by reading per-trial log lines and holding them in your head. Down and left is better
    on both axes, so a run that is working walks its marks towards the origin and a stuck one fills
    a corner.

    DRAWN WITH AXES BECAUSE A BARE SCATTER IS NOT A PLOT. The first version of this put dots on an
    unlabelled grid, which tells a reader the shape of the cloud and nothing about its size: two
    runs an order of magnitude apart in drift looked identical. The scale is the half that makes it
    readable, so the tick labels are part of the picture rather than decoration on it.

    Scaled to the DATA rather than to a fixed range, because the interesting spread differs per
    model and a fixed axis would put every mark in one cell on half of them.

    Rendered as text: it lives inside a panel redrawn several times a second on a terminal that may
    be 60 columns wide, and no plotting library does that better than arithmetic does. Returns a
    list of rows, empty when there is nothing yet to draw.
    """
    pts = [(x, y) for x, y in points if x is not None and y is not None]
    if not pts:
        return []
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    # One distinct value on an axis has no spread to scale to, so it sits mid-cell rather than
    # dividing by zero.
    x_span = (x_hi - x_lo) or 1.0
    y_span = (y_hi - y_lo) or 1.0

    gutter = 9                       # "    0.20 " before the tick column
    # ROOM FOR THE ANNOTATION IS RESERVED, NOT HOPED FOR. The arrow sits on the same row as the
    # point it names, so a plot drawn to the full width pushes it onto a line of its own, where it
    # names nothing. The plot gives the label its space instead.
    reserved = (len(best_label) + 5) if best_label else 0
    plot_w = max(10, width - gutter - 1 - reserved)
    rows = max(3, height)
    grid = [[" "] * plot_w for _ in range(rows)]

    def cell(x, y):
        col = min(plot_w - 1, max(0, round((x - x_lo) / x_span * (plot_w - 1))))
        # Row 0 is the TOP and refusal rising should rise, so the axis is inverted here.
        row = rows - 1 - min(rows - 1, max(0, round((y - y_lo) / y_span * (rows - 1))))
        return row, col

    for x, y in pts:
        r, c = cell(x, y)
        grid[r][c] = "·"
    best_rc = None
    if best is not None and best[0] is not None and best[1] is not None:
        best_rc = cell(*best)
        grid[best_rc[0]][best_rc[1]] = "●"

    out = []
    for i, row in enumerate(grid):
        bottom = i == rows - 1
        # The bottom row IS the x axis, so the gaps between marks are drawn as the axis line.
        body = "".join(ch if ch != " " else ("─" if bottom else " ") for ch in row)
        if i == 0:
            label, tick = f"{y_hi:.2f}", "┤"
        elif bottom:
            label, tick = f"{y_lo:.2f}", "┼"
        elif i == rows // 2:
            label, tick = f"{(y_lo + y_hi) / 2:.2f}", "┤"
        else:
            label, tick = "", "│"
        line = f"{label:>{gutter - 1}} {tick}{body}".rstrip()
        # THE BEST POINT SAYS SO, on its own row, because a reader looking at a cloud of identical
        # dots cannot otherwise tell which one the run will actually use.
        if best_rc and i == best_rc[0] and best_label:
            line = f"{line}  ← {best_label}"
        out.append(line)

    # The x scale, under the axis it belongs to. Four ticks spread across the plot.
    ticks = [""] * plot_w
    for n in range(4):
        value = x_lo + (x_span * n / 3.0)
        col = min(plot_w - 1, round((value - x_lo) / x_span * (plot_w - 1)))
        ticks[col] = f"{value:.2f}"
    scale = ""
    for col, text in enumerate(ticks):
        if not text:
            continue
        if len(scale) < col:
            scale += " " * (col - len(scale))
        scale += text
    out.append(" " * gutter + scale.rstrip())
    return out


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


def _cannot_draw_because(stream):
    """Which of the conditions in `_a_terminal_we_can_draw_in` actually failed, as a sentence."""
    try:
        interactive_stream = bool(stream.isatty())
    except (AttributeError, ValueError):
        interactive_stream = False
    if not interactive_stream:
        return "output is not an interactive terminal, so the plain log is used"
    if os.environ.get("TERM", "").lower() in ("dumb", ""):
        return "TERM says this terminal cannot move the cursor, so the plain log is used"
    for name in ("NO_COLOR", "CI"):
        if os.environ.get(name):
            return f"{name} is set, so the plain log is used"
    return (f"this terminal is narrower than {MIN_COLUMNS} columns, which is not enough for a "
            f"label and a number, so the plain log is used")


def why_not(args, stream=None):
    """The reason a panel will not be drawn, or None if it will be. Separated so it can be tested.

    Returned as a sentence rather than a flag, because "the panel did not appear" is otherwise the
    kind of thing somebody files a bug about.
    """
    stream = stream or sys.stdout
    if chosen_layout(args) == "off":
        return "--panel off was given" if getattr(args, "panel", None) else "--no-panel was given"
    if not _a_terminal_we_can_draw_in(stream):
        # TWO REASONS, NOT ONE SENTENCE FOR BOTH. This said "output is not an interactive
        # terminal" whatever the cause, including a terminal that is perfectly interactive and
        # merely narrower than the panel can be drawn in. Somebody reading that goes looking at
        # their pipes and their CI variables for a problem that is the width of their window,
        # which is the shape of message this project keeps removing: confident, and about the
        # wrong thing.
        return _cannot_draw_because(stream)
    try:
        import rich  # noqa: F401
    except ImportError:
        return "rich is not installed, so the plain log is used"
    return None


def chosen_layout(args, guided=None):
    """Which panel this run should draw: "full", "inline" or "off".

    ONE FLAG, TWO SPELLINGS, AND THE OLD ONE STILL WORKS. `--no-panel` shipped in 0.4.0 and is
    somebody's muscle memory and somebody's script, so it stays as a spelling of `off` rather than
    being replaced. `--panel` wins where both are given, because naming the thing you want beats
    naming the thing you do not.

    THE DEFAULT DEPENDS ON WHO IS WATCHING (decision Q-42 D4). The guided mode is a person who
    chose to be led, so it takes the screen. The flag path is what CI and scripts drive and what a
    person tails, so it keeps a compact panel beside its log. `guided` is passed by the caller that
    knows; nothing here guesses it from the environment.
    """
    explicit = getattr(args, "panel", None)
    if explicit:
        return explicit
    if getattr(args, "no_panel", False):
        return "off"
    if guided is None:
        guided = bool(getattr(args, "guided", False))
    return "full" if guided else "inline"


def attach(events, args, *, total_trials=None, log=None, stream=None, console=None,
           baseline=None, guided=None):
    """A live panel over this run's event stream, or a `NullPanel`.

    `events` is an `EventLog`. The panel registers as an observer and unregisters on exit. Nothing
    it does can reach the search: `EventLog.emit` drops an observer that raises.
    """
    reason = why_not(args, stream=stream)
    if reason is not None:
        # SAID, NOT JUST COMPUTED. `why_not` returns a sentence precisely so a reader can be told,
        # and its own docstring gives the reason: "the panel did not appear" is otherwise the kind
        # of thing somebody files a bug about. It was then thrown away here, so the one person who
        # needed the sentence never saw it, and on 2026-09-28 the question was asked out loud.
        # One line, on the way past, naming the flag that turns it off.
        if log:
            log(f"  no live panel: {reason}. The run is unaffected; --no-panel controls it.")
        return NullPanel()
    try:
        return _RichPanel(events, total_trials=total_trials, log=log,
                          stream=stream or sys.stdout, console=console,
                          args=args, baseline=baseline,
                          layout=chosen_layout(args, guided=guided))
    except Exception as e:
        # A panel that cannot be built is not a failed run. Said once, then forgotten.
        if log:
            log(f"  the live panel could not start ({type(e).__name__}: {e}); using the plain log")
        return NullPanel()


class _RichPanel:
    """The real thing. Constructed only when `why_not` returned None."""

    active = True

    def __init__(self, events, *, total_trials=None, log=None, stream=None, console=None,
                 args=None, baseline=None, layout="inline"):
        from rich.console import Console
        from rich.live import Live

        self._events = events
        self._log = log or (lambda _m: None)
        self._total = int(total_trials) if total_trials else None
        # `console` is accepted so this class can be driven without a terminal. rich decides colour
        # and width from the file it is given, and a StringIO has no file descriptor, so a test
        # cannot otherwise make it render: `Console.is_terminal` is a read-only property. Everything
        # in here was therefore unexercised, at 43% coverage, until a coverage floor caught it.
        self._console = console or Console(file=stream or sys.stdout)
        # The best row so far, which is the only thing a watcher actually wants.
        self._best = None
        self._last = None
        self._trials = 0
        self._notes = []
        self._stop = None
        # SCENE 8'S HEADER, from the arguments the run was actually given rather than from a
        # second source. Everything here is what the person typed or what the preset resolved, so
        # a panel and a log that disagree is not a state this can reach.
        self._args = args
        self._baseline = baseline
        self._started = time.monotonic()
        self._layout = layout
        self._spin = 0
        self._on_alt = False
        # Every (drift, refusal) seen, for the frontier. Bounded because a long search is
        # thousands of trials and the plot has a few hundred cells: past the cap the oldest go,
        # which is the right end to lose since the interesting marks are the recent ones.
        self._points = []
        self._stop = None
        # `get_renderable` RATHER THAN A STORED RENDERABLE. With one, `Live` calls back on every
        # refresh tick and the panel redraws four times a second whether or not a trial landed, so
        # the spinner turns and `elapsed` counts during the long silences this exists for. With a
        # stored renderable it would only change when `update` was called, which is once a trial.
        self._live = Live(console=self._console, get_renderable=self._render,
                          refresh_per_second=4, transient=False)

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def __enter__(self):
        self._stop = self._events.observe(self.event)
        self._enter_screen()
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
        # LAST, AND UNCONDITIONALLY. Leaving the alternate screen has to happen however the run
        # ended, including a crash and a Ctrl+C, because a process that exits without restoring
        # the buffer leaves the person looking at a dead dashboard with no shell prompt.
        self._leave_screen()
        return False

    # ── the alternate screen ────────────────────────────────────────────────────

    def _enter_screen(self):
        """Take the whole terminal, for the full layout only. Decision Q-42 D1.

        THE STACKING DEFECT IS A SHARED-SURFACE PROBLEM, so this removes the sharing rather than
        managing it. Measured under a pty: rich erases exactly its own panel height per redraw, and
        an interleaved log line longer than the terminal wraps to two rows while rich counts one,
        so the erase falls short and a dead copy of the panel is left in the scrollback. That is
        the ledger's Torch F1 by another route.

        On the alternate screen there is no scrollback to leave anything in, and the log is not
        sharing the surface. The inline layout keeps the shared surface on purpose, because that is
        where the log is the point, and the hard wrap in `cli.log` is what keeps it honest there.

        Failures are swallowed: a terminal that will not switch buffers is not a reason an
        abliteration stops, and the panel simply draws in place as it did before.
        """
        if self._layout != "full" or self._on_alt:
            return
        try:
            self._console.set_alt_screen(True)
            self._on_alt = True
        except Exception:
            self._on_alt = False

    def _leave_screen(self):
        if not self._on_alt:
            return
        try:
            self._console.set_alt_screen(False)
        except Exception:
            pass
        self._on_alt = False

    # ── input ───────────────────────────────────────────────────────────────────

    def event(self, rec):
        """One record from the stream. Only the kinds this panel draws are read."""
        if rec.get("kind") != "trial":
            return
        self._trials += 1
        self._last = rec
        if self._best is None or self._is_better(rec, self._best):
            self._best = rec
        try:
            self._points.append((float(rec.get("kl")), float(rec.get("refusals"))))
            del self._points[:-MAX_POINTS]
        except (TypeError, ValueError):
            # A trial that reported neither is still a trial; it just has no mark to draw.
            pass
        # No `update` call: `Live` pulls from `get_renderable`, so recording the state IS the
        # update and calling `update` here would replace the callback with a frozen snapshot.
        self._live.refresh()

    def note(self, text):
        """A line worth keeping on the panel, e.g. a stage boundary. The log gets it too, elsewhere."""
        self._notes.append(str(text))
        del self._notes[:-3]
        self._live.refresh()

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
        """One renderer, two containers. Decision Q-42 D4.

        The flag path is what CI and scripts drive, so it keeps a compact panel beside its log. The
        guided mode is a person watching a run that takes an hour, so it gets the whole screen. The
        numbers are identical either way; only how much room they are given differs.
        """
        # ADVANCED ONCE, HERE. Two call sites each advancing it would step the animation twice a
        # frame, which is not wrong so much as unreadable.
        self._spin = (self._spin + 1) % len(SPINNER_FRAMES)
        return self._render_full() if self._layout == "full" else self._render_inline()

    # ── the pieces both layouts share ───────────────────────────────────────────

    def _spinner(self):
        """The frame for this render.

        IT HAS TO TURN WHEN NOTHING IS HAPPENING, which is the whole point, and that is a fact
        about who drives the redraw rather than about this function. `Live.update` is only reached
        when a trial lands, and on a CPU run trials are minutes apart; a spinner stepping once a
        trial stands perfectly still exactly when somebody is wondering whether the process has
        wedged. So the panel hands `Live` a `get_renderable` and lets it redraw on its own timer.
        """
        return SPINNER_FRAMES[self._spin]

    def _elapsed(self):
        return time.monotonic() - self._started

    def _remaining(self):
        """An estimate, or None when there is not yet anything to estimate from.

        None rather than a guess for the first trial, because "about 0s" on a two-hour run is worse
        than an empty cell: the empty cell says nothing and the guess says something false.
        """
        if not self._total or self._trials < 1:
            return None
        per = self._elapsed() / self._trials
        return max(0.0, per * (self._total - self._trials))

    @staticmethod
    def _short(name):
        """`Qwen/Qwen3-1.7B` reads as `Qwen3-1.7B` in a header that already says what tool it is."""
        return str(name).rsplit("/", 1)[-1] if name else "?"

    def _headline(self):
        """`trial 48 / 200 ⠙`, which is the one place a reader looks to ask "is it alive".

        The count answers how far, and the spinner answers whether anything is still turning. They
        belong together because a stalled run and a slow run show the same count.
        """
        done = f"{self._trials}" + (f" / {self._total}" if self._total else "")
        return f"trial {done} {self._spinner()}"

    def _narrow(self):
        """Whether this render is under the design's first breakpoint: one column, no chart."""
        return self._console.width < ONE_COLUMN_BELOW

    def _grid(self):
        """An empty stat grid, two label/value pairs wide or one, depending on the terminal.

        LEFT-ALIGNED LABELS, as drawn. Right-aligning them lines up their last letters, which
        makes a ragged left edge down the one column a reader scans.
        """
        from rich.table import Table

        grid = Table.grid(padding=(0, 2))
        grid.add_column(justify="left", style="dim", min_width=8)
        grid.add_column()
        if not self._narrow():
            grid.add_column(justify="left", style="dim", min_width=6)
            grid.add_column()
        return grid

    @staticmethod
    def _cell(value):
        """A value that came from the user, as text rather than as markup.

        A grid cell is parsed for rich markup, so a model path or a track directory with square
        brackets in it has the bracketed part SILENTLY EATEN: `runs/[v2]/model` renders as
        `runs//model`, and the panel then shows a path that is not the one the run is using. On a
        screen whose whole job is to say what is being measured, that is the worst available way
        to be wrong. Every cell whose text came from outside this file goes through here.
        """
        from rich.text import Text

        return Text(str(value))

    def _stat_grid(self):
        a = getattr(self._args, "__dict__", {})
        used, total, temp, power, blocked = card_telemetry(a.get("device", "cpu"))
        grid = self._grid()

        if total:
            vram = f"{bar(used, total)}  {used / 1e9:.1f} / {total / 1e9:.1f} GB"
        else:
            # NOT A ZERO. There is no card, or it could not be read, and a bar reading 0.0 GB
            # would be a measurement of something that was never measured.
            vram = "not a cuda device"

        # Temperature and power are blank when `pynvml` will not import, and a blank is honest.
        # TEMP AND POWER ARE THEIR OWN LABELLED FIGURES, as drawn. Merging them under one heading
        # saved a word and lost which number was which. The label only appears when there is a
        # reading behind it: without `pynvml` both are always absent, and a permanently blank band
        # under a heading reads as a measurement that failed rather than one never available.
        heat = f"{temp} °C" if temp is not None else ""
        watts = f"{power} W" if power is not None else ""
        # "16, reduced from 24" when the governor cut it, because a batch that is not the one asked
        # for changes what every timing on this screen means.
        batch = str(a.get("gen_batch", "?"))
        asked = a.get("gen_batch_requested")
        if asked and str(asked) != batch:
            batch = f"{batch}, reduced from {asked}"

        if self._narrow():
            # STACKED, IN THE SAME READING ORDER. A narrow terminal loses the second column, not
            # the figures that were in it, so every cell above reappears on a row of its own.
            grid.add_row("model", self._cell(self._short(a.get("model"))))
            grid.add_row("VRAM", vram)
            grid.add_row("prompts", self._cell(a.get("track", "?")))
            if heat:
                grid.add_row("temp", heat)
            if watts:
                grid.add_row("power", watts)
            grid.add_row("device", self._cell(a.get("device", "?")))
            grid.add_row("batch", batch)
            # THE HEADLINE COMES INSIDE THE BOX HERE. `_title` drops the trial count when the top
            # border cannot hold the model name and the count without them colliding, which on a
            # narrow terminal is always, so the count and the spinner would be lost entirely: the
            # one place a reader looks to ask whether the run is alive. A narrow terminal drops the
            # chart, not a number.
            grid.add_row("trial", f"{self._trials} / {self._total or '?'} {self._spinner()}")
            grid.add_row("seed", str(a.get("seed", "?")))
        else:
            grid.add_row("model", self._cell(self._short(a.get("model"))), "VRAM", vram)
            if heat or watts:
                grid.add_row("prompts", self._cell(a.get("track", "?")), "temp",
                             f"{heat}        power   {watts}")
            else:
                grid.add_row("prompts", self._cell(a.get("track", "?")), "", "")
            grid.add_row("device", self._cell(a.get("device", "?")), "batch", batch)
            grid.add_row("trials", str(self._total or "?"), "seed", str(a.get("seed", "?")))

        # THE GAP SAYS WHY IT IS A GAP, and only where a card was asked for. On a CPU run the VRAM
        # cell already reads "not a cuda device" and a second line about missing bindings would be
        # noise; on a CUDA run two permanently empty cells with no explanation is the state that had
        # somebody assuming the card was unreadable rather than the bindings missing.
        if blocked and str(a.get("device", "")).startswith("cuda"):
            # AS `Text`, NOT AS A STRING: a grid cell given a `str` is parsed as rich markup, and
            # this cell carries a sentence rather than a figure, so anything bracketed in it would
            # be read as a style tag and eaten. It also wants to sit dimmer than the numbers.
            from rich.text import Text
            note = Text(blocked, style="dim")
            if self._narrow():
                grid.add_row("", note)
            else:
                grid.add_row("", note, "", "")
        return grid

    def _outcome_grid(self):
        grid = self._grid()

        started = f"{self._baseline * 100:.1f}% refusal" if self._baseline is not None else "?"
        if self._best is not None:
            # THE DESIGN'S OWN WORDING, and shorter than the inline panel's on purpose: the full
            # layout already carries the model, the device and the budget above this line, so
            # repeating the whole trial row here wrapped it onto two and buried the two figures
            # the run is actually judged on.
            best = self._headline_outcome(self._best)
        else:
            best = f"{self._spinner()} waiting for the first trial"

        # HEDGED, AS DRAWN. `remaining` is elapsed over trials done, times trials left, on a search
        # whose trials genuinely differ in cost: a K=3 trial at a wide weight range is not the same
        # work as a K=1 trial. Printing `16m 44s` puts four significant figures on an extrapolation.
        # This project withdraws numbers for that, and a dashboard whose job is saying what was
        # measured is the last place to start guessing to the second.
        left = self._remaining()
        remaining = _about(left) if left is not None else "not yet"
        if self._narrow():
            for label, value in (("started at", started), ("best so far", best),
                                 ("elapsed", _duration(self._elapsed())),
                                 ("remaining", remaining)):
                grid.add_row(label, value)
        else:
            grid.add_row("started at", started, "best so far", best)
            grid.add_row("elapsed", _duration(self._elapsed()), "remaining", remaining)
        return grid

    # ── the two containers ──────────────────────────────────────────────────────

    def _render_full(self):
        """Scene 8. A Panel, because the title has to live IN the top border.

        THE TRADE-OFF, since it is not obvious and the other way was tried. `Table.add_section`
        draws a divider that meets the frame, which is the `├────────┤` in the drawing, but `rich`
        renders a Table's title ABOVE the box rather than in its border, which loses the header
        line the design opens with. A Panel does the reverse. The header was the operator's own
        ask, so the Panel wins and the dividers span the content width instead, flush to the
        borders rather than through them. Getting both would mean drawing every frame character by
        hand and giving up `rich`'s wrapping, which is a bad trade for two junction glyphs.
        """
        from rich.console import Group
        from rich.panel import Panel
        from rich.rule import Rule
        from rich.text import Text

        head = [Text(""), self._stat_grid(), Text("")]
        tail = [Rule(style=BRAND_PURPLE), Text(""), self._outcome_grid()]
        tail.extend(Text("  " + note, style="dim") for note in self._notes)
        tail.append(Text(""))
        # The short spelling where the long one would wrap onto a second row and read as a
        # half-finished sentence rather than as a hint.
        tail.append(Text("  --panel off turns this off" if self._narrow() else
                         "  --panel inline for a smaller one, --panel off for none", style="dim"))

        chart = self._chart()
        # SHORT TERMINALS DROP THE CHART BEFORE THEY DROP A NUMBER, which is the design's rule and
        # was the exact opposite of what happened. The Panel is drawn to the terminal height, so
        # anything that does not fit is cropped from the BOTTOM, and the bottom is where the two
        # figures the run is judged on live: on a short window the plot survived and "best so far"
        # did not. Shedding the chart is what buys the numbers their rows back.
        if chart and self._rows_needed(head + chart + tail) > self._console.height - 2:
            chart = []

        # FILLS THE TERMINAL. Scene 8 says the run takes the whole screen, and the alternate buffer
        # is already in use, so a box that stops after its content leaves the rest blank and reads
        # as a fragment rather than as the run.
        return Panel(Group(*head, *chart, *tail), title=self._title(), title_align="left",
                     border_style=BRAND_PINK, padding=(0, 1),
                     height=self._console.height)

    def _chart(self):
        """The frontier block, or nothing when there is no room across or nothing to plot.

        Dropped outright under the first breakpoint (`ONE_COLUMN_BELOW`): a plot given twenty
        columns is a smudge with a scale beside it, and the rows it costs are rows the numbers want.
        """
        from rich.rule import Rule
        from rich.text import Text

        if self._narrow():
            return []
        rows = frontier([(x, y) for x, y in self._points],
                        best=self._best_point(), best_label=self._best_mark(),
                        width=max(24, self._console.width - 8))
        if not rows:
            return []
        block = [Rule(style=BRAND_PURPLE), Text(""),
                 Text("  FRONTIER            refusals ↓                    drift ↓", style="dim"),
                 Text("")]
        block += [Text("  " + r, style=BRAND_DEEP_PINK) for r in rows]
        block.append(Text(""))
        return block

    def _rows_needed(self, parts):
        """How many terminal rows `parts` would occupy inside the frame, measured not counted.

        Counted by hand this drifts the moment a grid wraps, which is precisely the narrow terminal
        the shedding rule is for, so it asks rich the same question rich will answer when it draws.
        Four columns go to the two borders and the panel's horizontal padding.
        """
        from rich.console import Group

        options = self._console.options.update(width=max(1, self._console.width - 4))
        return len(self._console.render_lines(Group(*parts), options, pad=False))

    def _title(self):
        """Both ends of the top border, joined by the border itself.

        `rich` gives one title and the gap has to be filled with the rule rather than with spaces,
        or the top border is cut in half and the frame stops reading as one box. Falls back to the
        name alone on a terminal too narrow to hold both, rather than letting them collide.
        """
        left = f"senbonzakura · {self._short(getattr(self._args, 'model', None))}"
        right = self._headline()
        room = self._console.width - 6 - len(left) - len(right)
        return f"{left} {'─' * max(1, room - 2)} {right}" if room >= 4 else left

    def _render_inline(self):
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column()

        done = f"{self._trials}" + (f" of {self._total}" if self._total else "")
        table.add_row("trials", done)
        for label, rec in (("best so far", self._best), ("latest", self._last)):
            if rec is None:
                table.add_row(label, f"{self._spinner()} waiting for the first trial")
                continue
            table.add_row(label, self._describe(rec))
        left = self._remaining()
        if left is not None:
            table.add_row("remaining", _duration(left))
        for note in self._notes:
            table.add_row("", note)
        table.add_row("", "[dim]the full log is printing beside this; --panel off turns this off[/dim]")
        return Panel(table, title="senbonzakura", border_style=BRAND_PINK)

    @staticmethod
    def _headline_outcome(rec):
        """The two figures a search is judged on, in the design's own words."""
        try:
            ref = f"{float(rec.get('refusals')) * 100:.1f}%"
        except (TypeError, ValueError):
            ref = "?"
        try:
            kl = f"{float(rec.get('kl')):.4f}"
        except (TypeError, ValueError):
            kl = "?"
        return f"{ref} at drift {kl}"

    def _best_mark(self):
        """What the arrow beside the best point says, or None when there is no best yet."""
        if self._best is None:
            return None
        return f"best, trial {self._best.get('number', '?')}"

    def _best_point(self):
        try:
            return (float(self._best.get("kl")), float(self._best.get("refusals")))
        except (AttributeError, TypeError, ValueError):
            return None

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
    """`--panel` and `--no-panel`, defined here so the flags and the behaviour live together."""
    parser.add_argument("--panel", dest="panel", choices=("full", "inline", "off"), default=None,
                        help="how much of the screen the live view takes. 'full' is the dashboard "
                             "with the frontier plot, which is what the guided mode uses; 'inline' "
                             "is a compact panel beside the log, which is the default everywhere "
                             "else; 'off' draws nothing. The log is identical in all three: the "
                             "same lines and the same markers are printed either way.")
    parser.add_argument("--no-panel", dest="no_panel", action="store_true",
                        help="the same as --panel off. Kept because it shipped first and is in "
                             "people's scripts; --panel wins if both are given.")
