# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The run dashboard: what it draws, what it refuses to draw, and which screen it takes.

WHAT PROMPTED IT, 2026-09-28. `private/design/interactive-mode-designs.md` Scene 8 was approved and
never built. What shipped instead was a three-row grid in a colour from no palette this project
owns, wrapping the search only, leaving a dead copy of itself in the scrollback after every log
line. The operator compared the two and said they were different products.

THE PROPERTIES WORTH GUARDING ARE NOT "IT RENDERS". They are: every figure on it was measured
rather than assumed, the plot carries its scale, the thing that says a silent run is alive actually
moves, and the screen it takes depends on who is watching.
"""
from __future__ import annotations

import argparse
import io
import pathlib
import sys
from typing import ClassVar

import pytest
from tomlread import tomllib

from senbonzakura import events, livedisplay


def _args(**kw):
    kw.setdefault("model", "Qwen/Qwen3-1.7B")
    kw.setdefault("track", "default")
    kw.setdefault("device", "cpu")
    kw.setdefault("gen_batch", 16)
    kw.setdefault("seed", 42)
    kw.setdefault("panel", None)
    kw.setdefault("no_panel", False)
    return argparse.Namespace(**kw)


def _panel(layout="full", width=94, total=200, baseline=0.578, height=None, **argkw):
    rich_console = pytest.importorskip("rich.console", reason="rich draws the panel")
    console = rich_console.Console(file=io.StringIO(), width=width, force_terminal=True)
    if height is not None:
        console.height = height
    return livedisplay._RichPanel(
        events.EventLog(None), total_trials=total, log=None, stream=io.StringIO(),
        console=console, args=_args(**argkw), baseline=baseline, layout=layout)


def _with_trials(panel, count=48):
    for i in range(count):
        panel.event({"kind": "trial", "number": i, "refusals": max(0.0, 0.2 - i * 0.004),
                     "soft": 0.0, "broken": 0.0, "kl": 0.03 + i * 0.004,
                     "objective": 0.5 - i * 0.002})
    return panel


def _drawn(panel):
    """The panel as plain text, which is what a reader actually sees."""
    panel._console.file = io.StringIO()
    panel._console.print(panel._render())
    return panel._console.file.getvalue()


# ── which screen, and who decides ────────────────────────────────────────────────

class TestWhichPanelThisRunGets:
    """Decision Q-42 D4. The flag path is driven by scripts; the guided mode is watched by a person."""

    def test_a_plain_run_gets_the_compact_panel(self):
        assert livedisplay.chosen_layout(_args()) == "inline"

    def test_the_guided_mode_gets_the_whole_screen(self):
        assert livedisplay.chosen_layout(_args(), guided=True) == "full"

    def test_the_flag_wins_over_the_guided_default(self):
        """Somebody who asks for the small one while being led gets the small one."""
        assert livedisplay.chosen_layout(_args(panel="inline"), guided=True) == "inline"

    def test_the_old_flag_still_means_off(self):
        """`--no-panel` shipped in 0.4.0 and is in people's scripts, so it keeps working."""
        assert livedisplay.chosen_layout(_args(no_panel=True)) == "off"

    def test_naming_what_you_want_beats_naming_what_you_do_not(self):
        assert livedisplay.chosen_layout(_args(panel="full", no_panel=True)) == "full"

    def test_off_is_a_reason_the_panel_can_state(self):
        reason = livedisplay.why_not(_args(panel="off"))
        assert reason and "off" in reason


# ── nothing on it is guessed ─────────────────────────────────────────────────────

class TestEveryFigureWasMeasured:
    """Q-42 D3. A blank is honest; a plausible number is the thing this project withdraws."""

    def test_an_unmeasurable_card_draws_an_empty_bar_rather_than_zero(self):
        assert set(livedisplay.bar(None, None)) == {"░"}
        assert set(livedisplay.bar(0, 0)) == {"░"}

    def test_the_bar_is_the_proportion_it_was_given(self):
        assert livedisplay.bar(0, 100, width=10) == "░" * 10
        assert livedisplay.bar(100, 100, width=10) == "▓" * 10
        half = livedisplay.bar(50, 100, width=10)
        assert half.count("▓") == 5 and half.count("░") == 5

    def test_telemetry_never_raises_and_says_nothing_it_did_not_read(self):
        """A card reading is decoration. It may not be the reason an abliteration stops."""
        used, total, temp, power, reason = livedisplay.card_telemetry("cpu")
        assert temp is None or isinstance(temp, int)
        assert power is None or isinstance(power, (int, float))
        assert (reason is None) == (temp is not None)

    def test_a_panel_with_no_card_does_not_print_a_vram_figure(self):
        text = _drawn(_panel())
        assert "not a cuda device" in text, (
            "a run with no card must say so rather than drawing a bar at 0.0 GB, which is a "
            "measurement of something never measured")

    def test_remaining_is_withheld_until_there_is_something_to_estimate_from(self):
        p = _panel()
        assert p._remaining() is None
        assert "not yet" in _drawn(p)


# ── the frontier is a plot, not a scatter ────────────────────────────────────────

class TestTheFrontierCarriesItsScale:
    """A bare grid shows the shape of the cloud and nothing about its size.

    Two runs an order of magnitude apart in drift rendered identically before the axes existed,
    which is the defect the operator caught by comparing the drawing with the screen.
    """

    POINTS: ClassVar = [(0.03, 0.19), (0.05, 0.12), (0.09, 0.10), (0.13, 0.047), (0.19, 0.0)]

    def test_nothing_yet_draws_nothing(self):
        assert livedisplay.frontier([]) == []
        assert livedisplay.frontier([(None, None)]) == []

    def test_the_axis_labels_are_the_data_it_was_given(self):
        rows = livedisplay.frontier(self.POINTS, width=78, height=5)
        text = "\n".join(rows)
        assert "0.19" in text, "the top tick does not state the highest refusal seen"
        assert "0.00" in text, "the bottom tick does not state the lowest refusal seen"
        assert "0.03" in text and "0.19" in text, "the drift scale is missing"

    def test_the_best_point_says_which_trial_it_is(self):
        rows = livedisplay.frontier(self.POINTS, best=(0.19, 0.0),
                                    best_label="best, trial 27", width=78)
        marked = [r for r in rows if "●" in r]
        assert marked, "the best point was not drawn"
        assert "← best, trial 27" in marked[0], (
            "the annotation is not on the row with the mark it names, so it names nothing")

    def test_the_annotation_gets_room_rather_than_overflowing(self):
        """Drawn to the full width it wrapped onto its own line, where it named nothing."""
        width = 78
        rows = livedisplay.frontier(self.POINTS, best=(0.19, 0.0),
                                    best_label="best, trial 27", width=width)
        assert all(len(r) <= width for r in rows), (
            f"a row exceeds the width it was given: {max(len(r) for r in rows)} > {width}")

    def test_one_distinct_value_on_an_axis_does_not_divide_by_zero(self):
        rows = livedisplay.frontier([(0.1, 0.5), (0.1, 0.5)], width=40)
        assert rows


# ── the thing that says a silent run is alive ────────────────────────────────────

class TestTheSpinner:
    """The phases this exists for have nothing to report.

    A CPU run printed `[ 231.8s]` and then nothing for 339 seconds while it extracted directions.
    A spinner that steps once per trial stands still through exactly that.
    """

    def test_it_turns_on_a_redraw_with_no_trial_landing(self):
        p = _panel()
        frames = []
        for _ in range(6):
            p._render()
            frames.append(p._spinner())
        assert len(set(frames)) > 1, (
            f"the spinner did not move across six redraws: {frames}. It steps on render, and "
            f"render is what the refresh tick calls, so this is the silent case it is for.")

    def test_it_sits_beside_the_trial_count(self):
        p = _panel()
        p._render()
        head = p._headline()
        assert head.startswith("trial "), head
        assert head[-1] in livedisplay.SPINNER_FRAMES, (
            f"the headline does not end in a spinner frame: {head!r}")


# ── both containers draw, at both ends of the range ──────────────────────────────

class TestBothLayoutsDraw:
    @pytest.mark.parametrize("layout", ["full", "inline"])
    @pytest.mark.parametrize("width", [60, 94, 200])
    def test_it_renders_before_any_trial_and_after_several(self, layout, width):
        p = _panel(layout=layout, width=width)
        assert _drawn(p), "an empty panel drew nothing at all"
        for i in range(5):
            p.event({"kind": "trial", "number": i, "refusals": 0.2 - i * 0.03,
                     "soft": 0.0, "broken": 0.0, "kl": 0.05 + i * 0.01,
                     "objective": 0.5 - i * 0.05})
        text = _drawn(p)
        assert "trial" in text

    def test_a_trial_that_reports_no_numbers_is_still_counted(self):
        """It has no mark to draw, which is not a reason to lose the trial or to raise."""
        p = _panel()
        p.event({"kind": "trial", "number": 0})
        assert p._trials == 1
        assert p._points == []
        assert _drawn(p)

    def test_the_full_layout_names_the_model_and_the_budget_on_one_border(self):
        p = _panel(layout="full")
        p._render()
        text = _drawn(p)
        assert "Qwen3-1.7B" in text, "the model is not named in the header"
        assert "/ 200" in text, "the trial budget is not on the border"

    def test_a_note_reaches_the_panel_and_the_oldest_are_dropped(self):
        p = _panel()
        for i in range(6):
            p.note(f"note {i}")
        assert len(p._notes) == 3
        assert "note 5" in _drawn(p)


# ── the alternate screen ─────────────────────────────────────────────────────────

class TestTakingTheScreen:
    """Q-42 D1. The full layout removes the shared surface; the inline one keeps it deliberately."""

    def test_only_the_full_layout_takes_the_screen(self):
        inline = _panel(layout="inline")
        inline._enter_screen()
        assert inline._on_alt is False
        inline._leave_screen()

    def test_the_screen_is_given_back_even_when_it_was_never_taken(self):
        """`__exit__` runs on every path including a crash, so this must be safe to call twice."""
        p = _panel(layout="full")
        p._leave_screen()
        p._leave_screen()
        assert p._on_alt is False


class TestTheLifecycleGivesTheTerminalBack:
    """However the run ends, the person gets their shell back.

    A process that exits without restoring the buffer leaves somebody looking at a dead dashboard
    with no prompt, and the two ways out of a search are a crash and a Ctrl+C.
    """

    def test_entering_and_leaving_restores_the_buffer(self):
        p = _panel(layout="full")
        with p:
            assert p._on_alt is True, "the full layout did not take the screen"
        assert p._on_alt is False, "the screen was not given back"

    def test_an_exception_inside_the_run_still_gives_it_back(self):
        p = _panel(layout="full")
        with pytest.raises(RuntimeError), p:
            raise RuntimeError("the search died")
        assert p._on_alt is False, (
            "a crash left the terminal on the alternate screen, so the person has no prompt")

    def test_a_keyboard_interrupt_still_gives_it_back(self):
        """The likeliest way out of an hour-long search, and the one that used to traceback."""
        p = _panel(layout="full")
        with pytest.raises(KeyboardInterrupt), p:
            raise KeyboardInterrupt
        assert p._on_alt is False

    def test_the_inline_layout_never_took_it_and_never_restores_it(self):
        p = _panel(layout="inline")
        with p:
            assert p._on_alt is False
        assert p._on_alt is False

    def test_the_observer_is_dropped_on_the_way_out(self):
        """An observer holding a closed display would raise on every event for the rest of the run."""
        log = events.EventLog(None)
        p = _panel(layout="inline")
        p._events = log
        with p:
            assert p._stop is not None
        assert p._stop is None
        # The proof that it really unsubscribed: an event after the exit reaches nothing.
        before = p._trials
        log.emit("trial", number=99, refusals=0.0, kl=0.0, objective=0.0)
        assert p._trials == before, "the panel was still observing after it closed"


class TestAttachRefusesRatherThanGuessing:
    def test_a_pipe_gets_a_null_panel_and_is_told_why(self):
        said = []
        panel = livedisplay.attach(events.EventLog(None), _args(), log=said.append,
                                   stream=io.StringIO())
        assert not panel.active
        assert any("no live panel" in s for s in said), (
            "the panel was skipped and the reason was computed and discarded, which is the defect "
            "that had somebody asking out loud why it never appeared")

    def test_a_panel_that_cannot_be_built_is_not_a_failed_run(self, monkeypatch):
        said = []

        def boom(*a, **k):
            raise RuntimeError("no terminal for you")

        monkeypatch.setattr(livedisplay, "_RichPanel", boom)
        monkeypatch.setattr(livedisplay, "why_not", lambda *a, **k: None)
        panel = livedisplay.attach(events.EventLog(None), _args(), log=said.append)
        assert not panel.active
        assert any("could not start" in s for s in said)


class TestTheSpinnerNeverMovesWhatIsBesideIt:
    """Operator direction, 2026-09-28: smooth, and it must not scatter the width of anything.

    A spinner is drawn into a border and into a grid cell. If its frames differ in rendered width
    the border shifts a column every frame, which is worse than no spinner: a moving border reads
    as the terminal misbehaving rather than as the run being alive.
    """

    def test_every_frame_is_exactly_one_column(self):
        import unicodedata

        wide = {ch: unicodedata.east_asian_width(ch) for ch in livedisplay.SPINNER_FRAMES
                if unicodedata.east_asian_width(ch) in {"W", "F", "A"}}
        assert not wide, (
            f"these frames can render wider than one column: {wide}. 'W' and 'F' always do and "
            f"'A' is ambiguous, which means some terminals do. Either shifts the border.")

    def test_every_frame_is_a_single_character(self):
        assert all(len(ch) == 1 for ch in livedisplay.SPINNER_FRAMES)

    def test_the_header_is_the_same_length_whatever_frame_is_showing(self):
        """The property that actually matters: the thing beside it does not move."""
        p = _panel(layout="full")
        lengths = set()
        for _ in range(len(livedisplay.SPINNER_FRAMES) * 2):
            p._render()
            lengths.add(len(p._headline()))
        assert len(lengths) == 1, (
            f"the header changes length as the spinner turns: {sorted(lengths)}")

    def test_it_reads_as_turning_rather_than_blinking(self):
        assert len(livedisplay.SPINNER_FRAMES) >= 6, (
            "too few frames to read as motion rather than as flicking between states")


class TestTheFullPanelTakesTheWholeTerminal:
    """Scene 8: "it becomes the run, and it takes the whole terminal"."""

    def test_the_panel_fills_the_height_it_was_given(self):
        p = _panel(layout="full", width=94)
        p._console.height = 30
        drawn = _drawn(p).rstrip("\n").split("\n")
        assert len(drawn) == 30, (
            f"the panel drew {len(drawn)} rows into a 30-row terminal, so the screen below it is "
            f"blank and it reads as a fragment rather than as the run")

    def test_the_top_border_is_not_cut_in_half_by_the_header(self):
        """Padding the gap with spaces breaks the box; it is filled with the border rule."""
        p = _panel(layout="full", width=94)
        p._render()
        top = _drawn(p).split("\n")[0]
        assert "senbonzakura" in top and "trial" in top, top
        assert "─" in top.split("senbonzakura")[1].split("trial")[0], (
            f"the gap between the name and the trial count is blank, so the border is broken: {top!r}")

    def test_a_terminal_too_narrow_for_both_keeps_the_name_rather_than_colliding(self):
        p = _panel(layout="full", width=30)
        p._render()
        top = _drawn(p).split("\n")[0]
        assert "senbonzakura" in top


class TestNothingBleedsPastTheFrame:
    """Every rendered line is exactly the terminal width, at every width, in every state.

    WHY IT IS MEASURED IN COLUMNS AND NOT IN CHARACTERS. A double-width glyph occupies two columns
    and counts as one character, so a length check in `len()` passes on a line that visibly runs
    past the border. The spinner, the `●`, the `←` and the box drawing are all candidates, so the
    count here is the one the terminal actually does.

    The two places a bleed would appear first are the top border, where the model name and the
    trial count are composed into one title, and the frontier row carrying `← best, trial N`, which
    is the only row whose content is not bounded by the plot. Both are covered by sweeping widths
    rather than by naming them, because the next one added will not be named here.

    WHY THE EXPECTED WIDTH COMES FROM THE CONSOLE AND NOT FROM THE NUMBER WE ASKED FOR. On a legacy
    Windows console, which is any Windows terminal without virtual-terminal sequences, writing into
    the last cell of a row wraps the cursor, so `rich` reserves that column: `Console.size` returns
    `self._width - self.legacy_windows`, and the frame is drawn one narrower than the terminal on
    purpose. The panel already composes itself against `console.width`, so comparing the drawn line
    against the constructor's number instead reported a bleed on Windows where there was none.
    """

    @staticmethod
    def _columns(text):
        import re
        import unicodedata

        plain = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in plain)

    def _lines(self, width, height, trials, legacy_windows=False):
        """The drawn rows, and the width `rich` believes it had to fill them to.

        `legacy_windows` is passed explicitly rather than detected, so the Windows geometry is
        reproducible on every machine instead of only on the one runner that has it.
        """
        rich_console = pytest.importorskip("rich.console")
        console = rich_console.Console(file=io.StringIO(), width=width, height=height,
                                       force_terminal=True, legacy_windows=legacy_windows)
        p = livedisplay._RichPanel(
            events.EventLog(None), total_trials=200, log=None, stream=io.StringIO(),
            console=console, args=_args(gen_batch_requested=24), baseline=0.578, layout="full")
        for i in range(trials):
            p.event({"kind": "trial", "number": i, "refusals": max(0.0, 0.2 - i * 0.004),
                     "soft": 0.0, "broken": 0.0, "kl": 0.03 + i * 0.004,
                     "objective": 0.5 - i * 0.002})
        console.file = io.StringIO()
        console.print(p._render())
        return console.width, console.file.getvalue().rstrip("\n").split("\n")

    @pytest.mark.parametrize("legacy_windows", [False, True])
    @pytest.mark.parametrize("width", [40, 60, 72, 80, 94, 120, 200])
    def test_no_line_is_wider_or_narrower_than_the_terminal(self, width, legacy_windows):
        drawable, lines = self._lines(width, 30, 48, legacy_windows=legacy_windows)
        for line in lines:
            assert self._columns(line) == drawable, (
                f"a line renders at {self._columns(line)} columns where {drawable} are drawable in "
                f"a {width}-column terminal (legacy windows: {legacy_windows}), so it bleeds past "
                f"the frame or falls short of it: {line!r}")

    @pytest.mark.parametrize("legacy_windows", [False, True])
    @pytest.mark.parametrize("trials", [0, 1, 48])
    def test_it_holds_before_during_and_after_the_search_fills_up(self, trials, legacy_windows):
        drawable, lines = self._lines(94, 30, trials, legacy_windows=legacy_windows)
        for line in lines:
            assert self._columns(line) == drawable, f"{trials} trial(s): {line!r}"

    def test_a_terminal_shorter_than_the_content_does_not_overflow_it(self):
        _drawable, lines = self._lines(94, 12, 48)
        assert len(lines) == 12, (
            f"the panel drew {len(lines)} rows into a 12-row terminal, which scrolls the top of it "
            f"off the alternate screen where there is nothing to scroll back to")


# ── the gap where a measurement would be says why it is a gap ─────────────────────

class TestTheMissingCardBindingsAreNamed:
    """`pynvml` was read by the code and declared in no packaging file at all.

    WHAT THAT ACTUALLY MEANT, and it went unnoticed because the two cells are meant to be allowed
    to be empty: temperature and power were blank on every install that has ever shipped, and a
    reader had no way to tell "this build cannot read your card" from "your card is idle". Q-42 D3
    makes the extra optional; it does not make the absence silent.
    """

    def test_absent_bindings_come_back_as_a_reason_rather_than_just_a_blank(self, monkeypatch):
        # A None in `sys.modules` is the documented way to make an import fail for one name, so
        # this holds on a machine that happens to have the bindings as well as one that does not.
        monkeypatch.setitem(sys.modules, "pynvml", None)
        _used, _total, temp, power, reason = livedisplay.card_telemetry("cuda:0")
        assert temp is None and power is None
        assert reason == livedisplay.NO_TELEMETRY
        assert "nvidia-ml-py" in reason, (
            "the advice must name the distribution pyproject.toml declares. `pip install pynvml` "
            "names the module and fetches a third-party wrapper, which is the wrong package")

    def test_a_card_that_will_not_answer_is_a_different_sentence(self, monkeypatch):
        """Bindings present and the card silent is not the same situation as bindings absent."""
        class _Sulky:
            @staticmethod
            def nvmlInit():  # noqa: N802  (the real binding's own spelling)
                raise RuntimeError("driver not loaded")

        monkeypatch.setitem(sys.modules, "pynvml", _Sulky)
        _used, _total, temp, power, reason = livedisplay.card_telemetry("cuda:0")
        assert temp is None and power is None
        assert reason and "pip install" not in reason
        assert "RuntimeError" in reason, reason

    def test_a_cuda_run_is_told_what_the_blank_cells_would_need(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        text = _drawn(_panel(device="cuda:0"))
        assert "nvidia-ml-py" in text, (
            "the panel left temp and power blank and said nothing about why, which is the state "
            "that had somebody assuming the card was unreadable rather than the bindings missing")

    def test_a_cpu_run_is_not_nagged_about_a_card_it_does_not_have(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        text = _drawn(_panel(device="cpu"))
        assert "not a cuda device" in text
        assert "nvidia-ml-py" not in text, (
            "there is no card to read, so the VRAM cell has already said everything true")


class TestThePackagingDeclaresWhatTheCodeImports:
    """A dependency that is present by luck is not a dependency, and this one was not even that.

    `livedisplay.card_telemetry` imported `pynvml` and no packaging file anywhere named it, so
    temperature and power were blank on every install that ever shipped. The operator's ruling on
    2026-09-28 was the base install rather than an extra, reversing Q-42 D3: one `pip install` and
    one `senbonzakura setup` is the whole story, and a 0.05 MB wheel is not worth a user decision.
    """

    @staticmethod
    def _project():
        root = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        with root.open("rb") as fh:
            return tomllib.load(fh)["project"]

    def test_the_base_install_declares_the_nvml_bindings(self):
        required = self._project()["dependencies"]
        assert any("nvidia-ml-py" in spec for spec in required), (
            "nothing declares the bindings the dashboard imports, so temp and power are blank on "
            "every install and no install command a user could run would change that")

    def test_it_names_the_distribution_and_not_the_module(self):
        """`pip install pynvml` fetches a third-party wrapper, not the bindings this imports."""
        required = self._project()["dependencies"]
        assert not any(spec.split(">")[0].strip() == "pynvml" for spec in required)

    def test_no_extra_is_left_standing_in_its_place(self):
        """Two answers on record is how a user ends up installing neither."""
        extras = self._project()["optional-dependencies"]
        assert "telemetry" not in extras
        for name, specs in extras.items():
            assert not any("nvidia-ml-py" in spec for spec in specs), name


# ── the breakpoints the design specifies ─────────────────────────────────────────

class TestItIsResponsiveRatherThanAbsent:
    """The design's table: under 60 is one column and no chart, not nothing at all.

    WHAT WAS WRONG. The width floor and the first breakpoint were the same number, so a terminal
    under 60 columns was refused a panel outright. The design never says a narrow terminal loses the
    dashboard; it says the dashboard loses its second column and its chart.
    """

    class _ATerminal(io.StringIO):
        def isatty(self):
            return True

    def test_a_narrow_terminal_is_still_offered_a_panel(self, monkeypatch):
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setenv("COLUMNS", "40")
        assert livedisplay.why_not(_args(), stream=self._ATerminal()) is None, (
            "40 columns is narrow, not undrawable, and the design gives it the one-column form")

    def test_a_terminal_with_no_room_for_a_row_is_still_declined(self):
        assert livedisplay.MIN_COLUMNS < livedisplay.ONE_COLUMN_BELOW, (
            "the floor and the first breakpoint are separate questions and must be separate numbers")

    def test_under_the_breakpoint_it_draws_the_numbers_and_drops_the_chart(self):
        text = _drawn(_with_trials(_panel(width=50)))
        assert "FRONTIER" not in text, "the chart was kept on a terminal too narrow to plot in"
        assert "48 / 200" in text, (
            "the trial count lives in the top border, which is too short to hold it here, so the "
            "one figure a reader checks to ask whether the run is alive went missing entirely")
        for figure in ("model", "device", "started at", "best so far", "elapsed", "remaining"):
            assert figure in text, (
                f"{figure!r} is missing from a 50-column panel, so the narrow layout dropped a "
                f"number, which is the one thing the design says it must not do")

    def test_above_the_breakpoint_the_chart_is_still_drawn(self):
        assert "FRONTIER" in _drawn(_with_trials(_panel(width=94, height=40)))


class TestAShortTerminalShedsTheChartFirst:
    """The design's rule, verbatim: "short terminals drop the chart before they drop a number".

    The panel is drawn to the terminal height, so anything that does not fit is cropped from the
    BOTTOM, and the bottom is where "best so far", "elapsed" and "remaining" live. On a short window
    the plot survived and every outcome figure went over the edge, which is the rule inverted.
    """

    def test_the_outcome_figures_survive_a_short_window(self):
        text = _drawn(_with_trials(_panel(width=94, height=18)))
        assert "best so far" in text, (
            "an 18-row terminal cropped the outcome figures off the bottom while keeping the plot")
        assert "FRONTIER" not in text, "the chart should have been shed to make that room"

    def test_it_still_fills_the_short_terminal_exactly(self):
        p = _with_trials(_panel(width=94, height=18))
        assert len(_drawn(p).rstrip("\n").split("\n")) == 18

    def test_a_tall_terminal_keeps_both(self):
        text = _drawn(_with_trials(_panel(width=94, height=40)))
        assert "FRONTIER" in text and "best so far" in text, (
            "there was room for everything and something was shed anyway")


class _AReadableCard:
    """A `pynvml` that answers, so the drawing path for a real reading is exercised too.

    The absent case is the one that was broken and the one most installs used to hit, so it is
    where the tests above sit. This is the other half: without it nothing ever asserts that a
    figure the bindings DO return reaches the screen, and a panel that draws the missing case
    perfectly and the present case not at all would pass every test above.
    """

    NVML_TEMPERATURE_GPU = 0

    @staticmethod
    def nvmlInit():  # noqa: N802  (the real binding's own spelling)
        return None

    @staticmethod
    def nvmlShutdown():  # noqa: N802
        return None

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index):  # noqa: N802
        return index

    @staticmethod
    def nvmlDeviceGetTemperature(_handle, _kind):  # noqa: N802
        return 71

    @staticmethod
    def nvmlDeviceGetPowerUsage(_handle):  # noqa: N802
        return 84_137


class TestACardThatAnswers:
    def test_the_reading_comes_back_with_no_reason_attached(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _AReadableCard)
        _used, _total, temp, power, reason = livedisplay.card_telemetry("cuda:1")
        assert (temp, power) == (71, 84), "milliwatts are reported as watts, to the watt"
        assert reason is None, "there is nothing missing, so there is nothing to explain"

    def test_the_device_index_is_the_one_the_run_is_using(self, monkeypatch):
        """`cuda:1` must not read card 0, which is the card somebody else's job is on."""
        asked = []
        monkeypatch.setitem(sys.modules, "pynvml", _AReadableCard)
        monkeypatch.setattr(_AReadableCard, "nvmlDeviceGetHandleByIndex",
                            staticmethod(lambda index: asked.append(index) or index))
        livedisplay.card_telemetry("cuda:1")
        assert asked == [1]

    @pytest.mark.parametrize("width", [50, 94])
    def test_both_widths_draw_the_figures_rather_than_the_advice(self, monkeypatch, width):
        monkeypatch.setitem(sys.modules, "pynvml", _AReadableCard)
        text = _drawn(_panel(width=width, device="cuda:0"))
        assert "71 °C" in text and "84 W" in text
        assert "nvidia-ml-py" not in text, "a card that answered was still told to install something"


class TestTheScreenShowsWhatWasActuallyGiven:
    """A panel that quietly alters the thing it is reporting.

    A grid cell is parsed for rich markup, so a path with square brackets in it loses the
    bracketed part: `runs/[v2]/model` renders as `runs//model`. The panel's whole job is to say
    what is being measured, so showing a path that is not the one the run is using is the worst
    available way for it to be wrong. Found by the agent that built the narrow layouts, in its own
    file, and reported rather than swallowed.
    """

    @staticmethod
    def _grid(**argkw):
        panel = _panel(**argkw)
        panel._console.file = io.StringIO()
        panel._console.print(panel._stat_grid())
        return panel._console.file.getvalue()

    def test_a_bracketed_model_path_survives_to_the_screen(self):
        out = self._grid(model="runs/model[v2]")
        assert "[v2]" in out, (
            "the brackets were parsed as markup and dropped, so the panel is naming a model the "
            "run is not using. `_short` keeps the last path segment, which is why the bracket is "
            "in that segment here rather than in a parent directory")

    def test_a_bracketed_track_path_survives_to_the_screen(self):
        out = self._grid(track="corpora/[held-out]")
        assert "[held-out]" in out

    def test_it_holds_on_the_narrow_layout_too(self):
        out = self._grid(width=50, model="runs/model[v2]")
        assert "[v2]" in out


class TestTheReasonNamesTheActualReason:
    """"The panel did not appear" is the kind of thing somebody files a bug about, so the tool
    says why. It said the same sentence for four different causes, including a terminal that is
    perfectly interactive and merely narrower than a panel can be drawn in. Somebody reading that
    goes hunting through their pipes and their CI variables for a problem that is the width of
    their window.
    """

    @staticmethod
    def _stream():
        """An interactive stream with no `fileno`, so COLUMNS decides the width."""
        class _Tty:
            def isatty(self):
                return True

            def fileno(self):
                raise OSError("no fileno in this test, so the environment decides the width")

        return _Tty()

    def test_a_narrow_terminal_is_told_it_is_narrow(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("COLUMNS", "12")
        why = livedisplay.why_not(_args(), stream=self._stream())
        assert why is not None
        assert "narrower" in why, f"a twelve column terminal was told: {why!r}"
        assert "not an interactive terminal" not in why, (
            "this terminal IS interactive, and sending somebody to look at their pipes over a "
            "window width is a confident message about the wrong problem")

    def test_a_pipe_is_still_told_it_is_a_pipe(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("CI", raising=False)

        class _Pipe:
            def isatty(self):
                return False

        why = livedisplay.why_not(_args(), stream=_Pipe())
        assert why and "not an interactive terminal" in why

    def test_an_unset_term_says_so(self, monkeypatch):
        """MET ON A REAL RUN, 2026-09-28. A guided abliteration driven through a pty over ssh drew
        no panel and reported "output is not an interactive terminal", about a pty. The cause was
        an unset TERM, which is ordinary over ssh, and the message sent the reader somewhere else.
        """
        monkeypatch.setenv("TERM", "")
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("CI", raising=False)
        why = livedisplay.why_not(_args(), stream=self._stream())
        assert why and "TERM" in why
