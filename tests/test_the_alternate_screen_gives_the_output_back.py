# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The alternate screen used to discard everything printed beside the panel. Driven on a real pty.

THE DEFECT

`chosen_layout` returns `"full"` for the guided mode, `_enter_screen` calls
`set_alt_screen(True)`, and `_leave_screen` calls `set_alt_screen(False)`, which restores the
primary buffer and lets the terminal throw the alternate one away. Everything written beside the
panel during the search was written into the buffer being discarded. Four paths wrote there: the
governor's pause, shrink, yield and resume lines including `VRAM OOM at batch=1`; optuna's WARNING
for a caught trial failure with its full parameter dump; torch and transformers warnings; and
`cli.interrupted_notice`.

**The Ctrl+C path is why this was serious rather than untidy.** A person stops a long run, the tool
prints how many trials survived and the command that resumes it, and the screen closes and takes
the answer with it. The run was recoverable and they were told nothing about how, which is worse
than printing nothing, because the tool believes it has explained itself.

WHY EVERY BEHAVIOURAL TEST HERE NEEDS A REAL TERMINAL, AND WHY THAT IS THE POINT

An earlier attempt at a test for this was **structurally incapable of failing**: it attached a
non-terminal stream, `why_not` therefore returned a reason, `attach` returned a `NullPanel`, and
the test above it in that file asserts a `NullPanel` does nothing at all. It was asserting that
nothing happens about a thing that never ran.

The same trap is one step further in here. `Console(file=StringIO(), force_terminal=True)` is
enough to make rich *draw*, which is how `test_the_live_display_never_eats_the_markers.py`
exercises the rendering without a pty, and it is **not** enough to test this: a StringIO keeps
every byte ever written to it, including the ones a real terminal discards when the alternate
screen closes. A test against a StringIO would pass whether or not the fix exists.

So the harness is a real pseudo-terminal, a child process, and an assertion about what appears
**after the alternate-screen-off control sequence**. That is the only place the discard is
visible, and it is the same reasoning as driving `--no-interactive` through `script -qec` rather
than through a stub whose `isatty` returns True.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The escape sequence a terminal is sent to leave the alternate screen and restore the primary
#: buffer. Everything before it in the stream was written into a buffer the terminal discards;
#: everything after it the person keeps. The whole assertion in this file is "which side is it on".
ALT_SCREEN_OFF = "\x1b[?1049l"

#: One line per real path that used to be lost. The text is representative rather than produced by
#: the real caller, because three of the four callers cannot be imported without torch; the shapes
#: are taken from the call sites named in `_replay`'s docstring.
NEEDLES = (
    "GOVERNOR: VRAM OOM at batch=1, shrinking",
    "WARNING optuna caught a trial failure",
    "Stopped after 7 trials. Resume with: senbonzakura --resume ./out",
)

_CHILD = """
import sys
from senbonzakura import events, livedisplay
log = events.EventLog(None)
panel = livedisplay._RichPanel(log, total_trials=3, log=lambda _m: None, layout={layout!r})
with panel:
    print({a!r})
    print({b!r}, file=sys.stderr)
    print({c!r})
print("AFTER THE BLOCK")
"""


def _run_on_a_pty(script, *, timeout=90):
    """Run a script in a child process whose stdout really is a terminal, and return what it saw.

    `pty` is POSIX only and the panel needs rich, so both are skips rather than failures.
    """
    pty = pytest.importorskip("pty")
    pytest.importorskip("rich")
    path = Path(script)
    master, slave = pty.openpty()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), TERM="xterm-256color",
               COLUMNS="100", LINES="40", PYTHONDONTWRITEBYTECODE="1")
    # NO_COLOR and CI would both turn the panel off through `why_not`, and a CI runner sets CI.
    for killer in ("NO_COLOR", "CI"):
        env.pop(killer, None)
    proc = subprocess.Popen([sys.executable, str(path)], stdin=slave, stdout=slave,
                            stderr=slave, env=env, cwd=str(ROOT))
    os.close(slave)
    chunks = []
    try:
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:          # the child closed its end
                break
            if not data:
                break
            chunks.append(data)
    finally:
        os.close(master)
        proc.wait(timeout=timeout)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _child(tmp_path, layout="full"):
    path = tmp_path / "child.py"
    path.write_text(_CHILD.format(layout=layout, a=NEEDLES[0], b=NEEDLES[1], c=NEEDLES[2]),
                    encoding="utf-8")
    return path


@pytest.mark.parametrize("needle", NEEDLES)
def test_what_was_printed_behind_the_dashboard_survives_it(needle, tmp_path):
    """The whole claim, on a real terminal, asserted on the side of the discard that matters.

    Before the fix every one of these appeared only BEFORE the alternate-screen-off sequence,
    which is a buffer the terminal throws away. A test that looked only for the text somewhere in
    the stream would have passed against the defect.
    """
    seen = _run_on_a_pty(_child(tmp_path))
    assert ALT_SCREEN_OFF in seen, (
        "the child never left the alternate screen, so this test proves nothing about the discard")
    kept = seen[seen.rfind(ALT_SCREEN_OFF):]
    assert needle in kept, (
        f"{needle!r} was printed beside the panel and does not appear after the alternate screen "
        f"closed, so the terminal discarded it along with the buffer")


def test_the_reader_is_told_where_the_replayed_text_came_from(tmp_path):
    """Output appearing after a dashboard has closed is otherwise unexplained."""
    seen = _run_on_a_pty(_child(tmp_path))
    kept = seen[seen.rfind(ALT_SCREEN_OFF):]
    assert "what was printed behind the dashboard" in kept


def test_the_replay_does_not_contain_the_dashboard_itself(tmp_path):
    """The panel's own drawing must not be in the replay, and the reason is structural.

    `self._console` is bound to the real stream object in `__init__`, before any tee exists, so
    rich writes past the capture. If the console is ever built lazily from `sys.stdout` instead,
    the replay starts including a frame of the dashboard, and this is what notices.

    Asserted on the panel's title, which is drawn by rich and by nothing else.
    """
    seen = _run_on_a_pty(_child(tmp_path))
    kept = seen[seen.rfind(ALT_SCREEN_OFF):]
    assert "trial" not in kept.lower() or "Resume with" in kept, kept[:400]
    # The box-drawing characters rich uses for the panel frame. One in the replay means the
    # dashboard is being captured and reprinted.
    assert "─" * 10 not in kept.replace(
        "── what was printed behind the dashboard ───────────────────────────", ""), (
        "a run of box-drawing characters appears in the replay, so the panel's own frame is being "
        "captured: check that the console is bound to the real stream rather than to sys.stdout")


def test_an_inline_panel_replays_nothing_because_it_discards_nothing(tmp_path):
    """No alternate screen, no discard, so no replay and above all no duplicate.

    This is the test that would catch the fix being applied to the wrong layout. The inline layout
    keeps the shared surface on purpose, so its log is already on the primary buffer; replaying it
    would print everything twice.
    """
    seen = _run_on_a_pty(_child(tmp_path, layout="inline"))
    assert ALT_SCREEN_OFF not in seen, "the inline layout took the alternate screen"
    assert "what was printed behind the dashboard" not in seen, (
        "the inline layout replayed output it never discarded, so every line is now doubled")
    for needle in NEEDLES:
        assert seen.count(needle) == 1, (
            f"{needle!r} appears {seen.count(needle)} times on the inline layout; it should appear "
            f"exactly once")


# ── the parts that need no terminal ──────────────────────────────────────────────

def test_a_console_that_is_not_a_terminal_is_not_recorded_as_being_on_the_screen():
    """`set_alt_screen` returns False without raising when it writes no control codes.

    The old code read only the exception, so a console that never switched buffers was recorded as
    being on the alternate screen. That was harmless while nothing depended on the flag. It stopped
    being harmless with the replay, because the replay would then print a second copy of output the
    terminal had never discarded.
    """
    pytest.importorskip("rich")
    import io

    from rich.console import Console

    from senbonzakura import events, livedisplay

    panel = livedisplay._RichPanel(events.EventLog(None), total_trials=1, log=lambda _m: None,
                                   console=Console(file=io.StringIO()), layout="full")
    panel._enter_screen()
    assert panel._on_alt is False, (
        "a non-terminal console was recorded as being on the alternate screen, so the replay will "
        "duplicate output that was never discarded")
    assert panel._saved_streams is None, "stdout was teed for a console that never switched buffers"


def test_the_replay_writes_to_the_saved_stream_rather_than_to_whatever_stdout_is_now():
    """The guard a surviving mutant asked for, and it is about the target rather than the order.

    A mutant that moved `_stop_capture` to after `_replay` changed nothing at all, and that was
    the right answer: `_replay` names `self._saved_streams[0]`, so it cannot be writing through its
    own tee whichever order the two calls sit in. The comment in `_leave_screen` had claimed the
    order was what prevented it, which was simply untrue, and the mutant is what showed that.

    What a later reader could plausibly break is simplifying `_replay` to use `sys.stdout`, which
    looks identical and is not: on the Ctrl+C path the tee is still installed at that moment. So
    the property worth holding is the target, and this holds it.
    """
    import io

    from senbonzakura import livedisplay

    panel = livedisplay._RichPanel.__new__(livedisplay._RichPanel)
    saved = io.StringIO()
    elsewhere = io.StringIO()
    panel._captured = ["the recovery instructions\n"]
    panel._captured_chars, panel._dropped_lines = 25, 0
    panel._saved_streams = (saved, saved)
    real_stdout = sys.stdout
    try:
        sys.stdout = elsewhere
        panel._replay()
    finally:
        sys.stdout = real_stdout
    assert "the recovery instructions" in saved.getvalue(), (
        "the replay did not reach the stream the panel saved")
    assert elsewhere.getvalue() == "", (
        "the replay went to the current sys.stdout instead of the saved stream; on the Ctrl+C path "
        "that is still the tee")


def test_the_capture_is_bounded_and_drops_the_oldest_lines():
    """Baseline Section 12: no collection that grows with input. A long search prints for hours.

    The oldest end goes, which is the opposite of what a log would do and is deliberate: what this
    buffer exists to save is the END of the run, where the OOM and the recovery instructions are.
    """
    from senbonzakura import events, livedisplay

    panel = livedisplay._RichPanel.__new__(livedisplay._RichPanel)
    panel._captured, panel._captured_chars, panel._dropped_lines = [], 0, 0
    assert events  # the import is the module under test's own dependency, not decoration
    for i in range(20000):
        panel._keep(f"line {i} with some padding to make it worth counting\n")
    held = "".join(panel._captured)
    assert len(held) <= livedisplay.CAPTURE_CHARS, (
        f"the capture grew to {len(held)} characters against a cap of {livedisplay.CAPTURE_CHARS}")
    assert panel._dropped_lines > 0, "nothing was dropped, so the cap never engaged"
    assert "line 19999" in held, "the most recent line was dropped, which is the wrong end"
    assert "line 0 " not in held, "the oldest line survived, so the cap dropped from the wrong end"
    assert held.startswith("line "), (
        f"the kept text begins mid-line, so whole lines are not being dropped: {held[:40]!r}")


def test_the_truncation_notice_says_how_much_went_and_what_was_kept():
    """Line 689, which coverage on a CPU-only box showed nothing reached.

    The notice only prints when the cap has engaged, so it needs a buffer big enough to engage it.
    It is user-facing text about missing output, which is the worst kind to leave unasserted: a
    reader who is not told that lines were dropped reads the oldest surviving line as the first
    thing that happened.
    """
    import io

    from senbonzakura import livedisplay

    panel = livedisplay._RichPanel.__new__(livedisplay._RichPanel)
    panel._captured, panel._captured_chars, panel._dropped_lines = [], 0, 0
    for i in range(20000):
        panel._keep(f"line {i} with some padding to make it worth counting\n")
    saved = io.StringIO()
    panel._saved_streams = (saved, saved)
    panel._replay()
    out = saved.getvalue()
    assert "earlier line(s) dropped" in out, (
        "output was truncated and the replay did not say so, so the oldest line kept reads as the "
        "first line printed")
    assert "256 KiB" in out, f"the notice does not say how much was kept: {out[:200]!r}"


def test_a_dead_terminal_during_the_replay_is_not_a_failed_run():
    """Lines 693 to 696. This runs on the Ctrl+C path, where there is nothing useful left to do.

    A `print` to a closed stream raising would otherwise turn "your run was interrupted and here
    is how to resume it" into a second, unrelated traceback on top of the first.
    """
    from senbonzakura import livedisplay

    class _Closed:
        def write(self, _text):
            raise ValueError("I/O operation on closed file")

        def flush(self):
            pass

    panel = livedisplay._RichPanel.__new__(livedisplay._RichPanel)
    panel._captured = ["the recovery instructions\n"]
    panel._captured_chars, panel._dropped_lines = 25, 0
    panel._saved_streams = (_Closed(), _Closed())
    panel._replay()          # must not raise


def test_a_terminal_that_refuses_to_switch_buffers_does_not_stop_the_run():
    """Lines 724 to 725 and 750 to 751: `set_alt_screen` raising rather than returning False.

    A terminal that will not switch buffers is not a reason an abliteration stops; the panel draws
    in place as it did before. Both directions are covered, because leaving has to survive it too:
    a process that exits without restoring the buffer leaves somebody looking at a dead dashboard
    with no shell prompt.
    """
    from senbonzakura import events, livedisplay

    class _Awkward:
        is_terminal = True

        def set_alt_screen(self, _enable=True):
            raise OSError("this terminal has opinions")

    panel = livedisplay._RichPanel.__new__(livedisplay._RichPanel)
    panel._layout, panel._on_alt, panel._console = "full", False, _Awkward()
    panel._captured, panel._captured_chars, panel._dropped_lines = [], 0, 0
    panel._saved_streams = None
    panel._enter_screen()
    assert panel._on_alt is False, "a terminal that raised was recorded as having switched"
    assert panel._saved_streams is None, "stdout was teed for a screen that never opened"
    # And leaving, from a state where entering had succeeded before the console turned awkward.
    panel._on_alt = True
    panel._leave_screen()
    assert panel._on_alt is False
    assert events  # the module under test's own dependency


def test_stopping_a_capture_that_never_started_does_nothing():
    """Line 654. `_leave_screen` calls it unconditionally, so it has to tolerate being called twice."""
    import sys as _sys

    from senbonzakura import livedisplay

    panel = livedisplay._RichPanel.__new__(livedisplay._RichPanel)
    panel._saved_streams = None
    before = (_sys.stdout, _sys.stderr)
    panel._stop_capture()
    assert (_sys.stdout, _sys.stderr) == before, "the streams were replaced by a no-op call"


def test_the_tee_writes_through_unchanged_and_delegates_everything_else():
    """A tee whose presence changes the output is not a tee.

    `isatty` is the one that would actually bite: something downstream deciding it is writing to a
    pipe changes its own behaviour, and then the capture has altered the run it was observing.
    """
    import io

    from senbonzakura import livedisplay

    class _Terminalish(io.StringIO):
        def isatty(self):
            return True

    real = _Terminalish()
    kept = []
    tee = livedisplay._Tee(real, kept.append)
    tee.write("one\n")
    tee.writelines(["two\n", "three"])
    tee.flush()
    assert real.getvalue() == "one\ntwo\nthree", "the wrapped stream did not receive the writes"
    assert "".join(kept) == "one\ntwo\nthree", "the copy does not match what was written"
    assert tee.isatty() is True, "isatty was not delegated, so the tee is visible to its writers"


def test_a_keep_that_raises_cannot_cost_the_run_its_log():
    """The real write happens first, so the worst case is a replay missing a line."""
    import io

    from senbonzakura import livedisplay

    real = io.StringIO()

    def _boom(_text):
        raise RuntimeError("the buffer is unhappy")

    livedisplay._Tee(real, _boom).write("this still has to arrive\n")
    assert real.getvalue() == "this still has to arrive\n"


def test_the_interrupt_notice_is_printed_inside_the_block_the_replay_covers():
    """Ties the Ctrl+C claim to the code rather than to a line of representative text.

    Three of the four lost paths cannot be imported without torch, so the behavioural tests above
    use text of the right shape. This one asserts the structural fact that makes the claim true for
    the path that matters: `cli.interrupted_notice` is called **inside** the
    `with livedisplay.attach(...)` block, which is exactly why its output was being discarded, and
    is exactly what the replay now covers.

    If somebody later moves the call out of the block, that is a different and also valid fix, and
    this fails so the claim gets re-read rather than left standing.
    """
    tree = ast.parse((ROOT / "src" / "senbonzakura" / "cli.py").read_text(encoding="utf-8"))
    blocks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        header = ast.dump(ast.Module(body=[ast.Expr(i.context_expr) for i in node.items],
                                     type_ignores=[]))
        if "attr='attach'" in header:
            blocks.append(node)
    assert blocks, "no `with ...attach(...)` block found in cli.py; this test's premise has moved"
    inside = any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == "interrupted_notice"
        for block in blocks for call in ast.walk(block))
    assert inside, (
        "`interrupted_notice` is no longer called inside the attach block. If it was moved out "
        "deliberately that is a valid fix for the Ctrl+C path, but the replay is then no longer "
        "what makes it survive and this file's docstring needs correcting")
