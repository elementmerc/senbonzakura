# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Drive the real tool through a real terminal, the way a person does.

WHY THIS EXISTS. On 2026-09-28 three defects were found by driving the binary and none of them was
visible to 6,500 unit tests, because each lived in a SEQUENCE rather than in a function: a
pre-flight board that said "all clear" about a run the tool then refused, a guided walk that
produced a command its own pre-flight rejects, and a menu answer that was a valid index until the
menu grew. A journey is the unit that catches those: a terminal, a fixed list of keystrokes, and
assertions on what a person would notice.

WHAT A JOURNEY MAY ASSERT, and this is the whole argument for the design. It asserts INVARIANTS,
never transcripts. A journey that pins the exact bytes breaks on every reworded sentence, and a
suite that cries wolf is one people learn to skip, at which point it is not there on the day it
matters. So: the exit code, a few phrases that carry meaning, no traceback, nothing wider than the
window it was given, and the rule this project already lives by, that the printed command is the
command that runs.

DETERMINISM IS THE OTHER HALF, and it is enforced here rather than remembered by each journey. Every
session gets its own working directory, its own Hugging Face cache, a fixed terminal size, a fixed
TERM, no banner (which is chosen at random), and the Hub switched off. A journey that reaches the
network is not a journey, it is a flake with a story.
"""
from __future__ import annotations

import os
import pty
import re
import select
import signal
import sys
import time

#: Answers are sent when the child is waiting for one, and this is how that is recognised: every
#: prompt this tool writes ends in a colon and a space and carries no newline, because the process
#: then blocks on `input`. Matched on the END of what has arrived rather than anywhere in it,
#: because the same text appears in prose above the prompt.
#:
#: COLON AND SPACE, NOT COLON AND ANYTHING. The pattern was `:\s?$`, which matched a colon
#: anywhere prose happened to put one, and a buffer ends wherever the last `read` returned rather
#: than where the tool stopped. The licence block carries the line
#: "Using the bundled evaluation track (CC BY-NC 4.0: attribution required", so a chunk boundary
#: landing after that colon made the harness answer a prompt that did not exist. The real prompt
#: then had no answer left, the newline that stands for the person's Enter was never injected, and
#: the next line of output ran on from the prompt. It failed about one run in three.
_WAITING = re.compile(r": $")

#: And the tool has to have STOPPED, which is the half the pattern cannot see. A prompt is the last
#: thing written before the process blocks on a read, so what it is DOING is the real signal and
#: the text is only the corroboration.
#:
#: MEASURED RATHER THAN ASSUMED, on a real walk, 2026-09-28. Polling the child's state during every
#: silence found fifty moments where the terminator was showing and the process was still RUNNING,
#: against seven where it was blocked in a terminal read, and seven is the number of prompts the
#: walk has. So "it went quiet and the text ends in a prompt" is not one coincidence away from
#: answering a prompt that does not exist, it is fifty. The longest mid-stream silence in that walk
#: was 1.5 seconds, ten times any tolerable timeout, which settles the question of whether a timing
#: threshold could ever have been the answer: it could not.
#:
#: `state` and `wchan` from `/proc`, then. Linux only, so it degrades rather than depends: a kernel
#: that cannot be asked falls back to the silence, which is what this did before and is right most
#: of the time. What the fallback never does is override a definite answer.
_TTY_READ = frozenset({"wait_woken", "n_tty_read"})

#: How long a silence stands in for "it has stopped", where the kernel cannot be asked.
_QUIET = 0.15


def _waiting_for_input(pid):
    """True if the child is blocked reading its terminal, False if it is busy, None if unknowable.

    None is the honest third answer and the caller treats it as such. Reporting a guess as a fact
    here would put an answer into a program that had not asked for one, and the symptom of that
    surfaces three screens later as a width complaint about a line nobody drew.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            state = f.read().rsplit(") ", 1)[1].split()[0]
    except (OSError, IndexError):                         # pragma: no cover - not Linux
        return None
    if state in ("R", "D"):
        # Running, or in an uninterruptible wait. Either way it is not asking anybody anything.
        return False
    if state != "S":
        return None
    try:
        with open(f"/proc/{pid}/wchan", encoding="utf-8") as f:
            return f.read().strip() in _TTY_READ
    except OSError:                                       # pragma: no cover - not Linux
        return None

#: Escape codes, stripped before anything is asserted. A journey is about what a person reads.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][B0]|\x1b[=>]|\r")

#: Ctrl+C as a byte an answer list can carry, so "the user interrupted it here" is just another
#: step rather than a separate mechanism.
INTERRUPT = "\x03"


class Result:
    """What a session produced, and the assertions worth making about it."""

    def __init__(self, output, status, columns):
        self.output = output
        self.status = status
        self.columns = columns

    @property
    def lines(self):
        return self.output.splitlines()

    def __repr__(self):                                   # pragma: no cover - a failure aid
        return f"<Result status={self.status} chars={len(self.output)}>"

    # ── the assertions ───────────────────────────────────────────────────────────────────

    def says(self, *phrases):
        for phrase in phrases:
            assert phrase in self.output, (
                f"the session never said {phrase!r}.\n"
                f"What it did say:\n{self._tail()}")
        return self

    def never_says(self, *phrases):
        for phrase in phrases:
            assert phrase not in self.output, (
                f"the session said {phrase!r}, which it must not.\n{self._tail()}")
        return self

    def exited(self, status):
        assert self.status == status, (
            f"the session exited {self.status}, not {status}.\n{self._tail()}")
        return self

    def carried_no_traceback(self):
        """A traceback is the tool failing to have an opinion about what went wrong.

        Every refusal this project ships is a sentence. One that arrives as eleven frames of our
        own internals is the defect `entry.py` exists to prevent, and it has reappeared twice
        through paths nobody had driven.
        """
        for marker in ("Traceback (most recent call last)", "\nTypeError:", "\nAttributeError:"):
            assert marker not in self.output, (
                f"a traceback reached the user:\n{self._tail(40)}")
        return self

    def fits_the_window(self, *, allowed=()):
        """Nothing runs past the edge except what cannot be folded.

        A path or a command carrying one unbreakable token wider than the window is not a wrapping
        failure and must not be folded: a folded path is a wrong path, and a folded command cannot
        be pasted. So a line is only a finding when its longest word would have fitted.
        """
        over = []
        for line in self.lines:
            if len(line) <= self.columns or any(a and a in line for a in allowed):
                continue
            if max((len(w) for w in line.split()), default=0) + 4 > self.columns:
                continue
            over.append(line)
        assert not over, (
            f"at {self.columns} columns these ran past the edge: {over}\n"
            f"The session, for context:\n{self._tail(40)}")
        return self

    def _tail(self, n=25):
        return "\n".join(f"  | {line}" for line in self.lines[-n:])


def _environment(home, columns, lines, extra):
    """The fixed world every journey runs in. See the module docstring on determinism."""
    env = dict(os.environ)
    # Removed rather than overridden: either of these turns the live panel off, and a journey that
    # passes because the machine running it happened to set CI is measuring the machine.
    for name in ("NO_COLOR", "CI", "SENBON_PANEL", "COLUMNS", "LINES"):
        env.pop(name, None)
    env.update({
        "COLUMNS": str(columns),
        "LINES": str(lines),
        "TERM": "xterm-256color",
        # The banner is one of several designs chosen at random, so it is off: a journey that
        # asserted on what it drew would fail one run in five.
        "SENBON_BANNER": "off",
        # NO NETWORK. A journey that reaches the Hub is a flake with a story, and the cache is
        # per session so one journey cannot see what another downloaded.
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(home / "hf"),
        "HOME": str(home),
        "PYTHONUNBUFFERED": "1",
    })
    env.update(extra or {})
    return env


def drive(tmp_path, argv, answers=(), *, columns=100, lines=40, timeout=120, env=None):
    """Run `senbonzakura <argv>` on a pty, answer its prompts in order, and return the Result.

    `answers` are sent one at a time, each when the child next stops with a prompt. Running out of
    answers is not an error: a journey that ends by letting the tool finish sends none at all.
    """
    # ONE LINE PER ANSWER, and the echo handling below depends on it. `\n` is put back into the
    # captured output once per answer to stand for the Enter the person pressed, so an answer
    # carrying its own newlines would insert the wrong number of breaks and every width assertion
    # after it would measure a line nobody drew. Nothing sends one today; this is here so that the
    # day something does, it says so rather than producing a quietly wrong measurement.
    for answer in answers:
        assert "\n" not in answer and "\r" not in answer, (
            f"the answer {answer!r} carries its own line break. A journey sends one line per "
            f"prompt; a multi-line answer needs the echo handling in this function revisited "
            f"first, not a newline smuggled through it.")

    home = tmp_path / "home"
    (home / "hf").mkdir(parents=True, exist_ok=True)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    environment = _environment(home, columns, lines, env)

    pid, fd = pty.fork()
    # ECHO OFF, so what comes back is what the PROGRAM printed.
    #
    # A pty echoes what is written to it, so every answer a journey sends reappears glued to the
    # end of the prompt it answered. `fits_the_window` then measured "Which model?: " plus a
    # forty character model id and called the tool's own prompt too wide, which is a journey
    # failing on its own keystrokes. A person's typing is not output, so it is not measured.
    if pid == 0:                                          # pragma: no cover - the child execs away
        os.chdir(work)
        # S606: no shell, on purpose. A shell between the test and the tool would mean the
        # journey is measuring the shell's quoting as well, and the whole point is the program a
        # person invokes.
        os.execve(sys.executable,  # noqa: S606
                  [sys.executable, "-m", "senbonzakura", *argv], environment)

    try:
        import termios
        mode = termios.tcgetattr(fd)
        mode[3] &= ~termios.ECHO                          # lflag
        termios.tcsetattr(fd, termios.TCSANOW, mode)
    except Exception:       # a pty that will not take it still runs; the echo is then measured
        pass

    out, sent, deadline = b"", 0, time.time() + timeout
    timed_out, quiet_since = False, None
    while True:
        if time.time() > deadline:
            timed_out = True
            os.kill(pid, signal.SIGKILL)
            break
        readable, _, _ = select.select([fd], [], [], _QUIET)
        if readable:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
            quiet_since = None
            continue
        # NOTHING ARRIVED. Ask the child what it is doing; only fall back to the length of the
        # silence where it cannot be asked.
        blocked = _waiting_for_input(pid)
        if blocked is False:
            quiet_since = None
            continue
        if quiet_since is None:
            quiet_since = time.time()
        if blocked is None and time.time() - quiet_since < _QUIET:
            continue
        if sent < len(answers) and _WAITING.search(_ANSI.sub("", out.decode(errors="replace"))):
            answer = answers[sent]
            # An interrupt is a byte, not a line: writing a newline after it would answer the
            # prompt as well as interrupting, and the two are different journeys.
            os.write(fd, answer.encode() if answer == INTERRUPT else answer.encode() + b"\n")
            # THE NEWLINE THE PERSON PRESSED, put back into what we measure.
            #
            # Echo is off above, so the answer itself never appears, which is right: somebody's
            # typing is not the program's output. But their Enter is what ENDS the line, and
            # without it the next prompt continues on the same one, so two prompts look like a
            # single over-wide line. This restores the break and nothing else.
            if answer != INTERRUPT:
                out += b"\n"
            sent += 1
            quiet_since = None

    os.close(fd)
    status = None
    try:
        _, raw = os.waitpid(pid, 0)
        status = os.waitstatus_to_exitcode(raw)
    except ChildProcessError:                             # pragma: no cover - already reaped
        pass
    text = _ANSI.sub("", out.decode(errors="replace"))
    assert not timed_out, (
        f"the session was still running after {timeout}s and was killed. It is either hung or "
        f"waiting for an answer nobody sent.\nWhat it had said:\n" +
        "\n".join(f"  | {line}" for line in text.splitlines()[-25:]))
    return Result(text, status, columns)
