# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The guided mode's crash and refusal screens, measured at three widths rather than read.

THE LEDGER ITEM, AND THE MEASUREMENT THAT MADE IT ACTIONABLE

"`interactive.py`'s failure paths still print hand wrapped literals. The screens a first time
user meets now go through `say`, so the walk reads correctly at 60 columns. The crash and refusal
screens below them do not, and they are laid out for 79."

Counted rather than estimated: **41 `log("  …")` calls**, of which **23 sat in 10 multi-line
hand-wrapped blocks** and the other 18 were single short lines that fit any terminal. The ten
blocks are what this covers, and the count is now 18.

THE FRAME WAS THE PART A READING WOULD HAVE MISSED

The failure screen's rules were two 61-character literals. On a 52-column terminal the rule wraps
onto a second line, so the box has three sides and a stray tail, which is worse than a long line
of prose because it reads as a rendering bug rather than as text. Found by driving the screen at
52 columns, not by counting the literals. `_rule` is now one function over `say.width`.

WHY THIS FILE ASSERTS AT THREE WIDTHS AND NOT ONE

79 alone proves nothing, since that is what the literals were already laid out for. 52 is where
the defect lived. 120 is there because `say.CEILING` caps the width at 79 on purpose, so a screen
that grew with the terminal would be a different defect in the other direction, and nothing was
checking that either.
"""
from __future__ import annotations

import pytest

from senbonzakura import interactive as it
from senbonzakura import say

#: PINNED, BECAUSE THIS FILE DRIVES THE WALK. `interactive.run` is the walk's front door and
#: reaches `plan_abliteration` one line in, so without this the questions asked depend on what the
#: box running the test happens to hold. Required by
#: `tests/test_the_walk_never_reads_the_machine.py`, which did not catch it until `.run(` and
#: `log_failure(` were added to its detector on 2026-10-01: the guard against reading the machine
#: could not see the commonest way of doing it.
pytestmark = pytest.mark.usefixtures("a_machine_with_nothing_on_it")


def _plan(out="./out"):
    """A plan whose `--out` is read from the filesystem, so the test has to own the directory.

    **NOT `"./out"`, and that cost a rewrite.** `log_failure` calls `_what_survived`, which reads
    the directory off the filesystem and is right to: the old version claimed a persisted study on
    every failure alike, including failures that happened before the search began. So the screen
    has two branches, and which one is drawn depends on what is actually on disk.

    With a relative `./out` this file was asking the directory the test runner happened to be
    started in. Proven rather than suspected: with a `./out/best-config.json` in the working
    directory, the recovery sentence is not printed at all and the "What is on disk" branch is
    drawn instead, so `test_the_recovery_sentence_survives_the_reflow` fails on a machine that has
    an unrelated directory called `out`.

    That is this project's most repeated defect, a test reading the machine it runs on, written
    into a test by somebody who had just spent the day cataloguing it. Found by predicting what
    the real `tests/conftest.py` pins and then checking, rather than by the test failing.

    **The cure is `monkeypatch.chdir(tmp_path)`, not an absolute path**, and the first attempt at
    an absolute one taught the difference. `tmp_path` is 70-odd characters, so passing it as
    `--out` made the screen print lines no terminal could fit and the width assertions failed on
    the path rather than on the prose. Worse, one of those lines is the `--resume` command, which
    `say` refuses to wrap on purpose because a wrapped command cannot be pasted, so the assertion
    was demanding something the module is right to decline.

    Changing directory instead gives both halves: the directory is private to the test, and
    `./out` is the length a real user's would be, so the widths being measured are the ones a
    person actually meets.
    """
    return {"command": "kageyoshi", "options": {"--model": "M", "--out": str(out)},
            "licence": "yours"}


def _lines_at(width, make, monkeypatch, tmp_path):
    monkeypatch.setenv("COLUMNS", str(width))
    monkeypatch.chdir(tmp_path)
    said = []
    make(said.append)
    return said


def _too_wide(said):
    """Lines over the terminal width that the module was willing to wrap in the first place.

    **The exemption is the module's own rule, not a convenience.** `say.is_verbatim` passes through
    anything indented, because in every message in this tool a deeper-indented line is a command or
    an example, and a wrapped command cannot be pasted. The two-space indent is prose under a
    heading and is wrapped; four spaces and beyond is content, and `say` is right to leave it.

    A flat "no line over the width" assertion is therefore stronger than the behaviour, and it was
    the first version of this file: it failed on the `--resume` command line, demanding something
    the module declines on purpose. An assertion that contradicts a deliberate design decision is
    not a stricter test, it is a wrong one.
    """
    return [line for line in said
            if len(line) > say.width() and not say.is_verbatim(line[2:])]


@pytest.mark.parametrize("width", [52, 60, 79, 120])
def test_the_failure_screen_fits_whatever_terminal_it_is_drawn_in(width, monkeypatch, tmp_path):
    """Every line, frame included, inside the width `say` reports.

    A failure screen is the worst possible place for a rendering fault: it is read by somebody who
    has just lost a run and is looking for the sentence that tells them what survived.
    """
    said = _lines_at(width, lambda log: it.log_failure(
        _plan(), "the card ran out of memory",
        step={"command": "kageyoshi"}, log=log), monkeypatch, tmp_path)
    assert said, "the failure screen printed nothing"
    over = _too_wide(said)
    assert not over, (
        f"at COLUMNS={width} (say.width()={say.width()}) these lines are too wide and will wrap "
        f"into ragged halves: {over}")


@pytest.mark.parametrize("width", [52, 79])
def test_the_recovery_sentence_survives_the_reflow(width, monkeypatch, tmp_path):
    """Wrapping must not be allowed to lose the content, which is the point of the screen.

    Asserted on the collapsed text rather than on a line, because where the breaks fall is not
    this test's subject and asserting on them would make every width change a failure.
    """
    said = _lines_at(width, lambda log: it.log_failure(
        _plan(), "the card ran out of memory",
        step={"command": "kageyoshi"}, log=log), monkeypatch, tmp_path)
    flat = " ".join(" ".join(said).split())
    for phrase in ("Nothing recoverable was written to ./out",
                   "run the same command again",
                   "--resume would have nothing to resume"):
        assert phrase in flat, f"{phrase!r} did not survive the reflow at {width} columns"


@pytest.mark.parametrize("width", [52, 79])
def test_the_frame_is_drawn_to_the_terminal_and_not_to_a_literal(width, monkeypatch, tmp_path):
    """The three-sided box, which is what a hardcoded 61-character rule produces at 52 columns."""
    said = _lines_at(width, lambda log: it.log_failure(
        _plan(), None, step={"command": "kageyoshi"}, log=log), monkeypatch, tmp_path)
    # `"─" in line` first, or a blank line matches `strip("─ ") == ""` and the test asserts a
    # frame is 52 characters about a line that is not a frame. It did, on the first run.
    rules = [line for line in said
             if "─" in line and (line.strip("─ ") == "" or line.startswith("── "))]
    assert rules, f"no frame was drawn: {said}"
    for rule in rules:
        assert len(rule) == say.width(), (
            f"a frame line is {len(rule)} characters against a terminal of {say.width()}: "
            f"{rule!r}")


@pytest.mark.parametrize("width", [52, 79])
def test_the_other_branch_fits_too_and_says_what_is_recoverable(width, monkeypatch, tmp_path):
    """The branch the machine-read was accidentally selecting, now driven on purpose.

    `log_failure` has two shapes: nothing recoverable on disk, and something recoverable. The
    second one carries the `--resume` command and the sentence about what continuing does, and it
    had no test at all: this file's first version only ever reached it by accident, on a machine
    with a stray `./out`.

    So the accident is now a case. Both branches are drawn, both are measured for width, and the
    one that tells somebody their hours of card time survived is asserted rather than assumed.
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "senbon-study.db").write_bytes(b"")
    (out / "best-config.json").write_text("{}", encoding="utf-8")
    said = _lines_at(width, lambda log: it.log_failure(
        _plan(), "the card ran out of memory",
        step={"command": "kageyoshi"}, log=log), monkeypatch, tmp_path)
    flat = " ".join(" ".join(said).split())
    assert "the persisted study, so completed trials are not lost" in flat, flat
    assert "best-config.json, the winning configuration" in flat, flat
    assert "--resume" in flat, "the command that picks the run up was not offered"
    over = _too_wide(said)
    assert not over, f"at COLUMNS={width} these lines are too wide: {over}"


def test_the_hand_wrapped_blocks_are_gone_and_the_count_is_the_guard():
    """A count, because the fix is a migration and the risk is a new one being added beside it.

    Single-line `log("  …")` calls are fine and 18 remain. What this refuses is a NEW multi-line
    hand-wrapped block, which is how the ten being fixed here got there one at a time. A block is
    two or more consecutive `log` calls whose text starts with indentation, which is the shape
    somebody reaches for when they wrap a paragraph by hand.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "senbonzakura"
              / "interactive.py").read_text(encoding="utf-8")
    runs, current = [], []
    for number, line in enumerate(source.splitlines(), 1):
        indented = re.match(r'\s*log\(f?"\s{2,}', line)
        continuing = current and re.match(r'\s*log\(f?"\s', line)
        if indented or continuing:
            current.append(number)
        else:
            if len(current) > 1:
                runs.append((current[0], current[-1]))
            current = []
    if len(current) > 1:
        runs.append((current[0], current[-1]))
    assert not runs, (
        f"these look like hand-wrapped paragraphs built from consecutive `log` calls, at lines "
        f"{runs}. Pass the whole sentence to `_say`, which wraps to the terminal, rather than "
        f"breaking it by hand to a width the reader may not have")
