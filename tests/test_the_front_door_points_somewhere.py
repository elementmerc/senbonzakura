# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Typing the tool's name is a question, and it has to be answered.

Bare `senbonzakura` used to print a 27-line argparse usage block listing every flag and then
refuse, because no model had been given. The first thing the tool ever said to anybody was a wall
of flags followed by an error, and the guided mode, which is what a newcomer wants, appeared
nowhere on it.

What these hold down is the page itself and the two properties that make it a front door rather
than a second help page: it names commands that exist, and the one it recommends can be typed.
"""
import re

import pytest

from senbonzakura import entry


@pytest.fixture
def page():
    return entry.short_help()


def test_the_first_thing_it_says_is_the_way_in(page):
    """`-i` is the recommended entry point, so it is on the first screen and it is first."""
    assert "senbonzakura -i" in page
    assert page.index("START HERE") < page.index("Or go direct")
    assert page.index("senbonzakura -i") < page.index("senbonzakura --model")


def test_bare_senbonzakura_answers_rather_than_refusing(capsys):
    assert entry.main([]) == 0, "asking a tool what it is is not a usage error"
    out = capsys.readouterr().out
    assert "senbonzakura -i" in out
    assert "--trials" not in out, "the flag wall belongs behind --help, not on the front door"
    assert len(out.splitlines()) < 25, (
        f"the front door is {len(out.splitlines())} lines. It exists because the 27-line usage "
        f"block was too long to read, so it cannot become one.")


def test_every_command_it_names_is_a_command_that_exists(page):
    """A SECOND LIST OF COMMAND NAMES IS A SECOND THING TO KEEP TRUE.

    This page is hand laid out, so the names on it are typed rather than derived. Rather than
    derive them and lose the layout, the names are checked against the real tables here: a command
    renamed or retired without this page being touched fails, which is the drift this project
    keeps finding.
    """
    from senbonzakura.parser import MODES, build_parser

    known = set(entry.DELEGATED) | set(entry.ALIASES) | set(MODES)
    flags = {s for a in build_parser(full=True)._actions for s in a.option_strings}
    flags.add("-i")

    # The table, not the rendered page. The masthead's `senbonzakura 0.4.0` is a version line
    # rather than an invocation, and reading it as one is how a guard ends up asserting about the
    # wrong text.
    for _group, rows in entry.WAYS_IN:
        for command, _note in rows:
            word = command.split()[1]
            assert word in known or word in flags, (
                f"the front door offers `senbonzakura {word}`, which is neither a command nor a "
                f"flag this tool has. Somebody following it gets a refusal from the page that "
                f"exists to stop that happening.")


def test_i_reaches_the_guided_mode_and_nothing_else(monkeypatch):
    """The page promises `-i`. This is the proof it is wired, not just printed."""
    seen = {}

    def fake_dispatch(name):
        seen["name"] = name
        return lambda rest: 0

    monkeypatch.setattr(entry, "dispatch", fake_dispatch)
    assert entry.main(["-i"]) == 0
    assert seen["name"] == "interactive"


def test_the_descriptions_share_one_edge():
    """ONE DESCRIPTION COLUMN, where there is room for one. The draft before this padded each line
    to its own command's width, so three columns started at three depths and every line had to be
    read on its own.
    """
    wide = entry.short_help(columns=100)
    starts = {m.start(2) for line in wide.splitlines()
              if (m := re.match(r"^(\s+senbonzakura \S+.*?\s{2,})(\S)", line))}
    assert len(starts) == 1, (
        f"the descriptions begin at {sorted(starts)}. A reader scanning the page has no single "
        f"edge to run down.")


@pytest.mark.parametrize("columns", [40, 52, 60, 79, 100, 160])
def test_the_page_fits_whatever_window_it_is_given(columns):
    """FOUND BY A JOURNEY, 2026-09-28, driving the real program in a fifty two column terminal.

    The page was one fixed block laid out for eighty columns, so in a half width window the thing
    written to be somebody's first screen was the thing that ran off the edge.
    """
    over = [line for line in entry.short_help(columns=columns).splitlines()
            if len(line) > columns and max((len(w) for w in line.split()), default=0) + 4 <= columns]
    assert not over, f"at {columns} columns these run past the edge: {over}"


def test_the_page_fits_an_eighty_column_terminal():
    over = [line for line in entry.short_help(columns=80).splitlines() if len(line) > 79]
    assert not over, f"these re-wrap on an 80-column terminal: {over}"


def test_it_says_which_version_answered(page):
    """The banner carries the wordmark and only ever draws to a terminal. Somebody piping this
    page into a bug report still needs the version, so the masthead is part of the page.
    """
    from senbonzakura._version import __version__
    assert __version__ in page
