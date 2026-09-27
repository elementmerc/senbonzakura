# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Asking about a mode word has to answer about that mode word.

WHAT PROMPTED IT, 2026-09-27

An output review ran the tool as a user and checksummed what it printed. `senbonzakura --help`,
`abliterate --help`, `kageyoshi --help` and `auto --help` were the same page byte for byte, md5
3f42c38e4d8087ee9d23697eda0576aa on all four. The page opened `usage: senbonzakura [-h] ...`, never
named the word that had been typed, and ended with the whole command list. So `kageyoshi --help`
answered a question about kageyoshi with a page that mentions it once, in a list of everything else,
and the difference between the three modes, which is the only reason the words exist, appeared
nowhere.

WHAT THESE TESTS HOLD

That each mode word prints a page naming itself; that the pages differ from each other and from the
top-level one; that the page which claims the preset stands down from hand-set flags lists which
flags those are; and that the list matches the one the preset actually reads. That last one is the
load-bearing test: the list lives in `parser.py`, which may not import `cli` because that costs
torch, so the two cannot be the same object and a duplicate that drifts is a help page lying about
behaviour.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest

from senbonzakura.parser import KAGEYOSHI_PRESET_FLAGS, MODES, mode_help

#: The widest terminal a page here is allowed to assume. Anything past it re-wraps into ragged
#: half-lines on an 80-column terminal, which is the default nearly everywhere.
WIDTH = 79


def _run(*argv):
    out = subprocess.run([sys.executable, "-m", "senbonzakura", *argv],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         check=False, timeout=300)
    assert out.returncode == 0, f"{argv} exited {out.returncode}\n{out.stderr[:400]}"
    return out.stdout


@pytest.mark.parametrize("mode", MODES)
def test_the_usage_line_names_the_word_that_was_typed(mode):
    first = _run(mode, "--help").splitlines()[0]
    assert first.startswith(f"usage: senbonzakura {mode}"), (
        f"`senbonzakura {mode} --help` opens with:\n  {first}\n"
        f"which is not the command that was typed to get it")


def test_the_four_pages_are_four_pages():
    """The original finding, kept as arithmetic: four identical checksums."""
    pages = {name: hashlib.md5(_run(*argv).encode()).hexdigest()   # noqa: S324 - not a credential
             for name, argv in (("top", ("--help",)), ("abliterate", ("abliterate", "--help")),
                                ("kageyoshi", ("kageyoshi", "--help")),
                                ("auto", ("auto", "--help")))}
    assert len(set(pages.values())) == 4, (
        f"two of these pages are the same document: {pages}")


@pytest.mark.parametrize("mode", MODES)
def test_the_page_points_at_the_full_flag_set(mode):
    """Each short page is a summary, so it has to say where the rest is."""
    assert "--help-all" in _run(mode, "--help")


@pytest.mark.parametrize("mode", MODES)
def test_the_page_carries_a_command_somebody_could_type(mode):
    text = _run(mode, "--help")
    assert f"senbonzakura {mode} Qwen" in text, (
        "the page has no worked example, which is the line people copy")


@pytest.mark.parametrize("mode", MODES)
def test_no_line_assumes_a_wide_terminal(mode):
    over = [ln for ln in mode_help(mode).splitlines() if len(ln) > WIDTH]
    assert not over, f"these lines are wider than {WIDTH} columns: {over}"


@pytest.mark.parametrize("mode", ["kageyoshi", "auto"])
def test_the_preset_page_lists_the_flags_it_stands_down_from(mode):
    """The top-level help promises that a hand-set flag wins and never says which flags.

    "Anything you set by hand WINS: the preset skips that knob and says so in the log" is only
    actionable if a reader can find out what the knobs are.
    """
    text = _run(mode, "--help")
    missing = [f for f in KAGEYOSHI_PRESET_FLAGS if f not in text]
    assert not missing, f"the page does not name {missing}"


def test_the_listed_flags_are_the_flags_the_preset_reads():
    """The duplicate that cannot be an import, pinned to the original.

    `parser.py` may not import `cli`: torch at module scope is 2.8 seconds and the whole reason
    that module was split out. So the list is copied, and a copy with no test is a help page that
    goes stale silently.
    """
    from senbonzakura.cli import _KAGEYOSHI_BUDGET_FLAGS

    actual = {flag for flags in _KAGEYOSHI_BUDGET_FLAGS.values() for flag in flags}
    assert set(KAGEYOSHI_PRESET_FLAGS) == actual, (
        "parser.KAGEYOSHI_PRESET_FLAGS and cli._KAGEYOSHI_BUDGET_FLAGS disagree about which flags "
        "the preset chooses, so `kageyoshi --help` describes a preset that is not the one that "
        f"runs. Only in the help: {sorted(set(KAGEYOSHI_PRESET_FLAGS) - actual)}. Only in the "
        f"preset: {sorted(actual - set(KAGEYOSHI_PRESET_FLAGS))}")


class TestAnAliasSaysThatItIsOne:
    """The same finding one door along: `harm-recognition --help` printed `compass --help` byte for
    byte, a page whose usage line, examples and artefact names all say `compass`, handed to somebody
    who typed a different word. Resolving the alias is right; doing it silently is what is not.
    """

    def test_the_note_names_both_words(self):
        from senbonzakura.entry import ALIASES

        for alias, real in ALIASES.items():
            text = _run(alias, "--help")
            assert f"`{alias}` is another name for `{real}`" in text, text

    def test_the_page_is_no_longer_identical_to_the_one_it_stands_for(self):
        from senbonzakura.entry import ALIASES

        for alias, real in ALIASES.items():
            assert _run(alias, "--help") != _run(real, "--help")

    def test_nothing_but_the_help_path_changed(self):
        """An extra line on every invocation would break captured logs and stdout checks."""
        from senbonzakura.entry import ALIASES

        for alias in ALIASES:
            out = subprocess.run([sys.executable, "-m", "senbonzakura", alias, "--no-such-flag"],
                                 capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                 check=False, timeout=300)
            assert "another name for" not in out.stdout + out.stderr


def test_help_all_still_reaches_the_flags_through_a_mode_word():
    """`--help` is intercepted for a mode word and `--help-all` must not be."""
    text = _run("kageyoshi", "--help-all")
    assert "--separation-statistic" in text, "the mode intercept swallowed --help-all"
