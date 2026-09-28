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


# ── the half of these commands that goes to the other stream ──────────────────────────────────────

@pytest.mark.parametrize("mode", MODES)
def test_a_mode_word_is_not_announced_as_a_model_id(mode):
    """WHAT PROMPTED IT, 2026-09-28. Eight tests drove this command and all of them read stdout.

    `_read_as_a_model` excluded the literal "abliterate", so `kageyoshi` and `auto` were not
    excluded and every invocation of either printed "reading `kageyoshi` as a model id, since it is
    not a command" to stderr. It is false: `split_mode` peels the word off as a mode a few lines
    later. `--help` calls `kageyoshi` the recommended way to run this tool, and `auto` exists for
    somebody meeting it for the first time, so the two words most likely to be a reader's first
    contact were the two that told them the tool had misunderstood.

    Nothing caught it because the notice goes to stderr by design, to leave a captured stdout
    unchanged, and every assertion in this file reads `out.stdout`. A guard that inspects one stream
    of a command that writes two reports clean on everything in the other.
    """
    # Deliberately NOT through `_run`, which returns `out.stdout` and drops the rest. That helper is
    # how eight tests came to drive this exact command without ever seeing what it wrote here.
    out = subprocess.run([sys.executable, "-m", "senbonzakura", mode, "--help"],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         check=False, timeout=300)
    assert out.returncode == 0, f"{mode} --help exited {out.returncode}"
    assert "as a model id" not in out.stderr, (
        f"`senbonzakura {mode} --help` claims the mode word is being read as a model:\n"
        f"{out.stderr}")
    assert out.stderr.strip() == "", (
        f"`senbonzakura {mode} --help` wrote to stderr, and a help page is not an error:\n"
        f"{out.stderr}")


def test_the_notice_still_fires_on_a_word_that_really_is_a_model():
    """The exclusion must not have been widened into silence.

    `senbonzakura frobnicate` is read as `--model frobnicate`, and saying so before anything
    expensive happens is the whole point of the notice. If this stops firing, the fix above has
    turned a false message into no message.
    """
    from senbonzakura import entry

    assert entry._read_as_a_model("frobnicate"), (
        "nothing is said about a bare word being taken as a model id any more")


def test_every_mode_word_can_be_suggested_for_a_typo():
    """`known` is what a near miss is matched against, and two of three modes were not in it.

    So a reader who typed `kageyosi` got no suggestion and a fetch attempt for a model of that
    name, for the command the help page recommends.
    """
    import difflib

    from senbonzakura import entry
    from senbonzakura.parser import MODES

    for mode in MODES:
        typo = mode[:-2] + mode[-1]
        said = entry._not_a_command(typo)
        assert said and mode in said, (
            f"`{typo}` produced no suggestion of `{mode}`: {said!r}")
        assert difflib.get_close_matches(typo, [mode], n=1, cutoff=0.8), (
            f"the typo {typo!r} is too far from {mode!r} for this test to be measuring anything")
