# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Prose wraps. Commands and machine markers do not, and that is the whole design.

WHAT PROMPTED IT, 2026-09-27

An audit of all 66 user-facing surfaces against a built wheel found every argparse page clean, not
one line over 79 columns across 26 help pages, and every hand-written message ignoring the terminal:
a 355-character `doctor` advisory, a 352-character `--resume` refusal, a 349-character
`--gen-tokens` refusal. Good content, emitted as a wall.

THE TWO THINGS THAT MUST SURVIVE UNTOUCHED, which is why this file is longer than the module

**Commands.** These messages are full of runnable lines. A wrapped command cannot be pasted, so
"fixing" readability by breaking `senbonzakura baseline result.json --seeds 42` across two lines
destroys the most useful thing in the message.

**Markers.** CI greps nine needles out of this tool's output to decide whether a release ships,
`head-to-head` parses fields off those same lines, and a runpod bootstrap gates an unattended job on
them. Wrapping them would be a change to a machine interface disguised as a readability fix. The
operator chose this split explicitly, having been told the alternative could break the head-to-head
harness.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from senbonzakura import say

LONG = ("the capability probe would take roughly 4.2 hours on this CPU, 200 items at 512 tokens "
        "each, and that estimate comes from a measurement of a 1.7B model so a larger one is worse")


def test_prose_is_wrapped_to_the_ceiling():
    out = say.lines(LONG, columns=79)
    assert len(out) > 1, "a 180-character sentence came back as one line"
    assert all(len(ln) <= 79 for ln in out), [ln for ln in out if len(ln) > 79]


def test_no_word_is_lost_or_duplicated_by_wrapping():
    """The property that matters more than the line lengths."""
    assert " ".join(" ".join(say.lines(LONG, columns=79)).split()) == " ".join(LONG.split())


@pytest.mark.parametrize("command", [
    "    senbonzakura baseline result.json --seeds 42 --out recorded.json",
    "  senbonzakura compass --model Qwen/Qwen3-0.6B --harmful default/bad_eval_ds --out c.json",
    "    pip install 'senbonzakura[completion]'",
])
def test_an_indented_command_is_passed_through_byte_for_byte(command):
    """A wrapped command cannot be pasted, which destroys the useful half of the message."""
    assert say.lines(command, columns=79) == [command], (
        "an indented line was altered. In every message in this tool, indentation marks an example "
        "or a command, and both have to survive exactly.")


@pytest.mark.parametrize("marker", [
    "MARGIN_DONE auc=0.8062 ci=[0.6904,0.9067] mean_h=1.723 mean_l=1.344 says_harmful_h=100.0%",
    "SCORE_DONE refusals=0.0625 soft=0.1250 n=32",
    "DRIFT_DONE kl=0.4076",
    "MARGIN_NULLS strongest=uppercase_ratio_auc=0.2573 against compass=0.8062",
    "MARGIN_READOUT_SUSPECT the verdict token is not where the compass reads it",
    "DONE",
])
def test_a_machine_marker_is_never_wrapped(marker):
    """CI greps these, head-to-head parses fields off them, a runpod bootstrap gates on them."""
    assert say.lines(marker, columns=40) == [marker], (
        f"{marker.split()[0]} was altered. That is a machine interface, and changing it under the "
        f"heading of readability is how a grep starts finding nothing.")


def test_a_marker_is_recognised_by_shape_not_by_a_list():
    """So a marker added next year is exempt without anybody remembering this file."""
    assert say.is_marker("NEW_MARKER_NOBODY_HAS_WRITTEN_YET x=1")
    assert say.is_marker("DONE")
    assert say.is_marker("REFUSED")
    # An ordinary sentence must not be mistaken for one, or prose stops wrapping.
    assert not say.is_marker("The capability probe would take four hours.")
    assert not say.is_marker("A run needs a track.")
    assert not say.is_marker("Qwen/Qwen3-0.6B is not a local folder")


def test_blank_lines_survive():
    """Whitespace between paragraphs is half of what the audit asked for."""
    assert say.lines("first para\n\nsecond para", columns=79) == ["first para", "", "second para"]


def test_a_long_unbreakable_token_survives_rather_than_vanishing():
    """A path or URL longer than the width must still be printed; an empty result would lose it."""
    url = "https://huggingface.co/" + "a" * 120
    out = say.lines(url, columns=79)
    assert any(url in ln for ln in out), "a long token was dropped instead of overflowing"


def test_width_honours_columns_then_caps():
    import os
    old = os.environ.get("COLUMNS")
    try:
        os.environ["COLUMNS"] = "200"
        assert say.width() == say.CEILING, "a very wide terminal must still be capped"
        os.environ["COLUMNS"] = "50"
        assert say.width() == 50, "a narrow terminal must be honoured"
        os.environ["COLUMNS"] = "10"
        assert say.width() == 40, "an absurd width must fall back to something usable"
        os.environ["COLUMNS"] = "not a number"
        assert 40 <= say.width() <= say.CEILING
    finally:
        os.environ.pop("COLUMNS", None)
        if old is not None:
            os.environ["COLUMNS"] = old


def test_indent_applies_to_continuations():
    out = say.lines(LONG, indent="  ", columns=60)
    assert all(ln.startswith("  ") for ln in out[1:])


def test_refusal_puts_the_head_first_and_indents_the_body(capsys):
    say.refusal("senbonzakura: that will not work", LONG, columns=79)
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "senbonzakura: that will not work"
    assert all(ln.startswith("  ") for ln in printed[1:] if ln)
    assert all(len(ln) <= 79 for ln in printed)


def test_a_marker_inside_a_multi_line_body_is_still_exempt():
    """The realistic shape: a sentence, then a marker line, then more prose."""
    body = f"some prose that is quite long and will certainly need wrapping at any width\nDONE\n{LONG}"
    out = say.lines(body, columns=60)
    assert "DONE" in out, "the marker was altered when it sat between two prose blocks"


def test_the_module_stays_cheap_to_import():
    """It is used in refusal paths and by `doctor`, so it must not pull anything heavy.

    `--help` paying for a heavy import is a regression this project has already fixed once, taking
    it from 2.80s to 0.06s, and the way it came back would be a new module in the refusal path.
    """
    source = Path(say.__file__).read_text(encoding="utf-8")
    for heavy in ("import torch", "import transformers", "import optuna", "import datasets",
                  "import numpy"):
        assert heavy not in source, f"say.py imports {heavy}, which every refusal would now pay for"


# ── the bug the first version of the marker rule had ─────────────────────────────────────────────

@pytest.mark.parametrize("prefix", [
    ("NOTE: the refusal-separation filter rejected NONE of 40 candidate axes, so the filter is not "
     "filtering and the search had nothing to choose between"),
    ("WARNING: none of 64 scored trials was intact, so the knee was picked from trials that were "
     "all broken and the number means nothing"),
    ("BROKEN FILTER: all 40 candidate axes scored a refusal separation of zero, which cannot "
     "happen unless the statistic is being read off the wrong tensor"),
    ("MATCHING ACHIEVED NOTHING: the harmless prompts chosen as nearest neighbours were no nearer "
     "than the harmless set at large"),
])
def test_a_shouted_prose_prefix_is_not_a_marker(prefix):
    """The bug the first marker rule had, and it disabled the module on its own worst case.

    The rule was `^[A-Z]{4,}`, on the theory that shouting was enough. That makes `NOTE:` a machine
    marker, and the longest single message in the whole tool, an 808-character explanation of a
    filter that rejected nothing, opens with `NOTE:`. So the function written to wrap it passed it
    through untouched, and the very worst line in the audit would have survived the fix intact.

    These four prefixes are emphasis. Emphasis is prose.
    """
    assert not say.is_marker(prefix), f"{prefix.split(':')[0]!r} is emphasis, not a machine marker"
    out = say.lines(prefix, columns=79)
    assert len(out) > 1, "a shouted prose prefix stopped the line wrapping"
    assert all(len(ln) <= 79 for ln in out)


def test_the_bare_markers_are_a_named_set_rather_than_a_pattern():
    """`DONE` has no underscore and is the needle the end-to-end smoke greps for."""
    assert "DONE" in say.BARE_MARKERS
    for bare in say.BARE_MARKERS:
        assert say.is_marker(bare)
        assert "_" not in bare, (
            f"{bare} contains an underscore, so the pattern already covers it and listing it here "
            f"is a second copy of the same rule")


# ── reflow, refusal_text and the printing helpers ─────────────────────────────────────────────────

REFLOWABLE = (
    "this looks like a summary rather than a single measurement: it carries `stages` and no "
    "`metrics` block.\n"
    "\n"
    "That is correct for what it is. It runs several instruments and writes one document describing "
    "all of them, so there is no single figure in it to record.\n"
    "\n"
    "    senbonzakura baseline coherence.json --seeds 42 --out b.json"
)


def test_reflow_wraps_prose_paragraphs_and_keeps_the_command():
    out = say.reflow(REFLOWABLE, columns=79)
    assert all(len(ln) <= 79 for ln in out), [ln for ln in out if len(ln) > 79]
    assert "    senbonzakura baseline coherence.json --seeds 42 --out b.json" in out, (
        "the four-space-indented command was altered, so it can no longer be pasted")
    assert "" in out, "the blank lines between paragraphs were lost"


def test_reflow_joins_a_two_space_continuation_into_one_paragraph():
    """A two-space indent means "this sentence continues", which is how these messages are written.

    Handed to `lines` directly, every such line counts as indented and nothing wraps at all. That is
    the defect this function exists for, and a 261-character line proved it in `baseline`.
    """
    out = say.reflow("first sentence here\n  and its continuation\n  and more of it", columns=79)
    assert len(out) == 1, f"a continuation was not joined into its paragraph: {out}"
    assert "first sentence here and its continuation and more of it" in out[0]
    assert out[0].startswith("  "), "reflow gives prose the two-space body indent"


def test_reflow_keeps_a_four_space_line_apart_from_the_paragraph_before_it():
    """Prose gets the two-space body indent; the command keeps its own four and nothing else."""
    out = say.reflow("a sentence\n    a command --flag", columns=79)
    assert out == ["  a sentence", "    a command --flag"]


def test_reflow_leaves_a_marker_alone_even_unindented():
    out = say.reflow("prose before\nMARGIN_DONE auc=0.8 ci=[0.7,0.9]\nprose after", columns=40)
    assert "MARGIN_DONE auc=0.8 ci=[0.7,0.9]" in out


def test_refusal_text_is_a_head_then_indented_paragraphs():
    text = say.refusal_text("senbonzakura: no.", "because of this reason which is fairly long and "
                            "will need to be wrapped at any sensible width at all", columns=60)
    got = text.split("\n")
    assert got[0] == "senbonzakura: no."
    assert got[1] == "", "there is no blank line between the head and the reason"
    assert all(len(ln) <= 60 for ln in got)
    assert all(ln.startswith("  ") for ln in got[2:] if ln)


def test_refusal_text_with_no_paragraphs_is_just_the_head():
    assert say.refusal_text("senbonzakura: no.") == "senbonzakura: no."


def test_say_prints_through_the_log_it_is_given():
    got = []
    say.say(LONG, indent="  ", log=got.append, columns=60)
    assert len(got) > 1 and all(len(ln) <= 60 for ln in got)


def test_say_defaults_to_print(capsys):
    say.say("a short line", columns=79)
    assert capsys.readouterr().out.strip() == "a short line"


def test_is_verbatim_covers_both_reasons():
    assert say.is_verbatim("    indented")
    assert say.is_verbatim("MARGIN_DONE x=1")
    assert not say.is_verbatim("ordinary prose")


def test_width_falls_back_when_the_terminal_cannot_be_measured(monkeypatch):
    """`shutil.get_terminal_size` raising must not take a refusal path down with it."""
    import shutil as _shutil
    monkeypatch.delenv("COLUMNS", raising=False)

    def _boom(fallback=(80, 24)):
        raise OSError("no terminal here")

    monkeypatch.setattr(_shutil, "get_terminal_size", _boom)
    assert 40 <= say.width() <= say.CEILING


# ── shorten: a cut value says it was cut ───────────────────────────────────────────────────────────

class TestShortenSaysItShortened:
    """A value trimmed to fit a column read as complete, which is how a path loses a component.

    WHAT PROMPTED IT, 2026-09-27

    A surface audit found values cut to a column budget with nothing to mark the cut: a path ended
    mid-component and a GPU's identifier ended mid-string, and both looked like the whole value. A
    reader copies a cut path. Worse, two cards whose names agree for sixty characters become one
    card to anybody shown neither the difference nor a sign that something was removed.
    """

    def test_a_value_that_fits_is_returned_unchanged(self):
        assert say.shorten("short", 20) == "short"

    def test_a_value_exactly_at_the_limit_is_not_cut(self):
        """The off-by-one that would put an ellipsis on a value that fitted."""
        assert say.shorten("abcde", 5) == "abcde"

    def test_a_longer_value_is_cut_to_the_limit_and_marked(self):
        got = say.shorten("abcdefghij", 8)
        assert got == "abcde" + say.CUT
        assert len(got) == 8, "the result overflows the budget it was given"

    def test_the_marker_is_ascii_because_this_goes_through_logs(self):
        """A single ellipsis character breaks in a terminal whose encoding we do not choose."""
        assert say.CUT.isascii() and say.CUT == "..."

    def test_a_limit_too_small_to_mark_returns_the_value_whole(self):
        """At that width there is nothing to report, so the layout gives rather than the truth.

        The alternative is returning two dots and no content, which tells the reader a value
        exists and nothing about it.
        """
        assert say.shorten("abcdefghij", 3) == "abcdefghij"
        assert say.shorten("abcdefghij", 0) == "abcdefghij"

    def test_a_non_string_is_shortened_rather_than_raising(self):
        """Callers pass whatever a check produced, and a crash in the formatter loses the report."""
        assert say.shorten(1234567890, 6) == "123" + say.CUT


# ── some_of: a cut list never contradicts its own count ────────────────────────────────────────────

class TestSomeOfNamesWhatItLeftOut:
    """A count in one clause and a shorter list in the next leaves the reader to guess which lies.

    WHAT PROMPTED IT, 2026-09-27

    `doctor` reported "7 failed to import:" and then named six of them. `convert` did the same with
    five. A JSON object with twenty keys was described by eight of them with no sign of the rest.
    Five other places in the package already said "and N more" by hand, so this is the ninth site
    and the first shared one.
    """

    def test_a_short_list_is_named_in_full_with_no_suffix(self):
        assert say.some_of(["a", "b"], 6) == "a, b"

    def test_a_list_exactly_at_the_limit_is_not_described_as_truncated(self):
        assert say.some_of(list("abcdef"), 6) == "a, b, c, d, e, f"

    def test_a_longer_list_names_the_limit_and_counts_the_rest(self):
        assert say.some_of(list("abcdefghij"), 6) == "a, b, c, d, e, f, and 4 more"

    def test_the_count_comes_from_the_items_rather_than_the_caller(self):
        """The half that was getting lost: the total and the list came from different expressions."""
        got = say.some_of(list(range(20)), 3)
        assert got.endswith("and 17 more"), got
        assert len([p for p in got.split(", ") if p.isdigit()]) == 3

    def test_an_empty_list_is_empty_rather_than_a_claim_about_nothing(self):
        assert say.some_of([], 6) == ""

    def test_items_that_are_not_strings_are_named_rather_than_crashing(self):
        """Callers pass whatever a check collected, and a formatter that raises loses the report."""
        assert say.some_of([1, None, 2.5], 6) == "1, None, 2.5"

    def test_the_joiner_can_be_changed_without_losing_the_count(self):
        assert say.some_of(list("abcd"), 2, joiner=" | ") == "a | b, and 2 more"
