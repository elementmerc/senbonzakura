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
