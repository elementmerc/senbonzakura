# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A worked example whose prose disagrees with the output printed beside it.

WHAT HAPPENED

`docs/guide/compass.md` shows a real command, its real output, and an argument built on that
output: a null ruler reading nothing but average word length "scores 1.0000 on this data, which is
exactly what the compass scored".

The padding-mask fix in `5fcd4c0` moved the compass to 0.9826. Nothing pinned the figures, so the
block went on printing 1.0000 and the sentence went on saying "exactly what the compass scored"
while the numbers directly above it no longer said that. The null now BEATS the compass, which is
a stronger version of the same lesson told in words that had stopped matching.

A published result would be worse. This is a documentation example, and it is still the first thing
a reader is invited to run, so a reproduction that does not reproduce teaches them to distrust the
next one.

WHY NOT JUST RE-RUN IT IN CI

The command downloads a 135M model. Doing that in every job on every platform to check a
documentation example is the wrong trade, and it would make the docs job depend on the network. So
this pins the property that actually broke: the prose and the numbers beside it agreeing. Re-running
the example against the real model is worth doing at release time and is recorded in DEFERRED.md.
"""
import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "docs" / "guide" / "compass.md"
TEXT = PAGE.read_text(encoding="utf-8")


def figure(pattern):
    m = re.search(pattern, TEXT)
    assert m, f"the documented output no longer contains {pattern!r}"
    return float(m.group(1))


def test_the_page_still_shows_the_worked_example():
    assert "MARGIN_DONE" in TEXT and "MARGIN_NULLS" in TEXT


def test_the_null_and_the_compass_are_described_as_the_numbers_have_them():
    """THE SENTENCE THAT WENT STALE, and the one thing a reader takes away from the page.

    Whether the strongest null MATCHES the compass or BEATS it changes what the example teaches,
    and the words for those two are different. Read from the block rather than assumed, so the day
    the block changes again this fails instead of quietly disagreeing with itself.
    """
    compass = figure(r"MARGIN_NULLS\s+strongest=\w+_auc=[\d.]+ against compass=([\d.]+)")
    null = figure(r"MARGIN_NULLS\s+strongest=\w+_auc=([\d.]+)")
    # The CLAIM paragraph only, which ends where the correction box starts. The box quotes the old
    # wording to explain what changed, and a window that swallowed it would flag the explanation as
    # the error it explains.
    prose = TEXT.split("The null panel is the loudest.")[1].split(":::")[0]

    # COMPARED AS DISTANCE FROM CHANCE, not as magnitude, which is what this read before 2026-10-02
    # and is a narrower question than the one being asked. An AUC of 0.5 is a coin flip and 0.0 is
    # a perfect separation in the other direction, so a null at 0.0 beats a compass at 0.9653 on
    # the only axis that means anything, and a raw `null > compass` reads it as the weakest result
    # on the board. The documented block has two rulers at exactly 0.0000, so this is the live
    # case rather than a hypothetical: the day the tie-break names one of them, the old comparison
    # would have failed the build with the opposite explanation.
    null_strength, compass_strength = abs(null - 0.5), abs(compass - 0.5)

    if null_strength > compass_strength:
        assert "better than the compass" in prose, (
            f"the strongest null scores {null} against the compass's {compass}, which is further "
            f"from chance, so it BEATS it and the prose must not say the two matched")
        assert "exactly what the compass scored" not in prose
    elif null_strength == compass_strength:
        assert "exactly what the compass scored" in prose
    else:
        pytest.fail(f"the strongest null ({null}, {null_strength:.4f} from chance) is nearer "
                    f"chance than the compass ({compass}, {compass_strength:.4f}); the page's "
                    f"whole argument is that a surface ruler matches or beats it")


def test_the_headline_count_matches_the_lines_that_justify_it():
    """"Four separate lines saying so" has to be four lines that are actually printed.

    It counts MARKER lines, which is why the `in words:` block added on 2026-09-27 does not change
    the number: that block is prose, printed after them, and saying the same thing in a sentence.
    """
    m = re.search(r"\*\*A near-perfect score you should not believe, and (\w+) separate lines",
                  TEXT)
    assert m, "the sentence introducing the diagnostic lines has changed shape"
    words = {"two": 2, "three": 3, "four": 4, "five": 5}
    claimed = words[m.group(1)]
    # COUNTED INSIDE THE OUTPUT BLOCK, not across the whole page. This read `TEXT` behind a dead
    # `if False`, so a marker merely NAMED in the surrounding prose counted as a line printed by the
    # command, and the guard would have agreed with a page that claimed four while showing three.
    # Fixed 2026-09-27 while adding the `in words:` block to the same example.
    fenced = [chunk for chunk in TEXT.split("```") if "MARGIN_DONE" in chunk]
    assert len(fenced) == 1, (
        f"expected exactly one fenced block carrying the compass output, found {len(fenced)}")
    block = fenced[0]
    printed = sum(1 for marker in ("MARGIN_CONTROLS", "MARGIN_NULLS", "MARGIN_READOUT ",
                                   "MARGIN_READOUT_SUSPECT")
                  if marker in block)
    assert claimed == printed, (
        f"the page claims {claimed} diagnostic lines and the output block shows {printed}")


def test_the_figures_say_which_environment_produced_them():
    """A date is not enough. "Measured twice, byte-identical" was true and still misled.

    The page used to carry that sentence with a date, and a review pass on 2026-09-10 ran the same
    command against the same cached model snapshot and got a different number, because the
    environment had moved to transformers 5.14.1. Both readings were real measurements. Neither
    said what it was read on, and a figure that does not name its environment cannot be checked by
    anyone who is not standing on the machine that produced it.
    """
    assert "constraints/ci.txt" in TEXT, (
        "the worked example must name the environment its figures were measured under, because "
        "the AUC moves with the transformers version and a bare number cannot be reproduced")
    assert re.search(r"transformers \d+\.\d+", TEXT), (
        "name the actual version, not just the constraints file")


def test_the_smoke_checks_this_page_against_the_tool():
    """The prose-versus-numbers checks in this file cannot catch the numbers going stale.

    They compare the page to itself. What closes the loop is `tools/ci/smoke_end_to_end.py`, which
    runs the command the page prints and compares the AUC it gets to the AUC the page promises.
    This asserts that check still points at this page, so removing it is a deliberate act rather
    than something that quietly stops happening.
    """
    smoke = (Path(__file__).resolve().parent.parent / "tools" / "ci" / "smoke_end_to_end.py"
             ).read_text(encoding="utf-8")
    assert "documented_compass_auc" in smoke
    assert "compass.md" in smoke, "the smoke must still be reading this page's figure"
