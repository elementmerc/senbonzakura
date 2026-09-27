# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`--inspect` marks a generation it shortened, because an unmarked cut reads as a broken model.

WHAT PROMPTED IT, 2026-09-27

Audit finding 15, the occurrence nobody had reached: values cut to a column budget with no ellipsis.
The one that had been fixed was a GPU's name in `senbonzakura setup`, where the string is the card's
whole `nvidia-smi` line and the cut landed inside its UUID. This is the other one that matters, and
it is worse, because of what the surface is for.

`--inspect` exists precisely because a KL number cannot tell "wrecked" from "a few benign first
tokens flipped": it prints the real text before and after one ablation window so a person can read
it and judge. It printed `a[:220]`. So a generation longer than 220 characters was shown stopping
mid-word with nothing to say it had been shortened, which is exactly what a wrecked model looks like.
The instrument built to judge coherence was manufacturing the evidence against it.

THE RULE

A value this tool shortens says it was shortened, and the prompt and both generations survive intact
when they already fit. A reader must never have to guess whether the model stopped or the display
did.
"""
from __future__ import annotations

from senbonzakura import cli, say

LONG = "Explain in detail, and at very great length indeed, the following: " + "wibble " * 200
SHORT = "a short answer."


def _block(prompts, pre, post):
    return cli.inspect_lines("HARMFUL", prompts, pre, post)


def test_a_shortened_generation_is_marked_as_shortened():
    lines = _block([SHORT], [LONG], [SHORT])
    pre = next(ln for ln in lines if ln.startswith("  PRE :"))
    assert say.CUT in pre, (
        f"a generation was cut with nothing to show it, which reads as a model that stopped: {pre!r}")


def test_a_shortened_prompt_is_marked_as_shortened():
    lines = _block([LONG], [SHORT], [SHORT])
    head = next(ln for ln in lines if "###" in ln)
    assert say.CUT in head


def test_a_generation_that_fits_is_shown_whole_and_unmarked():
    """The other half of the property: a marker that always appears carries no information."""
    lines = _block([SHORT], ["a perfectly ordinary reply."], [SHORT])
    pre = next(ln for ln in lines if ln.startswith("  PRE :"))
    assert "a perfectly ordinary reply." in pre
    assert say.CUT not in pre


def test_both_sides_of_the_edit_are_shown_for_every_prompt():
    """The comparison is the point; a block missing one side of it cannot be read at all."""
    lines = _block([SHORT, SHORT], [SHORT, SHORT], [SHORT, SHORT])
    assert sum(1 for ln in lines if ln.startswith("  PRE :")) == 2
    assert sum(1 for ln in lines if ln.startswith("  POST:")) == 2


def test_nothing_shown_exceeds_the_budget_it_was_given():
    """A marker added past the limit would just move the wall one character further out."""
    lines = _block([LONG], [LONG], [LONG])
    head = next(ln for ln in lines if "###" in ln)
    assert len(head.split(": ", 1)[1]) <= cli.INSPECT_PROMPT_CHARS
    for ln in lines:
        if ln.startswith(("  PRE :", "  POST:")):
            # eval() of a repr is not run here; the quoted length is what is compared.
            assert len(ln) <= len("  POST: ") + cli.INSPECT_TEXT_CHARS + 2
