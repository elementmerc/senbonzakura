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


# ── the caution that has to survive the next pass over this output ─────────────────
#
# Added 2026-10-10. `--inspect` is the one surface that puts a model's answers to harmful prompts
# on a screen. The acceptable-use policy says no completions are distributed with this tool, which
# is true of what ships and not of what this command makes, and until now nothing said so at the
# point the text appeared. A policy file the reader of this output may never have opened is not a
# disclosure to that reader.
#
# This is a test rather than a comment because the failure mode is specific and has happened twice
# in this project: somebody shortens a block of output for readability and takes a line of
# obligation out with the padding. The assertions below are about what the caution has to SAY, not
# how it is worded, so it can be rewritten without tripping them.

def test_the_inspect_caution_says_what_is_about_to_appear():
    from senbonzakura import cli
    text = cli.INSPECT_CAUTION.lower()
    assert "harmful" in text, "the caution has to name what the prompts are"
    assert "answers" in text or "completions" in text, (
        "the caution has to say these are the MODEL'S OUTPUT, not the prompt set. The prompts "
        "ship with the tool and the answers do not, which is the whole reason this needs saying")


def test_the_inspect_caution_says_the_text_is_not_saved_but_is_in_the_scrollback():
    from senbonzakura import cli
    text = cli.INSPECT_CAUTION.lower()
    assert "not saved" in text or "nothing here is saved" in text
    assert "scrollback" in text or "screen" in text, (
        "'nothing is saved' alone reads as reassurance. The point is that it is still on the "
        "screen and in the terminal's history, which is where somebody else may read it")


def test_the_inspect_caution_raises_the_shared_machine_question():
    from senbonzakura import cli
    text = cli.INSPECT_CAUTION.lower()
    assert "shared" in text or "institutional" in text, (
        "the acceptable-use policy asks this question about the corpus landing on a shared "
        "machine; the generations are the stronger case and the caution has to ask it too")


def test_the_caution_is_printed_before_the_first_harmful_block():
    """Order matters: a caution after the text is a caption, not a warning.

    Asserted on the source rather than by running the command, because reaching that code path
    needs a model on a card. A source-order check cannot prove the runtime order in general, and
    here the two statements sit in one straight-line function with nothing between them, so it
    does.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "senbonzakura" / "cli.py").read_text()
    caution_at = src.find("print(INSPECT_CAUTION)")
    harmful_at = src.find('show("HARMFUL"')
    assert caution_at != -1, "the caution is no longer printed anywhere"
    assert harmful_at != -1, "the harmful block is no longer shown, so this test needs rewriting"
    assert caution_at < harmful_at, (
        "the caution is printed AFTER the harmful generations, so the reader meets the text "
        "first and the warning second")
