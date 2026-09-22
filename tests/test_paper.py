# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The JOSS paper, held to the things that get a submission bounced or a claim wrong.

Two different kinds of assertion live here and the distinction matters. The word count is a
SUBMISSION requirement: JOSS asks for 250 to 1000 words and this project targets 700, so a paper
that drifts over is a paper that gets sent back. The others are CORRECTNESS requirements, and
they exist because this file makes claims about somebody else's work under a licence that
requires those claims to be right.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: The operator's target, 2026-09-22. Tighter than JOSS's own ceiling of 1000 on purpose: the
#: paper was 1,117 words when the panel finished with it, and a limit set at the outer bound is a
#: limit that gets met by trimming the caveats rather than the prose.
MAX_WORDS = 700


def _body():
    text = (ROOT / "paper.md").read_text(encoding="utf-8")
    assert text.startswith("---"), "paper.md carries no YAML frontmatter, which JOSS requires"
    return text.split("---", 2)[2]


def _flat(text):
    """Whitespace collapsed, because the prose is wrapped and a phrase lands across two lines.

    A line-based version of these assertions fired on correct text the first time it ran, which
    is the second time in one day that a guard over wrapped prose was written the naive way.
    """
    return " ".join(text.split())


def test_the_paper_is_within_the_word_target():
    words = len(re.findall(r"\S+", _body()))
    assert words <= MAX_WORDS, (
        f"paper.md is {words} words against a target of {MAX_WORDS}. Trim the prose rather than "
        f"the caveats: the ceiling paragraph and the withdrawn figures are the reason this paper "
        f"is worth submitting")


def test_the_paper_still_states_the_ceiling():
    """The honest ceiling is the first thing a word limit tempts you to cut, and it is the part
    that must survive. A paper claiming a measurement tool without saying where the tool stops
    working is the thing this project exists to argue against.
    """
    body = _flat(_body())
    assert "above 3B parameters" in body, "the paper no longer states the size ceiling"
    assert "Gemma figure is withdrawn" in body, "the paper no longer states the withdrawal"
    assert "length-only control" in body, "the paper no longer names the null control"


def test_the_paper_describes_the_borrowed_code_the_way_the_notices_do():
    """THE CLAIM THAT MUST MATCH THE LICENCE FILE.

    `THIRD-PARTY-NOTICES.md` was corrected on 2026-09-22: the keyword marker LIST is verbatim
    from Heretic and `_heretic_norm` is an adaptation, not a copy. The paper said the metric was
    used verbatim, and rewriting the paper is exactly the moment that correction gets undone by
    somebody reaching for a shorter phrase.

    Section 5(a) asks a modified work to say it was modified, so understating the modification is
    the one direction of error this cannot afford.
    """
    body = _flat(_body())
    assert "keyword marker list verbatim" in body, (
        "the paper no longer scopes 'verbatim' to the marker list. Heretic has no function called "
        "_heretic_norm; ours is an adaptation of its _is_match, and THIRD-PARTY-NOTICES.md says so")
    assert "adapts" in body or "adapted" in body, (
        "the paper no longer says the normalisation is adapted")


@pytest.mark.parametrize("claim", ["0.9887", "0.6564", "0.6616"])
def test_the_published_figures_in_the_paper_match_the_readme(claim):
    """One number, one value, wherever it is written. The README carries the same three compass
    figures, and a paper quoting a different one from the front page is the drift this project
    has already corrected twice in its own documentation.
    """
    assert claim in _flat(_body()), f"{claim} is no longer in the paper"
    assert claim in _flat((ROOT / "README.md").read_text(encoding="utf-8")), (
        f"{claim} is in the paper and not in the README")
