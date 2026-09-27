# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Three machine markers and no sentence is a reading nobody can act on.

WHAT PROMPTED IT, 2026-09-27

A surface audit ran `senbonzakura compass` and got `MARGIN_DONE`, `MARGIN_CONTROLS` and
`MARGIN_NULLS`, and nothing else. Every figure a reader needs was there; none of it said what an AUC
of 0.8062 MEANS, at what confidence, or whether it passed. The instrument's primary human output was
a machine interface.

The markers stay exactly as they are, by the operator's decision on 2026-09-27: they are parsed, by
`docs/guide/compass.md`'s own documented figures among other things, and their spacing is load
bearing. The sentences come after them.

THE RULE THIS FILE ENFORCES

The verdict is READ OFF the run's own flags and never recomputed. `self_invalidated` is already set
by the readout check and the null check, so the words cannot disagree with the markers they follow.
A summary that re-derives a verdict is a second source of truth about whether the figure is usable,
and the reason those checks exist at all is that this project shipped a correct diagnostic wired to
no consequence, twice.
"""
from __future__ import annotations

import pytest

from senbonzakura import margin


def _said(res, score):
    return " ".join(" ".join(margin.in_words(res, score)).split())


def test_the_number_is_explained_in_terms_of_what_it_ranks():
    """An AUC is a ranking statistic and almost nobody reads it as one without being told."""
    said = _said({"auc_ci": [0.72, 0.88]}, 0.8062)
    assert "0.8062" in said
    assert "80.6%" in said, "the figure is not restated as the plain-language quantity it is"
    assert "harmful one higher" in said
    assert "0.5 is a coin toss" in said, "nothing tells the reader where chance sits"


def test_the_confidence_level_is_named_rather_than_implied():
    """A bare interval leaves the reader to guess whether it is 90, 95 or 99 per cent."""
    said = _said({"auc_ci": [0.72, 0.88]}, 0.8062)
    assert margin.CONFIDENCE in said
    assert "0.7200" in said and "0.8800" in said
    assert "bootstrap" in said, "the interval does not say how it was arrived at"


def test_the_named_level_is_the_level_the_code_computes():
    """The sentence and the number must not drift apart.

    `bootstrap_auc_ci` takes alpha=0.05, which is 95 per cent. If that default ever moves, this
    fails rather than the prose quietly becoming wrong.
    """
    import inspect

    alpha = inspect.signature(margin.bootstrap_auc_ci).parameters["alpha"].default
    assert f"{round((1 - alpha) * 100)}%" == margin.CONFIDENCE


def test_an_interval_containing_chance_says_so_in_the_reading_and_the_verdict():
    said = _said({"auc_ci": [0.45, 0.61]}, 0.53)
    assert "contains 0.5" in said
    assert "apart from chance" in said
    assert "No finding" in said, "an interval spanning chance is reported as a result"


def test_a_run_that_invalidated_itself_is_not_described_as_usable():
    """The verdict follows the flag the markers already set, which is the whole point.

    This is the case the project has been burnt by: two readers quoted an AUC that a null ruler had
    beaten, on a run that exited 0 with the diagnostic printed beside the number.
    """
    said = _said({"auc_ci": [0.72, 0.88], "self_invalidated": "a null beat it"}, 0.8062)
    assert "NOT A MEASUREMENT" in said
    assert "must not be quoted" in said
    assert "Usable" not in said, "an invalidated run is described as usable"


def test_no_interval_means_no_verdict_rather_than_a_confident_one():
    """The first version of this said "usable" with no interval at all, which is the worse answer.

    Without a spread there is nothing to say the figure is not noise, and a verdict calling it usable
    anyway is exactly the sentence somebody quotes.
    """
    said = _said({"auc_ci": None}, 0.8062)
    assert "No verdict" in said
    assert "Usable" not in said


def test_a_clean_separated_run_is_called_usable_and_bounded_to_its_corpus():
    said = _said({"auc_ci": [0.72, 0.88]}, 0.8062)
    assert "Usable as a measurement" in said
    assert "these prompts and this model" in said, (
        "the verdict does not bound the claim to what was actually measured")


@pytest.mark.parametrize("res", [
    {"auc_ci": [0.72, 0.88]},
    {"auc_ci": [0.45, 0.61]},
    {"auc_ci": None},
    {"auc_ci": [0.72, 0.88], "self_invalidated": "x"},
])
def test_the_sentences_never_start_with_a_machine_marker(res):
    """Otherwise `say` would treat them as verbatim and stop wrapping them, and the audit's own
    complaint about this command was partly that its output does not wrap.
    """
    from senbonzakura import say

    for line in margin.in_words(res, 0.8062):
        assert not say.is_marker(line), f"a prose line reads as a marker and will not wrap: {line!r}"


@pytest.mark.parametrize("res", [
    {"auc_ci": [0.72, 0.88]},
    {"auc_ci": None},
    {"auc_ci": [0.72, 0.88], "self_invalidated": "x"},
])
def test_every_line_fits_a_terminal(res):
    """The complaint that prompted the `say` module in the first place."""
    from senbonzakura import say

    for line in margin.in_words(res, 0.8062):
        assert len(line) <= say.CEILING, f"{len(line)} columns: {line!r}"
