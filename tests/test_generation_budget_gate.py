# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A search cannot be steered by a signal too short to see the thing it is searching for.

THE DEFECT THIS WAS WRITTEN FOR, AND IT IS THE SHAPE THIS PROJECT KEEPS REPEATING.

Three commands carried three generation budgets, set independently, with nothing reconciling
them: the abliterator searched at 48 tokens, the scorer defaulted to 64, and the head-to-head
scored at 192. Only the last came from a measurement.

That is not three opinions about one tradeoff. The search scores every candidate at its budget,
so a configuration whose refusal merely arrives after token 48 wins the search, and the run then
publishes a number measured at 192 where that trick does not work. The optimiser was selecting
against a proxy that diverged fourfold from the published metric, and nothing downstream could
recover it, because the search had already crowned the winner.

The tool knew. `budget_warning` fired on every single run and the run proceeded, which is how it
fired on all ten arms of a comparison and changed nothing. A warning that never changes an outcome
is not a safeguard; it is noise that teaches people to scroll past it.
"""
import pytest

from senbonzakura import cli, lengthsweep


class _Args:
    def __init__(self, gen_tokens, short_budget_ok=False, model="some/model"):
        self.gen_tokens = gen_tokens
        self.short_budget_ok = short_budget_ok
        self.model = model


def test_the_three_budgets_are_now_one_number():
    """The regression that matters most: a second literal appearing anywhere.

    Each of these used to be written out separately. If one drifts, the search and the report
    disagree again and no other test in the suite would notice.
    """
    from senbonzakura import headtohead, parser, score

    assert headtohead.REFUSAL_MAX_NEW == lengthsweep.DEFAULT_BUDGET
    assert parser.build_parser().get_default("gen_tokens") == lengthsweep.DEFAULT_BUDGET
    assert score.build_parser().get_default("max_new") == lengthsweep.DEFAULT_BUDGET


def test_the_default_budget_is_above_the_floor_it_must_clear():
    """Otherwise a default run would trip its own gate, which has happened to other defaults here."""
    assert lengthsweep.DEFAULT_BUDGET >= lengthsweep.VISIBILITY_FLOOR


@pytest.mark.parametrize("budget", [1, 3, 16, 48, 64, 95])
def test_a_budget_below_the_floor_is_refused(budget):
    with pytest.raises(SystemExit) as caught:
        cli._preflight_generation_budget(_Args(budget), log=lambda *a: None)
    message = str(caught.value)
    assert "--short-budget-ok" in message, "the refusal must name the way past it"
    assert "score --length-sweep" in message, "and the way to answer the question properly"
    assert str(lengthsweep.DEFAULT_BUDGET) in message, "and what the default is"


@pytest.mark.parametrize("budget", [96, 128, 192, 256])
def test_a_budget_at_or_above_the_floor_passes_silently(budget):
    said = []
    assert cli._preflight_generation_budget(_Args(budget), log=said.append) == budget
    assert not said, f"a sound budget should say nothing, said: {said}"


def test_the_escape_hatch_allows_it_but_says_the_numbers_are_unquotable():
    said = []
    assert cli._preflight_generation_budget(_Args(16, short_budget_ok=True), log=said.append) == 16
    assert said, "a run that opted out of the floor must still be told what it gave up"
    assert "must not be quoted" in " ".join(said)


def test_the_artefact_carries_the_caveat_rather_than_only_the_console():
    """A warning printed at the start of a run is gone by the time the number gets quoted.

    The figure is read out of `abliteration.json` long after the log has scrolled, so the caveat
    has to travel in the same file. None when the budget is sound, so its presence is the signal.
    """
    assert cli._budget_warning(lengthsweep.DEFAULT_BUDGET) is None
    short = cli._budget_warning(48)
    assert short and "48" in short
