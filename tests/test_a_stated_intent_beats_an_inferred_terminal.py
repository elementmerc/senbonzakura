# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`interactive --no-interactive`: the case `isatty` cannot see, driven rather than parsed.

WHAT THE FLAG IS FOR (Q-55, decided 2026-10-01)

`is_tty` is an inference and it is wrong in one direction. Anything that allocates a
pseudo-terminal satisfies `isatty()` with nobody behind it: `docker run -t`, `expect`, several CI
agents. The inference says "terminal", the menu prints, and the run waits for an answer that is
never coming until something kills it.

WHY THE ACCEPTANCE TEST IS NOT "THE FLAG PARSES"

A test that only checked the flag was accepted would pass over a flag wired to nothing, and five
of the six fixes this project caught in a mutation pass were exactly that. So every test here
hands `run` a stdin stub whose `isatty()` returns **True**, which is the pseudo-terminal, and an
`ask_fn` that **fails the test if it is ever called**. The claim being checked is that a caller
which looks like a terminal and is not one reaches the non-interactive path.

`test_without_the_flag_the_same_stub_is_asked_a_question` is the one that keeps the rest honest.
If the stdin stub were not believed to be a terminal, every other test here would pass for the
wrong reason, because the old `is_tty` branch would be doing the work and the flag could be
deleted without a failure. That test asserts the stub does get the menu when the flag is absent,
so the refusals above it can only be the flag's.

WHY A FILE OF ITS OWN RATHER THAN `tests/test_interactive.py`

That file imports `tests.conftest`, which imports torch at module scope, so it does not collect on
the machine this was written on and these tests could not have been run there. Nothing here
imports conftest, which is what made it possible to drive them rather than reason about them. The
natural home is still `test_interactive.py`, and moving them there is on the torch-run handover
list in `private/plans/2026-10-01-deferred-triage.md`.
"""
from __future__ import annotations

import pytest

from senbonzakura import interactive as it

#: PINNED, BECAUSE THIS FILE DRIVES THE WALK. `interactive.run` is the walk's front door and
#: reaches `plan_abliteration` one line in, so without this the questions asked depend on what the
#: box running the test happens to hold. Required by
#: `tests/test_the_walk_never_reads_the_machine.py`, which did not catch it until `.run(` and
#: `log_failure(` were added to its detector on 2026-10-01: the guard against reading the machine
#: could not see the commonest way of doing it.
pytestmark = pytest.mark.usefixtures("a_machine_with_nothing_on_it")


class _LooksLikeATerminal:
    """A pseudo-terminal: `isatty()` is True and there is nobody behind it.

    This is the stub that matters. `_NotTty` in `test_interactive.py` covers the pipe, which the
    inference already gets right; this covers the case it gets wrong.
    """

    def isatty(self):
        return True


class _Asked(Exception):
    """Raised by the `ask_fn` below, so "a question was asked" is a visible event."""


def _must_not_ask(prompt):
    raise AssertionError(
        f"guided mode asked a question on a stream with nobody behind it: {prompt!r}. "
        f"This is the hang the flag exists to prevent")


def _records_that_it_asked(prompt):
    raise _Asked(prompt)


def test_a_pseudo_terminal_that_says_so_is_refused_rather_than_asked():
    """The whole claim, and the one a parse-only test would not make."""
    said = []
    code = it.run([it.NO_INTERACTIVE], ask_fn=_must_not_ask,
                  log=said.append, stdin=_LooksLikeATerminal())
    assert code == 2, f"expected the non-interactive refusal's exit 2, got {code}"
    assert any("not asking anything" in line for line in said), said


def test_without_the_flag_the_same_stub_is_asked_a_question():
    """The premise of every other test in this file, asserted rather than assumed.

    If this fails, the stdin stub is not being read as a terminal, and the refusals above are the
    old `is_tty` branch firing rather than the flag. That would make this whole file pass with the
    flag removed, which is the shape of test this project keeps finding in its own tree.
    """
    with pytest.raises(_Asked):
        it.run([], ask_fn=_records_that_it_asked,
               log=lambda _m: None, stdin=_LooksLikeATerminal())


def test_the_refusal_names_the_flag_rather_than_blaming_the_stream():
    """Two causes, two head lines, because they need different things from the reader.

    Telling somebody who passed `--no-interactive` that "this input is not a terminal" is false
    about their input: it was a terminal, and they said not to use it. The advice paragraph under
    the head line is deliberately the same for both, since the way back in does not depend on how
    the refusal was reached.
    """
    said = []
    it.run([it.NO_INTERACTIVE], ask_fn=_must_not_ask,
           log=said.append, stdin=_LooksLikeATerminal())
    joined = "\n".join(said)
    assert it.NO_INTERACTIVE in joined, joined
    assert "this input is not one" not in joined, (
        "the refusal blamed the stream, which was a terminal: the caller said not to use it")
    # Whitespace collapsed first, because `_say` wraps at the terminal width and the phrase being
    # looked for straddles a line break. Asserting on the unwrapped text would be asserting on the
    # wrap position, which is not this test's subject and would fail on a narrower terminal.
    assert "`senbonzakura --help` lists them" in " ".join(joined.split()), (
        "the shared advice paragraph did not come through, so the two routes have diverged")


def test_a_pipe_that_also_passes_the_flag_is_told_about_the_flag():
    """The only case where the order of the two checks is visible, and it was found by a mutant.

    A mutation run swapped the flag check and the terminal check, and **every test in this file
    still passed**, which was right: a pseudo-terminal satisfies `is_tty`, so the terminal branch
    does not fire there and the flag is reached in either order. The comment in the source claiming
    the flag would be unreachable was simply wrong, and the mutant is what showed it.

    What the order does decide is what a caller that is both piped and explicit gets told, and
    stated intent should be reported over inferred intent: somebody who passed the flag is told
    their flag was honoured rather than being told something about their stream they did not ask
    about. That is a small thing, and it is the only thing, so this is the test that holds it.
    """
    class _NotATerminal:
        def isatty(self):
            return False

    said = []
    assert it.run([it.NO_INTERACTIVE], ask_fn=_must_not_ask,
                  log=said.append, stdin=_NotATerminal()) == 2
    assert it.NO_INTERACTIVE in said[0], (
        f"a caller that passed the flag was told about its stream instead: {said[0]!r}")


def test_both_routes_give_the_same_status_and_the_same_advice():
    """One refusal reached two ways, so a fix to the advice cannot land on only one of them."""
    stated, inferred = [], []

    class _NotATerminal:
        def isatty(self):
            return False

    assert it.run([it.NO_INTERACTIVE], ask_fn=_must_not_ask,
                  log=stated.append, stdin=_LooksLikeATerminal()) == 2
    assert it.run([], ask_fn=_must_not_ask, log=inferred.append, stdin=_NotATerminal()) == 2
    assert stated[1:] == inferred[1:], (
        f"the two routes print different advice:\n stated: {stated[1:]}\n inferred: {inferred[1:]}")


def test_help_is_still_answered_before_either_check():
    """The property that was already there and must survive the flag being added in front of it.

    `senbonzakura interactive --help | less` has no terminal on stdin, and asking what a command
    does is the one question that has to be answerable without the conditions for running it.
    """
    class _NotATerminal:
        def isatty(self):
            return False

    said = []
    assert it.run(["--help"], ask_fn=_must_not_ask,
                  log=said.append, stdin=_NotATerminal()) == 0
    assert "A guided walk" in "\n".join(said)


def test_help_wins_over_the_flag_when_both_are_given():
    """`--no-interactive --help` is a reader asking what the flag does, not a run to refuse."""
    said = []
    assert it.run([it.NO_INTERACTIVE, "--help"], ask_fn=_must_not_ask,
                  log=said.append, stdin=_LooksLikeATerminal()) == 0
    assert "usage: senbonzakura interactive" in "\n".join(said)


def test_a_near_miss_spelling_is_refused_rather_than_ignored():
    """The hang reached by typing the flag almost right.

    With no parser in this command an unrecognised argument used to be dropped silently, so
    `--no-interactve` printed the menu: the exact outcome the flag prevents, in the hands of
    somebody who was trying to prevent it. A flag that fails open is worse than no flag.
    """
    said = []
    code = it.run(["--no-interactve"], ask_fn=_must_not_ask,
                  log=said.append, stdin=_LooksLikeATerminal())
    assert code == 2, f"a misspelt flag was ignored and the command carried on: {code}"
    joined = "\n".join(said)
    assert "--no-interactve" in joined, "the refusal did not name the argument it did not recognise"
    assert it.NO_INTERACTIVE in joined, "the refusal did not name the flag that was meant"


def test_several_unrecognised_arguments_are_all_named():
    """Naming one of three sends the reader round the loop twice more."""
    said = []
    assert it.run(["--loud", "--fast"], ask_fn=_must_not_ask,
                  log=said.append, stdin=_LooksLikeATerminal()) == 2
    joined = "\n".join(said)
    assert "--loud" in joined and "--fast" in joined, joined
    assert "arguments:" in joined, "the plural was not used for two arguments"


def test_the_help_page_documents_the_flag_and_its_usage_line_names_it():
    """A usage line that omits a flag the command takes is wrong in the one place a reader trusts.

    The file's own comment on `HELP` says so about `[-h]`, which is why it is asserted for the new
    flag too rather than left to a reading.
    """
    first = it.HELP.splitlines()[0]
    assert it.NO_INTERACTIVE in first, f"the usage line does not name the flag: {first}"
    assert "pseudo-terminal" in it.HELP, (
        "the help page does not say which case the flag is for, which is the only thing a reader "
        "needs to decide whether it applies to them")
    assert "takes no options beyond -h" not in it.HELP, (
        "the help page still claims the command takes no options beyond -h, which the flag made "
        "false")


def test_every_line_of_the_help_page_fits_an_eighty_column_terminal():
    """Hand-wrapped rather than formatted by argparse, so nothing but a test holds the width.

    The existing comment records this page having run to 93 columns, which re-wraps into ragged
    half-lines. The flag's description is the newest and longest prose on it.
    """
    long = [line for line in it.HELP.splitlines() if len(line) > 79]
    assert not long, f"these lines are over 79 columns: {long}"
