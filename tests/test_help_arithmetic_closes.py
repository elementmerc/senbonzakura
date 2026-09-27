# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The flag counts `--help` prints have to add up to the number `--help-all` claims.

WHAT PROMPTED IT, 2026-09-27

A first-time reader added them up. `--help` said "This page shows the 16 flags a run needs. 55 more
control the search, the scoring and the measurement", and `--help-all` said "there are 69 in total".
16 plus 55 is 71, not 69, and by the time it was checked on the development branch the computed pair
had moved to 16 and 56 against a still-typed 69.

Two separate faults in one sentence, which is why the fix is a test rather than a corrected number:

  * The total was a LITERAL in a help string while the other two were computed from argparse at
    run time, so the moment anybody added a flag the two disagreed. This project has a phrase for
    it: a flag can be forgotten, a version cannot. Same shape, one layer down.
  * The count of shown flags was `len(ap._actions)`, which includes the model POSITIONAL. It
    reported 16 flags where 15 flags and one positional were shown, so it was wrong even on the day
    it was written.

The numbers are trivia. A reader who checks one arithmetic claim in the help text and finds it
wrong has learned something true about how carefully the rest was checked, and this project's whole
argument is that its numbers can be checked.
"""
import re
from pathlib import Path

import pytest

from senbonzakura.parser import CORE_FLAGS, build_parser

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def parsers():
    return build_parser(full=False), build_parser(full=True)


def _flags(ap):
    """Every real flag: an action with option strings. The model positional is not a flag."""
    return [a for a in ap._actions if a.option_strings]


def test_the_positional_is_not_counted_as_a_flag(parsers):
    """The original miscount. `len(_actions)` includes the model, so it read one too many."""
    _, full = parsers
    positionals = [a for a in full._actions if not a.option_strings]
    assert positionals, "the model positional has gone; this parser takes a model to edit"
    assert len(_flags(full)) == len(full._actions) - len(positionals)


def test_the_epilog_and_the_total_agree(parsers):
    """Shown + hidden == total, read out of the strings a user actually sees."""
    short, full = parsers
    shown, hidden = (int(n) for n in re.search(
        r"shows the (\d+) flags a run needs, and the model it edits\. (\d+) more",
        short.epilog).groups())

    help_all = next(a for a in _flags(full) if a.dest == "help_all")
    total = int(re.search(r"there are (\d+) in total", help_all.help).group(1))

    assert shown + hidden == total, (
        f"`--help` says {shown} shown and {hidden} hidden, and `--help-all` says {total} in total. "
        f"{shown} + {hidden} is {shown + hidden}. A reader who adds them up is the person who "
        f"found this.")


def test_those_numbers_are_the_measured_ones(parsers):
    """Agreeing with each other is not enough; they have to agree with the parser.

    Two wrong numbers that happen to sum correctly would pass the test above.
    """
    short, full = parsers
    flags = _flags(full)
    core = sum(1 for a in flags if a.dest in CORE_FLAGS)

    shown, hidden = (int(n) for n in re.search(
        r"shows the (\d+) flags a run needs, and the model it edits\. (\d+) more",
        short.epilog).groups())
    help_all = next(a for a in flags if a.dest == "help_all")
    total = int(re.search(r"there are (\d+) in total", help_all.help).group(1))

    assert total == len(flags), f"the help claims {total} flags; the parser declares {len(flags)}"
    assert shown == core, f"the help claims {shown} core flags; CORE_FLAGS selects {core}"
    assert hidden == len(flags) - core, (
        f"the help claims {hidden} hidden flags; {len(flags) - core} are not in CORE_FLAGS")


def test_no_count_is_hardcoded_in_the_help_strings():
    """The fault was a literal beside two computed numbers, so the literal is what to forbid.

    Any two-digit number in these strings is a count that will not follow the parser. The strings
    are rebuilt per call, so this reads the live objects rather than the source.
    """
    short, full = build_parser(full=False), build_parser(full=True)
    help_all = next(a for a in _flags(full) if a.dest == "help_all")

    # SCOPED TO THE FOOTER SENTENCE, not to the whole epilog, which carries the command list and
    # with it the `judge` description's two mentions of 90%. Counting digits across all of it
    # measured the prose rather than the claim, and this assertion failed on its own first run for
    # that reason.
    footer = short.epilog[short.epilog.index("This page shows"):]
    assert len(re.findall(r"\b\d{2,}\b", footer)) == 2, (
        f"the footer should carry exactly the two computed counts and no third number: {footer!r}")
    assert len(re.findall(r"\b\d{2,}\b", help_all.help)) == 1, (
        f"the --help-all description should carry exactly the computed total: {help_all.help!r}")


def test_the_reference_pages_quote_the_measured_flag_count():
    """The same number, in prose, on two pages that went stale together.

    `docs/reference/cli.md` said "the default command has 69 flags" and
    `docs/reference/flags.md` said "there are 69 flags", while the parser declared 71. Same shape as
    the help text's own hardcoded total and as the install size that three pages disagreed about: a
    measured fact copied into prose, where nothing can notice it drifting.

    A reader who counts is the person who finds these, and this project's whole argument is that its
    numbers can be counted.
    """
    pages = [ROOT / "docs" / "reference" / "cli.md", ROOT / "docs" / "reference" / "flags.md"]
    total = len(_flags(build_parser(full=True)))

    wrong = {}
    for page in pages:
        if not page.is_file():
            continue
        for stated in re.findall(r"\b(\d{2,3})\s+flags\b", page.read_text(encoding="utf-8")):
            if int(stated) != total:
                wrong.setdefault(page.name, set()).add(stated)
    assert not wrong, (
        f"these pages state a flag count that is not the {total} the parser declares: "
        f"{ {k: sorted(v) for k, v in wrong.items()} }")
