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

#: The spellings a page uses to state the total number of flags. The second one exists because the
#: count is also written as "`--help-all` shows all N", with the word "flags" in the clause before
#: it, so a pattern anchored on "N flags" reads straight past it. No page under `docs/reference`
#: uses that form today; it is here so that one adopting it is covered on the day it does.
_A_STATED_FLAG_COUNT = [
    re.compile(r"\b(\d{2,3})\s+flags\b"),
    re.compile(r"`--help-all`\s+shows\s+all\s+(\d{2,3})\b"),
]


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
        r"shows the (\d+) flags a run needs, and the model it edits\.\s+(\d+) more",
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
        r"shows the (\d+) flags a run needs, and the model it edits\.\s+(\d+) more",
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
    # THE CHANGELOG IS DELIBERATELY NOT HERE, and it was added on 2026-09-28 and taken straight back
    # out. A reviewer found its 0.4.0 entry saying `--help-all` shows 69 while the parser declares
    # 73, which reads exactly like the drift this file exists to catch. It is not. At `v0.4.0` the
    # parser declared 69, so that sentence was true when it was written and the release it describes
    # still has 69. "Correcting" it to today's number puts a false statement into the record of a
    # shipped release, which is the opposite of the property being protected here.
    #
    # A released CHANGELOG entry is a historical claim about one artefact. These reference pages
    # describe the build in front of the reader. Only the second kind can be held to the count the
    # parser declares now, and conflating them is a mistake somebody will make again, which is why
    # this is written down rather than left as an absence.
    pages = [ROOT / "docs" / "reference" / "cli.md", ROOT / "docs" / "reference" / "flags.md"]
    total = len(_flags(build_parser(full=True)))

    wrong = {}
    for page in pages:
        if not page.is_file():
            continue
        text = page.read_text(encoding="utf-8")
        for pattern in _A_STATED_FLAG_COUNT:
            for stated in pattern.findall(text):
                if int(stated) != total:
                    wrong.setdefault(page.name, set()).add(stated)
    assert not wrong, (
        f"these pages state a flag count that is not the {total} the parser declares: "
        f"{ {k: sorted(v) for k, v in wrong.items()} }")


def test_the_count_patterns_catch_both_spellings():
    """Mutation test: the CHANGELOG's spelling has to be read, or adding the page changes nothing."""
    changelog = "**`--help` shows the flags a run needs; `--help-all` shows all 69.**"
    found = {n for pattern in _A_STATED_FLAG_COUNT for n in pattern.findall(changelog)}
    assert found == {"69"}, found
    reference = "the default command has 73 flags"
    found = {n for pattern in _A_STATED_FLAG_COUNT for n in pattern.findall(reference)}
    assert found == {"73"}, found
