# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A refusal for the commonest mistake there is has to say what to type, not only what is absent.

WHAT PROMPTED IT

Readers 3 and 5 of the 2026-09-26 user pass both came to set something up and stopped at the first
refusal. The top level prints a hand-written paragraph naming the next step; a subcommand printed
argparse's own line and nothing else:

    senbonzakura gate: error: the following arguments are required: --baseline, --current

That names what is missing and nothing about what to type, so the next move is to go away, read
`--help`, and work out the shape unaided. For a command whose whole value is being easy to put in
front of a build, that is the wrong place to send somebody.

WHY IT IS BUILT RATHER THAN WRITTEN PER COMMAND

Twenty delegated commands have required flags, and a per-command paragraph is twenty chances to
word it differently and twenty places to forget when a flag becomes required later. The shortest
accepted invocation is a fact about the parser, so it is read off the parser.

STRICTLY ADDITIVE, which is `ParserThatNamesUnknownFlags`'s own stated design and the reason this
could land without a decision: it never refuses anything that parses today, it only adds to a
refusal that was already happening.
"""
from __future__ import annotations

import argparse
import importlib

import pytest

from senbonzakura import argresolve
from senbonzakura.entry import typed_commands


def _parser(**over):
    p = argresolve.ParserThatNamesUnknownFlags(prog="senbonzakura thing", allow_abbrev=False)
    p.add_argument("--baseline", required=True)
    p.add_argument("--current", "--measurement", dest="current", required=True,
                   metavar="MEASUREMENT")
    p.add_argument("--quiet", action="store_true")
    for flag, kwargs in over.items():
        p.add_argument(flag, **kwargs)
    return p


# ── the shape itself ────────────────────────────────────────────────────────────────────────────


def test_the_shortest_invocation_names_the_program_and_every_required_flag():
    assert argresolve.shortest_invocation(_parser()) == (
        "senbonzakura thing --baseline BASELINE --current MEASUREMENT")


def test_the_value_shown_is_the_metavar_the_command_declared():
    """A command that named its own placeholder has already chosen the word its help uses."""
    shortest = argresolve.shortest_invocation(_parser())
    assert "MEASUREMENT" in shortest, "the declared metavar was replaced with the dest"


def test_an_optional_flag_is_not_in_the_shortest_invocation():
    assert "--quiet" not in argresolve.shortest_invocation(_parser())


def test_a_required_flag_that_takes_no_value_is_shown_without_one():
    p = argresolve.ParserThatNamesUnknownFlags(prog="p", allow_abbrev=False)
    p.add_argument("--yes", action="store_true", required=True)
    assert argresolve.shortest_invocation(p) == "p --yes"


def test_a_required_positional_comes_before_the_flags_because_that_is_the_typing_order():
    p = argresolve.ParserThatNamesUnknownFlags(prog="p", allow_abbrev=False)
    p.add_argument("source", metavar="SOURCE")
    p.add_argument("--out", required=True)
    assert argresolve.shortest_invocation(p) == "p SOURCE --out OUT"


def test_several_positionals_keep_their_declaration_order():
    """They cannot be sorted: the order they are declared in is the order they have to be typed."""
    p = argresolve.ParserThatNamesUnknownFlags(prog="p", allow_abbrev=False)
    p.add_argument("first")
    p.add_argument("second")
    assert argresolve.shortest_invocation(p) == "p FIRST SECOND"


def test_flags_are_sorted_so_two_runs_print_one_line():
    p = argresolve.ParserThatNamesUnknownFlags(prog="p", allow_abbrev=False)
    p.add_argument("--zebra", required=True)
    p.add_argument("--apple", required=True)
    assert argresolve.shortest_invocation(p) == "p --apple APPLE --zebra ZEBRA"


@pytest.mark.parametrize("nargs", ["?", "*"])
def test_an_optional_positional_is_not_required_however_argparse_spells_it(nargs):
    """Argparse sets no `required` on a positional, so the nargs is the only thing that says."""
    p = argresolve.ParserThatNamesUnknownFlags(prog="p", allow_abbrev=False)
    p.add_argument("maybe", nargs=nargs)
    p.add_argument("--out", required=True)
    assert argresolve.shortest_invocation(p) == "p --out OUT"


def test_a_parser_that_needs_nothing_has_no_shortest_invocation():
    """None rather than the bare program name: there is no missing-argument refusal to add to."""
    p = argresolve.ParserThatNamesUnknownFlags(prog="p", allow_abbrev=False)
    p.add_argument("--maybe")
    assert argresolve.shortest_invocation(p) is None


def test_a_subparsers_action_is_not_mistaken_for_a_positional():
    """A command word is not an argument, and printing it as one would say the wrong shape."""
    p = argresolve.ParserThatNamesUnknownFlags(prog="p", allow_abbrev=False)
    sub = p.add_subparsers(dest="verb")
    sub.add_parser("build")
    p.add_argument("--out", required=True)
    assert argresolve.shortest_invocation(p) == "p --out OUT"


# ── the refusal a reader actually meets ─────────────────────────────────────────────────────────


def _refusal(parser, argv):
    import contextlib
    import io

    buffer = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stderr(buffer):
        parser.parse_args(argv)
    return buffer.getvalue()


def test_the_refusal_shows_the_shape_as_well_as_what_is_missing():
    said = _refusal(_parser(), [])
    assert "the following arguments are required" in said, "argparse's own sentence must survive"
    assert "The shortest command line this accepts" in said
    assert "senbonzakura thing --baseline BASELINE --current MEASUREMENT" in said


def test_the_example_is_indented_four_spaces_so_it_stays_pasteable():
    """`say.reflow` joins two-space lines into a paragraph and passes four-space lines verbatim.

    The same rule `baseline`'s hand-written refusal relies on. Wrapped into a paragraph, a command
    line gets a newline in the middle of it and stops being something anybody can copy.
    """
    line = next(ln for ln in _refusal(_parser(), []).splitlines()
                if "--baseline BASELINE" in ln and "usage" not in ln)
    assert line.startswith("    ") and not line.startswith("     ")


def test_the_refusal_points_at_this_commands_own_help_and_not_the_tools():
    assert "senbonzakura thing --help" in _refusal(_parser(), [])


def test_the_typo_sentence_and_the_shape_both_appear_when_both_apply():
    """They answer different questions and a reader who typed a typo needs both."""
    said = _refusal(_parser(), ["--baselne", "x"])
    assert "--baselne" in said and "is not a flag" in said
    assert "The shortest command line this accepts" in said


def test_nothing_is_added_to_a_refusal_that_is_not_about_a_missing_argument():
    """Strictly additive means this fires on one refusal and leaves the others alone."""
    said = _refusal(_parser(), ["--baseline", "a", "--current", "b", "--nope"])
    assert "The shortest command line this accepts" not in said


def test_a_required_choice_between_flags_adds_nothing_rather_than_a_wrong_shape():
    """The one refusal that says "required" with no `required` action behind it.

    argparse sets `required` on the GROUP for a mutually exclusive choice and on none of its
    members, so there is no shortest invocation to build and showing an empty one would be worse
    than showing none. The group's own message already names both flags.
    """
    p = argresolve.ParserThatNamesUnknownFlags(prog="p", allow_abbrev=False)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--this")
    group.add_argument("--that")
    said = _refusal(p, [])
    assert "required" in said
    assert argresolve.shortest_invocation(p) is None
    assert "The shortest command line this accepts" not in said


def test_a_parser_with_subcommands_adds_nothing():
    """Its argv holds the subcommand's flags, which it does not declare and must not describe."""
    p = argresolve.ParserThatNamesUnknownFlags(prog="p", allow_abbrev=False)
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("build")
    assert "The shortest command line this accepts" not in _refusal(p, [])


# ── every delegated command, which is the point of doing it once ────────────────────────────────


def _delegated_parsers():
    for name, (module_name, *_rest) in sorted(typed_commands().items()):
        try:
            module = importlib.import_module(f"senbonzakura.{module_name}")
        except ImportError:  # pragma: no cover - depends on what is installed here
            continue
        build = getattr(module, "build_parser", None)
        if build is None:
            continue
        try:
            yield name, build()
        except TypeError:  # pragma: no cover - a builder that needs arguments
            continue


def test_there_are_commands_to_check():
    """An empty parametrisation passes and proves nothing, which this suite has been fooled by."""
    assert len(list(_delegated_parsers())) > 10


@pytest.mark.parametrize(("name", "parser"), list(_delegated_parsers()), ids=lambda x: str(x)[:20])
def test_every_command_with_a_required_argument_shows_its_shape(name, parser):
    """One implementation, so this is a sweep rather than twenty tests that could disagree."""
    required = [a for a in parser._actions
                if a.option_strings and a.required
                and not isinstance(a, argparse._SubParsersAction)]
    shortest = argresolve.shortest_invocation(parser)
    if not required:
        return
    assert shortest, f"{name} has required flags and no shortest invocation was built"
    for action in required:
        assert action.option_strings[0] in shortest, (
            f"{name} requires {action.option_strings[0]} and the shape does not show it")


@pytest.mark.parametrize(("name", "parser"), list(_delegated_parsers()), ids=lambda x: str(x)[:20])
def test_no_commands_shape_mentions_an_optional_flag(name, parser):
    """A shape that asks for more than the minimum teaches people to pass flags they do not need."""
    shortest = argresolve.shortest_invocation(parser)
    if not shortest:
        return
    optional = [a.option_strings[0] for a in parser._actions
                if a.option_strings and not a.required and a.option_strings[0] != "-h"]
    shown = [flag for flag in optional if f" {flag} " in f" {shortest} "]
    assert not shown, f"{name}'s shape asks for optional flags: {shown}"
