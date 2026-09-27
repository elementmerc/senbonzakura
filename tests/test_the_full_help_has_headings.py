# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`--help-all` is a reference page, so it needs somewhere for a reader to look.

WHAT PROMPTED IT, 2026-09-27

An output review measured the page: 497 lines, 37.5 KB, five blank lines in the whole thing, and
three headings, `positional arguments:`, `options:` and `commands:`. Lines 52 to 468 were one
unbroken run of 417 flags, with nothing between the VRAM throttle and the capability probe to tell a
reader they had moved from one subject to another.

The footer on the short page made it worse by promising structure that did not exist: "N more
control the search, the scoring and the measurement", three groups that appeared nowhere as
headings.

WHAT THESE TESTS HOLD

That the headings exist, that a blank line precedes each so the eye can find them, that no single
run of flags is long enough to get lost in, and that every flag is either core (so it belongs on the
short page under `options:`) or filed under a heading. The last one is what stops the next flag
being added into the void: the runtime leaves an unfiled flag where it is, on purpose, so that a
missing entry surfaces here rather than as a command that will not start.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys

import pytest

from senbonzakura.parser import _STAYS_IN_OPTIONS as STAYS_IN_OPTIONS
from senbonzakura.parser import CORE_FLAGS, FLAG_GROUPS, build_parser

#: The longest unbroken run of lines a reader should have to scan without a heading. Generous on
#: purpose: the point is that the page has joints, not that they fall in any particular place.
MAX_RUN = 120


@pytest.fixture(scope="module")
def page():
    run = subprocess.run([sys.executable, "-m", "senbonzakura", "--help-all"],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         check=False, timeout=300, env={"COLUMNS": "80", **_env()})
    assert run.returncode == 0, run.stderr[:400]
    return run.stdout.splitlines()


def _env():
    import os

    return dict(os.environ)


@pytest.mark.parametrize("title", sorted(FLAG_GROUPS))
def test_the_heading_is_on_the_page(page, title):
    assert f"{title}:" in page, f"no `{title}:` heading, so its flags are back in the flat run"


@pytest.mark.parametrize("title", sorted(FLAG_GROUPS))
def test_a_blank_line_precedes_the_heading(page, title):
    i = page.index(f"{title}:")
    assert i > 0 and page[i - 1].strip() == "", (
        f"`{title}:` is butted against the flag above it, so it does not read as a joint")


def test_no_run_of_flags_is_long_enough_to_get_lost_in(page):
    joints = [i for i, ln in enumerate(page) if ln.endswith(":") and not ln.startswith(" ")]
    runs = [b - a for a, b in itertools.pairwise(joints)]
    assert runs, "the page has no headings at all"
    assert max(runs) <= MAX_RUN, (
        f"the longest stretch between two headings is {max(runs)} lines, which is the flat wall "
        f"this was written for")


def test_every_flag_is_either_core_or_filed_under_a_heading():
    """The guard on the next flag somebody adds.

    `_file_flags_under_headings` leaves an unfiled flag under `options:` rather than raising,
    because a presentation detail must not stop the tool starting. This is where it is caught.
    """
    ap = build_parser(full=True)
    filed = {dest for dests in FLAG_GROUPS.values() for dest in dests}
    stray = sorted(a.dest for a in ap._actions
                   if a.option_strings and a.dest not in CORE_FLAGS
                   and a.dest not in filed and a.dest not in STAYS_IN_OPTIONS)
    assert not stray, (
        f"these flags are neither core nor filed under a heading in FLAG_GROUPS, so they sit in "
        f"the flat `options:` run: {stray}")


def test_no_core_flag_is_filed_under_a_heading():
    """A core flag stays on the short page, so filing one gives that page a one-flag section.

    `--trials` was filed under `the search` and `senbonzakura --help` ended with a `the search:`
    heading holding it alone, which reads as a page that lost the rest of its content.
    """
    filed = {dest for dests in FLAG_GROUPS.values() for dest in dests}
    assert not filed & set(CORE_FLAGS), (
        f"these are core and also filed under a heading, so the short page grows a section for "
        f"them: {sorted(filed & set(CORE_FLAGS))}")


def test_no_heading_claims_a_flag_the_parser_does_not_have():
    """The other direction: a renamed flag leaving a dead entry behind."""
    ap = build_parser(full=True)
    real = {a.dest for a in ap._actions}
    dead = {title: sorted(set(dests) - real) for title, dests in FLAG_GROUPS.items()
            if set(dests) - real}
    assert not dead, f"FLAG_GROUPS names flags that do not exist: {dead}"


def test_the_footer_names_the_headings_it_promises():
    """The original miswording: a promise of three groups that were not there.

    Now the promise is checkable, so it is checked. The footer names the groups in prose, and each
    name it uses has to be a heading the full page actually prints.
    """
    footer = build_parser(full=False).epilog
    footer = footer[footer.index("This page shows"):]
    missing = [t for t in FLAG_GROUPS if t.removeprefix("the ") not in footer]
    assert not missing, f"the footer does not mention {missing}, which the full page groups by"


def test_grouping_changes_nothing_about_what_is_accepted():
    """A presentation change that moved a flag out of the parse would be the worst outcome."""
    short, full = build_parser(full=False), build_parser(full=True)
    assert ({a.dest for a in short._actions} == {a.dest for a in full._actions}), \
        "the short and full parsers no longer accept the same arguments"
    for ap in (short, full):
        args = ap.parse_args(["some/model", "--trials", "7", "--gen-batch", "3",
                              "--capability-n", "5", "--base-licence", "mit"])
        assert (args.trials, args.gen_batch, args.capability_n, args.base_licence) == \
            (7, 3, 5, "mit")


def test_suppression_still_hides_every_non_core_flag():
    """The short page's contract, re-checked now that the flags live in several containers."""
    short = build_parser(full=False)
    shown = [a.dest for a in short._actions
             if a.help is not argparse.SUPPRESS and a.option_strings]
    assert set(shown) <= CORE_FLAGS, f"these leaked onto the short page: {sorted(set(shown) - CORE_FLAGS)}"
