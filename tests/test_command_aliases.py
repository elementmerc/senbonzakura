# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A plain-English second name for a command whose first name says nothing.

WHY ALIASES AND NOT RENAMES

`kageyoshi` and `compass` are names somebody has to be told before they mean anything, and the
tool is meant to be usable by people who have not read the project. `auto` has stood in for
`kageyoshi` since before this file; `harm-recognition` now stands in for `compass`.

Renaming was the other option and it was refused. Every run spec on record, every documented
example and every script anybody has written uses the existing names, and a tool that renames its
own commands breaks the record of what was already run. So the original stays canonical, is what
every artefact records, and the alias is a door rather than a replacement.

WHAT THESE TESTS HOLD

That an alias reaches a real command, that no alias shadows one, and that the list stays short:
a second name for a command that was already plain is surface with no reader, and surface is the
thing this whole round of work exists to reduce.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from senbonzakura.entry import ALIASES, DELEGATED


@pytest.mark.parametrize(("alias", "real"), sorted(ALIASES.items()))
def test_every_alias_names_a_command_that_exists(alias, real):
    assert real in DELEGATED, f"{alias} points at {real}, which is not a command"


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_no_alias_shadows_a_real_command(alias):
    """An alias with the same name as a command would take its place silently."""
    assert alias not in DELEGATED


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_an_alias_runs_the_command_it_stands_for(alias):
    """End to end through the entry point, because the resolution happens there and a unit test
    on the dictionary would pass whether or not anything reads it.
    """
    out = subprocess.run([sys.executable, "-m", "senbonzakura", alias, "--help"],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         check=False, timeout=300)
    assert out.returncode == 0, out.stderr
    assert f"senbonzakura {ALIASES[alias]}" in out.stdout, (
        "the help must name the real command, so a reader knows what their artefacts will say")


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_the_main_help_names_the_alias(alias):
    """An alias nobody can discover is a synonym in the source code."""
    from senbonzakura.parser import build_parser

    assert alias in build_parser(full=True).format_help()


def test_the_list_stays_short():
    """Not a rule against growth: a prompt to justify each one. Every alias is a second way to
    spell something, and this project's measured problem is that there are already too many.
    """
    assert len(ALIASES) <= 4, (
        "more than four aliases means the names themselves are the problem and should be fixed "
        "at a release boundary, not papered over one synonym at a time")
