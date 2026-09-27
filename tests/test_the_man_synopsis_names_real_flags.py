# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The installed manual's SYNOPSIS drifted from the parser.

FOUND BY A READER WITH NO KNOWLEDGE OF THE PROJECT, 2026-09-27. `man/senbonzakura.1` listed
`--search`, which `senbonzakura --help` does not show at all (it is a `--help-all` flag), and did
not list `--method`, which `--help` presents as a primary knob and which decides the whole ablation
recipe. A reader working from the manual could not find the flag that chooses what the tool does.

The SYNOPSIS is the first thing anybody reads and the last thing anybody updates. This is the
cheap gate: every long flag named there has to exist in the parser that actually runs.
"""
import re
from pathlib import Path

import pytest

MAN = Path(__file__).resolve().parent.parent / "man" / "senbonzakura.1"

pytestmark = pytest.mark.skipif(not MAN.exists(), reason="no man page in this tree")


def _synopsis():
    """The SYNOPSIS section only, with roff's backslash-escaped hyphens unescaped."""
    text = MAN.read_text(encoding="utf-8")
    found = re.search(r"^\.SH SYNOPSIS$(.*?)^\.SH ", text, re.DOTALL | re.MULTILINE)
    assert found, "the man page has no SYNOPSIS section"
    return found.group(1).replace("\\-", "-")


def _parser_options():
    from senbonzakura.parser import build_parser
    opts = set()
    for action in build_parser()._actions:
        opts.update(action.option_strings)
    return opts


def test_every_flag_in_the_synopsis_exists():
    named = set(re.findall(r"--[a-z0-9-]+", _synopsis()))
    assert named, "the SYNOPSIS names no flags at all, which means this test stopped testing"
    missing = sorted(named - _parser_options())
    assert not missing, (
        f"the SYNOPSIS of the installed manual names {missing}, which the parser does not accept. "
        f"The manual is what `pip install` puts on somebody's machine, so a flag that only exists "
        f"there is an instruction that fails.")


def test_the_synopsis_names_the_flag_that_chooses_the_recipe():
    """`--method` decides what the tool does; the SYNOPSIS omitted it while naming a search knob."""
    assert "--method" in _synopsis(), (
        "the SYNOPSIS does not name --method. It is the flag `senbonzakura --help` treats as "
        "primary, it selects the ablation recipe, and it is recorded in abliteration.json as the "
        "thing that makes two runs comparable arms.")
