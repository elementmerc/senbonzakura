# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A help page says what a flag does. It does not say what the flag used to do.

WHAT PROMPTED IT, 2026-09-27

An output review read the tool's screens as a user and called them "notes-to-self shipped as user
interface". The evidence was in the flag help:

  * `--matched-scoring` ran past 600 characters and included "ON BY DEFAULT since 2026-09-17", an
    account of what the default had been before, and "This was off pending Q-14, which has since
    reported...".
  * `--separation-statistic` said "Under measurement (Q-14): the default does not change until that
    measurement says it should." `Q-14` is an internal ticket number on a user-facing screen.
  * `senbonzakura validate --help` was 158 lines and opened with two paragraphs of bug history and a
    literature review carrying an arXiv number, before a single flag appeared.

None of it is wrong and none of it belongs there. A reader at `--help` is asking what to type. The
history is worth keeping and its home is the module docstring, the decision log and the
documentation, all of which this repository has.

WHAT THIS TEST HOLDS

The rendered help of every command, checked for the four tells: an internal ticket identifier, a
calendar date, an arXiv citation, and the phrase that introduces a description of a past default.
It reads what a user sees, not the source, because a string can be assembled from pieces.
"""
from __future__ import annotations

import re
import subprocess
import sys

import pytest

from senbonzakura.entry import DELEGATED

#: The modes `split_mode` peels off, which never appear in `DELEGATED`.
MODES = ("abliterate", "kageyoshi", "auto")

COMMANDS = [*sorted(DELEGATED), *MODES]

#: Each tell, as a pattern and the thing to do instead.
#:
#: `head-to-head` is the one exemption and it is narrow: the tool names the competing tools it runs
#: and their published figures, which is attribution rather than our own history. Nothing in the
#: patterns below matches that, so it needs no carve-out today; if one is ever needed it goes here
#: with a reason, not by loosening a pattern.
TELLS = (
    (r"\bQ-\d+\b",
     "an internal ticket identifier. Say what the flag does; the ticket means nothing to a user"),
    (r"\b20\d\d-[01]\d-[0-3]\d\b",
     "a date. 'on by default' does not need the date it became the default"),
    (r"\barXiv:\s*\d",
     "an arXiv citation. Cite it in the docs, not on a help screen"),
    (r"\b(?:as runs|runs) before 20\d\d",
     "a description of what the default used to be. State the current behaviour"),
)


@pytest.fixture(scope="module")
def pages():
    out = {}
    for command in COMMANDS:
        run = subprocess.run([sys.executable, "-m", "senbonzakura", command, "--help"],
                             capture_output=True, text=True, stdin=subprocess.DEVNULL,
                             check=False, timeout=300)
        assert run.returncode == 0, f"{command} --help exited {run.returncode}"
        out[command] = run.stdout
    run = subprocess.run([sys.executable, "-m", "senbonzakura", "--help-all"],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         check=False, timeout=300)
    assert run.returncode == 0
    out["--help-all"] = run.stdout
    return out


@pytest.mark.parametrize(("pattern", "why"), TELLS)
def test_no_help_page_explains_the_project_to_itself(pages, pattern, why):
    found = {}
    for command, text in pages.items():
        # Joined first: a tell can be split across argparse's wrapping, so a line-by-line search
        # would miss `Q-\n14` and report the page clean.
        hits = re.findall(pattern, " ".join(text.split()))
        if hits:
            found[command] = sorted(set(hits))
    assert not found, f"{why}.\nFound in: {found}"
