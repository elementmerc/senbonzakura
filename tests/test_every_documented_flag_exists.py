# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A flag the documentation tells you to pass has to be a flag the tool accepts.

`test_every_flag_is_documented.py` checks that every flag the tool has appears in the docs. This
checks the other direction, which is the one that bites a reader: a flag in the docs that the
tool rejects sends somebody to fix their own command line over a promise the tool never made.

FOUND BY A HOSTILE OUTSIDE REVIEW, 2026-09-17. Two pages, one of them titled "Your first run",
told the reader that an abliteration run writes per-prompt rows containing harmful prompts and
the model's replies, and to pass `--no-margins` if they would rather it did not:

    $ senbonzakura --model x --no-margins --out y
    senbonzakura: error: unrecognized arguments: --no-margins

`--no-margins` is real, on `senbonzakura compass`. It is not on the abliterate path, and no
abliterate run writes such a file. So the paragraph raised an alarm about something that does
not happen and then offered an escape hatch that does not exist, in the safety warning box. A
cautious reader following it gets a failure at exactly the moment they were being careful.
"""
import functools
import re
import subprocess
import sys
from pathlib import Path

import pytest

from senbonzakura.entry import DELEGATED

ROOT = Path(__file__).resolve().parent.parent
DOCS = [*sorted((ROOT / "docs" / "guide").glob("*.md")), ROOT / "README.md"]

#: Flags that appear in our prose while belonging to somebody else's tool, or that are shown as
#: placeholders rather than as something to type. Each one needs a reason, because the whole
#: point of this file is that an unexplained entry here is how the next one hides.
NOT_OURS = {
    "--flag",           # a placeholder in prose about flags in general
    "--help",           # argparse's own, present on every command
    "--version",        # likewise
}

FLAG = re.compile(r"`(--[a-z][a-z0-9-]+)")


@functools.cache
def _accepted():
    """Every flag any command accepts, read off the tool rather than off its source.

    Cached because it shells out once per command and this file parametrises over every flag the
    guide names, which would otherwise pay for the whole sweep on each one.
    """
    seen = set()
    for command in [*sorted(DELEGATED), "abliterate", "kageyoshi", "auto"]:
        out = subprocess.run([sys.executable, "-m", "senbonzakura", command, "--help"],
                             capture_output=True, text=True, stdin=subprocess.DEVNULL,
                             check=False, timeout=300)
        seen |= set(re.findall(r"(--[a-z][a-z0-9-]+)", out.stdout))
    return seen


def _documented():
    found = {}
    for page in DOCS:
        if not page.exists():
            continue
        for line in page.read_text(encoding="utf-8").splitlines():
            for flag in FLAG.findall(line):
                found.setdefault(flag, page.name)
    return found


@pytest.mark.parametrize(("flag", "page"), sorted(_documented().items()))
def test_a_flag_the_docs_name_is_a_flag_the_tool_takes(flag, page):
    if flag in NOT_OURS:
        pytest.skip(f"{flag} is documented as not ours, see NOT_OURS")
    assert flag in _accepted(), (
        f"{page} tells the reader to pass {flag}, and no command accepts it. Either the flag was "
        f"removed and the page was not, or the page describes a flag that belongs to a different "
        f"command. A reader who follows the documentation gets `unrecognized arguments`.")
