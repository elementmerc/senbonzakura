# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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

#: Commands that carry sub-verbs, whose flags live on the sub-verb rather than on the command.
#: Sweeping only the top level made `senbonzakura track build --no-balance` invisible to this
#: guard, which is the shape of defect this whole file exists to catch: a check that covers one
#: spelling of the surface and reports clean on the rest.
SUBCOMMANDS = {"track": ("build", "promote", "verify")}

FLAG = re.compile(r"`(--[a-z][a-z0-9-]+)")


@functools.cache
def _accepted():
    """Every flag any command accepts, read off the tool rather than off its source.

    Cached because it shells out once per command and this file parametrises over every flag the
    guide names, which would otherwise pay for the whole sweep on each one.
    """
    invocations = [[c, "--help"] for c in [*sorted(DELEGATED), "abliterate", "kageyoshi", "auto"]]
    invocations += [[c, sub, "--help"] for c, subs in sorted(SUBCOMMANDS.items()) for sub in subs]
    # THE LONG HELP FOR THE DEFAULT COMMAND. `--help` shows 16 of its 69 flags since the page was
    # split, so sweeping the short form would have quietly narrowed this guard to a quarter of the
    # surface it was written to cover, and every flag behind `--help-all` would have started
    # reading as one the tool does not accept.
    invocations += [[c, "--help-all"] for c in ("abliterate", "kageyoshi", "auto")]
    seen = set()
    for words in invocations:
        out = subprocess.run([sys.executable, "-m", "senbonzakura", *words],
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


# ── the short help and the long one ──────────────────────────────────────────────
#
# The default command carries 69 flags and its help was 470 lines, which is the research half of
# this tool standing in front of the door of the other half. `--help` now shows the flags a run
# needs and `--help-all` shows everything. The risk that creates is a flag that is real, accepted
# and reachable nowhere a reader looks, so these assert the two pages against each other rather
# than against a list somebody has to remember to update.

def _help(*words):
    out = subprocess.run([sys.executable, "-m", "senbonzakura", *words],
                         capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         check=False, timeout=300)
    assert out.returncode == 0, f"`senbonzakura {' '.join(words)}` exited {out.returncode}"
    return out.stdout


def test_the_short_help_is_shorter_and_says_how_to_see_the_rest():
    short, full = _help("--help"), _help("--help-all")
    assert len(short.splitlines()) < len(full.splitlines()) / 2, (
        "the short help is not meaningfully shorter than the long one, so the split buys the "
        "reader nothing")
    assert "--help-all" in short, (
        "the short help hides most of the surface without saying where it went, which is worse "
        "than showing all of it")


def test_every_flag_the_parser_accepts_appears_in_the_long_help():
    """THE FAILURE THE SPLIT MAKES POSSIBLE: a flag real enough to parse and visible nowhere.

    Read off the parser rather than off a list, so a flag added after this was written is covered
    by it automatically.
    """
    from senbonzakura.parser import build_parser

    full = _help("--help-all")
    missing = sorted(a.option_strings[0] for a in build_parser(full=True)._actions
                     if a.option_strings and a.option_strings[0] not in full)
    assert not missing, f"accepted by the parser and absent from `--help-all`: {missing}"


def test_the_short_help_keeps_the_flags_the_guide_tells_people_to_type():
    """`--out`, `--track` and `--model` are the three most-named flags in the user pages.

    A split that pushed one of those behind a second command would have moved the wall rather
    than removed it.
    """
    short = _help("--help")
    for flag in ("--model", "--track", "--out", "--device"):
        assert flag in short, f"{flag} is named all over the guide and is not on the first page"
