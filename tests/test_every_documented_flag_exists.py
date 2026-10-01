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


#: The fewest flags a sweep that really ran could come back with. The default command alone
#: carries 69, so a total anywhere near zero means the sweep did not happen rather than that the
#: tool lost its surface. Deliberately far below the real count: this is a did-it-run floor, not
#: a count the next flag addition has to keep up with.
SWEEP_FLOOR = 50


@functools.cache
def _sweep():
    """Every flag any command accepts, read off the tool rather than off its source.

    Returns the flags alongside the invocations that failed, because an invocation that exits
    non-zero contributes nothing to the set and the set alone cannot say why.

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
    seen, failures = set(), []
    for words in invocations:
        out = subprocess.run([sys.executable, "-m", "senbonzakura", *words],
                             capture_output=True, text=True, stdin=subprocess.DEVNULL,
                             check=False, timeout=300)
        if out.returncode != 0:
            failures.append((" ".join(words), out.returncode, out.stderr.strip()[-400:]))
        seen |= set(re.findall(r"(--[a-z][a-z0-9-]+)", out.stdout))
    return frozenset(seen), tuple(failures)


def _accepted():
    return _sweep()[0]


def _sweep_failure(accepted, failures):
    """Why a sweep looks like it never ran, or None when it plainly did.

    An empty sweep makes every documented flag look like a flag the tool rejects, so without this
    the whole file reports two dozen documentation defects when the real fault is that the tool
    would not start. That is the shape of defect this project keeps finding: a check answering a
    narrower question than the one it was asked, and reporting the narrow answer as the wide one.
    """
    if len(accepted) >= SWEEP_FLOOR:
        return None
    detail = "\n".join(f"  `senbonzakura {words}` exited {rc}\n    {err}"
                       for words, rc, err in failures) or "  (every invocation exited 0)"
    return (f"the help sweep found only {len(accepted)} flag(s), below the {SWEEP_FLOOR} a sweep "
            f"that ran would return, so this file cannot say anything about the documentation. "
            f"The tool did not start, most likely a missing dependency rather than a missing "
            f"flag:\n{detail}")


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
    if reason := _sweep_failure(*_sweep()):
        pytest.fail(reason)
    assert flag in _accepted(), (
        f"{page} tells the reader to pass {flag}, and no command accepts it. Either the flag was "
        f"removed and the page was not, or the page describes a flag that belongs to a different "
        f"command. A reader who follows the documentation gets `unrecognized arguments`.")


def test_the_help_sweep_actually_finds_flags():
    """A guard against the gate above passing, or failing, because it swept nothing.

    `test_every_flag_is_documented.py` has had the matching guard on its AST walk since it was
    written; this file went without one and paid for it. On a machine with no torch every help
    invocation exits 1, the sweep comes back empty, and all two dozen cases fail saying the
    documentation names a flag the tool rejects. The reader is then sent to edit a page that was
    never wrong.
    """
    accepted, failures = _sweep()
    assert not (reason := _sweep_failure(accepted, failures)), reason


def test_the_sweep_guard_catches_a_sweep_that_found_nothing():
    """The guard above is only worth having if it fires, so plant the empty sweep and check.

    Two plants, because the two ways a sweep dies read differently to a reader: every invocation
    failing loudly, and every invocation exiting 0 while printing nothing.
    """
    loud = _sweep_failure(set(), (("abliterate --help", 1, "ModuleNotFoundError: torch"),))
    assert loud and "ModuleNotFoundError" in loud and "abliterate --help" in loud
    quiet = _sweep_failure(set(), ())
    assert quiet and "every invocation exited 0" in quiet
    assert _sweep_failure({f"--flag{n}" for n in range(SWEEP_FLOOR)}, ()) is None


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
