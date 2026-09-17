# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Nothing tracked in this repository names the machines it was built on.

This repository has a public remote. Everything git tracks is published, including
the comments, the docstrings and the test names, and those are where private detail
leaks: not through a credential, which the commit hooks already scan for, but through
a reader slowly learning the shape of somebody's network from thirty incidental
mentions of where a bug was found.

The finding that produced this file, 2026-09-17: twenty-nine tracked files named
machines, a job runner and home paths. None of them was a secret on its own. Together
they described a fleet. A comment saying a defect was found on a particular box
carries no more information for the reader than one saying it was found on a Windows
laptop with a card in it, and the second is the useful sentence anyway, because it
says what about that machine mattered.

WHAT REPLACES A NAME. Describe the machine by what it has. "A CPU-only box", "a
Windows laptop", "a 6 GB card", "a second machine", "a peer session". Those are what
the reader needs to judge whether the measurement applies to them, which a nickname
never told them.

WHY THE TERMS ARE NOT IN THIS FILE, which is the part worth reading twice. A list of
the names that must not be published is a list of the names, and committing it to a
public repository publishes every one of them with a label explaining why they matter.
The check is tracked; what it looks for is not. The terms live in an untracked local
file, and when that file is absent the check says so loudly and skips rather than
passing quietly, because a guard that reports success having inspected nothing is the
failure this project has now hit four separate times.

The consequence, stated plainly: this check does NOT run in CI, because CI has no
copy of the list and must not be given one. It runs on a machine that has the list,
which is the same machine the leak would originate from. That is the whole enforceable
surface, and it is better than the alternative of a public denylist.
"""
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

#: One term per line, `#` for comments, each line a regular expression. Lives under the
#: private tree, which `.git/info/exclude` keeps out of every commit, so the list itself
#: is never published. See this module's docstring for why that is not optional.
TERMS_FILE = _ROOT / "private" / "infrastructure-terms.txt"

#: There is deliberately no exemption list. The first version of this file carried one,
#: for the operator tooling that was tracked despite living behind `.git/info/exclude`,
#: and the honest fix turned out to be untracking that tooling rather than teaching the
#: check to look away from it. If any of it comes back, this should fail.


def _load_terms():
    if not TERMS_FILE.is_file():
        return None
    out = []
    for raw in TERMS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _tracked_text_files():
    # NOT check=True. The suite is deliberately run from a `git archive` extract and inside
    # containers, where this is not a checkout and git exits non-zero. Raising there turns a
    # condition this check cannot inspect into a suite error, which is noise rather than a
    # finding; the empty result makes the fixture skip and say why. Found on the build box,
    # 2026-09-17, the first time this file ran anywhere but the machine that wrote it.
    try:
        proc = subprocess.run(
            ["git", "-C", str(_ROOT), "ls-files", "-z"],
            capture_output=True, text=True, check=False)
    except OSError:  # git absent entirely
        return
    if proc.returncode != 0:
        return
    out = proc.stdout
    here = Path(__file__).name
    for name in out.split("\0"):
        if not name or Path(name).name == here:
            continue
        p = _ROOT / name
        if not p.is_file():
            continue
        try:
            yield name, p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


@pytest.fixture(scope="module")
def terms():
    loaded = _load_terms()
    if loaded is None:
        pytest.skip(
            f"no term list at {TERMS_FILE.relative_to(_ROOT)}, so this check inspected "
            "NOTHING. It is deliberately not committed; create it locally, one regular "
            "expression per line, to arm this guard on this machine.")
    if not loaded:
        pytest.fail("the term list exists and is empty, which arms nothing")
    return loaded


@pytest.fixture(scope="module")
def tracked():
    files = list(_tracked_text_files())
    if not files:
        pytest.skip("not a git checkout, so there is nothing tracked to inspect")
    return files


def test_no_tracked_file_names_private_infrastructure(tracked, terms):
    """Every tracked line, against every term. One failure listing all of them.

    Reported as one assertion rather than one per term on purpose: the reader wants the
    whole surface in front of them when they start a sweep, not a test run that reveals
    the next name each time they fix the last one.
    """
    rx = re.compile("|".join(f"(?:{t})" for t in terms), re.IGNORECASE)
    hits = []
    for name, text in tracked:
        for n, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{name}:{n}: {line.strip()[:100]}")
    assert not hits, (
        f"{len(hits)} tracked line(s) name private infrastructure, and this repository "
        "is public. Describe the machine by what it has, not by what it is called:\n  "
        + "\n  ".join(hits[:20])
        + ("" if len(hits) <= 20 else f"\n  ... and {len(hits) - 20} more"))


def test_the_committed_ignore_file_does_not_advertise_the_private_trees(terms):
    """A public `.gitignore` naming a private directory tells every visitor it exists.

    The per-clone `.git/info/exclude` is where those rules belong. Same reasoning as the
    rest of this file, one layer down: the rule keeps the tree out, and where the rule is
    written decides whether the tree's name goes out.
    """
    gitignore = _ROOT / ".gitignore"
    if not gitignore.is_file():
        pytest.skip("no committed ignore file")
    rx = re.compile("|".join(f"(?:{t})" for t in terms), re.IGNORECASE)
    named = [
        line for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and rx.search(line)
    ]
    assert not named, (
        "the committed ignore file names a private tree: " + "; ".join(named))
