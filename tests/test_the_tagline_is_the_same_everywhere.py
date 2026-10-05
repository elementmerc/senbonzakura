# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The tagline reads the same on every surface a stranger meets it on.

WHAT PROMPTED IT, 2026-10-05

The tagline changed from "Precision abliteration" to "Precision uncensoring", because the word
people actually search for and respond to is the second one. It had to be changed by hand in
**eight** places: the README, the documentation landing page, the VitePress site description, the
Open Graph card, the PyPI description, both container image labels, the brand asset generator and
the banner the CLI prints.

Nothing guarded that. A previous session's note recorded the tagline as shipping to "all five
surfaces", and by today there were eight, so the count had already drifted once without anybody
noticing. Changing it in seven of eight is the obvious failure and the eighth is whichever one the
person doing it did not know about: most likely a container label, because nobody reads those until
a registry page looks wrong.

WHY A TEST RATHER THAN A CONSTANT

A constant would be the better engineering and is not available. These eight live in Python, in
Markdown, in TOML, in JavaScript and in two Dockerfiles, and four of them are read by tools that
cannot import anything of ours: `pip` reads the TOML, the registry reads the label, VitePress reads
the config at build time. So the string is genuinely duplicated by necessity, and what a test can
do is make the duplication honest.

THE PRIVATE TREE AND THE WORKTREES ARE EXCLUDED, deliberately. `private/` holds design notes and
plans that quote the old tagline as a record of what it used to be, and rewriting history to match
the present is how a project loses the ability to say what it once believed. A git worktree under
`.claude/` holds its own frozen copy of everything, which is not a published surface.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The one true wording. Changing the tagline means changing it here and then watching this test
#: name every surface that still disagrees, which is the whole point of the list below.
TAGLINE = "Precision uncensoring, with receipts"

#: Every surface a stranger meets the tagline on, with what reads it, because the reason a surface
#: matters is what consumes it rather than what language it is written in.
SURFACES = {
    "README.md": "the first thing a visitor to the repository reads",
    "docs/index.md": "the documentation site's hero",
    "docs/.vitepress/config.mjs": "the site description and the Open Graph card",
    "pyproject.toml": "the PyPI project page and `pip show`",
    "Dockerfile": "the container image label, which the registry page renders",
    "Dockerfile.cuda": "the GPU image's label, same reader",
    "tools/packaging/build_brand_assets.py": "the generated banner images",
    "src/senbonzakura/entry.py": "the banner the tool prints to its own users",
}

#: The superseded wording. Present anywhere outside the private tree, it means a surface was
#: missed rather than that somebody chose the old words.
RETIRED = "Precision abliteration"


@pytest.mark.parametrize("path", sorted(SURFACES))
def test_every_surface_carries_the_current_tagline(path):
    """Each listed surface contains the tagline, and the failure says what that surface feeds."""
    p = _ROOT / path
    if not p.is_file():
        pytest.skip(f"{path} is absent from this tree, so there is nothing to check")
    text = p.read_text(encoding="utf-8", errors="replace")
    assert TAGLINE in text, (
        f"{path} does not carry the tagline {TAGLINE!r}, and it is {SURFACES[path]}. "
        f"A tagline that differs by surface is one a reader meets twice and trusts less.")


def test_the_retired_tagline_survives_nowhere_public():
    """The old wording is gone from everything git publishes.

    Reported as one failure listing every file, because somebody fixing these one test run at a
    time is somebody who runs the suite eight times.

    WHAT "PUBLISHED" MEANS HERE, and the first version of this test got it wrong. It walked the
    working tree, which found `build/` and `src/senbonzakura.egg-info/PKG-INFO`: both generated,
    both untracked, both carrying a stale tagline because they were built before the rename. That
    is not a missed surface, it is yesterday's output, and failing on it trains a reader to ignore
    the test. Asking git what it tracks is the question actually being asked, and it excludes the
    private tree, the worktrees and every build directory for free rather than by a skip list that
    has to be maintained.
    """
    try:
        proc = subprocess.run(["git", "-C", str(_ROOT), "ls-files", "-z"],
                              capture_output=True, text=True, check=False)
    except OSError:
        pytest.skip("git is not available, so what is published cannot be determined")
    if proc.returncode != 0:
        pytest.skip("this is not a checkout, so what is published cannot be determined")
    tracked = [n for n in proc.stdout.split("\0") if n]
    assert tracked, "git tracks nothing here, so this check inspected nothing and says so"

    me = pathlib.Path(__file__).relative_to(_ROOT).as_posix()
    found = []
    for rel in tracked:
        # This file documents the retired wording on purpose, so it is the one exemption and it is
        # named rather than pattern matched.
        if rel == me:
            continue
        p = _ROOT / rel
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError, FileNotFoundError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if RETIRED in line:
                found.append(f"{rel}:{n}")
    assert not found, (
        f"the retired tagline {RETIRED!r} is still on {len(found)} published line(s), so a "
        f"rename stopped part way:\n  " + "\n  ".join(found))


def test_the_surface_list_has_not_quietly_shrunk():
    """A surface removed from the list above is a surface that stops being checked.

    The count is asserted rather than the names, because adding a surface should be easy and
    deleting one should need a reason. This is the check the 2026-10-05 rename wished it had: the
    previous note said five surfaces when there were eight, and nothing noticed.
    """
    assert len(SURFACES) >= 8, (
        f"the tagline surface list is down to {len(SURFACES)}. If a surface genuinely went away, "
        f"lower this number in the same commit that removes it and say which one in the message.")


def test_the_tagline_says_uncensoring_rather_than_the_technical_term():
    """The lay-facing wording is the point of the change, so it is pinned.

    `abliteration` is the correct technical term and stays in the command name, the artefact
    filenames and the prose that explains the method. It is not what a stranger searches for, and
    the tagline is the one line written entirely for strangers.
    """
    assert re.search(r"\buncensoring\b", TAGLINE), (
        "the tagline exists to be legible to somebody who has never heard the word abliteration. "
        "If the technical term belongs back in it, that is a decision worth recording rather than "
        "a test worth deleting.")
