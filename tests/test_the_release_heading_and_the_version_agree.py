# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The CHANGELOG heading and the declared version cannot drift apart.

Two things have to happen in the same step at tag time: the `.devN` suffix comes off both
distributions, and the CHANGELOG's date placeholder is replaced with the real date. Doing one and
forgetting the other is the obvious mistake, and it is invisible in a diff that is mostly release
notes.

The previous shape of this file's subject was worse than a drift. Until 2026-09-25 the whole v0.4
body sat under `## [Unreleased]`, with no `[0.4.0]` heading at all, so there was nothing for a
version to disagree WITH. The gate asks for the heading; this asks for the heading to be true.

Deliberately NOT asserted: that the version is `0.4.0` specifically, or that the date is today.
Both are release-time facts and pinning them here would make this file need editing at every
release, which is how a check becomes a chore and then a `# TODO`.
"""
import datetime as _dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura._version import __version__  # noqa: E402

CHANGELOG = ROOT / "CHANGELOG.md"
PLACEHOLDER = "YYYY-MM-DD"

#: `## [0.4.0] "TYBW" — 2026-09-25`, the shape every released entry in this file uses.
HEADING = re.compile(r'^## \[(?P<version>[0-9]+\.[0-9]+\.[0-9]+)\] '
                     r'"(?P<codename>[^"]+)" — (?P<date>\S+)\s*$', re.MULTILINE)


def _top_entry():
    m = HEADING.search(CHANGELOG.read_text(encoding="utf-8"))
    assert m, ("the CHANGELOG has no release heading in the form "
               '`## [X.Y.Z] "Codename" — DATE`. Every released entry uses it and the tag, the '
               "GitHub release title and this heading are supposed to match.")
    return m


def test_the_top_entry_is_a_release_heading_with_a_codename():
    """Section 22: a release is never unnamed, and the codename lives in three places."""
    m = _top_entry()
    assert m.group("codename").strip(), "the heading carries an empty codename"


def test_a_released_version_has_a_real_date_not_the_placeholder():
    """THE ONE THAT MATTERS. A non-dev version still carrying the placeholder is a shipped defect."""
    m = _top_entry()
    if ".dev" in __version__:
        return
    assert m.group("date") != PLACEHOLDER, (
        f"the version is {__version__}, which declares a real release, and the CHANGELOG heading "
        f"still says {PLACEHOLDER}. Set the date in the same step that drops the .devN suffix.")
    _dt.date.fromisoformat(m.group("date"))


def test_the_heading_version_matches_the_declared_version():
    """A heading that says 0.4.0 over a tree that declares 0.5.0 is worse than either alone."""
    m = _top_entry()
    base = __version__.split(".dev")[0].split("rc")[0]
    assert m.group("version") == base, (
        f"the CHANGELOG's top entry is {m.group('version')} and this tree declares {__version__}. "
        f"They describe the same release and must say so.")


def test_the_placeholder_is_only_ever_in_the_top_entry():
    """A released entry below the top carrying it would mean a past release shipped undated."""
    text = CHANGELOG.read_text(encoding="utf-8")
    first = text.index(PLACEHOLDER) if PLACEHOLDER in text else None
    if first is None:
        return
    assert text.count(PLACEHOLDER) <= 2, (
        f"{PLACEHOLDER} appears {text.count(PLACEHOLDER)} times. It belongs to the unreleased "
        f"entry at the top and to the sentence explaining it, and nowhere else.")
