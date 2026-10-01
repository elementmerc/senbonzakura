# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A `.gitleaksignore` entry must carry a reason and must not outlive what it was written for.

WHY THIS TEST EXISTS WHILE THE FILE IT POLICES DOES NOT

Measured 2026-10-01: `gitleaks git .` over all 854 commits of this repository's history, 842 of
them carrying additions, 9.25 MB, found **zero** findings. So there is no false-positive set to
collect and nothing to put in a `.gitleaksignore`, and the file is deliberately absent.

**An empty allowlist with a test pointed at nothing would be machinery pretending to be a
control**, which is the same judgement already made about `.github/actionlint.yaml`. This test is
the other thing: a guard that passes honestly while the file is absent and starts biting the
moment somebody writes the first entry. The alternative, writing the policy in a comment and
trusting the next person to find it, is the exact shape this project keeps watching fail.

So the policy is enforced rather than documented:

1. every entry carries a written reason, as a comment on the line above or beside it;
2. every entry names something that still exists, so an exemption cannot outlive its subject.

This mirrors `tests/test_accepted_advisories.py`, which fails the moment the accelerate pin an
accepted advisory was assessed against moves, forcing whoever bumps it to re-read the assessment
rather than inherit it.

WHAT A GITLEAKS IGNORE ENTRY LOOKS LIKE

`.gitleaksignore` holds one fingerprint per line, in the form
`commit:path/to/file:rule-id:line`, or a bare path. Comment lines start with `#`. The path is the
second colon-separated field when there are four, and the whole line when there is one.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
IGNORE = ROOT / ".gitleaksignore"


def _entries(text: str) -> list[tuple[int, str, str]]:
    """Return (line number, entry, the reason attached to it).

    A reason is a trailing `#` comment on the entry's own line, or a `#` comment line immediately
    above it. Either is a reason somebody wrote on purpose; a bare fingerprint is not.
    """
    out: list[tuple[int, str, str]] = []
    previous_comment = ""
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            previous_comment = ""
            continue
        if line.startswith("#"):
            previous_comment = line.lstrip("#").strip()
            continue
        entry, _, inline = line.partition("#")
        reason = inline.strip() or previous_comment
        out.append((number, entry.strip(), reason))
        previous_comment = ""
    return out


def _subject_path(entry: str) -> str | None:
    """The path an entry exempts, or None when the entry is a bare rule fingerprint.

    A four-field fingerprint is `commit:path:rule:line`. Anything with no separator is a path.
    A two or three field entry is ambiguous and is not guessed at: see the test below.
    """
    fields = entry.split(":")
    if len(fields) == 1:
        return fields[0] or None
    if len(fields) >= 4:
        return fields[1] or None
    return None


def test_the_allowlist_is_absent_or_every_entry_carries_a_reason():
    """The first requirement. An exemption nobody justified is how a gate becomes decoration."""
    if not IGNORE.exists():
        # Honest pass. The scan found nothing to exempt, so there is nothing to police yet.
        return
    unreasoned = [
        (number, entry)
        for number, entry, reason in _entries(IGNORE.read_text(encoding="utf-8"))
        if len(reason) < 15
    ]
    assert not unreasoned, (
        f".gitleaksignore has {len(unreasoned)} entry/entries with no written reason, or a reason "
        f"too short to be one: {unreasoned}. Every entry needs a sentence saying what the string "
        f"is and why it is not a secret, either as a comment on the line above it or after a `#` "
        f"on the entry itself. This repository deliberately contains token-shaped fixtures, and "
        f"the difference between a fixture and a leak is a human judgement somebody has to record."
    )


def test_no_allowlist_entry_outlives_the_thing_it_exempts():
    """The second requirement, and the one that stops permanent exceptions accumulating."""
    if not IGNORE.exists():
        return
    stale = []
    for number, entry, _reason in _entries(IGNORE.read_text(encoding="utf-8")):
        subject = _subject_path(entry)
        if subject is None:
            continue
        # A fingerprint pins a path AT A COMMIT, so the path may legitimately be gone from the
        # working tree while the fingerprint stays valid for history. Only a bare path entry,
        # which exempts the file as it is now, has to still exist.
        if ":" in entry:
            continue
        if not (ROOT / subject).exists():
            stale.append((number, subject))
    assert not stale, (
        f".gitleaksignore exempts {len(stale)} path(s) that no longer exist: {stale}. Either the "
        f"file moved, in which case the entry needs updating, or the reason it was exempted is "
        f"gone, in which case so should the entry be. An allowlist nobody is forced to revisit is "
        f"how a known hole becomes a permanent one."
    )


def test_an_ambiguous_entry_is_refused_rather_than_guessed_at():
    """A two or three field entry is neither a path nor a fingerprint, and must not be inferred.

    This is the narrower-question defect in miniature. If this test quietly treated a malformed
    entry as "not a path" and skipped it, a typo in a fingerprint would silently exempt nothing
    while reading as a working exemption, and the next reader would trust it.
    """
    if not IGNORE.exists():
        return
    malformed = [
        (number, entry)
        for number, entry, _ in _entries(IGNORE.read_text(encoding="utf-8"))
        if 2 <= len(entry.split(":")) <= 3
    ]
    assert not malformed, (
        f".gitleaksignore has {len(malformed)} entry/entries that are neither a bare path nor a "
        f"full `commit:path:rule:line` fingerprint: {malformed}. gitleaks will not match them, so "
        f"they exempt nothing while looking like they do. Write the full fingerprint from the "
        f"report, or the bare path."
    )
