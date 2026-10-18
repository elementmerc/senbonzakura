#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every plan is in the index, and no two plans share a number.

WHY THIS EXISTS. `private/plans/README.md` is the table a session reads to find out what has
already been decided. It has fallen behind twice and both times it was found by accident:

- On 2026-10-08 it was **22 files behind**, missing everything from 2026-09-11 onward, including
  the live build queue. Somebody asking "what are we building" would have been pointed at the
  ladder rather than at the queue.
- On 2026-10-15 it was **21 files behind** again, and the tree also had **two plan 24s** at once,
  because a plan written overnight took a number that was already taken.

The README has a section titled "Keeping this table true, which it has not been". A heading is not
a mechanism. This is the mechanism.

FAIL-OPEN BY DESIGN. `private/` is excluded from version control and is a symlink into a tree this
repository does not carry, so a fresh clone and most CI runners have no plans directory at all.
Finding nothing is therefore the normal case off the operator's machine and must never fail a
build. **An empty tree is reported as skipped rather than as passed**, because "nothing to check"
and "everything checked out" are different claims and only one of them is evidence.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: Files in the plans directory that are not plans and are not expected in the table.
NOT_A_PLAN = frozenset({"README.md"})

#: A plan whose name starts with digits owns that number. Date-named plans carry no number and are
#: exempt from the collision rule, which is why the pattern anchors on a short run of digits
#: followed by a hyphen rather than on anything date-shaped.
NUMBERED = re.compile(r"^(\d{1,3})-")


def plan_files(plans: Path) -> list[Path]:
    return sorted(p for p in plans.glob("*.md") if p.name not in NOT_A_PLAN)


def missing_from_index(index_text: str, files: list[Path]) -> list[str]:
    """Plan files the index never names. Substring, because the table styles entries in backticks."""
    return [p.name for p in files if p.name not in index_text]


def duplicate_numbers(files: list[Path]) -> dict[str, list[str]]:
    """Plan numbers claimed by more than one file."""
    seen: dict[str, list[str]] = {}
    for p in files:
        m = NUMBERED.match(p.name)
        if not m:
            continue
        # Leading zeros are cosmetic: `09-` and `9-` would be the same rung to a reader.
        seen.setdefault(str(int(m.group(1))), []).append(p.name)
    return {n: v for n, v in seen.items() if len(v) > 1}


def stale_index_entries(index_text: str, plans: Path) -> list[str]:
    """Names the index points at that are not there any more.

    The opposite failure from a missing entry and the more misleading one: a missing plan is
    invisible, while a dead row sends a reader to a file that does not exist and looks like the
    answer until they try to open it.
    """
    # ROWS OF THE TABLE ONLY, not every backticked filename in the document. The first version
    # scanned the whole text and flagged `DEFERRED.md`, which the prose mentions and which lives at
    # the repository root rather than here: a check that fires on a correct document trains people
    # to ignore it, which costs more than the thing it was looking for.
    named = set()
    for line in index_text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cell = line.lstrip().lstrip("|").split("|", 1)[0].strip()
        m = re.fullmatch(r"`([0-9A-Za-z][-0-9A-Za-z._]*\.md)`", cell)
        if m:
            named.add(m.group(1))
    return sorted(n for n in named if not (plans / n).is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="check_plan_index.py",
        description="Every plan is named in private/plans/README.md, and no number is claimed twice.")
    ap.add_argument("--plans", default="private/plans",
                    help="the plans directory (default: private/plans)")
    a = ap.parse_args(argv)

    plans = Path(a.plans)
    index = plans / "README.md"
    if not plans.is_dir() or not index.is_file():
        print(f"check_plan_index: SKIPPED, no plan tree at {plans}. This is expected anywhere but "
              f"the operator's machine, because `private/` is not version controlled. "
              f"NOTHING WAS CHECKED; this is not a pass.")
        return 0

    text = index.read_text(encoding="utf-8")
    files = plan_files(plans)
    if not files:
        print(f"check_plan_index: SKIPPED, {plans} holds no plan files. NOTHING WAS CHECKED.")
        return 0

    problems = 0

    missing = missing_from_index(text, files)
    if missing:
        problems += len(missing)
        print(f"check_plan_index: {len(missing)} plan(s) are not in the index:")
        for name in missing:
            print(f"    {name}")
        print("  Add a row to private/plans/README.md saying what each one answers. A plan nobody "
              "can find is a decision that gets taken twice.")

    dupes = duplicate_numbers(files)
    if dupes:
        problems += len(dupes)
        print(f"check_plan_index: {len(dupes)} plan number(s) claimed twice:")
        for number, names in sorted(dupes.items()):
            print(f"    {number}: {', '.join(names)}")
        print("  Renumber one of them to the next free number and fix its own heading. Two files "
              "with one number means a reference to that number points at either.")

    stale = stale_index_entries(text, plans)
    if stale:
        problems += len(stale)
        print(f"check_plan_index: {len(stale)} index row(s) point at a file that is gone:")
        for name in stale:
            print(f"    {name}")
        print("  Remove the row, or restore the file. A dead row looks like an answer until a "
              "reader tries to open it.")

    if problems:
        return 1
    print(f"check_plan_index: {len(files)} plans, all indexed, no number claimed twice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
