#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Validate every contributed probe in the repository, and say what a human still has to do.

WHY THIS RUNS IN CI RATHER THAN ON REQUEST

`probes/` holds probes contributed by other people and distributed with the project. Anyone who
installs it gets them. A format gate that somebody has to remember to run is a gate that runs
occasionally, and the moment it matters is the pull request nobody looked at closely.

WHAT IT CAN AND CANNOT DO, STATED HERE BECAUSE IT DECIDES WHAT THE REVIEW IS FOR

It checks SHAPE. A probe declares what it measures, names a grading rule that exists, carries a
licence and a pinned source, uses the item columns rather than prompt-shaped ones, and holds items
the rule can actually read.

It does not read the items and judge them. Doing that would need a harmfulness classifier, and
this project does not ship instruments it has not measured; an unvalidated one here would be worse
than none, because a green tick from it would be believed.

So the gate is necessary and is not sufficient, and this script says so on every run rather than
leaving it implied. The load-bearing control on contributed content is a person reading it.

    python tools/ci/check_probes.py             # every probe under probes/
    python tools/ci/check_probes.py PATH ...    # named directories
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBES = ROOT / "probes"

# The package is the source of truth for what a probe is. Importing it rather than restating the
# rules is deliberate: a CI copy of the validation would drift from the loader, and a probe that
# passes CI and then fails to load is the worst of both.
sys.path.insert(0, str(ROOT / "src"))


def _probe_dirs(paths):
    from senbonzakura import probe
    if paths:
        return [pathlib.Path(p) for p in paths]
    if not PROBES.is_dir():
        return []
    return sorted(d for d in PROBES.iterdir() if probe.is_probe(d))


def main(argv):
    from senbonzakura import probe

    directories = _probe_dirs(argv)
    if not directories:
        print("no contributed probes to check")
        return 0

    failed = 0
    for directory in directories:
        problems = probe.problems_with(directory)
        rel = directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory
        if problems:
            failed += 1
            print(f"REFUSED {rel}")
            for p in problems:
                print(f"  - {p}")
        else:
            loaded = probe.load(directory)
            print(f"ok      {rel}: {len(loaded.items)} items, graded by {loaded.task!r}, "
                  f"measuring {loaded.measures!r}")

    print()
    if failed:
        print(f"{failed} of {len(directories)} probe(s) refused.")
        return 1
    print(f"{len(directories)} probe(s) pass the format gate.")
    # Printed on success, deliberately. A gate that only speaks when it fails teaches readers that
    # silence means safe, and what silence means here is "the shape is right and nobody has read
    # the contents yet".
    print("The format gate checks SHAPE only. It cannot tell whether these items measure what "
          "their author says they measure.")
    print("A contributed probe is merged when a person has read it. See probes/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
