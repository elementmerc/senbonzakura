#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Whatever a built image carries, it must be honest about it.

WHY THIS EXISTS

The container job asserted that the built image reports the bundled corpora as present, and the
comment directly above that assertion explained why it could not hold: Docker's build context
ignores `.gitignore`, so an image built where the generated blobs exist has them and the same build
from a clone does not. A runner is always the second case. The check was unsatisfiable there from
the day it was written, and the job had been red since 2026-08-03.

Asserting the opposite would be no better: a RELEASE image is built where the blobs are, and would
then fail a check demanding their absence.

So this asserts the property that holds either way, which is the one that actually matters: an
image with corpora must load them at their declared size, and an image without must REFUSE and
name the tool that builds them. A silently empty image is the failure that ships, because it
builds, imports, answers `--help`, and fails `--track default` for every user.

This is the same discipline `tools/clean_room_checks.py` applies to a code-only wheel.

A FILE RATHER THAN A `python -c` ONE-LINER, because the first version of this check was one: an
indented block passed to `python -c`, which is an IndentationError, so the job failed on the check
instead of on the thing being checked. A file can be run, and tested, on its own.
"""
import sys

from senbonzakura import corpora
from senbonzakura.corpora import CorpusError


def main():
    try:
        rows = corpora.load("advbench")
    except CorpusError as e:
        if "build_corpora.py" not in str(e):
            print(f"FAILED: this image has no corpora and its refusal does not name the tool that "
                  f"builds them, so a user cannot act on it. The message was: {e}", file=sys.stderr)
            return 1
        print("no corpora in this image, and it says so and names the builder: OK")
        return 0

    want = corpora.CORPORA["advbench"].rows
    if len(rows) != want:
        print(f"FAILED: advbench loaded {len(rows)} rows and declares {want}. An image carrying a "
              f"corpus of the wrong size is worse than one carrying none, because nothing about "
              f"it looks wrong.", file=sys.stderr)
        return 1
    print(f"corpora present and loading at their declared size ({len(rows)} rows): OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
