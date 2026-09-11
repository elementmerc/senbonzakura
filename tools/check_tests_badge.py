#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The README's `tests-NNNN` badge states the number of tests that actually exist.

WHY THIS IS A SCRIPT AND NOT A TEST

The number to compare against is "how many tests does the whole suite have", and a test cannot
ask that about itself without running the whole suite a second time. Collection is the cheap
half of a run, so this does exactly one collection pass and compares.

It runs in CI on one matrix row, beside the Codecov upload, for the same reason that upload
exists: a figure printed to the public that nothing measures will drift, and it will drift in
the flattering direction, because nobody notices a number that went up.

WHAT MAKES THE COUNT STABLE

Skipped tests are collected. A runner with none of the release artefacts skips twenty-odd
tests and still collects them, so the figure does not depend on what the machine happens to
have. Parametrised counts are derived from files in the tree, so they are the same everywhere.

EXIT CODES
  0  the badge matches
  1  the badge is wrong, or there is no badge to check
  2  collection itself failed, which is a broken suite rather than a stale badge
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: `https://img.shields.io/badge/tests-3166-0A9EDC?...`
BADGE = re.compile(r"img\.shields\.io/badge/tests-(?P<count>\d+)-")

#: pytest's final collection line: `3172 tests collected in 4.21s`, or `1 test collected`.
COLLECTED = re.compile(r"^(?P<count>\d+) tests? collected", re.MULTILINE)


def collected_count():
    got = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-cov", "-p", "no:randomly"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    m = COLLECTED.search(got.stdout)
    if got.returncode != 0 or m is None:
        print("collection failed, so there is no number to compare the badge against:")
        print(got.stdout[-4000:])
        print(got.stderr[-2000:])
        sys.exit(2)
    return int(m.group("count"))


def main():
    text = README.read_text(encoding="utf-8")
    m = BADGE.search(text)
    if m is None:
        print("README has no `tests-NNNN` badge. If it was removed deliberately, remove this "
              "check with it; if it was renamed, teach the pattern here.")
        return 1

    claimed, real = int(m.group("count")), collected_count()
    if claimed == real:
        print(f"tests badge says {claimed}, suite collects {real}")
        return 0

    direction = "more" if real > claimed else "fewer"
    print(f"the README badge says {claimed} tests and the suite collects {real}, which is "
          f"{abs(real - claimed)} {direction}.\n"
          f"Update the badge URL in README.md to `tests-{real}-`.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
