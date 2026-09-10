#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Do the CUDA channels `senbonzakura setup` recommends still exist? Run this at every release.

THE OBLIGATION THIS ENFORCES

`envsetup.CUDA_CHANNELS` maps a driver's CUDA ceiling to a PyTorch channel, and `senbonzakura
setup` prints an index URL built from it. The list was read off download.pytorch.org on one day
and is otherwise a hardcoded snapshot of somebody else's release schedule. Two things go wrong as
it ages, and they are not the same fault:

  a channel we name that is GONE      the command prints a URL that 404s, and the user finds out
                                      by running it. Worse than no advice.
  a channel published that we MISS    a newer card gets an older build than it could have. A
                                      missed opportunity rather than a broken instruction.

WHY IT IS A TOOL AND NOT A TEST

It needs the network, and the same argument as `check_vendor_pins.py` applies: a test that
reaches the internet is slow in every run, fails on a train, and gets marked skip. So the
arithmetic lives in `senbonzakura.envsetup` as pure functions with tests, this fetches the index
and calls them, and RELEASING.md runs it once per release where the cost is paid deliberately.

FAILURE POSTURE

Exit 0: every channel we name is published, or only newer ones exist and are reported.
Exit 1: a channel we name has gone, or the table is past its staleness limit.
An unreachable index is reported as UNCHECKED and does NOT fail: that is a network problem, not a
stale table, and blocking a release on a flaky connection teaches people to pass --force.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from senbonzakura import envsetup

URL = "https://download.pytorch.org/whl/"

#: The index is a plain HTML directory listing. Anchors, not a parser: adding a dependency to
#: read eight href attributes once per release is the trade this project keeps declining.
_HREF = re.compile(r'href="\.?/?([a-z0-9_.]+)/?"', re.IGNORECASE)


def published_channels(url=URL, timeout=30):
    """Channel names the index actually offers, or None when it could not be asked."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 - fixed https URL
            body = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return {m.group(1).lower() for m in _HREF.finditer(body)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--today", help="ISO date, for testing the staleness arithmetic")
    a = p.parse_args(argv)

    today = date.fromisoformat(a.today) if a.today else datetime.now(UTC).date()

    age = envsetup.channels_age_days(today)
    print(f"  table checked  {envsetup.CHANNELS_CHECKED} ({age} days ago, limit "
          f"{envsetup.CHANNELS_STALE_AFTER_DAYS})")
    print(f"  channels       {', '.join(tag for _, tag in envsetup.CUDA_CHANNELS)}")

    failed = False
    if envsetup.channels_are_stale(today):
        print(f"  PROBLEM: the channel table has not been checked for {age} days. Re-read "
              f"{URL}, update envsetup.CUDA_CHANNELS and CHANNELS_CHECKED.")
        failed = True

    found = published_channels()
    if found is None:
        print(f"  UNCHECKED: {URL} could not be reached, so nothing is known about the channels "
              f"themselves. This is a network problem and does not fail the release.")
        return 1 if failed else 0

    problems = envsetup.channel_problems(found)
    for tag in problems["gone"]:
        print(f"  PROBLEM: `{tag}` is in our table and NOT published. `senbonzakura setup` would "
              f"print an index URL that 404s, and the reader would run it before finding out.")
        failed = True
    if problems["newer"]:
        print(f"  NOTE: newer channels exist and we do not offer them: "
              f"{', '.join(problems['newer'])}. A card whose driver could carry one of these gets "
              f"an older build than it could have. Not a failure; add them and refresh the date.")

    if failed:
        print("\nFAILED: the CUDA channel table cannot be trusted as it stands.")
        return 1
    print("\nOK: every channel this tool recommends is one PyTorch publishes.")
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
