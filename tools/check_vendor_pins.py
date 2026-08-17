#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Are the vendored third-party pins current? Run this at every release.

THE OBLIGATION THIS ENFORCES

Baseline Section 5 sets a cooldown: adopt nothing younger than seven days, because most
compromised releases are caught within days. The operator's rule of 2026-08-17 adds the half a
cooldown leaves out: **every release refreshes every pin to the newest version that has cleared
the cooldown.** Adopt nothing young; adopt everything that has aged. Without the second half a
pin freezes at whatever was current the day it was written and the project quietly ships a
year-old dependency with a year of known bugs.

WHY IT IS A TOOL AND NOT A GIT HOOK

It needs the network. A hook that reaches the internet is slow on every push, fails on a train,
and gets disabled. So the arithmetic lives in `senbonzakura.vendoring` as pure functions with
tests, this fetches the release list and calls them, and RELEASING.md runs it once per release
where the cost is paid deliberately.

FAILURE POSTURE

Exit 0: every pin current, or a refresh is due and reported. Exit 1: a pin is past the staleness
limit, or the manifest is unreadable. An unreachable upstream is reported as UNCHECKED and does
not fail: it is a network problem, not a stale pin, and blocking a release on a flaky connection
teaches people to pass --force. It is never reported as "current", because not being able to look
is not the same as having looked.

    python tools/check_vendor_pins.py
    python tools/check_vendor_pins.py --offline    # skip the fetch, report ages only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from senbonzakura import vendoring

RELEASES_URL = "https://api.github.com/repos/{repo}/releases?per_page=100"
TIMEOUT_S = 20


def fetch_releases(repo, *, timeout=TIMEOUT_S, opener=None):
    """`[(tag, published), ...]` for a GitHub repo, or None if it could not be read.

    None rather than an exception, because an unreachable upstream must degrade to UNCHECKED
    rather than take a release down. The distinction is made here so the caller cannot
    accidentally treat a failure as an empty list, which would read as "nothing upstream" and
    quietly report every pin as current.
    """
    url = RELEASES_URL.format(repo=repo)
    req = urllib.request.Request(url, headers={     # noqa: S310 - fixed https host
        "Accept": "application/vnd.github+json",
        "User-Agent": "senbonzakura-vendor-pin-check",
    })
    try:
        with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
            doc = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(doc, list):
        return None
    return [(d["tag_name"], d["published_at"]) for d in doc
            if d.get("tag_name") and d.get("published_at")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=None, help="pins.json to check (default: the shipped one)")
    ap.add_argument("--offline", action="store_true",
                    help="do not fetch; report each pin's age and check nothing against upstream")
    ap.add_argument("--today", default=None,
                    help="ISO date to evaluate against, for testing the boundaries")
    a = ap.parse_args(argv)

    today = a.today or dt.datetime.now(dt.timezone.utc).date().isoformat()
    try:
        manifest = vendoring.load_manifest(a.manifest)
    except vendoring.VendorError as e:
        print(f"vendor pins: {e}", file=sys.stderr)
        return 1

    candidates = {}
    if not a.offline:
        # One fetch per distinct repo: two pins commonly come from the same upstream, and
        # asking twice is a rate limit waiting to happen.
        by_repo = {}
        for key, pin in manifest["pins"].items():
            by_repo.setdefault(pin.get("repo"), []).append(key)
        for repo, keys in by_repo.items():
            if not repo:
                continue
            rels = fetch_releases(repo)
            if rels is None:
                continue
            for key in keys:
                candidates[key] = rels

    check = vendoring.check_all(manifest, candidates, today)
    print(f"vendored pins, checked {today} against a "
          f"{manifest.get('cooldown_days', vendoring.COOLDOWN_DAYS)}-day cooldown:")
    print(vendoring.format_report(check))
    return 0 if check["may_release"] else 1


if __name__ == "__main__":
    sys.exit(main())
