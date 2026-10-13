# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The public website cannot go on describing a release that is no longer the current one.

WHY THIS EXISTS

0.5.0 shipped on 2026-10-13. The live site went on saying "0.4.1 on PyPI" in its hero, linking the
v0.4.1 Colab notebook from its install section, and pointing its licence, acceptable-use,
contributing, changelog, reproduction and evidence links at the v0.4.1 tag. Eight references across
six components, every one of them a separate string literal somebody had to remember to change.

It was found by reading the site as an outsider would, not by any check, and it is worse than
merely stale: 0.4.0 and 0.4.1 are being withdrawn from PyPI for a token written into a generated
model card, so the page whose argument is that this project's numbers can be trusted was sending
every reader to a release we are in the middle of yanking.

WHAT IS ASSERTED, AND WHAT DELIBERATELY IS NOT

Asserted: the site declares its release in one place, that declaration matches the version the
project actually ships, and no component has gone back to hardcoding a tag of its own.

Not asserted: any particular version number. Pinning one here would make this file need editing at
every release, which is how a check becomes a chore and then a deleted line. It reads the version
the same way the rest of the tree does.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura._version import __version__  # noqa: E402

WEB = ROOT / "web"
RELEASE_TS = WEB / "lib" / "release.ts"

#: Where the site's own source lives. `out/` is the build output and `node_modules/` is somebody
#: else's code; neither is a place this project writes a version by hand.
SOURCE_DIRS = ("app", "components", "lib")


def _released_version():
    """The version without any `.devN` suffix, which is what a published tag carries."""
    return re.sub(r"\.dev\d+$", "", __version__)


def _sources():
    for d in SOURCE_DIRS:
        yield from sorted((WEB / d).rglob("*.ts*"))


def test_the_site_declares_its_release_in_one_place():
    assert RELEASE_TS.is_file(), f"{RELEASE_TS} is gone, and with it the single declaration"
    text = RELEASE_TS.read_text(encoding="utf-8")
    assert re.search(r"export const RELEASE = '[^']+'", text), \
        "release.ts no longer exports a RELEASE constant"


def test_the_declared_release_is_the_one_this_project_ships():
    text = RELEASE_TS.read_text(encoding="utf-8")
    declared = re.search(r"export const RELEASE = '([^']+)'", text).group(1)
    expected = _released_version()
    assert declared == expected, (
        f"the website says it describes {declared} and this project ships {expected}. "
        f"Update web/lib/release.ts as part of the release, not after somebody notices.")


def test_no_component_hardcodes_a_release_tag_of_its_own():
    """The defect was eight literals in six files, so the guard is against the shape, not the value.

    A `v<number>.<number>` inside a URL is a pinned tag by hand. `release.ts` is allowed it, being
    the one place that is supposed to know, and its prose explains the history.
    """
    offenders = []
    for p in _sources():
        if p == RELEASE_TS:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"/v\d+\.\d+(\.\d+)?/", line):
                offenders.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, (
        "these pin a release tag by hand instead of using atTag() from lib/release.ts:\n  "
        + "\n  ".join(offenders))


def test_no_component_prints_a_version_number_of_its_own():
    """The hero said "0.4.1 on PyPI" as literal text for the whole of 0.5.0's life."""
    offenders = []
    pattern = re.compile(r"\b\d+\.\d+\.\d+\b")
    for p in _sources():
        if p == RELEASE_TS:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            # Skip comments and the dependency-ish lines a version legitimately appears in.
            stripped = line.strip()
            if stripped.startswith(("*", "//", "/*")):
                continue
            if pattern.search(line) and "on PyPI" in line:
                offenders.append(f"{p.relative_to(ROOT)}:{n}: {stripped}")
    assert not offenders, (
        "these state a release version as literal copy instead of reading RELEASE:\n  "
        + "\n  ".join(offenders))


def test_the_built_output_is_not_left_describing_an_older_release():
    """`out/` is what is actually served, so a stale build is a stale site whatever the source says.

    Skipped rather than failed when there is no build here: a checkout that has never run the site
    build is a normal state for somebody working only on the engine.
    """
    import pytest

    index = WEB / "out" / "index.html"
    if not index.is_file():
        pytest.skip("the site has not been built in this checkout")
    html = index.read_text(encoding="utf-8", errors="replace")
    expected = _released_version()
    stale = sorted(set(re.findall(r"/blob/(v\d+\.\d+\.\d+)/", html)) - {f"v{expected}"})
    assert not stale, (
        f"the built site links these tags and this project ships {expected}: {stale}. "
        f"Rebuild web/ before publishing.")
