# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every `docker run` a reader can copy names the release that is actually published.

The v0.5.1 release repinned the README's image tag and missed `docs/guide/install.md`, which went
on telling readers to run `:v0.5.0`. A stale tag is worse than it looks: the paragraph around it
offers the container as the fix for a missing system library, so the one reader who follows it is
the reader already stuck, and they get the older image without being told.

The companion guard over the Colab link exists for the same class of mistake and caught its own
case. This one covers the other pinned reference a release has to move, and it is deliberately
written over text so a stale tag can be fed to it. A guard that has only ever been shown passing
documentation that is already correct has not been shown to work.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Public surfaces that carry a copyable `docker run`. A reader meets the image on one of these.
SURFACES = ("README.md", "docs/guide/install.md")

IMAGE = re.compile(r"ghcr\.io/elementmerc/senbonzakura:([^\s\"']+)")


def image_tags_in(text):
    """Every image tag a reader could copy out of this text, in the order they appear."""
    return IMAGE.findall(text)


def newest_release_tag():
    """The newest vX.Y.Z tag, or None where the clone has no tags (a shallow CI checkout)."""
    done = subprocess.run(["git", "tag", "--list", "v*", "--sort=-v:refname"],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    present = [t for t in done.stdout.split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    return present[0] if present else None


def test_the_documented_image_tag_is_the_newest_release():
    """The real surfaces, as committed.

    The shape claim holds in any checkout. The newest-tag claim is checked only where tags are
    present, and where they are absent it returns rather than skipping, because a skip reads
    exactly like a pass in a tail of the output and this file has already seen that cost once.
    """
    found = []
    for rel in SURFACES:
        path = ROOT / rel
        assert path.is_file(), f"{rel} does not exist, so this guard is measuring nothing"
        found.extend((rel, tag) for tag in image_tags_in(path.read_text(encoding="utf-8")))

    assert found, (
        f"no `ghcr.io/elementmerc/senbonzakura:` reference found in any of {list(SURFACES)}, so "
        f"this guard is measuring nothing. If the image moved, move this guard with it.")

    for rel, tag in found:
        assert tag not in ("latest", "dev", "main"), (
            f"{rel} names the moving tag `{tag}`, so a reader runs whatever was pushed last "
            f"instead of the release the documentation around it describes. Name a version.")
        assert re.fullmatch(r"v\d+\.\d+\.\d+", tag), (
            f"{rel} names the image tag `{tag}`, which is not a release tag of the form vX.Y.Z")

    newest = newest_release_tag()
    if newest is None:
        return
    stale = [(rel, tag) for rel, tag in found if tag != newest]
    assert not stale, (
        f"the newest release tag is {newest}, but these name an older image: {stale}. A release "
        f"has to repin every one of them, and forgetting one is what this guard is for.")


def test_the_guard_refuses_a_stale_tag():
    """Force the failure, so the test above is known to be capable of it.

    This is the actual v0.5.1 mistake, as text: the README moved and the install guide did not.
    """
    newest = newest_release_tag()
    if newest is None:
        return

    moved = f"docker run --rm ghcr.io/elementmerc/senbonzakura:{newest} doctor"
    left_behind = "docker run --rm ghcr.io/elementmerc/senbonzakura:v0.0.1 doctor"

    assert image_tags_in(moved) == [newest]
    assert image_tags_in(left_behind) == ["v0.0.1"]

    found = [("README.md", newest), ("docs/guide/install.md", "v0.0.1")]
    stale = [(rel, tag) for rel, tag in found if tag != newest]
    assert stale == [("docs/guide/install.md", "v0.0.1")], (
        "the comparison the guard above relies on did not single out the stale surface, so that "
        "guard cannot catch the mistake it was written for")


def test_the_guard_refuses_a_moving_tag():
    """`:latest` is the other way this goes wrong, and it passes a shape check that only tests digits."""
    assert image_tags_in("run ghcr.io/elementmerc/senbonzakura:latest doctor") == ["latest"]
    assert not re.fullmatch(r"v\d+\.\d+\.\d+", "latest")
