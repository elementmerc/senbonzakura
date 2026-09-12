# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The README's images point at something that exists, and keep pointing at it.

WHAT WAS BROKEN, and for how long

The banner was served from
`raw.githubusercontent.com/elementmerc/senbonzakura/main/assets/brand/readme-banner.png`. The
assets were added to `dev` on 2026-08-16 and `main` has not moved since well before that, so
`origin/main` carries no `assets/` directory at all and both URLs returned 404. README.md is also
`readme = "README.md"` in pyproject, so that is the PyPI project page: the first thing a stranger
saw was a broken image, on GitHub and on PyPI both.

`RELEASING.md` step 4a says "fast-forward `main` before the PyPI step, or the README banner ships
broken", which describes the failure exactly and treats it as a release-ordering problem. It is
not only that: `main` was 386 commits behind, so the banner was broken continuously rather than
during a release window.

THE FIX, AND WHY A COMMIT SHA

The URLs are pinned to the commit that holds the assets. A branch name in an asset URL is a
promise that a branch will always carry that file, which is the promise that failed here. A commit
SHA is immutable, works on GitHub and PyPI alike (both need absolute URLs, so a relative path is
not an option for the PyPI half), and matches this project's existing discipline of pinning CI
actions by hash and container images by digest rather than by tag.

WHAT THIS TEST DOES, AND DOES NOT

It checks the pin is a full SHA, that the commit exists in this repository, and that the file is
present in THAT commit's tree. It does not reach the network: a test that needs GitHub is slow on
every run, fails on a train, and gets marked skip. What it guarantees is the thing that actually
broke, which is a URL naming a place the file is not.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: `raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>`
RAW = re.compile(
    r"https://raw\.githubusercontent\.com/[^/]+/[^/]+/(?P<ref>[^/]+)/(?P<path>[^\"'\s)]+)")

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _refs():
    return list(RAW.finditer(README.read_text(encoding="utf-8")))


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=False)


def _require_checkout():
    """Skip rather than fail when this tree is not a git checkout.

    `git ls-files` outside a repository exits non-zero with EMPTY stdout, and a test that
    reads that as "the file is not tracked" is reporting on its environment rather than on
    the code. Found 2026-09-12 by running the suite from a `git archive` extract on atlas,
    which is the same configuration that caught four of these before and which nothing runs
    automatically. Matches the guard `test_shipped_files.py` already uses.
    """
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not a git checkout, so what is tracked cannot be asked here")


def test_the_readme_actually_references_some_images():
    """Zero matches would make every assertion below vacuously true."""
    assert _refs(), "no raw.githubusercontent URLs found; has the banner moved?"


@pytest.mark.parametrize("which", range(len(_refs())))
def test_every_raw_url_is_pinned_to_a_commit_not_a_branch(which):
    """A branch name here is a promise that the branch will always carry the file.

    That is the promise that failed: `main` is 386 commits behind `dev`, the assets were only ever
    added to `dev`, and both banner URLs 404ed on GitHub and on the PyPI project page.
    """
    m = _refs()[which]
    ref = m.group("ref")
    assert FULL_SHA.match(ref), (
        f"{m.group('path')} is served from {ref!r}, which is a branch or tag rather than a commit. "
        f"Pin it to a full 40-character SHA: a branch can stop carrying the file without anything "
        f"here changing, which is exactly how the banner came to be broken.")


@pytest.mark.parametrize("which", range(len(_refs())))
def test_every_pinned_commit_exists_and_carries_the_file(which):
    """The pin names a real commit, and the file is in THAT commit's tree.

    A well-formed SHA that no commit has, or a commit that does not contain the path, produces the
    identical 404 with none of the obviousness.
    """
    m = _refs()[which]
    ref, path = m.group("ref"), m.group("path")

    if _git("cat-file", "-e", f"{ref}^{{commit}}").returncode != 0:
        pytest.skip(f"commit {ref[:12]} is not in this clone (a shallow checkout, most likely), "
                    f"so the tree behind the pin cannot be read here")

    got = _git("cat-file", "-e", f"{ref}:{path}")
    assert got.returncode == 0, (
        f"README serves {path} from commit {ref[:12]}, and that commit's tree does not contain it. "
        f"The URL will 404 exactly as the branch-pinned one did.")


def test_the_pinned_files_are_still_tracked_on_this_branch():
    """A pin is not an excuse to delete the source.

    If the assets leave the working tree, the README keeps rendering from history and nobody
    notices they are gone until the pin needs moving and there is nothing to move it to.
    """
    _require_checkout()
    tracked = set(_git("ls-files", "assets/brand").stdout.split())
    for m in _refs():
        path = m.group("path")
        assert path in tracked, (
            f"{path} is served by the README and is no longer tracked on this branch")
