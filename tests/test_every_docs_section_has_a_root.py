# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A documentation section that is built has a page at its root, or its root is a 404.

WHAT PROMPTED IT, 2026-09-27

A reader found `/guide/` returned 404, and that `/guide` 301s to `/guide/`, so the extensionless
spelling of the directory landed on the error page. `/reference/` had the identical gap. Both were
fixed by adding a contents page, and the second one was found only because somebody went looking for
the other spelling after the first.

So this asks the question of every section rather than of the two that were reported, which is the
difference between fixing an instance and closing a class. A new section cannot ship without a root.

WHAT THIS DOES NOT CLAIM TO FIX

A trailing slash after a PAGE name, `/guide/what-we-know/`, still 404s and cannot be made to work on
GitHub Pages: there is no server-side rewrite, and VitePress emits one file per page, so turning
`cleanUrls` off would only move the 404 onto the extensionless spelling that every internal link and
every search result uses. That limitation is recorded rather than worked around, and it is a
different problem from this one.

WHY `srcExclude` IS READ RATHER THAN A LIST HERE

`writeups/` holds articles with their own publication route and is excluded from the build, so it has
no root and needs none. Hardcoding that exemption would go stale the moment the exclusion list
changes, and the failure would be this test demanding an index for a directory nobody publishes.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CONFIG = DOCS / ".vitepress" / "config.mjs"

#: Directories that are machinery rather than content, whatever they hold.
_NOT_CONTENT = {".vitepress", "public", "node_modules", "dist"}


def _excluded_globs():
    """The `srcExclude` patterns, read out of the config rather than copied into this file."""
    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(r"srcExclude:\s*\[([^\]]*)\]", text, re.DOTALL)
    assert match, "srcExclude is gone from the VitePress config; this test reads it to find what is built"
    return re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))


def _built_sections():
    """Every docs subdirectory that contributes at least one page to the site."""
    excluded = _excluded_globs()
    sections = []
    for child in sorted(p for p in DOCS.iterdir() if p.is_dir()):
        if child.name in _NOT_CONTENT:
            continue
        if any(child.match(glob.rstrip("/*")) or child.name == glob.split("/")[0]
               for glob in excluded):
            continue
        pages = [p for p in child.glob("*.md") if p.name != "index.md"]
        if pages:
            sections.append(child)
    return sections


def test_there_are_sections_to_check():
    """An empty list would pass every assertion below while measuring nothing."""
    sections = _built_sections()
    assert sections, (
        "no built documentation sections were found. Either docs/ moved or the srcExclude read "
        "above is now matching everything.")


@pytest.mark.parametrize("section", _built_sections(), ids=lambda p: p.name)
def test_the_section_root_is_a_page(section):
    index = section / "index.md"
    assert index.is_file(), (
        f"docs/{section.name}/ has pages and no index.md, so /{section.name}/ serves a 404 and "
        f"/{section.name} redirects into it. Add a contents page, or add the directory to "
        f"srcExclude if it is not meant to be published.")


@pytest.mark.parametrize("section", _built_sections(), ids=lambda p: p.name)
def test_the_root_links_somewhere_real(section):
    """A contents page full of dead links fails the docs build, which breaks the deploy.

    `ignoreDeadLinks` is deliberately unset, so this would be caught eventually. Caught here it
    names the file and the link instead of failing a node build in CI.
    """
    index = section / "index.md"
    if not index.is_file():
        pytest.skip("covered by the test above")
    text = index.read_text(encoding="utf-8")
    links = re.findall(r"\]\((/[^)#?]+)", text)
    assert links, f"docs/{section.name}/index.md links to nothing, so it is not a contents page"

    missing = []
    for link in links:
        target = link.rstrip("/")
        candidates = [
            DOCS / f"{target.lstrip('/')}.md",
            DOCS / target.lstrip("/") / "index.md",
        ]
        if not any(c.is_file() for c in candidates):
            missing.append(link)
    assert not missing, (
        f"docs/{section.name}/index.md links to pages that do not exist: {missing}. "
        f"The docs build fails on a dead link, so this would break the deploy.")
