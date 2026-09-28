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


# ── container fences, which the site build cannot check and never will ────────────────────────────
#
# WHAT PROMPTED IT, 2026-09-28
#
# Two pages shipped with a broken VitePress container. `docs/guide/quickstart.md` carried a closing
# `:::` that closed nothing, and `docs/guide/install.md` nested one warning box inside another at the
# same fence length, so the inner box's close ended the outer one and the outer's close had nothing
# left to end. Both printed a literal `:::` on the published page.
#
# The build is green either way: markdown-it treats an unmatched marker as ordinary text, which is
# exactly why a rendering defect of this class reaches a reader. So the check is arithmetic over the
# source, not the build.
#
# THE NESTING RULE, WHICH IS THE PART THAT IS NOT OBVIOUS
#
# A container opened with N colons is closed by the next line of N or more colons. An inner container
# at the same length as the one around it is therefore closed by a marker that also closes the outer,
# which is the install-guide defect. Nesting works only when the outer fence is LONGER than the inner:
# `::::` around `:::`.

_FENCE = re.compile(r"^(:{3,})\s*(.*)$")


def _content_pages():
    """Every Markdown page under `docs/` that is content rather than machinery."""
    pages = []
    for path in sorted(DOCS.rglob("*.md")):
        if any(part in _NOT_CONTENT for part in path.relative_to(DOCS).parts):
            continue
        pages.append(path)
    return pages


def _container_faults(text):
    """Every way a page's container fences fail to make a balanced, closable set.

    Fences inside a fenced code block are prose about containers rather than containers, so the
    scanner steps over them; the install guide shows the syntax in one.
    """
    faults = []
    stack = []
    in_code = False
    for number, line in enumerate(text.split("\n"), start=1):
        if line.startswith(("```", "~~~")):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = _FENCE.match(line)
        if not match:
            continue
        marker, label = match.group(1), match.group(2).strip()
        if label:
            if stack and len(marker) >= len(stack[-1][1]):
                faults.append(
                    f"line {number}: a container opened with {len(marker)} colons inside one opened "
                    f"with {len(stack[-1][1])} at line {stack[-1][0]}. The inner one's close would "
                    f"end the outer one too, so make the outer fence longer.")
            stack.append((number, marker))
        elif not stack:
            faults.append(f"line {number}: a closing `{marker}` with no container open.")
        else:
            opened_at, opener = stack.pop()
            if len(marker) < len(opener):
                faults.append(
                    f"line {number}: a closing `{marker}` is shorter than the `{opener}` opened at "
                    f"line {opened_at}, so it does not close it.")
    for opened_at, opener in stack:
        faults.append(f"line {opened_at}: `{opener}` is opened and never closed.")
    return faults


def test_there_are_pages_to_scan():
    """An empty list would pass the scan below while reading nothing."""
    pages = _content_pages()
    assert len(pages) > 10, f"only {len(pages)} content pages found under {DOCS}"


def test_every_page_has_balanced_and_closable_containers():
    broken = {}
    for page in _content_pages():
        faults = _container_faults(page.read_text(encoding="utf-8"))
        if faults:
            broken[page.relative_to(DOCS).as_posix()] = faults
    listing = "\n".join(f"  {name}\n    " + "\n    ".join(faults)
                        for name, faults in sorted(broken.items()))
    assert not broken, (
        "these pages would print a literal `:::` where a container box should be:\n" + listing)


def test_the_scan_catches_both_shapes_it_was_written_for():
    """Mutation test: the two defects it was written for, and one legitimate nesting.

    Without this the scanner could be silently permissive and read as clean.
    """
    stray = _container_faults("::: warning A\nbody\n:::\n\nmore prose\n:::\n")
    assert any("no container open" in f for f in stray), stray

    same_length = _container_faults("::: warning Outer\n::: tip Inner\nbody\n:::\n\ntail\n:::\n")
    assert any("inside one opened with" in f for f in same_length), same_length

    assert not _container_faults(":::: warning Outer\n::: tip Inner\nbody\n:::\n\ntail\n::::\n")
    assert not _container_faults("```\n::: warning shown as syntax\n```\n")
