# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`/guide/` on the published site was a 404.

FOUND BY A READER WITH NO KNOWLEDGE OF THE PROJECT, 2026-09-27. Confirmed against the live site:
`guide/what-we-know` returned 200, `guide` returned 301 to `guide/`, and `guide/` returned 404. The
guide had no root, so the obvious guess at its URL landed on an error page.

The trailing slash on a PAGE (`guide/what-we-know/`) cannot be fixed: GitHub Pages has no
server-side rewrite and VitePress emits one spelling per page. The directory ROOT can, with an
index page, and this checks the index is there and that everything it points at exists. A contents
page full of dead links would be worse than the 404 it replaced.
"""
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
INDEX = DOCS / "guide" / "index.md"

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="no docs tree in this wheel")


def test_the_guide_has_a_root_page():
    assert INDEX.is_file(), (
        "docs/guide/index.md is missing, so https://<site>/guide/ is a 404 again. VitePress serves "
        "a directory root from its index.md and from nothing else.")


@pytest.mark.skipif(not INDEX.is_file(), reason="no guide index to check")
def test_every_link_on_the_guide_root_resolves():
    body = INDEX.read_text(encoding="utf-8")
    # Site-absolute links only. These are the ones VitePress resolves against docs/, and the ones
    # this page is made of; external http links are somebody else's uptime.
    dead = []
    for target in re.findall(r"\]\((/[^)#\s]+)", body):
        rel = target.lstrip("/")
        if not ((DOCS / f"{rel}.md").is_file() or (DOCS / rel / "index.md").is_file()):
            dead.append(target)
    assert not dead, (
        f"the guide contents page links to {dead}, which have no page behind them. The docs build "
        f"fails on a dead link, so this would break the deploy rather than just the reader.")
