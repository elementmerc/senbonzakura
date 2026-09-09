#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Every documentation link in the README must resolve to a page the site actually builds.

WHY THIS EXISTS

The docs site is built by CI with `ignoreDeadLinks` deliberately unset, so a link from one page
of the site to another cannot break without failing the build. Nothing checked the links pointing
INTO the site from outside it, and the README is full of them: eleven, including the word
"Documentation" under the banner and four badges at the top.

That gap was not theoretical. The site was never deployed at all, GitHub Pages had never been
enabled, and every one of those links returned 404 for the life of the project, on the public
repository and on the PyPI project page, which renders the same README. The docs job was green
throughout, because building a site and publishing it are different things and only the first was
ever checked.

So this asks the question the build cannot: for each URL in the README that points at the docs
site, is there a file in the built output that serves it? A link into a page that no longer exists
after a rename is the same failure at a smaller scale, and it is the one that will actually recur.

Run it against a built site:

    npm --prefix docs ci && npm --prefix docs run build
    python tools/readme_links_resolve.py docs/.vitepress/dist
"""
import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

SITE = "https://elementmerc.github.io/senbonzakura"
# Anything that is not a URL character ends the match. The README carries these inside HTML
# `href="..."` attributes and inside Markdown `[text](...)`, so the closing delimiter varies.
_URL = re.compile(re.escape(SITE) + r"[^\s\"'<>)\]]*")

# Public prose that links into the docs site. The README is the one that reaches PyPI.
SOURCES = ("README.md",)


def links_in(text):
    """Every docs-site URL in `text`, deduplicated, in the order they first appear."""
    seen = {}
    for raw in _URL.findall(text):
        # A trailing full stop or comma belongs to the sentence, not the URL.
        url = raw.rstrip(".,;:")
        seen.setdefault(url, None)
    return list(seen)


def served_by(url, dist):
    """The file that would serve `url`, or None if the built site has nothing for it.

    `cleanUrls` is on, so `/guide/compass` is served by `guide/compass.html`. A directory with an
    `index.html` serves the same shape, and the site root serves `index.html`.
    """
    path = urlsplit(url).path
    rel = path[len(urlsplit(SITE).path):].strip("/")
    candidates = [f"{rel}.html", f"{rel}/index.html", rel] if rel else ["index.html"]
    for candidate in candidates:
        f = dist / candidate
        if f.is_file():
            return f
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dist", type=Path, help="the built site, normally docs/.vitepress/dist")
    ap.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    a = ap.parse_args(argv)

    if not a.dist.is_dir():
        print(f"no built site at {a.dist}: run `npm --prefix docs run build` first.",
              file=sys.stderr)
        return 2

    broken, checked = [], 0
    for name in SOURCES:
        source = a.root / name
        if not source.is_file():
            print(f"{name} is missing, and it is the file this check exists to read.",
                  file=sys.stderr)
            return 2
        for url in links_in(source.read_text(encoding="utf-8")):
            checked += 1
            if served_by(url, a.dist) is None:
                broken.append((name, url))

    if not checked:
        # Zero links found is not a pass. Either the README stopped linking to the docs, which is
        # a change worth noticing, or the pattern stopped matching and this check went quiet.
        print(f"FAILED: found no links to {SITE} in {', '.join(SOURCES)}. A check that examines "
              f"nothing reports success, which is how the docs came to be unpublished for the "
              f"life of the project. If the links were deliberately removed, update SOURCES.",
              file=sys.stderr)
        return 1

    if broken:
        print(f"FAILED: {len(broken)} of {checked} documentation links point at a page the site "
              f"does not build. A reader following one gets a 404, and on PyPI that reader has "
              f"not installed anything yet.", file=sys.stderr)
        for name, url in broken:
            print(f"  {name}: {url}", file=sys.stderr)
        return 1

    print(f"all {checked} documentation links resolve to a built page: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
