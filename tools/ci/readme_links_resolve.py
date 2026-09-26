#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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
    python tools/ci/readme_links_resolve.py docs/.vitepress/dist

THE SECOND HALF: LINKS INTO THE REPOSITORY

The same failure, one directory over. The whole reproducibility argument this project makes is
that a claim traces to a committed artefact, and the public route to an artefact is a
`github.com/elementmerc/senbonzakura/blob/<ref>/<path>` link. Those links named `main`, and on
2026-09-25 the published `main` was 611 commits behind `dev` and carried none of the artefacts
that matter: the head-to-head results, `CONTRACT.md`, the k-sweep drift file, `CONTRIBUTING.md`,
`ACCEPTABLE-USE.md`, `METHOD.md` and `REPRODUCING.md` itself. Every one of those links returned
404 for a reader on the docs site, on the issue-template menu and in the Colab notebook.

Promoting `dev` would fix today's list and not the class, because the links have to stay correct
against a ref that keeps moving. So this asks git the same question a reader's browser asks: does
`<ref>` contain `<path>`?

**Against the PUBLISHED ref, not the local one.** A local `main` that has been fast-forwarded but
not pushed answers yes to everything while the public branch answers no, which is exactly the
state that hid this. When a remote-tracking `origin/<ref>` exists, it is what gets asked.
"""
import argparse
import re
import subprocess
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


REPO = "https://github.com/elementmerc/senbonzakura"
# `blob` for a file, `tree` for a directory, `raw` for the bytes. All three take the same
# `<ref>/<path>` tail, and all three 404 identically when the ref does not carry the path.
_REPO_URL = re.compile(re.escape(REPO) + r"/(blob|tree|raw)/([^/\s\"'<>)\]]+)/([^\s\"'<>)\]]*)")

#: What counts as public prose for the repository-link half. The notebook is here because a Colab
#: reader meets those links before they have cloned anything, and `.github/` is here because the
#: issue-template menu is the first page somebody with a problem sees.
_PROSE_SUFFIXES = (".md", ".ipynb", ".yml", ".yaml")
_SKIP_PREFIXES = ("private/", "docs/node_modules/", "node_modules/")


def _git(root, *args, timeout=120):
    """Run git under a deadline and hand back (returncode, stdout). Never waits forever."""
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return 124, ""
    return done.returncode, done.stdout


def prose_sources(root):
    """Every tracked file that a reader could meet a repository link in."""
    code, out = _git(root, "ls-files")
    if code != 0:
        return []
    return [p for p in out.split()
            if p.endswith(_PROSE_SUFFIXES) and not p.startswith(_SKIP_PREFIXES)]


def repo_links_in(text):
    """Every `<REPO>/{blob,tree,raw}/<ref>/<path>` in `text`, as (ref, path) pairs, deduplicated."""
    seen = {}
    for _kind, ref, path in _REPO_URL.findall(text):
        # A fragment and a query are the browser's business, and a trailing full stop or comma
        # belongs to the sentence rather than to the path.
        clean = path.split("#", 1)[0].split("?", 1)[0].rstrip(".,;:").strip("/")
        if clean:
            seen.setdefault((ref, clean), None)
    return list(seen)


def published_ref(root, ref, allow_fetch=True):
    """The ref a reader's browser would see, or None if this clone cannot answer for it.

    A local branch is not the published one. `main` was 611 commits AHEAD of `origin/main` on the
    machine where this was written, so asking the local branch would have reported every broken
    link as fine. `origin/<ref>` is asked first, and only a ref with no remote-tracking form at
    all (a tag, a commit sha) falls back to the name as written.
    """
    for candidate in (f"origin/{ref}", ref):
        if _git(root, "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}")[0] == 0:
            return candidate
    if not allow_fetch:
        return None
    # A shallow checkout has no remote-tracking branches, so the honest move is to go and get the
    # one ref we need rather than to report a pass we cannot support. Bounded, and one attempt.
    if _git(root, "fetch", "--depth=1", "origin", f"+{ref}:refs/remotes/origin/{ref}",
            timeout=300)[0] == 0:
        return f"origin/{ref}"
    return None


def check_repo_links(root, allow_fetch=True):
    """(checked, broken, unresolvable). `broken` is (source, ref, path); the last is refs."""
    broken, unresolvable, checked = [], {}, 0
    resolved = {}
    for name in prose_sources(root):
        source = root / name
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"{name} could not be read, so its links were not checked: {e}", file=sys.stderr)
            return checked, [(name, "?", "?")], unresolvable
        for ref, path in repo_links_in(text):
            checked += 1
            if ref not in resolved:
                resolved[ref] = published_ref(root, ref, allow_fetch=allow_fetch)
            target = resolved[ref]
            if target is None:
                unresolvable.setdefault(ref, set()).add(name)
                continue
            if _git(root, "cat-file", "-e", f"{target}:{path}")[0] != 0:
                broken.append((name, ref, path))
    return checked, broken, unresolvable


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dist", type=Path, help="the built site, normally docs/.vitepress/dist")
    ap.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    ap.add_argument("--no-fetch", action="store_true",
                    help="never reach the network; a ref this clone cannot resolve is then a "
                         "failure rather than something to go and fetch")
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

    status = 0
    if broken:
        print(f"FAILED: {len(broken)} of {checked} documentation links point at a page the site "
              f"does not build. A reader following one gets a 404, and on PyPI that reader has "
              f"not installed anything yet.", file=sys.stderr)
        for name, url in broken:
            print(f"  {name}: {url}", file=sys.stderr)
        status = 1
    else:
        print(f"all {checked} documentation links resolve to a built page: OK")

    # ── and the links that point back into the repository ──────────────────────────
    #
    # Only where git can answer. `--root` is a bare directory in this file's own unit tests, and
    # asking a non-repository whether a ref carries a path has no meaningful answer. Every place
    # this runs for real is a checkout, so the skip is announced rather than quiet, and a root
    # that IS a repository still fails when it carries no repository links at all.
    if _git(a.root, "rev-parse", "--git-dir")[0] != 0:
        print(f"{a.root} is not a git checkout, so the repository links were NOT checked. In CI "
              f"this half must run: it is what catches a link to an artefact the published ref "
              f"does not carry.", file=sys.stderr)
        return status

    repo_checked, repo_broken, unresolvable = check_repo_links(a.root, allow_fetch=not a.no_fetch)

    if not repo_checked:
        print(f"FAILED: found no {REPO}/blob|tree|raw links in any tracked prose. Either every "
              f"artefact link was removed, which is worth noticing, or the pattern stopped "
              f"matching and this half went quiet.", file=sys.stderr)
        return 1

    if unresolvable:
        print("FAILED: these refs could not be resolved, so their links were not checked. A "
              "shallow checkout has no remote-tracking branches; give the job `fetch-depth: 0` "
              "or drop --no-fetch.", file=sys.stderr)
        for ref, names in sorted(unresolvable.items()):
            print(f"  {ref}: named in {', '.join(sorted(names))}", file=sys.stderr)
        status = 1

    if repo_broken:
        print(f"FAILED: {len(repo_broken)} of {repo_checked} repository links point at a path the "
              f"published ref does not carry, so a reader following one gets a 404. This project's "
              f"whole argument is that a claim traces to a committed artefact, and these are the "
              f"route to the artefacts.", file=sys.stderr)
        for name, ref, path in repo_broken:
            print(f"  {name}: {ref} has no {path}", file=sys.stderr)
        status = 1
    elif not unresolvable:
        print(f"all {repo_checked} repository links resolve on the ref they name: OK")

    return status


if __name__ == "__main__":
    sys.exit(main())
