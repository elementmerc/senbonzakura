# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The README's links into the documentation site.

The site's own build fails on a link from one page to another that does not resolve. Nothing
checked the links pointing INTO the site, and those are the ones a reader arrives by: the site was
never deployed at all, so all of them returned 404 on the public repository and on the PyPI project
page while the docs job stayed green.

What is tested here is the checker itself, against a built site made of empty files, because the
failure worth catching is a checker that reports success without examining anything.
"""
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "readme_links_resolve",
    Path(__file__).resolve().parent.parent / "tools" / "readme_links_resolve.py")
links = importlib.util.module_from_spec(_spec)
sys.modules["readme_links_resolve"] = links
_spec.loader.exec_module(links)

SITE = links.SITE


def build_site(root, pages):
    dist = root / "dist"
    for page in pages:
        f = dist / page
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("", encoding="utf-8")
    dist.mkdir(exist_ok=True)
    return dist


def write_readme(root, body):
    (root / "README.md").write_text(body, encoding="utf-8")


def test_it_finds_links_in_both_markdown_and_html_forms():
    # The README carries these in HTML `href` attributes in the banner and in Markdown links in
    # the prose, and the closing delimiter differs between them.
    body = (f'<a href="{SITE}/guide/compass">c</a>\n'
            f'[the track]({SITE}/guide/the-track)\n'
            f'plain {SITE}/guide/limits, mid-sentence.\n')
    found = links.links_in(body)
    assert found == [f"{SITE}/guide/compass", f"{SITE}/guide/the-track", f"{SITE}/guide/limits"]


def test_a_repeated_link_is_reported_once():
    body = f"{SITE}/guide/compass and again {SITE}/guide/compass\n"
    assert links.links_in(body) == [f"{SITE}/guide/compass"]


def test_clean_urls_resolve_to_the_html_file(tmp_path):
    dist = build_site(tmp_path, ["guide/compass.html"])
    assert links.served_by(f"{SITE}/guide/compass", dist) == dist / "guide/compass.html"


def test_the_site_root_resolves_to_the_index(tmp_path):
    dist = build_site(tmp_path, ["index.html"])
    assert links.served_by(SITE, dist) == dist / "index.html"
    assert links.served_by(f"{SITE}/", dist) == dist / "index.html"


def test_a_fragment_does_not_change_which_page_serves_it(tmp_path):
    dist = build_site(tmp_path, ["guide/compass.html"])
    assert links.served_by(f"{SITE}/guide/compass#the-method", dist) is not None


def test_a_page_the_site_does_not_build_is_not_served(tmp_path):
    dist = build_site(tmp_path, ["guide/compass.html"])
    assert links.served_by(f"{SITE}/guide/renamed", dist) is None


def test_a_readme_whose_links_all_resolve_passes(tmp_path, capsys):
    dist = build_site(tmp_path, ["index.html", "guide/compass.html"])
    write_readme(tmp_path, f"[docs]({SITE}) and [compass]({SITE}/guide/compass)\n")
    assert links.main([str(dist), "--root", str(tmp_path)]) == 0
    assert "all 2 documentation links resolve" in capsys.readouterr().out


def test_one_broken_link_fails_and_names_it(tmp_path, capsys):
    dist = build_site(tmp_path, ["index.html"])
    write_readme(tmp_path, f"[docs]({SITE}) and [gone]({SITE}/guide/renamed)\n")
    assert links.main([str(dist), "--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "guide/renamed" in err, "a reader has to be told which link to fix"
    assert "1 of 2" in err


def test_a_readme_with_no_docs_links_is_a_failure_not_a_pass(tmp_path, capsys):
    # A check that examines nothing reports success, which is the shape that let the docs go
    # unpublished for the life of the project. Zero links found has to be loud.
    dist = build_site(tmp_path, ["index.html"])
    write_readme(tmp_path, "no links here at all\n")
    assert links.main([str(dist), "--root", str(tmp_path)]) == 1
    assert "found no links" in capsys.readouterr().err


def test_a_missing_built_site_is_distinguished_from_a_broken_link(tmp_path, capsys):
    write_readme(tmp_path, f"[docs]({SITE})\n")
    assert links.main([str(tmp_path / "nope"), "--root", str(tmp_path)]) == 2
    assert "no built site" in capsys.readouterr().err


def test_a_missing_readme_is_refused_rather_than_counted_as_clean(tmp_path, capsys):
    dist = build_site(tmp_path, ["index.html"])
    assert links.main([str(dist), "--root", str(tmp_path)]) == 2
    assert "README.md is missing" in capsys.readouterr().err


def test_the_real_readme_only_links_to_pages_this_repository_holds():
    # Not a check that the site is built (that is the CI step's job), but that every docs link in
    # the shipped README names a page whose source file exists. A rename that updates the site and
    # forgets the README is caught here, on the machine that made it.
    root = Path(__file__).resolve().parent.parent
    docs = root / "docs"
    missing = []
    for url in links.links_in((root / "README.md").read_text(encoding="utf-8")):
        rel = url[len(SITE):].split("#")[0].strip("/")
        candidates = [docs / f"{rel}.md", docs / rel / "index.md"] if rel else [docs / "index.md"]
        if not any(c.is_file() for c in candidates):
            missing.append(url)
    assert not missing, f"the README links to pages with no source: {missing}"
