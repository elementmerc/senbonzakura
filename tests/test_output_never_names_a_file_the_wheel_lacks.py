# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A pointer an installed user cannot follow is worse than no pointer at all.

WHAT PROMPTED IT, 2026-09-27

`senbonzakura measure` closed its summary with "see docs/guide/what-we-know before quoting any of
it". `docs/` is in the repository and in no wheel, so every installed user was sent to a file they
do not have, at the exact moment they were being told not to quote a number without reading it. The
caveat that protects the reader was the line that could not be followed.

This is the same shape as two defects this project has already fixed: `doctor` telling an installed
user to run `tools/packaging/vendor_llama.py`, which exists only in a checkout, and the licence
notice pointing "inside this package" where the licences are not. Each time, output written while
looking at a checkout named something only a checkout has.

THE RULE

Prose that reaches a user names a URL, not a repository path, for anything under `docs/`. The
check reads string literals in the package rather than driving every command, because the defect is
in what the string says and there are sixty-odd surfaces; the URL helper it should be using is
`bundled.doc_url`, which is asserted here to produce a link that resolves in shape.
"""
from __future__ import annotations

import ast
import pathlib
import re

from senbonzakura import bundled

PACKAGE = pathlib.Path(bundled.__file__).resolve().parent

#: A repository path under `docs/`, as it would appear inside a string a user reads. Deliberately
#: not a bare "docs" match: `docs_commands_run.py` and a docstring mentioning the directory are not
#: defects, and a check that flags them would be routed around within a week.
A_DOCS_PATH = re.compile(r"docs/(guide|reference)/[a-z0-9-]+")


def _user_facing_strings():
    """(file, line, text) for every string literal in the package that is not a docstring.

    Docstrings are excluded because they are for whoever reads the source, who has the repository
    open by definition. Comments never reach a user and are not in the AST as strings anyway.
    """
    for path in sorted(PACKAGE.rglob("*.py")):
        if "vendor" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                first = node.body[0] if node.body else None
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    docstrings.add(id(first.value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings):
                yield path, node.lineno, node.value


def test_the_scan_reads_the_package_it_is_meant_to():
    """Without this the sweep below passes on an empty list, which is how a guard reports clean."""
    found = list(_user_facing_strings())
    assert len(found) > 500, f"only {len(found)} string literals found in {PACKAGE}"


def test_no_user_facing_string_names_a_documentation_path():
    offenders = [(p.name, line, text) for p, line, text in _user_facing_strings()
                 if A_DOCS_PATH.search(text)]
    listing = "\n".join(f"  {name}:{line}  {text[:100]}" for name, line, text in offenders)
    assert not offenders, (
        "these strings name a documentation path that is in the repository and in no wheel, so an "
        "installed reader cannot follow them:\n" + listing +
        "\n  Use bundled.doc_url('guide/<page>') instead.")


def test_the_helper_builds_a_link_and_not_a_path():
    url = bundled.doc_url("guide/what-we-know")
    assert url.startswith("https://"), url
    assert url.endswith("/docs/guide/what-we-know.md"), url
    assert not A_DOCS_PATH.search(url.split("://", 1)[0]), "the scheme should not look like a path"


def test_the_helper_pins_the_released_branch_rather_than_the_one_in_flight():
    """A reader following a link out of a released build lands on the released prose.

    `dev` moves under them: a page can say something different an hour later, and the number they
    were told to check it against is the one in their hand.
    """
    assert "/blob/main/" in bundled.doc_url("guide/what-we-know")


def test_a_page_given_with_slashes_or_without_produces_the_same_link():
    """Callers should not be able to half-write the URL, which is why they pass a name."""
    assert bundled.doc_url("guide/limits") == bundled.doc_url("/guide/limits/")


def test_the_sweep_would_fail_on_the_string_it_was_written_for():
    """Mutation test: the regex has to match the literal that prompted this.

    The sweep is only worth reading if it can fail, and the case it must catch is the exact
    sentence `measure` used to print.
    """
    assert A_DOCS_PATH.search(
        "None of this is a pass or a fail. See docs/guide/what-we-know before quoting any of it.")
    assert not A_DOCS_PATH.search(bundled.doc_url("guide/what-we-know").replace("docs/", "d/"))


#: Commands this project used to recommend and no longer should. Each entry is a thing a reader would
#: type and get an error or a deprecation from, which is worse than no advice: they followed us.
SUPERSEDED_COMMANDS = {
    # `hf auth login` replaced it. `hubmessage.ADVICE_FOR_THE_PYTHON_API` strips this very line out
    # of upstream's messages for being stale, and on 2026-09-27 the guided mode was still handing it
    # to a reader as our own advice, so the tool treated the same sentence as wrong from somebody
    # else and right from itself.
    "huggingface-cli login",
}


def test_no_user_facing_string_recommends_a_superseded_command():
    """Found by reading today's diff for hyphens and noticing two spellings of one instruction.

    The scan skips docstrings for the reason given above, and `hubmessage` is skipped entirely
    because its whole job is to hold the stale spellings so it can RECOGNISE them in somebody else's
    output. A list of what to filter is not a recommendation.
    """
    offenders = [(p.name, line, text) for p, line, text in _user_facing_strings()
                 if p.name != "hubmessage.py"
                 and any(bad in text for bad in SUPERSEDED_COMMANDS)]
    listing = "\n".join(f"  {name}:{line}  {text[:110]}" for name, line, text in offenders)
    assert not offenders, (
        "these strings tell a reader to run a command this project no longer recommends:\n"
        + listing + "\n  See SUPERSEDED_COMMANDS for what replaced it.")


def test_the_scan_would_catch_it_if_it_came_back():
    """Mutation test: the list is only worth reading if a match would be found."""
    assert any(bad in "needs a Hugging Face account and `huggingface-cli login` before it fetches"
               for bad in SUPERSEDED_COMMANDS)


def test_the_filter_that_recognises_the_stale_advice_still_holds_it():
    """The exemption above is load bearing: `hubmessage` must keep the string it strips.

    If this ever stops being true, the exemption is hiding nothing and should go, and upstream's
    stale advice is reaching readers again.
    """
    from senbonzakura import hubmessage

    assert any("huggingface-cli login" in phrase
               for phrase in hubmessage.ADVICE_FOR_THE_PYTHON_API), (
        "hubmessage no longer filters the superseded login command out of upstream messages")
