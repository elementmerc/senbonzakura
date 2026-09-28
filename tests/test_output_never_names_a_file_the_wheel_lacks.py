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
#:
#: WIDENED 2026-09-28. It used to be `docs/(guide|reference)/[a-z0-9-]+`, which reads only the two
#: subdirectories and misses a page sitting at the top of `docs/`. `trackbuild` named
#: `docs/evaluation-track-card.md` at an installed user in two places, one of them the refusal that
#: stops a build, and that page is the very one the 2026-09-10 attribution finding was about. So the
#: guard descended from that finding could not see the file that prompted it.
#:
#: The lookbehind is what keeps `bundled.doc_url`'s own output clean: in a link the segment is
#: preceded by a slash (`.../blob/main/docs/guide/x.md`), and in a repository path it is not.
A_DOCS_PATH = re.compile(r"(?<![/\w])docs/[a-z0-9][a-z0-9._/-]*")

#: Literal strings whose `docs/` mention is legitimate, each with the reason. Nothing in the package
#: needs one today, and the list exists so that the next legitimate case is written down here rather
#: than answered by loosening the pattern again. `tools/ci/docs_commands_run.py` is the shape of a
#: legitimate mention: it drives the commands printed in the guide and names pages by repository
#: path on purpose. It lives outside `PACKAGE`, so this scan never reads it.
ALLOWED_DOCS_MENTIONS: dict[str, str] = {}


def _names_a_docs_path(text):
    """Whether `text` sends a reader to a repository path under `docs/` they may not have."""
    if any(allowed in text for allowed in ALLOWED_DOCS_MENTIONS):
        return False
    return bool(A_DOCS_PATH.search(text))


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
                 if _names_a_docs_path(text)]
    listing = "\n".join(f"  {name}:{line}  {text[:100]}" for name, line, text in offenders)
    assert not offenders, (
        "these strings name a documentation path that is in the repository and in no wheel, so an "
        "installed reader cannot follow them:\n" + listing +
        "\n  Use bundled.doc_url('<page>') instead, and ALLOWED_DOCS_MENTIONS if the mention is "
        "genuinely right.")


def test_the_widened_pattern_catches_a_page_at_the_top_of_the_docs_tree():
    """Mutation test for the widening: the narrow pattern passed on both of these.

    `trackbuild` printed the first of them at the end of every corpus build and raised the second
    when an upstream licence changed, and the old `docs/(guide|reference)/...` pattern read neither.
    """
    assert _names_a_docs_path("anybody, including us. See docs/evaluation-track-card.md.")
    assert _names_a_docs_path("Update SOURCES and docs/evaluation-track-card.md together")


def test_the_widened_pattern_leaves_a_real_link_alone():
    """A link is the fix, so flagging one would make the guard unsatisfiable."""
    assert not _names_a_docs_path(
        "See " + bundled.doc_url("evaluation-track-card") + " for what the track may be used for.")
    assert not _names_a_docs_path("See " + bundled.doc_url("guide/what-we-know") + ".")


def test_the_allowlist_can_excuse_a_string_and_only_that_string():
    """The escape hatch has to work, or the next legitimate mention loosens the pattern instead."""
    allowed = "docs/evaluation-track-card.md"
    ALLOWED_DOCS_MENTIONS[allowed] = "test only"
    try:
        assert not _names_a_docs_path(f"See {allowed}.")
        assert _names_a_docs_path("See docs/guide/limits.md.")
    finally:
        del ALLOWED_DOCS_MENTIONS[allowed]


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


# ── a notice pointer has to resolve to the NOTICE, not merely to a file that ships ────────────────
#
# WHAT PROMPTED IT, 2026-09-28
#
# Every capability run printed "Attribution travels with it. See THIRD-PARTY-CORPORA.md." GSM8K is
# not in that file and cannot be: `corporabuild` regenerates it wholesale from the six-entry corpus
# table, so nothing a person adds by hand survives. The GSM8K attribution is in
# THIRD-PARTY-NOTICES.md, under "The bundled capability probe", and `bundled` gets the same
# distinction right for the evaluation track.
#
# Every guard in this file passed on it. The string named a file that ships, is spelled correctly
# and is in `license-files`, and the reader following it still found no attribution. So the check is
# the whole claim: the file ships AND it contains something identifying the thing being attributed.

#: Module in the package → the notice file it points a reader at, and a token that has to be in that
#: file for the pointer to have led anywhere. Lower-cased on both sides before comparing.
NOTICE_POINTERS = {
    "bundled.py": ("THIRD-PARTY-NOTICES.md", ("bundled evaluation track",)),
    "capability.py": ("THIRD-PARTY-NOTICES.md", ("gsm8k",)),
    "corpora.py": ("THIRD-PARTY-CORPORA.md", ("advbench", "harmbench")),
    "corporabuild.py": ("THIRD-PARTY-CORPORA.md", ("advbench", "harmbench")),
}

#: A third-party notice file as a user-facing string would spell it.
A_NOTICE_FILE = re.compile(r"\bTHIRD-PARTY-[A-Z]+\.md\b")

ROOT = PACKAGE.parent.parent


def _modules_that_name_a_notice_file():
    """(module name, set of notice files it names) for every module in the package that names one."""
    named = {}
    for path, _line, text in _user_facing_strings():
        for hit in A_NOTICE_FILE.findall(text):
            named.setdefault(path.name, set()).add(hit)
    return named


def test_every_module_naming_a_notice_file_is_declared_here():
    """A new pointer has to say what it is pointing at, or this guard grows a blind spot silently."""
    named = _modules_that_name_a_notice_file()
    assert named, (
        "no module names a third-party notice file any more. If the pointers were removed "
        "deliberately, remove this guard with them; otherwise the attribution has gone.")
    undeclared = sorted(set(named) - set(NOTICE_POINTERS))
    assert not undeclared, (
        f"these modules point a reader at a notice file and are not in NOTICE_POINTERS: "
        f"{undeclared}. Add the file it names and a token that proves the notice is in it.")


def test_each_module_names_the_notice_file_that_holds_its_attribution():
    wrong = {}
    for module, files in _modules_that_name_a_notice_file().items():
        expected = NOTICE_POINTERS[module][0]
        if files != {expected}:
            wrong[module] = sorted(files)
    assert not wrong, (
        f"these modules name a notice file that is not the one holding their attribution: {wrong}. "
        f"Expected, per module: "
        f"{ {m: f for m, (f, _) in NOTICE_POINTERS.items()} }")


def test_every_named_notice_file_ships():
    """`license-files` in pyproject is what puts these in the wheel, so that is what is read."""
    from tomlread import tomllib

    with (ROOT / "pyproject.toml").open("rb") as f:
        shipped = set(tomllib.load(f)["project"]["license-files"])
    for module, (notice, _tokens) in NOTICE_POINTERS.items():
        assert (ROOT / notice).is_file(), f"{module} names {notice}, which is not in the tree"
        assert notice in shipped, (
            f"{module} names {notice}, which is in the repository and not in `license-files`, so an "
            f"installed reader has no copy of it")


def test_every_notice_pointer_resolves_to_the_notice_and_not_just_to_the_file():
    """The finding itself: a pointer to the wrong notice file resolves and still leads nowhere."""
    missing = {}
    for module, (notice, tokens) in NOTICE_POINTERS.items():
        text = (ROOT / notice).read_text(encoding="utf-8").lower()
        absent = [t for t in tokens if t.lower() not in text]
        if absent:
            missing[module] = (notice, absent)
    assert not missing, (
        f"these pointers name a file that ships and does not carry the attribution they promise: "
        f"{missing}. A reader who follows one finds a notice about something else.")


def test_the_pointer_check_would_fail_on_the_string_it_was_written_for():
    """Mutation test: swapping capability's notice file for the wrong one has to be caught.

    Both files ship and both are spelled correctly, which is why nothing else here sees it.
    """
    corpora = (ROOT / "THIRD-PARTY-CORPORA.md").read_text(encoding="utf-8").lower()
    assert "gsm8k" not in corpora, (
        "THIRD-PARTY-CORPORA.md now mentions gsm8k, so the mutation this test relies on no longer "
        "fails. It is generated from the six-entry corpus table, so check what put it there.")
