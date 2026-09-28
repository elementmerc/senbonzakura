# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A flag the user did not type must never appear in the error that answers them.

WHAT PROMPTED IT, 2026-09-27

A surface audit ran `senbonzakura convert <model> --out out.gguf`. Nine commands in this tool spell
their output path `--out`, so that is the form a reader arrives with. On `convert` the output is the
second positional, and `convert` also carries `--outtype`. argparse abbreviates long options by
default, so `--out` prefix-matched `--outtype` and the refusal complained about a precision choice,
naming a flag the user never typed and a value that is not a precision. `quantise` had the same
collision against `--output-tensor-type`.

THE SHAPE OF THE RULE

Both parsers now run with `allow_abbrev=False`, and both answer `--out` with the positional form
that works. That fixes two commands.

The rule below is the one that matters for the next command somebody writes, because this defect is
invisible until a user meets it: adding a flag called `--outtype` silently changes what `--out`
means on that command alone, and no existing test reads the relationship between two flag names.

WHAT THE RULE IS NOT, because the first version of it was wrong in a way worth keeping written down

It is not "no flag may be a prefix of another flag in the same parser". That check fired on twelve
pairs across the package, `--base` against `--base-cache` among them, and every one of them is
safe: **argparse prefers an exact match over an abbreviation**, so a declared `--base` is never
abbreviated into `--base-cache`. Acting on that list would have meant twelve changes to fix nothing,
which is the same error as the defect it was chasing, a check answering a broader question than the
one being asked.

The defect needs a flag that is NOT declared on this command. So the rule is: a flag this tool uses
somewhere, absent from this parser, and a strict prefix of exactly one flag this parser does
declare. That is a form a reader plausibly types here having learned it elsewhere in the same tool,
and argparse will answer it with a different flag's name. `--out` is that, on nine other commands.

It walks the AST rather than importing the parsers, for the reason `test_every_flag_is_documented`
gives: several of these modules import torch, and the question is about the declarations.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "src" / "senbonzakura"


#: Every name this package builds a parser with. A list rather than a literal in the walk, because
#: this check went BLIND once before and its own sanity assertion is what caught it.
#:
#: WHAT HAPPENED, 2026-09-27. Every `argparse.ArgumentParser(` became
#: `argresolve.ParserThatNamesUnknownFlags(` so a mistyped flag would be named alongside a missing
#: one. The `allow_abbrev=False` arguments were all still there and the behaviour never changed, but
#: this scan matched on the constructor's NAME, so it found 3 sites out of 28 and would have reported
#: clean for ever on the other 25. `test_no_parser_in_this_tool_abbreviates_at_all` failed only
#: because it asserts how many sites it found before asserting anything about them.
#:
#: The lesson is the one this file already contains twice over: **a guard that can stop looking is
#: worse than no guard**, because the silence reads identically to a pass. Any future parser base
#: class goes here on the same day it is written.
BUILDS_A_PARSER = ("ArgumentParser", "ParserThatNamesUnknownFlags", "add_parser")


def _long_flags(node):
    """Every `--flag` string declared by an `add_argument` call inside this function."""
    flags = []
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        if not (isinstance(inner.func, ast.Attribute) and inner.func.attr == "add_argument"):
            continue
        flags.extend((arg.value, inner.lineno) for arg in inner.args
                     if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"))
    return flags


def _abbreviation_is_off(node):
    """True when this function passes `allow_abbrev=False` to a parser it builds."""
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        for kw in inner.keywords:
            if kw.arg == "allow_abbrev" and getattr(kw.value, "value", None) is False:
                return True
    return False


def _parser_builders():
    """(file, function, flags, abbreviation_is_off) for every function that declares flags."""
    for path in sorted(PACKAGE.rglob("*.py")):
        if "vendor" in path.parts:      # upstream llama.cpp code, not ours to restyle
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            flags = _long_flags(node)
            if flags:
                yield path, node.name, flags, _abbreviation_is_off(node)


def _parser_constructions():
    """(file, line, abbreviation_is_off) for every place in the package that builds a parser.

    Construction sites rather than enclosing functions, because a helper that attaches flags to a
    parser it was handed (`livedisplay.add_argument`) has no parser to configure, and a function
    checked for a setting it cannot own is a guard that fails on correct code.
    """
    for path in sorted(PACKAGE.rglob("*.py")):
        if "vendor" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            builds = (isinstance(func, ast.Attribute) and func.attr in BUILDS_A_PARSER)
            if not builds:
                continue
            off = any(kw.arg == "allow_abbrev" and isinstance(kw.value, ast.Constant)
                      and kw.value.value is False for kw in node.keywords)
            yield path, node.lineno, off


def _every_flag_this_tool_uses():
    """Every long flag declared anywhere in the package, so "a form a reader arrives with" is real."""
    return {name for _p, _f, flags, _a in _parser_builders() for name, _line in flags}


def _collisions(*, ignore_the_setting=False):
    """(file, function, elsewhere, here): a flag from elsewhere that this parser misreads.

    `elsewhere` is declared somewhere in this tool but not on this command, and prefix-matches
    exactly one flag that is. argparse will accept it silently as `here`, or name `here` in the
    refusal. Exactly one match is the harmful case: two or more and argparse says "ambiguous option"
    and lists both, which is a message the user can act on.
    """
    known = _every_flag_this_tool_uses()
    found = []
    for path, function, flags, abbreviation_is_off in _parser_builders():
        if abbreviation_is_off and not ignore_the_setting:
            continue
        declared = {name for name, _line in flags}
        for elsewhere in sorted(known - declared):
            matches = sorted(n for n in declared if n.startswith(elsewhere))
            if len(matches) == 1:
                found.append((path.name, function, elsewhere, matches[0]))
    return sorted(found)


def test_the_scan_finds_the_parsers_it_is_meant_to_read():
    """Without this the suite below passes on an empty list, which is how a guard reports clean."""
    builders = list(_parser_builders())
    assert len(builders) > 20, f"only found {len(builders)} flag-declaring functions"
    assert any(name == "quantise.py" for name, *_rest in
               ((p.name, f) for p, f, _fl, _a in builders)), "quantise.py was not read"


def test_the_abbreviation_setting_is_load_bearing_and_not_decoration():
    """The collisions are still there in the flag names; the setting is the only thing covering them.

    Once every parser turns abbreviation off, a sweep that skips those parsers reports clean for
    ever and stops being worth reading. So this one reads the declarations as if the setting were
    not there, and asserts the misreadings it would produce are real. If this list ever empties, the
    global rule has become belt-and-braces rather than the fix, and that is worth knowing.
    """
    would_misread = _collisions(ignore_the_setting=True)
    assert would_misread, (
        "no flag in this tool is a prefix of another any more, so `allow_abbrev=False` is no longer "
        "load-bearing. Check whether this test still describes the project before deleting it.")

    pairs = {(elsewhere, here) for _n, _f, elsewhere, here in would_misread}
    for elsewhere, here in (("--out", "--outtype"), ("--seed", "--seeds")):
        assert (elsewhere, here) in pairs, (
            f"{elsewhere} no longer prefix-matches {here} anywhere, so this test's example has "
            f"moved and its list needs rereading")


def test_no_parser_in_this_tool_abbreviates_at_all():
    """The stronger rule the collision sweep argued its way into, and the one actually enforced.

    Fixing only the colliding parsers leaves the defect one new flag away from returning, silently,
    on any command: adding `--no-cache` to a parser that has no `--n` changes what `--n` means there
    and nothing tells you. The collision sweep above would catch it at that point, which is later
    than it needs to be caught.

    No command in this tool documents an abbreviated flag, no script in the repository passes one
    (`measure` and `headtohead` build their child argv with full names), and nothing in the guide
    teaches one, so turning abbreviation off everywhere costs a feature nobody was offered. It buys
    the property that what argparse accepts is exactly what the help page lists.
    """
    sites = list(_parser_constructions())
    assert len(sites) > 25, f"only found {len(sites)} parser constructions, so this checked nothing"
    on = sorted((path.name, line) for path, line, is_off in sites if not is_off)
    assert not on, (
        "these parsers still abbreviate long flags, so a flag added later can silently change what "
        "a shorter one means:\n"
        + "\n".join(f"  {name}:{line}" for name, line in on)
        + "\n  Pass allow_abbrev=False where the parser is built.")


def test_convert_and_quantise_have_abbreviation_off_by_name():
    """The two the audit found, asserted directly rather than only through the sweep above."""
    off = {(path.name, function) for path, function, _flags, is_off in _parser_builders() if is_off}
    for module in ("convert.py", "quantise.py"):
        assert (module, "build_parser") in off, (
            f"{module} no longer turns argparse abbreviation off, so `--out` there prefix-matches "
            f"a flag the user did not type")


@pytest.mark.parametrize(("module", "positional"),
                         [("convert", "<model>"), ("quantise", "<source>")])
@pytest.mark.parametrize("form", ["--out", "--out=x"])
def test_the_two_audited_commands_answer_out_with_the_form_that_works(module, positional, form,
                                                                     capsys):
    """The audit's own case, driven through the real parser rather than asserted about the source.

    Both halves matter. The refusal must name `--out`, the flag the user typed, and must not name
    the precision flag they did not; and it must say what to type instead, because "unrecognized
    arguments: --out" is true and useless to somebody who has just read nine other commands that
    take `--out`.
    """
    built = importlib.import_module(f"senbonzakura.{module}").build_parser()
    argv = ["input", form] if form.endswith("=x") else ["input", form, "x"]
    with pytest.raises(SystemExit) as exit_code:
        built.parse_args(argv)
    assert exit_code.value.code == 2
    said = capsys.readouterr().err

    assert form in said, f"the refusal does not name the flag the user typed:\n{said}"
    assert "outtype" not in said.split("usage:")[-1].split("error:")[-1], (
        f"the refusal still names a precision flag nobody typed:\n{said}")
    assert "positional" in said and positional in said, (
        f"the refusal does not say what to type instead:\n{said}")


def test_argparse_really_does_prefer_an_exact_match(capsys):
    """The premise the rule rests on, asserted rather than believed.

    If this ever stops being true, `--base` against `--base-cache` becomes a real defect in twelve
    places and the sweep above would report clean on every one of them, because it excludes the
    declared-prefix case on the strength of exactly this behaviour.

    THAT argparse refuses an ambiguous abbreviation is the premise. HOW it refuses is a version
    detail, and this test asserted the how. From 3.11 the ambiguity is raised as `ArgumentError`,
    which `exit_on_error=False` hands back to the caller. On 3.10 the same condition goes straight
    to `parser.error()` and exits, because `exit_on_error` did not cover that path yet: the CI log
    shows it at `argparse.py:2594`, calling `self.exit(2, ...)`.

    So this passed on a development box at 3.14 and failed on the two rows that exist to catch
    exactly that, the 3.10 matrix row and the declared-minimums job. Both refusals are now accepted
    and the message is what gets checked, which is the part that is the same everywhere.
    """
    ap = argparse.ArgumentParser(prog="t", exit_on_error=False)
    ap.add_argument("--base")
    ap.add_argument("--base-cache")
    assert ap.parse_args(["--base", "X"]).base == "X"

    with pytest.raises((argparse.ArgumentError, SystemExit)) as refused:
        ap.parse_args(["--bas", "X"])
    # An `ArgumentError` carries its own message; a `SystemExit` from `parser.error` printed it to
    # stderr on the way out, so both are read.
    said = f"{refused.value}\n{capsys.readouterr().err}"
    assert "ambiguous" in said, (
        f"argparse accepted `--bas` where both `--base` and `--base-cache` are declared, or "
        f"refused it for some other reason:\n{said}")

    # AND THE EXITING ROUTE, ON EVERY VERSION. `exit_on_error=True` is the default and takes the
    # same path 3.10 takes for this condition, so running it here means the interpreter that found
    # this failure is covered on the interpreters that did not. A version-specific branch tested
    # only on the version that has it is how this reached CI in the first place.
    exiting = argparse.ArgumentParser(prog="t")
    exiting.add_argument("--base")
    exiting.add_argument("--base-cache")
    assert exiting.parse_args(["--base", "X"]).base == "X"
    with pytest.raises(SystemExit) as left:
        exiting.parse_args(["--bas", "X"])
    assert left.value.code == 2, f"argparse exited {left.value.code}, and a usage error is 2"
    assert "ambiguous" in capsys.readouterr().err


def test_the_collision_check_would_fail_on_the_defect_it_was_written_for():
    """Mutation test: the sweep is only worth reading if it can fail.

    Reconstructs the real shape, a parser that declares `--outtype` and not `--out` while `--out`
    exists elsewhere in the tool, and checks the helper's own predicate reports it.
    """
    source = ("def build_parser():\n"
              "    ap = argparse.ArgumentParser()\n"
              "    ap.add_argument('--outtype')\n"
              "    return ap\n")
    node = ast.parse(source).body[0]
    declared = {name for name, _line in _long_flags(node)}
    assert declared == {"--outtype"}
    assert not _abbreviation_is_off(node)
    assert "--out" in _every_flag_this_tool_uses(), (
        "`--out` is no longer used anywhere in this tool, so this test's premise has moved")
    matches = [n for n in declared if n.startswith("--out")]
    assert matches == ["--outtype"], "the defect is a single silent match, and it is not reported"


def test_the_constructor_names_are_read_from_the_code_rather_than_remembered():
    """The list above must contain whatever `argresolve` actually exports as its parser class.

    This is the guard on the guard. The scan went blind once because a rename moved the constructor
    out from under a hardcoded name, and a comment asking the next person to remember is not a
    control. Renaming the class again fails here, on the same day, rather than silently halving what
    the sweep above examines.
    """
    from senbonzakura import argresolve

    exported = [name for name, value in vars(argresolve).items()
                if isinstance(value, type) and issubclass(value, argparse.ArgumentParser)
                and value is not argparse.ArgumentParser]
    assert exported, "argresolve exports no parser class, so this test's premise has moved"
    for name in exported:
        assert name in BUILDS_A_PARSER, (
            f"{name} builds parsers in this package and the scan above does not look for it, so "
            f"every parser built with it is unchecked. Add it to BUILDS_A_PARSER.")
