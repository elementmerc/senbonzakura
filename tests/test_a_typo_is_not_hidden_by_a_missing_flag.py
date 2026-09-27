# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A mistyped flag stayed invisible whenever a required one was also absent.

WHAT PROMPTED IT, 2026-09-27

`senbonzakura gate --baselien x --current y` reported only `the following arguments are required:
--baseline`. The typo was never mentioned, so the obvious next move is to add `--baseline` and run
again, and now the command carries a correct flag AND one that is silently ignored. A surface audit
found it on every subcommand with a required flag.

The ordering is inside argparse: `parse_known_args` raises the required-argument error before it
returns anything about unrecognised tokens, so there is nothing to reorder at the call site.

THE SHAPE OF THE FIX, and what it deliberately is not

Operator decision, 2026-09-27: option A1, which is strictly additive. It never rejects anything that
parses today; it adds a sentence to a refusal that was already happening. The stronger version, a
pre-parse scan that refuses the unknown flag FIRST so the typo is the headline, is recorded for v0.5:
a scan that refuses can be wrong about a legitimate value beginning with a double dash, and this
cannot be wrong about anything.

That asymmetry is the whole argument, so the tests below check both halves: that the sentence appears
where it should, and that nothing which used to parse has stopped parsing.
"""
from __future__ import annotations

import argparse

import pytest

from senbonzakura import argresolve


def _parser(**kw):
    ap = argresolve.ParserThatNamesUnknownFlags(prog="t", allow_abbrev=False, **kw)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--threads", type=int)
    ap.add_argument("pos", nargs="?")
    return ap


def _refused(ap, argv, capsys):
    with pytest.raises(SystemExit) as e:
        ap.parse_args(argv)
    assert e.value.code == 2
    return " ".join(capsys.readouterr().err.split())


class TestTheTypoIsNamedAlongsideTheMissingFlag:
    def test_both_problems_are_reported(self, capsys):
        said = _refused(_parser(), ["x", "--sedes", "42"], capsys)
        assert "required: --seeds" in said, "the original complaint is gone"
        assert "--sedes" in said, "the typo is still invisible, which is the whole defect"

    def test_it_says_why_the_typo_matters(self, capsys):
        """Otherwise the reader fixes the missing flag and leaves the ignored one in place."""
        said = _refused(_parser(), ["x", "--sedes", "42"], capsys)
        assert "silently ignored" in said

    def test_several_unknown_flags_are_all_named(self, capsys):
        said = _refused(_parser(), ["x", "--sedes", "1", "--thraeds", "2"], capsys)
        assert "--sedes" in said and "--thraeds" in said

    def test_the_sentence_is_wrapped(self, capsys):
        from senbonzakura import say

        with pytest.raises(SystemExit):
            _parser().parse_args(["x", "--sedes", "42"])
        for line in capsys.readouterr().err.splitlines():
            assert len(line) <= say.CEILING, f"{len(line)} columns: {line!r}"

    def test_nothing_is_added_when_every_flag_is_spelled_correctly(self, capsys):
        """The addition must not fire on a plain missing-argument error."""
        said = _refused(_parser(), ["x", "--threads", "4"], capsys)
        assert "required: --seeds" in said
        assert "not a flag" not in said, f"a correct command was told it had a typo: {said}"


class TestItNeverRejectsWhatUsedToParse:
    """The property that makes A1 safe, and the reason the refusing version waits for v0.5."""

    def test_a_valid_command_still_parses(self):
        got = _parser().parse_args(["x", "--seeds", "42", "--threads", "4"])
        assert (got.pos, got.seeds, got.threads) == ("x", "42", 4)

    def test_a_negative_number_is_a_value_and_not_a_flag(self):
        """A single dash could be a negative number, which is why only `--` is considered."""
        assert _parser().parse_args(["--seeds", "1", "--threads", "-1"]).threads == -1

    def test_a_value_containing_a_double_dash_is_not_treated_as_a_flag(self):
        got = _parser().parse_args(["--seeds", "a=--b"])
        assert got.seeds == "a=--b"

    def test_an_unknown_flag_on_its_own_is_still_just_unrecognised(self, capsys):
        """With nothing required missing, argparse's own message is already the right one."""
        ap = argresolve.ParserThatNamesUnknownFlags(prog="t", allow_abbrev=False)
        ap.add_argument("--seeds")
        said = _refused(ap, ["--sedes", "1"], capsys)
        assert "unrecognized arguments" in said


class TestTheScanItself:
    def test_it_finds_only_undeclared_double_dash_tokens(self):
        ap = _parser()
        assert argresolve.unknown_long_flags(ap, ["--seeds", "1", "--sedes", "2"]) == ["--sedes"]
        assert argresolve.unknown_long_flags(ap, ["--threads=4"]) == []
        assert argresolve.unknown_long_flags(ap, ["--threads=4", "--thr=4"]) == ["--thr=4"]
        assert argresolve.unknown_long_flags(ap, ["-x", "pos", "-1"]) == []


class TestItDeclinesOnAParserWithSubcommands:
    """There, argv holds a subcommand's own flags, which the top-level parser does not declare.

    Reporting them as typos would be the defect this file is about, pointing the other way: a guard
    telling somebody a correct flag does not exist. argparse hands the subparser its own slice of
    argv, so the behaviour still applies where it can be right.
    """

    def test_a_subcommands_flags_are_not_reported_as_typos_by_the_parent(self, capsys):
        top = argresolve.ParserThatNamesUnknownFlags(prog="t", allow_abbrev=False)
        top.add_argument("--must", required=True)
        sub = top.add_subparsers(dest="cmd")
        run = sub.add_parser("run")
        run.add_argument("--tools")

        said = _refused(top, ["run", "--tools", "a"], capsys)
        assert "required: --must" in said
        assert "not a flag" not in said, (
            f"the parent reported a subcommand's own flag as a typo: {said}")

    def test_a_subparser_is_the_same_class_so_it_keeps_the_behaviour(self):
        """Argparse builds subparsers from the parent's own type, which is what carries this."""
        top = argresolve.ParserThatNamesUnknownFlags(prog="t", allow_abbrev=False)
        sub = top.add_subparsers(dest="cmd")
        assert isinstance(sub.add_parser("run"), argresolve.ParserThatNamesUnknownFlags)


class TestTheWholePackageUsesIt:
    def test_no_module_builds_a_bare_argument_parser(self):
        """Otherwise the behaviour reaches whichever commands somebody remembered.

        The same rule as `allow_abbrev=False`: a property that holds on most parsers is a property a
        reader cannot rely on, and the next command added is the one that misses it.
        """
        import pathlib

        package = pathlib.Path(argresolve.__file__).resolve().parent
        offenders = []
        for path in sorted(package.rglob("*.py")):
            if "vendor" in path.parts or path.name == "argresolve.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "argparse.ArgumentParser(" in text:
                offenders.append(path.name)
        assert not offenders, (
            "these modules build a plain ArgumentParser, so a mistyped flag stays hidden there: "
            + ", ".join(offenders)
            + ". Use argresolve.ParserThatNamesUnknownFlags.")

    def test_a_real_command_with_a_required_flag_reports_both(self, capsys):
        """END TO END on `gate`, which is where the audit met this."""
        from senbonzakura import gate

        with pytest.raises(SystemExit):
            gate.build_parser().parse_args(["--baselien", "x", "--current", "y"])
        said = " ".join(capsys.readouterr().err.split())
        assert "required: --baseline" in said
        assert "--baselien" in said


def test_the_class_is_still_an_argument_parser():
    """Anything that type-checks or subclasses further must keep working."""
    assert issubclass(argresolve.ParserThatNamesUnknownFlags, argparse.ArgumentParser)
