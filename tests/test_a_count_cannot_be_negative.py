# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A count or an offset is refused at parse time, on every command that takes one.

WHAT WAS WRONG

`score --skip -5` was accepted by everything it met. It is truthy, so it passed `if a.skip:`; it
is not `>= len(prompts)`, so it passed the bounds check; and `prompts[-5:]` is a perfectly good
slice. So the command reported a refusal rate measured over the LAST five prompts and stamped it
with the partition `rows-from--5`, a boundary that does not exist. `--n -5` dropped the last five
instead of taking the first five.

`capability` refused the same input correctly, which is the worse half of the finding: two
sibling commands disagreed about whether a negative sample size is a thing, and the one that
disagreed silently is the one a script reaches for.

So the check lives in `argresolve`, which exists precisely so each command does not re-decide the
same argument question, and these tests assert it from the parsers rather than from the helper
alone: a shared type nobody applied is the defect over again.
"""
from __future__ import annotations

import argparse

import pytest

from senbonzakura import capability
from senbonzakura.argresolve import whole_number


def test_a_negative_count_is_refused_with_a_sentence():
    parse = whole_number("--skip")
    with pytest.raises(argparse.ArgumentTypeError) as e:
        parse("-5")
    message = str(e.value)
    assert "--skip" in message and "-5" in message
    assert "cannot be below" in message, "the refusal has to say what would have been valid"


def test_zero_is_a_count(tmp_path):
    assert whole_number("--n")("0") == 0


def test_a_minimum_of_one_refuses_zero():
    """A batch of zero is `range(0, n, 0)`, which is a ValueError several screens from the flag."""
    with pytest.raises(argparse.ArgumentTypeError):
        whole_number("--batch", minimum=1)("0")
    assert whole_number("--batch", minimum=1)("1") == 1


def test_something_that_is_not_a_number_says_so():
    with pytest.raises(argparse.ArgumentTypeError, match="whole number"):
        whole_number("--n")("eight")


@pytest.mark.parametrize("flag", ["--n", "--skip", "--max-new", "--batch", "--bootstrap"])
def test_capability_refuses_a_negative_one_on_the_command_line(flag, capsys):
    with pytest.raises(SystemExit):
        capability.build_parser().parse_args(["m", flag, "-5"])
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--n", "--skip", "--max-new", "--batch", "--length-max"])
def test_score_refuses_a_negative_one_on_the_command_line(flag, capsys):
    """The command the defect was actually found on. It imports torch, so it is skipped rather
    than faked where torch is absent: a fake parser would test the fake.
    """
    pytest.importorskip("torch")
    from senbonzakura import score

    with pytest.raises(SystemExit):
        score.build_parser().parse_args(["--model", "m", "--eval", "e", "--out", "o", flag, "-5"])
    assert flag in capsys.readouterr().err


def test_the_two_commands_agree(capsys):
    """The finding was the disagreement, not either command on its own."""
    pytest.importorskip("torch")
    from senbonzakura import score

    for parser, argv in ((capability.build_parser(), ["m", "--skip", "-5"]),
                         (score.build_parser(),
                          ["--model", "m", "--eval", "e", "--out", "o", "--skip", "-5"])):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
