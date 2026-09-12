# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Every command says what it is when asked, and says it into a pipe.

THE DEFECT THIS WAS WRITTEN FOR. `senbonzakura interactive` never looked at its argv. On a
terminal `--help` started the interview; redirected, it printed the not-a-terminal refusal and
exited 2. So the one command written for somebody who has not read the flag list was the only
one that would not say what it was, and `senbonzakura interactive --help | less` (the way a
person actually reads help) told them nothing.

Asking what a command does must be answerable without the conditions for running it. A command
that needs a terminal, a GPU, a model or a network still has to answer `--help` in a pipe,
because that is where the answer gets read.

The operating assumption behind this file: neither a human nor an AI is going to read the
source, so anything discoverable only by grepping the code is undiscoverable in practice.
"""
import subprocess
import sys

import pytest

from senbonzakura.entry import DELEGATED

#: The modes `split_mode` peels off before the parser ever sees them, so they never appear in
#: `DELEGATED` and a table built from that alone would leave the default command untested.
MODES = ("abliterate", "kageyoshi", "auto")

COMMANDS = sorted(DELEGATED) + list(MODES)


def _help(command):
    """`--help` with stdout and stdin both pipes, which is the case this regressed on."""
    return subprocess.run([sys.executable, "-m", "senbonzakura", command, "--help"],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          check=False, timeout=300)


@pytest.mark.parametrize("command", COMMANDS)
def test_the_command_answers_help_into_a_pipe(command):
    out = _help(command)
    assert out.returncode == 0, (
        f"`senbonzakura {command} --help` exited {out.returncode} instead of describing itself.\n"
        f"stdout: {out.stdout[:400]}\nstderr: {out.stderr[:400]}")


@pytest.mark.parametrize("command", COMMANDS)
def test_the_help_describes_the_command_rather_than_only_naming_it(command):
    """A usage line alone is not a description. Every command owes a sentence of prose.

    The floor is deliberately low (a usage line plus real text) because the point is to catch a
    command that answers with nothing, not to grade the writing.
    """
    text = _help(command).stdout
    assert "usage:" in text, f"{command} printed no usage line"
    prose = [line for line in text.splitlines()
             if line.strip() and not line.startswith((" ", "\t")) and "usage:" not in line]
    assert prose, f"`senbonzakura {command} --help` gives a usage line and no sentence saying what it does"


@pytest.mark.parametrize("command", sorted(DELEGATED))
def test_the_usage_line_is_a_command_that_can_actually_be_run(command):
    """The usage line is the one line people copy, so it has to be the real invocation.

    THE DEFECT THIS WAS WRITTEN FOR. Five commands set `prog` to their MODULE path, so
    `senbonzakura compass --help` opened with `usage: senbonzakura.margin`. Copying that runs
    nothing, and `margin` does not share a name with `compass`, so it could not even be guessed
    at. `validate` was worse: it inherited `python -m senbonzakura` from sys.argv[0], a usage line
    with the subcommand missing entirely, which runs the ABLITERATOR if pasted.
    """
    first = _help(command).stdout.splitlines()[0]
    assert first.startswith(f"usage: senbonzakura {command}"), (
        f"`senbonzakura {command} --help` opens with:\n  {first}\n"
        f"which is not the command that was typed to get it")
