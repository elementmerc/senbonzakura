# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The checker can say which build it is, and names the command the reader actually typed.

WHAT PROMPTED IT, 2026-09-26

Two readers with no knowledge of this project used the checker and hit the same pair of small
things, both of which matter more to the audience this distribution exists for than to anyone else.

`senbonzakura-check --version` raised argparse's "unrecognised arguments" and exited 2. This package
is documented for continuous integration, where the entire point is recording what gated a build, so
a pipeline asking which checker ran got a usage block and a failure. The abliterator has carried
`--version` from the start; the gap survived in the smaller package because nobody ran it there.

The usage line said `senbonzakura check` whatever it was reached by. That is right when both
distributions are installed, where the two spellings are one command reached two ways. It is wrong
for the reader this package is FOR: the sceptic who installed the checker alone to avoid PyTorch,
who copies the usage line and finds no such command. One of them did copy it.

Neither is a wrong number. Both are the tool being unable to answer a question about itself, in a
tool whose whole argument is that claims should be checkable.
"""
import subprocess
import sys
from pathlib import Path

import pytest
from senbonzakura_check._version import __version__
from senbonzakura_check.cli import _invoked_as, build_parser

CHECKER_SRC = Path(__file__).resolve().parent.parent / "checker" / "src"


def test_the_parser_offers_a_version_flag():
    """Read off the parser, so this holds however the command was reached."""
    flags = {opt for action in build_parser()._actions for opt in action.option_strings}
    assert "--version" in flags, (
        "`--version` is gone. A build that cannot say which build it is cannot gate anything, and "
        "this distribution is documented for exactly that job.")


def test_the_version_is_the_checkers_own_name_and_number(capsys):
    """`action="version"` exits 0 through SystemExit; anything else is the old exit-2 failure."""
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0, (
        f"`--version` exited {exit_info.value.code}. It used to exit 2 with a usage block, which is "
        f"what a pipeline recording its gate received.")
    printed = capsys.readouterr().out.strip()
    assert printed == f"senbonzakura-check {__version__}", (
        f"`--version` printed {printed!r}. It names this distribution rather than the abliterator, "
        f"because the two ship separately and a bug report has to be traceable to one of them.")


@pytest.mark.parametrize(("argv0", "expected"), [
    pytest.param("/usr/local/bin/senbonzakura-check", "senbonzakura-check", id="hyphenated-binary"),
    pytest.param("senbonzakura-check", "senbonzakura-check", id="bare-hyphenated"),
    pytest.param("/usr/local/bin/senbonzakura", "senbonzakura check", id="via-the-abliterator"),
    pytest.param("", "senbonzakura check", id="no-argv0-at-all"),
])
def test_the_usage_line_names_how_it_was_reached(monkeypatch, argv0, expected):
    monkeypatch.setattr(sys, "argv", [argv0])
    assert _invoked_as() == expected


def test_it_really_names_itself_when_run_as_the_hyphenated_command():
    """The unit tests above monkeypatch argv. This one runs the module as the installed script would.

    Worth the subprocess: `prog` is computed at parser-build time from `sys.argv[0]`, and a refactor
    that moved the call earlier or cached the parser would pass every test above and still print the
    wrong name to a real user.
    """
    done = subprocess.run(
        [sys.executable, "-c",
         ("import sys; sys.argv = ['senbonzakura-check', '--help'];"
          "from senbonzakura_check.cli import build_parser; build_parser().print_usage()")],
        capture_output=True, text=True, timeout=60, check=False,
        env={"PYTHONPATH": str(CHECKER_SRC), "PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: senbonzakura-check"), done.stdout
