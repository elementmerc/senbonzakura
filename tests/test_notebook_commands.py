# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every command in the Colab notebook is one this package actually has.

WHY THIS EXISTS, AND IT IS NOT HYPOTHETICAL

The first draft of `notebooks/senbonzakura_colab.ipynb` called `senbonzakura compass` with flags
it does not take, `senbonzakura capability` without its required benchmark, and `senbonzakura
abliterate`, which is not a subcommand at all: abliteration is the bare invocation. Every one of
those would have failed on somebody's first run, in a browser, with no way to tell whether they
had done something wrong or the tool was broken.

Nothing would have caught it. The notebook is data, it is not imported, the suite never opens it,
and a reviewer reading the diff sees plausible command lines. This is the same shape as every
documentation defect this project has found: the words and the code drift, and the words are the
half nobody runs.

So the notebook's commands are parsed here, by the real parsers, with no model and no GPU. That
proves the flags exist and are spelled correctly. It does not prove the run succeeds, which needs
hardware, and the file says so rather than implying otherwise.
"""
import json
import pathlib
import shlex

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "senbonzakura_colab.ipynb"


def _shell_commands():
    """Every `!senbonzakura ...` line in the notebook, joined across line continuations."""
    doc = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    commands = []
    for cell in doc["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        # Line continuations first: the notebook wraps long invocations for readability, and a
        # reader of the raw source would otherwise see five broken commands.
        text = text.replace("\\\n", " ")
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("!senbonzakura"):
                commands.append(line[1:])
    return commands


def test_the_notebook_is_valid_and_has_commands():
    assert NOTEBOOK.is_file()
    commands = _shell_commands()
    assert commands, "no senbonzakura commands found, so this file is asserting nothing"


@pytest.mark.parametrize("command", _shell_commands(), ids=lambda c: c.split()[1][:24])
def test_every_notebook_command_parses(command):
    """THE ASSERTION. A flag that does not exist fails here rather than in somebody's browser."""
    from senbonzakura import capability, entry, parser

    argv = shlex.split(command)[1:]          # drop the program name
    if argv and argv[0] in entry.DELEGATED:
        # A subcommand. Each one owns its parser; `capability` is the only one the notebook uses,
        # and this looks it up rather than hard-coding it so a moved command fails loudly.
        assert argv[0] == "capability", (
            f"the notebook now uses the {argv[0]!r} subcommand and this test only knows how to "
            f"parse 'capability'. Teach it rather than dropping the assertion")
        args = capability.build_parser().parse_args(argv[1:])
        assert args.model
    else:
        # The bare invocation, which is abliteration. There is no `abliterate` subcommand, and the
        # first draft of the notebook invented one.
        assert argv[0] != "abliterate", (
            "abliteration is the bare invocation. `senbonzakura abliterate` is not a command and "
            "fails with an unrecognised-argument error")
        args = parser.build_parser().parse_args(argv)
        assert args.model


def test_the_notebook_does_not_promise_a_subcommand_that_does_not_exist():
    """A prose mention of a command is as wrong as a broken cell, and easier to miss.

    The markdown mentions `senbonzakura check`, which does exist. This catches the next one that
    does not.
    """
    from senbonzakura import entry

    doc = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    prose = "\n".join("".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "markdown")
    mentioned = set()
    for token in prose.replace("`", " ").split():
        if token in entry.DELEGATED or token == "abliterate":
            mentioned.add(token)
    assert "abliterate" not in mentioned, (
        "the notebook's prose names `abliterate` as a command. It is the bare invocation")
    unknown = mentioned - set(entry.DELEGATED)
    assert not unknown, f"the notebook names commands that do not exist: {sorted(unknown)}"


def test_the_notebook_does_not_send_a_reader_to_pypi_while_pypi_is_stale():
    """THE DEFECT THIS FILE DID NOT CATCH THE FIRST TIME.

    `test_docs_match_the_package.py` keeps a list of install surfaces and asserts each one warns
    that PyPI serves a withdrawn 0.3.0. The notebook is an install surface and was not on that
    list, so it shipped `%pip install senbonzakura` as its very first cell: a reader following it
    in a browser would have got July's version, silently, and then met commands that version does
    not have.

    Switching it to the repository alone is not enough either, and that is the second half. A
    clone carries no bundled corpora, because they are generated rather than committed, so
    `--track default` fails on a git install. The notebook therefore has to build them, and this
    asserts it does, because a reader cannot be expected to know that the failure two cells later
    is about a build step nobody ran.
    """
    doc = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    everything = "\n".join("".join(c["source"]) for c in doc["cells"])

    assert "pip install --quiet senbonzakura\n" not in everything, (
        "the notebook installs from PyPI, which serves a withdrawn 0.3.0 whose numbers are "
        "retracted and which has none of the measurement this notebook demonstrates")

    uses_bundled_track = "--track default" in everything
    if uses_bundled_track:
        assert "build_corpora.py" in everything, (
            "the notebook uses `--track default` and never builds the corpora. They are generated "
            "rather than committed, so a clone does not have them and the run fails on the first "
            "real command, which is the 0.3.0 defect in a browser")

    code = "\n".join("".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "code")
    assert "bundled.is_available()" in code, (
        "nothing in the notebook checks the track was actually built, so a reader whose build "
        "step failed finds out several minutes later from a command that looks unrelated")
