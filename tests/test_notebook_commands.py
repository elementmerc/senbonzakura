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
    import importlib

    from senbonzakura import entry, parser

    argv = shlex.split(command)[1:]          # drop the program name
    if argv and argv[0] in entry.DELEGATED:
        # A subcommand. The module is resolved through `entry.DELEGATED` rather than named here,
        # so a command that moves to a different module keeps being parsed instead of quietly
        # falling out of this test.
        #
        # `track` joined `capability` on 2026-09-23, when the notebook stopped using
        # `--track default`: the bundled track is packed from a held-out corpus outside this
        # repository, so a reader in Colab has to build their own.
        # A SUB-VERB IS A DIFFERENT PARSER, and parsing `track build --out X` with `track`'s own
        # parser would reject `build` rather than checking the flags that follow it. `corpora`
        # and `track build` both joined the notebook on 2026-09-23, when they stopped being
        # scripts under `tools/`: that directory ships in no wheel, so the notebook was cloning
        # the repository for two files.
        subverbs = {("track", "build"): ("trackbuild", "out")}
        key = (argv[0], argv[1] if len(argv) > 1 else None)
        if key in subverbs:
            module_name, required_arg = subverbs[key]
            rest = argv[2:]
        else:
            # `None` where a command requires nothing: the parse itself is the assertion.
            required = {"capability": "model", "track": "out", "corpora": None,
                        # `doctor` takes nothing and reports what the install can do. It replaced
                        # the two builder cells when the bundled track started arriving with the
                        # wheel, so the notebook now checks the install instead of assembling one.
                        "doctor": None}
            assert argv[0] in required, (
                f"the notebook now uses the {argv[0]!r} subcommand and this test does not know "
                f"which of its arguments must come out set. Teach it rather than dropping the "
                f"assertion")
            module_name, _attr = entry.DELEGATED[argv[0]]
            required_arg = required[argv[0]]
            rest = argv[1:]
        module = importlib.import_module(f"senbonzakura.{module_name}")
        if not hasattr(module, "build_parser"):
            # `doctor` builds its parser inside `main`, so there is nothing to parse against
            # without running it, and running it inspects the whole machine. What CAN be asserted
            # is that the command takes no arguments here: the moment the notebook passes one,
            # this fails and asks to be taught, which is the same bargain as the map above.
            assert hasattr(module, "main"), (
                f"senbonzakura.{module_name} has neither build_parser nor main, so nothing in "
                f"this test can say whether the notebook's invocation is valid")
            assert not rest, (
                f"the notebook now passes {rest} to {argv[0]!r}, whose parser is built inside "
                f"main and cannot be checked from here. Give it a build_parser, or teach this "
                f"test what those arguments mean")
            return
        args = module.build_parser().parse_args(rest)
        if required_arg:
            assert getattr(args, required_arg)
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


def test_the_notebook_installs_the_release_and_uses_the_track_that_comes_with_it():
    """INVERTED ON 2026-09-26, when 0.4.0 reached PyPI. The history is the point.

    This test used to assert the opposite of all three things below, and it was right to. PyPI
    served a withdrawn 0.3.0, `senbonzakura-check` was on no index at all, and the bundled track
    was packed from a held-out corpus outside this repository, so:

    - installing from PyPI handed a reader July's version, silently, and then commands it lacks;
    - `--track default` could not work, because no clone can build the packed track;
    - the notebook therefore had to build the corpora itself, and a reader who skipped that step
      met a failure several cells after the one that caused it.

    All three ended at the release. The wheel carries the packed track and the corpora, so
    `--track default` is the simplest correct thing a Colab reader can do, and the two builder
    cells that existed to work around the gap are now an optional aside.

    Left as a test rather than deleted, because the drift runs both ways: a notebook that goes
    back to `git+...@dev` would be exercising unreleased code in the surface most likely to be a
    stranger's first contact with this project.
    """
    doc = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "code")

    assert "%pip install --quiet senbonzakura" in code, (
        "the notebook no longer installs the published package. From 0.4.0 that is the whole "
        "install and it is what a reader in a browser should be running")
    assert "git+https://github.com/elementmerc/senbonzakura" not in code, (
        "the notebook installs from a branch. That was required while the index could not serve "
        "this project and is now a way to hand a stranger unreleased code")

    # ACTIVE lines only. The optional builder cells are commented out on purpose, and a check that
    # could not tell a live command from a commented one would fail on the explanation beside it.
    live = [ln for ln in code.splitlines() if not ln.lstrip().startswith("#")]
    assert any("--track default" in ln for ln in live), (
        "the notebook does not use `--track default`, which now ships inside the wheel. Building "
        "a corpus in Colab to avoid it is work the reader no longer has to do")
    assert "senbonzakura doctor" in "\n".join(live), (
        "the notebook does not run `doctor`, so a reader never sees what their install can and "
        "cannot do before the first command that depends on it")
