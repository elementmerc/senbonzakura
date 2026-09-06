# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""No message may recommend a flag the command it came from will not accept.

WHY THIS FILE EXISTS

The abliterator's own note about its weakest claim ended with "which needs a topic-matched
harmless set (--harmless-matched)". That flag lives on `margin`, not on the abliterator, so the
tool named the remedy for its own stated limitation and then exited 2 if you took the advice. It
printed on every run whose filter rejected nothing, which is most of them, and it survived a
dead-flag audit because that audit asks whether a declared flag is READ, and this is the mirror
image: a flag that is RECOMMENDED and does not exist.

Found on real hardware by a session running the shipped wheel, 2026-09-05, having already been
noticed and written down three days earlier without being fixed. Two independent discoveries of
the same defect is what an automated check is for.
"""
import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "senbonzakura"

#: A long flag as it appears inside a message: two dashes, then a lowercase word, then more
#: dash-joined words. Deliberately narrow, so `--` in prose and `---` rules are not swept up.
FLAG = re.compile(r"--[a-z][a-z0-9]*(?:-[a-z0-9]+)*")

#: Flags belonging to OTHER PROGRAMS. Ours must exist on one of our parsers; these never will,
#: and each carries the reason so a reviewer can tell an exemption from an oversight.
FOREIGN = {
    "--force-reinstall": "pip, in the damaged-install message",
    "--no-deps": "pip",
    "--network": "docker, in the clean-room documentation",
    "--gpus": "docker",
    "--no-verify": "git",
    "--always": "git describe, in code_version's subprocess argument list",
    "--dirty": "git describe",
    "--abbrev": "git describe",
    "--print-completion": "shtab installs this one itself",
}

#: How a message says "this flag belongs to a different command of ours". A flag from another
#: command is fine when the message names that command, and a dead end when it does not: the
#: reader has no way to discover which command to type.
NAMES_A_COMMAND = re.compile(r"senbonzakura[ .]([a-z][a-z-]+)")


def _parser_flags(build):
    out = set()
    for action in build()._actions:
        out.update(o for o in action.option_strings if o.startswith("--"))
    return out


def _docstring_nodes(tree):
    """Every docstring, so internal prose is not read as advice to a user."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _message_flags(path):
    """Every long flag a USER could be shown, with its line number and the text around it.

    Docstrings are excluded: they are notes to whoever edits the file, not advice to a reader of
    the output, and holding them to the same rule would make the check noisy enough to ignore.

    A flag ending an f-string fragment is skipped, because the next thing is an interpolation and
    the real flag is only known at runtime. `--skip-{what}` is `--skip-harmful` or
    `--skip-harmless`, and reading it as `--skip` would be the check inventing a defect.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_nodes(tree)
    # `ast.walk` yields an f-string's literal fragments as standalone Constants as well as through
    # the JoinedStr, so without this every fragment is scanned twice: once knowing an interpolation
    # follows and once not. The second reading is what resurrected `--skip` from `--skip-{what}`.
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            skip.update(id(p) for p in node.values if isinstance(p, ast.Constant))
    found = []

    def scan(text, lineno, truncated):
        for m in FLAG.finditer(text):
            # An interpolation follows, and nothing but the joining dash separates it from the
            # match, so the real flag is `--skip-` plus whatever is substituted. The regex cannot
            # include a trailing dash (that would match `--skip-` in ordinary prose too), which is
            # why the remainder is inspected rather than the match extended.
            if truncated and text[m.end():] in ("", "-"):
                continue
            found.append((m.group(), lineno, text))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            scan(node.value, node.lineno, truncated=False)
        elif isinstance(node, ast.JoinedStr):
            parts = node.values
            for i, part in enumerate(parts):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    follows = i + 1 < len(parts) and isinstance(parts[i + 1], ast.FormattedValue)
                    scan(part.value, node.lineno, truncated=follows)
    return found


def _offenders_in_text(text, accepted):
    """The rule applied to one string, so the detector can be tested without a file on disk."""
    return [f for f in FLAG.findall(text)
            if f not in accepted and f not in FOREIGN and not NAMES_A_COMMAND.search(text)]


def _offenders(path, accepted):
    """Flags this command would reject, in messages that do not name the command that accepts them."""
    bad = []
    for flag, line, text in _message_flags(path):
        if flag in accepted or flag in FOREIGN:
            continue
        if NAMES_A_COMMAND.search(text):
            continue              # cross-command advice, with the command spelled out
        bad.append((flag, line))
    return bad


def test_the_abliterator_never_recommends_a_flag_it_would_reject():
    """THE REGRESSION THIS FILE IS NAMED FOR.

    `cli.py` is the abliterate path. A flag it prints must be one `build_parser` accepts, or an
    explicitly reasoned exemption.
    """
    from senbonzakura.parser import build_parser
    accepted = _parser_flags(build_parser)
    offenders = _offenders(SRC / "cli.py", accepted)
    assert not offenders, (
        "cli.py recommends flags the abliterator does not accept, without naming the command "
        "that does, so a reader who follows the advice gets an argument error:\n  "
        + "\n  ".join(f"{flag} at cli.py:{line}" for flag, line in offenders))


@pytest.mark.parametrize("module", ["margin", "score", "coherence", "drift"])
def test_the_delegated_commands_do_not_recommend_flags_they_reject(module):
    """The same rule for every command with its own parser.

    These four build their parsers on top of `loader_parser`, so a flag named in one of their
    messages has to survive its own command line rather than the abliterator's.
    """
    import importlib
    mod = importlib.import_module(f"senbonzakura.{module}")
    build = getattr(mod, "build_parser", None)
    if build is None:
        pytest.skip(f"{module} builds its parser inline rather than through build_parser")
    accepted = _parser_flags(build)
    offenders = _offenders(SRC / f"{module}.py", accepted)
    assert not offenders, (
        f"{module}.py recommends flags it does not accept:\n  "
        + "\n  ".join(f"{flag} at {module}.py:{line}" for flag, line in offenders))


def test_the_check_would_catch_the_defect_it_was_written_for():
    """A gate only ever seen passing has not been shown to work.

    Reconstructs the shape of the string that shipped and requires the detector to flag it.

    THE ORIGINAL FLAG IS NO LONGER ABSENT. The shipped defect named `--harmless-matched`, and on
    2026-09-06 the abliterator gained that flag for real, so the historical string stopped being a
    dead end and this test stopped proving anything. The instruction left in the old assertion was
    to rewrite it around a flag that is still missing rather than to delete it, because what is
    under test is the DETECTOR, not that one string. `--skip-harmful` is the stand-in: it lives on
    `margin` and the abliterator has never accepted it.
    """
    from senbonzakura.parser import build_parser
    accepted = _parser_flags(build_parser)
    stand_in = "--skip-harmful"
    shipped = ("it is NOT evidence they carry refusal rather than topic, which needs a "
               f"topic-matched harmless set ({stand_in}).")
    named = FLAG.findall(shipped)
    assert named == [stand_in]
    assert named[0] not in accepted, (
        f"if the abliterator now accepts {stand_in}, this test needs rewriting around a flag that "
        f"is still absent, not deleting")
    assert named[0] not in FOREIGN, "the defect must not be exempted into invisibility"
    assert not NAMES_A_COMMAND.search(shipped), (
        "the shipped string named no command, which is exactly why it was a dead end")
    assert _offenders_in_text(shipped, accepted), "the detector no longer flags the shipped shape"


def test_the_original_defect_is_fixed_rather_than_merely_untestable():
    """The other half: `--harmless-matched` is now a real abliterator flag, not a dead end.

    Without this, the test above could be satisfied by a stand-in while the original defect
    quietly came back.
    """
    from senbonzakura.parser import build_parser
    assert "--harmless-matched" in _parser_flags(build_parser)


def test_the_exemptions_all_carry_a_reason():
    assert all(isinstance(v, str) and v.strip() for v in FOREIGN.values())


def test_naming_the_command_is_what_makes_a_foreign_flag_acceptable():
    """The rule in one line: a flag from another command needs that command named beside it."""
    assert NAMES_A_COMMAND.search("run python -m senbonzakura.score --load-in-4bit")
    assert NAMES_A_COMMAND.search("score it with `senbonzakura margin --harmless-matched`")
    assert not NAMES_A_COMMAND.search("which needs a topic-matched set (--harmless-matched)")


def test_an_interpolated_flag_is_not_read_as_a_shorter_one():
    """`--skip-{what}` is --skip-harmful or --skip-harmless, and is not a flag called --skip."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.py"
        p.write_text('def f(what):\n    return f"pass --skip-{what} please"\n', encoding="utf-8")
        assert [f for f, _l, _t in _message_flags(p)] == []


def test_the_pattern_does_not_sweep_up_prose():
    """`--` used as punctuation, and markdown rules, are not flags."""
    assert FLAG.findall("a long -- dash and --- a rule") == []
    assert FLAG.findall("pass --max-directions 3") == ["--max-directions"]
    assert FLAG.findall("--Uppercase is not a flag we emit") == []
