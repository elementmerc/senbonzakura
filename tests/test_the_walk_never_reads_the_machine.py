# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A test that reads the machine it runs on is this project's most repeated defect.

SIX OF THEM IN ONE DAY, 2026-09-28, and every one of them the same sentence: a stray `abliterated/`
left by an earlier run, the Hugging Face cache's contents reaching the model menu, the pre-flight
board running the real checks, the packed evaluation track (carried by a wheel, absent from a
checkout), the vendored converter and `llama-quantize` (present on one CI platform by accident),
and a directory holding a `best-config.json` turning up in the working directory. Each was fixed
where it was found, which is how the same fact ended up pinned in two test files and unpinned in
two others.

THE GUIDED WALK IS WHY THESE BITE HARDER HERE THAN ELSEWHERE. It is a sequence of questions
answered from a fixed list, so a probe that answers differently does not change one assertion, it
inserts or removes a QUESTION. Every canned answer after it then lines up against the wrong prompt
and the test dies somewhere far from the cause with `StopIteration`, on one platform, in a job
nobody was looking at.

SO THE FIX IS NOT ANOTHER FIX. Six occurrences of one sentence is a missing mechanism, and this
file is the mechanism. It holds two properties:

  1. every function in `interactive` is classified, as one that reads the machine or one that does
     not, so a seventh probe cannot be added without somebody deciding which it is; and
  2. every test file that drives the walk asks for the fixture that pins them.

Neither is a heuristic. There is no scan for filesystem calls here, because a scan knows the
spellings somebody thought of, and a guard that covers one spelling of a defect reports clean on
all the others. The set being closed is the whole point: the classification is exhaustive, so the
answer to "did we think of this one" is never needed.

The long-term shape is an injected seam (a `Machine` the walk is handed, rather than probes it
reaches for), which makes reading the machine structurally impossible rather than merely checked.
That is Q-43's follow-on, deferred to v0.5 by the operator on 2026-09-28; this file is what holds
the line until then.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from senbonzakura import interactive
from tests.conftest import MACHINE_READS

SOURCE = Path(inspect.getsourcefile(interactive))
TESTS = Path(__file__).parent

#: Everything else in `interactive`, declared to read nothing but its arguments.
#:
#: A NEW NAME FAILS UNTIL IT IS CLASSIFIED, and that refusal is the whole value of this list. The
#: cost is one line per function added to the module; the thing it buys is that nobody discovers
#: the answer from a red macOS job three weeks later.
DOES_NOT_READ_THE_MACHINE = frozenset({
    # `_no_terminal` prints a refusal and `_rule` draws a line to `say.width()`. Neither touches
    # the install; `_rule` reads COLUMNS, which is a terminal property rather than a machine one and
    # is what every width test in this project sets deliberately.
    "_no_terminal", "_rule",
    "_argv_for", "_board", "_bundled_entries", "_duration", "_log_bug_line", "_machine_rows",
    "_next_steps", "_probe_budget", "_say", "_size", "_what_it_cost", "_what_survived", "_wrap",
    "ask", "ask_output", "ask_trials", "by_side", "cached_models", "choose", "confirm",
    "edited_models", "finished", "is_tty", "licence_for_track", "log_failure", "offer_resume",
    "pick_device", "pick_eval", "pick_model", "pick_side", "pick_track", "plan_abliteration",
    "present", "quote", "render_command", "resume_plan", "run", "size_the_probe_for", "steps",
    "warn_if_unbundled",
})

#: Test files that import `interactive` without driving a walk through it, so the fixture would
#: pin screens they never reach. Each is listed deliberately rather than pattern-matched.
NOT_A_WALK = frozenset({
    "test_the_walk_never_reads_the_machine.py",
})


def _module_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}


def test_every_function_in_the_walk_is_classified():
    """Two lists, and between them they must cover the module exactly."""
    declared = set(MACHINE_READS) | DOES_NOT_READ_THE_MACHINE
    actual = _module_functions()

    unclassified = sorted(actual - declared)
    assert not unclassified, (
        f"{', '.join(unclassified)} is new in interactive.py and nobody has said whether it reads "
        f"the machine. If it does, add it to MACHINE_READS in tests/conftest.py with what it "
        f"should answer under test; if it does not, add it to DOES_NOT_READ_THE_MACHINE here. "
        f"Guessing is how six tests came to depend on whatever the runner happened to have.")

    gone = sorted(declared - actual)
    assert not gone, (
        f"{', '.join(gone)} is classified here but no longer exists in interactive.py. A stale "
        f"entry is a pin that protects nothing, and it reads as cover that is not there.")


@pytest.mark.parametrize("name", sorted(MACHINE_READS))
def test_each_pinned_probe_is_really_a_function_on_the_module(name):
    """A pin whose name has drifted silently stops pinning. `monkeypatch.setattr` would still
    accept it, because the attribute exists until it does not, so this asks directly.
    """
    assert callable(getattr(interactive, name, None)), (
        f"MACHINE_READS pins `{name}`, which is not a callable on interactive. The pin is doing "
        f"nothing and every walk under it is reading the real machine.")


def test_the_fixture_answers_for_every_probe_it_claims(a_machine_with_nothing_on_it):
    """THE PIN IS PROVEN, not assumed. Asking each probe through the fixture is the only way to
    tell a pin that took from one that named the wrong module.
    """
    assert interactive.models_on_this_machine() == []
    assert interactive.resumable_runs() == []
    assert interactive.missing_conversion_tools() == []
    assert interactive._checked_rows(None) == []
    assert interactive.device_available("cpu") is True
    assert interactive.device_available("cuda") is False
    assert interactive.bundled.is_available() is True


#: Calls that mean a test file has reached the walk.
#:
#: `run(` AND `log_failure(` WERE BOTH MISSING, and the first is the walk's own front door.
#: Found on 2026-10-01 by predicting what the real `conftest.py` would do to two new test files
#: rather than by this guard firing. `interactive.run` is what `senbonzakura interactive` calls and
#: it reaches `plan_abliteration` one line in, so a file that drives the entry point drives every
#: probe, and this list did not mention it: a guard against reading the machine that could not see
#: the commonest way of doing it.
#:
#: `log_failure` is the other kind. It calls `_what_survived`, which READS THE OUTPUT DIRECTORY off
#: the filesystem and is right to, so a test giving it a relative `--out` asks whatever directory
#: the runner was started in. One of the two new files did exactly that and the screen changed
#: branch when an unrelated `./out` existed.
_WALK_CALLS = ("plan_abliteration", "offer_resume", "pick_model", "pick_device", "pick_track",
               "pick_eval", "present(", "finished(", "log_failure(", "ask_output(")

#: The entry point, matched through whatever name the file imported the module under.
#:
#: `.run(` ON ITS OWN WAS TOO WIDE, which is the other way to get a guard wrong. A first attempt
#: used the bare substring and immediately claimed five files that have nothing to do with the
#: walk, because `subprocess.run(` and `cli.run(` match it too. A guard that fires on unrelated
#: files gets a blanket exemption added to it, and then it guards nothing. So the alias is read out
#: of the import rather than guessed.
_IMPORT_ALIAS = re.compile(
    r"^\s*from\s+senbonzakura\s+import\s+interactive(?:\s+as\s+(\w+))?"
    r"|^\s*from\s+\.\s+import\s+interactive(?:\s+as\s+(\w+))?", re.MULTILINE)


def _drives_a_walk(path):
    """Whether a test file reaches the walk, which is what makes the machine matter to it."""
    text = path.read_text(encoding="utf-8")
    if "interactive" not in text:
        return False
    if any(call in text for call in _WALK_CALLS):
        return True
    aliases = {m.group(1) or m.group(2) or "interactive" for m in _IMPORT_ALIAS.finditer(text)}
    return any(f"{alias}.run(" in text for alias in aliases)


def test_every_file_that_drives_the_walk_pins_the_machine():
    """THE SECOND HALF, and the one the copies were hiding. Four files drove the walk, two pinned
    the machine, and the two that did not were green only because the box they ran on happened to
    be bare. A file added next month gets the same treatment as the four here.
    """
    missing = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name in NOT_A_WALK or not _drives_a_walk(path):
            continue
        if 'usefixtures("a_machine_with_nothing_on_it")' not in path.read_text(encoding="utf-8"):
            missing.append(path.name)
    assert not missing, (
        f"{', '.join(missing)} drives the guided walk without pinning the machine, so it passes or "
        f"fails on what the box running it happens to hold. Add at module level:\n"
        f'    pytestmark = pytest.mark.usefixtures("a_machine_with_nothing_on_it")\n'
        f"If the file genuinely does not walk, name it in NOT_A_WALK here with the reason.")


# ── the pin that swallows the test it was protecting ─────────────────────────────────
#
# A PIN MAKES A TEST PASS, WHICH IS THE DANGER. Four tests here asserted `resumable_runs` returns
# nothing for an ordinary directory, a nested one, an unreadable root and an empty tree. Under the
# stand-in they still passed, instantly, having asked the stand-in rather than the probe: the
# negative cases of the one function they existed to hold down stopped being tested and nothing
# said so. That is this project's other recurring shape, a check answering a narrower question
# than the one being asked and reporting clean.
#
# So a test that NAMES a pinned probe must say it wants the real one. Naming it is the signal:
# nobody calls `resumable_runs` in a test body except to test `resumable_runs`.
def _tests_calling_a_pinned_probe(path):
    """Test functions in `path` that call a pinned probe, and which probes each one calls."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        called = {n.func.attr for n in ast.walk(node)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr in MACHINE_READS
                  and isinstance(n.func.value, ast.Name)
                  and n.func.value.id in {"interactive", "it"}}
        if called:
            out[node.name] = (called, node)
    return out


def _unpinned_by(node):
    """The probes a test's own decorators ask to have back."""
    asked = set()
    for dec in node.decorator_list:
        # BOTH SPELLINGS. `@...reads_the_real_install` is an attribute and
        # `@...reads_the_real_install("x")` is a call, the fixture accepts either, and a guard
        # that read only one of them would report a test uncovered while it was covered.
        call = dec if isinstance(dec, ast.Call) else None
        target = call.func if call is not None else dec
        if not isinstance(target, ast.Attribute) or target.attr != "reads_the_real_install":
            continue
        if call is None or not call.args:
            return set(MACHINE_READS)                     # naming none asks for all of them
        asked |= {a.value for a in call.args if isinstance(a, ast.Constant)}
    return asked


def test_a_test_that_asks_a_probe_a_question_gets_the_real_probe():
    findings = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name in NOT_A_WALK:
            continue
        if 'usefixtures("a_machine_with_nothing_on_it")' not in path.read_text(encoding="utf-8"):
            continue
        for name, (called, node) in _tests_calling_a_pinned_probe(path).items():
            short = called - _unpinned_by(node)
            if short:
                findings.append(f"{path.name}::{name} calls {', '.join(sorted(short))}")
    assert not findings, (
        "these tests call a pinned probe and get the stand-in, so they pass without asking the "
        "thing they name:\n  " + "\n  ".join(findings) +
        '\nAdd @pytest.mark.reads_the_real_install("<probe>") to each, naming only what it needs.')


def test_nobody_hand_writes_a_pin_that_conftest_already_owns():
    """THE THIRD HOME, and the one an AST could not see.

    A walk that runs in a subprocess is out of every fixture's reach, so it pinned its probes by
    assigning them in the source it builds as a string. It got two of the six, and the four it
    missed were read off whatever box ran the suite. Text rather than syntax here on purpose: the
    assignment that hid was inside a string literal, where an AST sees only a string.
    """
    pattern = re.compile(r"^\s*(?:it|interactive)\.(" + "|".join(MACHINE_READS) + r")\s*=",
                         re.MULTILINE)
    findings = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name in NOT_A_WALK:
            continue
        findings.extend(f"{path.name}: {m.group(0).strip()}"
                        for m in pattern.finditer(path.read_text(encoding="utf-8")))
    assert not findings, (
        "these pin a probe by hand, which is a second copy of what conftest already owns and will "
        "drift from it the way the last two copies did:\n  " + "\n  ".join(findings) +
        "\nUse the `a_machine_with_nothing_on_it` fixture, or `machine_pins()` where no fixture "
        "reaches, such as inside a subprocess probe.")
