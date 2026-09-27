# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A flag the caller typed wins over the preset, for every knob the preset has an opinion about.

WHY THE MECHANISM IS TESTED AND NOT THE THREE FLAGS

`_apply_kageyoshi` resolves the search budget from the model, which is the point of the preset. It
used to do that unconditionally, and silently discarded `--trials 200`: the run executed 100 trials
while its own command line, its spec and its published table all said 200. An equal-budget
comparison cannot survive that, and nothing in the log said it had happened. The fix was `preset()`,
which skips a knob the caller set by hand and logs that it did.

**That fix covered nine budget knobs and left three quality levers as plain assignments.** So
`--trials` was honoured and `--search`, `--per-component` and `--mlp-off` were overridden without a
word, which is this project's recurring shape: the guard covered one spelling and reported clean on
the others. Found 2026-09-27, by the operator asking whether the flags were respected at all.

Patching the three would leave a fourth. So the test below reads the function and asserts that
**every** assignment to the argument namespace inside it goes through `preset`, whatever the
function grows into. A new lever cannot be added as a plain assignment without failing here.
"""
import ast
import pathlib

import pytest

from senbonzakura.cli import _KAGEYOSHI_BUDGET_FLAGS

CLI = pathlib.Path(__file__).resolve().parent.parent / "src" / "senbonzakura" / "cli.py"

#: Assignments inside `_apply_kageyoshi` that are allowed to bypass `preset`, each with the reason.
#: A name here is a claim that the caller's value is honoured some OTHER way, and the claim is
#: checked below rather than trusted.
_GUARDED_BY_HAND = {
    # `if not args.hedge_ds and os.path.isdir(...)`: only set when the caller left it empty, which
    # is the same guarantee `preset` gives, expressed as a condition because it also has to check
    # the directory exists.
    "hedge_ds",
}


@pytest.fixture(scope="module")
def preset_function():
    tree = ast.parse(CLI.read_text(encoding="utf-8"), filename=str(CLI))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_kageyoshi":
            return node
    pytest.fail("_apply_kageyoshi is gone; if the preset moved, move this test with it")


def _namespace_assignments(function):
    """Every `args.<name> = ...` inside the function, as {name: lineno}."""
    found = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "args"):
                found[target.attr] = node.lineno
    return found


def test_every_knob_the_preset_sets_goes_through_the_honour_check(preset_function):
    direct = _namespace_assignments(preset_function)
    offenders = {name: line for name, line in direct.items() if name not in _GUARDED_BY_HAND}
    assert not offenders, (
        "these are assigned straight onto the namespace inside the preset, so a value the caller "
        "typed is overwritten without a word:\n  "
        + "\n  ".join(f"args.{name} at cli.py:{line}" for name, line in sorted(offenders.items()))
        + "\nRoute each through `preset(dest, value)` and add it to _KAGEYOSHI_BUDGET_FLAGS, or "
          "guard it by hand and record the reason in _GUARDED_BY_HAND here.")


def test_every_preset_call_names_a_flag_the_explicit_check_knows_about(preset_function):
    """`preset` can only honour a knob the argv scan looks for, so the two lists must agree.

    A `preset("foo", ...)` whose flag is missing from `_KAGEYOSHI_BUDGET_FLAGS` is silently back to
    the old behaviour: the scan never marks it explicit, so the preset always wins.
    """
    named = set()
    for node in ast.walk(preset_function):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "preset" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            named.add(node.args[0].value)
    unknown = named - set(_KAGEYOSHI_BUDGET_FLAGS)
    assert not unknown, (
        f"the preset sets {sorted(unknown)} but the argv scan does not look for them, so a "
        f"hand-set value is still overridden. Add them to _KAGEYOSHI_BUDGET_FLAGS.")


def test_the_guarded_exceptions_really_are_guarded(preset_function):
    """A name in the exemption list has to be inside an `if`, or the exemption is just a hole."""
    conditional = set()
    for node in ast.walk(preset_function):
        if not isinstance(node, ast.If):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                for target in inner.targets:
                    if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                            and target.value.id == "args"):
                        conditional.add(target.attr)
    unguarded = _GUARDED_BY_HAND & _namespace_assignments(preset_function).keys() - conditional
    assert not unguarded, (
        f"{sorted(unguarded)} are listed as guarded by hand and are assigned unconditionally, so "
        f"the exemption is recording something that is not true")


def test_the_quality_levers_are_among_them():
    """The three that were missing, pinned by name so a revert is loud rather than quiet."""
    for dest in ("search", "per_component", "mlp_off"):
        assert dest in _KAGEYOSHI_BUDGET_FLAGS, (
            f"{dest} was overridden by the preset until 2026-09-27 whatever the caller asked for")
