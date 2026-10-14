# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A factory that can hand back a do-nothing stand-in has to be tested on the real branch.

WHAT HAPPENED, 2026-10-01

`test_the_live_display_never_eats_the_markers.py` was written because the live panel captures
stdout, and a panel that swallows a progress marker would hide the one line a reader is watching
for. Its central test attached the panel with `stream=_NotATerminal()`, printed a needle, and
checked the needle survived.

`attach` returns a `NullPanel` for a stream that is not a terminal, and the test two functions
above it asserts that a `NullPanel` does nothing at all. So the subject of the test was an object
whose documented behaviour is to do nothing, and a thing that does nothing cannot eat a marker.
The test could not fail for any implementation. It was green for weeks while `_RichPanel`
captured every byte of stdout, which is the defect it was written to catch.

The file's own header names this shape in its own words, "a guard that cannot see the thing it
guards reports clean", and that is exactly what it then did.

WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT

It is NOT a detector of bad tests. There is no mechanical reading of "this assertion is vacuous",
and a check that claimed to find them would be the same defect one level up.

It is a REGISTRY WITH A SWEEP BEHIND IT, which is the shape this project already uses for the
unbounded-flag class (`test_a_count_cannot_be_negative.py`) and the dead-flag class. The sweep
finds every factory in the package that can return a do-nothing stand-in; the registry makes each
one name the test that drives its REAL branch. Neither half works alone: a registry nobody
compares to the code goes stale, and a sweep with nowhere to record an answer just fails.

So a new degradable factory cannot arrive without somebody writing down which test exercises the
half that does the work. That is the whole of the discipline, and it is the part that was missing.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = (ROOT / "src" / "senbonzakura", ROOT / "checker" / "src" / "senbonzakura_check")
TESTS = ROOT / "tests"

#: Every factory that can hand back a do-nothing stand-in, and the tests that drive the REAL one.
#:
#: The value is test function names this directory must contain. Names rather than a sentence,
#: because a sentence cannot be checked and a name can: the test below looks each one up, so a
#: rename or a deletion fails here rather than quietly leaving the factory unguarded.
#:
#: TWO NAMES AND NOT ONE, because two different things can go wrong and a single test covers one of
#: them. The first is that the factory takes the real branch at all; the second is that the real
#: object has the property the file is about. The 2026-10-01 defect was the first kind, and a
#: registry that accepted only the second would have been satisfied by the file as it stood.
#:
#: `module:function` as the key, because two modules may both have an `attach`.
DEGRADABLE: dict[str, tuple[str, ...]] = {
    # `attach` returns a `_RichPanel` on a terminal and a `NullPanel` everywhere else.
    "livedisplay:attach": (
        # The factory takes the real branch. "The success path of `attach`, which nothing reached
        # before", in its own docstring: this is the test written the day the defect was found.
        "test_attach_returns_a_real_panel_when_the_stream_looks_like_a_terminal",
        # And the real object keeps the property the file is named for. Driven against a panel that
        # actually draws, which is what caught `rich.live.Live` redirecting stdout by default.
        "test_every_smoke_needle_survives_a_real_drawing_panel",
    ),
}

#: Classes that are do-nothing stand-ins by construction and are not reached through a factory, so
#: there is no real branch for a test to take. Each needs a reason, because an unexplained entry
#: here is how the next one hides.
NOT_A_FACTORY: dict[str, str] = {
    "rented:Provider":
        "an interface, not a stand-in. Every method raises nothing and returns nothing because "
        "it is the shape a real provider implements, and no factory in this package returns one",
}


def _is_noop_body(body):
    """Whether a function body does nothing observable: a docstring, `pass`, or a constant return.

    Deliberately generous about what counts as nothing. The point is to find the stand-in half of a
    degradable pair, and a stand-in that returns `False` from `__exit__` or `self` from `__enter__`
    is still a stand-in.
    """
    statements = [s for s in body
                  if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                          and isinstance(s.value.value, str))]
    if not statements:
        return True
    for statement in statements:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, ast.Return):
            value = statement.value
            if value is None:
                continue
            if isinstance(value, ast.Constant) and isinstance(
                    value.value, (type(None), bool, int, float, str)):
                continue
            # `return self` is the identity an `__enter__` owes its caller and changes nothing.
            if isinstance(value, ast.Name) and value.id == "self":
                continue
        return False
    return True


def _modules():
    for package in PACKAGES:
        for py in sorted(package.rglob("*.py")):
            if "vendor" in py.parts:
                continue
            yield py, ast.parse(py.read_text(encoding="utf-8"))


def _standins():
    """`{module stem: {class names that do nothing}}`, found rather than listed.

    A class with fewer than two methods is not considered: one no-op method is an abstract hook or
    a default, and treating every one of them as a stand-in would bury the pairs that matter under
    a list nobody reads.
    """
    out = {}
    for py, tree in _modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = [n for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(methods) >= 2 and all(_is_noop_body(m.body) for m in methods):
                out.setdefault(py.stem, set()).add(node.name)
    return out


def _constructed_in(node):
    """Every class name this function body instantiates and returns."""
    names = set()
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Return) or not isinstance(inner.value, ast.Call):
            continue
        call = inner.value.func
        if isinstance(call, ast.Name):
            names.add(call.id)
        elif isinstance(call, ast.Attribute):
            names.add(call.attr)
    return names


def _degradable_factories():
    """`{"module:function": {class names it may return}}` for every factory with a stand-in branch.

    A factory here is a function that returns an instance of a do-nothing class on one path and an
    instance of something else on another. One branch alone is not a factory in this sense: a
    function that only ever returns the stand-in has no real path to test, and one that never
    returns a stand-in cannot degrade.
    """
    standins = _standins()
    out = {}
    for py, tree in _modules():
        here = standins.get(py.stem, set())
        if not here:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            returned = _constructed_in(node)
            if returned & here and returned - here:
                out[f"{py.stem}:{node.name}"] = returned
    return out


def _test_function_names():
    """Every test function name in this directory, read off the AST rather than imported.

    Reading the source rather than importing keeps this cheap and keeps it honest: a test file that
    cannot be imported on this machine still declares its functions, and the registry is a claim
    about what is written down.
    """
    names = set()
    for py in sorted(TESTS.glob("test_*.py")):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith("test_"):
                names.add(node.name)
    return names


# ── the sweep found something, which is the first thing to check ────────────────────────────────


def test_the_sweep_finds_the_pair_it_was_written_for():
    """Without this the two tests below pass on an empty list, which is how a guard reports clean.

    Named explicitly rather than counted, because the count is what the registry is for and the
    specific pair is what proves the detection works: `attach` returning `NullPanel` on one path and
    `_RichPanel` on another is the shape, and it is the one that shipped the defect.
    """
    standins = _standins()
    assert "NullPanel" in standins.get("livedisplay", set()), (
        "NullPanel is not being recognised as a do-nothing class, so this whole file is looking "
        f"for nothing. Found in livedisplay: {sorted(standins.get('livedisplay', set()))}")
    found = _degradable_factories()
    assert "livedisplay:attach" in found, (
        f"`attach` is not being recognised as a degradable factory: {sorted(found)}")
    assert "_RichPanel" in found["livedisplay:attach"], (
        "the real branch of `attach` is not being seen, so the pair reads as a stand-in only")


def test_every_standin_class_is_either_behind_a_registered_factory_or_explained():
    """A do-nothing class nobody accounts for is the next one of these to go unnoticed."""
    behind = {name for classes in _degradable_factories().values() for name in classes}
    loose = sorted(f"{module}:{name}"
                   for module, classes in _standins().items() for name in classes
                   if name not in behind)
    unexplained = [key for key in loose if key not in NOT_A_FACTORY]
    assert not unexplained, (
        f"these classes do nothing and no factory in the package returns one: {unexplained}\n"
        "  Either they are reached some other way, in which case record them in NOT_A_FACTORY "
        "with the reason, or they are dead and should go.")


# ── the registry and the code agree, in both directions ─────────────────────────────────────────


def test_every_degradable_factory_names_the_test_that_drives_its_real_branch():
    found = sorted(_degradable_factories())
    missing = [key for key in found if key not in DEGRADABLE]
    assert not missing, (
        f"these factories can hand back a do-nothing stand-in and nothing says which test drives "
        f"the half that does the work: {missing}\n"
        "  A test whose subject is the stand-in cannot fail for any implementation, which is how "
        "`test_the_live_display_never_eats_the_markers.py` stayed green while the panel it guarded "
        "captured every byte of stdout.\n"
        "  Write the test, then record it here as `module:function` -> test name.")


def test_the_registry_names_no_factory_the_package_does_not_have():
    """An entry for a factory nobody has is an entry covering the next factory of that name."""
    found = set(_degradable_factories())
    stale = sorted(set(DEGRADABLE) - found)
    assert not stale, (
        f"DEGRADABLE names factories the sweep cannot find: {stale}. Either the factory was "
        f"renamed, in which case rename the key, or it no longer degrades, in which case remove "
        f"the entry rather than leaving it as reassurance.")


@pytest.mark.parametrize(("factory", "test_name"),
                         sorted((key, name) for key, names in DEGRADABLE.items()
                                for name in names))
def test_the_named_test_exists(factory, test_name):
    """A registry pointing at a test nobody has is worse than an empty registry.

    It reads as covered. This is the same reasoning as `test_every_exemption_names_a_flag_that
    _exists` in the unbounded-count file: a record that cannot be resolved is a record that has
    stopped being a record.
    """
    assert test_name in _test_function_names(), (
        f"DEGRADABLE says {factory} is covered by {test_name}, and no test in {TESTS.name}/ has "
        f"that name. Either it was renamed or it was never written.")


def test_every_entry_names_at_least_two_tests():
    """One test covers one of the two failures, and the one it usually covers is not the one."""
    thin = sorted(key for key, names in DEGRADABLE.items() if len(names) < 2)
    assert not thin, (
        f"these name fewer than two tests: {thin}. One has to show the factory takes the real "
        f"branch and one has to show the real object keeps the property in question; the "
        f"2026-10-01 defect was the first kind and the file already had the second.")


def test_the_not_a_factory_table_carries_a_reason_for_each_entry():
    for key, reason in sorted(NOT_A_FACTORY.items()):
        assert len(reason.strip()) > 30, (
            f"{key} is excused with {reason!r}, which is a label rather than a reason. An "
            f"unexplained entry here is how the next stand-in hides.")


# ── the detection would have caught the shape it was written for ────────────────────────────────


def test_the_detection_fires_on_a_reconstruction_of_the_defect(tmp_path):
    """A guard only ever seen passing has not been shown to work. So plant the pair and look.

    The reconstruction is the shape `livedisplay.py` had: a null object whose methods do nothing, a
    real object that does something, and one function choosing between them on a condition a test
    can accidentally take the easy side of.
    """
    source = (
        "class NullThing:\n"
        "    def start(self):\n"
        "        pass\n"
        "\n"
        "    def stop(self):\n"
        "        return False\n"
        "\n"
        "\n"
        "class RealThing:\n"
        "    def start(self):\n"
        "        self.started = True\n"
        "\n"
        "    def stop(self):\n"
        "        self.started = False\n"
        "\n"
        "\n"
        "def attach(stream):\n"
        "    if not stream.isatty():\n"
        "        return NullThing()\n"
        "    return RealThing()\n")
    module = tmp_path / "planted.py"
    module.write_text(source, encoding="utf-8")

    tree = ast.parse(source)
    standins = {node.name for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
                and len([n for n in node.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]) >= 2
                and all(_is_noop_body(n.body) for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))}
    assert standins == {"NullThing"}, standins

    factory = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "attach")
    returned = _constructed_in(factory)
    assert returned == {"NullThing", "RealThing"}
    assert returned & standins and returned - standins, (
        "the planted pair is not recognised as degradable, so the sweep above is not looking for "
        "what this file says it looks for")


def test_a_factory_that_only_returns_the_standin_is_not_a_degradable_pair():
    """The other half of the detection: one branch is not a choice, so there is nothing to test."""
    source = ("class NullThing:\n"
              "    def start(self):\n"
              "        pass\n"
              "\n"
              "    def stop(self):\n"
              "        pass\n"
              "\n"
              "\n"
              "def attach(_stream):\n"
              "    return NullThing()\n")
    tree = ast.parse(source)
    factory = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "attach")
    returned = _constructed_in(factory)
    assert returned == {"NullThing"}
    assert not (returned - {"NullThing"}), "a single-branch factory must not count as degradable"
