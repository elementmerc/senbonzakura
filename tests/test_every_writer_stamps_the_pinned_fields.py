# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The automated half of a finding: four writers omitted the same five fields for the same reason.

THE DEFECT. `baseline.PINNED` names the fields that decide whether two measurements may be
compared, and `comparability` reports an absent one as a mismatch rather than skipping it, so a
figure missing any of them is comparable with nothing. On 2026-09-25 a self-review pass measured
the five writers and found `input_digest`, `partition`, `prompt_format`, `tool_version` and
`precision` absent from four of them, identically. Not four decisions: one omission repeated,
because each writer would have had to invent the same five answers on its own and four never got
round to it.

WHY THIS IS A STRUCTURAL TEST AND NOT A BEHAVIOURAL ONE. Three of the five writers cannot be run
without loading a model, so a test that exercises them is a test that does not run in CI, which is
exactly where this class of defect survived. What can be checked cheaply and completely is that
every `measurement.stamp` call in the package supplies the pinned fields, and that no sixth writer
can be added without them. A writer added next month is the case this owns.

It deliberately accepts either spelling, the shared `stamps.pinned(...)` or all five by hand, and
it fails loudly naming the file and line when a call supplies neither. A guard that insisted on one
spelling would report clean on the other, which is this project's most-repeated defect shape.
"""
import ast
import pathlib

from senbonzakura import baseline

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "senbonzakura"

#: The three `PINNED` fields that are NOT the writer's to supply, and where each comes from.
#:
#: `model` is read from the artefact's top level by `_pinned_from`; `metric` and `estimator` are the
#: stamp's own positional arguments and cannot be absent. The other five are what a writer has to
#: know about its own run, and are the five this test is about.
NOT_THE_WRITERS_TO_SUPPLY = {"model", "metric", "estimator"}

REQUIRED = set(baseline.PINNED) - NOT_THE_WRITERS_TO_SUPPLY


def _stamp_calls():
    """Every `measurement.stamp(...)` call in the package, with where it is.

    THE RECEIVER IS CHECKED, not just the method name. `marker.stamp` writes a checkpoint's
    provenance marker and has nothing to do with `baseline.PINNED`; matching on `.stamp` alone
    reported it as a writer missing five fields it was never going to carry. A guard that fires on
    the wrong thing gets an exception added to it, and the exception is where the next real one
    hides.
    """
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "stamp"
                    and isinstance(func.value, ast.Name) and func.value.id == "measurement"):
                yield path.name, node


def _fields_supplied(call):
    """What the call names, counting a `**stamps.pinned(...)` as supplying all five.

    A `**name` of anything else is counted as unknown rather than as satisfying the requirement:
    the writers that pass a dict through a helper pass one built by `stamps.pinned`, and accepting
    any `**` would turn this guard into a check that a writer passes something.
    """
    named = {kw.arg for kw in call.keywords if kw.arg}
    for kw in call.keywords:
        if kw.arg is not None:
            continue
        unpacked = kw.value
        if isinstance(unpacked, ast.Call) and isinstance(unpacked.func, ast.Attribute) \
                and unpacked.func.attr == "pinned":
            named |= REQUIRED
        elif isinstance(unpacked, ast.Name):
            # A local, as `score` and `margin` do: they build it once and stamp twice. The helper
            # that builds it is checked by the call site test below rather than here.
            named |= {"_via_local"}
    return named


def test_every_stamp_call_supplies_the_pinned_fields():
    missing_by_call = {}
    for filename, call in _stamp_calls():
        supplied = _fields_supplied(call)
        if "_via_local" in supplied:
            continue
        gap = REQUIRED - supplied
        if gap:
            missing_by_call[f"{filename}:{call.lineno}"] = sorted(gap)
    assert not missing_by_call, (
        f"these `measurement.stamp` calls do not supply every pinned field: {missing_by_call}.\n"
        f"`baseline.comparability` reports an absent pinned field as a mismatch, so a figure "
        f"missing any of them can never be compared with another one and the gate refuses it "
        f"without saying why. Pass `**stamps.pinned(...)`, which derives all five, or name them "
        f"explicitly if this writer genuinely knows better than the derivation does.")


def test_the_five_writers_are_all_still_here():
    """The guard above is only worth its runtime while it covers the writers it was built for."""
    files = {filename for filename, _ in _stamp_calls()}
    # Six, not the five the manual sweep counted. `cli.py` stamps the separation figure and was
    # missed by the sweep; the guard above found it on its first run, missing two pinned fields.
    expected = {"score.py", "margin.py", "drift.py", "capability.py", "coherence.py", "cli.py"}
    assert expected <= files, (
        f"{sorted(expected - files)} no longer stamps a measurement. Either the writer moved, in "
        f"which case update this list, or it stopped recording its figure's identity, which is "
        f"the defect this file exists to catch.")


def test_the_helpers_that_stamp_twice_pass_the_fields_through():
    """`score` and `margin` build the fields once and stamp two estimators with them."""
    import inspect

    from senbonzakura import margin, score
    for helper in (score._stamp_refusal, margin._stamp_compass):
        source = inspect.getsource(helper)
        assert "pinned or {}" in source, (
            f"{helper.__qualname__} no longer takes the pinned fields from its caller. Both "
            f"estimators it stamps are the same measurement under two rulers, so both need the "
            f"same identity or only one of them is gateable.")
        assert source.count("**fields") >= 2, (
            f"{helper.__qualname__} stamps more than one estimator and does not pass the pinned "
            f"fields to all of them. A figure and its control that disagree about their own "
            f"provenance are two measurements, not a comparison.")
