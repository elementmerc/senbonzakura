# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A recipe that fixes how directions are EXTRACTED, and what happens when the operator disagrees.

WHY THIS IS DIFFERENT FROM THE BAKE PROFILE

`single-pass` fixes a bake profile, which is applied where the bake happens and which nobody can
type on the command line. `searched-one-direction` fixes `--max-directions`, which is a flag the
operator can also set, and the two can therefore contradict each other.

Resolving that quietly is the dangerous option in both directions. Letting the command line win
produces a run whose artefact records `searched-one-direction` and which removed eight directions.
Letting the method win produces a run where the operator asked for eight, was not told no, and
reads the result as though they got them. Either one publishes a number that means something other
than what it says, which is the specific failure this project has already had to withdraw results
over.

So a genuine conflict is refused, loudly, before the model is downloaded. A command line that
merely REPEATS what the method pins is not a conflict and passes without comment.

WHY THE MISSING CELL MATTERS

`single-pass` differs from `searched` in two ways at once: no search, and one direction. Comparing
only those two cannot say which half did the work, and direction count is this project's entire
claim. `searched-one-direction` is our search with their direction budget, which makes the
comparison a 2x2 with a readable column.
"""
from __future__ import annotations

import types

import pytest

from senbonzakura import cli, methods


def _args(**over):
    a = dict(method="searched-one-direction", max_directions=8, track="default", good_ds=None,
             hedge_ds="", clean_ds="", harmless_matched="", text_column=None, hf_token=None,
             load_in_4bit=False)
    a.update(over)
    return types.SimpleNamespace(**a)


def _log(_m):
    pass


def _default_max_directions():
    from senbonzakura.parser import build_parser

    return {a.dest: a.default for a in build_parser()._actions}["max_directions"]


def test_the_missing_cell_exists_as_a_named_arm():
    """A comparison whose arms have no names is two runs, not two arms."""
    assert "searched-one-direction" in methods.CHOICES
    m = methods.get("searched-one-direction")
    assert m.settings["args"] == {"max_directions": 1}
    assert not m.settings.get("bake_profile"), (
        "this arm must differ from the default in the direction budget ALONE. Pinning a bake "
        "profile as well would make it two changes wearing one name, which is the defect the "
        "arm exists to remove.")


def test_it_differs_from_the_default_in_the_direction_budget_and_nothing_else():
    """If anything else differed, the column would not isolate the direction count.

    Compared on SETTINGS rather than on the prose fields, because the settings are what the run
    actually does and the prose is what a reader is told about it.
    """
    a = dict(methods.get("searched").settings)
    b = dict(methods.get("searched-one-direction").settings)
    assert b.pop("args") == {"max_directions": 1}
    assert a == b, f"the arm changes more than the direction budget: {a} against {b}"


def test_a_method_with_nothing_pinned_leaves_the_namespace_alone():
    args = _args(method="searched", max_directions=8)
    cli._apply_method_args(args, _log)
    assert args.max_directions == 8


def test_the_pin_applies_when_the_operator_left_the_default():
    args = _args(max_directions=_default_max_directions())
    cli._apply_method_args(args, _log)
    assert args.max_directions == 1


def test_a_contradicting_flag_is_refused_rather_than_overridden():
    with pytest.raises(SystemExit) as e:
        cli._apply_method_args(_args(max_directions=4), _log)
    msg = str(e.value)
    assert "--max-directions" in msg, "the message must name the flag the operator typed"
    assert "searched-one-direction" in msg, "and the method it fought with"
    assert "4" in msg and "1" in msg, "and both values, or it cannot be acted on"


def test_agreeing_with_the_pin_is_not_a_conflict():
    """Typing the value the method already fixes is a redundancy, not a contradiction."""
    args = _args(max_directions=1)
    cli._apply_method_args(args, _log)
    assert args.max_directions == 1


def test_the_pin_is_announced_when_it_changes_something():
    said = []
    cli._apply_method_args(_args(max_directions=_default_max_directions()), said.append)
    assert any("max-directions" in line for line in said), (
        "a run whose direction budget was changed under it must say so in its own log")


def test_nothing_is_announced_when_the_pin_changes_nothing():
    said = []
    cli._apply_method_args(_args(max_directions=1), said.append)
    assert not said


def test_the_conflict_is_refused_before_the_model_is_constructed(monkeypatch):
    """The order is the point: this fault is visible from the command line alone, and finding it
    after a download costs a rented card an hour for nothing.
    """
    built = []

    def _never(*_a, **_k):
        built.append(True)
        raise AssertionError("the model was constructed before the method was applied")

    monkeypatch.setattr(cli, "Abliterator", _never)
    with pytest.raises(SystemExit, match="--max-directions"):
        cli.run_parsed(_args(max_directions=4), None, [])
    assert not built
