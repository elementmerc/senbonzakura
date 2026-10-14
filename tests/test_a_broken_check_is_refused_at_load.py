# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A check file with a malformed rule is refused when the checks load, not when an artefact reaches it.

THE FAILURE THIS EXISTS TO CATCH: a broken `any_missing` in one check file aborted `senbonzakura
check` with an uncaught traceback, partway through a sweep, because `evaluate` raised only when an
artefact reached that rule. The check format is the one thing a stranger is meant to write, so the
engine has to say, before reading anything, which check cannot run and why.

These tests drive the refusal through the real loader and the real CLI, and they pin the drift guard
that keeps the validator's operator list the same as the evaluator's, so a new operator cannot be
added to one and forgotten in the other.
"""
from __future__ import annotations

import copy
import io
import json
import re
from pathlib import Path

import pytest
from senbonzakura_check import cli, registry
from senbonzakura_check.registry import CheckError, _validate_rule, load_checks

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "checker" / "src" / "senbonzakura_check" / "checks"
REGISTRY_SOURCE = ROOT / "checker" / "src" / "senbonzakura_check" / "registry.py"

#: A shipped check whose fields are all valid, used as the base for each broken copy.
BASE = json.loads((SHIPPED / "null-reported-as-zero.json").read_text(encoding="utf-8"))

BROKEN_RULES = [
    ("unknown operator", {"op": "lenient_equals", "path": "x", "value": 1}),
    ("no op at all", {"path": "x"}),
    ("op is not a string", {"op": 3, "path": "x"}),
    ("all_of with no rules", {"op": "all_of", "rules": []}),
    ("any_of with rules that are not a list", {"op": "any_of", "rules": {"op": "always"}}),
    ("not with no rule", {"op": "not"}),
    ("any_missing without a field", {"op": "any_missing", "path": "metrics"}),
    ("any_outside without a field", {"op": "any_outside", "path": "m", "min": 0}),
    ("any_outside with neither bound", {"op": "any_outside", "path": "m", "field": "v"}),
    ("any_entry with no conditions", {"op": "any_entry", "path": "m", "all": []}),
    ("any_entry condition with no field", {"op": "any_entry", "path": "m", "all": [{"equals": 1}]}),
    ("any_entry condition with an unknown predicate",
     {"op": "any_entry", "path": "m", "all": [{"field": "v", "is": "sort_of_zero"}]}),
    ("less_than with no path", {"op": "less_than", "value": 1}),
    ("less_than with a non-numeric value", {"op": "less_than", "path": "p", "value": "1"}),
    ("less_than with a boolean value", {"op": "less_than", "path": "p", "value": True}),
    ("disagrees with one path", {"op": "disagrees", "paths": ["a"]}),
    ("intersects with three paths", {"op": "intersects", "paths": ["a", "b", "c"]}),
    ("differs_in_more_than with a negative limit",
     {"op": "differs_in_more_than", "paths": ["a", "b"], "value": -1}),
    ("differs_by_more_than with a missing tolerance",
     {"op": "differs_by_more_than", "paths": ["a", "b"]}),
    ("equals with no path", {"op": "equals", "value": 1}),
    ("in with no values list", {"op": "in", "path": "p", "values": "x"}),
    ("not_in with no values list", {"op": "not_in", "path": "p"}),
]


@pytest.mark.parametrize(("label", "rule"), BROKEN_RULES, ids=[label for label, _ in BROKEN_RULES])
def test_a_malformed_rule_is_refused_before_any_artefact_is_read(label, rule):
    with pytest.raises(CheckError) as e:
        _validate_rule(rule, "check 'x', `rule`")
    assert "check 'x'" in str(e.value), f"the refusal must name the check: {e.value}"


def test_a_malformed_rule_nested_behind_one_that_short_circuits_is_still_refused():
    """The reason validation walks the tree rather than evaluating it.

    `any_of` stops at the first true sub-rule, so a broken sibling after it is never evaluated, and
    an evaluate-time check would miss it for every artefact that matched the first sibling. The
    validator reads every node, so it cannot be dodged by ordering.
    """
    rule = {"op": "any_of", "rules": [{"op": "always"}, {"op": "any_missing", "path": "m"}]}
    with pytest.raises(CheckError):
        _validate_rule(rule, "check 'x', `rule`")


@pytest.mark.parametrize("rule", [
    {"op": "always"},
    {"op": "never"},
    {"op": "present", "path": "a"},
    {"op": "equals", "path": "a", "value": None},
    {"op": "in", "path": "a", "values": [1, 2]},
    {"op": "any_missing", "path": "m", "field": "f"},
    {"op": "any_outside", "path": "m", "field": "f", "min": 0.0},
    {"op": "any_entry", "path": "m", "all": [{"field": "v", "is": "zero"}]},
    {"op": "less_than", "path": "p", "value": 0.5},
    {"op": "differs_by_more_than", "paths": ["a", "b"], "value": 0.01},
    {"op": "differs_in_more_than", "paths": ["a", "b"], "value": 0},
    {"op": "all_of", "rules": [{"op": "always"}, {"op": "not", "rule": {"op": "never"}}]},
])
def test_a_well_formed_rule_is_accepted(rule):
    _validate_rule(rule, "check 'x', `rule`")


def test_every_shipped_check_is_accepted_at_load():
    """The shipped set is the proof that the validator does not refuse what it should accept."""
    assert len(load_checks()) == len(list(SHIPPED.glob("*.json")))


def _write_check_dir(tmp_path, check):
    directory = tmp_path / "checks"
    directory.mkdir()
    (directory / "broken.json").write_text(json.dumps(check), encoding="utf-8")
    return directory


def test_a_check_file_with_a_broken_rule_fails_the_load_and_names_the_check(tmp_path):
    check = copy.deepcopy(BASE)
    check["id"] = "broken-any-missing"
    check["rule"] = {"op": "any_missing"}
    directory = _write_check_dir(tmp_path, check)
    with pytest.raises(CheckError) as e:
        load_checks(directory)
    assert "broken-any-missing" in str(e.value)
    assert "`rule`" in str(e.value)


def test_a_broken_applies_to_is_refused_too(tmp_path):
    check = copy.deepcopy(BASE)
    check["id"] = "broken-applies"
    check["applies_to"] = {"op": "not"}
    directory = _write_check_dir(tmp_path, check)
    with pytest.raises(CheckError) as e:
        load_checks(directory)
    assert "`applies_to`" in str(e.value)


def test_the_cli_refuses_a_broken_check_set_with_exit_two_and_no_traceback(tmp_path, monkeypatch):
    """The end-to-end defect. Before the validator, this run raised out of `main` with a traceback.

    Exit 2 is "could not check", so a CI step reads a broken install as a failure, never as a clean
    run, and the reader is told which check and which rule to fix.
    """
    check = copy.deepcopy(BASE)
    check["id"] = "broken-any-missing"
    check["rule"] = {"op": "any_missing"}
    monkeypatch.setattr(registry, "CHECKS_DIR", _write_check_dir(tmp_path, check))
    artefact = tmp_path / "result.json"
    artefact.write_text(json.dumps({"metric": "x", "value": 0}), encoding="utf-8")
    out = io.StringIO()
    code = cli.main([str(artefact)], out=out)
    assert code == 2
    assert "nothing was checked" in out.getvalue()
    assert "broken-any-missing" in out.getvalue()


def test_the_validator_knows_every_operator_the_evaluator_does():
    """The drift guard. Every `op == "x"` and `op in (...)` in the evaluator must be known here.

    Read from the source rather than typed, so adding an operator to `evaluate` without teaching the
    validator about it fails this test rather than letting a broken rule of the new kind through.
    """
    source = REGISTRY_SOURCE.read_text(encoding="utf-8")
    evaluated = set(re.findall(r'op == "([a-z_]+)"', source))
    for group in re.findall(r"op in \(([^)]*)\)", source):
        evaluated |= set(re.findall(r'"([a-z_]+)"', group))
    unknown = sorted(evaluated - registry._KNOWN_OPS)
    assert not unknown, f"the evaluator handles operators the validator refuses: {unknown}"
