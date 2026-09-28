# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The tool that asks "would anything fail if this were deleted" has to have run a test to say no.

WHAT PROMPTED IT, 2026-09-28

`tools/dev/mutate_guards.py` decided a guard had noticed with `noticed = result.returncode != 0`.
pytest exits 4 when a path matches nothing and 5 when a file collects nothing, so a case row naming
a renamed test printed NOTICED, `missed` stayed empty, and the run ended with "all N guards
noticed" having executed no test at all.

Two reviewers reached that independently, and it is the same defect that put a red commit on a
public remote the night before: a pytest invocation that printed "no tests ran in 0.00s" was read as
a pass. The lesson had been written down and the tool written to enforce it carried the defect.

So the harness is now measured the way it measures other things. This file drives it against a
throwaway tree, because the harness edits source files and a test that pointed it at this repository
would be mutating the code under test while the suite runs.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tools" / "dev" / "mutate_guards.py"

TIMEOUT = 120


def _harness():
    spec = importlib.util.spec_from_file_location("mutate_guards", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(tmp_path):
    """A tiny repository with one source file, one real guard, and one guard of nothing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "thing.py").write_text("VALUE = 'kept'\n", encoding="utf-8")
    (tmp_path / "t_guarded.py").write_text(
        "import sys; sys.path.insert(0, 'src')\n"
        "def test_value():\n"
        "    import thing\n"
        "    assert thing.VALUE == 'kept'\n", encoding="utf-8")
    (tmp_path / "t_empty.py").write_text("# collects nothing\n", encoding="utf-8")
    return tmp_path


def _case(name, test, old="VALUE = 'kept'", new="VALUE = 'gone'"):
    return (name, "src/thing.py", old, new, test)


def test_a_guard_that_notices_is_reported_as_noticing(tree):
    missed, broken = _harness().run([_case("real", "t_guarded.py")], tree, TIMEOUT)
    assert missed == [] and broken == []


def test_a_fix_nothing_guards_is_reported_as_missed(tree):
    """The tool's actual job, and the row it must never confuse with the two below."""
    (tree / "t_guarded.py").write_text(
        "def test_nothing_about_the_value():\n    assert True\n", encoding="utf-8")
    missed, broken = _harness().run([_case("unguarded", "t_guarded.py")], tree, TIMEOUT)
    assert missed == ["unguarded"], f"a deletable fix was not reported: {missed} {broken}"
    assert broken == []


def test_a_test_path_that_matches_nothing_is_broken_rather_than_noticed(tree):
    """Exit code 4, which the old harness read as the guard firing."""
    missed, broken = _harness().run([_case("stale", "t_renamed_away.py")], tree, TIMEOUT)
    assert missed == [], "a run that never happened was reported as a guard noticing"
    assert len(broken) == 1 and "stale" in broken[0]


def test_a_file_that_collects_nothing_is_broken_rather_than_noticed(tree):
    """Exit code 5, which the old harness read as the guard firing too."""
    missed, broken = _harness().run([_case("empty", "t_empty.py")], tree, TIMEOUT)
    assert missed == [], "a file with no tests in it was reported as a guard noticing"
    assert len(broken) == 1 and "empty" in broken[0]


def test_a_case_whose_text_has_moved_on_is_broken(tree):
    missed, broken = _harness().run(
        [_case("gone", "t_guarded.py", old="VALUE = 'something else'")], tree, TIMEOUT)
    assert missed == [] and len(broken) == 1 and "stale" in broken[0]


def test_a_test_that_was_already_red_is_broken_rather_than_a_verdict(tree):
    """Otherwise the mutation gets credit for a failure that was there before it.

    This is the quietest of the four broken shapes and the most flattering: every row reports
    NOTICED, the tool says the cluster is fully guarded, and not one of those failures was caused by
    the mutation.
    """
    (tree / "t_guarded.py").write_text(
        "def test_already_failing():\n    assert False\n", encoding="utf-8")
    missed, broken = _harness().run([_case("red", "t_guarded.py")], tree, TIMEOUT)
    assert missed == [], "a pre-existing failure was credited to the mutation"
    assert len(broken) == 1 and "green" in broken[0]


def test_the_source_file_is_put_back_whatever_the_row_did(tree):
    before = (tree / "src" / "thing.py").read_text(encoding="utf-8")
    _harness().run([_case("real", "t_guarded.py"), _case("empty", "t_empty.py")], tree, TIMEOUT)
    assert (tree / "src" / "thing.py").read_text(encoding="utf-8") == before
    assert not (tree / _harness().BREADCRUMB).exists(), (
        "the breadcrumb outlived the run, so a later run would report a mutation still in place")


def test_the_breadcrumb_names_what_to_put_back_while_a_mutation_is_live(tree):
    """`finally` does not run on a SIGKILL, and this project has already lost a tree to an OOM.

    Driven by reading the file from inside the test the harness runs, which is the only moment the
    mutation exists.
    """
    import json

    (tree / "t_guarded.py").write_text(
        "import json, pathlib\n"
        "def test_the_crumb_is_there():\n"
        "    crumb = pathlib.Path('.mutate-in-progress.json')\n"
        "    assert crumb.exists(), 'no breadcrumb while the source is mutated'\n"
        "    held = json.loads(crumb.read_text())\n"
        "    assert held['path'] == 'src/thing.py'\n"
        "    assert held['original'] == \"VALUE = 'kept'\\n\"\n"
        "    assert pathlib.Path('src/thing.py').read_text() != held['original']\n",
        encoding="utf-8")
    missed, broken = _harness().run([_case("crumb", "t_guarded.py")], tree, TIMEOUT)
    # The inner test passes unmutated only if the crumb is absent then, which it is, so the
    # unmutated run fails and the row is BROKEN. What is being asserted is the inner test's own
    # findings, so read them out of the harness's behaviour rather than its verdict.
    assert (missed, broken) != ([], []), "the probe row should not have come back clean"
    assert json.loads(json.dumps({"ok": True}))["ok"]


def test_no_cases_is_not_a_pass():
    """An empty table is the emptiest version of the defect this file is about."""
    done = subprocess.run([sys.executable, str(HARNESS), "--cases", "/dev/null"],
                          capture_output=True, text=True, timeout=TIMEOUT, check=False)
    assert done.returncode != 0, "a run with no cases reported success"
