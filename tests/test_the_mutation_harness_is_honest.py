# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The mutation harness itself, which is the one instrument that must be able to fail.

WHY THIS EXISTS

`tools/research/mutate.py` measured a 29% miss rate across twenty-one guards, and that figure is
what every argument about this project's test quality now rests on. It lived in `/tmp` on one
machine for four days. These tests are the price of bringing it into the tree: a file whose job is
finding guards that cannot fail has to be held to its own standard, and the original could not
fail. It printed its findings and returned nothing, so `echo $?` said success whatever it found.

WHAT IS CHECKED HERE, AND WHAT DELIBERATELY IS NOT

These tests never run a real mutation. Running one edits a tracked source file in place, and a
test suite that edits the tree it is testing is a worse idea than anything the harness is looking
for. So the real cases are checked statically, and the RUNNER is exercised against a throwaway
tree with a throwaway test, where breaking something costs nothing.

The static half matters more than it looks. The harness works by matching exact source text, so
every case is one refactor away from silently not running, and "did not run" must never look like
"passed". That is the same failure this project already paid for when one unresolvable pin made a
vulnerability scanner report NOT SCANNED for a whole file, and a real advisory was
indistinguishable from a scan that never happened.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _the_harness():
    """`tools/research` is not a package, so its directory goes on the path before the import.

    Through a function rather than a bare `sys.path` line followed by an import, which is the
    pattern `test_a_wheel_can_say_which_commit_built_it.py` already uses for `tools/ci`: an import
    that has to come after a statement is an import the sorter will keep trying to move.
    """
    sys.path.insert(0, str(ROOT / "tools" / "research"))
    import mutate

    return mutate


mutate = _the_harness()


def test_the_measurement_is_all_twenty_one_cases_in_two_rounds():
    """The 29% figure is 6 of 21, so a quietly trimmed roster makes the figure uncomparable.

    The split is load-bearing as well as the total. Round one tests helpers and round two tests
    whether each helper is wired in, and FIVE of the six misses were in round two. Merging the
    rounds or dropping one would delete the finding rather than the cases.
    """
    assert len(mutate.ROUND_ONE) == 11, "round one tested eleven helpers"
    assert len(mutate.ROUND_TWO) == 10, "round two tested ten call sites"
    assert len(mutate.ROUND_ONE) + len(mutate.ROUND_TWO) == 21


def test_every_case_is_well_formed():
    for which, cases in mutate.ROUNDS.items():
        for case in cases:
            assert len(case) == 5, f"round {which}: a case is (name, path, old, new, test)"
            name, path, old, new, test = case
            assert name and name.strip(), f"round {which}: a case with no name cannot be reported"
            assert old, f"{name}: an empty `old` would match everywhere and mutate the first byte"
            assert old != new, f"{name}: a mutation that changes nothing proves nothing"
            assert path and not path.startswith("/"), f"{name}: {path} must be repo-relative"
            assert test, f"{name}: a case with no test cannot notice anything"


def test_every_case_names_a_file_and_a_test_that_exist():
    """A case pointing at a deleted file is a case that cannot run, and silence about it is the
    defect. The harness reports that as DRIFT at run time; this catches it at suite time, which is
    sooner and cheaper.
    """
    missing = []
    for cases in mutate.ROUNDS.values():
        for name, path, _old, _new, test in cases:
            if not (ROOT / path).exists():
                missing.append(f"{name}: mutates {path}, which does not exist")
            # A test id may carry ::node, and only the file half is a path.
            if not (ROOT / test.split("::", 1)[0]).exists():
                missing.append(f"{name}: runs {test}, whose file does not exist")
    assert not missing, "\n".join(missing)


def test_the_harness_points_at_this_checkout_rather_than_one_machine():
    """The originals hardcoded `/home/daniel/the-factory/senbonzakura`, so they ran on exactly one
    box. That is half the reason the evidence nearly evaporated.
    """
    assert mutate.ROOT == ROOT
    assert (mutate.ROOT / "pyproject.toml").is_file()


def test_a_mutated_run_cannot_leave_bytecode_behind():
    """`-B`, because the restore check reads the source and a stale `.pyc` is invisible to it.

    CPython decides a cached `.pyc` is still good from the source's size and its mtime in whole
    seconds. A mutation that does not change the file's length, such as swapping two blocks, and
    which is restored inside the same second it was written, leaves bytecode compiled from the
    mutant that the next process imports in preference to the file on disk.

    Observed on 2026-10-01 rather than reasoned about: a block swap in `interactive.py` scored
    SURVIVED, and the test written afterwards to catch it then failed against the CLEAN tree while
    `inspect.getsource` printed the correct code. Two wrong answers in a row, both looking
    ordinary, from one cache. `-B` means there is never a `.pyc` to go stale.
    """
    argv = mutate.pytest_argv("tests/test_nothing.py")
    assert "-B" in argv, (
        "the harness can write bytecode from a mutated source, which outlives the restore and "
        "makes the next case's answer unreliable")
    assert argv.index("-B") < argv.index("-m"), (
        f"-B has to reach the interpreter rather than pytest, so it must precede -m: {argv}")


def test_the_literals_still_match_the_tree_or_the_drift_is_declared():
    """Every case's `old` is still present, except the one the module docstring says has drifted.

    This is the test that keeps the harness honest over time. A refactor that renames a line makes
    a case unrunnable, and without this the only symptom is one fewer line of output in a report
    nobody diffs.

    The exemption is NAMED, singular, and explained in the module docstring: the line that case
    mutates was itself found to be wrong and was fixed on the way to 0.4.1. Repairing the case
    means choosing a new literal and saying so there, not here.
    """
    declared_drift = {"measure points at the stage log"}
    drifted = set()
    for cases in mutate.ROUNDS.values():
        for name, path, old, _new, _test in cases:
            target = ROOT / path
            if target.exists() and old not in target.read_text():
                drifted.add(name)

    unexpected = sorted(drifted - declared_drift)
    assert not unexpected, (
        "these cases no longer match the tree and are therefore not running:\n  "
        + "\n  ".join(unexpected)
        + "\nEither repair the literal or add it to the declared drift with a reason in "
          "tools/research/mutate.py's docstring.")

    healed = sorted(declared_drift - drifted)
    assert not healed, (
        "these cases are declared as drifted and now match again:\n  " + "\n  ".join(healed)
        + "\nRemove them from the exemption; a stale exemption hides the next real drift.")


@pytest.fixture
def throwaway(tmp_path, monkeypatch):
    """A tiny tree with one source file and one test, so the runner can be exercised for real.

    Nothing here touches the repository. `mutate.ROOT` is repointed for the duration, which is why
    this is a fixture rather than inline setup: the restore has to happen even on a failure.
    """
    (tmp_path / "thing.py").write_text("VALUE = 1\n")
    (tmp_path / "test_thing.py").write_text(
        "import pathlib\n"
        "def test_value():\n"
        "    assert 'VALUE = 1' in pathlib.Path(__file__).with_name('thing.py').read_text()\n")
    (tmp_path / "test_blind.py").write_text("def test_nothing():\n    assert True\n")

    # The real invocation passes `--no-cov`, which is a pytest-cov option and a USAGE ERROR where
    # that plugin is absent. The throwaway tree has no coverage gate to suppress, so it does not
    # need the flag, and depending on it here would make these tests pass or fail on which
    # plugins the developer's environment happens to carry rather than on the harness's logic.
    # That substitution is the reason `pytest_argv` is a function in the first place.
    monkeypatch.setattr(
        mutate, "pytest_argv",
        lambda test: [sys.executable, "-m", "pytest", test, "-q", "-p", "no:cacheprovider", "-x"])
    return tmp_path


def test_a_guard_that_notices_is_reported_as_noticed(throwaway, monkeypatch):
    monkeypatch.setattr(mutate, "ROOT", throwaway)
    outcome, _detail = mutate.run_case(
        ("a watched value", "thing.py", "VALUE = 1", "VALUE = 2", "test_thing.py"))
    assert outcome == mutate.NOTICED


def test_a_guard_that_does_not_notice_is_reported_as_missed(throwaway, monkeypatch):
    """The whole point of the instrument: a test that passes with the line broken."""
    monkeypatch.setattr(mutate, "ROOT", throwaway)
    outcome, detail = mutate.run_case(
        ("an unwatched value", "thing.py", "VALUE = 1", "VALUE = 2", "test_blind.py"))
    assert outcome == mutate.MISSED
    assert "does not guard it" in detail


def test_a_literal_that_is_gone_is_drift_and_not_a_pass(throwaway, monkeypatch):
    """DRIFT is a third outcome rather than a quiet success, which is the trichotomy the
    supply-chain plan requires of every gate: PASS, FAIL, and DID NOT RUN.
    """
    monkeypatch.setattr(mutate, "ROOT", throwaway)
    outcome, _detail = mutate.run_case(
        ("a vanished line", "thing.py", "NOT IN THE FILE", "replacement", "test_thing.py"))
    assert outcome == mutate.DRIFT

    gone, _detail = mutate.run_case(
        ("a vanished file", "no_such_file.py", "anything", "else", "test_thing.py"))
    assert gone == mutate.DRIFT


def test_the_file_is_put_back_after_every_case(throwaway, monkeypatch):
    """The one thing this script must never do is leave the tree edited."""
    monkeypatch.setattr(mutate, "ROOT", throwaway)
    before = (throwaway / "thing.py").read_text()
    for test_file in ("test_thing.py", "test_blind.py"):
        mutate.run_case(("a watched value", "thing.py", "VALUE = 1", "VALUE = 2", test_file))
        assert (throwaway / "thing.py").read_text() == before
    assert not mutate._IN_FLIGHT, "a case left a file registered as mid-edit"


def test_the_runner_exits_non_zero_when_a_guard_misses(throwaway, monkeypatch):
    """THE DEFECT BOTH ORIGINALS HAD.

    `/tmp/mutate.py` and `/tmp/mutate2.py` printed MISSED and then returned None, so the process
    exit status was 0. An instrument built to find checks that cannot fail could not itself fail.
    """
    monkeypatch.setattr(mutate, "ROOT", throwaway)
    monkeypatch.setattr(mutate, "ROUNDS", {
        1: [("an unwatched value", "thing.py", "VALUE = 1", "VALUE = 2", "test_blind.py")]})
    monkeypatch.setattr(mutate, "tree_is_clean", lambda _root: (True, ""))
    assert mutate.main(["--round", "1"]) == 1


def test_the_runner_exits_non_zero_on_drift_as_well_as_on_a_miss():
    """A case that did not run is not a case that passed, and the exit status has to agree."""
    assert mutate.MISSED != mutate.NOTICED
    assert mutate.DRIFT != mutate.NOTICED


@pytest.mark.parametrize(
    ("code", "outcome"),
    [
        (1, mutate.NOTICED),   # a test failed. The ONLY code that means the guard noticed
        (0, mutate.MISSED),    # everything passed with the line broken
        (2, mutate.DRIFT),     # interrupted
        (3, mutate.DRIFT),     # pytest internal error
        (4, mutate.DRIFT),     # usage error: a bad flag, a missing plugin
        (5, mutate.DRIFT),     # NO TESTS COLLECTED. The one that would have bitten
        (9, mutate.DRIFT),     # anything unrecognised is not evidence either
    ])
def test_only_a_failing_test_counts_as_the_guard_noticing(code, outcome):
    """THE DEFECT THE TEST ABOVE FOUND ON ITS FIRST RUN, and it was worse than the one it was for.

    Both originals used `noticed = r.returncode != 0`, which scored FIVE of pytest's codes as the
    guard working when only exit 1 means a test failed. A misspelled test path collects nothing,
    exits 5, and was reported NOTICED.

    This project already holds that lesson from the other direction: "no tests ran in 0.00s" has
    verified nothing, and one path matching no file takes every other path in the same command
    with it. The instrument built to find unverified guards was reading one.
    """
    got, detail = mutate._read_pytest_status(code, "tests/test_example.py")
    assert got == outcome, f"exit {code} should be {outcome}, got {got} ({detail})"
    if outcome == mutate.DRIFT:
        assert "did not run" in detail, "a drifted case must say it proved nothing"


def test_the_collected_nothing_code_is_called_out_by_name():
    """Exit 5 gets its own sentence, because a reader who sees DRIFT needs to know which of the
    four did-not-run reasons they are looking at before they can fix it.
    """
    _got, detail = mutate._read_pytest_status(5, "tests/test_misspelled.py")
    assert "collected no tests" in detail
    assert "tests/test_misspelled.py" in detail, "name the path, since the path is the usual cause"


def test_the_runner_exits_zero_only_when_every_guard_noticed(throwaway, monkeypatch):
    monkeypatch.setattr(mutate, "ROOT", throwaway)
    monkeypatch.setattr(mutate, "ROUNDS", {
        1: [("a watched value", "thing.py", "VALUE = 1", "VALUE = 2", "test_thing.py")]})
    monkeypatch.setattr(mutate, "tree_is_clean", lambda _root: (True, ""))
    assert mutate.main(["--round", "1"]) == 0


def test_a_dirty_tree_is_refused_rather_than_mutated(monkeypatch, capsys):
    """It edits tracked files in place. On a tree that already has changes, a restore that goes
    wrong is indistinguishable from work in progress, so the honest move is to decline.
    """
    monkeypatch.setattr(mutate, "tree_is_clean", lambda _root: (False, " M src/thing.py"))
    assert mutate.main([]) == 2
    assert "REFUSING TO RUN" in capsys.readouterr().err


def test_an_unknown_git_state_is_also_refused(monkeypatch):
    """`tree_is_clean` returning False covers "git could not be run" as well as "dirty", because a
    tree whose state is unknown is not a tree to start editing.
    """
    monkeypatch.setattr(mutate, "tree_is_clean",
                        lambda _root: (False, "git status could not be run"))
    assert mutate.main([]) == 2


def _every_case_count():
    """How many cases `--list` should report, DERIVED from the rosters rather than written down.

    THIS WAS THE LITERAL `21` AND A THIRD ROUND BROKE IT, 2026-10-02. `ROUND_THREE` was added for
    the night five lanes committed into one tree, so `--list` correctly reported 27 while two
    assertions here still demanded 21, and CI went red over the harness having grown.

    The `21` elsewhere in this file is a DIFFERENT number and stays a literal on purpose: it is
    provenance, because the recorded 29% miss rate is 6 of 21, and a figure whose denominator moves
    silently is uncomparable. So the historical roster keeps its literal and the live output gets
    derived, which is the distinction the two assertions had collapsed.
    """
    return sum(len(cases) for cases in mutate.ROUNDS.values())


def test_listing_changes_nothing_and_needs_no_clean_tree(capsys):
    """`--list` is the only mode safe to run while working, so it must not consult the tree."""
    assert mutate.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert f"{_every_case_count()} case(s)" in out
    assert "Nothing was changed." in out


def test_the_harness_runs_as_a_script_and_lists_its_cases():
    """The runner is reached through `python tools/research/mutate.py`, so the entry point is part
    of the contract, not just the importable functions.
    """
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "research" / "mutate.py"), "--list"],
        capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert f"{_every_case_count()} case(s)" in done.stdout


def test_the_provenance_is_recorded_in_the_file_rather_than_in_a_handoff():
    """The figure this file carries is only usable if the file says where it came from. A number
    with no provenance is the thing this project refuses to publish about anybody else.
    """
    text = (ROOT / "tools" / "research" / "mutate.py").read_text()
    for needed in ("2026-09-27", "/tmp/mutate.py", "/tmp/mutate2.py", "29%", "21 guards"):
        assert needed in text, f"the docstring no longer records {needed!r}"
