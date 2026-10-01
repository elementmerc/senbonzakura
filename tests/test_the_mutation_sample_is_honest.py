# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The sampled mutation report must not be able to call a run that measured nothing clean.

Q-51 option B is a scheduled job that nobody watches, which is the worst place for a gate that
can report a false pass. These assertions are over the two properties that make its output
trustworthy: the ratchet compares like with like, and a partial run is refused rather than
summarised.

WHAT PROMPTED EACH ONE, because both were real

The completeness floor exists because this job had the defect it exists to find. While the
ratchet was being tested on 2026-10-01, a deliberately weakened test file was left syntactically
invalid. mutmut collected no tests, printed "Failed to collect list of tests" and exited 1. The
tool then read the thirteen stale verdicts sitting in the previous run's `.meta` file, found no
regression among them, printed "RATCHET HELD" and exited 0. mutmut records a null exit code for
a mutant it has not run, so a stale metadata file and a fresh one cannot be told apart by
content; only the count can tell them apart.

The ratchet's shape exists because a ratchet on a sampled percentage fires on noise. At a true
kill rate of 0.8 and a sample of 100 the standard error is about four percentage points, so
"must not fall below last night" would fail about half the level nights, and a gate that cries
wolf gets disabled within a fortnight. So the ratchet is on named mutants, where resampling
cannot change the answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "ci"))

import mutation_sample as ms  # noqa: E402

KEY = "senbonzakura.say.x_lines__mutmut_3"


def _ledger(verdict: str, function_hash: str = "abc123") -> dict:
    return {"mutants": {KEY: {"verdict": verdict, "function_hash": function_hash,
                              "last_seen": "2026-10-01"}}}


def test_a_killed_mutant_that_now_survives_is_a_regression():
    """The one thing the ratchet is for. Nothing but a weakened test produces it."""
    found = ms.ratchet(_ledger(ms.KILLED), {KEY: ms.SURVIVED}, {KEY: "abc123"})
    assert len(found) == 1, found
    assert KEY in found[0]
    assert "KILLED" in found[0] and "SURVIVES" in found[0]


def test_a_newly_discovered_survivor_is_not_a_regression():
    """A sample reaching new ground is the job working, not the tree getting worse.

    If discoveries failed the job, the first night would fail with hundreds of them and the job
    would be switched off before it ever reported anything useful.
    """
    assert ms.ratchet({"mutants": {}}, {KEY: ms.SURVIVED}, {KEY: "abc123"}) == []


def test_a_survivor_already_known_to_survive_is_not_a_regression():
    """Otherwise every known survivor would fail the job every night, for ever."""
    assert ms.ratchet(_ledger(ms.SURVIVED), {KEY: ms.SURVIVED}, {KEY: "abc123"}) == []


def test_a_regression_is_not_claimed_when_the_code_changed_underneath_it():
    """What keeps the ratchet free of sampling and of rewriting artefacts.

    A mutant named `x_lines__mutmut_3` after the function is rewritten is not the same mutant;
    comparing the two would report a regression every time anybody edited a module, which is a
    gate that punishes work.
    """
    found = ms.ratchet(_ledger(ms.KILLED, "OLD_HASH"), {KEY: ms.SURVIVED}, {KEY: "NEW_HASH"})
    assert found == [], f"a rewritten function was reported as a test regression: {found}"


def test_a_killed_mutant_that_stays_killed_is_silent():
    """The positive control. A gate only ever seen firing has not been shown to pass."""
    assert ms.ratchet(_ledger(ms.KILLED), {KEY: ms.KILLED}, {KEY: "abc123"}) == []


def test_no_tests_is_its_own_outcome_and_never_a_survivor():
    """Counting an unreachable mutant as surviving would inflate the miss rate with mutants
    nothing even attempted, which is the flattering direction for the tests and the misleading
    one for the reader.
    """
    assert ms.EXIT_VERDICT[33] == ms.NO_TESTS
    assert ms.NO_TESTS != ms.SURVIVED
    assert ms.ratchet(_ledger(ms.KILLED), {KEY: ms.NO_TESTS}, {KEY: "abc123"}) == [], (
        "a mutant that no test covers was reported as a regression")


def test_the_three_exit_codes_that_mean_caught_or_not_are_mapped():
    """Established by reading mutmut's source and confirmed against a real run of 153 mutants.

    0 means the suite passed with the mutant in place, so the mutant survived. 1 means a test
    failed, so it was caught. 33 means nothing covers the line. 37 means a type checker caught
    it, which is still caught, by a different instrument.
    """
    assert ms.EXIT_VERDICT[0] == ms.SURVIVED
    assert ms.EXIT_VERDICT[1] == ms.KILLED
    assert ms.EXIT_VERDICT[37] == ms.KILLED
    assert None not in ms.EXIT_VERDICT, (
        "a null exit code means mutmut never ran that mutant. Mapping it to any verdict would "
        "turn a stale metadata file into a reported result.")


def test_the_interval_is_wider_for_a_smaller_sample():
    """The figure is published with an interval precisely so it cannot read as a score."""
    small_lo, small_hi = ms.wilson_interval(8, 10)
    big_lo, big_hi = ms.wilson_interval(800, 1000)
    assert (small_hi - small_lo) > (big_hi - big_lo) * 3, (
        "the interval does not widen as the sample shrinks, so it is not doing its job")
    assert 0.0 <= small_lo <= small_hi <= 1.0


def test_the_interval_does_not_exceed_its_bounds_at_a_perfect_sample():
    """A naive normal approximation returns an upper bound above 1.0 here, and a figure like
    "kill rate 100% (95% interval 97% to 104%)" is the sort of thing a reader stops trusting.
    """
    lo, hi = ms.wilson_interval(50, 50)
    assert hi <= 1.0, hi
    assert lo < 1.0, "a perfect sample should still carry uncertainty"


def test_the_tool_refuses_a_sample_of_nothing():
    """`wilson_interval(0, 0)` must not raise, because a run can legitimately attempt nothing
    and the report has to say so rather than crash on the way to saying so.
    """
    assert ms.wilson_interval(0, 0) == (0.0, 0.0)


def test_the_completeness_floor_refuses_the_run_that_measured_almost_nothing():
    """The real incident, as a test. 13 of 60 verdicts must not be reportable.

    These are the actual numbers from the 2026-10-01 near-miss: a 60-mutant sample where
    mutmut collected no tests, 13 stale verdicts were read from the previous run's metadata, and
    the tool printed RATCHET HELD and exited 0.
    """
    assert ms.completeness_floor(60) > 13, (
        "a run that reached a verdict on 13 of 60 sampled mutants would be reported as a result")
    assert ms.completeness_floor(60) > 27, (
        "the second observed partial run, 27 of 60, would also have been reported")


def test_the_completeness_floor_allows_a_run_that_lost_a_mutant_or_two():
    """A gate that demands perfection fails on the per-mutant CPU limit and gets switched off."""
    assert ms.completeness_floor(60) <= 59
    assert ms.completeness_floor(1000) <= 1000


def test_the_completeness_floor_is_never_zero():
    """`int(0.9 * 0)` is 0, and `evaluated < 0` is never true, so a sample of nothing would
    sail through the one check built to catch a run that measured nothing.
    """
    assert ms.completeness_floor(0) >= 1
    assert ms.completeness_floor(1) >= 1
    assert ms.completeness_floor(0) > 0, (
        "zero verdicts would pass the completeness floor")


def test_the_vendored_tree_is_excluded_from_the_mutation_surface():
    """A surviving mutant in llama.cpp's `gguf-py` is a finding about llama.cpp's tests.

    We neither own that code nor ship a fix for it, and including it would have put 27,000
    mutants we cannot act on into a budgeted sample, crowding out the ones we can.
    """
    assert any("/vendor/" in x for x in ms.EXCLUDE)
    mods = [str(p) for p in ms.modules()]
    assert mods, "no modules found at all; has the package layout moved?"
    assert not [m for m in mods if "/vendor/" in m], "vendored modules reached the surface"


def test_the_ledger_path_is_private():
    """The ledger names surviving mutants, which is a map of where the tests are weakest.

    That belongs in `private/`, not in a public page, and not in the repository's published
    documentation. It is working material for the next person to strengthen a test, and in the
    wrong place it is a list of where to look.
    """
    assert "private" in ms.LEDGER.parts, ms.LEDGER
