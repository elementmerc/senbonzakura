# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The guard for a failure that has reached CI three times in one shape.

A pre-flight is added to `run_parsed`. It refuses when the machine lacks something: a card, the
corpora, an evaluation track. Every development machine here has all three, so the suite stays
green; no CI runner has any of them, so six or eight jobs go red on the next push, in tests whose
subject was never the environment.

The repair both previous times was to patch the new pre-flight into whichever tests had broken.
That fixes the instance and leaves the class alone, which is why it happened again. What closes it
is a single list of the environment pre-flights, in `conftest`, and this test refusing to let
`run_parsed` grow one that is not on it.

So the failure mode this owns is not "a test is broken". It is "somebody adds a pre-flight, all
local suites stay green, and the breakage is discovered by a runner". This test fails on the
machine that added it.
"""
import inspect
import re

from senbonzakura import cli

from conftest import ENVIRONMENT_PREFLIGHTS

#: Calls in `run_parsed` that can refuse, but on the strength of the ARGV rather than the machine.
#: A parsing test has no reason to silence these and several exist to assert that they fire, so
#: they are deliberately not in the fixture. Listed here so the difference is a decision on the
#: record rather than an omission.
ABOUT_THE_ARGUMENTS_NOT_THE_MACHINE = {
    "_preflight_numbers",
    "_preflight_model",
    "_preflight_generation_budget",
    "_preflight_recovery",
    "_preflight_dead_knobs",
    "_preflight_output",
}

#: Anything matching this in `run_parsed` is a pre-flight and has to be classified as one of the
#: two kinds above. `refuse_` is included because the track check is spelled that way, and a
#: refusal is what every one of these is.
PREFLIGHT_CALL = re.compile(r"\b(_preflight_[a-z_]+|refuse_without_[a-z_]+)\s*\(")


def _preflights_in_run_parsed():
    source = inspect.getsource(cli.run_parsed)
    return {m.group(1) for m in PREFLIGHT_CALL.finditer(source)}


def test_the_preflight_list_is_complete():
    found = _preflights_in_run_parsed()
    classified = set(ENVIRONMENT_PREFLIGHTS) | ABOUT_THE_ARGUMENTS_NOT_THE_MACHINE
    unclassified = found - classified
    assert not unclassified, (
        f"run_parsed calls {sorted(unclassified)}, which no list in the tests knows about.\n"
        f"If it can refuse because of what the MACHINE lacks (no card, no corpora, no track), add "
        f"it to ENVIRONMENT_PREFLIGHTS in tests/conftest.py, or every parsing test will pass here "
        f"and fail on a runner.\n"
        f"If it refuses on the strength of the ARGUMENTS, add it to "
        f"ABOUT_THE_ARGUMENTS_NOT_THE_MACHINE in this file instead.")


def test_every_environment_preflight_still_exists_on_the_module():
    """A rename would otherwise turn the fixture into a silent no-op.

    `monkeypatch.setattr(..., raising=True)` already fails loudly on a missing name, but only in
    tests that happen to use the fixture. This states it once, so a rename is reported as a rename.
    """
    missing = [n for n in ENVIRONMENT_PREFLIGHTS if not hasattr(cli, n)]
    assert not missing, (
        f"conftest.ENVIRONMENT_PREFLIGHTS names {missing}, which cli no longer has. If it was "
        f"renamed, rename it here; if it was removed, remove it here.")


def test_every_environment_preflight_is_actually_called_by_run_parsed():
    """The other direction: a pre-flight that stopped being called is a check nobody runs."""
    found = _preflights_in_run_parsed()
    stale = [n for n in ENVIRONMENT_PREFLIGHTS if n not in found]
    assert not stale, (
        f"{stale} is neutralised by the fixture but run_parsed no longer calls it. Either the "
        f"call was dropped, which means the check is gone, or it moved somewhere this test "
        f"cannot see.")
