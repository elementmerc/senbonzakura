# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The harness that runs every documented command, checked without needing an install.

WHY THIS EXISTS

`tools/ci/docs_commands_run.py` is deny-first: a documented command it cannot classify is a
failure, so adding a command to a page forces a decision to run it or to record why it cannot be
run. That property is only as good as the rule table, and the rule table has failed twice in ways
a passing CI job could not have shown.

The first was a catch-all `(r".*", SKIP, ...)` left in while the table was being written. Every
command in the documentation would have been skipped and the job would have printed a clean tick
over a check that ran nothing. The second is the one this file was written for: `man senbonzakura`
was grouped with `rm`, `mv` and `apt-get` under "changes the machine rather than exercising the
tool", which is not true of `man`, so the one documented command that proves the manual page ships
in the wheel was never run, under a reason that did not apply to it.

Both are the same shape, and it is the shape this project keeps meeting: **a check that answers a
narrower question than the one being asked reports clean.** The CI job needs a real install of the
package to say anything at all. These tests need nothing, so they run everywhere, on every commit.
"""
import importlib.util
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent / "tools" / "ci" / "docs_commands_run.py"


@pytest.fixture(scope="module")
def harness():
    """The runner, imported by path: `tools/` is repository-only and is not an importable package."""
    spec = importlib.util.spec_from_file_location("docs_commands_run", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def documented(harness):
    return list(harness.commands())


def test_the_documentation_contains_commands_to_check(documented):
    """A harness over an empty list passes every assertion below and measures nothing."""
    assert len(documented) > 20, (
        f"only {len(documented)} commands were extracted from the documentation, which is too few "
        f"for the rest of this file to mean anything. Either the pages lost their fenced blocks or "
        f"the extractor stopped recognising them.")


def test_every_documented_command_is_classified(harness, documented):
    """The deny-first property itself, checked with no install and on every platform."""
    unclassified = [(page, line, cmd) for page, line, cmd in documented
                    if harness.classify(cmd)[0] is None]
    assert not unclassified, (
        "these documented commands match no rule in the harness, so the job that runs the "
        "documentation would refuse:\n" + "\n".join(
            f"  {page}:{line}  {cmd}" for page, line, cmd in unclassified))


def test_the_rules_do_not_skip_everything(harness, documented):
    """A catch-all skip rule would leave the job green having executed nothing.

    The count is a floor rather than an exact figure, so adding a documented command that
    legitimately cannot run here does not fail this.
    """
    run = [cmd for _page, _line, cmd in documented if harness.classify(cmd)[0] == harness.RUN]
    assert len(run) >= 8, (
        f"only {len(run)} documented commands would actually be run. A rule that matches too "
        f"broadly turns this gate into a tick over nothing; check the table for a pattern that "
        f"swallowed its neighbours.\nCurrently runnable: {run}")


def test_the_manual_page_is_run_rather_than_skipped(harness):
    """`man senbonzakura` is the only check that the wheel's manual page exists and renders.

    `man` reads and does not write, and `run_one` puts the install under test first on `PATH`,
    which is where `man` builds its search path from, so it resolves to the page inside the
    environment being checked rather than to anything on the runner.
    """
    disposition, reason = harness.classify("man senbonzakura")
    assert disposition == harness.RUN, (
        f"`man senbonzakura` is classified {disposition!r} ({reason!r}). It is a documented "
        f"command, it works, and it is the only thing asserting the manual page ships.")


@pytest.mark.parametrize("command", [
    "rm -rf build/",
    "mkdir -p out",
    "apt-get install -y git",
])
def test_commands_that_change_the_machine_are_still_skipped(harness, command):
    """Taking `man` out of that group must not have taken the group with it."""
    disposition, _reason = harness.classify(command)
    assert disposition == harness.SKIP, (
        f"{command!r} is classified {disposition!r}; this harness must not carry out commands that "
        f"modify the machine it is checking.")


def test_every_skip_carries_a_reason(harness, documented):
    """A skip with no reason is indistinguishable from a command nobody thought about."""
    silent = [cmd for _page, _line, cmd in documented
              if harness.classify(cmd)[0] == harness.SKIP and not harness.classify(cmd)[1].strip()]
    assert not silent, f"these commands are skipped with no reason printed: {silent}"
