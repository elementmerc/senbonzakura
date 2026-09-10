# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The CI gate says which commit its verdict is about, and whether anything newer is still going.

WHY THIS FILE EXISTS

The gate was written after CI sat red on `dev` for thirty-seven days behind a green badge, and it
did its job. Then it produced the mirror-image error. At the start of a session it printed

    [ci gate] The last CI run on 'dev' FAILED (ddc1cd9, 2026-09-09).

naming a commit two behind HEAD, six minutes after the build for HEAD had started, and that build
went on to pass all ten jobs. Every word of it was true of the run it examined and the sentence a
reader takes away from it was false: the branch was green.

That is the shape this project keeps finding, and it is worth naming precisely. The check filtered
to COMPLETED runs, which is correct, because an unfinished run has no verdict. What it then did was
report that verdict as though it were the branch's. A verdict about one commit, printed as a
verdict about the branch, is a proxy standing in for the thing.

WHY THE REAL SCRIPT IS DRIVEN

Reimplementing the selection here would be a second copy agreeing with the first. The script shells
out to `gh`, so a stub on PATH is enough to hand it any run list worth asking about, and the words
it actually prints can be read.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "ci-check.sh"

#: The hook declines to run at all without a way to bound a network call, and says so in its own
#: words. macOS ships neither `timeout` nor `gtimeout` unless coreutils were installed, so the
#: assertions below would read that deliberate silence as a broken gate.
HAS_DEADLINE = bool(shutil.which("timeout") or shutil.which("gtimeout"))

#: Skipped inside the fixture rather than by a module mark, so the one check that needs no shell
#: at all still runs on every platform CI covers. A mark on a fixture is silently ignored.
NO_SHELL = ("drives a POSIX shell hook that needs timeout(1) to bound its own network call"
            if os.name == "nt" or not HAS_DEADLINE else "")


def test_the_hook_is_in_the_repository_at_all():
    """Not a formality. Its five sibling hooks were tracked and this one was not, so it lived on
    one machine and reached no clone: the gate about CI was the only gate a checkout did not get.

    Skipped where there is no git to ask, which is not the same as passing. A `git archive`
    extract, an sdist and a packaged wheel all have the files and none of them have the index, so
    the question is unanswerable there rather than answered yes. The first version of this asserted
    on the return code regardless and failed in a clean-room extract for a reason that had nothing
    to do with the hook, which is a check reporting on its own environment.
    """
    assert HOOK.exists(), f"{HOOK} is missing from the checkout"
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(HOOK)],
                             capture_output=True, text=True, check=False,
                             cwd=HOOK.parents[2])
    if "not a git repository" in (tracked.stderr or ""):
        pytest.skip("not a git checkout")
    assert tracked.returncode == 0, "the hook exists here but git does not track it"


def _run(name, sha, status="completed", conclusion="success", created="2026-09-09T23:00:00Z"):
    return {"conclusion": conclusion, "status": status, "headSha": sha,
            "createdAt": created, "url": f"https://example.invalid/{name}"}


@pytest.fixture
def gate(tmp_path):
    """A git repo with one commit, and a `gh` that answers with whatever run list is asked for."""
    if NO_SHELL:
        pytest.skip(NO_SHELL)
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e"}
    for cmd in (["git", "init", "-q", "-b", "dev"],
                ["git", "commit", "-q", "--allow-empty", "-m", "one"]):
        subprocess.run(cmd, cwd=repo, env=env, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def call(runs):
        (bin_dir / "runs.json").write_text(json.dumps(runs), encoding="utf-8")
        (bin_dir / "gh").write_text(
            '#!/bin/sh\n'
            'case "$1" in\n'
            '  auth) exit 0 ;;\n'
            f'  run)  cat "{bin_dir}/runs.json" ;;\n'
            '  *)    exit 1 ;;\n'
            'esac\n', encoding="utf-8")
        (bin_dir / "gh").chmod(0o755)
        p = subprocess.run(["bash", str(HOOK)], cwd=repo, capture_output=True, text=True,
                           timeout=60, check=False,
                           env={**env, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                                "CLAUDE_PROJECT_DIR": str(repo)})
        assert p.returncode == 0, f"the gate must never block a session: {p.stderr}"
        return p.stdout

    call.head = head
    return call


# ── the defect this file is named for ────────────────────────────────────────────────

def test_a_newer_run_still_going_is_not_reported_as_a_failing_branch(gate):
    """THE ONE THAT HAPPENED. An older failure plus a live build for HEAD is not "CI failed"."""
    out = gate([_run("new", gate.head, status="in_progress", conclusion=None),
                _run("old", "d" * 40, conclusion="failure")])
    assert "still going" in out
    assert gate.head[:7] in out, "the reader has to be told which commit is being built"
    assert "The last CI run on 'dev' FAILED" not in out, (
        "the branch was not failing; a run two commits behind it was")


def test_a_failure_on_the_current_head_is_still_stated_plainly(gate):
    """The gate must not get so careful that a genuine red HEAD reads as a hedge."""
    out = gate([_run("head", gate.head, conclusion="failure")])
    assert "CI FAILED on the current HEAD" in out
    assert gate.head[:7] in out


def test_a_failure_behind_an_unbuilt_head_says_head_has_no_run(gate):
    """Neither "the branch failed" nor "we are fine". HEAD simply has not been built."""
    out = gate([_run("old", "d" * 40, conclusion="failure")])
    assert "has no completed run of its own" in out
    assert gate.head[:7] in out


def test_a_green_head_prints_nothing_at_all(gate):
    """A quiet gate is the normal case, and noise is how a briefing stops being read."""
    assert gate([_run("head", gate.head)]) == ""


def test_an_in_flight_run_over_a_green_history_stays_quiet(gate):
    """Nothing to warn about, so nothing is said, even though a verdict is pending."""
    assert gate([_run("new", gate.head, status="in_progress", conclusion=None),
                 _run("old", "d" * 40)]) == ""


# ── the streak, which is what made the thirty-seven days legible ─────────────────────

def test_a_run_of_failures_is_counted_and_called_out(gate):
    out = gate([_run(f"f{i}", "d" * 40, conclusion="failure") for i in range(4)]
               + [_run("ok", "e" * 40)])
    assert "failed 4 runs in a row" in out
    assert "not a blip" in out


def test_a_streak_filling_the_whole_window_is_reported_as_at_least(gate):
    """Saying "5 in a row" when five is all we looked at is a number about the query."""
    out = gate([_run(f"f{i}", "d" * 40, conclusion="failure") for i in range(5)])
    assert "at least 5 runs in a row" in out
    assert "(all we looked at)" in out


def test_the_streak_counts_completed_runs_only(gate):
    """An in-flight run has no verdict and must not break or extend a streak."""
    out = gate([_run("live", gate.head, status="in_progress", conclusion=None)]
               + [_run(f"f{i}", "d" * 40, conclusion="failure") for i in range(3)])
    assert "failed 3 runs in a row" in out


def test_an_unfinished_run_between_two_failures_does_not_cut_the_streak_short(gate):
    """The case that tells the two implementations apart.

    An unfinished run listed first is filtered by any version of this, correct or not. One sitting
    between two failures is filtered only if unfinished runs are genuinely excluded rather than
    merely skipped over at the front, and a version that keeps it stops counting there and reports
    a five-run outage as a single failure. A re-run of an older commit puts one exactly there.
    """
    out = gate([_run("f0", "d" * 40, conclusion="failure"),
                _run("live", "e" * 40, status="in_progress", conclusion=None),
                _run("f1", "d" * 40, conclusion="failure"),
                _run("f2", "d" * 40, conclusion="failure")])
    assert "failed 3 runs in a row" in out, (
        "an unfinished run in the middle has no verdict and must not end the streak")


# ── fail-open, in every direction ────────────────────────────────────────────────────

def test_an_empty_run_list_says_nothing(gate):
    assert gate([]) == ""


def test_runs_that_have_all_never_finished_say_nothing(gate):
    """No completed run means no verdict, and a gate with no verdict keeps quiet."""
    assert gate([_run("a", gate.head, status="queued", conclusion=None)]) == ""


def test_a_cancelled_run_is_named_rather_than_called_a_failure(gate):
    out = gate([_run("c", gate.head, conclusion="cancelled")])
    assert "ended 'cancelled'" in out
    assert "FAILED" not in out
