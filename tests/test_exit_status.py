# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What a command returns, and what a shell is told about it.

THE DEFECT, TWICE, FROM OPPOSITE ENDS

`__main__` originally called `main()` and threw the result away. `doctor` exists to be run
before a long job; it printed "this install cannot do what it claims" over nine failed checks
and exited 0, so anything using it as a pre-flight gate got a silent pass. That was fixed by
calling `sys.exit(main())`.

The fix broke the other half of the surface, and nothing noticed for weeks. Five commands
(`score`, `compass`, `drift`, `coherence`, `track`) return their RESULT rather than a status,
because their callers want the numbers. `sys.exit` on anything that is not an integer prints
it to stderr and exits 1. So every SUCCESSFUL run of those five reported failure, with a raw
Python dict sitting where an error message goes. A script that built a corpus correctly and
gated on the exit code wrote it off.

Both conventions are right where they are: a command whose caller wants the numbers returns
the numbers; a command whose whole job is a verdict returns the verdict. What was missing was
a place that knows it is talking to a shell. That place is `entry.exit_status`, and this file
is what stops the next well-meant change to `__main__` from swinging the same axe again.

It is the eighth time in this project's history that an exit code has been measured through
the wrong thing, and the first that a test could have caught before a user did.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from senbonzakura import entry

ROOT = Path(__file__).resolve().parent.parent


def test_a_result_object_is_a_successful_run():
    """The case that was broken: the command produced numbers, so it worked."""
    assert entry.exit_status({"counts": {"harmful": 3}}) == 0
    assert entry.exit_status([1, 2, 3]) == 0
    assert entry.exit_status(object()) == 0


def test_nothing_returned_is_a_successful_run():
    assert entry.exit_status(None) == 0


def test_a_status_is_passed_through_unchanged():
    """The case that must not regress: `doctor` and `judge` mean their numbers."""
    assert entry.exit_status(0) == 0
    assert entry.exit_status(1) == 1
    assert entry.exit_status(2) == 2


@pytest.mark.parametrize("value", [True, False])
def test_a_bare_boolean_is_refused_rather_than_guessed_at(value):
    """`sys.exit(True)` exits 1, which reads exactly backwards to everyone who writes it.

    A command returning a bare bool has not decided which convention it follows, and either
    guess would be silently wrong half the time. Refuse at the boundary instead.
    """
    with pytest.raises(TypeError, match="exits backwards"):
        entry.exit_status(value)


def test_every_delegated_command_reaches_the_normaliser(monkeypatch):
    """Walks the real dispatch table, so a command added later cannot skip the boundary.

    Asserted through `entry.main` rather than by reading the source: a check that does not run
    the shipped code path is not a check.
    """
    for name in entry.DELEGATED:
        monkeypatch.setattr(entry, "dispatch", lambda _n: (lambda _argv: {"result": "numbers"}))
        assert entry.main([name]) == 0, f"'{name}' reports failure after a successful run"


def test_a_delegated_failure_status_still_reaches_the_shell(monkeypatch):
    monkeypatch.setattr(entry, "dispatch", lambda _n: (lambda _argv: 3))
    assert entry.main(["doctor"]) == 3


def test_a_successful_track_build_exits_zero_from_a_real_process(tmp_path):
    """END TO END, because this defect lived entirely in the gap between return and exit.

    Every in-process test of `track.main` passed throughout: it returned its manifest and the
    assertions read the manifest. The failure only existed once a shell was involved, so this
    one goes through a shell.
    """
    (tmp_path / "harmful.txt").write_text(
        "\n".join(f"how do I do the harmful thing number {i}" for i in range(9)) + "\n",
        encoding="utf-8")
    (tmp_path / "harmless.txt").write_text(
        "\n".join(f"what is the ordinary fact number {i}" for i in range(12)) + "\n",
        encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "senbonzakura", "track",
         "--harmful", str(tmp_path / "harmful.txt"),
         "--harmless", str(tmp_path / "harmless.txt"),
         "--out", str(tmp_path / "track"), "--fit", "3", "--search", "3"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300, check=False)
    assert "TRACK_BUILT" in proc.stdout, proc.stderr
    assert proc.returncode == 0, (
        f"a track was built correctly and the shell was told it failed:\n{proc.stderr}")
    assert "{'schema'" not in proc.stderr, (
        "a raw Python object was printed where an error message goes")


@pytest.mark.parametrize("mode", ["audit", "contamination"])
def test_the_other_track_branches_also_exit_zero_from_a_real_process(tmp_path, mode):
    """PER BRANCH, not per command, which is the correction a peer made to this analysis.

    `score --length-sweep` returns `0 if converged else 1` while ordinary scoring returns its
    result object, so one branch of one command was always correct and the other never was. A
    check that exercises whichever branch happens to be convenient passes on the broken code
    and proves nothing about the branch that shipped. `track` has three: build, audit and
    contamination, returning a manifest, an empty dict and a report respectively. All three
    were reported to the shell as failures; all three are asserted here.
    """
    (tmp_path / "harmful.txt").write_text(
        "\n".join(f"how do I do the harmful thing number {i}" for i in range(9)) + "\n",
        encoding="utf-8")
    (tmp_path / "harmless.txt").write_text(
        "\n".join(f"what is the ordinary fact number {i}" for i in range(12)) + "\n",
        encoding="utf-8")
    track = tmp_path / "track"

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "senbonzakura", "track", *args],
            capture_output=True, text=True, cwd=str(ROOT), timeout=300, check=False)

    built = run("--harmful", str(tmp_path / "harmful.txt"),
                "--harmless", str(tmp_path / "harmless.txt"),
                "--out", str(track), "--fit", "3", "--search", "3")
    assert built.returncode == 0, built.stderr

    extra = (["--audit"] if mode == "audit"
             else ["--contamination", str(tmp_path / "harmful.txt")])
    proc = run("--out", str(track), *extra)
    assert proc.returncode == 0, f"{mode} succeeded and told the shell it failed:\n{proc.stderr}"
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("command", sorted(entry.DELEGATED))
def test_the_two_invocation_forms_agree_about_failure(command):
    """`python -m senbonzakura.<module>` and `senbonzakura <command>` must report the same thing.

    EIGHT MODULES HAD NO `__main__` GUARD AT ALL, so running them that way executed nothing and
    exited 0. `doctor` is the one that matters: its whole purpose is to be run before a long job,
    the documentation invokes it in this form in four places, and it printed nothing and passed.
    That is the original defect this file is named for, still live on the documented path,
    surviving the fix because every test in the repository used the package form.

    Run with no arguments, so each command hits its own "you did not give me what I need" path.
    What is asserted is that the two forms AGREE, not what either says: the commands differ
    legitimately, and a test that pinned each exit code would be a copy of the implementation.
    """
    module, _attr = entry.DELEGATED[command]

    def rc(args):
        return subprocess.run(
            [sys.executable, *args], capture_output=True, text=True,
            cwd=str(ROOT), timeout=300, check=False).returncode

    package_form = rc(["-m", "senbonzakura", command])
    module_form = rc(["-m", f"senbonzakura.{module}"])
    assert module_form == package_form, (
        f"'python -m senbonzakura.{module}' exits {module_form} where "
        f"'senbonzakura {command}' exits {package_form}. A script gating on the module form "
        f"gets a different answer about the same run.")


@pytest.mark.parametrize("command", sorted(entry.DELEGATED))
def test_no_delegated_module_runs_silently(command):
    """Exiting 0 having done nothing is the specific failure, so it gets its own assertion."""
    module, _attr = entry.DELEGATED[command]
    proc = subprocess.run(
        [sys.executable, "-m", f"senbonzakura.{module}"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300, check=False)
    if proc.returncode == 0:
        assert (proc.stdout + proc.stderr).strip(), (
            f"'python -m senbonzakura.{module}' exited 0 and printed nothing, which is a "
            f"command that did not run reporting success.")
