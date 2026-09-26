# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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

Measuring an exit code through the wrong thing is a defect this project has shipped
repeatedly: a doctor gate that read a return value nobody returned, a shell `if` that
evaluated to zero when its command failed, a runner that read a pipeline's status instead of
its command's. This is the first one a test catches before a user does.
"""
from __future__ import annotations

import os
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


#: Commands that are COMPLETE with no arguments, so exiting 0 having printed a report is the
#: correct behaviour rather than the defect this file is named for. `setup` is the first: its whole
#: job is to look at this machine and say what it found, and requiring an argument would be
#: ceremony. The silence half of the check still applies to it, and is the half that catches a
#: module with no `__main__` guard.
#: Commands that legitimately do their whole job with no arguments, so exiting 0 is correct and
#: the assertion below would be asserting the opposite of the truth. Each one is named with its
#: reason, because an unexplained entry here is how a genuinely silent module would hide.
NEEDS_NO_ARGUMENTS = {
    "setup",     # reads the machine and prints what it found; changes nothing without --apply
    # Fetches and packs the corpora, which is the entire command. It became a command on
    # 2026-09-23, having been `tools/packaging/build_corpora.py`, and only the ubuntu-3.12 CI row
    # caught it: that is the one row given an authenticated `gh`, so it is the only place the
    # build SUCCEEDS with no arguments. Everywhere else it failed for want of a credential and
    # the assertion passed for the wrong reason, which is the shape this whole file is about.
    "corpora",
}


@pytest.mark.parametrize("command", sorted(entry.DELEGATED))
def test_no_delegated_module_runs_silently(command):
    """Exiting 0 having done nothing is the specific failure, so it gets its own assertion.

    THE ASSERTION USED TO SIT BEHIND `if proc.returncode == 0:`. Every one of these modules needs
    arguments, so with none they all exit 2 or 1, so the branch was never taken: sixteen green
    tests executing zero assertions, guarding the defect this file is named for. A test can be
    correct, run, and pass while asking nothing, and there is no way to tell from the report.

    So both halves are asserted unconditionally. A module with no `__main__` guard imports, runs
    nothing, prints nothing and exits 0, and either assertion catches it on its own.
    """
    module, _attr = entry.DELEGATED[command]
    proc = subprocess.run(
        [sys.executable, "-m", f"senbonzakura.{module}"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300, check=False)
    printed = (proc.stdout + proc.stderr).strip()

    assert printed, (
        f"'python -m senbonzakura.{module}' printed nothing at all. A module with no `__main__` "
        f"guard imports, runs nothing, prints nothing and exits 0, and this is what catches it.")
    if command in NEEDS_NO_ARGUMENTS:
        return
    assert proc.returncode != 0, (
        f"'python -m senbonzakura.{module}' exited 0 with no arguments. If this command genuinely "
        f"does useful work with none, exempt it here by name and say why; leaving it to pass "
        f"quietly is how a module with no `__main__` guard reported success for eight commands.")


def test_the_silent_module_check_actually_detects_a_silent_module(tmp_path):
    """The detector above, pointed at a module that has the defect.

    Every test in this file verifies that a command reports correctly. None verified that the
    checking would notice if one stopped, which is the gap that let the real assertion sit behind
    a branch nothing took. This builds a package whose module has no `__main__` guard, which is
    the original defect exactly, and confirms it exits 0 in silence.
    """
    pkg = tmp_path / "silentpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    # No `if __name__ == "__main__":` block, so `-m` imports this and runs nothing.
    (pkg / "quiet.py").write_text("def main():\n    print('work')\n", encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, "-m", "silentpkg.quiet"],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=300, check=False, env=env)

    assert proc.returncode == 0 and not (proc.stdout + proc.stderr).strip(), (
        "the shape this file exists to catch no longer behaves as described, so the assertions "
        "above are being checked against something other than the real defect")


@pytest.mark.parametrize("value", [256, 512, -1, 10_000])
def test_a_status_that_would_wrap_to_success_is_clamped(value):
    """`sys.exit(256)` exits 0: the shell keeps the low byte, so a failure reports success.

    No command returns one today. This is the same shape as every other defect in this file:
    a verdict that inverts silently, in a place nobody is looking.
    """
    assert entry.exit_status(value) == 1


def test_a_normal_status_is_untouched():
    for value in (0, 1, 2, 255):
        assert entry.exit_status(value) == value


# ── a result that declares its own figure invalid is not a success ────────────────────
#
# Found by the novice reader in the 2026-09-26 user pass, on their first real result. The compass
# printed THE AUC ABOVE IS NOT A MEASUREMENT OF HARM DISCRIMINATION, in those words, and exited 0.
#
# `python -m senbonzakura.margin` already exited 1 on that condition. `senbonzakura compass` did
# not, because the check lived in one module's `__main__` guard rather than here, and this
# function's own docstring says this is the one place that knows it is talking to a shell. Two
# doors into one command disagreeing about whether it succeeded is this project's most-repeated
# defect shape.

def test_a_self_invalidated_result_is_not_a_success():
    assert entry.exit_status({"self_invalidated": True}) == 1, (
        "a run that printed THE AUC ABOVE IS NOT A MEASUREMENT still exits 0, so a script gating "
        "on it cannot tell a measurement from the absence of one")


def test_an_ordinary_result_object_is_still_a_success():
    """The rule must not turn every returned dict into a failure: five commands return their result
    object rather than a status, and that convention is load-bearing.
    """
    assert entry.exit_status({"auc": 0.98}) == 0
    assert entry.exit_status({"self_invalidated": False}) == 0
    assert entry.exit_status({"self_invalidated": None}) == 0


def test_self_invalidation_is_status_one_and_not_two():
    """1 is "it ran and the figure is not usable". 2 is a refusal, where nothing ran. The artefact
    here is real and complete, so conflating the two tells a script the opposite of what happened.
    """
    assert entry.exit_status({"self_invalidated": True}) == 1


def test_the_marker_is_generic_rather_than_about_one_command():
    """`exit_status` must not need to know which command wrote the result.

    ASSERTED ON BEHAVIOUR, and the first version of this got that wrong in a way worth recording:
    it read `inspect.getsource` and asserted the word "readout" did not appear, which failed on the
    COMMENT explaining why the rule is generic. That is the defect this session had already fixed
    twice elsewhere, a guard measuring how code is spelled rather than what it does.
    """
    assert entry.exit_status(
        {"self_invalidated": True, "coherence": 2.61, "units": "nats-per-token"}) == 1
    assert entry.exit_status({"self_invalidated": "any truthy reason string"}) == 1
