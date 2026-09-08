# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The harness directory, and the three names it has been called at once.

WHAT HAPPENED

`6cb6149` renamed `bench/` to `head-to-head/` and changed the container mount to `/work/bench`.
It updated the directory and the shell script. Two functions in `headtohead.py` kept saying
`headtohead`, which is a THIRD name that has never existed on either side of the boundary: the
host directory has hyphens, so it is not an importable Python package at all, and the container
mounts somewhere else again.

So `python -u -m headtohead.run_heretic` could not resolve under any interpreter, on any
machine. Every Heretic arm of a five-seed comparison died at import without writing a byte. The
arm's output directory is created before the command runs, so from outside each one looked like
a finished arm, and all five were reported as complete for two hours.

WHY A TEST AND NOT A CAREFUL RENAME

Because a careful rename is what produced this. Three places name the same directory (a Python
constant, a shell script's `-v` mount, and the filesystem) and nothing compared them. A rename
is exactly the operation where a human updates two of three and the third fails silently on a
machine they are not looking at.

So this asks the filesystem and the shell script rather than trusting either. The container half
is checked by reading the mount out of `run-isolated.sh`, because that script is the thing that
actually decides where the files land, and a constant that disagrees with it is wrong however
carefully it was chosen.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from senbonzakura import headtohead

ROOT = Path(__file__).resolve().parent.parent


def test_the_harness_directory_the_code_names_exists():
    d = ROOT / headtohead.BENCH_DIR_NAME
    assert d.is_dir(), (
        f"the code looks for {headtohead.BENCH_DIR_NAME}/ and this repository has no such "
        f"directory. Every Heretic arm fails at launch when these disagree.")


@pytest.mark.parametrize("script", ["run_heretic.py", "best_of_n_heretic.py", "run-isolated.sh"])
def test_each_file_the_code_launches_is_actually_there(script):
    """Named individually, because the directory existing is not the question. What matters is
    that the specific file each argv builder points at is in it.
    """
    assert (ROOT / headtohead.BENCH_DIR_NAME / script).is_file(), (
        f"{script} is launched by name and is not in {headtohead.BENCH_DIR_NAME}/")


def test_the_guest_mount_matches_what_the_isolation_script_actually_mounts():
    """The container half, read out of the script rather than agreed by convention.

    `run-isolated.sh` is what decides where the harness lands inside the box. A Python constant
    that disagrees with it points the arm at an empty path, and the failure surfaces as a
    missing file inside a container, which is the least legible place this project has.
    """
    text = (ROOT / headtohead.BENCH_DIR_NAME / "run-isolated.sh").read_text(encoding="utf-8")
    # The SOURCE half of the mount is deliberately not matched. The harness line's source is
    # `$(cd "$(dirname "$0")" && pwd)`, which contains both spaces and nested quotes, and two
    # earlier versions of this pattern reported the constant as wrong because they tried to
    # match it. What matters is the destination, so only the destination is read.
    mounts = [m for line in text.splitlines() if "-v " in line
              for m in re.findall(r":(/work/[A-Za-z0-9_-]+):[a-z,]+", line)]
    assert headtohead.GUEST_BENCH in mounts, (
        f"the code expects the harness at {headtohead.GUEST_BENCH} inside the container, and "
        f"run-isolated.sh mounts {sorted(set(mounts))}. One of the two is wrong.")


def test_the_heretic_runner_is_launched_as_a_file_not_as_a_module():
    """THE DEFECT ITSELF. `head-to-head` has hyphens, so it can never be a module name.

    Asserted on the argv the shipped function builds, because that string is the whole bug: it
    was syntactically fine, it read plausibly, and it referred to something that has never
    existed.
    """
    argv = headtohead._heretic_argv(
        model="m", track="t", out="/tmp/out", seed=41, trials=10,
        slices=ROOT / "examples", extra=[])
    assert "-m" not in argv, f"the runner is still launched as a module: {argv}"
    launched = next(a for a in argv if a.endswith("run_heretic.py"))
    assert Path(launched).is_file(), f"the argv points at {launched}, which does not exist"


def test_a_missing_harness_directory_is_refused_rather_than_guessed_at(monkeypatch, tmp_path):
    """It used to return a bare relative path as a last resort, which turns "the harness is not
    where I expected" into a command that runs and fails later about a missing file, with the
    reader looking for the wrong problem.
    """
    monkeypatch.setattr(headtohead.Path, "cwd", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(headtohead.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(headtohead, "__file__", str(tmp_path / "a" / "b" / "headtohead.py"))
    with pytest.raises(headtohead.BenchError, match=headtohead.BENCH_DIR_NAME):
        headtohead._bench_dir_for("/some/host/out")


def test_the_guest_path_is_used_when_the_output_path_says_container():
    """The guest-or-host decision is made by the output path, and must not touch the filesystem."""
    assert headtohead._bench_dir_for(headtohead.GUEST_OUT) == Path(headtohead.GUEST_BENCH)
