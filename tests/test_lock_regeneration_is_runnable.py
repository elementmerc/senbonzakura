# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The lockfile's documented regeneration command and the script that runs it cannot disagree.

WHY THIS EXISTS

`.github/workflows/supply-chain.yml` goes red whenever a pin moves and `requirements.lock` was not
regenerated in the same commit. That is correct, and it is only fair if regenerating is a command
somebody can actually run. Before this, the recipe lived in a comment in the lockfile's header and
nothing checked that the comment stayed true: the day `uv` renamed a flag or the pinned Python
moved, the job would have gone red for a reason nobody could fix from the file it pointed at.

So `tools/packaging/regenerate_lock.sh` holds the command, the lockfile header documents it, and
this holds the two together. It is the same shape as `test_declared_floors.py`, which keeps
`constraints.txt` and `requirements.lock` naming the same versions, and the same shape as
`test_accepted_advisories.py`, which fails when the pin an accepted advisory was assessed against
moves. A rule enforced by somebody remembering it is the gate this project keeps watching fail.

WHY THE SCRIPT LIVES IN tools/packaging/ AND NOT IN scripts/

It was written in `scripts/` first, which would have made this whole test a lie. `/scripts/` is in
`.git/info/exclude` alongside `/.githooks/` and `/private/`, so nothing in it is tracked and
nothing in it reaches a clone. The script would have existed on one machine, `requirements.lock`
would have pointed at a path no reader has, and **this test would have failed in CI for the one
reason it was never meant to**: asserting the existence of a file the checkout cannot contain.
Caught by running `git check-ignore` on the new file rather than assuming a new path under a
familiar directory is tracked. `tools/` is tracked; `scripts/` is operator-local.

WHAT THIS DOES NOT DO

It does not run `uv`, and it does not regenerate anything. `uv` is not a declared dependency of
this project and a test that shells out to a package manager over the network is a test that fails
for reasons that have nothing to do with the assertion. This checks that the two recorded forms of
one command agree, which is the drift that actually happens.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "packaging" / "regenerate_lock.sh"
LOCK = ROOT / "requirements.lock"


def test_the_script_exists_and_is_executable():
    """A runbook that is not executable is a comment with a shebang."""
    assert SCRIPT.is_file(), f"{SCRIPT} is missing, and the lockfile header points at it"
    import os
    assert os.access(SCRIPT, os.X_OK), (
        f"{SCRIPT.name} is not executable, so the command the lockfile documents cannot be run "
        f"the way it is documented."
    )


def _script_command() -> str:
    """Ask the script itself, so this reads its definition rather than a copy of it."""
    proc = subprocess.run(
        [str(SCRIPT), "--print-command"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"{SCRIPT.name} --print-command exited {proc.returncode}: {proc.stderr.strip()}"
    )
    return proc.stdout.strip()


def _header_command() -> str:
    """The command as the lockfile's own header documents it."""
    header = LOCK.read_text(encoding="utf-8")
    # Only the header, which is everything above the first pin. A `uv pip compile` string further
    # down would not be the documented recipe, and matching it would make this test pass on the
    # wrong line. This is the slicing mistake that cost a false finding during the design of the
    # verification job: take the region deliberately, not a fixed number of lines.
    first_pin = re.search(r"^[A-Za-z0-9_.-]+==", header, flags=re.MULTILINE)
    assert first_pin, "requirements.lock names no pinned versions, so it has no header to read"
    found = re.findall(
        r"uv pip compile[^\n]*constraints\.txt", header[: first_pin.start()],
    )
    assert found, (
        "requirements.lock's header no longer documents a `uv pip compile ... constraints.txt` "
        "command. The header is what a reader reaches for when the verification job goes red."
    )
    assert len(found) == 1, (
        f"the header documents {len(found)} different compile commands: {found}. One of them is "
        f"the real one and a reader cannot tell which."
    )
    return found[0].strip()


@pytest.mark.skipif(sys.platform == "win32", reason="the script is bash; CI runs it on Linux")
def test_the_script_and_the_lockfile_header_agree_on_the_command():
    """The assertion this file exists for."""
    assert _script_command() == _header_command(), (
        f"tools/packaging/regenerate_lock.sh runs:\n    {_script_command()}\n"
        f"requirements.lock's header documents:\n    {_header_command()}\n"
        f"These have to be the same command. If the regeneration recipe changed, change it in "
        f"both places in the same commit."
    )


@pytest.mark.skipif(sys.platform == "win32", reason="the script is bash; CI runs it on Linux")
def test_the_command_still_carries_the_two_load_bearing_flags():
    """`--no-deps` and `--generate-hashes` are the whole design, not stylistic choices.

    Without `--generate-hashes` the file is a pin list and the verification job has nothing to
    verify. Without `--no-deps` the resolver adds every transitive dependency at whatever version
    is current today, which is a different environment wearing this filename, and the hashes then
    describe that instead. Measured 2026-10-01: a lock compiled with transitives present also
    breaks `pip download --require-hashes --no-deps`, which is what the CI job runs.
    """
    command = _script_command()
    for flag in ("--generate-hashes", "--no-deps"):
        assert flag in command, (
            f"{flag} is missing from the regeneration command. It is load-bearing: see this "
            f"test's docstring and the lockfile's header."
        )


def test_the_lockfile_documents_an_install_command_that_actually_works():
    """The header used to document `pip install -r requirements.lock`, which fails.

    Measured 2026-10-01: without `--no-deps`, hashes anywhere in the file put pip into
    `--require-hashes` mode, which then demands a hash for every transitive dependency too. This
    file is compiled `--no-deps` and holds only the direct pins, so the install fails naming
    something it has no hash for. A documented command that does not run is worse than no
    command, because a reader concludes the file is broken rather than the instruction.
    """
    header = LOCK.read_text(encoding="utf-8")
    first_pin = re.search(r"^[A-Za-z0-9_.-]+==", header, flags=re.MULTILINE)
    assert first_pin
    head = header[: first_pin.start()]
    for line in head.splitlines():
        if "pip install" in line and "requirements.lock" in line:
            assert "--no-deps" in line, (
                f"the header documents `{line.strip().lstrip('# ')}`, which fails in "
                f"--require-hashes mode because this lock holds no transitive hashes. It needs "
                f"--no-deps."
            )
