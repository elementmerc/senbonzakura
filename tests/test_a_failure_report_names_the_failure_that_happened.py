# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A crash report must describe the crash that happened, not a likelier one.

Two failures in `crashsafe.py` had one shape between them: the message named a cause the code
had not established and prescribed a remedy that could not work.

  * every save failure closed with "Free space or point --out at a larger volume first", including
    the ones the call site had already classified as NOT a space problem. Following that advice
    costs another bake on a rented card and ends in the same error.
  * every `FileNotFoundError` or `PermissionError` out of the final rename was reported as two
    writers racing, so an unmounted destination and a revoked permission were both answered with
    "give them separate output paths".

These check the remedy, because the remedy is the part an operator acts on.
"""
import os
from pathlib import Path

import pytest

from senbonzakura.crashsafe import (
    atomic_write,
    replace_failure_reason,
    save_failure_report,
)

# The one phrase that sends the reader to free disk space. Asserted as a substring rather than by
# re-quoting the whole sentence, so a rewording of the space branch does not read as a regression.
FREE_SPACE_ADVICE = "Free space or point --out at a larger volume"


# ── the save report ───────────────────────────────────────────────────────────────
def test_a_space_failure_still_says_to_free_space():
    msg = save_failure_report(OSError(28, "No space left on device"), "/out", free_bytes=1e6)
    assert FREE_SPACE_ADVICE in msg
    assert "NOT A SPACE PROBLEM" not in msg


def test_a_caller_that_knows_it_is_not_space_is_not_told_to_free_space():
    """The call site in cli.py has just taken its `not is_space_exhaustion(...)` branch."""
    msg = save_failure_report(RuntimeError("Some tensors share memory"), "/out",
                              free_bytes=900e9, space_related=False)
    assert FREE_SPACE_ADVICE not in msg
    assert "NOT A SPACE PROBLEM" in msg
    # The recovery is still offered, because the search really is still on disk.
    assert "--bake-config /out/best-config.json" in msg
    assert "NOT LOST" in msg


def test_a_permission_error_is_not_answered_with_a_larger_volume():
    msg = save_failure_report(PermissionError(13, "Permission denied: '/ro/out'"), "/ro/out",
                              free_bytes=900e9, space_related=False)
    assert FREE_SPACE_ADVICE not in msg
    assert "read-only" in msg


def test_an_unset_discriminator_is_measured_rather_than_assumed():
    """The default is `is_space_exhaustion(exc)`, so an un-updated caller still reports honestly."""
    assert FREE_SPACE_ADVICE in save_failure_report(
        RuntimeError("Disk quota exceeded"), "/out", free_bytes=1e6)
    assert FREE_SPACE_ADVICE not in save_failure_report(
        RuntimeError("Some tensors share memory"), "/out", free_bytes=900e9)


def test_the_caller_can_override_the_guess_in_either_direction():
    # A space failure whose message says nothing a matcher recognises is still a space failure
    # when the caller knows it, and the report must then offer the remedy that works.
    assert FREE_SPACE_ADVICE in save_failure_report(
        RuntimeError("write failed"), "/out", free_bytes=1e6, space_related=True)


# ── the rename failure ────────────────────────────────────────────────────────────
def test_a_lost_race_is_still_reported_as_a_lost_race(tmp_path):
    target = tmp_path / "result.json"
    gone = tmp_path / "result.json.part"          # the winner renamed it away
    why = replace_failure_reason(target, gone, FileNotFoundError(2, "No such file or directory"))
    assert "two processes writing the same path" in why
    assert "separate output paths" in why


def test_a_vanished_destination_directory_is_not_reported_as_a_race(tmp_path):
    missing = tmp_path / "unmounted"
    target = missing / "result.json"
    why = replace_failure_reason(target, missing / "result.json.part",
                                 FileNotFoundError(2, "No such file or directory"))
    assert "no longer there" in why
    assert "unmounted" in why
    assert "separate output paths" not in why
    assert "two processes" not in why


def test_a_refused_rename_with_the_temporary_file_intact_is_not_reported_as_a_race(tmp_path):
    target = tmp_path / "result.json"
    tmp = tmp_path / "result.json.part"
    tmp.write_text("complete output", encoding="utf-8")
    why = replace_failure_reason(target, tmp, PermissionError(13, "Permission denied"))
    assert "separate output paths" not in why
    assert "read-only" in why
    assert "still there" in why


def test_a_refused_rename_reaches_the_operator_through_atomic_write(tmp_path, monkeypatch):
    """End to end: the helper's verdict is what `atomic_write` actually raises.

    The rename is stubbed rather than provoked with a read-only directory, because a read-only
    directory refuses the OPEN and never reaches the branch under test. This is the Windows shape
    (the target held open by another process) and the permissions-revoked-mid-run shape, both of
    which used to be reported as this writer having lost a race.
    """
    target = tmp_path / "result.json"

    def refuse(_src, _dst):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(RuntimeError) as caught, atomic_write(target) as f:
        f.write("x")
    assert "separate output paths" not in str(caught.value)
    assert "read-only" in str(caught.value)


def test_a_genuine_race_message_survives_the_helper_being_handed_real_paths(tmp_path):
    """Guards the ordering: a missing parent is checked BEFORE a missing temporary file.

    Both are absent when a directory goes away, so a check in the other order would report every
    unmounted destination as a race, which is the defect this closes.
    """
    missing = Path(tmp_path) / "gone"
    why = replace_failure_reason(missing / "a.json", missing / "a.json.part",
                                 FileNotFoundError(2, "No such file or directory"))
    assert "no longer there" in why
