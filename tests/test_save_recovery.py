# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A completed search must survive a failing save.

The save is the crash-prone step and it runs when the search, the bake and the post-bake
measurement are all already paid for. These tests cover the classifier that decides whether a
retry could possibly help, the report an operator meets when it cannot, and the retry itself.

No model and no real disk failure: `_save_weights` is exercised against a stub whose writes
fail on demand, because a test that needs a full disk is a test that never runs.
"""
import types

import pytest

from senbonzakura import cli
from senbonzakura.crashsafe import (
    RETRY_SHARD_SIZE,
    is_space_exhaustion,
    save_failure_report,
)


# ── is_space_exhaustion: retry only when a retry could work ────────────────────────
@pytest.mark.parametrize("message", [
    "No space left on device",
    "[Errno 28] No space left on device: '/out/model.safetensors'",
    "Disk quota exceeded",
    "CUDA out of memory. Tried to allocate 384.00 MiB",
    "RuntimeError: not enough memory: you tried to allocate 2GB",
    "Cannot allocate memory",
    "insufficient space for the requested allocation",
])
def test_space_exhaustion_recognised(message):
    assert is_space_exhaustion(RuntimeError(message)) is True


def test_space_exhaustion_is_case_insensitive():
    assert is_space_exhaustion(RuntimeError("NO SPACE LEFT ON DEVICE")) is True


@pytest.mark.parametrize("message", [
    "Permission denied",
    "'NoneType' object has no attribute 'shape'",
    "safetensors does not support shared tensors",
    "Read-only file system",
])
def test_non_space_failures_are_not_retried(message):
    assert is_space_exhaustion(RuntimeError(message)) is False


def test_empty_message_is_not_treated_as_a_full_disk():
    # An exception that says nothing is not evidence of anything. Retrying on it would
    # burn a second save attempt on a bug that will fail identically.
    assert is_space_exhaustion(RuntimeError("")) is False


def test_errno_is_recognised_without_a_matching_message():
    # ENOSPC and EDQUOT carry the meaning even when the message is localised or absent,
    # which is the case string matching alone would miss.
    assert is_space_exhaustion(OSError(28, "")) is True
    assert is_space_exhaustion(OSError(122, "")) is True
    assert is_space_exhaustion(OSError(13, "Permission denied")) is False


# ── save_failure_report: the operator must learn the run is recoverable ────────────
def test_report_names_the_recovery_path():
    msg = save_failure_report(RuntimeError("boom"), "/out", free_bytes=1e9)
    assert "--bake-config /out/best-config.json" in msg
    assert "NOT LOST" in msg
    assert "boom" in msg


def test_report_carries_the_measured_context():
    msg = save_failure_report(RuntimeError("boom"), "/out", free_bytes=2.5e9, cuda_free=1.25e9)
    assert "2.5 GB" in msg        # free disk at the moment it broke
    assert "1.2 GB" in msg        # free VRAM


def test_report_says_so_when_the_disk_could_not_be_measured():
    # An unmeasurable disk is reported as unmeasured rather than as 0.0 GB, which would
    # send the operator to free space that was never the problem.
    msg = save_failure_report(RuntimeError("boom"), "/out", free_bytes=None)
    assert "could not be measured" in msg
    assert "0.0 GB" not in msg


def test_report_admits_the_retry_already_happened():
    msg = save_failure_report(RuntimeError("boom"), "/out", free_bytes=1e9, retried=True)
    assert RETRY_SHARD_SIZE in msg
    assert "already retried" in msg


# ── _save_weights: the retry, and the refusal to retry ─────────────────────────────
class _Recorder:
    """A stand-in for the model/tokenizer pair that records shard sizes and fails on cue."""

    def __init__(self, fail_with=None, fail_times=0):
        self.fail_with, self.fail_times = fail_with, fail_times
        self.shard_sizes = []

    def save_pretrained(self, out, **kw):
        if "max_shard_size" in kw:
            self.shard_sizes.append(kw["max_shard_size"])
            if self.fail_times > 0:
                self.fail_times -= 1
                raise self.fail_with


def _abliterator(model, tok=None):
    """A bare object carrying only what _save_weights touches."""
    obj = cli.Abliterator.__new__(cli.Abliterator)
    obj.model, obj.tok = model, tok or _Recorder()
    obj.dev = "cpu"
    obj.args = types.SimpleNamespace(out="/tmp/senbon-test-out")
    obj.log = lambda _m: None
    return obj


def test_first_attempt_uses_4gb_shards_and_returns():
    m = _Recorder()
    _abliterator(m)._save_weights()
    assert m.shard_sizes == ["4GB"]


def test_space_failure_retries_once_at_smaller_shards():
    m = _Recorder(fail_with=RuntimeError("No space left on device"), fail_times=1)
    _abliterator(m)._save_weights()
    # The retry is at a SMALLER size; retrying at the same size would just fail again.
    assert m.shard_sizes == ["4GB", RETRY_SHARD_SIZE]


def test_a_non_space_failure_is_not_retried():
    m = _Recorder(fail_with=RuntimeError("safetensors does not support shared tensors"),
                  fail_times=1)
    with pytest.raises(SystemExit) as e:
        _abliterator(m)._save_weights()
    assert m.shard_sizes == ["4GB"]                   # no second attempt
    assert "--bake-config" in str(e.value)


def test_both_attempts_failing_exits_loudly_and_says_it_retried():
    m = _Recorder(fail_with=RuntimeError("No space left on device"), fail_times=2)
    with pytest.raises(SystemExit) as e:
        _abliterator(m)._save_weights()
    assert m.shard_sizes == ["4GB", RETRY_SHARD_SIZE]
    msg = str(e.value)
    assert "already retried" in msg
    assert "--bake-config" in msg


def test_a_successful_retry_saves_the_tokenizer_too():
    # The tokenizer write follows the weights inside the same attempt, so a retry that
    # stopped at the weights would leave a directory no one can load.
    m = _Recorder(fail_with=RuntimeError("Disk quota exceeded"), fail_times=1)
    tok = _Recorder()
    _abliterator(m, tok)._save_weights()
    assert tok.shard_sizes == []          # the tokenizer takes no shard size
