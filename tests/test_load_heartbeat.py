# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A model load that stalls has to say which sub-step it is stuck in."""
from __future__ import annotations

import threading

import pytest

from senbonzakura import modelload


class Sink:
    """A log callable that records lines and lets a test wait for the nth one."""

    def __init__(self):
        self.lines = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def __call__(self, message):
        with self._cond:
            self.lines.append(message)
            self._cond.notify_all()

    def wait_for(self, count, timeout=10):
        with self._cond:
            ok = self._cond.wait_for(lambda: len(self.lines) >= count, timeout)
        assert ok, f"only {len(self.lines)} lines arrived: {self.lines}"

    def snapshot(self):
        with self._lock:
            return list(self.lines)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(modelload, "HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(modelload, "_resident_bytes", lambda: 3 << 30)
    monkeypatch.setattr(modelload, "_cuda_allocated_bytes", lambda: 1 << 30)


def _heartbeat_threads():
    return [t for t in threading.enumerate() if t.name == "load-heartbeat"]


def test_the_import_came_from_the_tree_under_test():
    import senbonzakura
    print("senbonzakura imported from", senbonzakura.__file__)


def test_a_slow_step_logs_its_name_elapsed_time_and_memory(fast):
    sink = Sink()
    with modelload.load_heartbeat(sink, "weights"):
        sink.wait_for(2)
    first = sink.snapshot()[0]
    assert "weights" in first
    assert "s since the load began" in first
    assert "resident memory 3.00 GiB" in first
    assert "cuda allocated 1.00 GiB" in first


def test_the_line_follows_the_step_as_it_changes(fast):
    sink = Sink()
    with modelload.load_heartbeat(sink, "tokenizer"):
        sink.wait_for(1)
        modelload.load_step("weights")
        seen = len(sink.snapshot())
        sink.wait_for(seen + 2)
    assert "tokenizer" in sink.snapshot()[0]
    assert any("weights" in line for line in sink.snapshot()[seen:])


def test_the_thread_stops_after_the_block(fast):
    sink = Sink()
    with modelload.load_heartbeat(sink, "weights"):
        sink.wait_for(1)
    assert _heartbeat_threads() == []
    count = len(sink.snapshot())
    threading.Event().wait(0.1)
    assert len(sink.snapshot()) == count
    assert modelload._active_heartbeat is None


def test_the_thread_stops_after_an_exception_which_propagates_unchanged(fast):
    sink = Sink()
    boom = RuntimeError("the load failed")
    with pytest.raises(RuntimeError) as caught:
        with modelload.load_heartbeat(sink, "weights"):
            sink.wait_for(1)
            raise boom
    assert caught.value is boom
    assert _heartbeat_threads() == []
    assert modelload._active_heartbeat is None


def test_a_systemexit_also_propagates_and_stops_the_thread(fast):
    sink = Sink()
    with pytest.raises(SystemExit):
        with modelload.load_heartbeat(sink, "weights"):
            raise SystemExit("refused")
    assert _heartbeat_threads() == []


def test_a_failing_reading_is_reported_as_unavailable_and_is_not_fatal(fast, monkeypatch):
    def broken():
        raise OSError("no /proc here")
    monkeypatch.setattr(modelload, "_resident_bytes", broken)
    monkeypatch.setattr(modelload, "_cuda_allocated_bytes", broken)
    sink = Sink()
    with modelload.load_heartbeat(sink, "weights"):
        sink.wait_for(2)
    line = sink.snapshot()[0]
    assert "resident memory unavailable (OSError: no /proc here)" in line
    assert "cuda allocated unavailable (OSError: no /proc here)" in line


def test_no_card_is_said_as_such(fast, monkeypatch):
    monkeypatch.setattr(modelload, "_cuda_allocated_bytes", lambda: None)
    sink = Sink()
    with modelload.load_heartbeat(sink, "weights"):
        sink.wait_for(1)
    assert "cuda allocated no card" in sink.snapshot()[0]


def test_the_real_readings_work_on_this_machine():
    assert modelload._resident_bytes() > 0
    assert modelload._cuda_allocated_bytes() is None or modelload._cuda_allocated_bytes() >= 0


def test_a_failing_log_sink_does_not_kill_the_load_or_the_thread(fast, capsys):
    calls = threading.Event()

    def bad(message):
        calls.set()
        raise OSError("disk full")
    with modelload.load_heartbeat(bad, "weights"):
        assert calls.wait(10)
    assert "the log could not be written: OSError: disk full" in capsys.readouterr().err


def test_stacks_are_dumped_once_after_the_threshold(fast, monkeypatch):
    monkeypatch.setattr(modelload, "STACK_DUMP_AFTER_BEATS", 3)
    sink = Sink()
    with modelload.load_heartbeat(sink, "weights"):
        sink.wait_for(6)
    dumps = [line for line in sink.snapshot() if "stacks of every thread follow" in line]
    assert len(dumps) == 1
    assert "no progress past 'weights'" in dumps[0]
    assert "--- thread" in dumps[0]
    assert sink.snapshot().index(dumps[0]) >= 3


def test_changing_step_restarts_the_stall_count_and_the_dump():
    beat = modelload._Heartbeat(lambda m: None, "a", 60, 3)
    beat._beats_in_step = 5
    beat._dumped = True
    beat.set_step("a")
    assert (beat._beats_in_step, beat._dumped) == (5, True)
    beat.set_step("b")
    assert (beat._beats_in_step, beat._dumped) == (0, False)


def test_a_nested_use_joins_the_running_heartbeat(fast):
    sink = Sink()
    with modelload.load_heartbeat(sink, "outer"):
        sink.wait_for(1)
        with modelload.load_heartbeat(sink, "inner"):
            assert len(_heartbeat_threads()) == 1
        assert len(_heartbeat_threads()) == 1
    assert _heartbeat_threads() == []


def test_without_a_log_nothing_runs():
    with modelload.load_heartbeat(None, "weights") as beat:
        assert beat is None
        assert _heartbeat_threads() == []
    modelload.load_step("ignored")


def test_the_loader_wraps_its_body_in_a_heartbeat(fast, monkeypatch):
    sink = Sink()
    seen = {}

    def inner(model_id, **kw):
        modelload.load_step("weights")
        sink.wait_for(1)
        seen["threads"] = len(_heartbeat_threads())
        return "model", "tok"
    monkeypatch.setattr(modelload, "_load_model_and_tokenizer", inner)
    assert modelload.load_model_and_tokenizer("m", log=sink) == ("model", "tok")
    assert seen["threads"] == 1
    assert _heartbeat_threads() == []
    assert "weights" in sink.snapshot()[0]
