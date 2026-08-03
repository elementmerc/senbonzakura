"""`atomic_write`: a result file is never observed half-written.

The defect this closes: every result was written with a plain `open(path, "w")`, which
truncates on the spot. A process killed between that and the last byte left a file that
existed, was not empty, and was not a result. A run spec's completeness check trusted
exactly that shape on 2026-08-03 and would have skipped re-measuring an arm that never
finished.

The subprocess test is the one that counts. Everything else here checks the contract; only
a real SIGKILL between the open and the close proves the file downstream sees is either the
old one or a complete new one.
"""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from senbonzakura.crashsafe import atomic_write


def test_it_writes_the_file(tmp_path):
    target = tmp_path / "r.json"
    with atomic_write(target) as f:
        json.dump({"auc": 0.98}, f)
    assert json.loads(target.read_text()) == {"auc": 0.98}


def test_it_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "deep" / "nested" / "r.json"
    with atomic_write(target) as f:
        f.write("{}")
    assert target.exists()


def test_it_accepts_a_string_path(tmp_path):
    target = tmp_path / "r.json"
    with atomic_write(str(target)) as f:
        f.write("{}")
    assert target.exists()


def test_it_leaves_no_part_file_behind_on_success(tmp_path):
    target = tmp_path / "r.json"
    with atomic_write(target) as f:
        f.write("{}")
    assert list(tmp_path.iterdir()) == [target]


def test_an_exception_leaves_the_previous_file_untouched(tmp_path):
    target = tmp_path / "r.json"
    target.write_text('{"auc": 0.5}')

    def write_then_fail():
        with atomic_write(target) as f:
            f.write('{"auc": 0.9')
            raise ValueError("the run died here")

    with pytest.raises(ValueError):
        write_then_fail()

    assert json.loads(target.read_text()) == {"auc": 0.5}, "the old result was destroyed"
    assert not (tmp_path / "r.json.part").exists(), "a partial file survived"


def test_a_keyboard_interrupt_also_cleans_up(tmp_path):
    """The likeliest way a long run is interrupted, and `except Exception` would miss it."""
    target = tmp_path / "r.json"

    def write_then_interrupt():
        with atomic_write(target) as f:
            f.write("partial")
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        write_then_interrupt()

    assert not target.exists()
    assert not (tmp_path / "r.json.part").exists()


def test_a_failed_write_does_not_create_the_target(tmp_path):
    target = tmp_path / "r.json"
    def write_then_raise():
        with atomic_write(target) as f:
            f.write("x")
            raise ZeroDivisionError("something unexpected, mid-write")

    with pytest.raises(ZeroDivisionError):
        write_then_raise()
    assert not target.exists(), "a target that never existed must not appear"


def test_it_replaces_rather_than_appends(tmp_path):
    target = tmp_path / "r.json"
    with atomic_write(target) as f:
        f.write("a" * 100)
    with atomic_write(target) as f:
        f.write("b")
    assert target.read_text() == "b"


KILL_SCRIPT = textwrap.dedent(
    """
    import os, sys, time
    from senbonzakura.crashsafe import atomic_write
    target = sys.argv[1]
    with open(target, "w", encoding="utf-8") as f:
        f.write('{"auc": 0.5}')          # a complete previous result
    print("READY", flush=True)
    with atomic_write(target) as f:
        f.write('{"auc": 0.9')           # a deliberately truncated new one
        f.flush()
        print("MIDWRITE", flush=True)
        time.sleep(30)                   # killed here
    """
)


def test_a_process_killed_mid_write_leaves_the_previous_result_intact(tmp_path):
    """The real thing: SIGKILL between the open and the close.

    This is the scenario that produced the finding. Under a plain `open(path, "w")` the
    target would be truncated to the 11 bytes written so far and would parse as neither the
    old result nor the new one.
    """
    target = tmp_path / "r.json"
    proc = subprocess.Popen(
        [sys.executable, "-c", KILL_SCRIPT, str(target)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "READY"
        assert proc.stdout.readline().strip() == "MIDWRITE"
        # The partial bytes are on disk by now; prove it, so the test cannot pass by
        # killing the child before it ever wrote anything.
        deadline = time.time() + 10
        while not (tmp_path / "r.json.part").exists() and time.time() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "r.json.part").exists(), "the child never reached the write"

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()

    assert json.loads(target.read_text()) == {"auc": 0.5}, (
        "the previous result was destroyed by a write that never completed"
    )
