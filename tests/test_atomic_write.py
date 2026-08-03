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


def test_a_losing_concurrent_writer_says_what_happened(tmp_path):
    """Two writers of one path is a race, and the loser must diagnose it, not just fail.

    The fixed `.part` name means the writer that finishes second finds its temporary file
    already renamed away by the first. The bare error is a FileNotFoundError naming a file
    the caller never created, which reads like a missing-output bug. It is a race, and the
    message says so.
    """
    import threading
    import time

    target = tmp_path / "race.json"
    errors = []

    def write(tag, delay):
        try:
            with atomic_write(target) as f:
                f.write(tag * 200)
                time.sleep(delay)
        except Exception as exc:
            errors.append(exc)

    slow = threading.Thread(target=write, args=("a", 0.30))
    fast = threading.Thread(target=write, args=("b", 0.02))
    slow.start()
    fast.start()
    slow.join()
    fast.join()

    assert len(errors) == 1, f"exactly one writer should lose, got {errors}"
    msg = str(errors[0])
    assert "two processes writing the same path" in msg
    assert "separate output paths" in msg

    # The surviving file is still one writer's complete output, never a mixture.
    content = target.read_text()
    assert len(set(content)) == 1, "the winner's file must not be interleaved"
    assert not (tmp_path / "race.json.part").exists(), "no temporary file may survive"


# ── durability, which the mutation sweep found nothing was asserting ───────────────────
def test_the_data_is_fsynced_before_the_rename(tmp_path, monkeypatch):
    """Found unprotected on 2026-08-03: dropping the fsync broke no test in the suite.

    Ordering is the whole guarantee. `os.replace` is atomic with respect to the directory
    entry, so after a crash the target is either the old file or the new one, never a mix. But
    that only holds if the new file's CONTENTS reached the disk first: rename the entry while
    the bytes are still in the page cache and a power loss leaves a complete-looking file full
    of zeroes, which is worse than the truncated file this whole helper exists to prevent.

    An fsync has no in-process observable effect, so this asserts the call and its order rather
    than the physics. That is weaker than a behavioural test and is the strongest available.
    """
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append("replace"), real_replace(a, b))[1])

    target = tmp_path / "out.json"
    with atomic_write(target) as f:
        f.write('{"a": 1}')

    assert order == ["fsync", "replace"], f"expected fsync before replace, got {order}"
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


def test_no_rename_happens_when_the_body_raises(tmp_path, monkeypatch):
    """The other half of the ordering contract: a failed write must never reach the target."""
    renamed = []
    monkeypatch.setattr(os, "replace", lambda a, b: renamed.append((a, b)))

    target = tmp_path / "out.json"
    def write_then_fail():
        with atomic_write(target) as f:
            f.write("half")
            raise ValueError("boom")

    with pytest.raises(ValueError):
        write_then_fail()

    assert not renamed, "a failed write was renamed over the target"
    assert not target.exists()
