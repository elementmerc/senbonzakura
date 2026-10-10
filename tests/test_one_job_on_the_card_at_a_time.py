# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The GPU interlock, and the two readings it must never confuse.

WHY THIS FILE EXISTS

On 2026-10-05 two driver scripts ran against the same 6 GB card for over two hours, scoring the
same model at the same time. Four of that queue's eight failures were `rc=124`, the 90 minute
timeout, and all four fell inside that window: six hours of card time spent producing nothing,
because neither script could ask whether the other was already using the card.

THE CHECK THAT LOOKS RIGHT AND IS NOT, which is the thing most of these tests are about

`memory.used < threshold` passed on 2026-10-07 while a small model was training, and put a second
job on the card. A threshold cannot tell a card holding a finished process's cached blocks from a
card holding a live job, because a small job does not use much memory. Occupancy is a PROCESS
question: `--query-compute-apps` lists what is computing, and only an empty list means nothing is.

And under WSL that query's `used_memory` column prints `[N/A]`, so a parser that insists on a
number drops a live process and reports a free card. The presence of a ROW is the signal.

The third confusion is the one this module made in its own refusal message before these tests
existed: an unreadable card is not an empty one. "Nothing is computing on the card" is a claim;
"I could not ask" is the fact.

Nothing here touches a real GPU. `nvidia-smi` is faked, because the behaviour under test is how the
output is read rather than what a driver reports.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "research"))

# SKIPPED WHERE `fcntl` DOES NOT EXIST, which is Windows, and this guard is a fix.
#
# `gpu_lock` is built on POSIX advisory locking and imports `fcntl` at module scope, so importing
# it on Windows raised ModuleNotFoundError during COLLECTION. That is an error rather than a
# failure, and pytest reports it in a separate block from the failure list, so for as long as
# this job had failing tests the error sat underneath them unread. Fixing the three failures on
# 2026-10-10 left it as the only thing keeping the Windows job red.
#
# Keyed on the missing module rather than on `sys.platform`, because the dependency is fcntl and
# naming the platform would be a guess about which platforms lack it.
pytest.importorskip(
    "fcntl",
    reason="gpu_lock is built on POSIX advisory locking, which this platform has no fcntl for. "
           "The interlock it provides is meaningless without one, so there is nothing here to "
           "test rather than something untested.")
import gpu_lock

TOOL = Path(__file__).resolve().parents[1] / "tools" / "research" / "gpu_lock.py"


class _Smi:
    """A stand-in for one `nvidia-smi` invocation."""

    def __init__(self, stdout="", returncode=0, stderr="", raises=None):
        self.stdout, self.returncode, self.stderr, self.raises = stdout, returncode, stderr, raises

    def __call__(self, *_a, **_k):
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            [], self.returncode, stdout=self.stdout, stderr=self.stderr)


def _fake_smi(monkeypatch, smi):
    monkeypatch.setattr(gpu_lock.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_lock.subprocess, "run", smi)


# ── reading the card ─────────────────────────────────────────────────────────────────

def test_a_row_is_a_running_process(monkeypatch):
    _fake_smi(monkeypatch, _Smi("12345, python, 1213\n"))
    apps = gpu_lock.compute_apps()
    assert len(apps) == 1
    assert apps[0]["pid"] == 12345
    assert apps[0]["process_name"] == "python"


def test_a_live_process_reporting_no_memory_is_still_a_live_process(monkeypatch):
    """THE WSL CASE, and the reason this reads rows rather than numbers.

    Under WSL the `used_memory` column is `[N/A]`. A parser that requires a number here drops the
    row, sees no processes, and reports a free card to the next job in the queue.
    """
    _fake_smi(monkeypatch, _Smi("4242, python, [N/A]\n"))
    apps = gpu_lock.compute_apps()
    assert len(apps) == 1, "a row with no memory figure is still a process on the card"
    assert apps[0]["pid"] == 4242
    assert apps[0]["used_memory"] == "[N/A]"


def test_an_empty_list_is_the_only_reading_that_means_free(monkeypatch):
    _fake_smi(monkeypatch, _Smi("\n"))
    assert gpu_lock.compute_apps() == []
    _fake_smi(monkeypatch, _Smi("No running processes found\n"))
    assert gpu_lock.compute_apps() == []


@pytest.mark.parametrize("smi", [
    _Smi(raises=FileNotFoundError("nvidia-smi")),
    _Smi(returncode=9, stderr="driver not loaded"),
    _Smi(raises=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=15)),
])
def test_a_card_that_cannot_be_read_raises_rather_than_reporting_free(monkeypatch, smi):
    """UNKNOWN IS NOT FREE. Returning `[]` here would be the same error as the memory threshold:
    answering a question that was not asked, in the direction that lets a second job through.
    """
    _fake_smi(monkeypatch, smi)
    with pytest.raises(gpu_lock.CardUnreadableError):
        gpu_lock.compute_apps()


def test_no_nvidia_smi_at_all_says_so_in_the_message(monkeypatch):
    monkeypatch.setattr(gpu_lock.shutil, "which", lambda _n: None)
    with pytest.raises(gpu_lock.CardUnreadableError, match="not on PATH"):
        gpu_lock.compute_apps()


# ── the lock itself ──────────────────────────────────────────────────────────────────

def test_the_lock_excludes_a_second_holder(tmp_path, monkeypatch):
    _fake_smi(monkeypatch, _Smi("\n"))
    lock = str(tmp_path / "gpu.lock")
    with gpu_lock.gpu_lock(lock, label="first", log=lambda _m: None), \
            pytest.raises(TimeoutError, match="still busy"), \
            gpu_lock.gpu_lock(lock, label="second", wait=0, log=lambda _m: None):
        pytest.fail("two holders at once, which is the whole defect")


def test_the_lock_is_released_even_when_the_body_raises(tmp_path, monkeypatch):
    """A lock that outlives its job turns one wasted slot into every slot after it."""
    _fake_smi(monkeypatch, _Smi("\n"))
    lock = str(tmp_path / "gpu.lock")
    with pytest.raises(ValueError, match="planted"), \
            gpu_lock.gpu_lock(lock, log=lambda _m: None):
        raise ValueError("planted")
    with gpu_lock.gpu_lock(lock, wait=0, log=lambda _m: None):
        pass


def test_the_holder_is_recorded_so_a_refusal_can_name_it(tmp_path, monkeypatch):
    _fake_smi(monkeypatch, _Smi("\n"))
    lock = tmp_path / "gpu.lock"
    with gpu_lock.gpu_lock(str(lock), label="pass2", log=lambda _m: None):
        doc = json.loads(lock.read_text(encoding="utf-8"))
    assert doc["pid"] == os.getpid()
    assert doc["label"] == "pass2"
    assert doc["since"] > 0


def test_a_free_lock_over_a_busy_card_still_refuses(tmp_path, monkeypatch):
    """THE CASE THE LOCK CANNOT SEE on its own: a hand-run script, another user, or a job that
    started before this tool existed. The lock and the card are two independent facts.
    """
    _fake_smi(monkeypatch, _Smi("777, someone_elses_training, 5800\n"))
    with pytest.raises(TimeoutError), gpu_lock.gpu_lock(str(tmp_path / "gpu.lock"), wait=0, log=lambda _m: None):
        pytest.fail("took a card that somebody else was computing on")


def test_sharing_can_be_asked_for_explicitly(tmp_path, monkeypatch):
    """Off by default, and the default is the lesson rather than a preference."""
    _fake_smi(monkeypatch, _Smi("777, someone_elses_training, 5800\n"))
    with gpu_lock.gpu_lock(str(tmp_path / "gpu.lock"), wait=0, require_idle=False,
                           log=lambda _m: None):
        pass


def test_our_own_process_does_not_count_as_a_competitor(tmp_path, monkeypatch):
    _fake_smi(monkeypatch, _Smi(f"{os.getpid()}, python, 1200\n"))
    with gpu_lock.gpu_lock(str(tmp_path / "gpu.lock"), wait=0, log=lambda _m: None):
        pass


def test_there_is_no_wait_value_meaning_forever(tmp_path, monkeypatch):
    """Baseline section 2: every lock acquisition has a deadline. A negative or absurd wait is
    clamped into one attempt rather than becoming an unbounded block.
    """
    _fake_smi(monkeypatch, _Smi("\n"))
    lock = str(tmp_path / "gpu.lock")
    with gpu_lock.gpu_lock(lock, log=lambda _m: None), pytest.raises(TimeoutError), \
            gpu_lock.gpu_lock(lock, wait=-5, log=lambda _m: None):
        pytest.fail("a negative wait became a block")


# ── the refusal message, which is what a person actually reads ───────────────────────

def test_an_unreadable_card_is_not_described_as_an_empty_one(tmp_path):
    """THE ERROR THIS MODULE MADE IN ITS OWN MESSAGE, before these tests existed.

    With `apps` as None, meaning the card could not be read, the message used to say "nothing is
    computing on the card", which is a claim rather than the fact. The fact is that it could not
    ask, and the two have to read differently or the module is making the mistake it exists to
    prevent, in the very sentence explaining the refusal.
    """
    lock = tmp_path / "gpu.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "label": "holder", "since": 1.0}),
                    encoding="utf-8")
    unknown = gpu_lock.describe_refusal(str(lock), None)
    empty = gpu_lock.describe_refusal(str(lock), [])
    assert "could not be read" in unknown
    assert "nothing is computing" not in unknown
    assert "nothing is computing" in empty


def test_a_stale_lock_is_named_as_stale(tmp_path):
    lock = tmp_path / "gpu.lock"
    # A pid that cannot be running: 0 is never a user process here.
    lock.write_text(json.dumps({"pid": 0, "label": "ghost", "since": 1.0}), encoding="utf-8")
    said = gpu_lock.describe_refusal(str(lock), [])
    assert "GONE" in said
    assert "stale" in said


def test_an_unparseable_lock_file_is_not_read_as_a_free_one(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("not json at all", encoding="utf-8")
    assert gpu_lock.read_holder(str(lock)) is None
    assert "recorded no details" in gpu_lock.describe_refusal(str(lock), [])


# ── the command line, where a dead flag has already shipped once ─────────────────────

def test_the_command_is_split_on_the_first_bare_separator():
    """NOT `argparse.REMAINDER`, and this is a defect this file shipped for one test run.

    REMAINDER is greedy from the first positional, so `run --label x --share-ok -- cmd` parsed
    `--share-ok` as part of the COMMAND and the flag silently did nothing. The one flag whose
    entire job is to override a refusal, quietly dead, which is this project's most repeated defect
    shape: `--chat-template` on the abliterate path, `--capability-eval` reading a flag that did
    not exist, and four tool-call measures with no caller.
    """
    options, cmd = gpu_lock.split_argv(
        ["run", "--label", "x", "--share-ok", "--", "./score.sh", "--device", "cuda"])
    assert options == ["run", "--label", "x", "--share-ok"]
    assert cmd == ["./score.sh", "--device", "cuda"]


def test_a_flag_after_the_separator_belongs_to_the_command():
    options, cmd = gpu_lock.split_argv(["run", "--", "./x.sh", "--share-ok"])
    assert options == ["run"]
    assert cmd == ["./x.sh", "--share-ok"]


def test_no_separator_means_no_command():
    options, cmd = gpu_lock.split_argv(["check", "--index", "1"])
    assert options == ["check", "--index", "1"]
    assert cmd == []


def test_the_share_flag_actually_reaches_the_parser():
    """The assertion the defect above would have failed."""
    options, _ = gpu_lock.split_argv(["run", "--share-ok", "--", "true"])
    a = gpu_lock.build_parser().parse_args(options)
    assert a.share_ok is True


def test_run_with_no_command_is_refused():
    with pytest.raises(SystemExit, match="needs a command"):
        gpu_lock.main(["run", "--share-ok"])


# ── end to end, as a shell driver would call it ──────────────────────────────────────

def _run_tool(args, lock):
    return subprocess.run([sys.executable, str(TOOL), *args[:1], "--lock", str(lock), *args[1:]],
                          capture_output=True, text=True, timeout=120, check=False)


# THE CHILD COMMANDS ARE THIS INTERPRETER, NOT /bin/true, and that is a fix.
#
# These tests used `/bin/true` and `/bin/false`, which do not exist on macOS: there `true` and
# `false` live in /usr/bin, so the child never started and the tool returned 1 for a reason that
# had nothing to do with what was being tested. The macOS job carried that single failure from
# before 2026-10-09 and it went unlooked-at because a louder failure was being chased.
#
# `sys.executable` is the one command guaranteed to exist wherever the suite runs, and the exit
# status it is asked for is explicit, which is what these tests are actually about.
def _exits(code):
    """A command that does nothing and exits with `code`, on any platform."""
    return [sys.executable, "-c", f"raise SystemExit({code})"]


def _prints(text):
    """A command that prints `text`, on any platform."""
    return [sys.executable, "-c", f"print({text!r})"]


def test_the_commands_own_exit_status_is_passed_through(tmp_path):
    lock = tmp_path / "gpu.lock"
    assert _run_tool(["run", "--share-ok", "--", *_exits(0)], lock).returncode == 0
    assert _run_tool(["run", "--share-ok", "--", *_exits(1)], lock).returncode == 1
    assert _run_tool(["run", "--share-ok", "--", *_exits(7)], lock).returncode == 7


def test_a_busy_card_exits_four_so_a_shell_can_tell_it_apart(tmp_path):
    """The exit codes are the interface for a driver: 4 is busy, 5 is unreadable, anything else is
    the command's own status. A driver that cannot tell those apart retries the wrong thing.
    """
    import time

    lock = str(tmp_path / "gpu.lock")
    # `with`, because `Popen` with a pipe opens file objects and the suite runs under
    # `-W error::ResourceWarning`. Killing the child does not close them, and the first version of
    # this test leaked one and failed the gate, which is the gate doing its job.
    with subprocess.Popen(
            [sys.executable, str(TOOL), "run", "--lock", lock, "--share-ok", "--", "sleep", "10"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as holder:
        try:
            # Wait for the holder to SAY it has the lock, rather than sleeping a guessed interval.
            # A fixed sleep here is a flaky test on a loaded machine, and this project does not
            # add retry loops to races.
            for _ in range(200):
                if os.path.exists(lock) and gpu_lock.read_holder(lock):
                    break
                time.sleep(0.05)
            else:
                pytest.fail("the holder never took the lock, so the refusal was never tested")
            done = subprocess.run(
                [sys.executable, str(TOOL), "run", "--lock", lock, "--share-ok", "--",
                 *_prints("SHOULD NOT RUN")],
                capture_output=True, text=True, timeout=60, check=False)
        finally:
            holder.kill()
            holder.wait(timeout=30)
    assert done.returncode == 4, done.stdout + done.stderr
    assert "SHOULD NOT RUN" not in done.stdout
