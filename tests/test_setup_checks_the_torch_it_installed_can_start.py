# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`setup --apply` verifies that the torch it just installed can initialise CUDA.

**MEASURED ON THE OPERATOR'S OWN MACHINE, 2026-10-12.** A Windows laptop with an RTX 3060 Laptop
GPU and a WSL2 Ubuntu inside. The driver reports a CUDA 13.4 ceiling (WSL's `nvidia-smi` spells it
`CUDA UMD Version: 13.4`, which this project's parser does read) and the card is compute 8.6.

`pick_cuda_channel((13, 4), (8, 6))` returns `cu132`. Both constraints it models are satisfied. The
resulting `torch 2.14.1+cu132` installs cleanly and then **dies with SIGSEGV** on
`torch.cuda.is_available()`. `cu128` on the same machine, same driver, same WSL, returns True and
names the card.

Ruled out before the channel was blamed: numpy absent (it was, and installing numpy 2.5.4 changed
nothing) and a broken GPU passthrough (`/dev/dxg` present, `libcuda.so.1` in `/usr/lib/wsl/lib/`,
`ldconfig` seeing it).

**Why this is worse than an environment quirk.** `senbonzakura setup` is the command this project
advertises as the remedy for a torch that cannot see the card, and `docs/guide/install.md` puts the
Windows row in bold because that platform gets a CPU-only build by default. Without this check,
following our own advice there replaces a working CPU torch with one that crashes, prints "done",
and sends the reader to `doctor`, which reports no CUDA device and reads like the problem they
started with.

**THE PROBE RUNS IN A SUBPROCESS AND THAT IS THE DESIGN.** A segfault takes the interpreter with
it, so an in-process check cannot report one and a `try/except` around the call never runs. The
test below named `test_a_probe_that_dies_on_a_signal_is_a_crash` is the whole reason this file
exists; everything else here is scaffolding around it.

The third constraint is not modelled on purpose. On WSL the usable CUDA version is capped by the
`libcuda` stub the Windows driver ships, which can lag `nvidia-smi`. Sniffing for WSL and capping
the channel would be a guess about a moving target. Verifying that the build we installed can start
is the thing we actually care about and cannot be wrong about.
"""

import subprocess
import sys

import pytest

from senbonzakura import envsetup

# ── the step down ────────────────────────────────────────────────────────────────

def test_the_next_channel_down_is_the_next_older_one():
    tags = [t for _, t in envsetup.CUDA_CHANNELS]
    assert envsetup.next_channel_below(tags[0]) == tags[1]
    assert envsetup.next_channel_below("cu132") == "cu130"


def test_the_oldest_channel_has_nothing_below_it():
    """A refusal that offers a channel which does not exist is worse than offering none."""
    tags = [t for _, t in envsetup.CUDA_CHANNELS]
    assert envsetup.next_channel_below(tags[-1]) is None


def test_a_channel_we_do_not_publish_has_no_neighbour():
    assert envsetup.next_channel_below("cu999") is None
    assert envsetup.next_channel_below("") is None


# ── the probe ────────────────────────────────────────────────────────────────────

def test_a_usable_card_reads_as_ok(monkeypatch):
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake)
    state, detail = envsetup.cuda_initialises()
    assert state == "ok"
    assert "usable" in detail


def test_torch_reporting_no_device_is_not_a_crash(monkeypatch):
    """The two have different remedies, so they must not collapse into one verdict."""
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, envsetup.CUDA_PROBE_NO_DEVICE, "", "")
    monkeypatch.setattr(subprocess, "run", fake)
    state, _ = envsetup.cuda_initialises()
    assert state == "no-device"


@pytest.mark.parametrize("rc", [-11, 139, -6, 134])
def test_a_probe_that_dies_on_a_signal_is_a_crash(monkeypatch, rc):
    """THE TEST THIS FILE EXISTS FOR.

    `subprocess` reports a signal death as the negated signal number; a shell in between turns it
    into 128 plus the signal. Both spellings are the same event, and the one observed on the ROG
    was SIGSEGV, which is 11 and therefore -11 or 139.
    """
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, rc, "", "")
    monkeypatch.setattr(subprocess, "run", fake)
    state, detail = envsetup.cuda_initialises()
    assert state == "crashed", f"return code {rc} was not read as a crash"
    assert "signal" in detail


def test_a_probe_that_hangs_is_reported_as_unverified(monkeypatch):
    def fake(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(subprocess, "run", fake)
    state, _ = envsetup.cuda_initialises()
    assert state == "timeout"


def test_a_probe_that_cannot_be_launched_is_an_error_not_a_crash(monkeypatch):
    """An unreadable instrument is not evidence that the thing it measures is broken."""
    def fake(cmd, **kw):
        raise OSError("no such interpreter")
    monkeypatch.setattr(subprocess, "run", fake)
    state, detail = envsetup.cuda_initialises()
    assert state == "error"
    assert "OSError" in detail


def test_the_probe_really_runs_in_another_interpreter():
    """Driven for real against this interpreter, with no mocking of subprocess at all.

    The point is that the mechanism works, not what the answer is: this machine may or may not have
    a card. Anything other than these five states would mean the probe is misreading real output.
    """
    state, detail = envsetup.cuda_initialises(timeout=120)
    assert state in {"ok", "no-device", "crashed", "timeout", "error"}
    assert detail


def test_the_probe_does_not_run_in_this_process(monkeypatch):
    """A regression guard on the design rather than the behaviour.

    If somebody rewrites this as an in-process `torch.cuda.is_available()` inside a try/except, the
    crash it was written for becomes invisible again and every test above still passes.
    """
    seen = {}

    def fake(cmd, **kw):
        seen["argv"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake)
    envsetup.cuda_initialises()
    assert seen["argv"][0] == sys.executable, "the probe no longer launches an interpreter"
    assert "-c" in seen["argv"]
    assert "torch.cuda.is_available()" in seen["argv"][-1]


def test_the_probe_distinguishes_no_device_by_a_code_of_its_own():
    """`0` and `1` are both taken: 1 is what an uncaught exception exits with, so a torch that
    fails to import would be indistinguishable from a torch reporting no card.
    """
    assert envsetup.CUDA_PROBE_NO_DEVICE not in (0, 1, 2)


def test_a_real_signal_death_is_read_as_a_crash(tmp_path):
    """END TO END, against a process that genuinely dies on SIGSEGV rather than a mocked code.

    Every other crash test above hands the function a return code. This one makes a real process
    kill itself with the real signal, because the thing being verified is the reading of what the
    operating system reports, and that is exactly what a mocked `CompletedProcess` cannot check.
    Verified on 2026-10-12 to report signal 11, which is what the ROG's cu132 torch did.
    """
    fake = tmp_path / "fakepython"
    fake.write_text("#!/bin/sh\nkill -SEGV $$\n", encoding="utf-8")
    fake.chmod(0o755)
    state, detail = envsetup.cuda_initialises(python=str(fake), timeout=60)
    assert state == "crashed"
    assert "signal 11" in detail


def test_a_real_clean_exit_is_read_as_ok(tmp_path):
    """The control for the test above: the same mechanism, a process that exits 0."""
    fake = tmp_path / "okpython"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    state, _ = envsetup.cuda_initialises(python=str(fake), timeout=60)
    assert state == "ok"
