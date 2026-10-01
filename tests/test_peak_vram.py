# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Measuring another process's VRAM, without changing the quantity being measured.

The arm being measured is a competing tool in a subprocess, often in a container. Reading the card
through `torch.cuda` would initialise a CUDA context in the measuring process, and a context costs
a few hundred MB. On the 6 GB card this project targets that is a measurable slice of the budget
the arm is being judged on, so the sampler shells out instead.

That is the one place in this repository where two mechanisms read the same fact on purpose, and
the reason is written at `resources.cuda_used_bytes_external`: the governor reads in-process at
4 Hz during our own run, and this reads out-of-process at well under 1 Hz during somebody else's.
"""
from __future__ import annotations

import contextlib
import threading

from senbonzakura import resources


def test_the_peak_is_the_largest_sample_not_the_last():
    values = iter([100, 900, 300])
    with resources.PeakVram(interval_s=0.001, reader=lambda: next(values, 300)) as p:
        for _ in range(60):          # let the thread get its samples in
            if p.samples >= 3:
                break
            threading.Event().wait(0.01)
    assert p.peak_bytes == 900, f"peak was {p.peak_bytes}, so a later smaller reading won"


def test_the_baseline_is_recorded_and_not_subtracted():
    """Reported BESIDE the peak, never taken off it. The baseline is what was resident a moment
    earlier, not what that other process held throughout, so a difference presented as "what this
    tool used" would be a guess wearing a measurement's clothes.
    """
    with resources.PeakVram(interval_s=10, reader=lambda: 500) as p:
        pass
    assert p.baseline_bytes == 500
    assert p.peak_bytes == 500, "the peak was adjusted by the baseline"


def test_a_card_that_cannot_be_read_yields_none_and_a_sample_count_of_zero():
    """NONE, NEVER ZERO. Zero reads as "used no memory", which is the exact shape of claim this
    project keeps withdrawing. The sample count is published so a reader can tell a measurement
    from a card that was never polled.
    """
    with resources.PeakVram(interval_s=10, reader=lambda: None) as p:
        pass
    assert p.peak_bytes is None
    assert p.baseline_bytes is None
    assert p.samples == 0
    assert p.as_record() == {"peak_vram_mib": None, "baseline_vram_mib": None, "vram_samples": 0}


def test_the_sampler_does_not_outlive_the_body_even_on_an_exception():
    """A thread still polling after the arm has gone would attribute the NEXT arm's memory to this
    one, which is the quiet version of the defect rather than the loud one.
    """
    def the_arm_dies():
        raise RuntimeError("the arm died")

    p = resources.PeakVram(interval_s=0.001, reader=lambda: 1)
    with contextlib.suppress(RuntimeError), p:
        the_arm_dies()
    assert p._thread is not None and not p._thread.is_alive(), (
        "the sampling thread survived the body it was measuring")


def test_the_record_is_mib_and_rounded():
    with resources.PeakVram(interval_s=10, reader=lambda: 3 * 1024 * 1024 * 1024) as p:
        pass
    assert p.as_record()["peak_vram_mib"] == 3072


def test_reading_the_card_never_imports_torch(monkeypatch):
    """THE WHOLE POINT OF THE SECOND MECHANISM, asserted rather than trusted to a comment.

    If this ever routes through `resources._torch`, the measuring process takes a CUDA context and
    the arm it is judging loses that memory. The test fails loudly rather than the number drifting.
    """
    def explode():
        raise AssertionError("the external VRAM reader imported torch")

    monkeypatch.setattr(resources, "_torch", explode)
    resources.cuda_used_bytes_external(0)        # must not raise


def test_a_missing_nvidia_smi_is_none_rather_than_an_exception(monkeypatch):
    """A machine with no card still has to finish the arm. The column goes blank; the run does not
    die halfway through a sweep because an instrument was absent.
    """
    import subprocess

    def no_such_binary(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", no_such_binary)
    assert resources.cuda_used_bytes_external(0) is None


def test_garbage_from_the_tool_is_none_rather_than_a_number(monkeypatch):
    import subprocess

    def garbage(*_a, **_k):
        return subprocess.CompletedProcess([], 0, stdout="not a number\n", stderr="")

    monkeypatch.setattr(subprocess, "run", garbage)
    assert resources.cuda_used_bytes_external(0) is None
