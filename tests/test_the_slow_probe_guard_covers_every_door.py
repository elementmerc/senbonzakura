# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The hours-long-probe refusal guards every way into the loop, and reads where the weights are.

TWO DEFECTS, ONE GUARD

First, it was called from one caller. `refuse_a_slow_probe` was written for the generation loop in
this module and wired into the abliterate path alone, while `senbonzakura capability MODEL
--device cpu` reached the same loop with the same 200 items at 512 tokens, which the guard's own
constant puts at about four hours, and started without a word. The guard exists because CI was
killed at 900 seconds with no way to tell working from wedged; a user on a laptop gets the same
silence and no timeout to rescue them.

Second, it read the wrong thing. `--device` is a request, not a statement about where the weights
are: the loader passes `device_map="auto"` and accelerate dispatches whatever will not fit in
VRAM to host RAM or to disk. Generation then runs at host speed on a run that never looked like a
CPU run. So there is a second check, after the load, reading the map accelerate actually built.
The shape is `preflight_snapshot_ram`'s: estimate from what is cheap to read, then check once for
real when the thing being estimated is resident.
"""
from __future__ import annotations

import inspect

import pytest

from senbonzakura import capability


class _Model:
    def __init__(self, dmap):
        self.hf_device_map = dmap


# ── the fraction that is not on an accelerator ───────────────────────────────────────
def test_a_model_wholly_on_the_card_is_not_offloaded():
    assert capability.offloaded_share(_Model({"model.layers.0": 0, "model.layers.1": "cuda:0"})) == 0


def test_a_model_wholly_in_host_ram_is_all_of_it():
    assert capability.offloaded_share(_Model({"": "cpu"})) == 1


def test_disk_counts_as_host_speed():
    """A layer paged off an SSD on every forward pass is slower than one in RAM, not faster."""
    assert capability.offloaded_share(_Model({"a": "disk", "b": "cuda:0"})) == 0.5


def test_a_model_with_no_map_says_nothing_rather_than_zero():
    """A single-device load has no map at all, and "cannot say" is not "all on the card"."""
    assert capability.offloaded_share(object()) is None


# ── the post-load refusal ────────────────────────────────────────────────────────────
def test_an_offloaded_model_is_refused_although_the_device_said_cuda():
    """THE DEFECT: `--device cuda` with half the layers in host RAM never reached the guard."""
    with pytest.raises(SystemExit) as e:
        capability.refuse_a_slow_probe_after_load(_Model({"a": "cpu", "b": "cuda:0"}), 200, 512)
    message = str(e.value)
    assert "50% of this model's layers" in message
    assert "--slow-probe-ok" in message and "hours" in message


def test_a_model_on_the_card_passes_straight_through():
    capability.refuse_a_slow_probe_after_load(_Model({"a": "cuda:0"}), 200, 512)


def test_a_small_enough_probe_passes_even_when_offloaded():
    """The refusal is about the wait, not about the offload."""
    capability.refuse_a_slow_probe_after_load(_Model({"a": "cpu"}), 2, 64)


def test_the_flag_turns_it_into_a_warning_that_still_names_the_cost():
    said = []
    capability.refuse_a_slow_probe_after_load(_Model({"a": "cpu"}), 200, 512,
                                              allowed=True, log=said.append)
    assert said and "hours" in said[0] and "--slow-probe-ok" in said[0]


def test_a_probe_that_is_switched_off_is_not_refused_for_hours_it_will_not_spend():
    """The same condition the probe itself uses, mirrored rather than approximated."""
    capability.refuse_a_slow_probe_after_load(_Model({"a": "cpu"}), 0, 512)
    capability.refuse_a_slow_probe_after_load(_Model({"a": "cpu"}), 200, 512, spec="")


# ── both doors ───────────────────────────────────────────────────────────────────────
def test_the_capability_command_calls_both_halves_in_the_right_order():
    """Asserted from the source because the alternative needs a loaded model.

    `main` reaches the loader through `.cli`, which imports torch, optuna and transformers at
    module scope, so a behavioural test here would be a test of the stub. What matters and is
    checkable is the ORDER: the cheap refusal before the load, the real one after it, and both
    before the generation loop.
    """
    body = inspect.getsource(capability.main)
    cheap = body.index("refuse_a_slow_probe(")
    load = body.index("load_model_and_tokenizer(")
    real = body.index("refuse_a_slow_probe_after_load(")
    generate = body.index("generate_with_truncation(")
    assert cheap < load < real < generate, (
        "the estimate belongs before the download and the real check after the load")


def test_the_command_can_answer_the_refusal_it_raises():
    """A refusal naming a flag the command does not have is a dead end."""
    a = capability.build_parser().parse_args(["m", "--slow-probe-ok"])
    assert a.slow_probe_ok is True
    assert capability.build_parser().parse_args(["m"]).slow_probe_ok is False
