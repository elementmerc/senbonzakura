# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The CPU-versus-accelerator parity gate's comparison logic.

The gate itself needs two devices and a few seconds of model time, so it is exercised on real
hardware rather than here. What IS tested here is the part that decides the verdict, and above all
the controls: a gate whose entire output is "these two agree" has not been shown to work until it
has been seen saying no.

One of these tests exists because the first version of the `drop` control silently removed nothing.
It zeroed direction slot -1, and a position that kept two of four directions already has two slots
of exact zeros, so the gate correctly reported no change and the control reported the gate as
toothless. The control was the broken part. That is the failure this file is shaped around.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "parity_check", Path(__file__).resolve().parent.parent / "tools" / "parity_check.py")
parity = importlib.util.module_from_spec(_spec)
sys.modules["parity_check"] = parity
_spec.loader.exec_module(parity)

POSITIONS, KMAX, H = 5, 4, 64


def _dirs(seed=0, filled=(1, 2, 2, 4, 4)):
    """A `dirs_multi`-shaped tensor: [positions, KMAX, H], padded with exact zeros."""
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(POSITIONS, KMAX, H)
    for p, k in enumerate(filled):
        block = torch.randn(k, H, generator=g)
        out[p, :k] = block / block.norm(dim=1, keepdim=True)
    return out


def test_a_run_agrees_with_itself():
    d = _dirs()
    rows = parity.compare_directions(d, d)
    assert len(rows) == POSITIONS
    assert all(c.same for _p, c in rows)


def test_padding_slots_do_not_become_shared_dimensions():
    """A position holding two of four directions must compare as rank 2, not rank 4."""
    rows = parity.compare_directions(_dirs(), _dirs())
    assert [c.rank_a for _p, c in rows] == [1, 2, 2, 4, 4]


def test_the_rotate_control_must_still_pass():
    """A rotated basis is the same subspace, and a gate that failed it would measure bookkeeping.

    This is the control that proves the gate is comparing subspaces rather than vectors. It is the
    one control here whose correct outcome is agreement.
    """
    d = _dirs()
    rows = parity.compare_directions(d, d, control="rotate")
    assert all(c.same for _p, c in rows), "a rotated basis was reported as a real difference"


@pytest.mark.parametrize("control", ["replace", "drop"])
def test_the_breaking_controls_are_caught(control):
    d = _dirs()
    rows = parity.compare_directions(d, d, control=control)
    disagreeing = [p for p, c in rows if not c.same]
    assert disagreeing, f"the '{control}' control changed nothing the gate could see"


def test_the_drop_control_removes_a_direction_that_was_actually_there():
    """THE BUG THIS FILE EXISTS FOR.

    Zeroing slot -1 removes nothing at a position that never filled it, so the control has to find
    an occupied slot. Asserted by counting: exactly one real direction must disappear.
    """
    d = _dirs()
    rows = parity.compare_directions(d, d, control="drop")
    ranks_before = [c.rank_a for _p, c in rows]
    ranks_after = [c.rank_b for _p, c in rows]
    assert sum(ranks_before) - sum(ranks_after) == 1, (
        f"expected exactly one direction to be dropped, ranks went {ranks_before} to {ranks_after}")


def test_the_drop_control_refuses_rather_than_silently_doing_nothing():
    """When no position holds two directions there is nothing to drop, and pretending otherwise is
    how the first version of this control failed.
    """
    thin = _dirs(filled=(1, 1, 1, 1, 1))
    with pytest.raises(RuntimeError, match="at least two directions"):
        parity.compare_directions(thin, thin, control="drop")


def test_an_unknown_control_is_refused():
    d = _dirs()
    with pytest.raises(ValueError, match="unknown control"):
        parity.compare_directions(d, d, control="wobble")


def test_a_genuinely_different_run_is_caught_without_any_control():
    """The case the gate is for: two devices that really do disagree."""
    rows = parity.compare_directions(_dirs(seed=1), _dirs(seed=2))
    assert not any(c.same for _p, c in rows)


def test_a_missing_device_is_a_skip_rather_than_a_pass(monkeypatch, capsys):
    """A machine with no card must not report the accelerated path as verified."""
    monkeypatch.setattr(parity.torch.cuda, "is_available", lambda: False)
    assert parity.main(["--device", "cuda"]) == 77
    assert "SKIP, not a pass" in capsys.readouterr().out


def test_weights_are_compared_by_magnitude_not_by_subspace():
    """The two halves need different instruments and must not be swapped.

    Two baked models are supposed to be the same tensors, so a rotation IS a real difference here,
    which is the opposite of what it means for directions.
    """
    a = {"w": torch.zeros(4, 4)}
    assert parity.compare_weights(a, {"w": torch.zeros(4, 4)}) == (0.0, "")
    worst, where = parity.compare_weights(a, {"w": torch.full((4, 4), 0.5)})
    assert worst == pytest.approx(0.5)
    assert where == "w"


def test_a_tensor_missing_from_the_second_bake_is_infinite_not_zero():
    """Absent must never read as identical, which a max over a shorter loop would give."""
    worst, where = parity.compare_weights({"w": torch.zeros(2)}, {})
    assert worst == float("inf")
    assert "missing" in where
