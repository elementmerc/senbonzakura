# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The planted models do what their docstring says, checked WITHOUT the instrument.

If a fixture's plant were wrong, the instrument would be graded against a false answer key and
could pass or fail for the wrong reason. So each plant is verified here by direct arithmetic on the
toy stack, importing nothing from the attribution engine.
"""
import pytest
import torch

from senbonzakura import attribution_fixtures as fx


def _readout(stack, x):
    with torch.no_grad():
        return (stack(x)[:, -1, :] @ stack.read)


def _build(name, seed=0):
    spec = fx.SPECS[name]
    return spec, fx.ToyStack(spec, seed=seed), fx.make_prompts(300, 300, seed=11)


def _auc(a, b):
    a, b = a[:, None], b[None, :]
    return float(((a > b).float() + 0.5 * (a == b).float()).mean())


def test_expected_statuses_are_bare_literals_that_agree_with_the_specs():
    assert set(fx.EXPECTED) == set(fx.SPECS)
    for name, spec in fx.SPECS.items():
        assert fx.EXPECTED[name] == spec.expected_status, name
    assert set(fx.EXPECTED.values()) == {
        "localised", "diffuse", "redundant", "absent", "direction_inert",
        "readout_invalid", "unresolved"}


def test_one_block_separates_the_classes_and_only_block_five_writes_the_behaviour():
    spec, stack, (harmful, harmless) = _build("one_block")
    assert _auc(_readout(stack, harmful), _readout(stack, harmless)) > 0.9
    assert set(spec.blocks) == {5}


def test_absent_has_no_separation_at_all():
    _, stack, (harmful, harmless) = _build("absent")
    assert 0.35 < _auc(_readout(stack, harmful), _readout(stack, harmless)) < 0.65


def test_spread_blocks_carry_equal_shares():
    spec, _, _ = _build("spread")
    amplitudes = {amp for _, amp, _ in spec.blocks.values()}
    assert len(spec.blocks) == fx.N_BLOCKS and len(amplitudes) == 1


def test_redundant_backup_covers_for_block_three_and_is_silent_otherwise():
    # Remove block 3's behaviour write by hand and the backup restores the readout; remove the
    # backup alone and the readout is unchanged; remove both and the behaviour is gone.
    spec, stack, (harmful, _) = _build("redundant")
    full = _readout(stack, harmful).mean()

    def without(*blocks):
        saved = {}
        for k in blocks:
            saved[k] = stack.layers[k].amplitude
            stack.layers[k].amplitude = 0.0
        out = _readout(stack, harmful).mean()
        for k, amp in saved.items():
            stack.layers[k].amplitude = amp
        return out

    assert without(3) == pytest.approx(float(full), abs=1e-5)
    assert without(11) == pytest.approx(float(full), abs=1e-5)
    # Zeroing the backup's amplitude means it writes max(0 - h, 0) = 0, so with block 3 also
    # zeroed nothing reaches the channel.
    assert without(3, 11) < float(full) - 1.0


def test_orthogonal_class_contrast_is_dominated_by_the_decoy_axis():
    spec, stack, (harmful, harmless) = _build("orthogonal")
    with torch.no_grad():
        diff = stack(harmful).mean((0, 1)) - stack(harmless).mean((0, 1))
    unit = diff / diff.norm()
    assert abs(float(unit[fx.DECOY])) > 0.999
    assert abs(float(unit[fx.BEH_B])) < 0.02
    # ...while the behaviour readout still separates the classes, so there IS a behaviour.
    assert _auc(_readout(stack, harmful), _readout(stack, harmless)) > 0.9


def test_plants_are_deterministic_for_a_seed():
    a = fx.make_prompts(10, 10, seed=3)
    b = fx.make_prompts(10, 10, seed=3)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_an_unknown_block_kind_fails_loudly():
    with pytest.raises(ValueError, match="unknown block kind"):
        fx.ToyBlock("nonsense", 1.0, 0, 0)(torch.zeros(1, 2, fx.WIDTH))


def test_a_mixer_writes_only_into_coordinates_nothing_reads():
    block = fx.ToyBlock("mixer", 0.0, 0, 0)
    h = torch.randn(4, 2, fx.WIDTH)
    out = block(h)[0]
    assert torch.equal(out[..., :fx.FIRST_FREE], h[..., :fx.FIRST_FREE])
