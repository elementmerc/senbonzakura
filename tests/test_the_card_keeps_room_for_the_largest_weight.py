# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A card that holds every activation can still fail on the copy of one weight.

accelerate places a weight that does not fit on the card in host RAM or on disk, and copies it onto the
card for each forward pass that uses it. The activation reserve covers the forward pass's own memory but
not that copy, so the largest weight is reserved as well. The output head of a wide-vocabulary model is
the usual largest one.
"""
import types

from senbonzakura import modelload, resources

GIB = 1 << 30


def config(layers=28, hidden=3584, vocab=152064, inter=18944, **extra):
    """A decoder config of the shape the arithmetic reads. Defaults are Qwen2.5-7B."""
    return types.SimpleNamespace(num_hidden_layers=layers, hidden_size=hidden,
                                 vocab_size=vocab, intermediate_size=inter, **extra)


def _one_card(monkeypatch, device_bytes):
    monkeypatch.setattr(resources, "host_memory_budget", lambda **k: (None, "unmeasured"))
    import accelerate.utils
    monkeypatch.setattr(accelerate.utils, "get_max_memory",
                        lambda *a, **k: {0: device_bytes, "cpu": 60 * 10**9})


def test_the_largest_weight_is_the_output_head_of_a_wide_vocabulary_model():
    got = modelload._largest_module_bytes(config(), 2)
    assert got == 152064 * 3584 * 2
    assert got > 18944 * 3584 * 2


def test_the_largest_weight_is_the_widest_mlp_projection_when_that_is_wider_than_the_vocabulary():
    got = modelload._largest_module_bytes(config(vocab=1000, inter=40000), 4)
    assert got == 40000 * 3584 * 4


def test_a_family_that_does_not_name_its_intermediate_size_is_measured_at_four_times_the_width():
    c = types.SimpleNamespace(num_hidden_layers=2, hidden_size=64, vocab_size=10)
    assert modelload._largest_module_bytes(c, 2) == max(10, 4 * 64) * 64 * 2


def test_a_config_that_does_not_name_the_shape_has_no_largest_weight():
    assert modelload._largest_module_bytes(types.SimpleNamespace(num_hidden_layers=2), 2) is None
    assert modelload._largest_module_bytes(config(vocab="152064"), 2) is None


def test_the_reserve_grows_by_exactly_the_largest_weight_when_the_dtype_is_known(monkeypatch):
    _one_card(monkeypatch, 48 * GIB)
    c = config()
    plain = modelload._corrected_max_memory([].append, c)
    wide = modelload._corrected_max_memory([].append, c, dtype_bytes=2)
    assert plain[0] - wide[0] == modelload._largest_module_bytes(c, 2)


def test_without_a_dtype_the_reserve_is_the_activations_alone_as_it_always_was(monkeypatch):
    _one_card(monkeypatch, 48 * GIB)
    c = config()
    out = modelload._corrected_max_memory([].append, c)
    assert 48 * GIB - out[0] == modelload._activation_headroom_bytes(c)


def test_the_largest_weight_is_reported_in_plain_words(monkeypatch):
    _one_card(monkeypatch, 48 * GIB)
    said = []
    modelload._corrected_max_memory(said.append, config(), dtype_bytes=2)
    assert any("the largest weight is" in line and "reserved beside the activations" in line for line in said)


def test_an_unnamed_shape_reserves_nothing_extra_and_changes_nothing(monkeypatch):
    _one_card(monkeypatch, 48 * GIB)
    unnamed = types.SimpleNamespace(num_hidden_layers=2)
    assert modelload._corrected_max_memory([].append, unnamed, dtype_bytes=2) is None


def test_the_reserve_on_a_small_card_is_still_capped_and_the_cap_is_still_said(monkeypatch):
    _one_card(monkeypatch, 6 * GIB)
    said = []
    out = modelload._corrected_max_memory(said.append, config(), dtype_bytes=2)
    assert out[0] == 6 * GIB - int(6 * GIB * modelload.HEADROOM_MAX_FRACTION)
    assert any("more than" in line and "left to the capture governor" in line for line in said)
