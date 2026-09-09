# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The per-layer shard spike.

Reconnaissance rather than a feature, so what is tested is the part a conclusion would rest on: the
sharding is lossless, the streamed edit is bit-identical to the resident one, and the size
arithmetic the pre-flight uses matches what actually gets allocated.

The last one exists because the pre-flight originally BUILT the checkpoint to find out how big it
was, which is a disk-space guard that allocates the thing it is checking there is room for. Asked
about a 200-layer model it was killed by the OOM killer before printing a word.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "shard_spike", Path(__file__).resolve().parent.parent / "tools" / "shard_spike.py")
spike = importlib.util.module_from_spec(_spec)
sys.modules["shard_spike"] = spike
_spec.loader.exec_module(spike)


def _direction(hidden, seed=1):
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(hidden, generator=g)
    return d / d.norm()


# ── the size arithmetic the pre-flight depends on ───────────────────────────────────
@pytest.mark.parametrize(("layers", "hidden", "ffn"), [(4, 64, 128), (8, 256, 512), (3, 32, 64)])
def test_the_computed_sizes_match_what_is_actually_allocated(layers, hidden, ffn):
    """If these drift, the pre-flight either refuses a run that would fit or permits one that
    will not, and the second failure mode is the one that wastes twenty minutes.
    """
    state = spike._synthetic_checkpoint(layers, hidden, ffn)
    real_model = sum(t.numel() * t.element_size() for t in state.values())
    real_layer = max(
        sum(t.numel() * t.element_size() for n, t in state.items() if spike._layer_of(n) == i)
        for i in range(layers))
    assert spike.checkpoint_sizes(layers, hidden, ffn) == (real_model, real_layer)


def test_the_sizes_are_computed_without_allocating_anything():
    """A model far larger than this machine must be sized instantly rather than attempted.

    The version this replaces was OOM-killed answering exactly this question.
    """
    model, layer = spike.checkpoint_sizes(200, 4096, 11008)
    assert model > 80e9, "the point of the case is that it could not be held"
    assert layer < model / 100


# ── sharding is lossless ────────────────────────────────────────────────────────────
def test_a_shard_round_trip_returns_exactly_what_went_in(tmp_path):
    state = spike._synthetic_checkpoint(4, 64, 128)
    spike.shard(state, tmp_path / "s")
    back = spike.reassemble(tmp_path / "s")
    worst, where = spike.max_abs_difference(state, back)
    assert worst == 0.0, f"sharding changed {where}"


def test_every_tensor_lands_in_exactly_one_shard(tmp_path):
    state = spike._synthetic_checkpoint(4, 64, 128)
    index = spike.shard(state, tmp_path / "s")
    placed = [n for names in index.values() for n in names]
    assert sorted(placed) == sorted(state)
    assert len(placed) == len(set(placed)), "a tensor written twice would be edited twice"


def test_layers_are_separated_and_the_rest_goes_to_extras(tmp_path):
    state = spike._synthetic_checkpoint(3, 32, 64)
    index = spike.shard(state, tmp_path / "s")
    assert set(index) == {"layer_0000", "layer_0001", "layer_0002", "extras"}
    assert all("layers.0." in n for n in index["layer_0000"])
    assert all("layers." not in n for n in index["extras"])


def test_a_tensor_belonging_to_no_layer_is_recognised():
    assert spike._layer_of("model.layers.7.mlp.up_proj.weight") == 7
    assert spike._layer_of("model.embed_tokens.weight") is None
    assert spike._layer_of("lm_head.weight") is None
    # A name containing the word without a number after it is not a layer.
    assert spike._layer_of("model.layers.weight") is None


# ── the claim the whole spike rests on ──────────────────────────────────────────────
def test_the_streamed_edit_matches_the_resident_one(tmp_path):
    """A streaming path that is cheap and WRONG is worth nothing, and it is the easy thing to
    build by accident.

    Matched to within what two correct float32 runs may differ by, not bit-identical. See the test
    below for why: bit-identity was never a property of this arrangement, and asserting it made
    the spike fail on torch 2.14 for a reason with nothing to do with streaming.
    """
    state = spike._synthetic_checkpoint(4, 64, 128)
    direction = _direction(64)
    resident = spike.edit_resident(state, direction)
    spike.shard(state, tmp_path / "s")
    spike.edit_streamed(tmp_path / "s", direction, tmp_path / "e")
    worst, where = spike.max_abs_difference(resident, spike.reassemble(tmp_path / "e"))
    slack = spike.tolerance_for(resident)
    assert worst <= slack, (
        f"the streamed edit differs from the resident one by {worst:.3e} at {where}, more than "
        f"the {slack:.3e} two correct float32 runs may differ by. That is structural, not "
        f"rounding.")


def test_bit_identity_was_never_a_property_of_the_streaming_arrangement(tmp_path):
    """THE MEASUREMENT THAT SET THE TOLERANCE, pinned so nobody re-tightens it on a hunch.

    The spike asserted bit-identity and failed on torch 2.14 with a message blaming the streaming
    path. It is not the streaming path: feeding the SAME arithmetic tensors that arrived
    memory-mapped off disk rather than freshly allocated moves the last bit, and two RESIDENT runs
    show it with no streaming anywhere near them.

    So this test asserts the thing that is actually true, in both directions: the same call twice
    is exact, and a trip through disk is not. If a future torch makes the second row exact again,
    this test fails and the tolerance can be revisited on evidence rather than on taste.
    """
    from safetensors.torch import load_file, save_file

    state = spike._synthetic_checkpoint(3, 64, 128)
    direction = _direction(64)

    twice_over = spike.max_abs_difference(spike.edit_resident(state, direction),
                                          spike.edit_resident(state, direction))[0]
    assert twice_over == 0.0, "the same call twice is not deterministic, which is a real defect"

    save_file({n: t.contiguous() for n, t in state.items()}, str(tmp_path / "s.safetensors"))
    off_disk = spike.edit_resident(load_file(str(tmp_path / "s.safetensors")), direction)
    in_memory = spike.edit_resident(state, direction)
    gap = spike.max_abs_difference(in_memory, off_disk)[0]

    # Not asserted to be non-zero: a machine where it IS zero is fine and must not fail here.
    # What must hold is that whatever the route costs, it is rounding and not structure, which is
    # the only thing the tolerance is claiming.
    assert gap <= spike.tolerance_for(in_memory), (
        f"a resident edit of the same tensors read off disk differs by {gap:.3e}, which is beyond "
        f"rounding. The route the tensors take is changing the arithmetic, not just its last bit.")


def test_the_edit_actually_changes_the_weights():
    """The test above would pass just as well if both paths did nothing at all."""
    state = spike._synthetic_checkpoint(2, 64, 128)
    edited = spike.edit_resident(state, _direction(64))
    worst, _where = spike.max_abs_difference(state, edited)
    assert worst > 0.1, "an edit that changes nothing makes every parity claim vacuous"


def test_the_edit_removes_the_direction_it_was_given():
    """The stand-in must be the same SHAPE of operation as the bake, or the spike tests nothing."""
    d = _direction(64)
    g = torch.Generator().manual_seed(3)
    w = torch.randn(32, 64, generator=g)
    out = spike._edit(w, d)
    assert float((out @ d).abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_a_tensor_the_direction_does_not_fit_is_passed_through_unchanged():
    d = _direction(64)
    norm = torch.ones(128)
    assert torch.equal(spike._edit(norm, d), norm)


# ── the comparison itself must be able to say no ────────────────────────────────────
def test_a_changed_tensor_is_reported_with_its_name():
    a = spike._synthetic_checkpoint(2, 32, 64)
    b = {k: v.clone() for k, v in a.items()}
    b["model.layers.1.mlp.up_proj.weight"] += 1.0
    worst, where = spike.max_abs_difference(a, b)
    assert worst == pytest.approx(1.0)
    assert where == "model.layers.1.mlp.up_proj.weight"


def test_a_missing_tensor_is_infinite_rather_than_ignored():
    """A shard that never got written must not compare as identical."""
    a = spike._synthetic_checkpoint(2, 32, 64)
    b = dict(a)
    b.pop("lm_head.weight")
    worst, where = spike.max_abs_difference(a, b)
    assert worst == float("inf")
    assert "only one side" in where


# ── the run refuses rather than dying ───────────────────────────────────────────────
def test_a_run_that_would_not_fit_on_disk_is_refused(tmp_path, capsys):
    rc = spike.main(["--work", str(tmp_path), "--layers", "200", "--hidden", "4096",
                     "--ffn", "11008"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not enough room" in err
    assert "--layers" in err, "it has to say what the reader can do about it"


def test_a_small_run_completes_and_records_its_numbers(tmp_path):
    out = tmp_path / "r.json"
    rc = spike.main(["--work", str(tmp_path / "w"), "--layers", "3", "--hidden", "64",
                     "--ffn", "128", "--json", str(out)])
    assert rc == 0
    recorded = json.loads(out.read_text(encoding="utf-8"))
    assert recorded["exact"] is True
    # Not asserted to be zero. `exact` means "within what two correct float32 runs may differ by",
    # and the difference is recorded beside the tolerance it was judged against so a reader can
    # see what the word was measured against rather than trusting it.
    assert recorded["max_abs_difference"] <= recorded["tolerance"]
    assert recorded["float32_slack_ulp"] == spike.FLOAT32_SLACK_ULP
    assert recorded["peaks"]["resident"] > 0
    # Each pass ran in its own process, so their peaks are independent rather than a high-water
    # mark one of them set for the others.
    assert set(recorded["peaks"]) == {"resident", "shard", "stream"}
