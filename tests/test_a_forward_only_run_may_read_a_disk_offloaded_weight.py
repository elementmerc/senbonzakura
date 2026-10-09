# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The write-safety guard must refuse a bake and must not refuse a forward pass.

THE DEFECT THIS CLOSES, and it was the whole of v0.5's feature gap. `stream-extract` exists so a
checkpoint too large for card plus host RAM can still have its refusal directions extracted: a
forward pass only READS weights, so accelerate's disk offload is available to it where it is not
available to the bake. It reaches the model through `Abliterator.__init__`, which walks every layer
to label its residual writers, and that walk resolved each weight through `_real_tensor`, which
REFUSES a disk-backed tensor because an in-place edit to one is written to a copy that is discarded
before the next forward pass. The refusal is correct for a bake and was guarding nothing on a
forward-only run, so the one path built for checkpoints this large could not load one.

HOW IT WAS FOUND, on 2026-10-09, on a rented A40 with a 145 GB checkpoint: the download finished,
the placement correction worked, 43.5 GB landed on the card, the rest spilled to disk, and then the
constructor raised `this decoder layer (type Qwen2DecoderLayer) has no residual-writing projection
in EITHER position`, with both positions reporting the disk-offload refusal.

WHY NO TEST CAUGHT IT, which is the part worth keeping. The feature's existing test proves the
directions a streamed run finds are byte-identical to a resident run's, and it has to use a model
small enough to stay resident, because a model that spills is a model the test machine cannot hold.
So the test proved the arithmetic and could not reach the path. The guard needs a tensor whose two
reads differ, and that is a property of the OFFLOAD MAP rather than of the model, which is what
makes it faked here: a fake owner whose `weights_map` returns a fresh tensor per read reproduces
the exact condition without a 145 GB download.
"""
import pytest
import torch

from senbonzakura import cli


class _FreshEveryRead:
    """An offload map that materialises a new tensor on each lookup, as a disk-backed one does."""

    def __init__(self, shape=(4, 4)):
        self.shape = shape
        self.reads = 0

    def __getitem__(self, name):
        self.reads += 1
        return torch.zeros(self.shape)


class _StableMap:
    """A CPU-offload map, which hands back one object, so in-place edits reach the next forward."""

    def __init__(self, shape=(4, 4)):
        self.tensor = torch.zeros(shape)

    def __getitem__(self, name):
        return self.tensor


class _Hook:
    def __init__(self, weights_map):
        self.weights_map = weights_map


class _Offloaded(torch.nn.Module):
    """A module whose weight is on meta with its real copy behind an accelerate hook."""

    def __init__(self, weights_map):
        super().__init__()
        self.weight = torch.zeros(4, 4, device="meta")
        self._hf_hook = _Hook(weights_map)


class _MetaOnly(torch.nn.Module):
    """On meta with no offload map at all, which is disk offload with no in-memory cache."""

    def __init__(self):
        super().__init__()
        self.weight = torch.zeros(4, 4, device="meta")


# ── the guard still refuses a bake, which is the half that must not regress ───────────

def test_a_bake_is_still_refused_a_disk_backed_weight():
    """THE ORIGINAL GUARD, asserted first and on purpose. An in-place edit to a tensor that is
    re-read from disk on every access lands on a temporary and is discarded, so the model would
    come out unabliterated with nothing reporting it. That is the failure the refusal exists for
    and the forward-only fix must not have widened it.
    """
    owner = _Offloaded(_FreshEveryRead())
    with pytest.raises(ValueError) as e:
        cli._real_tensor(owner, "weight")
    msg = str(e.value)
    assert "disk-offloaded" in msg
    assert "discarded before the next forward pass" in msg


def test_a_bake_is_still_refused_a_meta_weight_with_no_resident_copy():
    owner = _MetaOnly()
    with pytest.raises(ValueError) as e:
        cli._real_tensor(owner, "weight")
    assert "meta device with no resident offload copy" in str(e.value)


def test_a_bake_still_accepts_a_stably_mapped_weight():
    """CPU offload hands back one object, so it supports in-place editing and must keep working."""
    stable = _StableMap()
    owner = _Offloaded(stable)
    got = cli._real_tensor(owner, "weight")
    assert got is stable.tensor


def test_for_writing_defaults_to_true_so_nothing_is_relaxed_by_omission():
    """The safe value is the default. A caller that says nothing gets the bake's strictness, which
    is what every call site in the editing path relies on without passing anything.
    """
    owner = _Offloaded(_FreshEveryRead())
    with pytest.raises(ValueError):
        cli._real_tensor(owner, "weight")        # no keyword at all


# ── and now lets a forward-only caller through ────────────────────────────────────────

def test_a_forward_only_caller_gets_the_disk_backed_weight(recwarn):
    """The fix. A fresh copy per read is perfectly fine for somebody who only reads, and what the
    caller needs from it is shape, which a materialised copy has.
    """
    owner = _Offloaded(_FreshEveryRead())
    got = cli._real_tensor(owner, "weight", for_writing=False)
    assert got is not None
    assert tuple(got.shape) == (4, 4)
    assert not got.is_meta, "a materialised copy was available, so it should be the one returned"


def test_a_forward_only_caller_gets_the_meta_tensor_when_there_is_no_copy_at_all():
    """A meta tensor carries shape and dtype, which is all a structural walk reads. Refusing here
    would decline to describe a model we can perfectly well run a forward pass on.
    """
    owner = _MetaOnly()
    got = cli._real_tensor(owner, "weight", for_writing=False)
    assert got.is_meta
    assert tuple(got.shape) == (4, 4)


def test_the_forward_only_path_reads_rather_than_raising_and_does_not_write():
    """The map is consulted, which is the proof the value came from the offload copy and not from
    the meta placeholder, and nothing is assigned back onto the module.
    """
    fresh = _FreshEveryRead()
    owner = _Offloaded(fresh)
    cli._real_tensor(owner, "weight", for_writing=False)
    assert fresh.reads >= 1, "the offload map was never consulted"
    assert owner.weight.is_meta, "the module's own parameter must be left exactly as it was"


# ── the whole resolver chain carries the intent, not just the leaf ────────────────────

@pytest.mark.parametrize("fn", [
    "_real_tensor", "_owned_weight", "_attn_outproj", "_conv_outproj",
    "_block_outproj", "_mlp_downprojs", "layer_attn_writers", "layer_downproj", "layer_writers",
])
def test_every_function_on_the_path_takes_the_intent(fn):
    """A gap anywhere in the chain puts the refusal back, and the failure would look identical:
    the constructor's walk raises and the model appears to have no residual writers at all. The
    chain is asserted as a whole because testing only the leaf is what let this ship.
    """
    import inspect
    sig = inspect.signature(getattr(cli, fn))
    assert "for_writing" in sig.parameters, f"{fn} does not pass write-intent on"
    assert sig.parameters["for_writing"].default is True, (
        f"{fn} defaults to something other than the safe value")


def test_the_two_walkers_that_never_materialise_a_weight_do_not_take_it():
    """`refuse_unrecognised_writers` and `layer_composition` read parameters WITHOUT materialising
    them, so they work on a disk-offloaded model already and a flag on them would be dead. Asserted
    so nobody adds one for symmetry, and so that if either ever starts resolving tensors this test
    fails and says why.
    """
    import inspect
    for fn in ("refuse_unrecognised_writers", "layer_composition"):
        sig = inspect.signature(getattr(cli, fn))
        assert "for_writing" not in sig.parameters, (
            f"{fn} grew a write-intent parameter. If it now resolves real tensors it needs one "
            f"threaded properly; if it does not, the parameter is dead and misleads a reader")
