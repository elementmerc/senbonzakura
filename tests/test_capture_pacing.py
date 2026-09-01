# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Activation capture under memory pressure.

`collect_resid` is the first GPU-heavy thing an abliteration run does, and until this was written it
was the only batched path with a hard-coded chunk size: every other path went through the governor
that shrinks on out-of-memory and waits when the card is full. A run could therefore die at the very
first step, after paying to load the weights and before a single trial, and the failure would land
on whoever was renting the card.

What these tests hold down is not "the governor exists" but the three things that can go wrong when
a chunked path starts re-running its chunks: values must not change, ORDER must not change, and an
empty input must fail loudly rather than return an empty tensor that later reads as a direction.
"""
import pytest
import torch

from senbonzakura import cli
from senbonzakura.resources import ResourceGovernor


def _prompts(n):
    # Deliberately uneven lengths: left-padding means batch composition affects the numerics, so a
    # test that used identical prompts could not tell a re-chunked run from a correct one.
    return [f"prompt number {i} " + ("longer " * (i % 5)) for i in range(n)]


def _paced_gov(abl, *, free_seq, oom_on=None):
    """A governor wired to a fake card, so shrink and pause are exercised with no GPU."""
    return ResourceGovernor(
        "cuda:0", lambda _m: None,
        max_batch=cli.CAPTURE_BATCH,
        mem_fn=free_seq, reclaim_fn=lambda: 0, own_fn=lambda: 0,
        empty_cache_fn=lambda: None, sleep_fn=lambda _s: None)


# ── the shape of the answer is unchanged by pacing ───────────────────────────────
def test_capture_returns_one_column_per_prompt_in_order(abl):
    ps = _prompts(37)                      # not a multiple of the ceiling
    got = abl.collect_resid(ps)
    assert got.shape[0] == abl.NL + 1
    assert got.shape[1] == len(ps)
    assert got.shape[2] == abl.H
    assert got.dtype == torch.float32
    # Each column must be the activation of ITS prompt. Captured one at a time, the columns are
    # unambiguous, so this is the reference the chunked run has to reproduce.
    one_at_a_time = torch.cat([abl.collect_resid([p]) for p in ps], 1)
    assert torch.allclose(got, one_at_a_time, atol=1e-5)


def test_capture_chunks_at_the_ceiling_rather_than_all_at_once(abl):
    """The whole point is bounded memory. A path that quietly captured everything in one forward
    would pass every value test above and still be the bug.
    """
    abl.collect_resid(_prompts(40))
    sizes = abl.capture_gov.report()["batch_sizes_used"]
    assert sizes, "no chunk sizes recorded: the capture did not go through the governor"
    assert max(sizes) <= cli.CAPTURE_BATCH
    assert sum(n * c for n, c in sizes.items()) == 40


# ── the failure this was written for ─────────────────────────────────────────────
def test_an_out_of_memory_shrinks_the_chunk_and_still_completes(abl, monkeypatch):
    ps = _prompts(20)
    reference = abl.collect_resid(ps)

    abl.capture_gov = _paced_gov(abl, free_seq=lambda: (8 * 1024**3, 8 * 1024**3))
    real_model = abl.model
    seen = []

    def exploding(**enc):
        n = enc["input_ids"].shape[0]
        seen.append(n)
        # Fails while the chunk is large, succeeds once the governor has halved it. This is the
        # real shape of a card that another process is sitting on.
        if n > 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real_model(**enc)

    monkeypatch.setattr(abl, "model", exploding)
    got = abl.collect_resid(ps)

    assert abl.capture_gov.oom_shrinks > 0, "the OOM was not detected as one"
    assert max(abl.capture_gov.report()["batch_sizes_used"]) <= 4
    assert any(n > 4 for n in seen), "the large attempt never happened, so nothing was recovered from"
    assert torch.allclose(got, reference, atol=1e-5), "recovery changed the activations"


def test_order_survives_a_shrink_midway(abl, monkeypatch):
    """The subtle one. The governor re-runs a failed chunk at a smaller size, so the boundaries move
    underneath the caller. If results were appended per CHUNK rather than per PROMPT, a shrink would
    silently reorder the cloud, and a reordered cloud still produces a plausible direction: the
    difference-of-means is order-invariant, so nothing downstream would ever raise.
    """
    ps = _prompts(40)
    reference = abl.collect_resid(ps)

    abl.capture_gov = _paced_gov(abl, free_seq=lambda: (8 * 1024**3, 8 * 1024**3))
    real_model = abl.model
    state = {"calls": 0, "failed": False}

    def fail_once_partway(**enc):
        # Fail on the SECOND forward only, so the shrink happens after a full chunk of results is
        # already collected. A per-chunk accumulator would splice the retry in at the wrong offset.
        state["calls"] += 1
        if state["calls"] == 2:
            state["failed"] = True
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real_model(**enc)

    monkeypatch.setattr(abl, "model", fail_once_partway)
    got = abl.collect_resid(ps)

    assert state["failed"], "the mid-run OOM never fired, so this asserted nothing"
    for j in range(len(ps)):
        assert torch.allclose(got[:, j, :], reference[:, j, :], atol=1e-5), (
            f"prompt {j} landed in the wrong column after a shrink")


def test_an_empty_prompt_list_is_refused_loudly(abl):
    """Returning an empty tensor here would make `mean(1)` produce NaNs, and a NaN direction runs
    the entire search on garbage while every stage reports success.
    """
    with pytest.raises(ValueError, match="no activations captured"):
        abl.collect_resid([])


# ── the two governors stay separate ──────────────────────────────────────────────
def test_capture_and_generation_do_not_share_a_governor(abl):
    """The artefact describes `batch_sizes_used` under `generation` as a property of generation.
    One shared counter would make that field describe two different passes, which is the defect
    class this project has withdrawn results over.
    """
    assert abl.capture_gov is not abl.gov
    abl.collect_resid(_prompts(20))
    assert abl.capture_gov.report()["batch_sizes_used"], "capture recorded nothing"
    assert not abl.gov.report()["batch_sizes_used"], "capture leaked into the generation record"


def test_capture_reports_what_the_artefact_needs():
    gov = ResourceGovernor("cpu", lambda _m: None, max_batch=cli.CAPTURE_BATCH, enabled=False)
    r = gov.report()
    for key in ("throttled", "max_batch", "batch_sizes_used", "oom_shrinks", "pauses"):
        assert key in r, f"the artefact's direction_capture block would be missing {key}"
