# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The streaming budget at the size the acceptance run will meet, and at sizes far past it.

The checkpoint below is SYNTHETIC: Qwen3-30B-A3B's published shape (48 layers, hidden 2048, 128
experts of intermediate size 768, 32 query and 4 key/value heads of dimension 128, bfloat16),
worked out by hand into the byte counts `resources.Checkpoint` takes. No weight is read and no
model is downloaded. It is a statement about the arithmetic, not a measurement of that model, and
the writer figure is the fused-stack (conservative) reading of the expert layout.

Three claims, none of them a number a reader should quote as measured:

* the plan's goal is arithmetically reachable: the card pool of a 30B run asks for under 2 GB;
* a volume too small for the working copy plus the output is refused before the run starts;
* the budget grows only one way. Making a run larger never makes any pool need less, and the
  integer terms stay exact at sizes where a float would round.
"""
import pytest

from senbonzakura import resources

GB = 10 ** 9
BF16 = 2

ATTENTION = (2048 * 4096 + 2 * 2048 * 512 + 4096 * 2048) * BF16
EXPERTS = 128 * 3 * 2048 * 768 * BF16
LAYER = ATTENTION + EXPERTS
WRITER = 128 * 768 * 2048 * BF16
EMBEDDINGS = 2 * 151936 * 2048 * BF16


def thirty_b(**over):
    args = dict(layers=48, total_bytes=48 * LAYER + EMBEDDINGS, widest_layer_bytes=LAYER,
                widest_writer_bytes=WRITER,
                writer_bytes=48 * (WRITER + 4096 * 2048 * BF16),
                largest_shard_bytes=4 * GB, shards=16, writer_width=BF16, experts=128,
                kv_heads=4, head_dim=128, hidden=2048)
    args.update(over)
    return resources.Checkpoint(**args)


def a_run(**over):
    args = dict(prompts=48, tokens=48, trials=64, passes_per_trial=199)
    args.update(over)
    return resources.Run(**args)


def a_machine(**over):
    args = dict(vram_free=8 * GB, vram_total=8 * GB, ram_available=32 * GB,
                disk_free=500 * GB, pinned_ceiling=4 * GB, on_mains=True)
    args.update(over)
    return resources.Machine(**args)


def pool(budget, name):
    return next(p for p in budget.pools if p.name == name)


def test_a_thirty_billion_run_asks_the_card_for_well_under_eight_gigabytes():
    card = pool(resources.plan(thirty_b(), a_run(), a_machine()), "video memory")
    assert card.fits is True
    assert card.total < 2 * GB


def test_the_time_estimate_for_that_run_is_days_and_not_hours_or_months():
    days = resources.plan(thirty_b(), a_run(), a_machine()).seconds / 86400
    assert 1 < days < 30


def test_a_volume_too_small_for_the_copy_and_the_output_is_refused_naming_the_disk():
    small = a_machine(disk_free=100 * GB)
    with pytest.raises(resources.BudgetRefusedError, match="free disk"):
        resources.preflight(thirty_b(), a_run(), small, log=lambda *_: None)


def test_the_same_run_on_a_big_enough_volume_is_not_refused():
    resources.preflight(thirty_b(), a_run(), a_machine(), log=lambda *_: None)


def test_the_disk_refusal_is_about_the_disk_and_not_the_card():
    budget = resources.plan(thirty_b(), a_run(), a_machine(disk_free=100 * GB))
    assert pool(budget, "free disk").fits is False
    assert pool(budget, "video memory").fits is True
    assert pool(budget, "free disk").shortfall > 0


RUN_KNOBS = [("prompts", 96), ("tokens", 256), ("trials", 128), ("passes_per_trial", 400),
             ("expert_block", 32), ("sparsity", True)]
CHECKPOINT_KNOBS = [("layers", 96), ("total_bytes", 2 * (48 * LAYER)),
                    ("widest_layer_bytes", 2 * LAYER), ("widest_writer_bytes", 2 * WRITER),
                    ("writer_bytes", 10 ** 12), ("largest_shard_bytes", 8 * GB),
                    ("kv_heads", 8), ("head_dim", 256), ("hidden", 4096)]


def totals(budget):
    return ({p.name: p.total for p in budget.pools}, budget.read_bytes, budget.write_bytes)


def assert_no_term_shrinks(before, after):
    pools_before, read_before, write_before = totals(before)
    pools_after, read_after, write_after = totals(after)
    for name, need in pools_before.items():
        assert pools_after[name] >= need, name
    assert read_after >= read_before
    assert write_after >= write_before


@pytest.mark.parametrize(("knob", "bigger"), RUN_KNOBS)
def test_a_larger_run_never_asks_any_pool_for_less(knob, bigger):
    base = resources.plan(thirty_b(), a_run(), a_machine())
    assert_no_term_shrinks(base, resources.plan(thirty_b(), a_run(**{knob: bigger}), a_machine()))


@pytest.mark.parametrize(("knob", "bigger"), CHECKPOINT_KNOBS)
def test_a_larger_checkpoint_never_asks_any_pool_for_less(knob, bigger):
    base = resources.plan(thirty_b(), a_run(), a_machine())
    assert_no_term_shrinks(base, resources.plan(thirty_b(**{knob: bigger}), a_run(), a_machine()))


def test_a_smaller_expert_block_is_the_only_way_a_knob_lowers_the_card_demand():
    big = pool(resources.plan(thirty_b(), a_run(expert_block=32), a_machine()), "video memory")
    small = pool(resources.plan(thirty_b(), a_run(expert_block=1), a_machine()), "video memory")
    assert small.total < big.total


def test_the_integer_terms_stay_exact_at_a_quadrillion_bytes():
    total = 10 ** 18
    budget = resources.plan(thirty_b(total_bytes=total, largest_shard_bytes=10 ** 15),
                            a_run(), a_machine(disk_free=10 ** 19))
    disk = pool(budget, "free disk")
    assert disk.total == 2 * total + 10 ** 15 + 48 * resources.CHECKPOINT_BYTES_PER_LAYER
    assert budget.read_bytes == total * (2 + 64 * 200)
    assert budget.write_bytes == thirty_b().writer_bytes * 64 + total
    assert disk.fits is True
    assert pool(resources.plan(thirty_b(total_bytes=total), a_run(),
                               a_machine(disk_free=10 ** 18)), "free disk").fits is False
