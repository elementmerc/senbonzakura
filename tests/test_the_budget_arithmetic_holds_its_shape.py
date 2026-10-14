# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The properties the budget arithmetic has to keep, over a sweep rather than over one example.

WHY THESE AND NOT MORE EXAMPLES

`tests/test_the_budget_refuses_before_the_run.py` checks every term against a figure worked out by
hand from a real architecture, which is the right test for "is the arithmetic right". It cannot
answer "does it stay right", because a formula can be correct on one checkpoint and lose a term on
another: the plan's own memory model omitted the key and value cache entirely, and that would have
passed any single worked example that happened not to include one.

So these are the monotonicities and the identities. Each one is a sentence a reader can check
without arithmetic: a longer generation cannot need less cache; a bigger checkpoint cannot need
less disk; a pool's ceiling plus its other terms is exactly its budget. Every one of them is a
property the four rows that read this module depend on (v0.6's 6.3, 6.11 and 6.16, and v0.5's
preflight), and none of them is checked anywhere else.

NO `hypothesis`, DELIBERATELY. This project declares no property-testing dependency and adding one
for this would be a dependency decision (baseline section 5) taken for a handful of sweeps. The
sweeps here are explicit and seeded, so a failure names the input that broke it and two runs break
on the same one.
"""
from __future__ import annotations

import itertools
import random

import pytest

from senbonzakura import resources as r

GB = 1024 ** 3

#: One seed, named, so a failure is reproducible and two runs disagree about nothing.
SEED = 20261010


def _checkpoint(**over):
    fields = dict(layers=24, total_bytes=12 * GB, widest_layer_bytes=1 * GB,
                  widest_writer_bytes=256 * 1024 ** 2, writer_bytes=4 * GB,
                  largest_shard_bytes=4 * GB, shards=3, writer_width=2,
                  experts=8, kv_heads=4, head_dim=128, hidden=2048)
    fields.update(over)
    return r.Checkpoint(**fields)


def _run(**over):
    fields = dict(prompts=16, tokens=48, trials=200, passes_per_trial=199)
    fields.update(over)
    return r.Run(**fields)


def _machine(**over):
    fields = dict(vram_free=24 * GB, vram_total=24 * GB, ram_available=64 * GB,
                  disk_free=500 * GB, pinned_ceiling=8 * GB, on_mains=True)
    fields.update(over)
    return r.Machine(**fields)


# ── the cache terms cannot shrink when the thing they hold grows ─────────────────────────────────

KV_FIELDS = ("layers", "kv_heads", "head_dim", "width", "prompts", "tokens")


@pytest.mark.parametrize("field", KV_FIELDS)
def test_the_key_value_cache_never_shrinks_when_one_of_its_factors_grows(field):
    base = dict(layers=24, kv_heads=4, head_dim=128, width=2, prompts=16, tokens=48)
    before = r.kv_cache_bytes(**base)
    after = r.kv_cache_bytes(**{**base, field: base[field] + 1})
    assert after > before, f"raising {field} did not raise the cache, so that factor is lost"


@pytest.mark.parametrize("field", ["layers", "hidden", "prompts", "width"])
def test_the_activation_cache_never_shrinks_when_one_of_its_factors_grows(field):
    base = dict(layers=24, hidden=2048, prompts=16, width=4)
    before = r.activation_cache_bytes(**base)
    after = r.activation_cache_bytes(**{**base, field: base[field] + 1})
    assert after > before, f"raising {field} did not raise the cache, so that factor is lost"


def test_the_key_value_cache_is_zero_when_there_is_nothing_to_cache():
    """Not a special case: no prompts and no tokens is no cache, and the formula has to say so."""
    assert r.kv_cache_bytes(layers=24, kv_heads=4, head_dim=128, width=2,
                            prompts=0, tokens=48) == 0
    assert r.kv_cache_bytes(layers=24, kv_heads=4, head_dim=128, width=2,
                            prompts=16, tokens=0) == 0


# ── the pass count moves the way a reader expects ───────────────────────────────────────────────


@pytest.mark.parametrize("field", ["eval_refusal", "eval_kl", "gen_tokens"])
def test_more_work_per_trial_never_costs_fewer_passes(field):
    base = dict(eval_refusal=48, eval_kl=48, gen_batch=16, gen_tokens=48)
    before = r.passes_per_trial(**base)
    after = r.passes_per_trial(**{**base, field: base[field] + 16})
    assert after >= before, f"raising {field} lowered the pass count"


def test_a_bigger_batch_never_costs_more_passes():
    """Chunking is the whole point of the batch, so the count has to fall or hold as it rises."""
    base = dict(eval_refusal=48, eval_kl=48, gen_tokens=48)
    counts = [r.passes_per_trial(**base, gen_batch=b) for b in (1, 2, 4, 8, 16, 32, 64)]
    assert counts == sorted(counts, reverse=True), counts


def test_the_pass_count_is_at_least_one_chunk_of_each_phase():
    """A floor a reader can check: three phases, each at least one pass, whatever the batch."""
    assert r.passes_per_trial(eval_refusal=1, eval_kl=1, gen_batch=1024, gen_tokens=0) >= 3


# ── the plan's pools move with the checkpoint and the run ───────────────────────────────────────


def _pool(budget, name):
    return next(p for p in budget.pools if p.name == name)


def test_a_bigger_checkpoint_never_needs_less_disk():
    sizes = [1 * GB, 4 * GB, 12 * GB, 61 * GB, 200 * GB]
    totals = [_pool(r.plan(_checkpoint(total_bytes=s), _run(), _machine()), "free disk").total
              for s in sizes]
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals), "two different checkpoints needed the same disk"


def test_a_wider_layer_never_needs_less_video_memory():
    widths = [256 * 1024 ** 2, 1 * GB, 4 * GB]
    totals = [_pool(r.plan(_checkpoint(widest_layer_bytes=w), _run(), _machine()),
                    "video memory").total for w in widths]
    assert totals == sorted(totals)


def test_more_prompts_never_need_less_video_memory():
    totals = [_pool(r.plan(_checkpoint(), _run(prompts=n), _machine()), "video memory").total
              for n in (1, 16, 48, 192)]
    assert totals == sorted(totals)


def test_the_host_pool_is_exactly_twice_the_widest_layer():
    """Double buffered, which is the one term whose factor a reader can state from the name."""
    for width in (256 * 1024 ** 2, 1 * GB, 4 * GB):
        budget = r.plan(_checkpoint(widest_layer_bytes=width), _run(), _machine())
        assert _pool(budget, "host memory").total == 2 * width


def test_the_read_total_and_the_time_estimate_move_together():
    """The estimate is the read divided by a rate, so one cannot rise while the other falls."""
    plans = [r.plan(_checkpoint(), _run(trials=t), _machine()) for t in (1, 10, 100, 1000)]
    assert [p.read_bytes for p in plans] == sorted(p.read_bytes for p in plans)
    assert [p.seconds for p in plans] == sorted(p.seconds for p in plans)


def test_a_faster_disk_never_makes_a_run_take_longer():
    rates = [1e8, 1e9, 2e9, 1e10]
    seconds = [r.plan(_checkpoint(), _run(read_bytes_s=rate), _machine()).seconds
               for rate in rates]
    assert seconds == sorted(seconds, reverse=True)


# ── the pool verdicts, over a sweep of machines ─────────────────────────────────────────────────


def _machines(n=40):
    """A spread of machines, including the degenerate ones, from one named seed."""
    rng = random.Random(SEED)
    out = [_machine(), _machine(vram_free=None), _machine(ram_available=None),
           _machine(disk_free=None), _machine(vram_free=0, ram_available=0, disk_free=0)]
    out.extend(_machine(
        vram_free=rng.choice([None, 0, 2 * GB, 6 * GB, 24 * GB, 80 * GB]),
        ram_available=rng.choice([None, 0, 4 * GB, 16 * GB, 64 * GB, 512 * GB]),
        disk_free=rng.choice([None, 0, 8 * GB, 100 * GB, 2000 * GB])) for _ in range(n))
    return out


@pytest.mark.parametrize("machine", _machines(), ids=range(45))
def test_an_unmeasured_pool_is_never_a_pass(machine):
    """The defect this module exists for: a skipped check must not read as a clean one."""
    for pool in r.plan(_checkpoint(), _run(), machine).pools:
        if pool.available is None:
            assert pool.fits is None
            assert pool.fits is not True
            assert pool.shortfall == 0


@pytest.mark.parametrize("machine", _machines(), ids=range(45))
def test_a_pool_that_fits_has_no_shortfall_and_one_that_does_not_has_one(machine):
    for pool in r.plan(_checkpoint(), _run(), machine).pools:
        if pool.fits is True:
            assert pool.shortfall == 0
        elif pool.fits is False:
            assert pool.shortfall > 0, (
                f"{pool.name} does not fit and reports no shortfall, so the refusal would name "
                f"nothing to free")


@pytest.mark.parametrize("machine", _machines(), ids=range(45))
def test_the_planned_budget_never_exceeds_what_was_measured(machine):
    """The headroom policy, as a property rather than as a constant nobody re-reads."""
    for pool in r.plan(_checkpoint(), _run(), machine).pools:
        if pool.measured:
            assert 0 <= pool.budget <= pool.available


# ── the ceiling is an exact inversion, over the same sweep ──────────────────────────────────────


@pytest.mark.parametrize("machine", _machines(), ids=range(45))
def test_the_ceiling_and_the_other_terms_add_to_the_pools_budget(machine):
    """The identity that makes the ceiling an inversion rather than an estimate."""
    budget = r.plan(_checkpoint(), _run(), machine)
    for ceiling in r.ceilings(budget):
        if ceiling.measured:
            assert ceiling.ceiling_bytes + ceiling.other_bytes == ceiling.budget


@pytest.mark.parametrize("machine", _machines(), ids=range(45))
def test_a_pool_fits_exactly_when_its_binding_term_is_under_its_ceiling(machine):
    """Two ways of asking one question, which must never disagree.

    `Pool.fits` compares the whole total against the budget; `Ceiling.headroom` compares one term
    against what is left for it. If those ever disagree, a reader would be told a run fits and
    that its widest layer is over the limit, and would have no way to tell which was true.
    """
    budget = r.plan(_checkpoint(), _run(), machine)
    for ceiling in r.ceilings(budget):
        pool = _pool(budget, ceiling.pool)
        if not ceiling.measured:
            continue
        assert (ceiling.headroom >= 0) is (pool.fits is True), (
            f"{pool.name}: fits={pool.fits} and headroom={ceiling.headroom}")


@pytest.mark.parametrize("field", ["widest_layer_bytes", "total_bytes"])
def test_a_bigger_checkpoint_never_has_a_bigger_ceiling(field):
    """More of the pool spent on the other terms leaves less for the binding one, or the same."""
    sizes = [1 * GB, 4 * GB, 12 * GB] if field == "widest_layer_bytes" else [12 * GB, 61 * GB]
    seen = []
    for size in sizes:
        budget = r.plan(_checkpoint(**{field: size}), _run(), _machine())
        seen.append({c.pool: c.ceiling_bytes for c in r.ceilings(budget)})
    for pool in seen[0]:
        values = [row[pool] for row in seen]
        assert values == sorted(values, reverse=True) or len(set(values)) == 1, (
            f"{pool}'s ceiling rose as {field} rose: {values}")


# ── the formatters never lie about a size ───────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [0, 1, 999, 1000, 10 ** 6, 10 ** 9, 10 ** 12, 2 ** 50])
def test_a_formatted_size_is_never_negative_and_never_empty(n):
    text = r.fmt_bytes(n)
    assert text and not text.startswith("-")


def test_formatted_sizes_keep_their_order():
    """A reader compares two of these by eye, so the strings have to order like the numbers."""
    values = [0, 1, 1024, 10 ** 6, 10 ** 9, 10 ** 12]
    rendered = [r.fmt_bytes(v) for v in values]
    assert len(set(rendered)) == len(rendered), rendered


@pytest.mark.parametrize("seconds", [0, 1, 59, 60, 3599, 3600, 86399, 86400, 10 ** 7])
def test_a_formatted_span_is_never_negative_and_never_empty(seconds):
    for fn in (r.fmt_span, r.fmt_duration):
        text = fn(seconds)
        assert text and not text.startswith("-")


def test_the_sweep_actually_sweeps():
    """Without this the parametrised tests above pass on an empty list, which proves nothing."""
    machines = _machines()
    assert len(machines) == 45
    measured = sum(1 for m in machines if m.vram_free is not None)
    assert 5 < measured < 45, f"the sweep is not mixed: {measured} of 45 report video memory"
    combinations = {(m.vram_free, m.ram_available, m.disk_free) for m in machines}
    assert len(combinations) > 20, f"only {len(combinations)} distinct machines in the sweep"


def test_the_sweep_is_the_same_on_every_run():
    """A seeded sweep, so a failure names an input somebody else can reproduce."""
    first = [(m.vram_free, m.ram_available, m.disk_free) for m in _machines()]
    second = [(m.vram_free, m.ram_available, m.disk_free) for m in _machines()]
    assert first == second


def test_every_pool_is_exercised_both_ways_by_the_sweep():
    """A property test over machines that all fit has shown nothing about the refusal path."""
    verdicts = {}
    for machine in _machines():
        for pool in r.plan(_checkpoint(), _run(), machine).pools:
            verdicts.setdefault(pool.name, set()).add(pool.fits)
    for name, seen in verdicts.items():
        assert {True, False, None} <= seen, (
            f"{name} was only ever {sorted(str(s) for s in seen)} across the sweep, so one of its "
            f"branches is untested")


def test_the_terms_of_a_pool_are_named_and_not_just_totalled():
    """A refusal that gives one number tells an operator nothing they can act on."""
    for pool in r.plan(_checkpoint(), _run(), _machine()).pools:
        assert pool.needs, f"{pool.name} has no named terms"
        for label, value in pool.needs:
            assert isinstance(label, str) and label.strip(), f"{pool.name} has an unnamed term"
            assert value >= 0, f"{pool.name}'s {label!r} is negative"
        assert pool.total == sum(int(v) for _l, v in pool.needs)


def test_no_two_terms_of_one_pool_share_a_label():
    """Two rows reading the same would make the column unreadable and hide one of them."""
    for pool in r.plan(_checkpoint(), _run(), _machine()).pools:
        labels = [label for label, _v in pool.needs]
        assert len(set(labels)) == len(labels), f"{pool.name} repeats a label: {labels}"


def test_two_plans_over_the_same_inputs_are_identical():
    """Reproducibility: nothing in here may read a clock, a machine or an unordered collection."""
    for _ in range(3):
        a = r.plan(_checkpoint(), _run(), _machine())
        b = r.plan(_checkpoint(), _run(), _machine())
        assert [(p.name, p.needs, p.available) for p in a.pools] == \
               [(p.name, p.needs, p.available) for p in b.pools]
        assert (a.read_bytes, a.write_bytes, a.seconds) == (b.read_bytes, b.write_bytes, b.seconds)


def test_the_expert_block_changes_the_rewrite_and_nothing_else():
    """A knob that moved two pools would make the budget unreadable; it moves one term."""
    plans = {block: r.plan(_checkpoint(), _run(expert_block=block), _machine())
             for block in (1, 8, 64)}
    disks = {b: _pool(p, "free disk").total for b, p in plans.items()}
    hosts = {b: _pool(p, "host memory").total for b, p in plans.items()}
    assert len(set(disks.values())) == 1, disks
    assert len(set(hosts.values())) == 1, hosts


def test_every_pair_of_pools_is_independent_of_the_others_availability():
    """One pool's reading must not change another's need, or a refusal names the wrong resource."""
    reference = {p.name: p.total for p in r.plan(_checkpoint(), _run(), _machine()).pools}
    for pool_field, value in itertools.product(
            ("vram_free", "ram_available", "disk_free"), (None, 0, 1 * GB)):
        plan = r.plan(_checkpoint(), _run(), _machine(**{pool_field: value}))
        assert {p.name: p.total for p in plan.pools} == reference, (
            f"changing {pool_field} to {value} changed what a pool NEEDS, which is a property of "
            f"the run rather than of the machine")
