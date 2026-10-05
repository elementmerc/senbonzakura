# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The streaming run's memory budget, and the thing it must never say.

TWO SEPARATE CLAIMS, TESTED APART.

**The arithmetic is right.** Every term is checked against a figure worked out by hand from a real
architecture, because a budget that is internally consistent and wrong is a budget that refuses
runs that fit and starts runs that cannot finish. Each term is also checked for being LOAD
BEARING: a test that passes with a term deleted has not tested that term, and this project has
shipped three things that were never once executed.

**It never reports clean on a question it did not ask.** The defect this file exists for was found
by running the command on a machine with no card: a run needing 84 GB of key and value cache
printed "this run fits on this machine", because the video memory pool was unmeasured and an
unmeasured pool cannot refuse. Every line above the headline said SKIPPED and the headline
overruled all of them. That is this project's most recurring defect shape, and the guard against
it is the last group below.

NO TORCH ANYWHERE IN HERE, on purpose. The arithmetic lives in `resources` and reads nothing, the
checkpoint reading lives in `streaming` and computes nothing, and the synthetic shards below are
written with `struct` and `json`. So the whole budget is testable on a machine with no card and no
deep learning stack, which is the machine most of this suite runs on.
"""
import json
import pathlib
import struct

import pytest

from senbonzakura import resources, streaming

#: Bytes per element for the dtypes the fixtures use. Deliberately local: the point of a fixture
#: is to be an independent account of the format, so sharing the module's own table would make a
#: wrong table agree with itself.
WIDTHS = {"F32": 4, "F16": 2, "BF16": 2}


def write_shard(path, tensors):
    """A real safetensors file of zeros, from `{name: (dtype, shape)}`."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        size = count * WIDTHS[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % streaming.ALIGNMENT)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)
    return path


def build_checkpoint(root, *, layers=4, hidden=256, inter=512, experts=0, heads=8, kv_heads=2,
                     config=True, shards=1):
    """A checkpoint directory with a config and one or more shards, holding no real weights."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if config:
        (root / "config.json").write_text(json.dumps({
            "num_hidden_layers": layers, "hidden_size": hidden,
            "num_attention_heads": heads,
            **({"num_key_value_heads": kv_heads} if kv_heads else {})}), encoding="utf-8")
    shared = {"model.embed_tokens.weight": ("BF16", (1000, hidden)),
              "model.norm.weight": ("BF16", (hidden,))}
    per_layer = {}
    for i in range(layers):
        per_layer[i] = {
            f"model.layers.{i}.self_attn.o_proj.weight": ("BF16", (hidden, hidden)),
            f"model.layers.{i}.self_attn.q_proj.weight": ("BF16", (hidden, hidden)),
            f"model.layers.{i}.input_layernorm.weight": ("BF16", (hidden,)),
        }
        if experts:
            per_layer[i][f"model.layers.{i}.mlp.experts.down_proj.weight"] = (
                "BF16", (experts, inter, hidden))
        else:
            per_layer[i][f"model.layers.{i}.mlp.down_proj.weight"] = ("BF16", (hidden, inter))
    groups = [{} for _ in range(max(1, shards))]
    groups[0].update(shared)
    for i in range(layers):
        groups[i % len(groups)].update(per_layer[i])
    for n, group in enumerate(groups):
        write_shard(root / f"model-{n:05d}.safetensors", group)
    return root


@pytest.fixture
def dense(tmp_path):
    return build_checkpoint(tmp_path / "dense")


@pytest.fixture
def moe(tmp_path):
    return build_checkpoint(tmp_path / "moe", experts=16)


def a_run(**over):
    """A run record with every count named, because a default count is a budget for another run."""
    args = dict(prompts=48, tokens=48, trials=64, passes_per_trial=199)
    args.update(over)
    return resources.Run(**args)


def a_machine(**over):
    """A machine where everything is measured and generous, so a test can starve one pool."""
    args = dict(vram_free=6 * 10 ** 9, vram_total=6 * 10 ** 9, ram_available=32 * 10 ** 9,
                disk_free=500 * 10 ** 9, pinned_ceiling=4 * 10 ** 9, on_mains=True)
    args.update(over)
    return resources.Machine(**args)


# ── the arithmetic, against figures worked out by hand ───────────────────────────────────────────

def test_the_key_value_cache_matches_the_figure_the_plan_states():
    """Qwen3-30B-A3B: 48 layers, 4 key/value heads, head dimension 128, bfloat16.

    The plan states 98 kB a token and 402 MB at 16 prompts of 256 tokens. Both are reproduced
    here, which is what makes this an independent check rather than a restatement.
    """
    per_token = resources.kv_cache_bytes(layers=48, kv_heads=4, head_dim=128, width=2,
                                         prompts=1, tokens=1)
    assert per_token == 98304                       # 96 KiB, which is the plan's 98 kB
    at_scale = resources.kv_cache_bytes(layers=48, kv_heads=4, head_dim=128, width=2,
                                        prompts=16, tokens=256)
    assert at_scale == 402653184                    # 402 MB
    assert 400e6 < at_scale < 404e6


def test_the_activation_cloud_matches_the_seventy_seven_megabyte_figure():
    """48 layers, 192 prompts, hidden size 2048, float32. The plan states 77 MB a side."""
    nbytes = resources.activation_cache_bytes(layers=48, hidden=2048, prompts=192)
    assert nbytes == 49 * 192 * 2048 * 4
    assert 76e6 < nbytes < 78e6


def test_the_pass_count_is_one_hundred_and_ninety_nine_for_the_plans_case():
    """48 prompts, batch 16, 48 tokens. The plan recounted 198; the objective as written gives 199.

    The extra pass is the KL slice's third chunk. Pinned here rather than left as prose so that a
    change to the objective's shape has to come past this number.
    """
    assert resources.passes_per_trial(eval_refusal=48, eval_kl=48, gen_batch=16,
                                      gen_tokens=48) == 199


@pytest.mark.parametrize(("over", "why"), [
    ({"gen_tokens": 96}, "more tokens is more decode steps"),
    ({"gen_batch": 48}, "a wider batch is fewer chunks"),
    ({"eval_refusal": 96}, "more prompts is more chunks"),
    ({"eval_kl": 96}, "the KL slice is chunked too"),
])
def test_every_input_to_the_pass_count_actually_moves_it(over, why):
    """LOAD BEARING, EACH ONE. A formula that ignores an argument it takes is a formula whose
    caller is passing something into nothing, which is how a dead flag survives a review.
    """
    base = dict(eval_refusal=48, eval_kl=48, gen_batch=16, gen_tokens=48)
    assert resources.passes_per_trial(**{**base, **over}) != resources.passes_per_trial(**base), why


def test_the_rewrite_working_set_shrinks_with_the_expert_block():
    """The block loop is the lever the 2026-09-22 work added, so the budget has to feel it."""
    checkpoint = resources.Checkpoint(
        layers=4, total_bytes=10 ** 9, widest_layer_bytes=10 ** 8,
        widest_writer_bytes=128 * 10 ** 6, writer_bytes=4 * 128 * 10 ** 6,
        largest_shard_bytes=10 ** 9, shards=1, writer_width=2, experts=128)
    whole = resources.rewrite_working_bytes(checkpoint, a_run(expert_block=128))
    blocked = resources.rewrite_working_bytes(checkpoint, a_run(expert_block=8))
    assert blocked * 16 == pytest.approx(whole, rel=1e-6), "a block of 8 in 128 is a sixteenth"
    sparse = resources.rewrite_working_bytes(checkpoint, a_run(expert_block=8, sparsity=True))
    assert sparse > blocked, "sparse surgery holds one more float32 tensor live"


def test_the_card_pool_carries_every_term_and_each_one_is_its_own_arithmetic(dense):
    """THE TERMS ARE NAMED AND THE NUMBERS ARE CHECKED, which is two assertions and not one.

    An earlier version of this file tested `kv_cache_bytes` in isolation and tested that the term
    was ABSENT when the config said nothing, and a mutant that dropped the term from the pool
    unconditionally passed both. A term computed correctly by a function nothing calls is the
    exact shape of defect this project keeps finding.
    """
    checkpoint = streaming.describe(dense)
    run = a_run()
    pool = next(p for p in resources.plan(checkpoint, run, a_machine()).pools
                if p.name == "video memory")
    terms = dict(pool.needs)
    assert set(terms) == {"the widest layer, resident", "the rewrite's float32 working set",
                          "the key and value cache", "the captured activations"}
    assert terms["the widest layer, resident"] == checkpoint.widest_layer_bytes
    assert terms["the rewrite's float32 working set"] == resources.rewrite_working_bytes(
        checkpoint, run)
    assert terms["the key and value cache"] == resources.kv_cache_bytes(
        layers=4, kv_heads=2, head_dim=32, width=2, prompts=48, tokens=48)
    assert terms["the captured activations"] == resources.activation_cache_bytes(
        layers=4, hidden=256, prompts=48)
    assert pool.total == sum(terms.values()), "the total is the sum of what it named"


def test_a_bigger_cache_makes_a_bigger_demand_on_the_card(dense):
    """LOAD BEARING. If the cache term were dropped the prompt count would reach nothing."""
    checkpoint = streaming.describe(dense)
    small = resources.plan(checkpoint, a_run(prompts=8, tokens=8), a_machine()).pools[0].total
    large = resources.plan(checkpoint, a_run(prompts=512, tokens=512), a_machine()).pools[0].total
    assert large > small * 10


def test_a_dense_writer_has_no_expert_axis_for_the_block_to_divide():
    checkpoint = resources.Checkpoint(
        layers=4, total_bytes=10 ** 9, widest_layer_bytes=10 ** 8,
        widest_writer_bytes=10 ** 6, writer_bytes=4 * 10 ** 6,
        largest_shard_bytes=10 ** 9, shards=1, writer_width=2, experts=0)
    assert (resources.rewrite_working_bytes(checkpoint, a_run(expert_block=8))
            == resources.rewrite_working_bytes(checkpoint, a_run(expert_block=128)))


def test_the_restore_read_is_counted_as_its_own_term():
    """Dropping the resident snapshot costs one model read a trial, and the plan put that at
    3.9 TB over 64 trials of a 61 GB checkpoint.

    ASSERTED AGAINST THE FORMULA WRITTEN OUT BY HAND, not against the same function with one
    argument changed. The first version of this test compared 199 passes against 198 and a mutant
    that deleted the restore term entirely passed it, because deleting the term moved both sides
    by the same amount.
    """
    total = 61 * 10 ** 9
    checkpoint = resources.Checkpoint(
        layers=48, total_bytes=total, widest_layer_bytes=10 ** 9,
        widest_writer_bytes=10 ** 8, writer_bytes=20 * 10 ** 9,
        largest_shard_bytes=5 * 10 ** 9, shards=16, writer_width=2, experts=128)
    read = resources.streaming_read_bytes(checkpoint, a_run(trials=64, passes_per_trial=198))
    # capture + rescore, then per trial the 198 forwards AND the one restore read.
    assert read == total * 2 + 64 * total * 199
    restore_share = 64 * total
    assert 3.8e12 < restore_share < 4.0e12, "the plan's 3.9 TB"
    assert read - (total * 2 + 64 * total * 198) == restore_share


def test_the_bake_writes_the_writers_and_the_output_and_nothing_else():
    total, writers = 61 * 10 ** 9, 20 * 10 ** 9
    checkpoint = resources.Checkpoint(
        layers=48, total_bytes=total, widest_layer_bytes=10 ** 9,
        widest_writer_bytes=10 ** 8, writer_bytes=writers,
        largest_shard_bytes=5 * 10 ** 9, shards=16, writer_width=2, experts=128)
    assert resources.streaming_write_bytes(checkpoint, a_run(trials=64)) == writers * 64 + total


def test_the_five_day_estimate_for_a_thirty_billion_search_is_reproduced():
    """The plan's headline: about 1.7 hours a trial and about five days over 64, at 1.95 GB/s.

    The number this reproduces is the one the spike is told to measure against, so it is worth
    pinning in code where a change to the model has to argue with it.
    """
    checkpoint = resources.Checkpoint(
        layers=48, total_bytes=61 * 10 ** 9, widest_layer_bytes=10 ** 9,
        widest_writer_bytes=10 ** 8, writer_bytes=20 * 10 ** 9,
        largest_shard_bytes=5 * 10 ** 9, shards=16, writer_width=2, experts=128)
    one = resources.plan(checkpoint, a_run(trials=1, passes_per_trial=198), a_machine())
    assert 1.6 * 3600 < one.seconds < 1.9 * 3600, "about 1.7 hours a trial"
    many = resources.plan(checkpoint, a_run(trials=64, passes_per_trial=198), a_machine())
    assert 4.0 * 86400 < many.seconds < 5.5 * 86400, "about five days over 64 trials"


# ── refusing a run that cannot finish ────────────────────────────────────────────────────────────

def test_a_run_that_does_not_fit_the_card_is_refused_with_the_shortfall(dense):
    checkpoint = streaming.describe(dense)
    short = a_machine(vram_free=1024)
    with pytest.raises(resources.BudgetRefusedError) as caught:
        resources.preflight(checkpoint, a_run(), short, log=lambda _m: None)
    message = str(caught.value)
    assert "video memory" in message
    assert "short by" in message
    assert "nothing has been written" in message.lower()


@pytest.mark.parametrize(("starve", "pool"), [
    ({"vram_free": 1024}, "video memory"),
    ({"ram_available": 1024}, "host memory"),
    ({"disk_free": 1024}, "free disk"),
])
def test_every_pool_can_be_the_one_that_refuses(dense, starve, pool):
    """LOAD BEARING, EACH POOL. A budget that only ever refuses on one resource has two pools
    that are decoration, and decoration is what this project's gate audits keep finding.
    """
    budget = resources.plan(streaming.describe(dense), a_run(), a_machine(**starve))
    assert budget.fits is False
    assert [p.name for p in budget.pools if p.fits is False] == [pool]


def test_a_run_that_fits_is_not_refused(dense):
    budget = resources.preflight(streaming.describe(dense), a_run(), a_machine(),
                                 log=lambda _m: None)
    assert budget.fits is True
    assert budget.verdict == "this run fits on this machine"


def test_the_budget_plans_into_nine_tenths_rather_than_all_of_it(dense):
    """The same headroom the resident snapshot guard has always used. Two preflights over one
    machine disagreeing about what full means is how an operator learns to ignore both.
    """
    checkpoint = streaming.describe(dense)
    budget = resources.plan(checkpoint, a_run(), a_machine())
    card = next(p for p in budget.pools if p.name == "video memory")
    exact = resources.plan(checkpoint, a_run(), a_machine(vram_free=card.total))
    assert exact.pools[0].fits is False, "needing exactly what is free is not fitting"


# ── the page-locked ceiling, which the exit gate asks the preflight to report ────────────────────

def test_a_host_store_over_the_ceiling_is_reported_as_over(dense):
    checkpoint = streaming.describe(dense)
    tiny = a_machine(pinned_ceiling=1024)
    side, ceiling, store = resources.plan(checkpoint, a_run(), tiny).pinned_verdict
    assert side == "over"
    assert ceiling == 1024
    assert store == 2 * checkpoint.widest_layer_bytes, "double buffered, so two layers"


def test_a_host_store_under_the_ceiling_is_reported_as_under(dense):
    budget = resources.plan(streaming.describe(dense), a_run(),
                            a_machine(pinned_ceiling=4 * 10 ** 9))
    assert budget.pinned_verdict[0] == "under"


def test_an_unmeasured_ceiling_is_reported_as_unknown_rather_than_fine(dense):
    budget = resources.plan(streaming.describe(dense), a_run(), a_machine(pinned_ceiling=None))
    assert budget.pinned_verdict[0] is None
    lines = []
    resources.report(budget, log=lines.append)
    assert any("not measured" in line for line in lines)


def test_the_report_says_over_the_ceiling_loudly(dense):
    lines = []
    resources.report(resources.plan(streaming.describe(dense), a_run(),
                                    a_machine(pinned_ceiling=1024)), log=lines.append)
    joined = " ".join(lines)
    assert "OVER it" in joined
    assert "stop overlapping" in joined


def test_a_ceiling_the_probe_never_found_is_reported_as_a_floor(dense):
    """`hit_limit` False means the probe ran out of budget, so the honest word is "at least"."""
    lines = []
    budget = resources.plan(streaming.describe(dense), a_run(),
                            a_machine(pinned_ceiling=4 * 10 ** 9, pinned_is_floor=True))
    resources.report(budget, log=lines.append)
    assert any("at least" in line for line in lines)


# ── power, wear and the time estimate in the report ──────────────────────────────────────────────

def test_being_on_battery_is_said_out_loud(dense):
    lines = []
    resources.report(resources.plan(streaming.describe(dense), a_run(), a_machine(on_mains=False)),
                     log=lines.append)
    joined = " ".join(lines)
    assert "battery" in joined
    assert "third" in joined, "the reader needs the size of the effect, not just the fact"


def test_unknown_power_is_not_reported_as_mains(dense):
    lines = []
    resources.report(resources.plan(streaming.describe(dense), a_run(), a_machine(on_mains=None)),
                     log=lines.append)
    assert any("power: unknown" in line for line in lines)


def test_mains_power_says_nothing_at_all(dense):
    lines = []
    resources.report(resources.plan(streaming.describe(dense), a_run(), a_machine(on_mains=True)),
                     log=lines.append)
    assert not any("battery" in line for line in lines)


def test_the_wear_note_is_a_fraction_of_a_rated_lifetime(dense):
    budget = resources.plan(streaming.describe(dense), a_run(), a_machine())
    assert budget.wear_fraction == budget.write_bytes / resources.WRITE_ENDURANCE_BYTES
    lines = []
    resources.report(budget, log=lines.append)
    assert any("rated lifetime" in line for line in lines)


def test_the_report_names_the_rate_it_used_and_that_it_was_one_machine(dense):
    lines = []
    resources.report(resources.plan(streaming.describe(dense), a_run(), a_machine()),
                     log=lines.append)
    joined = " ".join(lines)
    assert "1.95 GB/s" in joined
    assert "one machine and one disk" in joined, "an inherited constant has to say whose it is"


def test_a_measured_read_rate_replaces_the_inherited_one(dense):
    checkpoint = streaming.describe(dense)
    slow = resources.plan(checkpoint, a_run(read_bytes_s=1e8), a_machine())
    fast = resources.plan(checkpoint, a_run(read_bytes_s=1e10), a_machine())
    assert slow.seconds > 10 * fast.seconds * 0.99


# ── the headline must not overrule the lines above it ────────────────────────────────────────────

def test_an_unmeasured_pool_never_claims_the_run_fits_this_machine(dense):
    """THE DEFECT THIS FILE EXISTS FOR, planted exactly as it was found.

    A card that cannot be read, and a key and value cache far larger than any card. The budget
    must not refuse, because an unmeasurable machine is not a small one, and it must not say the
    run fits either, because nothing checked the one pool that would have stopped it.
    """
    checkpoint = streaming.describe(dense)
    blind = a_machine(vram_free=None, vram_total=None)
    budget = resources.plan(checkpoint, a_run(prompts=20000, tokens=4096), blind)
    assert budget.unmeasured == ["video memory"]
    assert budget.fits is True, "an unmeasured pool cannot refuse a run"
    assert budget.verdict != "this run fits on this machine"
    assert "unchecked rather than passed" in budget.verdict
    assert "video memory" in budget.verdict


def test_a_pool_with_no_reading_is_skipped_rather_than_passed(dense):
    budget = resources.plan(streaming.describe(dense), a_run(), a_machine(ram_available=None))
    pool = next(p for p in budget.pools if p.name == "host memory")
    assert pool.fits is None, "None is not True: a skipped check is not a pass"
    assert pool.measured is False
    assert pool.budget is None
    assert pool.shortfall == 0
    lines = []
    resources.report(budget, log=lines.append)
    assert any("SKIPPED rather than passed" in line for line in lines)


def test_a_skipped_pool_does_not_hide_a_measured_one_that_is_short(dense):
    budget = resources.plan(streaming.describe(dense), a_run(),
                            a_machine(vram_free=None, disk_free=1024))
    assert budget.fits is False
    assert "short of free disk" in budget.verdict


# ── reading a checkpoint, which is the other half of the boundary ────────────────────────────────

def test_describe_reads_the_shape_without_reading_a_weight(dense):
    checkpoint = streaming.describe(dense)
    assert checkpoint.layers == 4
    assert checkpoint.shards == 1
    assert checkpoint.writer_width == 2, "bfloat16 is two bytes an element"
    assert checkpoint.experts == 0, "a dense writer has no expert axis"
    assert checkpoint.hidden == 256
    assert checkpoint.kv_heads == 2
    assert checkpoint.head_dim == 32, "256 hidden over 8 heads"
    # o_proj is 256x256 and down_proj is 256x512, so the widest writer is the larger of the two.
    assert checkpoint.widest_writer_bytes == 256 * 512 * 2
    assert checkpoint.writer_bytes == 4 * (256 * 256 + 256 * 512) * 2


def test_describe_sees_a_fused_expert_stack_and_its_expert_count(moe):
    checkpoint = streaming.describe(moe)
    assert checkpoint.experts == 16
    assert checkpoint.widest_writer_bytes == 16 * 512 * 256 * 2


def test_the_widest_layer_is_the_widest_and_not_the_mean(tmp_path):
    """A checkpoint with one fat layer. Budgeting the mean fits every layer but one."""
    root = build_checkpoint(tmp_path / "ragged", layers=3)
    write_shard(root / "model-00001.safetensors",
                {"model.layers.1.mlp.extra_down_proj.weight": ("F32", (4096, 4096))})
    index = streaming.index_layers(root)
    widest_i, widest = index.widest()
    assert widest_i == 1
    assert widest > index.nbytes(0) * 10


def test_the_largest_shard_is_the_one_a_rewrite_has_to_make_room_for(tmp_path):
    root = build_checkpoint(tmp_path / "split", layers=4, shards=2)
    checkpoint = streaming.describe(root)
    assert checkpoint.shards == 2
    sizes = sorted(p.stat().st_size for p in root.glob("*.safetensors"))
    assert checkpoint.largest_shard_bytes == sizes[-1]


def test_describe_refuses_a_checkpoint_with_no_recognised_writer(tmp_path):
    root = pathlib.Path(tmp_path / "strange")
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "hidden_size": 16}),
                                      encoding="utf-8")
    write_shard(root / "model.safetensors",
                {"model.layers.0.odd.thing.weight": ("F32", (4, 4)),
                 "model.layers.1.odd.thing.weight": ("F32", (4, 4))})
    with pytest.raises(streaming.ShardError, match="residual-writing"):
        streaming.describe(root)


def test_the_duplicate_shard_refusal_names_paths_rather_than_basenames(tmp_path):
    """Found by pointing the command at a directory holding two checkpoints: both shards are
    called the same thing, and the message read "model.safetensors and model.safetensors".
    """
    build_checkpoint(tmp_path / "parent" / "a")
    build_checkpoint(tmp_path / "parent" / "b")
    with pytest.raises(streaming.ShardError) as caught:
        streaming.index_layers(tmp_path / "parent")
    message = str(caught.value)
    assert "a/model-00000.safetensors" in message
    assert "b/model-00000.safetensors" in message


# ── the geometry, where a missing term is a term silently dropped ────────────────────────────────

def test_a_config_without_grouped_query_attention_uses_the_attention_head_count(tmp_path):
    root = build_checkpoint(tmp_path / "nogqa", kv_heads=None, heads=8)
    hidden, kv_heads, head_dim = streaming.geometry(root)
    assert (hidden, kv_heads, head_dim) == (256, 8, 32)


def test_an_explicit_head_dim_beats_the_division(tmp_path):
    root = build_checkpoint(tmp_path / "explicit")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["head_dim"] = 64
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert streaming.geometry(root)[2] == 64, "the config's own number, not hidden over heads"


def test_a_nested_text_config_is_read_from_one_section_only(tmp_path):
    """A multimodal checkpoint nests the text model. Taking the layer count from there and the
    hidden size from the top level would build a budget out of two different models.
    """
    root = pathlib.Path(tmp_path / "multimodal")
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "hidden_size": 9999, "num_attention_heads": 1,
        "text_config": {"num_hidden_layers": 2, "hidden_size": 128, "num_attention_heads": 4},
    }), encoding="utf-8")
    write_shard(root / "model.safetensors",
                {f"model.layers.{i}.self_attn.o_proj.weight": ("BF16", (128, 128))
                 for i in range(2)})
    hidden, kv_heads, head_dim = streaming.geometry(root)
    assert (hidden, kv_heads, head_dim) == (128, 4, 32), "the text section, not the top level"


def test_a_checkpoint_whose_config_says_nothing_about_geometry_drops_the_terms(tmp_path):
    """The budget then omits the cache terms and the report shows what it could size.

    Omitted, not invented. A term guessed from a default is wrong by an unknown amount, which is
    worse than a missing term because it looks right.
    """
    root = pathlib.Path(tmp_path / "bare")
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"num_hidden_layers": 2}), encoding="utf-8")
    write_shard(root / "model.safetensors",
                {f"model.layers.{i}.self_attn.o_proj.weight": ("BF16", (64, 64))
                 for i in range(2)})
    checkpoint = streaming.describe(root)
    assert checkpoint.hidden is None
    assert checkpoint.kv_heads is None
    budget = resources.plan(checkpoint, a_run(), a_machine())
    labels = [label for label, _n in budget.pools[0].needs]
    assert "the key and value cache" not in labels
    assert "the captured activations" not in labels


def test_geometry_on_a_directory_with_no_config_says_nothing(tmp_path):
    empty = pathlib.Path(tmp_path / "empty")
    empty.mkdir()
    assert streaming.geometry(empty) == (None, None, None)


# ── the probes, which must never raise and must distinguish none from zero ───────────────────────

def test_host_ram_available_reports_a_positive_number_on_this_machine():
    nbytes = resources.host_ram_available()
    assert nbytes is None or nbytes > 0


def test_host_ram_available_falls_through_to_the_posix_probe(monkeypatch):
    """The Linux file is the first probe and not the only one, and the fallback has to work.

    Only `/proc/meminfo` is made to fail, because patching `open` for everything would break
    whatever else the interpreter was reading at the time.
    """
    import builtins
    real = builtins.open

    def refuse_meminfo(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            raise OSError("no /proc here")
        return real(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", refuse_meminfo)
    nbytes = resources.host_ram_available()
    assert nbytes is None or nbytes > 0


def test_own_footprint_reports_a_positive_number_on_this_machine():
    nbytes = resources.own_footprint()
    assert nbytes is None or nbytes > 0


def test_own_footprint_says_unmeasured_rather_than_zero_without_proc(monkeypatch):
    """None means UNMEASURED and never "small".

    A caller that reads a missing footprint as zero prints a reassuring line about a run that
    could be anywhere, which is worse than printing nothing: the whole reason this probe exists
    is that a run growing in the background was found by somebody watching the machine instead of
    by the run's own log.

    Only `/proc/self/statm` is made to fail, because patching `open` for everything would break
    whatever else the interpreter was reading at the time.
    """
    import builtins
    real = builtins.open

    def refuse_statm(path, *args, **kwargs):
        if str(path) == "/proc/self/statm":
            raise OSError("no /proc here")
        return real(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", refuse_statm)
    assert resources.own_footprint() is None


def test_free_disk_walks_up_to_a_directory_that_exists(tmp_path):
    """The output directory of a run that has not started does not exist yet."""
    deep = tmp_path / "not" / "created" / "yet"
    assert resources.free_disk(deep) == resources.free_disk(tmp_path)


def test_free_disk_returns_none_rather_than_raising_on_nonsense():
    assert resources.free_disk("\0no such thing") is None


def test_on_mains_power_answers_with_one_of_three_things():
    assert resources.on_mains_power() in (True, False, None)


def a_power_tree(root, supplies):
    """A `/sys/class/power_supply` shaped directory. `supplies` is `{name: (type, online)}`."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, (kind, online) in supplies.items():
        supply = root / name
        supply.mkdir()
        (supply / "type").write_text(f"{kind}\n", encoding="utf-8")
        if online is not None:
            (supply / "online").write_text(f"{online}\n", encoding="utf-8")
    return root


def test_a_laptop_on_battery_is_read_as_on_battery(tmp_path):
    root = a_power_tree(tmp_path / "unplugged", {"AC": ("Mains", 0), "BAT0": ("Battery", None)})
    assert resources.on_mains_power(root) is False


def test_a_laptop_plugged_in_is_read_as_mains(tmp_path):
    root = a_power_tree(tmp_path / "plugged", {"AC": ("Mains", 1), "BAT0": ("Battery", None)})
    assert resources.on_mains_power(root) is True


def test_one_of_two_mains_supplies_being_online_is_enough(tmp_path):
    """A dock and a charger both present is two supplies and one of them offline."""
    root = a_power_tree(tmp_path / "docked", {"AC": ("Mains", 0), "ADP1": ("Mains", 1),
                                              "BAT0": ("Battery", None)})
    assert resources.on_mains_power(root) is True


def test_a_desktop_with_supplies_and_no_battery_is_mains(tmp_path):
    root = a_power_tree(tmp_path / "desktop", {"usbhid": ("USB", None)})
    assert resources.on_mains_power(root) is True


def test_a_machine_with_no_power_supply_tree_says_unknown(tmp_path):
    assert resources.on_mains_power(tmp_path / "nothing at all") is None


def test_a_battery_with_no_mains_supply_says_unknown_rather_than_battery(tmp_path):
    """A battery alone cannot say whether the charger is in, so the honest answer is unknown."""
    root = a_power_tree(tmp_path / "battery only", {"BAT0": ("Battery", None)})
    assert resources.on_mains_power(root) is None


def test_an_unreadable_supply_is_stepped_over_rather_than_fatal(tmp_path):
    root = a_power_tree(tmp_path / "broken", {"AC": ("Mains", 1)})
    (root / "nonsense").mkdir()                 # a directory with no `type` file at all
    assert resources.on_mains_power(root) is True


def test_the_pinned_probe_returns_none_when_it_cannot_measure(monkeypatch):
    """A diagnostic probe must never be the reason a run does not start."""
    from senbonzakura import doctor

    def explode(_budget):
        raise RuntimeError("no cuda here")
    monkeypatch.setattr(doctor, "_measure_pinned_ceiling", explode)
    assert resources.probe_pinned_ceiling() is None


def test_the_pinned_probe_reports_a_floor_when_it_ran_out_of_budget(monkeypatch):
    from senbonzakura import doctor
    monkeypatch.setattr(doctor, "_measure_pinned_ceiling",
                        lambda budget: (budget, False, ""))
    reached, hit_limit = resources.probe_pinned_ceiling(1234)
    assert (reached, hit_limit) == (1234, False)


def test_measuring_a_machine_never_raises_and_may_answer_nothing():
    """On a machine with no card every reading but two is None, and that is a valid answer."""
    machine = resources.Machine.measure(path=".", pinned=False)
    assert machine.pinned_ceiling is None
    assert machine.disk_free is None or machine.disk_free > 0


# ── the formatters, because a bare count of bytes is not a quantity anyone reads ─────────────────

@pytest.mark.parametrize(("nbytes", "shown"), [
    (0, "0 B"), (512, "512 B"), (2048, "2.0 kB"), (int(3.5e6), "3.5 MB"),
    (int(6.1e9), "6.1 GB"), (int(12.1e12), "12.1 TB"), (int(6e15), "6.0 PB"),
    (int(2e18), "2.0 EB"),
])
def test_bytes_are_shown_with_a_unit(nbytes, shown):
    assert resources.fmt_bytes(nbytes) == shown


def test_an_unknown_byte_count_says_unknown():
    assert resources.fmt_bytes(None) == "unknown"


@pytest.mark.parametrize(("seconds", "shown"), [
    (0, "0s"), (45, "45s"), (750, "12m 30s"), (7500, "2h 05m"),
    (86400, "1d 00h"), (400000, "4d 15h"),
])
def test_a_span_that_may_be_days_reads_as_days(seconds, shown):
    assert resources.fmt_span(seconds) == shown


def test_a_span_that_is_not_a_number_is_not_a_traceback():
    assert resources.fmt_span(float("nan")) == "0s"


# ── the command, as a person runs it ─────────────────────────────────────────────────────────────

def test_the_command_exits_zero_on_a_checkpoint_that_fits(dense, capsys):
    assert streaming.main(["--model", str(dense), "--no-pinned-probe"]) in (
        streaming.BUDGET_OK, streaming.BUDGET_SHORT)
    out = capsys.readouterr().out
    assert "verdict:" in out
    assert "layers" in out


def test_the_command_exits_two_on_a_directory_that_is_not_a_checkpoint(tmp_path, capsys):
    code = streaming.main(["--model", str(tmp_path), "--no-pinned-probe"])
    assert code == streaming.BUDGET_UNREADABLE
    assert "budget refused" in capsys.readouterr().out


def test_the_command_writes_its_budget_atomically(dense, tmp_path, capsys):
    out = tmp_path / "budget.json"
    streaming.main(["--model", str(dense), "--no-pinned-probe", "--out", str(out)])
    capsys.readouterr()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["layers"] == 4
    assert payload["passes_per_trial"] == 199
    assert payload["page_locked"]["side"] is None
    assert [p["name"] for p in payload["pools"]] == ["video memory", "host memory", "free disk"]
    assert not list(tmp_path.glob("*.part")), "the temporary file is renamed, never left behind"


def test_the_command_passes_the_run_shape_through_to_the_budget(dense, tmp_path, capsys):
    """A flag that reaches nothing is a dead knob, and this project has shipped three."""
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    streaming.main(["--model", str(dense), "--no-pinned-probe", "--trials", "4",
                    "--out", str(first)])
    streaming.main(["--model", str(dense), "--no-pinned-probe", "--trials", "64",
                    "--out", str(second)])
    capsys.readouterr()
    few = json.loads(first.read_text(encoding="utf-8"))
    many = json.loads(second.read_text(encoding="utf-8"))
    assert many["read_bytes"] > few["read_bytes"] * 10
    assert many["estimated_seconds"] > few["estimated_seconds"] * 10


def test_the_read_rate_flag_changes_the_estimate(dense, tmp_path, capsys):
    slow, fast = tmp_path / "s.json", tmp_path / "f.json"
    streaming.main(["--model", str(dense), "--no-pinned-probe", "--read-rate", "0.1",
                    "--out", str(slow)])
    streaming.main(["--model", str(dense), "--no-pinned-probe", "--read-rate", "10",
                    "--out", str(fast)])
    capsys.readouterr()
    assert (json.loads(slow.read_text(encoding="utf-8"))["estimated_seconds"]
            > json.loads(fast.read_text(encoding="utf-8"))["estimated_seconds"] * 50)


def test_the_report_folds_to_a_narrow_terminal(dense, monkeypatch, capsys):
    """The guided mode's screens read at 60 columns and the refusal screens below them do not.
    Nothing new here is allowed to join the second group.
    """
    monkeypatch.setenv("COLUMNS", "60")
    streaming.main(["--model", str(dense), "--no-pinned-probe"])
    for line in capsys.readouterr().out.splitlines():
        if line.startswith("      "):
            continue                    # the aligned need column, which `say` leaves verbatim
        if " " not in line.strip():
            continue                    # one unbreakable token, usually a path: it cannot fold
        assert len(line) <= 80, f"a {len(line)} character line: {line}"


def test_the_pinned_probe_says_nothing_when_the_doctor_module_is_unreachable(monkeypatch):
    """An install stripped down to the light commands has no `doctor` import to borrow."""
    import builtins
    real = builtins.__import__

    def refuse_doctor(name, *args, **kwargs):
        if name.endswith("doctor") or name == "doctor":
            raise ImportError("no doctor in this install")
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", refuse_doctor)
    assert resources.probe_pinned_ceiling() is None


def test_a_shard_that_vanishes_between_indexing_and_sizing_is_named(dense, monkeypatch):
    """The window is real: a sync or a tidy-up can remove a file between the two reads."""
    def vanish(_self):
        raise OSError("gone")
    monkeypatch.setattr(pathlib.Path, "stat", vanish)
    with pytest.raises(streaming.ShardError, match="indexed and then unreadable"):
        streaming.describe(dense)


def test_the_command_reports_an_unreadable_model_directory_as_a_sentence(tmp_path, capsys,
                                                                        monkeypatch):
    def refuse(_model, **_kw):
        raise OSError("permission denied")
    monkeypatch.setattr(streaming, "describe", refuse)
    code = streaming.main(["--model", str(tmp_path), "--no-pinned-probe"])
    assert code == streaming.BUDGET_UNREADABLE
    # Joined, because the refusal is wrapped and the reason can land across two lines. A test
    # that asserted against the raw output would be asserting the wrap width.
    out = " ".join(capsys.readouterr().out.split())
    assert "budget refused" in out
    assert "permission denied" in out


def test_the_command_tells_the_reader_that_stopping_here_is_free(dense, capsys, monkeypatch):
    """The refusal's second half, which is the part that stops it reading as a failure."""
    monkeypatch.setattr(resources.Machine, "measure",
                        classmethod(lambda _cls, **_kw: a_machine(disk_free=1024)))
    assert streaming.main(["--model", str(dense)]) == streaming.BUDGET_SHORT
    out = capsys.readouterr().out
    assert "short of free disk" in out
    assert "nothing has been written" in out.lower()
