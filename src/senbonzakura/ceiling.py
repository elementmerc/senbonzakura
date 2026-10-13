# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The largest model this machine can stream, with the method named beside every figure.

`resources.plan` answers "does THIS run fit". This answers the question people actually ask first:
"what is the biggest model I could do here". It solves `plan` backwards rather than keeping a
second copy of the arithmetic, so the two can never disagree.

TWO VIDEO MEMORY POLICIES, BOTH ALWAYS SHOWN. Decided by the operator on 2026-10-13.

  free plus reclaimable   the card's free memory plus the cache torch is holding for this process
                          and would hand straight back. `resources` documents why: raw free memory
                          counts that cache as used, so a warm box looks starved when it is not,
                          and a ceiling that halves itself because a job ran an hour ago is
                          confidently wrong about what the reader should do next.
  raw free                the card's free memory as the driver reports it. It is the figure that
                          can never over-promise, so it stays visible as the floor.

A figure without its method is the thing this project withdraws results over, so the policy name is
part of the output and of the JSON, not a docstring.

ONE HONEST LIMIT. The reclaimable term is THIS process's own cache. A fresh `ceiling` process holds
none, so there the two policies agree, and the report says so instead of implying a gain. The term
earns its place when the same arithmetic is called from inside a run that has been generating.

THE SHAPE ASSUMPTION. A ceiling over "models" needs a family of models. The answer is given at a
reference model's proportions: layer count and attention geometry stay fixed, every byte count is
scaled together. So "the ceiling is 74 GB" means "a model built like the reference, at 74 GB". The
reference is the checkpoint passed in, or the synthetic Qwen3-30B-A3B shape below, and the report
names which.
"""
from __future__ import annotations

import json
import sys

from . import resources

POLICY_RECLAIM = "free plus reclaimable"
POLICY_RAW = "raw free"

CEILING_OK, CEILING_NOTHING_MEASURED, CEILING_UNREADABLE = 0, 1, 2

#: Bisection depth. Each step halves the scale interval, so 200 steps is far past float precision
#: and the loop is bounded rather than trusting convergence to end it.
BISECT_STEPS = 200
#: Upper bound on the scale searched, in multiples of the reference. 10^6 times a 61 GB reference
#: is past any storage a person owns, so a ceiling reaching it is reported as unbounded.
MAX_SCALE = 1e6

_BYTE_FIELDS = ("total_bytes", "widest_layer_bytes", "widest_writer_bytes", "writer_bytes",
                "largest_shard_bytes")


def reference_checkpoint():
    """Qwen3-30B-A3B's published shape, worked out by hand. Synthetic: no weight was read."""
    bf16 = 2
    attention = (2048 * 4096 + 2 * 2048 * 512 + 4096 * 2048) * bf16
    experts = 128 * 3 * 2048 * 768 * bf16
    layer = attention + experts
    writer = 128 * 768 * 2048 * bf16
    return resources.Checkpoint(
        layers=48, total_bytes=48 * layer + 2 * 151936 * 2048 * bf16, widest_layer_bytes=layer,
        widest_writer_bytes=writer, writer_bytes=48 * (writer + 4096 * 2048 * bf16),
        largest_shard_bytes=4 * 10 ** 9, shards=16, writer_width=bf16, experts=128,
        kv_heads=4, head_dim=128, hidden=2048)


def scaled(checkpoint, factor):
    """The same model with every byte count multiplied by `factor` and the geometry unchanged."""
    fields = {name: getattr(checkpoint, name) for name in resources.Checkpoint.__slots__}
    for name in _BYTE_FIELDS:
        fields[name] = int(fields[name] * factor)
    return resources.Checkpoint(**fields)


def video_memory_figures(machine, reclaimable):
    """`{policy: bytes or None}` for the two policies. Reclaimable is capped at what is not free.

    Capped because free plus reclaimable can never exceed the card, and a driver racing a cache
    release has been seen to report a sum past the total.
    """
    raw = machine.vram_free
    if raw is None:
        return {POLICY_RECLAIM: None, POLICY_RAW: None}
    claim = int(raw) + max(0, int(reclaimable or 0))
    if machine.vram_total is not None:
        claim = min(claim, int(machine.vram_total))
    return {POLICY_RECLAIM: max(claim, int(raw)), POLICY_RAW: int(raw)}


def _with_vram(machine, vram):
    return resources.Machine(
        vram_free=vram, vram_total=machine.vram_total, ram_available=machine.ram_available,
        disk_free=machine.disk_free, pinned_ceiling=machine.pinned_ceiling,
        pinned_is_floor=machine.pinned_is_floor, on_mains=machine.on_mains)


def _fits_pool(reference, run, machine, factor, pool_name):
    budget = resources.plan(scaled(reference, factor), run, machine)
    pool = next(p for p in budget.pools if p.name == pool_name)
    return pool.fits is not False


def pool_ceiling(reference, run, machine, pool_name):
    """Largest checkpoint size, in bytes, this one pool allows. None if unmeasured.

    Returns `(bytes, bounded)`. `bounded` is False when the search reached MAX_SCALE, meaning no
    realistic model is stopped by this pool.
    """
    probe = resources.plan(reference, run, machine)
    if not next(p for p in probe.pools if p.name == pool_name).measured:
        return None
    if not _fits_pool(reference, run, machine, 0.0, pool_name):
        # The terms that do not scale with the model (cache, captured activations) already
        # exceed the pool, so no model fits however small.
        return 0, True
    if _fits_pool(reference, run, machine, MAX_SCALE, pool_name):
        return int(reference.total_bytes * MAX_SCALE), False
    low, high = 0.0, MAX_SCALE
    for _ in range(BISECT_STEPS):
        mid = (low + high) / 2
        if _fits_pool(reference, run, machine, mid, pool_name):
            low = mid
        else:
            high = mid
    return int(reference.total_bytes * low), True


def ceiling(reference, run, machine, reclaimable=0):
    """The ceiling under each policy. Pure: three records and a number in, a dict out."""
    result = {"reference_layers": reference.layers,
              "reference_bytes": reference.total_bytes,
              "reclaimable_bytes": int(reclaimable or 0),
              "policies": {}}
    for policy, vram in video_memory_figures(machine, reclaimable).items():
        view = _with_vram(machine, vram)
        pools = {}
        for name in ("video memory", "host memory", "free disk"):
            found = pool_ceiling(reference, run, view, name)
            pools[name] = None if found is None else {"bytes": found[0], "bounded": found[1]}
        measured = {n: v for n, v in pools.items() if v is not None}
        binding = min(measured, key=lambda n: measured[n]["bytes"]) if measured else None
        result["policies"][policy] = {
            "video_memory_bytes": vram, "pools": pools, "binding_pool": binding,
            "ceiling_bytes": measured[binding]["bytes"] if binding else None,
            "unmeasured": sorted(n for n, v in pools.items() if v is None)}
    return result


def render(result, log=print):
    from . import say

    def tell(text):
        say.say(text, indent="  ", log=log)

    fmt = resources.fmt_bytes
    tell(f"ceiling, in the proportions of a {result['reference_layers']} layer reference "
         f"({fmt(result['reference_bytes'])}). Every figure names its policy.")
    for policy, row in result["policies"].items():
        vram = row["video_memory_bytes"]
        head = "no video memory reading" if vram is None else f"{fmt(vram)} of video memory counted"
        if row["ceiling_bytes"] is None:
            tell(f"[{policy}] {head}: nothing could be measured, so no ceiling is claimed")
            continue
        tail = "" if row["pools"][row["binding_pool"]]["bounded"] else " (no realistic model hits it)"
        tell(f"[{policy}] {head}: up to {fmt(row['ceiling_bytes'])}, set by "
             f"{row['binding_pool']}{tail}")
        for name, found in row["pools"].items():
            log(f"      {fmt(found['bytes']) if found else 'unmeasured':>10}  {name}")
        if row["unmeasured"]:
            tell(f"not checked: {', '.join(row['unmeasured'])}, so the ceiling is an upper bound")
    if result["reclaimable_bytes"] == 0:
        tell("reclaimable cache: none held by this process, so the two policies agree here. "
             "The 'plus reclaimable' figure differs only inside a run that has been generating.")
    tell("raw free is the floor: it is the figure that can never over-promise.")
    tell("These are arithmetic over the memory budget, not a run: nothing was loaded.")


def build_parser():
    from . import argresolve

    p = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False, prog="senbonzakura ceiling",
        description="Work out the largest model this machine could stream.")
    p.add_argument("--model", default=None,
                   help="a checkpoint directory whose proportions the ceiling is given in "
                        "(default: a synthetic Qwen3-30B-A3B shape)")
    p.add_argument("--trials", type=argresolve.whole_number("--trials", minimum=1), default=64)
    p.add_argument("--prompts", type=argresolve.whole_number("--prompts", minimum=1), default=48)
    p.add_argument("--tokens", type=argresolve.whole_number("--tokens", minimum=1), default=48)
    p.add_argument("--gen-batch", type=argresolve.whole_number("--gen-batch", minimum=1),
                   default=16)
    p.add_argument("--expert-block", type=argresolve.whole_number("--expert-block", minimum=1),
                   default=8)
    p.add_argument("--sparsity", action="store_true")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    return p


def main(argv=None):
    from . import say, streaming

    a = build_parser().parse_args(argv)
    if a.model:
        try:
            reference = streaming.describe(a.model)
        except (streaming.ShardError, OSError) as exc:
            say.say(f"ceiling refused: cannot read {a.model}: {exc}", indent="  ")
            return CEILING_UNREADABLE
    else:
        reference = reference_checkpoint()
    run = resources.Run(
        prompts=a.prompts, tokens=a.tokens, trials=a.trials,
        passes_per_trial=resources.passes_per_trial(
            eval_refusal=a.prompts, eval_kl=a.prompts, gen_batch=a.gen_batch,
            gen_tokens=a.tokens),
        expert_block=a.expert_block, sparsity=a.sparsity)
    machine = resources.Machine.measure(path=a.model or ".", pinned=False)
    result = ceiling(reference, run, machine, reclaimable=resources.cuda_reclaimable())
    result["reference"] = a.model or "synthetic Qwen3-30B-A3B shape"
    if a.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        render(result)
    if all(row["ceiling_bytes"] is None for row in result["policies"].values()):
        return CEILING_NOTHING_MEASURED
    return CEILING_OK


if __name__ == "__main__":
    sys.exit(main())
