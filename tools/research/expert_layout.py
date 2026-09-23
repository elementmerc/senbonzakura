#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""SPIKE: what does a MoE checkpoint STORE, and what does transformers build from it?

RECONNAISSANCE, NOT A FEATURE. It answers one question and stops. Nothing in the tool calls it.

THE QUESTION

The v0.5 streaming plan says the checkpoint holds separate per-expert tensors and that transformers
fuses them at load, so a rewrite over the on-disk layout would peak at tens of megabytes where the
in-memory one peaks at gigabytes. The whole streaming bake design rests on that, and it had never
been checked against a real checkpoint.

WHAT IT MEASURED, 2026-09-22, transformers 5.14.1:

  LiquidAI/LFM2.5-8B-A1B        on disk: 32 x (2048, 1792) bf16, 7.0 MB each
                                in memory: ONE fused (32, 2048, 1792), 224 MB
  ibm-granite/granite-3.0-1b    on disk: fused (32, 1024, 512), 32 MB
                                in memory: the same fused tensor

**The plan's claim is true for LFM and false for Granite**, and a loader written to the plan's flat
version would have had nothing to read on a checkpoint that is already fused. Both layouts are
real and a streaming bake has to handle both.

WHAT IT COSTS, per MoE layer of LFM2.5-8B-A1B, for the residual writer alone. The rewrite converts
to float32 and holds about four intermediates of the same shape:

  Fused, whole stack at once      ~1.8 GB     what the bake did before 2026-09-22
  Fused, 8 experts at a time      ~448 MB     what it does now
  The on-disk per-expert tensors   ~56 MB     what a streaming bake could do

So for a per-expert checkpoint the streaming path should rewrite the stored tensors and never build
the fused parameter at all. The only reason it currently pays 1.8 GB for weights the checkpoint
holds in 7 MB pieces is that it goes through the transformers loader, which fuses on the way in.
For Granite there is no unfused form to fall back on, so blocking the fused rewrite is the only
lever available, which is why both exist.

HOW IT DETECTS EXPERTS, and why not by name

The first version of this script looked for the word "expert" in tensor names. Granite stores its
experts as `block_sparse_moe.output_linear.weight` and the word does not appear, so the script
reported a mixture-of-experts checkpoint as dense, confidently, in one line. That is this project's
oldest failure wearing a new hat: a guard that covers one spelling of a thing reports clean on the
others, and reporting clean is worse than not running.

So detection is by shape against the config instead. The expert count comes from the config, a
rank-3 weight whose leading dimension equals it is a fused stack, and a family of rank-2 weights
whose names differ only by an integer index is a per-expert group. Names are used for reporting,
never for deciding.
"""
import argparse
import collections
import json
import math
import pathlib
import re
import struct
import sys

#: Every spelling of "how many experts" seen across the architectures this tool supports. Read from
#: the config rather than inferred from the weights, so that a checkpoint whose expert tensors are
#: named in some new way is still measured rather than silently skipped.
EXPERT_COUNT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts",
                     "num_experts_per_tok_total", "n_expert")

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1,
               "BOOL": 1}

#: A name with an integer path component, which is how an unfused expert list is spelled:
#: `...experts.7.w2.weight`. The index is replaced to group the family.
_INDEXED = re.compile(r"\.(\d+)\.")


def read_header(path):
    """The safetensors header: 8 bytes of little-endian length, then JSON. No weights are read."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def _nbytes(meta):
    return math.prod(meta["shape"]) * DTYPE_BYTES.get(meta["dtype"], 2)


def expert_count(model_dir):
    for cfg_path in sorted(model_dir.rglob("config.json")):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # Some configs nest the text model's settings a level down.
        for scope in (cfg, cfg.get("text_config") or {}):
            for key in EXPERT_COUNT_KEYS:
                if isinstance(scope.get(key), int) and scope[key] > 1:
                    return scope[key], key
    return None, None


def stored_tensors(model_dir):
    out = {}
    for shard in sorted(model_dir.rglob("*.safetensors")):
        # Per shard rather than around the loop: one unreadable shard should cost its own tensors
        # and not the rest of the checkpoint's, because a partial answer that says which part is
        # missing beats no answer at all.
        try:
            header = read_header(shard)
        except (OSError, ValueError, struct.error) as exc:
            print(f"  could not read {shard.name}: {type(exc).__name__}: {exc}")
            continue
        out.update({name: meta for name, meta in header.items()
                    if name != "__metadata__" and isinstance(meta, dict) and "shape" in meta})
    return out


def _pattern(name):
    """Replace each integer path component with a distinct placeholder.

    Distinct rather than identical because a per-expert weight usually carries TWO indices, the
    layer and the expert, and rendering both as the same token makes the family count read as an
    expert count. On LFM2.5-8B-A1B that would print 704 where the model has 32 experts across 22
    layers, which is the kind of number somebody later quotes.
    """
    counter = iter("Lenxyz")
    return _INDEXED.sub(lambda _: f".{{{next(counter, 'k')}}}.", name)


def classify(tensors, experts):
    """Fused stacks and per-expert families, decided on shape rather than on the name."""
    fused = {n: m for n, m in tensors.items()
             if len(m["shape"]) == 3 and m["shape"][0] == experts}

    families = collections.defaultdict(list)
    for name, meta in tensors.items():
        if len(meta["shape"]) != 2 or not _INDEXED.search(name):
            continue
        families[_pattern(name)].append((name, meta))
    per_expert = {k: v for k, v in families.items() if len(v) % experts == 0}
    return fused, per_expert


def report(model_dir):
    print(f"\n=== {model_dir.name} ===")
    experts, key = expert_count(model_dir)
    if experts is None:
        print("  no expert count in any config, so this is dense or uses a spelling not listed in "
              "EXPERT_COUNT_KEYS. It is NOT safe to read this as 'dense'.")
        return
    print(f"  config says {experts} experts (via {key!r})")

    tensors = stored_tensors(model_dir)
    if not tensors:
        print("  no safetensors shards found")
        return
    fused, per_expert = classify(tensors, experts)

    if fused:
        print(f"  FUSED on disk: {len(fused)} rank-3 stacks with a leading dimension of {experts}")
        sizes = collections.Counter(
            (tuple(meta["shape"]), meta["dtype"], _nbytes(meta)) for meta in fused.values())
        for (shape, dtype, nbytes), count in sorted(sizes.items()):
            print(f"    {count:>4} x  {shape}  {dtype}  {nbytes / 1024 ** 2:8.1f} MB each")
    if per_expert:
        print(f"  PER-EXPERT on disk: {len(per_expert)} families of rank-2 tensors. The count is "
              f"every stored tensor in the family, so it is layers times the {experts} experts.")
        for pattern, members in sorted(per_expert.items())[:6]:
            shape = tuple(members[0][1]["shape"])
            each = _nbytes(members[0][1])
            layers = len(members) // experts
            print(f"    {len(members):>4} x  {shape}  {members[0][1]['dtype']}  "
                  f"{each / 1024 ** 2:8.1f} MB each  ({layers} layers x {experts})  {pattern}")
        if len(per_expert) > 6:
            print(f"    ... and {len(per_expert) - 6} more families")
    if not fused and not per_expert:
        print(f"  config declares {experts} experts and NO tensor matches either layout. That is a "
              f"third shape this script does not know, not an absence of experts.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="model directories; defaults to the HF cache")
    args = ap.parse_args()
    roots = [pathlib.Path(p) for p in args.paths]
    if not roots:
        cache = pathlib.Path.home() / ".cache/huggingface/hub"
        roots = sorted(p for p in cache.glob("models--*") if p.is_dir())
        if not roots:
            raise SystemExit(f"nothing to inspect: {cache} holds no cached models")
    for root in roots:
        if not root.is_dir():
            print(f"\n=== {root} ===\n  not a directory", file=sys.stderr)
            continue
        report(root)


if __name__ == "__main__":
    main()
