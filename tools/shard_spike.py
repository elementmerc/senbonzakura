#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""SPIKE: can a checkpoint be edited one layer at a time, never holding the whole model?

RECONNAISSANCE, NOT A FEATURE. It answers one question and stops. Nothing here is wired into the
tool, nothing depends on it, and it becomes v0.5 only if someone decides so on purpose.

THE QUESTION

`cli.py` refuses disk-offloaded weights, and it is right to: `accelerate`'s offload map hands back
a fresh tensor per read, so an in-place edit is written into a copy that is then discarded. The
refusal is correct and it is also the reason a model larger than host RAM cannot be abliterated at
all.

A per-layer on-disk format would make that refusal unnecessary rather than unavoidable: load one
layer, edit it, write it back, evict. The bake becomes a streaming operation over shards instead of
a resident operation over a model. Soup does exactly this for training and reports peak RSS while
sharding as one decoder layer rather than the model, which is the number that matters, because the
whole point is that the model does not fit.

WHAT IS ACTUALLY MEASURED

Peak resident set size, read from the kernel rather than estimated, for three passes over the same
checkpoint:

    resident   load it all, edit it, save it        <- what the tool does today
    shard      rewrite to one file per layer        <- the cost of getting into the format
    stream     edit each shard in turn              <- what v0.5 would do

EACH IN ITS OWN PROCESS, and that is not tidiness. `ru_maxrss` is a HIGH-WATER MARK: it never
falls. Run all three in one process and the streamed pass reports the peak the resident pass
already set, so "streaming added nothing" would mean "streaming did not exceed a number somebody
else reached", which is not the claim. The first version of this spike did exactly that and
reported +0.0 MB, which looked like the strongest possible result and was almost no evidence.

Plus a correctness check that is not optional: the streamed result must be bit-identical to the
resident one. A streaming path that is cheap and wrong is worth nothing, and it is the easy thing
to accidentally build.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from safetensors.torch import load_file, save_file

#: Written beside the shards so a reader can tell what a directory of them is.
INDEX_NAME = "shard-index.json"


def peak_rss_bytes():
    """Peak RSS this process has ever reached, from the kernel.

    `ru_maxrss` is a high-water mark and never falls, which is exactly right for the question:
    "did this approach ever need to hold the model" cannot be answered by sampling current usage
    and hoping to catch the peak.

    Linux reports kilobytes; macOS reports bytes.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def _is_tmpfs(path):
    """Is this path on a RAM-backed filesystem? Best effort, and a warning either way."""
    try:
        target = str(Path(path).resolve())
        best, kind = "", ""
        with open("/proc/mounts", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and target.startswith(parts[1]) and len(parts[1]) > len(best):
                    best, kind = parts[1], parts[2]
    except OSError:
        return False
    else:
        return kind in ("tmpfs", "ramfs")


def _layer_of(name):
    """Which decoder layer a tensor belongs to, or None for embeddings, the head and norms."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def shard(src_state, out_dir):
    """Rewrite a state dict into one file per decoder layer, plus one for everything else.

    Materialises ONE layer at a time on the way out. The caller hands in a state dict here because
    this is a spike; the real thing would read tensor by tensor from the source safetensors file
    and never build a dict at all, which is the difference between this measuring the idea and
    measuring the idea's best case.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for name in src_state:
        groups.setdefault(_layer_of(name), []).append(name)

    index = {}
    for layer, names in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
        stem = "extras" if layer is None else f"layer_{layer:04d}"
        piece = {n: src_state[n].contiguous() for n in names}
        save_file(piece, str(out_dir / f"{stem}.safetensors"))
        index[stem] = sorted(names)
        del piece
        gc.collect()
    (out_dir / INDEX_NAME).write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def edit_streamed(shard_dir, direction, out_dir):
    """Apply the edit shard by shard, holding one layer at a time.

    THE POINT OF THE SPIKE. Nothing here ever sees more than one shard, so peak RSS should track
    the largest layer rather than the model.
    """
    shard_dir, out_dir = Path(shard_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads((shard_dir / INDEX_NAME).read_text(encoding="utf-8"))
    for stem in sorted(index):
        piece = load_file(str(shard_dir / f"{stem}.safetensors"))
        edited = {n: _edit(t, direction) for n, t in piece.items()}
        save_file(edited, str(out_dir / f"{stem}.safetensors"))
        del piece, edited
        gc.collect()
    (out_dir / INDEX_NAME).write_text(json.dumps(index, indent=2), encoding="utf-8")


def _edit(tensor, direction):
    """A stand-in for the bake: remove a direction from every 2-D tensor's row space.

    Deliberately the same SHAPE of operation as the real orthogonalisation (a rank-one projection
    subtracted from the output rows) without being it. The spike is about whether the streaming
    arrangement holds together and stays exact, not about the arithmetic, which already has tests.
    """
    if tensor.ndim != 2 or tensor.shape[1] != direction.shape[0]:
        return tensor.clone()
    d = direction.to(tensor.dtype)
    return tensor - torch.outer(tensor @ d, d)


def edit_resident(state, direction):
    """What the tool does today: everything in memory at once."""
    return {n: _edit(t, direction) for n, t in state.items()}


def reassemble(shard_dir):
    """Every shard back into one state dict. Only for CHECKING the streamed result."""
    shard_dir = Path(shard_dir)
    index = json.loads((shard_dir / INDEX_NAME).read_text(encoding="utf-8"))
    out = {}
    for stem in sorted(index):
        out.update(load_file(str(shard_dir / f"{stem}.safetensors")))
    return out


def max_abs_difference(a, b):
    """Bit-identical or not, and where. A cheap streaming path that is wrong is worth nothing."""
    missing = set(a) ^ set(b)
    if missing:
        return float("inf"), f"{len(missing)} tensor(s) present on only one side: {sorted(missing)[:3]}"
    worst, where = 0.0, ""
    for name, left in a.items():
        d = float((left.float() - b[name].float()).abs().max())
        if d > worst:
            worst, where = d, name
    return worst, where


def checkpoint_sizes(layers, hidden, ffn, bytes_per_element=4):
    """Model and largest-layer sizes, computed from the SHAPES rather than by allocating them.

    THE PRE-FLIGHT USED TO BUILD THE CHECKPOINT TO FIND OUT HOW BIG IT WAS, which is a disk-space
    check that allocates the thing it is checking there is room for. Asked for a 200-layer model it
    was killed by the OOM killer before printing a word, so the guard against running out of
    resources was itself the thing that ran out of them. Arithmetic cannot do that.
    """
    per_layer = (hidden * hidden + hidden * ffn + ffn * hidden + hidden) * bytes_per_element
    extras = (512 * hidden * 2 + hidden) * bytes_per_element
    return per_layer * layers + extras, per_layer


def _synthetic_checkpoint(layers, hidden, ffn, dtype=torch.float32, seed=0):
    """A checkpoint-shaped state dict, so the spike needs no model download to answer anything."""
    g = torch.Generator().manual_seed(seed)
    state = {"model.embed_tokens.weight": torch.randn(512, hidden, generator=g, dtype=dtype),
             "lm_head.weight": torch.randn(512, hidden, generator=g, dtype=dtype),
             "model.norm.weight": torch.ones(hidden, dtype=dtype)}
    for i in range(layers):
        p = f"model.layers.{i}"
        state[f"{p}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden, generator=g, dtype=dtype)
        state[f"{p}.mlp.down_proj.weight"] = torch.randn(hidden, ffn, generator=g, dtype=dtype)
        state[f"{p}.mlp.up_proj.weight"] = torch.randn(ffn, hidden, generator=g, dtype=dtype)
        state[f"{p}.input_layernorm.weight"] = torch.ones(hidden, dtype=dtype)
    return state


def _run_one(mode, a, work, direction):
    """One pass, in a process that has done nothing else, so its peak is its own."""
    if mode == "resident":
        state = _synthetic_checkpoint(a.layers, a.hidden, a.ffn)
        edited = edit_resident(state, direction)
        save_file({n: t.contiguous() for n, t in edited.items()}, str(work / "resident.safetensors"))
    elif mode == "shard":
        state = _synthetic_checkpoint(a.layers, a.hidden, a.ffn)
        shard(state, work / "shards")
    elif mode == "stream":
        edit_streamed(work / "shards", direction, work / "edited")
    elif mode == "verify":
        resident = load_file(str(work / "resident.safetensors"))
        streamed = reassemble(work / "edited")
        worst, where = max_abs_difference(resident, streamed)
        print(json.dumps({"max_abs_difference": worst, "where": where}))
        return 0
    else:
        raise ValueError(f"unknown mode {mode!r}")
    print(json.dumps({"peak_rss_bytes": peak_rss_bytes()}))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--ffn", type=int, default=4096)
    p.add_argument("--work", default=None, help="where to put the shards (default: a temp dir)")
    p.add_argument("--json", default=None)
    p.add_argument("--mode", choices=["resident", "shard", "stream", "verify"], default=None,
                   help="run ONE pass and print its own peak. Used by the orchestrator to give "
                        "each pass a fresh process; ru_maxrss never falls, so passes sharing a "
                        "process cannot be told apart.")
    a = p.parse_args(argv)

    import subprocess
    import tempfile

    g = torch.Generator().manual_seed(1)
    direction = torch.randn(a.hidden, generator=g)
    direction = direction / direction.norm()

    if a.mode:
        return _run_one(a.mode, a, Path(a.work), direction)

    tmp = None
    if a.work is None:
        tmp = tempfile.TemporaryDirectory(prefix="senbon-shard-spike-")
        work = Path(tmp.name)
    else:
        work = Path(a.work)
        work.mkdir(parents=True, exist_ok=True)

    try:
        model_bytes, layer_bytes = checkpoint_sizes(a.layers, a.hidden, a.ffn)
        # PRE-FLIGHT, because the first run of this died two thirds of the way through with
        # "Disk quota exceeded" after twenty minutes of work. Three copies of the checkpoint land
        # on disk: the resident output, the shards, and the edited shards.
        import shutil
        need = model_bytes * 3
        free = shutil.disk_usage(work).free
        if free < need:
            print(f"not enough room in {work}: this needs about {need / 1e9:.1f} GB "
                  f"(three copies of a {model_bytes / 1e6:.0f} MB checkpoint) and there is "
                  f"{free / 1e9:.1f} GB free.\n"
                  f"Point --work at a disk with room, or shrink the checkpoint with --layers, "
                  f"--hidden and --ffn.", file=sys.stderr)
            return 2
        # A tmpfs is RAM, so writing gigabytes to one measures something other than what this
        # spike is about, and takes the machine down with it. This project has already lost an
        # afternoon to exactly that.
        if _is_tmpfs(work):
            print(f"WARNING: {work} looks like a tmpfs, which is RAM. The disk numbers will be "
                  f"meaningless and a large run may exhaust memory. Use --work on a real disk.",
                  file=sys.stderr)

        print(f"checkpoint: {a.layers} layers, hidden {a.hidden}, ffn {a.ffn}")
        print(f"  model     {model_bytes / 1e6:8.1f} MB")
        print(f"  one layer {layer_bytes / 1e6:8.1f} MB   "
              f"({model_bytes / layer_bytes:.1f}x smaller)\n")

        def child(mode):
            r = subprocess.run(
                [sys.executable, __file__, "--mode", mode, "--work", str(work),
                 "--layers", str(a.layers), "--hidden", str(a.hidden), "--ffn", str(a.ffn)],
                capture_output=True, text=True, timeout=1800, check=False)
            if r.returncode != 0:
                raise RuntimeError(f"the {mode} pass failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
            return json.loads(r.stdout.strip().splitlines()[-1])

        peaks = {m: child(m)["peak_rss_bytes"] for m in ("resident", "shard", "stream")}
        verdict = child("verify")
        worst, where = verdict["max_abs_difference"], verdict["where"]

        # An empty process, so the interpreter and torch are subtracted rather than counted as the
        # cost of an approach. Without it every number here is dominated by importing torch.
        empty = ("import json,resource,sys;import torch;"
                 "raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
                 "print(json.dumps({'peak_rss_bytes': raw if sys.platform=='darwin' "
                 "else raw*1024}))")
        floor = json.loads(subprocess.run(
            [sys.executable, "-c", empty],
            capture_output=True, text=True, timeout=600, check=True).stdout)["peak_rss_bytes"]

        print(f"{'pass':<12} {'own peak RSS':>14} {'above an empty process':>24}")
        for mode in ("resident", "shard", "stream"):
            print(f"{mode:<12} {peaks[mode] / 1e6:11.1f} MB {(peaks[mode] - floor) / 1e6:21.1f} MB")
        print(f"{'(empty)':<12} {floor / 1e6:11.1f} MB")
        print(f"\nstreamed vs resident: max abs difference {worst:.3e}"
              + (f" at {where}" if where else ""))

        resident_cost = peaks["resident"] - floor
        stream_cost = peaks["stream"] - floor
        exact = worst == 0.0
        # The claim is that streaming tracks a LAYER rather than the model. Three layers of
        # headroom, because a copy plus the tensor being written is legitimately more than one.
        cheap = stream_cost < layer_bytes * 3

        if a.json:
            Path(a.json).write_text(json.dumps({
                "layers": a.layers, "hidden": a.hidden, "ffn": a.ffn,
                "model_bytes": model_bytes, "layer_bytes": layer_bytes,
                "empty_process_bytes": floor, "peaks": peaks,
                "resident_cost_bytes": resident_cost, "stream_cost_bytes": stream_cost,
                "max_abs_difference": worst, "exact": exact, "cheap": cheap,
            }, indent=2), encoding="utf-8")

        print(f"\nthe resident edit needed {resident_cost / 1e6:.1f} MB, the streamed one "
              f"{stream_cost / 1e6:.1f} MB, and one layer is {layer_bytes / 1e6:.1f} MB")
        if not exact:
            print("\nFAILED: the streamed edit does not match the resident one. A streaming path "
                  "that is cheap and wrong is worth nothing, and it is the easy thing to build "
                  "by accident.")
            return 1
        if not cheap:
            print("\nINCONCLUSIVE: the two results match and the streamed pass did not stay "
                  "within a few layers' worth of memory. The idea is not refuted; THIS "
                  "arrangement of it has not been shown to pay.")
            return 2
        # A ratio against a cost of ~0 is a meaningless number, and printing "460554240x less
        # memory" would be the sort of figure that discredits everything beside it.
        if stream_cost < layer_bytes:
            print(f"\nOK: the streamed edit is bit-identical to the resident one, and never "
                  f"raised this process above what importing torch already required. The resident "
                  f"edit needed {resident_cost / 1e6:.0f} MB more than that.")
        else:
            print(f"\nOK: the streamed edit is bit-identical to the resident one and needed "
                  f"{resident_cost / stream_cost:.1f}x less memory.")
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
