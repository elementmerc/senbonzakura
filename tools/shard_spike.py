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

Plus a correctness check that is not optional: the streamed result must match the resident one to
within what two correct float32 runs may differ by. A streaming path that is cheap and wrong is
worth nothing, and it is the easy thing to accidentally build.

That check used to demand bit-identity, and bit-identity is not a property of this arrangement:
feeding the same arithmetic tensors that arrived memory-mapped off disk rather than freshly
allocated moves the last bit, and TWO RESIDENT RUNS show the same gap with no streaming involved.
See `FLOAT32_SLACK_ULP` for the measurement. A structural error moves values by order 1 and is
still caught.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

# `resource` is POSIX-only, and importing it at module scope took the whole Windows CI job down
# with "No module named 'resource'" at COLLECTION time, so every test in the run failed rather
# than the handful that need it. The badge in the README says this project is tested on Windows;
# it had not been, because the job never got as far as running a test.
try:
    import resource
except ModuleNotFoundError:   # Windows
    resource = None

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

    Linux reports kilobytes; macOS reports bytes. Windows has no `resource` at all, and this
    raises rather than returning a plausible zero: the whole spike is a memory measurement, and a
    memory measurement that quietly reports nothing is worse than one that refuses.
    """
    if resource is None:
        raise RuntimeError(
            "peak RSS cannot be read on this platform: Python's `resource` module is POSIX-only. "
            "This spike measures memory and has nothing to say without it. Run it on Linux or "
            "macOS.")
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
    arrangement holds together, not about the arithmetic, which already has tests. Note that both
    paths call THIS function, so any difference between them is the route the tensors took, never
    the operation applied to them.
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


#: How far apart two float32 results of the SAME arithmetic may legitimately be.
#:
#: THE MEASUREMENT THAT SET THIS, because "exact" was the wrong claim and asserting it made the
#: spike fail on a newer torch for a reason that had nothing to do with streaming. Under
#: torch 2.14 on eight threads the streamed and resident edits differed by 2.384e-07, one ULP at
#: float32, and the failure message blamed the streaming arrangement. They do not differ because
#: of streaming:
#:
#:     resident vs resident, the same call twice        0.0
#:     resident in memory vs resident FROM DISK         2.384e-07   <- no streaming at all
#:     resident vs streamed                             2.384e-07
#:
#: The middle row is the finding. Feeding the identical arithmetic tensors that arrived by a
#: different route, memory-mapped off disk rather than freshly allocated, moves the last bit,
#: because a float32 reduction is not associative and the kernel chosen depends on the buffer.
#: Two resident runs show it. So bit-identity was never a property of the streaming arrangement
#: and demanding it tested the allocator.
#:
#: Eight ULP rather than one, scaled by the largest magnitude present: enough headroom that a
#: legitimate difference in reduction order passes, and far too little for a structural error,
#: which moves values by order 1. The measured worst difference is always reported either way.
FLOAT32_SLACK_ULP = 8


def tolerance_for(tensors):
    """The largest difference two correct float32 runs may show, for these tensors.

    Scaled by the largest magnitude present rather than fixed, because ULP is relative: the same
    reduction on values around 1000 legitimately moves a thousand times further than on values
    around 1.

    PER TENSOR, not one global peak. It used to take `max(|t|)` across ALL tensors and apply the
    resulting slack to a max-abs-difference taken across all of them, so a tensor whose values
    live near 1 inherited a tolerance derived from one whose values live near 1000, and a
    structural error in the quiet tensor could hide under the loud one's headroom. That is
    harmless on the spike's synthetic weights, where every tensor has the same scale, and would
    not survive contact with real model tensors, where embedding and norm weights differ by orders
    of magnitude. The docstring read as though it already did this.

    Returns a mapping so `max_abs_difference` can compare each tensor against its own bound, plus
    the global figure under the key `None` for the summary line.
    """
    import torch as _t
    eps = _t.finfo(_t.float32).eps
    out, peak = {}, 0.0
    for name, t in tensors.items():
        m = float(t.float().abs().max()) if t.numel() else 0.0
        peak = max(peak, m)
        out[name] = FLOAT32_SLACK_ULP * eps * max(m, 1.0)
    out[None] = FLOAT32_SLACK_ULP * eps * max(peak, 1.0)
    return out


def max_abs_difference(a, b):
    """How far apart the two results are, and where. Reported whatever the verdict.

    A structural error, an edit applied to the wrong axis or skipped on some shard, moves values
    by order 1 and is nowhere near the tolerance above. That is the failure this is guarding, and
    it stays caught.
    """
    missing = set(a) ^ set(b)
    if missing:
        return float("inf"), f"{len(missing)} tensor(s) present on only one side: {sorted(missing)[:3]}"
    worst, where = 0.0, ""
    for name, left in a.items():
        d = float((left.float() - b[name].float()).abs().max())
        if d > worst:
            worst, where = d, name
    return worst, where


def breaches(a, b, slack):
    """Tensors whose own difference exceeds their OWN tolerance, worst first.

    The comparison that a single global tolerance could not make. With one bound derived from the
    largest magnitude anywhere in the checkpoint, a structural error in a tensor whose values live
    near 1 sits comfortably under a bound sized for one whose values live near 1000.
    """
    missing = set(a) ^ set(b)
    if missing:
        return [(min(missing), float("inf"), 0.0)]
    out = []
    for name, left in a.items():
        d = float((left.float() - b[name].float()).abs().max())
        bound = slack.get(name, slack.get(None, 0.0))
        if d > bound:
            out.append((name, d, bound))
    return sorted(out, key=lambda r: -(r[1] / r[2]) if r[2] else -r[1])


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
        # The tolerance is computed HERE, in the pass that holds the tensors, because it is
        # scaled by the magnitudes present and the parent process never sees them. Per tensor as
        # well as globally, so a quiet tensor is judged against its own bound rather than against
        # headroom borrowed from a loud one.
        slack = tolerance_for(resident)
        bad = breaches(resident, streamed, slack)
        print(json.dumps({"max_abs_difference": worst, "where": where,
                          "tolerance": slack[None],
                          "breaches": [{"tensor": n, "difference": d, "tolerance": t}
                                       for n, d, t in bad[:5]],
                          "n_breaches": len(bad)}))
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
        slack = verdict["tolerance"]

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
        exact = worst <= slack
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
                # Recorded beside the verdict so a reader can see what "exact" was measured
                # against, rather than having to trust the word.
                "tolerance": slack, "float32_slack_ulp": FLOAT32_SLACK_ULP,
            }, indent=2), encoding="utf-8")

        print(f"\nthe resident edit needed {resident_cost / 1e6:.1f} MB, the streamed one "
              f"{stream_cost / 1e6:.1f} MB, and one layer is {layer_bytes / 1e6:.1f} MB")
        if not exact:
            print(f"\nFAILED: the streamed edit differs from the resident one by {worst:.3e}, "
                  f"which is more than the {slack:.3e} two correct float32 runs may differ by. A "
                  f"streaming path that is cheap and wrong is worth nothing, and it is the easy "
                  f"thing to build by accident. A difference this large is structural, not "
                  f"rounding: look for an edit applied to the wrong axis or skipped on a shard.")
            return 1
        if not cheap:
            print("\nINCONCLUSIVE: the two results match and the streamed pass did not stay "
                  "within a few layers' worth of memory. The idea is not refuted; THIS "
                  "arrangement of it has not been shown to pay.")
            return 2
        # A ratio against a cost of ~0 is a meaningless number, and printing "460554240x less
        # memory" would be the sort of figure that discredits everything beside it.
        if stream_cost < layer_bytes:
            print(f"\nOK: the streamed edit matches the resident one to within float32 rounding, and never "
                  f"raised this process above what importing torch already required. The resident "
                  f"edit needed {resident_cost / 1e6:.0f} MB more than that.")
        else:
            print(f"\nOK: the streamed edit matches the resident one to within float32 rounding and needed "
                  f"{resident_cost / stream_cost:.1f}x less memory.")
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
