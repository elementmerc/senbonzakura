#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""SPIKE: how fast can one layer actually be read off this disk, and does the strategy matter?

RECONNAISSANCE, NOT A FEATURE. It answers one question and stops. Nothing in the tool calls it.

THE QUESTION

The v0.5 plan's whole performance model rests on one constant: a sustained 2 GB/s from a laptop
NVMe. Every estimate in that file is derived from it, including the headline "about 5 days of
disk-bound wall clock for one 30B search", and the plan says plainly that the constant has never
been checked:

    "2 GB/s assumes large sequential reads. If streaming.py uses safetensors mmap with per-tensor
     get_tensor, page-fault-driven reads on a laptop NVMe land well below that. Name the read
     strategy, because the whole performance model rests on that constant."

An estimate built on an unmeasured constant is not an estimate, it is a preference with arithmetic
attached. So this measures the constant, for each strategy a streaming loader could plausibly use,
against a real checkpoint.

THE THREE STRATEGIES, and why the third one exists

    pread     seek to each tensor and read it, which is what `streaming.py` does today
    mmap      map the shard and let page faults fetch it, which is what safetensors does
    span      one sequential read covering the layer's whole byte range, then slice

`span` is here because the other two are both random-access readings of a file whose layout may
well be sequential. A transformer layer's tensors are usually written adjacently, so the bytes a
layer needs may be one contiguous run, and reading it as one run is the case the 2 GB/s figure
actually describes. The spike reports how much of the span is waste, because a layer interleaved
with its neighbours would make `span` read far more than it uses and the strategy would be a bad
trade dressed as a good one.

THE CACHE IS THE TRAP, and it is the reason most numbers like this are wrong

Read a file twice and the second read measures the page cache, not the disk. On this machine that
is tens of gigabytes a second and it is not a lie, it is an answer to a different question. A
streaming run over a model far larger than RAM meets a cold cache on essentially every layer, so
the cold number is the one the estimate needs.

Dropping the cache normally wants root. `posix_fadvise(POSIX_FADV_DONTNEED)` evicts one file's
pages without it, which is both politer and more precise: it touches this checkpoint and nothing
else on the machine. The spike refuses to report a cold number it could not actually make cold,
rather than quietly reporting a warm one under a cold label.

BOTH NUMBERS ARE REPORTED, because both are real. The capture pass reads each layer once, which is
cold. The search re-reads the same layers every trial, and on a model that fits in RAM those are
warm. A tool that knew only one of them would mis-estimate one of the two phases badly.

WHAT IT MEASURED, 2026-09-23, on the ROG's ext4 NVMe, best of 3, GB/s of the layer's own bytes:

    Qwen3-4B      layer 0  192 MB  dense    cold  pread 0.96  mmap 0.90  span 0.86
    LFM2.5-8B-A1B layer 0  116 MB  dense    cold  pread 1.67  mmap 1.15  span 0.93
    LFM2.5-8B-A1B layer 3  704 MB  MoE      cold  pread 1.95  mmap 1.13  span 0.03

**The answer is `pread`**, which is what `streaming.py` already does. It is never worst and it is
dramatically best on the layout the streaming path exists for.

**`span` is dead, and the waste column is what killed it.** A dense layer's tensors are contiguous
and the span wastes nothing. The MoE layer's 103 tensors are scattered across 11.9 GB of shard to
collect 704 MB, 94.1% waste, so one sequential read moves seventeen times the data it needs. The
plan's reasoning that "a transformer layer's tensors are usually written adjacently" is true of
the dense layers and false of exactly the ones that matter.

**The plan's 2 GB/s assumption survives, narrowly, and only for `pread`.** `mmap` is what
safetensors does by default and measured 1.13 GB/s on the same layer, which would have stretched
the 30B estimate from about five days to about nine, for a reason no profile of our own code would
ever have shown.
"""
import argparse
import contextlib
import ctypes
import ctypes.util
import mmap
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from senbonzakura import streaming

GB = 1024 ** 3

#: POSIX_FADV_DONTNEED. Hard-coded rather than read from a module because Python does not expose
#: the constant on every platform it exposes `posix_fadvise` on, and a wrong value here would
#: silently advise something else and report a warm read as cold.
FADV_DONTNEED = 4


def evict(path):
    """Drop this file's pages from the cache. True when it actually happened.

    Returns False rather than raising on a platform without `posix_fadvise`, so the spike can
    still report its warm numbers there and say the cold ones are unavailable. Reporting a warm
    read under a cold label is the one outcome worth refusing.
    """
    if not hasattr(os, "posix_fadvise"):
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        # A write-back flush first: dirty pages cannot be dropped, and a checkpoint just written
        # by another process would otherwise stay resident and read back at RAM speed.
        with contextlib.suppress(OSError):
            os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


def _resident_pages(path):
    """Roughly how much of this file is in the page cache, in bytes, or None if it cannot be asked.

    The spike's own check on itself. `posix_fadvise` is ADVICE: the kernel is free to ignore it,
    and a cold measurement that was quietly warm is worse than no measurement, because it would
    make every strategy look equally fast and the conclusion would be "the read strategy does not
    matter", which is exactly the wrong answer to take away.
    """
    libc_name = ctypes.util.find_library("c")
    if not libc_name or not hasattr(mmap, "MAP_SHARED"):
        return None
    try:
        libc = ctypes.CDLL(libc_name, use_errno=True)
        mincore = libc.mincore
    except (OSError, AttributeError):
        return None

    size = os.path.getsize(path)
    if size == 0:
        return 0
    page = mmap.PAGESIZE
    # ACCESS_COPY rather than ACCESS_READ, only because `ctypes.from_buffer` refuses a read-only
    # buffer and there is no other way to get the mapping's address. Copy-on-write costs nothing
    # here because nothing below writes: pages stay shared with the page cache, which is exactly
    # the residency `mincore` is being asked about, and the file itself cannot be modified.
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY) as mm:
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        pages = (size + page - 1) // page
        vec = (ctypes.c_ubyte * pages)()
        if mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            return None
        # The low bit means resident. The rest are reserved and must be masked, or a kernel that
        # sets them turns this into a count of every page.
        return sum(1 for b in vec if b & 1) * page


# ── the three strategies ─────────────────────────────────────────────────────────────────────


def read_pread(shard, entries, data_start):
    """Seek to each tensor and read it. What `streaming.py` does today."""
    total = 0
    with open(shard, "rb") as fh:
        for tensor in entries:
            fh.seek(data_start + tensor.start)
            total += len(fh.read(tensor.nbytes))
    return total


def read_mmap(shard, entries, data_start):
    """Map the shard and touch each tensor's bytes. What safetensors does.

    The bytes are summed rather than merely sliced, because a slice of an mmap is lazy: without
    touching every page this would measure the cost of creating a view and report an absurd
    throughput. That is the shape of mistake this whole spike exists to avoid.
    """
    touched = 0
    with open(shard, "rb") as fh, mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
        view = memoryview(mm)
        for tensor in entries:
            lo = data_start + tensor.start
            # One byte per page is enough to fault the page in, and far cheaper than summing
            # every byte, which would measure Python's loop rather than the disk.
            for off in range(lo, lo + tensor.nbytes, mmap.PAGESIZE):
                if view[off]:
                    touched += 1
        view.release()
    return sum(t.nbytes for t in entries)


def read_span(shard, entries, data_start):
    """One sequential read covering the layer's whole byte range, then slice.

    Returns the bytes actually read, which is the SPAN and not the layer, so a caller comparing
    it against the layer's size can see the waste.
    """
    lo = min(t.start for t in entries)
    hi = max(t.end for t in entries)
    with open(shard, "rb") as fh:
        fh.seek(data_start + lo)
        blob = fh.read(hi - lo)
    # Slice each tensor out, so the strategy is compared doing the same job as the others rather
    # than doing less of it.
    #
    # THROUGH A MEMORYVIEW, and the first version did not. Slicing `bytes` COPIES, so `span` paid
    # for the whole layer twice and measured 1.4 GB/s against pread's 7.8 on a RAM-backed file,
    # which reads as "one big sequential read is five times slower than many small ones". That
    # conclusion is absurd on its face and would have been published as a measurement. What it
    # actually measured was an extra memcpy this strategy does not need: a reader hands the bytes
    # straight to a tensor constructor and never materialises a second copy.
    view = memoryview(blob)
    for tensor in entries:
        _ = view[tensor.start - lo:tensor.end - lo]
    view.release()
    return len(blob)


STRATEGIES = {"pread": read_pread, "mmap": read_mmap, "span": read_span}


def time_layer(shard, entries, data_start, strategy, cold):
    """One timed read of one layer by one strategy. Returns (seconds, bytes_read, was_cold)."""
    fn = STRATEGIES[strategy]
    was_cold = False
    if cold:
        was_cold = evict(shard)
        resident = _resident_pages(shard)
        if resident is not None and resident > os.path.getsize(shard) * 0.1:
            # More than a tenth still resident after the advice: the kernel declined, so this is
            # not a cold read and must not be labelled one.
            was_cold = False
    t0 = time.perf_counter()
    nbytes = fn(shard, entries, data_start)
    return time.perf_counter() - t0, nbytes, was_cold


def report(model_dir, reps, layers_to_try):
    print(f"\n=== {model_dir} ===")
    index = streaming.index_layers(model_dir)
    print(f"  {index.count} layers, {len(index.shards())} shard(s)")
    widest, widest_bytes = index.widest()
    print(f"  widest layer is {widest} at {widest_bytes / 1024 ** 2:.1f} MB")

    picks = sorted({0, widest, index.count - 1})[:layers_to_try]
    for i in picks:
        entries_by_shard = {}
        for shard, tensor in index.layers[i].values():
            entries_by_shard.setdefault(shard, []).append(tensor)
        if len(entries_by_shard) != 1:
            print(f"\n  layer {i}: spans {len(entries_by_shard)} shards, skipped. A layer split "
                  f"across shards is a real case and needs its own measurement, not this one.")
            continue
        shard, entries = next(iter(entries_by_shard.items()))
        entries.sort(key=lambda t: t.start)
        _header, data_start = streaming.read_header(shard)

        layer_bytes = sum(t.nbytes for t in entries)
        span = max(t.end for t in entries) - min(t.start for t in entries)
        waste = span - layer_bytes
        print(f"\n  -- layer {i}: {len(entries)} tensors, {layer_bytes / 1024 ** 2:.1f} MB, "
              f"span {span / 1024 ** 2:.1f} MB, waste {waste / 1024 ** 2:.1f} MB "
              f"({100 * waste / span if span else 0:.1f}%) --")

        for cold in (True, False):
            label = "cold" if cold else "warm"
            for strategy in STRATEGIES:
                times, honest = [], True
                for _ in range(reps):
                    dt, _nbytes, was_cold = time_layer(shard, entries, data_start, strategy, cold)
                    times.append(dt)
                    honest = honest and (was_cold or not cold)
                best = min(times)
                rate = layer_bytes / best / GB
                note = "" if honest else "   NOT ACTUALLY COLD: the kernel kept the pages"
                print(f"    {label:>4} {strategy:>5}: {best * 1000:8.1f} ms  "
                      f"{rate:6.2f} GB/s  (median {statistics.median(times) * 1000:.1f} ms)"
                      f"{note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="model directories to measure")
    ap.add_argument("--reps", type=int, default=3,
                    help="timed reads per strategy (default 3). Best-of is reported, because the "
                         "interesting failure is a floor and noise only ever adds")
    ap.add_argument("--layers", type=int, default=3,
                    help="how many layers to sample (default 3: first, widest, last)")
    args = ap.parse_args()

    if not hasattr(os, "posix_fadvise"):
        print("NOTE: no posix_fadvise on this platform, so no read here can be made cold. "
              "Every number below measures the page cache.", file=sys.stderr)

    for path in args.paths:
        root = pathlib.Path(path)
        if not root.is_dir():
            print(f"\n=== {root} ===\n  not a directory", file=sys.stderr)
            continue
        try:
            report(root, args.reps, args.layers)
        except streaming.ShardError as exc:
            print(f"\n=== {root} ===\n  refused: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
