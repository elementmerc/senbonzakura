#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""SPIKE: how much host memory will this machine page-lock, and what does exceeding it cost?

RECONNAISSANCE, NOT A FEATURE. It answers one question and stops. Nothing in the tool calls it.

THE QUESTION

The streaming design loads one layer at a time from a host-side store and hands it to the GPU
while the GPU is busy with the previous layer. That overlap is the entire performance argument,
and it only happens if the host buffer is page-locked: pinned memory can be moved by a DMA engine
alongside the compute units, while pageable memory is staged through a driver bounce buffer that
serialises against the work it was supposed to hide behind.

Page-locking is a finite resource. Prior art reports a ceiling around 7.1 GB on its reference box,
with a store above it silently falling back to pageable and GPU utilisation dropping to 79%. That
is the failure this measures, and the reason it is worth measuring rather than assuming: a run
that crosses the ceiling does not fail, it gets slow, and it gets slow for a reason that looks
exactly like streaming being a bad idea.

WHAT IT MEASURED, on an RTX 3060 Laptop (6 GB) in WSL2 with 23.5 GB of host RAM, torch 2.13/cu126:

  Largest single pinned allocation      4 GB      (6 GB refused: CUDA out of memory)
  Cumulative pinned, 1 GB at a time     4 GB      (the same wall, so it is a total, not a request)
  Host to device, pinned                6.11 GB/s
  Host to device, pageable              6.02 GB/s

The ceiling is CUDA's, not the shell's: `ulimit -l` on that box is 64 MB and pinned allocations
sailed past it to sixty-four times that.

The bandwidth rows are the trap this spike nearly fell into. Pinned and pageable are within 2% of
each other, which reads as "pinning buys nothing" and is a real measurement of the wrong quantity:
a throughput test synchronises, so it cannot see overlap either way. Running a transfer against a
compute load instead, and reporting how much of the shorter load disappeared into the longer one:

  balanced loads, pinned                hid 99.9% of the shorter load   (1.96x)
  balanced loads, pageable              hid  2.5% of the shorter load   (1.01x)

So on this hardware pinning is the difference between complete overlap and none at all, and the
4 GB ceiling is a hard design constraint rather than a tuning note. A host-side store larger than
that does not degrade gracefully; it loses essentially the whole benefit.

Re-run this on any machine before trusting a streaming time estimate on it. The ceiling is a
property of the driver, the platform and the host, and WSL2 is not Linux for this purpose.
"""
import argparse
import gc
import time

import torch

GB = 1024 ** 3
MB = 1024 ** 2


def _release(*tensors):
    for t in tensors:
        del t
    gc.collect()
    torch.cuda.empty_cache()


def _best_of(fn, reps=3):
    """Best of N, because the interesting failure is a floor and noise only ever adds."""
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def largest_single(cap_gb):
    """The biggest one-shot page-locked allocation that succeeds."""
    print(f"\n=== largest single pinned allocation, ramping to at most {cap_gb:.1f} GB ===")
    best = 0.0
    for gb in (0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32):
        if gb > cap_gb:
            print(f"  stopped at the {cap_gb:.1f} GB cap rather than the machine's limit")
            break
        try:
            t0 = time.perf_counter()
            buf = torch.empty(int(gb * GB), dtype=torch.uint8, pin_memory=True)
            dt = time.perf_counter() - t0
            # A driver that quietly hands back pageable memory is the exact failure being hunted,
            # so success is not enough: the allocation has to actually be pinned.
            pinned = buf.is_pinned()
            _release(buf)
            print(f"  {gb:>5} GB  ok  pinned={pinned}  {dt:5.2f}s")
            if not pinned:
                break
            best = gb
        except (RuntimeError, MemoryError) as exc:
            print(f"  {gb:>5} GB  REFUSED  {type(exc).__name__}: {str(exc)[:100]}")
            break
    return best


def cumulative(cap_gb, step_gb=1):
    """Total page-locked memory held at once.

    Separate from the ramp above because the limit that bites may be on total locked pages rather
    than on any one request, and a layer store is many buffers, not one.
    """
    print(f"\n=== cumulative pinned, {step_gb} GB at a time, capped at {cap_gb:.1f} GB ===")
    held, total = [], 0.0
    try:
        while total + step_gb <= cap_gb:
            buf = torch.empty(int(step_gb * GB), dtype=torch.uint8, pin_memory=True)
            if not buf.is_pinned():
                print(f"  fell back to pageable at {total + step_gb:.0f} GB")
                break
            held.append(buf)
            total += step_gb
            print(f"  holding {total:>5.0f} GB")
        else:
            print(f"  reached the {cap_gb:.1f} GB cap with no refusal, so this is a floor")
    except (RuntimeError, MemoryError) as exc:
        print(f"  REFUSED at {total + step_gb:.0f} GB  {type(exc).__name__}: {str(exc)[:100]}")
    finally:
        _release(*held)
    return total


def throughput(mb=512, reps=5):
    """Raw host to device rate. Reported for completeness, and it is NOT the interesting number."""
    print(f"\n=== host to device throughput, {mb} MB x {reps} ===")
    for label, pin in (("pageable", False), ("pinned", True)):
        try:
            src = torch.empty(mb * MB, dtype=torch.uint8, pin_memory=pin)
        except (RuntimeError, MemoryError) as exc:
            print(f"  {label:>9}: could not allocate: {str(exc)[:80]}")
            continue
        dst = torch.empty(mb * MB, dtype=torch.uint8, device="cuda")
        # Bound as defaults rather than captured: the lambda is consumed inside this iteration so
        # late binding is harmless here, but a closure over a loop variable in a benchmark is the
        # shape that later silently measures the last configuration three times.
        dt = _best_of(lambda d=dst, s=src, p=pin: d.copy_(s, non_blocking=p), reps)
        print(f"  {label:>9}: {dt * 1000:7.1f} ms  {mb / 1024 / dt:5.2f} GB/s")
        _release(src, dst)


def overlap(pin, mb=256, copies=8, matmul_n=2048, steps=60):
    """THE MEASUREMENT THIS SPIKE EXISTS FOR.

    Run a transfer and a compute load separately, then together, and report how much of the
    shorter one vanished into the longer one. Reported as a fraction rather than as the plain
    serial/together ratio, because that ratio is bounded by how balanced the two loads happen to
    be: perfect overlap of a 50 ms compute behind a 330 ms transfer is only 1.15x, and nothing
    about 1.15x says it was perfect.

    A figure slightly ABOVE 100% is measurement noise, not a discovery. The three timings are
    separate best-of-N passes, so the compute load can come out a few milliseconds cheaper inside
    the combined run than it did alone, and the fraction then overshoots. Read anything from about
    95% up as complete overlap; the interesting contrast here is against single digits.
    """
    src = torch.empty(mb * MB, dtype=torch.uint8, pin_memory=pin)
    dst = torch.empty(mb * MB, dtype=torch.uint8, device="cuda")
    a = torch.randn(matmul_n, matmul_n, device="cuda", dtype=torch.float16)
    b = torch.randn(matmul_n, matmul_n, device="cuda", dtype=torch.float16)
    copy_stream = torch.cuda.Stream()

    def compute():
        acc = a
        for _ in range(steps):
            acc = torch.mm(acc, b)
        return acc

    def transfer():
        with torch.cuda.stream(copy_stream):
            for _ in range(copies):
                dst.copy_(src, non_blocking=True)

    def both():
        transfer()
        return compute()

    t_compute = _best_of(compute)
    t_transfer = _best_of(transfer)
    t_both = _best_of(both)
    serial = t_compute + t_transfer
    hidden = (serial - t_both) / min(t_compute, t_transfer)
    print(f"  {'pinned' if pin else 'pageable':>9}: compute {t_compute * 1000:6.1f} ms | "
          f"transfer {t_transfer * 1000:6.1f} ms | together {t_both * 1000:6.1f} ms | "
          f"{serial / t_both:.2f}x | hid {hidden * 100:5.1f}% of the shorter load")
    _release(src, dst, a, b)
    return hidden


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap-fraction", type=float, default=0.5,
                    help="ramp to at most this fraction of host RAM (default 0.5). Page-locking "
                         "takes memory away from everything else on the machine, so this is a cap "
                         "on politeness rather than on capability")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: this spike measures a host-to-device property and has "
                         "nothing to say without one")

    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"vram: {free_b / GB:.2f} GB free of {total_b / GB:.2f} GB")
    with open("/proc/meminfo", encoding="utf-8") as fh:
        meminfo = dict(line.split(":", 1) for line in fh)
    host_gb = int(meminfo["MemTotal"].split()[0]) / (1024 ** 2)
    cap = max(1.0, host_gb * args.cap_fraction)
    print(f"host: {host_gb:.1f} GB total, ramping to at most {cap:.1f} GB")

    single = largest_single(cap)
    total = cumulative(cap)
    throughput()

    print("\n=== transfer against compute, at three balances ===")
    print("  hid 100% means the shorter load ran entirely inside the longer one.")
    for name, kw in (("transfer-heavy", dict(copies=8, steps=60)),
                     ("balanced", dict(copies=8, steps=380)),
                     ("compute-heavy", dict(copies=2, steps=380))):
        print(f"\n  -- {name} --")
        for pin in (False, True):
            overlap(pin, **kw)

    print(f"\nSUMMARY largest_single_pinned_gb={single} cumulative_pinned_gb={total:.0f} "
          f"host_gb={host_gb:.1f}")


if __name__ == "__main__":
    main()
