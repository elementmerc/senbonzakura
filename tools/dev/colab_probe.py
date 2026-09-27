#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Measure what a free Colab session can actually do, so the notebook can stop guessing.

WHY THIS EXISTS, 2026-09-27

`notebooks/senbonzakura_colab.ipynb` tells a reader the run takes "about fifteen minutes", and
nobody has ever run it in Colab. The figure was carried over from a 6 GB card on a desk. Three
separate numbers in that notebook (fifteen minutes at the top, ten in the install cell, ten in the
edit cell) are all unmeasured, and one of them is the first promise a stranger meets.

Extrapolating from our own hardware is what produced the "full abliteration in under three minutes
on a legacy CPU" claim, which measured at 857 seconds on sixteen modern cores. So this does not
extrapolate. It runs in the environment the reader will use and prints figures from it.

THE PART THAT IS NOT JUST TIMING

A free Colab session usually gets an NVIDIA T4. A T4 is Turing, compute capability 7.5, and it has
no hardware bfloat16; that arrived with Ampere at 8.0. `load_model_and_tokenizer` hardcodes
`dtype=torch.bfloat16` with no capability check and no flag to override it, so on the most common
free GPU in the world every matmul in this pipeline may be running emulated. That would be invisible
in the output and would show up only as everything being slow.

This probe measures the penalty rather than asserting it, because `torch.cuda.is_bf16_supported()`
has meant different things across torch versions and the honest answer is a wall-clock comparison.

WHAT IT DOES NOT DO

It does not change any weights outside the directory it is told to write to, does not print model
output, and does not upload anything. The edit phase writes an abliterated checkpoint to the Colab
machine's local disk, which Google deletes with the runtime.

HOW IT IS RUN

In a Colab cell, on a GPU runtime:

    !pip install --quiet senbonzakura
    !curl -sL https://raw.githubusercontent.com/elementmerc/senbonzakura/dev/tools/dev/colab_probe.py -o probe.py
    !python probe.py

Add `--skip-edit` to get the environment report and the measure-only tier in about four minutes,
without the twenty-minute edit. Paste the REPORT block back verbatim; it is written to be read by
somebody who was not there.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

#: The demo model. Measured 58.6% hard refusal on the bundled evaluation set at n=128, which is the
#: reason it was chosen over the Qwen3 pair the notebook currently uses: Qwen3-0.6B measured 4.7%,
#: under the 5% floor this tool now refuses below, so the notebook's own flagship command would be
#: refused by the tool it is demonstrating.
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

#: The reduced-budget edit, verbatim from the CPU floor run that exited 0 and moved refusal from
#: 58.6% to 28.1%. Cut any further and the search has nothing to choose between; this is the floor
#: of a demo that still shows motion, not a recommended configuration, and the notebook has to say
#: so wherever it quotes a number this produced.
DEMO_BUDGET = [
    "--trials", "2",
    "--dir-prompts", "16",
    "--eval-refusal", "32",
    "--capability-n", "0",
    "--max-directions", "1",
    "--method", "single-pass",
    "--track", "default",
]

#: A phase that has not finished inside this is not a phase, it is a hung session. Colab's own idle
#: timeout is the outer bound and it is not ours to set, so every subprocess here carries its own.
TIMEOUTS = {"doctor": 300, "capability": 900, "compass": 1200, "edit": 3600}


def _say(msg=""):
    print(msg, flush=True)


def _rule(title):
    _say()
    _say(f"── {title} " + "─" * max(0, 74 - len(title)))


# ── the environment, before anything is timed in it ──────────────────────────────

def environment():
    """What Google actually lent us. Every figure here is read, none is assumed."""
    env = {"platform": platform.platform(), "python": sys.version.split()[0]}

    try:
        import torch
    except ImportError:
        env["torch"] = None
        env["gpu"] = "torch is not installed, so nothing below can be measured"
        return env

    env["torch"] = torch.__version__
    env["cuda_build"] = torch.version.cuda
    if not torch.cuda.is_available():
        env["gpu"] = None
        return env

    major, minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    env.update({
        "gpu": torch.cuda.get_device_name(0),
        "capability": f"{major}.{minor}",
        "vram_gb": round(props.total_memory / 1024**3, 2),
        # Ampere is 8.0. Below it, bfloat16 has no hardware path and torch emulates, which is
        # correct and slow. The version-dependent helper is recorded beside the capability rather
        # than instead of it, because the two have disagreed.
        "bf16_hardware": (major, minor) >= (8, 0),
        "torch_says_bf16_supported": bool(torch.cuda.is_bf16_supported()),
    })

    env["disk_free_gb"] = round(shutil.disk_usage("/").free / 1024**3, 1)
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    env["ram_available_gb"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass
    return env


#: Square matmul side length. Large enough that kernel launch overhead is not what is being timed.
MATMUL_SIZE = 4096


def _time_one_matmul(torch, dtype):
    """One dtype, timed, or the reason it could not be. Isolated per dtype on purpose.

    A card that cannot do bf16 at all, or has no room for two 4096-square matrices, must still
    report the dtypes it can do: the comparison is the measurement, and a single OOM taking the
    whole table with it would leave nothing to compare.
    """
    size = MATMUL_SIZE
    try:
        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)
        for _ in range(3):                          # warm the kernels and the clocks
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()
        per = (time.perf_counter() - t0) / 20
        del a, b
        torch.cuda.empty_cache()
        # 2 * n^3 flops per matmul, reported in TFLOP/s so the figure is comparable to a spec sheet
        # rather than only to itself.
        return {"ms": round(per * 1000, 2), "tflops": round(2 * size**3 / per / 1e12, 1)}
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        return {"error": str(e)[:200]}


def dtype_penalty(env):
    """Wall-clock bf16 against fp16 on this card, which is the claim that matters.

    A ratio near 1.0 means bfloat16 costs nothing here. A ratio well above 1.0 means this pipeline's
    hardcoded bfloat16 is being emulated, and every timing below is paying for it.
    """
    import torch

    if not env.get("gpu"):
        return None

    out = {}
    for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        out[name] = _time_one_matmul(torch, dt)

    if "ms" in out.get("bf16", {}) and "ms" in out.get("fp16", {}):
        out["bf16_over_fp16"] = round(out["bf16"]["ms"] / out["fp16"]["ms"], 2)
    return out


# ── the timed phases ────────────────────────────────────────────────────────────

def run(label, argv, timeout, log_path):
    """One phase, timed, with its whole transcript kept for the phase breakdown.

    Output is streamed to a file rather than a pipe: a pipe that fills blocks the child, and this
    tool's progress bars are chatty.
    """
    _rule(label)
    _say(f"$ {' '.join(argv)}")
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as fh:
        try:
            code = subprocess.call(argv, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    seconds = time.perf_counter() - t0

    tail = ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-1500:]
    except OSError:
        pass

    if timed_out:
        _say(f"TIMED OUT after {timeout}s. This is the finding, not a failure to report.")
    else:
        _say(f"exit {code} in {seconds:.0f}s ({seconds / 60:.1f} min)")
    if code != 0:
        _say("last of its output:")
        _say(tail)
    return {"label": label, "seconds": round(seconds, 1), "exit": code,
            "timed_out": timed_out, "log": log_path}


def phase_breakdown(log_path):
    """Per-trial timings out of the tool's own transcript, so the total can be apportioned.

    Reads rather than instruments: the tool already prints its trial boundaries, and a probe that
    needed the tool changed to measure it would be measuring a different tool.
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return {}
    found = {}
    for pattern, key in (
        (r"trial\s+(\d+)\s*/\s*\d+", "trials_seen"),
        (r"baseline refusal[^\n]*?(\d+\.?\d*)\s*%", "baseline_refusal_pct"),
        (r"post[- ]bake refusal[^\n]*?(\d+\.?\d*)\s*%", "post_bake_refusal_pct"),
        (r"KL[^\n]*?(\d+\.\d+)", "kl"),
    ):
        hits = re.findall(pattern, text, re.IGNORECASE)
        if hits:
            found[key] = hits[-1] if key != "trials_seen" else len(set(hits))
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure a free Colab session against what the notebook promises.")
    ap.add_argument("--model", default=MODEL, help=f"the model to probe (default: {MODEL})")
    ap.add_argument("--out", default="/content/probe", help="where to write, including the edit")
    ap.add_argument("--skip-edit", action="store_true",
                    help="environment and the measure-only tier only, about four minutes")
    ap.add_argument("--report", default="colab-probe-report.json",
                    help="where to write the machine-readable copy of everything below")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    logs = os.path.join(a.out, "logs")
    os.makedirs(logs, exist_ok=True)

    _rule("ENVIRONMENT")
    env = environment()
    for k, v in env.items():
        _say(f"  {k:28} {v}")

    if not env.get("torch"):
        _say("\nStop here: install senbonzakura first, which brings torch with it.")
        return 2
    if not env.get("gpu"):
        _say("\nNO GPU. Runtime, Change runtime type, GPU, then run this again.")
        _say("Refusing the edit phase on CPU on purpose: it measured 857s on sixteen cores and")
        _say("a Colab CPU session is slower, so it would eat the session and prove nothing.")
        a.skip_edit = True

    _rule("DTYPE, because this pipeline hardcodes bfloat16")
    penalty = dtype_penalty(env)
    if penalty:
        for name in ("fp32", "fp16", "bf16"):
            row = penalty.get(name, {})
            if "ms" in row:
                _say(f"  {name:6} {row['ms']:8.2f} ms   {row['tflops']:6.1f} TFLOP/s")
            else:
                _say(f"  {name:6} {row.get('error', 'not measured')}")
        ratio = penalty.get("bf16_over_fp16")
        if ratio is not None:
            _say(f"\n  bfloat16 costs {ratio}x fp16 on this card.")
            if ratio > 1.3:
                _say("  THAT IS THE EMULATION PENALTY. Every timing below is paying it, and the")
                _say("  loader has no flag to avoid it. This is a defect, not a property of Colab.")
            else:
                _say("  No meaningful penalty, so the hardcoded bfloat16 is fine on this card.")

    phases = [run("doctor", ["senbonzakura", "doctor"], TIMEOUTS["doctor"],
                  os.path.join(logs, "doctor.log"))]

    # THE MEASURE-ONLY TIER, which is the part of the notebook that should survive a dead session.
    phases.append(run(
        "capability, n=40", ["senbonzakura", "capability", "--model", a.model, "--n", "40",
                             "--out", os.path.join(a.out, "before.json")],
        TIMEOUTS["capability"], os.path.join(logs, "capability.log")))

    # The bundled track is addressed as `default/<split>` rather than with `--track`: compass takes
    # the two sides explicitly, because which rows a figure came from is the whole question it
    # answers. Copied from the CPU example in `docs/guide/quickstart.md` so the two cannot drift.
    phases.append(run(
        "compass, the instrument with its own controls",
        ["senbonzakura", "compass", "--model", a.model,
         "--harmful", "default/bad_eval_ds", "--harmless", "default/good_ds",
         "--skip-harmful", "0", "--skip-harmless", "0", "--n", "64",
         "--out", os.path.join(a.out, "compass.json")],
        TIMEOUTS["compass"], os.path.join(logs, "compass.log")))

    edit = None
    if not a.skip_edit:
        edit = run("the reduced-budget edit",
                   ["senbonzakura", a.model, "--out", os.path.join(a.out, "abliterated"),
                    *DEMO_BUDGET],
                   TIMEOUTS["edit"], os.path.join(logs, "edit.log"))
        phases.append(edit)

    _rule("REPORT, paste this back")
    measured = sum(p["seconds"] for p in phases if not p["timed_out"])
    _say(f"  model                        {a.model}")
    _say(f"  gpu                          {env.get('gpu')} ({env.get('vram_gb')} GB, "
         f"capability {env.get('capability')})")
    _say(f"  bfloat16 in hardware         {env.get('bf16_hardware')}")
    if penalty and penalty.get("bf16_over_fp16") is not None:
        _say(f"  bfloat16 cost vs fp16        {penalty['bf16_over_fp16']}x")
    _say()
    for p in phases:
        state = "TIMED OUT" if p["timed_out"] else f"exit {p['exit']}"
        _say(f"  {p['label']:44} {p['seconds'] / 60:6.1f} min   {state}")
    _say(f"  {'TOTAL of the phases that completed':44} {measured / 60:6.1f} min")
    _say()
    _say("  The notebook currently promises fifteen minutes at the top, ten in the install cell")
    _say("  and ten in the edit cell. Compare those three against the lines above before any of")
    _say("  them is left in place.")

    if edit:
        found = phase_breakdown(edit["log"])
        if found:
            _say()
            _say("  out of the edit's own transcript:")
            for k, v in found.items():
                _say(f"    {k:26} {v}")

    report = {"environment": env, "dtype": penalty, "phases": phases,
              "model": a.model, "budget": DEMO_BUDGET,
              "edit_breakdown": phase_breakdown(edit["log"]) if edit else None}
    try:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        _say(f"\n  written to {a.report}, and the per-phase transcripts are under {logs}")
    except OSError as e:
        _say(f"\n  could not write {a.report}: {e}")

    failed = [p["label"] for p in phases if p["exit"] != 0]
    if failed:
        _say(f"\n  THESE DID NOT SUCCEED: {', '.join(failed)}")
        _say("  That is a result too. Their transcripts are the most useful thing here.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
