# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura convert`: turn edited weights into a GGUF, and check what came out.

WHAT THIS REPLACES

`gguf2.sh`, `gguf3.sh` and `gguf-atlas.sh`, three files whose header comment is identical word for
word ("Convert the abliterated releases to GGUF and publish them") and which differed by a digit
and by which machine's paths were baked into them. The house rule that produced this file: the
moment a script gets a digit it has earned a subcommand.

WHY THE CONVERTER IS VENDORED RATHER THAN CALLED FROM THE ENVIRONMENT

Conversion lives in llama.cpp's Python and nowhere else this package can reach. The `gguf` pip
package writes GGUF files but does not know how to read a transformers checkpoint; the mapping
from one architecture's tensor names to GGUF's is 85 modules of upstream code.

WHY IT ALSO VENDORS `gguf-py`, WHICH IS THE PART THAT LOOKS REDUNDANT

The converter's first real statement is `import gguf`, and **the PyPI package of that name is not
the same code as the `gguf-py` inside the llama.cpp tree**. Measured on 2026-08-18: both declare
version 0.19.0, and the PyPI one is missing constants the pinned converter uses. The visible
consequence was that `conversion/lfm2.py` raised on import while `--print-supported-models` went on
advertising `Lfm2MoeForCausalLM`, because that list is built from a static registry rather than
from what actually imported. Support claimed, support absent, exit code zero.

No version constraint can separate those two packages, because they share a version number and
differ in contents. Vendoring `gguf-py/` from the same tag makes the converter and its library
consistent by construction, and the upstream entry point picks the sibling up on its own.

WHAT THIS ADDS OVER CALLING THE SCRIPT BY HAND

The pre-flight, and reading the output back. A conversion that fails halfway leaves a file that is
exactly the shape of a real one, and this project has already recorded a 987 MB fragment of a
5.16 GB GGUF as model quality once.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import gguf_io
from .crashsafe import free_bytes_for
from .vendored import VendorError, find_script

#: What the vendored converter can be asked to write. Deliberately not every value it accepts:
#: each of these is a type `gguf_io` can name in the output header, so a conversion cannot quietly
#: produce something other than what was asked for.
OUT_TYPES = ("bf16", "f16", "f32", "q8_0", "auto")

#: How the converter's `--outtype` names map to what the GGUF header records, for the read-back.
#: `auto` is absent on purpose: it resolves to whichever 16-bit type suits the weights, so there is
#: nothing to assert against, and the check falls back to "some float type".
HEADER_TYPE = {"bf16": "BF16", "f16": "F16", "f32": "F32", "q8_0": "Q8_0"}

#: GGUF is a little larger than the safetensors it came from at the same precision (it carries the
#: vocabulary and the metadata), so this is a floor for the disk check rather than an estimate.
SIZE_HEADROOM = 1.20

#: The pre-flight architecture check costs one subprocess and about two seconds, against a
#: conversion measured in minutes and a rented card measured in money.
SUPPORTED_TIMEOUT_S = 180


class ConvertError(Exception):
    """A conversion that cannot be trusted, phrased for a person."""


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura convert",
        description="Convert a transformers checkpoint to GGUF with the pinned converter, then "
                    "verify what was written.")
    ap.add_argument("model", help="a directory of safetensors, such as an --out from a run")
    ap.add_argument("out", nargs="?", default=None,
                    help="output path (default: the model directory's name, beside it)")
    ap.add_argument("--outtype", default="bf16", choices=OUT_TYPES,
                    help="precision to write (default: bf16, which is lossless for the bf16 "
                         "checkpoints this tool edits)")
    ap.add_argument("--quantise", default=None, metavar="TYPE",
                    help="after converting, quantise to this type (e.g. Q4_K_M) and remove the "
                         "intermediate. One command from edited weights to something llama.cpp "
                         "will serve")
    ap.add_argument("--imatrix", default=None, metavar="FILE",
                    help="with --quantise, apply this importance matrix (an `i1-` quantisation)")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="with --quantise, keep the full-precision GGUF as well")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    ap.add_argument("--use-temp-file", action="store_true",
                    help="stream through a temporary file: slower, and the way to convert a model "
                         "larger than memory")
    ap.add_argument("--skip-arch-check", action="store_true",
                    help="skip the pre-flight architecture check. For an architecture the pinned "
                         "converter supports under a name this cannot read from config.json")
    return ap


def read_config(model_dir):
    """The checkpoint's `config.json`, or a failure that says which of the two problems it is."""
    d = Path(model_dir)
    if not d.is_dir():
        raise ConvertError(f"no model directory at {d}.")
    cfg = d / "config.json"
    if not cfg.is_file():
        raise ConvertError(
            f"{d} has no config.json, so it is not a transformers checkpoint. If it is already a "
            f"GGUF, `senbonzakura quantise` is the command you want.")
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise ConvertError(f"{cfg} is not readable JSON ({e}).") from e


def weight_files(model_dir):
    """The weight shards, which are what makes this a checkpoint rather than a directory."""
    d = Path(model_dir)
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix in (".safetensors", ".bin"))


def supported_architectures(script, *, timeout=SUPPORTED_TIMEOUT_S):
    """Every architecture the VENDORED converter will accept, as it reports them itself.

    Asked of the script rather than kept as a list here, because a list here would be a copy that
    goes stale at the next pin bump, and the failure mode of a stale copy is refusing a model the
    converter would happily have taken.
    """
    r = subprocess.run([sys.executable, str(script), "--print-supported-models"],
                       capture_output=True, timeout=timeout, check=False)
    out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    names = {ln.strip().lstrip("-").strip() for ln in out.splitlines()
             if ln.strip().startswith("- ")}
    # A module that failed to import is NOT support, whatever the registry above says. This is the
    # exact gap that hid the missing gguf-py: the list is static and the imports are not.
    broken = sorted({ln.split(":")[0].split()[-1] for ln in out.splitlines()
                     if "failed to load model module" in ln.lower()})
    return names, broken


def preflight(model_dir, out, *, force, skip_arch_check, log=print):
    """Everything checkable before a long job, because none of it is worth finding halfway."""
    cfg = read_config(model_dir)
    d = Path(model_dir)

    shards = weight_files(d)
    if not shards:
        raise ConvertError(
            f"{d} holds a config.json and no .safetensors or .bin weights. A config-only directory "
            f"is what a cache leaves behind when only the metadata was ever downloaded, and it "
            f"converts to nothing.")

    if "quantization_config" in cfg:
        raise ConvertError(
            f"{d} is already quantised ({cfg['quantization_config'].get('quant_method', 'unknown')}"
            f"). Converting it would bake one lossy step into another; convert the original "
            f"weights instead.")

    arch = (cfg.get("architectures") or [None])[0]
    if not arch:
        raise ConvertError(f"{d}/config.json names no architecture, so nothing can be dispatched.")

    if Path(out).resolve() == d.resolve():
        raise ConvertError(f"the output path is the model directory ({d}). Name a file instead.")
    if Path(out).exists() and not force:
        raise ConvertError(f"{out} exists. Pass --force to overwrite it, or choose another path.")

    try:
        script = find_script("convert_hf_to_gguf.py")
    except VendorError as e:
        raise ConvertError(str(e)) from e

    if not skip_arch_check:
        names, broken = supported_architectures(script)
        if broken:
            # Loud, because this is the failure that shipped: a module that did not import while
            # its architectures stayed on the advertised list.
            raise ConvertError(
                f"the vendored converter could not import {len(broken)} of its architecture "
                f"modules ({', '.join(broken[:5])}), so the architectures they provide are "
                f"advertised and absent. This is a broken vendoring rather than anything about "
                f"your model: re-run `python tools/vendor_llama.py`.")
        if names and arch not in names:
            raise ConvertError(
                f"the pinned converter does not support {arch}. It supports {len(names)} "
                f"architectures and this is not among them. Either the pin predates support for "
                f"this model, in which case refreshing it at the next release reaches it, or the "
                f"architecture genuinely is not supported upstream yet.")

    total = sum(p.stat().st_size for p in shards)
    need = int(total * SIZE_HEADROOM)
    free = free_bytes_for(out)
    if free is not None and free < need:
        raise ConvertError(
            f"only {free / 1e9:.1f} GB free where the output goes and the weights are "
            f"{total / 1e9:.1f} GB. A GGUF at the same precision is slightly larger than the "
            f"checkpoint, not smaller: free space or choose an output on a larger volume.")
    return {"architecture": arch, "shards": len(shards), "bytes": total, "script": script}


def default_output(model_dir, outtype):
    d = Path(model_dir).resolve()
    return d.parent / f"{d.name}-{outtype}.gguf"


def run(argv=None, log=print):
    a = build_parser().parse_args(argv)
    out = Path(a.out) if a.out else default_output(a.model, a.outtype)

    try:
        pre = preflight(a.model, out, force=a.force, skip_arch_check=a.skip_arch_check, log=log)
    except ConvertError as e:
        raise SystemExit(f"cannot convert: {e}") from e

    log(f"convert {Path(a.model).name} ({pre['architecture']}, {pre['shards']} shard(s), "
        f"{pre['bytes'] / 1e9:.2f} GB) -> {out.name} [{a.outtype}]")
    log(f"  using the vendored converter at {pre['script']}")

    argv_c = [sys.executable, str(pre["script"]), str(a.model),
              "--outfile", str(out), "--outtype", a.outtype]
    if a.use_temp_file:
        argv_c.append("--use-temp-file")

    started = time.monotonic()
    # No timeout, for the same reason `quantise` has none: converting a large model is genuinely
    # long, and a ceiling here would kill a job with its work nearly done.
    r = subprocess.run(argv_c, check=False)
    took = time.monotonic() - started

    if r.returncode != 0:
        if out.exists():
            out.unlink()
            log(f"  removed the partial {out.name}")
        raise SystemExit(
            f"the converter exited {r.returncode} after {took:.0f}s. Nothing usable was written.")

    # READ IT BACK. An exit code is a statement about a process; everything that matters here is a
    # statement about a file.
    expect = HEADER_TYPE.get(a.outtype)
    try:
        got = gguf_io.verify(out, expect_quant=expect or False)
    except gguf_io.GGUFError as e:
        raise SystemExit(
            f"the converter reported success and the output does not verify: {e}\n"
            f"Left at {out} for inspection rather than deleted, because what is wrong with it is "
            f"the interesting part.") from e

    log(f"  wrote {out.name}: {out.stat().st_size / 1e9:.2f} GB, {got['tensor_count']} tensors, "
        f"{took:.0f}s")
    log(f"  verified: {got['file_type']}, architecture {got['architecture']}")

    if a.quantise:
        from . import quantise
        log(f"quantising to {a.quantise}")
        q_argv = [str(out), "--type", a.quantise]
        if a.imatrix:
            q_argv += ["--imatrix", a.imatrix]
        if a.force:
            q_argv.append("--force")
        if not a.keep_intermediate:
            q_argv.append("--prune-source")
        rc = quantise.run(q_argv, log=log)
        if rc != 0:
            return rc
    return 0


def main(argv=None):
    return run(argv)
