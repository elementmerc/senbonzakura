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
import shutil
import subprocess
import sys
import tempfile
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
    ap.add_argument("--keep-tied-head", action="store_true",
                    help="keep a separate output head even when the config declares tied "
                         "embeddings and the head is byte-identical to them. The default drops it, "
                         "because a lone embedding is also the output projection and is quantised "
                         "accurately as a result: measured at Q3_K_L, dropping the duplicate moves "
                         "the embedding from Q3_K to Q6_K, which is 8.5x closer to the original, "
                         "in a smaller file. Use this to reproduce a file made before that "
                         "behaviour, or to compare the two")
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


#: The two names for one tensor when a config declares tied embeddings.
TIED_HEAD = "lm_head.weight"
TIED_EMBED = "model.embed_tokens.weight"


def tied_head_state(model_dir):
    """Whether this checkpoint ships an output head its own config says is redundant.

    Returns (verdict, shard, detail). The verdict is one of:

      "absent"      the config does not declare tying, or there is no separate head. Nothing to do,
                    and this covers most large models: Qwen3-30B-A3B declares tie_word_embeddings
                    false and its head is real.
      "redundant"   the config declares tying and the head is byte-identical to the embedding, so
                    it is a duplicate and dropping it changes nothing about the model.
      "contradicts" the config declares tying and the head DIFFERS. One of the two is wrong and
                    this code cannot know which, so the caller refuses rather than picking.

    WHY THIS EXISTS. `llama-quantize` decides the embedding's precision from whether a separate
    output head is present, because a lone `token_embd` is also the output projection and must
    stay accurate. Measured on a tied model at Q3_K_L: with the duplicate present the embedding
    lands at Q3_K and its error against f16 is 0.1509; with it absent the embedding lands at Q6_K
    and the error is 0.0177, EIGHT AND A HALF TIMES better, in a file that is also smaller. The
    output projection is Q6_K either way. So this is not a trade: the duplicate costs accuracy and
    disk at once, and which you get is decided by whether the checkpoint happened to ship a tensor
    its own config calls redundant.
    """
    cfg = read_config(model_dir)
    if not cfg.get("tie_word_embeddings", False):
        return "absent", None, "the config does not declare tied embeddings"

    d = Path(model_dir)
    shards = [p for p in weight_files(d) if p.suffix == ".safetensors"]
    if not shards:
        # A .bin checkpoint. Detectable, but rewriting a pickle is a different job and torch
        # checkpoints are not what this project produces or expects.
        return "absent", None, "no safetensors shards to inspect"

    from safetensors import safe_open
    head_shard = embed_shard = None
    for p in shards:
        with safe_open(p, framework="np") as h:
            keys = set(h.keys())
        if TIED_HEAD in keys:
            head_shard = p
        if TIED_EMBED in keys:
            embed_shard = p
    if head_shard is None:
        return "absent", None, "the config declares tying and no separate head is present"
    if embed_shard is None:
        return "contradicts", head_shard, (
            f"{TIED_HEAD} is present and {TIED_EMBED} is not, so there is nothing to tie it to")

    import numpy as np
    with safe_open(head_shard, framework="np") as h:
        head = h.get_tensor(TIED_HEAD)
    with safe_open(embed_shard, framework="np") as h:
        embed = h.get_tensor(TIED_EMBED)
    if head.shape != embed.shape:
        return "contradicts", head_shard, (
            f"{TIED_HEAD} is {head.shape} and {TIED_EMBED} is {embed.shape}")
    if not np.array_equal(head, embed):
        # Byte equality, not a tolerance. The claim being checked is "these are the same tensor",
        # and two tensors that merely round to each other are not that.
        diff = float(np.abs(head.astype(np.float64) - embed.astype(np.float64)).max())
        return "contradicts", head_shard, (
            f"{TIED_HEAD} and {TIED_EMBED} have the same shape and different values "
            f"(largest difference {diff:g})")
    return "redundant", head_shard, f"{TIED_HEAD} is byte-identical to {TIED_EMBED}"


def without_tied_head(model_dir, workdir, shard, log=print):
    """A view of the checkpoint with the redundant head removed, built without copying weights.

    Every file is symlinked into `workdir` and only the one shard holding the duplicate is
    rewritten, so the cost is that shard rather than the model. On a tied checkpoint the head sits
    beside the embedding in the first shard, and tied models are small ones: large models declare
    tying false, so the expensive case does not arise.

    The safetensors index, if there is one, is rewritten too. The converter reads the tensors
    present in each part and checks them against the index, so an index still promising a tensor
    that has been removed would fail the converter's own consistency check.
    """
    from safetensors.numpy import load_file, save_file

    src, work = Path(model_dir), Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    index_name = "model.safetensors.index.json"
    for p in src.iterdir():
        if not p.is_file() or p.name in (shard.name, index_name):
            continue
        (work / p.name).symlink_to(p.resolve())

    tensors = load_file(shard)
    dropped = tensors.pop(TIED_HEAD)
    save_file(tensors, work / shard.name, metadata={"format": "pt"})
    log(f"  dropped {TIED_HEAD} {tuple(dropped.shape)}: the config declares tied embeddings and "
        f"it is byte-identical to {TIED_EMBED}")

    index = src / index_name
    if index.is_file():
        doc = json.loads(index.read_text(encoding="utf-8"))
        wmap = doc.get("weight_map", {})
        wmap.pop(TIED_HEAD, None)
        meta = doc.get("metadata")
        if isinstance(meta, dict) and "total_size" in meta:
            meta["total_size"] = int(meta["total_size"]) - int(dropped.nbytes)
        (work / index_name).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return work


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

    source = Path(a.model)
    tmp_view = None
    verdict, shard, detail = tied_head_state(a.model)
    if verdict == "contradicts":
        raise SystemExit(
            f"this checkpoint's config says tie_word_embeddings is true, and its weights disagree: "
            f"{detail}.\nOne of the two is wrong and nothing here can tell which, so the conversion "
            f"stops rather than choosing. Fix the config or the weights, or pass "
            f"--keep-tied-head to convert it exactly as it is.")
    if verdict == "redundant" and a.keep_tied_head:
        log(f"  NOTE: {detail}, and --keep-tied-head was given, so it is kept. The embedding will "
            f"be quantised as an ordinary embedding rather than as an output projection.")
    elif verdict == "redundant":
        tmp_view = Path(tempfile.mkdtemp(prefix="senbonzakura-tied-", dir=out.parent))
        # The VIEW keeps the source's own directory name. The converter derives `general.name`
        # from the directory it is pointed at, so converting from `senbonzakura-tied-zkxhdz01/`
        # stamped that into the model's metadata: a published GGUF called
        # "Senbonzakura Tied Zkxhdz01". The random part is the parent and the name is the leaf.
        source = without_tied_head(a.model, tmp_view / Path(a.model).resolve().name, shard,
                                   log=log)

    argv_c = [sys.executable, str(pre["script"]), str(source),
              "--outfile", str(out), "--outtype", a.outtype]
    if a.use_temp_file:
        argv_c.append("--use-temp-file")

    started = time.monotonic()
    # No timeout, for the same reason `quantise` has none: converting a large model is genuinely
    # long, and a ceiling here would kill a job with its work nearly done.
    try:
        r = subprocess.run(argv_c, check=False)
    finally:
        if tmp_view is not None:
            # Symlinks and one rewritten shard. Removed on every path, including a converter crash,
            # because a stray view of a checkpoint is confusing to find later and it sits beside
            # the output rather than in a temp directory someone would think to clear.
            shutil.rmtree(tmp_view, ignore_errors=True)
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
