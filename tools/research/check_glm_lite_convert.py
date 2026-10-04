#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Can an abliterated GLM-4.7-Flash checkpoint be turned into a GGUF that llama.cpp will load?

WHY THIS EXISTS

GLM-4.7-Flash carries one multi-token-prediction layer, declared in its config as
`num_nextn_predict_layers: 1`. transformers does not load that layer, so a checkpoint this tool
saves keeps the config value and loses the tensors. The pinned converter counts the prediction
layer into the block total unless it is given `--no-mtp` (`vendor/src/conversion/glm.py`), and
`senbonzakura convert` never passes it. The likely result is a GGUF whose header declares one
more block than it holds: conversion exits 0, the file looks perfect, and the failure arrives at
load time on whatever machine tries to serve it. Finding that on a rented GPU after a paid run is
the expensive version of this check.

WHAT IT DOES, ON NOTHING BIGGER THAN A FEW MEGABYTES

0. Pre-flight: the libraries know `glm4_moe_lite`, the pinned converter and `llama-imatrix` are
   present (the converter is fetched at build time, so a checkout may lack it), and the scratch
   directory is on real disk.
1. Build a tiny but structurally real `glm4_moe_lite` from a small config with one prediction
   layer declared, save it the way a run saves its output, and assert the scenario actually holds:
   the config still declares the layer and no prediction-layer tensor was written. Then add the
   real GLM tokenizer, because the converter checks it against a known fingerprint.
2. Convert it through `senbonzakura convert`, exactly as shipped, and read the GGUF header back:
   the declared block count must equal the blocks the tensors cover.
3. Load it with the pinned `llama-imatrix`, which does a full model load. `llama-quantize` would
   not do: it copies tensors and never notices a missing block.

The last line of output is either `GLM_LITE_CONVERT: PASS ...` with exit 0, or
`GLM_LITE_CONVERT: FAIL step=<n> ...` with exit 1. Nothing else counts as a pass.

No model weights are downloaded. The only network access is the two tokenizer files, pinned to
one revision so a rerun reads the same bytes.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ID = "zai-org/GLM-4.7-Flash"
#: The snapshot whose config.json was read to build the tiny config below.
REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

#: From the real config at REVISION. The vocabulary has to match the real tokenizer, and the
#: special-token ids are what the converter looks up.
VOCAB_SIZE = 154880
PAD_ID = 154820
EOS_IDS = [154820, 154827, 154829]

#: Three decoder layers: the first dense (`first_k_dense_replace=1`), the other two MoE, which is
#: the real model's layout in miniature. The MLA ranks are small but keep the real ratios' shape.
TINY = dict(
    hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
    num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=3,
    first_k_dense_replace=1, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
    n_group=1, topk_group=1,
    q_lora_rank=32, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
    num_nextn_predict_layers=1,
    vocab_size=VOCAB_SIZE, pad_token_id=PAD_ID, eos_token_id=EOS_IDS,
    tie_word_embeddings=False,
)

#: Prediction-layer tensor names as they appear in the published checkpoint and in llama.cpp.
_MTP = re.compile(r"(nextn|\bmtp\b|eh_proj|\benorm\b|\bhnorm\b|shared_head)")

CONVERT_TIMEOUT_S = 900
IMATRIX_TIMEOUT_S = 600


class StepError(Exception):
    """One step's failure, with the step number the final line reports."""

    def __init__(self, step, reason):
        super().__init__(reason)
        self.step = step
        self.reason = reason


def log(msg):
    print(msg, flush=True)


def _tail(text, lines=25):
    return "\n".join((text or "").strip().splitlines()[-lines:])


def _on_tmpfs(path):
    """Whether `path` lives on a RAM-backed filesystem, read from the mount table."""
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    best, fstype = "", ""
    target = str(path.resolve())
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mnt = parts[1]
        if (target == mnt or target.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
            best, fstype = mnt, parts[2]
    return fstype in ("tmpfs", "ramfs")


def preflight(workdir):
    log("step 0: pre-flight")
    free = os.statvfs(workdir).f_bavail * os.statvfs(workdir).f_frsize
    if free < 2 * 1024**3:
        raise StepError(0, f"{workdir} has {free / 1024**3:.1f} GB free and this needs 2 GB")
    try:
        import torch  # noqa: F401
        import transformers
        from transformers import AutoConfig
    except ImportError as e:
        raise StepError(0, f"cannot import torch or transformers ({e}); run with senbonzakura's venv") from e
    try:
        AutoConfig.for_model("glm4_moe_lite")
    except (ValueError, KeyError) as e:
        raise StepError(0, f"transformers {transformers.__version__} does not know glm4_moe_lite ({e}); "
                      "it needs 5.x") from e
    from senbonzakura.vendored import VendorError, find_binary, find_script
    try:
        script = find_script("convert_hf_to_gguf.py")
        imatrix, source = find_binary("llama-imatrix", search_path=False)
    except VendorError as e:
        raise StepError(0, f"the pinned toolchain is not in this checkout: {e}") from e
    log(f"  transformers {transformers.__version__}, converter {script}, imatrix {imatrix} ({source})")
    return imatrix


def build_checkpoint(run):
    log("step 1: build a tiny glm4_moe_lite with one prediction layer declared")
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    ckpt = run / "ckpt"
    cfg = AutoConfig.for_model("glm4_moe_lite", **TINY)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
    model.save_pretrained(ckpt)

    saved = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    if saved.get("num_nextn_predict_layers") != 1:
        raise StepError(1, "SCENARIO NOT REPRODUCED: the saved config no longer declares the prediction "
                      f"layer (num_nextn_predict_layers={saved.get('num_nextn_predict_layers')!r}), "
                      "so this would not test what an abliterated checkpoint looks like")
    names = []
    for shard in sorted(ckpt.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            names += list(f.keys())
    if not names:
        raise StepError(1, "save_pretrained wrote no safetensors")
    beyond = [n for n in names if re.search(rf"\blayers\.{TINY['num_hidden_layers']}\.", n)]
    mtp = [n for n in names if _MTP.search(n)]
    if beyond or mtp:
        raise StepError(1, "SCENARIO NOT REPRODUCED: the checkpoint holds prediction-layer tensors "
                      f"({(beyond + mtp)[:3]}), so it is not shaped like an abliterated one")
    log(f"  saved {len(names)} tensors, config declares 1 prediction layer, none written")

    try:
        for name in TOKENIZER_FILES:
            # token=False: the repository is public, and a stale token in the environment would
            # turn a public download into a 401 while sending a credential nobody needed to send.
            hf_hub_download(REPO_ID, name, revision=REVISION, local_dir=str(ckpt), token=False)
    except Exception as e:  # any Hub failure ends the check with its reason, never a pass
        raise StepError(1, f"could not fetch the tokenizer from {REPO_ID}@{REVISION[:8]}: "
                            f"{type(e).__name__}: {e}") from e
    log(f"  added {', '.join(TOKENIZER_FILES)} from {REPO_ID}@{REVISION[:8]}")
    return ckpt, len(names)


def convert(ckpt, run):
    log("step 2: senbonzakura convert, as shipped (no --no-mtp)")
    from senbonzakura import gguf_io

    out = run / "tiny-glm.gguf"
    r = subprocess.run([sys.executable, "-m", "senbonzakura", "convert", str(ckpt), str(out),
                        "--outtype", "f16"],
                       capture_output=True, text=True, timeout=CONVERT_TIMEOUT_S, check=False)
    (run / "convert.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not out.is_file():
        raise StepError(2, f"convert exited {r.returncode}, output {'present' if out.is_file() else 'absent'}"
                      f"\n{_tail(r.stdout + r.stderr)}")

    head = gguf_io.read_header(out)
    arch = head.get("architecture")
    declared = head.get("metadata", {}).get(f"{arch}.block_count")
    tensors = gguf_io.read_tensor_info(out)
    covered = {int(m.group(1)) for t in tensors if (m := re.match(r"blk\.(\d+)\.", t["name"]))}
    log(f"  architecture {arch}, header declares {declared} blocks, tensors cover "
        f"{len(covered)} ({sorted(covered)}), {len(tensors)} tensors")
    mismatch = None
    if not isinstance(declared, int) or covered != set(range(declared)):
        mismatch = (f"the header declares {declared} blocks and the tensors cover {sorted(covered)}: "
                    "the prediction layer was counted without being written")
        log(f"  MISMATCH: {mismatch}")
    return out, declared, len(tensors), mismatch


def load(gguf, imatrix_exe, run):
    log("step 3: load the GGUF with the pinned llama-imatrix (a full model load)")
    text = run / "calibration.txt"
    sentence = ("The quick brown fox jumps over the lazy dog while the river runs past the old "
                "mill and the bells ring out across the valley at noon. ")
    text.write_text("\n".join(sentence * 2 for _ in range(400)), encoding="utf-8")
    out = run / "tiny-glm.imatrix"
    env = dict(os.environ, LD_LIBRARY_PATH=str(Path(imatrix_exe).parent))
    r = subprocess.run([str(imatrix_exe), "-m", str(gguf), "-f", str(text), "-o", str(out),
                        "--chunks", "1", "-ngl", "0"],
                       capture_output=True, text=True, timeout=IMATRIX_TIMEOUT_S, check=False,
                       env=env)
    (run / "imatrix.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not out.is_file():
        raise StepError(3, f"llama-imatrix exited {r.returncode}, output "
                      f"{'present' if out.is_file() else 'absent'}\n{_tail(r.stdout + r.stderr)}")
    log("  loaded and ran one chunk")


def _confirm(mismatch, load_error):
    """The verdict once the load has run, which is the second witness to a header mismatch."""
    if mismatch and load_error:
        # The predicted cause, confirmed: report it as the finding, the load failure as evidence.
        return StepError(2, f"{mismatch}. Confirmed at load: {load_error.reason}")
    if mismatch:
        return StepError(2, f"{mismatch}, yet llama-imatrix loaded it; read imatrix.log before "
                             "trusting either half")
    return load_error


def run_steps(workdir, run):
    """All four steps. Returns (tensor count, declared blocks) or raises StepError."""
    imatrix_exe = preflight(workdir)
    ckpt, _n = build_checkpoint(run)
    gguf, declared, n_tensors, mismatch = convert(ckpt, run)
    load_error = None
    try:
        load(gguf, imatrix_exe, run)
    except StepError as e:
        load_error = e
    verdict = _confirm(mismatch, load_error)
    if verdict is not None:
        raise verdict
    return n_tensors, declared


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workdir", required=True,
                    help="scratch directory on real disk; a fresh run directory is made inside it")
    a = ap.parse_args(argv)

    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # Before anything is written, so a refused run leaves nothing in RAM behind it.
    if _on_tmpfs(workdir):
        log(f"GLM_LITE_CONVERT: FAIL step=0 {workdir} is on a RAM-backed filesystem; "
            "pass a --workdir on real disk")
        return 1
    run = workdir / f"run-{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run.mkdir()
    # Children inherit this, so nothing the converter or the Hub client writes lands in /tmp.
    (run / "tmp").mkdir()
    os.environ["TMPDIR"] = str(run / "tmp")
    log(f"run directory: {run}")

    try:
        n_tensors, declared = run_steps(workdir, run)
    except StepError as e:
        log(f"GLM_LITE_CONVERT: FAIL step={e.step} {e.reason}")
        return 1
    except subprocess.TimeoutExpired as e:
        log(f"GLM_LITE_CONVERT: FAIL step=? timed out after {e.timeout}s: {e.cmd[:3]}")
        return 1
    except Exception as e:  # an unexpected fault is a failure with its reason, never a pass
        log(f"GLM_LITE_CONVERT: FAIL step=? unexpected {type(e).__name__}: {e}")
        return 1
    log(f"GLM_LITE_CONVERT: PASS tensors={n_tensors} blocks={declared} run={run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
