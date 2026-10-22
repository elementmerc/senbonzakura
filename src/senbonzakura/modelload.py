# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King
"""Which kind of checkpoint is this, can this install load it, and what should the result say.

WHY THIS EXISTS

People bring the model in whatever form they downloaded it: plain safetensors, a 4-bit copy from a
quantiser's own repository, a GPTQ or AWQ folder, or a single GGUF file. Before this module only
plain safetensors and on-the-fly 4-bit loaded on every command, and `attribute` had neither of the
flags, so a measurement tool refused the models people actually run. The decisions about format
live here, in one place, and `cli.load_model_and_tokenizer` (the loader every measurement command
already shares) asks this module instead of each command growing its own copy.

THE PRINCIPLES

1. A format is detected from the checkpoint, not guessed from its name. A pre-quantised folder says
   so in the `quantization_config` of its config.json; a GGUF says so in its magic bytes.
2. A missing backend is one sentence naming the package and the install command, raised before the
   weights are fetched. Nothing here lets a traceback through.
3. The result records what was measured: format, method, bits, whether the weights were
   dequantised and from which quantisation, and the backend's version. A figure taken on a Q4_K_M
   file that was expanded to bfloat16 is not the figure for the original model, and the record says
   so.
4. A format this tool cannot guarantee the measurement on is refused with a sentence, never loaded
   and hoped about.

The decoder-block hook that `attribute` relies on sits on the block's OUTPUT, so the quantised
linear layers inside the block do not change what it reads. What can change it is a quantiser that
replaces the block itself, which is why an unrecognised method is refused for attribution only.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from . import argresolve, checkpoint, hubmessage, loadreport
from .chattemplate import ensure_chat_template

#: Packages each format needs, as (import name, distribution name), and the extra that brings them.
BACKENDS = {
    "bnb": ((("bitsandbytes", "bitsandbytes"),), "quant"),
    "gptq": ((("optimum", "optimum"), ("gptqmodel", "gptqmodel")), "gptq"),
    "awq": ((("gptqmodel", "gptqmodel"),), "awq"),
    # gguf is a core dependency, so its fix is a reinstall rather than an extra.
    "gguf": ((("gguf", "gguf"),), None),
}

_ANY_GGUF = "*.gguf"


class LoadRefusedError(ValueError):
    """A checkpoint this run will not load. The message is the whole sentence the reader sees."""


@dataclass(frozen=True)
class Plan:
    """What a load will do, decided before any weight is fetched."""

    source: str                  # what to hand transformers: a directory or a hub id
    kind: str                    # plain | bnb | gptq | awq | gguf | other
    method: str | None = None    # the quantiser's own name
    bits: int | None = None
    quant_type: str | None = None
    on_the_fly: bool = False
    gguf_file: str | None = None
    dequantised_from: str | None = None
    kernel: str | None = None    # the GPTQ or AWQ kernel actually used; None when left to the library

    @property
    def quantised(self):
        return self.kind != "plain"

    @property
    def dequantised(self):
        return self.kind == "gguf"


def _installed(import_name):
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_command(kind):
    extra = BACKENDS.get(kind, ((), None))[1]
    return f'pip install "senbonzakura[{extra}]"' if extra else "pip install --force-reinstall senbonzakura"


def require_backend(kind):
    """Refuse, in one sentence, when the package a format needs is not installed."""
    needed = BACKENDS.get(kind, ((), None))[0]
    missing = [dist for imp, dist in needed if not _installed(imp)]
    if missing:
        names = " and ".join(missing)
        raise LoadRefusedError(
            f"senbonzakura: this model is stored as {kind_label(kind)}, which needs {names} "
            f"and it isn't installed. Install it with: {install_command(kind)}")


def kind_label(kind):
    return {"bnb": "bitsandbytes 4-bit or 8-bit weights", "gptq": "GPTQ weights",
            "awq": "AWQ weights", "gguf": "a GGUF file"}.get(kind, kind)


def backend_of(kind):
    """The backend package's name and version, for the record; None for plain weights."""
    needed, _ = BACKENDS.get(kind, ((), None))
    if not needed:
        return None
    dist = (needed[-1] if kind == "gptq" else needed[0])[1]
    return {"package": dist, "version": _version(dist)}


#: The kernels the loader picks for GPTQ and AWQ on a card, fastest first. Both lists are kernels that
#: need no compiler: gptqmodel's own default (Marlin) and AWQ's `gemm` build a C++ extension on first use,
#: which fails on a machine without a build toolchain even when the checkpoint is fine. Measured on an
#: RTX 3060: all four of these loaded and ranked the right token first.
CUDA_KERNELS = {"gptq": ("gptq_triton", "gptq_torch"), "awq": ("gemm_triton", "torch_awq")}


def kernel_choices(p, device):
    """Kernels to try for this load, in order; empty when the library should choose (CPU, other formats)."""
    if p.kind in CUDA_KERNELS and not p.on_the_fly and str(device).startswith("cuda"):
        return CUDA_KERNELS[p.kind]
    return ()


def kernel_config(p, kernel):
    """The quantisation config that pins one kernel; the checkpoint's own settings fill in the rest."""
    from transformers import AwqConfig, GPTQConfig
    if p.kind == "gptq":
        return GPTQConfig(bits=p.bits or 4, backend=kernel)
    return AwqConfig(bits=p.bits or 4, backend=kernel)


def kernel_fallback_notice(failed, using, error):
    first = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    return (f"the {failed} kernel couldn't load this model ({first[:120]}), so it was loaded with the "
            f"{using} kernel instead; the result records which one ran")


# ----------------------------------------------------------------------------- reading the checkpoint

def _config_quantisation(source, revision):
    """The `quantization_config` block of a checkpoint's config.json, or None.

    Read as plain JSON through transformers' own resolver, so a model that needs custom code can
    still be inspected without executing any of it. A config that cannot be read is not fatal here:
    the load that follows fails with a better message than this could give.
    """
    from transformers import PretrainedConfig
    try:
        raw, _ = PretrainedConfig.get_config_dict(
            source, **({"revision": revision} if revision else {}))
    except (OSError, ValueError):
        return None
    qc = raw.get("quantization_config")
    return qc if isinstance(qc, dict) else None


def _classify(qc):
    method = str(qc.get("quant_method") or "").lower()
    if method == "bitsandbytes" or "load_in_4bit" in qc or "load_in_8bit" in qc:
        four = bool(qc.get("load_in_4bit") or qc.get("_load_in_4bit"))
        return ("bnb", "bitsandbytes", 4 if four else 8,
                str(qc.get("bnb_4bit_quant_type", "fp4")) if four else "int8")
    if method == "gptq":
        return ("gptq", "gptq", int(qc.get("bits", 4)), None)
    if method == "awq":
        return ("awq", "awq", int(qc.get("bits", 4)), None)
    return ("other", method or "unknown", None, None)


def resolve_gguf(model_id, gguf_file):
    """Split a GGUF reference into (directory or hub id, file name), or None if it isn't one.

    Three shapes: a path to a .gguf file; a local directory holding exactly one; a hub id with
    `--gguf-file` naming the file. Two files in one directory is refused, because choosing for the
    reader would be choosing which quantisation gets measured.
    """
    path = Path(model_id)
    try:
        is_file, is_dir = path.is_file(), path.is_dir()
    except (OSError, ValueError):
        is_file = is_dir = False
    if is_file:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
        if gguf_file and gguf_file != path.name:
            raise LoadRefusedError(f"senbonzakura: --gguf-file {gguf_file!r} doesn't match the file "
                              f"{path.name!r} given as the model. Pass one or the other.")
        return str(path.parent), path.name
    if is_dir:
        found = sorted(p.name for p in path.glob(_ANY_GGUF))
        if gguf_file:
            if gguf_file not in found:
                raise LoadRefusedError(f"senbonzakura: there is no {gguf_file!r} in {model_id}. "
                                  f"It holds: {', '.join(found) or 'no GGUF files'}.")
            return str(path), gguf_file
        if not found or (path / "config.json").exists():
            return None
        if len(found) > 1:
            raise LoadRefusedError(
                f"senbonzakura: {model_id} holds {len(found)} GGUF files ({', '.join(found)}). "
                f"Pick one with --gguf-file, because each is a different quantisation and the "
                f"figure belongs to one of them.")
        return str(path), found[0]
    if gguf_file:
        return model_id, gguf_file
    return None


def _gguf_quant(source, name, revision=None):
    """The quantisation a GGUF file declares, read from its header; None when it can't be read.

    A hub id is fetched first (the load that follows would fetch the same file, so nothing extra is
    downloaded), because the header is the only honest source: the file name is a claim.
    """
    from . import gguf_io
    path = Path(source) / name
    if not Path(source).is_dir():
        try:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(source, name, **({"revision": revision} if revision else {})))
        except Exception:
            return None
    try:
        return gguf_io.read_header(path)["file_type"]
    except (gguf_io.GGUFError, OSError, ValueError, KeyError):
        return None


def plan(model_id, *, load_in_4bit=False, load_in_8bit=False, gguf_file=None, revision=None,
         forward_only=True):
    """Decide what kind of load this is, and refuse every combination that can't work.

    Runs before any weight is fetched. `forward_only=False` is the weight editor, which rewrites
    tensors in place and so refuses every quantised or dequantised source.
    """
    if load_in_4bit and load_in_8bit:
        raise LoadRefusedError("senbonzakura: --load-in-4bit and --load-in-8bit were both given. "
                          "Choose one.")
    gguf = resolve_gguf(model_id, gguf_file)
    if gguf is not None:
        if load_in_4bit or load_in_8bit:
            raise LoadRefusedError("senbonzakura: a GGUF is already quantised, so --load-in-4bit and "
                              "--load-in-8bit can't apply to it. Drop the flag.")
        if not forward_only:
            raise _editor_refusal("a GGUF file")
        require_backend("gguf")
        source, name = gguf
        return Plan(source=source, kind="gguf", method="gguf", gguf_file=name,
                    dequantised_from=_gguf_quant(source, name, revision))
    qc = _config_quantisation(model_id, revision)
    if qc is not None:
        kind, method, bits, qtype = _classify(qc)
        if load_in_4bit or load_in_8bit:
            raise LoadRefusedError(
                f"senbonzakura: '{model_id}' is already stored quantised ({method}"
                f"{f', {bits}-bit' if bits else ''}), so --load-in-4bit and --load-in-8bit "
                f"can't apply. Drop the flag, or point at the unquantised original.")
        if not forward_only:
            raise _editor_refusal(f"a {method} checkpoint")
        if kind != "other":
            require_backend(kind)
        return Plan(source=model_id, kind=kind, method=method, bits=bits, quant_type=qtype)
    if load_in_4bit or load_in_8bit:
        if not forward_only:
            raise _editor_refusal("on-the-fly quantisation")
        require_backend("bnb")
        return Plan(source=model_id, kind="bnb", method="bitsandbytes",
                    bits=4 if load_in_4bit else 8, quant_type="nf4" if load_in_4bit else "int8",
                    on_the_fly=True)
    return Plan(source=model_id, kind="plain")


def _editor_refusal(what):
    return LoadRefusedError(
        f"senbonzakura abliterates by rewriting weights, which needs full precision, so {what} "
        f"is not supported here. Use it with a measurement command (score, drift, attribute) "
        f"instead, or start from the unquantised model.")


# ----------------------------------------------------------------------------- what to hand transformers

def bnb_config(p, torch_module):
    """The on-the-fly BitsAndBytesConfig for this plan, or None when it isn't one."""
    if not (p.on_the_fly and p.kind == "bnb"):
        return None
    from transformers import BitsAndBytesConfig
    if p.bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch_module.bfloat16,
                              bnb_4bit_use_double_quant=True)


def dtype_for(p, torch_module):
    """GPTQ and AWQ kernels run in float16; everything else keeps the loader's bfloat16."""
    return torch_module.float16 if p.kind in ("gptq", "awq") else torch_module.bfloat16


# ----------------------------------------------------------------------------- what the result records

def precision_label(p, dtype_name):
    if p.kind == "plain":
        return dtype_name
    if p.kind == "bnb":
        return p.quant_type or f"int{p.bits}"
    if p.kind in ("gptq", "awq"):
        return f"{p.kind}-{p.bits}bit"
    if p.kind == "gguf":
        return f"{dtype_name} dequantised from {p.dequantised_from or 'unknown quantisation'}"
    return f"{p.method}"


def _backend_record(p):
    backend = backend_of(p.kind)
    return None if backend is None else {**backend, "kernel": p.kernel}


def provenance(p, dtype_name):
    """The fields a result carries about how the weights were loaded. Always the same keys."""
    return {
        "format": "safetensors" if p.kind == "plain" else p.kind,
        "quant_method": p.method,
        "quant_bits": p.bits,
        "quant_type": p.quant_type,
        "on_the_fly": p.on_the_fly,
        "dequantised": p.dequantised,
        "dequantised_from": p.dequantised_from,
        "gguf_file": p.gguf_file,
        "backend": _backend_record(p),
        "precision": precision_label(p, dtype_name),
    }


#: What an older artefact, written before these fields, is read as.
LEGACY = {"format": "safetensors", "quant_method": None, "quant_bits": None, "quant_type": None,
          "on_the_fly": False, "dequantised": False, "dequantised_from": None, "gguf_file": None,
          "backend": None}


def with_defaults(identity):
    """An identity block from any era, with the loading fields present.

    The adapter for artefacts written before the loading fields existed: they were all plain
    full-precision loads, so that is what their absence means.
    """
    out = dict(LEGACY)
    out.update(identity or {})
    return out


def attach(model, p, dtype_name):
    """Carry the provenance on the model so a writer far from the loader can record it."""
    info = provenance(p, dtype_name)
    try:
        model.senbon_load = info
    except (AttributeError, TypeError):
        pass
    return info


def check_hookable(info):
    """Refuse, for `attribute`, a loading form the block-output hook can't be guaranteed on."""
    if info.get("format") == "other":
        raise LoadRefusedError(
            f"senbonzakura: the quantisation method {info.get('quant_method')!r} hasn't been "
            f"checked against the attribution hook, which reads each decoder block's output. "
            f"Use safetensors, bitsandbytes, GPTQ, AWQ or a GGUF, or the unquantised original.")


def probe_block_output(layers, run_forward):
    """Run one forward pass and confirm a decoder block's output is a tensor the hook can read.

    The hook sits on the block, so the quantised layers inside it are invisible to it; what this
    proves is that the block itself still returns hidden states of the expected kind on this load.
    """
    import torch
    seen = []
    handle = layers[0].register_forward_hook(lambda _m, _a, out: seen.append(out))
    try:
        run_forward()
    finally:
        handle.remove()
    if not seen:
        raise LoadRefusedError(
            "senbonzakura: the first decoder block never ran during a forward pass, so the "
            "attribution hook would never fire on this model. This loading form isn't supported "
            "for attribute.")
    out = seen[0]
    if isinstance(out, tuple):
        out = out[0] if out else None
    if not isinstance(out, torch.Tensor):
        raise LoadRefusedError(
            "senbonzakura: after loading, the first decoder block didn't return a tensor, so "
            "the attribution hook can't read what it writes. This loading form isn't "
            "supported for attribute.")
    return True


# ----------------------------------------------------------------------------- readable failures

def gguf_tokenizer_refusal(source, name, error):
    """The sentence for a GGUF whose tokenizer couldn't be built.

    transformers rebuilds a tokenizer from the GGUF's metadata when no tokenizer files sit beside
    it, and that conversion needs sentencepiece or tiktoken. Without them the error it raised was
    "Couldn't instantiate the backend tokenizer", which names neither the cause nor the fix.
    """
    return (f"senbonzakura: the GGUF '{name}' carries its tokenizer inside it, and building it "
            f"failed ({type(error).__name__}).\n"
            f"  Either install the converters with: pip install sentencepiece tiktoken\n"
            f"  or put the original model's tokenizer.json (or tokenizer.model) in {source} "
            f"beside the GGUF.")


def import_refusal(p, error):
    """A backend that was present at the check but failed on use (a version mismatch)."""
    first = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    hint = f" Install a matching version with: {install_command(p.kind)}" if p.kind in BACKENDS else ""
    return (f"senbonzakura: loading {kind_label(p.kind)} failed because a package it needs "
            f"couldn't be used: {first}.{hint}")


def remote_code_refusal(model_id, error):
    """The sentence for a model that ships its own code when `--trust-remote-code` wasn't given."""
    if "trust_remote_code" not in str(error):
        return None
    return (f"senbonzakura: '{model_id}' ships custom modelling code, which this tool won't run "
            f"unless you say so. If you trust the repository, add --trust-remote-code (off by "
            f"default).")


# ----------------------------------------------------------------------------- the loader itself
#
# Moved here from `cli` so that `attribute` can use the loader every other command uses without
# importing the module that holds the weight edit (the attribution firewall forbids that import).
# `cli` re-exports every name, so its own callers are unchanged.

#: The prompt length the activation headroom is budgeted for, in tokens. The corpora this tool
#: ships are short (the longest bundled refusal prompt is well under 200 tokens), so this is
#: generous rather than tight, and it is a documented assumption rather than a measurement because
#: the headroom is a floor and overshooting it costs a little card space while undershooting it
#: costs the run. A prompt longer than this does not fail: it makes the reserve smaller than the
#: pass needs, and the capture governor then shrinks the chunk exactly as it does for any other
#: shortfall.
HEADROOM_SEQ_TOKENS = 512

#: The floor under any computed headroom. CUDA's own context, cuBLAS and cuDNN workspaces and
#: allocator fragmentation all live outside the arithmetic below and together run to hundreds of
#: megabytes before a single activation exists.
HEADROOM_FLOOR_BYTES = 1 << 30

#: The most of any one device's budget the headroom may claim. Reserving most of a small card to
#: guarantee a forward pass leaves too little for weights to be worth placing, and a loud failure
#: at load beats a run that technically starts and then thrashes. Past this the reserve is capped
#: and the cap is reported.
HEADROOM_MAX_FRACTION = 0.25


def _inner_text_config(config):
    """The sub-config that carries the decoder's shape, which on a multimodal config is nested."""
    inner = getattr(config, "text_config", None)
    return inner if inner is not None and hasattr(inner, "num_hidden_layers") else config


def _activation_headroom_bytes(config, *, seq=HEADROOM_SEQ_TOKENS):
    """VRAM one single-prompt forward pass needs beyond the weights, or None if unknowable.

    THE DEFECT THIS EXISTS FOR, and it is the same one as `_corrected_max_memory` one device over.
    accelerate's `get_max_memory` reports each card's FREE memory and `infer_auto_device_map` then
    fills it with weights, because placement is the only question it is being asked. A forward pass
    needs room on top of the weights, and `output_hidden_states=True` needs a lot of it: every
    layer's activations for the whole batch exist at once. With the card packed to the last hundred
    megabytes there is nowhere for them to go.

    WHY A ONE-PROMPT BUDGET AND NOT THE WHOLE CHUNK. `collect_resid` runs behind a
    `ResourceGovernor` that halves the chunk on out-of-memory and waits when the card is too full,
    so a chunk that does not fit is already handled. What the governor CANNOT do is go below one
    prompt, and that floor is what has to be guaranteed in advance. Budgeting the full
    `CAPTURE_BATCH` instead would reserve about 108 GB on a 72B model, which is not a reserve, it
    is a refusal to use the card.

    Found on 2026-10-09 on a rented A40: the host budget correction worked, 44.19 GiB of weights
    landed on a 44.42 GiB card, and the first capture died asking for 462 MiB with 227 MiB free.

    THE ARITHMETIC, per prompt at `seq` tokens, every term named so a reader can check it:

      hidden states   (L + 1) x seq x H x 2      every layer's output, kept, in bf16
      logits          seq x V x 4                computed in float32 on most families
      transients      seq x I x 2 x 3            MLP intermediates, a few live at once

    Doubled, because fragmentation and the attention implementation's own scratch are real and are
    not in the list above, and floored at `HEADROOM_FLOOR_BYTES`, because on a small model the
    three terms together come to a couple of hundred megabytes while CUDA's context and the BLAS
    workspaces, which are in none of them, do not shrink with the model. Returns None when the
    config does not name the shape, which is a statement that the headroom is unmeasured rather
    than a guess at it.
    """
    inner = _inner_text_config(config)
    layers = getattr(inner, "num_hidden_layers", None)
    hidden = getattr(inner, "hidden_size", None)
    vocab = getattr(inner, "vocab_size", None)
    if not isinstance(layers, int) or not isinstance(hidden, int) or not isinstance(vocab, int):
        return None
    # `intermediate_size` is absent on some hybrid and MoE configs. Four times the hidden size is
    # the conventional ratio and is used only for a transient term, so a family that does not name
    # it gets a reasonable figure rather than no reserve at all.
    inter = getattr(inner, "intermediate_size", None)
    if not isinstance(inter, int):
        inter = 4 * hidden
    states = (layers + 1) * seq * hidden * 2
    logits = seq * vocab * 4
    transients = seq * inter * 2 * 3
    return max(2 * (states + logits + transients), HEADROOM_FLOOR_BYTES)


def _largest_module_bytes(config, dtype_bytes):
    """Bytes of the largest single weight the config names, or None when its shape is not in the config.

    The embedding and the output head are `vocab x hidden`, and the widest MLP projection is
    `intermediate x hidden`. The larger of the two is the most accelerate can copy onto the card at
    once. Read from the same fields as the activation arithmetic, and None for the same reasons.
    """
    inner = _inner_text_config(config)
    hidden = getattr(inner, "hidden_size", None)
    vocab = getattr(inner, "vocab_size", None)
    if not isinstance(hidden, int) or not isinstance(vocab, int):
        return None
    inter = getattr(inner, "intermediate_size", None)
    if not isinstance(inter, int):
        inter = 4 * hidden
    return max(vocab, inter) * hidden * dtype_bytes


def _reserve_activation_headroom(budgets, headroom, log):
    """Subtract `headroom` from every compute device in `budgets`. Returns True if any changed.

    Mutates `budgets`. Only the keys accelerate uses for real devices are touched: `"cpu"` has
    already been corrected against the cgroup by the caller and `"disk"` is not a device a forward
    pass allocates on. A non-integer budget (accelerate accepts `"10GiB"` strings) is left alone
    and reported, because parsing a unit string to subtract from it would be guessing at a format
    we do not own.
    """
    changed = False
    for key, value in list(budgets.items()):
        if key in ("cpu", "disk"):
            continue
        if not isinstance(value, int):
            log(f"  device {key}: its placement budget is {value!r} rather than a byte count, so "
                f"no activation headroom was reserved on it. A capture that runs out of room will "
                f"be shrunk by the governor rather than refused here.")
            continue
        take = min(headroom, int(value * HEADROOM_MAX_FRACTION))
        if take < headroom:
            log(f"  device {key}: a forward pass wants {headroom / 1e9:.1f} GB of headroom but that "
                f"is more than {HEADROOM_MAX_FRACTION:.0%} of its {value / 1e9:.1f} GB, so "
                f"{take / 1e9:.1f} GB is reserved and the rest is left to the capture governor.")
        if take <= 0:
            continue
        budgets[key] = value - take
        changed = True
        log(f"  device {key}: {take / 1e9:.1f} GB of its {value / 1e9:.1f} GB held back for "
            f"activations, so weights are placed against {budgets[key] / 1e9:.1f} GB.")
    return changed


def _corrected_max_memory(log, config=None, dtype_bytes=None):
    """The placement budget accelerate computed, corrected, or None to leave it alone.

    Two corrections, both of the same shape: accelerate is answering "where do the weights go" and
    neither the host's real limit nor the room a forward pass needs is part of that question. The
    host entry is corrected against the cgroup, and each compute device keeps back enough for one
    prompt's activations. `config` is the model's own config and may be omitted, in which case only
    the host correction is made.

    `dtype_bytes` is the width of one weight. When it is given, the reserve also covers the largest
    single weight. accelerate copies a weight that lives in host RAM or on disk onto the card for each
    forward pass that uses it, and that copy needs room beside the activations. A card packed to the
    last activation byte can hold every activation and still fail on the copy of the output head. The
    FAILURE was a 2.03 GiB output head on a 24 GB card, with 1.56 GiB free after the placement. Without
    `dtype_bytes` the reserve is the activations alone, which is every earlier caller's figure.

    THE DEFECT. accelerate decides how much of a model goes in host RAM from
    `psutil.virtual_memory().available`, which reads `/proc/meminfo`. Inside a container that is
    the HOST's memory and not the limit the process is held to, so on a cgroup-capped box the
    placement believes in memory that is not there. Without an offload directory the ending is an
    OOM kill partway through loading, with no Python traceback; with one, it is worse in a quieter
    way, because the weights go to RAM that seems to exist and the disk path the run was relying on
    never executes at all.

    Found on 2026-10-09 by trying to demonstrate disk offload on rented hardware. The box reported
    540.6 GB of RAM against a provider spec sheet saying 50 GB, so a 145 GB checkpoint read as
    something that comfortably fits and nothing would have spilled.

    WHY THIS STARTS FROM accelerate's OWN DICT rather than building one. `max_memory` is used
    verbatim when it is given, so a dict naming only `cpu` would place the entire model on the host
    and never touch the card. Taking `get_max_memory()` and overriding one key keeps every
    per-device figure accelerate would have chosen, which is the part we have no business
    second-guessing.

    IT ONLY EVER CORRECTS DOWNWARD. When no cgroup caps us, `host_memory_budget` returns the same
    `/proc/meminfo` reading accelerate started from, and this returns None so the call site passes
    nothing at all. Raising a budget on the strength of our own arithmetic would turn a
    conservative default into an OOM, so the one direction this moves is the safe one.
    """
    from . import resources
    budget, source = resources.host_memory_budget()
    headroom = _activation_headroom_bytes(config) if config is not None else None
    largest = _largest_module_bytes(config, dtype_bytes) if headroom is not None and dtype_bytes else None
    if largest is not None:
        headroom += largest
        log(f"  the largest weight is {largest / 1e9:.2f} GB, and it is reserved beside the activations, "
            f"because an offloaded weight is copied onto the card for each forward pass that uses it.")
    if budget is None and headroom is None:
        log("  host memory: unmeasured on this platform, so accelerate's own placement budget "
            "stands. A model that does not fit will fail at load rather than being refused here.")
        return None
    if budget is None:
        log("  host memory: unmeasured on this platform, so accelerate's own host budget stands.")
    elif source != "cgroup":
        budget = None                                   # nothing caps us; accelerate was right
    if headroom is None and config is not None:
        log("  activations: this config does not name the layer count, width and vocabulary "
            "together, so the headroom a forward pass needs is unmeasured and none is reserved. "
            "A capture that runs out of room is shrunk by the governor instead.")
    if budget is None and headroom is None:
        return None
    try:
        from accelerate.utils import get_max_memory
        budgets = dict(get_max_memory())
    except Exception as e:
        # Graceful degradation with a visible warning, per the robustness mandate: a placement we
        # could not correct is accelerate's default, which is what every previous run used.
        log(f"  placement: accelerate's own budget could not be read to correct it "
            f"({type(e).__name__}: {e}), so its default stands. A load that overshoots a cgroup cap "
            f"will be OOM-killed rather than spilling to disk, and a packed card will run out of "
            f"room on the first forward pass.")
        return None
    changed = False
    if budget is not None:
        theirs = budgets.get("cpu")
        if not (isinstance(theirs, int) and theirs <= budget):
            limit, current = resources.cgroup_memory_limit(), resources.cgroup_memory_current()
            log(f"  host memory: a cgroup limits this process to {limit / 1e9:.1f} GB with "
                f"{(current or 0) / 1e9:.1f} GB already counted against it, so the placement budget "
                f"is {budget / 1e9:.1f} GB and not the "
                f"{('%.1f GB' % (theirs / 1e9)) if isinstance(theirs, int) else theirs} that "
                f"/proc/meminfo reports. Inside a container that file describes the host.")
            budgets["cpu"] = budget
            changed = True
    if headroom is not None:
        changed = _reserve_activation_headroom(budgets, headroom, log) or changed
    return budgets if changed else None


#: Phrases in `transformers`' load failure that give advice for the PYTHON API, to somebody who is
#: holding a command line. Its message is otherwise the best part of the refusal: it names the id
#: and says whether it was a local folder or a listed model, which is exactly what the reader needs.
#: But it closes by telling them to pass `token=<your_token>`, a keyword argument to a function they
#: are not calling, and to use `huggingface-cli login`, which `hf auth login` replaced.
#:
#: So the useful half is kept verbatim and the advice is dropped, because the line under it already
#: says the same thing in the two spellings that work here. Leaving both in gave a reader two
#: instructions where one was impossible to follow, and the impossible one came first.
#:
#: THE LIST ITSELF NOW LIVES IN `hubmessage`, because this was the only one of three surfaces that
#: consulted it: `fetch` and the snapshot pre-flight printed the same advice unfiltered. The name is
#: kept here because it is what this file's readers and its tests already look for.
_THEIR_ADVICE_IS_FOR_THE_PYTHON_API = hubmessage.ADVICE_FOR_THE_PYTHON_API


#: How a git-lfs pointer file begins. The spec line is the first thing in the file and it is
#: fixed text, so this is a reliable test rather than a guess.
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

#: The files worth examining for the pointer. Not a glob over the whole directory: a model
#: directory can hold hundreds of shards, and the point is a cheap answer, not a complete audit.
_LFS_SUSPECTS = ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")


def _unfetched_lfs_files(model_dir):
    """The files in a local model directory that are git-lfs pointers rather than their contents.

    A `git clone` of a model repository without `git lfs pull`, or with lfs absent, leaves every
    large file as a 130-byte text stub naming where the real one lives. The directory then looks
    complete: the file names are right, `config.json` may even be real, and the failure arrives
    from inside `transformers` as whatever that stub fails to parse as. The streaming journey of
    2026-10-13 met it as `json.decoder.JSONDecodeError: Expecting value: line 1 column 1` out of
    `tokenization_utils_base.py`, which is eleven frames describing our dependency's parser and
    not one word about the actual problem.

    Cheap on purpose: a bounded list of names, 64 bytes read from each.
    """
    found = []
    for name in _LFS_SUSPECTS:
        candidate = Path(model_dir) / name
        try:
            if not candidate.is_file():
                continue
            with open(candidate, "rb") as fh:
                if fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                    found.append(name)
        except OSError:
            # This function runs while another failure is being explained. It must not be able to
            # replace that explanation with one of its own.
            continue
    return found


def _looks_like_a_local_path(model_id):
    """Whether this string names the filesystem in a way a hub model id cannot.

    A hub id is `org/model`: one slash, no leading dot or tilde, no trailing separator, no backslash
    and no colon. So an absolute path, an explicitly relative one, a `~` path, a trailing separator,
    a Windows path and three or more segments are all shapes an id cannot take, and anything else is
    left ambiguous on purpose.
    """
    text = str(model_id or "").strip()
    if not text:
        return False
    # A backslash is the Windows separator and no hub id has one, so a backslash anywhere names the
    # filesystem. A drive letter (`C:` or `C:/models`) does too. Without these, a Windows reader who
    # mistyped a folder was told to run `hf auth login`, and the path check only ever saw `/`.
    if "\\" in text or (text[:1].isalpha() and text[1:3] in (":/", ":\\")):
        return True
    if text.startswith(("/", "./", "../", "~")) or text.endswith("/"):
        return True
    return text.count("/") >= 2


def _could_not_load(model_id, error):
    """One refusal for a model that will not load, shared by both loaders.

    Two identical copies of this message had already drifted apart once elsewhere in this file, and
    a refusal is a user-facing string like any other.
    """
    # JOINED ONLY IF THERE IS SOMETHING TO JOIN. An OSError with an empty message, or one whose
    # every line was advice, left a blank line where the detail should be, which reads as output
    # that got cut off.
    detail = "".join(f"  {ln}\n" for ln in hubmessage.useful_lines(error))
    # THE TOKEN ADVICE IS FOR THE HUB, and it used to be printed for a local path too. A reader who
    # passed a file that is already on their disk was told to authenticate to a service that is not
    # involved, which sends them to fix something that is not broken. Found by a user journey on
    # 2026-10-13, where the path given was a .jsonl.
    #
    # The test is "does this name something on disk", not "does it look like an id": an id with a
    # slash and a path with a slash are the same shape, and a relative path that exists is the case
    # this has to catch.
    try:
        local = Path(model_id).exists()
    except (OSError, ValueError):
        # An over-long path or one with a null byte is not a thing on disk, and raising here would
        # replace a refusal the reader can act on with a traceback.
        local = False
    # AND THE THIRD CASE, WHICH THE PAIR ABOVE SENT TO THE WRONG FIX, 2026-10-13. A string that
    # plainly names the filesystem and is not there got the hub's token advice, so a reader who
    # mistyped a directory was told to go and authenticate. The two cases above were "it is on
    # disk" and "everything else", and a mistyped path falls in the second.
    #
    # Deliberately conservative about what counts as plainly a path. A hub id carries exactly one
    # slash, so `org/model` stays a hub id and only the shapes an id cannot have are claimed here.
    # The ambiguous middle, a bare `name` or a single-slash `a/b` that is not on disk, keeps the
    # hub advice, because that genuinely is the likelier reading of it.
    if local:
        # AND IF THE FILES ARE STUBS, SAY SO, because "about the files themselves" is true and
        # unhelpful next to the one cause that accounts for most of it.
        stubs = _unfetched_lfs_files(model_id) if Path(model_id).is_dir() else []
        if stubs:
            one = len(stubs) == 1
            advice = (f"  {', '.join(stubs)} "
                      f"{'is a git-lfs pointer' if one else 'are git-lfs pointers'} rather than "
                      f"{'the real file' if one else 'the real files'}, so this checkpoint was "
                      f"cloned without its large files. Fetch them with `git lfs pull` in that "
                      f"directory, or download the model with `senbonzakura fetch` instead of "
                      f"git.\n")
        else:
            advice = ("  That path exists, so this is about the files themselves rather than "
                      "about access.\n")
    elif _looks_like_a_local_path(model_id):
        advice = ("  Nothing exists at that path, and it is a path rather than a model id, so "
                  "the hub is not involved. Check the spelling, and that the directory holds a "
                  "config.json beside its weights.\n")
    else:
        advice = "  If it is gated or private, pass --hf-token or log in with `hf auth login`.\n"
    return (f"senbonzakura: could not load the model '{model_id}'.\n"
            f"{detail}"
            f"{advice}".rstrip("\n"))


def load_tokenizer(model_id, *, trust_remote_code=False, log=None, chat_template=None,
                   needs_chat_template=True, revision=None, gguf_file=None):
    """The tokenizer half of the loader, on its own.

    Split out so a caller that needs only the chat template does not have to put a model on the
    device to get it. `drift` did: it loaded the candidate, then loaded the base while the
    candidate was still resident, so the peak was two full copies. Two bf16 copies of a 1.7B model
    do not fit the 6 GB card this project is sized for.

    Left-padding is set here because `firsttoken.last_token_logits` reads position -1, which is
    the last REAL token only under left padding.
    """
    _log = log or (lambda m: None)
    # The tokenizer loads FIRST, so this is where a bad model reference actually surfaces, and
    # wrapping only the weight load left the traceback exactly as it was. Same treatment, same
    # reason: transformers' sentence is good, the frames around it are not.
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code,
                                            **({"revision": revision} if revision else {}),
                                            **({"gguf_file": gguf_file} if gguf_file else {}))
    # `ValueError` BESIDE `OSError`, 2026-10-13. A missing file raises `OSError` and that was the
    # only shape handled; a file that is PRESENT and is not what it claims raises out of whatever
    # parser `transformers` hands it to, and `json.JSONDecodeError` is a `ValueError`. So a
    # checkpoint cloned without its large files reached the user as eleven frames ending inside
    # `tokenization_utils_base.py`. Both are the same question to the reader, "this model will not
    # load, why", and `_could_not_load` now answers the git-lfs case by name.
    except (OSError, ValueError) as e:
        if gguf_file:
            raise SystemExit(gguf_tokenizer_refusal(model_id, gguf_file, e)) from e
        raise SystemExit(remote_code_refusal(model_id, e)
                         or _could_not_load(model_id, e)) from e
    # A GGUF's tokenizer is rebuilt from its metadata, and that conversion fails with these when
    # sentencepiece or tiktoken is missing or the vocabulary is of a kind it can't convert.
    except (ImportError, RuntimeError, TypeError) as e:
        if not gguf_file:
            raise
        raise SystemExit(gguf_tokenizer_refusal(model_id, gguf_file, e)) from e
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Validated here, at the single boundary every entry point passes through, rather than
    # in each of the three. The provenance rides on the tokenizer so callers can put it in
    # their result files without the loader having to change what it returns.
    # Scoped to the callers whose measurement depends on prompt format. The perplexity of a
    # fixed neutral passage does not: it never renders a chat prompt, so requiring a template
    # there refused to measure base models for a reason that did not apply to them.
    tok.senbon_chat_template = (ensure_chat_template(tok, chat_template, _log)
                                if needs_chat_template else None)
    return tok


#: The one declared dependency that is imported from inside a function rather than at module top.
#:
#: `senbonzakura_check` owns the measurement stamp, so every command that writes a figure reaches
#: for it at the END of its run. The imports are deferred on purpose, to keep `--help` fast, and
#: that is exactly why nothing caught its absence early: `entry.dispatch` only sees imports that
#: happen while a module is LOADING, and the handler around the run catches the late failure and
#: says the right thing far too late.
CHECKER = "senbonzakura_check"


def _refuse_without_the_checker():
    """Refuse before the weights, if the package that will write the result is missing.

    MEASURED 2026-10-11 on sixteen cores. `senbonzakura compass` spent 3 minutes 45 seconds and 28
    CPU-minutes loading Qwen3-1.7B and scoring sixty prompts, then refused with "it needs
    senbonzakura_check, which is not installed". True, actionable, and after all the work. Section
    2.1 asks for dependency presence to be checked "before a long job starts, not halfway
    through", and this project already does that for the CUDA device, the skip boundaries and the
    capability-probe budget.

    HERE, AND NOT IN A TABLE IN `entry`. The first attempt listed which commands defer the import
    and pre-flighted them before dispatch. A test derived the same list from the source and showed
    the hand-written one was wrong in both directions, then showed that neither a direct-import
    walk nor a transitive one can produce the right answer: direct misses `leak`, which reaches
    the checker through `residualleak`, and transitive reachability sweeps in `doctor`, which must
    run with nothing installed. Predicting the set was the wrong idea.

    The loader is the honest place. Everything expensive is downstream of it, every command that
    writes a stamped figure passes through it, and the commands that never load a model, `doctor`
    and `diff` and `track` among them, never reach it and keep working with the checker absent.
    No table, nothing to drift, and the cost avoided is the whole load plus the scoring.

    `find_spec` rather than an import, so this costs microseconds and executes nothing.
    """
    import importlib.util

    try:
        present = importlib.util.find_spec(CHECKER) is not None
    except (ImportError, ValueError):
        present = False
    if present:
        return
    raise SystemExit(
        f"senbonzakura: {CHECKER} is not installed, and every measurement this writes is stamped "
        f"through it.\n"
        f"  Refusing now rather than after the model loads: on a processor that wait is minutes, "
        f"and the answer at the end of it would be this same sentence.\n"
        f"  Install it with:\n"
        f"    pip install senbonzakura\n"
        f"  `senbonzakura doctor` lists everything this install is missing in one go, and it runs "
        f"without any of it.")


#: Seconds between heartbeat lines while a model load is in progress. A load on a rented pod once went
#: silent for an hour and nothing said where it was stuck, so a stalled load now names its sub-step.
HEARTBEAT_SECONDS = 60.0

#: Heartbeats in the SAME sub-step after which every thread's stack is written to the log, once.
#: At the default interval that is ten minutes, long past any load that is merely slow.
STACK_DUMP_AFTER_BEATS = 10

_JOIN_SECONDS = 5.0
_active_heartbeat = None


def _resident_bytes():
    """This process's resident memory in bytes, from the kernel's own account of it."""
    with open("/proc/self/statm", "rb") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def _cuda_allocated_bytes():
    """Bytes torch has allocated on the current device, or None before the card is started.

    `is_initialized` only looks, where `is_available` can start the card itself: if starting the card is what
    the load is stuck on, a heartbeat that started it too would hang beside it and say nothing.
    """
    if not torch.cuda.is_initialized():
        return None
    return torch.cuda.memory_allocated()


def _reading(read):
    # A failed reading must show up in the line as unavailable, with its reason, and never raise.
    try:
        value = read()
    except Exception as e:
        return f"unavailable ({type(e).__name__}: {e})"
    if value is None:
        return "no card"
    return f"{value / (1 << 30):.2f} GiB"


class _Heartbeat:
    def __init__(self, log, step, interval, dump_after):
        self._log = log
        self.step = step
        self._interval = interval
        self._dump_after = dump_after
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._step_started = self._started
        self._beats_in_step = 0
        self._dumped = False
        self._thread = threading.Thread(target=self._run, name="load-heartbeat", daemon=True)

    def set_step(self, name):
        if name != self.step:
            self.step = name
            self._step_started = time.monotonic()
            self._beats_in_step = 0
            self._dumped = False

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(_JOIN_SECONDS)

    def _say(self, message):
        try:
            self._log(message)
        except Exception as e:
            # The log sink failed; the one other place a reader can look is stderr.
            try:
                sys.stderr.write(f"{message}\n[load heartbeat] the log could not be written: "
                                 f"{type(e).__name__}: {e}\n")
            except Exception:
                pass

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                self._beat()
            except Exception as e:
                self._say(f"  [load] heartbeat failed ({type(e).__name__}: {e}); the load continues")

    def _beat(self):
        now = time.monotonic()
        self._beats_in_step += 1
        step = self.step
        # The stacks first: a reading below could block on whatever the load is stuck on, and the dump is
        # the one line a stall must leave.
        if self._beats_in_step >= self._dump_after and not self._dumped:
            self._dumped = True
            self._say(f"  [load] no progress past '{step}' for {self._beats_in_step} heartbeats; "
                      f"stacks of every thread follow\n{_all_stacks()}")
        self._say(
            f"  [load] still working: {step}, {now - self._started:.0f}s since the load began "
            f"({now - self._step_started:.0f}s in this step), resident memory "
            f"{_reading(_resident_bytes)}, cuda allocated {_reading(_cuda_allocated_bytes)}")


def _all_stacks():
    names = {t.ident: t.name for t in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():
        parts.append(f"--- thread {names.get(ident, '?')} ({ident})\n"
                     + "".join(traceback.format_stack(frame)))
    return "\n".join(parts).rstrip("\n")


@contextlib.contextmanager
def load_heartbeat(log, step):
    """While the block runs, log one line every HEARTBEAT_SECONDS naming the current sub-step.

    A nested use joins the heartbeat already running instead of starting a second thread, so the
    outer caller and the loader can each name their steps. Silent when `log` is None.
    """
    global _active_heartbeat
    if _active_heartbeat is not None:
        _active_heartbeat.set_step(step)
        yield _active_heartbeat
        return
    if log is None:
        yield None
        return
    beat = _Heartbeat(log, step, HEARTBEAT_SECONDS, STACK_DUMP_AFTER_BEATS)
    _active_heartbeat = beat
    beat.start()
    try:
        yield beat
    finally:
        try:
            beat.stop()
        finally:
            _active_heartbeat = None


def load_step(name):
    """Name the sub-step now running; does nothing when no heartbeat is active."""
    beat = _active_heartbeat
    if beat is not None:
        beat.set_step(name)


def load_model_and_tokenizer(model_id, device="cuda", load_in_4bit=False,
                             trust_remote_code=False, attn_impl=None, log=None,
                             chat_template=None, needs_chat_template=True,
                             offload_dir=None, revision=None, load_in_8bit=False,
                             gguf_file=None, forward_only=True, dtype=None):
    with load_heartbeat(log, "checks before the weights"):
        return _load_model_and_tokenizer(
            model_id, device=device, load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code, attn_impl=attn_impl, log=log,
            chat_template=chat_template, needs_chat_template=needs_chat_template,
            offload_dir=offload_dir, revision=revision, load_in_8bit=load_in_8bit,
            gguf_file=gguf_file, forward_only=forward_only, dtype=dtype)


def _load_model_and_tokenizer(model_id, device="cuda", load_in_4bit=False,
                             trust_remote_code=False, attn_impl=None, log=None,
                             chat_template=None, needs_chat_template=True,
                             offload_dir=None, revision=None, load_in_8bit=False,
                             gguf_file=None, forward_only=True, dtype=None):
    # Shared model loader for the abliterator and the scorer. Left-pads the tokenizer and sets a pad
    # token, loads in bf16 (or 4-bit via bitsandbytes when asked), and honours trust_remote_code and
    # a chosen attention implementation. Placement uses accelerate's device_map so multi-GPU and a
    # specific cuda:N both work. 4-bit is for the pure-forward paths only (scoring / measurement):
    # the weight bake rewrites tensors in place and needs full precision.
    _log = log or (lambda m: None)
    _refuse_without_the_checker()
    # BEFORE accelerate SEES IT. `device_map` below routes placement through accelerate's
    # sharded-checkpoint loader, which takes shard names out of the index and opens them relative
    # to this directory without sanitising them (PYSEC-2026-3804, no fixed release). This is the
    # only point we control. It applies to a local directory; a bare model id is fetched and
    # loaded inside transformers, with no point between the two to stand.
    # WHAT KIND OF CHECKPOINT THIS IS, DECIDED BEFORE ANYTHING IS FETCHED. A missing backend
    # package, a flag that can't apply to this format, or a GGUF folder with two files in it is a
    # sentence now and not a download followed by a traceback. See `modelload`.
    try:
        chosen = plan(model_id, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
                              gguf_file=gguf_file, revision=revision, forward_only=forward_only)
    except LoadRefusedError as e:
        raise SystemExit(str(e)) from e
    source = chosen.source
    if Path(source).is_dir():
        checkpoint.refuse_unsafe_index(source)
    elif chosen.kind != "gguf":
        checkpoint.refuse_unsafe_hub_index(model_id, revision=revision)
    # AND THE DEVICE, AFTER THE INDEX CHECK AND BEFORE ANY WEIGHTS. `--device` defaults to `cuda`,
    # and `_preflight_device` refused an unusable one on the abliterate path and the guided mode
    # only. Every other command reached `.to(device)` with the weights already fetched and loaded,
    # and died there under fourteen frames of torch. On a metered connection that download is the
    # whole cost of finding out. Found by the constrained-user journey, 2026-10-11.
    #
    # HERE RATHER THAN AT EACH `main`, which is where it went first and was wrong. A surface check
    # preempts every refusal the command does for itself, so eighteen tests that assert a specific
    # early refusal got this one instead. Worse, they only got it on a machine with no card, which
    # is CI: the suite's behaviour would have depended on the test host's hardware.
    #
    # AND AFTER THE INDEX CHECK RATHER THAN BEFORE IT, which is the second ordering mistake in the
    # same fix. Put first, it preempted the PYSEC-2026-3804 refusal above, so a hostile checkpoint
    # went unexamined whenever the device string was also wrong. A convenience check must not be
    # able to stand in front of a security one. The weights are still downstream of this line, so
    # the cost it exists to save is still saved.
    argresolve.refuse_unusable_device(device, offload_dir=offload_dir)
    load_step("tokenizer")
    tok = load_tokenizer(source, trust_remote_code=trust_remote_code, log=log,
                         chat_template=chat_template, needs_chat_template=needs_chat_template,
                         revision=revision, gguf_file=chosen.gguf_file)
    load_step("memory budget")
    dtype = dtype or dtype_for(chosen, torch)
    kw = dict(dtype=dtype, trust_remote_code=trust_remote_code)
    if chosen.gguf_file:
        kw["gguf_file"] = chosen.gguf_file
    if revision:
        kw["revision"] = revision
    if attn_impl:
        kw["attn_implementation"] = attn_impl
    if load_in_4bit or load_in_8bit:
        kw["quantization_config"] = bnb_config(chosen, torch)
        # "auto" on a machine asked for the processor makes accelerate dispatch layers to a
        # card that isn't there and bitsandbytes then refuses; the placement follows the request.
        kw["device_map"] = ("auto" if device == "cuda" else {"": int(device.split(":", 1)[1])}
                            if device.startswith("cuda:") else {"": "cpu"})
        _log(f"loading in {'4-bit (nf4, double-quant)' if load_in_4bit else '8-bit (int8)'}; "
             f"measurement path only, the bake needs full precision")
    elif chosen.quantised and chosen.kind != "gguf" and device == "cpu":
        # A stored-quantised model is placed by its quantiser, and a later `.to()` on it can fail.
        kw["device_map"] = {"": "cpu"}
    elif device == "cuda":
        kw["device_map"] = "auto"                       # accelerate places / shards; supports big models
    elif device.startswith("cuda:"):
        kw["device_map"] = {"": int(device.split(":", 1)[1])}
    # A MODEL THAT CANNOT BE FETCHED IS A SENTENCE, NOT A TRACEBACK. transformers' own message is
    # good (it names the id and says it is neither a local folder nor a listed model), so it is
    # kept verbatim rather than reworded; what is dropped is the twenty frames of our internals
    # and theirs wrapped around it. Found by typing a model name with a character missing, which
    # is the most likely user error there is.
    # DISK OFFLOAD IS OPT-IN, AND ONLY FOR A FORWARD PASS. Without `offload_folder`, a
    # `device_map="auto"` fills the card and then host RAM and then fails; with it, accelerate
    # spills the rest to disk and reads each group back as the pass reaches it. That makes a model
    # larger than card plus RAM loadable for READING, which is all extraction needs.
    #
    # It is not offered to the bake and must not be. A disk-offloaded weight reads back as a fresh
    # copy on every access, so an in-place write lands on a temporary and is discarded; see
    # `_real_tensor`, which raises rather than pretending, and `capability.disk_offloaded_entries`,
    # which is the check that first found this distinction mattered. The caller decides, because
    # only the caller knows whether it is about to write.
    if offload_dir:
        if "device_map" not in kw:
            raise ValueError(
                f"offload_dir={offload_dir!r} was passed with device={device!r}, which places the "
                f"model with a plain `.to()` and never consults accelerate. Disk offload needs a "
                f"device_map, so this combination would create the directory and silently not use "
                f"it.")
        kw["offload_folder"] = str(offload_dir)
        # The state dict goes to disk as it is read rather than being held whole in host RAM,
        # which is the other half of the problem: a 200 GB checkpoint cannot be staged in RAM on
        # the way to being placed, even when the placement itself would have fitted.
        kw["offload_state_dict"] = True
    # THE HOST BUDGET, CORRECTED WHEN A CGROUP CAPS US. Applies to every accelerate placement and
    # not only to the offload path, because the wrong number is just as wrong without it: there the
    # overshoot is an OOM kill instead of a spill to disk, which is the same defect with a worse
    # ending. See `_corrected_max_memory` for why the figure accelerate computes can be wrong.
    if "device_map" in kw:
        # The config is read separately and BEFORE the weights, because the headroom has to be
        # subtracted from the budget the placement is computed against. Reading it is a metadata
        # fetch against a cache transformers is about to consult anyway. A config that cannot be
        # read is not fatal here: the load below will fail with a better message than anything
        # this could say, so the correction degrades to the host-only one.
        cfg = None
        try:
            cfg = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code,
                                             **({"revision": revision} if revision else {}),
                                             **({"gguf_file": chosen.gguf_file} if chosen.gguf_file else {}))
        except Exception as e:
            _log(f"  the model config could not be read before the weights "
                 f"({type(e).__name__}: {e}), so no activation headroom is reserved.")
        corrected = _corrected_max_memory(_log, cfg, dtype_bytes=torch.empty((), dtype=dtype).element_size())
        if corrected is not None:
            kw["max_memory"] = corrected
    # SAID BEFORE THE WAIT, NOT AFTER IT. Measured 2026-10-11 while reproducing a journey
    # finding: `senbonzakura compass` on a processor sat for six minutes having printed exactly
    # one line, and that line was huggingface's own rate-limit warning. Nothing of ours said a
    # model was being fetched, so the only thing on screen was somebody else's prose and the run
    # looked wedged. Section 2 asks for a heartbeat on every long wait; a fetch over a slow or
    # throttled link is one of the longest this tool has, and it had no entry line at all.
    #
    # The download note is added only for a Hub id. A local directory loads at disk speed, so
    # telling that reader about a download would be wrong about what they are waiting for, and
    # being confidently wrong about the cause is the shape of message this project keeps removing.
    _log(f"  loading {model_id} on {device}"
         f"{'' if Path(source).is_dir() else ' (a first run downloads it, which is the slow part)'}"
         f"{'' if chosen.kind == 'plain' else f' as {chosen.kind}'}")
    if chosen.kind == "gguf":
        _log(f"  a GGUF is expanded to {str(dtype).removeprefix('torch.')} to be measured here, so "
             f"the figure is for the dequantised weights "
             f"({chosen.dequantised_from or 'quantisation unknown'}), not the file as llama.cpp runs it")
    # The load report is asked for so a checkpoint that left weights missing is refused here; see
    # `loadreport` for the run that measured random numbers because nothing read it.
    kw["output_loading_info"] = True
    # On a card, GPTQ and AWQ are loaded with a kernel that needs no compiler, the faster one first.
    # A failure with the first is retried once with the second and said in a sentence; with none left,
    # or on any other failure, the usual refusals below apply.
    kernels = kernel_choices(chosen, device) or (None,)
    load_step("weights")
    for position, kernel in enumerate(kernels):
        attempt = dict(kw)
        if kernel:
            attempt["quantization_config"] = kernel_config(chosen, kernel)
        more = position + 1 < len(kernels)
        try:
            model, info = AutoModelForCausalLM.from_pretrained(source, **attempt)
            if kernel:
                chosen = replace(chosen, kernel=kernel)
            break
        # `ValueError` beside `OSError` for the same reason as the tokenizer above: a file that is
        # present and malformed fails in a parser rather than in the filesystem. A missing file is
        # never retried with another kernel; a kernel that could not be built or run (which surfaces
        # as a ValueError, an ImportError or a RuntimeError) is.
        except (OSError, ValueError) as e:
            if kernel and more and isinstance(e, ValueError):
                _log("  " + kernel_fallback_notice(kernel, kernels[position + 1], e))
                continue
            raise SystemExit(remote_code_refusal(model_id, e)
                             or _could_not_load(model_id, e)) from e
        except (ImportError, RuntimeError) as e:
            if kernel and more:
                _log("  " + kernel_fallback_notice(kernel, kernels[position + 1], e))
                continue
            if isinstance(e, ImportError) and chosen.kind != "plain":
                raise SystemExit(import_refusal(chosen, e)) from e
            raise
    load_step("after the weights")
    refusal = loadreport.missing_weights(model_id, info)
    if refusal:
        raise SystemExit(refusal)
    if "device_map" not in kw:                          # cpu path
        model = model.to(device)
    model.eval()
    attach(model, chosen, str(dtype).removeprefix("torch."))
    return model, tok
