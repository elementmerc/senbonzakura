# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from pathlib import Path

#: The help text for `--trust-remote-code`, held here so every command says the same thing.
TRUST_REMOTE_CODE_HELP = ("allow models that ship custom modelling code (some Hub models need "
                          "it); off by default.")

#: Packages each format needs, as (import name, distribution name), and the extra that brings them.
BACKENDS = {
    "bnb": ((("bitsandbytes", "bitsandbytes"),), "quant"),
    "gptq": ((("optimum", "optimum"), ("gptqmodel", "gptqmodel")), "gptq"),
    "awq": ((("gptqmodel", "gptqmodel"),), "awq"),
    "gguf": ((("gguf", "gguf"),), "gguf"),
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


def require_backend(kind):
    """Refuse, in one sentence, when the package a format needs is not installed."""
    needed, extra = BACKENDS.get(kind, ((), None))
    missing = [dist for imp, dist in needed if not _installed(imp)]
    if missing:
        names = " and ".join(missing)
        raise LoadRefusedError(
            f"senbonzakura: this model is stored as {kind_label(kind)}, which needs {names} "
            f"and it isn't installed. Install it with: pip install \"senbonzakura[{extra}]\"")


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


def _gguf_quant(source, name):
    from . import gguf_io
    try:
        return gguf_io.read_header(Path(source) / name)["file_type"]
    except (gguf_io.GGUFError, OSError, ValueError, KeyError):
        # A hub id has nothing on disk to read yet; the record says so rather than guessing.
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
                    dequantised_from=_gguf_quant(source, name))
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
        "backend": backend_of(p.kind),
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
    extra = BACKENDS.get(p.kind, ((), None))[1]
    hint = f' Install a matching version with: pip install "senbonzakura[{extra}]"' if extra else ""
    return (f"senbonzakura: loading {kind_label(p.kind)} failed because a package it needs "
            f"couldn't be used: {first}.{hint}")


def remote_code_refusal(model_id, error):
    """The sentence for a model that ships its own code when `--trust-remote-code` wasn't given."""
    if "trust_remote_code" not in str(error):
        return None
    return (f"senbonzakura: '{model_id}' ships custom modelling code, which this tool won't run "
            f"unless you say so. If you trust the repository, add --trust-remote-code (off by "
            f"default).")
