# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura convert`: turn edited weights into a GGUF, and check what came out.

WHAT THIS REPLACES

`gguf2.sh`, `gguf3.sh` and a third named after a machine, three files whose header comment is identical word for
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
from ._version import __version__
from .crashsafe import free_bytes_for, provenance
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


#: The GGUF side lives in `gguf_io`, because `quantise` needs the same key and two spellings of
#: one metadata field is how a check starts passing on a file it is no longer reading.
GGUF_CHAT_TEMPLATE_KEY = gguf_io.CHAT_TEMPLATE_KEY
HF_TOKENIZER_CONFIG = "tokenizer_config.json"

#: The standalone file transformers writes the template to on a modern save, holding raw Jinja
#: rather than JSON. It is the DEFAULT location now, not an alternative one.
HF_CHAT_TEMPLATE_JINJA = "chat_template.jinja"

#: The intermediate convention, a JSON object with the template under the same key it used to
#: occupy in the tokeniser config. Processors and multimodal checkpoints still ship it.
HF_CHAT_TEMPLATE_JSON = "chat_template.json"


def source_chat_template(model_dir):
    """Does the checkpoint declare a chat template? True, False, or None for cannot tell.

    None is a real third answer and not a failure: a checkpoint that says nothing anywhere tells
    us nothing about what the output should carry, and a check that guesses in that case would
    fire on models it knows nothing about.

    THREE PLACES, AND THE ONE THIS ORIGINALLY READ IS NO LONGER THE USUAL ONE. Until 2026-09-21
    this looked only in `tokenizer_config.json`, which is where transformers used to keep the
    template and is where it keeps it least often now. A modern save writes raw Jinja to
    `chat_template.jinja` beside the config, and the config then has no `chat_template` key at
    all.

    That made this function return False, meaning "definitely none", for checkpoints carrying a
    perfectly good template. False is the answer that switches the whole check off: `chat_template_lost`
    only fires when the source is a definite True, so the export verification was a no-op on
    exactly the checkpoints anyone would convert today. Found on the ROG against LFM2.5-350M,
    whose template is 5,487 bytes in `chat_template.jinja` and absent from the tokeniser config.

    A guard returning a confident negative is worse than one returning None, because None is
    visible as an unknown and False is indistinguishable from a checked, clean result.
    """
    root = Path(model_dir)
    for name in (HF_CHAT_TEMPLATE_JINJA, HF_CHAT_TEMPLATE_JSON):
        path = root / name
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return None
        if name == HF_CHAT_TEMPLATE_JINJA:
            if raw.strip():
                return True
            continue
        try:
            loaded = json.loads(raw)
        except ValueError:
            return None
        if isinstance(loaded, dict) and loaded.get("chat_template"):
            return True

    cfg = root / HF_TOKENIZER_CONFIG
    if not cfg.is_file():
        # No tokeniser config AND no template file. Nothing here has an opinion, and the
        # standalone files were already checked above, so this is genuinely unknown.
        return None
    try:
        loaded = json.loads(cfg.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError):
        return None
    if not isinstance(loaded, dict):
        return None
    return bool(loaded.get("chat_template"))


def chat_template_lost(model_dir, header):
    """The prompt format went in and did not come out. Returns a sentence, or None.

    WHY THIS IS CHECKED AT ALL, since the conversion reported success and the tensors verify.
    A GGUF carries the chat template in its own metadata, and llama.cpp, Ollama and vLLM read it
    from there rather than from the checkpoint the file came from. If it is lost in conversion the
    file still loads, still answers, and answers badly: the model is handed raw text where it was
    trained to expect turn markers, so the failure looks like a bad model rather than a bad export.
    Nothing about the tensors is wrong, which is exactly why the existing receipt cannot see it.

    Compared against the SOURCE rather than asserted outright, because a base model legitimately
    has no template and refusing one would be refusing a correct file. The claim here is narrow and
    provable: this checkpoint had one, this output does not.
    """
    if source_chat_template(model_dir) is not True:
        return None
    if gguf_io.has_chat_template(header):
        return None
    return (f"the checkpoint declares a chat template and the converted file carries no "
            f"{GGUF_CHAT_TEMPLATE_KEY}. It will load and generate, and every conversational "
            f"result from it will be measured on a prompt format the model was not trained on. "
            f"Check the converter's output above for a tokeniser it did not recognise.")


#: What a conversion record is called, beside the file it describes. One per output, rather than
#: one per directory, because two conversions of one checkpoint at different precisions live side
#: by side and a shared name would leave the second silently overwriting the first's receipt.
RECORD_SUFFIX = ".conversion.json"

#: The marker an adapter detects this shape by. Two fields together rather than one, for the same
#: reason the abliteration record needs two: a single common field could plausibly appear in
#: somebody else's log, and an adapter that claims a foreign file reads it in the wrong language.
RECORD_KIND = "gguf-conversion"


def record_path(out):
    return Path(str(out) + RECORD_SUFFIX)


def build_conversion_record(model_dir, out, pre, got, *, outtype, quantise_to, took_s, lost):
    """Everything this conversion claims, as a value rather than a side effect.

    WHY THIS EXISTS AT ALL, when the conversion already prints what it did.

    `chat_template_lost` produced a log line and nothing else. A GGUF carries its chat template in
    its own metadata, and llama.cpp, Ollama and vLLM read it from there rather than from the
    checkpoint: lose it and the file still loads, still answers, and answers badly, because the
    model is handed raw text where it was trained to expect turn markers. The failure looks like a
    bad model rather than a bad export, and the one sentence saying otherwise scrolled past an
    operator once and was gone. That is the same shape as the `budget_warning` this project wrote
    into three artefacts and read none of: a caveat that lives only in a terminal is lost exactly
    where the number gets quoted from.

    So it goes in a file, beside the output, in the vocabulary the checker reads.

    WHY NO FIELD HERE IS CALLED `generation`, `prompt`, `output`, `text` OR `response`, and this
    is not a stylistic choice. `tools/ci/check_prompt_artefacts.py` refuses any committed JSON
    carrying one of those names at any depth, because that is what a retained model reply is
    called everywhere else in this project, and it was already refusing this project's own
    primary artefact until a field moved on 2026-09-21. The gate is the one control between a
    harmful prompt and a public push; a record that could not be committed would be a record
    nobody keeps.

    WRITTEN AFTER VERIFICATION AND BEFORE QUANTISATION, which is a real choice. It describes the
    conversion, from a checkpoint to a GGUF, and that is finished and read back by this point. A
    quantisation that follows produces a different file with its own properties, so the record
    names it as requested rather than claiming to describe it.
    """
    source = Path(model_dir).resolve()
    return {
        "record": RECORD_KIND,
        "tool_version": __version__,
        "source": {
            "path": str(source),
            "name": source.name,
            "architecture": pre["architecture"],
            "shards": pre["shards"],
            "bytes": pre["bytes"],
            # True, False, or None for "nothing here has an opinion". None is a real third answer
            # and the check that reads it stays quiet on it: a confident negative on a checkpoint
            # nobody could read is indistinguishable from a checked, clean result.
            "declares_chat_template": source_chat_template(source),
        },
        "target": {
            "path": str(Path(out).resolve()),
            "name": Path(out).name,
            "architecture": got.get("architecture"),
            "file_type": got.get("file_type"),
            "tensor_count": got.get("tensor_count"),
            "bytes": Path(out).stat().st_size if Path(out).exists() else None,
            "carries_chat_template": gguf_io.has_chat_template(got),
        },
        "converter": {
            "script": str(pre["script"]),
            "outtype": outtype,
            "seconds": round(took_s, 1),
        },
        # The sentence `chat_template_lost` returns, or None when nothing was lost. Its presence
        # is the signal, exactly as `budget_warning`'s is on the abliteration record.
        "chat_template_warning": lost,
        # What was ASKED to follow, not what happened. See the docstring: this record describes
        # the conversion, and a quantised file is a different file.
        "quantise_requested": quantise_to,
        "provenance": provenance(),
    }


def write_conversion_record(out, record, *, log=print):
    """Write the record beside its output, atomically, and never fail the conversion for it.

    ATOMIC BECAUSE A HALF-WRITTEN RECEIPT IS WORSE THAN NONE: it parses as far as the reader gets
    and then stops, which is a file that looks like evidence. Baseline section 2.1.

    BEST EFFORT BECAUSE THE GGUF IS THE PRODUCT. A conversion that took forty minutes and
    verified must not be reported as a failure because a read-only directory refused a receipt.
    The refusal is loud in the log, which is the honest outcome: the file is good and the
    paperwork is missing, and the reader is told which.
    """
    path = record_path(out)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        log(f"  NOTE: the GGUF is fine and its conversion record could not be written to {path} "
            f"({e}). Nothing downstream will be able to check what this file was converted from.")
        return None
    return path


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
    # THE EXIT STATUS, which was read and discarded. The script imports torch at module level, so
    # on an install without it the process dies with a traceback and this returned an empty set,
    # which `doctor` then reported as "the vendored package is incomplete; re-run the vendoring
    # tool". Every clause of that was untrue, and the remedy it named is `tools/packaging/vendor_llama.py`,
    # which no wheel contains. Same class as telling the operator to update a driver that had run
    # CUDA all night: a refusal for a reason that is not true sends the reader to the wrong thing.
    died = None
    if r.returncode != 0 and not names:
        tail = [ln for ln in out.splitlines() if ln.strip()][-3:]
        died = " / ".join(tail) if tail else f"exited {r.returncode} with no output"
    return names, broken, died


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
        names, broken, _died = supported_architectures(script)
        if broken:
            # Loud, because this is the failure that shipped: a module that did not import while
            # its architectures stayed on the advertised list.
            raise ConvertError(
                f"the vendored converter could not import {len(broken)} of its architecture "
                f"modules ({', '.join(broken[:5])}), so the architectures they provide are "
                f"advertised and absent. This is a broken vendoring rather than anything about "
                f"your model: re-run `python tools/packaging/vendor_llama.py`.")
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

#: How much of a tensor to compare at a time. An embedding table is hundreds of megabytes on a
#: real model and there is no reason to hold two of them to answer a yes/no question.
COMPARE_CHUNK = 8 << 20


def safetensors_header(path):
    """The tensor index of a safetensors file: name -> {dtype, shape, data_offsets}.

    Read from the format directly rather than through `safetensors.safe_open`, and that is the
    point rather than an optimisation. `safe_open(..., framework="np")` cannot materialise a bf16
    tensor at all ("data type 'bfloat16' not understood"), because numpy has no bfloat16, and bf16
    is what most modern checkpoints ship. The header is plain JSON and carries everything needed
    to compare two tensors without decoding either.

    It also keeps this module free of torch and numpy, which the clean room checks: `convert` has
    to give a clean refusal on an install that has neither, not an ImportError.
    """
    p = Path(path)
    try:
        with p.open("rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise ConvertError(f"{p.name} is too short to be safetensors.")
            n = int.from_bytes(raw, "little")
            if n <= 0 or n > 100 << 20:
                raise ConvertError(f"{p.name} declares an implausible header of {n} bytes.")
            doc = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise ConvertError(f"{p.name} is not readable as safetensors ({e}).") from e
    # `__metadata__` is the format's own free-text slot, not a tensor.
    return {k: v for k, v in doc.items() if k != "__metadata__"}


def raw_bytes_equal(p_a, entry_a, p_b, entry_b, *, chunk=COMPARE_CHUNK):
    """Whether two tensors hold identical bytes. Returns (equal, first differing offset or None).

    Byte equality is what the caller actually means, stated exactly: a tensor that merely rounds to
    another is not the same tensor. Comparing the stored bytes says that in any dtype, including
    the ones no numeric library here can represent, and it is correct for NaN payloads where value
    comparison is not.
    """
    (a0, a1), (b0, b1) = entry_a["data_offsets"], entry_b["data_offsets"]
    if (a1 - a0) != (b1 - b0):
        return False, 0
    with Path(p_a).open("rb") as fa, Path(p_b).open("rb") as fb:
        # The data section starts after the 8-byte length and the header itself, and both files
        # are already open, so the bases are read here rather than through a second open apiece.
        base_a = 8 + int.from_bytes(fa.read(8), "little")
        base_b = 8 + int.from_bytes(fb.read(8), "little")
        fa.seek(base_a + a0)
        fb.seek(base_b + b0)
        remaining, seen = a1 - a0, 0
        while remaining > 0:
            n = min(chunk, remaining)
            ba, bb = fa.read(n), fb.read(n)
            if len(ba) != n or len(bb) != n:
                raise ConvertError("a safetensors shard ended before its header said it would.")
            if ba != bb:
                for i, (x, y) in enumerate(zip(ba, bb, strict=True)):
                    if x != y:
                        return False, seen + i
            remaining -= n
            seen += n
    return True, None


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

    head_shard = embed_shard = None
    headers = {}
    for p in shards:
        try:
            headers[p] = safetensors_header(p)
        except ConvertError:
            return "absent", None, f"{p.name} is not readable as safetensors"
        if TIED_HEAD in headers[p]:
            head_shard = p
        if TIED_EMBED in headers[p]:
            embed_shard = p
    if head_shard is None:
        return "absent", None, "the config declares tying and no separate head is present"
    if embed_shard is None:
        return "contradicts", head_shard, (
            f"{TIED_HEAD} is present and {TIED_EMBED} is not, so there is nothing to tie it to")

    a = headers[head_shard][TIED_HEAD]
    b = headers[embed_shard][TIED_EMBED]
    if a["shape"] != b["shape"]:
        return "contradicts", head_shard, (
            f"{TIED_HEAD} is {tuple(a['shape'])} and {TIED_EMBED} is {tuple(b['shape'])}")
    if a["dtype"] != b["dtype"]:
        return "contradicts", head_shard, (
            f"{TIED_HEAD} is {a['dtype']} and {TIED_EMBED} is {b['dtype']}")
    same, where = raw_bytes_equal(head_shard, a, embed_shard, b)
    if not same:
        return "contradicts", head_shard, (
            f"{TIED_HEAD} and {TIED_EMBED} are both {a['dtype']}{tuple(a['shape'])} and their "
            f"bytes differ, first at offset {where}")
    return "redundant", head_shard, f"{TIED_HEAD} is byte-identical to {TIED_EMBED}"


def copy_shard_without(src_shard, dst_shard, drop, *, chunk=COMPARE_CHUNK):
    """Write `src_shard` to `dst_shard` with one tensor removed, copying bytes and decoding none.

    Returns the dropped tensor's header entry.

    Byte-level for the same reason the comparison is: `safetensors.numpy.load_file` cannot
    represent bf16, which is what most modern checkpoints ship, and decoding a tensor to write it
    back unchanged would be work done only to lose precision on the way. It also means a shard
    larger than memory streams rather than being materialised, and every surviving tensor is
    byte-identical to its source by construction.
    """
    header = safetensors_header(src_shard)
    if drop not in header:
        raise ConvertError(f"{Path(src_shard).name} does not hold {drop}.")
    with Path(src_shard).open("rb") as f:
        src_base = 8 + int.from_bytes(f.read(8), "little")
    dropped = header[drop]

    # Offsets are recomputed from scratch, because removing a tensor moves everything after it.
    # Order is preserved: a reader may not care, but a diff of two headers is far easier to read
    # when the only change is the absence.
    out, cursor, plan = {}, 0, []
    for name, entry in header.items():
        if name == drop:
            continue
        a0, a1 = entry["data_offsets"]
        size = a1 - a0
        out[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                     "data_offsets": [cursor, cursor + size]}
        plan.append((src_base + a0, size))
        cursor += size

    blob = json.dumps(out, separators=(",", ":")).encode("utf-8")
    # The format wants the data section 8-byte aligned, and the reference writer pads the header
    # with spaces to achieve it. A header that is merely valid JSON but misaligned is readable by
    # some tools and not others, which is the worst of both.
    pad = (-len(blob)) % 8
    blob += b" " * pad

    tmp = Path(dst_shard).with_suffix(".part")
    try:
        with Path(src_shard).open("rb") as fin, tmp.open("wb") as fout:
            fout.write(len(blob).to_bytes(8, "little"))
            fout.write(blob)
            for offset, size in plan:
                fin.seek(offset)
                remaining = size
                while remaining > 0:
                    piece = fin.read(min(chunk, remaining))
                    if not piece:
                        raise ConvertError(
                            f"{Path(src_shard).name} ended before its header said it would.")
                    fout.write(piece)
                    remaining -= len(piece)
        tmp.replace(dst_shard)      # atomic: a half-written shard must never look like a whole one
    finally:
        tmp.unlink(missing_ok=True)
    return dropped


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
    src, work = Path(model_dir), Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    index_name = "model.safetensors.index.json"
    for p in src.iterdir():
        if not p.is_file() or p.name in (shard.name, index_name):
            continue
        (work / p.name).symlink_to(p.resolve())

    dropped = copy_shard_without(shard, work / shard.name, TIED_HEAD)
    log(f"  dropped {TIED_HEAD} {tuple(dropped['shape'])}: the config declares tied embeddings "
        f"and it is byte-identical to {TIED_EMBED}")

    index = src / index_name
    if index.is_file():
        doc = json.loads(index.read_text(encoding="utf-8"))
        wmap = doc.get("weight_map", {})
        wmap.pop(TIED_HEAD, None)
        meta = doc.get("metadata")
        if isinstance(meta, dict) and "total_size" in meta:
            a0, a1 = dropped["data_offsets"]
            meta["total_size"] = int(meta["total_size"]) - (a1 - a0)
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

    lost = chat_template_lost(a.model, got)
    if lost:
        log(f"  NOTE: {lost}")

    # THE RECEIPT, which is the half that outlives the terminal. Written whether or not anything
    # went wrong, because a record that appears only on a bad conversion tells a reader nothing
    # about a good one: absence would have to be read as either "fine" or "this build is too old
    # to say", and those are different claims.
    written = write_conversion_record(
        out, build_conversion_record(a.model, out, pre, got, outtype=a.outtype,
                                     quantise_to=a.quantise, took_s=took, lost=lost),
        log=log)
    if written:
        log(f"  conversion record: {written.name}")

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


if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(main)
