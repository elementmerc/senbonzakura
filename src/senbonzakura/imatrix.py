# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura imatrix`: compute an importance matrix, so a quantisation can be compared fairly.

WHAT AN IMPORTANCE MATRIX IS, IN ONE PARAGRAPH

Quantising a model means storing each weight in fewer bits, and the question is which weights can
afford to lose precision. A plain quantisation answers "all of them equally". An importance matrix
answers it with evidence: run some text through the model, watch which weights actually move the
activations, and spend the bits there. Same file size, better output, and llama.cpp names the
result `i1-` by convention.

WHY THIS IS HERE RATHER THAN LEFT TO WHOEVER SERVES THE MODEL

Because the difference is large enough to be mistaken for a result. Take an abliterated model
quantised WITH an importance matrix, compare it against its stock sibling quantised without one,
and the measured gap is two variables wearing one name: part of it is the edit and part of it is
the quantiser. That comparison has been made, on this family, and it read as an effect of the
edit. Nothing careless happened; the only place the difference was recorded was the filename, one
carrying `i1-` and the other not.

A tool that edits models and then hands them to be quantised elsewhere cannot make that comparison
one-variable. A tool that owns both halves can.

THE CALIBRATION SET IS A CHOICE, AND THIS COMMAND REFUSES TO MAKE IT SILENTLY

An importance matrix is computed FROM text, and it preserves the weights that mattered for THAT
text. Calibrate on harmful prompts and you have preserved the machinery those prompts exercise,
which on a refusal-abliteration tool is very much not a neutral act: it is plausible that it
partially preserves, or partially erases, exactly the behaviour under measurement. Nobody here has
measured which.

So there is no default corpus. The command asks, names the trade-off, and records what was used in
a sidecar file so a later reader can tell what a given `i1-` file was calibrated on.
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
from .vendored import VendorError, find_binary

#: Chunks of calibration text to process. llama.cpp's own guidance is that a few hundred is
#: plenty; the default here is deliberately modest because this runs on a laptop card.
DEFAULT_CHUNKS = 128

#: The sidecar that travels with an imatrix, so "what was this calibrated on" has an answer that
#: does not depend on someone remembering. Written beside the matrix, same stem.
SIDECAR_SUFFIX = ".calibration.json"


class ImatrixError(Exception):
    """An importance matrix that cannot be trusted, phrased for a person."""


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura imatrix",
        description="Compute an importance matrix for a GGUF, for a fairer quantisation.",
        epilog="The calibration text decides which weights keep their precision, so it is part of "
               "the result rather than a detail. Pick it deliberately.")
    ap.add_argument("model", help="the GGUF to calibrate (full precision, not already quantised)")
    ap.add_argument("-o", "--out", default=None,
                    help="output path (default: the model's name with -imatrix.gguf)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", default=None,
                     help="a bundled corpus name to calibrate on. NOTE: calibrating on harmful "
                          "prompts preserves the machinery those prompts exercise, which is not "
                          "neutral for a refusal measurement")
    src.add_argument("--file", default=None,
                     help="a plain text file of calibration text. The usual choice for a general "
                          "purpose quantisation is broad prose rather than task prompts")
    ap.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS,
                    help=f"chunks of text to process (default: {DEFAULT_CHUNKS})")
    ap.add_argument("--gpu-layers", type=int, default=0,
                    help="layers to offload to the GPU. 0 keeps it on the CPU, which is slower and "
                         "always works")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    return ap


def default_output(model):
    p = Path(model)
    stem = p.name[: -len(p.suffix)] if p.suffix else p.name
    return p.with_name(f"{stem}-imatrix.gguf")


def calibration_text(corpus=None, path=None):
    """The text to calibrate on, and a record of where it came from.

    Returns `(text, provenance)`. The provenance is written to the sidecar rather than being
    reconstructed later from a filename, because a filename is exactly where the last confound of
    this kind hid.
    """
    if path:
        p = Path(path)
        if not p.is_file():
            raise ImatrixError(f"no calibration file at {p}.")
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise ImatrixError(f"{p} is empty, so there is nothing to calibrate on.")
        return text, {"source": "file", "path": str(p.resolve()), "bytes": len(text)}

    from .corpora import CORPORA, CorpusError, resolve_name
    try:
        key = resolve_name(corpus)
        from .corpora import load
        rows = load(key)
    except CorpusError as e:
        raise ImatrixError(str(e)) from e
    c = CORPORA[key]
    return "\n".join(rows), {"source": "bundled corpus", "corpus": key, "arm": c.arm,
                             "rows": len(rows), "upstream": c.upstream, "commit": c.commit}


def preflight(model, out, *, force):
    src = Path(model)
    if not src.is_file():
        raise ImatrixError(f"no model at {src}.")
    if Path(out).resolve() == src.resolve():
        raise ImatrixError(f"the output path is the model ({src}). Name a different --out.")
    if Path(out).exists() and not force:
        raise ImatrixError(f"{out} exists. Pass --force to overwrite it, or choose another path.")
    try:
        head = gguf_io.verify(src, expect_quant=False)
    except gguf_io.GGUFError as e:
        raise ImatrixError(f"cannot read {src}: {e}") from e
    if head["file_type"] not in ("F32", "F16", "BF16"):
        # Calibrating an already-quantised model measures the quantised weights, which is not what
        # the matrix is for: it is meant to decide how to quantise the full-precision ones.
        raise ImatrixError(
            f"{src} is already a {head['file_type']}. An importance matrix is computed from "
            f"full-precision weights so it can decide which of them to keep; computing one from "
            f"an already-quantised model measures the damage rather than guiding it.")
    free = free_bytes_for(out)
    if free is not None and free < 1 << 30:
        raise ImatrixError(f"only {free / 1e9:.1f} GB free where the matrix goes.")
    return head


def run(argv=None, log=print):
    a = build_parser().parse_args(argv)
    out = Path(a.out) if a.out else default_output(a.model)

    try:
        head = preflight(a.model, out, force=a.force)
        text, provenance = calibration_text(corpus=a.corpus, path=a.file)
    except ImatrixError as e:
        raise SystemExit(f"cannot compute an importance matrix: {e}") from e

    try:
        exe, source_of = find_binary("llama-imatrix", log=log)
    except VendorError as e:
        raise SystemExit(str(e)) from e

    log(f"imatrix for {Path(a.model).name} ({head['file_type']}, {head['architecture']})")
    log(f"  calibrating on {provenance.get('corpus') or provenance.get('path')} "
        f"({len(text):,} characters, {a.chunks} chunks)")
    if provenance.get("arm") == "harmful":
        # Said out loud, every time. A quiet default here would be the same shape of mistake as the
        # comparison this command exists to make honest.
        log("  NOTE: this is a harmful-prompt corpus. The matrix will preserve the weights those "
            "prompts exercise, which is not a neutral choice for a refusal measurement.")
    log(f"  using {source_of} llama-imatrix at {exe}")

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        cal = f.name

    argv_i = [str(exe), "-m", str(a.model), "-f", cal, "-o", str(out),
              "--chunks", str(a.chunks), "-ngl", str(a.gpu_layers)]
    started = time.monotonic()
    try:
        r = subprocess.run(argv_i, check=False)
    finally:
        Path(cal).unlink(missing_ok=True)
    took = time.monotonic() - started

    if r.returncode != 0 or not out.is_file():
        if out.exists():
            out.unlink()
        raise SystemExit(
            f"llama-imatrix exited {r.returncode} after {took:.0f}s. Nothing usable was written.")

    sidecar = out.with_name(out.name + SIDECAR_SUFFIX)
    sidecar.write_text(json.dumps(
        {"schema": "senbonzakura-imatrix/1", "model": str(Path(a.model).resolve()),
         "architecture": head["architecture"], "chunks": a.chunks,
         "calibration": provenance}, indent=2) + "\n", encoding="utf-8")

    log(f"  wrote {out.name}: {out.stat().st_size / 1e6:.1f} MB, {took:.0f}s")
    log(f"  wrote {sidecar.name}: what this was calibrated on, so a later reader can tell")
    return 0


def main(argv=None):
    return run(argv)


def describe(path):
    """What an imatrix was calibrated on, or None if it carries no sidecar.

    Used by `quantise` so the artefact of a quantisation can name the calibration rather than
    just the fact that some matrix was applied.
    """
    p = Path(str(path) + SIDECAR_SUFFIX)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


if __name__ == "__main__":
    sys.exit(main())
