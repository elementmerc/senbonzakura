# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura quantise`: turn a GGUF into a smaller one, and check what came out.

WHAT THIS REPLACES

Four shell scripts that differed by a digit: `gguf2.sh`, `gguf3.sh`, `gguf-atlas.sh` and
`gguf-after-kl.sh`, each a hand-written pipeline with paths for one machine baked into it. The
house rule that produced this file: the moment a script gets a digit it has earned a subcommand,
and the superseded copy is deleted in the same change rather than left beside it.

WHY IT WRAPS A BINARY RATHER THAN DOING THE ARITHMETIC

k-quant quantisation exists in llama.cpp's C++ and nowhere this package can reach. The `gguf`
package implements quantisation for Q4_0, Q8_0 and BF16 but only DEQUANTISATION for Q4_K, Q5_K and
Q6_K, and a Q4_K_M is a mixture of Q4_K and Q6_K tensors. Reimplementing the k-quant search in
numpy was considered and rejected: its failure mode is a model that loads and is quietly slightly
worse, which is the defect class this project has withdrawn results over twice.

So `llama-quantize` is vendored, pinned, hash-verified and refreshed every release, and this
drives it.

WHAT IT ADDS OVER CALLING THE BINARY BY HAND

The binary's exit code is a statement about a process. Everything here that matters is a statement
about a FILE, and the two have already come apart once on this project: a 987 MB fragment of a
5.16 GB GGUF passed "exists and is non-empty", loaded, served, and answered nonsense that was
recorded as model quality. So the output is read back: magic, version, tensor count, architecture,
and the quantisation actually written, which must be the one that was asked for.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from . import gguf_io
from .crashsafe import free_bytes_for
from .vendored import VendorError, find_binary

#: Types this command will produce. Deliberately not "whatever the binary accepts": every entry
#: here is a name `gguf_io` can verify in the output header, so a typo cannot silently produce
#: something other than what was asked for. The k-quants are the reason the binary is vendored.
QUANT_TYPES = ("Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
               "Q6_K", "Q8_0", "Q4_0", "Q5_0", "IQ4_XS", "IQ4_NL", "F16", "BF16")

#: How much bigger than the source the output could conceivably be. Quantisation shrinks, so this
#: is a sanity floor for the disk check rather than an estimate: asking for F16 output from an F32
#: input is the one case where "smaller" is the wrong assumption.
SIZE_HEADROOM = 1.15



#: Written beside the output GGUF. Mirrors `imatrix`'s `.calibration.json`: the provenance of a
#: file lives next to the file, so a figure measured on it can name the toolchain that made it.
SIDECAR_SUFFIX = ".provenance.json"


def _now():
    """One UTC timestamp per operation, to seconds. Matches `track.py`'s `promoted_at`."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

#: llama.cpp prints this on the way into a real operation. The number is the upstream build, which
#: is the same number our pin carries as `bNNNNN`, so the two can be checked against each other.
_BUILD_RE = re.compile(r"build\s*=\s*(\d+)\s*\(([0-9a-f]+)\)")


def build_info(exe, *, timeout=20):
    """What the binary says its own build is, or None when it will not say.

    The identity of a quantiser is not the tag we believe we vendored; it is what the executable
    that actually ran reports about itself. Those can disagree, and the whole point of recording
    provenance is to be able to notice when they do.

    llama-quantize prints the line only once it is past argument parsing, so this asks it to
    dry-run a path that does not exist: the build banner is emitted, nothing is read, nothing is
    written, and the non-zero exit is expected rather than a failure. Returns None on anything
    unexpected, because a provenance field nobody can trust is worse than an absent one.
    """
    try:
        r = subprocess.run([str(exe), "--dry-run", "/nonexistent.senbonzakura.probe.gguf",
                            "/nonexistent.senbonzakura.probe.out.gguf", "Q4_K_M"],
                           capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    # getattr rather than attribute access: this is a probe, and anything unexpected about
    # the result means "cannot say", never a crash on the way into a real quantisation.
    text = (getattr(r, "stdout", "") or "") + (getattr(r, "stderr", "") or "")
    m = _BUILD_RE.search(text if isinstance(text, str) else "")
    return {"build": int(m.group(1)), "commit": m.group(2)} if m else None


def _sha256(path, *, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def pinned_tag():
    """The llama.cpp tag this install claims to vendor, or None if it cannot be read."""
    from . import vendoring
    try:
        manifest = vendoring.load_manifest()
    except (vendoring.VendorError, OSError):
        # A missing or malformed manifest is a real problem, and it is `doctor`'s to report.
        # Here it means only that this field cannot be filled, and an absent field is honest.
        return None
    pin = (manifest.get("pins") or {}).get("llama.cpp") or {}
    return pin.get("tag")


def quantiser_identity(exe, source_of, log=print):
    """Who did the quantising, stated so a reader can check it rather than trust it.

    Carries the binary's self-reported build AND the tag we believe we pinned, deliberately as
    two separate fields. Recording only one would make the interesting case, the case where they
    disagree, unrepresentable.
    """
    info = build_info(exe)
    tag = pinned_tag()
    ident = {
        "tool": "llama-quantize",
        "source": source_of,
        "path": str(exe),
        "pinned_tag": tag,
        "reported_build": info,
        "sha256": None,
    }
    try:
        ident["sha256"] = _sha256(exe)
    except OSError:
        pass

    # `b10355` against a reported build of 10355. A mismatch means the binary on disk is not the
    # one the pin names, which is exactly the substitution the pins exist to prevent, so it is
    # said out loud rather than left for whoever later reads the JSON.
    if info and tag and re.fullmatch(r"b\d+", str(tag)) and int(str(tag)[1:]) != info["build"]:
        ident["pin_mismatch"] = True
        log(f"  WARNING: the pin says {tag} and the binary reports build {info['build']}. "
            f"The quantiser that ran is not the one this install claims to vendor.")
    return ident


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura quantise",
        description="Quantise a GGUF with the pinned llama-quantize, then verify what was written.")
    ap.add_argument("source", help="an f16 or f32 GGUF to quantise")
    ap.add_argument("out", nargs="?", default=None,
                    help="output path (default: the source with its quant name substituted)")
    ap.add_argument("--type", default="Q4_K_M", choices=QUANT_TYPES,
                    help="target quantisation (default: Q4_K_M)")
    ap.add_argument("--threads", type=int, default=0,
                    help="worker threads; 0 lets llama-quantize choose")
    ap.add_argument("--allow-requantize", action="store_true",
                    help="quantise an already-quantised source. Lossy on top of lossy, and the "
                         "reason it is off by default")
    ap.add_argument("--imatrix", default=None, metavar="FILE",
                    help="apply an importance matrix, producing the better-quality quantisation "
                         "llama.cpp names `i1-`. Build one with `senbonzakura imatrix`. An i1 and "
                         "a plain quant of the SAME weights are not comparable, so a comparison "
                         "that mixes them is measuring the quantiser as well as the model")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    ap.add_argument("--keep-source", action="store_true",
                    help="do not offer to remove the source afterwards (it never removes it "
                         "without this being absent AND --prune-source given)")
    ap.add_argument("--prune-source", action="store_true",
                    help="delete the source once the output has been verified. The f16 halfway "
                         "file is usually the largest thing on the disk and is reproducible")
    return ap


def default_output(source, quant):
    """`model-f16.gguf` plus Q4_K_M becomes `model-Q4_K_M.gguf`.

    Named so that `gguf_io.verify` can check the file against its own name afterwards, which is
    the check that catches a Q8_0 sitting under a Q4_K_M name.
    """
    p = Path(source)
    stem = p.name[: -len(p.suffix)] if p.suffix else p.name
    claimed = gguf_io.claimed_quant(stem)
    stem = (stem.replace(claimed, quant).replace(claimed.lower(), quant) if claimed
            else f"{stem}-{quant}")
    return p.with_name(f"{stem}{p.suffix or '.gguf'}")


def preflight(source, out, quant, *, allow_requantize, force):
    """Everything checkable before a long job starts, because none of it is worth finding halfway.

    Returns the source's header so the caller does not read it twice.
    """
    src = Path(source)
    if not src.is_file():
        raise SystemExit(f"no source GGUF at {src}")

    # BEFORE the exists check, because it is the more specific and the more destructive of the
    # two. Ordered the other way, the only case that reaches it is `--force` on the same path,
    # where "it exists, pass --force" has already been printed and taken.
    if Path(out).resolve() == src.resolve():
        raise SystemExit(
            f"the output path is the source path ({src}), so the run would read a file it is "
            f"overwriting. Name a different --out.")

    # Translated rather than propagated. A GGUFError is a readable sentence already, and a
    # traceback in front of it is not the plain-language failure a user is owed.
    try:
        head = gguf_io.verify(src, expect_quant=False)   # its own name may say anything
    except gguf_io.GGUFError as e:
        raise SystemExit(f"cannot quantise {src}: {e}") from e

    if head["file_type"] not in ("F32", "F16", "BF16") and not allow_requantize:
        raise SystemExit(
            f"{src} is already a {head['file_type']}, and quantising a quantised model stacks one "
            f"lossy step on another. Convert from the original weights instead, or pass "
            f"--allow-requantize if you genuinely want that and will label the result.")

    if Path(out).exists() and not force:
        raise SystemExit(f"{out} exists. Pass --force to overwrite it, or choose another path.")

    need = int(src.stat().st_size * SIZE_HEADROOM)
    free = free_bytes_for(out)
    if free is not None and free < need:
        raise SystemExit(
            f"only {free / 1e9:.1f} GB free where the output goes and the source is "
            f"{src.stat().st_size / 1e9:.1f} GB. Quantising shrinks a model, but not before it has "
            f"written it, so free space or choose an --out on a larger volume.")
    return head


def run(argv=None, log=print):
    a = build_parser().parse_args(argv)
    out = Path(a.out) if a.out else default_output(a.source, a.type)

    head = preflight(a.source, out, a.type, allow_requantize=a.allow_requantize, force=a.force)
    log(f"quantise {Path(a.source).name} ({head['file_type']}, {head['tensor_count']} tensors, "
        f"{head['architecture']}) -> {out.name} [{a.type}]")

    try:
        exe, source_of = find_binary("llama-quantize", log=log)
    except VendorError as e:
        raise SystemExit(str(e)) from e
    log(f"  using {source_of} llama-quantize at {exe}")
    # Read BEFORE the work, so a run that dies mid-quantise has still said what was about to do it.
    identity = quantiser_identity(exe, source_of, log=log)

    argv_q = [str(exe)]
    if a.allow_requantize:
        argv_q.append("--allow-requantize")
    if a.imatrix:
        im = Path(a.imatrix)
        if not im.is_file():
            raise SystemExit(f"no importance matrix at {im}. Build one with `senbonzakura imatrix`.")
        argv_q += ["--imatrix", str(im)]
        from . import imatrix as _imatrix
        cal = _imatrix.describe(im)
        # Named, not just applied. Which text a matrix was calibrated on changes which weights keep
        # their precision, and a quantisation whose provenance is only in a filename is how the
        # imatrix confound hid in the first place.
        log(f"  applying the importance matrix at {im.name}"
            + (f", calibrated on {cal['calibration'].get('corpus') or cal['calibration'].get('path')}"
               if cal else " (no calibration sidecar; its provenance is unknown)"))
        imatrix_record = {"path": str(im), "name": im.name, "calibration": cal}
    else:
        imatrix_record = None
    argv_q += [str(a.source), str(out), a.type]
    if a.threads:
        argv_q.append(str(a.threads))

    started = time.monotonic()
    # No timeout: quantising a large model is genuinely long and a ceiling here would kill a job
    # with its work nearly done, which is the failure that cost a completed head-to-head on
    # 2026-08-06. Interrupting it is the operator's call, and the partial output is cleaned below.
    r = subprocess.run(argv_q, check=False)
    took = time.monotonic() - started

    if r.returncode != 0:
        # A failed quantisation leaves a partial file that is exactly the shape of a real one.
        if out.exists():
            out.unlink()
            log(f"  removed the partial {out.name}")
        raise SystemExit(
            f"llama-quantize exited {r.returncode} after {took:.0f}s. Nothing usable was written.")

    # THE OUTPUT IS READ BACK. An exit code is a statement about a process; every claim that
    # matters here is a statement about a file, and the two came apart on a 987 MB fragment of a
    # 5.16 GB GGUF that loaded, served, and answered nonsense.
    try:
        got = gguf_io.verify(out, expect_quant=a.type, expect_arch=head["architecture"])
    except gguf_io.GGUFError as e:
        raise SystemExit(
            f"llama-quantize reported success and the output does not verify: {e}\n"
            f"The file has been left at {out} for inspection rather than deleted, because what is "
            f"wrong with it is the interesting part.") from e

    src_size, out_size = Path(a.source).stat().st_size, out.stat().st_size
    log(f"  wrote {out.name}: {out_size / 1e9:.2f} GB from {src_size / 1e9:.2f} GB "
        f"({out_size / src_size * 100:.0f}%), {got['tensor_count']} tensors, {took:.0f}s")
    log(f"  verified: {got['file_type']}, architecture {got['architecture']}")

    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    record = {
        "schema": "senbonzakura-quantisation/1",
        "created": _now(),
        "quantiser": identity,
        "quant_type": a.type,
        "imatrix": imatrix_record,
        "allow_requantize": bool(a.allow_requantize),
        "source": {"name": Path(a.source).name, "bytes": src_size,
                   "file_type": head["file_type"], "architecture": head["architecture"]},
        "output": {"name": out.name, "bytes": out_size, "file_type": got["file_type"],
                   "architecture": got["architecture"], "tensor_count": got["tensor_count"]},
        "seconds": round(took, 1),
    }
    try:
        sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        log(f"  wrote {sidecar.name}: the toolchain that produced this file")
    except OSError as e:
        # The GGUF is good and is the point; losing its provenance is a degradation, not a
        # failure, and it degrades LOUDLY rather than leaving a silent gap in the record.
        log(f"  WARNING: could not write {sidecar.name} ({e}). The quantisation is fine and "
            f"its provenance is unrecorded.")

    if a.prune_source and not a.keep_source:
        # Only after the output has verified. The f16 halfway file is usually the largest thing on
        # the disk and it is reproducible from the original weights, but deleting it before the
        # replacement is known good is how one bad run costs both files.
        Path(a.source).unlink()
        log(f"  removed the source {Path(a.source).name} now that the output verifies")
    return 0


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
