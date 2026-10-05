# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura quantise`: turn a GGUF into a smaller one, and check what came out.

WHAT THIS REPLACES

Four shell scripts that differed by a digit: `gguf2.sh`, `gguf3.sh`, one named after a machine and
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

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import argresolve, gguf_io, say, vendored
from .crashsafe import atomic_write, digest_for_the_record, free_bytes_for
from .vendored import VendorError, find_binary

#: Types this command will produce. Deliberately not "whatever the binary accepts": every entry
#: here is a name `gguf_io` can verify in the output header, so a typo cannot silently produce
#: something other than what was asked for. The k-quants are the reason the binary is vendored.
QUANT_TYPES = ("Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
               "Q6_K", "Q8_0", "Q4_0", "Q5_0", "IQ4_XS", "IQ4_NL", "F16", "BF16")

#: The base recipe when nobody names one. Spelled once, because `--like` can also supply it and a
#: default written out twice is a default that drifts.
DEFAULT_TYPE = "Q4_K_M"

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
#:
#: TWO SPELLINGS, because upstream changed the banner between b10355 and b11046:
#:
#:     b10355   build = 10355 (0a1b2c3)
#:     b11046   version: 0.4.1-dev (build 11046, commit 60081bb2b)
#:
#: Both are matched rather than the newest only. The binary that runs is not always the one we
#: vendored: `source_of` can be a system llama.cpp of any age, and the field exists precisely to
#: record what actually ran. Reading only the current spelling would report "cannot say" for every
#: older build, which is indistinguishable from a binary that refused to identify itself.
#:
#: Found on 2026-09-25 by re-verifying the pin bump against a real model rather than trusting it:
#: the sidecar had started writing `reported_build: null` while every other field stayed correct,
#: so a quantisation looked fully provenanced and had lost the one field that can catch a binary
#: disagreeing with its pin.
_BUILD_RE = re.compile(
    r"build\s*=\s*(\d+)\s*\(([0-9a-f]+)\)"
    r"|build\s+(\d+)\s*,\s*commit\s+([0-9a-f]+)")


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
    if not m:
        return None
    build, commit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return {"build": int(build), "commit": commit}


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
    ap = argresolve.ParserThatNamesUnknownFlags(
        prog="senbonzakura quantise",
        # See the note in convert.build_parser: `--out` prefix-matches `--output-tensor-type` here,
        # so an abbreviation turns a plausible typo into a complaint about an unrelated flag.
        allow_abbrev=False,
        description="Quantise a model with the pinned llama-quantize, then verify what was "
                    "written. A transformers checkpoint is converted to GGUF on the way in, so "
                    "edited weights reach something llama.cpp will serve in one command.")
    ap.add_argument("source",
                    help="an f16 or f32 GGUF to quantise, or a transformers checkpoint "
                         "directory, which is converted to GGUF first with the pinned converter")
    ap.add_argument("out", nargs="?", default=None,
                    help="output path (default: the source with its quant name substituted)")
    # `metavar` so the usage line reads `[--type TYPE]`. Spelling all sixteen choices there made it
    # 136 columns wide, which wraps into four lines of quantisation names on any real terminal and
    # buries the two positional arguments a reader is actually looking for. The choices are still
    # enforced, and listed in the help below where there is room for them.
    # `default=None` rather than the literal, so `--like` can supply the base recipe from the
    # reference's own header when the user has not named one. `resolved_type` turns None into
    # DEFAULT_TYPE, and nothing downstream of it ever sees None.
    ap.add_argument("--type", default=None, choices=QUANT_TYPES, metavar="TYPE",
                    help=f"target quantisation (default: {DEFAULT_TYPE}, or the reference's own "
                         f"type when --like is given). One of: " + ", ".join(QUANT_TYPES))
    ap.add_argument("--like", default=None, metavar="REFERENCE.gguf",
                    help="copy the per-tensor precision schedule out of REFERENCE and replay it "
                         "onto this run. A GGUF records a type per tensor, so the schedule of a "
                         "published build is readable out of the file. Its importance matrix is "
                         "NOT, so the output is that schedule with your own matrix or none, and is "
                         "not the reference. The output is never named after the reference's label")
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
    ap.add_argument("--output-tensor-type", dest="output_tensor_type", default=None,
                    choices=gguf_io.OVERRIDE_TYPES, metavar="TYPE",
                    help="keep the output head at this precision instead of whatever the recipe "
                         "chose. Quantisation damage shows there first, so Q8_0 or F16 costs little "
                         "size and buys back most of it. Refused on a model with tied embeddings, "
                         "which has no separate head")
    ap.add_argument("--token-embedding-type", dest="token_embedding_type", default=None,
                    choices=gguf_io.OVERRIDE_TYPES, metavar="TYPE",
                    help="the same for the token embedding table")
    ap.add_argument("--tensor-type", dest="tensor_type", action="append", default=[],
                    metavar="NAME=TYPE",
                    help="pin any tensor whose name matches NAME to TYPE. Repeatable. NAME is "
                         "matched by llama-quantize as a pattern, so `attn_v=Q6_K` reaches every "
                         "layer's value projection")
    ap.add_argument("--verbose", action="store_true",
                    help="show every line the quantiser and the vendored converter print. Both are "
                         "summarised by default, one line per tensor being hundreds of lines on a "
                         "real model; this is the flag for watching a run that is behaving oddly")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output")
    ap.add_argument("--keep-source", action="store_true",
                    help="do not offer to remove the source afterwards (it never removes it "
                         "without this being absent AND --prune-source given)")
    ap.add_argument("--prune-source", action="store_true",
                    help="delete the source once the output has been verified. The f16 halfway "
                         "file is usually the largest thing on the disk and is reproducible")
    return argresolve.explain_that_the_output_is_positional(ap, takes="<source>")


#: The only suffix that is an EXTENSION here. Anything else after a dot is part of the name.
#:
#: `Path.suffix` answers "what follows the last dot", which is not the same question. It calls
#: `.7b` the extension of `stock-1.7b`, and a model name carrying a size or a version is the
#: normal case rather than a corner: `stock-1.7b` became `stock-1-Q3_K_L.7b` and `v0.3.0-model`
#: became `v0.3-Q3_K_L.0-model`. Both are files llama.cpp tooling will not recognise, produced
#: silently. Found on real hardware 2026-09-05.
GGUF_SUFFIX = ".gguf"


def default_output(source, quant):
    """`model-f16.gguf` plus Q4_K_M becomes `model-Q4_K_M.gguf`.

    Named so that `gguf_io.verify` can check the file against its own name afterwards, which is
    the check that catches a Q8_0 sitting under a Q4_K_M name.
    """
    p = Path(source)
    has_ext = p.name.lower().endswith(GGUF_SUFFIX)
    stem = p.name[: -len(GGUF_SUFFIX)] if has_ext else p.name
    claimed = gguf_io.claimed_quant(stem)
    stem = (stem.replace(claimed, quant).replace(claimed.lower(), quant) if claimed
            else f"{stem}-{quant}")
    # Always ends in .gguf, including when the source did not. A quantised file with no extension
    # is a file the rest of the ecosystem declines to open.
    return p.with_name(f"{stem}{GGUF_SUFFIX}")


#: The tensors the two named overrides act on. llama-quantize hard-codes these names, so a model
#: that spells them differently, or does not have them, cannot be served by those flags.
OUTPUT_TENSOR = "output.weight"
EMBED_TENSOR = "token_embd.weight"


def parse_tensor_type(spec):
    """`attn_v=Q6_K` becomes ("attn_v", "Q6_K"), or a readable refusal.

    Validated here rather than left to llama-quantize, which reports a bad type by listing every
    type it knows and exiting, after the operator has waited for the job to start.
    """
    name, sep, kind = spec.partition("=")
    if not sep or not name.strip():
        raise SystemExit(
            f"--tensor-type wants NAME=TYPE and got {spec!r}. The name is matched against tensor "
            f"names, so `attn_v=Q6_K` reaches every layer's value projection.")
    kind = kind.strip().upper()
    if kind not in gguf_io.OVERRIDE_TYPES:
        raise SystemExit(
            f"--tensor-type {spec!r} names the type {kind!r}, which is not one this tool will "
            f"pass on. Choose from: {', '.join(gguf_io.OVERRIDE_TYPES)}.")
    return name.strip(), kind


#: No machine has this many cores, and llama-quantize does not treat the number as a request it
#: can decline. Measured 2026-09-17 on a 270 MB model: the default finished in 1s, `--threads
#: 99999` was still on tensor 84 of 272 after 200s with 1.69 GB resident, in ONE thread. It
#: serialises and allocates rather than refusing, so the ceiling has to be ours.
_MAX_THREADS = 1024


def _preflight_arguments(a, log=print):
    """Refuse or flag an argument that is wrong on its face, before the header line is printed.

    FOUND BY ADVERSARIAL USER TESTING, 2026-09-17, from an installed wheel with no source:

      --threads -5      quantised, verified, and reported DONE. The vendored binary clamps, so
                        the output was fine and the operator's number was discarded silently.
      --threads 99999   turned a one-second job into an open-ended hang with no heartbeat.
      --imatrix FILE    a missing file was refused only after the header, the binary lookup and
                        the quantiser identity read had all been done and announced.
      --tensor-type X   a malformed pair was refused after the run had announced itself, so the
                        tool said what it was about to do and then declined to do it.

    All four are decidable from the command line, so they belong ahead of the announcement.
    """
    if a.threads < 0:
        raise SystemExit(
            f"--threads is {a.threads}, and a worker count cannot be negative. Pass 0 to let "
            f"llama-quantize choose, or a positive count to pin it.")
    if a.threads > _MAX_THREADS:
        raise SystemExit(
            f"--threads is {a.threads}, which is past the {_MAX_THREADS} ceiling this refuses at. "
            f"llama-quantize does not decline a number it cannot use: it serialises and allocates "
            f"instead, so a mistyped count reads as a hang rather than as an error. Pass 0 to let "
            f"it choose.")
    cores = os.cpu_count()
    if cores and a.threads > cores:
        log(f"  WARNING: --threads {a.threads} is above the {cores} cores this machine reports. "
            f"Past the core count the extra workers cost memory and contention rather than speed.")
    if a.imatrix and not Path(a.imatrix).is_file():
        raise SystemExit(
            f"no importance matrix at {a.imatrix}. Build one with `senbonzakura imatrix`.")
    # Same argument as the imatrix above, and the same shape: decidable from the command line, so it
    # belongs in front of the announcement rather than after the conversion.
    if a.like and not Path(a.like).is_file():
        raise SystemExit(say.refusal_text(
            "there is no reference GGUF at the path given to --like.",
            f"Nothing is at {a.like}. --like reads that file's per-tensor schedule and replays it, "
            f"so the run cannot start without it.",
            "Point --like at a finished GGUF whose precision schedule you want copied."))
    # THE TWO FLAGS CONTRADICT EACH OTHER AND BOTH WERE ACCEPTED, 2026-10-01. `--keep-source` says
    # do not remove the source and `--prune-source` says delete it, so giving both asks for two
    # opposite things about somebody's largest file. The code resolved it by precedence rather than
    # by refusing, which means the answer depended on reading this function. Decidable from the
    # command line, so it belongs up here with the other four.
    if a.keep_source and a.prune_source:
        raise SystemExit(say.refusal_text(
            "--keep-source and --prune-source ask for opposite things, and this run would have to "
            "choose for you.",
            "--prune-source deletes the source once the output is verified. --keep-source says "
            "not to. The source is usually the largest file on the disk, so which one wins is not "
            "a detail to settle by precedence.",
            "Pass whichever you meant, and only that one."))
    for s in a.tensor_type:
        parse_tensor_type(s)


#: llama-quantize's own count of tensors the recipe could not be applied to. It reports this once,
#: in the middle of several hundred per-tensor lines, and then exits 0.
_FALLBACK = re.compile(r"WARNING:\s*(\d+)\s+of\s+(\d+)\s+tensor\(s\)\s+required fallback")


def _run_quantiser(argv_q, *, verbose=False, log=print):
    """Run llama-quantize, pass its output through, and keep the one line that matters.

    FOUND BY ADVERSARIAL USER TESTING, 2026-09-17. The binary printed

        WARNING: 180 of 272 tensor(s) required fallback quantization

    and this tool's own summary, four lines later, printed `verified: Q4_K_M` and nothing else.
    Two thirds of the tensors were not at the requested precision and the line written to be read
    said the opposite. The warning was never lost, only buried: the subprocess wrote straight to
    the terminal, so nothing here ever saw it and nothing could carry it into the summary.

    The output is streamed rather than captured because a large quantisation is long and its
    per-tensor progress is the only sign of life it gives.
    """
    hit = None
    tail = []
    proc = subprocess.Popen(argv_q, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1)
    with proc:
        matches, tail = vendored.relay(proc.stdout, verbose=verbose, log=log,
                                       watch=(_FALLBACK,))
    for _pattern, found in matches:
        hit = (int(found.group(1)), int(found.group(2)))
    return proc, hit, tail


def _preflight_overrides(src, a, log):
    """Refuse an override that cannot possibly land, before the hours rather than after them.

    The expensive case is a model with TIED embeddings. It has no `output.weight`, so
    `--output-tensor-type` is accepted by llama-quantize, does nothing at all, and leaves a file
    whose `general.file_type` is exactly what it would have been anyway. Nothing downstream can
    tell that apart from success, which is how the operator ends up publishing a model they
    believe has a high-precision head.

    Returns the pairs from --tensor-type, parsed.
    """
    pairs = [parse_tensor_type(s) for s in a.tensor_type]
    if not (a.output_tensor_type or a.token_embedding_type or pairs):
        return pairs

    try:
        names = [t["name"] for t in gguf_io.read_tensor_info(src)]
    except gguf_io.GGUFError as e:
        # The overrides cannot be checked, and saying so is better than either silently skipping
        # the check or refusing a file that quantises perfectly well.
        log(f"  WARNING: the tensor list could not be read ({e}), so the overrides below are "
            f"passed on unchecked and their effect is confirmed only after the run.")
        return pairs

    for flag, tensor, wanted in (("--output-tensor-type", OUTPUT_TENSOR, a.output_tensor_type),
                                 ("--token-embedding-type", EMBED_TENSOR, a.token_embedding_type)):
        if wanted and tensor not in names:
            raise SystemExit(
                f"{flag} {wanted} was asked for and {Path(src).name} has no tensor called "
                f"{tensor}. llama-quantize would accept the flag, do nothing, and produce a file "
                f"indistinguishable from one where it was never passed. A model with tied "
                f"embeddings has no separate output head, which is the usual reason.")
    for name, kind in pairs:
        if not any(name in n for n in names):
            raise SystemExit(
                f"--tensor-type {name}={kind} matches no tensor in {Path(src).name}. It is "
                f"matched against tensor names, and nothing here contains {name!r}, so the flag "
                f"would quietly do nothing.")
    return pairs


def _verify_overrides(out, a, pairs, log):
    """Read the finished file and confirm every override actually landed.

    The receipt. An exit code says a process finished; this says the tensor on disk carries the
    precision that was asked for, and those two came apart on a 987 MB fragment of a 5.16 GB file
    that loaded and served.
    """
    wanted = {}
    if a.output_tensor_type:
        wanted[OUTPUT_TENSOR] = a.output_tensor_type
    if a.token_embedding_type:
        wanted[EMBED_TENSOR] = a.token_embedding_type
    if not wanted and not pairs:
        return None

    info = gguf_io.read_tensor_info(out)
    got = {t["name"]: t["type"] for t in info}
    wrong = []
    for name, kind in wanted.items():
        if got.get(name) != kind:
            wrong.append(f"{name} was asked for as {kind} and is {got.get(name) or 'absent'}")
        else:
            log(f"  verified: {name} is {kind}")
    for name, kind in pairs:
        matched = [n for n in got if name in n]
        off = [n for n in matched if got[n] != kind]
        if off:
            wrong.append(f"{len(off)} of {len(matched)} tensors matching {name!r} are not {kind} "
                         f"(for example {off[0]} is {got[off[0]]})")
        else:
            log(f"  verified: all {len(matched)} tensors matching {name!r} are {kind}")
    if wrong:
        raise SystemExit(
            "llama-quantize reported success and the per-tensor precision asked for is not what "
            "is in the file:\n  " + "\n  ".join(wrong) + f"\nThe file is left at {out} for "
            f"inspection. Nothing else would have caught this: an override does not change "
            f"general.file_type, so the file verifies against its own name either way.")
    return {"requested": {**wanted, **dict(pairs)},
            "census": gguf_io.type_census(out)}


# ── --like: replaying one file's per-tensor schedule, and saying what is still missing ──────
#
# WHAT IS COPYABLE AND WHAT IS NOT, because the whole honesty of this feature is the difference.
#
# A GGUF stores a ggml type for EVERY tensor in its own header (`gguf_io.read_tensor_info` reads
# it without touching a byte of tensor data), and `llama-quantize --tensor-type NAME=TYPE` applies
# an arbitrary per-tensor type. So the per-tensor SCHEDULE of any published build, Unsloth Dynamic
# included, is recoverable out of the artefact and replayable. That is a fact about the format, not
# a reverse-engineering guess.
#
# The IMPORTANCE MATRIX is not recoverable. It is a calibration pass over a corpus, it leaves no
# record in the output file, and Unsloth does not publish the corpus they use. So a `--like` output
# carries the reference's schedule and OUR importance matrix, or none, and the two files are
# different builds of the same recipe shape.
#
# That distinction decides whether a figure measured on the reference transfers to the file we
# would actually ship, which is the entire reason anybody asked for this. It is therefore stated in
# the terminal on every run, recorded in the sidecar on every run, and kept out of the filename on
# every run, rather than left in this comment where only a maintainer meets it.

#: Stamped into the output name in place of the reference's own label. A name like `UD-Q3_K_XL` is a
#: claim about who built the file and what they calibrated it on, and this is not that file. The
#: filename is what travels furthest and is read by the most people, so it is the last place the
#: claim may appear. The base recipe stays in the name so `gguf_io.claimed_quant` and `verify` still
#: have something to check the file against, which a bare `-like` suffix would have taken away.
SCHEDULE_MARKER = "copied-schedule"

#: How much the two tensor-name sets must coincide before a schedule is replayed, as a Jaccard
#: ratio over the two sets rather than as a containment check.
#:
#: CONTAINMENT WAS TRIED FIRST AND IS WRONG. The dangerous pair is one model's schedule replayed
#: onto a smaller model of the SAME architecture: a 4B's tensor names are a strict subset of a 27B's
#: because the difference is the block count, so "every name I have is in yours" scores a perfect
#: 1.0 on exactly the mismatch that matters most. Jaccard counts the names only ONE side has, which
#: is where a block-count difference lives, and scores that pair around 0.58.
#:
#: 0.95 rather than 1.0 because two honest builds of one model legitimately differ by a tensor or
#: two: `rope_freqs.weight` is written by some converter versions and not others, and a tied-
#: embedding export has no `output.weight`. Refusing those would refuse the normal case.
MIN_NAME_AGREEMENT = 0.95

#: How many differing tensors to name in the terminal before summarising the rest. A whole-file
#: schedule is hundreds of tensors and a wall of them is a wall nobody reads; the full list goes in
#: the sidecar, which is where a reader who wants all of it can get all of it.
_SCHEDULE_DIFFS_SHOWN = 8


def _named_type(tensor):
    """A tensor's type as a string, NEVER None, matching `gguf_io.type_census`'s own convention.

    `read_tensor_info` reports `None` for a type id it does not know, which is the right answer
    there: naming an unknown number would make a wrong answer look like an answer. It is the wrong
    SHAPE here, because a None flows into a `', '.join(...)` of the types a bucket holds and crashes
    the explanation of why those tensors were skipped. A reference carrying a tensor type newer than
    the pinned `gguf` package is a file this will meet, not a hypothetical.

    An unknown type is not in `OVERRIDE_TYPES`, so it lands in `not-passable` and is reported, which
    is the honest outcome: a type this cannot name is a type it cannot replay.
    """
    return tensor["type"] or f"unknown type {tensor['type_id']}"


def read_schedule(reference):
    """The per-tensor type of a reference GGUF, as {name: type}. The whole of what --like copies.

    Reads the header only, so pointing this at a 20 GB file costs a few hundred kilobytes.
    """
    try:
        info = gguf_io.read_tensor_info(reference)
    except gguf_io.GGUFError as e:
        raise SystemExit(say.refusal_text(
            "the file passed to --like could not be read as a GGUF.",
            str(e),
            "--like wants a finished GGUF whose per-tensor schedule you want copied. Only its "
            "header is read, so a truncated download fails here rather than halfway through a "
            "quantisation.")) from e
    return {t["name"]: _named_type(t) for t in info}


def schedule_disagreement(reference, source, *, reference_head, source_head,
                          reference_names, source_names):
    """Why this reference's schedule must not be replayed onto this source, or None if it may be.

    A schedule applied to the wrong model is worse than no feature at all: llama-quantize matches
    `--tensor-type` by pattern, so a 27B's schedule aimed at a 4B does not fail, it lands on the
    blocks that happen to share a name and leaves the rest on the base recipe. The result is a file
    that is neither the reference's schedule nor the base recipe, produced silently.

    Two tests, and both have to pass. The architecture, because one architecture's tensor names mean
    different things from another's even where they coincide. Then the name sets, because the same
    architecture at a different size shares every name it has.
    """
    ref_arch, src_arch = reference_head.get("architecture"), source_head.get("architecture")
    if not ref_arch or not src_arch:
        unnamed = Path(reference).name if not ref_arch else Path(source).name
        return say.refusal_text(
            "--like cannot check the reference against the source, because one of them does not "
            "say what architecture it is.",
            f"{unnamed} records no readable general.architecture, so there is no way to tell "
            f"whether its tensor names mean the same thing as the other file's. Replaying a "
            f"schedule on that basis would be a guess wearing a receipt.",
            "Convert the file again with the pinned converter, which writes the key, or pick a "
            "reference that carries it.")
    if ref_arch != src_arch:
        return say.refusal_text(
            "--like was given a reference from a different architecture.",
            f"{Path(reference).name} is {ref_arch} and {Path(source).name} is {src_arch}. Tensor "
            f"names that look alike across two architectures do not hold the same thing, and "
            f"llama-quantize would apply the overlapping ones rather than refuse.",
            "Pass a reference built from the same architecture as the source.")

    shared = reference_names & source_names
    union = reference_names | source_names
    agreement = len(shared) / len(union) if union else 0.0
    if agreement < MIN_NAME_AGREEMENT:
        only_ref = sorted(reference_names - source_names)
        only_src = sorted(source_names - reference_names)
        return say.refusal_text(
            "--like was given a reference whose tensors do not correspond to the source's.",
            f"{len(shared)} names are shared out of {len(union)}, which is "
            f"{agreement * 100:.0f}% agreement, below the {MIN_NAME_AGREEMENT * 100:.0f}% this "
            f"will replay a schedule on. Both files say they are {ref_arch}, so the usual cause is "
            f"two different SIZES of the same architecture: the smaller one's names are a subset of "
            f"the larger one's, and the schedule would land on the blocks they share and leave the "
            f"rest alone.",
            f"{len(only_ref)} tensors are only in the reference"
            + (f" (for example {only_ref[0]})" if only_ref else "")
            + f" and {len(only_src)} only in the source"
            + (f" (for example {only_src[0]})" if only_src else "") + ".",
            "Pass a reference built from the same model as the source.")
    return None


def plan_schedule(schedule, source_names, *, patterns=(), exactly=()):
    """Which of the reference's tensors this run can actually pin, and what it has to leave behind.

    Returns (pairs, skipped), where `pairs` is [(name, type)] for llama-quantize and `skipped` is
    {reason: [names]} for the record and for the terminal. Nothing is dropped quietly: every tensor
    the schedule names and this run will not pin appears in one of those buckets.

    Three reasons a tensor is left out, and they are different facts rather than one:

      `absent`         the source has no such tensor, so the pattern would match nothing. The
                       commonest honest case: a reference carrying `rope_freqs.weight` replayed onto
                       a source converted without it.
      `not-passable`   the type is real and `llama-quantize` will not take it as a per-tensor
                       override. A UD-IQ2_M reference holds IQ2_S and IQ2_XXS tensors, and
                       `gguf_io.OVERRIDE_TYPES` deliberately does not include them, so those tensors
                       take the base recipe and the output is NOT that reference's schedule.
      `spoken-for`     the user named this tensor themselves, with --tensor-type,
                       --output-tensor-type or --token-embedding-type.

    The last of those exists so the schedule and the user's own flags never both describe one
    tensor. llama-quantize's precedence between two matching patterns is its business and not
    something this should depend on, so the conflict is removed rather than resolved: the explicit
    flag wins because it is the more specific statement of intent, and the schedule entry is dropped.

    `patterns` AND `exactly` ARE SEPARATE, and collapsing them into one substring test was a real
    defect in the first version of this function. `--tensor-type NAME` is matched by llama-quantize
    as a pattern, so `attn_v` legitimately claims every layer's value projection. The other two
    flags act on ONE hard-coded tensor each, and `output.weight` tested as a substring also matches
    `blk.0.attn_output.weight`: `--output-tensor-type F16` would then have dropped every attention
    output projection out of the copied schedule and left it on the base recipe, silently, with the
    `spoken-for` bucket reporting the loss as the user's own choice.
    """
    pairs, skipped = [], {}
    exactly = set(exactly)

    def leave(reason, name):
        skipped.setdefault(reason, []).append(name)

    for name in sorted(schedule):
        kind = schedule[name]
        if name not in source_names:
            leave("absent", name)
        elif name in exactly or any(pattern in name for pattern in patterns):
            leave("spoken-for", name)
        elif kind not in gguf_io.OVERRIDE_TYPES:
            leave("not-passable", name)
        else:
            pairs.append((name, kind))
    return pairs, skipped


def describe_skipped(skipped, schedule, log):
    """Say out loud which of the reference's tensors this run is not pinning, and why.

    Each bucket is a different statement about how far the output is from the reference, and
    `not-passable` is the one that changes the answer: those tensors take the base recipe, so the
    file is the reference's schedule only where the schedule could be expressed.
    """
    reasons = {
        "absent": "are not in the source at all, so the schedule has nothing to pin there",
        "spoken-for": "were named by your own flags, which win over the copied schedule",
        "not-passable": "carry a type llama-quantize will not accept as a per-tensor override, so "
                        "they take the base recipe instead and the output is NOT this reference's "
                        "schedule on those tensors",
    }
    for reason, names in sorted(skipped.items()):
        kinds = sorted({schedule[n] for n in names})
        say.say(f"{len(names)} of {len(schedule)} tensors in the reference {reasons[reason]} "
                f"({', '.join(kinds)}; for example {names[0]}).", indent="  ", log=log)


def verify_schedule(out, schedule, log):
    """Read the finished file back and compare every tensor against the reference's schedule.

    THE ONLY THING STANDING BETWEEN THIS FEATURE AND A CONFIDENT LIE. llama-quantize accepts a flag,
    does nothing with it, and produces a file anyway: that is why `_verify_overrides` above exists
    and why `--output-tensor-type` is refused up front on a tied-embedding model. A whole-file
    schedule is the same exposure several hundred times over, and the exit code says nothing about
    any of it.

    WHY A DIFFERENCE IS REPORTED RATHER THAN REFUSED, unlike `_verify_overrides`. A hand-picked
    override is a tensor the operator chose, so not getting it is a failed request. A whole-file
    schedule includes tensors the recipe genuinely cannot honour, block dimensions the quantiser's
    block size does not divide being the common one, and refusing those would refuse every honest
    run on a real model. So the COUNT is printed and the full difference is recorded, because the
    count is the thing that decides whether a figure transfers, and a reader can act on a number
    they can see.

    Zero matches IS refused, because that is not a partial honouring: it means the overrides were
    not applied at all.
    """
    # `_named_type` on BOTH sides, so an unknown type id is compared as the same string here as it is
    # in the schedule. Comparing a None against an "unknown type 99" would report a difference
    # between a tensor and itself.
    got = {t["name"]: _named_type(t) for t in gguf_io.read_tensor_info(out)}
    honoured, differed = {}, {}
    for name, kind in sorted(schedule.items()):
        if name not in got:
            continue
        (honoured if got[name] == kind else differed)[name] = {"reference": kind, "output": got[name]}

    checked = len(honoured) + len(differed)
    if checked and not honoured:
        raise SystemExit(say.refusal_text(
            "not one tensor in the output carries the type the reference has for it.",
            f"All {checked} tensors compared against {Path(out).name} came out at a different "
            f"precision from the reference's. llama-quantize reported success, so the overrides "
            f"were accepted and then had no effect at all rather than partially landing.",
            f"The file is left at {out} for inspection. Re-run with --verbose to see what the "
            f"quantiser said about each tensor."))

    log(f"  schedule: {len(honoured)} of {checked} tensors carry the reference's type, "
        f"{len(differed)} differ.")
    for name in list(differed)[:_SCHEDULE_DIFFS_SHOWN]:
        d = differed[name]
        log(f"    {name}: the reference has {d['reference']} and this file has {d['output']}")
    if len(differed) > _SCHEDULE_DIFFS_SHOWN:
        log(f"    and {len(differed) - _SCHEDULE_DIFFS_SHOWN} more, all of them in the sidecar.")
    return {"honoured": len(honoured), "compared": checked, "differed": differed}


def schedule_output_name(source, quant, reference):
    """Where a `--like` run writes when the user has not said: the base type, plus the marker.

    `model-BF16.gguf` with `--type Q3_K_M --like X-UD-Q3_K_XL.gguf` becomes
    `model-Q3_K_M-copied-schedule.gguf`. The reference's own label never appears, for the reason
    `SCHEDULE_MARKER` gives; which reference it was is in the sidecar, with its hash.
    """
    plain = default_output(source, quant)
    # `reference` is taken so the signature says this name depends on there being one, and so a
    # future scheme can use it without every caller changing. Deliberately unused for now.
    del reference
    return plain.with_name(f"{plain.name[:-len(GGUF_SUFFIX)]}-{SCHEDULE_MARKER}{GGUF_SUFFIX}")


def _preflight_schedule(a, source_head, log):
    """Read the reference, refuse a mismatched pair, and work out what this run can pin.

    Returns (schedule, pairs, skipped), all three empty when `--like` was not given, so the caller
    has no branch to forget. Everything expensive about `--like` happens here, before the quantiser
    is started: a mismatched pair is the one failure that would otherwise produce a plausible file.
    """
    if not a.like:
        return {}, [], {}

    schedule = read_schedule(a.like)
    try:
        reference_head = gguf_io.read_header(a.like)
        source_names = {t["name"] for t in gguf_io.read_tensor_info(a.source)}
    except gguf_io.GGUFError as e:
        raise SystemExit(
            f"the tensor lists needed to check --like against the source could not be read: {e}"
        ) from e

    refusal = schedule_disagreement(
        a.like, a.source, reference_head=reference_head, source_head=source_head,
        reference_names=set(schedule), source_names=source_names)
    if refusal:
        raise SystemExit(refusal)

    # Every tensor the user named themselves, so the schedule never describes a tensor one of their
    # own flags already describes. `--tensor-type` is a pattern and the other two name exactly one
    # tensor each; see `plan_schedule` for why that distinction is load-bearing.
    patterns = [name for name, _ in (parse_tensor_type(s) for s in a.tensor_type)]
    exactly = ([OUTPUT_TENSOR] if a.output_tensor_type else []) \
        + ([EMBED_TENSOR] if a.token_embedding_type else [])

    pairs, skipped = plan_schedule(schedule, source_names, patterns=patterns, exactly=exactly)
    say.say(f"--like {Path(a.like).name}: {len(schedule)} tensors in the reference, "
            f"{len(pairs)} of them pinned onto this run ({a.type} underneath).",
            indent="  ", log=log)
    describe_skipped(skipped, schedule, log)
    if not pairs:
        raise SystemExit(say.refusal_text(
            "nothing in the reference's schedule can be replayed onto this source.",
            f"All {len(schedule)} tensors in {Path(a.like).name} were skipped for the reasons "
            f"above, so --like would have no effect and the output would be a plain {a.type} under "
            f"a name saying a schedule had been copied.",
            "Drop --like and ask for the base recipe directly, or pick a reference whose types "
            "this tool can pass on."))
    return schedule, pairs, skipped


def schedule_record(a, schedule, pairs, skipped, verified, log=print):
    """The `--like` half of the provenance sidecar, and the honesty of the feature lives here.

    WRITTEN ON EVERY RUN, `null` when `--like` was not used. A field that appears only when
    somebody remembered to ask for it is a field nothing downstream can rely on, which is the same
    argument the source and output hashes above are always recorded under.

    `importance_matrix_copied` is always `false` and is recorded anyway, which looks redundant and
    is the point. The reference's importance matrix is not in the reference: it is a calibration
    pass over a corpus that leaves no trace in the file, and Unsloth does not publish theirs. So the
    one question a reader of this record will have, "is this the same build as the reference", has a
    fixed answer, and a fixed answer stated explicitly is worth more than an absent field a reader
    has to infer from.
    """
    if not a.like:
        return None
    head = gguf_io.read_header(a.like)
    return {
        "path": str(a.like),
        "name": Path(a.like).name,
        "sha256": digest_for_the_record(a.like, what="--like reference", log=log),
        "architecture": head.get("architecture"),
        "file_type": head.get("file_type"),
        # The schedule that was READ, and the subset that was actually PINNED. Two different facts:
        # the first is what the reference is, the second is what this run could express.
        "schedule": schedule,
        "pinned": dict(pairs),
        "skipped": skipped,
        "verified": verified,
        # THE GAP, stated rather than left to be worked out from what is absent.
        "importance_matrix_copied": False,
        "bit_identical_to_reference": False,
        "difference_from_reference":
            "the per-tensor type schedule was copied out of the reference's header, which is where "
            "a GGUF records it. The reference's importance matrix was NOT copied, because a "
            "calibration pass leaves no trace in the file it produced and the corpus behind a "
            "published build is generally not published with it. This file therefore carries the "
            "reference's schedule and this run's own importance matrix, or none, and is not "
            "bit-identical to the reference. A figure measured on the reference does not transfer "
            "to this file on the strength of the matching schedule alone.",
    }


def say_what_is_still_different(a, log):
    """The sentence the operator actually needs, in the terminal, on every `--like` run.

    They are weighing a checkpoint against a number benched on a `UD-` build. The schedule matching
    is what they asked for; the importance matrix not matching is what decides whether the number
    carries over, and that half is invisible in the file, the filename and the census. So it is said
    here, where somebody reading the run meets it, rather than only in the sidecar.
    """
    if not a.like:
        return
    mine = (f"the importance matrix at {Path(a.imatrix).name}" if a.imatrix
            else "NO importance matrix")
    # Through `say` rather than one `log` call, because this is the longest thing a `--like` run
    # prints and an unwrapped paragraph is a wall, and a wall is what a reader skips. That is the
    # finding `say` was written for, and skipping it here would be skipping it on the one message
    # whose whole purpose is to be read.
    say.say(f"NOT COPIED: {Path(a.like).name}'s importance matrix. A calibration pass leaves no "
            f"trace in the file it produced, so there is nothing in the reference to read it out "
            f"of, and the corpus behind a published build is generally not published with it. This "
            f"output is that reference's SCHEDULE built with {mine}, which is a different build of "
            f"the same recipe shape rather than the same file. A figure measured on the reference "
            f"does not transfer here on the matching schedule alone. Recorded in the sidecar "
            f"beside it.", indent="  ", log=log)


def resolved_type(a):
    """`--type`, or the base recipe this run should use when the user did not name one.

    With `--like` that is the REFERENCE'S OWN file-level type, because the point of the flag is to
    land as close to the reference as the format allows and the base recipe decides every tensor the
    schedule could not pin. A reference whose type this tool cannot produce is a refusal rather than
    a silent fall back to Q4_K_M: quietly overlaying a Q3_K schedule on a Q4_K_M base would produce
    a file nobody asked for under a name that looked deliberate.
    """
    if a.type:
        return a.type
    if not getattr(a, "like", None):
        return DEFAULT_TYPE
    try:
        head = gguf_io.read_header(a.like)
    except gguf_io.GGUFError as e:
        raise SystemExit(say.refusal_text(
            "the file passed to --like could not be read as a GGUF.", str(e),
            "--like reads only the header, so this fails before any work rather than after it."
        )) from e
    claimed = head.get("file_type")
    if claimed in QUANT_TYPES:
        return claimed
    return _refuse_an_unproducible_base(a.like, claimed)


def _refuse_an_unproducible_base(reference, claimed):
    """`--like <something whose own file type we cannot make>` needs an explicit --type.

    `UD-IQ2_M` declares IQ2_M, which `gguf_io` can read and `QUANT_TYPES` cannot produce. There is
    no honest default here: the base recipe decides every tensor the schedule does not pin, so
    choosing one for the operator would decide part of the output's quality for them.
    """
    raise SystemExit(say.refusal_text(
        "--like needs a --type here, because the reference's own type is not one this tool can "
        "produce.",
        f"{Path(reference).name} declares {claimed or 'no readable general.file_type'}, and "
        f"`quantise` can make: {', '.join(QUANT_TYPES)}. The base recipe decides every tensor the "
        f"copied schedule cannot pin, so picking one for you would decide part of the output "
        f"without saying so.",
        "Name the base recipe yourself, for example:\n"
        f"  senbonzakura quantise <source> --like {reference} --type Q4_K_M"))


def source_conversion_record(source):
    """The conversion receipt beside the input, read as a value so it can outlive the input.

    A two-step run converts a checkpoint to f16 and then quantises it, and `--prune-source`
    deletes the f16 once the output verifies. That left the receipt describing a file that no
    longer exists, and left the file the user keeps with no record of the checkpoint it came
    from: the provenance was broken in both directions at once by a step meant to save disk.

    So the chain is copied forward before anything is deleted. A source this tool did not
    convert has no receipt, which is reported as None rather than as an empty record, because
    "no conversion step" and "a conversion that recorded nothing" are different facts.
    """
    path = Path(str(source) + convert_record_suffix())
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def convert_record_suffix():
    """Read from `convert` rather than restated, so the two cannot drift apart."""
    from .convert import RECORD_SUFFIX
    return RECORD_SUFFIX


def preflight_output(source, out, *, force):
    """The checks that need only the two paths and the flag, so they can run before a conversion.

    SPLIT OUT OF `preflight` ON 2026-09-25, and the two-step path is the reason. A checkpoint
    reaches `preflight` only after `_quantise_a_checkpoint` has converted it: tens of minutes and
    tens of gigabytes through a temporary directory, and only then "the output exists, pass
    --force". The commit that moved `_preflight_arguments` ahead of the conversion made exactly
    this argument in its own comment and stopped one function short of the refusal a user is most
    likely to meet.

    The source-side checks stay where they are because they genuinely need the converted file: a
    header cannot be read before the file exists. These do not need it, and anything that does not
    need the expensive step belongs in front of it.
    """
    # BEFORE the exists check, because it is the more specific and the more destructive of the
    # two. Ordered the other way, the only case that reaches it is `--force` on the same path,
    # where "it exists, pass --force" has already been printed and taken.
    if Path(out).resolve() == Path(source).resolve():
        raise SystemExit(
            f"the output path is the source path ({source}), so the run would read a file it is "
            f"overwriting. Name a different --out.")

    if Path(out).exists() and not force:
        raise SystemExit(f"{out} exists. Pass --force to overwrite it, or choose another path.")


def refuse_without_room(out, need, *, describing):
    """Refuse when the volume the output goes to cannot hold `need` bytes.

    One place, because the two-step path asks the same question about a total it works out
    differently: a checkpoint run writes an intermediate GGUF and THEN the quantised file, and
    sizing only the second of those is how a run converts for half an hour into a volume that was
    never going to hold both.
    """
    free = free_bytes_for(out)
    if free is not None and free < need:
        raise SystemExit(
            f"only {free / 1e9:.1f} GB free where the output goes and {describing} needs about "
            f"{need / 1e9:.1f} GB. Quantising shrinks a model, but not before it has written it, "
            f"so free space or choose an output path on a larger volume.")


def preflight(source, out, quant, *, allow_requantize, force):
    """Everything checkable before a long job starts, because none of it is worth finding halfway.

    Returns the source's header so the caller does not read it twice.
    """
    src = Path(source)
    if not src.is_file():
        raise SystemExit(
            f"no source GGUF at {src}.\n"
            f"  This takes either a GGUF or a transformers checkpoint directory, and {src} is "
            f"neither: nothing is there.\n"
            f"  If you have a checkpoint, pass it straight to this command; it converts first.\n"
            f"  If you want the intermediate GGUF kept, make it yourself:\n"
            f"    senbonzakura convert <model directory> {src}")

    # Both output-side refusals BEFORE the header read, and the same-path one first within them:
    # a run that is about to overwrite its own source should not have read it first.
    preflight_output(src, out, force=force)

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

    refuse_without_room(out, int(src.stat().st_size * SIZE_HEADROOM),
                        describing=f"the output of a {src.stat().st_size / 1e9:.1f} GB source")
    return head


#: What a transformers checkpoint directory looks like from outside. `config.json` alone, because
#: a checkpoint may hold safetensors, shards, a `.bin`, or nothing this tool can read, and the
#: converter is the thing qualified to say which; this only has to decide which command owns it.
def looks_like_a_checkpoint(path):
    p = Path(path)
    return p.is_dir() and (p / "config.json").is_file()


def checkpoint_bytes(directory):
    """How much a checkpoint weighs on disk, for sizing the two files this route writes.

    Symlinks are FOLLOWED rather than skipped: a Hub snapshot directory is a tree of links into
    the blob store, and counting those as nothing would report a 60 GB model as a few kilobytes,
    which is an under-estimate in the one direction a disk check must never be wrong in. A tree
    that cannot be walked to the end yields the partial total rather than raising, and the caller
    reads zero as "cannot say" rather than as "nothing there": this feeds a pre-flight, not a
    correctness claim.
    """
    total = 0
    # One try around the WALK rather than one per entry: `rglob` itself can raise on a directory
    # that disappears underneath it, and a per-entry guard is both slower and blind to that.
    try:
        for path in Path(directory).rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    except OSError:
        # A partial total is still worth more than none: it can only under-estimate, and the
        # caller treats zero as "cannot say" rather than as "nothing there".
        pass
    return total


#: Roughly how many bits per weight each recipe spends, so the two-step route can size the file
#: it has not written yet. Upstream's own published figures, rounded: they are used to decide
#: whether a volume can hold the conversion and the quantised file at once, not to promise a size.
#:
#: A SINGLE PESSIMISTIC FRACTION WAS TRIED FIRST AND REJECTED. Taking Q8_0's share for every
#: recipe refuses a Q4_K_M run on a volume that would have held it comfortably, and a pre-flight
#: that refuses work which would have succeeded is a worse failure than the one it prevents: it
#: teaches people to reach for the flag that turns it off.
_BITS_PER_WEIGHT = {
    "Q2_K": 2.6, "Q3_K_S": 3.4, "Q3_K_M": 3.7, "Q3_K_L": 3.9,
    "Q4_K_S": 4.6, "Q4_K_M": 4.9, "Q4_0": 4.6, "IQ4_XS": 4.3, "IQ4_NL": 4.5,
    "Q5_K_S": 5.5, "Q5_K_M": 5.7, "Q5_0": 5.5, "Q6_K": 6.6, "Q8_0": 8.5,
    "F16": 16.0, "BF16": 16.0,
}

def checkpoint_output_path(a):
    """Where `quantise <checkpoint>` writes, computed in one place because two callers need it.

    BESIDE THE SOURCE, like the GGUF route, and until 2026-09-25 it was not: this read
    `Path(a.source).name`, which drops the directory, so a checkpoint at `/models/X` wrote
    `X-Q4_K_M.gguf` into whatever directory the command happened to be run from. It was logged,
    so it was not silent, but one command that puts a file in two different places depending on
    the shape of its input is a command nobody can script against.
    """
    if a.out:
        return Path(a.out)
    # `resolved_type` rather than `a.type`, because `--type` defaults to None so `--like` can supply
    # it. Called here as well as in `run` so a direct caller of this function gets a real name.
    quant = resolved_type(a)
    source = str(Path(a.source)) + GGUF_SUFFIX
    if a.like:
        return schedule_output_name(source, quant, a.like)
    return default_output(source, quant)


def preflight_a_checkpoint(a, log=print):
    """The output-side refusals for the two-step route, before a byte is converted.

    Both halves are sized here rather than one: this route writes an intermediate GGUF the size of
    the checkpoint and then a quantised file beside it, and a volume that can hold the second but
    not the first refuses halfway through, with the conversion already paid for.

    The intermediate's size is an estimate from the checkpoint's bytes on disk, which is exact for
    a 16-bit checkpoint and generous for a 32-bit one. Where the free space cannot be read, the
    check reports nothing rather than guessing, which is `free_bytes_for`'s own contract.
    """
    out = checkpoint_output_path(a)
    preflight_output(a.source, out, force=a.force)

    weights = checkpoint_bytes(a.source)
    if not weights:
        return out
    # The intermediate is 16-bit, so it weighs what the checkpoint does when the checkpoint is
    # already 16-bit and less when it is 32-bit. Taking the checkpoint's own size for it errs
    # towards asking for more room than the run needs, which is the safe direction here.
    share = _BITS_PER_WEIGHT.get(a.type, 16.0) / 16.0
    need = int(weights * (1 + share) * SIZE_HEADROOM)
    if not a.keep_source:
        # The intermediate is deleted once the quantised file is written, so the PEAK is both
        # files at once and that is what has to fit. Stated rather than left implicit: somebody
        # reading the refusal with `df` in the other window should be able to reconcile it.
        log(f"  this route writes about {weights / 1e9:.1f} GB of intermediate GGUF and then the "
            f"quantised file, so about {need / 1e9:.1f} GB has to be free at once.")
    refuse_without_room(out, need, describing="the conversion and the quantised file together")
    return out


def _quantise_a_checkpoint(a, *, log=print):
    """Convert, then quantise, with the caller's `--out` naming the file they asked for.

    TWO EXPLICIT STEPS RATHER THAN `convert --quantise`, and the reason is a defect found by
    running it. Handing the whole job to `convert` meant the converter's positional named the
    INTERMEDIATE, and the quantised file took a name derived from that: a run given an explicit
    output path wrote 0.73 GB to a directory the user had not named, under a log line saying it
    would be somewhere else. A command that states where a file will be and then puts it
    elsewhere is worse than one that never said.

    So the conversion is asked for on its own, into a temporary GGUF beside the output, and the
    quantisation is this module's ordinary path with the source it was always going to have. The
    converter is still the only thing that converts.
    """
    import tempfile

    from . import convert

    out = checkpoint_output_path(a)
    out.parent.mkdir(parents=True, exist_ok=True)

    log(f"{a.source} is a transformers checkpoint rather than a GGUF, so it is converted first, "
        f"then quantised to {a.type}. Two commands, run for you.")
    log(f"  the quantised model will be at {out}")

    # Beside the OUTPUT rather than beside the checkpoint: the checkpoint may be in a read-only
    # cache, and the output directory is the one the user has just said they can write to.
    tmp_dir = tempfile.mkdtemp(prefix=".senbonzakura-convert-", dir=str(out.parent))
    intermediate = Path(tmp_dir) / (Path(a.source).name + "-bf16.gguf")
    converted = quantised = False
    try:
        # `--verbose` reaches the converter too: somebody who asked to see the work means all
        # of it, and the checkpoint path runs two vendored tools rather than one.
        convert_argv = [str(a.source), str(intermediate)]
        if a.verbose:
            convert_argv.append("--verbose")
        rc = convert.run(convert_argv, log=log)
        if rc != 0:
            return rc
        # A conversion that FAILED leaves something that is not a GGUF, and keeping that would be
        # offering the user a resume point that cannot be resumed from. Only a finished
        # conversion is worth rescuing.
        converted = True
        # EVERY QUANTISATION FLAG IS FORWARDED, not the four somebody remembered.
        #
        # This rebuilt the inner command line by hand and carried `--type`, `--imatrix`,
        # `--force` and `--threads`. So `--output-tensor-type`, `--token-embedding-type`,
        # `--tensor-type` and `--allow-requantize` were accepted on the way in, silently dropped,
        # and the run reported DONE having produced a file the user had not asked for. Worse for
        # `--tensor-type`: its value is validated by `parse_tensor_type` inside
        # `_preflight_arguments`, which this path skips, so a malformed one was accepted and
        # ignored rather than refused.
        #
        # Driven off the parser's own actions rather than a second hand-written list, because a
        # hand-written list is what was wrong: a flag added later would be dropped again and
        # nothing would say so.
        q_argv = [str(intermediate), str(out), "--type", a.type]
        for flag, value in _forwardable_quantiser_flags(a):
            q_argv += [flag] if value is True else [flag, str(value)]
        rc = run(q_argv, log=log)
        quantised = rc == 0
        return rc
    finally:
        # The intermediate is scaffolding when the run SUCCEEDED and is the expensive half of the
        # work when it did not, and this deleted it either way.
        #
        # THE RUN THAT PRODUCED THIS. A 30B checkpoint converts for tens of minutes into about
        # 60 GB, the quantisation's own pre-flight then finds the volume short for the output,
        # and this `finally` deleted the conversion on the way out. The user frees space and
        # starts again from the checkpoint, paying the conversion a second time for a failure
        # that happened AFTER it. A failed step must never destroy a completed one.
        #
        # `--keep-source` still asks for it on a successful run, and either way it moves next to
        # the output where the user can find it rather than staying in a temporary directory
        # named after this function.
        if converted and (a.keep_source or not quantised) and intermediate.is_file():
            kept = out.parent / intermediate.name
            intermediate.replace(kept)
            if quantised:
                log(f"  kept the intermediate GGUF at {kept}")
            else:
                log(f"  the quantisation did not finish, and the conversion it needed is kept at "
                    f"{kept} rather than deleted with it.")
                log("  Resume from there once the reason is dealt with, and the conversion is "
                    "not paid for twice:")
                log(f"    senbonzakura quantise {kept} {out} --type {a.type}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


#: Flags the checkpoint path must hand to the inner quantisation, and the ones it must NOT.
#:
#: `source` and `out` are positional and are replaced. `--type` is passed explicitly. The three
#: source-disposal flags belong to the OUTER run: the checkpoint is the user's source, and the
#: intermediate GGUF is scaffolding this function owns and deletes, so forwarding them would
#: point `--keep-source` or `--prune-source` at the wrong file entirely.
_NOT_FORWARDED = frozenset({"source", "out", "type", "keep_source", "prune_source", "help"})


def _forwardable_quantiser_flags(a):
    """(flag, value) for every quantiser flag the user set, read off the parser itself.

    Read off the parser rather than listed by hand, because the hand-written list is the defect:
    four flags were carried and four were dropped, and a flag added later would have been dropped
    too with nothing to say so. `--tensor-type` is `append`, so it yields once per occurrence.
    """
    for action in build_parser()._actions:   # noqa: SLF001 - argparse exposes no public accessor
        dest = action.dest
        if dest in _NOT_FORWARDED or not action.option_strings:
            continue
        value = getattr(a, dest, None)
        if value in (None, False, 0, [], ""):
            continue
        flag = max(action.option_strings, key=len)
        if isinstance(value, list):
            for item in value:
                yield flag, item
        else:
            yield flag, value


def run(argv=None, log=print):
    a = build_parser().parse_args(argv)
    # Resolved once, here, before anything reads it. `--type` defaults to None so `--like` can
    # supply the base recipe from the reference's header, and nothing below this line sees None.
    a.type = resolved_type(a)

    # A CHECKPOINT, NOT A GGUF. `convert --quantise` has always done both steps in one command,
    # and this is the name people reach for when they want a quantised model: they typed
    # `quantise`, were told to run a different command first, and went away to learn a file
    # format in order to be handed back to the command they started with. The conversion is
    # delegated rather than reimplemented, so there is one converter and one set of checks.
    if looks_like_a_checkpoint(a.source):
        # BEFORE the conversion, not after it. This branch used to return straight into
        # `_quantise_a_checkpoint`, whose inner `run()` reaches `_preflight_arguments` only once
        # the checkpoint has been converted. So `--imatrix /nope` or `--threads 99999` was
        # refused after tens of minutes and tens of gigabytes of writes, for a fault decidable
        # from the command line before anything started. That is precisely what this function's
        # own docstring says it exists to prevent.
        _preflight_arguments(a, log=log)
        # AND THE OUTPUT-SIDE CHECKS, which the same argument reaches: `quantise <checkpoint>
        # <a file that is already there>` converted first and refused afterwards, having spent
        # tens of minutes and tens of gigabytes on a fault visible from the command line.
        preflight_a_checkpoint(a, log=log)
        return _quantise_a_checkpoint(a, log=log)

    _preflight_arguments(a, log=log)
    if a.out:
        out = Path(a.out)
    elif a.like:
        out = schedule_output_name(a.source, a.type, a.like)
    else:
        out = default_output(a.source, a.type)

    head = preflight(a.source, out, a.type, allow_requantize=a.allow_requantize, force=a.force)
    log(f"quantise {Path(a.source).name} ({head['file_type']}, {head['tensor_count']} tensors, "
        f"{head['architecture']}) -> {out.name} [{a.type}]")
    pairs = _preflight_overrides(a.source, a, log)
    # Kept SEPARATE from `pairs` rather than appended to it, and the difference is the verification.
    # `_verify_overrides` treats every pair as a pattern the operator chose and raises on anything
    # that did not land, which is right for a hand-picked override and wrong for a whole-file
    # schedule: see `verify_schedule`. Appending here would also print one "verified" line per
    # tensor, several hundred of them, which buries the one line that matters.
    schedule, schedule_pairs, skipped = _preflight_schedule(a, head, log)

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

    # After the imatrix and before the positionals, which is where llama-quantize expects every
    # option. Announced individually because a per-tensor precision is exactly the kind of choice
    # that gets forgotten between running a command and reading its output a week later.
    if a.output_tensor_type:
        argv_q += ["--output-tensor-type", a.output_tensor_type]
        log(f"  keeping {OUTPUT_TENSOR} at {a.output_tensor_type}")
    if a.token_embedding_type:
        argv_q += ["--token-embedding-type", a.token_embedding_type]
        log(f"  keeping {EMBED_TENSOR} at {a.token_embedding_type}")
    for name, kind in pairs:
        argv_q += ["--tensor-type", f"{name}={kind}"]
        log(f"  pinning tensors matching {name!r} to {kind}")
    # The copied schedule, after the user's own pairs so the command line reads in the order the
    # choices were made. Not announced per tensor: `_preflight_schedule` has already said how many
    # there are, and several hundred lines naming one tensor each is noise, not observability.
    for name, kind in schedule_pairs:
        argv_q += ["--tensor-type", f"{name}={kind}"]

    argv_q += [str(a.source), str(out), a.type]
    if a.threads:
        argv_q.append(str(a.threads))

    started = time.monotonic()
    # No timeout: quantising a large model is genuinely long and a ceiling here would kill a job
    # with its work nearly done, which is the failure that cost a completed head-to-head on
    # 2026-08-06. Interrupting it is the operator's call, and the partial output is cleaned below.
    r, fallback, tail = _run_quantiser(argv_q, verbose=a.verbose, log=log)
    took = time.monotonic() - started

    if r.returncode != 0:
        # A failed quantisation leaves a partial file that is exactly the shape of a real one.
        if out.exists():
            out.unlink()
            log(f"  removed the partial {out.name}")
        # ITS LAST WORDS, because summarising the output must not cost the diagnostic. Hiding
        # hundreds of per-tensor lines is only an improvement while a failure still explains itself.
        said = "\n".join(f"    {line}" for line in tail)
        raise SystemExit(
            f"llama-quantize exited {r.returncode} after {took:.0f}s. Nothing usable was written.\n"
            + (f"  Its last {len(tail)} line(s):\n{said}\n" if tail else "")
            + "  Re-run with --verbose to see everything it printed.")

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

    # The per-tensor receipt, and it is a separate check from the one above on purpose: `verify`
    # judges the file against its own name, and an override leaves that judgement unchanged.
    try:
        overrides = _verify_overrides(out, a, pairs, log)
    except gguf_io.GGUFError as e:
        raise SystemExit(
            f"the output quantised and its tensor list could not be read back to confirm the "
            f"per-tensor precision that was asked for: {e}\nThe file is left at {out}.") from e

    # The copied schedule's own receipt, tensor for tensor against the reference. Separate from
    # both checks above for the reason `verify_schedule` gives: `verify` judges the file against its
    # own name, `_verify_overrides` refuses anything the operator picked and did not get, and this
    # one reports a count because a whole-file schedule legitimately does not land everywhere.
    schedule_verified = None
    if a.like:
        try:
            schedule_verified = verify_schedule(out, schedule, log)
        except gguf_io.GGUFError as e:
            raise SystemExit(
                f"the output quantised and its tensor list could not be read back to compare it "
                f"against the schedule copied from {Path(a.like).name}: {e}\nThe file is left at "
                f"{out}, and whether it carries that schedule is unknown rather than confirmed."
            ) from e

    src_size, out_size = Path(a.source).stat().st_size, out.stat().st_size
    log(f"  wrote {out.name}: {out_size / 1e9:.2f} GB from {src_size / 1e9:.2f} GB "
        f"({out_size / src_size * 100:.0f}%), {got['tensor_count']} tensors, {took:.0f}s")
    log(f"  verified: {got['file_type']}, architecture {got['architecture']}")
    if fallback:
        fell, total = fallback
        log(f"  NOTE: {fell} of {total} tensors ({fell / total * 100:.0f}%) could not take the "
            f"{a.type} recipe and fell back to another type. The file is still a valid {a.type} "
            f"and its name is honest, but 'verified: {got['file_type']}' is a statement about what "
            f"the file declares, not about every tensor in it. The usual cause is tensor "
            f"dimensions the recipe's block size does not divide, which is benign and common in "
            f"small models; a high proportion on a large model is worth looking into.")

    # Provable, so it is stated: the file that came in had a prompt format and the file going out
    # does not. Quantisation copies metadata, so this should never fire; if it does, the output is
    # a model that loads, generates, and generates against a format it was not trained on, and
    # nothing downstream of here would say so.
    if gguf_io.has_chat_template(head) and not gguf_io.has_chat_template(got):
        log(f"  NOTE: the source carries {gguf_io.CHAT_TEMPLATE_KEY} and this output does not. "
            f"Every conversational result from this file will be measured on a prompt format the "
            f"model was not trained on, and it will look like a weak model rather than a broken "
            f"export. Recorded in the sidecar beside it.")

    say_what_is_still_different(a, log)

    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    record = {
        # /2 CARRIES THE sha256 OF THE SOURCE AND THE OUTPUT; /1 did not, and a reader cannot tell
        # a /1 record from a /2 one where the hash happened to fail unless the version says so. The
        # field is additive, so a /1 reader loses nothing, and the bump is what lets anything
        # downstream REQUIRE the hash rather than hope for it.
        #
        # /3 CARRIES `schedule_reference`, on the same argument: a reader cannot tell a /2 record
        # from a /3 one that happened to have no `--like` unless the version says so, and the field
        # is the only place the importance-matrix gap is written down.
        "schema": "senbonzakura-quantisation/3",
        "created": _now(),
        "quantiser": identity,
        "quant_type": a.type,
        "imatrix": imatrix_record,
        # ALWAYS PRESENT, `null` when `--like` was not used. See `schedule_record`.
        "schedule_reference": schedule_record(a, schedule, schedule_pairs, skipped,
                                              schedule_verified, log=log),
        # What was asked for AND what the finished file actually holds. A recipe name is a
        # statement about intent; the census is a statement about the file.
        "tensor_overrides": overrides,
        # None when the quantiser said nothing, which is not the same as zero. A reader comparing
        # two files has to be able to tell "no tensor fell back" from "we were not watching".
        "fallback_tensors": None if fallback is None else {"fell_back": fallback[0],
                                                           "of": fallback[1]},
        "allow_requantize": bool(a.allow_requantize),
        # THE PROMPT FORMAT, RECORDED AS A FACT ON BOTH SIDES rather than as a verdict.
        # Nothing this project ships measures a GGUF: `score`, `capability` and `drift` all read
        # a transformers checkpoint, so the place a lost template turns into a wrong number is
        # somebody else's llama.cpp, weeks later. The sidecar is the only thing that travels with
        # the file, so the fact goes here where a downstream reader can act on it, and the
        # comparison below is made only where it is provable.
        "source": {"name": Path(a.source).name, "bytes": src_size,
                   "sha256": digest_for_the_record(a.source, what="source", log=log),
                   "file_type": head["file_type"], "architecture": head["architecture"],
                   "chat_template": gguf_io.has_chat_template(head)},
        # Carried forward so the kept file still names the checkpoint it came from after
        # --prune-source removes the file this receipt describes.
        "source_conversion": source_conversion_record(a.source),
        "output": {"name": out.name, "bytes": out_size,
                   "sha256": digest_for_the_record(out, what="output", log=log),
                   "file_type": got["file_type"],
                   "architecture": got["architecture"], "tensor_count": got["tensor_count"],
                   "chat_template": gguf_io.has_chat_template(got)},
        "seconds": round(took, 1),
    }
    try:
        with atomic_write(sidecar) as fh:
            fh.write(json.dumps(record, indent=2) + "\n")
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
        # Its receipt goes with it. A record describing a file that is not there reads as
        # evidence about something a reader cannot inspect, and its content has already been
        # copied into the sidecar above.
        orphan = Path(str(a.source) + convert_record_suffix())
        if orphan.is_file():
            orphan.unlink()
            log(f"  removed {orphan.name}; its contents are in {sidecar.name}")
    return 0


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
