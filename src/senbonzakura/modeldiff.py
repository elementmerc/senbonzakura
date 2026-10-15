# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""What changed between a checkpoint and the base it claims to come from.

WHAT THIS ANSWERS, AND WHAT IT CANNOT

It answers: which tensors differ from the claimed base, where in the stack, and by how much. That
is a fact about two files and it needs no GPU, no generated token and no judge.

It does NOT answer whether a model is safe, and it cannot. A backdoored checkpoint's defining
property is that ordinary behaviour does not move: the published demonstration (ProjectDiscovery,
2026-10-06) scored 100% on clean accuracy while firing on its trigger every time. The attacker
picks one trigger out of an unbounded space and the defender has to guess a key nobody handed
them, so no tool finds it, including this one. **The word "safe" does not appear in this command's
output and must not be added to it.**

WHY IT EXISTS ANYWAY

Because the thing the field recommends and nobody runs is exactly this. That same write-up's
advice to defenders is to check "whether the weights diff cleanly against the base", in the same
breath as saying most teams will not. A model card saying "abliterated from X" is, today, checked
by reading the sentence. This checks the files.

Three callers wanted it in one week, which is why it is a command rather than a helper: the
provenance plan needs it as its first rung, the hosted free tier needs it as a report that costs no
compute, and the v0.5 pod window needed it on 2026-10-09 and got twenty inline lines in a TOML
string because nothing existed.

WHY IT IS TORCH-FREE

`entry.py` dispatches without importing `cli`, so a command that avoids torch starts in
milliseconds and works on a machine where torch is not installed. Comparing two files does not
need a tensor library: identity is a byte comparison, and the one place a number is wanted
(how big the change is) is a decode this module does itself for the three float widths that
actually occur. A dtype it cannot decode still gets an exact changed-or-not verdict, which is the
load-bearing half, and the magnitude is reported as unavailable rather than guessed.

WHAT IS REFUSED RATHER THAN REPORTED

A dtype or shape mismatch on a shared tensor name. A Q4 build of a model differs from its bf16
base in every byte, and reporting that as "changed everywhere" would be true and useless, so a
mismatch stops the comparison and says which tensors disagree. Same discipline as the quantiser's
provenance refusal.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

from . import argresolve, checkpoint, crashsafe, say, streaming

HELP = """\
Compare a checkpoint against the base it claims to come from, tensor by tensor.

  senbonzakura diff --base Qwen/Qwen2.5-0.5B-Instruct --candidate ./edited

Reports which tensors differ, where in the stack, and by how much. It streams, so a pair of
checkpoints larger than memory is fine: peak memory is one tensor from each side.

This is a statement about two files. It is NOT a safety verdict: a backdoored checkpoint behaves
normally until its trigger appears, and no tool finds the trigger. A clean comparison means the
weights match the base, nothing more.
"""

#: Tensors whose name carries one of these is a residual-writing projection, which is what a
#: refusal edit rewrites. Used only to GROUP the report, never to decide what to compare: a diff
#: that only looked where it expected change would be unable to report change anywhere else,
#: which is the half that matters when the claimed method is not the method used.
WRITER_HINTS = ("o_proj", "out_proj", "down_proj", "wo", "dense_4h_to_h", "c_proj")

#: Per-tensor rows kept in the printed summary before it switches to counts. The artefact keeps
#: every row regardless; this is a terminal budget, not a data decision.
PRINT_ROWS = 25

#: WHICH READING A REPORT CARRIES, declared in the report rather than inferred from which fields
#: happen to be populated. `relative` is the full comparison, which needs both sets of weights.
#: `touched` is the cheap one, which needs only a fingerprint of the base.
#:
#: A reader that guessed the tier from whether `relative_change` was null would get it wrong for a
#: perfectly ordinary reason: a full comparison also leaves that field null on a tensor it could
#: not decode. The tier is a fact about how the report was made, so the report states it.
READING_RELATIVE, READING_TOUCHED = "relative", "touched"

#: What a BASE FINGERPRINT is: one hash per tensor, with the shape and dtype beside it.
#:
#: WHY IT EXISTS. The two channels that compare a model against its base need the base, and the
#: base is gigabytes. But "which tensors changed" is decided by a BYTE COMPARISON, so a hash per
#: tensor answers it exactly, and a whole model's worth of hashes is about twenty kilobytes. So the
#: question "which blocks did this edit touch, and how much of each block did it touch" can be
#: answered without the base weights ever being present. How FAR each one moved still cannot: that
#: needs the values, and no hash carries them.
FINGERPRINT_SCHEMA = "senbonzakura-modelfingerprint/1"

#: What this report IS, declared in the report, because another surface now reads it. The card's
#: edit-magnitude channel is drawn from these rows, and a reader that guesses at an undeclared
#: shape is a reader that draws a picture from whatever JSON it was handed. Every other artefact
#: this project writes declares its schema; this one did not, because until now nothing but a
#: human read it.
SCHEMA = "senbonzakura-modeldiff/1"

#: Float dtypes this module decodes itself, to the struct format and the width in bytes. Anything
#: else gets an exact identity verdict and no magnitude.
DECODABLE = {"F64": ("<d", 8), "F32": ("<f", 4), "F16": ("<e", 2)}


#: How many elements are decoded at a time. The bound that makes this command's own `--help` true.
#:
#: 4,096 BY MEASUREMENT, NOT BY TASTE, on 2026-10-13 over 4-million-element BF16 tensors:
#:
#:     chunk      peak      rate
#:     4,096     0.94 MB    4.45 million elements/second
#:     16,384    3.74 MB    4.10
#:     65,536   14.98 MB    3.73
#:     262,144  59.99 MB    3.53
#:
#: The expected trade is smaller chunks against more per-chunk overhead, and it does not appear:
#: the smallest chunk is both the leanest and the fastest, which is what cache locality looks like
#: when the working set stops fitting. So this is the cheap end of both axes rather than a
#: compromise between them. Peak is flat in the tensor's size at every row, which is the property
#: that matters; the figure is which constant it is flat at.
CHUNK_ELEMENTS = 4_096

#: Elements per second this decodes and compares, measured on 2026-10-13 on an eight-core laptop:
#: 4.15 million. Rounded down to 4 million because the figure is only ever used to warn somebody
#: that a run is long, and a warning that overstates the wait is the harmless direction.
#:
#: It is pure Python arithmetic on purpose. numpy would be an order of magnitude faster and is not
#: a dependency of this package, and torch is, but pulling torch in here would put a multi-second
#: import in front of a command whose promise is that it is cheap.
ELEMENTS_PER_SECOND = 4_000_000

#: Past this much work, say how long it might take. One billion elements is about four minutes,
#: which is where a reader starts wondering whether the command has hung.
SLOW_ELEMENTS = 1_000_000_000

#: How often to say something from inside one tensor. The baseline asks for 30 to 60 seconds on
#: any long-running loop, and this is the loop that can sit silent for minutes on one tensor.
HEARTBEAT_SECONDS = 30


def _count(n):
    """A count a reader can take in at a glance, with units that suit its size.

    The first version printed millions unconditionally, so a progress line about a tensor smaller
    than a million elements read "0 of 0 million elements", which looks like a bug in the thing
    reporting progress rather than progress.
    """
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} billion"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f} million"
    return f"{n:,}"


def _width(dtype):
    """Bytes per element, or None when this cannot decode the dtype at all."""
    if dtype == "BF16":
        return 2
    return DECODABLE[dtype][1] if dtype in DECODABLE else None


def _decode_chunks(raw, dtype):
    """Floats from a tensor's raw bytes, a bounded chunk at a time.

    **CHUNKED, AND THAT IS THE WHOLE POINT.** This returned one list holding the entire tensor,
    and a Python float costs about 24 bytes against BF16's 2, so the list cost more than an order
    of magnitude more than the data it came from. Measured 2026-10-13 on 8-million-element tensors:
    **967 MB resident for 16 MB of raw data per side, 60 times.** This command's `--help` says peak
    memory is one tensor from each side, and that sentence was false by that factor on every run.

    The arithmetic in `_magnitude` is three running sums, all of which are associative, so chunking
    changes the bound and not the answer.

    BF16 is handled by hand because it is not a struct format and not a numpy dtype: it is the top
    sixteen bits of an IEEE single, so shifting each pair left into a four-byte word and reading
    that as a float is exact rather than approximate. Doing it by hand also keeps this module free
    of a numeric dependency it would otherwise pull in for one conversion.
    """
    width = _width(dtype)
    if width is None:
        return
    total = len(raw) // width
    for start in range(0, total, CHUNK_ELEMENTS):
        n = min(CHUNK_ELEMENTS, total - start)
        piece = raw[start * width:(start + n) * width]
        if dtype == "BF16":
            words = struct.unpack(f"<{n}H", piece)
            yield struct.unpack(f"<{n}f", struct.pack(f"<{n}I", *(w << 16 for w in words)))
        else:
            yield struct.unpack(f"<{n}{DECODABLE[dtype][0][1]}", piece)


def _which_side_is_not_finite(a, b, dtype):
    """Which of the two tensors holds a non-finite value: "the candidate", "the base" or "both".

    `a` is the candidate and `b` the base, matching the call in `compare`.
    """
    bad_a = any(not math.isfinite(v) for chunk in _decode_chunks(a, dtype) for v in chunk)
    bad_b = any(not math.isfinite(v) for chunk in _decode_chunks(b, dtype) for v in chunk)
    if bad_a and bad_b:
        return "both sides hold"
    return "the candidate holds" if bad_a else "the base holds"


def _squared_norm(raw, dtype):
    """The sum of squares of a tensor's elements, or None where the dtype is not decodable.

    Chunked through the same decoder `_magnitude` uses, so a norm taken here and a norm taken
    there cannot disagree, and so a single large tensor never has to be held as floats all at once.
    """
    if _width(dtype) is None:
        return None
    total = 0.0
    for chunk in _decode_chunks(raw, dtype):
        for value in chunk:
            total += value * value
    return total if math.isfinite(total) else None


def _magnitude(a, b, dtype, on_progress=None):
    """How big the change is, as a pair: the relative Frobenius norm and the largest element move.

    `on_progress(done, total)` is called as the chunks go by, so a caller can keep a long single
    tensor from looking like a hang. One tensor of an 8B model runs to hundreds of millions of
    elements, which is minutes of arithmetic with nothing printed.

    RELATIVE, because an absolute norm over a tensor says more about the tensor's size than about
    the edit. The denominator is the base's own norm, so 0.01 reads as "a one per cent change to
    this tensor" on every layer of every model and the figures are comparable down the stack.

    Returns `(relative, peak, why_not, sums)`, where `why_not` is None on success and otherwise
    names WHICH of the three unavailable cases this was, and `sums` is the pair of squared sums
    this tensor contributed, or None where there was no magnitude to contribute.

    THE SUMS ARE RETURNED RATHER THAN THE RATIO ALONE because the whole-model figure cannot be
    rebuilt from the ratios. A mean of per-tensor ratios weights a 4,000 element bias the same as
    a 20 million element projection, so it is not the model's relative change and must not be
    printed as one. Summing the numerators and the denominators and dividing once gives the real
    thing, and that needs the two sums rather than what they divided into.

    `why_not` NAMES WHICH of the three unavailable cases happened. They used to be
    indistinguishable, and the row reported all three as "this command does not decode {dtype}":
    false for two of them, and most wrong for the one a reader most needs, because a non-finite
    value in a candidate is exactly what a diff is for. The identity verdict beside it is still
    exact in every case, because that one is a byte comparison.
    """
    width = _width(dtype)
    if width is None:
        return None, None, (f"this command does not decode {dtype}, so whether the tensor changed "
                            f"is exact and how much it changed is unmeasured"), None
    if len(a) != len(b):
        return None, None, (f"the two tensors hold {len(a)} and {len(b)} bytes, so there is no "
                            f"element-by-element comparison to make. They are both {dtype} of the "
                            f"same declared shape, so one of the two files disagrees with its own "
                            f"header"), None
    num = base = peak = 0.0
    done = 0
    # `strict=True` on both: the outer zip would otherwise stop at the shorter stream and the inner
    # one at the shorter chunk, and a comparison that silently ignored a tail would report a
    # smaller change than the real one, which is the direction that matters here.
    total = len(a) // width
    for xs, ys in zip(_decode_chunks(a, dtype), _decode_chunks(b, dtype), strict=True):
        for x, y in zip(xs, ys, strict=True):
            d = x - y
            num += d * d
            base += y * y
            peak = max(peak, abs(d))
        done += len(xs)
        if on_progress is not None:
            on_progress(done, total)
    if not math.isfinite(num) or not math.isfinite(base):
        # WHICH SIDE, because they mean different things. A non-finite value in the CANDIDATE is a
        # broken edit, which is the single most useful thing this command can tell anybody. One in
        # the BASE means the checkpoint being compared against was already broken, and no edit made
        # it so.
        #
        # ASKED OF THE DATA, NOT INFERRED FROM THE SUMS, because the sums cannot answer it: a NaN
        # in the base poisons `base` AND `num`, since every difference against it is also a NaN,
        # so "both are NaN" does not mean both sides hold one. The first version of this reported
        # a broken base as "both sides" for exactly that reason.
        #
        # The extra pass is affordable because it runs only when a magnitude has already failed,
        # which is the rare path. Testing finiteness per element in the loop above would cost a
        # comparison on every element of every tensor to answer a question almost nobody asks.
        where = _which_side_is_not_finite(a, b, dtype)
        return None, None, (
            f"{where} a value that is not a finite number (a NaN or an infinity), so a "
            f"magnitude cannot be computed. This is not a dtype this command cannot read: "
            f"{dtype} is decoded, and what came out of it is not arithmetic. A checkpoint with "
            f"non-finite weights will not generate text, so this is worth looking at before "
            f"anything else in this report"), None
    return (math.sqrt(num) / math.sqrt(base) if base > 0 else None), peak, None, (num, base)


def resolve_checkpoint(spec, *, what):
    """A local directory holding safetensors shards, from a path or a cached Hub id.

    LOCAL ONLY, AND DELIBERATELY. A diff of a 145 GB pair would otherwise start a download that
    costs the user bandwidth they did not ask for, in a command whose whole promise is that it is
    cheap. A Hub id that is not already cached is a refusal naming the command that would fetch it.
    """
    path = Path(spec).expanduser()
    if path.is_dir():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            f"{what} {spec!r} is not a directory on this machine, and huggingface_hub is not "
            f"installed, so it cannot be looked up in the Hub cache either ({e}).") from e
    try:
        return Path(snapshot_download(spec, local_files_only=True))
    except Exception as e:
        raise SystemExit(
            f"{what} {spec!r} is neither a directory nor a model already in the Hub cache "
            f"({type(e).__name__}). This command does not download: a diff that quietly fetched "
            f"a checkpoint would cost you the bandwidth the comparison was supposed to save. "
            f"Fetch it first (`senbonzakura fetch {spec}`, or any tool that populates the cache) "
            f"and run this again.") from e


def index(model_dir):
    """A map from tensor name to (shard path, Tensor), read from the headers alone.

    Raises on a duplicate name across shards rather than taking the first. Two shards claiming the
    same tensor is a broken checkpoint, and picking one would make the comparison depend on
    directory listing order, which is exactly the kind of thing that makes a figure irreproducible.
    """
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise SystemExit(
            f"{model_dir} holds no .safetensors shards, so there is nothing to compare. A "
            f"checkpoint in another format (a .bin pickle, a GGUF) is not read by this command.")
    where = {}
    for shard in shards:
        for t in streaming.tensors(str(shard)):
            if t.name in where:
                raise SystemExit(
                    f"{model_dir}: {t.name!r} appears in both {where[t.name][0].name} and "
                    f"{shard.name}. Which one is the tensor is undecidable, so this refuses "
                    f"rather than letting the answer depend on which file was listed first.")
            where[t.name] = (shard, t)
    return where


def _layer_of(name):
    """The layer number in a tensor's name, or None. Used only for grouping."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if (p in ("layers", "h", "blocks", "block", "decoder")
                and i + 1 < len(parts) and parts[i + 1].isdigit()):
            return int(parts[i + 1])
    return None


def fingerprint(model_dir, *, log=None):
    """One hash per tensor, so a later comparison can tell what changed without these weights.

    Hashed from the tensor's RAW BYTES, which is the same thing `compare` compares when it decides
    whether a tensor changed. So a hash mismatch and a byte mismatch are the same fact, and the
    cheap reading cannot disagree with the full one about WHICH tensors moved. It can only decline
    to say how far.

    The shape travels with the hash because the cheap reading needs it: "how much of this block
    changed" is a share of the block's parameters, and a hash alone cannot weigh a bias against a
    projection.

    EACH TENSOR'S SQUARED NORM TRAVELS WITH IT TOO, and this is the field that makes the expensive
    reading cheap. A whole-model relative change is the delta norm over the BASE's norm, and the
    base's norm needs every tensor decoded, including the untouched ones that a comparison
    deliberately never reads. That is the whole reason the honest whole-model figure cost an order
    of magnitude more than the fast one.

    Measuring it HERE moves that cost to the base model, where it is paid once and reused by every
    comparison ever made against it, instead of to the candidate, where it would be paid again on
    every run. The same field also makes the exactly-correct PER BLOCK reading available, which
    `compare` could not produce for the same reason.

    So this function is deliberately the expensive one: it decodes everything. It is meant to be
    run once per base checkpoint and kept.
    """
    say_ = log or (lambda m: None)
    found = index(model_dir)
    tensors, read, undecodable = {}, 0, 0
    for n, (name, (shard, t)) in enumerate(sorted(found.items()), 1):
        raw = streaming.read_tensor(str(shard), t)
        read += len(raw)
        entry = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "shape": list(t.shape), "dtype": t.dtype,
        }
        # A dtype this command cannot decode gets a hash and no norm. Named as null rather than
        # left out, and counted below, because a missing norm silently dropped from a denominator
        # would make the whole-model figure read HIGHER than the truth, which is the direction
        # that matters: it would overstate the edit.
        squared = _squared_norm(raw, t.dtype)
        entry["squared_norm"] = squared
        if squared is None:
            undecodable += 1
        tensors[name] = entry
        if n % 200 == 0:
            say_(f"  read {n} of {len(found)} tensors ({read / 1e9:.1f} GB)")
    total = sum(e["squared_norm"] for e in tensors.values() if e["squared_norm"] is not None)
    return {
        "schema": FINGERPRINT_SCHEMA,
        "model": str(model_dir),
        "tensors": tensors,
        "total_squared_norm": total,
        "tensors_without_a_norm": undecodable,
        "bytes_read": read,
        "what_this_does_not_say": (
            "These are hashes of a checkpoint's tensors. They say whether a tensor is byte for "
            "byte the same as this one, and nothing about what it contains, what it does, or "
            "whether either checkpoint is safe."),
    }


def compare_to_fingerprint(cand_dir, base_fingerprint, *, log=None, base_dir=None):
    """The cheap reading: which tensors changed, from a fingerprint rather than from the weights.

    Returns the same report shape as `compare`, with `reading` set to `touched` and every
    `relative_change` null. That is not a degraded full comparison with fields missing, it is a
    different and exact answer to a smaller question, and the `reading` field is what tells a
    consumer which one it is holding.

    `base_dir` UPGRADES IT TO THE FULL READING AT A FRACTION OF THE COST, and this is the path
    worth understanding, because the obvious one is slower rather than faster.

    `compare` reads BOTH checkpoints' bytes for every tensor, because a byte comparison is how it
    decides what changed. Handing it a fingerprint removes the arithmetic on the untouched tensors
    and leaves the reading, so it still moves twice the bytes and measures no faster: that was
    measured at 114% of a plain full decode, not an improvement.

    This path decides what changed by hashing the CANDIDATE against the fingerprint, so the base is
    not read at all for that. Only the tensors that actually differ need the base's values, and an
    abliteration touches a few per cent of them. Every untouched tensor contributes zero to the
    numerator and its stored norm to the denominator, so the whole-model figure stays exact. With
    `base_dir` given, the reading is `relative` and carries everything `compare` would report.
    """
    say_ = log or (lambda m: None)
    declared = base_fingerprint.get("schema")
    if declared != FINGERPRINT_SCHEMA:
        raise SystemExit(
            f"the base fingerprint declares {declared!r} rather than {FINGERPRINT_SCHEMA!r}, so "
            f"it is not a fingerprint this command can read. Comparing against it would be "
            f"guessing at somebody else's format.")
    base = base_fingerprint.get("tensors")
    if not isinstance(base, dict) or not base:
        raise SystemExit("the base fingerprint names no tensors, so there is nothing to compare "
                         "this checkpoint against.")
    cand = index(cand_dir)
    only_base = sorted(set(base) - set(cand))
    only_cand = sorted(set(cand) - set(base))
    shared = sorted(set(base) & set(cand))
    source = index(base_dir) if base_dir is not None else None
    rows, read = [], 0
    moved_num = moved_base = whole_num = whole_base = 0.0
    by_block_num: dict[int, float] = {}
    by_block_base: dict[int, float] = {}
    for n, name in enumerate(shared, 1):
        shard, t = cand[name]
        raw = streaming.read_tensor(str(shard), t)
        read += len(raw)
        identical = hashlib.sha256(raw).hexdigest() == base[name].get("sha256")
        stored = base[name].get("squared_norm")
        rel = peak = why_not = None
        if not identical and source is not None and name in source:
            bshard, bt = source[name]
            braw = streaming.read_tensor(str(bshard), bt)
            read += len(braw)
            rel, peak, why_not, sums = _magnitude(raw, braw, bt.dtype)
            if sums is not None:
                moved_num += sums[0]
                moved_base += sums[1]
                whole_num += sums[0]
                whole_base += sums[1]
                _add(by_block_num, by_block_base, _layer_of(name), sums[0], sums[1])
        elif identical and stored is not None:
            whole_base += stored
            _add(by_block_num, by_block_base, _layer_of(name), 0.0, stored)
        rows.append({
            "tensor": name, "layer": _layer_of(name), "dtype": t.dtype,
            "shape": list(t.shape), "changed": not identical,
            "relative_change": None if rel is None else round(rel, 6),
            "largest_element_move": None if peak is None else float(peak),
            "magnitude_unavailable_because": (
                why_not if why_not is not None else
                None if identical or source is not None else
                "this comparison was made against a fingerprint of the base rather than against "
                "its weights, so whether the tensor changed is exact and how far it moved is "
                "not measured"),
            "is_residual_writer": any(h in name for h in WRITER_HINTS),
        })
        if n % 200 == 0:
            say_(f"  hashed {n} of {len(shared)} tensors ({read / 1e9:.1f} GB read)")
    changed = [r for r in rows if r["changed"]]
    layers = sorted({r["layer"] for r in changed if r["layer"] is not None})
    return {
        "schema": SCHEMA,
        "reading": READING_TOUCHED if source is None else READING_RELATIVE,
        "base": base_fingerprint.get("model"), "candidate": str(cand_dir),
        "tensors": {"shared": len(shared), "changed": len(changed),
                    "only_in_base": only_base, "only_in_candidate": only_cand},
        "changed_layers": layers,
        "changed_layer_span": None if not layers else [layers[0], layers[-1]],
        "relative_change_whole_model": (
            None if whole_base <= 0 else round(math.sqrt(whole_num) / math.sqrt(whole_base), 8)),
        "blocks": [
            {"index": layer,
             "relative_change": (None if by_block_base[layer] <= 0 else
                                 round(math.sqrt(by_block_num[layer])
                                       / math.sqrt(by_block_base[layer]), 8))}
            for layer in sorted(by_block_base)],
        "relative_change_over_changed_tensors": (
            None if moved_base <= 0 else round(math.sqrt(moved_num) / math.sqrt(moved_base), 6)),
        "changed_residual_writers": sum(1 for r in changed if r["is_residual_writer"]),
        "changed_elsewhere": sum(1 for r in changed if not r["is_residual_writer"]),
        "bytes_read": read,
        "rows": sorted(rows, key=lambda r: r["tensor"]),
        "what_this_does_not_say": (
            "This says which tensors differ from the base and nothing about how far they moved, "
            "because it was made from hashes rather than from the base's weights. It is not a "
            "safety verdict and not a backdoor check."),
    }


def _add(num_by, base_by, layer, num, base):
    """Accumulate a tensor's numerator and denominator into its block, ignoring unplaced tensors."""
    if layer is None:
        return
    num_by[layer] = num_by.get(layer, 0.0) + num
    base_by[layer] = base_by.get(layer, 0.0) + base


def compare(base_dir, cand_dir, *, log=None, base_norms=None):
    """The whole comparison, as a dict ready to serialise. Peak memory is two tensors.

    `base_norms` is an optional fingerprint of the BASE, as `fingerprint` writes it. Given one,
    this also reports the whole-model relative change and the exact per-block reading, neither of
    which it can produce on its own.

    WHY IT CANNOT PRODUCE THEM ON ITS OWN, since this is the question that keeps being asked. Both
    figures divide by the base's norm over tensors this command never decodes: it settles an
    unchanged tensor with a byte comparison and moves on, which is exactly where its speed comes
    from. Decoding them anyway would cost an order of magnitude to measure the size of weights
    nobody touched. A fingerprint has already measured them, once, so passing one in buys the
    expensive figures at no extra cost here.
    """
    say_ = log or (lambda m: None)
    base, cand = index(base_dir), index(cand_dir)
    only_base = sorted(set(base) - set(cand))
    only_cand = sorted(set(cand) - set(base))
    shared = sorted(set(base) & set(cand))

    # THE REFUSAL GOES FIRST, before a single byte is read. A dtype mismatch means the two
    # checkpoints are not comparable at all (a 4-bit build against a bf16 base differs
    # everywhere), and discovering that halfway through a 145 GB walk would have cost the walk.
    incomparable = []
    for name in shared:
        bt, ct = base[name][1], cand[name][1]
        if bt.dtype != ct.dtype or bt.shape != ct.shape:
            incomparable.append({"tensor": name,
                                 "base": {"dtype": bt.dtype, "shape": list(bt.shape)},
                                 "candidate": {"dtype": ct.dtype, "shape": list(ct.shape)}})
    if incomparable:
        shown = ", ".join(f"{e['tensor']} ({e['base']['dtype']}{list(e['base']['shape'])} against "
                          f"{e['candidate']['dtype']}{list(e['candidate']['shape'])})"
                          for e in incomparable[:3])
        raise SystemExit(
            f"these two checkpoints are not comparable: {len(incomparable)} of {len(shared)} "
            f"shared tensors differ in dtype or shape. {shown}"
            f"{'' if len(incomparable) <= 3 else ', and more'}.\n\n"
            f"The usual cause is a quantised candidate against a full-precision base, where every "
            f"byte differs for a reason that has nothing to do with an edit. Reporting that as "
            f"'changed everywhere' would be true and useless, so this refuses. Compare like with "
            f"like, or dequantise first.")

    # HOW LONG THIS MIGHT TAKE, SAID BEFORE IT STARTS. Only CHANGED tensors are decoded, so an
    # abliteration that rewrote a handful of projections finishes in seconds while a full finetune
    # decodes the whole model, and nothing said which of those the reader was about to sit through.
    # The ceiling is the honest number to quote because this cannot know what changed until it has
    # read it, and a reader who is warned about an hour and waits a minute is not harmed.
    # From `nbytes` and the dtype's width rather than from the shape, so this uses the same two
    # public facts the decode itself uses and cannot disagree with it about how big a tensor is.
    decodable = sum(base[name][1].nbytes // _width(base[name][1].dtype) for name in shared
                    if _width(base[name][1].dtype) is not None)
    if decodable > SLOW_ELEMENTS:
        say_(f"  up to {_count(decodable)} elements to compare, which is roughly "
             f"{max(1, round(decodable / ELEMENTS_PER_SECOND / 60))} minutes at the rate this was "
             f"measured at "
             f"if every tensor changed. Only changed tensors are decoded, so an edit that touched "
             f"a few projections will finish far sooner than that.")

    rows, read = [], 0
    # Accumulated across the CHANGED tensors only, which is what makes the figure below cheap and
    # is also exactly what its name has to say. A whole-model relative change would need the base
    # norm of every tensor including the untouched ones, and the untouched ones are never decoded:
    # this command settles them with a byte comparison, which is where its speed comes from. So
    # the honest cheap figure is "across the tensors that changed", and the expensive one is not
    # computed rather than approximated by the cheap one under a name that implies it.
    moved_num = moved_base = 0.0
    whole_num = whole_base = 0.0
    by_block_num: dict[int, float] = {}
    by_block_base: dict[int, float] = {}
    norms = None
    if base_norms is not None:
        if base_norms.get("schema") != FINGERPRINT_SCHEMA:
            raise SystemExit(
                f"the base fingerprint declares {base_norms.get('schema')!r} rather than "
                f"{FINGERPRINT_SCHEMA!r}, so the whole-model figure cannot be taken from it.")
        norms = base_norms.get("tensors") or {}
    last_beat = time.monotonic()

    def beat(done, total):
        nonlocal last_beat
        now = time.monotonic()
        if now - last_beat >= HEARTBEAT_SECONDS:
            last_beat = now
            say_(f"    still inside {name}: {_count(done)} of {_count(total)} elements")

    for n, name in enumerate(shared, 1):
        bshard, bt = base[name]
        cshard, ct = cand[name]
        braw = streaming.read_tensor(str(bshard), bt)
        craw = streaming.read_tensor(str(cshard), ct)
        read += len(braw)
        identical = braw == craw
        rel, peak, why_not, sums = ((None, None, None, None) if identical
                                    else _magnitude(craw, braw, bt.dtype, on_progress=beat))
        if sums is not None:
            moved_num += sums[0]
            moved_base += sums[1]
        if norms is not None:
            # The numerator is this tensor's own squared delta, which is zero when it did not move.
            # The denominator is its base norm, taken from the fingerprint when the tensor was not
            # decoded here and from the decode when it was. A tensor the fingerprint has no norm
            # for is left out of BOTH, so it cannot quietly shrink the denominator.
            whole = norms.get(name, {}).get("squared_norm")
            if identical and whole is not None:
                whole_num += 0.0
                whole_base += whole
                _add(by_block_num, by_block_base, _layer_of(name), 0.0, whole)
            elif sums is not None:
                whole_num += sums[0]
                whole_base += sums[1]
                _add(by_block_num, by_block_base, _layer_of(name), sums[0], sums[1])
        rows.append({
            "tensor": name, "layer": _layer_of(name), "dtype": bt.dtype,
            "shape": list(bt.shape), "changed": not identical,
            "relative_change": None if rel is None else round(rel, 6),
            "largest_element_move": None if peak is None else float(peak),
            # Named rather than left to a reader comparing nulls: a magnitude we could not compute
            # and a magnitude of zero are different facts, and the second is a measurement.
            "magnitude_unavailable_because": why_not,
            "is_residual_writer": any(h in name for h in WRITER_HINTS),
        })
        if n % 200 == 0:
            say_(f"  compared {n} of {len(shared)} tensors ({read / 1e9:.1f} GB read)")

    changed = [r for r in rows if r["changed"]]
    layers = sorted({r["layer"] for r in changed if r["layer"] is not None})
    return {
        "schema": SCHEMA,
        "reading": READING_RELATIVE,
        "base": str(base_dir), "candidate": str(cand_dir),
        "tensors": {"shared": len(shared), "changed": len(changed),
                    "only_in_base": only_base, "only_in_candidate": only_cand},
        "changed_layers": layers,
        "changed_layer_span": None if not layers else [layers[0], layers[-1]],
        # ONE number for how far the edit moved the weights it touched, as the relative Frobenius
        # norm: sqrt(sum of squared differences) over sqrt(sum of squared base values), summed
        # across the changed tensors and divided once. NOT a mean of the per-tensor ratios in
        # `rows`, which would weight a small bias the same as a large projection.
        #
        # THE DENOMINATOR IS THE CHANGED TENSORS, not the model, and the field name says so
        # because the difference is large: an abliteration touches a handful of projections, so a
        # whole-model denominator is far bigger and the same edit reads as a far smaller number.
        # Quoting this one as "x% of the weights moved" would overstate it by whatever fraction of
        # the model was left alone.
        # THE WHOLE-MODEL FIGURE, present only when a base fingerprint was supplied. This is the
        # one that may honestly be worded "x% of the weights moved": its denominator is every
        # tensor of the base, not only the ones the edit touched.
        "relative_change_whole_model": (
            None if (norms is None or whole_base <= 0)
            else round(math.sqrt(whole_num) / math.sqrt(whole_base), 8)),
        # Per block, the same reading: the block's delta norm over the block's own base norm. This
        # is the exactly-correct per-block number, as opposed to the parameter-weighted
        # approximation a reader has to fall back on without a fingerprint.
        "blocks": (
            None if norms is None else
            [{"index": layer,
              "relative_change": (None if by_block_base[layer] <= 0 else
                                  round(math.sqrt(by_block_num[layer])
                                        / math.sqrt(by_block_base[layer]), 8))}
             for layer in sorted(by_block_base)]),
        "relative_change_over_changed_tensors": (
            None if moved_base <= 0 else round(math.sqrt(moved_num) / math.sqrt(moved_base), 6)),
        "changed_residual_writers": sum(1 for r in changed if r["is_residual_writer"]),
        "changed_elsewhere": sum(1 for r in changed if not r["is_residual_writer"]),
        "bytes_read": read,
        # Sorted at the serialisation boundary so two runs over the same pair produce byte
        # identical output, which is what makes a published report re-derivable.
        "rows": sorted(rows, key=lambda r: r["tensor"]),
        "what_this_does_not_say": (
            "This compares two sets of weights. It is not a safety verdict and not a backdoor "
            "check. A poisoned checkpoint behaves normally until its trigger appears, the trigger "
            "is chosen from an unbounded space, and no tool recovers it. A comparison showing no "
            "change means the weights match the base and nothing more."),
    }


def render(report):
    """The terminal summary. Counts first, then the heaviest rows, then what it cannot say."""
    t = report["tensors"]
    out = [f"compared {t['shared']} shared tensors, {t['changed']} changed"]
    if t["only_in_base"]:
        out.append(f"  {len(t['only_in_base'])} tensor(s) are in the base and not the candidate, "
                   f"reported as absent rather than as changed: "
                   f"{', '.join(t['only_in_base'][:3])}"
                   f"{'' if len(t['only_in_base']) <= 3 else ', ...'}")
    if t["only_in_candidate"]:
        out.append(f"  {len(t['only_in_candidate'])} tensor(s) are in the candidate and not the "
                   f"base: {', '.join(t['only_in_candidate'][:3])}"
                   f"{'' if len(t['only_in_candidate']) <= 3 else ', ...'}")
    if not t["changed"]:
        out.append("  no shared tensor differs by a single byte")
    else:
        span = report["changed_layer_span"]
        if span:
            out.append(f"  changed layers span {span[0]} to {span[1]} "
                       f"({len(report['changed_layers'])} layers touched)")
        out.append(f"  of the changed tensors, {report['changed_residual_writers']} are "
                   f"residual-writing projections and {report['changed_elsewhere']} are not")
        heavy = sorted((r for r in report["rows"] if r["changed"]),
                       key=lambda r: (r["relative_change"] is None, -(r["relative_change"] or 0)))
        out.append(f"  the {min(PRINT_ROWS, len(heavy))} largest changes:")
        for r in heavy[:PRINT_ROWS]:
            # A CHANGED TENSOR MUST NOT PRINT AS 0.000%. Every row here changed by definition, so
            # a figure that rounds to zero at three places would read as "no change" next to the
            # count that says otherwise, and the reader would have to decide which to believe.
            # Seen immediately on the first real run: a one-byte edit to a 0.5B o_proj is about
            # 4e-6 of the tensor's norm.
            rel = r["relative_change"]
            if rel is None:
                amount = "unmeasured"
            elif rel * 100 < 0.001:
                amount = "<0.001%"
            else:
                amount = f"{rel * 100:.3f}%"
            out.append(f"    {r['tensor']}  {amount}")
    out.append("")
    out.append(report["what_this_does_not_say"])
    return "\n".join(out)


def build_parser():
    p = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False, prog="senbonzakura diff", description=HELP)
    p.add_argument("--base", required=True,
                   help="the checkpoint the candidate claims to come from: a local directory, or "
                        "a Hub id already in the local cache")
    p.add_argument("--candidate", required=True,
                   help="the checkpoint to compare against it")
    p.add_argument("--out", default=None,
                   help="write the full per-tensor report here as JSON. Left out, only the "
                        "summary is printed and the per-tensor rows are not kept")
    p.add_argument("--json", action="store_true",
                   help="print the whole report as JSON on stdout instead of the summary")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    def log(m):
        say.say(m)

    base = resolve_checkpoint(args.base, what="the base")
    cand = resolve_checkpoint(args.candidate, what="the candidate")
    # The same index guard every accelerate placement in this project gets, for the same reason:
    # a shard name taken out of an index and opened relative to a directory is a path traversal
    # unless somebody checks it, and this command opens both directories' shards.
    checkpoint.refuse_unsafe_index(str(base))
    checkpoint.refuse_unsafe_index(str(cand))
    if os.path.realpath(base) == os.path.realpath(cand):
        raise SystemExit(
            f"the base and the candidate are the same directory ({base}), so the comparison can "
            f"only report that a checkpoint matches itself. Name two checkpoints.")
    log(f"comparing {cand} against {base}")
    try:
        report = compare(base, cand, log=log)
    except streaming.ShardError as exc:
        # A SENTENCE, NOT A TRACEBACK, which is the answer `budget` already gave to the same
        # question through the same reader. One malformed checkpoint got a clean refusal from one
        # command and a stack trace from the other, and the one that crashed is the one whose whole
        # job is inspecting files you have reason to doubt.
        say.say(f"diff refused: {exc}", indent="  ")
        return DIFF_UNREADABLE
    return _write(args, report, log)


#: Exit codes, matching `budget`'s so a script can branch on them the same way. 0 the comparison
#: ran, 2 a checkpoint could not be read at all. There is no 1 here: "they differ" is the ordinary
#: result of this command rather than a failure, and a reader scripting against it wants the report.
DIFF_OK, DIFF_UNREADABLE = 0, 2


def _write(args, report, log):
    """The reporting tail, split out so the refusal above returns without duplicating it."""
    if args.out:
        # THROUGH THE SHARED WRITER, which this used to reimplement badly. The hand-rolled version
        # here renamed on close and did nothing else: no fsync, and no cleanup, so a failure
        # between writing and renaming left a `.part` behind for ever, which is the pattern
        # baseline 2.1 forbids. `crashsafe.atomic_write` fsyncs, removes the partial on any
        # exception, and explains a failed replace by looking at what is on disk. It imports
        # nothing heavy, so it costs this command none of the startup time it promises.
        with crashsafe.atomic_write(args.out) as fh:
            fh.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        log(f"wrote {args.out}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))
    return DIFF_OK


if __name__ == "__main__":
    sys.exit(main())
