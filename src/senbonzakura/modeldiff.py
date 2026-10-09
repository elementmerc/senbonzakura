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

import json
import math
import os
import struct
import sys
from pathlib import Path

from . import argresolve, checkpoint, say, streaming

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

#: Float dtypes this module decodes itself, to the struct format and the width in bytes. Anything
#: else gets an exact identity verdict and no magnitude.
DECODABLE = {"F64": ("<d", 8), "F32": ("<f", 4), "F16": ("<e", 2)}


def _decode(raw, dtype):
    """A list of floats from a tensor's raw bytes, or None when the dtype is not decodable.

    BF16 is handled by hand because it is not a struct format and not a numpy dtype: it is the top
    sixteen bits of an IEEE single, so shifting each pair left into a four-byte word and reading
    that as a float is exact rather than approximate. Doing it by hand also keeps this module free
    of a numeric dependency it would otherwise pull in for one conversion.
    """
    if dtype == "BF16":
        n = len(raw) // 2
        words = struct.unpack(f"<{n}H", raw[:n * 2])
        return list(struct.unpack(f"<{n}f", struct.pack(f"<{n}I", *(w << 16 for w in words))))
    if dtype in DECODABLE:
        fmt, width = DECODABLE[dtype]
        n = len(raw) // width
        return list(struct.unpack(f"<{n}{fmt[1]}", raw[:n * width]))
    return None


def _magnitude(a, b, dtype):
    """How big the change is, as a pair: the relative Frobenius norm and the largest element move.

    RELATIVE, because an absolute norm over a tensor says more about the tensor's size than about
    the edit. The denominator is the base's own norm, so 0.01 reads as "a one per cent change to
    this tensor" on every layer of every model and the figures are comparable down the stack.

    Returns (None, None) for a dtype this cannot decode, which is reported as unavailable. The
    identity verdict beside it is still exact, because that one is a byte comparison.
    """
    xs, ys = _decode(a, dtype), _decode(b, dtype)
    if xs is None or ys is None or len(xs) != len(ys):
        return None, None
    num = base = peak = 0.0
    for x, y in zip(xs, ys, strict=True):
        d = x - y
        num += d * d
        base += y * y
        peak = max(peak, abs(d))
    if not math.isfinite(num) or not math.isfinite(base):
        return None, None
    return (math.sqrt(num) / math.sqrt(base) if base > 0 else None), peak


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


def compare(base_dir, cand_dir, *, log=None):
    """The whole comparison, as a dict ready to serialise. Peak memory is two tensors."""
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

    rows, read = [], 0
    for n, name in enumerate(shared, 1):
        bshard, bt = base[name]
        cshard, ct = cand[name]
        braw = streaming.read_tensor(str(bshard), bt)
        craw = streaming.read_tensor(str(cshard), ct)
        read += len(braw)
        identical = braw == craw
        rel, peak = (None, None) if identical else _magnitude(craw, braw, bt.dtype)
        rows.append({
            "tensor": name, "layer": _layer_of(name), "dtype": bt.dtype,
            "shape": list(bt.shape), "changed": not identical,
            "relative_change": None if rel is None else round(rel, 6),
            "largest_element_move": None if peak is None else float(peak),
            # Named rather than left to a reader comparing nulls: a magnitude we could not compute
            # and a magnitude of zero are different facts, and the second is a measurement.
            "magnitude_unavailable_because": (
                None if identical or rel is not None
                else f"this command does not decode {bt.dtype}, so whether the tensor changed is "
                     f"exact and how much it changed is unmeasured"),
            "is_residual_writer": any(h in name for h in WRITER_HINTS),
        })
        if n % 200 == 0:
            say_(f"  compared {n} of {len(shared)} tensors ({read / 1e9:.1f} GB read)")

    changed = [r for r in rows if r["changed"]]
    layers = sorted({r["layer"] for r in changed if r["layer"] is not None})
    return {
        "base": str(base_dir), "candidate": str(cand_dir),
        "tensors": {"shared": len(shared), "changed": len(changed),
                    "only_in_base": only_base, "only_in_candidate": only_cand},
        "changed_layers": layers,
        "changed_layer_span": None if not layers else [layers[0], layers[-1]],
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
    report = compare(base, cand, log=log)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(f"{args.out}.part")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        tmp.replace(args.out)          # atomic: a reader never sees a half-written report
        log(f"wrote {args.out}")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
