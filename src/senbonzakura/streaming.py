# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Read and rewrite a checkpoint one tensor at a time, never holding the model.

THE PROBLEM THIS SOLVES. Abliterating a model today means loading it: `load_model_and_tokenizer`
hands the whole checkpoint to transformers, which decides where it goes. That is fine while the
model fits and is a hard wall when it does not, and the wall is capacity rather than throughput, so
no amount of tuning moves it. A 6 GB laptop card cannot edit a 30B model, and renting a cloud GPU
to get around that is the exact bypass this project exists to remove.

The bake does not actually need the model. It needs one weight at a time: read the tensor, remove
the refusal direction from it, write it back. This module is that, and nothing more.

WHY IT IS NOT AN EXTENSION OF THE EXISTING LOADER. `load_model_and_tokenizer` asks transformers for
a model, and transformers fuses what it reads. Measured on 2026-09-22 with
`tools/research/expert_layout.py`: LFM2.5-8B-A1B stores 32 separate residual writers of
(2048, 1792) per layer, 7.0 MB each, and transformers builds one fused (32, 2048, 1792) of 224 MB
from them. Editing through the loader therefore materialises a tensor the checkpoint does not hold,
and pays roughly 1.8 GB per layer for weights stored in 7 MB pieces. Reading the shards directly
pays about 56 MB.

Granite-3.0-1b-a400m stores the fused (32, 1024, 512) stack itself, so there is no unfused form to
fall back on there and the blocked rewrite in `cli.orthogonalize_np_3d_` is the lever instead. Both
layouts are real. Anything here that assumes one of them is a defect.

HOW A SHARD IS REWRITTEN, and why not in place. A safetensors file is eight bytes of header length,
then a JSON map of tensor name to dtype, shape and byte range, then the tensor bytes. An edit that
preserves dtype and shape preserves the byte range, so the bytes could be written back where they
sit. This does not do that. It streams the shard to a sibling file, substituting the tensors that
changed, and renames at the end, because a crash halfway through an in-place rewrite leaves a
checkpoint that is neither the original nor the edited one and says nothing about which. The cost
is one shard of extra disk, transiently, rather than one model. `marker.py` rewrites shard headers
the same way and for the same reason.

WHAT IS NOT HERE YET. The capture half: pushing calibration prompts through one layer at a time to
extract the direction in the first place. That needs a forward pass and therefore a model, and it
is the open half of the v0.5 spike. This module is the bake half, which needs neither a GPU nor a
model, and is the part that can be tested on any machine.
"""
import contextlib
import itertools
import json
import os
import struct

#: safetensors pads its header to this boundary. Writing a header that is not padded produces a
#: file every reader still accepts, which is worse than one they reject: it diverges quietly.
ALIGNMENT = 8

#: Bytes per element, by the dtype names safetensors writes into the header. Deliberately explicit
#: rather than derived from numpy or torch: this module reads a file format, and a checkpoint
#: written by something that had a dtype we cannot size must be refused loudly rather than
#: mis-measured against whatever the local torch happens to think the name means.
DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
}

#: Copy buffer for the byte ranges that are passing through unchanged. Large enough that the copy
#: is not syscall bound, small enough that it is not part of the memory story this module exists
#: to fix.
CHUNK = 1 << 22


class ShardError(Exception):
    """A shard this module will not read or will not rewrite, with the reason."""


class Tensor:
    """One tensor's entry in a shard header: where it is, and what shape it claims to be."""

    __slots__ = ("dtype", "end", "name", "shape", "start")

    def __init__(self, name, dtype, shape, start, end):
        self.name = name
        self.dtype = dtype
        self.shape = tuple(shape)
        self.start = start
        self.end = end

    @property
    def nbytes(self):
        return self.end - self.start

    def __repr__(self):
        return f"Tensor({self.name!r}, {self.dtype}, {self.shape}, {self.nbytes} bytes)"


def _element_count(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def read_header(path):
    """The header of a safetensors shard, and the offset its tensor data starts at.

    Validated rather than trusted. The failure being guarded against is not a corrupt file, which
    announces itself; it is a header whose declared byte ranges disagree with the shapes beside
    them, because a rewrite driven by those ranges would then write a correct tensor into the wrong
    place and produce a checkpoint that loads.
    """
    # The read and the judgement are kept apart: only the read can raise OSError, and wrapping the
    # validation in the same try would turn a future bug in it into "cannot be read".
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            size = os.fstat(fh.fileno()).st_size
            length = struct.unpack("<Q", raw)[0] if len(raw) == 8 else None
            blob = fh.read(length) if length is not None and 0 < length <= size else b""
    except OSError as exc:
        raise ShardError(f"{path}: cannot be read: {exc}") from exc

    if length is None:
        raise ShardError(f"{path}: too short to be a safetensors file")
    if length <= 0 or 8 + length > size:
        raise ShardError(f"{path}: header says it is {length} bytes and the file is {size}")
    if len(blob) != length:
        raise ShardError(f"{path}: header is truncated")

    try:
        header = json.loads(blob)
    except ValueError as exc:
        raise ShardError(f"{path}: header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ShardError(f"{path}: header is not a JSON object")
    return header, 8 + length


def tensors(path):
    """Every tensor in a shard, in the order its bytes appear, with the header checked first.

    Ordered by offset rather than by name because the rewrite copies the file forwards in one pass.
    A reader that took them in header order would seek backwards on most checkpoints.
    """
    header, data_start = read_header(path)
    size = os.path.getsize(path)
    found = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise ShardError(f"{path}: entry {name!r} is not an object")
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            raise ShardError(f"{path}: {name!r} has dtype {dtype!r}, which this reader cannot size")
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise ShardError(f"{path}: {name!r} has shape {shape!r}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(o, int) for o in offsets)):
            raise ShardError(f"{path}: {name!r} has data_offsets {offsets!r}")
        start, end = offsets
        declared = _element_count(shape) * DTYPE_BYTES[dtype]
        if start < 0 or end < start:
            raise ShardError(f"{path}: {name!r} spans {start} to {end}")
        if end - start != declared:
            raise ShardError(
                f"{path}: {name!r} is {shape} of {dtype}, which is {declared} bytes, and its "
                f"range covers {end - start}. The header disagrees with itself")
        if data_start + end > size:
            raise ShardError(f"{path}: {name!r} ends past the end of the file")
        found.append(Tensor(name, dtype, shape, start, end))

    found.sort(key=lambda t: t.start)
    for previous, nxt in itertools.pairwise(found):
        if nxt.start < previous.end:
            raise ShardError(
                f"{path}: {previous.name!r} and {nxt.name!r} overlap, so editing either would "
                f"corrupt the other")
    return found


def read_tensor(path, tensor):
    """The raw bytes of one tensor. Peak memory is that tensor, which is the whole point."""
    _, data_start = read_header(path)
    with open(path, "rb") as fh:
        fh.seek(data_start + tensor.start)
        raw = fh.read(tensor.nbytes)
    if len(raw) != tensor.nbytes:
        raise ShardError(f"{path}: {tensor.name!r} is shorter on disk than its header claims")
    return raw


def rewrite_shard(path, edit, *, log=None):
    """Stream a shard through, replacing the tensors `edit` chooses to replace.

    `edit` is called with each `Tensor` and must return the replacement bytes, or None to leave it
    alone. It is called in offset order and one tensor at a time, so a caller that edits every
    tensor still never holds more than one.

    Returns the number of tensors replaced. The header, including `__metadata__`, is written back
    byte for byte: an edit that preserved dtype and shape did not change any of it, and rewriting
    it anyway would silently drop a marker `marker.py` had put there.
    """
    entries = tensors(path)
    header_blob, data_start = _header_blob(path)
    tmp = f"{path}.rewriting"
    replaced = 0
    try:
        with open(path, "rb") as src, open(tmp, "wb") as out:
            out.write(struct.pack("<Q", len(header_blob)))
            out.write(header_blob)
            cursor = 0
            for tensor in entries:
                # Whatever sits between the previous tensor and this one is copied untouched.
                # safetensors does not normally leave gaps, and a reader that assumed so would
                # silently drop padding a future writer decides to add.
                _copy_range(src, out, data_start + cursor, tensor.start - cursor)
                cursor = tensor.start
                src.seek(data_start + tensor.start)
                raw = _exactly(src.read(tensor.nbytes), tensor, path)
                new = edit(tensor, raw)
                if new is None:
                    out.write(raw)
                else:
                    out.write(_same_size(new, tensor, path))
                    replaced += 1
                    if log:
                        log(f"  rewrote {tensor.name} {tensor.shape}")
                cursor = tensor.end
            # And whatever follows the last tensor.
            src.seek(0, os.SEEK_END)
            _copy_range(src, out, data_start + cursor, src.tell() - (data_start + cursor))
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        # No `.rewriting` left behind, per baseline 2.1. The failure this matters for is the disk
        # filling during the copy, where the leftover is itself part of what filled it.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

    # Rename last. An interrupted rewrite leaves the original shard exactly as it was, rather than
    # a file that is neither the original nor the edited one and cannot say which.
    os.replace(tmp, path)
    return replaced


def _exactly(raw, tensor, path):
    """The tensor's bytes, or a refusal. A short read here means the file lost bytes since its
    header was validated, which is rare and is not something to write a shorter tensor about.
    """
    if len(raw) != tensor.nbytes:
        raise ShardError(f"{path}: {tensor.name!r} is truncated on disk")
    return raw


def _same_size(new, tensor, path):
    """A replacement has to occupy exactly what it replaces.

    Refused rather than accommodated: a tensor that changed size would move every tensor after it,
    which means rewriting every offset in the header and is a different operation from this one.
    """
    if len(new) != tensor.nbytes:
        raise ShardError(
            f"{path}: the edit returned {len(new)} bytes for {tensor.name!r}, which occupies "
            f"{tensor.nbytes}. A rewrite cannot change a tensor's size without moving every "
            f"tensor after it")
    return new


def _header_blob(path):
    """The header's bytes as they will be written back: unchanged, and padded as they were."""
    with open(path, "rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        return fh.read(length), 8 + length


def _copy_range(src, out, offset, count):
    if count <= 0:
        return
    src.seek(offset)
    while count:
        chunk = src.read(min(CHUNK, count))
        if not chunk:
            raise ShardError("the shard ended while copying a range its header said was there")
        out.write(chunk)
        count -= len(chunk)
