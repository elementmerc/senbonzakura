# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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
import collections
import contextlib
import itertools
import json
import os
import pathlib
import re
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

#: Every spelling of "how many transformer layers" across the architectures this tool supports.
#: Read from the config rather than inferred from the weights, for the reason the whole module
#: turns on: a checkpoint whose layers are named in some new way must be measured rather than
#: silently skipped. Same list-from-the-config discipline as `tools/research/expert_layout.py`,
#: which was rewritten after name matching reported a mixture-of-experts checkpoint as dense.
LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "n_layers", "num_layers",
                    "num_decoder_layers", "n_block")

#: An integer path component, which is how every architecture spells an index:
#: `model.layers.3.mlp.experts.7.w2.weight` has two, and only the first of them is the layer.
_INDEXED = re.compile(r"(?<=\.)(\d+)(?=\.)")


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


# ── the layer index ──────────────────────────────────────────────────────────────────────────
#
# WHAT IT IS FOR. Everything downstream of here works one layer at a time, and none of it can
# start without knowing which tensors belong to which layer and which file they sit in:
#
#   * the capture pass loads layer i, runs the prompts through it, and evicts it;
#   * the bake rewrites layer i's residual writers and writes a completion marker, so a run that
#     dies on layer 30 of 48 resumes at 30 rather than at 0;
#   * the preflight sizes the largest layer to say, before anything starts, whether the run fits
#     and roughly how long it will take.
#
# It reads HEADERS ONLY. A whole 30B checkpoint costs a few hundred kilobytes of JSON to index
# and no weights at all, which is what makes it usable in a preflight that has to answer before
# the user has committed to anything.


class LayerIndex:
    """Which tensors make up each layer, and where their bytes are.

    `layers[i]` maps a tensor name to `(shard_path, Tensor)`. `shared` holds everything that is
    not part of any layer: embeddings, the final norm, the language-model head.
    """

    __slots__ = ("count", "layers", "root", "shared")

    def __init__(self, root, count, layers, shared):
        self.root = pathlib.Path(root)
        self.count = count
        self.layers = layers
        self.shared = shared

    def nbytes(self, i):
        """On-disk bytes of layer `i`. What one layer costs to read, and the preflight's unit."""
        return sum(t.nbytes for _shard, t in self.layers[i].values())

    def writers(self, i, ablate_conv=True):
        """The tensors in layer `i` that write to the residual stream, which are the edited ones.

        From `writers.py`, which is the SAME list the editor walks and the snapshot estimator
        sizes against. A streaming bake that kept its own idea of which tensors to edit would be
        the third hand-kept copy of a set of architecture names, and the second one already
        drifted far enough to refuse a model the editor could edit perfectly well.

        A name match is an estimate and cannot apply the structural conditions the editor applies
        with the module in hand. It is what walks the checkpoint; on an architecture the resident
        path would refuse, `refuse_unrecognised_writers` is still the one that refuses it.
        """
        from .writers import is_writer_tensor
        return {name: entry for name, entry in self.layers[i].items()
                if is_writer_tensor(name, ablate_conv)}

    def widest(self):
        """(index, bytes) of the largest layer.

        The memory budget is set by the WORST layer, not the average. Architectures that put a
        dense mixture-of-experts block on some layers and not others exist, and budgeting the mean
        would fit every layer but one and fail hours in.
        """
        sizes = [self.nbytes(i) for i in range(self.count)]
        widest = max(range(self.count), key=sizes.__getitem__)
        return widest, sizes[widest]

    def shards(self):
        """Every shard the checkpoint spans, in a stable order."""
        seen = {shard for layer in self.layers for shard, _t in layer.values()}
        seen |= {shard for shard, _t in self.shared.values()}
        return sorted(seen)

    def __repr__(self):
        return (f"LayerIndex({self.count} layers, {len(self.shards())} shard(s), "
                f"{len(self.shared)} shared tensor(s))")


def layer_count(model_dir):
    """How many layers the CONFIG says there are, and which key said so.

    From the config rather than from the weights, deliberately. Counting distinct indices in the
    tensor names would agree on every checkpoint that is already understood and would quietly
    agree with itself on one that is not: a checkpoint whose last layer is stored under a name
    this reader does not match would report one fewer layer and edit one fewer, and nothing
    anywhere would say so. The config is a second, independent account, which is the whole point
    of consulting it.
    """
    model_dir = pathlib.Path(model_dir)
    for cfg_path in sorted(model_dir.rglob("config.json")):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(cfg, dict):
            continue
        # Some configs nest the text model's settings a level down, which is where a multimodal
        # checkpoint keeps the part this tool edits.
        for scope in (cfg, cfg.get("text_config") or {}):
            if not isinstance(scope, dict):
                continue
            for key in LAYER_COUNT_KEYS:
                if isinstance(scope.get(key), int) and scope[key] > 0:
                    return scope[key], key
    raise ShardError(
        f"{model_dir}: no config declares a layer count under any of {list(LAYER_COUNT_KEYS)}. "
        f"A streaming run has to know how many layers it is walking before it starts, and "
        f"counting them from the tensor names would agree with whatever the names happen to "
        f"spell rather than with the model")


def _layer_axis(names, count):
    """Which integer position in a tensor name is the LAYER index.

    Decided against the config's count rather than by matching the word "layers", because the
    name for that block is `layers` on most architectures, `h` on GPT-2 descendants and
    `blocks` elsewhere, and a reader that knows three spellings reports clean on the fourth.

    A name can carry several indices: `model.layers.3.mlp.experts.7.w2.weight` has the layer and
    the expert. The layer is the FIRST, on every architecture this tool has met, because the
    layer stack is the outer structure and everything indexed inside it is nested underneath. So
    candidate positions are tried in order and the earliest whose values cover exactly
    `0..count-1` wins. A mixture-of-experts checkpoint with as many experts as layers would
    otherwise make position 1 look just as good, and it is the position that is ambiguous there,
    not the answer.

    Returns None when nothing matches, which is a refusal for the caller to make rather than an
    assumption for this to paper over.
    """
    wanted = set(range(count))
    seen = collections.defaultdict(set)
    for name in names:
        for position, match in enumerate(_INDEXED.finditer(name)):
            seen[position].add(int(match.group(1)))
    for position in sorted(seen):
        if seen[position] == wanted:
            return position
    return None


def _nth_index(name, position):
    """The integer at `position` among the name's integer components, or None."""
    for i, match in enumerate(_INDEXED.finditer(name)):
        if i == position:
            return int(match.group(1))
    return None


def index_layers(model_dir):
    """Map a checkpoint to its layers, reading headers and no weights.

    Refuses loudly rather than returning a partial answer. A streaming run that silently skipped
    a layer would produce a model that loads, generates, and was edited in 47 places out of 48,
    and nothing downstream could tell. That is the failure this whole module is shaped around.
    """
    model_dir = pathlib.Path(model_dir)
    count, _key = layer_count(model_dir)

    located = {}
    for shard in sorted(model_dir.rglob("*.safetensors")):
        for tensor in tensors(shard):
            if tensor.name in located:
                raise ShardError(
                    f"{model_dir}: {tensor.name!r} appears in more than one shard "
                    f"({located[tensor.name][0].name} and {shard.name}). Editing it would leave "
                    f"the other copy behind, and a loader may read either")
            located[tensor.name] = (shard, tensor)
    if not located:
        raise ShardError(f"{model_dir}: no safetensors shards, so there is nothing to index")

    axis = _layer_axis(located, count)
    if axis is None:
        raise ShardError(
            f"{model_dir}: the config declares {count} layers and no position in the tensor "
            f"names carries exactly the indices 0 to {count - 1}. This is a naming layout this "
            f"reader does not know, not a checkpoint without layers, and guessing which tensors "
            f"belong to which layer is how a streaming run edits 47 layers of 48 in silence")

    layers = [{} for _ in range(count)]
    shared = {}
    for name, entry in located.items():
        i = _nth_index(name, axis)
        if i is None:
            shared[name] = entry
        elif 0 <= i < count:
            layers[i][name] = entry
        else:
            # An index at the layer position outside the declared range. Refused rather than
            # filed under `shared`, because the two readings ("the config is wrong" and "this
            # position is not the layer after all") have opposite fixes and picking one quietly
            # would hide whichever it was.
            raise ShardError(
                f"{model_dir}: {name!r} carries {i} where the layer index sits, and the config "
                f"declares {count} layers. Either the config disagrees with the weights or this "
                f"reader picked the wrong index position")

    empty = [i for i, layer in enumerate(layers) if not layer]
    if empty:
        raise ShardError(
            f"{model_dir}: the config declares {count} layers and layer(s) {empty} hold no "
            f"tensors. A streaming run would walk straight past them and report success")
    return LayerIndex(model_dir, count, layers, shared)
