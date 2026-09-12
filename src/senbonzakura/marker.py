#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What was done to this checkpoint, written where it cannot be separated from the weights.

WHY THIS EXISTS

Every model this tool produces already ships an `abliteration.json` beside it, recording the
configuration, the seed and the result. That file binds only what reads it, and copying one
`model.safetensors` out of a directory is exactly how a checkpoint comes to outlive its
provenance. It was raised on 2026-08-15 while reviewing the control arm of the hybrid experiment:
a PARTIAL abliteration, produced deliberately to test whether refusal travels through a
convolution path, is a model whose refusal behaviour is only half removed. It must never be able
to pass for a whole one, and a sibling JSON file is not what stops that.

So the marker goes in two places that travel differently:

- **`config.json`**, which is mandatory to load a model at all, so anything that loads the weights
  has already read it. Measured rather than assumed: transformers attaches the unknown key to the
  config object and carries it through a subsequent `save_pretrained`, so it propagates to
  derivatives rather than being dropped at the first re-save.
- **The safetensors header**, which survives somebody lifting a single shard out of the directory.

HOW THE HEADER IS REWRITTEN, AND WHY NOT WITH TORCH

`save_pretrained` builds its metadata internally and offers no hook, so the shards are stamped
afterwards. A safetensors file is `[u64 header length][JSON header][tensor data]`, and every
offset in the header is relative to the start of the data block, so replacing the header moves
nothing: the data is copied through byte for byte and no tensor is deserialised, reshaped or
re-quantised on the way. Loading each shard with torch and saving it again would do the same job
while holding a shard in memory and putting every dtype through a round trip, which is a great
deal of risk for a metadata edit.

HOW STRICT IT IS, AND WHY THAT IS ASYMMETRIC

A whole abliteration that loses its provenance line is an annoyance. A partial one that loses its
marker is the failure this module was written for. So `required=True` (used for partial models)
raises, and a whole model warns loudly and keeps the run, because the alternative is killing a
save whose GPU work is already spent over an informational field.
"""
import contextlib
import json
import os
import shutil
import struct

NAMESPACE = "senbonzakura"

#: safetensors puts the tensor data immediately after the header and readers expect that boundary
#: to be 8-byte aligned, so a header shorter than a multiple of 8 is padded with spaces. JSON
#: ignores trailing whitespace, so the padding costs nothing to parse.
ALIGNMENT = 8


def fields(*, version, ablate_conv, partial_layers, num_directions=None, dir_mode=None,
                  seed=None, base_model=None):
    """The flat string map that goes in both places.

    Flat and string-valued because the safetensors header allows nothing else, and the same shape
    is used in `config.json` so the two cannot drift into saying different things.
    """
    fields = {
        "tool": "senbonzakura",
        "version": str(version),
        "abliterated": "true",
        # The question a reader most needs answered, phrased so that reading it wrongly is hard.
        # `partial` is the word, not `complete`, because a missing field then reads as "unknown"
        # rather than as "fine".
        "partial": "true" if partial_layers else "false",
        "ablate_conv": "true" if ablate_conv else "false",
    }
    if partial_layers:
        fields["unedited_layers"] = ",".join(str(i) for i in sorted(partial_layers))
        fields["warning"] = ("PARTIAL ABLITERATION: the layers listed in unedited_layers write the "
                             "residual stream through a module this run deliberately left alone. "
                             "This model's refusal behaviour is only partly removed. It is a "
                             "control, not a result.")
    if num_directions is not None:
        fields["num_directions"] = str(num_directions)
    if dir_mode is not None:
        fields["dir_mode"] = str(dir_mode)
    if seed is not None:
        fields["seed"] = str(seed)
    if base_model:
        fields["base_model"] = str(base_model)
    return fields


def _rewrite_header(path, fields):
    """Add the marker to one shard's metadata, leaving every tensor byte where it was."""
    tmp = f"{path}.stamping"
    try:
        _write_stamped_copy(path, tmp, fields)
    except BaseException:
        # NO `.stamping` LEFT BEHIND. Baseline 2.1: a partial-state file is cleaned up on the
        # way out, not left for the next run to find. The failure this matters for is the disk
        # filling mid-stamp, where the leftover copy is itself part of what filled it.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    # Rename last, so an interrupted stamp leaves the original shard intact rather than a
    # half-written one. The whole point of this module is that a checkpoint says what it is, and a
    # truncated checkpoint that says so is not an improvement.
    os.replace(tmp, path)


def _write_stamped_copy(path, tmp, fields):
    """The copy itself: header replaced, every tensor byte passed through unchanged."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        meta = dict(header.get("__metadata__") or {})
        meta.update({f"{NAMESPACE}.{k}": v for k, v in fields.items()})
        header["__metadata__"] = meta
        blob = json.dumps(header, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % ALIGNMENT)
        with open(tmp, "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            while True:
                chunk = f.read(1 << 22)
                if not chunk:
                    break
                out.write(chunk)


class PartialStampError(Exception):
    """Some shards in this directory carry the marker and some do not."""


def _preflight_room(directory, shards, log):
    """Refuse before the first shard if there is not room for the largest copy.

    Stamping writes a COMPLETE copy of a shard next to it and renames over the original, one at
    a time, so the requirement is the largest single shard rather than the whole model. Nothing
    reserved it: the model save's own pre-flight reserves 5% of the model size, and on a 30B
    with 5 GB shards that is not one shard. With `--free-base-model` the base is already gone by
    this point, so recovering from a half-stamped directory means downloading it again.

    A margin on top because the filesystem needs somewhere to put the metadata, and because a
    stamp that fits with nothing to spare is a stamp that fails on the next run.
    """
    if not shards:
        return
    largest = max(os.path.getsize(os.path.join(directory, n)) for n in shards)
    need = largest + (1 << 26)                      # 64 MiB of elbow room
    try:
        free = shutil.disk_usage(directory).free
    except OSError as e:
        log(f"  provenance: cannot measure free space on this filesystem ({e}), so the "
            f"{need / 1e9:.1f} GB pre-flight was skipped, not passed")
        return
    if free < need:
        raise OSError(
            f"stamping needs {need / 1e9:.2f} GB free beside the checkpoint (the largest shard "
            f"is {largest / 1e9:.2f} GB and is copied before being renamed over), and "
            f"{free / 1e9:.2f} GB is available. Free space and re-run the stamp; the weights "
            f"themselves are already written and are not affected.")


def stamp_safetensors(directory, fields, log=print):
    """Stamp every safetensors shard in a saved model directory. Returns how many it touched."""
    shards = sorted(n for n in os.listdir(directory) if n.endswith(".safetensors"))
    _preflight_room(directory, shards, log)
    # NAME THE PARTIAL STATE. Stopping part way leaves earlier shards stamped and later ones
    # not, and a directory that disagrees with itself cannot say what was done to it. The
    # try sits outside the loop, which also means the first failure ends the stamp rather than
    # the run limping on to leave a wider gap.
    done = 0
    try:
        for name in shards:
            _rewrite_header(os.path.join(directory, name), fields)
            done += 1
    except OSError as e:
        raise OSError(
            f"stamping failed on {shards[done]} after {done} of {len(shards)} shard(s) were "
            f"already stamped, so this directory is now PART marked: {e}. Re-run the stamp "
            f"once there is room; stamping an already-stamped shard is safe."
        ) from e
    log(f"  provenance: stamped {len(shards)} safetensors shard(s)")
    return len(shards)


def stamp_config(directory, fields):
    """Add the marker to `config.json`, which every loader reads by construction."""
    path = os.path.join(directory, "config.json")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc[NAMESPACE] = fields
    tmp = f"{path}.stamping"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)


def read(directory):
    """Whatever this checkpoint says about itself, preferring the header over the sibling file.

    The header is preferred because it is the copy that cannot be separated from the tensors, and
    a disagreement between the two is worth surfacing rather than resolving silently.
    """
    from safetensors import safe_open  # imported here so the module stays torch-free
    # EVERY SHARD IS READ, not the first one carrying a marker. Stamping rewrites shards one at
    # a time, so an interrupted stamp leaves some marked and some not; answering from the first
    # hit reports such a directory as fully marked, which for a PARTIAL abliteration is exactly
    # the thing this module exists to prevent.
    marked, unmarked, first = {}, [], None
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(directory, name), framework="pt") as f:
            meta = f.metadata() or {}
        found = {k[len(NAMESPACE) + 1:]: v for k, v in meta.items()
                 if k.startswith(f"{NAMESPACE}.")}
        if found:
            marked[name] = found
            first = first if first is not None else found
        else:
            unmarked.append(name)
    if marked and unmarked:
        raise PartialStampError(
            f"{len(marked)} of {len(marked) + len(unmarked)} shard(s) in {directory} carry a "
            f"provenance marker and the rest do not, so this checkpoint cannot say what was "
            f"done to it. That is what an interrupted stamp leaves behind. Unmarked: "
            f"{', '.join(unmarked[:3])}{' ...' if len(unmarked) > 3 else ''}. Re-run the stamp.")
    if first is not None:
        return first
    path = os.path.join(directory, "config.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get(NAMESPACE)
    return None


def stamp(directory, fields, required=False, log=print):
    """Write the marker to both places.

    `required` is the partial-ablation case and is the whole reason this is not best-effort
    everywhere: a model whose refusal behaviour is half removed and that cannot say so is the
    thing this module exists to prevent, so it takes the run down rather than shipping.
    """
    try:
        stamp_config(directory, fields)
        stamp_safetensors(directory, fields, log=log)
    except (OSError, ValueError, KeyError, struct.error) as e:
        if required:
            raise RuntimeError(
                f"could not write the provenance marker into {directory}: {e}. This model is a "
                f"PARTIAL abliteration and refusing to leave it unmarked, because an unmarked "
                f"partial model is indistinguishable from a whole one.") from e
        log(f"  provenance: WARNING, could not stamp this checkpoint ({e}). The model is saved "
            f"and usable; it just does not carry a record of what produced it.")
        return False
    return True
