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
import json
import os
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
    # Rename last, so an interrupted stamp leaves the original shard intact rather than a
    # half-written one. The whole point of this module is that a checkpoint says what it is, and a
    # truncated checkpoint that says so is not an improvement.
    os.replace(tmp, path)


def stamp_safetensors(directory, fields, log=print):
    """Stamp every safetensors shard in a saved model directory. Returns how many it touched."""
    shards = sorted(n for n in os.listdir(directory) if n.endswith(".safetensors"))
    for name in shards:
        _rewrite_header(os.path.join(directory, name), fields)
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
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(directory, name), framework="pt") as f:
            meta = f.metadata() or {}
        found = {k[len(NAMESPACE) + 1:]: v for k, v in meta.items()
                 if k.startswith(f"{NAMESPACE}.")}
        if found:
            return found
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
