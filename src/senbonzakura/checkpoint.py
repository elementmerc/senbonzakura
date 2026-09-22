# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What a checkpoint directory has to look like before anything is allowed to load it.

WHY THIS EXISTS

This tool's job is downloading other people's model weights and loading them. "Untrusted input
from a stranger" is not an edge case here, it is the main path, and the format has a field that
names files to open.

A sharded checkpoint carries an index, `model.safetensors.index.json` or its `.bin` equivalent,
whose `weight_map` maps every tensor name to the file holding it. The loader reads those names
and opens them relative to the checkpoint directory. `accelerate` PYSEC-2026-3804 is exactly
this: `load_checkpoint_in_model` and `load_checkpoint_and_dispatch` do not sanitise the entries,
so an index naming `../../../etc/passwd` or an absolute path sends the loader outside the
directory it was pointed at. There is no fixed release at the time of writing, and we reach that
code through `device_map`, so the only lever we control is refusing the checkpoint first.

WHAT IS CHECKED, AND WHY IT IS THE DECLARED STRING RATHER THAN THE RESOLVED PATH

The entries are validated AS WRITTEN. It is tempting to resolve each one and assert it lands
inside the directory, and that would be wrong here: `huggingface_hub` populates a snapshot with
SYMLINKS into a shared `blobs/` store, so every shard of a normally downloaded model resolves
outside its own directory. A resolve-and-contain check would refuse every model from the Hub
cache while passing a crafted index whose traversal happens to land back inside, which is the
worst of both. What the advisory is about is the path the index DECLARES, so that is what this
reads.

WHAT IT DOES NOT COVER, stated because a guard whose reach is assumed is worse than none

It needs a directory. When a model id is handed straight to `transformers`, the download and the
load happen inside somebody else's call and there is no point between them to stand. Closing
that needs the snapshot to be fetched deliberately and the local path passed on, which is a
change to how models are fetched rather than a check, and it is recorded in DEFERRED.md.
"""
import json
import posixpath
from pathlib import Path

#: The index files a checkpoint can carry. Both shapes exist in the wild: safetensors is the
#: modern one and `pytorch_model.bin.index.json` is still published by older repositories, and a
#: guard that read one of them would be the same defect this project keeps finding.
INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


class UnsafeCheckpointError(Exception):
    """A checkpoint asked for something a checkpoint is not allowed to ask for."""


def unsafe_entry(name):
    """Why this `weight_map` value is not a plain file beside the index, or None if it is.

    Returns a sentence rather than a bool so the refusal can say which rule was broken. Every
    branch here is a shape that makes a loader open a file the checkpoint's own directory does
    not contain.
    """
    if not isinstance(name, str) or not name:
        return "is not a filename"
    # Backslashes first: on Windows `..\\..\\x` traverses, and on POSIX the whole thing is one
    # odd but legal filename, so a check that only split on "/" would pass it on one platform
    # and not the other. A shard name has no business containing one either way.
    if "\\" in name:
        return "contains a backslash, which traverses on Windows"
    if name.startswith("/") or posixpath.isabs(name):
        return "is an absolute path"
    # A Windows drive-relative name ("C:x") is not caught by isabs on POSIX.
    if len(name) > 1 and name[1] == ":":
        return "names a drive"
    parts = name.split("/")
    if ".." in parts:
        return "walks up out of the checkpoint directory"
    if any(p in ("", ".") for p in parts[:-1]) or name.endswith("/"):
        return "is not a plain relative filename"
    return None


def refuse_unsafe_index(model_dir):
    """Raise before a loader is pointed at a checkpoint whose index names files outside it.

    Silent on a checkpoint with no index: a single-file model has no `weight_map` and nothing to
    validate. Silent, too, on an index that cannot be read here, because deciding what a
    malformed index means belongs to the loader that has to use it; this refuses the one thing it
    can judge, which is an entry that is a traversal whatever the rest of the file says.
    """
    d = Path(model_dir)
    for index_name in INDEX_NAMES:
        index = d / index_name
        if not index.is_file():
            continue
        try:
            doc = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        weight_map = doc.get("weight_map") if isinstance(doc, dict) else None
        if not isinstance(weight_map, dict):
            continue
        # EVERY OFFENDING ENTRY, not the first. A crafted index carries more than one, and a
        # refusal naming one of them invites fixing that one and running it again.
        bad = []
        for tensor, target in sorted(weight_map.items()):
            why = unsafe_entry(target)
            if why:
                bad.append(f"  {tensor!r} -> {target!r}: {why}")
        if bad:
            listed = "\n".join(bad[:10])
            more = f"\n  ... and {len(bad) - 10} more" if len(bad) > 10 else ""
            raise UnsafeCheckpointError(
                f"{index} names files outside the checkpoint directory, so loading it would read "
                f"from somewhere you did not point this at:\n{listed}{more}\n"
                f"A shard name must be a plain filename beside the index. This is refused rather "
                f"than sanitised, because a checkpoint that asks for this is not one with a typo "
                f"in it.")
