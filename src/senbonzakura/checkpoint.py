# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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

THE HUB CASE, AND WHY IT IS NOT A WHOLE SNAPSHOT

A model id handed straight to `transformers` gets downloaded and loaded inside one call, with no
point between them to stand. The obvious answer, fetching the snapshot ourselves and passing the
local path on, is the wrong one: `snapshot_download` with no patterns takes everything in the
repository, and plenty of repositories publish both `.bin` and `.safetensors` copies of the same
weights, so a guard bolted on that way would double the download of a twenty gigabyte model to
read one small JSON file.

The index IS that one small JSON file, so it is fetched on its own and judged before any weights
move. A repository with no index, a network that is not there, a gated repository, a revision
that does not exist: none of those are this function's business, and all of them leave the load
to proceed exactly as it would have. The check answers one question, and answers it only when it
can: does this checkpoint's index ask a loader to open something outside the checkpoint.
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


def unsafe_entries(index_path):
    """Every offending `weight_map` entry in this index file, as sentences. Empty means clean.

    Returns rather than raises, so the two callers can name the checkpoint the way their own
    caller knows it: a directory on disk, or a repository id that has not been downloaded yet.
    Sharing the judgement is the point. Two copies of a rule about what a path may look like is
    the shape this project keeps finding in its own guards.

    An index that cannot be read counts as clean here. What a malformed index MEANS belongs to
    the loader that has to use it, and raising would turn a truncated download into a security
    refusal, which sends the reader looking for an attacker who is not there.
    """
    try:
        doc = json.loads(Path(index_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    weight_map = doc.get("weight_map") if isinstance(doc, dict) else None
    if not isinstance(weight_map, dict):
        return []
    # EVERY OFFENDING ENTRY, not the first. A crafted index carries more than one, and a refusal
    # naming one of them invites fixing that one and running it again.
    out = []
    for tensor, target in sorted(weight_map.items()):
        why = unsafe_entry(target)
        if why:
            out.append(f"  {tensor!r} -> {target!r}: {why}")
    return out


def _refusal(where, bad):
    listed = "\n".join(bad[:10])
    more = f"\n  ... and {len(bad) - 10} more" if len(bad) > 10 else ""
    return (f"{where} names files outside the checkpoint directory, so loading it would read "
            f"from somewhere you did not point this at:\n{listed}{more}\n"
            f"A shard name must be a plain filename beside the index. This is refused rather "
            f"than sanitised, because a checkpoint that asks for this is not one with a typo "
            f"in it.")


def refuse_unsafe_index(model_dir):
    """Raise before a loader is pointed at a checkpoint directory whose index escapes it.

    Silent on a checkpoint with no index: a single-file model has no `weight_map` and nothing to
    validate.
    """
    d = Path(model_dir)
    for index_name in INDEX_NAMES:
        index = d / index_name
        if not index.is_file():
            continue
        bad = unsafe_entries(index)
        if bad:
            raise UnsafeCheckpointError(_refusal(str(index), bad))


def _fetch_index(repo_id, index_name, revision, token):
    """The index file from a Hub repository, or None when it cannot be had.

    Every failure here is a reason to say nothing rather than to refuse: no such file, not
    authorised, no network, no such revision. None of those is evidence about whether the
    checkpoint is hostile. The catch is broad because the client's failure modes are broad and
    this function has no opinion on any of them.
    """
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=index_name,
                               revision=revision, token=token)
    except Exception:
        return None


def refuse_unsafe_hub_index(repo_id, *, revision=None, token=None):
    """The same judgement for a Hub id, made before any weights are downloaded.

    Fetches only the index, a few kilobytes, so the cost of asking is not measured against the
    size of the model. The obvious alternative, fetching the whole snapshot and checking it on
    disk, would double the download of a repository that publishes both `.bin` and `.safetensors`
    copies of its weights, which many do.

    Raises only `UnsafeCheckpointError`, and only when an index was read and names something
    outside the checkpoint.
    """
    for index_name in INDEX_NAMES:
        got = _fetch_index(repo_id, index_name, revision, token)
        if got is None:
            continue
        bad = unsafe_entries(Path(got))
        if bad:
            raise UnsafeCheckpointError(_refusal(f"{repo_id} ({index_name})", bad))
