# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Write an abliteration recipe for an edit senbonzakura has just baked.

A recipe is the small JSON file that says what a bake removed and how, so that somebody else can
reproduce the same edit with the reference applier (`recipes/apply_recipe.py`, Apache-2.0) without
running this tool. The format is specified in `private/plans/13-the-recipe-format.md`.

WHY THIS READS THE DIRECTIONS FILE AND NOT THE RUN

The recipe has to describe the edit that was actually applied, so every field that affects the
weights is taken from the same inputs the bake used: the directions file's own bytes (for its hash
and direction count) and the arguments the bake was given. A recipe written from memory of a run
would be a second account of the same edit, and it would drift.

WHAT IT REFUSES

A bake whose settings the recipe format cannot express is refused rather than recorded as if it
were the standard edit. The applier implements one convention set; a recipe claiming that set for
an edit made with a different one would be a recipe that applies to a different model.
"""

import hashlib
import json
import pathlib

from . import crashsafe, streambake

#: The schema string the reference applier reads. The major version is what it checks.
SCHEMA = "abliteration-recipe/1"

#: The layer-to-direction mapping and the target names this writer emits. Written here so the
#: recipe and the bake agree on which tensors are edited, by construction.
ATTN_TARGET = "model.layers.*.self_attn.o_proj.weight"
MLP_TARGET = "model.layers.*.mlp.down_proj.weight"


class RecipeError(Exception):
    """A recipe that cannot be written honestly, with a sentence a person can act on."""


def _sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def base_hash(model_dir):
    """`(sha256 or None, reason or None)` for the checkpoint's index, as the recipe records it.

    The index file is hashed when there is one, because it names every shard. A single-shard
    checkpoint has no index, so its one `.safetensors` file is hashed instead. Several shards with
    no index cannot be pinned by one hash, and the reason says so rather than hashing a guess.
    """
    model_dir = pathlib.Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return _sha256(index), None
    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) == 1:
        return _sha256(shards[0]), None
    return None, (f"{len(shards)} shards and no index file, so no single hash pins the checkpoint")


def build(
    *,
    name,
    model_dir,
    model_id,
    revision,
    directions_path,
    layer_count,
    strength,
    restore_norms,
    measured,
    command,
    edit_tensors=("attn", "mlp"),
):
    """The recipe dictionary for one bake.

    `measured` is either the `measured` block itself or None, which records that the edit was not
    measured and says why, rather than leaving the field out.
    """
    arr, meta = streambake.load_directions(directions_path)
    positions, K, hidden = arr.shape
    if positions < layer_count + 1:
        raise RecipeError(
            f"the direction set covers {positions} positions and {layer_count} layers need "
            f"{layer_count + 1}. A recipe for it would describe layers it cannot edit."
        )
    counts = streambake.real_counts(arr)
    recorded_found = int(meta.get("directions_found", sum(counts)))
    if recorded_found != sum(counts):
        raise RecipeError(
            f"the directions file records {recorded_found} directions and holds {sum(counts)}. "
            f"Refusing to write a recipe from a file that disagrees with itself."
        )

    targets = []
    if "attn" in edit_tensors:
        targets.append(ATTN_TARGET)
    if "mlp" in edit_tensors:
        targets.append(MLP_TARGET)

    sha, reason = base_hash(model_dir)
    base = {"id": str(model_id), "revision": revision, "weights_sha256": sha}
    if sha is None:
        base["weights_sha256_reason"] = reason

    if measured is None:
        measured_block = {"status": "not_measured", "why": "the edit was written without an evaluation run attached"}
    else:
        measured_block = measured

    return {
        "schema": SCHEMA,
        "name": str(name),
        "base": base,
        "edit": {
            "method": "projection",
            "tensor_naming": "huggingface",
            "projection_side": "output",
            "sign_convention": "projection",
            "restore_norms": bool(restore_norms),
            "dtype": "float32",
            "targets": targets,
            "layers": {"from": 0, "to": layer_count - 1},
            "strength": float(strength),
            "directions": {
                "encoding": "safetensors",
                "file": pathlib.Path(directions_path).name,
                "sha256": _sha256(directions_path),
                "shape": [positions, K, hidden],
                "directions_found": sum(counts),
                "directions_found_per_position": counts,
            },
        },
        "measured": measured_block,
        "reproduce": {"tool": "senbonzakura", "command": str(command)},
    }


def write(path, recipe):
    """Atomically, through the same writer as the rest of this project's artefacts."""
    text = json.dumps(recipe, indent=2, sort_keys=True) + "\n"
    with crashsafe.atomic_write(path, binary=False) as fh:
        fh.write(text)
    return pathlib.Path(path)
