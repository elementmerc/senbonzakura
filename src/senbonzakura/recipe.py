# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Write an abliteration recipe for an edit senbonzakura has just baked, in profile core/1.

A recipe is the small JSON file that says what a bake removed and how, so that somebody else can
reproduce the same edit with the reference applier (`recipes/apply_recipe.py`, Apache-2.0) without
running this tool. The format is specified in `private/plans/13-the-recipe-format.md`.

WHY THIS READS THE DIRECTIONS FILE AND NOT THE RUN

The recipe has to describe the edit that was actually applied, so every field that affects the
weights is taken from the same inputs the bake used: the directions file's own bytes (for its hash
and direction count) and the arguments the bake was given. A recipe written from memory of a run
would be a second account of the same edit, and it would drift.

WHAT IT REFUSES

A bake whose settings the profile cannot express is refused rather than recorded as if it were the
standard edit. Sparsity and rounds are the two that exist in this tool and not in core/1.
"""

import hashlib
import json
import pathlib
import re

from . import crashsafe, streambake

#: The schema string the reference applier reads. The major version is what it checks.
SCHEMA = "abliteration-recipe/1"

#: The profile this writer claims. Every field it emits is one core/1 defines.
PROFILE = "core/1"

#: The layer-to-direction mapping and the target names this writer emits. Written here so the
#: recipe and the bake agree on which tensors are edited, by construction.
ATTN_TARGET = "model.layers.*.self_attn.o_proj.weight"
MLP_TARGET = "model.layers.*.mlp.down_proj.weight"

#: The residual metric and its definition. An applier and a writer must not disagree about which
#: figure this is, so the definition is written out in every recipe that records one.
RESIDUAL_METRIC = "max_abs_projection"
RESIDUAL_DEFINITION = (
    "max over the edited tensors, over output rows, of |R_layer W| in float64, where R_layer is "
    "the layer's non-empty direction rows and W is the tensor before the edit (before) or after it "
    "(after)"
)


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
    return None, f"{len(shards)} shards and no index file, so no single hash pins the checkpoint"


def _layer_of(name, targets):
    for pattern in targets:
        pre, post = pattern.split("*")
        m = re.match("^" + re.escape(pre) + "([0-9]+)" + re.escape(post) + "$", name)
        if m:
            return int(m.group(1))
    return None


def _max_abs_projection(tensor, R):
    import numpy as np

    if R.shape[0] == 0:
        return 0.0
    W = tensor.detach().to("cpu").float().numpy().astype(np.float64)
    return float(np.abs(R @ W).max())


def measure_residual(source_dir, baked_dir, directions_path, *, layer_count, targets):
    """The residual figures for a bake, computed from the source and baked checkpoints in float64.

    This is the writer's implementation of the residual definition in core/1. The reference applier
    computes the same figures on its own while it edits, and refuses a recipe whose recorded figures
    disagree with its own. Two implementations agreeing on a number is the cross-check; neither is
    trusted alone.
    """
    import numpy as np
    from safetensors.torch import load_file

    arr, _meta = streambake.load_directions(directions_path)
    before_max = after_max = 0.0
    edited = 0
    for shard in sorted(pathlib.Path(source_dir).glob("*.safetensors")):
        before_tensors = load_file(str(shard))
        after_tensors = load_file(str(pathlib.Path(baked_dir) / shard.name))
        for name, source in before_tensors.items():
            layer = _layer_of(name, targets)
            if layer is None or not 0 <= layer < layer_count:
                continue
            R = np.asarray(arr[layer + 1], dtype=np.float64)
            R = R[np.linalg.norm(R, axis=1) > streambake.DIRECTION_EPSILON]
            if source.shape[0] != arr.shape[2]:
                raise RecipeError(
                    f"{name!r} has output width {source.shape[0]}, but the direction is "
                    f"{arr.shape[2]} wide. This target does not write to the residual stream, so "
                    f"the direction does not act on it and no residual can be measured for it."
                )
            before_max = max(before_max, _max_abs_projection(source, R))
            after_max = max(after_max, _max_abs_projection(after_tensors[name], R))
            edited += 1
    if edited == 0 or before_max <= 0:
        raise RecipeError("no edited tensor carries the direction, so there is no residual to measure")
    return {
        "status": "measured",
        "metric": RESIDUAL_METRIC,
        "definition": RESIDUAL_DEFINITION,
        "before": before_max,
        "after": after_max,
        "fraction_remaining": after_max / before_max,
        "rounds": 0,
    }


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
    sparsity=0.0,
    rounds=0,
    residual=None,
):
    """The recipe dictionary for one bake, in profile core/1.

    `measured` is the `measured` block itself, or None, which records that the edit was not measured
    and says why. `residual` is the figures from `measure_residual`, or None, which records that the
    residual was not measured and says why. Both follow the same not_measured convention.

    A bake with sparsity or rounds is refused rather than recorded. core/1 has no field for either,
    and a recipe that omitted them would describe a different edit from the one that was baked.
    """
    if sparsity != 0.0:
        raise RecipeError(
            f"sparsity {sparsity} is not expressible in profile {PROFILE}. Recording this bake "
            f"would describe a dense edit it did not make."
        )
    if rounds != 0:
        raise RecipeError(
            f"rounds {rounds} is not expressible in profile {PROFILE}. Recording this bake would "
            f"describe a single-pass edit it did not make."
        )
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
        measured_block = {
            "status": "not_measured",
            "why": "the edit was written without an evaluation run attached",
        }
    else:
        measured_block = measured

    if residual is None:
        residual_block = {
            "status": "not_measured",
            "why": "the residual was not measured when this recipe was written",
        }
    else:
        residual_block = residual

    return {
        "schema": SCHEMA,
        "profile": PROFILE,
        "name": str(name),
        "base": base,
        "extensions": [],
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
            "direction_residual": residual_block,
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
