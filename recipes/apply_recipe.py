# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Apply an abliteration recipe to a safetensors checkpoint, and write the edited weights.

This is the reference applier for the abliteration recipe format. It is deliberately small so a
stranger can read it end to end in one sitting and decide whether it does what the recipe says.
It depends on numpy and the Python standard library only. It does not import senbonzakura.

WHAT IT DOES

    recipe.json + checkpoint directory + directions.safetensors  ->  edited checkpoint directory

For every tensor named by a recipe target, it removes the directions for that layer from the
tensor's output side, and writes the result back with the same dtype, shape and file layout.

WHAT IT REFUSES, AND WHY REFUSING IS THE POINT

A recipe that is applied with a silently different reading is worse than an error. The same
numbers can describe two different edits, and the only thing that tells them apart is the
convention fields. So the applier stops rather than guesses on:

  - an unknown schema major version, or an unknown key inside `edit`
  - any convention value it does not implement
  - a directions file whose hash, shape or direction count does not match the recipe
  - directions that are not orthonormal, because the projection is only a projection for an
    orthonormal basis
  - a target pattern that matches no tensor in the requested layers
  - a base checkpoint whose index hash does not match the recipe's `base.weights_sha256`
  - a tensor whose axis does not carry the hidden size, so there is nothing to project

CONVENTIONS THIS APPLIER IMPLEMENTS

  - Layer `i` is edited with direction position `i + 1`. Position 0 is the embedding output.
  - A target is a pattern with exactly one `*`, which stands for the layer index. Example:
    `model.layers.*.mlp.down_proj.weight`.
  - `layers.from` and `layers.to` are inclusive.
  - A direction row whose length is at most 1e-6 is an unused slot and is ignored.
  - Non-zero direction rows must be orthonormal to within 1e-4.
  - Arithmetic is float32. The output is cast back to the tensor's stored dtype (float32,
    float16 or bfloat16), with round to nearest, ties to even.
  - `restore_norms: true` restores each output row to its original L2 length (the norm over the
    input axis). `restore_norms: false` removes the direction and lets row lengths fall.

The directions file is a safetensors file holding one float32 tensor named `dirs_per_position`
with shape [positions, K, hidden].

USAGE

    python apply_recipe.py --check  recipe.json MODEL_DIR DIRECTIONS
    python apply_recipe.py          recipe.json MODEL_DIR DIRECTIONS OUT_DIR

`--check` validates everything that can be checked without writing a weight. The full run runs
the same checks first, so a run that starts has already passed them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import struct
import sys

import numpy as np

SCHEMA_NAME = "abliteration-recipe"
SCHEMA_MAJOR = 1
DIRECTIONS_KEY = "dirs_per_position"
DIRECTION_EPSILON = 1e-6
ORTHONORMAL_TOL = 1e-4
NORM_FLOOR = 1e-8
MAX_HEADER_BYTES = 100 * 1024 * 1024

#: The `edit` keys this applier implements. Anything else under `edit` is refused, because a key
#: the applier does not understand is a key whose effect it would silently drop.
EDIT_KEYS = frozenset(
    {
        "method",
        "tensor_naming",
        "projection_side",
        "sign_convention",
        "restore_norms",
        "dtype",
        "targets",
        "layers",
        "strength",
        "directions",
    }
)
DIRECTIONS_KEYS = frozenset(
    {"encoding", "file", "sha256", "shape", "directions_found", "directions_found_per_position"}
)

#: safetensors dtype name -> bytes per element. BF16 has no numpy dtype, so it is decoded by hand.
ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}


class RecipeError(Exception):
    """A refusal. The message is a sentence a person can act on."""


# ── safetensors container, read and written directly ─────────────────────────────────────────


def read_safetensors_header(path):
    """`(header, data_start, file_size)` for a safetensors file, validated.

    Validated because a header is the only thing that says where each tensor's bytes are, and a
    bad header read as-is would edit the wrong bytes without any error.
    """
    path = pathlib.Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise RecipeError(f"{path} is {size} bytes, too short to be a safetensors file.")
        (n,) = struct.unpack("<Q", prefix)
        if n == 0 or n > MAX_HEADER_BYTES or 8 + n > size:
            raise RecipeError(f"{path} declares a {n} byte header, which is outside what a safetensors file can hold.")
        raw = f.read(n)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise RecipeError(f"{path} has a header that is not JSON ({e}).") from e
    if not isinstance(header, dict):
        raise RecipeError(f"{path} has a header that is not an object.")
    data_start = 8 + n
    data_len = size - data_start

    spans = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise RecipeError(f"{path}: tensor {name!r} has a header entry that is not an object.")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in ITEMSIZE:
            raise RecipeError(
                f"{path}: tensor {name!r} has dtype {dtype!r}, which this applier "
                f"does not read. It reads F32, F16 and BF16."
            )
        if (
            not isinstance(shape, list)
            or not all(isinstance(d, int) and d >= 0 for d in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            raise RecipeError(f"{path}: tensor {name!r} has a malformed shape or offset.")
        begin, end = offsets
        elements = math.prod(shape)
        if end < begin or end - begin != elements * ITEMSIZE[dtype]:
            raise RecipeError(
                f"{path}: tensor {name!r} claims {end - begin} bytes for shape "
                f"{shape} of {dtype}, which is {elements * ITEMSIZE[dtype]}."
            )
        spans.append((begin, end, name))
    # The spans must tile the data region exactly: no gap, no overlap, nothing past the end.
    spans.sort()
    cursor = 0
    for begin, end, name in spans:
        if begin != cursor:
            raise RecipeError(
                f"{path}: tensor {name!r} starts at byte {begin} where byte "
                f"{cursor} was expected. The offsets are not contiguous."
            )
        cursor = end
    if cursor != data_len:
        raise RecipeError(f"{path}: the tensors cover {cursor} bytes and the file holds {data_len} after the header.")
    return header, data_start, size


def read_tensor_bytes(f, data_start, entry):
    begin, end = entry["data_offsets"]
    f.seek(data_start + begin)
    raw = f.read(end - begin)
    if len(raw) != end - begin:
        raise RecipeError("a tensor's bytes ended early, so the file was truncated while reading.")
    return raw


def decode(raw, dtype, shape):
    """Raw safetensors bytes to a float32 array."""
    if dtype == "F32":
        values = np.frombuffer(raw, dtype="<f4")
    elif dtype == "F16":
        values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        values = bits.view(np.float32)
    else:
        raise RecipeError(f"no decoder for {dtype}")
    return values.reshape(shape).astype(np.float32, copy=True)


def encode(values, dtype):
    """float32 array to the raw little-endian bytes of `dtype`, rounding to nearest, ties to even.

    Rounding is spelt out for BF16 because numpy has no bfloat16. The rule is the one PyTorch and
    the hardware use, so a bfloat16 written here matches one written by a framework.
    """
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if dtype == "F32":
        return flat.astype("<f4").tobytes()
    if dtype == "F16":
        return flat.astype("<f2").tobytes()
    if dtype == "BF16":
        bits = flat.view(np.uint32)
        rounded = (bits + (((bits >> 16) & 1) + np.uint32(0x7FFF))) >> 16
        return rounded.astype("<u2").tobytes()
    raise RecipeError(f"no encoder for {dtype}")


# ── the recipe ───────────────────────────────────────────────────────────────────────────────


def load_recipe(path):
    path = pathlib.Path(path)
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RecipeError(f"{path} could not be read as JSON ({e}).") from e
    if not isinstance(recipe, dict):
        raise RecipeError(f"{path} is not a JSON object.")

    schema = recipe.get("schema")
    match = re.fullmatch(rf"{SCHEMA_NAME}/([0-9]+)", schema) if isinstance(schema, str) else None
    if match is None:
        raise RecipeError(f"the schema is {schema!r}, which is not an {SCHEMA_NAME} recipe.")
    if int(match.group(1)) != SCHEMA_MAJOR:
        raise RecipeError(
            f"the schema is {schema!r}. This applier reads major version "
            f"{SCHEMA_MAJOR} only, and applying a different major would be a guess."
        )

    for key in ("name", "base", "edit", "measured"):
        if key not in recipe:
            raise RecipeError(f"the recipe has no {key!r} field, which the format requires.")
    check_measured(recipe["measured"])
    return recipe


def check_measured(measured):
    """A measured block must say it was measured, or say why it was not, with `n` on each figure."""
    if not isinstance(measured, dict) or measured.get("status") not in ("measured", "not_measured"):
        raise RecipeError('`measured.status` must be "measured" or "not_measured".')
    if measured["status"] == "not_measured":
        if not isinstance(measured.get("why"), str) or not measured["why"].strip():
            raise RecipeError("a `not_measured` recipe must say why in `measured.why`.")
        return
    for key in ("refusal_before", "refusal_after", "kl"):
        figure = measured.get(key)
        if not isinstance(figure, dict) or not isinstance(figure.get("n"), int):
            raise RecipeError(
                f"`measured.{key}` has no sample size `n`. A figure without `n` cannot be read as a measurement."
            )


def parse_edit(recipe):
    edit = recipe["edit"]
    if not isinstance(edit, dict):
        raise RecipeError("`edit` must be an object.")
    unknown = sorted(set(edit) - EDIT_KEYS)
    if unknown:
        raise RecipeError(
            f"`edit` has keys this applier does not implement: {unknown}. Applying "
            f"the recipe without them would be a different edit, so it is refused."
        )
    expect = {
        "method": "projection",
        "tensor_naming": "huggingface",
        "projection_side": "output",
        "sign_convention": "projection",
        "dtype": "float32",
    }
    for key, value in expect.items():
        if edit.get(key) != value:
            raise RecipeError(f"`edit.{key}` is {edit.get(key)!r}. This applier implements only {value!r} for it.")
    if not isinstance(edit.get("restore_norms"), bool):
        raise RecipeError("`edit.restore_norms` must be true or false, not a string.")
    strength = edit.get("strength")
    if (
        not isinstance(strength, (int, float))
        or isinstance(strength, bool)
        or not math.isfinite(strength)
        or strength < 0
    ):
        raise RecipeError(
            "`edit.strength` must be a finite number, zero or above. A negative "
            "strength adds the direction back, which is a different intervention."
        )
    targets = edit.get("targets")
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
        raise RecipeError("`edit.targets` must be a non-empty list of pattern strings.")
    compiled = []
    for pattern in targets:
        if pattern.count("*") != 1:
            raise RecipeError(f"target {pattern!r} must contain exactly one `*`, which stands for the layer index.")
        pre, post = pattern.split("*")
        compiled.append((pattern, re.compile("^" + re.escape(pre) + "([0-9]+)" + re.escape(post) + "$")))
    layers = edit.get("layers")
    if (
        not isinstance(layers, dict)
        or not isinstance(layers.get("from"), int)
        or not isinstance(layers.get("to"), int)
        or layers["from"] < 0
        or layers["to"] < layers["from"]
    ):
        raise RecipeError("`edit.layers` must hold integers `from` and `to`, with 0 <= from <= to.")
    return {
        "restore_norms": edit["restore_norms"],
        "strength": float(strength),
        "targets": compiled,
        "layers": (layers["from"], layers["to"]),
        "directions": edit.get("directions"),
    }


# ── directions ───────────────────────────────────────────────────────────────────────────────


def sha256_file(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_directions(path, spec):
    """The float32 direction array, checked against what the recipe says it holds."""
    if not isinstance(spec, dict):
        raise RecipeError("`edit.directions` must be an object.")
    unknown = sorted(set(spec) - DIRECTIONS_KEYS)
    if unknown:
        raise RecipeError(f"`edit.directions` has keys this applier does not implement: {unknown}.")
    if spec.get("encoding") != "safetensors":
        raise RecipeError('`edit.directions.encoding` must be "safetensors".')
    actual = sha256_file(path)
    if spec.get("sha256") != actual:
        raise RecipeError(
            f"the directions file hashes to {actual}, and the recipe records "
            f"{spec.get('sha256')!r}. These are not the directions the recipe means."
        )

    header, data_start, _size = read_safetensors_header(path)
    entry = header.get(DIRECTIONS_KEY)
    if not isinstance(entry, dict):
        raise RecipeError(f"{path} holds no tensor named {DIRECTIONS_KEY!r}, so it is not a direction set.")
    if entry.get("dtype") != "F32":
        raise RecipeError(
            f"the direction tensor is {entry.get('dtype')}, and this applier reads float32 directions only."
        )
    shape = entry.get("shape")
    if shape != spec.get("shape"):
        raise RecipeError(f"the directions file has shape {shape} and the recipe records {spec.get('shape')}.")
    if not isinstance(shape, list) or len(shape) != 3:
        raise RecipeError("a direction set has shape [positions, K, hidden].")
    with pathlib.Path(path).open("rb") as f:
        raw = read_tensor_bytes(f, data_start, entry)
    dirs = decode(raw, "F32", shape)
    if not np.isfinite(dirs).all():
        raise RecipeError("the direction set holds a NaN or infinity.")

    counts = [int(c) for c in (np.linalg.norm(dirs, axis=-1) > DIRECTION_EPSILON).sum(axis=-1)]
    if "directions_found" in spec and spec["directions_found"] != sum(counts):
        raise RecipeError(
            f"the recipe says {spec['directions_found']} directions were found and "
            f"the file holds {sum(counts)} non-empty rows."
        )
    if "directions_found_per_position" in spec and spec["directions_found_per_position"] != counts:
        raise RecipeError("the recipe's per-position direction counts do not match the file.")
    return dirs


def check_orthonormal(rows, where):
    live = rows[np.linalg.norm(rows, axis=-1) > DIRECTION_EPSILON]
    if live.shape[0] == 0:
        return
    gram = live.astype(np.float64) @ live.astype(np.float64).T
    deviation = float(np.abs(gram - np.eye(live.shape[0])).max())
    if deviation > ORTHONORMAL_TOL:
        raise RecipeError(
            f"the directions for {where} are not orthonormal (worst deviation "
            f"{deviation:.2e}, tolerance {ORTHONORMAL_TOL:.0e}). The projection "
            f"removes the wrong subspace unless the basis is orthonormal."
        )


# ── the edit ─────────────────────────────────────────────────────────────────────────────────


def project_output_side(W, R, strength, restore_norms):
    """Remove the span of the rows of R from the output side of W, which is shaped [hidden, in].

    With restore_norms, each output row is rescaled to its original length after the projection.
    The rescaling is done on a row-normalised copy, so the result does not depend on the order
    in which the two steps are done.
    """
    if not restore_norms:
        return W - strength * (R.T @ (R @ W))
    row_norm = np.maximum(np.linalg.norm(W, axis=1, keepdims=True), NORM_FLOOR)
    unit = W / row_norm
    unit = unit - strength * (R.T @ (R @ unit))
    unit_norm = np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), NORM_FLOOR)
    return unit / unit_norm * row_norm


def plan_edits(headers, targets, layers, positions):
    """`{tensor name: layer index}` for every tensor the recipe edits, across all shards.

    A pattern is checked against the whole checkpoint, not one shard: a target whose tensors
    happen to live in the second shard must not be refused for matching nothing in the first.
    """
    lo, hi = layers
    plan = {}
    for pattern, rx in targets:
        matched_in_range = 0
        for header in headers:
            for name in header:
                if name == "__metadata__":
                    continue
                m = rx.match(name)
                if m is None:
                    continue
                layer = int(m.group(1))
                if not lo <= layer <= hi:
                    continue
                matched_in_range += 1
                if name in plan and plan[name] != layer:
                    raise RecipeError(
                        f"{name!r} is named by two targets with two layers. "
                        f"The recipe is ambiguous about it, so it is refused."
                    )
                plan[name] = layer
        if matched_in_range == 0:
            raise RecipeError(
                f"target {pattern!r} matches no tensor in layers {lo} to {hi} of "
                f"this checkpoint. A pattern that matches nothing would produce an "
                f"unedited model and report success, so it is refused."
            )
    for layer in sorted(set(plan.values())):
        if layer + 1 >= positions:
            raise RecipeError(
                f"layer {layer} is edited with direction position {layer + 1}, but "
                f"the direction set holds {positions} positions. Its deepest layers "
                f"would go unedited while the run reported success."
            )
    return plan


def find_shards(model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.is_dir():
        raise RecipeError(f"{model_dir} is not a directory holding a checkpoint.")
    shards = sorted(p for p in model_dir.glob("*.safetensors") if p.is_file())
    if not shards:
        raise RecipeError(f"{model_dir} holds no top-level .safetensors shards.")
    nested = [p for p in model_dir.rglob("*.safetensors") if p.parent != model_dir]
    if nested:
        raise RecipeError(
            f"{nested[0]} sits in a subdirectory. This applier reads top-level "
            f"shards only, and would otherwise edit part of a checkpoint."
        )
    return shards


def check_base(recipe, model_dir):
    base = recipe["base"]
    if not isinstance(base, dict):
        raise RecipeError("`base` must be an object.")
    recorded = base.get("weights_sha256")
    if recorded is None:
        if not isinstance(base.get("weights_sha256_reason"), str) or not base["weights_sha256_reason"].strip():
            raise RecipeError("`base.weights_sha256` is null and `base.weights_sha256_reason` does not say why.")
        return
    index = pathlib.Path(model_dir) / "model.safetensors.index.json"
    if index.is_file():
        actual = sha256_file(index)
    else:
        shards = find_shards(model_dir)
        if len(shards) != 1:
            raise RecipeError(
                "the base hash names the checkpoint index, and this checkpoint has several shards and no index to hash."
            )
        actual = sha256_file(shards[0])
    if actual != recorded:
        raise RecipeError(
            f"the base checkpoint hashes to {actual}, and the recipe was made from "
            f"{recorded}. Applying it would edit a different model."
        )


def apply(recipe_path, model_dir, directions_path, out_dir, *, check_only=False, log=print):
    """Validate everything, then (unless check_only) write the edited checkpoint to out_dir.

    Returns a summary dict. Raises RecipeError with a sentence a person can act on.
    """
    recipe = load_recipe(recipe_path)
    edit = parse_edit(recipe)
    check_base(recipe, model_dir)
    dirs = load_directions(directions_path, edit["directions"])
    positions, K, hidden = dirs.shape
    if positions < edit["layers"][1] + 2:
        raise RecipeError(
            f"layers run to {edit['layers'][1]} and the direction set holds only "
            f"{positions} positions. Layer i reads position i + 1, so this needs "
            f"{edit['layers'][1] + 2}."
        )

    shards = find_shards(model_dir)
    located = {shard: read_safetensors_header(shard) for shard in shards}
    plan = plan_edits([located[s][0] for s in shards], edit["targets"], edit["layers"], positions)
    if not plan:
        raise RecipeError("the recipe edits no tensor in this checkpoint.")

    for layer in sorted(set(plan.values())):
        check_orthonormal(dirs[layer + 1], f"layer {layer}")
    for shard in shards:
        for name, entry in located[shard][0].items():
            if name not in plan:
                continue
            width = entry["shape"]
            if len(width) != 2 or width[0] != hidden:
                raise RecipeError(
                    f"{name!r} has shape {width}. The direction set is {hidden} "
                    f"wide, and an output-side projection needs a 2-D tensor whose "
                    f"first axis is that width."
                )

    summary = {
        "tensors_to_edit": len(plan),
        "layers": sorted(set(plan.values())),
        "directions": f"{positions} positions, K={K}, hidden={hidden}",
        "restore_norms": edit["restore_norms"],
        "strength": edit["strength"],
    }
    log(f"  {len(plan)} tensor(s) to edit across layers {summary['layers']}; {summary['directions']}")
    if check_only:
        summary["checked_only"] = True
        return summary

    out = pathlib.Path(out_dir)
    refuse_unsafe_output(model_dir, out)
    out.mkdir(parents=True, exist_ok=True)

    strength, restore = edit["strength"], edit["restore_norms"]
    edited = 0
    for shard in shards:
        header, data_start, _size = located[shard]
        target = out / shard.name
        part = target.with_name(target.name + ".part")
        try:
            with shard.open("rb") as src, part.open("wb") as dst:
                prefix = src.read(8)
                (n,) = struct.unpack("<Q", prefix)
                src.seek(0)
                dst.write(src.read(8 + n))
                for name, entry in sorted(
                    ((k, v) for k, v in header.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0]
                ):
                    raw = read_tensor_bytes(src, data_start, entry)
                    if name in plan:
                        W = decode(raw, entry["dtype"], entry["shape"])
                        if not np.isfinite(W).all():
                            raise RecipeError(f"{name!r} holds a NaN or infinity in the source.")
                        R = dirs[plan[name] + 1]
                        new = project_output_side(
                            W.astype(np.float32), R.astype(np.float32), np.float32(strength), restore
                        )
                        raw = encode(new, entry["dtype"])
                        edited += 1
                    dst.write(raw)
            os.replace(part, target)
        finally:
            if part.exists():
                part.unlink()
        log(f"  wrote {shard.name}")

    for item in pathlib.Path(model_dir).iterdir():
        if item.is_file() and not item.name.endswith(".safetensors"):
            shutil.copyfile(item, out / item.name)
    summary["tensors_edited"] = edited
    summary["checked_only"] = False
    return summary


def refuse_unsafe_output(model_dir, out_dir):
    """The output must not be the model, inside it, or contain it, and must start empty."""
    src = pathlib.Path(model_dir).resolve()
    dst = pathlib.Path(out_dir).resolve()
    if src == dst or dst.is_relative_to(src) or src.is_relative_to(dst):
        raise RecipeError(
            f"the output {dst} overlaps the model directory {src}. The edit writes "
            f"a new copy, so the two must be separate."
        )
    if dst.exists() and any(dst.iterdir()):
        raise RecipeError(
            f"the output directory {dst} already holds files. Choose an empty "
            f"directory, so nothing is overwritten by accident."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="apply_recipe.py", description="Apply an abliteration recipe to a checkpoint."
    )
    parser.add_argument(
        "--check", action="store_true", help="validate the recipe and inputs without writing any weights"
    )
    parser.add_argument("recipe")
    parser.add_argument("model_dir")
    parser.add_argument("directions")
    parser.add_argument("out_dir", nargs="?")
    args = parser.parse_args(argv)
    if not args.check and args.out_dir is None:
        parser.error("an output directory is needed unless --check is given")
    try:
        summary = apply(
            args.recipe,
            args.model_dir,
            args.directions,
            args.out_dir,
            check_only=args.check or args.out_dir is None,
            log=lambda s: print(s, file=sys.stderr),
        )
    except RecipeError as e:
        print(f"apply_recipe: {e}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
