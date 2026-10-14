#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Apply an abliteration recipe to a safetensors checkpoint, and write the edited weights.

This is the reference applier for the abliteration recipe format, profile `core/1`. It is small
enough to read end to end in one sitting. It depends on numpy and the Python standard library only,
and it does not import senbonzakura.

WHAT IT DOES

    recipe.json + checkpoint directory + directions.safetensors  ->  edited checkpoint directory

For each tensor a recipe targets, it removes the layer's directions from the tensor's output side,
with the strength the recipe gives for that layer and pattern, then writes the tensor back with the
same dtype, shape and file layout. It then checks the residual the recipe recorded.

PROFILES AND EXTENSIONS

A recipe declares the profile it claims in `profile`. This applier implements `core/1` and refuses
any other profile, because it cannot know what an unknown profile promises.

A recipe may also declare extensions, in `extensions`, each with a `id` in the form
`x_<vendor>_<name>` and a boolean `edit_affecting`:

  - An edit-affecting extension changes the weights. This applier refuses it, because applying the
    recipe without it would be a different edit.
  - A descriptive extension only describes the recipe. This applier applies the recipe and reports
    the extension as ignored, so the omission is visible in the output rather than silent.

The `edit` block is closed. A key inside it that this applier does not know is refused, never
ignored, because a dropped key such as `sparsity` silently changes the edit.

REFUSALS

Every refusal carries a stable code (see `RecipeError.code`). The conformance harness in
`recipes/conformance/` checks codes, so a tool's refusal behaviour can be tested the same way as its
output. A refusal prints one JSON line on stdout and exits with status 2.

CONVENTIONS

  - Layer `i` is edited with direction position `i + 1`. Position 0 is the embedding output.
  - A target is a pattern with exactly one `*`, which stands for the layer index, for example
    `model.layers.*.mlp.down_proj.weight`. The pattern must match the whole tensor name.
  - `layers.from` and `layers.to` are inclusive. A tensor is edited only when its name matches a
    target and its layer is in range.
  - A strength is either one number for every layer (`strength`), or a `schedule` giving one list of
    per-layer strengths for each target. Exactly one of the two is present.
  - A direction row with norm at most 1e-6 is an unused slot and is ignored.
  - Non-empty direction rows must have a Gram matrix within 1e-4 of the identity.
  - Arithmetic is float32. Each tensor is written back in its stored dtype (float32, float16 or
    bfloat16), rounding to nearest, ties to even.
  - `restore_norms: true` rescales each output row to its original length, where a row is the norm
    over the input axis. The removal is then approximate, which is why the residual is recorded.
  - The residual, `edit.direction_residual`, is the largest absolute value of R W over the edited
    tensors, measured on the input (`before`) and the output (`after`), in float64. It is required.
    It may say `"not_measured"` with a reason. When measured, this applier recomputes the figures and
    refuses a recipe whose recorded figures disagree.
  - `provenance` is required. It names the tool and version that wrote the recipe and the commit it ran
    from, or says why the commit is unknown.
  - `measured` is required. It either says `"not_measured"` with a reason, or names its instrument
    (scorer, pinned corpus, partition, seed), gives every figure its `n` and an interval, lists what the
    figures do not say, and carries `edit_sha256`, the hash of the `edit` block it was measured on. The
    applier checks that hash, so a figure cannot be attached to an edit it did not come from.

The directions file is a safetensors file holding one float32 tensor named `dirs_per_position`, with
shape [positions, K, hidden].

USAGE

    python apply_recipe.py --check  recipe.json MODEL_DIR DIRECTIONS
    python apply_recipe.py          recipe.json MODEL_DIR DIRECTIONS OUT_DIR
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

PROFILE = "core/1"
SUPPORTED_PROFILES = ("core/1",)
#: Extensions this applier can honour. None yet. An edit-affecting extension outside this set is refused.
IMPLEMENTED_EXTENSIONS = frozenset()
EXTENSION_ID = re.compile(r"^x_[a-z0-9]+_[a-z0-9_]+$")

SCHEMA_NAME = "abliteration-recipe"
SCHEMA_MAJOR = 1
DIRECTIONS_KEY = "dirs_per_position"
DIRECTION_EPSILON = 1e-6
ORTHONORMAL_TOL = 1e-4
NORM_FLOOR = 1e-8
MAX_HEADER_BYTES = 100 * 1024 * 1024
RESIDUAL_METRIC = "max_abs_projection/1"
RESIDUAL_REL_TOL = 1e-3
RESIDUAL_ABS_TOL = 1e-6

#: The `edit` keys of profile core/1. Anything else under `edit` is refused.
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
        "schedule",
        "directions",
        "direction_residual",
    }
)
DIRECTIONS_KEYS = frozenset(
    {"encoding", "file", "sha256", "shape", "directions_found", "directions_found_per_position"}
)

#: safetensors dtype name -> bytes per element. BF16 has no numpy dtype, so it is decoded by hand.
ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2}


class RecipeError(Exception):
    """A refusal. `code` is stable and machine readable; the message is for a person."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


# ── safetensors container, read and written directly ─────────────────────────────────────────


def read_safetensors_header(path):
    """`(header, data_start, file_size)` for a safetensors file, validated."""
    path = pathlib.Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise RecipeError("model-invalid", f"{path} is too short to be a safetensors file.")
            (n,) = struct.unpack("<Q", prefix)
            if n == 0 or n > MAX_HEADER_BYTES or 8 + n > size:
                raise RecipeError(
                    "model-invalid",
                    f"{path} declares a {n} byte header, which is outside what a safetensors file can hold.",
                )
            raw = f.read(n)
    except OSError as e:
        raise RecipeError("io-error", f"{path} could not be read ({e}).") from e
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise RecipeError("model-invalid", f"{path} has a header that is not JSON ({e}).") from e
    if not isinstance(header, dict):
        raise RecipeError("model-invalid", f"{path} has a header that is not an object.")
    data_start = 8 + n
    data_len = size - data_start

    spans = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise RecipeError("model-invalid", f"{path}: tensor {name!r} has a header entry that is not an object.")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in ITEMSIZE:
            raise RecipeError(
                "model-invalid",
                f"{path}: tensor {name!r} has dtype {dtype!r}, which "
                f"this applier does not read. It reads F32, F16 and BF16.",
            )
        if (
            not isinstance(shape, list)
            or not all(isinstance(d, int) and d >= 0 for d in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            raise RecipeError("model-invalid", f"{path}: tensor {name!r} has a malformed shape or offset.")
        begin, end = offsets
        elements = math.prod(shape)
        if end < begin or end - begin != elements * ITEMSIZE[dtype]:
            raise RecipeError(
                "model-invalid",
                f"{path}: tensor {name!r} claims {end - begin} bytes "
                f"for shape {shape} of {dtype}, which is {elements * ITEMSIZE[dtype]}.",
            )
        spans.append((begin, end, name))
    spans.sort()
    cursor = 0
    for begin, end, name in spans:
        if begin != cursor:
            raise RecipeError(
                "model-invalid",
                f"{path}: tensor {name!r} starts at byte {begin} where "
                f"byte {cursor} was expected. The offsets are not contiguous.",
            )
        cursor = end
    if cursor != data_len:
        raise RecipeError(
            "model-invalid", f"{path}: the tensors cover {cursor} bytes and the file holds {data_len} after the header."
        )
    return header, data_start, size


def read_tensor_bytes(f, data_start, entry):
    begin, end = entry["data_offsets"]
    f.seek(data_start + begin)
    raw = f.read(end - begin)
    if len(raw) != end - begin:
        raise RecipeError("io-error", "a tensor's bytes ended early, so the file was truncated.")
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
        raise RecipeError("model-invalid", f"no decoder for {dtype}")
    return values.reshape(shape).astype(np.float32, copy=True)


def encode(values, dtype):
    """float32 array to the raw little-endian bytes of `dtype`, rounding to nearest, ties to even.

    Spelt out for BF16 because numpy has no bfloat16. The rule is the one PyTorch and the hardware use.
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
    raise RecipeError("model-invalid", f"no encoder for {dtype}")


# ── the recipe ───────────────────────────────────────────────────────────────────────────────


def load_recipe(path):
    path = pathlib.Path(path)
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RecipeError("recipe-invalid", f"{path} could not be read as JSON ({e}).") from e
    if not isinstance(recipe, dict):
        raise RecipeError("recipe-invalid", f"{path} is not a JSON object.")

    schema = recipe.get("schema")
    match = re.fullmatch(rf"{SCHEMA_NAME}/([0-9]+)", schema) if isinstance(schema, str) else None
    if match is None:
        raise RecipeError("schema-unknown", f"the schema is {schema!r}, which is not an {SCHEMA_NAME} recipe.")
    if int(match.group(1)) != SCHEMA_MAJOR:
        raise RecipeError(
            "schema-unknown", f"the schema is {schema!r}. This applier reads major version {SCHEMA_MAJOR} only."
        )

    if "profile" not in recipe:
        raise RecipeError(
            "profile-missing",
            "the recipe does not declare a `profile`. A recipe must "
            "say which profile it claims, so an applier knows what it promises.",
        )
    if recipe["profile"] not in SUPPORTED_PROFILES:
        raise RecipeError(
            "profile-unsupported",
            f"the recipe claims profile {recipe['profile']!r}. "
            f"This applier implements {list(SUPPORTED_PROFILES)} and cannot honour "
            f"another profile.",
        )

    for key in ("name", "base", "edit", "measured", "extensions"):
        if key not in recipe:
            raise RecipeError("recipe-invalid", f"the recipe has no {key!r} field, which the format requires.")
    if "provenance" not in recipe:
        raise RecipeError(
            "provenance-missing",
            "the recipe has no `provenance` block. A recipe must say which tool and version wrote it, "
            "so a stranger can tell what produced it.",
        )
    check_provenance(recipe["provenance"])
    check_measured(recipe["measured"], recipe["edit"])
    return recipe


def _text(value):
    return isinstance(value, str) and value.strip() != ""


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def edit_sha256(edit):
    """SHA-256 of the `edit` block in canonical JSON: sorted keys, no spaces, no NaN or infinity.

    A measured block carries this hash so that its figures are bound to the exact edit they were
    taken from. Copying a figure onto a recipe with a different strength or target fails here.
    """
    text = json.dumps(edit, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_provenance(provenance):
    """Which tool wrote the recipe, and from which commit, or why the commit is unknown."""
    if not isinstance(provenance, dict):
        raise RecipeError("provenance-invalid", "`provenance` must be an object.")
    tool = provenance.get("tool")
    if not isinstance(tool, dict) or not _text(tool.get("name")) or not _text(tool.get("version")):
        raise RecipeError(
            "provenance-invalid",
            "`provenance.tool` must name the tool and its version, so a reader can find the code that wrote it.",
        )
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise RecipeError(
            "provenance-invalid", "`provenance.source` must say which commit the tool ran from, or why it cannot."
        )
    commit = source.get("commit")
    if commit is None:
        if not _text(source.get("reason")):
            raise RecipeError(
                "provenance-invalid", "`provenance.source` has no commit, so it must say why in `reason`."
            )
    elif not _text(commit) or not _text(source.get("via")):
        raise RecipeError(
            "provenance-invalid",
            "`provenance.source.commit` must be a commit, and `via` must say where it was read from.",
        )
    if source.get("dirty") not in (True, False, None):
        raise RecipeError("provenance-invalid", "`provenance.source.dirty` must be true, false or null.")


def check_extensions(recipe):
    """`[ids ignored]` for descriptive extensions. Refuses any edit-affecting one it cannot honour."""
    entries = recipe["extensions"]
    if not isinstance(entries, list):
        raise RecipeError("extension-malformed", "`extensions` must be a list.")
    seen = set()
    ignored = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RecipeError("extension-malformed", "every extension must be an object.")
        ext_id = entry.get("id")
        if not isinstance(ext_id, str) or not EXTENSION_ID.match(ext_id):
            raise RecipeError("extension-malformed", f"extension id {ext_id!r} must look like x_<vendor>_<name>.")
        if ext_id in seen:
            raise RecipeError("extension-malformed", f"extension {ext_id!r} is declared twice.")
        seen.add(ext_id)
        if not isinstance(entry.get("edit_affecting"), bool):
            raise RecipeError(
                "extension-malformed", f"extension {ext_id!r} must say whether it is edit_affecting, as true or false."
            )
        if ext_id in IMPLEMENTED_EXTENSIONS:
            continue
        if entry["edit_affecting"]:
            raise RecipeError(
                "extension-unsupported",
                f"the recipe depends on extension {ext_id!r}, "
                f"which changes the weights and which this applier does not implement. "
                f"Applying the recipe without it would be a different edit, so it is "
                f"refused.",
            )
        ignored.append(ext_id)
    return ignored


def check_measured(measured, edit):
    """A measured block says it was measured, with its instrument and figures, or says why not.

    Every figure carries its sample size `n` and an interval that contains the point. A rate must sit
    between 0 and 1. The block must name the instrument precisely enough to rerun it, and it must be
    bound to this edit by `edit_sha256`, so a figure cannot be attached to a recipe it did not measure.
    """
    if not isinstance(measured, dict) or measured.get("status") not in ("measured", "not_measured"):
        raise RecipeError("measured-invalid", '`measured.status` must be "measured" or "not_measured".')
    if measured["status"] == "not_measured":
        if not _text(measured.get("why")):
            raise RecipeError("measured-invalid", "a `not_measured` recipe must say why in `measured.why`.")
        return
    _check_instrument(measured.get("instrument"))
    _check_figure(measured.get("refusal_before"), "measured.refusal_before", rate=True)
    _check_figure(measured.get("refusal_after"), "measured.refusal_after", rate=True)
    _check_figure(measured.get("kl"), "measured.kl", rate=False)
    _check_capability(measured.get("capability"))
    if "self_invalidated" not in measured:
        raise RecipeError(
            "measured-invalid",
            "`measured.self_invalidated` must be present: null when the producing run did not invalidate "
            "itself, otherwise the reason it did.",
        )
    if measured["self_invalidated"] is not None and not _text(measured["self_invalidated"]):
        raise RecipeError("measured-invalid", "`measured.self_invalidated` must be null or a reason.")
    limits = measured.get("limits")
    if not isinstance(limits, list) or not limits or not all(_text(item) for item in limits):
        raise RecipeError(
            "measured-invalid",
            "`measured.limits` must list, in plain words, what the figures do not say. It cannot be empty.",
        )
    if measured.get("edit_sha256") != edit_sha256(edit):
        raise RecipeError(
            "measured-mismatch",
            "`measured.edit_sha256` is not the hash of this recipe's `edit` block. These figures were not "
            "measured on this edit, so they cannot be attached to it.",
        )


def _check_instrument(instrument):
    if not isinstance(instrument, dict):
        raise RecipeError("measured-invalid", "`measured.instrument` must name the scorer, corpus, partition and seed.")
    for key in ("scorer", "partition"):
        if not _text(instrument.get(key)):
            raise RecipeError("measured-invalid", f"`measured.instrument.{key}` must be named.")
    corpus = instrument.get("corpus")
    if not isinstance(corpus, dict) or not _text(corpus.get("id")) or not _text(corpus.get("revision")):
        raise RecipeError(
            "measured-invalid",
            "`measured.instrument.corpus` must hold a pinned `id` and `revision`. A corpus that can change "
            "under the same name cannot back a figure.",
        )
    seed = instrument.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RecipeError(
            "measured-invalid", "`measured.instrument.seed` must be a whole number, so the run can be repeated."
        )


def _check_figure(figure, where, *, rate):
    if not isinstance(figure, dict):
        raise RecipeError("measured-invalid", f"`{where}` must be an object holding point, n and ci.")
    point, n, ci = figure.get("point"), figure.get("n"), figure.get("ci")
    if not _finite(point):
        raise RecipeError("measured-invalid", f"`{where}.point` must be a finite number.")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise RecipeError(
            "measured-invalid",
            f"`{where}` has no sample size `n` of one or more. A figure without `n` cannot be read as a measurement.",
        )
    if not isinstance(ci, list) or len(ci) != 2 or not all(_finite(v) for v in ci) or not ci[0] <= point <= ci[1]:
        raise RecipeError("measured-invalid", f"`{where}.ci` must be [low, high] and must contain the point.")
    if rate and not (ci[0] >= 0 and ci[1] <= 1 and point >= 0 and point <= 1):
        raise RecipeError(
            "measured-invalid", f"`{where}` is a rate, so its point and interval must lie between 0 and 1."
        )


def _check_capability(capability):
    if isinstance(capability, dict):
        if capability.get("status") != "not_measured" or not _text(capability.get("why")):
            raise RecipeError(
                "measured-invalid",
                "`measured.capability` must be a list of benchmark figures, or say `not_measured` with a reason.",
            )
        return
    if not isinstance(capability, list) or not capability:
        raise RecipeError(
            "measured-invalid",
            "`measured.capability` must be a list of benchmark figures, or say `not_measured` with a reason.",
        )
    for entry in capability:
        if not isinstance(entry, dict) or not _text(entry.get("name")):
            raise RecipeError("measured-invalid", "each capability entry must name its benchmark.")
        _check_figure(entry.get("before"), f"capability {entry['name']!r} before", rate=True)
        _check_figure(entry.get("after"), f"capability {entry['name']!r} after", rate=True)


def _finite_nonnegative(value, where):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise RecipeError(
            "strength-invalid",
            f"{where} must be a finite number, zero or above. A "
            f"negative strength adds the direction back, which is a different "
            f"intervention.",
        )
    return float(value)


def parse_edit(recipe):
    edit = recipe["edit"]
    if not isinstance(edit, dict):
        raise RecipeError("recipe-invalid", "`edit` must be an object.")
    unknown = sorted(set(edit) - EDIT_KEYS)
    if unknown:
        raise RecipeError(
            "unknown-edit-key",
            f"`edit` has keys this applier does not implement: "
            f"{unknown}. Applying the recipe without them would be a different edit, "
            f"so it is refused.",
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
            raise RecipeError(
                "convention-unsupported",
                f"`edit.{key}` is {edit.get(key)!r}. This applier implements only {value!r} for it.",
            )
    if not isinstance(edit.get("restore_norms"), bool):
        raise RecipeError("convention-unsupported", "`edit.restore_norms` must be true or false, not a string.")

    targets = edit.get("targets")
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
        raise RecipeError("targets-invalid", "`edit.targets` must be a non-empty list of pattern strings.")
    compiled = []
    for pattern in targets:
        if pattern.count("*") != 1:
            raise RecipeError(
                "targets-invalid", f"target {pattern!r} must contain exactly one `*`, which stands for the layer index."
            )
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
        raise RecipeError(
            "layers-invalid",
            "`edit.layers` must hold integers `from` and `to`, with 0 <= from <= to. Both ends are inclusive.",
        )
    lo, hi = layers["from"], layers["to"]

    has_scalar = "strength" in edit
    has_schedule = "schedule" in edit
    if has_scalar == has_schedule:
        raise RecipeError(
            "strength-invalid",
            "`edit` must hold exactly one of `strength` (one number "
            "for every layer) or `schedule` (a list of per-layer strengths per target).",
        )
    strengths = {}
    if has_scalar:
        scalar = _finite_nonnegative(edit["strength"], "`edit.strength`")
        for pattern in targets:
            strengths[pattern] = [scalar] * (hi - lo + 1)
    else:
        schedule = edit["schedule"]
        if not isinstance(schedule, list):
            raise RecipeError("strength-invalid", "`edit.schedule` must be a list.")
        for entry in schedule:
            if not isinstance(entry, dict) or set(entry) != {"target", "strengths"}:
                raise RecipeError(
                    "strength-invalid", "each `schedule` entry must hold exactly `target` and `strengths`."
                )
            target = entry["target"]
            if target not in targets or target in strengths:
                raise RecipeError(
                    "strength-invalid",
                    f"schedule target {target!r} must name each `edit.targets` pattern exactly once.",
                )
            values = entry["strengths"]
            if not isinstance(values, list) or len(values) != hi - lo + 1:
                raise RecipeError(
                    "strength-invalid",
                    f"schedule for {target!r} must give one strength "
                    f"per layer, {hi - lo + 1} in all, from layer {lo} to {hi}.",
                )
            strengths[target] = [_finite_nonnegative(v, f"a schedule value for {target!r}") for v in values]
        missing = [t for t in targets if t not in strengths]
        if missing:
            raise RecipeError("strength-invalid", f"the schedule gives no strengths for {missing}.")

    return {
        "restore_norms": edit["restore_norms"],
        "targets": compiled,
        "layers": (lo, hi),
        "strengths": strengths,
        "directions": edit.get("directions"),
        "residual": edit.get("direction_residual"),
    }


def check_residual_field(residual):
    """The residual must be present and say measured or not_measured, with a reason when not."""
    if residual is None:
        raise RecipeError(
            "residual-missing",
            "`edit.direction_residual` is missing. Every recipe "
            "records how much of the direction survived, or says why it was not "
            "measured.",
        )
    if not isinstance(residual, dict) or residual.get("status") not in ("measured", "not_measured"):
        raise RecipeError("residual-invalid", '`edit.direction_residual.status` must be "measured" or "not_measured".')
    if residual["status"] == "not_measured":
        if not isinstance(residual.get("why"), str) or not residual["why"].strip():
            raise RecipeError("residual-invalid", "a `not_measured` residual must say why.")
        return None
    if residual.get("metric") != RESIDUAL_METRIC:
        raise RecipeError(
            "residual-invalid",
            f"`metric` is {residual.get('metric')!r}. This applier implements only {RESIDUAL_METRIC!r}. "
            "A different version is a different number, so it is refused rather than compared.",
        )
    for key in ("before", "after", "fraction_remaining"):
        value = residual.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise RecipeError("residual-invalid", f"`direction_residual.{key}` must be a finite number, zero or above.")
    return residual


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
        raise RecipeError("directions-mismatch", "`edit.directions` must be an object.")
    unknown = sorted(set(spec) - DIRECTIONS_KEYS)
    if unknown:
        raise RecipeError(
            "directions-mismatch", f"`edit.directions` has keys this applier does not implement: {unknown}."
        )
    if spec.get("encoding") != "safetensors":
        raise RecipeError("directions-mismatch", '`edit.directions.encoding` must be "safetensors".')
    actual = sha256_file(path)
    if spec.get("sha256") != actual:
        raise RecipeError(
            "directions-mismatch",
            f"the directions file hashes to {actual}, and the "
            f"recipe records {spec.get('sha256')!r}. These are not the directions the "
            f"recipe means.",
        )

    header, data_start, _size = read_safetensors_header(path)
    entry = header.get(DIRECTIONS_KEY)
    if not isinstance(entry, dict):
        raise RecipeError(
            "directions-mismatch", f"{path} holds no tensor named {DIRECTIONS_KEY!r}, so it is not a direction set."
        )
    if entry.get("dtype") != "F32":
        raise RecipeError(
            "directions-mismatch",
            f"the direction tensor is {entry.get('dtype')}, and this applier reads float32 directions only.",
        )
    shape = entry.get("shape")
    if shape != spec.get("shape"):
        raise RecipeError(
            "directions-mismatch", f"the directions file has shape {shape} and the recipe records {spec.get('shape')}."
        )
    if not isinstance(shape, list) or len(shape) != 3:
        raise RecipeError("directions-mismatch", "a direction set has shape [positions, K, hidden].")
    with pathlib.Path(path).open("rb") as f:
        raw = read_tensor_bytes(f, data_start, entry)
    dirs = decode(raw, "F32", shape)
    if not np.isfinite(dirs).all():
        raise RecipeError("directions-mismatch", "the direction set holds a NaN or infinity.")

    counts = [int(c) for c in (np.linalg.norm(dirs, axis=-1) > DIRECTION_EPSILON).sum(axis=-1)]
    if "directions_found" in spec and spec["directions_found"] != sum(counts):
        raise RecipeError(
            "directions-mismatch",
            f"the recipe says {spec['directions_found']} directions "
            f"were found and the file holds {sum(counts)} non-empty rows.",
        )
    if "directions_found_per_position" in spec and spec["directions_found_per_position"] != counts:
        raise RecipeError("directions-mismatch", "the recipe's per-position direction counts do not match the file.")
    return dirs


def check_orthonormal(rows, where):
    live = rows[np.linalg.norm(rows, axis=-1) > DIRECTION_EPSILON]
    if live.shape[0] == 0:
        return
    gram = live.astype(np.float64) @ live.astype(np.float64).T
    deviation = float(np.abs(gram - np.eye(live.shape[0])).max())
    if deviation > ORTHONORMAL_TOL:
        raise RecipeError(
            "directions-not-orthonormal",
            f"the directions for {where} are not "
            f"orthonormal (worst deviation {deviation:.2e}, tolerance "
            f"{ORTHONORMAL_TOL:.0e}). The projection removes the wrong subspace unless "
            f"the basis is orthonormal.",
        )


# ── the edit ─────────────────────────────────────────────────────────────────────────────────


def project_output_side(W, R, strength, restore_norms):
    """Remove the span of the rows of R from the output side of W, which is shaped [hidden, in].

    With restore_norms, each output row is rescaled to its original length after the projection, on
    a row-normalised copy, so the result does not depend on the order of the two steps.
    """
    if not restore_norms:
        return W - strength * (R.T @ (R @ W))
    row_norm = np.maximum(np.linalg.norm(W, axis=1, keepdims=True), NORM_FLOOR)
    unit = W / row_norm
    unit = unit - strength * (R.T @ (R @ unit))
    unit_norm = np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), NORM_FLOOR)
    return unit / unit_norm * row_norm


def residual_of(W, R):
    """Max |R W| in float64, for the residual figure. R is the layer's non-empty rows."""
    if R.shape[0] == 0 or W.size == 0:
        return 0.0
    return float(np.abs(R.astype(np.float64) @ W.astype(np.float64)).max())


def plan_edits(headers, targets, layers, positions):
    """`{tensor name: (layer, target pattern)}` for every tensor the recipe edits, across all shards.

    A pattern is checked against the whole checkpoint, not one shard.
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
                if name in plan and plan[name][0] != layer:
                    raise RecipeError(
                        "targets-invalid",
                        f"{name!r} is named by two targets with two "
                        f"layers. The recipe is ambiguous about it, so it is refused.",
                    )
                plan[name] = (layer, pattern)
        if matched_in_range == 0:
            raise RecipeError(
                "no-target-matches",
                f"target {pattern!r} matches no tensor in layers "
                f"{lo} to {hi} of this checkpoint. A pattern that matches nothing would "
                f"produce an unedited model and report success, so it is refused.",
            )
    for layer in sorted({v[0] for v in plan.values()}):
        if layer + 1 >= positions:
            raise RecipeError(
                "layer-without-direction",
                f"layer {layer} is edited with direction "
                f"position {layer + 1}, but the direction set holds {positions} "
                f"positions.",
            )
    return plan


def find_shards(model_dir):
    model_dir = pathlib.Path(model_dir)
    if not model_dir.is_dir():
        raise RecipeError("model-invalid", f"{model_dir} is not a directory holding a checkpoint.")
    shards = sorted(p for p in model_dir.glob("*.safetensors") if p.is_file())
    if not shards:
        raise RecipeError("model-invalid", f"{model_dir} holds no top-level .safetensors shards.")
    nested = [p for p in model_dir.rglob("*.safetensors") if p.parent != model_dir]
    if nested:
        raise RecipeError(
            "model-invalid", f"{nested[0]} sits in a subdirectory. This applier reads top-level shards only."
        )
    return shards


def check_base(recipe, model_dir):
    base = recipe["base"]
    if not isinstance(base, dict):
        raise RecipeError("recipe-invalid", "`base` must be an object.")
    if not _text(base.get("id")):
        raise RecipeError("base-invalid", "`base.id` must name the base checkpoint.")
    if base.get("revision") is None:
        if not _text(base.get("revision_reason")):
            raise RecipeError(
                "base-invalid",
                "`base.revision` is null and `base.revision_reason` does not say why. A local checkpoint has "
                "no hub revision, so the reason must say what identifies it instead.",
            )
    elif not _text(base.get("revision")):
        raise RecipeError("base-invalid", "`base.revision` must be a hub revision or a git sha, or null with a reason.")
    recorded = base.get("weights_sha256")
    if recorded is None:
        reason = base.get("weights_sha256_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise RecipeError(
                "base-mismatch", "`base.weights_sha256` is null and `base.weights_sha256_reason` does not say why."
            )
        return
    index = pathlib.Path(model_dir) / "model.safetensors.index.json"
    if index.is_file():
        actual = sha256_file(index)
    else:
        shards = find_shards(model_dir)
        if len(shards) != 1:
            raise RecipeError(
                "base-mismatch",
                "the base hash names the checkpoint index, and this "
                "checkpoint has several shards and no index to hash.",
            )
        actual = sha256_file(shards[0])
    if actual != recorded:
        raise RecipeError(
            "base-mismatch",
            f"the base checkpoint hashes to {actual}, and the recipe "
            f"was made from {recorded}. Applying it would edit a different model.",
        )


def apply(recipe_path, model_dir, directions_path, out_dir, *, check_only=False, log=print):
    """Validate everything, then (unless check_only) write the edited checkpoint to out_dir.

    Returns a summary dict. Raises RecipeError, with a stable code and a sentence for a person.
    """
    recipe = load_recipe(recipe_path)
    ignored = check_extensions(recipe)
    edit = parse_edit(recipe)
    recorded_residual = check_residual_field(edit["residual"])
    check_base(recipe, model_dir)
    dirs = load_directions(directions_path, edit["directions"])
    positions, K, hidden = dirs.shape
    if positions < edit["layers"][1] + 2:
        raise RecipeError(
            "layer-without-direction",
            f"layers run to {edit['layers'][1]} and the "
            f"direction set holds only {positions} positions. Layer i reads position "
            f"i + 1, so this needs {edit['layers'][1] + 2}.",
        )

    shards = find_shards(model_dir)
    located = {shard: read_safetensors_header(shard) for shard in shards}
    plan = plan_edits([located[s][0] for s in shards], edit["targets"], edit["layers"], positions)
    if not plan:
        raise RecipeError("no-target-matches", "the recipe edits no tensor in this checkpoint.")

    for layer in sorted({v[0] for v in plan.values()}):
        check_orthonormal(dirs[layer + 1], f"layer {layer}")
    for shard in shards:
        for name, entry in located[shard][0].items():
            if name not in plan:
                continue
            width = entry["shape"]
            if len(width) != 2 or width[0] != hidden:
                raise RecipeError(
                    "layer-without-direction",
                    f"{name!r} has shape {width}. The "
                    f"direction set is {hidden} wide, and an output-side projection "
                    f"needs a 2-D tensor whose first axis is that width.",
                )

    layers_used = sorted({v[0] for v in plan.values()})
    summary = {
        "status": "checked" if check_only else "applied",
        "profile": PROFILE,
        "tensors_to_edit": len(plan),
        "layers": layers_used,
        "directions": f"{positions} positions, K={K}, hidden={hidden}",
        "restore_norms": edit["restore_norms"],
        "ignored_extensions": ignored,
    }
    log(f"  {len(plan)} tensor(s) to edit across layers {layers_used}; {summary['directions']}")
    if check_only:
        return summary

    out = pathlib.Path(out_dir)
    refuse_unsafe_output(model_dir, out)
    out.mkdir(parents=True, exist_ok=True)

    restore = edit["restore_norms"]
    edited = 0
    before_max, after_max = 0.0, 0.0
    written = []
    try:
        for shard in shards:
            header, data_start, _size = located[shard]
            target = out / shard.name
            part = target.with_name(target.name + ".part")
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
                        layer, pattern = plan[name]
                        strength = edit["strengths"][pattern][layer - edit["layers"][0]]
                        if strength > 0:
                            W = decode(raw, entry["dtype"], entry["shape"])
                            _require_finite(W, name)
                            R = dirs[layer + 1]
                            live = R[np.linalg.norm(R, axis=1) > DIRECTION_EPSILON]
                            before_max = max(before_max, residual_of(W, live))
                            new = project_output_side(
                                W.astype(np.float32), R.astype(np.float32), np.float32(strength), restore
                            )
                            after_max = max(after_max, residual_of(new, live))
                            raw = encode(new, entry["dtype"])
                            edited += 1
                    dst.write(raw)
            os.replace(part, target)
            written.append(target)
            log(f"  wrote {shard.name}")
        _require_an_edit(edited)
        residual = _verify_residual(recorded_residual, before_max, after_max)
    except BaseException:
        # A refused or interrupted run leaves no partial output behind: finished shards and any
        # half-written .part file are removed, so the directory cannot be mistaken for an edit.
        for path in written:
            path.unlink(missing_ok=True)
        for part in out.glob("*.part"):
            part.unlink(missing_ok=True)
        raise

    for item in pathlib.Path(model_dir).iterdir():
        if item.is_file() and not item.name.endswith(".safetensors"):
            shutil.copyfile(item, out / item.name)
    summary.update({"tensors_edited": edited, "residual": residual})
    return summary


def _require_finite(W, name):
    if not np.isfinite(W).all():
        raise RecipeError("nonfinite-weight", f"{name!r} holds a NaN or infinity in the source.")


def _require_an_edit(edited):
    if edited == 0:
        raise RecipeError(
            "no-target-matches",
            "every edit has strength zero, so the recipe does nothing and would report a success it did not earn.",
        )


def _verify_residual(recorded, before, after):
    """Compare the figures measured on this run with the recipe's. Refuses a disagreement."""
    figures = {"before": before, "after": after, "fraction_remaining": (after / before) if before > 0 else None}
    if recorded is None:
        return {"measured": False, **figures}
    if before <= RESIDUAL_ABS_TOL:
        raise RecipeError(
            "residual-invalid",
            "the direction is absent from the edited tensors "
            "before the edit, so there is no residual to measure. The recipe edits "
            "nothing it could claim to remove.",
        )
    for key in ("before", "after"):
        measured = figures[key]
        if abs(measured - recorded[key]) > RESIDUAL_ABS_TOL + RESIDUAL_REL_TOL * abs(recorded[key]):
            raise RecipeError(
                "residual-mismatch",
                f"the recipe records residual {key} = "
                f"{recorded[key]:.6g}, and this run measures {measured:.6g}. The "
                f"recipe's claim about what the edit removed is wrong, so the output "
                f"is removed.",
            )
    fraction = figures["fraction_remaining"]
    if abs(fraction - recorded["fraction_remaining"]) > RESIDUAL_REL_TOL + 1e-9:
        raise RecipeError(
            "residual-mismatch",
            f"the recipe records fraction_remaining = "
            f"{recorded['fraction_remaining']:.6g}, and this run measures "
            f"{fraction:.6g}.",
        )
    return {"measured": True, **figures}


def refuse_unsafe_output(model_dir, out_dir):
    """The output must not be the model, inside it, or contain it, and must start empty."""
    src = pathlib.Path(model_dir).resolve()
    dst = pathlib.Path(out_dir).resolve()
    if src == dst or dst.is_relative_to(src) or src.is_relative_to(dst):
        raise RecipeError(
            "output-invalid",
            f"the output {dst} overlaps the model directory {src}. "
            f"The edit writes a new copy, so the two must be separate.",
        )
    if dst.exists() and any(dst.iterdir()):
        raise RecipeError(
            "output-invalid",
            f"the output directory {dst} already holds files. Choose "
            f"an empty directory, so nothing is overwritten by accident.",
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
        print(json.dumps({"status": "refused", "code": e.code, "message": str(e)}))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
