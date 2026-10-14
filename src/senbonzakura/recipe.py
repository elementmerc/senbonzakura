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
import math
import pathlib
import re
import shlex

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
RESIDUAL_METRIC = "max_abs_projection/1"
RESIDUAL_DEFINITION = (
    "max over the edited tensors, over output rows, of |R_layer W| in float64, where R_layer is "
    "the layer's non-empty direction rows and W is the tensor before the edit (before) or after it "
    "(after)"
)


class RecipeError(Exception):
    """A recipe that cannot be written honestly, with a sentence a person can act on.

    `code` is the same stable refusal code the reference applier uses for the same defect, so the two
    implementations can be held to the same answers by a test.
    """

    def __init__(self, message, code="recipe-invalid"):
        super().__init__(message)
        self.code = code


def _text(value):
    return isinstance(value, str) and value.strip() != ""


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def edit_sha256(edit):
    """The same canonical hash the reference applier computes. The two must not drift apart."""
    text = json.dumps(edit, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: Asks `provenance_block` to find the commit itself. A distinct object, because `None` already means
#: "the commit could not be found", and the two must not be confused.
_DETECT = object()


def provenance_block(source=_DETECT):
    """Which tool wrote the recipe, and the commit it ran from, or why that is unknown.

    Deliberately narrower than `crashsafe.provenance`. A recipe is a small file people copy and diff, so it
    carries no platform, package list or device, which would make two identical edits differ by machine.
    """
    from . import __version__

    if source is _DETECT:
        source = crashsafe.git_commit()
    if source is None:
        commit = {
            "commit": None,
            "dirty": None,
            "via": None,
            "reason": "no source knew which commit this tool ran from",
        }
    else:
        commit = {"commit": source["commit"], "dirty": source["dirty"], "via": source["source"]}
    return {"tool": {"name": "senbonzakura", "version": __version__}, "source": commit}


#: Absolute paths in a recorded command. The bake records its own argv, so a path given on the command line
#: would otherwise travel into every recipe, and a recipe is meant to leave the machine that made it.
_ABSOLUTE_PATH = re.compile(r"^(/|~|[A-Za-z]:[\\/])")


def public_command(command):
    """The command with each absolute path reduced to its last component.

    The recipe says what was run, and the names of the files are enough to rerun it beside those files.
    Where a path sits in a `--flag=value` token, only the value is reduced.
    """
    words = []
    for word in shlex.split(command):
        flag, sep, value = word.partition("=")
        if sep and _ABSOLUTE_PATH.match(value):
            words.append(f"{flag}={pathlib.PurePath(value.replace(chr(92), '/')).name or value}")
        elif not sep and _ABSOLUTE_PATH.match(word):
            words.append(pathlib.PurePath(word.replace(chr(92), "/")).name or word)
        else:
            words.append(word)
    return shlex.join(words)


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


def _max_abs_projection(W, R):
    """Maximum |R W| in float64. `W` is a float64 array of the tensor's shape, `R` the non-empty rows."""
    import numpy as np

    if R.shape[0] == 0:
        return 0.0
    return float(np.abs(R @ W).max())


def _float64_of(raw, tensor):
    """One tensor's raw bytes as a float64 array. Decoded here so the bytes are read exactly once."""
    import numpy as np
    import torch

    from . import streaming

    dtype = streaming.torch_dtype(tensor.dtype)
    values = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(tensor.shape)
    return values.float().numpy().astype(np.float64)


def measure_residual(source_dir, baked_dir, directions_path, *, layer_count, targets):
    """The residual figures for a bake, computed from the source and baked checkpoints in float64.

    This is the writer's implementation of the residual definition in core/1. The reference applier
    computes the same figures on its own while it edits, and refuses a recipe whose recorded figures
    disagree with its own. Two implementations agreeing on a number is the cross-check; neither is
    trusted alone.
    """
    import numpy as np

    from . import streaming

    arr, _meta = streambake.load_directions(directions_path)
    before_max = after_max = 0.0
    edited = 0
    # STREAMED, ONE TENSOR AT A TIME. Loading a shard whole would hold a 200 GB checkpoint in memory
    # to measure one number. The residual is the cheapest pass this tool makes: one read of the
    # source and one of the baked copy, each tensor decoded and discarded before the next.
    for shard in sorted(pathlib.Path(source_dir).glob("*.safetensors")):
        baked_shard = pathlib.Path(baked_dir) / shard.name
        baked_by_name = {t.name: t for t in streaming.tensors(baked_shard)}
        for tensor in streaming.tensors(shard):
            layer = _layer_of(tensor.name, targets)
            if layer is None or not 0 <= layer < layer_count:
                continue
            R = np.asarray(arr[layer + 1], dtype=np.float64)
            R = R[np.linalg.norm(R, axis=1) > streambake.DIRECTION_EPSILON]
            if len(tensor.shape) != 2 or tensor.shape[0] != arr.shape[2]:
                raise RecipeError(
                    f"{tensor.name!r} has shape {tensor.shape}, but the direction is "
                    f"{arr.shape[2]} wide and must sit on the first axis. This target does not "
                    f"write to the residual stream, so no residual can be measured for it."
                )
            before = _float64_of(streaming.read_tensor(shard, tensor), tensor)
            after = _float64_of(streaming.read_tensor(baked_shard, baked_by_name[tensor.name]), tensor)
            before_max = max(before_max, _max_abs_projection(before, R))
            after_max = max(after_max, _max_abs_projection(after, R))
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


# ── the checks the reference applier makes on the blocks this writer creates ─────────────────
# The applier is the reader of record. These mirror its rules for provenance, base identity and the
# measured block, with the same refusal codes, so a writer cannot produce a block the applier would
# refuse. `tests/test_recipe_writer.py` runs both over the same defective recipes and holds them to
# the same answers.


def validate_document(doc):
    """Refuses a recipe document whose provenance, base identity or measured block the applier would refuse."""
    if not isinstance(doc, dict):
        raise RecipeError("a recipe is a JSON object.")
    if "provenance" not in doc:
        raise RecipeError(
            "the recipe has no `provenance` block. It must say which tool and version wrote it.",
            code="provenance-missing",
        )
    _check_provenance(doc["provenance"])
    _check_base_identity(doc.get("base"))
    _check_measured(doc.get("measured"), doc.get("edit"))


def _check_provenance(provenance):
    if not isinstance(provenance, dict):
        raise RecipeError("`provenance` must be an object.", code="provenance-invalid")
    tool = provenance.get("tool")
    if not isinstance(tool, dict) or not _text(tool.get("name")) or not _text(tool.get("version")):
        raise RecipeError("`provenance.tool` must name the tool and its version.", code="provenance-invalid")
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise RecipeError(
            "`provenance.source` must say which commit the tool ran from, or why not.", code="provenance-invalid"
        )
    commit = source.get("commit")
    if commit is None:
        if not _text(source.get("reason")):
            raise RecipeError(
                "`provenance.source` has no commit, so it must say why in `reason`.", code="provenance-invalid"
            )
    elif not _text(commit) or not _text(source.get("via")):
        raise RecipeError(
            "`provenance.source.commit` must be a commit, with `via` naming its origin.", code="provenance-invalid"
        )
    if source.get("dirty") not in (True, False, None):
        raise RecipeError("`provenance.source.dirty` must be true, false or null.", code="provenance-invalid")


def _check_base_identity(base):
    if not isinstance(base, dict) or not _text(base.get("id")):
        raise RecipeError("`base.id` must name the base checkpoint.", code="base-invalid")
    if base.get("revision") is None:
        if not _text(base.get("revision_reason")):
            raise RecipeError(
                "`base.revision` is null and `base.revision_reason` does not say why.", code="base-invalid"
            )
    elif not _text(base.get("revision")):
        raise RecipeError(
            "`base.revision` must be a hub revision or a git sha, or null with a reason.", code="base-invalid"
        )


def _check_measured(measured, edit):
    if not isinstance(measured, dict) or measured.get("status") not in ("measured", "not_measured"):
        raise RecipeError('`measured.status` must be "measured" or "not_measured".', code="measured-invalid")
    if measured["status"] == "not_measured":
        if not _text(measured.get("why")):
            raise RecipeError("a `not_measured` recipe must say why in `measured.why`.", code="measured-invalid")
        return
    _check_instrument(measured.get("instrument"))
    _check_figure(measured.get("refusal_before"), "measured.refusal_before", rate=True)
    _check_figure(measured.get("refusal_after"), "measured.refusal_after", rate=True)
    _check_figure(measured.get("kl"), "measured.kl", rate=False)
    _check_capability(measured.get("capability"))
    if "self_invalidated" not in measured:
        raise RecipeError("`measured.self_invalidated` must be present, null or a reason.", code="measured-invalid")
    if measured["self_invalidated"] is not None and not _text(measured["self_invalidated"]):
        raise RecipeError("`measured.self_invalidated` must be null or a reason.", code="measured-invalid")
    limits = measured.get("limits")
    if not isinstance(limits, list) or not limits or not all(_text(item) for item in limits):
        raise RecipeError(
            "`measured.limits` must list what the figures do not say. It cannot be empty.", code="measured-invalid"
        )
    if measured.get("edit_sha256") != edit_sha256(edit):
        raise RecipeError(
            "`measured.edit_sha256` is not the hash of this recipe's `edit` block, so these figures were not "
            "measured on this edit.",
            code="measured-mismatch",
        )


def _check_instrument(instrument):
    if not isinstance(instrument, dict):
        raise RecipeError(
            "`measured.instrument` must name the scorer, corpus, partition and seed.", code="measured-invalid"
        )
    for key in ("scorer", "partition"):
        if not _text(instrument.get(key)):
            raise RecipeError(f"`measured.instrument.{key}` must be named.", code="measured-invalid")
    corpus = instrument.get("corpus")
    if not isinstance(corpus, dict) or not _text(corpus.get("id")) or not _text(corpus.get("revision")):
        raise RecipeError(
            "`measured.instrument.corpus` must hold a pinned `id` and `revision`.", code="measured-invalid"
        )
    seed = instrument.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RecipeError("`measured.instrument.seed` must be a whole number.", code="measured-invalid")


def _check_figure(figure, where, *, rate):
    if not isinstance(figure, dict):
        raise RecipeError(f"`{where}` must be an object holding point, n and ci.", code="measured-invalid")
    point, n, ci = figure.get("point"), figure.get("n"), figure.get("ci")
    if not _finite(point):
        raise RecipeError(f"`{where}.point` must be a finite number.", code="measured-invalid")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise RecipeError(f"`{where}` has no sample size `n` of one or more.", code="measured-invalid")
    if not isinstance(ci, list) or len(ci) != 2 or not all(_finite(v) for v in ci) or not ci[0] <= point <= ci[1]:
        raise RecipeError(f"`{where}.ci` must be [low, high] and must contain the point.", code="measured-invalid")
    if rate and not (ci[0] >= 0 and ci[1] <= 1 and point >= 0 and point <= 1):
        raise RecipeError(f"`{where}` is a rate, so it must lie between 0 and 1.", code="measured-invalid")


def _check_capability(capability):
    if isinstance(capability, dict):
        if capability.get("status") != "not_measured" or not _text(capability.get("why")):
            raise RecipeError(
                "`measured.capability` must be a list of figures, or `not_measured` with a reason.",
                code="measured-invalid",
            )
        return
    if not isinstance(capability, list) or not capability:
        raise RecipeError(
            "`measured.capability` must be a list of figures, or `not_measured` with a reason.", code="measured-invalid"
        )
    for entry in capability:
        if not isinstance(entry, dict) or not _text(entry.get("name")):
            raise RecipeError("each capability entry must name its benchmark.", code="measured-invalid")
        _check_figure(entry.get("before"), f"capability {entry['name']!r} before", rate=True)
        _check_figure(entry.get("after"), f"capability {entry['name']!r} after", rate=True)


def build(
    *,
    name,
    model_dir,
    model_id,
    directions_path,
    layer_count,
    strength,
    restore_norms,
    measured,
    command,
    revision=None,
    revision_reason=None,
    provenance=None,
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
    if revision is None:
        base["revision_reason"] = revision_reason
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

    doc = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "name": str(name),
        "base": base,
        "provenance": provenance if provenance is not None else provenance_block(),
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
        "reproduce": {"tool": "senbonzakura", "command": public_command(str(command))},
    }
    validate_document(doc)
    return doc


def write(path, recipe):
    """Atomically, through the same writer as the rest of this project's artefacts."""
    text = json.dumps(recipe, indent=2, sort_keys=True) + "\n"
    with crashsafe.atomic_write(path, binary=False) as fh:
        fh.write(text)
    return pathlib.Path(path)


#: A bake reads a local directory, which has no hub revision. The weights hash is what identifies it, so the
#: recipe says so rather than writing a placeholder that looks like a revision.
LOCAL_REVISION_REASON = "a local checkpoint directory has no hub revision, so weights_sha256 identifies it"

#: The reason recorded when the operator skips the residual measurement. Named so a reader of the
#: recipe can tell a choice from an omission.
SKIPPED_REASON = "the operator skipped the residual measurement with --skip-residual"


def _require_expressible(index, plan, layer_count, targets):
    """The bake must edit exactly the tensors the recipe's targets name, and no others.

    The check is by set equality, both ways. A tensor the bake edits and the targets do not name
    would be an edit the recipe omits. A tensor the targets name and the bake does not edit would be
    an edit the recipe invents. Either would make the recipe describe a different model.
    """
    named = set()
    for layer_names in index.layers:
        for name in layer_names:
            layer = _layer_of(name, targets)
            if layer is not None and 0 <= layer < layer_count:
                named.add(name)
    planned = set(plan)
    if planned != named:
        differing = sorted(planned ^ named)
        raise RecipeError(
            f"this bake and the recipe's targets disagree about which tensors are edited (for "
            f"example {differing[0]!r}), so a recipe would describe a different model. Only the "
            f"attention output and MLP down projections are expressible in {PROFILE} today."
        )


def _require_plain_bake(mode, K, arr):
    """A shared-direction bake, or one that used a subset of the directions, is not the core edit."""
    if mode != "per_layer":
        raise RecipeError(
            f"this bake used the {mode!r} direction mode, where one direction set is shared across "
            f"every layer. {PROFILE} applies each layer's own set, so a recipe would describe a "
            f"different edit."
        )
    if K is not None and int(K) < arr.shape[1]:
        raise RecipeError(
            f"this bake used the first {K} of {arr.shape[1]} directions per layer. The recipe "
            f"carries the whole set, so it would describe directions the bake did not use."
        )


def write_for_bake(
    *,
    model,
    out,
    directions,
    strength,
    restore_norms,
    sparsity,
    rounds,
    K,
    mode,
    ablate_conv,
    prediction_head,
    skip_residual,
    command,
    log=print,
):
    """Write `recipe.json` beside a finished bake, measuring the residual unless told not to.

    A recipe is the better artefact, so this runs by default. It never fails the bake: a bake the
    profile cannot express is still a bake, and the reason no recipe was written is printed plainly.
    Returns the recipe path, or None.
    """
    from . import streaming

    try:
        index = streaming.index_layers(model)
        layer_count = index.count
        plan = streambake.plan_edits(index, ablate_conv=ablate_conv, prediction_head=prediction_head)
        targets = [ATTN_TARGET, MLP_TARGET]
        _require_expressible(index, plan, layer_count, targets)
        arr, _meta = streambake.load_directions(directions)
        _require_plain_bake(mode, K, arr)
        if skip_residual:
            residual = {"status": "not_measured", "why": SKIPPED_REASON}
        else:
            residual = measure_residual(model, out, directions, layer_count=layer_count, targets=targets)
        doc = build(
            name=pathlib.Path(out).name,
            model_dir=model,
            model_id=pathlib.Path(model).name,
            revision=None,
            revision_reason=LOCAL_REVISION_REASON,
            directions_path=directions,
            layer_count=layer_count,
            strength=strength,
            restore_norms=restore_norms,
            measured=None,
            command=command,
            sparsity=sparsity,
            rounds=rounds,
            residual=residual,
        )
    except (RecipeError, streaming.ShardError) as e:
        log(f"  no recipe written: {e}")
        return None
    path = write(pathlib.Path(out) / "recipe.json", doc)
    log(f"  recipe written to {path}")
    return path
