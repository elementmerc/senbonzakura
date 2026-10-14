# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Assemble a recipe into a pack, and check a pack before anyone applies it.

A pack is the directory a person hands to someone else or publishes. It holds four files and nothing else:

    recipe.json        the edit, its provenance and its measured result
    <directions file>  the direction set the recipe names, under the name the recipe records
    README.md          the figures, what they do not say, and how to apply the recipe
    MANIFEST.json      the sha256 and size of the three files above

A pack never holds model weights, prompts or generations. The manifest proves the bytes are the ones the pack was
built from. The README is rendered again from the recipe when the pack is checked, and it must match, so the README
cannot say more than the recipe records.
"""

import copy
import hashlib
import json
import os
import pathlib
import re
import shutil
import uuid

from . import __version__, crashsafe
from . import recipe as recipe_mod
from .recipe import RecipeError

#: The schema the manifest declares. A reader refuses a pack whose manifest it does not understand.
PACK_SCHEMA = "abliteration-recipe-pack/1"

RECIPE_NAME = "recipe.json"
README_NAME = "README.md"
MANIFEST_NAME = "MANIFEST.json"
#: The names a direction file may not take, because the pack already uses them.
RESERVED_NAMES = frozenset({RECIPE_NAME, README_NAME, MANIFEST_NAME})


def build_pack(recipe_path, directions_path, out_dir, *, measured=None, log=print):
    """Write a pack for one recipe into `out_dir`, which must not exist yet. Returns that directory.

    `measured`, when given, is a measured block bound to this recipe's edit. It is attached only to a recipe
    that says it was not measured. A recipe that already carries a result keeps it, and a second result is
    refused rather than shown beside the first.
    """
    out = pathlib.Path(out_dir)
    if out.exists() or out.is_symlink():
        raise RecipeError(
            f"{out} already exists. A pack is written once, and an existing one is never overwritten.",
            code="pack-exists",
        )
    doc = _load_json(recipe_path, "the recipe", code="pack-invalid")
    recipe_mod.validate_document(doc)
    if measured is not None:
        if doc["measured"].get("status") == "measured":
            raise RecipeError(
                "the recipe already carries a measured result. A pack will not show a second one beside it.",
                code="pack-measured-conflict",
            )
        doc["measured"] = copy.deepcopy(measured)
        recipe_mod.validate_document(doc)
    _require_pinned(doc)
    name = _directions_name(doc)
    _require_directions_match(doc, directions_path)
    _refuse_local_paths(doc, where="the recipe")
    readme = render_readme(doc)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".pack-{uuid.uuid4().hex}"
    os.mkdir(tmp)
    try:
        recipe_mod.write(tmp / RECIPE_NAME, doc)
        shutil.copyfile(directions_path, tmp / name)
        with crashsafe.atomic_write(tmp / README_NAME, binary=False) as fh:
            fh.write(readme)
        files = {entry: _entry(tmp / entry) for entry in (RECIPE_NAME, name, README_NAME)}
        _write_manifest(tmp, files)
        _publish(tmp, out)
    except BaseException:
        try:
            shutil.rmtree(tmp)
        except OSError as cleanup:
            log(f"  could not remove the unfinished pack at {tmp}: {cleanup}")
        raise
    log(f"  pack written to {out}")
    return out


def verify_pack(pack_dir):
    """Check a pack's bytes, its recipe and its README. Returns a summary, or raises RecipeError with a code.

    This reads the recipe's claims and the manifest. It does not apply the edit: that needs the base checkpoint,
    and the reference applier does it with `--check`.
    """
    root = pathlib.Path(pack_dir)
    if not root.is_dir() or root.is_symlink():
        raise RecipeError(f"{root} is not a directory holding a pack.", code="pack-invalid")
    manifest = _load_json(root / MANIFEST_NAME, "the manifest", code="pack-invalid")
    if manifest.get("schema") != PACK_SCHEMA:
        raise RecipeError(
            f"the manifest declares {manifest.get('schema')!r}, which is not a pack this tool reads.",
            code="pack-invalid",
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not RESERVED_NAMES - {MANIFEST_NAME} <= set(files):
        raise RecipeError("the manifest does not list the recipe and the README.", code="pack-invalid")
    listed = set(files)
    on_disk = {p.name for p in root.iterdir()} - {MANIFEST_NAME}
    if on_disk != listed:
        extra, missing = sorted(on_disk - listed), sorted(listed - on_disk)
        raise RecipeError(
            f"the pack holds files the manifest does not list ({extra}) or lacks files it lists ({missing}). "
            "A file outside the manifest is not covered by it, so the pack is refused.",
            code="pack-mismatch",
        )
    for name in sorted(listed):
        _check_entry(root / name, name, files[name])
    doc = _load_json(root / RECIPE_NAME, "the recipe", code="pack-invalid")
    recipe_mod.validate_document(doc)
    _require_pinned(doc)
    name = _directions_name(doc)
    if name not in listed:
        raise RecipeError(
            f"the recipe names {name!r} as its direction file, and the pack does not hold it.", code="pack-mismatch"
        )
    if files[name]["sha256"] != doc["edit"]["directions"]["sha256"]:
        raise RecipeError(
            "the direction file in the pack is not the one the recipe records, so the pack is refused.",
            code="pack-mismatch",
        )
    _refuse_local_paths(doc, where="the recipe")
    if (root / README_NAME).read_text(encoding="utf-8") != render_readme(doc):
        raise RecipeError(
            "README.md does not match the recipe it sits beside. The README may not say more than the recipe records.",
            code="pack-mismatch",
        )
    return {
        "status": "verified",
        "name": doc["name"],
        "base_sha256": doc["base"]["weights_sha256"],
        "measured": doc["measured"]["status"],
        "files": sorted(listed),
    }


def render_readme(doc):
    """The README for a recipe. A pure function of the recipe, so the README can be checked by rendering again."""
    base, edit, measured = doc["base"], doc["edit"], doc["measured"]
    directions = edit["directions"]
    lines = [
        f"# {doc['name']}",
        "",
        (
            "This pack holds an abliteration recipe. The recipe describes an edit that removes a direction from a "
            "language model, and what the edit did when it was measured. The pack holds no model weights. Apply "
            "the recipe to a copy of the base checkpoint to make the edited model."
        ),
        "",
        "## The base checkpoint",
        "",
        f"- Name: {base['id']}",
    ]
    if base["revision"] is None:
        lines.append(f"- Revision: not recorded. Reason recorded: {base['revision_reason']}")
    else:
        lines.append(f"- Revision: {base['revision']}")
    lines += [f"- Weights sha256: {base['weights_sha256']}", "", "## The edit", ""]
    lines.append("- Method: projection, removing the recorded directions from the output side of each target.")
    lines += [f"- Target: {target}" for target in edit["targets"]]
    lines.append(f"- Layers: {edit['layers']['from']} to {edit['layers']['to']}, both included.")
    if "strength" in edit:
        lines.append(f"- Strength: {edit['strength']} in every layer.")
    else:
        lines.append("- Strength: a schedule, given per target and per layer in the recipe.")
    lines.append(f"- Restore norms: {'yes' if edit['restore_norms'] else 'no'}.")
    positions, per_position, hidden = directions["shape"]
    lines.append(
        f"- Directions: {directions['file']}, sha256 {directions['sha256']}, with {directions['directions_found']} "
        f"directions found across {positions} positions of up to {per_position} each, hidden size {hidden}."
    )
    residual = edit["direction_residual"]
    if residual["status"] == "measured":
        lines.append(
            f"- Residual (a weight measurement, not a behaviour measurement): the largest component of the removed "
            f"direction is {residual['after']:.6g} after the edit, against {residual['before']:.6g} before it. "
            f"That leaves a fraction of {residual['fraction_remaining']:.6g} (metric {residual['metric']}, "
            "the maximum over the edited tensors)."
        )
    else:
        lines.append(f"- Residual: not measured. Reason recorded: {residual['why']}")
    lines += ["", "## What was measured", ""]
    lines += _measured_lines(measured)
    lines += ["", "## What the figures do not say", ""]
    lines += [f"- {item}" for item in measured.get("limits", [])] or [
        "- Nothing was measured, so there is nothing to limit."
    ]
    lines += [
        "",
        "## Provenance",
        "",
        (
            f"Written by {doc['provenance']['tool']['name']} {doc['provenance']['tool']['version']}. "
            f"{_source_line(doc['provenance']['source'])}"
        ),
        "",
        "## Checking and applying",
        "",
        (
            "Every file must match its sha256 and size in MANIFEST.json, and this README must match the recipe. "
            "The reference applier, recipes/apply_recipe.py in the senbonzakura repository, checks the recipe "
            "against the base checkpoint and the direction file before it writes anything:"
        ),
        "",
        "    python apply_recipe.py --check recipe.json MODEL_DIR directions.safetensors",
        "    python apply_recipe.py recipe.json MODEL_DIR directions.safetensors OUT_DIR",
        "",
    ]
    return "\n".join(lines)


def _measured_lines(measured):
    if measured["status"] == "not_measured":
        return [
            f"Not measured. Reason recorded: {measured['why']}",
            "",
            "Nothing in this pack says what the edit did to refusal, to a model's answers, or to any task.",
        ]
    instrument = measured["instrument"]
    level = measured["ci_level"]
    lines = [
        f"- Scorer: {instrument['scorer']}",
        f"- Corpus: {instrument['corpus']['id']}, revision {instrument['corpus']['revision']}",
        f"- Partition: {instrument['partition']}",
        f"- Seed: {instrument['seed']}",
        f"- Every interval is at the {level} level.",
    ]
    for key, label in (
        ("refusal_before", "Refusal before the edit"),
        ("refusal_after", "Refusal after the edit"),
        ("kl", "KL divergence from the base"),
    ):
        lines.append(f"- {label}: {_figure(measured[key], level)}")
    capability = measured["capability"]
    if isinstance(capability, dict):
        lines.append(f"- Capability: not measured. {capability['why']}")
    else:
        lines.extend(
            f"- Capability, {entry['name']}: before {_figure(entry['before'], level)}; "
            f"after {_figure(entry['after'], level)}"
            for entry in capability
        )
    if measured["self_invalidated"] is None:
        lines.append("- The producing run did not invalidate itself.")
    else:
        lines.append(f"- The producing run recorded that its measurement is invalid: {measured['self_invalidated']}")
    return lines


def _figure(figure, level):
    low, high = figure["ci"]
    return f"{figure['point']} (n = {figure['n']}, {level} interval {low} to {high})"


def _source_line(source):
    if source["commit"] is None:
        return f"No commit could be named, because {source['reason']}."
    state = {
        True: "with uncommitted changes",
        False: "from a clean tree",
        None: "tree state not checked",
    }[source["dirty"]]
    return f"It was run from commit {source['commit']}, {state}."


def _publish(tmp, out):
    if out.exists():
        raise RecipeError(
            f"{out} appeared while the pack was being written, so nothing was replaced.", code="pack-exists"
        )
    os.rename(tmp, out)


def _require_pinned(doc):
    if doc["base"].get("weights_sha256") is None:
        raise RecipeError(
            f"a pack must pin its base checkpoint, and this recipe records no base hash: "
            f"{doc['base'].get('weights_sha256_reason')}. Without one, nobody can tell which model the edit "
            "was made to.",
            code="pack-unpinned",
        )


def _directions_name(doc):
    name = doc["edit"]["directions"]["file"]
    if not isinstance(name, str) or name in ("", ".", "..") or "/" in name or "\\" in name or name in RESERVED_NAMES:
        raise RecipeError(
            f"the recipe names its direction file {name!r}, which a pack cannot hold beside the recipe.",
            code="pack-invalid",
        )
    return name


def _require_directions_match(doc, directions_path):
    path = pathlib.Path(directions_path)
    if not path.is_file() or path.is_symlink():
        raise RecipeError(f"{path} is not a regular file, so it cannot be packed.", code="pack-invalid")
    digest, _size = _digest(path)
    if digest != doc["edit"]["directions"]["sha256"]:
        raise RecipeError(
            f"the direction file hashes to {digest}, and the recipe records {doc['edit']['directions']['sha256']}. "
            "These are not the directions the recipe means, so no pack is written.",
            code="pack-mismatch",
        )


#: A token that begins like a local path. The same test the writer applies to the recorded command.
_LOCAL = recipe_mod.LOCAL_PATH


def _refuse_local_paths(doc, *, where):
    for text in _strings(doc):
        for token in re.split(r"[\s=]+", text):
            if _LOCAL.match(token):
                raise RecipeError(
                    f"{where} records a path from the machine that made it. A pack leaves that machine, so the "
                    "recipe must name files rather than locations.",
                    code="pack-local-path",
                )


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _digest(path):
    h = hashlib.sha256()
    size = 0
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _entry(path):
    digest, size = _digest(path)
    return {"sha256": digest, "bytes": size}


def _check_entry(path, name, entry):
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("sha256"), str)
        or not isinstance(entry.get("bytes"), int)
    ):
        raise RecipeError(f"the manifest's entry for {name!r} is not a sha256 and a size.", code="pack-invalid")
    if path.is_symlink() or not path.is_file():
        raise RecipeError(
            f"{name} is not a plain file in the pack. A link could point anywhere, so it is refused.",
            code="pack-mismatch",
        )
    digest, size = _digest(path)
    if size != entry["bytes"] or digest != entry["sha256"]:
        raise RecipeError(
            f"{name} does not match the manifest. The pack has changed since it was written, so it is refused.",
            code="pack-mismatch",
        )


def _write_manifest(root, files):
    manifest = {"schema": PACK_SCHEMA, "tool": {"name": "senbonzakura", "version": __version__}, "files": files}
    with crashsafe.atomic_write(root / MANIFEST_NAME, binary=False) as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _load_json(path, what, *, code):
    try:
        value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RecipeError(f"{what} could not be read as JSON ({e}).", code=code) from e
    if not isinstance(value, dict):
        raise RecipeError(f"{what} is not a JSON object.", code=code)
    return value
