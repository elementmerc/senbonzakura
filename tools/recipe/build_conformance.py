#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Rebuild the conformance fixture in `recipes/conformance/` from senbonzakura's own bake.

The fixture is the expected answer that any implementation of the recipe format is checked against,
so it has to come from the tool whose behaviour the format describes: this bake. It also holds a set
of defective recipes, each with the refusal code a conforming applier must give. Run it after any
change to the bake, the direction format, the recipe writer or the profile, then commit the result,
so a change shows up as a change to the fixture rather than as silent drift.

    python tools/recipe/build_conformance.py

It writes into `recipes/conformance/`. Nothing else is touched.
"""

import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "recipes"))

import apply_recipe  # noqa: E402  (the reference applier, found beside this repo's recipes)

from senbonzakura import recipe as recipe_mod  # noqa: E402
from senbonzakura import streambake  # noqa: E402

FIXTURE = REPO / "recipes" / "conformance"
HIDDEN, INTER, VOCAB, LAYERS, KV = 8, 12, 16, 2, 4


def build_model(root):
    """A two-layer checkpoint. Layer 0's o_proj is bfloat16 so the rounding path is exercised."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    arrays = {"model.embed_tokens.weight": rng.standard_normal((VOCAB, HIDDEN))}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        arrays[p + "input_layernorm.weight"] = 1 + 0.1 * rng.standard_normal(HIDDEN)
        arrays[p + "self_attn.q_proj.weight"] = rng.standard_normal((HIDDEN, HIDDEN))
        arrays[p + "self_attn.k_proj.weight"] = rng.standard_normal((KV, HIDDEN))
        arrays[p + "self_attn.v_proj.weight"] = rng.standard_normal((KV, HIDDEN))
        arrays[p + "self_attn.o_proj.weight"] = rng.standard_normal((HIDDEN, HIDDEN))
        arrays[p + "post_attention_layernorm.weight"] = 1 + 0.1 * rng.standard_normal(HIDDEN)
        arrays[p + "mlp.gate_proj.weight"] = rng.standard_normal((INTER, HIDDEN))
        arrays[p + "mlp.up_proj.weight"] = rng.standard_normal((INTER, HIDDEN))
        arrays[p + "mlp.down_proj.weight"] = rng.standard_normal((HIDDEN, INTER))
    arrays["model.norm.weight"] = np.ones(HIDDEN)
    arrays["lm_head.weight"] = rng.standard_normal((VOCAB, HIDDEN))

    tensors = {}
    for name, arr in arrays.items():
        dtype = "BF16" if name == "model.layers.0.self_attn.o_proj.weight" else "F32"
        tensors[name] = (dtype, arr.astype(np.float32))
    header, blobs, offset = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = apply_recipe.encode(arr, dtype)
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    (root / "model.safetensors").write_bytes(len(blob).to_bytes(8, "little") + blob + b"".join(blobs))
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": HIDDEN,
                "num_hidden_layers": LAYERS,
                "intermediate_size": INTER,
                "vocab_size": VOCAB,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_directions(path, *, scale_row=None):
    """Two orthonormal rows per layer, one unused slot. `scale_row` makes one row non-orthonormal."""
    rng = np.random.default_rng(20261015)
    dirs = np.zeros((LAYERS + 1, 2, HIDDEN), dtype=np.float32)
    q, _ = np.linalg.qr(rng.standard_normal((HIDDEN, 2)))
    dirs[1, :2] = q.T
    dirs[2, :1] = q[:, 0]
    if scale_row is not None:
        dirs[1, 0] *= scale_row
    path.parent.mkdir(parents=True, exist_ok=True)
    streambake.save_directions(
        path, dirs, model="conformance-tiny", mode="per_layer", provenance={"fixture": "recipes/conformance"}
    )


def _sha(path):
    return apply_recipe.sha256_file(path)


def main():
    work = pathlib.Path(tempfile.mkdtemp(prefix="senbon-conformance-"))
    try:
        model = work / "model"
        build_model(model)
        directions = work / "directions.safetensors"
        build_directions(directions)

        expected = streambake.prepare_output(model, work / "expected", log=lambda *_: None)
        arr, _meta = streambake.load_directions(directions)
        streambake.bake(expected, arr, strength=1.0, restore_norms=True, log=lambda *_: None)

        targets = [recipe_mod.ATTN_TARGET, recipe_mod.MLP_TARGET]
        residual = recipe_mod.measure_residual(model, expected, directions, layer_count=LAYERS, targets=targets)
        doc = recipe_mod.build(
            name="conformance-tiny",
            model_dir=model,
            model_id="senbonzakura/conformance-tiny",
            revision="fixture",
            directions_path=directions,
            layer_count=LAYERS,
            strength=1.0,
            restore_norms=True,
            measured={
                "status": "not_measured",
                "why": "conformance fixture: a weight check, not a behaviour measurement",
            },
            residual=residual,
            command="senbonzakura stream-bake --model MODEL --directions DIRS --out OUT",
        )

        FIXTURE.mkdir(parents=True, exist_ok=True)
        # The builder owns `cases/`: clearing it means a rebuilt fixture cannot carry a stale case.
        shutil.rmtree(FIXTURE / "cases", ignore_errors=True)
        for sub in ("model", "expected", "cases"):
            (FIXTURE / sub).mkdir(exist_ok=True)
        shutil.copyfile(model / "model.safetensors", FIXTURE / "model" / "model.safetensors")
        shutil.copyfile(model / "config.json", FIXTURE / "model" / "config.json")
        shutil.copyfile(directions, FIXTURE / "directions.safetensors")
        shutil.copyfile(expected / "model.safetensors", FIXTURE / "expected" / "model.safetensors")
        recipe_mod.write(FIXTURE / "recipe.json", doc)

        cases = write_cases(doc, work)
        (FIXTURE / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote the conformance fixture and {len(cases)} cases to {FIXTURE}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


def write_cases(valid, work):
    """Every case the harness runs.

    Each defective case is a PATCH against the valid recipe, not a copy of it: one `set` or `delete`
    naming the field it breaks. A reviewer reads the defect in one line, and the fixture stays small.
    The harness materialises each patch into a temporary recipe before running the tool.
    """
    cases = []

    def add(case_id, *, expect, patch=(), code=None, output=None, ignored=None, directions=None):
        entry = {"id": case_id, "expect": expect, "patch": list(patch)}
        if expect == "refused":
            entry["code"] = code
        if output:
            entry["output"] = output
        if ignored:
            entry["ignored_extensions"] = ignored
        if directions:
            entry["directions"] = directions
        cases.append(entry)

    def set_(path, value):
        return {"op": "set", "path": path, "value": value}

    def delete(path):
        return {"op": "delete", "path": path}

    add("valid", expect="applied", output="expected")
    add(
        "descriptive-extension-ignored",
        expect="applied",
        output="expected",
        ignored=["x_example_notes"],
        patch=[
            set_(
                ["extensions"],
                [
                    {
                        "id": "x_example_notes",
                        "edit_affecting": False,
                        "description": "a note the applier does not need to honour",
                    }
                ],
            )
        ],
    )
    add("profile-missing", expect="refused", code="profile-missing", patch=[delete(["profile"])])
    add("profile-unsupported", expect="refused", code="profile-unsupported", patch=[set_(["profile"], "core/2")])
    add(
        "edit-affecting-extension-refused",
        expect="refused",
        code="extension-unsupported",
        patch=[set_(["extensions"], [{"id": "x_example_sparse", "edit_affecting": True}])],
    )
    add(
        "extension-without-edit-flag",
        expect="refused",
        code="extension-malformed",
        patch=[set_(["extensions"], [{"id": "x_example_notes"}])],
    )
    add("unknown-edit-key", expect="refused", code="unknown-edit-key", patch=[set_(["edit", "sparsity"], 0.3)])
    add("strength-and-schedule-both", expect="refused", code="strength-invalid", patch=[set_(["edit", "schedule"], [])])
    add("negative-strength", expect="refused", code="strength-invalid", patch=[set_(["edit", "strength"], -1.0)])
    add(
        "projection-side-input",
        expect="refused",
        code="convention-unsupported",
        patch=[set_(["edit", "projection_side"], "input")],
    )
    add("residual-missing", expect="refused", code="residual-missing", patch=[delete(["edit", "direction_residual"])])
    recorded = valid["edit"]["direction_residual"]
    add(
        "residual-mismatch",
        expect="refused",
        code="residual-mismatch",
        patch=[set_(["edit", "direction_residual", "after"], recorded["after"] * 0.5)],
    )
    add(
        "residual-metric-unknown-version",
        expect="refused",
        code="residual-invalid",
        patch=[set_(["edit", "direction_residual", "metric"], "max_abs_projection/2")],
    )
    add(
        "base-hash-mismatch", expect="refused", code="base-mismatch", patch=[set_(["base", "weights_sha256"], "0" * 64)]
    )
    add(
        "target-matches-nothing",
        expect="refused",
        code="no-target-matches",
        patch=[set_(["edit", "targets"], ["model.layers.*.mlp.gate_proj.bias"])],
    )
    add(
        "residual-not-measured-allowed",
        expect="applied",
        output="expected",
        patch=[
            set_(
                ["edit", "direction_residual"],
                {"status": "not_measured", "why": "conformance case: the residual is recorded as unmeasured"},
            )
        ],
    )

    # A direction set whose layer-0 row is scaled. The recipe pins this file's own hash, so the only
    # defect is the basis: the orthonormality check must be what refuses it.
    bad_dir = FIXTURE / "cases" / "bad-basis" / "directions.safetensors"
    build_directions(bad_dir, scale_row=3.0)
    add(
        "directions-not-orthonormal",
        expect="refused",
        code="directions-not-orthonormal",
        directions="cases/bad-basis/directions.safetensors",
        patch=[set_(["edit", "directions", "sha256"], _sha(bad_dir))],
    )
    return cases


if __name__ == "__main__":
    raise SystemExit(main())
