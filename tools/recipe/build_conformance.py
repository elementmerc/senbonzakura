#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Rebuild the conformance fixture in `recipes/conformance/` from senbonzakura's own bake.

The fixture is the expected answer that any implementation of the recipe format is checked
against, so it has to come from the tool whose behaviour the format describes: this bake. Run it
after any change to the bake, the direction format or the recipe writer, then commit the result,
so a change in the bake shows up as a change to the fixture rather than as silent drift.

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


def build_directions(path):
    rng = np.random.default_rng(20261015)
    dirs = np.zeros((LAYERS + 1, 2, HIDDEN), dtype=np.float32)
    q, _ = np.linalg.qr(rng.standard_normal((HIDDEN, 2)))
    dirs[1, :2] = q.T
    dirs[2, :1] = q[:, 0]
    streambake.save_directions(
        path, dirs, model="conformance-tiny", mode="per_layer", provenance={"fixture": "recipes/conformance"}
    )


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
            command="senbonzakura stream-bake --model MODEL --directions DIRS --out OUT",
        )
        recipe_mod.write(work / "recipe.json", doc)

        FIXTURE.mkdir(parents=True, exist_ok=True)
        (FIXTURE / "model").mkdir(exist_ok=True)
        (FIXTURE / "expected").mkdir(exist_ok=True)
        shutil.copyfile(model / "model.safetensors", FIXTURE / "model" / "model.safetensors")
        shutil.copyfile(model / "config.json", FIXTURE / "model" / "config.json")
        shutil.copyfile(directions, FIXTURE / "directions.safetensors")
        shutil.copyfile(work / "recipe.json", FIXTURE / "recipe.json")
        shutil.copyfile(expected / "model.safetensors", FIXTURE / "expected" / "model.safetensors")
        print(f"wrote the conformance fixture to {FIXTURE}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
