#!/usr/bin/env python3
"""Does this tool see a real checkpoint the way its tests predict?

WHY THIS EXISTS

Every architecture claim in this repository is tested against models built from a config in
milliseconds: no download, no GPU, no weights. That is deliberate and it covers the wiring that
real models are too large to exercise on the hardware here. It cannot cover one thing: whether the
checkpoint on disk actually has the shape its config advertises.

That gap is not hypothetical. The withdrawn gemma numbers came from a walker that found the right
tensors on a real model and an edit that never reached the model's running state. And a sister
project lost a day in August 2026 to a checkpoint whose file was the exact right length and the
wrong content.

So this is the pre-flight to run on real weights before spending card time: it reports the layer
inventory the walkers actually find, checks it against the config, exercises the partial-ablation
guard in both directions, and states how much of the model a conv-blind tool would miss.

It loads on the CPU and touches no GPU, so it is safe to run while the card is busy.

Usage:
    python tools/inspect_model.py ~/models/LFM2.5-350M
"""
import argparse
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from senbonzakura import cli

# The walkers are module-private because nothing outside the package should depend on their
# shape. This tool is inside the repository and exists precisely to exercise them against real
# weights, so it reaches for them deliberately rather than through a public surface that would
# have to be invented for one caller.
_decoder_layers = cli._decoder_layers      # noqa: SLF001
_conv_outproj = cli._conv_outproj          # noqa: SLF001


def main(argv=None):
    ap = argparse.ArgumentParser(prog="inspect_model", description=__doc__.split("\n")[0])
    ap.add_argument("model", help="a local checkpoint directory")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args(argv)

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    types = list(getattr(cfg, "layer_types", []) or [])
    print(f"config     : {cfg.model_type}  hidden={cfg.hidden_size}  layers={cfg.num_hidden_layers}")
    if types:
        print(f"config     : {types.count('conv')} conv, {types.count('full_attention')} attention")

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, trust_remote_code=a.trust_remote_code)
    layers = _decoder_layers(model)
    if len(layers) != cfg.num_hidden_layers:
        print(f"FAIL: the walker found {len(layers)} decoder layers, the config declares "
              f"{cfg.num_hidden_layers}", file=sys.stderr)
        return 1

    # The guard, in both directions. Passing with the convolutions included and REFUSING without
    # them is what makes the control arm a deliberate choice rather than an oversight.
    cli.refuse_unrecognised_writers(layers, cfg.hidden_size, ablate_conv=True)
    print("guard      : passes with the convolution path included")
    try:
        cli.refuse_unrecognised_writers(layers, cfg.hidden_size, ablate_conv=False)
        conv_layers = 0
        print("guard      : nothing to refuse without it, so this model has no convolution blocks")
    except ValueError as e:
        conv_layers = int(str(e).split(" of ")[0])
        print(f"guard      : refuses without it, naming {conv_layers} layer(s)")

    convs = attn = fused = dense = listed = 0
    for i, layer in enumerate(layers):
        writers = cli.layer_attn_writers(layer)
        if len(writers) != 1:
            print(f"FAIL: layer {i} yields {len(writers)} attention-position writers, expected 1",
                  file=sys.stderr)
            return 1
        w = writers[0]
        if tuple(w.shape) != (cfg.hidden_size, cfg.hidden_size):
            print(f"FAIL: layer {i} writer is {tuple(w.shape)}, expected "
                  f"({cfg.hidden_size}, {cfg.hidden_size})", file=sys.stderr)
            return 1
        if _conv_outproj(layer) is not None:
            convs += 1
        else:
            attn += 1
        entries = cli.layer_downproj(layer)
        if not entries:
            print(f"FAIL: layer {i} has no residual-writing down-projection", file=sys.stderr)
            return 1
        for kind, _obj in entries:
            fused += kind == "fused3d"
            dense += kind == "dense"
            listed += kind == "list"

    print(f"walker     : {convs} convolution writers, {attn} attention writers, "
          f"all [{cfg.hidden_size},{cfg.hidden_size}]")
    print(f"walker     : down-projections {dense} dense, {fused} fused expert, {listed} unfused")
    if types and convs != types.count("conv"):
        print(f"FAIL: the walker found {convs} convolution writers, the config declares "
              f"{types.count('conv')}", file=sys.stderr)
        return 1
    if types:
        print("walker     : matches the config exactly")

    total = sum(len(cli.layer_attn_writers(layer))
                + sum(len(o) if k == "list" else 1 for k, o in cli.layer_downproj(layer))
                for layer in layers)
    if convs:
        print(f"bake reach : {total} residual writers; a conv-blind tool misses {convs} "
              f"({100 * convs / total:.0f}%)")
    else:
        print(f"bake reach : {total} residual writers")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
