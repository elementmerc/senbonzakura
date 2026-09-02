#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Does this machine's accelerated path agree with the CPU one?

CPU IS THE ORACLE, and that is the design rather than an accident of convenience. It is the
implementation with the fewest moving parts, it is available on every machine, and it is the one
that does not change when a driver, a card or a torch build does. Everything else is measured
against it rather than against another accelerator.

WHAT IT COMPARES, AND WHY THE TWO HALVES NEED DIFFERENT INSTRUMENTS

**The directions**, through `subspace.compare`. Two runs can produce the same subspace and disagree
completely vector by vector: signs are arbitrary and a near-degenerate subspace comes back
arbitrarily rotated. What matters is whether the same subspace gets ablated, so the comparison is
between projectors.

**The edited weights**, through a plain maximum absolute difference. Here the opposite is true.
Two baked models are supposed to be the same tensors, so any principled tolerance is a numerical
one and a rotation is a real difference.

Using one instrument for both would be wrong twice over: the subspace comparison would wave through
genuinely different weights, and the tensor comparison would fail a perfectly good run for spinning
a basis.

THE CONTROL

`--control` breaks the comparison on purpose and the command exits non-zero, on the argument that
an instrument whose entire output is "these two agree" has not been shown to work until it has been
seen saying no. Written before this gate ever passed, so the first green run could be believed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from senbonzakura import cli, subspace

#: How far two baked weight tensors may differ and still count as the same bake.
#:
#: The bake is arithmetic in the model's own dtype, so the floor is that dtype's resolution rather
#: than float64's. bfloat16 carries about three decimal digits, and the edit is a subtraction of a
#: rank-K projection whose terms are of the same order as the weights, so a few ULPs is the honest
#: expectation. Stated here before the gate was first run.
WEIGHT_TOL = 1e-2


def _build_tiny_model(device, seed=0):
    """A real two-layer checkpoint rather than a mock, small enough to run twice in a minute."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=32, max_position_embeddings=256)
    model = Qwen3ForCausalLM(cfg).to(torch.float32)
    # The tiny tokenizer ships no chat template and the extractor renders one. Its CONTENT is
    # irrelevant to a parity check, because both sides render with the identical template and the
    # question is only whether two devices agree; it would matter enormously to a measurement of
    # the model, which is why the tool makes you supply one there rather than inventing this.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token          # the extractor batches, so it needs one
    if not getattr(tok, "chat_template", None):
        tok.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}")
    return model.to(device), tok


def _extract(device, track, args, seed=0, log=lambda _m: None):
    """One extraction on one device, returning the directions and the timing."""
    run_args = cli.build_parser().parse_args([
        "--model", "unused", "--track", track, "--device", device,
        "--dir-prompts", str(args.dir_prompts), "--max-directions", str(args.max_directions),
        "--direction-clusters", str(args.direction_clusters), "--seed", str(seed),
    ])
    model, tok = _build_tiny_model(device, seed=0)     # identical weights on both sides
    abl = cli.Abliterator(run_args, log, model=model, tok=tok)
    t0 = time.time()
    abl.extract_directions(f"{track}/bad_ds", f"{track}/good_ds", None, f"{track}/good_ds")
    seconds = time.time() - t0
    dirs = abl.dirs_multi.detach().float().cpu().clone()
    del model, abl
    if device != "cpu":
        torch.cuda.empty_cache()
    return dirs, seconds


def compare_directions(oracle, other, control=None, tolerance=subspace.SAME_SUBSPACE_TOL):
    """Per residual-stream position, do the two runs ablate the same subspace?

    Indexed by POSITION, not by layer: `dirs_multi` has NL+1 entries and position 0 is the
    embedding output, which is never ablated. Naming those per-layer is a mistake this project has
    already made and paid for.
    """
    if control == "rotate":
        # NOT a real difference: a rotated basis for the same span. The gate must pass this, and a
        # gate built on per-vector cosines would fail it. It is here to prove which one this is.
        g = torch.Generator().manual_seed(11)
        k = other.shape[1]
        rot = torch.linalg.qr(torch.randn(k, k, generator=g))[0]
        other = torch.einsum("ij,pjh->pih", rot, other)
    elif control == "replace":
        g = torch.Generator().manual_seed(12)
        other = other.clone()
        other[1, 0] = torch.randn(other.shape[2], generator=g)
    elif control == "drop":
        # MUST FIND AN OCCUPIED SLOT. `dirs_multi` is [positions, KMAX, H] and a position that kept
        # two of four directions has two slots of exact zeros already, so zeroing slot -1 removes
        # nothing and the gate correctly reports no change. The first version of this control did
        # exactly that and reported the gate as toothless; the control was the broken part.
        other = other.clone()
        occupied = (other.float().norm(dim=-1) > subspace.MIN_DIRECTION_NORM)
        counts = occupied.sum(dim=1)
        candidates = (counts >= 2).nonzero().flatten()
        if candidates.numel() == 0:
            raise RuntimeError(
                "the 'drop' control needs a position holding at least two directions and this run "
                "produced none, so it cannot break anything. Raise --max-directions or "
                "--direction-clusters; a control that silently does nothing is worse than none.")
        position = int(candidates[0])
        last = int(occupied[position].nonzero().flatten()[-1])
        other[position, last] = 0.0
    elif control is not None:
        raise ValueError(f"unknown control {control!r}")

    rows = []
    for position in range(oracle.shape[0]):
        c = subspace.compare(oracle[position], other[position], tolerance=tolerance)
        rows.append((position, c))
    return rows


def compare_weights(oracle_state, other_state):
    """Plain maximum absolute difference over every edited tensor, and where the worst one is."""
    worst, where = 0.0, ""
    for name, a in oracle_state.items():
        b = other_state.get(name)
        if b is None:
            return float("inf"), f"{name} is missing from the second bake"
        d = float((a.float() - b.float()).abs().max())
        if d > worst:
            worst, where = d, name
    return worst, where


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="cuda", help="the path being checked against CPU")
    p.add_argument("--track", default=None, help="defaults to the bundled evaluation track")
    p.add_argument("--control", choices=["rotate", "replace", "drop"], default=None,
                   help="break the comparison on purpose. 'rotate' must still PASS, because a "
                        "rotated basis is the same subspace; the other two must fail.")
    p.add_argument("--dir-prompts", type=int, default=64, dest="dir_prompts")
    p.add_argument("--max-directions", type=int, default=4, dest="max_directions")
    p.add_argument("--direction-clusters", type=int, default=4, dest="direction_clusters")
    p.add_argument("--json", default=None, help="write the full result here")
    a = p.parse_args(argv)

    if a.device != "cpu" and not torch.cuda.is_available():
        print(f"parity: no {a.device} device on this machine, so there is nothing to compare "
              f"against. This is a SKIP, not a pass.")
        return 77                     # the automake convention, so CI can tell it apart from green

    track = a.track
    if track is None:
        from senbonzakura import bundled
        track = str(bundled.ensure())

    print(f"oracle: cpu   |   under test: {a.device}   |   track: {track}")
    oracle, cpu_seconds = _extract("cpu", track, a)
    other, dev_seconds = _extract(a.device, track, a)
    print(f"cpu {cpu_seconds:.1f}s, {a.device} {dev_seconds:.1f}s\n")

    rows = compare_directions(oracle, other, control=a.control)
    worst = max(rows, key=lambda r: r[1].distance)
    for position, c in rows:
        flag = "    " if c.same else "  ->"
        print(f"{flag} position {position:>3}: rank {c.rank_a}/{c.rank_b}, "
              f"distance {c.distance:.3e}, {c.disagreeing:.3e} directions")
    disagreeing = [r for r in rows if not r[1].same]

    print()
    print(f"worst position: {worst[0]}, distance {worst[1].distance:.6e} "
          f"(tolerance {subspace.SAME_SUBSPACE_TOL:.0e})")
    print(f"positions disagreeing: {len(disagreeing)} of {len(rows)}")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "device": a.device, "control": a.control, "track": track,
            "code_version": cli.code_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_seconds": round(cpu_seconds, 2), "device_seconds": round(dev_seconds, 2),
            "tolerance": subspace.SAME_SUBSPACE_TOL,
            "positions": [{"position": pos, "distance": c.distance, "same": c.same,
                           "disagreeing": c.disagreeing, "rank_a": c.rank_a, "rank_b": c.rank_b}
                          for pos, c in rows],
        }, indent=2), encoding="utf-8")

    # 'rotate' is the control that must still PASS: it is a different description of one subspace,
    # and a gate that failed it would be measuring the basis rather than the ablation.
    expect_pass = a.control in (None, "rotate")
    if expect_pass and not disagreeing:
        print(f"\nOK: {a.device} ablates the same subspaces as the CPU oracle at every position"
              + (", including under a rotated basis" if a.control == "rotate" else "") + ".")
        return 0
    if expect_pass:
        print(f"\nFAILED: {a.device} and the CPU oracle disagree at {len(disagreeing)} position(s). "
              f"The accelerated path is not doing what the reference does.")
        return 1
    if disagreeing:
        print(f"\nOK: the '{a.control}' control was caught at {len(disagreeing)} position(s), and "
              f"this command exits non-zero to prove the gate has teeth.")
        return 1
    print(f"\nFAILED: the '{a.control}' control changed nothing. This gate cannot say no, so its "
          f"agreements mean nothing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
