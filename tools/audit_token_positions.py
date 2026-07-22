#!/usr/bin/env python3
"""Audit WHERE refusal lives along the token sequence, not just at the last token.

Senbonzakura (like Arditi and Heretic) reads the refusal direction from the LAST prompt token's
residual stream. That is an assumption, not a law: on some models the refusal signal peaks a few
tokens earlier (a trigger token), and a mean-difference taken at the wrong site leaves signal on the
table. This is a read-only diagnostic (it never edits weights). It loads a model and the same
harmful/harmless contrast the search uses, measures the harmful-vs-harmless separation (Cohen's d
along the mean-difference direction) at each of the last few positions, per layer, and reports where
the separation peaks. If the peak is not the last token, the last-token extraction is suboptimal on
that model and worth revisiting.

Usage:
  python tools/audit_token_positions.py --model <hf-id> --track <dir with bad_ds/good_ds> \
      [--device cuda] [--positions 8] [--n 64] [--layer-lo 0.3] [--layer-hi 0.9]
"""
import argparse

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(path, n):
    ds = load_from_disk(path)
    return [ds[i]["text"] for i in range(min(n, len(ds)))]


def chat(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt


@torch.no_grad()
def tail_residuals(model, tok, prompts, positions, device):
    # Return [n_prompts, n_layers, positions, H]: the last `positions` token residuals at every
    # layer, per prompt. One prompt at a time (batch 1) so no padding token ever pollutes a slot;
    # a prompt shorter than `positions` is padded at the FRONT with its own earliest token so every
    # slot is a real token (position 0 = last token, 1 = second-last, ...).
    out = []
    for p in prompts:
        enc = tok(chat(tok, p), return_tensors="pt", add_special_tokens=False).to(device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states  # tuple[L] of [1,T,H]
        stack = torch.stack([h[0] for h in hs], 0)                                   # [L, T, H]
        T = stack.shape[1]
        take = min(positions, T)
        tail = stack[:, -take:, :]                                                   # [L, take, H]
        if take < positions:                                                         # front-pad short prompts
            pad = tail[:, :1, :].expand(-1, positions - take, -1)
            tail = torch.cat([pad, tail], 1)
        out.append(tail.float().cpu())
    return torch.stack(out, 0)                                                       # [N, L, positions, H]


def cohens_d(bad, good):
    # Separation along the mean-difference direction (the site's own refusal axis).
    dm = bad.mean(0) - good.mean(0)
    u = dm / dm.norm().clamp_min(1e-8)
    pb, pg = bad @ u, good @ u
    pooled = ((pb.var(unbiased=False) + pg.var(unbiased=False)) / 2).clamp_min(1e-12).sqrt()
    return ((pb.mean() - pg.mean()).abs() / pooled).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", required=True, help="dir holding bad_ds / good_ds (save_to_disk, 'text' column)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--positions", type=int, default=8, help="how many trailing token positions to profile")
    ap.add_argument("--n", type=int, default=64, help="prompts per side")
    ap.add_argument("--layer-lo", type=float, default=0.3)
    ap.add_argument("--layer-hi", type=float, default=0.9)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device).eval()

    bad = load_prompts(f"{args.track}/bad_ds", args.n)
    good = load_prompts(f"{args.track}/good_ds", args.n)
    print(f"model {args.model}: {len(bad)} harmful / {len(good)} harmless, last {args.positions} positions")

    Rb = tail_residuals(model, tok, bad, args.positions, args.device)   # [Nb, L, P, H]
    Rg = tail_residuals(model, tok, good, args.positions, args.device)  # [Ng, L, P, H]
    L = Rb.shape[1]
    lo, hi = max(1, int(L * args.layer_lo)), min(L - 1, int(L * args.layer_hi))

    # Aggregate the separation over the searched layer band, per trailing position.
    print(f"\n  {'pos-from-end':>12}  {'mean Cohen d (layers ' + str(lo) + '..' + str(hi) + ')':>28}")
    per_pos = []
    for m in range(args.positions):
        ds = [cohens_d(Rb[:, li, m, :], Rg[:, li, m, :]) for li in range(lo, hi + 1)]
        d = float(np.mean(ds))
        per_pos.append(d)
        label = "LAST" if m == 0 else f"-{m}"
        print(f"  {label:>12}  {d:>28.3f}")

    peak = int(np.argmax(per_pos))
    if peak == 0:
        print("\nVERDICT: refusal separation peaks at the LAST token. Senbon's last-token extraction is well sited.")
    else:
        print(f"\nVERDICT: refusal separation peaks at position -{peak} (d={per_pos[peak]:.3f}), NOT the last token "
              f"(d={per_pos[0]:.3f}). The last-token mean-difference is leaving signal on the table on this model; "
              f"consider extracting at -{peak} or averaging the last {peak + 1} positions.")


if __name__ == "__main__":
    main()
