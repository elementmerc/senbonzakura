#!/usr/bin/env python3
"""Canonical fixed-eval scorer for the <5% chase.

Every abliteration variation is scored HERE, on the SAME eval-fixed set, with
the SAME metrics.py detectors. That is the whole point: the old per-track eval
made variations incomparable. Reports the full breakdown (hard refusal / soft
refusal / noncompliance / broken / heretic-keyword) so we can see WHICH axis a
lever moves. Getting hard-refusal down is easy; the residual is usually soft
refusal + evasion (noncompliance), which is the real wall before <5%.

Run with cwd = ~/track2 so `import metrics` resolves.
"""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from . import metrics

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--eval", required=True, help="path to eval-fixed (column 'text')")
ap.add_argument("--out", required=True, help="results json path")
ap.add_argument("--label", default="")
ap.add_argument("--n", type=int, default=0, help="0 = all prompts")
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = (
    AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    .to(a.device)
    .eval()
)

ds = load_from_disk(a.eval)
prompts = [r["text"] for r in ds]
if a.n:
    prompts = prompts[: a.n]

gens = []
for i in range(0, len(prompts), a.batch):
    batch = prompts[i : i + a.batch]
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in batch
    ]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(
        a.device
    )
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=a.max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    for j in range(len(batch)):
        new = out[j][enc.input_ids.shape[1] :]
        gens.append(tok.decode(new, skip_special_tokens=True))

res = {
    "label": a.label,
    "model": a.model,
    "eval": a.eval,
    "n": len(gens),
    "refusal": round(metrics.refusal_rate(gens), 4),
    "soft_refusal": round(metrics.soft_refusal_rate(gens), 4),
    "noncompliant": round(metrics.noncompliance_rate(gens), 4),
    "broken": round(metrics.broken_rate(gens), 4),
    "heretic": round(metrics.heretic_keyword_rate(gens), 4),
}
with open(a.out, "w") as f:
    json.dump(res, f, indent=2)
print(
    f"SCORE_DONE {a.label} refusal={res['refusal']*100:.1f}% "
    f"soft={res['soft_refusal']*100:.1f}% noncompliant={res['noncompliant']*100:.1f}% "
    f"broken={res['broken']*100:.1f}% heretic={res['heretic']*100:.1f}% n={res['n']}"
)
