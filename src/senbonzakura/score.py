#!/usr/bin/env python3
"""Canonical fixed-eval scorer.

Every abliteration variation is scored HERE, on the SAME eval-fixed set, with the SAME
metrics.py detectors. That is the whole point: a per-track eval makes variations
incomparable. Reports the full breakdown (hard refusal / soft refusal / noncompliance /
broken / heretic-keyword) so we can see WHICH axis a lever moves. Getting hard-refusal down
is easy; the residual is usually soft refusal + evasion (noncompliance), the real wall.
"""
import argparse
import json

import torch
from datasets import load_from_disk

from . import metrics
from .cli import load_model_and_tokenizer


def build_parser():
    ap = argparse.ArgumentParser(prog="senbonzakura.score",
                                 description="Score a model's refusal / coherence on a fixed eval set.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True, help="path to eval-fixed dataset (column 'text')")
    ap.add_argument("--out", required=True, help="results json path")
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=0, help="0 = all prompts")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true",
                    help="load in 4-bit (bitsandbytes nf4) to score a large model on low VRAM. Scoring "
                         "is pure forward passes, so 4-bit is safe here (unlike the abliterator's bake).")
    ap.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true",
                    help="allow models that ship custom modelling code.")
    return ap


def generate(model, tok, prompts, device, batch=16, max_new=64):
    gens = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            gens.append(tok.decode(out[j][enc.input_ids.shape[1]:], skip_special_tokens=True))
    return gens


def score(gens, label="", model="", eval_path=""):
    return {
        "label": label, "model": model, "eval": eval_path, "n": len(gens),
        "refusal": round(metrics.refusal_rate(gens), 4),
        "soft_refusal": round(metrics.soft_refusal_rate(gens), 4),
        "noncompliant": round(metrics.noncompliance_rate(gens), 4),
        "broken": round(metrics.broken_rate(gens), 4),
        "heretic": round(metrics.heretic_keyword_rate(gens), 4),
    }


def main(argv=None):
    a = build_parser().parse_args(argv)
    model, tok = load_model_and_tokenizer(
        a.model, device=a.device, load_in_4bit=a.load_in_4bit, trust_remote_code=a.trust_remote_code)
    ds = load_from_disk(a.eval)
    prompts = [r["text"] for r in ds]
    if a.n:
        prompts = prompts[:a.n]
    gens = generate(model, tok, prompts, a.device, batch=a.batch, max_new=a.max_new)
    res = score(gens, label=a.label, model=a.model, eval_path=a.eval)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"SCORE_DONE {a.label} refusal={res['refusal']*100:.1f}% "
          f"soft={res['soft_refusal']*100:.1f}% noncompliant={res['noncompliant']*100:.1f}% "
          f"broken={res['broken']*100:.1f}% heretic={res['heretic']*100:.1f}% n={res['n']}")
    return res


if __name__ == "__main__":   # pragma: no cover
    main()
