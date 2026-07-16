#!/usr/bin/env python3
"""Canonical coherence probe.

Coherence, measured the same way every time: the negative log-likelihood (and its
perplexity) the model assigns to one fixed, neutral passage of ordinary English. A
clean model predicts this prose comfortably; a model that has been cut too deep
during abliteration stumbles on it, and the perplexity climbs. Reported next to
the refusal breakdown from `senbonzakura.score`, this is the "did the surgery keep
the patient coherent" half of the ruler.

One passage, one number, loaded through the SAME shared loader the scorer uses, so
the flags behave identically (`--load-in-4bit`, `--device`, `--trust-remote-code`)
and no caller can trip an unknown-argument error that masquerades as a coherence
failure. Lower perplexity is better; compare a bake against its own stock model,
not against an absolute threshold.
"""
import argparse
import json
import math

import torch

from .cli import load_model_and_tokenizer

# A fixed, deliberately unremarkable passage. It touches no refusal-adjacent topic,
# so a rising perplexity here is coherence damage, not the model balking at content.
NEUTRAL = (
    "The printing press was among the most consequential inventions of the last "
    "thousand years. Before it, a book had to be copied out by hand, a process that "
    "could take a scribe the better part of a year for a single volume. Errors crept "
    "in with every copy, and no two manuscripts were ever quite the same. When "
    "movable type arrived in Europe in the middle of the fifteenth century, the cost "
    "of producing a book collapsed, and the number of books in circulation grew "
    "faster than anyone had thought possible. Ideas that had once been confined to a "
    "handful of monasteries could now travel across a continent in a matter of "
    "months. Literacy spread, first among merchants and clerks, then more widely, "
    "and with it came an appetite for news, argument, and instruction. The technology "
    "did not care what it printed. The same press that turned out prayer books turned "
    "out pamphlets, almanacs, and eventually newspapers. Standardisation followed: "
    "spelling settled, page numbers appeared, tables of contents and indexes made it "
    "possible to find a passage without reading the whole work. In time the book "
    "became an object designed to be searched as much as read. Centuries later the "
    "same pattern would repeat with the network, where the cost of copying fell to "
    "almost nothing and the difficulty shifted from making copies to deciding which "
    "of them deserved attention."
)


def build_parser():
    ap = argparse.ArgumentParser(prog="senbonzakura.coherence",
                                 description="Measure a model's coherence as neutral-passage perplexity.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="results json path")
    ap.add_argument("--label", default="")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true",
                    help="load in 4-bit (bitsandbytes nf4) to measure a large model on low VRAM. This is a "
                         "pure forward pass, so 4-bit is safe here (unlike the abliterator's bake).")
    ap.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true",
                    help="allow models that ship custom modelling code.")
    return ap


def coherence(model, tok, text=NEUTRAL):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        nll = model(ids, labels=ids).loss.item()
    return {"nll": nll, "ppl": math.exp(nll), "n_tokens": int(ids.shape[1])}


def main(argv=None):
    a = build_parser().parse_args(argv)
    model, tok = load_model_and_tokenizer(
        a.model, device=a.device, load_in_4bit=a.load_in_4bit, trust_remote_code=a.trust_remote_code)
    res = {"label": a.label, "model": a.model, **coherence(model, tok)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"COHERENCE_DONE {a.label} ppl={res['ppl']:.2f} nll={res['nll']:.4f} n_tokens={res['n_tokens']}")
    return res


if __name__ == "__main__":   # pragma: no cover
    main()
