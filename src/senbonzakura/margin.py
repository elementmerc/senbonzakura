#!/usr/bin/env python3
"""Criterion-free harm judgement: the HARMFUL vs BENIGN logit margin.

Why this exists
---------------
The compass asked the model for a verdict and counted the answers. That makes the
score depend on where the model happens to put its decision threshold, and small
instruct models put it in wild places. One model in the 2026-07 sweep answered
HARMFUL to all 200 harmless prompts, scoring a perfect 100% on an all-harmful
eval; another answered HARMFUL to 93.5% of harmless prompts while still being a
genuinely useful discriminator underneath. Counting verdicts cannot tell those two
apart, and balanced accuracy only papers over it: it mixes discriminability with
criterion, so a pinned threshold reads as ignorance.

This measures the thing underneath. One forward pass per prompt, no generation.
Take the logit of the first token of HARMFUL and of BENIGN at the position where
the verdict would begin, and use their difference as a continuous score. Rank all
the harmful prompts against all the harmless ones and you get AUC, which is
invariant to any monotonic shift in the threshold: a model that always SAYS
harmful can still rank harmful above harmless, and AUC will say so.

It also deletes an entire class of bug. No first line parsing, no reasoning
blocks to strip, no verdict-absent fallback, no token budget, no truncation. Both
scoring errors found in the 2026-07 sweep were parsing errors that only existed
because the pipeline read prose.
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk

from .cli import last_token_logits, load_model_and_tokenizer
from .resources import ResourceGovernor
from .score import JUDGE_TEMPLATE


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura.margin",
        description="Criterion-free harm discrimination via the HARMFUL/BENIGN logit margin.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--harmful", required=True, help="dataset of harmful prompts (column 'text')")
    ap.add_argument("--harmless", required=True, help="dataset of harmless prompts (column 'text')")
    ap.add_argument("--out", required=True, help="results json path")
    # Retention is on by default. Both scoring errors in this project's history were
    # invisible in the percentages and obvious in the per-prompt rows, so the rows are
    # kept unless someone asks for them not to be.
    ap.add_argument("--margins", default=None,
                    help="jsonl of every per-prompt margin (default: alongside --out, as "
                         "<out>.margins.jsonl). These rows hold the prompts, so they belong in "
                         "the ignored results/ tree, not in committed evidence.")
    ap.add_argument("--no-margins", dest="margins", action="store_const", const="",
                    help="do not retain the per-prompt margins")
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--skip-harmless", type=int, default=320,
                    help="drop the head of the harmless set, which is where the abliteration "
                         "directions and the drift check were fitted from")
    ap.add_argument("--skip-harmful", type=int, default=128,
                    help="drop the head of the harmful set, which is where the search selected "
                         "its winning trial from. 128 is the largest --eval-refusal-final any "
                         "auto preset uses, so it covers every prompt the search could have seen "
                         "(0 to score the selection set too, which is not a held-out number)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    return ap


def label_token_ids(tok, word):
    """First-token ids for a verdict word, across the spellings a model may use.

    Tokenizers disagree about leading spaces and case, and picking the wrong
    variant silently measures a token the model never emits. Collect every
    plausible first token and take the best one at scoring time.
    """
    ids = set()
    for variant in (word, " " + word, word.capitalize(), " " + word.capitalize(),
                    word.lower(), " " + word.lower()):
        enc = tok.encode(variant, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


@torch.no_grad()
def margins(model, tok, prompts, harmful_ids, benign_ids, device, batch=16, gov=None, log=None):
    """Per prompt: max logit over HARMFUL spellings minus max over BENIGN spellings.

    Chunked by the shared ResourceGovernor rather than a fixed stride, so the compass
    shrinks its batch under VRAM pressure and pauses instead of dying, the same way
    every other batched pass in this project already did. On CPU the governor is
    inert and chunks at the ceiling, so the numbers are identical either way.
    """
    def _do(chunk):
        texts = [tok.apply_chat_template([{"role": "user", "content": JUDGE_TEMPLATE.format(p)}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048, add_special_tokens=False).to(device)
        logits = last_token_logits(model, enc, log)
        h = logits[:, harmful_ids].max(dim=-1).values
        b = logits[:, benign_ids].max(dim=-1).values
        return (h - b).cpu().tolist()

    gov = gov or ResourceGovernor(device, log, max_batch=batch)
    return gov.run(_do, list(prompts))


def auc(pos, neg):
    """Mann-Whitney U as AUC: P(a harmful prompt outranks a harmless one).

    0.5 is chance. Ties count a half, which matters because a saturated model can
    produce identical margins across many prompts.
    """
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def load_prompts(path, what):
    """Read the 'text' column of a save_to_disk dataset, failing readably at the boundary.

    load_from_disk raises a bare IndexError on an empty dataset and a KeyError on a
    missing column, neither of which names the file or the flag that pointed at it.
    """
    try:
        ds = load_from_disk(path)
    except Exception as e:
        raise SystemExit(f"could not load the {what} dataset at {path}: {e}. Expected a "
                         f"datasets.save_to_disk directory with a 'text' column.") from e
    cols = list(getattr(ds, "column_names", None) or [])
    if "text" not in cols:
        raise SystemExit(f"the {what} dataset at {path} has columns {cols} and no 'text' column")
    # No empty-set check here on purpose: an empty save_to_disk directory fails inside
    # load_from_disk above, and a set too small for --n is caught by the fit check in
    # main(), which reports the row count and both flags.
    return [r["text"] for r in ds]


def main(argv=None):
    a = build_parser().parse_args(argv)
    model, tok = load_model_and_tokenizer(a.model, device=a.device,
                                          trust_remote_code=a.trust_remote_code)
    harmful_all = load_prompts(a.harmful, "harmful")
    harmless_all = load_prompts(a.harmless, "harmless")
    harmful = harmful_all[a.skip_harmful:a.skip_harmful + a.n]
    harmless = harmless_all[a.skip_harmless:a.skip_harmless + a.n]
    # Both arms, the same check. Only the harmless arm was checked before, so asking
    # for 200 harmful prompts against a 150-row set quietly scored 150 and printed a
    # result that looked like every other one.
    for what, got, rows, skip in (("harmful", harmful, harmful_all, a.skip_harmful),
                                  ("harmless", harmless, harmless_all, a.skip_harmless)):
        if len(got) < a.n:
            raise SystemExit(f"{what} set has {len(rows)} rows; --skip-{what} {skip} + --n {a.n} "
                             f"does not fit (it would score {len(got)})")

    hid = label_token_ids(tok, "HARMFUL")
    bid = label_token_ids(tok, "BENIGN")
    if not hid or not bid:
        raise SystemExit("could not resolve verdict token ids for this tokenizer")
    # Overlapping ids would take the same logit as the max on both sides of the
    # subtraction, pulling every margin toward zero and toward AUC 0.5. It would look
    # like a model that cannot discriminate rather than a ruler that cannot measure.
    shared = sorted(set(hid) & set(bid))
    if shared:
        raise SystemExit(f"this tokenizer maps HARMFUL and BENIGN onto shared first-token ids "
                         f"{shared}, so the margin cannot separate them")

    # None means "not asked either way", so it takes the default beside --out. An
    # explicit empty string is --no-margins, and must not be overwritten by the default.
    margins_path = a.margins
    if margins_path is None:
        margins_path = str(Path(a.out).with_suffix("")) + ".margins.jsonl"

    mh = margins(model, tok, harmful, hid, bid, a.device, a.batch)
    ml = margins(model, tok, harmless, hid, bid, a.device, a.batch)
    score = auc(mh, ml)

    res = {
        "label": a.label, "model": a.model, "mode": "logit_margin",
        "n_harmful": len(mh), "n_harmless": len(ml),
        # Both skips, recorded, because "held out" is a claim about these two numbers
        # and a reader cannot recover them from the AUC.
        "skip_harmful": a.skip_harmful, "skip_harmless": a.skip_harmless,
        "margins_path": margins_path,
        "auc": round(score, 4),
        "mean_margin_harmful": round(sum(mh) / len(mh), 4),
        "mean_margin_harmless": round(sum(ml) / len(ml), 4),
        # A model whose margin never crosses zero always says the same thing. That
        # is exactly the case the verdict count could not distinguish from
        # ignorance, so record it rather than leaving it to be inferred.
        "frac_harmful_positive": round(sum(m > 0 for m in mh) / len(mh), 4),
        "frac_harmless_positive": round(sum(m > 0 for m in ml) / len(ml), 4),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    if margins_path:
        Path(margins_path).parent.mkdir(parents=True, exist_ok=True)
        with open(margins_path, "w", encoding="utf-8") as f:
            for kind, ms, ps in (("harmful", mh, harmful), ("harmless", ml, harmless)):
                # strict=True: a margin count that has drifted from its prompt count means the
                # rows are misaligned, and every margin after the drift is attributed to the
                # wrong prompt. Fail rather than write a file that reads as valid.
                for i, (m, p) in enumerate(zip(ms, ps, strict=True)):
                    f.write(json.dumps({"i": i, "set": kind, "margin": m, "prompt": p}) + "\n")

    print(f"MARGIN_DONE {a.label} auc={score:.4f} "
          f"mean_h={res['mean_margin_harmful']:.3f} mean_l={res['mean_margin_harmless']:.3f} "
          f"says_harmful_h={res['frac_harmful_positive']*100:.1f}% "
          f"says_harmful_l={res['frac_harmless_positive']*100:.1f}%")
    return res


if __name__ == "__main__":   # pragma: no cover
    main()
