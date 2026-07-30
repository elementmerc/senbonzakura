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

from .cli import accelerator_name, last_token_logits, load_model_and_tokenizer
from .crashsafe import provenance
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
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the bootstrap resampling, recorded in the result")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="bootstrap resamples for the AUC interval (0 to skip). At n=200 the "
                         "analytic standard error is about 0.029, so an AUC without an interval "
                         "invites a reader to believe a difference the data does not carry")
    ap.add_argument("--compare-to", default="",
                    help="a margins jsonl from a previous run on the same prompts (typically the "
                         "unabliterated model). Adds the PAIRED interval on the change, which is "
                         "much tighter than comparing two separate intervals by eye")
    ap.add_argument("--chat-template", dest="chat_template", default="",
                    help="Jinja chat template file, for a model that ships none. Prompt "
                         "format drives every measurement here, so a missing template is an "
                         "input you supply and the run records, not something the tool invents.")
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

    Computed from tie-averaged ranks rather than by comparing every pair. The two
    are identical by definition of U, and the tests assert it against a naive
    reference; the reason to care is the bootstrap, which needs thousands of AUCs.
    Pairwise is 200 x 200 = 40,000 comparisons each, so 2,000 resamples would be
    80 million of them in Python. Ranking is 400 log 400 instead.
    """
    if not pos or not neg:
        return None
    n, m = len(pos), len(neg)
    x = torch.tensor([*pos, *neg], dtype=torch.float64)
    # Tie-averaged 1-based ranks: for a group of `c` equal values whose last rank is
    # `e`, every member ranks (e - (c - 1) / 2). Exactly the half-credit for ties.
    _, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    last_rank = torch.cumsum(counts, 0).to(torch.float64)
    ranks = (last_rank - (counts.to(torch.float64) - 1) / 2)[inverse]
    rank_sum_pos = float(ranks[:n].sum())
    return (rank_sum_pos - n * (n + 1) / 2) / (n * m)


def _resampled_auc(pos, neg, gen):
    """One bootstrap replicate: resample prompts with replacement, within each arm."""
    n, m = len(pos), len(neg)
    pi = torch.randint(n, (n,), generator=gen)
    ni = torch.randint(m, (m,), generator=gen)
    return auc([pos[int(i)] for i in pi], [neg[int(i)] for i in ni])


def bootstrap_auc_ci(pos, neg, seed=0, resamples=2000, alpha=0.05):
    """Percentile confidence interval for one AUC, by resampling prompts.

    Prompts are the sampling unit, not pairs: the uncertainty being estimated is
    "what if we had drawn a different 200 prompts", and resampling pairs would
    answer a question nobody asked and give an interval that is too narrow.

    Seeded, so the interval is reproducible. Returns (low, high), or None when
    either arm is empty.
    """
    if not pos or not neg:
        return None
    gen = torch.Generator().manual_seed(int(seed))
    draws = sorted(_resampled_auc(pos, neg, gen) for _ in range(resamples))
    lo = draws[int((alpha / 2) * (resamples - 1))]
    hi = draws[int((1 - alpha / 2) * (resamples - 1))]
    return (round(lo, 4), round(hi, 4))


def paired_bootstrap_delta_ci(before, after, seed=0, resamples=2000, alpha=0.05):
    """Interval on the AUC change between two models measured on the SAME prompts.

    The pairing is the whole point. Two independent intervals that overlap do not
    mean the difference is uncertain: before and after share every prompt, so the
    prompt-to-prompt variation cancels and the paired interval is much tighter than
    subtracting two unpaired ones would suggest. So each replicate draws ONE set of
    prompt indices and applies it to both models.

    `before` and `after` are each (harmful_margins, harmless_margins), aligned by
    prompt. Returns a dict with the point delta and its interval, or None if the
    shapes do not line up, which the caller is expected to have refused already.
    """
    (bp, bn), (ap, an) = before, after
    if not bp or not bn or len(bp) != len(ap) or len(bn) != len(an):
        return None
    gen = torch.Generator().manual_seed(int(seed))
    n, m = len(bp), len(bn)
    draws = []
    for _ in range(resamples):
        pi = [int(i) for i in torch.randint(n, (n,), generator=gen)]
        ni = [int(i) for i in torch.randint(m, (m,), generator=gen)]
        a = auc([ap[i] for i in pi], [an[i] for i in ni])
        b = auc([bp[i] for i in pi], [bn[i] for i in ni])
        draws.append(a - b)
    draws.sort()
    lo = draws[int((alpha / 2) * (resamples - 1))]
    hi = draws[int((1 - alpha / 2) * (resamples - 1))]
    return {
        "delta_auc": round(auc(ap, an) - auc(bp, bn), 4),
        "delta_ci": (round(lo, 4), round(hi, 4)),
        # A sign that is not consistent across the resamples is the honest way to say
        # "this delta is not distinguishable from no change", without a p-value.
        "delta_crosses_zero": bool(lo <= 0.0 <= hi),
        "resamples": resamples, "seed": int(seed),
    }


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
    rows = [r["text"] for r in ds]
    if not rows:
        # Reachable, and only on part of the supported range: datasets 5.x raises inside
        # load_from_disk on an empty save_to_disk directory, while 2.15 loads it happily
        # and hands back nothing. Without this the same input produces a different error
        # depending on the installed version, which the dependency-floor job caught.
        raise SystemExit(f"the {what} dataset at {path} is empty")
    return rows


def load_margins_jsonl(path, harmful_prompts, harmless_prompts):
    """Read a previous run's per-prompt margins, aligned to this run's prompts.

    The alignment is checked rather than assumed. A paired interval computed on rows
    that are not actually the same prompts is not conservative, it is wrong: it
    reports a tight interval around a meaningless difference. So every row's prompt
    text must match, and a mismatch is fatal.
    """
    by_set = {"harmful": {}, "harmless": {}}
    try:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{lineno} is not valid JSON ({e})") from e
                kind = row.get("set")
                if kind not in by_set:
                    raise SystemExit(f"{path}:{lineno} has set={kind!r}, expected harmful or harmless")
                by_set[kind][row.get("i")] = (row.get("margin"), row.get("prompt"))
    except OSError as e:
        raise SystemExit(f"could not read --compare-to {path}: {e}") from e

    out = []
    for kind, prompts in (("harmful", harmful_prompts), ("harmless", harmless_prompts)):
        rows = by_set[kind]
        if len(rows) != len(prompts):
            raise SystemExit(f"--compare-to {path} has {len(rows)} {kind} rows but this run scores "
                             f"{len(prompts)}; a paired interval needs the same prompts")
        arm = []
        for i, prompt in enumerate(prompts):
            if i not in rows:
                raise SystemExit(f"--compare-to {path} is missing {kind} row {i}")
            margin_value, previous = rows[i]
            # A row without a usable margin would reach the bootstrap as None and die there,
            # thousands of resamples deep, with a TypeError naming neither the file nor the
            # row. The boundary is here.
            if not isinstance(margin_value, (int, float)) or isinstance(margin_value, bool):
                raise SystemExit(f"--compare-to {path} {kind} row {i} has margin={margin_value!r}, "
                                 f"which is not a number, so no interval can come from it")
            if previous is not None and previous != prompt:
                raise SystemExit(f"--compare-to {path} {kind} row {i} is a different prompt than "
                                 f"this run scores, so the two are not paired")
            arm.append(margin_value)
        out.append(arm)
    return tuple(out)


def main(argv=None):
    a = build_parser().parse_args(argv)
    model, tok = load_model_and_tokenizer(a.model, device=a.device,
                                          trust_remote_code=a.trust_remote_code,
                                          chat_template=a.chat_template)
    # Slice arithmetic makes a nonsense argument silently produce a plausible file rather
    # than an error: --n 0 scores nothing and then dies inside round(None), and a negative
    # skip reads the TAIL of the set, which is real data from the wrong partition.
    if a.n < 1:
        raise SystemExit(f"--n {a.n} scores no prompts; an AUC needs at least one per arm")
    for flag, value in (("--skip-harmful", a.skip_harmful), ("--skip-harmless", a.skip_harmless)):
        if value < 0:
            raise SystemExit(f"{flag} {value} is negative, which would read the end of the set "
                             f"instead of skipping its head")
    if a.compare_to and not a.bootstrap:
        raise SystemExit("--compare-to exists to produce the paired interval, and --bootstrap 0 "
                         "asks for no resampling; pick one")

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
        "auc_ci": bootstrap_auc_ci(mh, ml, seed=a.seed, resamples=a.bootstrap) if a.bootstrap else None,
        "bootstrap_resamples": a.bootstrap, "seed": a.seed,
        # Which prompt format produced these numbers, so two runs can be told apart.
        "chat_template": getattr(tok, "senbon_chat_template", None),
        # Which tokens were actually scored, decoded. The AUC is a difference between
        # two logits and there is no way to check it later without knowing which two:
        # a tokenizer change that moves HARMFUL onto a different first token produces a
        # perfectly plausible number for a different measurement.
        "verdict_tokens": {
            "HARMFUL": [{"id": i, "text": tok.decode([i])} for i in hid],
            "BENIGN": [{"id": i, "text": tok.decode([i])} for i in bid],
        },
        "provenance": provenance(device=a.device, accelerator=accelerator_name(a.device)),
        "mean_margin_harmful": round(sum(mh) / len(mh), 4),
        "mean_margin_harmless": round(sum(ml) / len(ml), 4),
        # A model whose margin never crosses zero always says the same thing. That
        # is exactly the case the verdict count could not distinguish from
        # ignorance, so record it rather than leaving it to be inferred.
        "frac_harmful_positive": round(sum(m > 0 for m in mh) / len(mh), 4),
        "frac_harmless_positive": round(sum(m > 0 for m in ml) / len(ml), 4),
    }
    if a.compare_to:
        before = load_margins_jsonl(a.compare_to, harmful, harmless)
        res["paired"] = paired_bootstrap_delta_ci((*before,), (mh, ml),
                                                  seed=a.seed, resamples=a.bootstrap)
        res["compared_to"] = a.compare_to

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

    ci = f" ci=[{res['auc_ci'][0]:.4f},{res['auc_ci'][1]:.4f}]" if res["auc_ci"] else ""
    print(f"MARGIN_DONE {a.label} auc={score:.4f}{ci} "
          f"mean_h={res['mean_margin_harmful']:.3f} mean_l={res['mean_margin_harmless']:.3f} "
          f"says_harmful_h={res['frac_harmful_positive']*100:.1f}% "
          f"says_harmful_l={res['frac_harmless_positive']*100:.1f}%")
    if res.get("paired"):
        p = res["paired"]
        verdict = "CROSSES ZERO" if p["delta_crosses_zero"] else "excludes zero"
        print(f"MARGIN_PAIRED {a.label} delta={p['delta_auc']:+.4f} "
              f"ci=[{p['delta_ci'][0]:+.4f},{p['delta_ci'][1]:+.4f}] {verdict}")
    return res


if __name__ == "__main__":   # pragma: no cover
    main()
