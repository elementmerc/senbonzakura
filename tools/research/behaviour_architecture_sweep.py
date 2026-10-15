#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every behaviour against every architecture, with a verdict per cell.

WHAT THIS ANSWERS, which is not what the 2026-10-14 sweep answered

That sweep ran thirteen behaviours on two models and both were dense. The gap it left is
architecture: a behaviour that extracts on a dense stack and not on a hybrid or a mixture of
experts is a behaviour this product cannot ship, and nothing so far could tell the difference.
So the axes here are behaviour by architecture, and the deliverable is a map rather than a result.

THE FOUR MEASUREMENTS PER CELL, cheapest first, and each one can rule a cell out

    1. SEPARATION      does a direction fitted on one half tell the arms apart on the other half?
                       Residuals only, no generation. Reported with a bootstrap interval over
                       groups and against a shuffled-label null computed on the same residuals.
    2. LENGTH CONTROL  can prompt length alone tell the arms apart? If it can, the separation
                       score may be reading the word count, which caught six candidates last time.
    3. TRANSFER        for an instructed contrast only: extract the direction from one wording of
                       the instruction and score it on another. A direction that was reading the
                       literal words collapses here.
    4. ELICITATION     does the instruction change what the model WRITES? Generation, but no judge:
                       the refusal rate per arm (the shipped ruler), how far apart the two arms'
                       replies are, and how much they differ in length. This is the measurement
                       the toxicity work said was missing, and it is behaviour-agnostic.

WHY ELICITATION IS HERE AND WHY IT MATTERS MOST

The toxicity channel's own report found that "be rude" means different things to different
architectures: contempt for the asker on one, disparagement of the subject on another, a refusal
on a third, and nothing at all on a fourth. That is upstream of every channel. A separation score
cannot see it, because the arms' PROMPTS differ whether or not the model does anything about them.
So every instructed cell carries an elicitation reading, and a cell whose arms produce near
identical replies is reported as not elicited however well it separates.

WHAT THIS DOES NOT DO

No causal test and no rate. A rate needs a judge per behaviour, and this project has exactly one
validated judge (`toxicity`). Saying which cells deserve a judge is this sweep's job; building
fourteen of them is not. Every cell's causal column therefore reads "not tested", and the one
exception is recorded with a pointer to where it was tested.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from senbonzakura import attribution as A
from senbonzakura import attribution_hf as H
from senbonzakura import contrasts, metrics

#: Bootstrap resamples for every interval, matching `metrics.DEFAULT_RESAMPLES` so a reader meets
#: one instrument across this project's artefacts.
RESAMPLES = metrics.DEFAULT_RESAMPLES

#: Shuffled-label replicates for the separation null. Pure arithmetic on cached residuals: the
#: labels are permuted, the direction refitted and rescored, and nothing touches the model, so
#: twenty replicates cost milliseconds rather than forward passes.
NULL_REPLICATES = 20

#: Seeds. A seed changes which groups land in the fit half and which in the held-out half, so
#: agreement across seeds is what "reproducible" means in the matrix and disagreement is the
#: honest reason to distrust a cell.
SEEDS = (0, 1, 2)

#: How many groups the elicitation reading generates over, and how many tokens it writes. Small on
#: purpose: it asks whether the instruction does ANYTHING, which is visible in a few replies, and
#: it runs for every behaviour on every model, so the cost multiplies by the size of the matrix.
ELICIT_GROUPS, ELICIT_TOKENS = 12, 48

#: Below this held-out AUC a cell has not separated the arms. 0.6 is `Conventions.absent_auc_floor`,
#: the same floor `attribution.classify` calls `absent` beneath, so the matrix and the shipped
#: instrument agree about what "no behaviour here" means.
SEPARATION_FLOOR = 0.6

#: A separation this close to a length-only score is reported as suspect however high it is.
LENGTH_CONFOUND_MARGIN = 0.1

#: Share of above-chance separation a direction must keep across a paraphrase. 0.5 is the line the
#: 2026-10-14 sweep's own table drew between "transfers" and "collapses", and it is declared here
#: rather than chosen once the numbers are in.
TRANSFER_FLOOR = 0.5

#: Two arms whose replies overlap more than this are the same text in different clothes, which
#: means the instruction was ignored. Declared before the run.
ELICIT_OVERLAP_CEILING = 0.9

#: Above this refusal rate on either arm, the contrast is substantially a refusal contrast and a
#: direction fitted from it is partly the refusal direction. The toxicity work measured 0.70 on
#: one model and recorded the trap; this is that reading made a rule.
ELICIT_REFUSAL_CEILING = 0.10


class SweepError(RuntimeError):
    """The sweep cannot proceed, for a reason stated in the message."""


def halves(groups, seed):
    """Split group labels into a fit half and a held-out half, by seed.

    By GROUP and never by prompt: template variants of one request are near copies, so splitting
    them across halves would score a direction on prompts it has effectively already seen.
    """
    keys = sorted(set(groups))
    order = torch.randperm(len(keys), generator=torch.Generator().manual_seed(seed)).tolist()
    cut = max(1, len(keys) // 2)
    fit = {keys[i] for i in order[:cut]}
    return [i for i, g in enumerate(groups) if g in fit], \
           [i for i, g in enumerate(groups) if g not in fit]


def auc_interval(scores_a, scores_b, groups_a, groups_b, *, seed, resamples=RESAMPLES):
    """An AUC with a bootstrap interval that resamples GROUPS rather than prompts."""
    import random

    point = A.auc(scores_a, scores_b)
    idx_a, idx_b = {}, {}
    for i, g in enumerate(groups_a):
        idx_a.setdefault(g, []).append(i)
    for i, g in enumerate(groups_b):
        idx_b.setdefault(g, []).append(i)
    keys_a, keys_b = sorted(idx_a), sorted(idx_b)
    rng = random.Random(int(seed))  # noqa: S311  # resampling a statistic, not a key
    draws = []
    for _ in range(min(resamples, 500)):
        rows_a = [i for _ in keys_a for i in idx_a[keys_a[rng.randrange(len(keys_a))]]]
        rows_b = [i for _ in keys_b for i in idx_b[keys_b[rng.randrange(len(keys_b))]]]
        draws.append(A.auc([scores_a[i] for i in rows_a], [scores_b[i] for i in rows_b]))
    draws.sort()
    return {"auc": round(float(point), 4),
            "ci": [round(draws[int(0.025 * (len(draws) - 1))], 4),
                   round(draws[int(0.975 * (len(draws) - 1))], 4)]}


def fit_and_score(res_a, res_b, groups, seed, position=None):
    """Fit a mean-difference direction on half the groups and score it on the other half.

    Returns the held-out AUC with its interval, the position used and the unit direction. When
    `position` is None the position is chosen by held-out AUC, which is the engine's own rule.
    """
    fit_rows, held_rows = halves(groups, seed)
    if not fit_rows or not held_rows:
        raise SweepError("a seed split every group into one half, so nothing was held out.")
    fit_a, fit_b = res_a[:, fit_rows], res_b[:, fit_rows]
    held_a, held_b = res_a[:, held_rows], res_b[:, held_rows]
    if position is None:
        direction, position, _ = A.fit_direction(fit_a, fit_b, held_a, held_b)
    else:
        direction, _ = A.fit_direction_at(fit_a, fit_b, held_a, held_b, position)
    scores_a = (held_a[position].double() @ direction.double()).tolist()
    scores_b = (held_b[position].double() @ direction.double()).tolist()
    held_groups = [groups[i] for i in held_rows]
    out = auc_interval(scores_a, scores_b, held_groups, held_groups, seed=seed)
    out["position"] = int(position)
    return out, direction, position, (fit_rows, held_rows)


def shuffled_null(res_a, res_b, groups, seed):
    """What this pipeline scores when the labels are noise. Arithmetic only, no model calls."""
    pooled = torch.cat([res_a, res_b], 1)
    pooled_groups = [("a", g) for g in groups] + [("b", g) for g in groups]
    gen = torch.Generator().manual_seed(seed + 7919)
    scores = []
    keys = sorted(set(pooled_groups))
    for _ in range(NULL_REPLICATES):
        order = torch.randperm(len(keys), generator=gen).tolist()
        side = {keys[i]: (i in order[:len(keys) // 2]) for i in range(len(keys))}
        mask = torch.tensor([side[g] for g in pooled_groups])
        if bool(mask.all()) or not bool(mask.any()):
            continue
        fake_a, fake_b = pooled[:, mask], pooled[:, ~mask]
        fake_groups = [g for g, m in zip(pooled_groups, mask.tolist(), strict=True) if m]
        try:
            got, _, _, _ = fit_and_score(fake_a, fake_b, fake_groups, seed)
        except (SweepError, A.AttributionError):
            continue
        scores.append(got["auc"])
    if not scores:
        return None
    scores.sort()
    return {"replicates": len(scores), "p95": round(scores[int(0.95 * (len(scores) - 1))], 4),
            "max": round(scores[-1], 4), "median": round(scores[len(scores) // 2], 4)}


def length_auc(prompts_a, prompts_b, tok):
    """Can prompt length alone tell the arms apart? The control that caught six candidates."""
    len_a = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts_a]
    len_b = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts_b]
    return round(float(A.auc(len_a, len_b)), 4)


def overlap(reply_a, reply_b):
    """Jaccard overlap of the word sets of two replies: 1.0 means the instruction changed nothing."""
    wa, wb = set(reply_a.lower().split()), set(reply_b.lower().split())
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / max(len(wa | wb), 1)


def elicitation(subject, pairs, *, log):
    """Does the instruction change what the model writes? Generation, and no judge anywhere.

    Three readings, none of which needs to know what the behaviour IS:
      - the refusal rate on each arm, from the project's own ruler, which catches a contrast that
        is really a refusal contrast;
      - the word overlap between the two arms' replies to the SAME request, which catches an
        instruction the model ignored;
      - the reply length on each arm, which is the one behaviour (verbosity) whose elicitation is
        visible without reading anything.
    """
    chosen, seen = [], set()
    for pair in pairs:
        if pair.group in seen:
            continue
        seen.add(pair.group)
        chosen.append(pair)
        if len(chosen) >= ELICIT_GROUPS:
            break
    a_prompts = [p.a for p in chosen]
    b_prompts = [p.b for p in chosen]
    began = time.monotonic()
    a_replies = subject.replies(a_prompts)
    b_replies = subject.replies(b_prompts)
    overlaps = [overlap(x, y) for x, y in zip(a_replies, b_replies, strict=True)]
    out = {
        "n_groups": len(chosen), "seconds": round(time.monotonic() - began, 1),
        "refusal_rate_a": round(metrics.refusal_rate(a_replies), 4),
        "refusal_rate_b": round(metrics.refusal_rate(b_replies), 4),
        "broken_rate_a": round(metrics.broken_rate(a_replies), 4),
        "broken_rate_b": round(metrics.broken_rate(b_replies), 4),
        "mean_overlap": round(sum(overlaps) / len(overlaps), 4),
        "max_overlap": round(max(overlaps), 4),
        "mean_chars_a": round(sum(len(r) for r in a_replies) / len(a_replies), 1),
        "mean_chars_b": round(sum(len(r) for r in b_replies) / len(b_replies), 1),
    }
    out["length_ratio"] = round(out["mean_chars_a"] / max(out["mean_chars_b"], 1.0), 3)
    refused = max(out["refusal_rate_a"], out["refusal_rate_b"]) > ELICIT_REFUSAL_CEILING
    ignored = out["mean_overlap"] > ELICIT_OVERLAP_CEILING
    out["elicited"] = bool(not refused and not ignored)
    out["why_not"] = ("the model refuses one arm, so the contrast is substantially a refusal "
                      "contrast" if refused else
                      "the two arms produce near identical replies, so the instruction did "
                      "nothing" if ignored else None)
    log(f"      elicited={out['elicited']} overlap={out['mean_overlap']} "
        f"refusal={out['refusal_rate_a']}/{out['refusal_rate_b']} "
        f"len={out['length_ratio']} ({out['seconds']}s)")
    return out


def run_cell(subject, tok, contrast, *, elicit, log):
    """Every measurement for one behaviour on one model."""
    try:
        pairs = contrast.build()
    except Exception as e:  # a contrast that cannot build is a cell, not a crash
        return {"ran": False, "why": f"the contrast could not be built: {e}"}
    groups = [p.group for p in pairs]
    if len({*groups}) < 8:
        return {"ran": False,
                "why": f"only {len({*groups})} groups, too few to split into halves and bootstrap"}

    began = time.monotonic()
    res_a = subject.residuals([p.a for p in pairs])
    res_b = subject.residuals([p.b for p in pairs])
    seconds = round(time.monotonic() - began, 1)

    per_seed, directions = {}, {}
    for seed in SEEDS:
        got, direction, position, _ = fit_and_score(res_a, res_b, groups, seed)
        per_seed[str(seed)] = got
        directions[seed] = (direction, position)
    aucs = [per_seed[str(s)]["auc"] for s in SEEDS]
    null = shuffled_null(res_a, res_b, groups, SEEDS[0])

    cell = {
        "ran": True,
        "instructed": contrast.instructed,
        "tier": contrast.tier,
        "pairs": len(pairs), "groups": len({*groups}),
        "residual_seconds": seconds,
        "separation_by_seed": per_seed,
        "separation_median": round(sorted(aucs)[len(aucs) // 2], 4),
        "separation_spread": round(max(aucs) - min(aucs), 4),
        "shuffled_null": null,
        "length_only_auc": length_auc([p.a for p in pairs], [p.b for p in pairs], tok),
        "sweep_verdict_2026_10_14": contrast.sweep_verdict,
    }

    # THE TRANSFER TEST, over EVERY ORDERED PAIR OF PHRASINGS rather than one of them.
    #
    # TWO CORRECTIONS, both made after model one's numbers disagreed with the 2026-10-14 sweep on
    # its two headline cases, and both recorded in the plan because a rule changed mid-flight has
    # to be. Neither can manufacture a pass: the first makes the comparison harder to flatter and
    # the second takes the WORST pair rather than the best.
    #
    #   1. `across` is scored on the HELD-OUT groups only, exactly as `within` is. It used to be
    #      scored on every group, so the two numbers being divided were taken over different
    #      prompt sets and their ratio was not a like-for-like share.
    #   2. Every ordered pair of phrasings is measured, not just 0 to 1. If a behaviour's verdict
    #      depends on which paraphrase was written for it, then the verdict is a property of our
    #      wording rather than of the behaviour, and the only way to see that is to try them all.
    #      `transfers` is true only when the WEAKEST pair clears the floor.
    if contrast.instructed and contrast.phrasings >= 2:
        fit_rows, held_rows = halves(groups, SEEDS[0])
        res_by_phrasing = {0: (res_a, res_b)}
        for q in range(1, contrast.phrasings):
            other = contrast.build(phrasing=q)
            res_by_phrasing[q] = (subject.residuals([p.a for p in other]),
                                  subject.residuals([p.b for p in other]))
        pairs_out, retained_values = {}, []
        for p in range(contrast.phrasings):
            fit_a, fit_b = res_by_phrasing[p]
            direction, position, _ = A.fit_direction(
                fit_a[:, fit_rows], fit_b[:, fit_rows], fit_a[:, held_rows], fit_b[:, held_rows])
            within = A.auc((fit_a[position][held_rows].double() @ direction.double()).tolist(),
                           (fit_b[position][held_rows].double() @ direction.double()).tolist())
            for q in range(contrast.phrasings):
                if p == q:
                    continue
                score_a, score_b = res_by_phrasing[q]
                across = A.auc(
                    (score_a[position][held_rows].double() @ direction.double()).tolist(),
                    (score_b[position][held_rows].double() @ direction.double()).tolist())
                keep = ((across - 0.5) / (within - 0.5)) if within > 0.5 else None
                pairs_out[f"{p}->{q}"] = {
                    "within": round(float(within), 4), "across": round(float(across), 4),
                    "retained": None if keep is None else round(float(keep), 4)}
                if keep is not None:
                    retained_values.append(float(keep))
        worst = min(retained_values) if retained_values else None
        best = max(retained_values) if retained_values else None
        cell["transfer"] = {
            "phrasings": contrast.phrasings, "pairs": pairs_out,
            "retained": None if worst is None else round(worst, 4),
            "retained_best_pair": None if best is None else round(best, 4),
            "spread_across_pairs": None if worst is None else round(best - worst, 4),
            "floor": TRANSFER_FLOOR, "rule": "the weakest ordered pair must clear the floor",
            "transfers": bool(worst is not None and worst >= TRANSFER_FLOOR),
        }
    else:
        cell["transfer"] = {"ran": False,
                            "why": "uninstructed: there is no instruction to rephrase, so a "
                                   "separation score means what the label says without one"}

    cell["elicitation"] = elicitation(subject, pairs, log=log) if elicit else {
        "ran": False, "why": "not requested on this pass"}
    cell["verdict"] = verdict_for(cell)
    return cell


def verdict_for(cell):
    """The cell's one-line answer, from its own numbers and the floors declared above."""
    if not cell.get("ran"):
        return "could not run"
    sep = cell["separation_median"]
    null = (cell.get("shuffled_null") or {}).get("p95")
    reasons = []
    if sep < SEPARATION_FLOOR:
        reasons.append(f"separation {sep:.3f} below the {SEPARATION_FLOOR} floor")
    if null is not None and sep <= null:
        reasons.append(f"separation {sep:.3f} does not clear the shuffled null {null:.3f}")
    if cell["length_only_auc"] >= sep - LENGTH_CONFOUND_MARGIN and sep >= SEPARATION_FLOOR:
        reasons.append(f"prompt length alone scores {cell['length_only_auc']:.3f} against "
                       f"{sep:.3f}, so the direction may be reading length")
    if cell["separation_spread"] > 0.15:
        reasons.append(f"seed spread {cell['separation_spread']:.3f} is wide")
    transfer = cell.get("transfer") or {}
    if transfer.get("ran") is not False and transfer.get("transfers") is False:
        reasons.append(f"collapses under paraphrase, weakest pair retaining "
                       f"{transfer.get('retained')}")
    if (transfer.get("spread_across_pairs") or 0) > 0.4:
        reasons.append(f"the transfer verdict depends on which paraphrase is used (spread "
                       f"{transfer.get('spread_across_pairs')}), so it is a fact about our "
                       f"wording rather than about the behaviour")
    elicit = cell.get("elicitation") or {}
    if elicit.get("ran") is not False and elicit.get("elicited") is False:
        reasons.append(f"not elicited: {elicit.get('why_not')}")
    return "USABLE" if not reasons else "; ".join(reasons)


def build_parser():
    ap = argparse.ArgumentParser(
        allow_abbrev=False, prog="behaviour_architecture_sweep",
        description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="how this model appears in the matrix")
    ap.add_argument("--architecture", required=True, choices=("hybrid", "dense", "moe"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--behaviours", default="", help="comma separated; default is all of them")
    ap.add_argument("--no-elicit", action="store_true", help="skip the generation reading")
    ap.add_argument("--force", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device asks for cuda and torch reports no cuda device.")
    keys = ([k.strip() for k in args.behaviours.split(",") if k.strip()]
            or sorted(contrasts.REGISTRY))
    unknown = [k for k in keys if k not in contrasts.REGISTRY]
    if unknown:
        raise SystemExit(f"unknown behaviour(s) {unknown}; known: {sorted(contrasts.REGISTRY)}")

    log = print
    log(f"loading {args.model} on {args.device} at {args.dtype}")
    model, tok, identity = H.load(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    subject = H.HFSubject(model, tok, batch=args.batch, log=lambda *_: None)
    log(f"  {args.label}: {args.architecture}, {subject.n_blocks} blocks")

    result = {
        "model": identity, "label": args.label, "architecture": args.architecture,
        "blocks": subject.n_blocks,
        "machine": {"node": platform.node(), "torch": torch.__version__},
        "seeds": list(SEEDS),
        "floors": {"separation": SEPARATION_FLOOR, "transfer": TRANSFER_FLOOR,
                   "length_confound_margin": LENGTH_CONFOUND_MARGIN,
                   "elicit_overlap_ceiling": ELICIT_OVERLAP_CEILING,
                   "elicit_refusal_ceiling": ELICIT_REFUSAL_CEILING},
        "not_built": contrasts.NOT_BUILT,
        "cells": {},
    }
    path = out / f"{args.label}.json"
    if path.is_file() and not args.force:
        result["cells"] = json.loads(path.read_text(encoding="utf-8")).get("cells", {})
        log(f"  resuming: {len(result['cells'])} cell(s) already on disk")

    for key in keys:
        if key in result["cells"]:
            log(f"    {key:22s} already done")
            continue
        log(f"    {key:22s} ...")
        began = time.monotonic()
        try:
            cell = run_cell(subject, tok, contrasts.get(key), elicit=not args.no_elicit, log=log)
        except Exception as e:  # one behaviour failing must not lose the others
            cell = {"ran": False, "why": f"{type(e).__name__}: {e}",
                    "verdict": "could not run"}
        cell["seconds"] = round(time.monotonic() - began, 1)
        result["cells"][key] = cell
        # Written after every cell, so an interrupted night keeps everything measured so far.
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"      {cell.get('verdict')}")
    log(f"artefact: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
