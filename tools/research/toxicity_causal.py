#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The test the toxicity channel owes: does removing the direction change the behaviour?

WHY THIS FILE EXISTS

The behaviour sweep of 2026-10-14 showed that a rude-register direction EXISTS (AUC 0.990 on two
models) and that it is not an artefact of our own wording (it retained 0.90 and 0.79 of its
above-chance separation when extracted from one phrasing and scored on another). Neither of those
is a causal claim. A direction that merely correlates with a behaviour separates the arms perfectly
and moves nothing when it is removed, and this project has withdrawn a published result for a
smaller mistake than that.

So: remove the direction from every block's write while the model GENERATES, judge the replies, and
compare the fall in rude-register output against controls. That is the only evidence that the
channel measures something the model does rather than something our prompts say.

THE THREE STAGES, AND WHY THEY RUN IN THIS ORDER

    judge    Validate the judge before anything is graded with it. A rate produced by an
             unvalidated judge is unfalsifiable, and every number in the later stages is a rate
             this judge produced. Agreement is measured against the INSTRUCTION the model was
             given, which is a label this project did not author per item, and optionally against
             a hand-labelled sample.
    causal   The owed test. Rude-register rate on the measure partition with nothing ablated,
             with the register direction ablated from every block, with random directions drawn
             from where the blocks write, and with a topic direction fitted from a contrast that
             is real and has nothing to do with register. Every rate carries a bootstrap interval
             over SUBJECTS rather than prompts, because four templates over one subject are
             variants of one request.
    nulls    The project's own instrument, `attribution.attribute`, with this channel's judge in
             place of the refusal ruler. It brings the shuffled-label null, the random-direction
             null, the length-direction control and the seven-way status with it, so the channel
             clears the bar refusal cleared rather than a lower one.

WHAT IS MATCHED BY CONSTRUCTION RATHER THAN BY CARE

"A random direction of the same norm" needs no norm matching here: `attribution.ablating`
normalises whatever direction it is handed before projecting it out, so every arm of this
experiment removes a unit direction and the arms differ only in which one.

WHAT NEVER LEAVES THE MACHINE

The artefact carries counts, rates, intervals and digests. It carries no prompt and no generation.
`--replies-out` writes replies for hand labelling and is the one path that holds model output; it
belongs outside the repository, and the artefact records only the digest of what was written.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from senbonzakura import attribution as A
from senbonzakura import attribution_hf as H
from senbonzakura import judge, metrics, toxicity

#: How many random control directions to draw. Five, which is what `attribution_hf`'s own
#: predictive check uses for the same job: enough to average away one unlucky draw, few enough
#: that the generation budget stays inside an hour on a processor.
RANDOM_DIRECTIONS = 5

#: Principal directions of the block writes that a random control is drawn from. The same number
#: `attribution.Conventions.random_direction_pcs` uses, and for the reason written there: a random
#: direction drawn from the whole space is nearly orthogonal to everything the model does, so
#: removing it is a weaker intervention than removing the real direction and the control flatters
#: the result.
RANDOM_DIRECTION_PCS = 8

#: Resamples for every interval here. `metrics.DEFAULT_RESAMPLES`, so a reader meeting an interval
#: in this artefact meets the same instrument as everywhere else in the project.
RESAMPLES = metrics.DEFAULT_RESAMPLES

#: Subjects in each partition. Fit and search are what the direction is extracted and positioned
#: with; measure is where every published rate is taken. Measure is the largest because it is the
#: only one a figure is quoted from, and 30 subjects is 120 prompts, four times
#: `metrics.MIN_REPORTABLE_N`.
FIT_SUBJECTS, SEARCH_SUBJECTS = 15, 15

#: The decision rule, declared here before the run. The causal claim is supported when the lower
#: end of the interval on (fall with the register direction minus fall with the control) is above
#: zero, for BOTH controls. One control clearing it and the other not is a result that is reported
#: as such and is not a pass.
DECISION_RULE = (
    "supported when the lower end of the 95% interval on (the fall in rude-register rate with the "
    "register direction removed, minus the fall with the control direction removed) is above zero, "
    "for the random control and for the topic control alike")


class RunError(RuntimeError):
    """The run cannot proceed, for a reason stated in the message."""


# ── partitions ────────────────────────────────────────────────────────────────────────────────────

def partition_subjects(seed):
    """Fit, search and measure subject lists, drawn once by a seeded permutation.

    By SUBJECT and never by prompt: the four templates over one subject are variants of one
    request, so splitting them across partitions would put a near-copy of a fit prompt into the
    measure arm and the held-out numbers would be held out of nothing.
    """
    subjects = list(toxicity.SUBJECTS)
    order = torch.randperm(len(subjects), generator=torch.Generator().manual_seed(seed)).tolist()
    picked = [subjects[i] for i in order]
    fit = picked[:FIT_SUBJECTS]
    search = picked[FIT_SUBJECTS:FIT_SUBJECTS + SEARCH_SUBJECTS]
    measure = picked[FIT_SUBJECTS + SEARCH_SUBJECTS:]
    if len(measure) * len(toxicity.TEMPLATES) < metrics.MIN_REPORTABLE_N:
        raise RunError(
            f"the measure partition would hold {len(measure)} subjects, which is "
            f"{len(measure) * len(toxicity.TEMPLATES)} prompts, below the reportable floor of "
            f"{metrics.MIN_REPORTABLE_N}. Either widen `toxicity.SUBJECTS` or lower FIT_SUBJECTS "
            f"and SEARCH_SUBJECTS; do not publish a rate from fewer.")
    return {"fit": fit, "search": search, "measure": measure}


def arms(subjects, *, phrasing=0):
    """The rude and polite arms over one subject list, grouped by subject."""
    rows = toxicity.pairs(phrasing=phrasing, subjects=subjects)
    return {"harmful": A.arm([p.rude for p in rows], [p.group for p in rows]),
            "harmless": A.arm([p.polite for p in rows], [p.group for p in rows])}


def topic_arms(subjects):
    """The topic control's arms, restricted to whichever declared pairs touch these subjects."""
    rows = [p for p in toxicity.topic_pairs()
            if any(s in p.group.split(" | ") for s in subjects)]
    if not rows:
        raise RunError("no topic control pairs cover the fit subjects, so no topic direction can "
                       "be fitted. This means the subject lists have drifted out of step.")
    return {"harmful": A.arm([p.rude for p in rows], [p.group for p in rows]),
            "harmless": A.arm([p.polite for p in rows], [p.group for p in rows])}


# ── intervals over subjects ───────────────────────────────────────────────────────────────────────

def group_bootstrap(groups, per_item, statistic, *, seed, resamples=RESAMPLES, alpha=0.05):
    """An interval on a statistic of several aligned per-item lists, resampling SUBJECTS whole.

    `metrics.paired_rate_bootstrap` resamples items, which is right when items are independent and
    wrong here: four templates over one subject move together, and resampling prompts would treat
    them as four observations and give an interval about twice too narrow. So the draw is over
    subjects, with every prompt of a drawn subject coming along.

    `per_item` maps a name to a list of booleans aligned with `groups`. `statistic` receives
    `{name: rate}` and returns a float or None.
    """
    names = sorted(per_item)
    lengths = {len(per_item[k]) for k in names} | {len(groups)}
    if len(lengths) != 1 or not names:
        raise RunError(f"the arms of this interval have lengths {sorted(lengths)}, so they are "
                       f"not aligned item for item and no paired interval can be taken.")
    index = {}
    for i, g in enumerate(groups):
        index.setdefault(g, []).append(i)
    keys = sorted(index)

    def rates(rows):
        return {k: (sum(1 for i in rows if per_item[k][i]) / len(rows)) if rows else None
                for k in names}

    point = statistic(rates(list(range(len(groups)))))
    rng = random.Random(int(seed))  # noqa: S311  # resampling a statistic, not a key
    draws, undefined = [], 0
    for _ in range(resamples):
        rows = []
        for _ in keys:
            rows.extend(index[keys[rng.randrange(len(keys))]])
        value = statistic(rates(rows))
        if value is None:
            undefined += 1
        else:
            draws.append(value)
    if not draws:
        return {"point": point, "ci": None, "crosses_zero": None, "groups": len(keys),
                "resamples": resamples, "undefined_resamples": undefined, "seed": int(seed)}
    draws.sort()
    lo = draws[int((alpha / 2) * (len(draws) - 1))]
    hi = draws[int((1 - alpha / 2) * (len(draws) - 1))]
    return {"point": None if point is None else round(point, 4),
            "ci": (round(lo, 4), round(hi, 4)), "crosses_zero": bool(lo <= 0.0 <= hi),
            "groups": len(keys), "resamples": resamples, "undefined_resamples": undefined,
            "seed": int(seed)}


# ── the directions ────────────────────────────────────────────────────────────────────────────────

def random_pc_directions(fit_residuals, *, replicates, seed):
    """Unit directions drawn at random from the top principal directions of the block writes.

    Label free, and built exactly as `attribution._null_replicates` builds its own random-direction
    null, including the reason it is drawn from the top principal components rather than from the
    whole space: a uniformly random direction in a 576-dimensional space is nearly orthogonal to
    anything the blocks actually write, so removing it does nothing and the control is no control
    at all.
    """
    pooled = torch.cat([fit_residuals["harmful"], fit_residuals["harmless"]], 1)
    writes = (pooled[1:] - pooled[:-1]).double()
    flat = writes.reshape(-1, writes.shape[-1])
    flat = flat - flat.mean(0)
    _, _, vt = torch.linalg.svd(flat, full_matrices=False)
    basis = vt[:RANDOM_DIRECTION_PCS].float()
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(replicates):
        coef = torch.randn(basis.shape[0], generator=gen)
        pick = (coef / coef.norm()) @ basis
        out.append(pick / pick.norm())
    return out, writes


def alignment(a, b):
    """Cosine between two directions. A control aligned with the real direction is not a control,
    so the number belongs in the artefact whether it is flattering or not.
    """
    return float(torch.dot(a.double().flatten() / a.double().norm(),
                           b.double().flatten() / b.double().norm()))


# ── stages ────────────────────────────────────────────────────────────────────────────────────────

def read_replies(path, *, expected):
    """Replies from an earlier `--replies-out` file, back in the order the stage generated them.

    THE RE-ANALYSIS PATH, and it exists because hand labelling is a second pass over the SAME
    replies. Without it, adding labels costs another full generation run, and a run that has to be
    repeated to be analysed is one somebody analyses once and then stops. Greedy decoding is
    deterministic, so a regenerated set would be identical, which is exactly why paying for it
    twice is waste rather than rigour.
    """
    rows = [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_item = {int(r["item"]): r["reply"] for r in rows if "item" in r and "reply" in r}
    missing = [i for i in range(expected) if i not in by_item]
    if missing or len(by_item) != expected:
        raise RunError(
            f"{path} holds {len(by_item)} replies for the {expected} prompts this partition has, "
            f"and item(s) {missing[:5]} are absent. A reply file from a different model, seed or "
            f"corpus cannot be re-analysed against this one: the item numbers would line up and "
            f"mean different prompts.")
    return [by_item[i] for i in range(expected)]


def stage_judge(subject, parts, *, arm_names, replies_out, replies_in, hand_labels, seed, log):
    """Validate the judge before it grades anything, against the instruction and against hands.

    The instruction label is the primary validation and it is the one worth trusting: the model was
    told to be rude or told to be polite, and whether the judge recovers that is a question this
    project did not answer per item. The hand labels are the secondary one, and they are weaker for
    a reason recorded in `toxicity.judge_record`: the same author wrote the markers and the labels.
    """
    out = {"rule": "the reference label is the register the prompt instructed"}
    rude_prompts = list(parts["search"]["harmful"].prompts)
    polite_prompts = list(parts["search"]["harmless"].prompts)
    began = time.monotonic()
    if replies_in:
        log(f"  judge: re-reading {len(rude_prompts) + len(polite_prompts)} replies from "
            f"{replies_in}")
        every = read_replies(replies_in, expected=len(rude_prompts) + len(polite_prompts))
        rude_replies, polite_replies = every[:len(rude_prompts)], every[len(rude_prompts):]
        out["replies_read_from"] = str(replies_in)
    else:
        log(f"  judge: generating {len(rude_prompts) + len(polite_prompts)} replies")
        rude_replies = subject.replies(rude_prompts)
        polite_replies = subject.replies(polite_prompts)
    out["generation_seconds"] = round(time.monotonic() - began, 1)

    texts = rude_replies + polite_replies
    reference = ([toxicity.RUDE] * len(rude_replies)) + ([toxicity.POLITE] * len(polite_replies))
    for arm in arm_names:
        verdicts = [toxicity.verdict(t, arm=arm) for t in texts]
        out[arm] = {
            "against_the_instruction": judge.validate(verdicts, reference),
            "declaration": toxicity.ARM_DECLARATIONS[arm],
        }
    scores = [toxicity.score(t) for t in texts]
    out["score_auc_rude_over_polite"] = A.auc(scores[:len(rude_replies)], scores[len(rude_replies):])
    out["rates"] = {
        "rude_arm": toxicity.rate(rude_replies),
        "polite_arm": toxicity.rate(polite_replies),
    }
    out["broken_rate"] = {
        "rude_arm": round(metrics.broken_rate(rude_replies), 4),
        "polite_arm": round(metrics.broken_rate(polite_replies), 4),
    }
    # THE CONFOUND THIS MEASUREMENT EXISTS BECAUSE OF, found on 2026-10-15. Qwen2.5-0.5B-Instruct
    # answers a large share of the rude-instructed prompts with "I'm sorry, but I can't assist with
    # that", so the two arms of this contrast differ by whether the model REFUSED as much as by
    # register, and a direction fitted from them is partly the refusal direction the first channel
    # already measures. A second channel that is a re-reading of the first is worse than no second
    # channel, so the refusal rate per arm travels with every figure taken on this corpus.
    out["refusal_rate"] = {
        "rude_arm": round(metrics.refusal_rate(rude_replies), 4),
        "polite_arm": round(metrics.refusal_rate(polite_replies), 4),
        "ruler": "metrics.is_refusal, the project's own refusal ruler",
        "why_it_is_here": ("a rude-register instruction is a thing a safety-trained model can "
                           "refuse. Where it does, this contrast is a refusal contrast wearing "
                           "another name, and the toxicity channel would be re-measuring the "
                           "refusal channel."),
    }
    out["mean_reply_chars"] = {
        "rude_arm": round(sum(len(t) for t in rude_replies) / max(len(rude_replies), 1), 1),
        "polite_arm": round(sum(len(t) for t in polite_replies) / max(len(polite_replies), 1), 1),
    }

    if replies_out:
        rng = random.Random(seed)  # noqa: S311  # shuffling a labelling sheet, not a key
        order = list(range(len(texts)))
        rng.shuffle(order)
        path = Path(replies_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for i in order:
                f.write(json.dumps({"item": i, "reply": texts[i]}, ensure_ascii=False) + "\n")
        out["replies_written"] = {
            "path": str(path), "rows": len(texts),
            "shuffled_with_seed": seed,
            "note": ("the reference register is NOT in this file, so a labeller cannot read the "
                     "answer off it. It holds model output and must stay outside the repository."),
            "digest": toxicity.digest(texts)}
        log(f"  judge: {len(texts)} replies written for hand labelling -> {path}")

    if hand_labels:
        rows = [json.loads(line) for line in
                Path(hand_labels).read_text(encoding="utf-8").splitlines() if line.strip()]
        by_item = {int(r["item"]): r.get("label") for r in rows if "item" in r}
        unknown = sorted(set(by_item.values()) - {toxicity.RUDE, toxicity.POLITE, None})
        if unknown:
            raise RunError(
                f"the hand labels contain {unknown}, and the only labels this comparison "
                f"understands are {toxicity.RUDE!r}, {toxicity.POLITE!r} and null for undecided.")
        items = sorted(by_item)
        out["against_hand_labels"] = {
            "n_labelled": len(items),
            "sampled_from": len(texts),
            "weaker_because": ("the author of the marker lists is the author of these labels, so "
                               "this agreement is an upper bound on how the judge would fare "
                               "against an independent labeller"),
            **{arm: judge.validate([toxicity.verdict(texts[i], arm=arm) for i in items],
                                   [by_item[i] for i in items]) for arm in arm_names},
            "instruction_agreement_on_the_same_items": judge.validate(
                [by_item[i] for i in items], [reference[i] for i in items]),
        }
    return out


def stage_causal(subject, parts, direction, controls, *, judge_arm, seed, log):
    """The owed test: does removing the direction move the generated behaviour?

    Every arm generates on the SAME measure prompts, so the comparison is paired and the interval
    is taken over subjects. The polite arm runs under the same conditions, because a direction that
    flattens every reply has not removed rudeness, it has broken the model, and the only way to see
    the difference is to watch what happens where there was no rudeness to remove.
    """
    nb = subject.n_blocks
    every = tuple(range(nb))
    rude = list(parts["measure"]["harmful"].prompts)
    polite = list(parts["measure"]["harmless"].prompts)
    groups = list(parts["measure"]["harmful"].groups)

    conditions = {"none": None, "register": direction}
    conditions["topic"] = controls["topic"]
    for i, d in enumerate(controls["random"]):
        conditions[f"random:{i}"] = d

    # EVERY JUDGE ARM IS SCORED ON EVERY CONDITION, from the same generations, and that removes a
    # choice that would otherwise be made after the fact. Generation is the whole cost here;
    # judging is substring matching over text already in memory, so the arms are free and leaving
    # one out would only mean choosing between them once the numbers were visible.
    arms_ = sorted(toxicity.ARM_DECLARATIONS)
    flags = {arm: {} for arm in arms_}
    record = {}
    for name, d in conditions.items():
        began = time.monotonic()
        blocks = () if d is None else every
        rude_replies = subject.replies(rude, d, blocks)
        polite_replies = subject.replies(polite, d, blocks)
        for arm in arms_:
            flags[arm][name] = [toxicity.is_rude(t, arm=arm) for t in rude_replies]
            flags[arm][f"polite_arm:{name}"] = [toxicity.is_rude(t, arm=arm)
                                                for t in polite_replies]
        record[name] = {
            "blocks_ablated": len(blocks),
            "by_arm": {arm: {"rude_arm": toxicity.rate(rude_replies, arm=arm),
                             "polite_arm": toxicity.rate(polite_replies, arm=arm)}
                       for arm in arms_},
            "broken_rate": {"rude_arm": round(metrics.broken_rate(rude_replies), 4),
                            "polite_arm": round(metrics.broken_rate(polite_replies), 4)},
            "refusal_rate": {"rude_arm": round(metrics.refusal_rate(rude_replies), 4),
                             "polite_arm": round(metrics.refusal_rate(polite_replies), 4)},
            "mean_reply_chars": round(sum(len(t) for t in rude_replies)
                                      / max(len(rude_replies), 1), 1),
            "mean_score": round(sum(toxicity.score(t) for t in rude_replies)
                                / max(len(rude_replies), 1), 3),
            "seconds": round(time.monotonic() - began, 1),
        }
        log(f"  causal: {name:12s} rude rate "
            f"{record[name]['by_arm'][judge_arm]['rude_arm']['rate']} "
            f"({record[name]['seconds']}s)")

    random_names = [n for n in conditions if n.startswith("random:")]

    def mean_random(r):
        return sum(r["none"] - r[n] for n in random_names) / len(random_names)

    def analysis(arm):
        f = flags[arm]
        falls = {name: group_bootstrap(groups, f, lambda r, n=name: r["none"] - r[n], seed=seed)
                 for name in conditions if name != "none"}
        contrasts = {
            "register_minus_random_mean": group_bootstrap(
                groups, f, lambda r: (r["none"] - r["register"]) - mean_random(r), seed=seed + 1),
            "register_minus_topic": group_bootstrap(
                groups, f, lambda r: (r["none"] - r["register"]) - (r["none"] - r["topic"]),
                seed=seed + 2),
        }
        side_effect = {
            name: group_bootstrap(groups, f,
                                  lambda r, n=name: r["polite_arm:none"] - r[f"polite_arm:{n}"],
                                  seed=seed + 3)
            for name in conditions if name != "none"
        }
        return {
            "declaration": toxicity.ARM_DECLARATIONS[arm],
            "supported": bool(all(c["ci"] is not None and c["ci"][0] > 0.0
                                  for c in contrasts.values())),
            "falls": falls, "contrasts": contrasts, "polite_arm_falls": side_effect,
        }

    by_arm = {arm: analysis(arm) for arm in arms_}
    return {
        "decision_rule": DECISION_RULE,
        "primary_judge_arm": judge_arm,
        "supported": by_arm[judge_arm]["supported"],
        "supported_by_arm": {arm: by_arm[arm]["supported"] for arm in arms_},
        "n_prompts_per_arm": len(rude),
        "n_subjects": len({*groups}),
        "conditions": record,
        "analysis_by_judge_arm": by_arm,
        "alignments": {
            "register_with_topic": alignment(direction, controls["topic"]),
            "register_with_random": [alignment(direction, d) for d in controls["random"]],
        },
    }


def stage_nulls(subject, parts, *, seed, log):
    """The project's own nulls, through the shipped instrument with this channel's judge."""
    log("  nulls: running attribution.attribute")
    began = time.monotonic()
    res = A.attribute(subject, parts, seed=seed, log=log,
                      predictive=H.predictive_check if "confirm" in parts else None)
    res["seconds"] = round(time.monotonic() - began, 1)
    res["selftest"] = A.selftest()
    return res


# ── the run ───────────────────────────────────────────────────────────────────────────────────────

def preflight(args):
    """Refuse before the first forward pass rather than partway through it."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probe = out / ".writable"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        raise RunError(f"the output directory {out} cannot be written to: {e}. Every stage writes "
                       f"its result there as it finishes, so a run with nowhere to write would "
                       f"lose everything it measured.") from e
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RunError("--device asks for cuda and torch reports no cuda device. Pass "
                       "--device cpu, or run this where the card is.")
    if args.replies_in and not Path(args.replies_in).is_file():
        raise RunError(f"there is no file at {args.replies_in}. --replies-in wants the jsonl an "
                       f"earlier run wrote with --replies-out.")
    if args.hand_labels and not Path(args.hand_labels).is_file():
        raise RunError(f"there is no file at {args.hand_labels}. --hand-labels wants the jsonl "
                       f"written by an earlier --replies-out run with a 'label' field added.")
    bad = [r for r in toxicity.instruction_lengths() if not r["words_match"]]
    if bad:
        raise RunError(
            f"the register instruction pairs {[r['phrasing'] for r in bad]} do not match in word "
            f"count, so prompt length alone would partly separate the arms and the direction could "
            f"be a direction for length. Fix the wording in `toxicity.py`.")


def build_parser():
    ap = argparse.ArgumentParser(
        allow_abbrev=False, prog="toxicity_causal",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a model id or a path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="directory for the stage artefacts")
    ap.add_argument("--stages", default="judge,causal,nulls",
                    help="comma separated: judge, causal, nulls")
    ap.add_argument("--judge-arm", default=toxicity.PRIMARY_ARM,
                    choices=sorted(toxicity.ARM_DECLARATIONS))
    ap.add_argument("--replies-out", default="",
                    help="write the judge stage's replies here for hand labelling. Holds model "
                         "output: keep it outside the repository")
    ap.add_argument("--replies-in", default="",
                    help="re-read the judge stage's replies from an earlier --replies-out file "
                         "instead of generating them again")
    ap.add_argument("--hand-labels", default="",
                    help="a jsonl of {'item': n, 'label': 'rude'|'polite'|null} to compare against")
    ap.add_argument("--force", action="store_true",
                    help="re-run a stage whose artefact is already on disk")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ("judge", "causal", "nulls")]
    if unknown:
        raise SystemExit(f"--stages names {unknown}, and the stages are judge, causal and nulls.")
    preflight(args)
    out = Path(args.out)
    log = print

    log(f"loading {args.model} on {args.device} at {args.dtype}")
    model, tok, identity = H.load(args.model, device=args.device,
                                  dtype=getattr(torch, args.dtype))
    subject = H.HFSubject(model, tok, batch=args.batch, log=log,
                          label=lambda t: toxicity.is_rude(t, arm=args.judge_arm))

    split = partition_subjects(args.seed)
    parts = {name: arms(split[name]) for name in ("fit", "search", "measure")}
    header = {
        "experiment": "toxicity-as-rude-register, causal test",
        "model": identity,
        "machine": {"node": platform.node(), "python": platform.python_version(),
                    "torch": torch.__version__},
        "seed": args.seed,
        "judge": toxicity.judge_record(),
        "corpus": toxicity.provenance(),
        "partitions": {name: {"subjects": len(split[name]),
                              "prompts_per_arm": len(parts[name]["harmful"].prompts),
                              "rude_digest": toxicity.digest(parts[name]["harmful"].prompts),
                              "polite_digest": toxicity.digest(parts[name]["harmless"].prompts)}
                       for name in parts},
        "decision_rule": DECISION_RULE,
    }
    (out / "header.json").write_text(json.dumps(header, indent=2), encoding="utf-8")

    def done(name):
        return (out / f"{name}.json").is_file() and not args.force

    if "judge" in stages:
        if done("judge"):
            log("judge: already on disk, skipping (pass --force to re-run)")
        else:
            log("stage: judge")
            res = stage_judge(subject, parts, arm_names=sorted(toxicity.ARM_DECLARATIONS),
                              replies_out=args.replies_out, replies_in=args.replies_in,
                              hand_labels=args.hand_labels, seed=args.seed, log=log)
            (out / "judge.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    direction = controls = None
    if "causal" in stages and not done("causal"):
        log("fitting the register direction")
        res_fit = {s: subject.residuals(list(parts["fit"][s].prompts))
                   for s in ("harmful", "harmless")}
        res_search = {s: subject.residuals(list(parts["search"][s].prompts))
                      for s in ("harmful", "harmless")}
        direction, position, position_aucs = A.fit_direction(
            res_fit["harmful"], res_fit["harmless"], res_search["harmful"], res_search["harmless"])
        topic = topic_arms(split["fit"])
        res_topic = {s: subject.residuals(list(topic[s].prompts)) for s in ("harmful", "harmless")}
        topic_direction, topic_auc = A.fit_direction_at(
            res_topic["harmful"], res_topic["harmless"],
            res_topic["harmful"], res_topic["harmless"], position)
        random_dirs, _ = random_pc_directions(res_fit, replicates=RANDOM_DIRECTIONS,
                                              seed=args.seed + 7919)
        controls = {"topic": topic_direction, "random": random_dirs}
        fitted = {
            "position": int(position), "n_positions": len(position_aucs),
            "held_out_auc": round(float(position_aucs[position]), 4),
            "held_out_auc_by_position": [round(float(a), 4) for a in position_aucs],
            "topic_control_in_sample_auc": round(float(topic_auc), 4),
            "topic_control_note": ("fitted at the same position as the register direction and "
                                   "scored in sample, because it is a control direction rather "
                                   "than a reported separation"),
            "random_directions": RANDOM_DIRECTIONS, "random_direction_pcs": RANDOM_DIRECTION_PCS,
        }
        (out / "direction.json").write_text(json.dumps(fitted, indent=2), encoding="utf-8")
        log(f"  direction: position {position} of {len(position_aucs) - 1}, "
            f"held-out AUC {fitted['held_out_auc']}")

    if "causal" in stages:
        if done("causal"):
            log("causal: already on disk, skipping (pass --force to re-run)")
        else:
            log("stage: causal")
            res = stage_causal(subject, parts, direction, controls,
                               judge_arm=args.judge_arm, seed=args.seed, log=log)
            (out / "causal.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            log(f"  causal: supported = {res['supported']}")

    if "nulls" in stages:
        if done("nulls"):
            log("nulls: already on disk, skipping (pass --force to re-run)")
        else:
            log("stage: nulls")
            res = stage_nulls(subject, parts, seed=args.seed, log=log)
            (out / "nulls.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            log(f"  nulls: status {res['status']}")
            log(f"  {res['status_reason']}")
    log(f"artefacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
