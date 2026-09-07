# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What the edit cost, measured on a task the model either gets right or does not.

WHY THIS EXISTS

Everything else this project measures is about refusal. `refusal_rate`, `soft_refusal_rate`,
`heretic_keyword_rate` are refusal rulers; `broken_rate` catches a model that has stopped forming
sentences; KL is a distributional proxy taken on harmless prompts; the compass measures whether
harm is still RECOGNISED. **Not one of them asks the model to do anything hard.** A model can sit
at KL 0.128 with `broken=0%`, which is exactly what a real run reported, and have lost multi-step
arithmetic, because nothing in the pipeline ever asks it to add up.

A tool whose pitch is receipts was missing the receipt that matters most.

WHY GRADE-SCHOOL ARITHMETIC FIRST

It is the benchmark reported as most sensitive to this class of edit, on the reading that
mathematical reasoning and refusal representations share circuitry. It is also the one that needs
no judge: the answer is a number, so grading is exact-match and there is no model-in-the-loop whose
own reliability would have to be established first. This project has withdrawn results over judges
before.

THE TRAP THIS FILE IS BUILT AROUND

A generation with no extractable number must be INDETERMINATE, never wrong. If truncation counted
as a wrong answer, then setting `--max-new` too low would look exactly like capability loss, and
the resulting number would be wrong in the same direction as the claim it was built to test. The
same mistake in a different costume already reached a published figure here once, when a thinking
model's verdict was read at the position it emits `<think>`.
"""
from __future__ import annotations

import json
import re

#: A number as a model writes one: optional sign, digits with optional thousands separators, an
#: optional decimal part. Deliberately does not accept a bare `.5`, because the things that look
#: like that in a worked solution are usually sentence-ending full stops.
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

#: How GSM8K marks the gold answer in its solution text.
GOLD_MARKER = "####"

#: Answers this far apart are the same answer. Grade-school answers are exact, and this exists only
#: so that 12.0 and 12 agree rather than to admit "close enough".
TOLERANCE = 1e-6


def parse_number(text):
    """The numeric value of a token as a model writes it, or None.

    Thousands separators are stripped because models emit them and the gold answers do not.
    """
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", "").rstrip(".")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def gold_answer(solution):
    """The reference answer from a GSM8K solution, which marks it with `####`."""
    if solution is None:
        return None
    text = str(solution)
    if GOLD_MARKER in text:
        text = text.rsplit(GOLD_MARKER, maxsplit=1)[-1]
    found = NUMBER.findall(text)
    return parse_number(found[-1]) if found else None


def predicted_answer(generation):
    """The model's answer: the LAST number it wrote, or None when it wrote none.

    Last rather than first, because a worked solution restates quantities from the question before
    it reaches a result, so the first number is usually the question's and the last is usually the
    answer.

    None is a real outcome and the caller must keep it separate from a wrong answer. A truncated
    generation, a refusal, and a model that answered in words all land here, and none of them is
    evidence that the arithmetic is gone.
    """
    if generation is None:
        return None
    found = NUMBER.findall(str(generation))
    return parse_number(found[-1]) if found else None


def grade_one(generation, solution, truncated=False):
    """One item, as "correct", "wrong" or "indeterminate".

    `truncated` says the generation stopped because it ran out of token budget rather than because
    the model finished, and it is REQUIRED for this to measure what it claims. Text alone cannot
    tell: a solution cut off at "he had 5 and bought 3 more, so he" has numbers in it, so the last
    one it managed to write gets read as the answer and the item scores WRONG. Every such item is
    a token budget being reported as a capability loss, in the same direction as the claim this
    module exists to test, which is the worst direction for an error to point.
    """
    if truncated:
        return "indeterminate"
    gold = gold_answer(solution)
    if gold is None:
        # The dataset row is unusable, not the model's fault, and silently scoring it wrong would
        # make a corpus defect look like a capability loss.
        return "indeterminate"
    got = predicted_answer(generation)
    if got is None:
        return "indeterminate"
    return "correct" if abs(got - gold) <= TOLERANCE else "wrong"


def grade(generations, solutions, truncated=None):
    """Every item's verdict, in order. Lengths must match; the caller has already checked.

    `truncated` is a parallel sequence of flags. It defaults to None only so a caller grading
    already-complete text is not forced to fabricate one, and a caller that has the information
    and does not pass it is measuring its own token budget.
    """
    flags = [False] * len(generations) if truncated is None else list(truncated)
    return [grade_one(g, s, t)
            for g, s, t in zip(generations, solutions, flags, strict=True)]


def summarise(verdicts):
    """Counts and the two rates that matter, kept apart on purpose.

    `accuracy` is over items that could be graded at all, which is the number a reader means by
    accuracy. `indeterminate` is reported beside it rather than folded in, because a run with a
    high indeterminate rate has not measured capability, it has measured its own token budget, and
    a single blended figure would hide exactly that.
    """
    n = len(verdicts)
    correct = sum(1 for v in verdicts if v == "correct")
    wrong = sum(1 for v in verdicts if v == "wrong")
    graded = correct + wrong
    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "indeterminate": n - graded,
        "graded": graded,
        "accuracy": round(correct / graded, 4) if graded else None,
        "indeterminate_rate": round((n - graded) / n, 4) if n else None,
    }


def paired_change(before, after, seed=0, resamples=2000, alpha=0.05):
    """The change in accuracy between two models graded on the SAME items, with an interval.

    The pairing is the point, exactly as it is for the compass. Two independent intervals that
    overlap do not mean the difference is uncertain: both models saw every item, so item difficulty
    cancels and the paired interval is far tighter than subtracting two unpaired ones suggests.
    Each replicate therefore draws ONE set of item indices and applies it to both.

    Items indeterminate under EITHER model are dropped from the comparison rather than scored,
    because a pair where one side could not be graded says nothing about the difference. The count
    of those is returned so a reader can see how much of the set the comparison rests on.

    The field this measures against currently publishes a claimed degradation with no interval at
    all, so shipping the point estimate alone would be joining the problem.
    """
    if len(before) != len(after):
        return None
    pairs = [(b, a) for b, a in zip(before, after, strict=True)
             if b in ("correct", "wrong") and a in ("correct", "wrong")]
    if not pairs:
        return None

    # McNemar's counts: the two disagreement cells are what a change actually consists of, and
    # they say something a delta cannot, namely whether items moved both ways or only one.
    fixed = sum(1 for b, a in pairs if b == "wrong" and a == "correct")
    broke = sum(1 for b, a in pairs if b == "correct" and a == "wrong")

    import torch  # local: this module is import-light for the same reason

    gen = torch.Generator().manual_seed(int(seed))
    n = len(pairs)
    draws = []
    for _ in range(resamples):
        idx = [int(i) for i in torch.randint(n, (n,), generator=gen)]
        b_acc = sum(1 for i in idx if pairs[i][0] == "correct") / n
        a_acc = sum(1 for i in idx if pairs[i][1] == "correct") / n
        draws.append(a_acc - b_acc)
    draws.sort()
    lo = draws[int((alpha / 2) * (resamples - 1))]
    hi = draws[int((1 - alpha / 2) * (resamples - 1))]
    point = (sum(1 for b, a in pairs if a == "correct")
             - sum(1 for b, a in pairs if b == "correct")) / n
    return {
        "compared_on": n,
        "dropped_indeterminate": len(before) - n,
        "delta_accuracy": round(point, 4),
        "delta_ci": (round(lo, 4), round(hi, 4)),
        # A sign that is not consistent across the resamples is the honest way to say "this is not
        # distinguishable from no change", and it is the sentence the field's published figures
        # are missing.
        "distinguishable_from_zero": bool(lo > 0 or hi < 0),
        "items_fixed": fixed,
        "items_broken": broke,
    }


def report(summary, change=None):
    """The result in the words a reader needs, rather than a dump of the dict."""
    lines = [
        f"  graded {summary['graded']} of {summary['n']} items",
        f"  accuracy {summary['accuracy']}" if summary["accuracy"] is not None
        else "  accuracy: nothing could be graded",
    ]
    if summary["indeterminate"]:
        lines.append(
            f"  INDETERMINATE {summary['indeterminate']} ({summary['indeterminate_rate']:.1%}): "
            f"no number in the generation. Usually the token budget, not the model. Raise "
            f"--max-new before reading anything into the accuracy above.")
    if change:
        lo, hi = change["delta_ci"]
        lines += [
            "",
            f"  change against the reference, on {change['compared_on']} shared items:",
            f"    {change['delta_accuracy']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]",
            f"    {change['items_broken']} items it used to get right and now does not",
            f"    {change['items_fixed']} the other way",
        ]
        if not change["distinguishable_from_zero"]:
            lines.append("    the interval spans zero, so this is not distinguishable from no "
                         "change")
    return lines


def load_reference(path):
    """A previous run's verdicts, for the paired comparison. Returns None when not asked for."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc.get("verdicts")


def generate_with_truncation(model, tok, prompts, device, batch=8, max_new=320):
    """Generate, and say for each item whether it finished or ran out of budget.

    Returns (generations, truncated_flags).

    Separate from `score.generate` because that one returns text alone, and text alone cannot
    answer the question this module turns on. An item is truncated when the model produced the
    whole budget without emitting an end-of-sequence token, which is exactly the condition under
    which the last number in the text is not an answer.

    `max_new` defaults far higher than the scorer's 64. A worked arithmetic solution is long, and a
    budget that cuts most of them off would make every model look equally incapable, which is a
    measurement of this function rather than of the model.
    """
    import torch

    from .cli import render_chat

    gens, truncated = [], []
    eos = tok.eos_token_id
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [render_chat(tok, p) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        start = enc.input_ids.shape[1]
        for j in range(len(chunk)):
            new_tokens = out[j][start:]
            gens.append(tok.decode(new_tokens, skip_special_tokens=True))
            # An end-of-sequence token anywhere in the new tokens means the model chose to stop.
            # Without one, it was still going when the budget ran out.
            finished = eos is not None and bool((new_tokens == eos).any())
            truncated.append(not finished)
    return gens, truncated
