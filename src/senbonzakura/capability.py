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
    from .metrics import reportable_rate

    n = len(verdicts)
    correct = sum(1 for v in verdicts if v == "correct")
    wrong = sum(1 for v in verdicts if v == "wrong")
    graded = correct + wrong
    acc = reportable_rate(correct, graded)
    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "indeterminate": n - graded,
        "graded": graded,
        "accuracy": round(acc["rate"], 4) if acc["rate"] is not None else None,
        # The interval, always, and the reason when the rate is withheld. A point estimate over a
        # handful of items is what reversed direction on the ROG when the sample grew, and the
        # interval is what would have said so at the time.
        "accuracy_ci": acc["ci"],
        "accuracy_reportable": acc["reportable"],
        "accuracy_withheld_because": acc["why_not"],
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
    lines = [f"  graded {summary['graded']} of {summary['n']} items"]
    if summary["accuracy"] is not None:
        lo, hi = summary["accuracy_ci"]
        lines.append(f"  accuracy {summary['correct']}/{summary['graded']} = "
                     f"{summary['accuracy']}  95% CI [{lo}, {hi}]")
    elif summary.get("accuracy_withheld_because"):
        lines.append(f"  accuracy: {summary['correct']}/{summary['graded']} and NOT REPORTED as a "
                     f"rate: {summary['accuracy_withheld_because']}")
    else:
        lines.append("  accuracy: nothing could be graded")
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
    # getattr, because a tokenizer without one is a real thing and crashing on it would be worse
    # than the conservative answer. No end-of-sequence token means finished and cut off cannot be
    # told apart, so everything is marked truncated and therefore indeterminate, which reports
    # nothing rather than reporting wrong answers.
    eos = getattr(tok, "eos_token_id", None)
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


#: How the question is put to the model. Deliberately plain and deliberately fixed: a prompt that
#: varies between the two arms would make the comparison a measurement of the prompt.
PROMPT = ("{}\n\nWork through it, then give the final answer as a number on the last line.")


def build_parser():
    import argparse

    from .cli import loader_parser

    ap = argparse.ArgumentParser(
        prog="senbonzakura capability",
        description="Measure what an edit cost, on a task the model either gets right or does "
                    "not. Refusal rates and KL cannot see capability loss; this can.",
        parents=[loader_parser()])
    ap.add_argument("--eval", required=True,
                    help="a graded benchmark with a question column and an answer column, such "
                         "as openai/gsm8k::test. A plain prompt list will not do: marking needs "
                         "the reference answer")
    ap.add_argument("--hf-token", dest="hf_token", default=None,
                    help="token for a gated or private Hub dataset; defaults to $HF_TOKEN")
    ap.add_argument("--question-column", dest="question_column", default=None,
                    help="name the question column when it cannot be detected")
    ap.add_argument("--answer-column", dest="answer_column", default=None,
                    help="name the answer column when it cannot be detected")
    ap.add_argument("--out", required=True, help="where the verdicts and summary are written")
    ap.add_argument("--label", default="", help="a name for this arm, recorded in the output")
    ap.add_argument("--n", type=int, default=200,
                    help="how many items (default 200). A FIXED subset, taken from the head, so "
                         "two arms are compared on the same questions")
    ap.add_argument("--skip", type=int, default=0, help="drop this many items from the head first")
    ap.add_argument("--max-new", dest="max_new", type=int, default=320,
                    help="token budget per answer (default 320). A worked solution is long, and "
                         "a budget that truncates most of them measures the budget rather than "
                         "the model. Truncated items are reported as indeterminate, never wrong")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--compare-to", dest="compare_to", default="",
                    help="a previous run's output, typically the stock model. Adds the PAIRED "
                         "change with its interval, which is much tighter than comparing two "
                         "separate runs by eye and is the number that says what the edit cost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="resamples for the interval on the change (0 disables it)")
    ap.add_argument("--save-generations", dest="save_generations", default="",
                    help="write every question, answer and verdict, so a disputed grade can be "
                         "checked without the GPU back")
    return ap


def main(argv=None):
    import json as _json

    from . import dataset
    from .cli import load_model_and_tokenizer

    a = build_parser().parse_args(argv)
    if a.n is not None and a.n < 1:
        raise SystemExit("--n must be at least 1.")

    try:
        questions, answers = dataset.resolve_pairs(
            a.eval, question_column=a.question_column, answer_column=a.answer_column,
            token=a.hf_token or None)
    except dataset.DatasetError as e:
        raise SystemExit(str(e)) from e

    if a.skip >= len(questions):
        raise SystemExit(f"--skip {a.skip} leaves nothing: the set has {len(questions)} items.")
    questions, answers = questions[a.skip:], answers[a.skip:]
    if a.n > len(questions):
        raise SystemExit(f"--n {a.n} exceeds the {len(questions)} items available after --skip.")
    questions, answers = questions[:a.n], answers[:a.n]

    reference = load_reference(a.compare_to)
    if reference is not None and len(reference) != len(questions):
        # Refused rather than truncated to fit. Two arms compared on different item sets is not a
        # paired comparison, and silently aligning them by position would produce a number that
        # looks paired and is not.
        raise SystemExit(
            f"--compare-to {a.compare_to} holds {len(reference)} items and this run has "
            f"{len(questions)}. A paired comparison needs the same items in the same order; "
            f"re-run with matching --n and --skip.")

    model, tok = load_model_and_tokenizer(
        a.model, device=a.device, load_in_4bit=a.load_in_4bit,
        trust_remote_code=a.trust_remote_code, chat_template=a.chat_template)

    prompts = [PROMPT.format(q) for q in questions]
    gens, truncated = generate_with_truncation(
        model, tok, prompts, a.device, batch=a.batch, max_new=a.max_new)
    verdicts = grade(gens, answers, truncated)
    summary = summarise(verdicts)
    change = None
    if reference is not None and a.bootstrap:
        change = paired_change(reference, verdicts, seed=a.seed, resamples=a.bootstrap)

    print(f"capability: {a.label or a.model} on {a.eval}")
    for line in report(summary, change):
        print(line)

    result = {"label": a.label, "model": a.model, "eval": a.eval, "n": len(questions),
              "max_new": a.max_new, "seed": a.seed, "summary": summary,
              "verdicts": verdicts, "compare_to": a.compare_to or None, "change": change}
    with open(a.out, "w", encoding="utf-8") as f:
        _json.dump(result, f, indent=2)
    if a.save_generations:
        with open(a.save_generations, "w", encoding="utf-8") as f:
            for q, gold, gen, t, v in zip(questions, answers, gens, truncated, verdicts,
                                          strict=True):
                f.write(_json.dumps({"question": q, "gold": gold, "generation": gen,
                                     "truncated": t, "verdict": v}, ensure_ascii=False) + "\n")
    # Non-zero when nothing could be graded, because a run that measured nothing must not look
    # like a run that measured a zero.
    return 0 if summary["graded"] else 1
