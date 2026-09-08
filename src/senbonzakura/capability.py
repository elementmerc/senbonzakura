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
from dataclasses import dataclass

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


# ── the tasks, and why there is more than one ────────────────────────────────────────
# Arithmetic is one capability. A model can keep it and lose instruction-following, or lose the
# ability to pick the right option from a list, and a single benchmark would report none of that.
# Every task here is graded by CODE against a reference answer: no model in the loop, so no judge
# whose own reliability would have to be established before the result meant anything.
#
# Adding a task means adding an entry here. The three rules never move: an answer that cannot be
# graded is indeterminate, truncation is decided by the generator rather than guessed from text,
# and counts are reported beside every rate.

#: A multiple-choice answer as a model writes one: a letter, optionally in brackets or after a
#: label. Deliberately anchored so a stray capital in prose is not read as a choice.
CHOICE = re.compile(r"(?:^|[^A-Za-z])\(?([A-J])\)?(?:[.):]|\b)")

#: The two option letters that are also ordinary English words. Matched on the ORIGINAL text
#: rather than the upper-cased copy, because that is the only place the distinction survives.
#:
#: THE DEFECT THIS FIXES, and it is not a symmetric one. The pattern upper-cased the whole
#: generation first, so the article "a" and the pronoun "I" both matched, and taking the LAST
#: match meant "The answer is A. I hope that helps." graded as I, and "B, because it is a fact."
#: graded as A. Neither is indeterminate: both return a letter, so both are scored WRONG.
#:
#: The bias runs one way. A safety-tuned model hedges and self-refers ("I hope this helps",
#: "I'd note that"); an abliterated one is terser and less first-person. So the stock arm
#: collects more spurious matches than the edited arm, and `paired_change` reports abliteration
#: IMPROVING accuracy on general-knowledge questions. Found by a review pass running the grader
#: on realistic replies, which the existing tests never did.
_WORD_LETTERS = ("A", "I")


def choice_answer(text):
    """The LAST option letter in the text, upper-cased, or None.

    Last for the same reason the numeric task takes the last number: a model that reasons aloud
    names the options it is rejecting before it names the one it picks.
    """
    if text is None:
        return None
    raw = str(text)
    out = []
    for m in CHOICE.finditer(raw.upper()):
        letter = m.group(1)
        if letter in _WORD_LETTERS:
            # Lower case in the ORIGINAL is the article "a": a word, never a choice.
            if raw[m.start(1)].islower():
                continue
            # Upper case is ambiguous, and "I" is always capitalised in English, so case cannot
            # separate the pronoun. A real ANSWER carries a delimiter or ends the reply: "(A)",
            # "A.", "A:", "the answer is A". A pronoun is followed by its verb. Requiring that
            # much of the two word-letters keeps "The answer is A" and drops "I hope this helps".
            rest = raw[m.end(1):].lstrip(")")
            if rest.strip() and rest[0] not in ".):,;":
                continue
        out.append(letter)
    return out[-1] if out else None


def normalised_text(text):
    """A short free-text answer with the things that are not the answer removed.

    Case, surrounding punctuation and articles. Anything more aggressive starts deciding that two
    different answers are the same, which is a judge wearing a regex.
    """
    if text is None:
        return None
    s = " ".join(str(text).strip().lower().split())
    s = s.strip(".,;:!?\"'()[]")
    for article in ("the ", "a ", "an "):
        s = s.removeprefix(article)
    return s or None


def last_line_answer(text):
    """The final non-empty line, for tasks whose prompt asks for the answer on its own line."""
    if text is None:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return normalised_text(lines[-1]) if lines else None


# ── agentic behaviour, and how much of it needs no judge ─────────────────────────────
# "Agentic" sounds like the thing that finally forces a judge, and most of it does not. Whether a
# model emitted a well-formed tool call, named the right tool, passed the right arguments, and
# obeyed a format instruction are all questions code can answer exactly. A judge is only needed for
# the part that is genuinely a matter of opinion, and that part is smaller than it first looks.
#
# Reaching for a judge early would have made every number here inherit its reliability, which is
# what `senbonzakura judge` exists to stop. So these two tasks close as much of the gap as can be
# closed without one.

#: A JSON object as a model embeds one in prose or a code fence. Non-greedy from the last opening
#: brace, because a model that reasons aloud writes its call last.
JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)


def tool_call(text):
    """The tool call a model emitted, as (name, arguments), or None.

    Accepts the shapes models actually produce: a bare JSON object, one inside a ```json fence, or
    one embedded in prose. `arguments` may itself arrive as a JSON string rather than an object,
    which is common and is not the model getting it wrong.
    """
    if text is None:
        return None
    m = JSON_BLOB.search(str(text))
    if not m:
        return None
    try:
        doc = json.loads(m.group())
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    name = doc.get("name") or doc.get("tool") or doc.get("function")
    if isinstance(name, dict):                       # {"function": {"name": ...}}
        name = name.get("name")
    if not isinstance(name, str):
        return None
    args = doc.get("arguments", doc.get("args", doc.get("parameters", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {"__unparsed__": args}
    if not isinstance(args, dict):
        args = {"__value__": args}
    return (name, args)


def same_tool_call(got, want):
    """Same tool, same arguments. Argument ORDER is not part of the answer; presence and value are.

    Compared as parsed objects rather than as strings, because two calls that differ only in
    whitespace or key order are the same call, and marking them different would report a
    formatting change as a capability loss.
    """
    return got[0] == want[0] and got[1] == want[1]


#: The constraint checks a format instruction can be graded by. Deliberately small: every one is a
#: rule a reader could apply by hand and get the same answer, which is what keeps this out of
#: judge territory.
CONSTRAINTS = {
    "max_words": lambda text, v: len(text.split()) <= int(v),
    "min_words": lambda text, v: len(text.split()) >= int(v),
    "contains": lambda text, v: str(v).lower() in text.lower(),
    "excludes": lambda text, v: str(v).lower() not in text.lower(),
    "lines": lambda text, v: len([ln for ln in text.splitlines() if ln.strip()]) == int(v),
    "lowercase": lambda text, _v: text == text.lower(),
    "uppercase": lambda text, _v: text == text.upper(),
    "json": lambda text, _v: _is_json(text),
    "ends_with": lambda text, v: text.rstrip().endswith(str(v)),
    "starts_with": lambda text, v: text.lstrip().startswith(str(v)),
}


def _is_json(text):
    try:
        json.loads(JSON_BLOB.search(text).group() if JSON_BLOB.search(text) else text)
    except (ValueError, AttributeError):
        return False
    return True


def parse_constraints(spec):
    """`"max_words:50;lowercase"` into a list of (check, value). Unknown checks are a refusal.

    An unknown constraint is not silently skipped. A skipped check is one the model is graded as
    having passed, so a typo in a benchmark would quietly make every item easier.
    """
    if spec is None:
        return None
    out = []
    for raw in str(spec).split(";"):
        part = raw.strip()
        if not part:
            continue
        name, _, value = part.partition(":")
        name = name.strip()
        if name not in CONSTRAINTS:
            raise KeyError(
                f"unknown constraint {name!r}. Available: {', '.join(sorted(CONSTRAINTS))}.")
        out.append((name, value.strip()))
    return out or None


def meets_constraints(text, checks):
    """Every check, or nothing. A response that obeys three instructions and breaks the fourth has
    not followed the instruction.
    """
    return all(CONSTRAINTS[name](text, value) for name, value in checks)


@dataclass(frozen=True)
class Task:
    """One graded task: how to ask, how to read the answer, and how to compare it."""

    name: str
    prompt: str
    extract: object          # generation -> answer or None
    reference: object        # dataset answer -> answer or None
    same: object             # (got, want) -> bool
    grades: str              # what a reader should understand this measures
    needs_no_judge: str      # why the grading is trustworthy without a model in the loop
    #: Whether a generation the extractor cannot read is a WRONG answer rather than an ungradeable
    #: one. Usually False: a model that answers "eight" in words did the arithmetic and formatted
    #: it unexpectedly, and scoring that wrong would count a formatting habit as lost reasoning.
    #: True where producing the format IS the capability under test, as it is for a tool call:
    #: a model that replies in prose when asked for a call has failed the thing being measured,
    #: and calling that indeterminate would hide the most common way tool use breaks.
    #: Truncation is handled before this either way, so a cut-off generation never lands here.
    unparseable_is_wrong: bool = False


TASKS = {
    "numeric": Task(
        name="numeric",
        prompt="{}\n\nWork through it, then give the final answer as a number on the last line.",
        extract=predicted_answer,
        reference=gold_answer,
        same=lambda got, want: abs(got - want) <= TOLERANCE,
        grades="multi-step arithmetic and the reasoning that gets there",
        needs_no_judge="the answer is a number, so agreement is exact",
    ),
    "multiple-choice": Task(
        name="multiple-choice",
        prompt="{}\n\nAnswer with the letter of the correct option, on its own line.",
        extract=choice_answer,
        reference=lambda s: choice_answer(s) if s else None,
        same=lambda got, want: got == want,
        grades="knowledge and discrimination, without needing the model to compose an answer",
        needs_no_judge="the answer is one letter from a fixed set",
    ),
    "tool-call": Task(
        name="tool-call",
        prompt="{}\n\nRespond with a single JSON object naming the tool and its arguments, and "
               "nothing else.",
        extract=tool_call,
        reference=tool_call,
        same=same_tool_call,
        grades="whether the model can pick the right tool and pass it the right arguments, which "
               "is the part of agentic behaviour that fails first",
        needs_no_judge="the tool name and the argument values are compared as parsed data, so "
                       "agreement is exact and whitespace or key order cannot change it",
        unparseable_is_wrong=True,
    ),
    "constraints": Task(
        name="constraints",
        prompt="{}",
        extract=lambda gen: gen if gen and gen.strip() else None,
        reference=parse_constraints,
        same=lambda text, checks: meets_constraints(text, checks),
        grades="whether the model still follows a format instruction, which abliteration may "
               "damage independently of its reasoning",
        needs_no_judge="each constraint is a rule a reader could apply by hand and get the same "
                       "answer: a word count, a substring, a letter case",
        unparseable_is_wrong=True,
    ),
    "exact": Task(
        name="exact",
        prompt="{}\n\nAnswer as briefly as possible, on the last line and nothing else.",
        extract=last_line_answer,
        reference=normalised_text,
        same=lambda got, want: got == want,
        grades="short factual recall and whether the model can follow a format instruction",
        needs_no_judge="agreement is string equality after case and article normalisation, which "
                       "is narrow enough to be a rule rather than a judgement",
    ),
}

DEFAULT_TASK = "numeric"
TASK_CHOICES = sorted(TASKS)


def get_task(name):
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}. Available: {', '.join(TASK_CHOICES)}.") from None


def grade_one(generation, solution, truncated=False, task=DEFAULT_TASK):
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
    t = get_task(task)
    want = t.reference(solution)
    if want is None:
        # The dataset row is unusable, not the model's fault, and silently scoring it wrong would
        # make a corpus defect look like a capability loss.
        return "indeterminate"
    got = t.extract(generation)
    if got is None:
        return "wrong" if t.unparseable_is_wrong else "indeterminate"
    return "correct" if t.same(got, want) else "wrong"


def grade(generations, solutions, truncated=None, task=DEFAULT_TASK):
    """Every item's verdict, in order. Lengths must match; the caller has already checked.

    `truncated` is a parallel sequence of flags. It defaults to None only so a caller grading
    already-complete text is not forced to fabricate one, and a caller that has the information
    and does not pass it is measuring its own token budget.
    """
    flags = [False] * len(generations) if truncated is None else list(truncated)
    return [grade_one(g, s, cut, task)
            for g, s, cut in zip(generations, solutions, flags, strict=True)]


#: Above this share of ungradeable answers, the accuracy is not a statement about the model.
#:
#: Not because a smaller sample is imprecise, which the interval already reports, but because the
#: items that fail to finish are NOT MISSING AT RANDOM. A long worked solution truncates and a
#: short one does not, so the graded subset is biased toward the easy questions and the accuracy
#: over it is measured on a different, easier exam than the one that was set.
#:
#: 10% is a convention rather than a measurement, and it is stated as one. The case that forced
#: it was 41 of 200 (20.5%) on every arm of a real run INCLUDING the unedited reference, where a
#: 2.5 point drop was about to be read as a capability cost.
MAX_INDETERMINATE = 0.10


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
        # The verdict, in the record rather than left to a reader comparing two decimals. A run
        # this far past the threshold has measured its token budget, and every figure beside it
        # inherits that.
        "budget_suspect": bool(n and (n - graded) / n > MAX_INDETERMINATE),
        "budget_threshold": MAX_INDETERMINATE,
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
    if summary.get("budget_suspect"):
        lines.append(
            f"  BUDGET, NOT MODEL. {summary['indeterminate']} of {summary['n']} answers "
            f"({summary['indeterminate_rate']:.1%}) never finished, past the "
            f"{summary['budget_threshold']:.0%} this tool will report through. The ones that fail "
            f"to finish are the LONG ones, so what was graded is an easier exam than the one set "
            f"and the accuracy above is not a statement about this model. Raise --max-new and "
            f"run it again; do not quote any figure from this run.")
    elif summary["indeterminate"]:
        lines.append(
            f"  indeterminate {summary['indeterminate']} ({summary['indeterminate_rate']:.1%}): "
            f"no number in the generation, usually the token budget rather than the model. Below "
            f"the {summary['budget_threshold']:.0%} threshold, so the accuracy above stands.")
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
        if summary.get("budget_suspect"):
            lines.append("    AND this change is NOT QUOTABLE: it is a difference between two "
                         "accuracies measured on whichever items happened to finish.")
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
                         "as openai/gsm8k:main::test. A plain prompt list will not do: marking needs "
                         "the reference answer")
    ap.add_argument("--task", choices=TASK_CHOICES, default=DEFAULT_TASK,
                    help="how the answers are graded. 'numeric' (default) reads the last number, "
                         "for arithmetic sets like GSM8K; 'multiple-choice' reads the last option "
                         "letter; 'exact' compares the last line as text after normalising case "
                         "and articles. Every one is graded by code against a reference answer, "
                         "so no judge model is involved and none has to be validated first. The "
                         "task is recorded in the output, because an accuracy means nothing "
                         "without knowing what was asked.")
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

    task = get_task(a.task)
    prompts = [task.prompt.format(q) for q in questions]
    gens, truncated = generate_with_truncation(
        model, tok, prompts, a.device, batch=a.batch, max_new=a.max_new)
    verdicts = grade(gens, answers, truncated, task=a.task)
    summary = summarise(verdicts)
    change = None
    if reference is not None and a.bootstrap:
        change = paired_change(reference, verdicts, seed=a.seed, resamples=a.bootstrap)

    print(f"capability: {a.label or a.model} on {a.eval}")
    print(f"  task {task.name}: {task.grades}")
    for line in report(summary, change):
        print(line)

    from .crashsafe import atomic_write, provenance

    result = {"label": a.label, "model": a.model, "eval": a.eval, "task": a.task,
              "n": len(questions),
              "max_new": a.max_new, "seed": a.seed, "summary": summary,
              "verdicts": verdicts, "compare_to": a.compare_to or None, "change": change,
              # Which build produced this. Every other artefact in this project carries it and
              # this one did not, so a night of arm results came back citable everywhere except
              # the capability figures, which are the ones the whole experiment exists for.
              "provenance": provenance(device=a.device)}
    # Atomic, like every other result here. A capability run is a generation pass over hundreds of
    # prompts and a kill partway through the write used to leave a file that exists, is not empty,
    # and is not a result.
    with atomic_write(a.out) as f:
        _json.dump(result, f, indent=2)
    if a.save_generations:
        with open(a.save_generations, "w", encoding="utf-8") as f:
            for q, gold, gen, t, v in zip(questions, answers, gens, truncated, verdicts,
                                          strict=True):
                f.write(_json.dumps({"question": q, "gold": gold, "generation": gen,
                                     "truncated": t, "verdict": v}, ensure_ascii=False) + "\n")
    # Non-zero when nothing could be graded, because a run that measured nothing must not look
    # like a run that measured a zero.
    # Non-zero when the budget ate too much of the sample, so a pipeline cannot collect the
    # number and carry on. `tools/e2_arms.sh` did exactly that: three arms of capability figures
    # at 20.5% indeterminate, on every arm including the unedited reference.
    return 0 if summary["graded"] and not summary.get("budget_suspect") else 1


if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(main)
