# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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

THE TOOL-CALL VALIDITY MEASURES ARE MECHANICAL AND ARE NOT A VERDICT ON TOOL USE

`tool_call_validity` and `tool_call_validity_block` answer one question: is the reply a well-formed
call against the schema the harness put in the prompt. Did it emit a call at all, does it name a
function on offer, are the required arguments there, are their types the declared ones, did it
invent a parameter. **Every one of those is about shape, and none of them reads the question.** A
model that answers everything by calling the wrong tool, with every argument present and correctly
typed and every value nonsense, scores a perfect 100% on all of them.

Whether calling THAT tool with THOSE values was the right decision is a judgement, and a judgement
needs a certified grader (`senbonzakura judge`, with its kappa floor) and a labelled set this
repository does not have. The split is the point: the mechanical half needs no labels at all,
because the schema is already in the harness's hand, so it can ship now and be believed, while the
judgement half waits for a grader whose own reliability has been established. Anything quoting a
figure from this block alongside the word "capable" has merged the two.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

from . import argresolve

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

#: KEPT, UNUSED BY `tool_call`, AND THE COMMENT IT CARRIED WAS FALSE. It read "non-greedy from the
#: last opening brace", and `.*` is greedy while `.search` finds the FIRST brace, so it matched
#: everything from the first `{` to the last `}`. A model that reasons aloud and writes a brace in
#: its prose therefore had its perfect call read as NO CALL, and because the `tool-call` task sets
#: `unparseable_is_wrong=True` that scored WRONG rather than unparsed.
#:
#: THE BIAS RAN ONE WAY AND IT WAS THE FLATTERING ONE, which is the same shape, direction and cause
#: as the `choice_answer` defect documented above: a safety-tuned model reasons aloud and writes
#: braces, an abliterated one is terser, so the stock arm collected more of these false failures
#: and `paired_change` would have reported abliteration IMPROVING tool use.
#:
#: Safe to fix without a re-run, checked before fixing: `tool-call` is one of five GRADING RULES
#: and the only dataset this package ships is a 256 row GSM8K subset, graded `numeric`. No
#: published figure was taken through it. `docs/comparison.md` is exact about that distinction.
#:
#: The name survives because it is public and something may import it. Nothing here calls it.
JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)


def tool_call(text):
    """The tool call a model emitted, as (name, arguments), or None.

    Accepts the shapes models actually produce: a bare JSON object, one inside a ```json fence, or
    one embedded in prose. `arguments` may itself arrive as a JSON string rather than an object,
    which is common and is not the model getting it wrong.

    Scanned for brace-balanced substrings rather than matched with a regular expression, and the
    LAST one that parses into a call wins, because a model that reasons aloud writes its call last.
    That is what the regular expression above claimed to do and did not.
    """
    candidates, _ = _balanced_json_objects(text)
    for blob in reversed(candidates):
        try:
            doc = json.loads(blob)
        except ValueError:
            continue
        call = _call_from_object(doc)
        if call is not None:
            return call
    return None


#: What an `arguments` value that could not be read as an object gets stored under, so that the
#: failure survives into the record instead of being dropped. `__unparsed__` is an `arguments`
#: string that is not JSON; `__value__` is an `arguments` that parsed to a list, a number or a
#: bare string. Both mean the ARGUMENTS are malformed rather than that the model passed a
#: parameter by those names, and `tool_call_validity` reports them as such.
ARGS_SENTINELS = ("__unparsed__", "__value__")


def _call_from_object(doc):
    """(name, arguments) from an already-parsed JSON object, or None if it names no function.

    The key aliases live here in one copy because two extractors read them: `tool_call`, which
    grades against a reference call, and `emitted_tool_call`, which grades against a schema. Two
    copies of the alias list would let the two measures disagree about what counts as a call.
    """
    if not isinstance(doc, dict):
        return None
    name = doc.get("name") or doc.get("tool") or doc.get("function")
    # THE ARGUMENTS LIVE BESIDE THE NAME, WHICH IS NOT ALWAYS THE TOP LEVEL. OpenAI's shape is
    # `{"function": {"name": ..., "arguments": ...}}`, and reading the name from the nested object
    # while reading the arguments only from the outer one returned ("f", {}) for every such call:
    # every argument silently dropped, and `same_tool_call` compares arguments, so a correct call
    # scored WRONG rather than unparsed. Found 2026-10-08.
    #
    # The nested object is PREFERRED and the outer one is a fallback, not a replacement. Models
    # also emit `{"function": {"name": ...}, "parameters": {...}}`, with the name nested and the
    # arguments outside it, and an existing test pins that shape. Taking the nested object
    # wholesale broke it.
    nested = name if isinstance(name, dict) else None
    if nested is not None:                           # {"function": {"name": ...}}
        name = nested.get("name")
    if not isinstance(name, str):
        return None

    def _args_from(obj):
        for key in ("arguments", "args", "parameters"):
            if key in obj:
                return obj[key], True
        return {}, False

    args, found = ({}, False) if nested is None else _args_from(nested)
    if not found:
        args, _ = _args_from(doc)
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


# ── mechanical validity of a tool call, which needs no judge and no labels ────────────
# WHAT THIS BLOCK IS AND IS NOT. It answers "is this a well-formed call against the schema the
# harness handed the model": did it emit a call at all, does it name a function the schema offers,
# are the required arguments present, are their types the declared ones, and did it invent a
# parameter. Every one of those is a question about SHAPE.
#
# NONE OF IT ASKS WHETHER CALLING THAT TOOL WAS THE RIGHT MOVE, or whether the values passed were
# the right values. A model that answers every request by calling the wrong function with
# beautifully typed arguments scores 100% here. That half needs a certified grader, which is what
# `senbonzakura judge` exists for, and the labelled set this repository does not have.
#
# WHY IT IS WORTH HAVING ANYWAY. The schema is already in the harness's hand, because it is what
# was put in the prompt, so this costs no annotation at all. `tool_call` above grades against a
# REFERENCE call, so it needs a gold answer per item; this needs only the schema. That is the whole
# difference, and it is what moves the mechanical half of the measure off the labelled-data gate.

#: Why a reply carries no gradeable call. Three different failures, kept apart because collapsing
#: them loses the only information a reader can act on: a model that answers in prose needs a
#: different fix from one that emits a call with a comma missing.
NO_CALL_NO_JSON = "no_json"                   # prose, with nothing object-shaped in it at all
NO_CALL_UNPARSEABLE = "unparseable_json"      # something object-shaped that will not parse
NO_CALL_NAMES_NOTHING = "names_no_function"   # parsed cleanly, but names no function

#: What each JSON type name admits, as Python types. `boolean` is checked BEFORE the numeric names
#: and `bool` is excluded from both of them on purpose: `isinstance(True, int)` is True in Python,
#: so a model that passes `true` where an integer was declared would otherwise be scored as having
#: got the type right. That is a real shape of tool-call failure and the language hides it.
JSON_TYPES = {
    "string": (str,),
    "boolean": (bool,),
    "integer": (int,),
    "number": (int, float),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _balanced_json_objects(text):
    """Every brace-balanced substring of `text`, in order, and whether any brace was seen at all.

    Hand-scanned rather than matched with a regular expression because the question "was anything
    object-shaped emitted" has to be answerable separately from "did it parse", and a regex that
    fails to match cannot tell those apart. Quotes are only honoured once a brace is open, so a
    quotation mark in the surrounding prose cannot swallow the call that follows it.
    """
    s = "" if text is None else str(text)
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(s):
        if depth and in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if depth and ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(s[start:i + 1])
    return out, ("{" in s)


def emitted_tool_call(text):
    """(name, arguments) if the reply emitted a call, else (None, why it did not).

    THE LAST BALANCED OBJECT WINS, not the first. A model that reasons aloud writes its call at the
    end, and often writes braces before it, so taking the first opening brace reads the reasoning
    and taking everything between the first and the last reads neither.

    Returns the reason as one of the `NO_CALL_*` codes when there is no call, because "answered in
    prose", "emitted malformed JSON" and "emitted clean JSON that names no function" are three
    different defects and a single None would merge them.
    """
    candidates, saw_brace = _balanced_json_objects(text)
    parsed_any = False
    for blob in reversed(candidates):
        try:
            doc = json.loads(blob)
        except ValueError:
            continue
        parsed_any = True
        call = _call_from_object(doc)
        if call is not None:
            return call, None
    if parsed_any:
        return None, NO_CALL_NAMES_NOTHING
    return None, (NO_CALL_UNPARSEABLE if saw_brace else NO_CALL_NO_JSON)


def tool_schema(schema):
    """A tool schema in the shapes providers write them, as (name, declared types, required).

    Returns None when the schema names no function, which makes the item ungradeable rather than
    failed: an unusable schema is the harness's defect, and scoring the model against it would
    report our own bug as a capability loss. `grade_one` takes the same line on an unusable dataset
    row. A schema that names a function and declares no parameters is USABLE, not broken: a
    zero-argument tool is a real tool, and against it every argument supplied is an undeclared one.
    """
    if not isinstance(schema, dict):
        return None
    body = schema.get("function") if isinstance(schema.get("function"), dict) else schema
    name = body.get("name")
    if not isinstance(name, str) or not name:
        return None
    params = body.get("parameters", body.get("input_schema", body))
    if not isinstance(params, dict):
        return None
    props = params.get("properties", {})
    if not isinstance(props, dict):
        return None
    declared = {}
    for key, spec in props.items():
        kind = spec.get("type") if isinstance(spec, dict) else None
        declared[key] = kind if isinstance(kind, (str, list)) else None
    required = params.get("required", [])
    required = [r for r in required if isinstance(r, str)] if isinstance(required, list) else []
    return (name, declared, required)


def _toolbox(schema):
    """Every usable schema in what the caller passed, keyed by function name.

    One schema or a sequence of them, because a harness that offers the model a choice of tools is
    the normal case and "named a function the schema offers" is only meaningful against the whole
    set it was offered.
    """
    items = schema if isinstance(schema, (list, tuple)) else [schema]
    out = {}
    for one in items:
        parsed = tool_schema(one)
        if parsed is not None:
            out[parsed[0]] = parsed
    return out


def _type_mismatch(value, declared):
    """Whether `value` violates a declared JSON type. An undeclared type cannot be violated."""
    if declared is None:
        return False
    names = [declared] if isinstance(declared, str) else list(declared)
    admitted = tuple(t for n in names for t in JSON_TYPES.get(n, ()))
    if not admitted:
        # A type name this does not know is not a licence to call the value wrong. Reporting a
        # mismatch here would turn an unsupported schema keyword into a model failure.
        return False
    if isinstance(value, bool):
        return not any(n == "boolean" for n in names)
    return not isinstance(value, admitted)


def tool_call_validity(reply, schema, truncated=False):
    """Whether one reply is a MECHANICALLY VALID call against `schema`. Not whether it was right.

    THE CAVEAT IS THE RETURN VALUE'S FIRST KEY and it is there in every record this produces. A
    call that names the wrong tool for the question, with every argument present, correctly typed
    and holding a nonsense value, is `valid: True` here. Nothing in this function reads the
    question, so nothing in it is evidence that the model chose or filled the tool well.

    `truncated` says the generation ran out of token budget rather than finishing, and it is
    REQUIRED for this to measure what it claims: a call cut off mid-object is indistinguishable
    from malformed JSON by looking at the text, and scoring it as malformed would report a small
    `--max-new` as a tool-use failure, in the same direction as the claim the measure exists to
    test. A caller that holds the flag and does not pass it is measuring its own token budget.
    """
    out = {
        "measures": ("mechanical validity against the schema, NOT whether calling that tool with "
                     "those values was correct"),
        "indeterminate": False,
        "indeterminate_because": None,
        "emitted_call": False,
        "no_call_because": None,
        "name": None,
        "known_function": None,
        "arguments_malformed": None,
        "missing_required": None,
        "wrong_typed": None,
        "undeclared": None,
        "valid": None,
    }
    if truncated:
        out["indeterminate"] = True
        out["indeterminate_because"] = "the generation ran out of token budget before it finished"
        return out
    box = _toolbox(schema)
    if not box:
        out["indeterminate"] = True
        out["indeterminate_because"] = (
            "the schema names no usable function, so there was nothing to check the reply against")
        return out

    call, why = emitted_tool_call(reply)
    if call is None:
        out["no_call_because"] = why
        out["valid"] = False
        return out

    name, args = call
    out["emitted_call"] = True
    out["name"] = name
    out["known_function"] = name in box
    if not out["known_function"]:
        # The argument checks are left as None rather than filled in against nothing. There is no
        # schema for a function the schema does not offer, so "no arguments missing" would be true
        # in the way that an empty list is true, and would read as a pass.
        out["valid"] = False
        return out

    malformed = sorted(k for k in ARGS_SENTINELS if k in args)
    out["arguments_malformed"] = malformed or []
    if malformed:
        # The sentinels are not parameter names and must not be reported as undeclared ones: the
        # failure is that the `arguments` value could not be read, not that the model invented a
        # parameter called `__unparsed__`.
        out["valid"] = False
        return out

    declared, required = box[name][1:]
    out["missing_required"] = sorted(k for k in required if k not in args)
    out["undeclared"] = sorted(k for k in args if k not in declared)
    out["wrong_typed"] = [
        {"argument": k, "declared": declared[k], "got": type(v).__name__}
        for k, v in sorted(args.items())
        if k in declared and _type_mismatch(v, declared[k])
    ]
    out["valid"] = not (out["missing_required"] or out["undeclared"] or out["wrong_typed"])
    return out


def tool_call_validity_block(replies, schema, truncated=None):
    """The mechanical validity measures over many replies, or None when there are none.

    EVERY RATE CARRIES ITS OWN DENOMINATOR, and they are not the same denominator. `valid` is over
    the replies that could be graded at all. The argument checks are over the replies that named a
    function the schema offers, which is a smaller set, because a required argument has no meaning
    against a function that does not exist. A single `n` across all of them would be wrong for most
    of the rows under it.

    MECHANICAL, which the per-reply record also says and which does not get weaker by repetition:
    these say whether the calls were well formed, not whether they were the right calls. A model
    that reaches for the wrong tool every time, with clean arguments, scores perfectly on every
    figure in this block, so nothing here is evidence that its tool use is good.
    """
    from .metrics import reportable_rate

    rows = list(replies)
    if not rows:
        return None
    flags = [False] * len(rows) if truncated is None else list(truncated)
    reports = [tool_call_validity(r, schema, cut) for r, cut in zip(rows, flags, strict=True)]

    n = len(reports)
    graded = [r for r in reports if not r["indeterminate"]]
    called = [r for r in graded if r["emitted_call"]]
    checkable = [r for r in called if r["known_function"]]

    out = {
        "measures": ("mechanical validity against the schema, NOT whether calling that tool with "
                     "those values was correct"),
        "n": n,
        "indeterminate": n - len(graded),
        "graded": len(graded),
        # The same ceiling the accuracy measures use, and for the same reason: the replies that
        # fail to finish are not missing at random, so past this point the rates below are a
        # statement about the token budget rather than about the model.
        "budget_suspect": bool(n and (n - len(graded)) / n > MAX_INDETERMINATE),
        "budget_threshold": MAX_INDETERMINATE,
        "valid": reportable_rate(sum(1 for r in graded if r["valid"]), len(graded)),
        "emitted_call": reportable_rate(len(called), len(graded)),
        # Why there was no call, as counts over the graded replies. Not rates: the interesting
        # thing about these is which one it was, and three rates over the same small denominator
        # invite reading a one-reply difference as a trend.
        "no_call_because": {
            code: sum(1 for r in graded if r["no_call_because"] == code)
            for code in (NO_CALL_NO_JSON, NO_CALL_UNPARSEABLE, NO_CALL_NAMES_NOTHING)
        },
        "invented_function": reportable_rate(
            sum(1 for r in called if not r["known_function"]), len(called)),
    }
    for key in ("arguments_malformed", "missing_required", "wrong_typed", "undeclared"):
        # Over the calls that named a real function, which is the only set these are defined on.
        # `reportable_rate` returns a None rate under the floor rather than a bare decimal, and a
        # caller that renders that None as 0.0 has put the defect back.
        out[key] = reportable_rate(sum(1 for r in checkable if r[key]), len(checkable))
    out["checkable"] = len(checkable)
    out["reports"] = reports
    return out


# ── offering the tools to the model, which is what makes the block above mean anything ───
#
# THE MEASURES ABOVE WERE UNREACHABLE and that is the defect this section closes. `tool_call_
# validity_block` was written, tested and called by nothing, which is this project's most
# frequently repeated failure: a measure that exists, looks finished, and has never once run. It
# has caught us on `--capability-eval` reading a flag that did not exist, on an in-search
# capability gate that had never executed, and on a compass print whose deletion broke no test.
#
# Wiring it needs one thing the command did not have: a schema. And a schema the model never saw
# cannot be graded against, because "passed an argument the tool does not declare" is only a
# failure if the model was told what the tool declares. Grading an unprompted reply against a
# hidden schema would score the model on a guess and report it as malformed tool use, in the
# flattering direction for an edited model, which is the direction this module exists to distrust.
#
# So the schema goes into the prompt and into the grading from one source, and the exam fingerprint
# covers it, because two runs offered DIFFERENT tools are not the same exam and pairing them would
# compare item i against a different item i.


def load_tool_schema(path):
    """The tool declarations to offer and to grade against, from a JSON file.

    One schema object or a list of them. Refused loudly rather than half-read: a schema that names
    no function would make every item ungradeable, and a run that produced nothing but
    indeterminates because of a typo in a path is a wasted generation pass over hundreds of prompts.
    """
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except OSError as e:
        raise SystemExit(f"--tool-schema {path}: cannot be read ({e}).") from e
    except ValueError as e:
        raise SystemExit(f"--tool-schema {path}: is not valid JSON ({e}).") from e
    if isinstance(doc, dict) and isinstance(doc.get("tools"), list):
        # The shape a provider request body uses, accepted because that is the file somebody
        # already has rather than one they have to write out again by hand.
        doc = doc["tools"]
    items = doc if isinstance(doc, list) else [doc]
    box = _toolbox(items)
    if not box:
        raise SystemExit(
            f"--tool-schema {path}: holds no usable tool declaration. Each one needs a function "
            f"name, either at the top level or under a 'function' key. A tool that declares no "
            f"parameters is fine; a tool with no name is not, because every grade below depends "
            f"on whether the model named a tool that was offered.")
    return items, box


def offered_tools(box):
    """The tool declarations as the model is shown them, or "" when none were offered.

    DETERMINISTIC, sorted by name and by parameter name, because this text goes into the prompt and
    therefore into the exam fingerprint. A dictionary's iteration order leaking in here would make
    two identical runs look like two different exams and refuse to pair them.

    Written out in prose rather than as pasted JSON on purpose. A JSON block invites a small model
    to continue the block instead of answering, and the thing being measured is whether it emits a
    call, so a prompt that makes a transcription failure look like a tool-use failure is measuring
    the prompt.
    """
    if not box:
        return ""
    lines = ["You have these tools:"]
    for name in sorted(box):
        _, declared, required = box[name]
        if not declared:
            lines.append(f"- {name}: takes no arguments.")
            continue
        parts = []
        for key in sorted(declared):
            kind = declared[key]
            if isinstance(kind, list):
                kind = " or ".join(str(k) for k in kind)
            shown = f"{key} ({kind})" if kind else key
            parts.append(f"{shown} required" if key in required else f"{shown} optional")
        lines.append(f"- {name}: " + ", ".join(parts) + ".")
    return "\n".join(lines)


def tool_validity_report(block):
    """The validity measures as lines, or the reason there are none.

    A WITHHELD RATE IS PRINTED AS WITHHELD, never as zero. `reportable_rate` returns None for a
    rate whose denominator is below the floor, and a renderer that formats that as 0.0% would tell
    a reader the model never once passed an undeclared argument when what happened is that too few
    replies could be checked to say.
    """
    if block is None:
        return ["  tool calls: nothing to grade, so no validity figures were taken."]
    out = [f"  tool calls, MECHANICAL ONLY: {block['measures']}."]

    def pct(key, label):
        # The denominator comes out of the measure rather than being worked out again here. Each
        # of these rates is over a different set and recomputing the sets at the rendering layer
        # is how a figure ends up printed over the wrong n.
        row = block[key]
        if row["rate"] is None:
            out.append(f"    {label}: {row['count']} of {row['n']}, and no rate, because "
                       f"{row['why_not']}")
        else:
            lo, hi = row["ci"]
            out.append(f"    {label}: {row['rate'] * 100:.1f}% ({row['count']}/{row['n']}, "
                       f"95% CI {lo * 100:.1f} to {hi * 100:.1f}%)")

    pct("emitted_call", "emitted a call at all")
    pct("valid", "mechanically valid")
    pct("invented_function", "named a tool that was not offered")
    for key, label in (("missing_required", "missing a required argument"),
                       ("wrong_typed", "argument of the wrong declared type"),
                       ("undeclared", "passed an argument the tool does not declare"),
                       ("arguments_malformed", "arguments could not be read")):
        pct(key, label)
    reasons = {k: v for k, v in block["no_call_because"].items() if v}
    if reasons:
        out.append("    no call because: "
                   + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    if block["indeterminate"]:
        out.append(f"    {block['indeterminate']} of {block['n']} replies could not be graded "
                   f"because they ran out of token budget.")
    if block["budget_suspect"]:
        out.append(f"    TOOL_CALL_BUDGET_SUSPECT: more than "
                   f"{block['budget_threshold'] * 100:.0f}% of replies were cut off, so the rates "
                   f"above describe --max-new rather than the model. Raise it and re-run.")
    out.append("    None of these says the model chose the right tool or filled it sensibly. A "
               "model that calls the wrong tool every time, cleanly, scores perfectly here.")
    return out


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
        # handful of items is what reversed direction on real hardware when the sample grew, and the
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
    # WHAT THE READER CAN ACTUALLY SEE ABOVE, carried forward rather than assumed. The budget
    # warnings below used to say "the accuracy above" and "the accuracy above stands" whichever of
    # these three branches had run, so a run that graded nothing printed "accuracy: nothing could
    # be graded" and then told the reader that the accuracy above was not a statement about this
    # model. A caveat about a figure that is not there reads as a figure that is.
    figure_above = None
    if summary["accuracy"] is not None:
        lo, hi = summary["accuracy_ci"]
        lines.append(f"  accuracy {summary['correct']}/{summary['graded']} = "
                     f"{summary['accuracy']}  95% CI [{lo}, {hi}]")
        figure_above = "the accuracy above"
    elif summary.get("accuracy_withheld_because"):
        lines.append(f"  accuracy: {summary['correct']}/{summary['graded']} and NOT REPORTED as a "
                     f"rate: {summary['accuracy_withheld_because']}")
        figure_above = "the counts above"
    else:
        lines.append("  accuracy: nothing could be graded")
    ref_rate = (change or {}).get("reference_indeterminate_rate")
    if change and ref_rate is not None and ref_rate > MAX_INDETERMINATE:
        lines.append(
            f"  THE COMPARISON IS NOT QUOTABLE, because the run it is compared AGAINST left "
            f"{ref_rate:.1%} of its answers ungraded, past the {MAX_INDETERMINATE:.0%} ceiling. "
            f"A pair is dropped when EITHER arm failed to grade it, and the ones that fail are "
            f"the long answers, so the pairs that survive are dominated by the reference's easy "
            f"items. That understates the cost in the flattering direction. Re-run the reference "
            f"with a larger --max-new before reading the change.")
    if summary.get("budget_suspect"):
        cost = (f"so what was graded is an easier exam than the one set and {figure_above} is not "
                f"a statement about this model"
                if figure_above else
                "and here nothing could be graded at all, so this run has produced no figure "
                "about this model to read")
        lines.append(
            f"  BUDGET, NOT MODEL. {summary['indeterminate']} of {summary['n']} answers "
            f"({summary['indeterminate_rate']:.1%}) never finished, past the "
            f"{summary['budget_threshold']:.0%} this tool will report through. The ones that fail "
            f"to finish are the LONG ones, {cost}. Raise --max-new and "
            f"run it again; do not quote any figure from this run.")
    elif summary["indeterminate"]:
        stands = {
            "the accuracy above": "so the accuracy above stands",
            "the counts above": "so it is not why the rate above was withheld",
        }.get(figure_above, "though nothing could be graded here, so there is no rate either way")
        lines.append(
            f"  indeterminate {summary['indeterminate']} ({summary['indeterminate_rate']:.1%}): "
            f"no number in the generation, usually the token budget rather than the model. Below "
            f"the {summary['budget_threshold']:.0%} threshold, {stands}.")
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


def not_a_measurement(summary, change=None, *, ceiling=MAX_INDETERMINATE):
    """Why this run produced no figure about this model, or None when it produced one.

    THE VERDICT WAS ALREADY BEING MADE, THREE TIMES, AND IT WENT NOWHERE A MACHINE COULD READ IT.
    `report` has printed all three of these in capitals since before the ledger entry, and
    `main`'s exit status covered two of them, and the artefact said nothing at all. So
    `senbonzakura measure`, which reads the file rather than the exit status, printed the accuracy
    from a run whose own log said not to quote any figure from it, and the checker's rule for a
    figure a file calls invalid could not fire because no capability file ever carried the field.
    This returns the reason in the one vocabulary every consumer already speaks.

    Pure, and deliberately reading only the summary and the change dicts, so each branch is
    testable without a model, a card or a generation pass.

    WHAT IS NOT IN HERE, and why each one is a caveat rather than a refusal:

      - An accuracy withheld because fewer than `metrics.MIN_REPORTABLE_N` items could be graded.
        `summarise` already withholds the rate itself and records why, so there is no figure to
        misquote: the counts go out as counts. A run that grades 12 of 12 items has measured
        something real about 12 items and saying so is honest.
      - A comparison whose pairing could not be verified, which is `change["items_verified"]`
        false. That is an unchecked claim rather than a known-false one, it is already recorded in
        the artefact and printed as a NOTE, and every reference artefact written before the exam
        fingerprint existed would trip it, including this project's own published arms. A refusal
        that fires on correct historical runs is a refusal somebody switches off.
      - A change whose interval spans zero. That is a result.
    """
    if not summary.get("graded"):
        return (f"nothing in this run could be graded, so there is no accuracy here and no "
                f"figure about this model: {summary.get('n')} items were asked and not one came "
                f"back with a readable answer in it. The budget or the prompt is the cause far "
                f"more often than the model is. Raise --max-new, and keep --save-generations on "
                f"the next run so you can read what the model actually said.")
    if summary.get("budget_suspect"):
        rate = summary.get("indeterminate_rate")
        return (f"{summary.get('indeterminate')} of {summary.get('n')} answers never finished, "
                f"which is past the {ceiling:.0%} this tool reports through"
                f"{'' if rate is None else f' (this run: {rate:.1%})'}. The answers that fail to "
                f"finish are the long ones, so what was graded is an easier exam than the one "
                f"that was set, and an accuracy over it is a statement about the budget rather "
                f"than about the model. Raise --max-new and run it again.")
    reference_rate = (change or {}).get("reference_indeterminate_rate")
    if reference_rate is not None and reference_rate > ceiling:
        return (f"the run this one is compared against left {reference_rate:.1%} of its own "
                f"answers ungraded, past the {ceiling:.0%} ceiling. A pair is dropped when either "
                f"arm failed to grade it, and the ones that fail are the long answers, so the "
                f"surviving pairs are dominated by the reference's easy items and the change "
                f"understates the cost in the flattering direction. This run's own accuracy "
                f"stands; the change figure beside it does not. Re-run the reference with a "
                f"larger --max-new before reading the comparison.")
    return None


def load_reference(path):
    """A previous run's verdicts AND its summary, for the paired comparison.

    THE SUMMARY WAS THROWN AWAY, and with it the reference run's indeterminate rate. The
    comparison drops any item either arm failed to grade, and the warning that says the graded
    subset is an easier exam was guarded on the CURRENT run's `budget_suspect` alone. So a stock
    arm at 35% indeterminate compared against an edited arm at 2% produced a change figure with
    no warning at all.

    The bias is the flattering one. Indeterminates are the long answers, and abliteration makes
    a model terser, so the dropped pairs are dominated by the BASE model's truncations, which are
    its hard items. What is left is an easier exam for the arm that was already struggling, and
    the measured capability loss is systematically understated. That is the same missing-not-at-
    random argument this module is built around, one level up, in the function that produces the
    number the module exists to produce.

    Returns `(verdicts, summary, items_digest)`; all are None when no reference was asked for.
    `items_digest` is None for an artefact written before the field existed, which is a different
    answer from "the items differ" and is reported differently. See `items_digest()`.
    """
    if not path:
        return None, None, None
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc.get("verdicts"), doc.get("summary"), doc.get("items_digest")


def items_digest(questions, offered=""):
    """A fingerprint of the exam, so a paired comparison can check WHICH items it paired.

    WHY A COUNT IS NOT ENOUGH, and this is the defect it closes.

    `paired_change` pairs the reference run's verdicts with this run's by POSITION: item i against
    item i. The only thing that guarded it was `len(reference) != len(questions)`, and a length is
    not an identity. Two runs with the same number of items and a different selection, a different
    `--skip`, a different `--eval` file, or a dataset whose order moved between downloads, pair
    item i against a different item i and produce a number that looks paired and is not.

    Nothing downstream would notice. The lengths match, the verdicts are valid strings, McNemar's
    counts compute, the bootstrap returns an interval. There is no signal anywhere, and it would
    surface much later as a capability change nobody can account for.

    Raised by the stegcore session on 2026-09-21, which had just found the same shape in its own
    builder: a JPEG cover pool indexed positionally, where deleting one file shifted every cover
    after it down a place and paired each stego half against a clean half it never came from. Its
    generalisation is the one worth keeping: **an index that is positional rather than keyed is a
    silent mispairing waiting for its first gap.** A key that names the thing survives a deletion;
    a position does not.

    Sixteen hex characters, matching the width of the other digests this project's artefacts
    carry, so a reader meets one kind of thing rather than three.

    `offered` is the tool declarations the model was shown, which are part of the exam and not part
    of the question. The same hundred questions asked with a different toolbox are a different
    paper, and without this a run offered one tool would pair item by item against a run offered
    five and report the difference as a change in capability. An empty `offered`, which is every
    run of every other task, hashes exactly as it did before this argument existed, so artefacts
    already on disk still pair.
    """
    h = hashlib.sha256()
    for q in questions:
        # Length-prefixed, so ["ab", "c"] and ["a", "bc"] cannot collide. Two different exams
        # hashing alike would put the check back where it started.
        raw = str(q).encode("utf-8")
        h.update(str(len(raw)).encode("ascii") + b"\0" + raw)
    if offered:
        # Under its own separator so a toolbox can never be mistaken for one more question, which
        # is the collision that would make the length prefixing above pointless.
        raw = str(offered).encode("utf-8")
        h.update(b"tools\0" + str(len(raw)).encode("ascii") + b"\0" + raw)
    return h.hexdigest()[:16]


#: Seconds to generate one token for one item, on a CPU, measured 2026-09-23 on an RTX 3060
#: laptop's host CPU with Qwen3-1.7B in float32 at batch 4: 73.1 s per item at 512 new tokens.
#: A larger model is worse, and this is not scaled by model size on purpose. It is used to say
#: roughly how long a run will take before it starts, not to predict it; a number that is right
#: to the order of magnitude is what decides whether somebody should be asked first.
CPU_SECONDS_PER_TOKEN = 73.1 / 512

#: How long a capability probe may take on a CPU before it has to be asked for. Half an hour is
#: chosen as the point past which somebody would reasonably assume the tool had hung, which is
#: exactly what happened to CI: the probe's own notice was the last line before the 900 second
#: kill, and nothing said whether it was working.
CPU_REFUSE_AFTER_SECONDS = 30 * 60


def cpu_probe_estimate(n, max_new):
    """Roughly how long `n` items at `max_new` tokens will take on a CPU, in seconds."""
    return int(n) * int(max_new) * CPU_SECONDS_PER_TOKEN


def refuse_a_slow_probe(device, n, max_new, *, spec="bundled", allowed=False, log=print):
    """Refuse a CPU probe measured in hours unless somebody asked for one.

    THE SAME SHAPE AS `--short-budget-ok`, and for the same reason. The capability probe became a
    default on 2026-09-22 at 200 items and 512 new tokens, which are sensible on a GPU. Measured
    on a CPU the same defaults take about four hours for a 1.7B model, and this project is aimed
    at people on laptops.

    A heartbeat was added first, and it is not enough: it makes a long wait legible rather than
    short, and somebody watching a progress line crawl for four hours reaches for Ctrl+C on a run
    that was working. The alternatives were all worse. Shrinking the default weakens every
    measurement to suit the slowest machine, and 200 paired items is what makes the number worth
    quoting. Varying the default by device is the one that looks most helpful and is the most
    dangerous: two people running an identical command would no longer be running an identical
    experiment, which is precisely the comparability hole this codebase has spent months closing.

    So the default stays honest and the run says what it will cost before spending it.
    """
    # THE SAME CONDITION THE PROBE ITSELF USES, mirrored rather than approximated. `_score_items`
    # returns without measuring anything when the spec is empty or `n` is zero, so a run that
    # turned the probe off with `--capability-eval ""` was being refused for hours it was never
    # going to spend. The first version of this guard read `n` and `max_new` alone and did exactly
    # that, which is a guard inventing a cost for work that does not happen.
    if not spec or int(n or 0) <= 0:
        return
    if str(device).lower() not in ("cpu", "", "none"):
        return
    seconds = cpu_probe_estimate(n, max_new)
    if seconds <= CPU_REFUSE_AFTER_SECONDS:
        return
    hours = seconds / 3600
    if allowed:
        log(f"WARNING: the capability probe is {int(n)} items at {int(max_new)} tokens on a CPU, "
            f"roughly {hours:.1f} hours, and --slow-probe-ok was given. It reports progress every "
            f"30 seconds; a line that has not moved for several minutes is a fault, not the wait.")
        return
    raise SystemExit(
        f"senbonzakura: the capability probe would take roughly {hours:.1f} hours on this CPU "
        f"({int(n)} items at {int(max_new)} tokens each).\n"
        f"  That estimate comes from a measurement of a 1.7B model, so a larger one is worse. It "
        f"is stopping here rather than starting, because a run that looks identical to a hung one "
        f"for four hours is how somebody kills work that was fine.\n"
        f"  What to do:\n"
        f"    run it on a GPU with --device cuda, where these defaults are minutes, or\n"
        f"    measure less of it:  --capability-n 40 --capability-max-new 256\n"
        f"    turn it off entirely with --capability-n 0, and get no capability number at all\n"
        f"    if you meant it and will leave it running, add --slow-probe-ok")


#: Device strings in a `hf_device_map` that mean "not on an accelerator". `disk` is worse than
#: `cpu` rather than better: a layer paged off an SSD every forward pass is slower than one in
#: host RAM, so counting it as host speed under-estimates rather than over-estimates.
_HOST_DEVICES = ("cpu", "disk", "meta")


def offloaded_share(model):
    """The fraction of a loaded model's entries that are NOT on an accelerator, or None.

    Read from `hf_device_map`, which is what accelerate actually built, rather than from the
    `--device` string, which is what the user asked for. The two are different facts and the
    guard above only had the second one.
    """
    dmap = getattr(model, "hf_device_map", None) or {}
    if not dmap:
        return None
    host = sum(1 for d in dmap.values() if str(d).lower().split(":")[0] in _HOST_DEVICES)
    return host / len(dmap)


def disk_offloaded_entries(model):
    """How many of a loaded model's entries accelerate put on DISK, and the total.

    SEPARATE FROM `offloaded_share` BECAUSE DISK IS A DIFFERENT OUTCOME, not a slower one.
    `_HOST_DEVICES` groups `cpu`, `disk` and `meta` because for the purpose of pricing generation
    they are all "not the accelerator". For the purpose of telling somebody whether their run
    works they are not alike at all: a CPU-offloaded weight is editable in place, and a
    disk-offloaded one makes `cli._real_tensor` raise, because its offload map returns a fresh
    tensor on every read and the bake would write into a copy that is discarded.

    So a notice built on `offloaded_share` alone can only ever say "slower". That is what it said,
    for a placement under which the edit refuses outright.
    """
    dmap = getattr(model, "hf_device_map", None) or {}
    if not dmap:
        return 0, 0
    on_disk = sum(1 for d in dmap.values() if str(d).lower().split(":")[0] == "disk")
    return on_disk, len(dmap)


def report_offload_cost_for_a_search(model, *, trials, prompts_per_trial, gen_tokens, log=print):
    """Say that the SEARCH is running partly on the host, and roughly what that costs.

    WHY THE SEARCH AND NOT JUST THE PROBE. `offloaded_share` had exactly one caller: the capability
    probe, which refuses when placement makes it slow. The search is the same fact applied to a job
    one to two orders of magnitude longer, and it said nothing at all. So a run on a card too small
    for its model got a careful warning about the four-minute probe and silence about the four-hour
    search, which is this project's recurring shape: a guard that covers one spelling of a defect
    reports clean on the others.

    IT REPORTS, IT DOES NOT REFUSE. Decided by the operator on 2026-09-27, and it is the right call
    for this path: an offloaded search is slow, whereas an offloaded capability probe buys a number
    nobody needs in order to finish an edit. Refusing the long job would leave somebody with a 6 GB
    card unable to run the tool at all, and a tool that refuses the hardware it is aimed at has
    chosen purity over use.

    THE RATE IS A PROJECTION, and crude in a stated direction. It multiplies the same measured
    per-token CPU constant the probe uses by the offloaded fraction, so it treats an offloaded layer
    as costing host time and a resident one as costing nothing. The true figure is worse, because a
    partly offloaded forward pass also moves activations across the bus every step, and the constant
    was measured on a 1.7B model so a larger one is worse again. Under-stating is the right
    direction here for the opposite reason to the probe's: this number is advice, not a gate, and an
    over-stated one gets dismissed.
    """
    share = offloaded_share(model)
    if not share:
        return
    # DISK FIRST, AND ABOVE THE BUDGET GUARDS. Everything below this prices a slower run, and
    # on a disk placement there is no run to price: `cli._real_tensor` raises on the first
    # disk-offloaded writer it is asked to edit, because the offload map hands back a fresh tensor
    # each read and the bake would write into a copy. This notice said "The run works" for that
    # placement, which is the `doctor` CPU advisory's defect in a second place: a reassurance built
    # on a check that never tested the thing it was reassuring about. `offloaded_share` cannot tell
    # the two apart by design, so the disk reading is taken separately.
    #
    # ABOVE THE BUDGET GUARDS, because this needs no arithmetic: a disk placement refuses the edit
    # at any token budget, and the early return on `tokens <= 0` was silencing it on exactly the
    # paths that inject a model without one.
    on_disk, total = disk_offloaded_entries(model)
    if on_disk:
        log(f"NOTE: {on_disk} of {total} module groups are on DISK, not in host RAM, because the "
            f"card and the host together have less free memory than the model needs.")
        log("  The edit will not run. A disk-offloaded weight is handed back as a fresh copy on "
            "every read, so the bake would write into something discarded before the next forward "
            "pass and the model would come out unedited with nothing saying so. It is refused "
            "instead, when the first such weight is reached.")
        log("  To make it a run: free the card, use one with more memory, add host RAM, or load "
            "smaller with --load-in-4bit. Host-RAM offload is fine; disk is not.")
        return
    # EVERY BUDGET IS COERCED, because this runs on the way into the longest job the tool has and
    # must never be the reason a run fails to start. `gen_tokens` arrives as None on the paths that
    # inject a model, and an unguarded `int(None)` here crashed a real search in a test written for
    # exactly that. A notice is not worth a traceback.
    tokens = int(gen_tokens or 0)
    generations = max(1, int(trials or 1)) * max(1, int(prompts_per_trial or 1))
    if tokens <= 0:
        return
    seconds = cpu_probe_estimate(generations, tokens) * share
    hours = seconds / 3600
    log(f"NOTE: {share * 100:.0f}% of this model's layers are in host RAM, not on the "
        f"GPU, because the card has less free memory than the model needs.")
    log(f"  The run works. It generates at host speed for that share, and the search is the long "
        f"part: roughly {hours:.1f} hours for about {generations} generations at {int(gen_tokens)} "
        f"tokens, on top of the bake.")
    log("  That is a projection from a per-token CPU measurement of a 1.7B model, not a "
        "measurement of this one, and it under-states rather than over-states.")
    log("  To make it a GPU run: free the card, use one with more memory, or load smaller with "
        "--load-in-4bit on the measurement paths. --device cuda alone does not do it.")


def refuse_a_slow_probe_after_load(model, n, max_new, *, spec="bundled", allowed=False,
                                   log=print):
    """The same refusal again, decided from where the weights ended up rather than from a flag.

    WHY BOTH HALVES EXIST. The command-line check above reads `--device`, and `--device cuda` is
    not a statement about where the weights are: the loader passes `device_map="auto"`, and
    accelerate dispatches whatever will not fit in VRAM to host RAM or to disk. Generation then
    runs at host speed on a run that never looked like a CPU run, so the guard written for
    exactly that wait never fired.

    The layering is `preflight_snapshot_ram`'s: estimate from what can be read cheaply, then check
    once for real when the thing being estimated is actually resident. A model fully on an
    accelerator returns immediately and costs a dictionary walk.

    The offloaded fraction is the multiplier on the CPU constant, which is deliberately crude. It
    treats an offloaded layer as costing CPU time and a resident one as costing nothing, and the
    true figure is worse than that, because a partly offloaded forward pass also pays to move
    activations across the bus. Under-estimating is the right direction for a guard that refuses:
    it fires late rather than wrongly.
    """
    if not spec or int(n or 0) <= 0:
        return
    share = offloaded_share(model)
    if not share:
        return
    seconds = cpu_probe_estimate(n, max_new) * share
    if seconds <= CPU_REFUSE_AFTER_SECONDS:
        return
    hours = seconds / 3600
    placed = f"{share * 100:.0f}% of this model's layers are on the CPU or on disk, not on the GPU"
    if allowed:
        log(f"WARNING: {placed}, so the capability probe will generate at host speed: roughly "
            f"{hours:.1f} hours for {int(n)} items at {int(max_new)} tokens. --slow-probe-ok was "
            f"given, so it is running. It reports progress every 30 seconds.")
        return
    raise SystemExit(
        f"senbonzakura: {placed}, so the capability probe would generate at host speed and take "
        f"roughly {hours:.1f} hours ({int(n)} items at {int(max_new)} tokens each).\n"
        f"  The card has less free memory than this model needs, so accelerate put the rest in "
        f"host RAM. That is a working model and a very slow one, and --device cuda does not make "
        f"it a GPU run.\n"
        f"  What to do:\n"
        f"    free the card, or use one with more memory, or load smaller with --load-in-4bit\n"
        f"    measure less of it:  --n 40 --max-new 256\n"
        f"    if you meant it and will leave it running, add --slow-probe-ok")


def generate_with_truncation(model, tok, prompts, device, batch=8, max_new=320, *, log=None,
                             heartbeat=30.0):
    """Generate, and say for each item whether it finished or ran out of budget.

    Returns (generations, truncated_flags).

    SAY SOMETHING WHILE IT WORKS. This loop had no output of any kind, and once the capability
    probe became a default on 2026-09-22 that turned every plain `abliterate` on a CPU into a run
    that prints the probe's attribution notice and then goes silent for as long as it takes to
    generate 200 worked solutions. CI found it the hard way: the end-to-end smoke was killed at
    900 seconds with the notice as its last line, and nothing in the log said whether the job was
    working or wedged. A user on a laptop gets exactly the same thing and no timeout to rescue
    them, so they reach for Ctrl+C, which is the worst available reading of a run that was fine.

    Baseline 2.1 asks for a heartbeat every 30 to 60 seconds on every long-running loop. This is
    that, emitted per batch and rate limited by `heartbeat` seconds so a fast GPU run does not
    paper the log with a line per batch. The first batch reports unconditionally, because the
    useful moment is the one where a person is deciding whether anything is happening at all.

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
    started = last_beat = time.monotonic()
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
        now = time.monotonic()
        if log and (i == 0 or now - last_beat >= heartbeat or len(gens) == len(prompts)):
            last_beat = now
            done, total = len(gens), len(prompts)
            elapsed = now - started
            # An ETA from the rate so far, which is honest on this loop: every batch does the same
            # work, so the estimate does not drift the way it would on a search whose trials get
            # cheaper. Omitted on the first batch, where one sample is not a rate.
            rest = (elapsed / done) * (total - done) if done else 0.0
            eta = f", about {rest:.0f}s left" if done < total and done > batch else ""
            log(f"  capability probe: {done}/{total} items, {elapsed:.0f}s elapsed{eta}")
    return gens, truncated


#: How the question is put to the model. Deliberately plain and deliberately fixed: a prompt that
#: varies between the two arms would make the comparison a measurement of the prompt.
PROMPT = ("{}\n\nWork through it, then give the final answer as a number on the last line.")


#: Where the result goes when nobody said. Named after the command rather than after the
#: model, because a directory of results from one command is what `senbonzakura measure`
#: and `report` both read.
DEFAULT_OUT = "capability.json"


def build_parser():

    # FROM `.parser`, NOT FROM `.cli`, which re-exports it. `.cli` imports torch, optuna and
    # transformers at module scope, so reaching `loader_parser` through it made BUILDING THE
    # PARSER need the whole abliteration stack, and `capability --help` raised ImportError on
    # exactly the install where a person is trying to find out what to install.
    from .argresolve import whole_number
    from .parser import loader_parser

    ap = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog="senbonzakura capability",
        description="Measure what an edit cost, on a task the model either gets right or does "
                    "not. Refusal rates and KL cannot see capability loss; this can.",
        parents=[loader_parser(model_required=False)])
    # THE MODEL WITHOUT A FLAG, as the default command already takes it. `--model` still works and
    # is what every run spec on record passes; this is a second spelling of the same argument, and
    # `resolve_model` refuses rather than picking a winner when the two disagree.
    ap.add_argument("model_positional", nargs="?", default=None, metavar="MODEL",
                    help="the model to measure, given without a flag. Equivalent to --model.")
    ap.add_argument("--eval", default="bundled",
                    help="what to measure against. 'bundled' (the default) is the probe that "
                         "ships with the package, so this works offline. Otherwise a graded "
                         "benchmark with a question column and an answer column, such as "
                         "openai/gsm8k:main::test, or a contributed probe directory. A plain "
                         "prompt list will not do: marking needs the reference answer")
    # `metavar` so the usage line reads `[--task TASK]`. Spelling the five choices there pushed it
    # past 80 columns, and the choices are listed in the help below, where there is room to say
    # what each one means.
    ap.add_argument("--task", choices=TASK_CHOICES, default=DEFAULT_TASK, metavar="TASK",
                    help="how the answers are graded, and it is recorded in the output. 'numeric' "
                         "(default) reads the last number; 'multiple-choice' reads the last option "
                         "letter; 'exact' compares the last line as text, ignoring case and "
                         "articles; 'tool-call' compares a JSON tool name and its arguments; "
                         "'constraints' checks a format instruction was followed. All grade by "
                         "code against a reference answer, so no judge model is involved.")
    ap.add_argument("--tool-schema", dest="tool_schema", default="",
                    help="a JSON file of tool declarations, for --task tool-call. The tools are "
                         "OFFERED to the model in the prompt and the replies are then graded "
                         "against them mechanically: did it emit a call, name a tool that exists, "
                         "pass the required arguments, and get the declared types right. Those "
                         "are separate from the graded accuracy and say nothing about whether the "
                         "call was the right one to make")
    ap.add_argument("--hf-token", dest="hf_token", default=None,
                    help="token for a gated or private Hub dataset; defaults to $HF_TOKEN")
    ap.add_argument("--question-column", dest="question_column", default=None,
                    help="name the question column when it cannot be detected")
    ap.add_argument("--answer-column", dest="answer_column", default=None,
                    help="name the answer column when it cannot be detected")
    ap.add_argument("--out", default=None,
                    help=f"where the verdicts and summary are written (default: "
                         f"./{DEFAULT_OUT}). A default that already exists is refused rather "
                         f"than replaced, because two runs' numbers in one filename are "
                         f"indistinguishable afterwards")
    ap.add_argument("--label", default="", help="a name for this arm, recorded in the output")
    ap.add_argument("--n", type=whole_number("--n"), default=200,
                    help="how many items (default 200). A FIXED subset, taken from the head, so "
                         "two arms are compared on the same questions")
    ap.add_argument("--skip", type=whole_number("--skip"), default=0,
                    help="drop this many items from the head first")
    ap.add_argument("--max-new", dest="max_new", type=whole_number("--max-new", minimum=1),
                    default=512,
                    help="token budget per answer (default 512). A worked solution is long, and "
                         "a budget that truncates most of them measures the budget rather than "
                         "the model. Truncated items are reported as indeterminate, never wrong")
    ap.add_argument("--batch", type=whole_number("--batch", minimum=1), default=8,
                    help="prompts per generation batch (default: 8). Lower than the other "
                         "commands because graded answers are longer; lower it further if the "
                         "card runs out of memory")
    ap.add_argument("--compare-to", dest="compare_to", default="",
                    help="a previous run's output, typically the stock model. Adds the PAIRED "
                         "change with its interval, which is much tighter than comparing two "
                         "separate runs by eye and is the number that says what the edit cost")
    ap.add_argument("--seed", type=argresolve.whole_number("--seed", minimum=0), default=0,
                    help="seed for the bootstrap resampling behind the reported interval "
                         "(default: 0). Generation itself is greedy, so this changes the "
                         "interval, not the answers")
    ap.add_argument("--bootstrap", type=whole_number("--bootstrap"), default=2000,
                    help="resamples for the interval on the change (0 disables it)")
    ap.add_argument("--save-generations", dest="save_generations", default="",
                    help="write every question, answer and verdict, so a disputed grade can be "
                         "checked without the GPU back")
    # THE SAME FLAG THE ABLITERATE PATH CARRIES, because this command reaches the same generation
    # loop with the same defaults and the refusal has to be answerable from here too. Without it,
    # the only way past the guard on this command would be to measure less than you meant to.
    ap.add_argument("--slow-probe-ok", dest="slow_probe_ok", action="store_true",
                    help="run the probe even when it will take hours: on a CPU, or on a model "
                         "the card could not hold and accelerate put in host RAM. It is refused "
                         "by default because a run that looks identical to a hung one for four "
                         "hours is how somebody kills work that was fine")
    return ap


def main(argv=None):
    import json as _json

    # PARSED BEFORE THE HEAVY IMPORTS, and the order is the fix. `.cli` imports torch, optuna and
    # transformers at module scope, so importing it up here meant `capability --help` raised
    # ImportError on any install without the abliterate extra: the one command a person runs to
    # find out what they need told them nothing and traced. `--help` and a usage error both exit
    # inside `parse_args`, so nothing heavy has to exist for either.
    a = build_parser().parse_args(argv)
    from .argresolve import pick_model, refuse_to_overwrite
    a.model = pick_model(a.model_positional, a.model, command="senbonzakura capability")
    if a.out is None:
        a.out = refuse_to_overwrite(DEFAULT_OUT, what="capability result")

    # THE TOOLBOX, resolved before ANYTHING ELSE happens, because a bad path here is a typo and a
    # typo should cost a second rather than a dataset download, a model load and a generation
    # pass over hundreds of prompts. The same argument as every other pre-flight in this file.
    tool_items, tool_box, offered = None, None, ""
    if a.tool_schema:
        if a.task != "tool-call":
            # Refused rather than ignored. A flag that is silently dead on the task somebody
            # passed it with is how a run gets reported as having measured tool validity when
            # nothing measured anything, and this command has shipped that defect before.
            raise SystemExit(
                f"--tool-schema is for --task tool-call and this run is --task {a.task}. The "
                f"validity measures grade a reply against the tools it was offered, which only "
                f"means something when the task is to call one.")
        tool_items, tool_box = load_tool_schema(a.tool_schema)
        offered = offered_tools(tool_box)
        print(f"offering {len(tool_box)} tool(s) from {a.tool_schema}: "
              f"{', '.join(sorted(tool_box))}")

    from . import dataset
    from .cli import load_model_and_tokenizer
    if a.n is not None and a.n < 1:
        raise SystemExit("--n must be at least 1.")

    # THREE THINGS `--eval` CAN BE, resolved here rather than by `resolve_pairs`, because two of
    # them are this project's own shapes and one is a general table reader. Keeping the package's
    # own formats out of the dataset layer is what stops that layer growing a special case per
    # release.
    from . import probe as probefmt
    if a.eval == "bundled":
        questions, answers = zip(*load_probe(), strict=True)
        questions, answers = list(questions), list(answers)
        probe_notice(used=min(a.n, len(questions)) if a.n else None)
    elif probefmt.is_probe(a.eval):
        try:
            loaded = probefmt.load(a.eval)
        except probefmt.ProbeError as e:
            raise SystemExit(str(e)) from e
        pairs = loaded.pairs()
        questions, answers = [q for q, _ in pairs], [r for _, r in pairs]
        # The probe names its own grading rule. A probe graded by a rule it was not written for
        # produces a number that looks fine and means nothing.
        a.task = loaded.task
        print(f"probe {loaded.name!r}: {loaded.measures!r}, graded by {loaded.task!r}")
        print("  The format gate checks shape, not contents. Read a contributed probe before "
              "reporting a number from it.")
    else:
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

    reference, reference_summary, reference_digest = load_reference(a.compare_to)
    if reference is not None and len(reference) != len(questions):
        # Refused rather than truncated to fit. Two arms compared on different item sets is not a
        # paired comparison, and silently aligning them by position would produce a number that
        # looks paired and is not.
        raise SystemExit(
            f"--compare-to {a.compare_to} holds {len(reference)} items and this run has "
            f"{len(questions)}. A paired comparison needs the same items in the same order; "
            f"re-run with matching --n and --skip.")

    # THE OTHER HALF OF THAT REFUSAL, and until 2026-09-21 it did not exist. The check above
    # compares COUNTS, and the message beside it promises "the same items in the same order"
    # while verifying only the first word of it. Two runs of the same size over different items
    # pair item i against a different item i, and nothing downstream can tell: the verdicts are
    # valid, McNemar's counts compute, an interval comes back.
    mine = items_digest(questions, offered)
    if reference is not None and reference_digest and reference_digest != mine:
        raise SystemExit(
            f"--compare-to {a.compare_to} was measured on a DIFFERENT set of {len(reference)} "
            f"items (exam {reference_digest}, this run {mine}). The counts match and the items do "
            f"not, so pairing them by position would compare item 1 against somebody else's item "
            f"1 and report it as a change in capability. Re-run both arms with the same --eval, "
            f"--task, --n and --skip.")

    # THE SLOW-PROBE GUARD, ON THIS COMMAND TOO, and until 2026-09-25 it was not. It was written
    # for a generation loop that three doors lead into, and it was called from one of them: the
    # abliterate path. `senbonzakura capability Qwen/Qwen3-1.7B --device cpu` takes the same 200
    # items at 512 tokens, which this guard's own constant puts at about four hours, and it
    # started without a word. A guard that covers one caller of a shared loop is a guard the next
    # caller does not have.
    #
    # Here rather than at the top of `main` because the item count is only known once the eval set
    # is resolved and sliced: estimating from `--n` alone would invent a cost for items a smaller
    # set does not have.
    refuse_a_slow_probe(a.device, len(questions), a.max_new, spec=a.eval,
                        allowed=a.slow_probe_ok)

    model, tok = load_model_and_tokenizer(
        a.model, device=a.device, load_in_4bit=a.load_in_4bit,
        trust_remote_code=a.trust_remote_code, chat_template=a.chat_template)
    # AND AGAIN FROM WHERE THE WEIGHTS ACTUALLY LANDED. The check above reads `--device`, which is
    # a request; this reads the map accelerate built, which is the fact.
    refuse_a_slow_probe_after_load(model, len(questions), a.max_new, spec=a.eval,
                                   allowed=a.slow_probe_ok)

    task = get_task(a.task)
    prompts = [task.prompt.format(q) for q in questions]
    if offered:
        # Before the question, because the model should know what it has to work with before it
        # reads what it is being asked to do, and after nothing, because a preamble buried under a
        # long question is a preamble a small model has forgotten by the time it answers.
        prompts = [f"{offered}\n\n{p}" for p in prompts]
    gens, truncated = generate_with_truncation(
        model, tok, prompts, a.device, batch=a.batch, max_new=a.max_new, log=print)
    verdicts = grade(gens, answers, truncated, task=a.task)
    summary = summarise(verdicts)
    change = None
    if reference is not None and a.bootstrap:
        change = paired_change(reference, verdicts, seed=a.seed, resamples=a.bootstrap)
        # The reference run's OWN ungraded rate, carried into the comparison so the warning can
        # read both arms rather than only this one. `None` when the reference predates the field,
        # which is a different thing from zero and is reported as unknown rather than as fine.
        if change is not None:
            change["reference_indeterminate_rate"] = (
                (reference_summary or {}).get("indeterminate_rate"))
            # WHETHER THE PAIRING WAS VERIFIED, carried with the number rather than assumed.
            # An artefact written before `items_digest` existed cannot say which items it
            # measured, so the comparison rests on position and a matching count alone. That is
            # not the same as a checked pairing and it must not read like one: absent has to be
            # reported as unknown, because treating it as fine is how a silent mispairing gets a
            # clean bill of health. Re-run the reference arm to get a checked comparison.
            change["items_verified"] = bool(reference_digest)
            if not reference_digest:
                print(f"  NOTE: {a.compare_to} predates the exam fingerprint, so this "
                      f"comparison paired the two arms BY POSITION and could only check that "
                      f"the counts match. If the two runs used different items, the change "
                      f"figure is not a paired comparison. Re-run the reference arm to have "
                      f"this checked rather than assumed.")

    print(f"capability: {a.label or a.model} on {a.eval}")
    print(f"  task {task.name}: {task.grades}")
    for line in report(summary, change):
        print(line)

    # THE MECHANICAL BLOCK, beside the graded accuracy and not instead of it. They answer
    # different questions and a reader who takes one for the other has the wrong number: accuracy
    # says whether the call matched the reference, validity says whether it could have been
    # executed at all. A model can score 0% accuracy with 100% validity by calling a real tool
    # correctly and wrongly every time.
    tool_calls = tool_call_validity_block(gens, tool_items, truncated) if tool_items else None
    if tool_items:
        for line in tool_validity_report(tool_calls):
            print(line)

    from .crashsafe import atomic_write, provenance

    result = {"label": a.label, "model": a.model, "eval": a.eval, "task": a.task,
              "n": len(questions),
              # WHICH items, not just how many. A later run comparing against this artefact
              # checks this before it pairs anything: a count is not an identity, and two runs
              # of equal size over different exams pair item i against a different item i and
              # report the difference as a change in capability.
              "items_digest": mine,
              # WHICH tools were offered, by name and by the exact text the model was shown.
              # The digest above already covers it, but a digest says two runs differ without
              # saying how, and the text is what somebody needs to reproduce the exam.
              "tool_schema": a.tool_schema or None,
              "tools_offered": sorted(tool_box) if tool_box else None,
              "tools_offered_text": offered or None,
              "tool_calls": tool_calls,
              "max_new": a.max_new, "seed": a.seed, "summary": summary,
              "verdicts": verdicts, "compare_to": a.compare_to or None, "change": change,
              # Which build produced this. Every other artefact in this project carries it and
              # this one did not, so a night of arm results came back citable everywhere except
              # the capability figures, which are the ones the whole experiment exists for.
              "provenance": provenance(device=a.device)}
    # IN THE FILE, not only in the exit status. The status is read by a shell; the artefact is read
    # by `measure`, by the checker and by whoever opens it in six months, and those three were the
    # ones quoting the number. Set only when it fires, which is the shape `margin` established and
    # every consumer was written against.
    invalid = not_a_measurement(summary, change)
    if invalid:
        result["self_invalidated"] = invalid
    # The canonical metrics block, beside the fields this command has always written rather than
    # instead of them. Withheld deliberately when the accuracy was: a run whose graded subset is
    # too small to report is a run with no capability number, and stamping `None` under a
    # declared metric would hand a reader an identity for a measurement that was not taken.
    if summary.get("accuracy") is not None:
        from senbonzakura_check import measurement

        from . import stamps
        measurement.stamp(result, "capability", summary["accuracy"], "code-graded",
                          n=summary.get("graded"), interval=summary.get("accuracy_ci"),
                          # THE FIVE PINNED FIELDS, absent until 2026-09-25. The digest is the
                          # exam's own, which this command already computes and which identifies
                          # WHICH items were asked rather than how many: a count is not an
                          # identity, and two runs of equal size over different exams pair item i
                          # against a different item i. The partition carries `--skip`, because an
                          # exam scored from row 0 includes items a later run skips.
                          **stamps.pinned(prompts=prompts, model=model, tok=tok,
                                          load_in_4bit=a.load_in_4bit,
                                          input_digest=result["items_digest"],
                                          skip=getattr(a, "skip", 0)))
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
    # number and carry on. `tools/research/e2_arms.sh` did exactly that: three arms of capability figures
    # at 20.5% indeterminate, on every arm including the unedited reference.
    #
    # Read off `not_a_measurement` rather than recomputed here, so the status, the artefact and
    # the printed verdict cannot say three different things. It adds one case the status used to
    # miss: a comparison against a reference that could not grade its own answers, which `report`
    # has always printed as NOT QUOTABLE and which exited 0.
    if invalid:
        print(f"  CAPABILITY_NOT_A_MEASUREMENT: this run has recorded, in {a.out}, that its own "
              f"figure is not a measurement of this model's capability, for the reason above. It "
              f"exits non-zero so a pipeline cannot collect the number and carry on.")
        return 1
    return 0


# ── the probe that ships in the package ──────────────────────────────────────────────────────
#
# WHY THERE IS ONE AT ALL. Until 2026-09-22 `--capability-eval` defaulted to empty, so the gate
# that matters most was the one nobody switched on, and a run could report a clean bake on a model
# that had quietly lost multi-step arithmetic. A default that needs a download is not a default:
# this tool is meant to work with no network, which is why the corpora ship inside the wheel.
#
# WHY IT IS COMMITTED WHERE THE CORPORA ARE GENERATED. They hold harmful prompts and are published
# as a gated dataset on purpose. This is grade-school arithmetic under MIT. Generating it would
# mean a clone install could not run the default gate, which is the `--track default` defect found
# that same morning, reproduced on the gate whose absence is hardest to notice.

#: The bundled probe's filename. Rebuilt by `tools/packaging/build_capability_probe.py`, which
#: pins the upstream revision and can check this file against what the pin rebuilds to.
PROBE_NAME = "capability-gsm8k.jsonl"

#: The probe's own column names, and they are deliberately not "question" and "answer".
#: `tools/ci/check_prompt_artefacts.py` treats those two as prompt-shaped field names and
#: refuses to let them reach a public tree, which is exactly its job: it is the one control
#: between a harmful prompt and a public push and it does not get weakened for our
#: convenience. It carries a carve-out for files whose SCHEMA belongs to somebody else, and
#: this schema is ours, so the fix is to rename our own fields rather than to claim the
#: exemption. The upstream rows keep their own names; only what we write changes.
PROBE_QUESTION_KEY = "problem"
PROBE_REFERENCE_KEY = "reference"

#: What the bundled probe is and what using it obliges. MIT is permissive and still requires the
#: notice to travel, which is why it is stated rather than assumed to be common knowledge.
PROBE_SOURCE = "GSM8K (Cobbe et al. 2021), openai/gsm8k, MIT licence"

#: Held in a mutable holder rather than a bare global, matching `bundled.py`, so tests can reset
#: it without reaching into module internals.
_probe_state = {"notified": False}


def probe_path():
    """Where the bundled probe is, whether or not it is there."""
    import pathlib
    return pathlib.Path(__file__).resolve().parent / "data" / PROBE_NAME


def probe_is_available():
    return probe_path().is_file()


def probe_notice(log=print, used=None):
    """Say what the bundled probe is and where it came from. Once per process.

    Somebody who installs a package and runs a default has not read a dataset card. The bundled
    track prints the same kind of notice for the same reason; the licence here is permissive
    rather than restrictive, which changes what the notice asks of them and not whether it is
    owed.

    `used` is how many items this run will actually score. It is reported because the first
    version said "256 bundled items" on a run using 200 of them, and a number in a log is a
    number somebody quotes. The bundle is deliberately larger than any sensible sample, so the
    two will usually differ.
    """
    if _probe_state["notified"]:
        return
    _probe_state["notified"] = True
    have = len(load_probe())
    scope = f"{used} of {have}" if used is not None and used != have else f"{have}"
    log(f"Capability probe: {scope} bundled items from {PROBE_SOURCE}.")
    log('  Attribution travels with it: THIRD-PARTY-NOTICES.md, under "The bundled capability '
        'probe".')


def load_probe(limit=None):
    """The bundled probe as (question, answer) pairs, in file order.

    Raises rather than returning an empty list when the file is missing. An empty probe would make
    every candidate score identically and the run would report that as no capability cost, which
    is the most flattering possible reading of an absent measurement.
    """
    import json
    path = probe_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"the bundled capability probe is not in this install ({path}). A build from a clone "
            f"carries it, so this is an install that lost it rather than a clone; rebuild it with "
            f"tools/packaging/build_capability_probe.py, or pass --capability-eval to name your "
            f"own benchmark")
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pairs.append((row[PROBE_QUESTION_KEY], row[PROBE_REFERENCE_KEY]))
    return pairs if limit is None else pairs[:limit]


# AT THE END OF THE FILE, AND IT HAS TO BE. This block used to sit directly under `main`, with the
# bundled probe's loader a hundred lines below it, and `python -m senbonzakura.capability` therefore
# ran `main` before `load_probe` had been defined. Every invocation of the default `--eval bundled`
# over that module path died with a bare `NameError: name 'load_probe' is not defined`, which names
# neither the command nor the cause. The console script was fine, because importing a module runs
# all of it before anything calls `main`, so the two documented ways to reach the same command did
# not behave the same way.
#
# `tests/test_main_block_is_last.py` holds the shape so it cannot come back.
if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(main)
