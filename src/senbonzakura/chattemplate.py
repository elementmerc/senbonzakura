# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""What a chat template actually did to a conversation, checked rather than assumed.

WHY THIS EXISTS

Two of the three states a chat template can be in were already handled. An ABSENT template is
refused by the loader, which will not invent a prompt format because every number here moves with
it. A template LOST during quantisation is reported by `quantise`, which can prove the input had
one and the output does not. The third state was unguarded: a template that is **present and
wrong**.

That is the worst shape of defect this project has, and it has met it before in another costume.
A prompt renderer that drifted produced a published table and a passing build, because a wrong
prompt format does not crash. It renders. The model answers. Every rate, every divergence figure
and every compass reading comes out looking ordinary and means something else. One prospect's
public complaint about a shipped model is, in full, "Jinja Script not good", and nothing in this
tool could have told them whether that was true.

WHAT A VERDICT LOOKS LIKE, AND WHY IT IS NOT A BOOLEAN

A template is not valid or invalid. It can render perfectly and drop the system message. It can
keep every message and never mark whose turn it is. It can be correct today and contain today's
date, which makes two runs a week apart two different prompts. So this reports FINDINGS, each
naming what was observed, what it does to a measurement, and how much it matters:

    refuse   a measurement taken through this template is not trustworthy
    suspect  it changes what a number means, and a reader has to be told
    note     worth knowing, and it does not undermine the figure

"This renders but the system message vanished" is the useful output. "Invalid" is not.

HOW IT CHECKS

By rendering probe conversations and reading what comes back, never by parsing the Jinja. A
template is a program; the only honest question about a program is what it does when it runs, and
the attribute holding its source has moved and been deprecated across transformers versions while
`apply_chat_template`'s contract has not. Parsing the source would also miss the whole class of
defect that lives in the tokeniser rather than the template: a marker the template emits that the
vocabulary does not contain renders as literal letters, and no amount of reading the Jinja shows
that.

Nothing here imports torch or transformers. The tokeniser arrives as an argument, and the only
thing asked of it is `apply_chat_template`, which is the contract the rest of this package
already depends on.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

#: A measurement taken through this template cannot be trusted.
REFUSE = "refuse"
#: The number still means something, and not what a reader would assume.
SUSPECT = "suspect"
#: Worth recording, and it does not undermine the figure.
NOTE = "note"

_RANK = {NOTE: 0, SUSPECT: 1, REFUSE: 2}

#: The probe texts. Distinctive rather than natural, because the check is whether each one comes
#: back, and a probe that reads like ordinary prose can match something the template emits itself.
SYSTEM_TEXT = "SYSPROBE-be-careful"
USER_TEXT = "USERPROBE-first-question"
ASSISTANT_TEXT = "ASSISTANTPROBE-the-answer"
SECOND_USER_TEXT = "USERPROBE-second-question"

#: Characters a Jinja environment with autoescaping on would rewrite. If this comes back changed,
#: the prompt the model reads is not the prompt the caller supplied, and every reply is a reply to
#: something else.
MARKUP_TEXT = "MARKUPPROBE a & b < c > d \"quoted\" it's"

#: Spans in a rendered prompt that are meant to be single tokens. Matched broadly on purpose: a
#: marker this does not recognise is simply not checked, whereas a pattern that misses
#: `<|im_start|>` would pass a template whose markers the vocabulary has never seen.
_MARKER = re.compile(r"<\|[^<>|]{1,64}\|>|<\/?[A-Za-z_][A-Za-z0-9_\-]{0,32}>|\[/?[A-Z]{2,16}\]")

#: Words a template uses to say whose turn it is. Only used to decide whether SOMETHING marks a
#: turn boundary, never to require a particular vocabulary.
_ROLE_WORD = re.compile(r"\b(user|assistant|system|human|model|ai)\b", re.IGNORECASE)

#: Markers that open a model's private reasoning. The compass reads its verdict at the position
#: the generation prompt ends on, and when that position is a reasoning opener the two verdict
#: sets hold almost no probability there: measured at 0.0% of prompts on two Qwen3 sizes before
#: `firsttoken.render_chat` was shared. So a template that appends one is not broken; it changes
#: where a verdict can be read from, and the reader has to be told.
_THINKING = re.compile(r"<\|?/?(think|thinking|reasoning|analysis)\|?>", re.IGNORECASE)

#: Jinja calls that put the current date into the rendered prompt. Llama 3's own template does
#: this. It is not a defect, and it does mean two runs on different days rendered different
#: prompts, which matters to a project that compares a measurement with one taken weeks earlier.
_DATED = re.compile(r"strftime_now|datetime\.now|\bnow\(\)")


@dataclass(frozen=True)
class Finding:
    """One thing observed about the template, and what it does to a measurement."""

    code: str
    severity: str
    what: str
    why: str


def _render(tok, messages, **kwargs):
    """Render, or hand back the exception rather than a guess at what it meant."""
    try:
        out = tok.apply_chat_template(messages, tokenize=False, **kwargs)
    except Exception as e:
        # Deliberately broad. The template is somebody else's program and a Jinja error arrives
        # as whatever that program raised; every one of them means the same thing here, which is
        # that this conversation did not render.
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(out, str):
        return None, f"apply_chat_template returned {type(out).__name__} rather than text"
    return out, None


def _conversation(*roles):
    return [{"role": r, "content": c} for r, c in roles]


def _known_token(tok, marker):
    """Does the vocabulary hold this marker as one token?

    `convert_tokens_to_ids` is the question that matters, because a marker the vocabulary lacks
    is not an error: it renders as letters, the model reads the letters, and the prompt format it
    was trained on never appears. Returns None when the tokeniser cannot answer, which is not the
    same as a no and must not be reported as one.
    """
    convert = getattr(tok, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        ident = convert(marker)
    except Exception:
        return None
    if ident is None:
        return False
    unk = getattr(tok, "unk_token_id", None)
    if unk is not None and ident == unk:
        return False
    return True


def _check_renders(tok):
    """Nothing else can be asked of a template that will not render one user turn."""
    out, error = _render(tok, _conversation(("user", USER_TEXT)), add_generation_prompt=True)
    if error is not None:
        return None, [Finding(
            "renders", REFUSE,
            f"one user turn did not render: {error}",
            "No prompt can be built for this model, so no measurement through it exists. Supply a "
            "template with --chat-template, or fix the one in tokenizer_config.json.")]
    if not out.strip():
        return None, [Finding(
            "renders", REFUSE, "one user turn rendered as nothing but whitespace",
            "The model would be asked an empty question and its reply scored as an answer.")]
    if USER_TEXT not in out:
        return out, [Finding(
            "user_kept", REFUSE, "the user's own words are not in the rendered prompt",
            "Whatever the model is being asked, it is not the prompt supplied. Every rate "
            "measured through this template is a rate on a different question.")]
    return out, []


def _check_generation_prompt(tok, rendered):
    """Is the model told that it is its turn?"""
    plain, error = _render(tok, _conversation(("user", USER_TEXT)), add_generation_prompt=False)
    if error is not None:
        return [Finding(
            "generation_prompt", SUSPECT,
            f"the template would not render without a generation prompt: {error}",
            "Not fatal, and it means the template cannot be used to score a completed "
            "conversation, only to ask for a new turn.")]
    if plain != rendered:
        return []
    # TWO DIFFERENT TEMPLATES PRODUCE THIS ONE OBSERVATION, and the first version of the check
    # called both of them fatal. A template that NEVER opens the assistant's turn cannot be
    # measured at all. A template that ALWAYS opens it, ignoring the flag, generates perfectly
    # well and merely cannot render a finished conversation. Telling them apart is the tail: if
    # something after the last piece of content marks a turn, the turn is open.
    tail = rendered[rendered.rindex(USER_TEXT) + len(USER_TEXT):] if USER_TEXT in rendered else ""
    if _MARKER.search(tail) or _ROLE_WORD.search(tail):
        return [Finding(
            "generation_prompt", SUSPECT,
            "the template opens the assistant's turn whether or not a generation prompt is asked "
            "for",
            "Generation is unaffected, because the model is told it is its turn either way. What "
            "this template cannot do is render a finished conversation, so anything that scores "
            "an existing transcript rather than asking for a new turn will see an empty "
            "assistant turn appended to it.")]
    return [Finding(
        "generation_prompt", REFUSE,
        "asking for a generation prompt changed nothing, and nothing in the rendered text opens "
        "the assistant's turn",
        "The model is not told it is its turn. It will usually continue the user instead, and a "
        "reply scored that way measures the template rather than the model.")]


def _check_system(tok):
    """A system message either arrives, is refused out loud, or vanishes."""
    out, error = _render(tok, _conversation(("system", SYSTEM_TEXT), ("user", USER_TEXT)),
                         add_generation_prompt=True)
    if error is not None:
        return [Finding(
            "system_role", NOTE,
            f"this template refuses a system message rather than dropping it: {error}",
            "That is a legitimate design and several published models do it. It is recorded "
            "because anything that passes a system prompt here has to put it in the user turn "
            "instead.")]
    if SYSTEM_TEXT not in out:
        return [Finding(
            "system_role", REFUSE,
            "a system message was accepted and does not appear in the rendered prompt",
            "It renders, so nothing fails, and the instruction silently never reaches the model. "
            "This is the exact shape that produces a plausible number from a prompt nobody "
            "asked for.")]
    return []


def _check_turns(tok):
    """Does a multi-turn conversation keep every turn, and can the model tell who spoke?"""
    messages = _conversation(("user", USER_TEXT), ("assistant", ASSISTANT_TEXT),
                             ("user", SECOND_USER_TEXT))
    out, error = _render(tok, messages, add_generation_prompt=True)
    if error is not None:
        return [Finding(
            "turns", REFUSE, f"a three-turn conversation did not render: {error}",
            "Multi-turn measurement is not available for this model, and a single-turn figure "
            "cannot stand in for it, because the whole claim is the difference between them.")]
    missing = [name for name, text in (("the first user turn", USER_TEXT),
                                       ("the assistant turn", ASSISTANT_TEXT),
                                       ("the second user turn", SECOND_USER_TEXT))
               if text not in out]
    if missing:
        return [Finding(
            "turns", REFUSE, f"the template dropped {', '.join(missing)}",
            "A conversation the model never sees in full. The multi-turn escalation rate "
            "measured through it is a rate on a shorter conversation than the one recorded.")]
    # WHAT THIS CHECK IS NOT ALLOWED TO DEMAND, because the first version of it did. It asked for
    # a per-speaker NAME in the rendered text, and several published formats do not have one:
    # a Mistral-style template wraps the user's turn in `[INST] ... [/INST]` and leaves the
    # assistant's bare, so alternation carries the attribution and no role word appears anywhere.
    # Refusing that would be refusing a format half the field uses. What is left is the narrow
    # and real case: nothing at all between one turn and the next.
    between = out[out.index(USER_TEXT) + len(USER_TEXT):out.index(ASSISTANT_TEXT)]
    if not _MARKER.search(between) and not _ROLE_WORD.search(between):
        return [Finding(
            "roles_marked", SUSPECT,
            "nothing separates one turn from the next in the rendered text "
            f"({between!r} sits between them)",
            "The model is handed a transcript with no turn boundaries in it, so a multi-turn "
            "result may be measuring that confusion rather than the escalation it claims to.")]
    return []


def _check_markup(tok):
    """Does the caller's text arrive as the caller wrote it?"""
    out, error = _render(tok, _conversation(("user", MARKUP_TEXT)), add_generation_prompt=True)
    if error is not None:
        return [Finding(
            "content_verbatim", REFUSE,
            f"a prompt containing punctuation and angle brackets did not render: {error}",
            "Harmful prompt corpora are full of quotes and brackets, so a template that cannot "
            "take them cannot score a corpus.")]
    if MARKUP_TEXT not in out:
        return [Finding(
            "content_verbatim", REFUSE,
            "the prompt text was altered on the way into the rendered output",
            "Autoescaping or a filter has rewritten the caller's characters, so the model is "
            "answering something adjacent to the prompt rather than the prompt. A refusal rate "
            "measured this way is a rate on text nobody wrote.")]
    return []


def _check_markers(tok, rendered):
    """Are the template's own markers tokens the model knows, or just letters?

    THE CALLER'S TEXT IS REMOVED FIRST, and this was a real false positive rather than a
    precaution. A template that wraps content in angle brackets rendered the probe as
    `<USERPROBE-first-question>`, the marker pattern matched it, the vocabulary had never heard of
    it, and a healthy tokeniser was refused over a string this module had supplied itself. A
    marker is text the TEMPLATE emitted; anything the caller passed in is not a candidate.
    """
    body = rendered
    for probe in (SYSTEM_TEXT, USER_TEXT, ASSISTANT_TEXT, SECOND_USER_TEXT, MARKUP_TEXT):
        body = body.replace(probe, " ")
    markers, unknown, unanswerable = [], [], False
    for marker in dict.fromkeys(_MARKER.findall(body)):
        markers.append(marker)
        known = _known_token(tok, marker)
        if known is None:
            unanswerable = True
        elif known is False:
            unknown.append(marker)
    if unknown:
        return [Finding(
            "markers_in_vocabulary", REFUSE,
            f"the rendered prompt contains {len(unknown)} marker(s) the vocabulary does not hold: "
            f"{', '.join(sorted(unknown))}",
            "They reach the model as ordinary letters rather than as the control tokens it was "
            "trained on, which is a prompt format the model has never seen. This is what a "
            "mismatched template and tokeniser looks like from the outside, and it does not "
            "raise.")]
    if unanswerable and markers:
        return [Finding(
            "markers_in_vocabulary", NOTE,
            "this tokeniser cannot say whether the template's markers are in its vocabulary",
            "The check was skipped rather than passed. An inspected pass is the only kind worth "
            "recording.")]
    return []


def _check_deterministic(tok):
    """Two renderings of one conversation, and whether the template holds a date."""
    found = []
    first, error = _render(tok, _conversation(("user", USER_TEXT)), add_generation_prompt=True)
    if error is None:
        second, _ = _render(tok, _conversation(("user", USER_TEXT)), add_generation_prompt=True)
        if second != first:
            found.append(Finding(
                "deterministic", REFUSE,
                "the same conversation rendered differently on two consecutive calls",
                "Two runs of one command are then two different measurements, and no figure from "
                "this model can be compared with another."))
    source = getattr(tok, "chat_template", None)
    if isinstance(source, str) and _DATED.search(source):
        found.append(Finding(
            "dated_template", SUSPECT,
            "the template puts the current date into every prompt",
            "Rendering is stable within a day and not across days, so a figure measured today "
            "and one measured last week were measured on different prompts. Published models do "
            "this, so it is not a defect; it is a condition on comparing two numbers."))
    return found


def _check_thinking(tok, rendered):
    """Does the generation prompt end by opening a reasoning block?"""
    tail = rendered[-80:]
    match = _THINKING.search(tail)
    if not match:
        return []
    return [Finding(
        "thinking_opener", SUSPECT,
        f"the generation prompt ends by opening a reasoning block ({match.group(0)!r})",
        "Refusal scoring is unaffected. The compass is: it reads its verdict at the position the "
        "prompt ends on, and on a thinking model that position holds the opener rather than a "
        "verdict. Measured at 0.0% of prompts carrying a verdict there on two model sizes. Render "
        "with thinking disabled, which `firsttoken.render_chat` already does where the model "
        "supports it.")]


def audit(tok, *, source=None):
    """Every finding about this tokeniser's chat template, with the probes that produced them.

    The return value is a record meant to be stamped into an artefact beside a number, because a
    figure whose prompt format nobody checked is a figure with an unexamined dependency, and this
    is the check that was missing.
    """
    rendered, found = _check_renders(tok)
    if rendered is not None and not any(f.severity == REFUSE for f in found):
        found = found + _check_generation_prompt(tok, rendered)
        found = found + _check_system(tok)
        found = found + _check_turns(tok)
        found = found + _check_markup(tok)
        found = found + _check_markers(tok, rendered)
        found = found + _check_deterministic(tok)
        found = found + _check_thinking(tok, rendered)
    template = getattr(tok, "chat_template", None)
    digest = (hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
              if isinstance(template, str) and template else None)
    return {
        "source": source,
        "sha256": digest,
        "verdict": worst(found),
        "trustworthy": not any(f.severity == REFUSE for f in found),
        "findings": [{"code": f.code, "severity": f.severity, "what": f.what, "why": f.why}
                     for f in found],
        "rendered_probe": rendered,
    }


def worst(findings):
    """The highest severity present, or None when a template came through clean."""
    if not findings:
        return None
    return max((f.severity if isinstance(f, Finding) else f["severity"] for f in findings),
               key=lambda s: _RANK.get(s, 0))


def compare_renderings(left, right, *, left_label="transformers", right_label="the GGUF"):
    """Two renderers on one conversation, and where they part company.

    THIS IS THE CHECK THAT MAKES TWO PATHS ONE MEASUREMENT. A figure from a GGUF and a figure
    from a safetensors checkpoint are the same measurement only if the model read the same text,
    and the two formats carry their own copies of the template. Measured on 2026-10-05: for one
    model the llama.cpp server's `/apply-template` and this package's own renderer agreed byte
    for byte on all 40 prompts of a run, which is a result worth having rather than an assumption
    worth making.

    Returns a finding, or None when the two agree exactly.
    """
    if left == right:
        return None
    for i, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            at = i
            break
    else:
        at = min(len(left), len(right))
    window = 40
    return Finding(
        "renderers_agree", REFUSE,
        f"{left_label} and {right_label} render this conversation differently from character "
        f"{at}: {left[max(0, at - window):at + window]!r} against "
        f"{right[max(0, at - window):at + window]!r}",
        "A number from one path cannot be quoted beside a number from the other, because the "
        "two models were asked different things. Whichever template is wrong, the comparison is "
        "between prompt formats rather than between runners.")


def describe(record):
    """The audit in the words somebody debugging a template needs.

    Ordered worst first, because the reader wants the thing that invalidates their number before
    the thing that is merely worth knowing.
    """
    findings = record.get("findings") or []
    if not findings:
        return [("chat template: no findings. It renders, keeps every turn, marks the "
                 "assistant's turn, and its markers are in the vocabulary.")]
    head = ("chat template: NOT TRUSTWORTHY for measurement"
            if not record.get("trustworthy") else "chat template: usable, with conditions")
    lines = [f"{head}  ({len(findings)} finding(s), worst: {record.get('verdict')})"]
    for f in sorted(findings, key=lambda f: -_RANK.get(f["severity"], 0)):
        lines.append(f"  [{f['severity']}] {f['code']}: {f['what']}")
        lines.append(f"      {f['why']}")
    return lines


def refuse_if_untrustworthy(record, *, how_to_fix=""):
    """Stop a measurement that would be taken through a template this found broken.

    Separate from `audit` on purpose. A checker that both decides and enforces gives a caller no
    way to look before it leaps, and the loader's note explains why reporting comes first here.
    This is the enforcement half, for the caller who wants it.

    `renders` is excluded, and only that one: the loader answers an absent or unusable template
    with a longer message naming every way to supply one, and refusing it here as well would mean
    the shorter message wins.
    """
    fatal = [f for f in (record.get("findings") or [])
             if f["severity"] == REFUSE and f["code"] != "renders"]
    if not fatal:
        return
    raise SystemExit(
        "this model's chat template renders, and what it renders cannot be measured:\n"
        + "\n".join(f"  * {f['code']}: {f['what']}\n    {f['why']}" for f in fatal)
        + "\n  A template that is present and wrong does not fail: it produces an ordinary "
          "looking number from a prompt nobody asked for, so the run stops here instead."
        + (f"\n  {how_to_fix}" if how_to_fix else ""))
