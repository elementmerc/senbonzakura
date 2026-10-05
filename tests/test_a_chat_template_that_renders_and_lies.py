# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The chat template validator: one test per way a template can render and still be wrong.

WHY THE FAKES ARE THE RIGHT SHAPE HERE

The thing under test is not Jinja. It is what comes back out of `apply_chat_template`, which is
the only surface the rest of this package touches and the only place the defects show. So most of
these tokenisers are a few lines that return a string, each one standing for a template defect
seen in the wild: a system message dropped, a generation prompt that opens nothing, markers the
vocabulary has never heard of, a role silently ignored.

Two tests use REAL Jinja rather than a fake, because one defect class lives in the rendering
engine rather than in the template: an environment with autoescaping on rewrites the caller's
quotes and ampersands, and a fake that returns a string by hand could never show it. Those two
need no model and no download.

The validator was also run against every chat template on the development machine, and the
result is recorded in the report rather than here: two of seven findings on real published
tokenisers, both on a model whose markers are genuinely absent from its vocabulary, and no
findings on either model this project measures.
"""
import pytest

from senbonzakura import chattemplate
from senbonzakura.chattemplate import NOTE, REFUSE, SUSPECT


class _Tok:
    """A tokeniser whose rendering is whatever the test says it is."""

    unk_token_id = 0

    def __init__(self, render, *, vocabulary=(), template=None, convert=True):
        self._render = render
        self._vocabulary = set(vocabulary)
        self.chat_template = template
        self._convert = convert

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return self._render(messages, **kwargs)

    def convert_tokens_to_ids(self, token):
        if not self._convert:
            raise RuntimeError("this tokeniser cannot answer")
        return 7 if token in self._vocabulary else self.unk_token_id


#: Every marker the healthy renderer below emits, so the vocabulary check passes by default.
MARKERS = ("<|im_start|>", "<|im_end|>")


def _healthy(messages, add_generation_prompt=False, **_kwargs):
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def _tok(render=_healthy, **kwargs):
    kwargs.setdefault("vocabulary", MARKERS)
    return _Tok(render, **kwargs)


def _codes(record):
    return {(f["code"], f["severity"]) for f in record["findings"]}


# ── a template with nothing wrong with it ────────────────────────────────────────
def test_a_healthy_template_produces_no_findings():
    record = chattemplate.audit(_tok(), source="tokenizer")
    assert record["findings"] == []
    assert record["trustworthy"] is True
    assert record["verdict"] is None
    assert chattemplate.describe(record) == [
        ("chat template: no findings. It renders, keeps every turn, marks the assistant's turn, "
         "and its markers are in the vocabulary.")]


def test_the_record_carries_the_digest_of_the_template_it_audited():
    record = chattemplate.audit(_tok(template="{{ messages }}"), source="bundled:plain")
    assert record["source"] == "bundled:plain"
    assert len(record["sha256"]) == 16
    assert record["rendered_probe"].endswith("<|im_start|>assistant\n")


def test_a_template_held_as_something_other_than_text_has_no_digest():
    assert chattemplate.audit(_tok(template=None))["sha256"] is None


# ── it will not render at all ────────────────────────────────────────────────────
def test_a_template_that_raises_is_refused_and_the_error_is_passed_on():
    def boom(_messages, **_kwargs):
        raise ValueError("Conversation roles must alternate")

    record = chattemplate.audit(_Tok(boom))
    assert _codes(record) == {("renders", REFUSE)}
    assert "Conversation roles must alternate" in record["findings"][0]["what"]
    assert record["trustworthy"] is False
    assert record["rendered_probe"] is None


def test_a_template_that_returns_something_other_than_text_is_refused():
    record = chattemplate.audit(_Tok(lambda _m, **_k: [1, 2, 3]))
    assert _codes(record) == {("renders", REFUSE)}
    assert "returned list rather than text" in record["findings"][0]["what"]


def test_a_template_that_renders_nothing_is_refused():
    record = chattemplate.audit(_Tok(lambda _m, **_k: "   \n  "))
    assert _codes(record) == {("renders", REFUSE)}
    assert "whitespace" in record["findings"][0]["what"]


def test_a_template_that_loses_the_user_is_refused_before_anything_else_is_asked():
    """Nothing further is worth probing once the prompt is not the prompt."""
    record = chattemplate.audit(_Tok(lambda _m, **_k: "<|im_start|>assistant\n"))
    assert _codes(record) == {("user_kept", REFUSE)}
    assert "rate on a different question" in record["findings"][0]["why"]


# ── it renders and the assistant's turn is never opened ──────────────────────────
def test_a_template_that_never_opens_the_assistants_turn_is_refused():
    """Nothing after the last content marks a turn, so the model is never prompted to answer."""
    def continues(messages, **_kwargs):
        return "".join(f"{m['content']}. " for m in messages)

    record = chattemplate.audit(_Tok(continues))
    assert ("generation_prompt", REFUSE) in _codes(record)
    assert any("not told it is its turn" in f["why"] for f in record["findings"])
    assert record["trustworthy"] is False


def test_a_template_that_always_opens_the_turn_is_suspect_rather_than_refused():
    """The two cases this observation covers, and only one of them is fatal.

    A template that ignores the flag and ALWAYS opens the assistant's turn generates perfectly
    well: the model is told it is its turn either way. What it cannot do is render a finished
    conversation. The first version of this check called that fatal, which would have refused a
    template that works.
    """
    record = chattemplate.audit(_tok(lambda messages, **_k: _healthy(messages,
                                                                     add_generation_prompt=True)))
    assert ("generation_prompt", SUSPECT) in _codes(record)
    assert record["trustworthy"] is True
    assert any("Generation is unaffected" in f["why"] for f in record["findings"])


def test_a_template_that_refuses_to_render_without_a_generation_prompt_is_only_suspect():
    def picky(messages, add_generation_prompt=False, **_kwargs):
        if not add_generation_prompt:
            raise ValueError("this template only builds prompts for a new turn")
        return _healthy(messages, add_generation_prompt=True)

    record = chattemplate.audit(_tok(picky))
    assert ("generation_prompt", SUSPECT) in _codes(record)
    assert record["trustworthy"] is True, "a usable template was called untrustworthy"


# ── it renders and the system message vanishes ───────────────────────────────────
def test_a_system_message_that_disappears_is_refused():
    def drops_system(messages, add_generation_prompt=False, **_kwargs):
        kept = [m for m in messages if m["role"] != "system"]
        return _healthy(kept, add_generation_prompt=add_generation_prompt)

    record = chattemplate.audit(_tok(drops_system))
    assert ("system_role", REFUSE) in _codes(record)
    assert any("silently never reaches the model" in f["why"] for f in record["findings"])
    assert record["trustworthy"] is False


def test_a_template_that_refuses_a_system_message_out_loud_is_only_noted():
    """Several published models do this, so it is a condition rather than a defect."""
    def no_system(messages, add_generation_prompt=False, **_kwargs):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return _healthy(messages, add_generation_prompt=add_generation_prompt)

    record = chattemplate.audit(_tok(no_system))
    assert ("system_role", NOTE) in _codes(record)
    assert record["trustworthy"] is True
    assert any("put it in the user turn instead" in f["why"] for f in record["findings"])


# ── it renders and a turn is lost ────────────────────────────────────────────────
def test_a_dropped_assistant_turn_is_refused_and_named():
    def drops_assistant(messages, add_generation_prompt=False, **_kwargs):
        kept = [m for m in messages if m["role"] != "assistant"]
        return _healthy(kept, add_generation_prompt=add_generation_prompt)

    record = chattemplate.audit(_tok(drops_assistant))
    assert ("turns", REFUSE) in _codes(record)
    assert any("the assistant turn" in f["what"] for f in record["findings"])
    assert any("shorter conversation than the one recorded" in f["why"]
               for f in record["findings"])


def test_a_multi_turn_conversation_that_will_not_render_is_refused():
    def single_turn_only(messages, add_generation_prompt=False, **_kwargs):
        if len(messages) > 1:
            raise ValueError("only one turn is supported")
        return _healthy(messages, add_generation_prompt=add_generation_prompt)

    record = chattemplate.audit(_tok(single_turn_only))
    assert ("turns", REFUSE) in _codes(record)
    assert any("Multi-turn measurement is not available" in f["why"] for f in record["findings"])


def test_a_transcript_nobody_can_attribute_is_suspect():
    def unmarked(messages, add_generation_prompt=False, **_kwargs):
        out = "".join(f"{m['content']}\n" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    record = chattemplate.audit(_tok(unmarked))
    assert ("roles_marked", SUSPECT) in _codes(record)
    assert any("no turn boundaries in it" in f["why"] for f in record["findings"])


def test_a_format_that_names_no_speaker_is_not_refused_for_it():
    """A Mistral-style template wraps the user turn and leaves the assistant's bare.

    Half the field renders this way and alternation carries the attribution, so demanding a
    per-speaker name would be refusing a format that works. The first version of the check did
    exactly that, which is why its control is pinned here.
    """
    def instruct(messages, add_generation_prompt=False, **_kwargs):
        out = ""
        for m in messages:
            out += f"[INST] {m['content']} [/INST]" if m["role"] == "user" else m["content"]
        return out

    record = chattemplate.audit(_Tok(instruct, vocabulary=("[INST]", "[/INST]")))
    assert not [f for f in record["findings"] if f["code"] == "roles_marked"]


# ── it renders and the caller's text is altered ──────────────────────────────────
def test_text_rewritten_on_the_way_in_is_refused():
    def escapes(messages, add_generation_prompt=False, **_kwargs):
        fixed = [{"role": m["role"],
                  "content": m["content"].replace("&", "&amp;").replace("<", "&lt;")}
                 for m in messages]
        return _healthy(fixed, add_generation_prompt=add_generation_prompt)

    record = chattemplate.audit(_tok(escapes))
    assert ("content_verbatim", REFUSE) in _codes(record)
    assert any("a rate on text nobody wrote" in f["why"] for f in record["findings"])


def test_a_template_that_cannot_take_punctuation_is_refused():
    def fragile(messages, add_generation_prompt=False, **_kwargs):
        if any("<" in m["content"] for m in messages):
            raise ValueError("unexpected character in content")
        return _healthy(messages, add_generation_prompt=add_generation_prompt)

    record = chattemplate.audit(_tok(fragile))
    assert ("content_verbatim", REFUSE) in _codes(record)
    assert any("cannot score a corpus" in f["why"] for f in record["findings"])


def test_real_jinja_with_autoescaping_on_is_caught():
    """The one defect that lives in the engine rather than in the template.

    No fake could show this: the template is correct, and the environment rewrites the caller's
    characters on the way through it. Uses real Jinja, no model and no download.
    """
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(autoescape=True)  # the defect under test
    template = env.from_string(
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

    def render(messages, add_generation_prompt=False, **_kwargs):
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)

    record = chattemplate.audit(_tok(render))
    assert ("content_verbatim", REFUSE) in _codes(record)


def test_the_same_template_without_autoescaping_passes():
    """The control on the test above: the only difference is the environment."""
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(autoescape=False)  # the correct setting
    template = env.from_string(
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

    def render(messages, add_generation_prompt=False, **_kwargs):
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)

    assert chattemplate.audit(_tok(render))["findings"] == []


# ── it renders markers the model has never seen ──────────────────────────────────
def test_markers_outside_the_vocabulary_are_refused():
    record = chattemplate.audit(_tok(vocabulary=()))
    assert ("markers_in_vocabulary", REFUSE) in _codes(record)
    finding = next(f for f in record["findings"] if f["code"] == "markers_in_vocabulary")
    assert "<|im_start|>" in finding["what"]
    assert "ordinary letters" in finding["why"]


def test_a_marker_the_tokeniser_maps_to_nothing_counts_as_absent():
    class _NoneTok(_Tok):
        def convert_tokens_to_ids(self, token):
            return None

    record = chattemplate.audit(_NoneTok(_healthy))
    assert ("markers_in_vocabulary", REFUSE) in _codes(record)


def test_a_tokeniser_that_cannot_answer_is_recorded_as_skipped_rather_than_passed():
    record = chattemplate.audit(_tok(convert=False))
    assert ("markers_in_vocabulary", NOTE) in _codes(record)
    assert any("skipped rather than passed" in f["why"] for f in record["findings"])
    assert record["trustworthy"] is True


def test_a_tokeniser_with_no_lookup_at_all_is_recorded_as_skipped():
    class _Bare:
        chat_template = None

        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            return _healthy(messages, **kwargs)

    record = chattemplate.audit(_Bare())
    assert ("markers_in_vocabulary", NOTE) in _codes(record)


def test_a_prompt_with_no_markers_in_it_asks_nothing_of_the_vocabulary():
    def bare(messages, add_generation_prompt=False, **_kwargs):
        out = "".join(f"{m['role'].upper()}: {m['content']}\n" for m in messages)
        return out + ("ASSISTANT: " if add_generation_prompt else "")

    record = chattemplate.audit(_tok(bare, vocabulary=(), convert=False))
    assert not [f for f in record["findings"] if f["code"] == "markers_in_vocabulary"]


def test_a_vocabulary_without_an_unknown_token_still_accepts_a_known_marker():
    class _NoUnk(_Tok):
        unk_token_id = None

    record = chattemplate.audit(_NoUnk(_healthy, vocabulary=MARKERS))
    assert not [f for f in record["findings"] if f["code"] == "markers_in_vocabulary"]


# ── it renders differently twice, or carries a date ──────────────────────────────
def test_a_rendering_that_changes_between_calls_is_refused():
    state = {"n": 0}

    def drifting(messages, add_generation_prompt=False, **_kwargs):
        state["n"] += 1
        return _healthy(messages, add_generation_prompt=add_generation_prompt) + str(state["n"])

    record = chattemplate.audit(_tok(drifting))
    assert ("deterministic", REFUSE) in _codes(record)
    assert any("two different measurements" in f["why"] for f in record["findings"])


def test_a_template_carrying_todays_date_is_a_condition_rather_than_a_defect():
    record = chattemplate.audit(_tok(template="{{ strftime_now('%d %b %Y') }}{{ messages }}"))
    assert ("dated_template", SUSPECT) in _codes(record)
    assert record["trustworthy"] is True
    assert any("different prompts" in f["why"] for f in record["findings"])


# ── it opens a reasoning block where the compass reads its verdict ───────────────
def test_a_generation_prompt_that_opens_a_reasoning_block_is_suspect():
    def thinking(messages, add_generation_prompt=False, **_kwargs):
        out = _healthy(messages, add_generation_prompt=add_generation_prompt)
        return out + "<think>\n" if add_generation_prompt else out

    record = chattemplate.audit(_tok(thinking, vocabulary=(*MARKERS, "<think>")))
    assert ("thinking_opener", SUSPECT) in _codes(record)
    finding = next(f for f in record["findings"] if f["code"] == "thinking_opener")
    assert "Refusal scoring is unaffected" in finding["why"]
    assert "0.0%" in finding["why"], "the measured consequence is not quoted"
    assert record["trustworthy"] is True


def test_a_reasoning_marker_far_from_the_end_is_not_the_compass_problem():
    """The compass reads the position the prompt ENDS on, so only the tail matters."""
    def closed(messages, add_generation_prompt=False, **_kwargs):
        body = "<think>a long earlier reasoning block</think>" + "x" * 200
        return body + _healthy(messages, add_generation_prompt=add_generation_prompt)

    record = chattemplate.audit(_tok(closed, vocabulary=(*MARKERS, "<think>", "</think>")))
    assert not [f for f in record["findings"] if f["code"] == "thinking_opener"]


# ── two renderers on one conversation ────────────────────────────────────────────
def test_two_renderers_that_agree_exactly_produce_no_finding():
    assert chattemplate.compare_renderings("<|im_start|>user\nhi", "<|im_start|>user\nhi") is None


def test_two_renderers_that_differ_name_the_character_and_show_both_sides():
    finding = chattemplate.compare_renderings(
        "<|im_start|>user\nhi<|im_end|>", "<|im_start|>user\nhi<|eot_id|>",
        left_label="transformers", right_label="the GGUF")
    assert finding.severity == REFUSE
    assert finding.code == "renderers_agree"
    assert "transformers and the GGUF render" in finding.what
    assert "character 21" in finding.what
    assert "prompt formats rather than between runners" in finding.why


def test_one_rendering_being_a_prefix_of_the_other_is_still_a_difference():
    finding = chattemplate.compare_renderings("<|im_start|>user\nhi", "<|im_start|>user\nhi\n")
    assert finding is not None
    assert "character 19" in finding.what


# ── how a finding reads ──────────────────────────────────────────────────────────
def test_the_worst_severity_is_what_a_caller_gates_on():
    assert chattemplate.worst([]) is None
    assert chattemplate.worst([{"severity": NOTE}, {"severity": REFUSE}, {"severity": SUSPECT}]) \
        == REFUSE
    assert chattemplate.worst([chattemplate.Finding("a", NOTE, "w", "y")]) == NOTE
    assert chattemplate.worst([{"severity": "something new"}]) == "something new"


def test_the_description_leads_with_the_finding_that_invalidates_the_number():
    record = chattemplate.audit(_tok(vocabulary=(), template="{{ strftime_now('%Y') }}"))
    lines = chattemplate.describe(record)
    assert "NOT TRUSTWORTHY" in lines[0]
    assert "markers_in_vocabulary" in lines[1], "a note was printed above a refusal"


def test_a_template_with_only_conditions_on_it_does_not_read_as_broken():
    record = chattemplate.audit(_tok(template="{{ strftime_now('%Y') }}"))
    lines = chattemplate.describe(record)
    assert "usable, with conditions" in lines[0]
    assert "NOT TRUSTWORTHY" not in lines[0]


def test_a_template_that_starts_failing_part_way_through_the_audit_is_not_called_stable():
    """An intermittent template renders for the early probes and raises for a later one.

    The determinism check has to survive that rather than crash on it, and it must not report
    "renders identically twice" about a conversation that did not render at all.
    """
    state = {"n": 0}

    def flaky(messages, add_generation_prompt=False, **_kwargs):
        state["n"] += 1
        if state["n"] >= 6:
            raise ValueError("the template gave up")
        return _healthy(messages, add_generation_prompt=add_generation_prompt)

    record = chattemplate.audit(_tok(flaky))
    assert not [f for f in record["findings"] if f["code"] == "deterministic"], (
        "a conversation that never rendered was reported as rendering stably")


# ── the enforcement half, which the loader does not yet use ──────────────────────
def test_a_clean_record_passes_the_refusal_gate():
    chattemplate.refuse_if_untrustworthy(chattemplate.audit(_tok()))


def test_the_refusal_gate_names_every_fatal_finding_and_the_way_out():
    record = chattemplate.audit(_tok(vocabulary=()))
    with pytest.raises(SystemExit) as e:
        chattemplate.refuse_if_untrustworthy(record, how_to_fix="Supply one with --chat-template.")
    said = str(e.value)
    assert "markers_in_vocabulary" in said
    assert "ordinary letters" in said
    assert "Supply one with --chat-template." in said
    assert "produces an ordinary looking number from a prompt nobody asked for" in said


def test_the_refusal_gate_leaves_an_absent_template_to_the_loader():
    """`renders` has a longer and better message of its own one layer up."""
    chattemplate.refuse_if_untrustworthy(
        {"findings": [{"code": "renders", "severity": REFUSE, "what": "w", "why": "y"}]})


def test_the_refusal_gate_on_a_record_with_no_findings_key():
    chattemplate.refuse_if_untrustworthy({})
