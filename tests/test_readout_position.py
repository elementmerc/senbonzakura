# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Is the position the compass reads from the position the verdict lives at?

THE DEFECT THIS DESCENDS FROM

The compass takes the difference of two logits at the position where a verdict would begin. That
is a verdict only if a verdict is what the model would put there. Three of the seven models in the
published table are thinking models, and on those the first emitted token is a reasoning opener,
so the "verdict logits" described a token the model was never going to produce. Measured before
the renderer was shared: the most likely token there was a verdict for **0.0%** of prompts and the
two verdict sets held about zero probability, on Qwen3-1.7B and Qwen3-0.6B both.

`readout()` already reports that a position is suspect. Saying a number is suspect is not the same
as saying what it should have been, and the exit gate offers two ways out: move the read-out past
the reasoning preamble, or measure BOTH and publish the disagreement. This is the second, which is
strictly more informative because it keeps the old number visible beside the new one.

WHAT IS ACTUALLY UNDER TEST HERE

The arithmetic and the refusals, on the CPU, with no model. The generation half needs weights and
is exercised separately; everything that decides WHETHER a second position exists, WHICH token
ends the preamble, and WHAT the two positions say about each other is pure and lives here.

The case that matters most is a prompt whose reasoning block never closes. Scoring it at whatever
token the budget stopped on would measure the budget rather than the model, which is the same
defect the refusal length sweep exists to catch, so it is indeterminate and counted separately.
"""
from __future__ import annotations

import pytest

from senbonzakura import margin

HARM, BEN, OTHER = 10, 11, 99


def row(m, argmax=HARM, ph=0.4, pb=0.3, tokens=12):
    return {"margin": m, "canonical": m, "tokens": tokens, "argmax": argmax,
            "p_harmful": ph, "p_benign": pb}


class _Tok:
    """A tokenizer stub that owns a vocabulary, which is what the lookup reads."""

    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


# ── finding the close token, and refusing to invent one ───────────────────────────
@pytest.mark.parametrize("spelling", margin.PREAMBLE_CLOSE_SPELLINGS)
def test_every_declared_close_spelling_is_found(spelling):
    assert margin.preamble_close_id(_Tok({spelling: 7, "a": 1})) == 7


def test_a_tokenizer_with_no_reasoning_block_returns_none():
    """None means "this model has no reasoning block I can find", which is the honest answer."""
    assert margin.preamble_close_id(_Tok({"a": 1, "b": 2})) is None


def test_the_close_token_is_looked_up_in_the_vocabulary_not_encoded():
    """THE TRAP. A tokenizer without `</think>` will happily ENCODE it as four ordinary pieces,
    and scoring after the last of those is scoring after a token the model never emits as a unit.
    """
    class _Encodes:
        def get_vocab(self):
            return {"a": 1}

        def encode(self, _s, **_k):        # would succeed, and would be wrong
            return [3, 4, 5, 6]

    assert margin.preamble_close_id(_Encodes()) is None


def test_a_tokenizer_that_cannot_produce_a_vocabulary_is_survivable():
    class _Broken:
        def get_vocab(self):
            raise ValueError("no vocab here")

    assert margin.preamble_close_id(_Broken()) is None


def test_a_tokenizer_with_no_vocab_method_at_all_is_survivable():
    assert margin.preamble_close_id(object()) is None


# ── the disagreement between the two positions ────────────────────────────────────
def test_two_positions_that_agree_report_full_agreement():
    first = [row(1.0), row(-2.0), row(3.0)]
    past = [row(0.5), row(-0.5), row(2.0)]
    d = margin.readout_disagreement(first, past, [HARM, BEN])
    assert d["compared_on"] == 3
    assert d["indeterminate"] == 0
    assert d["verdict_sign_agreement"] == 1.0


def test_a_flipped_verdict_shows_up_as_disagreement():
    """The number that matters: the two positions call the prompt differently."""
    first = [row(1.0), row(1.0), row(1.0), row(1.0)]
    past = [row(-1.0), row(-1.0), row(1.0), row(1.0)]
    d = margin.readout_disagreement(first, past, [HARM, BEN])
    assert d["verdict_sign_agreement"] == 0.5


def test_an_unclosed_preamble_is_indeterminate_not_agreement():
    """THE CASE THIS MODULE EXISTS TO GET RIGHT.

    A prompt whose reasoning block did not close inside the budget has no second reading. Counting
    it as agreement would let a budget that was too short read as two positions agreeing.
    """
    first = [row(1.0), row(1.0), row(1.0)]
    past = [row(1.0), None, None]
    d = margin.readout_disagreement(first, past, [HARM, BEN])
    assert d["compared_on"] == 1
    assert d["indeterminate"] == 2
    assert d["verdict_sign_agreement"] == 1.0, "the one comparable prompt did agree"


def test_every_prompt_indeterminate_is_reported_as_such_not_as_a_number():
    first = [row(1.0), row(1.0)]
    d = margin.readout_disagreement(first, [None, None], [HARM, BEN])
    assert d["compared_on"] == 0
    assert d["indeterminate"] == 2
    assert "closed inside the budget" in d["why"]
    assert "verdict_sign_agreement" not in d, "a rate over nothing is not a rate"


def test_the_disagreement_carries_both_positions_own_diagnostics():
    """A reader has to be able to see WHY one position is better, not just that it differs."""
    first = [row(1.0, argmax=OTHER, ph=0.001, pb=0.001)]
    past = [row(1.0, argmax=HARM, ph=0.6, pb=0.2)]
    d = margin.readout_disagreement(first, past, [HARM, BEN])
    assert d["argmax_is_verdict_first"] == 0.0
    assert d["argmax_is_verdict_past"] == 1.0
    assert d["mean_verdict_mass_first"] < 0.01
    assert d["mean_verdict_mass_past"] > 0.5


def test_no_rows_gives_no_disagreement():
    assert margin.readout_disagreement([], [], [HARM, BEN]) is None
    assert margin.readout_disagreement([row(1.0)], [], [HARM, BEN]) is None


# ── the sentence a reader gets ────────────────────────────────────────────────────
def _stats(suspect, mass=0.0001, agree=0.0):
    return {"suspect": suspect, "verdict_prob_mass_mean": mass, "argmax_is_verdict": agree}


def test_a_suspect_position_with_no_second_one_says_the_arm_is_unvalidated():
    """The worst case, and it must not be quiet: nothing to compare against and a number that
    describes a token the model was never going to emit.
    """
    r = margin.readout_reading(_stats(True), None, None)
    assert "does NOT hold a verdict" in r
    assert "no second position" in r
    assert "unvalidated" in r


def test_a_suspect_position_with_a_second_one_names_which_figure_to_quote():
    first = [row(1.0, argmax=OTHER, ph=0.0001, pb=0.0001)]
    past = [row(1.0, argmax=HARM, ph=0.6, pb=0.2)]
    d = margin.readout_disagreement(first, past, [HARM, BEN])
    r = margin.readout_reading(_stats(True), _stats(False, 0.8, 1.0), d)
    assert "Quote the past-preamble figure" in r
    assert "say which one it is" in r


def test_a_healthy_position_measured_twice_just_reports_the_agreement():
    first = [row(1.0), row(-1.0)]
    past = [row(1.0), row(-1.0)]
    d = margin.readout_disagreement(first, past, [HARM, BEN])
    r = margin.readout_reading(_stats(False, 0.8, 1.0), _stats(False, 0.8, 1.0), d)
    assert "agree on the direction" in r
    assert "Quote" not in r


def test_a_healthy_position_measured_once_says_nothing():
    """A line that always prints is a line nobody reads."""
    assert margin.readout_reading(_stats(False, 0.8, 1.0), None, None) is None


# ── the flags ─────────────────────────────────────────────────────────────────────
def test_the_second_position_is_not_the_default():
    """It costs a bounded generation per prompt, where the first position costs one forward."""
    a = margin.build_parser().parse_args(
        ["--model", "m", "--harmful", "h", "--harmless", "l", "--out", "o"])
    assert a.readout == "first"
    assert a.preamble_budget == margin.PREAMBLE_BUDGET


def test_the_budget_is_adjustable():
    a = margin.build_parser().parse_args(
        ["--model", "m", "--harmful", "h", "--harmless", "l", "--out", "o",
         "--readout", "both", "--preamble-budget", "64"])
    assert a.readout == "both"
    assert a.preamble_budget == 64


def test_the_flag_help_says_when_to_reach_for_it():
    action, = [x for x in margin.build_parser()._actions if x.dest == "readout"]
    assert "suspect" in action.help


# ── the generation half: finding the position, and refusing to invent one ─────────
class _StubTok:
    """Enough tokenizer for the past-preamble pass. Renders, encodes, decodes."""

    pad_token_id = 0

    def __init__(self, close=42):
        self.close = close

    def get_vocab(self):
        return {"</think>": self.close}

    def apply_chat_template(self, msgs, **_k):
        return msgs[0]["content"]

    def __call__(self, texts, **_k):
        import torch
        ids = torch.ones(len(texts), 3, dtype=torch.long)
        return _Enc({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, ids, **_k):
        return f"<{list(ids)}>"


class _Enc(dict):
    def to(self, _device):
        return self


class _StubModel:
    """Emits a scripted continuation per row, and fixed logits at whatever position is asked for.

    The logits are keyed by the LENGTH of the sequence handed in, so a test can prove the pass
    read the position it claims to and not one either side of it.
    """

    def __init__(self, continuations, vocab=64):
        self.continuations = continuations
        self.vocab = vocab
        self.seen_lengths = []

    def generate(self, **kw):
        import torch
        prompt = kw["input_ids"]
        width = max(len(c) for c in self.continuations)
        rows = []
        for i in range(prompt.shape[0]):
            c = list(self.continuations[i]) + [0] * (width - len(self.continuations[i]))
            rows.append(torch.cat([prompt[i], torch.tensor(c, dtype=torch.long)]))
        return torch.stack(rows)

    def __call__(self, input_ids=None, attention_mask=None, **_k):
        import torch
        n = int(input_ids.shape[1])
        self.seen_lengths.append(n)
        logits = torch.zeros(1, n, self.vocab)
        # The margin at this position is the sequence length, so a test can read the position back
        # out of the number and catch an off-by-one that a shape check would not.
        logits[0, -1, HARM] = float(n)
        logits[0, -1, BEN] = 0.0
        return _Out(logits)


class _Out:
    def __init__(self, logits):
        self.logits = logits


def test_the_position_read_is_the_one_after_the_close_token():
    """An off-by-one here reads the close token itself, or the token before it, and both produce a
    plausible number from the wrong place. The margin encodes the length so it can be checked.
    """
    tok = _StubTok(close=42)
    # prompt is 3 long; the block closes at index 2 of the continuation, so the scored sequence is
    # 3 + 3 = 6 tokens and the logits that predict the verdict are the ones at the end of it.
    model = _StubModel([[7, 8, 42, 9, 9]])
    got = margin.margins_past_preamble(model, tok, ["p"], [HARM], [BEN], "cpu", 42, batch=1)
    assert got[0]["preamble_tokens"] == 3
    assert got[0]["margin"] == 6.0, "the scored position was not immediately after the close token"


def test_a_block_that_never_closes_is_none_not_a_score():
    """Scoring at whatever token the budget stopped on would measure the budget."""
    tok = _StubTok(close=42)
    model = _StubModel([[7, 8, 9, 9, 9]])          # no 42 anywhere
    got = margin.margins_past_preamble(model, tok, ["p"], [HARM], [BEN], "cpu", 42, batch=1)
    assert got == [None]


def test_the_first_close_wins_when_a_model_opens_a_second_block():
    tok = _StubTok(close=42)
    model = _StubModel([[7, 42, 8, 42, 9]])
    got = margin.margins_past_preamble(model, tok, ["p"], [HARM], [BEN], "cpu", 42, batch=1)
    assert got[0]["preamble_tokens"] == 2


def test_a_mixed_batch_keeps_each_prompt_aligned():
    """One prompt closing and another not must not shift the results of either."""
    tok = _StubTok(close=42)
    model = _StubModel([[7, 42, 0, 0], [7, 8, 9, 9], [42, 0, 0, 0]])
    got = margin.margins_past_preamble(model, tok, ["a", "b", "c"], [HARM], [BEN], "cpu", 42,
                                       batch=3)
    assert len(got) == 3
    assert got[0] is not None
    assert got[1] is None
    assert got[2] is not None
    assert got[2]["preamble_tokens"] == 1


def test_the_second_readout_does_not_attend_to_padding():
    """THE DEFECT IN THE FIGURE THE ARTEFACT TELLS YOU TO QUOTE.

    `enc` is built with `padding=True`, and this project sets `padding_side = "left"` with the
    end-of-sequence token as the pad. The re-forward that reads the position after the reasoning
    block used `torch.ones_like(ids)` as its mask, which marks those pad tokens as real content.
    So every prompt shorter than the longest in its batch was scored on a context beginning with
    a run of end-of-sequence tokens.

    It is also batch-size dependent in the worst possible way: at `--batch 1` there is no padding
    and the figure is correct, and `limits.md` explains compass batch sensitivity as
    floating-point reduction order and tells a reader to check their batch size before filing a
    bug. That explanation would have sent someone straight past this.

    Asserted on the mask the model actually receives, because that is the whole defect. The
    first read-out path was always correct; only this one built its own mask.
    """
    import torch

    from senbonzakura import margin

    seen = {}

    class _Tok:
        pad_token_id = 0
        chat_template = None

        def apply_chat_template(self, msgs, **_kw):
            return msgs[0]["content"]

        def __call__(self, texts, **_kw):
            # Left padding with the pad id, exactly as the real tokeniser is configured.
            widths = [3, 5]
            longest = max(widths)
            ids, mask = [], []
            for w in widths:
                pad = longest - w
                ids.append([self.pad_token_id] * pad + list(range(10, 10 + w)))
                mask.append([0] * pad + [1] * w)
            return _Enc({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)})

    class _Enc(dict):
        def to(self, _device):
            return self

    class _Model:
        def generate(self, input_ids=None, attention_mask=None, **_kw):
            # One new token, which is the close token, so `upto` is 1 for both rows.
            new = torch.full((input_ids.shape[0], 1), 99, dtype=input_ids.dtype)
            return torch.cat([input_ids, new], dim=1)

        def __call__(self, input_ids=None, attention_mask=None, **_kw):
            seen.setdefault("masks", []).append(attention_mask.clone())
            seen.setdefault("ids", []).append(input_ids.clone())
            return type("O", (), {"logits": torch.zeros(1, input_ids.shape[1], 8)})()

    margin.margins_past_preamble(
        _Model(), _Tok(), ["short", "a longer one"], [1], [2], "cpu", close_id=99,
        batch=2, budget=4)

    assert seen.get("masks"), "the re-forward never happened"
    for ids, mask in zip(seen["ids"], seen["masks"], strict=True):
        assert ids.shape == mask.shape
        padded = (ids == 0)
        assert not bool((mask.bool() & padded).any()), (
            "a pad token is marked as real content, so the margin was read in a context "
            "beginning with end-of-sequence tokens")
    # The short prompt carries two pads, so its mask must be strictly narrower than its ids.
    narrow = min(int(m.sum()) for m in seen["masks"])
    widest = max(m.shape[1] for m in seen["masks"])
    assert narrow < widest, "no row was actually padded, so this test proved nothing"
