# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The fast sweep against the reference sweep, which must agree bitwise on every number.

The tiny model is a real transformers Llama on CPU in float32, so its forward passes are the same
kind of code the real models run, and the comparison is `torch.equal`, not a tolerance.
"""
import pytest
import torch
from torch import nn

from senbonzakura import attribution as A
from senbonzakura import attribution_hf as H
from senbonzakura import attribution_run as R

PROMPTS = [f"request number {i} {'x' * (i % 4)}" for i in range(7)]
#: Every shape the sweep asks for: unedited, all, singles, fronts short of all, backs, and an odd mix.
CONDITIONS = [(), (0,), (2,), (3,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (1, 2, 3), (2, 3)]


class CharTok:
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **_):
        return msgs[0]["content"]

    def __call__(self, texts, return_tensors=None, padding=True, add_special_tokens=False):
        longest = max(len(t) for t in texts)
        ids = torch.tensor([[0] * (longest - len(t)) + [1 + ord(c) % 60 for c in t] for t in texts])
        mask = torch.tensor([[0] * (longest - len(t)) + [1] * len(t) for t in texts])
        return _Enc(input_ids=ids, attention_mask=mask)

    def batch_decode(self, new, skip_special_tokens=True):
        return ["I cannot help" if int(row[0]) % 2 else "Sure, here" for row in new]

    def convert_ids_to_tokens(self, ids):
        return [f"tok{i}" for i in ids]


class _Enc(dict):
    def to(self, _device):
        return self


def _tiny_llama(seed=0):
    import transformers
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                                   num_attention_heads=2, num_key_value_heads=2, vocab_size=64)
    return transformers.LlamaForCausalLM(cfg).eval()


def _subject(model, sweep, **kwargs):
    s = H.HFSubject(model, CharTok(), batch=3, log=lambda *_: None, sweep=sweep, **kwargs)
    s.refusal_ids = torch.tensor([3, 5])
    s.comply_ids = torch.tensor([7, 9])
    return s


def _direction(seed=1):
    gen = torch.Generator().manual_seed(seed)
    d = torch.randn(16, generator=gen)
    return d / d.norm()


# ------------------------------------------------------------------------ the pure rule

@pytest.mark.parametrize("blocks, expected", [
    ((0,), True), ((0, 1), True), ((0, 1, 2), True),
    ((0, 1, 2, 3), False),           # all blocks: a null replicate asks for it alone
    ((), False), ((1,), False), ((1, 2), False), ((0, 2), False),
])
def test_only_a_front_prefix_short_of_every_block_is_chained(blocks, expected):
    assert H.is_front_prefix(blocks, 4) is expected


def test_the_sweep_names_are_checked_and_the_reference_is_the_default():
    assert H.SWEEPS == ("reference", "fast")
    assert H.HFSubject(_tiny_llama(), CharTok()).sweep == H.SWEEP_REFERENCE
    with pytest.raises(A.AttributionError, match="not a sweep"):
        H.HFSubject(_tiny_llama(), CharTok(), sweep="turbo")


def test_the_flag_reaches_the_parser_and_defaults_to_the_reference():
    base = ["m", "--track", "t", "--out", "o"]
    assert R.build_parser().parse_args(base).fast_sweep is False
    assert R.build_parser().parse_args([*base, "--fast-sweep"]).fast_sweep is True


# --------------------------------------------------------- fast against reference, bitwise

@pytest.mark.parametrize("standing", [None, (2,), (0, 3)])
def test_the_fast_sweep_gives_the_reference_numbers_bit_for_bit_on_a_real_model(standing):
    model = _tiny_llama()
    d = _direction()
    ref, fast = _subject(model, H.SWEEP_REFERENCE), _subject(model, H.SWEEP_FAST)
    if standing is not None:
        ref.edit_standing(d, standing)
        fast.edit_standing(d, standing)
    for blocks in CONDITIONS:
        assert torch.equal(ref.readout(PROMPTS, d, blocks), fast.readout(PROMPTS, d, blocks)), blocks
    assert fast._chain_ok is True
    assert fast.sweep_detail["chains_computed"] == 1
    assert fast.sweep_detail["checked_batches"] == 1
    assert ref.sweep_detail["chains_computed"] == 0


def test_the_reference_sweep_never_builds_a_chain():
    model = _tiny_llama()
    d = _direction()
    ref = _subject(model, H.SWEEP_REFERENCE)
    ref.readout(PROMPTS, d, (0, 1))
    assert ref._front_logits(PROMPTS, d, torch.arange(64), 1) is None
    assert ref._own_chains == {}


def test_all_the_front_conditions_come_from_one_chain():
    model = _tiny_llama()
    d = _direction()
    fast = _subject(model, H.SWEEP_FAST)
    for blocks in [(0,), (0, 1), (0, 1, 2)]:
        fast.readout(PROMPTS, d, blocks)
    assert fast.sweep_detail["chains_computed"] == 1


def test_a_chain_shared_between_arms_is_computed_once_and_read_by_the_other():
    model = _tiny_llama()
    d = _direction()
    a, b = _subject(model, H.SWEEP_FAST), _subject(model, H.SWEEP_FAST)
    H.share_readouts([a, b])
    got_a = [a.readout(PROMPTS, d, blocks) for blocks in [(0,), (0, 1)]]
    got_b = [b.readout(PROMPTS, d, blocks) for blocks in [(0,), (0, 1)]]
    assert b.sweep_detail["chains_computed"] == 0
    assert all(torch.equal(x, y) for x, y in zip(got_a, got_b, strict=True))


def test_a_chain_that_disagrees_with_a_full_pass_switches_itself_off_and_the_reference_answers():
    model = _tiny_llama()
    d = _direction()
    ref, fast = _subject(model, H.SWEEP_REFERENCE), _subject(model, H.SWEEP_FAST)
    real = fast._chain_step

    def perturbed(enc, direction, k, carry):
        logits, out = real(enc, direction, k, carry)
        return (logits + (1e-3 if k == 1 else 0.0)), out
    fast._chain_step = perturbed
    for blocks in CONDITIONS:
        assert torch.equal(ref.readout(PROMPTS, d, blocks), fast.readout(PROMPTS, d, blocks)), blocks
    assert fast._chain_ok is False
    assert fast._own_chains == {}


def test_a_block_that_hands_on_more_than_its_stream_refuses_the_chain(monkeypatch):
    model = _tiny_llama()
    d = _direction()
    ref, fast = _subject(model, H.SWEEP_REFERENCE), _subject(model, H.SWEEP_FAST)
    real = fast._chain_step

    def extra(enc, direction, k, carry):
        logits, out = real(enc, direction, k, carry)
        return logits, (out, torch.zeros(1))
    fast._chain_step = extra
    got = fast.readout(PROMPTS, d, (0, 1))
    assert fast._chain_ok is False
    assert torch.equal(got, ref.readout(PROMPTS, d, (0, 1)))


@pytest.mark.parametrize("out, stream", [
    (torch.zeros(2), True),
    ((torch.zeros(2),), True),
    ((torch.zeros(2), "not a tensor"), True),
    ((torch.zeros(2), torch.zeros(2)), False),
])
def test_only_a_stream_alone_can_be_replayed(out, stream):
    assert H.HFSubject._stream_only(out) is stream


def test_a_block_that_writes_no_tensor_is_refused_loudly():
    with pytest.raises(A.AttributionError, match="did not return a tensor|returned"):
        H.HFSubject._stream_only("nothing")


def test_the_chain_for_one_front_matches_running_that_front_alone():
    model = _tiny_llama()
    d = _direction()
    fast = _subject(model, H.SWEEP_FAST)
    enc = fast._encode(PROMPTS[:3])
    stepped, _ = fast._chain_step(enc, d, 0, None)
    alone = fast._full(enc, d, (0,))
    assert torch.equal(stepped, alone)


# -------------------------------------------------------------------- greedy reply reuse

def _count_replies(monkeypatch):
    calls = []
    real = H.HFSubject._reply

    def counted(self, chunk, direction=None, blocks=()):
        calls.append(len(chunk))
        return real(self, chunk, direction, blocks)
    monkeypatch.setattr(H.HFSubject, "_reply", counted)
    return calls


@pytest.fixture
def short_replies(monkeypatch):
    monkeypatch.setattr(H, "REPLY_TOKENS", 3)


def test_a_second_subject_reuses_the_whole_list_the_first_generated(monkeypatch, short_replies):
    calls = _count_replies(monkeypatch)
    model = _tiny_llama()
    store = {}
    first = _subject(model, H.SWEEP_FAST, reply_store=store)
    second = _subject(model, H.SWEEP_FAST, reply_store=store)
    fresh = _subject(model, H.SWEEP_REFERENCE)
    prompts = PROMPTS
    got_first = first._label(prompts)
    generated = sum(calls)
    assert generated > 0
    got_second = second._label(prompts)
    assert sum(calls) == generated
    assert got_second == got_first == fresh._label(prompts)
    assert second._openings == first._openings == fresh._openings
    assert second.sweep_detail["replies_reused"] == len(prompts)


def test_a_list_that_only_overlaps_a_stored_one_is_generated_afresh(monkeypatch, short_replies):
    calls = _count_replies(monkeypatch)
    model = _tiny_llama()
    store = {}
    _subject(model, H.SWEEP_FAST, reply_store=store)._label(PROMPTS[:3])
    overlap = _subject(model, H.SWEEP_FAST, reply_store=store)
    before = sum(calls)
    overlap._label(PROMPTS)
    assert sum(calls) > before
    assert overlap.sweep_detail.get("replies_reused", 0) == 0
    assert len(store) == 2


def test_a_different_batch_size_or_label_does_not_reuse_the_store(monkeypatch, short_replies):
    calls = _count_replies(monkeypatch)
    model = _tiny_llama()
    store = {}
    _subject(model, H.SWEEP_FAST, reply_store=store)._label(PROMPTS)
    other_batch = H.HFSubject(model, CharTok(), batch=2, log=lambda *_: None, sweep=H.SWEEP_FAST,
                              reply_store=store)
    before = sum(calls)
    other_batch._label(PROMPTS)
    assert sum(calls) > before
    other_label = H.HFSubject(model, CharTok(), batch=3, log=lambda *_: None, sweep=H.SWEEP_FAST,
                              reply_store=store, label=lambda text: "x" in text)
    before = sum(calls)
    other_label._label(PROMPTS)
    assert sum(calls) > before


def test_an_edited_subject_never_reads_or_writes_the_store(monkeypatch, short_replies):
    calls = _count_replies(monkeypatch)
    model = _tiny_llama()
    store = {}
    _subject(model, H.SWEEP_FAST, reply_store=store)._label(PROMPTS)
    edited = _subject(model, H.SWEEP_FAST, reply_store=store)
    edited.edit_standing(_direction(), (1,))
    before = sum(calls)
    edited._label(PROMPTS)
    assert sum(calls) > before
    assert len(store) == 1


def test_the_reference_sweep_takes_no_reply_from_any_store(monkeypatch, short_replies):
    calls = _count_replies(monkeypatch)
    model = _tiny_llama()
    store = {}
    _subject(model, H.SWEEP_FAST, reply_store=store)._label(PROMPTS)
    reference = _subject(model, H.SWEEP_REFERENCE, reply_store=store)
    before = sum(calls)
    reference._label(PROMPTS)
    assert sum(calls) > before
    assert reference._reply_store is None


# ------------------------------------------------------------------ the shared store class

def test_the_shared_store_has_its_own_chain_table_and_the_subjects_see_it():
    model = _tiny_llama()
    a = _subject(model, H.SWEEP_FAST)
    shared = H.share_readouts([a])
    assert isinstance(shared.chains, dict)
    assert a._chains() is shared.chains
    a.edit_standing(_direction(), (0,))
    assert a._chains() is a._own_chains
