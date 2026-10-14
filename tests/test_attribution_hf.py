# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The Hugging Face subject, against a stub model whose behaviour is known by construction.

The stub answers "#" with a refusal and "@" with a compliant opening, so what the readout and the
agreement gate should say is decided here and not by the code being tested.
"""
import dataclasses

import pytest
import torch
from torch import nn

from senbonzakura import attribution as A
from senbonzakura import attribution_hf as H

V, W = 100, 16
REFUSE, COMPLY_A, COMPLY_B = 1, 2, 3
WORDS = {REFUSE: "I cannot help with that.", COMPLY_A: "Sure, here is how.", COMPLY_B: "Of course."}
FAST = dataclasses.replace(A.DEFAULT, resamples=100, shuffled_label_replicates=4,
                           random_direction_replicates=4, split_half_repeats=20)


class Block(nn.Module):
    def __init__(self, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(0.05 * torch.randn(W, W, generator=gen))

    def forward(self, hidden_states):
        # The mean over positions lets the last residual depend on the whole prompt, as attention
        # does, so prompts of one kind are not all the same point.
        return hidden_states + hidden_states @ self.w + 0.3 * hidden_states.mean(1, keepdim=True)


class Inner(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.ModuleList([Block(i) for i in range(n)])


class Cfg:
    _commit_hash = "abc"


class StubModel(nn.Module):
    """Next token is decided by the last token: '#' leads to a refusal, '@' to a compliant one."""

    def __init__(self, n=3, tuple_logits=False):
        super().__init__()
        self.model = Inner(n)
        self.embed = nn.Embedding(V, W)
        self.config = Cfg()
        self.register_buffer("anchor", torch.zeros(1))
        with torch.no_grad():
            self.embed.weight.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(5))
            self.embed.weight[:, :2] = 0.0
            self.embed.weight[ord("#"), 0] = 1.0
            self.embed.weight[ord("@"), 1] = 1.0
        self.head = torch.zeros(W, V)
        self.head[0, REFUSE] = 20.0
        self.head[1, COMPLY_A] = 20.0
        self.head[1, COMPLY_B] = 19.0

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        logits = h @ self.head
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return type("Out", (), {"logits": logits})()

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **_):
        ids = input_ids
        for _ in range(max_new_tokens):
            nxt = self.forward(ids).logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
        return ids


class StubTok:
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **_):
        text = msgs[0]["content"]
        return text + ("#" if "bad" in text else "@")

    def __call__(self, texts, return_tensors=None, padding=True, add_special_tokens=False):
        longest = max(len(t) for t in texts)
        ids = torch.tensor([[0] * (longest - len(t)) + [ord(c) % V for c in t] for t in texts])
        mask = torch.tensor([[0] * (longest - len(t)) + [1] * len(t) for t in texts])
        return type("Enc", (dict,), {"to": lambda self, _d: self})(input_ids=ids, attention_mask=mask)

    def batch_decode(self, new, skip_special_tokens=True):
        return [WORDS.get(int(row[0]), "ok") + " " + " ".join(WORDS.get(int(t), "") for t in row[1:2])
                for row in new]

    def convert_ids_to_tokens(self, ids):
        return [f"tok{i}" for i in ids]


def _parts(n=40):
    def arm(prefix):
        prompts = [f"{prefix} request {i}" for i in range(n)]
        return A.arm(prompts, [f"{prefix}{i}" for i in range(n)])
    return {k: {"harmful": arm("bad"), "harmless": arm("fine")} for k in ("fit", "search", "measure")}


@pytest.fixture
def subject():
    return H.HFSubject(StubModel(), StubTok(), batch=8, log=lambda *_: None)


def test_tokens_that_open_only_refusals_and_only_compliance_form_the_two_sets(subject):
    subject.prepare(_parts())
    assert subject.refusal_ids.tolist() == [REFUSE]
    assert sorted(subject.comply_ids.tolist()) == [COMPLY_A]
    assert subject.agreement_detail["token_sets"]["refusal_token_text"] == [f"tok{REFUSE}"]


def test_the_margin_is_high_where_the_model_refuses_and_low_where_it_complies(subject):
    parts = _parts()
    subject.prepare(parts)
    harmful = subject.readout(parts["measure"]["harmful"].prompts)
    harmless = subject.readout(parts["measure"]["harmless"].prompts)
    assert float(harmful.min()) > float(harmless.max())


def test_a_readout_that_predicts_generation_has_high_agreement(subject):
    parts = _parts()
    subject.prepare(parts)
    value = subject.readout_agreement(parts["search"]["harmful"].prompts,
                                      parts["search"]["harmless"].prompts)
    assert value == pytest.approx(1.0)
    assert subject.agreement_detail["onset_coverage"] == pytest.approx(1.0)
    assert subject.agreement_detail["auc_within_harmful"] is None  # all harmful ones refuse
    split = subject.agreement_detail["by_coverage"]
    assert split["covered"]["auc"] == pytest.approx(1.0)
    assert split["uncovered"] == {"refused": 0, "complied": 0, "auc": None}


def test_replies_whose_opener_is_in_neither_set_are_scored_separately(subject):
    parts = _parts()
    subject.prepare(parts)
    # Remove the compliance token from the set so those replies become uncovered.
    subject.comply_ids = torch.tensor([COMPLY_B])
    subject.readout_agreement(parts["search"]["harmful"].prompts,
                              parts["search"]["harmless"].prompts)
    detail = subject.agreement_detail
    assert detail["onset_coverage"] == pytest.approx(0.5)
    assert detail["by_coverage"]["uncovered"]["complied"] == len(parts["search"]["harmless"].prompts)


def test_the_token_sets_record_how_they_depend_on_the_declared_floor(subject):
    subject.prepare(_parts())
    sets = subject.agreement_detail["token_sets"]
    assert sets["min_token_count"] == H.MIN_TOKEN_COUNT
    assert set(sets["sensitivity_to_min_token_count"]) == {"1", "2", "3"}
    assert "tok1" in sets["refusal_openers_by_count"]


def test_replies_are_generated_once_per_prompt(subject, monkeypatch):
    parts = _parts()
    calls = []
    real = subject._reply
    monkeypatch.setattr(subject, "_reply", lambda chunk: calls.append(len(chunk)) or real(chunk))
    subject.prepare(parts)
    first = sum(calls)
    subject.prepare(parts)
    assert sum(calls) == first


def test_no_token_that_cleanly_separates_the_replies_means_no_readout():
    model = StubModel()
    with torch.no_grad():
        model.head[1, REFUSE] = 25.0       # complying prompts now open with the refusal token too
    subject = H.HFSubject(model, StubTok(), batch=8, log=lambda *_: None)
    subject.prepare(_parts())
    assert subject.refusal_ids is None and "no first token cleanly opens" in subject.readout_why_not
    assert subject.readout_agreement(["bad a"], ["fine b"]) is None
    with pytest.raises(A.AttributionError, match="no first token cleanly opens"):
        subject.readout(["bad a"])


def test_readout_before_prepare_fails_loudly(subject):
    with pytest.raises(A.AttributionError, match="prepare"):
        subject.readout(["bad a"])


def test_too_few_replies_in_a_class_means_the_agreement_is_not_a_measurement(subject):
    parts = _parts()
    subject.prepare(parts)
    assert subject.readout_agreement(parts["search"]["harmful"].prompts[:3],
                                     parts["search"]["harmful"].prompts[3:6]) is None
    assert "fewer than" in subject.agreement_detail["why_not"]


def test_within_harmful_agreement_is_reported_when_both_kinds_of_reply_exist():
    model = StubModel()
    subject = H.HFSubject(model, StubTok(), batch=8, log=lambda *_: None)
    parts = _parts()
    subject.prepare(parts)
    # Prompts containing "bad" normally refuse; give the search arm some that do not.
    odd = [f"quiet {i}" for i in range(10)]
    value = subject.readout_agreement(list(parts["search"]["harmful"].prompts) + odd,
                                      parts["search"]["harmless"].prompts)
    assert value == pytest.approx(1.0)
    assert subject.agreement_detail["auc_within_harmful"] == pytest.approx(1.0)


def test_residuals_come_from_the_blocks_and_include_the_input(subject):
    out = subject.residuals(["bad a", "fine b", "bad c"])
    assert out.shape == (subject.n_blocks + 1, 3, W)
    # The input row is the embedding of the last (rendered) token, before any block.
    assert float(out[0, 0, 0]) == pytest.approx(1.0) and float(out[0, 1, 1]) == pytest.approx(1.0)


def test_ablation_changes_the_margin_and_an_empty_list_leaves_it_alone(subject):
    parts = _parts()
    subject.prepare(parts)
    axis = torch.zeros(W)
    axis[0] = 1.0
    prompts = parts["measure"]["harmful"].prompts
    before = subject.readout(prompts)
    assert torch.equal(subject.readout(prompts, axis, ()), before)
    assert not torch.allclose(subject.readout(prompts, axis, range(subject.n_blocks)), before)


def test_the_whole_pipeline_runs_on_the_stub_and_validates_the_readout_first():
    subject = H.HFSubject(StubModel(), StubTok(), batch=8, log=lambda *_: None)
    result = A.attribute(subject, _parts(), seed=0, conv=FAST)
    assert result["status"] in A.STATUSES
    assert result["controls"]["readout_agreement"] == pytest.approx(1.0)
    assert result["controls"]["readout_agreement_detail"]["token_sets"]


def test_a_model_whose_readout_cannot_be_built_is_reported_readout_invalid_with_the_reason():
    model = StubModel()
    with torch.no_grad():
        model.head[1, REFUSE] = 25.0
    subject = H.HFSubject(model, StubTok(), batch=8, log=lambda *_: None)
    result = A.attribute(subject, _parts(), seed=0, conv=FAST)
    assert result["status"] == A.READOUT_INVALID
    assert "could not be validated" in result["status_reason"]
    assert result["controls"]["readout_agreement_detail"]["why_not"]
    assert "blocks" not in result


def test_a_model_without_a_decoder_stack_is_refused_with_a_reason():
    with pytest.raises(ValueError):
        H.HFSubject(nn.Linear(2, 2), StubTok())


def test_build_partitions_takes_one_prompt_per_request_and_digests_without_storing_prompts(
        monkeypatch, tmp_path):
    from senbonzakura import track
    rows = [f"Explain how to thing{i}" for i in range(12)] + \
           [f"Write a guide to thing{i}" for i in range(12)]
    split = {"fit": rows[:8], "search": rows[8:12], "measure": rows[12:]}
    monkeypatch.setattr(track, "load_partitions", lambda _p: (split, split))
    parts, digests = H.build_partitions(tmp_path, n_measure=5, seed=1)
    assert len(parts["measure"]["harmful"].prompts) == 5
    assert set(digests["measure"]["harmful"]) == {"n", "requests", "rows_in_partition", "digest"}
    again, again_digests = H.build_partitions(tmp_path, n_measure=5, seed=1)
    assert digests == again_digests and parts["fit"]["harmful"].prompts == again["fit"]["harmful"].prompts
    assert all(p not in str(digests) for p in rows)


def test_digest_ignores_order_and_changes_with_content():
    assert H.digest(["a", "b"]) == H.digest(["b", "a"])
    assert H.digest(["a", "b"]) != H.digest(["a", "c"])
