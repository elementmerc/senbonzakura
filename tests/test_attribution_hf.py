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
    hidden_size = W


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
        # A floor for the neutral token, so a small leak into the refusal axis cannot win by itself.
        logits[..., 0] = logits[..., 0] + 3.0
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


def _parts(n=40, confirm=False):
    def arm(prefix, tag=""):
        prompts = [f"{prefix} request {i}{tag}" for i in range(n)]
        return A.arm(prompts, [f"{prefix}{i}{tag}" for i in range(n)])
    out = {k: {"harmful": arm("bad"), "harmless": arm("fine")}
           for k in ("fit", "search", "measure")}
    if confirm:
        out["confirm"] = {"harmful": arm("bad", " c"), "harmless": arm("fine", " c")}
    return out


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


# ----------------------------------------------------------------------- the two readout arms

def test_the_disjoint_rule_drops_the_commonest_refusal_opener_if_it_ever_opens_a_compliant_reply():
    refused, complied = {10: 66, 11: 7}, {10: 3, 12: 30}
    ref, com, dropped = H.token_sets(refused, complied, H.ARM_DISJOINT)
    assert ref == [11] and com == [12] and dropped == 1


def test_the_purity_rule_admits_it_at_the_declared_threshold():
    refused, complied = {10: 66, 11: 7}, {10: 3, 12: 30}
    ref, com, dropped = H.token_sets(refused, complied, H.ARM_PURITY)
    assert ref == [10, 11] and com == [12] and dropped == 0
    assert H.PURITY_THRESHOLD == 0.9


def test_the_purity_rule_drops_a_token_below_the_threshold_either_way():
    ref, com, dropped = H.token_sets({1: 8}, {1: 2}, H.ARM_PURITY)       # 0.8 refusals
    assert (ref, com, dropped) == ([], [], 1)
    ref, com, dropped = H.token_sets({1: 9}, {1: 1}, H.ARM_PURITY)       # exactly 0.9
    assert ref == [1] and dropped == 0


def test_tokens_below_the_count_floor_are_ignored_by_both_arms():
    for arm in (H.ARM_DISJOINT, H.ARM_PURITY):
        assert H.token_sets({1: 1}, {2: 1}, arm) == ([], [], 0)


def test_an_unknown_arm_is_refused_by_the_rule_and_the_subject():
    with pytest.raises(A.AttributionError, match="not a readout arm"):
        H.token_sets({}, {}, "clever")
    with pytest.raises(A.AttributionError, match="not a readout arm"):
        H.HFSubject(StubModel(), StubTok(), arm="clever")


def test_the_arms_declare_when_they_were_added_and_the_artefact_carries_it(subject):
    assert H.ARM_DECLARATIONS[H.ARM_DISJOINT]["declared"].startswith("before")
    assert H.ARM_DECLARATIONS[H.ARM_PURITY]["declared"].startswith("after")
    subject.prepare(_parts())
    sets = subject.agreement_detail["token_sets"]
    assert sets["arm"] == H.ARM_DISJOINT and sets["declared"].startswith("before")
    purity = H.HFSubject(StubModel(), StubTok(), batch=8, arm=H.ARM_PURITY, log=lambda *_: None)
    purity.prepare(_parts())
    assert purity.agreement_detail["token_sets"]["declared"].startswith("after")
    assert purity.agreement_detail["token_sets"]["refusal_set_covers_fit_refusals"] == 1.0


def test_uncovered_openers_are_recorded_as_decoded_single_tokens(subject):
    parts = _parts()
    subject.prepare(parts)
    subject.comply_ids = torch.tensor([COMPLY_B])
    subject.readout_agreement(parts["search"]["harmful"].prompts,
                              parts["search"]["harmless"].prompts)
    assert subject.agreement_detail["uncovered_openers"] == {f"tok{COMPLY_A}": 40}


# ------------------------------------------------------------------ the predictive check

class _Wire(Block):
    """Converts the harm marker into a refusal signal on axis 0, so removing it stops refusals."""

    def forward(self, hidden_states):
        out = super().forward(hidden_states)
        out = out.clone()
        out[..., 0] = out[..., 0] + 5.0 * hidden_states[..., 2]
        return out


def _wired_model():
    model = StubModel(n=3)
    with torch.no_grad():
        model.embed.weight[ord("#"), 0] = 0.0
        model.embed.weight[ord("#"), 2] = 1.0
        model.embed.weight[ord("@"), 0] = 0.0
        model.model.layers[1] = _Wire(1)
    return model


def _axis0():
    d = torch.zeros(W)
    d[0] = 1.0
    return d


def test_refusal_flags_follow_the_condition_the_replies_were_written_under():
    subject = H.HFSubject(_wired_model(), StubTok(), batch=8, log=lambda *_: None)
    prompts = [f"bad {i}" for i in range(10)]
    assert all(subject.refusal_flags(prompts))
    assert not any(subject.refusal_flags(prompts, _axis0(), [1]))
    assert all(subject.refusal_flags(prompts, _axis0(), [0, 2]))


def test_the_predictive_check_is_supported_when_the_claimed_block_is_the_one_that_matters():
    subject = H.HFSubject(_wired_model(), StubTok(), batch=8, log=lambda *_: None)
    parts = _parts(confirm=True)
    result = H.predictive_check(subject, parts, [1], _axis0(), [1, 0, 2], seed=0,
                                conv=dataclasses.replace(A.DEFAULT, resamples=200))
    assert result["ran"] and result["supported"] is True
    rates = result["harmful_refusal_rate"]
    assert rates["none"] == 1.0 and rates["claimed"] == 0.0 and rates["random_same_size_mean"] == 1.0
    assert result["contrast"]["ci"][0] > 0
    assert set(result["harmless_refusal_rate"]) == {"none", "claimed", "all_blocks"}


def test_the_predictive_check_is_not_supported_when_the_claim_names_the_wrong_block():
    subject = H.HFSubject(_wired_model(), StubTok(), batch=8, log=lambda *_: None)
    result = H.predictive_check(subject, _parts(confirm=True), [0], _axis0(), [0, 1, 2], seed=0,
                                conv=dataclasses.replace(A.DEFAULT, resamples=200))
    assert result["ran"] and result["supported"] is False


def test_the_predictive_check_says_why_it_did_not_run():
    subject = H.HFSubject(_wired_model(), StubTok(), batch=8, log=lambda *_: None)
    assert "no fresh requests" in H.predictive_check(subject, _parts(), [1], _axis0(), [1])["why"]
    assert "too few others" in H.predictive_check(subject, _parts(confirm=True), [0, 1],
                                                  _axis0(), [0, 1, 2])["why"]
    assert not H.predictive_check(subject, _parts(confirm=True), [], _axis0(), [])["ran"]


def test_confirm_requests_are_disjoint_from_the_measure_requests(monkeypatch, tmp_path):
    from senbonzakura import track
    rows = [f"Explain how to thing{i}" for i in range(60)]
    split = {"fit": rows[:10], "search": rows[10:20], "measure": rows[20:]}
    monkeypatch.setattr(track, "load_partitions", lambda _p: (split, split))
    parts, digests = H.build_partitions(tmp_path, n_measure=15, n_confirm=10, seed=2)
    measure, confirm = parts["measure"]["harmful"], parts["confirm"]["harmful"]
    assert len(confirm.prompts) == 10 and not set(measure.groups) & set(confirm.groups)
    assert "confirm" in digests


# --------------------------------------------------------- a standing edit, and the style control

def test_a_standing_edit_applies_to_every_later_call_and_clears_the_reply_cache(subject):
    parts = _parts()
    subject.prepare(parts)
    before = subject.readout(parts["measure"]["harmful"].prompts)
    subject.edit_standing(_axis0(), range(subject.n_blocks))
    assert not subject._replies
    after = subject.readout(parts["measure"]["harmful"].prompts)
    assert not torch.allclose(before, after)


def test_the_length_direction_control_runs_on_text_and_is_reported_beside_the_total():
    def arm(prefix, base):
        prompts = [f"{prefix} request {i} " + "x" * (i % base) for i in range(40)]
        return A.arm(prompts, [f"{prefix}{i}" for i in range(40)])
    parts = {k: {"harmful": arm("bad", 7), "harmless": arm("fine", 5)}
             for k in ("fit", "search", "measure")}
    result = A.attribute(H.HFSubject(StubModel(), StubTok(), batch=8, log=lambda *_: None),
                         parts, seed=0, conv=FAST)
    style = result["controls"]["style_direction"]
    assert style["ran"] is True and style["flag_ratio"] == A.STYLE_FLAG_RATIO
    assert set(style) >= {"total", "max_single", "flagged"}


# ------------------------------------------------------------- forward passes shared by the arms

def _two_arms():
    out = {}
    for arm in (H.ARM_DISJOINT, H.ARM_PURITY):
        sub = H.HFSubject(StubModel(), StubTok(), batch=8, arm=arm, log=lambda *_: None)
        sub.prepare(_parts())
        out[arm] = sub
    return out


def test_sharing_forward_passes_changes_no_number_and_the_second_arm_reuses_them():
    prompts = _parts()["measure"]["harmful"].prompts
    axis = _axis0()
    alone = {a: s.readout(prompts, axis, [0, 1]) for a, s in _two_arms().items()}
    arms = _two_arms()
    shared = H.share_readouts(arms.values())
    together = {a: s.readout(prompts, axis, [0, 1]) for a, s in arms.items()}
    for arm in arms:
        assert torch.equal(alone[arm], together[arm])
    assert (shared.misses, shared.hits) == (1, 1)


def test_an_arm_asking_a_different_question_misses_instead_of_getting_a_wrong_answer():
    arms = _two_arms()
    shared = H.share_readouts(arms.values())
    prompts = _parts()["measure"]["harmful"].prompts
    arms[H.ARM_DISJOINT].readout(prompts, _axis0(), [0])
    arms[H.ARM_PURITY].readout(prompts, _axis0(), [1])
    arms[H.ARM_PURITY].readout(prompts[:5], _axis0(), [0])
    arms[H.ARM_PURITY].readout(prompts, None, ())
    assert (shared.misses, shared.hits) == (4, 0)


def test_a_subject_with_no_readout_is_left_out_of_the_shared_cache():
    arms = _two_arms()
    arms[H.ARM_PURITY].refusal_ids = None
    shared = H.share_readouts(arms.values())
    assert shared.ids.tolist() == sorted(set(arms[H.ARM_DISJOINT].refusal_ids.tolist()
                                             + arms[H.ARM_DISJOINT].comply_ids.tolist()))
    assert H.share_readouts([arms[H.ARM_PURITY]]) is None
