# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The wavefront executor against the reference executor, which must agree bitwise on every number.

The tiny models are real transformers models on CPU in float32, of three kinds: plain attention (Llama),
hybrid linear and full attention (Qwen3.5, whose two kinds of block are handed different masks) and a
mixture of experts (Qwen3 MoE). The comparison is `torch.equal`, not a tolerance, and the micro batches
have different padding so a stale captured argument would show.
"""
import json
import types

import pytest
import torch
from torch import nn

from senbonzakura import attribution as A
from senbonzakura import attribution_hf as H
from senbonzakura import attribution_wavefront as W

#: Different lengths, so the sorted micro batches of three carry different padding.
PROMPTS = [f"request number {i} {'x' * (i * 3 % 11)}" for i in range(11)]
#: Every shape of condition the map asks for, and an odd mix.
CONDITIONS = [(), (0,), (2,), (3,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (1, 2, 3), (2, 3), (0, 2)]
KINDS = ["llama", "qwen35", "moe"]


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


def tiny(kind, seed=0):
    import transformers
    torch.manual_seed(seed)
    if kind == "llama":
        cfg = transformers.LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                                       num_attention_heads=2, num_key_value_heads=2, vocab_size=64)
        return transformers.LlamaForCausalLM(cfg).eval()
    if kind == "qwen35":
        cfg = transformers.Qwen3_5TextConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=2, head_dim=16, vocab_size=64, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
        return transformers.Qwen3_5ForCausalLM(cfg).eval()
    cfg = transformers.Qwen3MoeConfig(
        hidden_size=16, intermediate_size=32, moe_intermediate_size=8, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, vocab_size=64, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=1, mlp_only_layers=[])
    return transformers.Qwen3MoeForCausalLM(cfg).eval()


def width(model):
    return model.config.get_text_config().hidden_size


def subject(model, executor=H.EXECUTOR_REFERENCE, log=None, **kwargs):
    s = H.HFSubject(model, CharTok(), batch=3, log=log or (lambda *_: None), executor=executor, **kwargs)
    s.refusal_ids = torch.tensor([3, 5])
    s.comply_ids = torch.tensor([7, 9])
    return s


def direction(model, seed=1):
    gen = torch.Generator().manual_seed(seed)
    d = torch.randn(width(model), generator=gen)
    return d / d.norm()


def reference_rows(model, d, standing=None, conditions=CONDITIONS, prompts=PROMPTS):
    ref = subject(model)
    if standing is not None:
        ref.edit_standing(d, standing)
    return torch.stack([ref.readout(prompts, d, b) for b in conditions])


# ------------------------------------------------------------------------------ bitwise against the reference

@pytest.mark.parametrize("executor", [H.EXECUTOR_WAVEFRONT, H.EXECUTOR_RESIDENT])
@pytest.mark.parametrize("standing", [None, (2,), (0, 3)])
@pytest.mark.parametrize("kind", KINDS)
def test_the_wavefront_gives_the_reference_numbers_bit_for_bit(kind, standing, executor):
    model = tiny(kind)
    d = direction(model)
    want = reference_rows(model, d, standing)
    wave = subject(model, executor)
    if standing is not None:
        wave.edit_standing(d, standing)
    assert torch.equal(wave.readouts(PROMPTS, d, CONDITIONS), want)
    check = wave.executor_summary()["self_check"]
    assert check["passed"] is True and check["max_abs_diff"] == 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_one_condition_at_a_time_through_readout_gives_the_same_numbers(kind):
    model = tiny(kind)
    d = direction(model)
    want = reference_rows(model, d)
    wave = subject(model, H.EXECUTOR_WAVEFRONT)
    for i, blocks in enumerate(CONDITIONS):
        assert torch.equal(wave.readout(PROMPTS, d, blocks), want[i]), blocks


@pytest.mark.parametrize("kind", KINDS)
def test_the_blocks_and_the_weights_are_exactly_as_they_were_after_a_run(kind):
    model = tiny(kind)
    d = direction(model)
    layers = list(model.model.layers)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    params = {id(p): p for p in model.parameters()}
    subject(model, H.EXECUTOR_WAVEFRONT).readouts(PROMPTS, d, CONDITIONS)
    assert all(a is b for a, b in zip(layers, model.model.layers, strict=True))
    assert all(params[id(p)] is p for p in model.parameters())
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())


def test_a_hybrid_model_hands_its_two_kinds_of_block_different_masks():
    model = tiny("qwen35")
    ex = subject(model, H.EXECUTOR_WAVEFRONT)
    enc = ex._encode(PROMPTS[:3])
    caps = ex._wavefront_executor()._capture(enc)
    masks = [caps[j][1]["attention_mask"].dim() for j in range(4)]
    assert masks == [2, 2, 2, 4]


def test_the_arguments_are_captured_per_micro_batch_so_changing_padding_cannot_go_stale():
    model = tiny("llama")
    ex = subject(model, H.EXECUTOR_WAVEFRONT)._wavefront_executor()
    short, long = subject(model)._encode(["ab", "abc", "abcd"]), subject(model)._encode(["x" * 9, "y" * 12, "z" * 7])
    a, b = ex._capture(short), ex._capture(long)
    assert a[0][1]["attention_mask"].shape[-1] == 4 and b[0][1]["attention_mask"].shape[-1] == 12


def test_a_tight_state_budget_splits_the_walk_into_groups_and_changes_no_number():
    model = tiny("llama")
    d = direction(model)
    want = reference_rows(model, d)
    roomy = subject(model, H.EXECUTOR_WAVEFRONT, wavefront_state_bytes=10 ** 9)
    assert torch.equal(roomy.readouts(PROMPTS, d, CONDITIONS), want)
    # One micro batch of the largest size and not a byte more: no two of them can share a walk.
    ex = roomy._wavefront_executor()
    encs = [roomy._encode(c) for c in roomy._batches(roomy._by_length(PROMPTS)[1])]
    live = len([c for c in CONDITIONS if c])
    tight_bytes = max(ex._state_bytes(ex._capture(e)) for e in encs) * (live + 1)
    tight = subject(model, H.EXECUTOR_WAVEFRONT, wavefront_state_bytes=tight_bytes)
    assert torch.equal(tight.readouts(PROMPTS, d, CONDITIONS), want)
    assert tight.executor_summary()["sweeps"] > roomy.executor_summary()["sweeps"] >= 2


def test_groups_are_made_in_order_and_each_stays_inside_the_budget():
    model = tiny("llama")
    ex = subject(model, H.EXECUTOR_WAVEFRONT, wavefront_state_bytes=100)._wavefront_executor()
    caps = [{"x0": torch.zeros(n)} for n in (1, 1, 2, 1, 1)]
    ex.state_bytes = 16
    assert ex._groups(caps, 1) == [[0, 1], [2], [3, 4]]
    ex.state_bytes = 10 ** 9
    assert ex._groups(caps, 1) == [[0, 1, 2, 3, 4]]


def test_a_budget_too_small_for_one_micro_batch_stops_before_any_walk():
    model = tiny("llama")
    wave = subject(model, H.EXECUTOR_WAVEFRONT, wavefront_state_bytes=1)
    with pytest.raises(A.AttributionError, match=r"needs .* GiB of state, over the .* the wavefront may use"):
        wave.readouts(PROMPTS, direction(model), CONDITIONS)
    assert wave.executor_summary()["sweeps"] == 0


def test_the_host_memory_free_caps_the_state_budget():
    model = tiny("llama")
    wave = subject(model, H.EXECUTOR_WAVEFRONT, spill_guard=lambda: 100)
    with pytest.raises(A.AttributionError, match="may use here"):
        wave.readouts(PROMPTS, direction(model), CONDITIONS)
    unknown = subject(model, H.EXECUTOR_WAVEFRONT, spill_guard=lambda: None)
    assert unknown.readouts(PROMPTS, direction(model), [(), (1,)]).shape == (2, len(PROMPTS))


# --------------------------------------------------------------------------------- the self check stops a run

def test_a_difference_from_the_reference_stops_the_run_and_puts_everything_back():
    model = tiny("llama")
    d = direction(model)
    layers = list(model.model.layers)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    wave = subject(model, H.EXECUTOR_WAVEFRONT)
    ex = wave._wavefront_executor()
    real = ex._call

    def perturbed(j, cap, x, watch):
        out = real(j, cap, x, watch)
        return out + 1e-4 if j == 2 else out
    ex._call = perturbed
    said = r"different numbers from the reference path.*rerun with --executor reference"
    with pytest.raises(A.AttributionError, match=said):
        wave.readouts(PROMPTS, d, CONDITIONS)
    assert ex.check["passed"] is False and ex.check["logits_that_differ"] > 0
    assert all(a is b for a, b in zip(layers, model.model.layers, strict=True))
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())
    ex._call = real
    with pytest.raises(A.AttributionError, match="already failed its self check"):
        wave.readouts(PROMPTS, d, CONDITIONS)


def test_the_self_check_runs_once_and_covers_a_bounded_spread_of_conditions():
    model = tiny("llama")
    d = direction(model)
    calls = []
    wave = subject(model, H.EXECUTOR_WAVEFRONT)
    real = wave._full
    wave._full = lambda enc, dirn, blocks: calls.append(blocks) or real(enc, dirn, blocks)
    wave._wavefront = None
    wave.readouts(PROMPTS, d, CONDITIONS)
    first = list(calls)
    assert 0 < len(first) <= W.CHECK_CONDITIONS and () in first
    wave.readouts(PROMPTS, d, [(1,), (0, 1, 2)])
    assert calls == first


def test_the_first_micro_batch_gets_a_walk_to_itself_so_a_disagreement_stops_early():
    assert W.WavefrontExecutor._with_first_alone([[0, 1, 2], [3, 4]]) == [[0], [1, 2], [3, 4]]
    assert W.WavefrontExecutor._with_first_alone([[0], [1]]) == [[0], [1]]


def test_the_check_conditions_are_the_same_every_time_and_spread_early_to_late():
    conds = [(), *[(k,) for k in range(20)], *[tuple(range(k + 1)) for k in range(1, 20)]]
    picked = W.pick_check_conditions(conds)
    assert picked == W.pick_check_conditions(list(reversed(conds)))
    assert len(picked) == W.CHECK_CONDITIONS and () in picked
    assert min(min(c) for c in picked if c) == 0 and max(min(c) for c in picked if c) >= 15
    assert W.pick_check_conditions([(1,), (1,), ()]) == [(), (1,)]


# ----------------------------------------------------------------------------------------- refusals, by name

@pytest.mark.parametrize(("kwargs", "words"), [
    ({"past_key_values": object()}, "handed a cache in `past_key_values`"),
    ({"cache_params": [1]}, "handed a cache in `cache_params`"),
    ({"use_cache": True}, "asked to keep a cache"),
    ({"state": object()}, "handed `state`, which is a object"),
    ({"nested": ({"deeper": [object()]},)}, "handed `nested`"),
])
def test_a_block_handed_state_from_outside_is_refused_by_name(kwargs, words):
    with pytest.raises(A.AttributionError, match=r"block 3 \(Probe\).*" + words.replace("`", "`")):
        W.vet_arguments("block 3 (Probe)", (), kwargs)


def test_plain_data_and_tensors_in_any_nesting_are_accepted():
    W.vet_arguments("block 0 (Probe)", (torch.zeros(1),), {
        "position_embeddings": (torch.zeros(1), torch.zeros(1)), "use_cache": False, "past_key_values": None,
        "flags": {"a": [1, 2.0, "x", None, torch.Size([1]), torch.float32]}})


def test_an_argument_nested_deeper_than_the_cap_is_refused_rather_than_trusted():
    deep = [[[[[object()]]]]]
    with pytest.raises(A.AttributionError, match="handed `x`"):
        W.vet_arguments("block 0 (Probe)", (), {"x": deep})


def test_a_positional_argument_that_is_not_plain_data_is_refused():
    with pytest.raises(A.AttributionError, match="handed argument 1"):
        W.vet_arguments("block 0 (Probe)", (torch.zeros(1), object()), {})


class Wrap(nn.Module):
    """A block with something else done to it, to see what the walk makes of it."""

    def __init__(self, inner, mode):
        super().__init__()
        self.inner, self.mode = inner, mode
        self.register_buffer("counter", torch.zeros(1))

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if self.mode == "extra":
            return out, torch.zeros(1)
        if self.mode == "count":
            self.counter.add_(1)
        if self.mode == "shape":
            return out[:, :-1]
        return out


def wrapped(mode, at=1):
    model = tiny("llama")
    model.model.layers[at] = Wrap(model.model.layers[at], mode)
    return model


def walk_one(model, watch=False):
    ex = subject(model, H.EXECUTOR_RESIDENT)._wavefront_executor()
    enc = subject(model)._encode(PROMPTS[:3])
    cap = ex._capture(enc)
    return ex._call(1, cap[1], cap["x0"], watch)


def test_a_block_that_hands_on_more_than_its_stream_is_refused_by_name():
    with pytest.raises(A.AttributionError, match=r"block 1 \(Wrap\) hands on more than its stream"):
        walk_one(wrapped("extra"))


def test_a_block_that_changes_a_buffer_while_it_runs_is_refused_by_name():
    with pytest.raises(A.AttributionError, match=r"block 1 \(Wrap\) changed one of its own weights or buffers"):
        walk_one(wrapped("count"), watch=True)
    assert torch.equal(walk_one(wrapped("count"), watch=False), walk_one(wrapped("none")))


def test_a_block_whose_shapes_are_not_frozen_is_refused_by_name():
    with pytest.raises(A.AttributionError, match=r"block 1 \(Wrap\) returned a stream of shape"):
        walk_one(wrapped("shape"))


def test_a_block_with_state_changing_buffers_is_refused_on_a_real_run_too():
    model = wrapped("count")
    wave = subject(model, H.EXECUTOR_RESIDENT)
    with pytest.raises(A.AttributionError, match=r"block 1 \(Wrap\) changed one of its own"):
        wave.readouts(PROMPTS, direction(model), [(), (1,)])


def test_a_block_offloaded_by_the_loader_is_refused_by_name():
    model = tiny("llama")
    model.model.layers[2]._hf_hook = object()
    with pytest.raises(A.AttributionError, match=r"block 2 \(LlamaDecoderLayer\) is offloaded by the loader"):
        subject(model, H.EXECUTOR_WAVEFRONT).readouts(PROMPTS, direction(model), [()])


def test_a_block_with_unloaded_weights_is_refused_by_name():
    model = tiny("llama")
    model.model.layers[3].to("meta")
    with pytest.raises(A.AttributionError, match=r"block 3 \(LlamaDecoderLayer\) has weights that aren't loaded"):
        subject(model, H.EXECUTOR_WAVEFRONT).readouts(PROMPTS, direction(model), [()])


def test_two_blocks_sharing_a_weight_are_refused_by_name():
    model = tiny("llama")
    model.model.layers[1].input_layernorm.weight = model.model.layers[0].input_layernorm.weight
    with pytest.raises(A.AttributionError, match=r"block 1 \(LlamaDecoderLayer\) shares a weight with block 0"):
        subject(model, H.EXECUTOR_WAVEFRONT).readouts(PROMPTS, direction(model), [()])


def test_a_weight_shared_with_something_outside_the_blocks_is_refused_by_name():
    model = tiny("llama")
    model.model.norm.weight = model.model.layers[0].input_layernorm.weight
    with pytest.raises(A.AttributionError, match=r"outside the blocks is shared with block 0"):
        subject(model, H.EXECUTOR_WAVEFRONT).readouts(PROMPTS, direction(model), [()])


def test_a_condition_naming_a_block_that_does_not_exist_is_refused_before_the_walk():
    model = tiny("llama")
    wave = subject(model, H.EXECUTOR_WAVEFRONT)
    with pytest.raises(A.AttributionError, match=r"block\(s\) \[9\] do not exist: this model has blocks 0 to 3"):
        wave.readouts(PROMPTS, direction(model), [(9,)])
    assert wave.executor_summary()["sweeps"] == 0


def test_a_condition_with_no_direction_given_is_refused():
    model = tiny("llama")
    with pytest.raises(A.AttributionError, match="no direction was given"):
        subject(model, H.EXECUTOR_WAVEFRONT).readouts(PROMPTS, None, [(1,)])


def test_a_direction_that_points_nowhere_is_refused_before_the_walk():
    model = tiny("llama")
    with pytest.raises(A.AttributionError, match="points nowhere"):
        subject(model, H.EXECUTOR_WAVEFRONT).readouts(PROMPTS, torch.zeros(16), [(1,)])


def test_a_block_that_never_runs_in_the_capture_pass_is_refused_by_name(monkeypatch):
    model = tiny("llama")
    ex = subject(model, H.EXECUTOR_WAVEFRONT)._wavefront_executor()
    enc = subject(model)._encode(PROMPTS[:3])
    monkeypatch.setattr(W, "last_token_logits", lambda *_a, **_k: None)
    with pytest.raises(A.AttributionError, match=r"block 0 \(LlamaDecoderLayer\) never ran in the capture pass"):
        ex._capture(enc)
    assert all(isinstance(layer, type(model.model.layers[0])) for layer in model.model.layers)


def test_a_block_called_with_no_hidden_states_is_refused():
    stand = W._Capture(nn.Identity(), {}, 5)
    with pytest.raises(A.AttributionError, match="block 5 was called with no hidden states"):
        stand()


def test_a_block_called_by_name_is_walked_by_name():
    record = {}
    stand = W._Capture(nn.Identity(), record, 0)
    x = torch.ones(1, 2, 3)
    assert stand(hidden_states=x, mask=1) is x
    assert record[0] == ((), {"mask": 1}, True) and torch.equal(record["x0"], x)


def test_a_stand_in_answers_for_the_attributes_of_the_block_it_replaces():
    class Block(nn.Module):
        layer_type = "linear_attention"
    stand = W._Capture(Block(), {}, 0)
    assert stand.layer_type == "linear_attention"
    with pytest.raises(AttributeError):
        getattr(stand, "nothing_like_this")  # noqa: B009 - the lookup itself is under test
    bare = W._Stand.__new__(W._Stand)
    nn.Module.__init__(bare)
    with pytest.raises(AttributeError):
        getattr(bare, "anything")  # noqa: B009 - a stand-in with no block behind it has nothing to fall back to


@pytest.mark.parametrize(("out", "stream"), [
    (torch.zeros(2), True), ((torch.zeros(2),), True), ((torch.zeros(2), "x"), True),
    ((torch.zeros(2), torch.zeros(2)), False)])
def test_only_a_stream_alone_can_be_carried(out, stream):
    assert W.stream_only(out) is stream


# ----------------------------------------------------------------------- the window, the copying and the timeline

def test_at_most_two_blocks_are_ever_on_the_card_and_the_next_copy_starts_before_the_current_block_runs():
    model = tiny("llama")
    d = direction(model)
    wave = subject(model, H.EXECUTOR_WAVEFRONT)
    ex = wave._wavefront_executor()
    ex.check = {"passed": True}
    events = []
    window = ex.window
    for name in ("prefetch", "release"):
        real = getattr(window, name)
        setattr(window, name, lambda j, real=real, name=name: (events.append((name, j)), real(j))[1])
    for j, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(
            lambda _m, _a, j=j: events.append(("compute", j)) if events[-1:] != [("compute", j)] else None)
    wave.readouts(PROMPTS, d, [(), (1,), (0, 1)])
    assert window.peak_resident == 2
    first = [events.index(("compute", j)) for j in range(4)]
    for j in range(3):
        assert events.index(("prefetch", j + 1)) < first[j] < events.index(("release", j))
    assert events.index(("prefetch", 0)) < first[0]


def test_resident_blocks_are_never_copied():
    model = tiny("llama")
    wave = subject(model, H.EXECUTOR_RESIDENT)
    wave.readouts(PROMPTS, direction(model), [(), (1,)])
    summary = wave.executor_summary()
    assert summary["streaming"] is False and summary["peak_blocks_resident"] == 0 and summary["copy_s"] == 0.0


def test_each_block_logs_its_copy_and_compute_seconds_and_how_much_was_hidden():
    model = tiny("llama")
    lines = []
    wave = subject(model, H.EXECUTOR_WAVEFRONT, log=lines.append)
    wave.readouts(PROMPTS, direction(model), [(), (1,)])
    per_block = [line for line in lines if "wavefront block" in line]
    sweeps = wave.executor_summary()["sweeps"]
    assert len(per_block) == 4 * sweeps
    assert all("copy" in line and "compute" in line and "overlap" in line for line in per_block)
    assert any("self check passed" in line for line in lines)
    blocks = wave.executor_summary()["blocks"]
    assert len(blocks) == 4 and all(b["compute_s"] > 0 and b["copy_s"] >= 0 for b in blocks)
    json.dumps(wave.executor_summary())


def test_resident_blocks_log_nothing_per_block():
    model = tiny("llama")
    lines = []
    subject(model, H.EXECUTOR_RESIDENT, log=lines.append).readouts(PROMPTS, direction(model), [(), (1,)])
    assert not [line for line in lines if "wavefront block" in line]


def test_the_timeline_counts_only_the_copy_that_ran_inside_the_previous_blocks_compute():
    assert W.overlap_seconds((1.0, 3.0), (2.0, 5.0)) == 1.0
    assert W.overlap_seconds((1.0, 2.0), (2.0, 5.0)) == 0.0
    assert W.overlap_seconds((0.0, 9.0), (2.0, 5.0)) == 3.0
    assert W.overlap_seconds((6.0, 7.0), (2.0, 5.0)) == 0.0


def test_the_account_turns_marks_into_copy_compute_and_overlap_per_block():
    model = tiny("llama")
    ex = subject(model, H.EXECUTOR_WAVEFRONT)._wavefront_executor()
    marks = {("copy", 0, "start"): 0.0, ("copy", 0, "end"): 1.0,
             ("compute", 0, "start"): 1.0, ("compute", 0, "end"): 4.0,
             ("copy", 1, "start"): 1.0, ("copy", 1, "end"): 3.0,
             ("compute", 1, "start"): 4.0, ("compute", 1, "end"): 5.0,
             ("copy", 2, "start"): 4.0, ("copy", 2, "end"): 5.5,
             ("compute", 2, "start"): 5.5, ("compute", 2, "end"): 6.0,
             ("compute", 3, "start"): 6.0, ("compute", 3, "end"): 7.0}
    lines = []
    ex.log = lines.append
    ex._account(marks, 4)
    got = [(b["copy_s"], b["compute_s"], b["overlap_s"]) for b in ex.blocks]
    assert got == [(1.0, 3.0, 0.0), (2.0, 1.0, 2.0), (1.5, 0.5, 1.0), (0.0, 1.0, 0.0)]
    assert "100% of the copy hidden" in lines[1] and "67% of the copy hidden" in lines[2]


def test_the_summary_adds_up_across_sweeps():
    model = tiny("llama")
    wave = subject(model, H.EXECUTOR_WAVEFRONT, wavefront_state_bytes=60_000)
    wave.readouts(PROMPTS, direction(model), CONDITIONS)
    s = wave.executor_summary()
    assert s["sweeps"] >= 2
    assert s["compute_s"] == pytest.approx(sum(b["compute_s"] for b in s["blocks"]))


def test_parking_that_fails_halfway_puts_every_block_back(monkeypatch):
    model = tiny("llama")
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    ex = subject(model, H.EXECUTOR_WAVEFRONT)._wavefront_executor()
    real = W._put
    seen = {"n": 0}

    def flaky(owner, name, is_param, tensor):
        seen["n"] += 1
        if seen["n"] == 9:
            raise RuntimeError("out of pinned memory")
        return real(owner, name, is_param, tensor)
    monkeypatch.setattr(W, "_put", flaky)
    with pytest.raises(RuntimeError, match="out of pinned memory"):
        ex.window.park()
    monkeypatch.setattr(W, "_put", real)
    assert ex.window.masters is None
    assert all(torch.equal(saved[k], v) for k, v in model.state_dict().items())


def test_a_second_park_is_a_no_op_and_a_resident_window_never_parks():
    model = tiny("llama")
    ex = subject(model, H.EXECUTOR_WAVEFRONT)._wavefront_executor()
    ex.window.park()
    masters = ex.window.masters
    ex.window.park()
    assert ex.window.masters is masters
    ex.window.unpark()
    ex.window.unpark()
    resident = subject(model, H.EXECUTOR_RESIDENT)._wavefront_executor()
    resident.window.park()
    assert resident.window.masters is None
    resident.window.prefetch(0)
    resident.window.ready(0)
    resident.window.release(0)


def test_an_empty_list_of_micro_batches_walks_nothing():
    model = tiny("llama")
    ex = subject(model, H.EXECUTOR_RESIDENT)._wavefront_executor()
    assert ex.run([], direction(model), [(), (1,)], torch.tensor([3])) == {(): [], (1,): []}


def test_a_zero_state_budget_is_refused():
    model = tiny("llama")
    with pytest.raises(A.AttributionError, match="state budget is 0 bytes"):
        W.WavefrontExecutor(model, model.model.layers, standing=None, reference=None, state_bytes=0)


def test_describing_a_condition_in_words():
    assert W.describe(()) == "no blocks" and W.describe((4,)) == "block 4"
    assert W.describe((1, 2)) == "blocks 1, 2" and W.describe(tuple(range(9))) == "9 blocks from 0 to 8"


# ------------------------------------------------------------------------------- the subject and the pipeline

def test_the_executor_names_are_checked_and_the_reference_is_the_default():
    assert H.EXECUTORS == ("reference", "wavefront", "wavefront-resident")
    model = tiny("llama")
    assert H.HFSubject(model, CharTok()).executor == H.EXECUTOR_REFERENCE
    with pytest.raises(A.AttributionError, match="not an executor"):
        H.HFSubject(model, CharTok(), executor="turbo")
    s = subject(model)
    s.use_executor(H.EXECUTOR_WAVEFRONT)
    assert s.executor == H.EXECUTOR_WAVEFRONT and s.executor_summary() is None


def test_the_reference_executor_answers_many_conditions_exactly_as_one_by_one():
    model = tiny("llama")
    d = direction(model)
    s = subject(model)
    assert torch.equal(s.readouts(PROMPTS, d, CONDITIONS), reference_rows(model, d))
    assert s.executor_summary() is None


def test_readouts_before_prepare_fail_loudly_under_every_executor():
    for executor in H.EXECUTORS:
        s = H.HFSubject(tiny("llama"), CharTok(), executor=executor)
        with pytest.raises(A.AttributionError, match="prepare"):
            s.readouts(PROMPTS, None, [()])
        with pytest.raises(A.AttributionError, match="prepare"):
            s.readout(PROMPTS)


def test_the_arms_share_one_walk_and_the_second_arm_computes_nothing():
    model = tiny("llama")
    d = direction(model)
    one, two = subject(model, H.EXECUTOR_RESIDENT), subject(model, H.EXECUTOR_RESIDENT)
    two.refusal_ids, two.comply_ids = torch.tensor([5, 11]), torch.tensor([7, 13])
    shared = H.share_readouts([one, two])
    first = one.readouts(PROMPTS, d, CONDITIONS)
    second = two.readouts(PROMPTS, d, CONDITIONS)
    assert shared.misses == len({tuple(c) for c in CONDITIONS}) and shared.hits == shared.misses
    assert two.executor_summary() is None
    alone = subject(model)
    alone.refusal_ids, alone.comply_ids = two.refusal_ids, two.comply_ids
    assert torch.equal(second, torch.stack([alone.readout(PROMPTS, d, b) for b in CONDITIONS]))
    assert torch.equal(first, reference_rows(model, d))


def test_a_subject_under_a_standing_edit_never_reads_or_writes_the_shared_walk():
    model = tiny("llama")
    d = direction(model)
    plain, edited = subject(model, H.EXECUTOR_RESIDENT), subject(model, H.EXECUTOR_RESIDENT)
    shared = H.share_readouts([plain, edited])
    edited.edit_standing(d, (1,))
    got = edited.readouts(PROMPTS, d, [(), (2,)])
    assert shared.misses == 0 and torch.equal(got, reference_rows(model, d, (1,), [(), (2,)]))
    plain_rows = plain.readouts(PROMPTS, d, [(), (2,)])
    assert not torch.equal(plain_rows, got) and shared.misses == 2


def test_changing_the_standing_edit_forgets_what_the_subject_had_walked():
    model = tiny("llama")
    d = direction(model)
    s = subject(model, H.EXECUTOR_RESIDENT)
    before = s.readouts(PROMPTS, d, [(), (2,)])
    s.edit_standing(d, (0,))
    after = s.readouts(PROMPTS, d, [(), (2,)])
    assert not torch.equal(before, after)
    assert torch.equal(after, reference_rows(model, d, (0,), [(), (2,)]))


def test_a_repeated_question_is_answered_from_what_was_walked():
    model = tiny("llama")
    d = direction(model)
    s = subject(model, H.EXECUTOR_RESIDENT)
    s.readouts(PROMPTS, d, CONDITIONS)
    sweeps = s.executor_summary()["sweeps"]
    s.readouts(PROMPTS, d, CONDITIONS)
    assert s.executor_summary()["sweeps"] == sweeps


def test_a_readout_position_past_an_opening_is_walked_with_the_opening_fed_in():
    model = tiny("llama")
    d = direction(model)
    ref, wave = subject(model), subject(model, H.EXECUTOR_RESIDENT)
    ref.opener = wave.opener = (4, 6)
    want = torch.stack([ref.readout(PROMPTS, d, b) for b in CONDITIONS])
    assert torch.equal(wave.readouts(PROMPTS, d, CONDITIONS), want)


def test_the_whole_attribution_is_identical_under_the_wavefront_on_the_stub_model():
    from tests.test_attribution_hf import FAST, StubModel, StubTok, _parts
    parts = _parts()

    def run(executor):
        s = H.HFSubject(StubModel(), StubTok(), batch=8, log=lambda *_: None, executor=executor)
        return A.attribute(s, parts, seed=0, conv=FAST), s
    plain, _ = run(H.EXECUTOR_REFERENCE)
    for executor in (H.EXECUTOR_WAVEFRONT, H.EXECUTOR_RESIDENT):
        wave, s = run(executor)
        assert json.dumps(wave, sort_keys=True, default=str) == json.dumps(plain, sort_keys=True, default=str)
        assert s.executor_summary()["self_check"]["passed"] is True


def test_the_reference_path_is_untouched_by_the_readouts_hook():
    from tests.test_attribution_hf import FAST, StubModel, StubTok, _parts
    parts = _parts()
    one = A.attribute(H.HFSubject(StubModel(), StubTok(), batch=8, log=lambda *_: None), parts, seed=0, conv=FAST)
    two = A.attribute(H.HFSubject(StubModel(), StubTok(), batch=8, log=lambda *_: None), parts, seed=0, conv=FAST)
    assert json.dumps(one, sort_keys=True, default=str) == json.dumps(two, sort_keys=True, default=str)


def test_a_subject_without_a_batched_readout_is_asked_one_condition_at_a_time():
    asked = []

    class Plain:
        def readout(self, prompts, direction, blocks=()):
            asked.append(blocks)
            return torch.zeros(len(prompts), dtype=torch.float64)
    arm = types.SimpleNamespace(prompts=["a", "b"])
    A._readouts(Plain(), arm, None, {"none": (), "single:0": (0,)})
    assert asked == [(), (0,)]


def test_a_block_with_buffers_of_its_own_is_streamed_and_put_back_exactly():
    model = wrapped("none")
    d = direction(model)
    want = reference_rows(model, d)
    saved = model.model.layers[1].counter.clone()
    got = subject(model, H.EXECUTOR_WAVEFRONT).readouts(PROMPTS, d, CONDITIONS)
    assert torch.equal(got, want) and torch.equal(model.model.layers[1].counter, saved)


def test_a_tensor_a_block_holds_under_two_names_is_one_slot():
    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.zeros(2)
            self.register_buffer("a", shared)
            self.register_buffer("b", shared)
            self.w = nn.Parameter(torch.zeros(2))
    assert len(W._slots(Twice())) == 2


def test_asking_for_the_same_condition_twice_walks_it_once():
    model = tiny("llama")
    d = direction(model)
    wave = subject(model, H.EXECUTOR_RESIDENT)
    got = wave.readouts(PROMPTS, d, [(1,), (1,), ()])
    assert torch.equal(got[0], got[1]) and torch.equal(got, reference_rows(model, d, None, [(1,), (1,), ()]))


def test_no_micro_batches_make_no_groups():
    ex = subject(tiny("llama"), H.EXECUTOR_RESIDENT)._wavefront_executor()
    assert ex._groups([], 3) == []
