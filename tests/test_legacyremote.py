# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The adapter that lets a model's own transformers 4 code meet transformers 5.

Every model class here is a tiny stand in written in this file. No model's real code is read or run, and
nothing is fetched. The stand ins have the two transformers 4 habits the adapter exists for: a tied weights
LIST, and a mask call written with `input_embeds=` and `cache_position=`.

A class has to be CREATED while the adapter is on, because that is when real remote code is created (it's
imported during the load), so each test builds its classes through `make_classes()` inside the `with`.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest
import torch
import transformers
import transformers.masking_utils as masks
import transformers.modeling_utils as modeling
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from senbonzakura import legacyremote as L
from senbonzakura import modelload


@pytest.fixture(autouse=True)
def _fresh_once_only_notes():
    L._SAID.clear()
    yield
    L._SAID.clear()


def make_classes(embedding="embedding"):
    """The stand in model, created now (so inside whatever adapter is on). Returns (config class, model class).

    It declares no tied list itself; `make_with_list` adds the transformers 4 list the way a model file does.
    """
    from transformers.masking_utils import create_causal_mask

    class TinyConfig(PretrainedConfig):
        model_type = "tiny_legacy_standin"

        def __init__(self, vocab_size=48, hidden_size=16, num_hidden_layers=1, tie_word_embeddings=True, **kw):
            self.vocab_size, self.hidden_size, self.num_hidden_layers = vocab_size, hidden_size, num_hidden_layers
            super().__init__(tie_word_embeddings=tie_word_embeddings, **kw)

    class TinyBase(PreTrainedModel):
        config_class = TinyConfig

        def __init__(self, config):
            super().__init__(config)
            setattr(self, embedding, nn.Embedding(config.vocab_size, config.hidden_size))
            self.mix = nn.Linear(config.hidden_size, config.hidden_size)
            self.post_init()

        def get_input_embeddings(self):
            return getattr(self, embedding)

        def forward(self, input_ids):
            x = getattr(self, embedding)(input_ids)
            position = torch.arange(0, x.shape[1])
            mask = create_causal_mask(config=self.config, input_embeds=x, attention_mask=None,
                                      cache_position=position, past_key_values=DynamicCache(config=self.config),
                                      position_ids=position.unsqueeze(0))
            return self.mix(x), mask

    class TinyForCausalLM(PreTrainedModel):
        config_class = TinyConfig
        base_model_prefix = "model"

        def __init__(self, config):
            super().__init__(config)
            self.model = TinyBase(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids):
            hidden, mask = self.model(input_ids)
            weight = self.model.get_input_embeddings().weight if self.config.tie_word_embeddings else self.lm_head.weight
            return CausalLMOutputWithPast(logits=torch.nn.functional.linear(hidden, weight)), mask

    return TinyConfig, TinyForCausalLM


def make_with_list(keys=("lm_head.weight",), embedding="embedding"):
    """Same, but with the list in the class body, which is what a real model's file has."""
    from transformers.masking_utils import create_causal_mask   # noqa: F401  (bound here, as remote code binds it)
    config_cls, model_cls = make_classes(embedding)
    body = {"_tied_weights_keys": list(keys)}
    return config_cls, type(model_cls.__name__, (model_cls,), body)


def build(model_cls, config_cls, **kw):
    return model_cls(config_cls(**kw))


# ----------------------------------------------------------------------------- the failure it exists for
def test_the_list_form_fails_without_the_adapter():
    config_cls, model_cls = make_with_list()
    with pytest.raises(AttributeError, match="'list' object has no attribute 'keys'"):
        build(model_cls, config_cls)


def test_a_mask_call_written_the_old_way_fails_without_the_adapter():
    config_cls, model_cls = make_classes()
    model_cls._tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}   # fix the tying by hand
    model = build(model_cls, config_cls)
    with pytest.raises(TypeError, match="input_embeds"):
        model(torch.tensor([[1, 2, 3]]))


# ----------------------------------------------------------------------------- the adapter, working
def test_it_builds_ties_logs_once_and_records_what_it_did():
    said = []
    with L.adapting(True, said.append) as report:
        config_cls, model_cls = make_with_list()
        model = build(model_cls, config_cls)
        assert model.all_tied_weights_keys == {"lm_head.weight": "model.embedding.weight"}
        model.tie_weights()
        assert model.lm_head.weight.data_ptr() == model.model.embedding.weight.data_ptr()
        build(model_cls, config_cls)                                  # a second build says nothing new
    assert report.applied == [L.TIED_WEIGHTS]
    assert len([line for line in said if "tied weights" in line]) == 1
    assert "'lm_head.weight': 'model.embedding.weight'" in said[0]
    assert "legacy remote code" in said[0]


def test_the_mask_call_is_adapted_logged_once_and_recorded_even_after_the_load():
    said = []
    with L.adapting(True, said.append) as report:
        config_cls, model_cls = make_with_list()
        model = build(model_cls, config_cls)
    assert report.applied == [L.TIED_WEIGHTS]                         # the mask has not been called yet
    ids = torch.tensor([[1, 2, 3, 4]])
    logits, mask = model(ids)                                         # the adapter is OFF now; the model's own module holds the wrapper
    model(ids)
    assert logits.logits.shape == (1, 4, 48) and mask is not None
    assert report.applied == [L.TIED_WEIGHTS, L.MASK_CALL]
    assert len([line for line in said if "was called the transformers 4 way" in line]) == 1


def test_save_and_reload_gives_the_same_logits(tmp_path):
    with L.adapting(True) as report:
        config_cls, model_cls = make_with_list()
        model = build(model_cls, config_cls)
    ids = torch.tensor([[3, 1, 4, 1, 5]])
    before = model(ids)[0].logits
    model.save_pretrained(tmp_path)
    from safetensors import safe_open
    keys = {k for f in tmp_path.glob("*.safetensors") for k in safe_open(f, "pt").keys()}  # noqa: SIM118
    assert "lm_head.weight" not in keys and "model.embedding.weight" in keys
    again = model_cls.from_pretrained(tmp_path)
    assert again.lm_head.weight.data_ptr() == again.model.embedding.weight.data_ptr()
    assert torch.equal(again(ids)[0].logits, before)
    assert report.applied.count(L.TIED_WEIGHTS) == 1


def test_an_embedding_with_another_name_is_found_from_the_getters_not_from_a_name():
    with L.adapting(True):
        config_cls, model_cls = make_with_list(embedding="wte")
        model = build(model_cls, config_cls)
    assert model.all_tied_weights_keys == {"lm_head.weight": "model.wte.weight"}


def test_untied_config_gives_an_empty_mapping():
    with L.adapting(True):
        config_cls, model_cls = make_with_list()
        model = build(model_cls, config_cls, tie_word_embeddings=False)
    assert model.all_tied_weights_keys == {}


def test_a_dict_form_class_is_left_alone_and_nothing_is_logged():
    said = []
    with L.adapting(True, said.append) as report:
        config_cls, model_cls = make_classes()
        model_cls._tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}
        model = build(model_cls, config_cls)
    assert model.all_tied_weights_keys == {"lm_head.weight": "model.embedding.weight"}
    assert not isinstance(model_cls.__dict__["_tied_weights_keys"], L._TiedKeysFromList)
    assert said == [] and report.applied == []


def test_reading_the_stand_in_on_the_class_gives_the_list_back():
    with L.adapting(True):
        _, model_cls = make_with_list()
    assert model_cls._tied_weights_keys == ["lm_head.weight"]


def test_the_same_layer_both_ways_has_nothing_to_tie():
    with L.adapting(True):
        config_cls, model_cls = make_with_list()
        model = build(model_cls, config_cls)
        model.get_output_embeddings = model.get_input_embeddings
        assert model.get_expanded_tied_weights_keys() == {}


# ----------------------------------------------------------------------------- refusals
def test_a_list_naming_the_wrong_weight_is_refused_in_a_sentence():
    with L.adapting(True):
        config_cls, model_cls = make_with_list(keys=("somewhere_else.weight",))
        with pytest.raises(L.Refusal, match=r"lists \['somewhere_else.weight'\] as tied, but its output embedding is 'lm_head.weight'"):
            build(model_cls, config_cls)


@pytest.mark.parametrize("failure", [None, NotImplementedError("no head"), AttributeError("no getter")])
def test_a_model_that_does_not_name_two_layers_to_tie_is_refused(failure):
    with L.adapting(True):
        config_cls, model_cls = make_with_list()
        model_cls.get_output_embeddings = (lambda self: None) if failure is None else (lambda self: (_ for _ in ()).throw(failure))
        with pytest.raises(L.Refusal, match="don't give two layers to tie"):
            build(model_cls, config_cls)


def test_a_layer_that_is_not_in_the_model_is_refused():
    with L.adapting(True):
        config_cls, model_cls = make_with_list()
        model_cls.get_input_embeddings = lambda self: nn.Embedding(4, 4)       # not one of its own modules
        with pytest.raises(L.Refusal, match="don't give two layers to tie"):
            build(model_cls, config_cls)


def _wrapped(name="create_causal_mask"):
    seen = {}
    report = L.Report()
    original = lambda *a, **k: seen.update(k) or "built"               # noqa: E731
    original.__name__ = name
    return L._mask_wrapper(original, lambda m: None, report), seen, report


def test_both_embedding_arguments_at_once_are_refused():
    wrapper, _, _ = _wrapped()
    with pytest.raises(L.Refusal, match="both input_embeds and inputs_embeds"):
        wrapper(input_embeds=torch.zeros(1, 2, 3), inputs_embeds=torch.zeros(1, 2, 3))


def test_a_cache_position_that_does_not_match_the_cache_is_refused():
    wrapper, _, _ = _wrapped()
    embeds = torch.zeros(1, 3, 4)
    with pytest.raises(L.Refusal, match="isn't the one transformers 5 works out"):
        wrapper(input_embeds=embeds, cache_position=torch.tensor([5, 6, 7]), past_key_values=None)
    with pytest.raises(L.Refusal, match="isn't the one transformers 5 works out"):
        wrapper(input_embeds=embeds, cache_position=torch.tensor([0, 1]), past_key_values=None)   # wrong length


def test_a_cache_position_without_its_embeddings_cannot_be_checked_so_is_refused():
    wrapper, _, _ = _wrapped()
    with pytest.raises(L.Refusal, match="without the embeddings"):
        wrapper(cache_position=torch.tensor([0, 1, 2]))


def test_a_matching_cache_position_is_dropped_whether_or_not_there_is_a_cache():
    wrapper, seen, report = _wrapped()
    embeds = torch.zeros(1, 3, 4)
    assert wrapper(input_embeds=embeds, cache_position=torch.arange(3), past_key_values=None) == "built"
    assert "cache_position" not in seen and seen["inputs_embeds"] is embeds
    cache = types.SimpleNamespace(get_seq_length=lambda: 2)
    wrapper(input_embeds=embeds, cache_position=torch.arange(2, 5), past_key_values=cache)
    assert report.applied == [L.MASK_CALL]


def test_only_the_renamed_argument_is_enough_to_adapt_the_call():
    wrapper, seen, report = _wrapped()
    embeds = torch.zeros(1, 2, 3)
    wrapper(input_embeds=embeds)
    assert seen == {"inputs_embeds": embeds} and report.applied == [L.MASK_CALL]


def test_a_cache_position_of_none_is_just_dropped():
    wrapper, seen, _ = _wrapped()
    wrapper(inputs_embeds=torch.zeros(1, 2, 3), cache_position=None)
    assert "cache_position" not in seen


def test_a_call_made_the_transformers_5_way_goes_straight_through():
    wrapper, seen, report = _wrapped()
    wrapper(inputs_embeds=torch.zeros(1, 2, 3), attention_mask=None)
    assert "attention_mask" in seen and report.applied == []


def test_the_real_mask_function_still_works_inside_the_adapter_for_a_new_style_call():
    config_cls, _ = make_classes()
    with L.adapting(True):
        mask = masks.create_causal_mask(config=config_cls(), inputs_embeds=torch.zeros(1, 3, 16),
                                        attention_mask=None, past_key_values=None)
    assert mask is None or mask.shape[-1] == 3


# ----------------------------------------------------------------------------- the version guard and the shape guard
def test_a_version_it_was_not_tested_on_does_nothing_and_says_so_once():
    said = []
    original = masks.create_causal_mask
    with L.adapting(True, said.append, known=("0.0.1",)) as report:
        assert masks.create_causal_mask is original                   # nothing patched
        config_cls, model_cls = make_with_list()
        with pytest.raises(AttributeError):                            # so the old failure is the honest outcome
            build(model_cls, config_cls)
    with L.adapting(True, said.append, known=("0.0.1",)):
        pass
    assert report.skipped and transformers.__version__ in report.skipped and report.applied == []
    assert len(said) == 1 and "did nothing" in said[0] and "0.0.1" in said[0]


def test_the_version_this_was_written_against_is_on_the_list():
    assert "5.20.1" in L.KNOWN_TRANSFORMERS
    assert transformers.__version__ in L.KNOWN_TRANSFORMERS, "install a tested transformers, or test and list the new one"


def test_internals_that_are_not_the_expected_shape_do_nothing_and_say_so_once(monkeypatch):
    said = []
    monkeypatch.setattr(L, "_internals", lambda: None)
    with L.adapting(True, said.append) as report:
        pass
    with L.adapting(True, said.append):
        pass
    assert report.skipped and len(said) == 1 and "did nothing" in said[0]


def test_what_counts_as_the_expected_shape(monkeypatch):
    assert L._internals() is not None
    # a mask function that still takes cache_position is the old shape
    monkeypatch.setattr(masks, "create_causal_mask", lambda config, inputs_embeds, cache_position: None)
    assert L._internals() is None
    monkeypatch.undo()
    # a mask function with the old argument name
    monkeypatch.setattr(masks, "create_causal_mask", lambda config, input_embeds: None)
    assert L._internals() is None
    monkeypatch.undo()
    # a missing function
    monkeypatch.setattr(masks, "create_sliding_window_causal_mask", None)
    assert L._internals() is None
    monkeypatch.undo()
    # a transformers module that can't be imported
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", None)
    assert L._internals() is None


def test_an_init_subclass_that_is_not_a_classmethod_is_not_the_expected_shape():
    base = modeling.PreTrainedModel
    raw = base.__dict__["__init_subclass__"]
    try:
        base.__init_subclass__ = lambda **kw: None                     # a plain function in the class dict
        assert L._internals() is None
    finally:
        base.__init_subclass__ = raw
    assert L._internals() is not None


# ----------------------------------------------------------------------------- nothing leaks
def test_it_is_off_unless_the_models_own_code_is_being_run():
    original, raw = masks.create_causal_mask, modeling.PreTrainedModel.__dict__["__init_subclass__"]
    with L.adapting(False) as report:
        assert masks.create_causal_mask is original
        assert modeling.PreTrainedModel.__dict__["__init_subclass__"] is raw
        assert not L._LOCK.locked()
    assert report.applied == [] and report.skipped is None


def test_everything_is_put_back_afterwards_and_after_a_failure():
    originals = {n: getattr(masks, n) for n in ("create_causal_mask", "create_sliding_window_causal_mask")}
    raw = modeling.PreTrainedModel.__dict__["__init_subclass__"]
    with L.adapting(True) as report:
        assert masks.create_causal_mask is not originals["create_causal_mask"]
        assert masks.create_causal_mask.legacy_adapter is True
        assert modeling.PreTrainedModel.__dict__["__init_subclass__"] is not raw
        assert L._LOCK.locked()
    assert {n: getattr(masks, n) for n in originals} == originals
    assert modeling.PreTrainedModel.__dict__["__init_subclass__"] is raw and not L._LOCK.locked()
    with pytest.raises(RuntimeError, match="the load failed"):
        with L.adapting(True):
            raise RuntimeError("the load failed")
    assert {n: getattr(masks, n) for n in originals} == originals
    assert modeling.PreTrainedModel.__dict__["__init_subclass__"] is raw and not L._LOCK.locked()
    assert report.skipped is None


def test_a_class_made_after_the_adapter_is_off_gets_no_stand_in():
    with L.adapting(True):
        pass
    config_cls, model_cls = make_with_list()
    assert isinstance(model_cls.__dict__["_tied_weights_keys"], list)
    with pytest.raises(AttributeError, match="'list' object has no attribute 'keys'"):
        build(model_cls, config_cls)


def test_a_second_load_that_cannot_get_the_lock_in_time_is_refused():
    started, release = threading.Event(), threading.Event()

    def holder():
        with L.adapting(True):
            started.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert started.wait(10)
        with pytest.raises(L.Refusal, match="Another model load in this process"):
            with L.adapting(True, lock_seconds=0.05):
                pytest.fail("the second load must not get in")
    finally:
        release.set()
        thread.join(10)
    with L.adapting(True):                                             # and the lock came free
        pass


# ----------------------------------------------------------------------------- a model whose code was loaded before
def _model_in(module_name, held):
    module = types.ModuleType(module_name)
    if held is not None:
        module.create_causal_mask = held
    sys.modules[module_name] = module
    return type("Modelled", (), {"__module__": module_name})()


def test_a_model_loaded_before_the_adapter_was_on_is_named_once(monkeypatch):
    said = []
    name = "standin_remote_module"
    monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        model = _model_in(name, masks.create_causal_mask)             # the plain function: bound before
        L.note_if_not_adapted(model, L.Report(), said.append)
        L.note_if_not_adapted(model, L.Report(), said.append)
        assert len(said) == 1 and "loaded before the adapter was on" in said[0]
        said.clear()
        L._SAID.clear()
        skipped = L.Report()
        skipped.skipped = "not this version"
        L.note_if_not_adapted(model, skipped, said.append)            # a skipped adapter says nothing more
        sys.modules[name].create_causal_mask = L._mask_wrapper(masks.create_causal_mask, lambda m: None, L.Report())
        L.note_if_not_adapted(model, L.Report(), said.append)          # holds the wrapper: fine
        sys.modules[name].create_causal_mask = lambda *a, **k: None    # the model's own function: not ours to judge
        L.note_if_not_adapted(model, L.Report(), said.append)
        L.note_if_not_adapted(object(), L.Report(), said.append)       # no such module at all
        L.note_if_not_adapted(_model_in("standin_without_mask", None), L.Report(), None)
        assert said == []
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop("standin_without_mask", None)


# ----------------------------------------------------------------------------- how the loader uses it
@pytest.fixture
def loader(monkeypatch, tiny_tok):
    monkeypatch.setattr(modelload, "_refuse_without_the_checker", lambda: None)
    monkeypatch.setattr(modelload, "_config_quantisation", lambda *a, **k: None)
    monkeypatch.setattr(modelload, "AutoConfig", types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace()))
    monkeypatch.setattr(modelload, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: tiny_tok))

    def install(from_pretrained):
        monkeypatch.setattr(modelload, "AutoModelForCausalLM", types.SimpleNamespace(from_pretrained=from_pretrained))
    return install


def _remote_loader(seen):
    def from_pretrained(source, **kw):
        seen.append(kw.get("trust_remote_code"))
        config_cls, model_cls = make_with_list()
        return model_cls(config_cls()), {"missing_keys": []}
    return from_pretrained


def test_a_remote_code_load_is_adapted_and_the_provenance_names_it(loader):
    seen, said = [], []
    loader(_remote_loader(seen))
    model, _ = modelload.load_model_and_tokenizer("org/m", device="cpu", trust_remote_code=True, log=said.append)
    assert seen == [True]
    assert model.senbon_load["remote_code_adapters"] == [L.TIED_WEIGHTS]
    model(torch.tensor([[1, 2, 3]]))
    assert model.senbon_load["remote_code_adapters"] == [L.TIED_WEIGHTS, L.MASK_CALL]   # the same live list
    assert any("legacy remote code" in line for line in said)


def test_a_load_without_remote_code_never_reaches_the_adapter(loader, tiny_model):
    loader(lambda source, **kw: (tiny_model, {"missing_keys": []}))
    model, _ = modelload.load_model_and_tokenizer("m", device="cpu")
    assert model.senbon_load["remote_code_adapters"] == []


def test_a_refusal_during_the_load_is_one_sentence(loader):
    def from_pretrained(source, **kw):
        config_cls, model_cls = make_with_list(keys=("not_the_head.weight",))
        return model_cls(config_cls()), {"missing_keys": []}
    loader(from_pretrained)
    with pytest.raises(SystemExit) as stop:
        modelload.load_model_and_tokenizer("org/m", device="cpu", trust_remote_code=True)
    said = str(stop.value.code)
    assert said.startswith("senbonzakura: 'org/m' can't be loaded.") and "not_the_head.weight" in said
    assert "Traceback" not in said


def test_the_load_notes_a_model_whose_code_was_imported_before(loader):
    said, name = [], "standin_early_module"
    sys.modules[name] = types.ModuleType(name)
    sys.modules[name].create_causal_mask = masks.create_causal_mask

    class Model(nn.Module):
        pass
    Model.__module__ = name
    loader(lambda source, **kw: (Model(), {"missing_keys": []}))
    try:
        modelload.load_model_and_tokenizer("org/m", device="cpu", trust_remote_code=True, log=said.append)
    finally:
        sys.modules.pop(name, None)
    assert any("loaded before the adapter was on" in line for line in said)


def test_the_provenance_has_the_new_field_in_every_era():
    plain = modelload.provenance(modelload.Plan(source="m", kind="plain"), "bfloat16")
    assert plain["remote_code_adapters"] == []
    assert modelload.with_defaults(None)["remote_code_adapters"] == []
    assert modelload.with_defaults({"format": "gguf"})["remote_code_adapters"] == []
