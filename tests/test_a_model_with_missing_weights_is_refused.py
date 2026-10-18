# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A checkpoint that leaves weights missing is refused by both loaders, before anything is measured.

transformers fills a missing weight with random values and carries on, so a run on such a model
measures noise. Found 2026-10-18, when a checkpoint stored in a packed quantised form came up with
every layer's weights missing and two timing runs measured nothing. See `loadreport`.

The checkpoints here are built on disk from a tiny random model, so the real `from_pretrained`
runs on every one of them and nothing is downloaded.
"""
from __future__ import annotations

import types

import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file

from senbonzakura import attribution as A
from senbonzakura import attribution_hf as H
from senbonzakura import cli, loadreport

DROPPED = "model.layers.0.mlp.up_proj.weight"


def _checkpoint(root, name, change):
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=2, num_attention_heads=2,
                                   num_key_value_heads=2, tie_word_embeddings=True)
    torch.manual_seed(0)
    base = root / "base"
    if not base.exists():
        transformers.LlamaForCausalLM(cfg).save_pretrained(base)
    out = root / name
    out.mkdir()
    cfg.save_pretrained(out)
    tensors = change(load_file(base / "model.safetensors"))
    save_file({k: v.contiguous() for k, v in tensors.items()}, out / "model.safetensors",
              metadata={"format": "pt"})
    return str(out)


def _intact(sd):
    return sd


def _dropped(sd):
    return {k: v for k, v in sd.items() if k != DROPPED}


def _packed(sd):
    """The shape that was found: each layer weight renamed to the `._data` half of a packed pair."""
    return {(k + "._data" if "layers.1" in k and k.endswith("proj.weight") else k): v
            for k, v in sd.items()}


@pytest.fixture
def models(tmp_path):
    return {name: _checkpoint(tmp_path, name, change)
            for name, change in (("intact", _intact), ("dropped", _dropped), ("packed", _packed))}


# --------------------------------------------------------------------------- the message on its own

def test_a_load_with_nothing_missing_is_not_refused():
    assert loadreport.missing_weights("m", {"missing_keys": set(), "unexpected_keys": {"x"}}) is None
    assert loadreport.missing_weights("m", {}) is None
    assert loadreport.missing_weights("m", None) is None


def test_an_incomplete_file_is_told_to_download_again():
    said = loadreport.missing_weights("org/m", {"missing_keys": {DROPPED}, "unexpected_keys": set()})
    assert "'org/m' loaded with 1 of its weights missing" in said
    assert "random numbers" in said
    assert DROPPED in said
    assert "Download it again" in said
    assert "packed" not in said


def test_a_packed_file_is_told_why_and_not_to_download_again():
    info = {"missing_keys": {"a.weight"}, "unexpected_keys": {"a.weight._data", "a.weight._scale"}}
    said = loadreport.missing_weights("m", info)
    assert "packed, quantised form" in said
    assert "Download it again" not in said


def test_a_long_list_is_cut_to_a_few_names_in_a_fixed_order():
    missing = {f"layer.{i:02d}.weight" for i in range(12)}
    said = loadreport.missing_weights("m", {"missing_keys": missing})
    assert "with 12 of its weights missing" in said
    assert "layer.00.weight, layer.01.weight, layer.02.weight, layer.03.weight, layer.04.weight" in said
    assert "(and 7 more)" in said
    assert "layer.05.weight" not in said
    assert said == loadreport.missing_weights("m", {"missing_keys": sorted(missing, reverse=True)})


# --------------------------------------------------------------------------- the command line loader

def _cli_load(path, monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_refuse_without_the_checker", lambda: None)
    return cli.load_model_and_tokenizer(path, device="cpu", log=lambda m: None)


def test_the_command_line_loader_loads_an_intact_checkpoint(models, monkeypatch):
    model, _ = _cli_load(models["intact"], monkeypatch)
    assert isinstance(model, torch.nn.Module)
    assert not model.training


@pytest.mark.parametrize(("name", "advice"), [("dropped", "Download it again"),
                                              ("packed", "packed, quantised form")])
def test_the_command_line_loader_refuses_missing_weights(models, monkeypatch, name, advice):
    with pytest.raises(SystemExit) as stop:
        _cli_load(models[name], monkeypatch)
    said = str(stop.value.code)
    assert "of its weights missing" in said
    assert advice in said


# --------------------------------------------------------------------------- the attribution loader

def _stub_tokenizer(monkeypatch):
    end = "</s>"
    tok = types.SimpleNamespace(pad_token=None, eos_token=end, padding_side="right")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    # This file is about the weights; the prompt format is checked in test_attribute_takes_a_chat_template.py.
    monkeypatch.setattr(cli, "ensure_chat_template", lambda *_a, **_k: {"source": "stub", "sha256": None})


def test_the_attribution_loader_loads_an_intact_checkpoint(models, monkeypatch):
    _stub_tokenizer(monkeypatch)
    model, tok, identity = H.load(models["intact"], device="cpu", dtype=torch.float32)
    assert not model.training
    assert tok.pad_token == "</s>"
    assert identity["id"] == models["intact"]


def test_the_attribution_loader_stops_in_a_sentence_when_the_files_are_not_there(tmp_path):
    """A folder with a config and nothing else, the shape a structure-only read leaves behind."""
    (tmp_path / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    with pytest.raises(A.AttributionError, match="could not be loaded"):
        H.load(str(tmp_path), device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("name", ["dropped", "packed"])
def test_the_attribution_loader_refuses_missing_weights(models, monkeypatch, name):
    _stub_tokenizer(monkeypatch)
    with pytest.raises(A.AttributionError, match="of its weights missing"):
        H.load(models[name], device="cpu", dtype=torch.float32)
