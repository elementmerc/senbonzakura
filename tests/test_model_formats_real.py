# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""One real checkpoint per format, through the shared loader and through `attribute`'s loader.

The unit tests in `test_modelload.py` prove every branch with stand-ins. This file proves the
branches meet real files: the smallest public checkpoint of each format, all of the same 0.5B
model, so the formats are compared on one set of weights.

SKIPPED CLEANLY, never failed, when the thing it needs is not here: the backend package for that
format, or the network. Nothing is downloaded unless the test is going to run. A run that skips
all five has verified nothing, and the final summary line says how many ran.

Set SENBON_REAL_FORMATS=0 to skip them all, for instance on a metered connection.
"""
import importlib.util
import os

import pytest
import torch

from senbonzakura import attribution as A
from senbonzakura import attribution_hf as H
from senbonzakura import cli, modelload

PLAIN = "Qwen/Qwen2.5-0.5B-Instruct"
GGUF_REPO, GGUF_FILE = "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"

FORMATS = {
    "safetensors": dict(model=PLAIN, needs=(), flags={}, fmt="safetensors"),
    "bnb-4bit-on-the-fly": dict(model=PLAIN, needs=("bitsandbytes",), flags={"load_in_4bit": True},
                                fmt="bnb"),
    "bnb-8bit-on-the-fly": dict(model=PLAIN, needs=("bitsandbytes",), flags={"load_in_8bit": True},
                                fmt="bnb"),
    "bnb-4bit-stored": dict(model="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
                            needs=("bitsandbytes",), flags={}, fmt="bnb"),
    "gptq": dict(model="Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4", needs=("optimum", "gptqmodel"),
                 flags={}, fmt="gptq"),
    "awq": dict(model="Qwen/Qwen2.5-0.5B-Instruct-AWQ", needs=("gptqmodel",), flags={}, fmt="awq"),
    "gguf": dict(model=GGUF_REPO, needs=("gguf",), flags={"gguf_file": GGUF_FILE}, fmt="gguf"),
}


def _skip_unless_possible(spec):
    if os.environ.get("SENBON_REAL_FORMATS") == "0":
        pytest.skip("SENBON_REAL_FORMATS=0")
    for package in spec["needs"]:
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"{package} is not installed")
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(spec["model"], timeout=15)
    except Exception as e:
        pytest.skip(f"{spec['model']} is not reachable here ({type(e).__name__})")


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=sorted(FORMATS))
def loaded(request):
    spec = FORMATS[request.param]
    _skip_unless_possible(spec)
    try:
        model, tok = cli.load_model_and_tokenizer(spec["model"], device=_device(), log=lambda m: None,
                                                  **spec["flags"])
    except SystemExit as e:
        # A backend that is installed but cannot run here (GPTQ and AWQ kernels with no toolchain)
        # is a skip with its reason; anything else is a real failure.
        said = str(e.code)
        if "couldn't be used" in said or "needs" in said and "isn't installed" in said:
            pytest.skip(said.splitlines()[0])
        raise
    return request.param, spec, model, tok


def test_it_loads_and_says_how(loaded):
    name, spec, model, _ = loaded
    info = model.senbon_load
    assert info["format"] == spec["fmt"], name
    assert info["precision"]
    if spec["fmt"] != "safetensors":
        assert info["backend"]["version"], f"{name}: the backend's version must be recorded"
    if spec["fmt"] == "gguf":
        assert info["dequantised"] is True and info["dequantised_from"] == "Q4_K_M"
    else:
        assert info["dequantised"] is False


def test_it_answers_sensibly(loaded):
    _, _, model, tok = loaded
    ids = tok("The capital of France is", return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    assert torch.isfinite(logits).all()
    top = [tok.decode([int(i)]).strip() for i in logits.topk(5).indices]
    assert "Paris" in top, top


def test_the_attribution_hook_changes_the_forward_pass_on_this_format(loaded):
    """The hook is on the block output, so it must bite on quantised weights exactly as on plain."""
    name, _, model, tok = loaded
    from senbonzakura import modelmap
    layers = modelmap.stack_or_raise(model)
    dev = next(model.parameters()).device
    ids = tok("The capital of France is", return_tensors="pt").to(dev)
    with torch.no_grad():
        before = model(**ids).logits[0, -1].float().cpu()
        direction = torch.randn(model.config.hidden_size)
        with A.ablating(layers, direction, [len(layers) // 2]):
            during = model(**ids).logits[0, -1].float().cpu()
        after = model(**ids).logits[0, -1].float().cpu()
    assert not torch.allclose(before, during), f"{name}: ablating a block changed nothing"
    assert torch.allclose(before, after), f"{name}: the hook was not removed"


def test_attribute_loads_it_and_records_the_provenance(loaded):
    name, spec, _, _ = loaded
    if "Qwen2.5-0.5B-Instruct-bnb" in spec["model"] and _device() == "cpu":
        pytest.skip("stored bitsandbytes weights need a card")
    _model, _tok, identity = H.load(spec["model"], device=_device(), **spec["flags"])
    assert identity["format"] == spec["fmt"], name
    assert identity["precision"]
    assert set(modelload.LEGACY) <= set(identity)


def test_a_gguf_folder_without_tokenizer_files_is_a_sentence_or_loads(tmp_path):
    """The LFM2.5 shape: a folder holding only the GGUF. It must load or say what to do, never trace."""
    spec = FORMATS["gguf"]
    _skip_unless_possible(spec)
    from huggingface_hub import hf_hub_download
    hf_hub_download(spec["model"], GGUF_FILE, local_dir=tmp_path)
    try:
        model, tok = cli.load_model_and_tokenizer(str(tmp_path), device=_device(), log=lambda m: None)
    except SystemExit as e:
        said = str(e.code)
        assert "tokenizer" in said and "pip install" in said or "tokenizer.json" in said, said
        assert "Traceback" not in said
        return
    assert model.senbon_load["format"] == "gguf" and tok is not None
