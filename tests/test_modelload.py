# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every branch of the shared model loader: which format it is, what is missing, what is recorded.

Nothing here downloads a model or needs a GPU. Backend packages are made present or absent by
patching `find_spec`, a checkpoint's stored quantisation is supplied by patching the config reader,
and the Hugging Face classes are replaced the way the rest of the suite replaces them. The
real-model checks, one per format, are in `test_model_formats_real.py` and skip cleanly.
"""
import importlib.util
import types

import pytest
import torch

from senbonzakura import attribution as A
from senbonzakura import attribution_followups as F
from senbonzakura import attribution_hf as H
from senbonzakura import attribution_run as R
from senbonzakura import cli, modelload, parser, stamps


@pytest.fixture
def backends(monkeypatch):
    """Control which backend packages look installed. Returns a set the test may edit."""
    present = {"bitsandbytes", "optimum", "gptqmodel", "gguf"}
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name in {"bitsandbytes", "optimum", "gptqmodel", "gguf"}:
            return object() if name in present else None
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    return present


def _stored(monkeypatch, qc):
    monkeypatch.setattr(modelload, "_config_quantisation", lambda *a, **k: qc)


# --------------------------------------------------------------------------- plan: plain and on the fly

def test_a_plain_checkpoint_is_plain(monkeypatch):
    _stored(monkeypatch, None)
    p = modelload.plan("org/m")
    assert (p.kind, p.quantised, p.dequantised, p.source) == ("plain", False, False, "org/m")


def test_both_bit_widths_at_once_is_refused():
    with pytest.raises(modelload.LoadRefusedError, match="Choose one"):
        modelload.plan("m", load_in_4bit=True, load_in_8bit=True)


@pytest.mark.parametrize(("flags", "bits", "qtype"), [({"load_in_4bit": True}, 4, "nf4"),
                                                      ({"load_in_8bit": True}, 8, "int8")])
def test_on_the_fly_quantisation_is_planned(monkeypatch, backends, flags, bits, qtype):
    _stored(monkeypatch, None)
    p = modelload.plan("m", **flags)
    assert (p.kind, p.bits, p.quant_type, p.on_the_fly) == ("bnb", bits, qtype, True)


def test_on_the_fly_without_bitsandbytes_names_the_package_and_the_command(monkeypatch, backends):
    _stored(monkeypatch, None)
    backends.discard("bitsandbytes")
    with pytest.raises(modelload.LoadRefusedError) as e:
        modelload.plan("m", load_in_4bit=True)
    said = str(e.value)
    assert "bitsandbytes" in said and 'pip install "senbonzakura[quant]"' in said
    assert "Traceback" not in said


def test_the_editor_refuses_on_the_fly_quantisation(monkeypatch, backends):
    _stored(monkeypatch, None)
    with pytest.raises(modelload.LoadRefusedError, match="rewriting weights"):
        modelload.plan("m", load_in_4bit=True, forward_only=False)


# --------------------------------------------------------------------------- plan: stored quantisation

@pytest.mark.parametrize(("qc", "kind", "bits", "qtype"), [
    ({"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}, "bnb", 4, "nf4"),
    ({"quant_method": "bitsandbytes", "load_in_4bit": True}, "bnb", 4, "fp4"),
    ({"quant_method": "bitsandbytes", "load_in_8bit": True}, "bnb", 8, "int8"),
    ({"quant_method": "gptq", "bits": 4}, "gptq", 4, None),
    ({"quant_method": "gptq"}, "gptq", 4, None),
    ({"quant_method": "awq", "bits": 4}, "awq", 4, None),
])
def test_a_stored_quantisation_is_recognised(monkeypatch, backends, qc, kind, bits, qtype):
    _stored(monkeypatch, qc)
    p = modelload.plan("m")
    assert (p.kind, p.bits, p.quant_type, p.on_the_fly) == (kind, bits, qtype, False)
    assert p.quantised and not p.dequantised


def test_an_unrecognised_method_is_kept_and_named(monkeypatch, backends):
    _stored(monkeypatch, {"quant_method": "hqq"})
    p = modelload.plan("m")
    assert (p.kind, p.method) == ("other", "hqq")
    _stored(monkeypatch, {"something": 1})
    assert modelload.plan("m").method == "unknown"


@pytest.mark.parametrize(("qc", "missing", "extra"), [
    ({"quant_method": "gptq", "bits": 4}, "optimum", "gptq"),
    ({"quant_method": "gptq", "bits": 4}, "gptqmodel", "gptq"),
    ({"quant_method": "awq", "bits": 4}, "gptqmodel", "awq"),
    ({"quant_method": "bitsandbytes", "load_in_4bit": True}, "bitsandbytes", "quant"),
])
def test_each_missing_backend_is_one_sentence(monkeypatch, backends, qc, missing, extra):
    _stored(monkeypatch, qc)
    backends.discard(missing)
    with pytest.raises(modelload.LoadRefusedError) as e:
        modelload.plan("m")
    said = str(e.value)
    assert missing in said and f'pip install "senbonzakura[{extra}]"' in said
    assert said.count("\n") == 0


def test_a_flag_on_an_already_quantised_checkpoint_is_refused(monkeypatch, backends):
    _stored(monkeypatch, {"quant_method": "awq", "bits": 4})
    with pytest.raises(modelload.LoadRefusedError, match="already stored quantised"):
        modelload.plan("m", load_in_4bit=True)


def test_the_editor_refuses_a_stored_quantisation(monkeypatch, backends):
    _stored(monkeypatch, {"quant_method": "gptq", "bits": 4})
    with pytest.raises(modelload.LoadRefusedError, match="rewriting weights"):
        modelload.plan("m", forward_only=False)


def test_the_config_reader_survives_an_unreadable_config(monkeypatch, tmp_path):
    import transformers
    assert modelload._config_quantisation(str(tmp_path / "absent"), None) is None
    (tmp_path / "config.json").write_text('{"model_type": "llama", "quantization_config": '
                                          '{"quant_method": "gptq", "bits": 4}}', encoding="utf-8")
    assert modelload._config_quantisation(str(tmp_path), None)["bits"] == 4
    (tmp_path / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    assert modelload._config_quantisation(str(tmp_path), "main") is None
    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict",
                        classmethod(lambda cls, *a, **k: ({"quantization_config": "nope"}, {})))
    assert modelload._config_quantisation("x", None) is None


# --------------------------------------------------------------------------- GGUF

def _gguf(path, body=b"GGUF\x03\x00\x00\x00"):
    path.write_bytes(body)
    return path


def test_a_gguf_file_is_split_into_folder_and_name(tmp_path, monkeypatch, backends):
    f = _gguf(tmp_path / "m-Q4_K_M.gguf")
    monkeypatch.setattr("senbonzakura.gguf_io.read_header", lambda p: {"file_type": "Q4_K_M"})
    p = modelload.plan(str(f))
    assert (p.kind, p.source, p.gguf_file, p.dequantised_from) == (
        "gguf", str(tmp_path), "m-Q4_K_M.gguf", "Q4_K_M")
    assert p.dequantised and p.quantised


def test_a_file_that_is_not_a_gguf_is_not_treated_as_one(tmp_path, monkeypatch):
    _stored(monkeypatch, None)
    f = _gguf(tmp_path / "x.gguf", b"NOPE")
    assert modelload.resolve_gguf(str(f), None) is None


def test_a_file_and_a_different_name_disagree(tmp_path):
    f = _gguf(tmp_path / "a.gguf")
    with pytest.raises(modelload.LoadRefusedError, match="doesn't match"):
        modelload.resolve_gguf(str(f), "b.gguf")
    assert modelload.resolve_gguf(str(f), "a.gguf") == (str(tmp_path), "a.gguf")


def test_a_folder_with_one_gguf_picks_it_and_with_two_refuses(tmp_path):
    _gguf(tmp_path / "a.gguf")
    assert modelload.resolve_gguf(str(tmp_path), None) == (str(tmp_path), "a.gguf")
    _gguf(tmp_path / "b.gguf")
    with pytest.raises(modelload.LoadRefusedError, match="holds 2 GGUF files"):
        modelload.resolve_gguf(str(tmp_path), None)
    assert modelload.resolve_gguf(str(tmp_path), "b.gguf") == (str(tmp_path), "b.gguf")
    with pytest.raises(modelload.LoadRefusedError, match="no 'c.gguf'"):
        modelload.resolve_gguf(str(tmp_path), "c.gguf")


def test_a_folder_with_a_config_or_no_gguf_is_not_a_gguf_folder(tmp_path):
    assert modelload.resolve_gguf(str(tmp_path), None) is None
    _gguf(tmp_path / "a.gguf")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert modelload.resolve_gguf(str(tmp_path), None) is None


def test_a_hub_id_with_a_file_name_is_a_gguf_and_without_is_not(tmp_path):
    assert modelload.resolve_gguf("org/repo-GGUF", "m.Q4_K_M.gguf") == ("org/repo-GGUF", "m.Q4_K_M.gguf")
    assert modelload.resolve_gguf("org/repo", None) is None


def test_an_unreadable_path_string_is_not_a_gguf():
    assert modelload.resolve_gguf("bad\x00path", None) is None


def test_the_quantisation_of_a_gguf_that_cannot_be_read_is_recorded_as_unknown(tmp_path, backends):
    f = _gguf(tmp_path / "broken.gguf")          # magic only: the header reader refuses it
    p = modelload.plan(str(f))
    assert p.kind == "gguf" and p.dequantised_from is None
    assert "unknown quantisation" in modelload.provenance(p, "bfloat16")["precision"]


def test_a_hub_gguf_is_fetched_once_to_read_its_header_and_a_hub_failure_is_left_to_the_load(
        tmp_path, monkeypatch):
    import huggingface_hub
    f = _gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(f))
    monkeypatch.setattr("senbonzakura.gguf_io.read_header", lambda p: {"file_type": "Q8_0"})
    assert modelload._gguf_quant("org/repo-GGUF", "m.gguf", "rev") == "Q8_0"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert modelload._gguf_quant("org/repo-GGUF", "m.gguf") is None


def test_a_gguf_with_a_load_flag_the_editor_or_no_package_is_refused(tmp_path, backends):
    f = _gguf(tmp_path / "a.gguf")
    with pytest.raises(modelload.LoadRefusedError, match="already quantised"):
        modelload.plan(str(f), load_in_4bit=True)
    with pytest.raises(modelload.LoadRefusedError, match="rewriting weights"):
        modelload.plan(str(f), forward_only=False)
    backends.discard("gguf")
    with pytest.raises(modelload.LoadRefusedError) as e:
        modelload.plan(str(f))
    assert "gguf" in str(e.value) and "pip install --force-reinstall senbonzakura" in str(e.value)


# --------------------------------------------------------------------------- what is handed on and recorded

def test_dtype_and_quantisation_config(monkeypatch, backends):
    _stored(monkeypatch, None)
    plain = modelload.plan("m")
    assert modelload.dtype_for(plain, torch) is torch.bfloat16
    assert modelload.bnb_config(plain, torch) is None
    gptq = modelload.Plan(source="m", kind="gptq", method="gptq", bits=4)
    assert modelload.dtype_for(gptq, torch) is torch.float16
    pytest.importorskip("bitsandbytes", reason="building the config needs the quant extra on older transformers")
    four = modelload.bnb_config(modelload.plan("m", load_in_4bit=True), torch)
    assert four.load_in_4bit and four.bnb_4bit_quant_type == "nf4"
    assert modelload.bnb_config(modelload.plan("m", load_in_8bit=True), torch).load_in_8bit


@pytest.mark.parametrize(("p", "label"), [
    (modelload.Plan(source="m", kind="plain"), "bfloat16"),
    (modelload.Plan(source="m", kind="bnb", bits=4, quant_type="nf4"), "nf4"),
    (modelload.Plan(source="m", kind="bnb", bits=8), "int8"),
    (modelload.Plan(source="m", kind="gptq", bits=4), "gptq-4bit"),
    (modelload.Plan(source="m", kind="awq", bits=4), "awq-4bit"),
    (modelload.Plan(source="m", kind="gguf", dequantised_from="Q4_K_M"), "bfloat16 dequantised from Q4_K_M"),
    (modelload.Plan(source="m", kind="other", method="hqq"), "hqq"),
])
def test_the_precision_says_how_the_weights_came_in(p, label):
    assert modelload.provenance(p, "bfloat16")["precision"] == label


def test_provenance_carries_the_same_keys_for_every_format(monkeypatch, backends):
    plain = modelload.provenance(modelload.Plan(source="m", kind="plain"), "bfloat16")
    quant = modelload.provenance(modelload.Plan(source="m", kind="gptq", method="gptq", bits=4), "float16")
    assert set(plain) == set(quant) == set(modelload.LEGACY) | {"precision"}
    assert plain["format"] == "safetensors" and plain["backend"] is None
    assert quant["format"] == "gptq" and quant["backend"]["package"] == "gptqmodel"


def test_the_backend_version_comes_from_the_installed_package(monkeypatch):
    monkeypatch.setattr(modelload, "_version", lambda dist: "9.9.9")
    assert modelload.backend_of("bnb") == {"package": "bitsandbytes", "version": "9.9.9"}
    assert modelload.backend_of("gguf")["package"] == "gguf"
    assert modelload.backend_of("plain") is None
    assert modelload.backend_of("other") is None


def test_an_uninstalled_package_has_no_version():
    assert modelload._version("definitely-not-a-package-name") is None


def test_an_old_identity_reads_as_a_plain_load():
    old = modelload.with_defaults({"id": "m", "precision": "bfloat16"})
    assert old["format"] == "safetensors" and old["dequantised"] is False and old["backend"] is None
    assert modelload.with_defaults(None) == modelload.LEGACY


def test_attach_survives_an_object_that_refuses_attributes():
    model = torch.nn.Linear(1, 1)
    info = modelload.attach(model, modelload.Plan(source="m", kind="plain"), "bfloat16")
    assert model.senbon_load is info
    modelload.attach(object(), modelload.Plan(source="m", kind="plain"), "bfloat16")


def test_the_stamped_precision_follows_what_the_loader_recorded():
    model = types.SimpleNamespace(dtype=torch.float32)
    assert stamps.model_precision(model, False) == "float32"
    assert stamps.model_precision(model, True) == "nf4"
    model.senbon_load = {"precision": "gptq-4bit"}
    assert stamps.model_precision(model, False) == "gptq-4bit"


# --------------------------------------------------------------------------- the attribution hook

def test_an_unchecked_quantiser_is_refused_for_attribution_and_a_known_one_is_not():
    with pytest.raises(modelload.LoadRefusedError, match="hqq"):
        modelload.check_hookable({"format": "other", "quant_method": "hqq"})
    for fmt in ("safetensors", "bnb", "gptq", "awq", "gguf"):
        modelload.check_hookable({"format": fmt})


def test_the_block_probe_accepts_a_tensor_or_a_tuple_and_refuses_anything_else():
    class Block(torch.nn.Module):
        def __init__(self, out):
            super().__init__()
            self.out = out

        def forward(self, x):
            return self.out(x)

    for out in (lambda x: x, lambda x: (x, None)):
        layers = [Block(out)]
        assert modelload.probe_block_output(layers, lambda: layers[0](torch.zeros(1))) is True
    for bad in (lambda x: None, lambda x: ()):
        layers = [Block(bad)]
        with pytest.raises(modelload.LoadRefusedError, match="didn't return a tensor"):
            modelload.probe_block_output(layers, lambda: layers[0](torch.zeros(1)))
    with pytest.raises(modelload.LoadRefusedError, match="never ran"):
        modelload.probe_block_output([Block(lambda x: x)], lambda: None)


# --------------------------------------------------------------------------- readable failures

def test_the_gguf_tokenizer_sentence_names_the_cause_and_both_fixes():
    said = modelload.gguf_tokenizer_refusal("/m", "a.gguf", ValueError("x"))
    assert "pip install sentencepiece tiktoken" in said and "tokenizer.json" in said and "/m" in said


def test_a_backend_that_fails_on_use_is_one_sentence_with_the_install_command():
    p = modelload.Plan(source="m", kind="awq")
    said = modelload.import_refusal(p, ImportError("needs gptqmodel>=7\nsecond line"))
    assert "needs gptqmodel>=7" in said and "second line" not in said and "senbonzakura[awq]" in said
    assert "senbonzakura[" not in modelload.import_refusal(modelload.Plan(source="m", kind="other"), ImportError())


def test_remote_code_is_named_only_when_that_is_the_reason():
    said = modelload.remote_code_refusal("org/m", ValueError("set trust_remote_code=True to run it"))
    assert "--trust-remote-code" in said and "org/m" in said
    assert modelload.remote_code_refusal("org/m", ValueError("something else")) is None


# --------------------------------------------------------------------------- the shared loader

class _Auto:
    """Stand-in for transformers' Auto classes that records what it was asked for."""

    def __init__(self, result=None, error=None):
        self.seen = {}
        self.result, self.error = result, error

    def from_pretrained(self, *a, **k):
        self.seen = {"args": a, **k}
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def plain_stubs(monkeypatch, tiny_model, tiny_tok):
    monkeypatch.setattr(modelload, "_refuse_without_the_checker", lambda: None)
    monkeypatch.setattr(modelload, "_config_quantisation", lambda *a, **k: None)
    tok = _Auto(tiny_tok)
    mod = _Auto((tiny_model, {"missing_keys": []}))
    monkeypatch.setattr(modelload, "AutoTokenizer", tok)
    monkeypatch.setattr(modelload, "AutoModelForCausalLM", mod)
    monkeypatch.setattr(modelload, "AutoConfig", _Auto(types.SimpleNamespace()))
    return tok, mod


def test_the_loader_records_a_plain_load(plain_stubs, tiny_model):
    model, _ = cli.load_model_and_tokenizer("m", device="cpu")
    assert model.senbon_load["format"] == "safetensors" and model.senbon_load["precision"] == "bfloat16"


def test_a_dtype_override_wins(plain_stubs):
    cli.load_model_and_tokenizer("m", device="cpu", dtype=torch.float32)
    assert plain_stubs[1].seen["dtype"] is torch.float32


def test_the_loader_refuses_before_loading_anything(plain_stubs):
    with pytest.raises(SystemExit, match="Choose one"):
        cli.load_model_and_tokenizer("m", device="cpu", load_in_4bit=True, load_in_8bit=True)
    assert plain_stubs[0].seen == {} and plain_stubs[1].seen == {}


def test_eight_bit_asks_for_an_int8_config(plain_stubs, monkeypatch, backends):
    pytest.importorskip("bitsandbytes", reason="building the config needs the quant extra on older transformers")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    cli.load_model_and_tokenizer("m", device="cuda", load_in_8bit=True)
    seen = plain_stubs[1].seen
    assert seen["quantization_config"].load_in_8bit and seen["device_map"] == "auto"


@pytest.mark.parametrize(("device", "placement"), [("cpu", {"": "cpu"}), ("cuda:1", {"": 1})])
def test_on_the_fly_quantisation_follows_the_requested_device(plain_stubs, monkeypatch, backends,
                                                              device, placement):
    pytest.importorskip("bitsandbytes", reason="building the config needs the quant extra on older transformers")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    cli.load_model_and_tokenizer("m", device=device, load_in_4bit=True)
    assert plain_stubs[1].seen["device_map"] == placement


def test_a_stored_quantised_model_on_the_processor_is_placed_by_its_quantiser(plain_stubs, monkeypatch, backends):
    monkeypatch.setattr(modelload, "_config_quantisation",
                        lambda *a, **k: {"quant_method": "gptq", "bits": 4})
    model, _ = cli.load_model_and_tokenizer("m", device="cpu")
    seen = plain_stubs[1].seen
    assert seen["device_map"] == {"": "cpu"} and seen["dtype"] is torch.float16
    assert model.senbon_load["format"] == "gptq"


def test_a_gguf_passes_its_file_name_to_both_loads_and_says_it_was_dequantised(
        plain_stubs, tmp_path, monkeypatch, backends):
    f = _gguf(tmp_path / "m-Q4_K_M.gguf")
    monkeypatch.setattr("senbonzakura.gguf_io.read_header", lambda p: {"file_type": "Q4_K_M"})
    said = []
    model, _ = cli.load_model_and_tokenizer(str(f), device="cpu", log=said.append)
    assert plain_stubs[0].seen["gguf_file"] == "m-Q4_K_M.gguf"
    assert plain_stubs[1].seen["gguf_file"] == "m-Q4_K_M.gguf"
    assert plain_stubs[1].seen["args"] == (str(tmp_path),)
    assert model.senbon_load["dequantised"] is True and model.senbon_load["dequantised_from"] == "Q4_K_M"
    assert any("expanded to bfloat16" in line and "Q4_K_M" in line for line in said)


@pytest.mark.parametrize("error", [ValueError("Couldn't instantiate the backend tokenizer"),
                                   ImportError("sentencepiece"), RuntimeError("boom"), TypeError("t")])
def test_a_gguf_whose_tokenizer_cannot_be_built_is_one_sentence_with_the_fix(
        plain_stubs, tmp_path, monkeypatch, backends, error):
    f = _gguf(tmp_path / "lfm.gguf")
    monkeypatch.setattr(modelload, "AutoTokenizer", _Auto(error=error))
    with pytest.raises(SystemExit) as stop:
        cli.load_model_and_tokenizer(str(f), device="cpu")
    said = str(stop.value.code)
    assert "carries its tokenizer inside it" in said and "pip install sentencepiece tiktoken" in said
    assert "Traceback" not in said


def test_a_plain_tokenizer_import_failure_is_not_disguised(plain_stubs, monkeypatch):
    monkeypatch.setattr(modelload, "AutoTokenizer", _Auto(error=ImportError("sentencepiece")))
    with pytest.raises(ImportError):
        cli.load_model_and_tokenizer("m", device="cpu")


def test_a_model_that_needs_remote_code_says_which_flag(plain_stubs, monkeypatch):
    err = ValueError("requires you to set trust_remote_code=True")
    monkeypatch.setattr(modelload, "AutoTokenizer", _Auto(error=err))
    with pytest.raises(SystemExit, match="--trust-remote-code"):
        cli.load_model_and_tokenizer("org/m", device="cpu")


def test_remote_code_on_the_weights_is_named_too(plain_stubs, monkeypatch):
    monkeypatch.setattr(modelload, "AutoModelForCausalLM",
                        _Auto(error=ValueError("set trust_remote_code=True")))
    with pytest.raises(SystemExit, match="ships custom modelling code"):
        cli.load_model_and_tokenizer("org/m", device="cpu")


def test_a_backend_that_fails_when_used_is_a_sentence_and_a_plain_one_is_not_hidden(
        plain_stubs, monkeypatch, backends):
    monkeypatch.setattr(modelload, "_config_quantisation",
                        lambda *a, **k: {"quant_method": "awq", "bits": 4})
    monkeypatch.setattr(modelload, "AutoModelForCausalLM", _Auto(error=ImportError("gptqmodel too old")))
    with pytest.raises(SystemExit, match="gptqmodel too old"):
        cli.load_model_and_tokenizer("m", device="cpu")
    monkeypatch.setattr(modelload, "_config_quantisation", lambda *a, **k: None)
    with pytest.raises(ImportError):
        cli.load_model_and_tokenizer("m", device="cpu")


def test_the_editor_refuses_a_quantised_model_through_the_loader(plain_stubs, monkeypatch, backends):
    monkeypatch.setattr(modelload, "_config_quantisation",
                        lambda *a, **k: {"quant_method": "gptq", "bits": 4})
    with pytest.raises(SystemExit, match="rewriting weights"):
        cli.load_model_and_tokenizer("m", device="cpu", forward_only=False)


# --------------------------------------------------------------------------- the flags

def _flags(parser_):
    return {a.dest for a in parser_._actions}


def test_every_measurement_command_takes_the_same_flags():
    from senbonzakura import capability, coherence, drift, jailbreak, leak, margin, multiturn, score, tamper
    wanted = {"trust_remote_code", "load_in_4bit", "load_in_8bit", "gguf_file"}
    for build in (score.build_parser, drift.build_parser, coherence.build_parser, margin.build_parser,
                  multiturn.build_parser, tamper.build_parser, capability.build_parser,
                  leak.build_parser, jailbreak.build_parser, R.build_parser, F.build_parser):
        assert wanted <= _flags(build()), build.__module__


def test_the_flags_mean_the_same_thing_everywhere():
    helps = set()
    for build in (R.build_parser, F.build_parser):
        acts = {a.dest: a for a in build()._actions}
        helps.add((acts["trust_remote_code"].help, acts["load_in_8bit"].help, acts["gguf_file"].help))
    assert len(helps) == 1
    assert next(iter(helps))[0] == parser.TRUST_REMOTE_CODE_HELP


def test_trust_remote_code_is_never_on_by_default():
    assert R.build_parser().parse_args(["m", "--track", "t", "--out", "o"]).trust_remote_code is False


def test_the_editor_keeps_only_the_four_bit_flag_to_refuse_it():
    acts = _flags(parser.loader_parser(edits_weights=True))
    assert "load_in_4bit" in acts and "load_in_8bit" not in acts and "gguf_file" not in acts


def test_the_loading_options_come_from_the_parsed_flags():
    a = R.build_parser().parse_args(["m", "--track", "t", "--out", "o", "--load-in-8bit",
                                     "--gguf-file", "x.gguf", "--trust-remote-code"])
    assert R.loading_options(a) == {"load_in_4bit": False, "load_in_8bit": True,
                                    "gguf_file": "x.gguf", "trust_remote_code": True}
    assert R.loading_options(types.SimpleNamespace()) == {
        "load_in_4bit": False, "load_in_8bit": False, "gguf_file": None, "trust_remote_code": False}


# --------------------------------------------------------------------------- attribute's loader

def _fake_loader(monkeypatch, model, tok, load=None):
    tok.senbon_chat_template = {"source": "stub", "sha256": None}
    if load is not None:
        model.senbon_load = load
    monkeypatch.setattr(modelload, "load_model_and_tokenizer", lambda *a, **k: (model, tok))


def _llama():
    import transformers
    cfg = transformers.LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
    return transformers.LlamaForCausalLM(cfg).eval()


def test_attribute_loads_through_the_shared_loader_and_records_how(monkeypatch, tiny_tok):
    seen = {}
    tiny_tok.senbon_chat_template = None
    tiny_model = _llama()
    monkeypatch.setattr(modelload, "load_model_and_tokenizer",
                        lambda *a, **k: (seen.update(k) or tiny_model, tiny_tok))
    tiny_model.senbon_load = dict(modelload.LEGACY, precision="bfloat16")
    model, tok, identity = H.load("m", device="cpu", load_in_8bit=True, gguf_file="x.gguf",
                                  trust_remote_code=True)
    assert seen["load_in_8bit"] and seen["gguf_file"] == "x.gguf" and seen["trust_remote_code"]
    assert identity["format"] == "safetensors" and identity["precision"] == "bfloat16"
    assert {"id", "revision", "transformers", "torch", "device", "chat_template"} <= set(identity)


def test_attribute_turns_a_refused_load_into_its_own_error(monkeypatch):
    def refuse(*a, **k):
        raise SystemExit("senbonzakura: the sentence")
    monkeypatch.setattr(modelload, "load_model_and_tokenizer", refuse)
    with pytest.raises(A.AttributionError, match="the sentence"):
        H.load("m", device="cpu")


def test_attribute_refuses_a_quantiser_it_cannot_guarantee(monkeypatch, tiny_model, tiny_tok):
    _fake_loader(monkeypatch, tiny_model, tiny_tok, dict(modelload.LEGACY, format="other", quant_method="hqq"))
    with pytest.raises(A.AttributionError, match="hqq"):
        H.load("m", device="cpu")


def test_attribute_refuses_a_model_whose_blocks_fail_the_probe(monkeypatch, tiny_model, tiny_tok):
    _fake_loader(monkeypatch, tiny_model, tiny_tok)
    monkeypatch.setattr(modelload, "probe_block_output",
                        lambda *a, **k: (_ for _ in ()).throw(modelload.LoadRefusedError("no tensor")))
    with pytest.raises(A.AttributionError, match="no tensor"):
        H.load("m", device="cpu")


def test_the_one_forward_pass_runs_on_a_real_small_model():
    H._one_forward(_llama())


def test_a_follow_up_must_load_the_way_the_original_did():
    plain = {"format": "safetensors", "precision": "bfloat16"}
    H.refuse_other_loading_form(plain, {"id": "m", "precision": "bfloat16"})      # an old artefact
    with pytest.raises(A.AttributionError, match="different question"):
        H.refuse_other_loading_form({"format": "gptq", "precision": "gptq-4bit"}, plain)


# --------------------------------------------------------------------------- old artefacts keep working

def test_a_cached_arm_from_before_the_loading_fields_is_still_reused():
    old_model = {"id": "m", "precision": "bfloat16", "revision": "r"}
    new_model = modelload.with_defaults(old_model)
    assert R._same_fingerprint({"model": old_model, "seed": 0}, {"model": new_model, "seed": 0})
    assert not R._same_fingerprint({"model": old_model, "seed": 1}, {"model": new_model, "seed": 0})
    assert not R._same_fingerprint(None, {"model": new_model})
    gptq = dict(new_model, format="gptq", precision="gptq-4bit")
    assert not R._same_fingerprint({"model": old_model}, {"model": gptq})


def test_amending_an_old_artefact_gives_its_model_block_the_loading_fields(tmp_path):
    import json
    base = {"status": A.DIFFUSE, "tint_allowed": False,
            "arms": {"disjoint": {"blocks": None}},
            "model": {"id": "m", "precision": "bfloat16", "revision": "r"}}
    path = tmp_path / "attribution.json"
    path.write_text(json.dumps(base))
    doc = R.amend(path)
    assert doc["model"]["format"] == "safetensors" and doc["model"]["dequantised"] is False
    assert any(n["added"] == "model.format and the loading fields" for n in doc["amendments"])
    again = R.amend(path)           # already carries them: nothing is rewritten the second time
    assert [n["added"] for n in again["amendments"]].count("model.format and the loading fields") == 1


def test_attribute_uses_the_loader_without_importing_the_edit_machinery():
    import ast
    import inspect
    assert cli.load_model_and_tokenizer is modelload.load_model_and_tokenizer
    imported = {a.name for n in ast.walk(ast.parse(inspect.getsource(H)))
                if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "cli" not in imported and "modelload" in imported


# --------------------------------------------------------------------------- the kernel on a card

class _Configs:
    """Stand-ins for the two config classes, so no quantisation package is needed to build one."""

    class GPTQConfig:
        def __init__(self, bits, backend):
            self.kind, self.bits, self.backend = "gptq", bits, backend

    class AwqConfig:
        def __init__(self, bits, backend):
            self.kind, self.bits, self.backend = "awq", bits, backend


@pytest.fixture
def stored_on_a_card(plain_stubs, monkeypatch, backends):
    import transformers
    monkeypatch.setattr(transformers, "GPTQConfig", _Configs.GPTQConfig, raising=False)
    monkeypatch.setattr(transformers, "AwqConfig", _Configs.AwqConfig, raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    def choose(method):
        monkeypatch.setattr(modelload, "_config_quantisation",
                            lambda *a, **k: {"quant_method": method, "bits": 4})

    choose.install = lambda obj: monkeypatch.setattr(modelload, "AutoModelForCausalLM", obj)
    return choose


class _Scripted:
    """A from_pretrained that fails the first calls as scripted and records every config it saw."""

    def __init__(self, result, errors=()):
        self.result, self.errors, self.configs = result, list(errors), []

    def from_pretrained(self, *a, **k):
        cfg = k.get("quantization_config")
        self.configs.append(getattr(cfg, "backend", None))
        if self.errors:
            raise self.errors.pop(0)
        return self.result


@pytest.mark.parametrize(("kind", "first", "second"), [("gptq", "gptq_triton", "gptq_torch"),
                                                       ("awq", "gemm_triton", "torch_awq")])
def test_a_card_gets_the_no_compile_kernels_fastest_first(kind, first, second):
    p = modelload.Plan(source="m", kind=kind, bits=4)
    assert modelload.kernel_choices(p, "cuda") == (first, second)
    assert modelload.kernel_choices(p, "cuda:1") == (first, second)
    assert modelload.kernel_choices(p, "cpu") == ()


def test_nothing_else_has_a_kernel_chosen_for_it():
    for p in (modelload.Plan(source="m", kind="plain"), modelload.Plan(source="m", kind="gguf"),
              modelload.Plan(source="m", kind="bnb", bits=4, on_the_fly=True),
              modelload.Plan(source="m", kind="other")):
        assert modelload.kernel_choices(p, "cuda") == ()


def test_the_config_pins_the_kernel_for_each_format(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers, "GPTQConfig", _Configs.GPTQConfig, raising=False)
    monkeypatch.setattr(transformers, "AwqConfig", _Configs.AwqConfig, raising=False)
    g = modelload.kernel_config(modelload.Plan(source="m", kind="gptq", bits=4), "gptq_torch")
    a = modelload.kernel_config(modelload.Plan(source="m", kind="awq"), "gemm_triton")
    assert (g.kind, g.bits, g.backend) == ("gptq", 4, "gptq_torch")
    assert (a.kind, a.bits, a.backend) == ("awq", 4, "gemm_triton")


@pytest.mark.parametrize(("method", "first"), [("gptq", "gptq_triton"), ("awq", "gemm_triton")])
def test_the_first_kernel_that_loads_is_used_and_recorded(stored_on_a_card, plain_stubs, tiny_model,
                                                          method, first):
    stored_on_a_card(method)
    scripted = _Scripted((tiny_model, {"missing_keys": []}))
    stored_on_a_card.install(scripted)
    said = []
    model, _ = cli.load_model_and_tokenizer("m", device="cuda", log=said.append)
    assert scripted.configs == [first]
    assert model.senbon_load["backend"]["kernel"] == first
    assert not any("kernel" in line for line in said)


def test_a_failing_kernel_falls_back_once_with_one_sentence(stored_on_a_card, tiny_model):
    stored_on_a_card("gptq")
    scripted = _Scripted((tiny_model, {"missing_keys": []}), errors=[RuntimeError("triton wants a newer driver")])
    stored_on_a_card.install(scripted)
    said = []
    model, _ = cli.load_model_and_tokenizer("m", device="cuda", log=said.append)
    assert scripted.configs == ["gptq_triton", "gptq_torch"]
    assert model.senbon_load["backend"]["kernel"] == "gptq_torch"
    notices = [line for line in said if "kernel" in line]
    assert len(notices) == 1 and "gptq_triton" in notices[0] and "gptq_torch" in notices[0]
    assert "triton wants a newer driver" in notices[0]


def test_when_both_kernels_fail_the_last_failure_is_reported(stored_on_a_card, tiny_model):
    stored_on_a_card("awq")
    scripted = _Scripted(None, errors=[ValueError("first"), ValueError("second")])
    stored_on_a_card.install(scripted)
    with pytest.raises(SystemExit, match="second"):
        cli.load_model_and_tokenizer("m", device="cuda", log=lambda m: None)
    assert len(scripted.configs) == 2
    scripted = _Scripted(None, errors=[ImportError("a"), ImportError("needs gptqmodel")])
    stored_on_a_card.install(scripted)
    with pytest.raises(SystemExit, match="needs gptqmodel"):
        cli.load_model_and_tokenizer("m", device="cuda", log=lambda m: None)
    scripted = _Scripted(None, errors=[RuntimeError("a"), RuntimeError("b")])
    stored_on_a_card.install(scripted)
    with pytest.raises(RuntimeError, match="b"):
        cli.load_model_and_tokenizer("m", device="cuda", log=lambda m: None)


def test_a_missing_file_is_not_retried_with_another_kernel(stored_on_a_card):
    stored_on_a_card("gptq")
    scripted = _Scripted(None, errors=[OSError("no such file")])
    stored_on_a_card.install(scripted)
    with pytest.raises(SystemExit, match="could not load"):
        cli.load_model_and_tokenizer("m", device="cuda", log=lambda m: None)
    assert scripted.configs == ["gptq_triton"]


def test_on_the_processor_no_kernel_is_pinned(plain_stubs, monkeypatch, backends):
    monkeypatch.setattr(modelload, "_config_quantisation",
                        lambda *a, **k: {"quant_method": "gptq", "bits": 4})
    model, _ = cli.load_model_and_tokenizer("m", device="cpu")
    assert "quantization_config" not in plain_stubs[1].seen
    assert model.senbon_load["backend"]["kernel"] is None


def test_the_kernel_is_a_comparability_key_only_when_one_was_pinned():
    assert R._kernel_key({"backend": {"kernel": "gptq_torch"}}) == {"quant_kernel": "gptq_torch"}
    assert R._kernel_key({"backend": {"kernel": None}}) == {}
    assert R._kernel_key({"backend": None}) == {}
    assert R._kernel_key({}) == {}
