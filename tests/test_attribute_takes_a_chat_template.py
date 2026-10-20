# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`attribute` takes `--chat-template` and records the prompt format, the way `score` does.

A model that ships no chat template used to fail in `attribute` deep inside the tokeniser, and a model that did have
one was measured with no record of which format it was. The flag, the validation and the provenance are now the same
code `score` uses (`cli.ensure_chat_template`), and the provenance lands in the artefact's model block.
"""
import hashlib
import json

import pytest
import transformers
from test_a_model_with_missing_weights_is_refused import models  # noqa: F401
from test_attribution_hf import StubModel, StubTok, _parts
from tokenizers import Tokenizer, models as tk_models, pre_tokenizers

from senbonzakura import attribution as A
from senbonzakura import attribution_hf as H
from senbonzakura import attribution_run as R

TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")


def _tokenizer(template=None):
    core = Tokenizer(tk_models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", eos_token="[UNK]")
    if template is not None:
        tok.chat_template = template
    return tok


def _serve(monkeypatch, tok):
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)


def test_the_flag_is_there_and_defaults_to_none_given():
    parser = R.build_parser()
    assert parser.parse_args(["m", "--track", "t", "--out", "o"]).chat_template == ""
    assert parser.parse_args(["m", "--track", "t", "--out", "o", "--chat-template", "f.jinja"]).chat_template == "f.jinja"


def test_a_models_own_template_is_checked_and_recorded(models, monkeypatch):  # noqa: F811
    _serve(monkeypatch, _tokenizer(TEMPLATE))
    _model, tok, identity = H.load(models["intact"], device="cpu")
    record = identity["chat_template"]
    assert record["source"] == "tokenizer" and record["sha256"] == hashlib.sha256(TEMPLATE.encode()).hexdigest()[:16]
    assert tok.senbon_chat_template is record


def test_a_supplied_template_is_used_and_recorded_with_its_digest(models, monkeypatch, tmp_path):  # noqa: F811
    _serve(monkeypatch, _tokenizer())
    path = tmp_path / "t.jinja"
    path.write_text(TEMPLATE, encoding="utf-8")
    _model, tok, identity = H.load(models["intact"], device="cpu", chat_template=str(path))
    assert identity["chat_template"]["source"] == str(path)
    assert identity["chat_template"]["sha256"] == hashlib.sha256(TEMPLATE.encode()).hexdigest()[:16]
    assert "<|assistant|>" in tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                                      add_generation_prompt=True)


def test_the_provenance_can_be_written_into_the_artefact(models, monkeypatch):  # noqa: F811
    _serve(monkeypatch, _tokenizer(TEMPLATE))
    _model, _tok, identity = H.load(models["intact"], device="cpu")
    assert json.loads(json.dumps(identity))["chat_template"]["source"] == "tokenizer"


def test_a_model_with_no_template_and_none_supplied_is_refused_in_a_sentence(models, monkeypatch):  # noqa: F811
    _serve(monkeypatch, _tokenizer())
    with pytest.raises(A.AttributionError, match="ships no chat template"):
        H.load(models["intact"], device="cpu")


@pytest.mark.parametrize(("content", "fragment"), [("", "is empty"), ("   ", "is empty")])
def test_an_empty_supplied_template_is_refused(models, monkeypatch, tmp_path, content, fragment):  # noqa: F811
    _serve(monkeypatch, _tokenizer())
    path = tmp_path / "t.jinja"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(A.AttributionError, match=fragment):
        H.load(models["intact"], device="cpu", chat_template=str(path))


def test_a_template_file_that_is_not_there_is_refused(models, monkeypatch, tmp_path):  # noqa: F811
    _serve(monkeypatch, _tokenizer())
    with pytest.raises(A.AttributionError, match="could not read --chat-template"):
        H.load(models["intact"], device="cpu", chat_template=str(tmp_path / "nowhere.jinja"))


def test_a_template_that_renders_nothing_usable_is_refused(models, monkeypatch, tmp_path):  # noqa: F811
    _serve(monkeypatch, _tokenizer())
    path = tmp_path / "t.jinja"
    path.write_text("{{ 1 + 'a' }}", encoding="utf-8")
    with pytest.raises(A.AttributionError, match="did not produce a usable prompt"):
        H.load(models["intact"], device="cpu", chat_template=str(path))


def test_the_command_passes_the_flag_to_the_loader_and_the_artefact_carries_the_record(monkeypatch, tmp_path):
    seen = {}
    record = {"source": "supplied.jinja", "sha256": "0123456789abcdef", "audit": {}}
    identity = {"id": "stub", "revision": "abc", "precision": "bfloat16", "chat_template": record,
                "torch": "2.9.1", "transformers": "5.20.1"}

    def load(*_a, **kwargs):
        seen.update(kwargs)
        return StubModel(), StubTok(), identity

    monkeypatch.setattr(H, "load", load)
    monkeypatch.setattr(H, "build_partitions", lambda *_a, **_k: (_parts(confirm=True), {"measure": {"n": 40}}))
    monkeypatch.setattr(A, "selftest", lambda **_k: {"passed": True, "cases": {}})
    args = R.build_parser().parse_args(["stub", "--track", "t", "--out", str(tmp_path), "--n-measure", "40",
                                        "--batch", "8", "--chat-template", "supplied.jinja"])
    R.run(args, log=lambda *_: None)
    assert seen["chat_template"] == "supplied.jinja"
    written = json.loads((tmp_path / "attribution.json").read_text())
    assert written["model"]["chat_template"] == record
