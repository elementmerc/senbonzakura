# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`score` takes `--revision` and records which revision its figures are about, the way `attribute` does.

Before this a score file named the model and nothing about which version of it, so a figure could not be tied to the
attribution of the same model by anything but a name. The value is the commit transformers resolved, else the one
given; a local folder has no commit, so it is recorded as given. The loader is handed the value so that a Hub model is
the revision that is recorded.
"""
import json

import pytest
from datasets import Dataset

from senbonzakura import cli, modelload, score


def _eval(tmp_path):
    path = str(tmp_path / "eval")
    Dataset.from_dict({"text": ["do the bad thing", "what is the capital city"]}).save_to_disk(path)
    return path


def _run(monkeypatch, tiny_model, tiny_tok, tmp_path, *flags, seen=None):
    def load(*_a, **kwargs):
        if seen is not None:
            seen.update(kwargs)
        return tiny_model, tiny_tok

    monkeypatch.setattr(cli, "load_model_and_tokenizer", load)
    out = tmp_path / "res.json"
    score.main(["--model", "x", "--eval", _eval(tmp_path), "--out", str(out), "--max-new", "3", "--device", "cpu",
                *flags])
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_flag_exists_and_defaults_to_none():
    parser = score.build_parser()
    assert parser.parse_args(["--model", "m", "--eval", "e", "--out", "o"]).revision is None
    assert parser.parse_args(["--model", "m", "--eval", "e", "--out", "o", "--revision", "abc"]).revision == "abc"


def test_a_given_revision_is_recorded_and_handed_to_the_loader(monkeypatch, tiny_model, tiny_tok, tmp_path):
    seen = {}
    res = _run(monkeypatch, tiny_model, tiny_tok, tmp_path, "--revision", "local-sha256:abc", seen=seen)
    assert res["revision"] == "local-sha256:abc" and seen["revision"] == "local-sha256:abc"


def test_the_commit_transformers_resolved_wins_over_the_value_given(monkeypatch, tiny_model, tiny_tok, tmp_path):
    tiny_model.config._commit_hash = "c" * 40
    res = _run(monkeypatch, tiny_model, tiny_tok, tmp_path, "--revision", "main")
    assert res["revision"] == "c" * 40


def test_with_nothing_given_and_nothing_resolved_the_revision_is_null_not_missing(monkeypatch, tiny_model, tiny_tok,
                                                                                  tmp_path):
    tiny_model.config._commit_hash = None
    res = _run(monkeypatch, tiny_model, tiny_tok, tmp_path)
    assert "revision" in res and res["revision"] is None


def test_the_compass_pass_records_it_too(monkeypatch, tiny_model, tiny_tok, tmp_path):
    res = _run(monkeypatch, tiny_model, tiny_tok, tmp_path, "--harm-recognition", "--revision", "r1")
    assert res["revision"] == "r1"


def test_a_length_sweep_records_it_too(monkeypatch, tiny_model, tiny_tok, tmp_path):
    from senbonzakura import lengthsweep
    monkeypatch.setattr(lengthsweep, "still_climbing", lambda *a, **k: True)
    out = tmp_path / "res.json"
    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    score.main(["--model", "x", "--eval", _eval(tmp_path), "--out", str(out), "--device", "cpu", "--length-sweep",
                "--length-max", "16", "--revision", "r2"])
    assert json.loads(out.read_text(encoding="utf-8"))["revision"] == "r2"


@pytest.mark.parametrize("flag", [[], ["--revision", "r"]])
def test_the_loader_is_called_the_same_way_when_no_revision_is_given(monkeypatch, tiny_model, tiny_tok, tmp_path,
                                                                     flag):
    seen = {}
    _run(monkeypatch, tiny_model, tiny_tok, tmp_path, *flag, seen=seen)
    assert seen["revision"] == (flag[1] if flag else None)


def test_the_loader_passes_a_revision_to_every_from_pretrained_and_a_local_folder_still_loads(tmp_path, monkeypatch):
    """The three loads take the revision only when one is given, and a local folder ignores it, as attribute's does."""
    import transformers

    calls = {}

    class Tok:
        pad_token = "x"
        eos_token = "x"
        padding_side = "left"
        chat_template = None

    def tokenizer(model_id, **kwargs):
        calls["tokenizer"] = kwargs
        return Tok()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(modelload, "AutoTokenizer", transformers.AutoTokenizer)
    monkeypatch.setattr(modelload, "ensure_chat_template", lambda *_a, **_k: {"source": "stub"})
    monkeypatch.setattr(modelload, "_refuse_without_the_checker", lambda: None)
    monkeypatch.setattr(cli.argresolve, "refuse_unusable_device", lambda *a, **k: None)
    tok = cli.load_tokenizer(str(tmp_path), revision="rev1", needs_chat_template=False)
    assert calls["tokenizer"]["revision"] == "rev1" and tok.pad_token == "x"
    cli.load_tokenizer(str(tmp_path), needs_chat_template=False)
    assert "revision" not in calls["tokenizer"]
