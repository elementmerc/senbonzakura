"""Tests for the fixed-eval scorer (senbonzakura.score)."""
import json
import os

from senbonzakura import score


def test_score_dict_helper():
    r = score.score(["I cannot help with that.", "Sure, here you go.", ""], label="x",
                    model="m", eval_path="e")
    assert r["n"] == 3
    assert 0.0 <= r["refusal"] <= 1.0
    assert r["broken"] > 0.0            # the empty string
    assert set(r) >= {"refusal", "soft_refusal", "noncompliant", "broken", "heretic"}


def test_score_main_end_to_end(monkeypatch, tiny_model, tiny_tok, tmp_path):
    from datasets import Dataset
    ev = str(tmp_path / "eval")
    Dataset.from_dict({"text": ["do the bad thing", "what is the capital city"]}).save_to_disk(ev)
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    out = str(tmp_path / "res.json")
    res = score.main(["--model", "x", "--eval", ev, "--out", out, "--max-new", "3", "--device", "cpu"])
    assert res["n"] == 2
    assert os.path.exists(out)
    with open(out) as f:
        assert "heretic" in json.load(f)


def test_score_main_respects_n(monkeypatch, tiny_model, tiny_tok, tmp_path):
    from datasets import Dataset
    ev = str(tmp_path / "eval")
    Dataset.from_dict({"text": [f"prompt {i}" for i in range(6)]}).save_to_disk(ev)
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    out = str(tmp_path / "res.json")
    res = score.main(["--model", "x", "--eval", ev, "--out", out, "--n", "3",
                      "--max-new", "2", "--device", "cpu"])
    assert res["n"] == 3
