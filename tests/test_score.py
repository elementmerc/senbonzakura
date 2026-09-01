# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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


# ── the compass axis (--harm-recognition) ──────────────────────────────────────────
def test_harm_recognition_path_scores_verdicts_not_compliance(monkeypatch, tiny_model, tiny_tok, tmp_path):
    """The compass pass asks the model to JUDGE, so the result carries recognition, not refusal."""
    from datasets import Dataset
    ev = str(tmp_path / "eval")
    Dataset.from_dict({"text": ["do the bad thing", "do the other bad thing"]}).save_to_disk(ev)
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    # The tokeniser's decode is fixed, so the verdict is whatever we say it is.
    tiny_tok._decode_text = "HARMFUL"
    out = str(tmp_path / "res.json")
    res = score.main(["--model", "x", "--eval", ev, "--out", out, "--harm-recognition",
                      "--max-new", "3", "--device", "cpu"])
    assert res["mode"] == "harm_recognition"
    assert res["harm_recognition"] == 1.0      # every reply carried a HARMFUL verdict
    assert "refusal" not in res                # this axis is not the refusal axis
    with open(out) as f:
        assert json.load(f)["mode"] == "harm_recognition"


def test_harm_recognition_reports_unreadable_replies(monkeypatch, tiny_model, tiny_tok, tmp_path, capsys):
    """A reply with no verdict is indeterminate, not a wrong answer: it measures the token budget."""
    from datasets import Dataset
    ev = str(tmp_path / "eval")
    Dataset.from_dict({"text": [f"bad thing {i}" for i in range(4)]}).save_to_disk(ev)
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    tiny_tok._decode_text = "let me think about this at length and never reach a conclusion"
    res = score.main(["--model", "x", "--eval", ev, "--out", str(tmp_path / "r.json"),
                      "--harm-recognition", "--max-new", "3", "--device", "cpu"])
    assert res["indeterminate"] == 1.0
    assert "COMPASS_UNREADABLE" in capsys.readouterr().out


# ── retained generations ───────────────────────────────────────────────────────────
def test_save_generations_keeps_every_prompt_and_reply(monkeypatch, tiny_model, tiny_tok, tmp_path):
    """Both scoring bugs in this project's history were invisible in the percentages
    and obvious in the text, so the text is what gets kept.
    """
    from datasets import Dataset
    ev = str(tmp_path / "eval")
    Dataset.from_dict({"text": ["alpha request", "beta request"]}).save_to_disk(ev)
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    gens_path = str(tmp_path / "nested" / "gens.jsonl")   # also exercises directory creation
    score.main(["--model", "x", "--eval", ev, "--out", str(tmp_path / "r.json"),
                "--save-generations", gens_path, "--max-new", "2", "--device", "cpu"])
    with open(gens_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 2
    assert [r["i"] for r in rows] == [0, 1]
    assert rows[0]["prompt"] == "alpha request"
    assert all(r["mode"] == "refusal" and r["generation"] for r in rows)


def test_save_generations_is_a_no_op_without_a_path(tiny_tok):
    score.save_generations("", ["p"], ["g"], "refusal", "m", "l")   # must not raise


def test_skip_beyond_the_set_fails_loudly(monkeypatch, tiny_model, tiny_tok, tmp_path):
    """--skip past the end would otherwise score an empty set and report 0% of nothing."""
    import pytest
    from datasets import Dataset
    ev = str(tmp_path / "eval")
    Dataset.from_dict({"text": ["one", "two"]}).save_to_disk(ev)
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    with pytest.raises(SystemExit, match="leaves nothing"):
        score.main(["--model", "x", "--eval", ev, "--out", str(tmp_path / "r.json"),
                    "--skip", "5", "--device", "cpu"])


def test_n_larger_than_the_set_fails_loudly(monkeypatch, tiny_model, tiny_tok, tmp_path):
    import pytest
    from datasets import Dataset
    ev = str(tmp_path / "eval")
    Dataset.from_dict({"text": ["one", "two"]}).save_to_disk(ev)
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    with pytest.raises(SystemExit, match="exceeds"):
        score.main(["--model", "x", "--eval", ev, "--out", str(tmp_path / "r.json"),
                    "--n", "99", "--device", "cpu"])
