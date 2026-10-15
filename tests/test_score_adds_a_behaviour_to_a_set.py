# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`score --behaviour` joins a set instead of replacing the file, and refuses before the model loads.

Two separate properties, and the second is the one that gets dropped. Every refusal `--behaviour`
can decide from the command line is decided before `load_model_and_tokenizer` is reached, because
finding out after a multi-gigabyte load that the pass cannot join a set is what makes an operator
stop passing the flag. The fake loader below raises, so a test that reaches it fails loudly rather
than quietly loading nothing.
"""
import json

import pytest
from datasets import Dataset

from senbonzakura import behaviourset, cli, score


@pytest.fixture
def evalset(tmp_path):
    path = str(tmp_path / "eval")
    Dataset.from_dict({"text": ["do the bad thing", "what is the capital city"]}).save_to_disk(path)
    return path


@pytest.fixture
def never_loads(monkeypatch):
    """A loader that fails the test if it is reached, so "refused early" is actually asserted."""
    def refuse(*a, **k):
        raise AssertionError("the model was loaded before --behaviour was checked")

    monkeypatch.setattr(cli, "load_model_and_tokenizer", refuse)


def run(evalset, out, *extra, model="org/tiny"):
    return score.main(["--model", model, "--eval", evalset, "--out", str(out),
                       "--max-new", "3", "--device", "cpu", *extra])


# ── the refusals, all of them before the model loads ──────────────────────────────────────────────

def test_a_key_shaped_wrong_is_refused_before_the_model_loads(evalset, tmp_path, never_loads):
    with pytest.raises(SystemExit, match="not a usable behaviour key"):
        run(evalset, tmp_path / "s.json", "--behaviour", "NOT A KEY")


def test_an_unregistered_key_is_refused_before_the_model_loads(evalset, tmp_path, never_loads):
    with pytest.raises(SystemExit, match="not a behaviour a card can draw"):
        run(evalset, tmp_path / "s.json", "--behaviour", "telepathy")


def test_the_sweep_stamps_nothing_so_it_cannot_join_a_set(evalset, tmp_path, never_loads):
    with pytest.raises(SystemExit, match="cannot be used with --length-sweep"):
        run(evalset, tmp_path / "s.json", "--behaviour", "refusal", "--length-sweep")


def test_the_compass_stamps_nothing_so_it_cannot_join_a_set(evalset, tmp_path, never_loads):
    with pytest.raises(SystemExit, match="cannot be used with --harm-recognition"):
        run(evalset, tmp_path / "s.json", "--behaviour", "refusal", "--harm-recognition")


def test_the_gguf_path_stamps_nothing_so_it_cannot_join_a_set(evalset, tmp_path, never_loads):
    with pytest.raises(SystemExit, match="cannot be used with a GGUF"):
        run(evalset, tmp_path / "s.json", "--behaviour", "refusal",
            model=str(tmp_path / "model.gguf"))


def test_no_behaviour_flag_leaves_the_check_silent(evalset, tmp_path, monkeypatch, tiny_model,
                                                   tiny_tok):
    """The flag is opt-in: without it the command writes exactly the file it always wrote."""
    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    out = tmp_path / "s.json"
    run(evalset, out)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert behaviourset.SCHEMA_FIELD not in doc
    assert "refusal" in doc


# ── the merge, end to end ─────────────────────────────────────────────────────────────────────────

def test_the_pass_joins_a_set_and_names_the_roster(evalset, tmp_path, monkeypatch, tiny_model,
                                                   tiny_tok, capsys):
    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    out = tmp_path / "score.json"
    run(evalset, out, "--behaviour", "refusal")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc[behaviourset.SCHEMA_FIELD] == behaviourset.SCHEMA
    assert doc[behaviourset.ROSTER_KEY] == ["refusal"]
    assert "refusal_rate.senbonzakura-ruler" in doc["metrics"]
    assert "BEHAVIOUR_SET" in capsys.readouterr().out


def test_re_running_the_same_behaviour_is_idempotent(evalset, tmp_path, monkeypatch, tiny_model,
                                                     tiny_tok):
    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    out = tmp_path / "score.json"
    run(evalset, out, "--behaviour", "refusal")
    first = out.read_bytes()
    run(evalset, out, "--behaviour", "refusal")
    assert out.read_bytes() == first


def test_the_card_reads_a_set_exactly_as_it_reads_a_single_result(evalset, tmp_path, monkeypatch,
                                                                  tiny_model, tiny_tok):
    """The reason the merged shape was chosen: the card's own reader needed no edit at all."""
    from senbonzakura.workbench.card import behaviours

    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    run(evalset, tmp_path / "score.json", "--behaviour", "refusal")
    measured = behaviours.measure_refusal(tmp_path)
    # Two replies is below the reportable floor, so the honest answer is a named refusal rather
    # than a rate. What matters here is that the reader got to the stamped block at all.
    assert "below the floor" in measured.figure.reason
    assert behaviours.read_model(tmp_path) == {"name": "org/tiny", "revision": None}


def test_a_set_declaring_a_schema_loses_the_no_schema_caveat(evalset, tmp_path, monkeypatch,
                                                             tiny_model, tiny_tok):
    from senbonzakura.workbench.core import artefacts

    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    run(evalset, tmp_path / "score.json", "--behaviour", "refusal")
    read = artefacts.read_json(tmp_path, "score.json")
    assert read.state == "present"
    assert read.schema_declared == behaviourset.SCHEMA
    assert artefacts.NO_SCHEMA_CAVEAT not in artefacts.schema_caveats(read)


def test_a_result_that_cannot_join_the_set_is_kept_rather_than_thrown_away(
        evalset, tmp_path, monkeypatch, tiny_model, tiny_tok):
    """A run already paid for must not be lost because the file beside it was wrong."""
    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda *a, **k: (tiny_model, tiny_tok))
    out = tmp_path / "score.json"
    out.write_text(json.dumps({"schema": "something-else/1"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a behaviour set"):
        run(evalset, out, "--behaviour", "refusal")
    rescued = json.loads((tmp_path / "score.json.refusal.json").read_text(encoding="utf-8"))
    assert rescued["n"] == 2
    assert "refusal_rate.senbonzakura-ruler" in rescued["metrics"]
    # And the file that was already there is untouched, which is the whole point.
    assert json.loads(out.read_text(encoding="utf-8")) == {"schema": "something-else/1"}
