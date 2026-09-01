# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for the coherence ruler (senbonzakura.drift).

This module exists because two tools reported a KL divergence a hundredfold apart and neither
number could be put in a column with the other. It is the instrument that decides every published
coherence figure, so the things guarded here are the ones that would make a whole table wrong in
the same direction without anything looking broken:

- the cache is refused unless it was built for THIS base, THESE prompts, THIS chat template and
  THIS batch size, because a cache keyed on nothing scores ten models against the wrong reference;
- the divergence runs base against candidate and not the other way round;
- a row count that disagrees with the prompt count is refused rather than compared.
"""
import json

import pytest
import torch

from senbonzakura import drift


# ── the prompt file ───────────────────────────────────────────────────────────────────
def test_blank_lines_are_dropped_and_whitespace_stripped(tmp_path):
    p = tmp_path / "prompts.txt"
    p.write_text("  first  \n\n second\n   \n", encoding="utf-8")
    assert drift.read_prompts(str(p)) == ["first", "second"]


def test_a_prompt_file_with_nothing_in_it_is_refused(tmp_path):
    """Measuring on zero prompts would report a drift of nan and print it as a result."""
    p = tmp_path / "empty.txt"
    p.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        drift.read_prompts(str(p))
    assert "no prompts" in str(e.value)


# ── the fingerprint, which is what the cache is allowed to be reused for ──────────────
def test_the_fingerprint_moves_with_everything_it_covers():
    base = drift.fingerprint("base-a", ["one", "two"], "tpl")
    assert base == drift.fingerprint("base-a", ["one", "two"], "tpl")
    assert base != drift.fingerprint("base-b", ["one", "two"], "tpl")
    assert base != drift.fingerprint("base-a", ["one", "three"], "tpl")
    # The template is in the key because the same model gives different first-token
    # distributions under different chat formats, and this project has already published a
    # number selected under one prompt format and reported under another.
    assert base != drift.fingerprint("base-a", ["one", "two"], "other-tpl")


def test_prompt_order_and_boundaries_change_the_fingerprint():
    """Two prompt sets with the same characters are not the same prompt set."""
    assert (drift.fingerprint("b", ["ab", "c"], "t")
            != drift.fingerprint("b", ["a", "bc"], "t"))
    assert (drift.fingerprint("b", ["one", "two"], "t")
            != drift.fingerprint("b", ["two", "one"], "t"))


# ── the divergence itself ─────────────────────────────────────────────────────────────
def _lp(rows):
    return torch.log(torch.tensor(rows, dtype=torch.float))


def test_identical_distributions_have_zero_divergence():
    lp = _lp([[0.5, 0.5], [0.25, 0.75]])
    assert drift.kl(lp, lp) == pytest.approx(0.0, abs=1e-6)


def test_the_divergence_matches_the_arithmetic_by_hand():
    base = _lp([[0.5, 0.5]])
    cand = _lp([[0.25, 0.75]])
    expected = (0.5 * torch.log(torch.tensor(0.5 / 0.25))
                + 0.5 * torch.log(torch.tensor(0.5 / 0.75)))
    assert drift.kl(base, cand) == pytest.approx(float(expected), abs=1e-6)


def test_the_divergence_is_averaged_over_prompts_not_summed():
    """Otherwise the figure would depend on how many prompts were scored."""
    base = _lp([[0.5, 0.5], [0.5, 0.5]])
    cand = _lp([[0.25, 0.75], [0.25, 0.75]])
    one = drift.kl(base[:1], cand[:1])
    assert drift.kl(base, cand) == pytest.approx(one, abs=1e-6)


def test_the_direction_is_base_against_candidate():
    """KL is not symmetric, and the reverse direction rewards an edited model for collapsing
    onto a few tokens the original also liked.
    """
    base = _lp([[0.9, 0.1]])
    cand = _lp([[0.5, 0.5]])
    assert drift.kl(base, cand) != pytest.approx(drift.kl(cand, base), abs=1e-3)


# ── the forward pass ──────────────────────────────────────────────────────────────────
class _Enc:
    def to(self, _device):
        return self


class _FakeTok:
    chat_template = "tpl"

    def __call__(self, chunk, **kw):
        self.last_chunk = list(chunk)
        return _Enc()


class _FakeModel:
    def __init__(self, vocab=3):
        self.device = torch.device("cpu")
        self.vocab = vocab
        self.calls = 0

    def __call__(self, *a, **kw):     # pragma: no cover - the logits helper is stubbed
        raise AssertionError("the logits helper should be stubbed in these tests")


@pytest.fixture
def stub_forward(monkeypatch):
    """Render and last-token-logits stubbed, so the batching is what is under test."""
    seen = {"batches": []}

    monkeypatch.setattr(drift, "render_chat", lambda tok, p: f"<{p}>")

    def _logits(model, enc, log):
        n = len(model_tok[1].last_chunk)
        seen["batches"].append(n)
        model.calls += 1
        return torch.zeros(n, model.vocab)

    model_tok = (_FakeModel(), _FakeTok())
    monkeypatch.setattr(drift, "last_token_logits", _logits)
    return model_tok, seen


def test_the_forward_pass_returns_one_row_per_prompt(stub_forward):
    (model, tok), _ = stub_forward
    out = drift.first_token_logprobs(model, tok, ["a", "b", "c"], batch=2)
    assert out.shape == (3, model.vocab)
    # Uniform logits mean a uniform distribution: log(1/3) in every column.
    assert float(out[0][0]) == pytest.approx(torch.log(torch.tensor(1 / 3)), abs=1e-6)


def test_the_prompts_go_through_the_chat_template(stub_forward):
    (model, tok), _ = stub_forward
    drift.first_token_logprobs(model, tok, ["a", "b"], batch=8)
    assert tok.last_chunk == ["<a>", "<b>"]


def test_the_batch_size_is_honoured(stub_forward):
    """It is recorded in the artefact because the reduction order depends on it, so a
    comparison that changed it between models would not be one.
    """
    (model, tok), seen = stub_forward
    drift.first_token_logprobs(model, tok, ["a", "b", "c", "d", "e"], batch=2)
    assert seen["batches"] == [2, 2, 1]


# ── the cache, and what it is allowed to be reused for ────────────────────────────────
def _args(tmp_path, **kw):
    a = drift.build_parser().parse_args(
        ["--model", "cand", "--base", "base", "--prompts", str(tmp_path / "p.txt"),
         "--out", str(tmp_path / "o.json"), *[str(x) for x in kw.pop("extra", [])]])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _write_cache(path, fingerprint, batch, rows):
    torch.save({"schema": drift.CACHE_SCHEMA, "fingerprint": fingerprint,
                "batch": batch, "logprobs": torch.zeros(rows, 2)}, path)


def test_a_matching_cache_is_reused_without_loading_the_base(tmp_path, monkeypatch):
    cache = str(tmp_path / "base.pt")
    _write_cache(cache, "fp-1", 16, 4)
    monkeypatch.setattr(drift, "load_model_and_tokenizer",
                        lambda *a, **k: pytest.fail("the base was loaded despite a valid cache"))
    said = []
    out = drift.load_base_logprobs(_args(tmp_path, base_cache=cache, batch=16),
                                   ["a"], "fp-1", log=said.append)
    assert out.shape == (4, 2)
    assert any("reusing" in s for s in said)


@pytest.mark.parametrize(("cached_fp", "cached_batch", "why"), [
    ("fp-other", 16, "a different base, prompt set or chat template"),
    ("fp-1", 8, "a different batch size"),
])
def test_a_cache_built_for_a_different_question_is_rebuilt_loudly(
        tmp_path, monkeypatch, stub_forward, cached_fp, cached_batch, why):
    """Silence here is the worst case: every model gets compared to the wrong reference and
    the whole table is wrong in the same direction, which the output cannot show.
    """
    (model, tok), _ = stub_forward
    cache = str(tmp_path / "base.pt")
    _write_cache(cache, cached_fp, cached_batch, 4)
    monkeypatch.setattr(drift, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    said = []
    out = drift.load_base_logprobs(_args(tmp_path, base_cache=cache, batch=16),
                                   ["a", "b"], "fp-1", log=said.append)
    assert out.shape == (2, model.vocab), "the recomputed rows, not the cached ones"
    assert any("built for a different" in s for s in said), why


def test_the_recomputed_base_is_written_back_for_the_next_model(tmp_path, monkeypatch,
                                                               stub_forward):
    (model, tok), _ = stub_forward
    cache = str(tmp_path / "base.pt")
    monkeypatch.setattr(drift, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    drift.load_base_logprobs(_args(tmp_path, base_cache=cache, batch=4), ["a"], "fp-1",
                             log=lambda _s: None)
    doc = torch.load(cache, map_location="cpu", weights_only=False)
    assert doc["schema"] == drift.CACHE_SCHEMA
    assert doc["fingerprint"] == "fp-1" and doc["batch"] == 4


def test_no_cache_path_means_no_cache_file(tmp_path, monkeypatch, stub_forward):
    (model, tok), _ = stub_forward
    monkeypatch.setattr(drift, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    drift.load_base_logprobs(_args(tmp_path, base_cache=None, batch=4), ["a"], "fp-1",
                             log=lambda _s: None)
    assert list(tmp_path.glob("*.pt")) == []


# ── main(), end to end ────────────────────────────────────────────────────────────────
@pytest.fixture
def prompts_file(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("first\nsecond\n", encoding="utf-8")
    return p


def test_main_writes_the_result_with_what_produced_it(tmp_path, monkeypatch, prompts_file,
                                                      capsys):
    """The provenance fields are the point: the whole reason this module exists is that two KL
    figures measured on different slices were put in one column.
    """
    monkeypatch.setattr(drift, "load_model_and_tokenizer", lambda *a, **k: (_FakeModel(), _FakeTok()))
    monkeypatch.setattr(drift, "first_token_logprobs",
                        lambda m, t, prompts, batch=16, log=None: _lp([[0.5, 0.5]] * len(prompts)))
    out = str(tmp_path / "drift.json")
    res = drift.main(["--model", "cand", "--base", "base", "--prompts", str(prompts_file),
                      "--out", out, "--label", "arm-1", "--batch", "4"])

    assert res["kl"] == pytest.approx(0.0, abs=1e-6)
    assert res["label"] == "arm-1" and res["n_prompts"] == 2 and res["batch"] == 4
    assert res["prompts"] == str(prompts_file)
    assert res["base"] == "base" and res["model"] == "cand"
    assert "KL(base||candidate)" in res["instrument"]
    with open(out, encoding="utf-8") as f:
        assert json.load(f) == res
    assert "DRIFT_DONE arm-1" in capsys.readouterr().out


def test_main_refuses_a_base_that_covers_a_different_number_of_prompts(tmp_path, monkeypatch,
                                                                      prompts_file):
    """A cache from a longer prompt file would otherwise be compared row by row against this
    one, which silently pairs each model with somebody else's question.
    """
    monkeypatch.setattr(drift, "load_model_and_tokenizer", lambda *a, **k: (_FakeModel(), _FakeTok()))
    monkeypatch.setattr(drift, "load_base_logprobs",
                        lambda *a, **k: _lp([[0.5, 0.5]] * 5))
    with pytest.raises(SystemExit) as e:
        drift.main(["--model", "cand", "--base", "base", "--prompts", str(prompts_file),
                    "--out", str(tmp_path / "d.json")])
    assert "5 prompts" in str(e.value) and "2" in str(e.value)


def test_a_model_with_no_chat_template_still_fingerprints(tmp_path, monkeypatch, prompts_file):
    """`getattr(tok, "chat_template", None) or ""` is the only thing between this and a
    TypeError on a base model that has none.
    """
    class _NoTemplate(_FakeTok):
        chat_template = None

    monkeypatch.setattr(drift, "load_model_and_tokenizer",
                        lambda *a, **k: (_FakeModel(), _NoTemplate()))
    monkeypatch.setattr(drift, "first_token_logprobs",
                        lambda m, t, prompts, batch=16, log=None: _lp([[0.5, 0.5]] * len(prompts)))
    res = drift.main(["--model", "cand", "--base", "base", "--prompts", str(prompts_file),
                      "--out", str(tmp_path / "d.json")])
    assert res["fingerprint"] == drift.fingerprint("base", ["first", "second"], "")
