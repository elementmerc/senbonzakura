# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for the coherence probe (senbonzakura.coherence).

The regression these guard is concrete: the neutral-passage coherence scorer used
to be a loose script that did not accept --load-in-4bit, so a bench chain that
passed the flag (the same flag senbonzakura.score accepts) died on an
unknown-argument error and reported a false "coherence failure". The canonical
module accepts the scorer's full flag set, so that class cannot recur.
"""
import json
import math

import pytest
import torch

from senbonzakura import coherence


def test_parser_accepts_load_in_4bit():
    # The flag that used to be fatal is now honoured, matching senbonzakura.score.
    a = coherence.build_parser().parse_args(
        ["--model", "m", "--out", "o.json", "--load-in-4bit", "--device", "cuda"])
    assert a.load_in_4bit is True
    a2 = coherence.build_parser().parse_args(["--model", "m", "--out", "o.json"])
    assert a2.load_in_4bit is False


def test_parser_shares_scorer_flags():
    from senbonzakura import score
    coh = {a.dest for a in coherence.build_parser()._actions}
    sco = {a.dest for a in score.build_parser()._actions}
    # coherence's flags are a subset of the scorer's, so no caller is surprised.
    assert {"model", "out", "label", "device", "load_in_4bit", "trust_remote_code"} <= coh
    assert coh <= sco | {"help"}


def test_neutral_passage_is_content_neutral():
    t = coherence.NEUTRAL.lower()
    assert len(coherence.NEUTRAL) > 800
    for w in ("refuse", "cannot help", "harmful", "weapon", "illegal"):
        assert w not in t


# ── the probe itself, and main() end to end ────────────────────────────────────────
class _LossOut:
    def __init__(self, loss):
        self.loss = loss


class _FakeLM:
    """The smallest thing coherence() actually needs: a device and a loss."""

    def __init__(self, nll=1.5):
        self.device = torch.device("cpu")
        self._nll = nll
        self.seen = None

    def __call__(self, ids, labels=None):
        self.seen = (ids, labels)
        return _LossOut(torch.tensor(self._nll))


class _FakeTok:
    def __init__(self, n_tokens=7):
        self._n = n_tokens

    def __call__(self, text, return_tensors="pt"):
        class _E:
            pass
        e = _E()
        e.input_ids = torch.arange(self._n).unsqueeze(0)
        return e


def test_coherence_reports_perplexity_as_exp_of_the_loss():
    m, t = _FakeLM(nll=1.5), _FakeTok(n_tokens=7)
    r = coherence.coherence(m, t)
    assert r["nll"] == pytest.approx(1.5)
    assert r["ppl"] == pytest.approx(math.exp(1.5))
    assert r["n_tokens"] == 7
    # Scoring a passage means labels are the inputs: this is a language-model loss,
    # not a classification one, and passing labels=None would silently return no loss.
    ids, labels = m.seen
    assert labels is not None and torch.equal(ids, labels)


def test_coherence_main_writes_the_result(monkeypatch, tmp_path):
    monkeypatch.setattr(coherence, "load_model_and_tokenizer",
                        lambda *a, **k: (_FakeLM(nll=2.0), _FakeTok(n_tokens=5)))
    out = str(tmp_path / "coh.json")
    res = coherence.main(["--model", "m", "--out", out, "--label", "post", "--device", "cpu"])
    assert res["label"] == "post" and res["model"] == "m"
    assert res["ppl"] == pytest.approx(math.exp(2.0))
    with open(out) as f:
        on_disk = json.load(f)
    assert on_disk == res      # what is printed is what is written


def test_coherence_main_accepts_the_scorers_loader_flags(monkeypatch, tmp_path):
    """The regression this module was fixed for: a bench chain passing --load-in-4bit
    died on an unknown argument and reported it as a coherence failure.
    """
    seen = {}

    def _loader(model, device=None, load_in_4bit=False, trust_remote_code=False, **kw):
        seen.update(load_in_4bit=load_in_4bit, trust_remote_code=trust_remote_code)
        return _FakeLM(), _FakeTok()

    monkeypatch.setattr(coherence, "load_model_and_tokenizer", _loader)
    coherence.main(["--model", "m", "--out", str(tmp_path / "c.json"), "--device", "cpu",
                    "--load-in-4bit", "--trust-remote-code"])
    assert seen == {"load_in_4bit": True, "trust_remote_code": True}
