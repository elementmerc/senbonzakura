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


# ── the metric's identity: what was measured, on what, and in what units ───────────────
#
# WHY THESE EXIST. A coherence file used to hold `nll` and `ppl` as bare floats beside a label
# somebody typed. Nothing in it said which passage produced the number, whether a chat template
# had been applied, or which build of the tool took it, so two coherence figures could only be
# compared by someone who happened to remember all three. That is the shape of the defect that
# withdrew four published claims on 2026-08-05: arms that did different work compared as one
# measurement, with no field saying otherwise.

def _run_main(monkeypatch, tmp_path, nll=2.0, n_tokens=5):
    monkeypatch.setattr(coherence, "load_model_and_tokenizer",
                        lambda *a, **k: (_FakeLM(nll=nll), _FakeTok(n_tokens=n_tokens)))
    out = tmp_path / "coh.json"
    res = coherence.main(["--model", "m", "--out", str(out), "--device", "cpu"])
    return res, out


def test_the_written_artefact_carries_the_metric_identity(monkeypatch, tmp_path):
    """THE WIRING TEST. Driven through main(), not through the stamping helper.

    A stamping helper with no call site is a green suite, and this project has shipped three
    separate features that were never once executed. So the assertion is made against the file
    on disk, which is the only thing that outlives the run.
    """
    res, out = _run_main(monkeypatch, tmp_path)
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == res, "what is returned is what is written"

    block = on_disk["metrics"]["coherence"]
    assert block["metric"] == "coherence"
    assert block["estimator"] == "neutral-passage-nll"
    assert block["units"] == "nats-per-token"
    assert block["higher_is_better"] is False
    assert block["value"] == pytest.approx(2.0)
    assert len(block["estimator_description"]) > 30, (
        "a reader holding only the artefact has to be able to tell this apart from another "
        "coherence number taken some other way")


def test_the_identity_names_the_passage_the_number_was_taken_on(monkeypatch, tmp_path):
    """`NEUTRAL` is a constant in a source file: edit one word and every later figure moves."""
    _, out = _run_main(monkeypatch, tmp_path)
    block = json.loads(out.read_text(encoding="utf-8"))["metrics"]["coherence"]
    assert block["passage_digest"] == coherence.passage_digest()
    assert block["passage_digest"] != coherence.passage_digest(coherence.NEUTRAL + " One more."), (
        "a digest that does not move when the passage moves records nothing")


def test_the_identity_says_no_chat_template_was_applied(monkeypatch, tmp_path):
    """Stated, not inferred from an absent flag. A raw-format figure and a templated one are
    not the same measurement, and the probe deliberately renders no template at all.
    """
    _, out = _run_main(monkeypatch, tmp_path)
    block = json.loads(out.read_text(encoding="utf-8"))["metrics"]["coherence"]
    assert block["prompt_format"] == "raw"


def test_the_identity_names_the_build_that_took_it(monkeypatch, tmp_path):
    from senbonzakura._version import __version__
    _, out = _run_main(monkeypatch, tmp_path)
    block = json.loads(out.read_text(encoding="utf-8"))["metrics"]["coherence"]
    assert block["tool_version"] == __version__


def test_the_sample_size_is_the_passage_and_not_its_tokens(monkeypatch, tmp_path):
    """One passage is one observation. Recording 250 tokens as n would hand every reader, and
    the sample-size check in the checker, a denominator that is not a sample.
    """
    _, out = _run_main(monkeypatch, tmp_path, n_tokens=250)
    block = json.loads(out.read_text(encoding="utf-8"))["metrics"]["coherence"]
    assert block["n"] == 1
    assert block["n_tokens"] == 250


def test_the_perplexity_is_not_stamped_as_a_second_measurement(monkeypatch, tmp_path):
    """`ppl` is `exp(nll)`: one measurement in two units, and stamping both would leave a
    reader unable to say which number is being claimed. It stays in the document unchanged.
    """
    _, out = _run_main(monkeypatch, tmp_path)
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert list(on_disk["metrics"]) == ["coherence"]
    assert on_disk["ppl"] == pytest.approx(math.exp(2.0))
    assert on_disk["nll"] == pytest.approx(2.0)


def test_the_stamp_is_additive_and_moves_nothing_that_was_there_before(monkeypatch, tmp_path):
    res, _ = _run_main(monkeypatch, tmp_path)
    assert {"label", "model", "nll", "ppl", "n_tokens"} <= set(res)


def test_the_checker_reads_the_artefact_and_finds_nothing_wrong(monkeypatch, tmp_path):
    """THE FAR END OF THE WIRE. Before the stamp, a coherence file was not even RECOGNISED by
    the checker: it carried no metric name the adapter looks for and no metrics block, so
    `senbonzakura check` on it said it could not read the artefact at all. A tool whose whole
    pitch is "here is how your number could be wrong" could not read its own coherence output.
    """
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    _, out = _run_main(monkeypatch, tmp_path)
    findings, skipped, problem = inspect_file(str(out), load_checks())
    assert problem is None, f"the checker could not read our own artefact: {problem}"
    assert findings == [], f"unexpected findings on a clean artefact: {[f.check_id for f in findings]}"
    # A SKIPPED CHECK IS NOT A PASS, which is the outcome the v0.8 plan names as the worst one
    # available: indistinguishable from a clean bill of health. The one that has to have looked
    # is the estimator check, because the control test below depends on it examining this file.
    assert "metric-reported-without-its-estimator" not in skipped


def test_stripping_the_estimator_makes_the_checker_say_so(monkeypatch, tmp_path):
    """The control that makes the test above able to fail.

    A check nobody has watched fail is a check nobody has tested, so the estimator is removed
    from a real stamped artefact and the finding it is supposed to produce is named.
    """
    from senbonzakura_check.cli import inspect_file
    from senbonzakura_check.registry import load_checks

    _, out = _run_main(monkeypatch, tmp_path)
    doc = json.loads(out.read_text(encoding="utf-8"))
    del doc["metrics"]["coherence"]["estimator"]
    out.write_text(json.dumps(doc), encoding="utf-8")

    findings, _skipped, problem = inspect_file(str(out), load_checks())
    assert problem is None
    assert [f.check_id for f in findings] == ["metric-reported-without-its-estimator"]


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
