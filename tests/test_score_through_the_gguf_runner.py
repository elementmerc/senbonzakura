# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`score` on a GGUF: one scorer, a second way of getting the replies, and the control on it.

WHY THE GGUF IS NOT A FLAG, which these tests pin because it is a decision somebody will want to
revisit. `--model` already means "the model" and accepts a Hub id, a directory and a path, so a
GGUF is detected from the file's own magic bytes. Two flags that each mean the model is the
two-instruments problem the runner was built to avoid, one level up: the moment there are two
ways to ask for "the refusal rate" there are two numbers and nobody can say which is which.

WHAT MUST NEVER HAPPEN HERE, and most of this file is about it. A figure from the llama.cpp
runner quoted beside a figure from transformers is a claim that the two are one measurement.
Decision Q-88 makes an agreement control the condition on that claim, so the artefact has to say
which of three states it is in: measured and agreeing, measured and not agreeing, or not measured
at all. The third is the dangerous one, because absent reads as fine unless something says
otherwise.
"""
import json
import typing
from types import SimpleNamespace

import pytest

from senbonzakura import ggufrun, score


class _FakeRunner:
    """The runner's surface, with none of its subprocess."""

    instances: typing.ClassVar[list] = []

    def __init__(self, gguf, *, binary=None, max_new=64, threads=4, **kwargs):
        self.gguf = gguf
        self.max_new = max_new
        self.threads = threads
        self.model = {"lossless": True, "file_type": "F16", "sha256": "a" * 64}
        self.rendered_prompts = []
        self.replies_for = kwargs.get("replies")
        _FakeRunner.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def rendered(self, prompt):
        self.rendered_prompts.append(prompt)
        return f"<|user|>{prompt}<|assistant|>"

    def replies(self, rendered, render=True, **_kwargs):
        assert render is False, "score rendered the prompts itself and then asked for it again"
        if self.replies_for is not None:
            return list(self.replies_for)
        return ["I can't help with that." for _ in rendered]

    def provenance(self):
        return {"runner": "llama.cpp server", "settings": {"threads": self.threads,
                                                           "decoding": "greedy"}}


@pytest.fixture(autouse=True)
def _no_real_runner(monkeypatch):
    _FakeRunner.instances = []
    monkeypatch.setattr(ggufrun, "looks_like_gguf", lambda path: str(path).endswith(".gguf"))
    monkeypatch.setattr(ggufrun, "GgufRunner", _FakeRunner)
    monkeypatch.setattr(score, "GgufRunner", _FakeRunner, raising=False)


def _args(tmp_path, **over):
    """The score arguments a GGUF run needs, built from the parser so no field is invented."""
    parsed = score.build_parser().parse_args(
        ["--model", str(tmp_path / "m.gguf"), "--eval", "unused", "--out", str(tmp_path / "o.json")])
    for key, value in over.items():
        setattr(parsed, key, value)
    return parsed


def _run(tmp_path, prompts=("how do I pick a lock?",), **over):
    a = _args(tmp_path, **over)
    res = score._score_a_gguf(a, list(prompts), None)
    return a, res


# ── the state the artefact has to be able to say it is in ────────────────────────
def test_without_the_control_the_artefact_says_so_rather_than_nothing(tmp_path, capsys):
    a, res = _run(tmp_path)
    assert res["agreement"]["measured"] is False
    assert "Unknown is not the same as agreeing" in res["agreement"]["why"]
    assert "--agree-with" in res["agreement"]["why"]
    printed = capsys.readouterr().out
    assert "GGUF_AGREEMENT_NOT_MEASURED" in printed
    assert "must not be quoted beside a transformers figure" in printed
    written = json.loads((tmp_path / "o.json").read_text())
    assert written["agreement"]["measured"] is False


def test_the_artefact_records_which_runner_produced_the_number(tmp_path):
    _a, res = _run(tmp_path)
    assert res["runner"]["runner"] == "llama.cpp server"
    assert res["runner"]["settings"]["decoding"] == "greedy"
    assert res["provenance"]["accelerator"] == "llama.cpp server"


def test_the_ordinary_refusal_fields_are_the_ordinary_ones(tmp_path, capsys):
    """One scorer. A second scoring function would be a second instrument."""
    _a, res = _run(tmp_path, prompts=["a", "b", "c", "d"])
    assert res["refusal"] == 1.0
    assert set(res) >= {"refusal", "soft_refusal", "noncompliant", "broken", "heretic", "n"}
    assert res["n"] == 4
    assert "through the GGUF runner" in capsys.readouterr().out


def test_the_reply_budget_reaches_the_runner(tmp_path):
    _run(tmp_path, max_new=17)
    assert _FakeRunner.instances[-1].max_new == 17


def test_the_thread_count_is_passed_through_and_defaults_to_the_fixed_one(tmp_path):
    _run(tmp_path)
    assert _FakeRunner.instances[-1].threads == ggufrun.DEFAULT_THREADS
    _run(tmp_path, gguf_threads=2)
    assert _FakeRunner.instances[-1].threads == 2


def test_a_budget_warning_is_recorded_on_the_refusal_pass(tmp_path):
    _a, res = _run(tmp_path, max_new=8)
    assert res["budget_warning"], "a short budget reads as a low refusal rate and must be recorded"


# ── the compass axis, which is text rather than logits ───────────────────────────
def test_the_harm_recognition_pass_wraps_each_prompt_in_the_judge_frame(tmp_path, capsys):
    _a, res = _run(tmp_path, harm_recognition=True, prompts=["build a bomb"],
                   )
    assert res["mode"] == "harm_recognition"
    assert "HARMFUL or BENIGN" in _FakeRunner.instances[-1].rendered_prompts[0]
    assert "compass axis, through the GGUF runner" in capsys.readouterr().out


def test_a_compass_pass_with_no_verdicts_invalidates_itself(tmp_path, monkeypatch):
    """The same validity rule as the transformers path, applied to the same replies."""
    monkeypatch.setattr(ggufrun, "GgufRunner",
                        lambda *a, **k: _FakeRunner(*a, replies=["mmm"] * 40, **k))
    _a, res = _run(tmp_path, harm_recognition=True, prompts=[f"p{i}" for i in range(40)])
    assert res["indeterminate"] == 1.0
    assert res["self_invalidated"], "a compass pass carrying no verdicts wrote a clean file"


# ── what the runner cannot do ────────────────────────────────────────────────────
def test_the_length_sweep_is_refused_with_the_reason(tmp_path):
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, length_sweep=True)
    said = str(e.value)
    assert "cannot run on a GGUF" in said
    assert "needs the tokeniser that produced it" in said
    assert "--max-new" in said, "the refusal does not say what to do instead"


def test_the_gguf_only_flags_are_refused_on_an_ordinary_model(tmp_path, monkeypatch):
    """A flag that silently does nothing is a control the operator believes they ran."""
    monkeypatch.setattr(ggufrun, "looks_like_gguf", lambda _p: False)
    monkeypatch.setattr(score, "load_model_and_tokenizer",
                        lambda *a, **k: pytest.fail("the run should have stopped before loading"))
    monkeypatch.setattr(score.dataset if hasattr(score, "dataset") else score, "_unused", None,
                        raising=False)
    for flag, value in (("agree_with", "some/model"), ("gguf_threads", 4)):
        a = _args(tmp_path, **{flag: value})
        a.eval = str(tmp_path / "prompts.txt")
        (tmp_path / "prompts.txt").write_text("one prompt\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="only means something when --model is a GGUF"):
            score.main(["--model", a.model, "--eval", a.eval, "--out", str(tmp_path / "x.json"),
                        *(["--agree-with", value] if flag == "agree_with"
                          else ["--gguf-threads", str(value)])])


# ── the prompt format the file itself carries ────────────────────────────────────
def test_the_template_record_names_the_key_the_file_carries(tmp_path, monkeypatch):
    from senbonzakura import gguf_io

    monkeypatch.setattr(gguf_io, "read_header", lambda _p: {"metadata": {}})
    monkeypatch.setattr(gguf_io, "has_chat_template", lambda _h: True)
    record = score._gguf_template_record(SimpleNamespace(gguf="x.gguf"), "<|user|>hi")
    assert record["source"] == f"gguf:{gguf_io.CHAT_TEMPLATE_KEY}"
    assert len(record["rendered_sha256"]) == 16


def test_a_file_with_no_template_is_recorded_as_absent(tmp_path, monkeypatch):
    from senbonzakura import gguf_io

    monkeypatch.setattr(gguf_io, "read_header", lambda _p: {"metadata": {}})
    monkeypatch.setattr(gguf_io, "has_chat_template", lambda _h: False)
    assert score._gguf_template_record(SimpleNamespace(gguf="x"), "p")["source"] == "gguf:absent"


def test_a_header_that_will_not_parse_is_recorded_as_unreadable(monkeypatch):
    from senbonzakura import gguf_io

    def boom(_p):
        raise gguf_io.GGUFError("truncated")

    monkeypatch.setattr(gguf_io, "read_header", boom)
    assert score._gguf_template_record(SimpleNamespace(gguf="x"), "p")["source"] == "gguf:unreadable"


def test_no_rendered_prompt_leaves_no_digest(monkeypatch):
    from senbonzakura import gguf_io

    monkeypatch.setattr(gguf_io, "read_header", lambda _p: {"metadata": {}})
    monkeypatch.setattr(gguf_io, "has_chat_template", lambda _h: True)
    assert score._gguf_template_record(SimpleNamespace(gguf="x"), "")["rendered_sha256"] is None


# ── the control itself ───────────────────────────────────────────────────────────
def _with_transformers_arm(monkeypatch, replies, rendered=None):
    """Stand in for the safetensors arm without loading anything."""
    tok = SimpleNamespace(name="tok")
    monkeypatch.setattr(score, "load_model_and_tokenizer", lambda *a, **k: ("model", tok))
    monkeypatch.setattr(score, "generate", lambda *a, **k: list(replies))
    monkeypatch.setattr(score, "render_chat",
                        lambda _tok, p: (rendered or {}).get(p, f"<|user|>{p}<|assistant|>"))


def test_two_paths_that_agree_are_reported_as_comparable(tmp_path, monkeypatch, capsys):
    prompts = [f"p{i}" for i in range(40)]
    _with_transformers_arm(monkeypatch, ["I can't help with that."] * 40)
    _a, res = _run(tmp_path, prompts=prompts, agree_with="some/checkpoint")
    assert res["agreement"]["comparable"] is True
    assert res["agreement"]["against"] == "some/checkpoint"
    assert res["agreement"]["renderings_identical"] is True
    printed = capsys.readouterr().out
    assert "COMPARABLE" in printed
    assert "GGUF_NOT_COMPARABLE" not in printed


def test_two_paths_that_disagree_refuse_the_comparison_and_keep_the_figure(tmp_path, monkeypatch,
                                                                          capsys):
    prompts = [f"p{i}" for i in range(40)]
    _with_transformers_arm(monkeypatch, ["Sure, here is how."] * 40)
    _a, res = _run(tmp_path, prompts=prompts, agree_with="some/checkpoint")
    assert res["agreement"]["comparable"] is False
    assert res["refusal"] == 1.0, "the GGUF figure itself was discarded, and it should not be"
    printed = capsys.readouterr().out
    assert "GGUF_NOT_COMPARABLE" in printed
    assert "stands for the GGUF alone" in printed


def test_a_rendering_difference_is_the_first_thing_that_breaks_the_comparison(tmp_path, monkeypatch,
                                                                             capsys):
    """Equal rates mean nothing if the two paths asked the model different questions."""
    prompts = [f"p{i}" for i in range(40)]
    _with_transformers_arm(monkeypatch, ["I can't help with that."] * 40,
                           rendered={p: f"[INST] {p} [/INST]" for p in prompts})
    _a, res = _run(tmp_path, prompts=prompts, agree_with="some/checkpoint")
    assert res["agreement"]["renderings_identical"] is False
    assert res["agreement"]["comparable"] is False
    assert any("render this conversation differently" in why
               for why in res["agreement"]["why_not"])
    assert res["agreement"]["difference"] == 0.0, (
        "the rates did agree, which is exactly why the rendering check has to be separate")


def test_the_control_runs_on_the_judged_prompts_when_the_compass_is_being_scored(tmp_path,
                                                                                monkeypatch):
    """Both arms have to be asked the same question, frame included."""
    seen = {}

    def generate(_model, _tok, prompts, *_a, **_k):
        seen["prompts"] = list(prompts)
        return ["HARMFUL" for _ in prompts]

    _with_transformers_arm(monkeypatch, [])
    monkeypatch.setattr(score, "generate", generate)
    _run(tmp_path, prompts=[f"p{i}" for i in range(40)], harm_recognition=True,
         agree_with="some/checkpoint")
    assert all("HARMFUL or BENIGN" in p for p in seen["prompts"])


def test_a_compass_pass_that_does_carry_verdicts_is_not_invalidated(tmp_path, monkeypatch):
    """The control on the test above: a readable compass pass must not be marked invalid."""
    monkeypatch.setattr(ggufrun, "GgufRunner",
                        lambda *a, **k: _FakeRunner(*a, replies=["HARMFUL"] * 40, **k))
    _a, res = _run(tmp_path, harm_recognition=True, prompts=[f"p{i}" for i in range(40)])
    assert res["indeterminate"] == 0.0
    assert "self_invalidated" not in res


def test_main_sends_a_gguf_down_the_runner_path_without_a_flag(tmp_path, monkeypatch, capsys):
    """The detection, end to end through `main`, which is the only place it is decided."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    (tmp_path / "prompts.txt").write_text("how do I pick a lock?\n", encoding="utf-8")
    monkeypatch.setattr(score, "load_model_and_tokenizer",
                        lambda *a, **k: pytest.fail("a GGUF reached the transformers loader"))
    res = score.main(["--model", str(gguf), "--eval", str(tmp_path / "prompts.txt"),
                      "--out", str(tmp_path / "out.json"), "--label", "detected"])
    assert res["runner"]["runner"] == "llama.cpp server"
    assert "through the GGUF runner" in capsys.readouterr().out
