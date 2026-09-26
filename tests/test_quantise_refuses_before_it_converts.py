# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The checkpoint route refuses what it can decide up front, and keeps what it has already paid for.

THREE DEFECTS, ALL ON THE TWO-STEP PATH

1. `quantise <checkpoint> <a file that is already there>` converted first and refused afterwards.
   The output-exists refusal lived inside `preflight`, which a checkpoint reaches only once
   `_quantise_a_checkpoint` has run the conversion: minutes to tens of minutes, and tens of
   gigabytes through a temporary directory, for a fault visible from the command line. The commit
   that moved `_preflight_arguments` ahead of the conversion made this argument in its own comment
   and stopped one function short.

2. A failed quantisation deleted the conversion it depended on. The intermediate was removed in a
   `finally`, so the failure path destroyed it too: a 30B checkpoint converts for tens of minutes
   into about 60 GB, the quantiser's pre-flight then finds the volume short for the output, and
   the conversion goes with it. The user frees space and starts from zero. A failed step must not
   destroy a completed one.

3. The two routes of one command disagreed about where a file goes. The checkpoint path used
   `Path(a.source).name`, dropping the directory, so a checkpoint at `/models/X` wrote into
   whatever directory the command was run from while the GGUF path writes beside its source.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from senbonzakura import quantise


def _checkpoint(tmp_path, name="edited"):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (d / "model.safetensors").write_bytes(b"0" * 4096)
    return d


def _never(*_args, **_kwargs):
    raise AssertionError("the conversion ran before the output-side checks refused the run")


# ── 1: refused before the conversion ─────────────────────────────────────────────────
def test_an_existing_output_is_refused_before_a_byte_is_converted(tmp_path, monkeypatch):
    monkeypatch.setattr("senbonzakura.convert.run", _never)
    out = tmp_path / "already.gguf"
    out.write_bytes(b"somebody's previous result")
    with pytest.raises(SystemExit, match="Pass --force"):
        quantise.run([str(_checkpoint(tmp_path)), str(out)], log=lambda _m: None)
    assert out.read_bytes() == b"somebody's previous result"


def test_force_still_gets_past_it(tmp_path, monkeypatch):
    out = tmp_path / "already.gguf"
    out.write_bytes(b"previous")
    reached = []
    monkeypatch.setattr("senbonzakura.convert.run", lambda *a, **k: reached.append(a) or 1)
    quantise.run([str(_checkpoint(tmp_path)), str(out), "--force"], log=lambda _m: None)
    assert reached, "--force must not be blocked by the new check"


def test_a_volume_too_small_for_both_steps_is_refused_before_the_first(tmp_path, monkeypatch):
    """Both halves are sized, because this route writes an intermediate AND a quantised file."""
    monkeypatch.setattr("senbonzakura.convert.run", _never)
    monkeypatch.setattr(quantise, "free_bytes_for", lambda _p: 512)
    with pytest.raises(SystemExit, match="free where the output goes"):
        quantise.run([str(_checkpoint(tmp_path)), str(tmp_path / "m.gguf")], log=lambda _m: None)


def test_unknown_free_space_is_not_a_refusal(tmp_path, monkeypatch):
    """Not knowing is a reason not to claim, never a reason to refuse."""
    monkeypatch.setattr(quantise, "free_bytes_for", lambda _p: None)
    a = quantise.build_parser().parse_args([str(_checkpoint(tmp_path)), str(tmp_path / "m.gguf")])
    assert quantise.preflight_a_checkpoint(a, log=lambda _m: None) == tmp_path / "m.gguf"


def test_the_output_side_checks_need_no_converted_file():
    """The split is the fix: whatever does not need the expensive step goes in front of it."""
    quantise.preflight_output("/models/nowhere", "/models/nowhere/out.gguf", force=False)


def test_writing_over_the_source_is_still_refused_first(tmp_path):
    src = tmp_path / "m.gguf"
    src.write_bytes(b"GGUF")
    with pytest.raises(SystemExit, match="output path is the source path"):
        quantise.preflight_output(src, src, force=True)


# ── 3: one command, one answer about where the file goes ─────────────────────────────
def test_the_default_output_stays_beside_the_checkpoint(tmp_path):
    ck = _checkpoint(tmp_path / "models")
    a = quantise.build_parser().parse_args([str(ck)])
    out = quantise.checkpoint_output_path(a)
    assert out.parent == ck.parent, "a checkpoint's output belongs beside it, as a GGUF's does"
    assert out.name == "edited-Q4_K_M.gguf"


def test_an_explicit_output_still_wins(tmp_path):
    a = quantise.build_parser().parse_args([str(_checkpoint(tmp_path)), str(tmp_path / "mine.gguf")])
    assert quantise.checkpoint_output_path(a) == tmp_path / "mine.gguf"


# ── 2: a failure keeps the conversion ────────────────────────────────────────────────
def _stub(monkeypatch, *, quantise_result):
    """The converter writes an intermediate; the inner quantisation does what the test asks."""
    def _convert(argv, log=print):
        Path(argv[1]).write_bytes(b"GGUF" + b"0" * 1024)
        return 0

    real_run = quantise.run

    def _run(argv=None, log=print):
        if argv and str(argv[0]).endswith("-bf16.gguf"):
            return quantise_result()
        return real_run(argv, log=log)

    monkeypatch.setattr("senbonzakura.convert.run", _convert)
    monkeypatch.setattr(quantise, "run", _run)
    return _run


def _kept(directory):
    return sorted(p for p in Path(directory).glob("*-bf16.gguf"))


def test_a_failed_quantisation_keeps_the_conversion(tmp_path, monkeypatch):
    def _fail():
        raise SystemExit("only 1.0 GB free where the output goes")

    run = _stub(monkeypatch, quantise_result=_fail)
    out = tmp_path / "out" / "m.gguf"
    said = []
    with pytest.raises(SystemExit):
        run([str(_checkpoint(tmp_path)), str(out)], log=said.append)
    assert _kept(out.parent), "the conversion was deleted by the step that failed after it"
    joined = " ".join(said)
    assert "did not finish" in joined
    assert "senbonzakura quantise" in joined, "the user is owed the command that resumes from it"


def test_a_non_zero_quantisation_keeps_it_too(tmp_path, monkeypatch):
    run = _stub(monkeypatch, quantise_result=lambda: 3)
    out = tmp_path / "out" / "m.gguf"
    assert run([str(_checkpoint(tmp_path)), str(out)], log=lambda _m: None) == 3
    assert _kept(out.parent)


def test_a_successful_run_still_clears_the_scaffolding(tmp_path, monkeypatch):
    run = _stub(monkeypatch, quantise_result=lambda: 0)
    out = tmp_path / "out" / "m.gguf"
    assert run([str(_checkpoint(tmp_path)), str(out)], log=lambda _m: None) == 0
    assert not _kept(out.parent), "the intermediate is scaffolding when the run succeeded"


def test_a_failed_conversion_leaves_no_resume_point_it_cannot_honour(tmp_path, monkeypatch):
    """Keeping a half-written conversion would offer a resume that cannot be resumed from."""
    def _convert(argv, log=print):
        Path(argv[1]).write_bytes(b"partial")
        return 2

    monkeypatch.setattr("senbonzakura.convert.run", _convert)
    out = tmp_path / "out" / "m.gguf"
    assert quantise.run([str(_checkpoint(tmp_path)), str(out)], log=lambda _m: None) == 2
    assert not _kept(out.parent)
