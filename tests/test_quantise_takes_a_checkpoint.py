# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura quantise <checkpoint>` converts first rather than sending the user away.

WHAT WAS WRONG

`convert --quantise Q4_K_M` has always done both steps in one command. `quantise` is the name
somebody reaches for when they want a quantised model, and it refused a checkpoint directory with
an instruction to run a different command first. So the shortest route from edited weights to
something llama.cpp will serve was behind a word describing an intermediate file format, and the
person who typed the obvious thing was sent to learn GGUF in order to be handed back to the
command they started with.

WHAT IT MUST NOT BECOME

A second conversion pipeline. The converter is pinned, its output is verified, and a copy of that
in this module would be a second set of checks to keep in step. So this delegates, and these tests
assert the delegation rather than the result: what reaches `convert` is the command line a person
would have typed.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from senbonzakura import quantise


def _checkpoint(tmp_path, name="edited"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    return d


def test_a_directory_with_a_config_is_a_checkpoint(tmp_path):
    assert quantise.looks_like_a_checkpoint(_checkpoint(tmp_path))


def test_a_gguf_file_is_not(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF")
    assert not quantise.looks_like_a_checkpoint(f)


def test_a_directory_without_a_config_is_not(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert not quantise.looks_like_a_checkpoint(d)


def _stub_convert_and_quantise(monkeypatch, tmp_path, seen):
    """Stand in for both halves: the converter writes a file, the quantiser records its argv.

    `quantise.run` calls itself for the second half, so the recursion is intercepted rather than
    the whole function replaced; otherwise the test would assert nothing about the step that
    names the output file, which is the step the defect was in.
    """
    def _convert(argv, log=print):
        seen["convert"] = list(argv)
        Path(argv[1]).write_bytes(b"GGUF")      # the intermediate the second half consumes
        return 0

    real_run = quantise.run

    def _run(argv=None, log=print):
        if seen.get("convert") is not None and "--type" in (argv or []):
            seen["quantise"] = list(argv)
            return 0
        return real_run(argv, log=log)

    monkeypatch.setattr("senbonzakura.convert.run", _convert)
    monkeypatch.setattr(quantise, "run", _run)
    return _run


def test_a_checkpoint_is_converted_then_quantised_to_the_type_asked_for(tmp_path, monkeypatch):
    seen = {}
    run = _stub_convert_and_quantise(monkeypatch, tmp_path, seen)
    d = _checkpoint(tmp_path)
    assert run([str(d), "--type", "Q5_K_M"], log=lambda _m: None) == 0
    assert seen["convert"][0] == str(d)
    assert seen["quantise"][seen["quantise"].index("--type") + 1] == "Q5_K_M"


def test_the_output_path_names_the_quantised_file_not_the_intermediate(tmp_path, monkeypatch):
    """THE DEFECT A REAL RUN FOUND, and a log line could not.

    Handing the whole job to `convert --quantise` meant the converter's positional named the
    INTERMEDIATE and the quantised file took a name derived from it. A run given an explicit
    output path wrote 0.73 GB into a directory the user had not named, under a log line saying
    it would be somewhere else. Measured on the ROG, 2026-09-23.
    """
    seen = {}
    run = _stub_convert_and_quantise(monkeypatch, tmp_path, seen)
    wanted = tmp_path / "somewhere" / "mine.gguf"
    assert run([str(_checkpoint(tmp_path)), str(wanted), "--type", "Q4_K_M"],
               log=lambda _m: None) == 0
    assert seen["quantise"][1] == str(wanted), (
        f"the quantised file would go to {seen['quantise'][1]}, not where the user asked")
    assert str(wanted) != seen["convert"][1], "the intermediate must not take the output's name"


def test_the_intermediate_is_not_left_beside_the_checkpoint(tmp_path, monkeypatch):
    """The checkpoint may sit in a read-only Hub cache, and it is not the user's output
    directory either way.
    """
    seen = {}
    run = _stub_convert_and_quantise(monkeypatch, tmp_path, seen)
    d = _checkpoint(tmp_path)
    out = tmp_path / "out" / "m.gguf"
    run([str(d), str(out), "--type", "Q4_K_M"], log=lambda _m: None)
    assert not list(d.glob("*.gguf")), "the intermediate was written into the checkpoint directory"
    assert Path(seen["convert"][1]).parent.parent == out.parent, (
        "the intermediate belongs in a temporary directory beside the OUTPUT, which is "
        "the directory the user has just said they can write to")


def test_the_converter_is_not_reimplemented_here():
    """The delegation is the design. A second converter is a second set of checks to keep in
    step with a pinned binary, and the pin is what makes the output reproducible.
    """
    body = inspect.getsource(quantise._quantise_a_checkpoint)
    assert "from . import convert" in body, "the conversion must be delegated, not rebuilt"


def test_an_importance_matrix_survives_the_hand_off(tmp_path, monkeypatch):
    """An i1 quantisation and a plain one of the same weights are not comparable, so a matrix
    quietly dropped on this route would produce a file whose name says something it is not.
    """
    seen = {}
    d = _checkpoint(tmp_path)
    im = tmp_path / "calib.imatrix"
    im.write_bytes(b"x")
    run = _stub_convert_and_quantise(monkeypatch, tmp_path, seen)
    run([str(d), "--imatrix", str(im)], log=lambda _m: None)
    argv = seen["quantise"]
    assert "--imatrix" in argv
    assert argv[argv.index("--imatrix") + 1] == str(im)


def test_keep_source_moves_the_intermediate_somewhere_findable(tmp_path, monkeypatch):
    """Kept means kept where the user can see it, not left in a temporary directory named after
    an internal function.
    """
    seen = {}
    run = _stub_convert_and_quantise(monkeypatch, tmp_path, seen)
    out = tmp_path / "out" / "m.gguf"
    run([str(_checkpoint(tmp_path)), str(out), "--keep-source"], log=lambda _m: None)
    assert list(out.parent.glob("*-bf16.gguf")), (
        f"nothing kept in {out.parent}: {sorted(p.name for p in out.parent.iterdir())}")


def test_the_route_is_announced_rather_than_taken_silently(tmp_path, monkeypatch, capsys):
    """Two commands ran where one was typed. A user reading the log has to be able to see that,
    or the provenance of the file they end up with is a guess.
    """
    seen = {}
    run = _stub_convert_and_quantise(monkeypatch, tmp_path, seen)
    said = []
    run([str(_checkpoint(tmp_path)), str(tmp_path / "o" / "m.gguf")], log=said.append)
    joined = " ".join(said)
    assert "checkpoint" in joined and "converted first" in joined
    assert str(tmp_path / "o" / "m.gguf") in joined, (
        "the log must name where the file will be, and it must be true")


def test_a_source_that_is_neither_is_still_refused_and_names_both_routes(tmp_path):
    """A path that is neither a GGUF nor a checkpoint. The refusal has to say what this command
    takes, and it still names `convert`, which is what makes the first of the two.
    """
    with pytest.raises(SystemExit) as e:
        quantise.run([str(tmp_path / "nothing.gguf")], log=lambda _m: None)
    message = str(e.value)
    assert "nothing is there" in message
    assert "checkpoint" in message and "senbonzakura convert" in message
