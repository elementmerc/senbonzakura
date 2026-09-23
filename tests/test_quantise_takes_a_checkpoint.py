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

import pytest

from senbonzakura import quantise


def _record(seen):
    """A stand-in for `convert.run` that keeps its argv and reports success."""
    def _run(argv, log=print):
        seen["argv"] = list(argv)
        return 0
    return _run


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


def test_a_checkpoint_is_handed_to_the_converter_with_the_quantisation_asked_for(
        tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("senbonzakura.convert.run", _record(seen))
    d = _checkpoint(tmp_path)
    assert quantise.run([str(d), "--type", "Q5_K_M"], log=lambda _m: None) == 0
    argv = seen["argv"]
    assert argv[0] == str(d)
    assert argv[argv.index("--quantise") + 1] == "Q5_K_M"


def test_the_converter_is_not_reimplemented_here():
    """The delegation is the design. A second converter is a second set of checks to keep in
    step with a pinned binary, and the pin is what makes the output reproducible.
    """
    body = inspect.getsource(quantise.run)
    assert "from . import convert" in body, "the conversion must be delegated, not rebuilt"


def test_an_importance_matrix_survives_the_hand_off(tmp_path, monkeypatch):
    """An i1 quantisation and a plain one of the same weights are not comparable, so a matrix
    quietly dropped on this route would produce a file whose name says something it is not.
    """
    seen = {}
    monkeypatch.setattr("senbonzakura.convert.run", _record(seen))
    d = _checkpoint(tmp_path)
    im = tmp_path / "calib.imatrix"
    im.write_bytes(b"x")
    quantise.run([str(d), "--imatrix", str(im)], log=lambda _m: None)
    assert "--imatrix" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--imatrix") + 1] == str(im)


def test_keep_source_keeps_the_intermediate(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("senbonzakura.convert.run", _record(seen))
    quantise.run([str(_checkpoint(tmp_path)), "--keep-source"], log=lambda _m: None)
    assert "--keep-intermediate" in seen["argv"]


def test_the_route_is_announced_rather_than_taken_silently(tmp_path, monkeypatch, capsys):
    """Two commands ran where one was typed. A user reading the log has to be able to see that,
    or the provenance of the file they end up with is a guess.
    """
    monkeypatch.setattr("senbonzakura.convert.run", lambda argv, log=print: 0)
    said = []
    quantise.run([str(_checkpoint(tmp_path))], log=said.append)
    joined = " ".join(said)
    assert "convert" in joined and "checkpoint" in joined


def test_a_source_that_is_neither_is_still_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        quantise.run([str(tmp_path / "nothing.gguf")], log=lambda _m: None)
    assert "does not exist" in str(e.value)
