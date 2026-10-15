# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`quantise --imatrix auto` computes the matrix instead of being handed one.

The `imatrix` command folds into `quantise` as a second spelling of a flag that already existed,
which is the whole reason this file is careful: `--imatrix FILE` means "read this one" and
`--imatrix auto` means "make one", and the two must never be confused by the code that applies it.

Three properties are worth more than the rest.

**There is no default calibration corpus, and the refusal is the feature.** An importance matrix
keeps the precision of whatever the calibration text exercises, so the text is part of the result.
Five of the six bundled corpora are harmful-prompt sets, which is the wrong thing to calibrate on
before measuring refusal, and the sixth is task prompts rather than prose. A quiet default here
would be the same confound the `imatrix` command was written to make visible.

**The matrix is computed after the conversion, not before it.** `llama-imatrix` reads a GGUF. A
transformers checkpoint reaches `run` as a checkpoint and re-enters it as the converted
intermediate, so computing the matrix any earlier works for a GGUF input and fails for the input
the command spends most of its time on.

**The calibration flags are refused where they cannot be read.** A flag accepted and ignored is
worse than one that does not exist, because the run then reports success having dropped what the
user asked for. That is the `--chat-template` defect, one command along.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura import quantise  # noqa: E402


def _gguf(tmp_path, name="m.gguf"):
    """A file with a GGUF name. Nothing here reaches a parser that reads its bytes."""
    p = tmp_path / name
    p.write_bytes(b"GGUF\x00\x00\x00\x00")
    return p


def _run(tmp_path, *argv):
    """`run` up to its first refusal, with the arguments a real invocation would carry."""
    src = _gguf(tmp_path)
    return quantise.run([str(src), str(tmp_path / "out.gguf"), "--type", "Q4_K_M", *argv])


# --------------------------------------------------------------------------------------------
# There is no default, on purpose.
# --------------------------------------------------------------------------------------------

def test_auto_with_no_calibration_source_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, "--imatrix", "auto")
    message = str(e.value)
    assert "calibrate" in message
    assert "--imatrix-file" in message and "--imatrix-corpus" in message, (
        "the refusal has to name both ways out, because a refusal with no escape is worse than "
        "the default it is avoiding")


def test_the_refusal_says_why_there_is_no_default(tmp_path):
    """The reason is the point. Without it this reads as a missing feature rather than a choice."""
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, "--imatrix", "auto")
    message = str(e.value)
    assert "harmful" in message, "the harmful-corpus majority is the reason, so it has to be said"


def test_the_bundled_corpora_still_justify_that_refusal():
    """The refusal's reasoning is a claim about the corpora, so it is checked against them.

    If a broad-prose corpus is ever bundled, this fails and the refusal above should be revisited
    rather than left asserting something that stopped being true.
    """
    from senbonzakura.corpora import CORPORA

    benign = {k for k, c in CORPORA.items() if c.arm != "harmful"}
    assert len(benign) <= 1, (
        f"more than one non-harmful corpus is bundled now ({sorted(benign)}), so `--imatrix auto` "
        f"may have an honest default and the refusal should be reconsidered")


def test_two_calibration_sources_are_refused(tmp_path):
    text = tmp_path / "prose.txt"
    text.write_text("ordinary prose\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, "--imatrix", "auto", "--imatrix-corpus", "xstest-safe",
             "--imatrix-file", str(text))
    assert "both" in str(e.value)


def test_a_missing_calibration_file_is_refused_by_path(tmp_path):
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, "--imatrix", "auto", "--imatrix-file", str(tmp_path / "nope.txt"))
    assert "nope.txt" in str(e.value), "the refusal names the path, so it can be fixed"


# --------------------------------------------------------------------------------------------
# A flag that cannot be read is refused, not ignored.
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("flag", "value"), [
    ("--imatrix-corpus", "xstest-safe"),
    ("--imatrix-file", "prose.txt"),
    ("--imatrix-chunks", "64"),
])
def test_a_calibration_flag_without_auto_is_refused(tmp_path, flag, value):
    if flag == "--imatrix-file":
        (tmp_path / value).write_text("prose\n", encoding="utf-8")
        value = str(tmp_path / "prose.txt")
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, flag, value)
    message = str(e.value)
    assert flag in message, "the refusal names the flag that did nothing"
    assert "--imatrix auto" in message, "and the thing that would make it mean something"


def test_a_calibration_flag_is_refused_even_beside_a_real_matrix(tmp_path):
    """The dangerous spelling: a matrix IS applied, so the run would otherwise look correct."""
    matrix = tmp_path / "m.imatrix"
    matrix.write_bytes(b"\x00")
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, "--imatrix", str(matrix), "--imatrix-chunks", "64")
    assert "--imatrix-chunks" in str(e.value)


# --------------------------------------------------------------------------------------------
# The old spelling still works, and now says the new one exists.
# --------------------------------------------------------------------------------------------

def test_a_missing_matrix_still_refuses_and_offers_auto(tmp_path):
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, "--imatrix", str(tmp_path / "nope.imatrix"))
    message = str(e.value)
    assert "nope.imatrix" in message
    assert "--imatrix auto" in message, (
        "somebody who has no matrix is exactly the person the new spelling is for")


def test_auto_is_not_mistaken_for_a_path(tmp_path):
    """`auto` must never be looked for on disk, including when a file called `auto` exists."""
    (tmp_path / "auto").write_bytes(b"\x00")
    with pytest.raises(SystemExit) as e:
        quantise.run([str(_gguf(tmp_path)), str(tmp_path / "out.gguf"), "--type", "Q4_K_M",
                      "--imatrix", "auto"], log=lambda *_: None)
    assert "calibrate" in str(e.value), (
        "a file named `auto` in the working directory must not turn the keyword back into a path")


# --------------------------------------------------------------------------------------------
# The ordering property: computed from a GGUF, never from a checkpoint.
# --------------------------------------------------------------------------------------------

def test_the_calibration_flags_are_forwarded_through_the_checkpoint_path():
    """A checkpoint converts and re-enters `run`, so the flags must survive the hop.

    Read off the forwarder rather than asserted against a list, which is the derivation that file
    exists to protect: the matrix is computed on the second pass, so a flag dropped on the first
    one means `--imatrix auto` quietly loses its calibration text on exactly the input this
    command spends most of its time on.
    """
    a = quantise.build_parser().parse_args(
        ["ckpt", "out.gguf", "--type", "Q4_K_M", "--imatrix", "auto",
         "--imatrix-corpus", "xstest-safe", "--imatrix-chunks", "64"])
    forwarded = dict(quantise._forwardable_quantiser_flags(a))
    assert forwarded.get("--imatrix") == "auto"
    assert forwarded.get("--imatrix-corpus") == "xstest-safe"
    assert forwarded.get("--imatrix-chunks") == 64


def test_the_computed_matrix_is_named_after_the_source_it_was_computed_from(tmp_path, monkeypatch):
    """And the source at that point is the GGUF, which is what `llama-imatrix` can read."""
    seen = {}

    def fake_run(argv, log=print):
        seen["argv"] = argv
        Path(argv[argv.index("-o") + 1]).write_bytes(b"\x00")
        return 0

    import senbonzakura.imatrix as imatrix_module
    monkeypatch.setattr(imatrix_module, "run", fake_run)

    src = _gguf(tmp_path, "edited.gguf")
    a = quantise.build_parser().parse_args(
        [str(src), str(tmp_path / "out.gguf"), "--type", "Q4_K_M",
         "--imatrix", "auto", "--imatrix-corpus", "xstest-safe"])
    out = quantise._computed_imatrix(a, log=lambda *_: None)

    assert out == src.with_suffix(".imatrix")
    assert out.is_file()
    assert seen["argv"][0] == str(src), "computed from the GGUF this run is quantising"
    assert "--corpus" in seen["argv"] and "xstest-safe" in seen["argv"]


def test_a_failed_matrix_step_stops_the_run(tmp_path, monkeypatch):
    """It is not optional: the user asked for an i1 quantisation and would otherwise get a plain
    one reported as success, which is the silent-degradation shape this project refuses.
    """
    import senbonzakura.imatrix as imatrix_module
    monkeypatch.setattr(imatrix_module, "run", lambda argv, log=print: 1)

    a = quantise.build_parser().parse_args(
        [str(_gguf(tmp_path)), str(tmp_path / "out.gguf"), "--type", "Q4_K_M",
         "--imatrix", "auto", "--imatrix-corpus", "xstest-safe"])
    with pytest.raises(SystemExit) as e:
        quantise._computed_imatrix(a, log=lambda *_: None)
    assert "nothing was quantised" in str(e.value)


def test_a_matrix_step_that_claims_success_without_writing_is_caught(tmp_path, monkeypatch):
    """Exit 0 is not evidence a file exists, and the quantiser would fail confusingly later."""
    import senbonzakura.imatrix as imatrix_module
    monkeypatch.setattr(imatrix_module, "run", lambda argv, log=print: 0)

    a = quantise.build_parser().parse_args(
        [str(_gguf(tmp_path)), str(tmp_path / "out.gguf"), "--type", "Q4_K_M",
         "--imatrix", "auto", "--imatrix-file", str(tmp_path / "p.txt")])
    (tmp_path / "p.txt").write_text("prose\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        quantise._computed_imatrix(a, log=lambda *_: None)
