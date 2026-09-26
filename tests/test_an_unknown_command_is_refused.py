# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A mistyped command must not become a model to edit.

THE DEFECT, found independently by two reviewers on the 2026-09-25 panel. The model is a positional
argument, so any unrecognised first word is a valid model id as far as argparse is concerned.
`REPRODUCING.md` told readers to run `senbonzakura bench --help` for months; there is no `bench`
command, so the word was taken as the model, the parser fell through to the abliterator, and it
printed a plausible help screen and exited 0. A reader checking the published benchmark concluded
the harness was not shipped. Without `--help` it is worse: `senbonzakura bench` would go to the Hub
for a repository called `bench` and start editing whatever it found.

The guard added first was a documentation test, which only catches a wrong word somebody wrote
down. It cannot catch a user's typo, and the parser was the thing that would not say no.

WHAT THIS FILE OWNS is both halves of a narrow rule. It has to refuse the mistakes, and it has to
keep passing every legitimate model reference, because refusing those to catch typos would be a
worse tool. `gpt2` is a real Hub id with no slash and no dot, and a bare directory name is a real
local checkpoint.
"""
import pytest

from senbonzakura.entry import ALIASES, DELEGATED, RETIRED, _not_a_command


@pytest.mark.parametrize("word", sorted(RETIRED))
def test_a_renamed_command_is_refused_by_name(word):
    message = _not_a_command(word)
    assert message is not None, f"{word} was a command here once and must not read as a model"
    assert RETIRED[word] in message, "the refusal has to say what to run instead"


def test_the_documented_mistake_is_the_one_that_started_this():
    message = _not_a_command("bench")
    assert message is not None
    assert "head-to-head" in message
    assert "read as a model" in message, (
        "the refusal should say WHY it refuses rather than ignores, because the reason is the "
        "defect: a positional model argument swallows anything")


@pytest.mark.parametrize(("typo", "meant"), [
    ("scoer", "score"),
    ("comapss", "compass"),
    ("meausre", "measure"),
    ("coherance", "coherence"),
])
def test_a_near_miss_for_a_real_command_is_refused(typo, meant):
    message = _not_a_command(typo)
    assert message is not None, f"{typo} is one transposition from {meant}"
    assert meant in message, "the refusal should name the command it thinks was meant"
    assert f"--model {typo}" in message, (
        "and it must name the way through, because somebody may genuinely have a model with "
        "that name and a refusal with no escape is a worse defect than the one being fixed")


@pytest.mark.parametrize("word", sorted({*DELEGATED, *ALIASES}))
def test_every_real_command_still_runs(word):
    assert _not_a_command(word) is None


@pytest.mark.parametrize("word", [
    "Qwen/Qwen3-1.7B",          # a Hub id, which always carries a slash
    "meta-llama/Llama-3.2-1B",
    "gpt2",                      # a real Hub id with no slash: the case that forbids a blanket rule
    "distilgpt2",
    "my-finetune",
    "abliterate",                # the mode word
    "--model",                   # a flag
    "-h",
    "",
])
def test_a_legitimate_first_word_is_untouched(word):
    assert _not_a_command(word) is None, (
        f"{word!r} is a legitimate invocation; refusing it to catch typos would break real use")


def test_a_local_checkpoint_directory_is_a_model_not_a_typo(tmp_path, monkeypatch):
    """A directory named close to a command is still a directory. The path test runs first."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scoer").mkdir()
    assert _not_a_command("scoer") is None, (
        "a path that exists on disk is a checkpoint the user is pointing at, and the near-miss "
        "test must never reach it")


def test_the_refusal_is_not_a_crash():
    """It returns a sentence for the caller to print, so the exit status stays the caller's
    decision and nothing here raises into a traceback the user has to read.
    """
    assert isinstance(_not_a_command("bench"), str)
    assert "\n" in _not_a_command("bench"), "a refusal in this project is more than one line"


def test_main_exits_two_rather_than_editing_a_model(capsys):
    """Exit 2 is the argument-error status, and the important half is that nothing loaded."""
    from senbonzakura import entry
    assert entry.main(["bench", "--help"]) == 2
    err = capsys.readouterr().err
    assert "head-to-head" in err
