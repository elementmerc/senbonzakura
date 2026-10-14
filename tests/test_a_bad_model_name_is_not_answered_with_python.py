# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A command-line user who mistypes a model gets command-line advice, not a Python keyword argument.

WHAT PROMPTED IT, 2026-09-27

`load_model_and_tokenizer` kept `transformers`' own load failure verbatim, deliberately and for a
good reason: that sentence names the model id and says whether it was a local folder or a listed
model, which is precisely what a reader who typed a name with a character missing needs. The frames
around it were the noise, and dropping them was right.

What came with it was the advice at the end, which tells the reader to pass `token=<your_token>` and
to run `huggingface-cli login`. The first is a keyword argument to a function they are not calling.
The second is the old name for `hf auth login`. Our own line underneath already gave the two
spellings that work here, so a reader met two instructions, one of them impossible, and the
impossible one came first.

The fix keeps the half that identifies the problem and drops the half that answers the wrong
question. It is the same discipline as the corpus banners: cut the padding, keep the obligation.
"""
from __future__ import annotations

import pytest

from senbonzakura.cli import _could_not_load

#: The real shape of the message, as `transformers` emits it for a model id that does not exist.
THEIRS = (
    "Qwen/Qwen2.5-0.5B-Instruc is not a local folder and is not a valid model identifier "
    "listed on 'https://huggingface.co/models'\n"
    "If this is a private repository, make sure to pass a token having permission to this repo "
    "either by logging in with `huggingface-cli login` or by passing `token=<your_token>`"
)


@pytest.fixture
def message():
    return _could_not_load("Qwen/Qwen2.5-0.5B-Instruc", OSError(THEIRS))


def test_the_part_that_identifies_the_problem_is_kept(message):
    """The whole reason the upstream sentence was kept verbatim in the first place."""
    assert "is not a local folder" in message
    assert "not a valid model identifier" in message, (
        "the reader needs to know it was looked for in both places and found in neither")


def test_the_model_it_could_not_load_is_named(message):
    """A typo is the likeliest cause, and a typo is only visible when the id is echoed back."""
    assert "Qwen/Qwen2.5-0.5B-Instruc" in message


@pytest.mark.parametrize("wrong", ["token=<your_token>", "use_auth_token", "huggingface-cli login"])
def test_advice_for_the_python_api_is_gone(message, wrong):
    """None of these can be followed by somebody holding a command line."""
    assert wrong not in message, (
        f"{wrong!r} survived into a command-line refusal. It is advice for a Python caller, and it "
        f"sits above the line that gives the two spellings which do work here.")


def test_the_advice_that_does_work_is_present(message):
    """Dropping the wrong instruction is only half of it; the right one has to be there."""
    assert "--hf-token" in message
    assert "hf auth login" in message, "the current spelling, not the one upstream still prints"


def test_an_unrecognised_message_is_passed_through_rather_than_swallowed(message):
    """A future upstream wording must degrade to "kept" and never to "dropped".

    The filter names phrases. If upstream rewrites its advice, the new wording will not match, and
    the correct outcome is that the reader sees a little too much rather than nothing at all. A
    filter that silently emptied the message would turn a helpful refusal into a bare sentence.
    """
    novel = _could_not_load("some/model", OSError("a wording nobody has seen before\nand a second line"))
    assert "a wording nobody has seen before" in novel
    assert "and a second line" in novel


def test_an_empty_upstream_message_still_gives_a_usable_refusal():
    """Fail loud, not blank. An error with nothing in it is the case most likely to print nothing."""
    bare = _could_not_load("some/model", OSError(""))
    assert "could not load the model 'some/model'" in bare
    assert "--hf-token" in bare
    assert "\n\n" not in bare, "an empty upstream message must not leave a hole in the output"


# ── the token advice is for the Hub, not for a file already on disk ──────────────────
#
# Journey 1, 2026-10-13: a reader passed a local .jsonl as --model, it would not load, and the
# refusal told them to authenticate to a service that was not involved. Advice for a problem the
# reader does not have sends them to fix something that is not broken, which is expensive in
# exactly the moment they are already stuck.
def test_a_path_that_exists_is_not_told_to_authenticate(tmp_path):
    local = tmp_path / "prompts.jsonl"
    local.write_text('{"prompt": "hello"}\n')
    message = _could_not_load(str(local), OSError("not a model directory"))
    assert "--hf-token" not in message, f"told to authenticate about a local file: {message}"
    assert "hf auth login" not in message
    assert "That path exists" in message
    assert "not a model directory" in message, "the useful half of the refusal is still there"


def test_a_hub_id_still_gets_the_token_advice():
    """The common case is unchanged, and the shape of an id and of a path is the same."""
    message = _could_not_load("meta-llama/Llama-3-8B", OSError("gated repo"))
    assert "--hf-token" in message
    assert "That path exists" not in message


def test_an_ambiguous_name_that_does_not_exist_is_treated_as_an_id():
    """A mistyped path and a mistyped id are indistinguishable, so the Hub advice is still right.

    THE PREMISE IS RIGHT AND THIS TEST'S EXAMPLE WAS NOT, corrected 2026-10-13. It asserted the
    rule with an ABSOLUTE path out of `tmp_path`, and an absolute path is one of the few shapes a
    hub id cannot take: an id is `org/model`, with one slash and no leading separator. So the
    example was not an instance of the ambiguity the sentence describes, and the test was pinning
    the Hub advice onto a case where it is simply wrong.

    The journey that found it typed a directory that was not there and was told to go and
    authenticate, which is the same wrong fix this file's other tests exist to prevent, arriving
    through the one input nobody had questioned: the test's own example.

    So the rule is kept and stated against a name that genuinely is ambiguous.
    """
    message = _could_not_load("meta-llma/Llama-3-8B", OSError("no such thing"))
    assert "--hf-token" in message


def test_a_name_that_can_only_be_a_path_is_not_treated_as_an_id(tmp_path):
    """And the unambiguous half of the same question, which is what the old example really was."""
    message = _could_not_load(str(tmp_path / "nothing-here"), OSError("no such thing"))
    assert "--hf-token" not in message, f"told to authenticate about a path: {message}"
    assert "hf auth login" not in message
    assert "Nothing exists at that path" in message
    assert "no such thing" in message, "the useful half of the refusal is still there"


#: Paths as a Windows reader types or pastes them. Literal strings, so this runs on every platform:
#: the defect was in the string's shape, and a Windows runner only ever found it by accident.
@pytest.mark.parametrize("typed", [
    r"C:\models\nothing-here",
    r"d:\work\qwen\nothing-here",
    "D:/nothing-here",
    r"\\server\share\nothing-here",
    r"models\nothing-here",
    r".\nothing-here",
    r"..\nothing-here",
])
def test_a_windows_path_that_is_not_there_is_not_treated_as_an_id(typed):
    message = _could_not_load(typed, OSError("no such thing"))
    assert "--hf-token" not in message, f"told to authenticate about a path: {message}"
    assert "hf auth login" not in message
    assert "Nothing exists at that path" in message


def test_an_unusable_path_does_not_raise_from_inside_the_refusal():
    """A null byte makes `Path.exists` raise, and a traceback here replaces a usable refusal."""
    message = _could_not_load("some/model\x00name", OSError("boom"))
    assert "could not load the model" in message
    assert "--hf-token" in message


def test_the_local_refusal_has_no_trailing_blank_line(tmp_path):
    """Same property the empty-upstream case asserts: no hole in the output."""
    local = tmp_path / "weights"
    local.mkdir()
    assert not _could_not_load(str(local), OSError("")).endswith("\n")
