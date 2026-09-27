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
