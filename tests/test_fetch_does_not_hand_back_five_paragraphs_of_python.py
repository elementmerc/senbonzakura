# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`fetch` says what is wrong in a sentence, rather than forwarding the library's five paragraphs.

WHAT PROMPTED IT, 2026-09-27

Audit finding 11. `senbonzakura fetch owner/does-not-exist:model.gguf` printed this, verbatim:

    the Hub refused Nonexistent-Owner-xyz/definitely-not-a-real-repo-xyz:model.gguf: 401 Client
    Error. (Request ID: Root=1-6ab9552c-...;568a7158-...)

    Repository Not Found for url: https://huggingface.co/.../resolve/main/model.gguf.
    Please make sure you specified the correct `repo_id` and `repo_type`.
    If you are trying to access a private or gated repo, make sure you are authenticated and your
    token has the required permissions.
    For more details, see https://huggingface.co/docs/huggingface_hub/authentication
    Invalid username or password.. A gated or private repository needs a token; set $HF_TOKEN or run
    `hf auth login` rather than passing one as an argument.

Five paragraphs, and the reader meets four wrong explanations before the right one. `repo_id` and
`repo_type` are keyword arguments of a function they are not calling. The request id is for somebody
else's logs. "Invalid username or password." is not merely useless at a command line, it is FALSE
for the case that produces it most often: nobody typed a password, the id had a character missing,
and a reader told their credentials are wrong goes hunting for a token they never needed. There was
a doubled full stop in there too.

The repository being absent and a private repository you cannot see both answer 401, so the Hub is
asked which it was instead of being guessed at. When it says the repository is not there, that is
the whole message: a token cannot help, and offering one is the same wrong answer in a quieter
voice.

THE RULE

  * a repository the Hub does not have is named, and the message says so and stops;
  * nothing meant for a Python caller survives, and neither does the credentials line;
  * the message is wrapped;
  * a failure that is NOT a missing repository keeps the library's own useful words, because a
  filter that emptied the message would be worse than one that left too much.
"""
from __future__ import annotations

import pytest

from senbonzakura import fetch, hubmessage, say

REPO = "Nonexistent-Owner-xyz/definitely-not-a-real-repo-xyz"
FILENAME = "model.gguf"

#: The real shape of what `huggingface_hub` raises, reproduced on a CPU-only box on 2026-09-27.
THEIRS = (
    "401 Client Error. (Request ID: Root=1-6ab9552c-64feb10467035dd1310d3112;568a7158-123e)\n"
    "\n"
    f"Repository Not Found for url: https://huggingface.co/{REPO}/resolve/main/{FILENAME}.\n"
    "Please make sure you specified the correct `repo_id` and `repo_type`.\n"
    "If you are trying to access a private or gated repo, make sure you are authenticated and your "
    "token has the required permissions.\n"
    "For more details, see https://huggingface.co/docs/huggingface_hub/authentication\n"
    "Invalid username or password."
)


@pytest.fixture
def refusal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(hubmessage, "repo_is_missing", lambda *a, **k: True)
    return fetch._the_hub_said_no(REPO, FILENAME, OSError(THEIRS))


def test_the_repository_is_named_and_said_to_be_absent(refusal):
    assert REPO in refusal
    assert "no repository called" in refusal


def test_it_does_not_claim_a_credentials_problem(refusal):
    """The one line in the original that was actively false for the likeliest cause."""
    assert "Invalid username or password" not in refusal
    assert "password" not in refusal


@pytest.mark.parametrize("python", ["repo_id", "repo_type", "Request ID", "huggingface.co/docs"])
def test_nothing_meant_for_a_python_caller_survives(refusal, python):
    assert python not in refusal, f"{python!r} cannot be acted on by somebody holding a shell"


def test_it_is_short_and_wrapped(refusal):
    lines = refusal.split("\n")
    assert len(lines) <= 8, f"five paragraphs became {len(lines)} lines:\n{refusal}"
    for line in lines:
        assert len(line) <= say.CEILING, f"{len(line)} columns: {line!r}"


def test_there_is_no_doubled_full_stop(refusal):
    assert ".." not in refusal


def test_a_missing_file_in_a_real_repository_is_a_file_problem_not_a_token_problem(monkeypatch):
    """The repository answered, so it was read, so no token can change the outcome.

    Offering one anyway is the same defect as "Invalid username or password" in a politer voice: a
    second theory about credentials, arriving in front of the one thing that is actually wrong.
    """
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(hubmessage, "repo_is_missing", lambda *a, **k: False)
    said = fetch._the_hub_said_no(
        "real/repo", FILENAME,
        OSError("404 Client Error. Entry Not Found for url: .../model.gguf.\n"
                "Please make sure you specified the correct `repo_id` and `repo_type`."))
    assert "real/repo" in said and FILENAME in said
    assert "no file called" in said
    assert "no repository called" not in said, "the repository is there; only the file is not"
    assert "hf auth login" not in said, "a token cannot conjure a file the repository does not have"
    assert "repo_type" not in said


def test_an_unfamiliar_failure_keeps_the_libraries_own_reason(monkeypatch):
    """Degrade to "kept", never to "dropped". If upstream rewords, the reader must see a little too
    much rather than nothing: our own branches only recognise the shapes we have met.
    """
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(hubmessage, "repo_is_missing", lambda *a, **k: False)
    said = fetch._the_hub_said_no(
        "real/repo", FILENAME,
        OSError("503 Server Error: the Hub is in read-only maintenance\n"
                "Please make sure you specified the correct `repo_id` and `repo_type`."))
    assert "read-only maintenance" in said, "the only words that say what happened"
    assert "repo_type" not in said
    assert "hf auth login" in said, "a gated repo is still a live possibility on this branch"


def test_an_error_with_nothing_usable_in_it_still_refuses_loudly(monkeypatch):
    """The case most likely to print an empty message: every line was advice."""
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(hubmessage, "repo_is_missing", lambda *a, **k: False)
    # The blank line is deliberate: upstream separates its paragraphs with one, and an empty line is
    # neither advice to drop nor a reason to keep, so it must not become an empty line in ours.
    said = fetch._the_hub_said_no("real/repo", FILENAME,
                                  OSError("Invalid username or password.\n\n"))
    assert "real/repo" in said
    assert "without saying why" in said
    assert said.strip(), "a refusal that prints nothing is the worst outcome of a filter"
