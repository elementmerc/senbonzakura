# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A model id that does not exist is told to the reader as a wrong id, not as a broken checkpoint.

WHAT PROMPTED IT, 2026-09-27

Audit finding 10. Running the abliterator against `Qwen/definitely-not-a-real-model-xyz` printed
this, on one line, measured at 325 characters:

    snapshot pre-flight SKIPPED, not passed: NotASafetensorsRepoError:
    'Qwen/definitely-not-a-real-model-xyz' is not a safetensors repo. Couldn't find
    'model.safetensors.index.json' or 'model.safetensors' files.. The host-RAM check still runs
    once the model is loaded, which on rented hardware is after you have paid for the download.

Three separate defects in one sentence. It named a Python exception class, which is the one fact in
there the reader can do nothing with. It had a doubled full stop, because the library's message ends
in one and ours adds another. And what it actually told them was false in the way that matters: the
repository is not missing its safetensors, the repository is not there at all, and a reader who
believes the first reading goes looking for a checkpoint format problem in a model that does not
exist.

The Hub cannot tell the library which it was, because a repository that is absent and a private one
you cannot see both answer 401. So the question is asked directly, once, and only after something
has already failed.

THE RULE

  * a model id the Hub says it does not have is reported as an id problem, and the id is echoed
    back, because a typo is only visible when the reader sees what was typed;
  * no exception class name reaches a user;
  * no doubled full stop;
  * the line is wrapped, so the reason is read rather than skipped;
  * an unreachable Hub is NOT reported as a missing model, because that sends somebody to correct
    something that was already correct.
"""
from __future__ import annotations

import pytest

from senbonzakura import cli, hubmessage, say

MISSING = "Qwen/definitely-not-a-real-model-xyz"

#: What `huggingface_hub` actually raises for that id, class name and trailing full stop included.
class NotASafetensorsRepoError(Exception):
    pass


THEIRS = NotASafetensorsRepoError(
    f"'{MISSING}' is not a safetensors repo. Couldn't find 'model.safetensors.index.json' or "
    f"'model.safetensors' files.")


@pytest.fixture
def reason(monkeypatch):
    monkeypatch.setattr(hubmessage, "repo_is_missing", lambda *a, **k: True)
    return cli._why_the_headers_could_not_be_read(MISSING, THEIRS)


def test_the_id_is_named_as_the_thing_that_is_wrong(reason):
    """A typo is the likeliest cause and the only one the reader can act on."""
    assert MISSING in reason
    assert "no model called" in reason


def test_it_is_not_described_as_a_safetensors_problem(reason):
    """The exact wrong reading the finding was raised about."""
    assert "safetensors" not in reason, (
        "a repository that does not exist was reported as a repository whose weights are the wrong "
        "format, which sends the reader to fix a checkpoint that is not there")


def test_no_exception_class_name_reaches_the_user(reason):
    assert "NotASafetensorsRepoError" not in reason
    assert "Error:" not in reason


def test_an_unreachable_hub_is_not_reported_as_a_missing_model(monkeypatch):
    """The asymmetric cost: guessing "missing" on a network failure is the worse wrong answer."""
    monkeypatch.setattr(hubmessage, "repo_is_missing", lambda *a, **k: False)
    said = cli._why_the_headers_could_not_be_read(
        MISSING, OSError("the Hub is unreachable"))
    assert "no model called" not in said
    assert "unreachable" in said, "the reason still has to reach the reader"


def test_a_local_folder_is_not_described_as_a_hub_problem(monkeypatch):
    """A path that is a directory was never an id, so none of the id advice applies to it."""
    monkeypatch.setattr(hubmessage, "repo_is_missing",
                        lambda *a, **k: pytest.fail("the Hub must not be asked about a folder"))
    said = cli._why_the_headers_could_not_be_read("/models/mine", OSError("boom"), local=True)
    assert "/models/mine" in said
    assert "Hub" not in said


# ── the line the user actually sees ───────────────────────────────────────────────────────────

class _Args:
    model = MISSING
    hf_token = None
    skip_conv_ablation = False


@pytest.fixture
def printed(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(cli, "estimate_snapshot_bytes",
                        lambda *a, **k: (None, hubmessage.no_such_model(MISSING)))
    lines = []
    assert cli.preflight_snapshot_ram(_Args(), log=lines.append) is None
    return lines


def test_the_skipped_line_is_wrapped(printed):
    """325 characters on one line is a wall, and a wall is what a reader skips."""
    assert len(printed) > 1, "it was emitted as a single line again"
    for line in printed:
        assert len(line) <= say.CEILING, f"{len(line)} columns: {line!r}"


def test_skipped_is_still_distinguishable_from_passed(printed):
    """Wrapping must not break the phrase that carries the whole meaning of the message."""
    assert any("SKIPPED, not passed" in line for line in printed)


def test_there_is_no_doubled_full_stop(printed):
    joined = " ".join(printed)
    assert ".." not in joined, f"a doubled full stop survived: {joined!r}"


def test_the_reason_and_the_consequence_both_survive(printed):
    joined = " ".join(printed)
    assert MISSING in joined
    assert "host-RAM check still runs" in joined, (
        "the point of skipped-not-passed is that the reader knows what still has to happen")


# ── asking the Hub, which is the half that decides which sentence gets printed ─────────────────
#
# Every test above stubs `repo_is_missing`, so without these it would never run. It is the seam the
# whole wording turns on, and the answers are deliberately lopsided: only a positive "not there"
# from the Hub counts, because the other reading sends a reader to correct a correct id.

class _Boom:
    def __init__(self, error, *, accept_timeout=True):
        self.error, self.accept_timeout, self.seen = error, accept_timeout, {}

    def repo_info(self, repo_id, **kw):
        if "timeout" in kw and not self.accept_timeout:
            raise TypeError("repo_info() got an unexpected keyword argument 'timeout'")
        self.seen.update(kw, repo_id=repo_id)
        if self.error is not None:
            raise self.error
        return object()


def _hub(monkeypatch, api):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: api)
    return api


def test_a_repository_the_hub_denies_is_missing(monkeypatch):
    from huggingface_hub.errors import RepositoryNotFoundError
    # Built without `__init__` on purpose: this class requires the `response` object it was raised
    # from, and what is under test is the type, not anything the response carries.
    _hub(monkeypatch, _Boom(RepositoryNotFoundError.__new__(RepositoryNotFoundError)))
    assert hubmessage.repo_is_missing("owner/nope") is True


def test_a_repository_the_hub_returns_is_not_missing(monkeypatch):
    api = _hub(monkeypatch, _Boom(None))
    sentinel = "not-a-real-token"
    assert hubmessage.repo_is_missing("owner/real", token=sentinel) is False
    assert api.seen["token"] == sentinel, (
        "a private repository is only visible to a request that carries one")


def test_an_unreachable_hub_is_not_missing(monkeypatch):
    """The asymmetric answer, at the seam rather than at the wording."""
    _hub(monkeypatch, _Boom(OSError("connection refused")))
    assert hubmessage.repo_is_missing("owner/real") is False


def test_the_question_is_asked_with_a_deadline(monkeypatch):
    """The Hub is only asked to improve the wording of a failure that has already happened, so it is
    not allowed to be the slowest part of it. `repo_exists` takes no timeout, which is why it is not
    what gets called.
    """
    api = _hub(monkeypatch, _Boom(None))
    hubmessage.repo_is_missing("owner/real")
    assert api.seen["timeout"] == hubmessage.ASK_TIMEOUT


def test_a_library_too_old_for_a_deadline_is_still_asked(monkeypatch):
    """Degrade, do not fail: an older `huggingface_hub` whose repo_info takes no timeout must still
    get an answer, or the good sentence would be lost on exactly the installs that are hardest to
    debug.
    """
    api = _hub(monkeypatch, _Boom(None, accept_timeout=False))
    assert hubmessage.repo_is_missing("owner/real") is False
    assert "timeout" not in api.seen


def test_an_install_without_huggingface_hub_says_nothing_about_the_id(monkeypatch):
    """`huggingface_hub` is an extra. An install that cannot ask must not answer anyway."""
    import sys
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert hubmessage.repo_is_missing("owner/nope") is False
