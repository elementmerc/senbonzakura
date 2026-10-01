# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura track build` reports a Hub failure in words, not as `(ClassName: str(e))`.

WHAT PROMPTED IT, 2026-10-01

A CLI surface sweep item: "`trackbuild.py` has five refusals of the shape
`({type(e).__name__}: {e})`." Exactly five, and the module never imported `hubmessage`, so the one
command whose whole job is fetching from the Hub was the one command not using the project's Hub
vocabulary. That is the duplicated-fact shape in reverse: not two copies disagreeing, one copy and
a surface that never found it.

Both halves of `({type(e).__name__}: {e})` are unusable. The class name is ours to know and theirs
to ignore, since `HfHubHTTPError` tells somebody with a mistyped repository id nothing. And
`str(e)` from `huggingface_hub` runs to five paragraphs including a request id, a link to the
authentication docs, the `repo_id` and `repo_type` keyword arguments of a function the reader is
not calling, and, for an id that was simply mistyped, the sentence "Invalid username or password."

THE GUESS THAT WAS WRONG IN THE WORST CASE

Both revision refusals said "a pinned revision that has been deleted upstream is the likeliest
cause". When the repository itself has gone, or was never there, that sends somebody to re-pin a
revision inside a repository that does not exist. `hubmessage.repo_is_missing` answers that
question positively or not at all, so a machine with no network is never told its id is wrong,
which is the asymmetry that function was written for.

WHAT IS NOT ASSERTED HERE

The exact wording of `hubmessage`'s own filtering. That is `test_hubmessage.py`'s subject, and
duplicating it here would make two tests that have to be changed together.
"""
from __future__ import annotations

import pytest

from senbonzakura import hubmessage, trackbuild

#: A real `huggingface_hub` 404 body, which is the shape the old refusals pasted whole.
HUB_404 = """404 Client Error. (Request ID: Root=1-68dc0000-1234567890abcdef)

Repository Not Found for url: https://huggingface.co/api/datasets/owner/nope/revision/0000.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated.
For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password."""


class HfHubHTTPError(Exception):
    """Stands in for the upstream class, so the test does not need huggingface_hub installed."""


def _src(repo="owner/nope"):
    return {"repo": repo, "side": "harmful", "split": "train", "revision": "a" * 40,
            "licence": "apache-2.0", "note": ""}


@pytest.fixture(autouse=True)
def hub_is_silent(monkeypatch):
    """`repo_is_missing` answers False by default, which is the offline answer.

    Defaulted rather than left real, because the real one makes a network call and a test that
    quietly reaches the Hub is a test that fails on a train.
    """
    monkeypatch.setattr(hubmessage, "repo_is_missing", lambda *a, **kw: False)


def _refusal(error=None, repo="owner/nope"):
    """What `_repo_files` exits with when the listing call raises.

    Driven through `_repo_files` rather than through `_hub_refusal`, because what is under test is
    that the call site uses the helper. A test on the helper alone would pass with all five call
    sites still pasting a class name.
    """
    import sys
    import types

    error = error or HfHubHTTPError(HUB_404)

    def explode(*_a, **_kw):
        raise error

    fake = types.ModuleType("huggingface_hub")
    fake.list_repo_files = explode
    sys.modules["huggingface_hub"] = fake
    try:
        with pytest.raises(SystemExit) as exit_info:
            trackbuild._repo_files(_src(repo))
    finally:
        del sys.modules["huggingface_hub"]
    return str(exit_info.value)


def test_the_class_name_is_not_in_the_refusal():
    """The defect, stated as the property a reader cares about."""
    said = _refusal()
    assert "HfHubHTTPError" not in said, said


def test_the_advice_for_python_callers_is_dropped():
    """Every one of these answers a question the reader did not ask.

    Dropped by `hubmessage`, so this test is really asserting that the refusal goes through it at
    all. The four phrases are the ones a mistyped id actually produced.
    """
    said = _refusal()
    for noise in ("Request ID:", "repo_type", "Invalid username or password",
                  "https://huggingface.co/docs/huggingface_hub/authentication"):
        assert noise not in said, f"{noise!r} survived into: {said}"


def test_what_the_hub_said_is_still_there():
    """The other half, and the one that is easy to lose.

    The upstream text is the only thing in the refusal that knows what actually went wrong, so a
    fix that dropped all of it would trade an unusable refusal for a confident wrong one.
    """
    said = _refusal()
    assert "Repository Not Found" in said, said
    assert "What the Hub said" in said, said


def test_the_diagnosis_and_the_next_step_survive():
    said = _refusal()
    assert "re-pinning" in said, said
    assert "SOURCES" in said, said


def test_a_missing_repository_is_named_as_missing_rather_than_as_a_stale_revision(monkeypatch):
    """The wrong guess, which is worse than no guess.

    Re-pinning a revision inside a repository that does not exist is work that cannot succeed, so
    when the Hub positively says the repository has gone, that is what the refusal says.
    """
    monkeypatch.setattr(hubmessage, "repo_is_missing", lambda *a, **kw: True)
    said = _refusal()
    assert "there is no dataset called 'owner/nope'" in said, said
    assert "re-pinning" not in said, (
        f"it still tells the reader to re-pin a revision in a repository that is gone: {said}")


def test_an_unreachable_hub_is_not_reported_as_a_missing_repository():
    """The asymmetry `repo_is_missing` exists for, asserted at this call site.

    The autouse fixture answers False, which is what offline, gated and too-old-to-ask all give.
    None of them may become "your id is wrong".
    """
    said = _refusal()
    assert "there is no dataset" not in said, said


def test_a_long_hub_error_is_bounded_and_says_so():
    """`huggingface_hub` can produce five paragraphs; a refusal that pastes them is the defect."""
    many = HfHubHTTPError("\n".join(f"upstream detail line {i}" for i in range(30)))
    said = _refusal(error=many)
    shown = sum("upstream detail line" in line for line in said.splitlines())
    assert shown == trackbuild._HUB_LINES, f"{shown} lines shown: {said}"
    assert f"and {30 - trackbuild._HUB_LINES} more line(s)" in said, said


def test_a_short_hub_error_gets_no_cut_notice():
    """A notice that always appears carries no information."""
    said = _refusal(error=HfHubHTTPError("the volume is full"))
    assert "the volume is full" in said
    assert "more line(s)" not in said, said


def test_an_error_with_nothing_usable_left_does_not_print_an_empty_heading():
    """Every line filtered away must not leave "What the Hub said:" standing over nothing."""
    said = _refusal(error=HfHubHTTPError("Request ID: Root=1-abc"))
    assert "What the Hub said" not in said, said
    assert "re-pinning" in said, "the diagnosis went with it"


def test_no_call_site_still_pastes_a_class_name():
    """The count, because this item was five occurrences and a partial fix is the usual outcome.

    Source text rather than behaviour, deliberately and as a supplement: four of the five call
    sites need a network stub each to drive, and the property worth guarding cheaply is that none
    of them came back. The one occurrence allowed is `_hub_refusal`'s own docstring, which quotes
    the shape it replaced.
    """
    from pathlib import Path

    text = Path(trackbuild.__file__).read_text(encoding="utf-8")
    assert text.count("type(e).__name__") == 1, (
        "a refusal is pasting an exception class name again; `_hub_refusal` is the shared way "
        "to report a Hub failure in this module")
    assert text.count("_hub_refusal(") == 6, (
        f"expected the definition plus five call sites, found {text.count('_hub_refusal(')}")
