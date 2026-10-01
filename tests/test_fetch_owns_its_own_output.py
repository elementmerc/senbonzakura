# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura fetch` says the anonymous-download thing itself, and loses nothing upstream says.

WHAT PROMPTED IT, 2026-10-01

A CLI surface sweep item: `huggingface_hub` prints "You are sending unauthenticated requests ..."
to stderr above this command's own output, unwrapped and in upstream's voice, for the ordinary case
of downloading a public file. Same class as the vendored converter's 350 lines of per-tensor
chatter, which is what `vendored.relay` was written for.

WHY NOT `vendored.relay`, WHICH IS WHAT THE LEDGER ITEM NAMED

`relay` reads a subprocess's pipe. `huggingface_hub` runs in this process and writes through
`logging`, so the idea transfers and the function does not: take the library's handlers for the
duration, collect what it emits, and re-emit it through `say` afterwards.

THE PROPERTY THAT MATTERS MOST HERE IS THE ONE ABOUT NOT LOSING THINGS

Setting the library's verbosity to error would be one line and would hide a genuine warning, and
this project refuses that trade everywhere else. So the test that carries the most weight in this
file is `test_an_unrelated_warning_is_not_swallowed`, not the one about the nudge: dropping the
nudge is cosmetic, and dropping a Xet-storage or disk warning is a silent wrong answer.

The match on the nudge is loose and fails safe by design, so an upstream rewording we no longer
recognise is passed through. The worst outcome is a reader told twice, never a warning lost, and
`test_an_unrecognised_rewording_is_passed_through_rather_than_dropped` holds that direction.
"""
from __future__ import annotations

import logging

import pytest

from senbonzakura import fetch, say

#: Verbatim from huggingface_hub 0.34, which is the floor `pyproject.toml` pins.
NUDGE = ("You are sending unauthenticated requests to Hugging Face Hub. Please log in to get "
         "higher rate limits.")

#: Not a credential. A literal here would be flagged as one, and the point of the tests that use
#: it is only that `token` is truthy.
A_TOKEN = "hf_" + "notarealtoken"

#: A warning that is nothing to do with credentials and must survive.
REAL_WARNING = ("Xet Storage is enabled for this repo but the cache is on a filesystem that does "
                "not support it, so transfers fall back to the HTTP path")


def _run(records, *, token=None, columns="80", monkeypatch=None):
    """Drive the context manager and return the lines it produced, in order.

    `records` is a list of (level, message) the library emits while the transfer is in flight.
    """
    if monkeypatch is not None:
        monkeypatch.setenv("COLUMNS", columns)
    out = []
    logger = logging.getLogger("huggingface_hub")
    with fetch.hub_logging_through_us(out.append, token=token):
        for level, message in records:
            logger.log(level, message)
    return out


def test_the_nudge_is_replaced_by_our_own_sentence(monkeypatch):
    lines = _run([(logging.WARNING, NUDGE)], monkeypatch=monkeypatch)
    said = " ".join(" ".join(lines).split())
    assert "no Hugging Face token is set" in said, said
    assert "You are sending unauthenticated requests" not in said, said


def test_our_sentence_names_the_cost_and_not_only_the_remedy(monkeypatch):
    """A reader downloading a public file does not need an account.

    Nudging them towards one without saying what it buys is how a notice becomes an instruction
    somebody follows for no reason.
    """
    said = " ".join(" ".join(_run([], monkeypatch=monkeypatch)).split())
    assert "works for public files" in said, said
    assert "rate limit" in said, said
    assert "HF_TOKEN" in said, said


def test_nothing_is_said_about_anonymity_when_a_token_is_set(monkeypatch):
    lines = _run([], token=A_TOKEN, monkeypatch=monkeypatch)
    assert lines == [], lines


def test_an_unrelated_warning_is_not_swallowed(monkeypatch):
    """The test that carries the weight. A lost warning is a silent wrong answer."""
    said = " ".join(" ".join(_run([(logging.WARNING, REAL_WARNING)],
                                  monkeypatch=monkeypatch)).split())
    assert "Xet Storage" in said, said
    assert "huggingface_hub" in said, "the line no longer says whose voice it is"


def test_an_unrecognised_rewording_is_passed_through_rather_than_dropped(monkeypatch):
    """The failure direction is chosen, not accidental.

    If upstream rewrites the nudge into words none of `_LOGIN_NUDGE`'s markers match, the reader
    is told twice. That is the right way round: a notice repeated is a cosmetic defect and a
    warning deleted is not.
    """
    reworded = "Anonymous access detected; quotas may apply to this transfer."
    said = " ".join(" ".join(_run([(logging.WARNING, reworded)], monkeypatch=monkeypatch)).split())
    assert "quotas may apply" in said, said


def test_the_nudge_is_kept_when_a_token_was_given(monkeypatch):
    """With a token set, "unauthenticated" stops being noise and becomes information.

    It means the token was not used, which is a real finding about the run and the opposite of
    the case this filter exists for.
    """
    said = " ".join(" ".join(_run([(logging.WARNING, NUDGE)], token=A_TOKEN,
                                  monkeypatch=monkeypatch)).split())
    assert "unauthenticated requests" in said, said


def test_an_info_line_is_not_promoted_to_a_warning(monkeypatch):
    """Collected at DEBUG so nothing is lost, reported from WARNING so nothing is noise."""
    lines = _run([(logging.INFO, "resolving the revision")], token=A_TOKEN, monkeypatch=monkeypatch)
    assert lines == [], lines


@pytest.mark.parametrize("columns", ["80", "120", "40"])
def test_every_line_fits_the_terminal(columns, monkeypatch):
    """The first draft of the fix called `log` directly and printed a 227-column line.

    That is the defect the function exists to fix, moved one voice over, which is why this is
    asserted over both the notice and a long re-emitted warning.
    """
    lines = _run([(logging.WARNING, REAL_WARNING)], columns=columns, monkeypatch=monkeypatch)
    assert lines
    for line in lines:
        assert len(line) <= say.CEILING, f"{len(line)} columns at COLUMNS={columns}: {line!r}"


# ── the part that has to leave the process as it found it ───────────────────────
def _logger_state():
    logger = logging.getLogger("huggingface_hub")
    return (list(logger.handlers), logger.propagate, logger.level, logger.disabled)


def test_the_logger_is_left_exactly_as_it_was():
    """The handler swap is global to the process while it is held.

    A leak here is not a cosmetic defect: every later `huggingface_hub` message in the process
    would be collected by a handler nobody reads, which is the silent-suppression failure this
    function was written to avoid.
    """
    before = _logger_state()
    with fetch.hub_logging_through_us(lambda _line: None, token=A_TOKEN):
        pass
    assert _logger_state() == before


def test_the_logger_is_restored_even_when_the_transfer_raises():
    before = _logger_state()
    with pytest.raises(RuntimeError), fetch.hub_logging_through_us(lambda _l: None, token=A_TOKEN):
        raise RuntimeError("the transfer died")
    assert _logger_state() == before


def test_a_record_with_a_broken_format_string_does_not_take_the_run_with_it():
    """`record.getMessage()` interpolates, so upstream's own bug must not become ours.

    Not hypothetical defensiveness: this code reports on records it did not create, which is the
    boundary where validating is the rule rather than the exception.
    """
    out = []
    logger = logging.getLogger("huggingface_hub")
    with fetch.hub_logging_through_us(out.append, token=A_TOKEN):
        logger.warning("a count: %d and %s", "not a number")  # noqa: PLE1206 - the point
    assert out, "the record was dropped entirely"
    assert "a count" in " ".join(out)


def test_the_download_path_wraps_the_call_rather_than_the_module():
    """The context manager has to be around the transfer, not merely defined.

    Source text, deliberately and as a supplement: driving `download` needs `huggingface_hub`
    installed and a network, and this file runs where neither is guaranteed. The behaviour is
    covered by every test above; what this adds is that the call site uses it.
    """
    from pathlib import Path

    text = Path(fetch.__file__).read_text(encoding="utf-8")
    assert "with hub_logging_through_us(log, token=token):" in text, (
        "the hub download no longer runs inside the logging context, so upstream's output goes "
        "back over this command's own")
    body = text[text.index("def download("):]
    assert body.index("with hub_logging_through_us") < body.index("hf_hub_download(repo_id="), (
        "the context is entered after the transfer starts, which is too late to catch anything")
