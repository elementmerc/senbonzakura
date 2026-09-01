# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`--json-events`: the machine-readable stream a GUI or a supervisor consumes.

The reason it exists at all is that the human log is written for a person and changes whenever
someone improves a sentence. Anything parsing it would be regex-matching prose, and would break on
a wording change with no test able to notice. These tests hold down the two properties that make
the second stream worth having: it never contaminates stdout, and it cannot take a run down.
"""
import json

import pytest

from senbonzakura import events


def _lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ── the envelope ─────────────────────────────────────────────────────────────────
def test_every_event_carries_the_envelope(tmp_path):
    p = tmp_path / "e.jsonl"
    with events.EventLog(p) as log:
        log.emit("trial", number=3, kl=0.04)
    (rec,) = _lines(p)
    assert rec["schema"] == events.SCHEMA
    assert rec["kind"] == "trial"
    assert rec["seq"] == 1
    assert "elapsed" in rec and "ts" in rec
    assert rec["number"] == 3 and rec["kl"] == 0.04


def test_sequence_numbers_are_monotonic_and_gapless(tmp_path):
    """A consumer has to be able to tell a dropped event from a slow one."""
    p = tmp_path / "e.jsonl"
    with events.EventLog(p) as log:
        for i in range(5):
            log.emit("trial", number=i)
    assert [r["seq"] for r in _lines(p)] == [1, 2, 3, 4, 5]


def test_elapsed_is_monotonic_from_the_log_opening(tmp_path):
    p = tmp_path / "e.jsonl"
    ticks = iter([100.0, 100.5, 102.0])
    log = events.EventLog(p, clock=lambda: next(ticks), wall=lambda: 0.0)
    log.emit("a")
    log.emit("b")
    log.close()
    assert [r["elapsed"] for r in _lines(p)] == [0.5, 2.0]


def test_a_call_site_cannot_overwrite_the_envelope(tmp_path):
    """An event claiming its own `seq` or `kind` would make the stream unorderable. Refused rather
    than accepted, because a consumer trusting `seq` has no way to detect the lie.
    """
    p = tmp_path / "e.jsonl"
    with events.EventLog(p) as log:
        log.emit("trial", seq=999, kind="something-else", schema="not-ours", number=1)
    (rec,) = _lines(p)
    assert rec["seq"] == 1 and rec["kind"] == "trial" and rec["schema"] == events.SCHEMA
    assert rec["number"] == 1


# ── the property that protects every published number ────────────────────────────
def test_nothing_is_written_when_the_flag_is_absent(capsys, tmp_path):
    """The whole point: a run without the flag is byte-identical to one before this existed."""
    log = events.EventLog(None)
    assert not log.enabled
    log.emit("trial", number=1)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_stderr_is_the_only_stream_it_will_ever_write_to(capsys):
    """`-` means stderr, never stdout. A run spec's success check is `stdout-contains` and its log
    is the evidence behind a published number; one JSON line in there is a corrupted record.
    """
    log = events.EventLog("-")
    log.emit("trial", number=1)
    out = capsys.readouterr()
    assert out.out == "", "an event reached stdout"
    assert json.loads(out.err)["kind"] == "trial"


# ── it cannot take a run down ────────────────────────────────────────────────────
def test_a_destination_that_dies_mid_run_does_not_raise(tmp_path):
    """A GUI closing its pipe must not kill an abliteration twenty minutes in holding a model."""
    p = tmp_path / "e.jsonl"
    msgs = []
    log = events.EventLog(p, log=msgs.append)
    log.emit("first")
    log._fh.close()                      # the consumer went away
    log.emit("second")                   # must not raise
    log.emit("third")
    assert not log.enabled, "a broken sink kept claiming to be enabled"
    assert any("continues without them" in m for m in msgs)
    assert len(msgs) == 1, "the failure was reported more than once"


def test_an_unserialisable_field_becomes_text_rather_than_an_exception(tmp_path):
    p = tmp_path / "e.jsonl"

    class Odd:
        def __repr__(self):
            return "<an odd object>"

    with events.EventLog(p) as log:
        log.emit("thing", weird=Odd(), path=tmp_path)
    (rec,) = _lines(p)
    assert rec["weird"] == "<an odd object>"
    assert rec["path"] == str(tmp_path)


def test_closing_twice_is_safe(tmp_path):
    log = events.EventLog(tmp_path / "e.jsonl")
    log.close()
    log.close()
    log.emit("after")   # no destination, no error


def test_it_does_not_close_a_stream_it_does_not_own(capsys):
    """Stderr belongs to the process, not to us."""
    log = events.EventLog("-")
    log.close()
    import sys
    assert not sys.stderr.closed


# ── the flag, worded once ────────────────────────────────────────────────────────
def test_the_flag_is_added_with_a_consistent_name():
    import argparse
    ap = argparse.ArgumentParser()
    events.add_argument(ap)
    a = ap.parse_args(["--json-events", "x.jsonl"])
    assert a.json_events == "x.jsonl"
    assert ap.parse_args([]).json_events is None


def test_open_for_reads_the_flag_off_a_namespace(tmp_path):
    import types
    log = events.open_for(types.SimpleNamespace(json_events=str(tmp_path / "e.jsonl")))
    assert log.enabled
    log.close()


def test_open_for_is_a_no_op_when_the_attribute_is_missing():
    """Callers that predate the flag, and tests building their own args, must not break."""
    import types
    assert not events.open_for(types.SimpleNamespace()).enabled


# ── the run emits the events a progress display needs ────────────────────────────
@pytest.mark.parametrize("kind", ["model_loaded", "phase", "trial", "done"])
def test_the_abliterator_emits_the_documented_kinds(kind):
    """Held down by name because a consumer keys off them: renaming one is a schema change, and
    this test is what makes that a deliberate act rather than a surprise.
    """
    import inspect

    from senbonzakura import cli
    src = inspect.getsource(cli)
    assert f'emit("{kind}"' in src, f"nothing emits a {kind!r} event any more"
