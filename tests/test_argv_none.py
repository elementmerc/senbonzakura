# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A flag handed the word "None", and the run that died at argument parsing after 108 minutes.

WHAT HAPPENED

`head-to-head --skip-harmful` defaults to None, meaning "read the held-out boundary from the
track's own manifest". Two places compose a child command from that value. The compass path
omits the flag when it is None, correctly. The refusal path interpolated it, so the child
received the four characters N-o-n-e:

    senbonzakura.score: error: argument --skip: invalid int value: 'None'

Ten arms, all trained, all scored zero times. The run had already spent 108 minutes and every
model was on disk; what died was the cheap half, at parsing, in under a second, ten times.

WHY THE FIX IS AT THE RUNNER RATHER THAN AT THE COMPOSER

Fixing that one composer fixes that one composer, and there are around thirty places in this
codebase that build an argv by putting `str(value)` next to a flag. `str()` renders anything.
So the guard sits at the choke point every launch goes through, and a composer written next
month is covered without anybody remembering this happened. The specific composer is fixed too,
and it resolves the boundary from the manifest rather than defaulting: `score --skip` defaults
to 0, which would have counted refusals on the very rows the search selected on, so an
unresolved boundary has to be a refusal rather than a fallback.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from senbonzakura import headtohead


def test_a_flag_given_the_word_none_is_refused_before_anything_is_launched():
    with pytest.raises(headtohead.BenchError) as e:
        headtohead.check_argv(["python", "-m", "senbonzakura", "score", "--skip", "None"])
    assert "--skip" in str(e.value), "the message must name the flag that was not resolved"


def test_the_guard_runs_before_the_child_starts(monkeypatch):
    """Catching it after launch would be a slower way of learning the same thing."""
    started = []
    monkeypatch.setattr(headtohead.subprocess, "run",
                        lambda *a, **k: started.append(a) or (_ for _ in ()).throw(AssertionError))
    with pytest.raises(headtohead.BenchError):
        headtohead.default_runner(["python", "--skip", "None"], log=lambda _m: None)
    assert not started


def test_a_bare_none_that_is_not_a_flag_value_is_left_alone():
    """A label or a filename may legitimately be the word None, and refusing those is a worse
    rule than no rule: it would fail runs that are correct.
    """
    headtohead.check_argv(["python", "-m", "x", "None"])
    headtohead.check_argv(["python", "--label", "arm", "None"])


def test_a_flag_that_looks_like_a_short_option_is_not_matched():
    """Only `--flag None` is the pattern; `-n None` would be, but a single dash is also how
    negative numbers and file arguments arrive, so the rule stays narrow and says so.
    """
    headtohead.check_argv(["python", "-x", "None"])


def test_the_refusal_argv_refuses_an_unresolved_boundary(tmp_path):
    """THE ORIGINAL DEFECT, at the composer, asserted on the shipped function.

    Not "does it produce the right string" but "does it refuse the wrong input", because the
    wrong input is what it silently accepted for as long as this code has existed.
    """
    with pytest.raises(headtohead.BenchError, match="not a count"):
        headtohead.refusal_argv(model=tmp_path, harmful=tmp_path, out=tmp_path,
                                label="x", skip=None, batch=16)


@pytest.mark.parametrize("field", ["skip", "batch", "n"])
def test_every_count_it_interpolates_is_checked_not_just_the_one_that_broke(field, tmp_path):
    kw = {"model": tmp_path, "harmful": tmp_path, "out": tmp_path, "label": "x",
          "skip": 128, "batch": 16, "n": 200}
    kw[field] = None
    with pytest.raises(headtohead.BenchError, match=field):
        headtohead.refusal_argv(**kw)


def test_a_boolean_is_not_accepted_as_a_count(tmp_path):
    """`isinstance(True, int)` is True in Python, so a bool sails through a naive check and
    renders as the word 'True'. Same class of defect, one type along.
    """
    with pytest.raises(headtohead.BenchError):
        headtohead.refusal_argv(model=tmp_path, harmful=tmp_path, out=tmp_path,
                                label="x", skip=True, batch=16)


def test_a_resolved_boundary_produces_the_command_it_should(tmp_path):
    argv = headtohead.refusal_argv(model=Path("m"), harmful=Path("h"), out=Path("o"),
                                   label="senbon-seed41", skip=128, batch=16, n=200)
    assert "--skip" in argv
    assert argv[argv.index("--skip") + 1] == "128"
    headtohead.check_argv(argv), "the composer must produce something the guard accepts"
