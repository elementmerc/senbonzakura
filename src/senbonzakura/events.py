# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`--json-events`: a machine-readable account of a run, beside the human one.

WHY A SECOND STREAM RATHER THAN PARSING THE FIRST

The log this tool prints is written for a person: it changes when someone improves a sentence, it
reorders when a phase moves, and it carries numbers formatted for reading rather than for parsing.
Anything that consumed it would be regex-matching prose like `trial 31/200 ... refusals=25.0%`, and
it would break on the next wording change with no test able to notice.

So the structured stream is emitted deliberately at the points that matter, and the human log is
left alone. Both come from the same call sites, which is what stops them drifting apart: a phase
that emits an event and forgets to log, or logs and forgets to emit, is visible in one place.

WHY IT NEVER TOUCHES STDOUT

A run spec's success check is `stdout-contains`, and its log is the evidence behind a published
number. One JSON line in the middle of that is a failed run at best and a corrupted record at
worst. Events go to a file, or to stderr with `-`, and stdout stays byte-identical to a run without
this flag.

WHAT IT IS FOR

A GUI wanting a live progress bar, a supervisor deciding whether to intervene, and a later
reconstruction of what a run did and when. It is NOT the artefact: `abliteration.json` remains the
record of what a run PRODUCED, and nothing here is required to interpret it.
"""
from __future__ import annotations

import json
import os
import sys
import time

#: Bumped when a consumer would have to change. New optional fields do not bump it; a renamed or
#: removed field, or a changed meaning, does.
SCHEMA = "senbonzakura-events/1"

#: Every event carries these. `elapsed` is monotonic seconds since the log opened, which is what a
#: progress display actually wants; `ts` is wall-clock for correlating with other systems' logs.
#: Both are deliberately absent from the artefact, which has to stay reproducible.
_ALWAYS = ("schema", "kind", "seq", "elapsed", "ts")


class EventLog:
    """A JSONL sink that cannot take a run down with it.

    Deliberately forgiving in one specific way: if the destination goes away mid-run, a GUI having
    closed its pipe should not kill an abliteration that is twenty minutes in and holding a model.
    The failure is reported once, on the human log, and the run continues without events. That is
    the graceful-degradation rule rather than an exception to fail-loud: the events are a view of
    the work, not the work.
    """

    def __init__(self, dest=None, *, log=None, clock=time.monotonic, wall=time.time):
        self._log = log or (lambda _m: None)
        self._clock, self._wall = clock, wall
        self._t0 = clock()
        self._seq = 0
        self._fh = None
        self._own = False
        self._broken = False
        if dest is None:
            return
        if dest == "-":
            self._fh = sys.stderr
        else:
            # Line-buffered so a consumer tailing the file sees each event as it happens rather
            # than in 8 KB bursts, which would make a progress bar lurch.
            # Not a context manager on purpose: the handle lives as long as the run does, and
            # `close()` is what ends it. A `with` here would close the file before the first event.
            self._fh = open(dest, "w", buffering=1, encoding="utf-8")  # noqa: SIM115
            self._own = True

    @property
    def enabled(self):
        return self._fh is not None and not self._broken

    def emit(self, kind, /, **fields):
        """One event. Unknown fields are passed through, so a call site can add detail freely.

        `kind` is POSITIONAL-ONLY, and that slash is load-bearing. Without it, a call site passing
        `kind=` as a field raises TypeError before any filtering happens, which would crash a run
        from inside the one class whose whole promise is that it cannot. With it, the collision
        becomes an ordinary field name that the envelope guard below drops.
        """
        if not self.enabled:
            return
        self._seq += 1
        rec = {"schema": SCHEMA, "kind": kind, "seq": self._seq,
               "elapsed": round(self._clock() - self._t0, 3), "ts": round(self._wall(), 3)}
        # Call-site fields never overwrite the envelope: an event claiming its own `kind` or `seq`
        # would make the stream unorderable, and that is worth refusing rather than accepting.
        rec.update({k: v for k, v in fields.items() if k not in _ALWAYS})
        try:
            self._fh.write(json.dumps(rec, default=_stringify) + "\n")
        except (OSError, ValueError) as e:
            self._broken = True
            self._log(f"  json-events: the destination stopped accepting writes ({e}); "
                      f"the run continues without them")

    def close(self):
        if self._own and self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _stringify(o):
    """Anything json cannot take becomes its repr rather than an exception.

    A run must not die because a field held a Path or a tensor. The consumer sees a string and can
    say so; the alternative is a crash at minute twenty of a job holding a model.
    """
    try:
        return str(os.fspath(o))
    except TypeError:
        return repr(o)


def open_for(args, *, log=None):
    """The log an argv-driven command should use, from `--json-events` if it is set."""
    return EventLog(getattr(args, "json_events", None), log=log)


def add_argument(parser):
    """The flag, worded the same wherever it appears."""
    parser.add_argument(
        "--json-events", default=None, metavar="PATH",
        help="write a machine-readable JSONL account of the run to PATH, or to stderr with '-'. "
             "Never written to stdout, so a redirected run stays byte-identical without it")
