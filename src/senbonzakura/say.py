# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""How this tool says a long thing: wrapped prose, verbatim commands, untouched markers.

WHY THIS EXISTS, 2026-09-27

An audit of all 66 user-facing surfaces against a built wheel found that every argparse-generated
page wraps correctly, with not one line over 79 columns across 26 help pages, and that every
hand-written body ignores the terminal entirely. The worst were a 355-character `doctor` advisory, a
352-character `--resume` refusal and a 349-character `--gen-tokens` refusal. The CONTENT of those is
the best in the tool: they explain what a thing costs, name the flag to change, and give a command
you can run. They were simply emitted as one line each, which a terminal turns into a wall, and a
wall is what a reader skips.

It also found four separate wrapping implementations already in the tree, at three different widths,
one of them 94 columns. That is the duplicated-fact problem this project keeps finding, so this is
one definition and the others become callers.

THREE KINDS OF LINE, AND ONLY ONE OF THEM GETS WRAPPED

**Prose** wraps. That is the point.

**A command does not.** These messages are full of runnable lines like
`senbonzakura baseline result.json --seeds 42`, and a wrapped command cannot be pasted, which
destroys the most useful thing in the message in the name of tidiness. Any line that arrives already
indented is passed through untouched, because in every one of these messages indentation is what
marks an example.

**A marker does not.** `MARGIN_DONE auc=0.8062 ci=[0.6904,0.9067] ...` is read by machines: CI greps
nine such needles to decide whether a release is fit to ship, `head-to-head` parses fields off them,
and a runpod bootstrap gates an unattended job on them. Wrapping those would be a change to a
machine interface dressed up as a readability fix. They stay on one line, deliberately, and the
exemption is declared here and asserted by a test rather than left as an accident.

The operator chose this split explicitly on 2026-09-27 over the alternative of wrapping everything,
having been told that the alternative could break the head-to-head harness.
"""
from __future__ import annotations

import os
import re
import shutil
import textwrap

#: Nothing this tool prints assumes more than this. 80 is the floor a terminal is allowed to be, and
#: the last column is left alone because a line that exactly fills the width wraps anyway on some
#: terminals and produces a blank line on others.
CEILING = 79

#: A machine marker: SHOUTED, at the start of a line, at least four characters. Deliberately matched
#: on shape rather than on a list of known names, so a marker added later is exempt without anybody
#: having to remember this file. `DONE`, `MARGIN_DONE`, `SCORE_DONE`, `MARGIN_NULLS_DOMINATE` and
#: `MARGIN_READOUT_SUSPECT` all match; an ordinary sentence starting with a capital does not.
MARKER = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b|^[A-Z]{4,}\b")


def width(fallback=80):
    """The usable width: the terminal's, capped at `CEILING`, never below 40.

    `COLUMNS` is honoured first because that is what a caller sets to ask for a specific width, and
    every test in this project that checks wrapping sets it. A terminal that cannot be measured
    yields `fallback`, which is the conservative answer rather than an unbounded one.
    """
    try:
        columns = int(os.environ["COLUMNS"])
    except (KeyError, ValueError):
        try:
            columns = shutil.get_terminal_size(fallback=(fallback, 24)).columns
        except (OSError, ValueError):
            columns = fallback
    return max(40, min(columns, CEILING))


def is_marker(line):
    """Whether this line is a machine marker and must be left exactly as it is."""
    return bool(MARKER.match(line.lstrip()))


def is_verbatim(line):
    """Whether this line must be passed through untouched.

    An indented line is an example or a command in every message in this tool, and a wrapped command
    cannot be pasted. A marker is a machine interface. Neither is prose.
    """
    return line != line.lstrip() or is_marker(line)


def lines(text, *, indent="", first=None, columns=None):
    """`text` as a list of lines, none wider than the terminal, commands and markers intact.

    `indent` prefixes continuation lines, `first` the opening one (defaulting to `indent`). Blank
    lines are preserved, because the whitespace between paragraphs is half of what makes these
    messages readable and the audit that prompted this file complained about its absence as often as
    about the width.
    """
    limit = columns or width()
    lead = indent if first is None else first
    out = []
    for block in str(text).split("\n"):
        if not block.strip():
            out.append("")
            continue
        if is_verbatim(block):
            # Passed through EXACTLY, including its own indentation. Not re-indented: an example's
            # alignment is information, and a command's text is something to paste.
            out.append(block)
            continue
        wrapped = textwrap.wrap(" ".join(block.split()), width=limit,
                                initial_indent=lead if not out else indent,
                                subsequent_indent=indent,
                                break_long_words=False, break_on_hyphens=False)
        # A block of only a very long unbreakable token (a path, a URL) survives as itself rather
        # than vanishing, which is what an empty wrap result would mean.
        out.extend(wrapped or [lead + block.strip()])
    return out


def say(text, *, indent="", first=None, log=print, columns=None):
    """Print `text`, wrapped. The one call every long message in this tool should go through."""
    for line in lines(text, indent=indent, first=first, columns=columns):
        log(line)


def refusal(head, body, *, log=print, columns=None):
    """A refusal: one shouted line, then the reason and the next step, indented and wrapped.

    The shape every good refusal in this tool already had before they were wrapped, made reusable so
    the next one does not have to reinvent the layout. `head` is expected to be short; it is printed
    as given so a caller can put a marker there.
    """
    log(head)
    say(body, indent="  ", log=log, columns=columns)
