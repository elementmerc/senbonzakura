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

#: A machine marker: a SHOUTED first token CARRYING AN UNDERSCORE. Matched on shape rather than
#: against a list of names, so `COHERENCE_DONE` or whatever is added next year is exempt without
#: anybody having to remember this file.
#:
#: THE UNDERSCORE IS THE WHOLE RULE, and the first version of it did not have one. It read
#: `^[A-Z]{4,}` on the theory that shouting was enough, which made `NOTE:` a machine marker. The
#: longest single message in the tool, an 808-character explanation of a filter that rejected
#: nothing, opens with `NOTE:` and was therefore passed through unwrapped by the very function
#: written to wrap it. `WARNING:`, `BROKEN FILTER:` and `MATCHING ACHIEVED NOTHING:` are the same
#: shape: they are emphasis, which is prose, and they were the reason this module was written.
#:
#: Every real marker in this project contains an underscore except the two below, so the rule is
#: cheap and the exceptions are named rather than inferred.
MARKER = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")

#: The markers with no underscore in them. Short, closed, and worth stating: `DONE` is the needle the
#: end-to-end smoke greps for to decide an abliteration finished.
BARE_MARKERS = frozenset({"DONE", "REFUSED", "OK", "FAIL"})


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
    stripped = line.lstrip()
    if MARKER.match(stripped):
        return True
    first = stripped.split(" ", 1)[0] if stripped else ""
    return first in BARE_MARKERS


def is_verbatim(line):
    """Whether this line must be passed through untouched.

    An indented line is an example or a command in every message in this tool, and a wrapped command
    cannot be pasted. A marker is a machine interface. Neither is prose.
    """
    return line != line.lstrip() or is_marker(line)


#: What a cut value ends with, so a reader can tell a shortened string from a short one. Three
#: ASCII dots rather than the single ellipsis character, because this travels through logs and
#: terminals whose encoding we do not control.
CUT = "..."


def shorten(text, limit):
    """`text` cut to `limit` characters, saying so, or unchanged when it already fits.

    WHY THE MARKER IS NOT OPTIONAL, 2026-09-27

    A surface audit found values cut to a column budget with nothing to show it: a path lost its
    last component and a GPU's identifier lost its tail, and both still read as complete. A cut
    path is worse than a long one, because a reader copies it, and a cut identifier is worse still,
    because two cards whose names share a prefix become one card.

    A limit too small to hold the marker is a caller error rather than a case to handle: at that
    width there is nothing to report, so the value is returned whole and the caller's layout is
    the thing that gives, not the truth of what is shown.
    """
    text = str(text)
    if len(text) <= limit or limit <= len(CUT):
        return text
    return text[:limit - len(CUT)] + CUT


def some_of(items, limit, *, joiner=", "):
    """`limit` of `items` and a count of the rest, so a cut list cannot read as a whole one.

    WHY THIS IS SHARED, 2026-09-27

    Five places in this package already said "and N more" by hand, and four others cut a list with
    nothing at all: `doctor` reported "7 failed to import" and then named six, `convert` named five,
    and a JSON object with twenty keys was described by eight of them. A reader who counts the names
    and gets a different number than the count they were given has to work out which one is lying.

    The count comes from the items rather than from the caller, which is the half that was getting
    lost: `f"{len(broken)} failed: {', '.join(broken[:6])}"` states the total in one clause and
    contradicts it in the next.
    """
    shown = [str(i) for i in items][:limit]
    rest = len(items) - len(shown)
    return joiner.join(shown) + (f", and {rest} more" if rest > 0 else "")


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


#: How deep an indent has to be before it means "verbatim" rather than "this is a continuation".
#: Messages in this tool are written with two-space continuation indents for prose and four or more
#: for a command or an example, so that is where the line sits.
VERBATIM_INDENT = 4


def reflow(text, *, columns=None):
    """A message written with two-space continuation indents, rewrapped, commands left alone.

    WHY THIS IS NEEDED ON TOP OF `lines`, 2026-09-27

    `is_verbatim` treats ANY indented line as something to pass through, which is right for a command
    and wrong for the way these messages are actually written: most of them are one logical paragraph
    per sentence with two-space continuation indents already baked into the string literal. Handed
    straight to `lines`, every one of those is "indented", so nothing wraps.

    A blast-radius pass predicted this before it happened, and the first attempt at rewrapping
    `baseline`'s refusals then printed a 261-character line to prove it.

    So a two-space indent is treated as prose to be rewrapped, and four or more as verbatim. Blank
    lines separate paragraphs.
    """
    out, para = [], []

    def flush():
        if para:
            out.extend(lines(" ".join(para), indent="  ", columns=columns))
            para.clear()

    for raw in str(text).split("\n"):
        if not raw.strip():
            flush()
            out.append("")
        elif raw.startswith(" " * VERBATIM_INDENT) or is_marker(raw):
            flush()
            out.append(raw)
        else:
            para.append(raw.strip())
    flush()
    return out


def refusal_text(head, *paragraphs, columns=None):
    """A refusal as a STRING, wrapped, for `raise SystemExit(...)`.

    The shape every good refusal in this tool already had: a short head, then the reason, then what
    to type, each its own paragraph with a blank line between. Blank lines matter as much as the
    width; the audit that prompted this module complained about their absence just as often.

    The head is wrapped too, so keep it short: a head long enough to wrap leaves the
    `senbonzakura:` prefix alone on the first line, which reads like a crash. Put interpolated paths
    in a paragraph of their own rather than in the head, for that reason.
    """
    out = list(lines(head, columns=columns))
    for para in paragraphs:
        out.append("")
        out.extend(lines(para, indent="  ", columns=columns))
    return "\n".join(out)


def refusal(head, body, *, log=print, columns=None):
    """A refusal: one shouted line, then the reason and the next step, indented and wrapped.

    The shape every good refusal in this tool already had before they were wrapped, made reusable so
    the next one does not have to reinvent the layout. `head` is expected to be short; it is printed
    as given so a caller can put a marker there.
    """
    log(head)
    say(body, indent="  ", log=log, columns=columns)
