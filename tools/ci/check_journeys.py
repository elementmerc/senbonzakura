#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Count the journey suite, measure what it covers, and hold the ratchet.

WHY THIS EXISTS

On 2026-09-28 three defects were found by driving the real CLI through a pty, and none of them
was visible to 6,500 unit tests, because each lived in a SEQUENCE rather than in a function: a
pre-flight board that said "all clear" about a run the tool then refused, a guided walk that
produced a command its own pre-flight rejects, and a menu answer that was a valid index until the
menu grew. Decision Q-43 answered that with a suite of scripted sessions, plus a ratchet, because
a suite with no floor shrinks the first week somebody is in a hurry.

WHAT A JOURNEY IS

A pytest test function in `tests/journeys/test_*.py` carrying a `journey` mark whose positional
arguments name the surfaces it walks:

    @pytest.mark.journey("command:doctor", "state:no-tty")
    def test_asking_what_the_tool_is_answers_rather_than_refusing(session):
        ...

A surface string is `command:<name>` or `state:<name>` and nothing else.

WHY THIS PARSES RATHER THAN IMPORTS

Importing the journey modules would import the package, which imports torch, which is minutes of
wall clock and gigabytes of wheel on a machine whose only job here is to count. So everything
below reads source with `ast`. That also means this gate runs on a box that cannot run the suite,
which is the case on at least one machine in this project already.

WHAT THE RATCHET RATCHETS, AND WHAT IT DELIBERATELY DOES NOT

Two high-water marks, each of which may only rise: the number of journeys, and the number of
DISTINCT surfaces they cover. It does NOT ratchet the coverage PERCENTAGE. Registering a new
command would then fail the gate the moment it landed, before anybody could have written its
journey, and a gate that fires on work nobody could have done yet is one people learn to bypass.
The percentage is printed, with the uncovered names next to it, because a percentage nobody can
act on is decoration.

    python tools/ci/check_journeys.py             # the gate: counts may not have fallen
    python tools/ci/check_journeys.py --update    # record today's numbers as the new floor
    python tools/ci/check_journeys.py --update --force   # ... even if that lowers one
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The hand-kept half of the universe. Nothing in the code enumerates "the user pressed Ctrl+C at
#: the confirm", so this list cannot be derived the way the command half is, and it is EXPECTED TO
#: GROW: every time a journey walks a situation that is not named here, the name is added here
#: first. Keeping it in one typed list is the cost of covering the sequences that have no symbol.
STATE_SURFACES = (
    "no-tty",
    # Distinct from `no-tty`, and the distinction is the whole reason the state exists: this is a
    # pseudo-terminal with nobody behind it, which `isatty` reports as a terminal. `docker run -t`,
    # `expect` and some CI agents all produce it, and it is the one case the inference cannot see.
    "fake-tty",
    "narrow-terminal",
    "cancelled-at-confirm",
    "interrupted-mid-run",
    "answered-wrong-then-right",
    "resumed",
    "nothing-cached",
    "failing-preflight",
)

#: Surfaces with no entry in any command table: the bare invocation, and the short flag that opens
#: the guided mode. Both are things a person types, so both are things a journey can walk.
EXTRA_COMMAND_SURFACES = ("senbonzakura", "-i")

SURFACE_RE = re.compile(r"^(?:command|state):[A-Za-z0-9._-]+$")


class JourneyError(Exception):
    """A journey, a command table or a ratchet file that this gate refuses to read past."""


# ── reading the journeys ─────────────────────────────────────────────────────────────────────

def _journey_mark_args(decorator):
    """The mark's argument nodes, or None when this decorator is not a `journey` mark.

    Returns an empty list for a bare `@pytest.mark.journey` with no call, which is a declaration
    of nothing and is refused by the caller rather than silently counted as covering nothing.
    """
    node = decorator.func if isinstance(decorator, ast.Call) else decorator
    if not isinstance(node, ast.Attribute) or node.attr != "journey":
        return None
    owner = node.value
    if not isinstance(owner, ast.Attribute) or owner.attr != "mark":
        return None
    return list(decorator.args) if isinstance(decorator, ast.Call) else []


def journeys_in_file(path):
    """`[(function_name, [surface, ...]), ...]` for one journey module.

    Raises JourneyError naming the file and the function for a test with no mark, a mark with no
    surfaces, a non-literal argument, or a surface string that is not `command:x` or `state:x`.
    An undeclared journey is the failure this check exists for: it silently lowers the measured
    coverage while looking on the page exactly like work.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as e:
        raise JourneyError(f"{path}: cannot be parsed: {e}") from e

    found = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue

        marks = [args for args in (_journey_mark_args(d) for d in node.decorator_list)
                 if args is not None]
        if not marks:
            raise JourneyError(
                f"{path}::{node.name} is a journey with no `journey` mark.\n"
                f"  Every journey declares the surfaces it walks, or it lowers the measured\n"
                f"  coverage while looking like work. Add, for example:\n"
                f'    @pytest.mark.journey("command:doctor", "state:no-tty")')

        surfaces = []
        for args in marks:
            if not args:
                raise JourneyError(
                    f"{path}::{node.name} carries a `journey` mark that names no surfaces.\n"
                    f'  Pass one or more, as in @pytest.mark.journey("command:doctor").')
            for arg in args:
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    raise JourneyError(
                        f"{path}::{node.name} declares a surface this gate cannot read at line "
                        f"{getattr(arg, 'lineno', node.lineno)}.\n"
                        f"  Surfaces are written as plain string literals, because this gate "
                        f"reads the file rather than importing it.")
                if not SURFACE_RE.match(arg.value):
                    raise JourneyError(
                        f"{path}::{node.name} declares the surface {arg.value!r}, which is "
                        f"neither `command:<name>` nor `state:<name>`.\n"
                        f"  Those two shapes are the whole vocabulary; anything else covers "
                        f"nothing and is counted as nothing.")
                surfaces.append(arg.value)
        found.append((node.name, surfaces))
    return found


def read_journeys(directory):
    """`[(path, function, [surface, ...]), ...]` over `test_*.py`, sorted for a stable report."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("test_*.py")):
        for name, surfaces in journeys_in_file(path):
            out.append((path, name, surfaces))
    return out


# ── building the universe ────────────────────────────────────────────────────────────────────

def _literal_table(path, name):
    """The module-level literal assigned to `name`, read with `ast`.

    RETIRED is deliberately not among the callers. A retired word is kept so the tool can say
    what a command was renamed to; it is not a surface anybody should be writing a journey for.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as e:
        raise JourneyError(f"{path}: cannot be parsed: {e}") from e

    for node in tree.body:
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else node.targets if isinstance(node, ast.Assign) else [])
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                return node.value
    raise JourneyError(
        f"{path}: no module-level `{name}`. The command half of the surface universe is DERIVED "
        f"from that table, so this gate has stopped measuring rather than quietly reporting a "
        f"smaller universe and a higher percentage.")


def _string_keys(node, path, name):
    if isinstance(node, ast.Dict):
        items = node.keys
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = node.elts
    else:
        raise JourneyError(f"{path}: `{name}` is not a literal dict, tuple, list or set, so this "
                           f"gate cannot read it without importing the package.")
    words = []
    for item in items:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            raise JourneyError(f"{path}: `{name}` holds an entry that is not a string literal.")
        words.append(item.value)
    return words


def command_surfaces(src):
    """Every word a person can type as the first argument, derived from the real tables.

    Derived rather than typed, so that registering a command in `entry.DELEGATED` puts it in this
    universe without anybody editing this file. A hand-typed inventory rots, and an inventory that
    has rotted reports a percentage that is too high, which is the one failure mode that makes a
    coverage gate worse than no gate.
    """
    src = Path(src)
    entry = src / "senbonzakura" / "entry.py"
    parser = src / "senbonzakura" / "parser.py"
    words = set()
    for name in ("DELEGATED", "ALIASES"):
        words.update(_string_keys(_literal_table(entry, name), entry, name))
    words.update(_string_keys(_literal_table(parser, "MODES"), parser, "MODES"))
    words.update(EXTRA_COMMAND_SURFACES)
    return {f"command:{w}" for w in words}


def universe(src):
    return command_surfaces(src) | {f"state:{s}" for s in STATE_SURFACES}


# ── the ratchet ──────────────────────────────────────────────────────────────────────────────

def read_ratchet(path):
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise JourneyError(
            f"{path} does not exist, so there is no floor to hold. Create it with "
            f"`python tools/ci/check_journeys.py --update`.") from e
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise JourneyError(f"{path}: cannot be read: {e}") from e
    if not isinstance(doc, dict):
        raise JourneyError(f"{path}: expected an object with `journeys` and `surfaces`.")
    counts = {}
    for key in ("journeys", "surfaces"):
        value = doc.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise JourneyError(f"{path}: `{key}` must be a non-negative whole number, not "
                               f"{value!r}. A floor nobody can read is not a floor.")
        counts[key] = value
    return counts


def write_ratchet(path, counts, today):
    """Rewrite the ratchet atomically, so an interrupted update cannot leave half a floor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "journeys": counts["journeys"],
        "surfaces": counts["surfaces"],
        "updated": today,
        "why": "High-water marks for the journey suite (decision Q-43). Each may only rise. "
               "Raised by `python tools/ci/check_journeys.py --update`; lowering one needs "
               "--force and a reason in the commit message.",
    }
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ── the report ───────────────────────────────────────────────────────────────────────────────

def _coverage_line(covered, whole):
    if not whole:
        return "surfaces: 0 of 0"
    return (f"surfaces: {len(covered)} of {len(whole)} "
            f"({round(100 * len(covered) / len(whole))}%)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="refuse if either high-water mark has fallen (the default)")
    ap.add_argument("--update", action="store_true",
                    help="record today's numbers as the new floor")
    ap.add_argument("--force", action="store_true",
                    help="with --update, allow a number to go down")
    ap.add_argument("--journeys", default=str(ROOT / "tests" / "journeys"),
                    help="directory holding test_*.py journey modules")
    ap.add_argument("--ratchet", default=str(ROOT / "journeys" / "ratchet.json"),
                    help="the ratchet file to read and write")
    ap.add_argument("--src", default=str(ROOT / "src"),
                    help="source root the command universe is derived from")
    ap.add_argument("--today", default=None, help="ISO date to stamp an update with")
    a = ap.parse_args(argv)

    if a.update and a.check:
        print("check_journeys: pass --check or --update, not both.", file=sys.stderr)
        return 2

    try:
        whole = universe(a.src)
        found = read_journeys(a.journeys)
    except JourneyError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1

    declared = {s for _, _, surfaces in found for s in surfaces}
    stray = sorted(declared - whole)
    if stray:
        print("REFUSED: these journeys declare surfaces that are in no universe:", file=sys.stderr)
        for surface in stray:
            where = sorted({f"{p}::{n}" for p, n, ss in found if surface in ss})
            print(f"  {surface}  ({', '.join(where)})", file=sys.stderr)
        print("  A command surface is derived from the real command tables and a state surface "
              "comes from\n  STATE_SURFACES in this file. A misspelt one covers nothing while "
              "counting as something.", file=sys.stderr)
        return 1

    counts = {"journeys": len(found), "surfaces": len(declared)}
    print(f"journeys: {counts['journeys']} in {a.journeys}")
    print(_coverage_line(declared, whole))
    uncovered = sorted(whole - declared)
    if uncovered:
        print("uncovered: " + ", ".join(uncovered))
    else:
        print("uncovered: none")

    if a.update:
        try:
            before = read_ratchet(a.ratchet)
        except JourneyError:
            before = {"journeys": 0, "surfaces": 0}
        fell = {k: before[k] - counts[k] for k in counts if counts[k] < before[k]}
        if fell and not a.force:
            print()
            for key, drop in sorted(fell.items()):
                print(f"REFUSED: {key} would go from {before[key]} to {counts[key]} "
                      f"(down {drop}).", file=sys.stderr)
            print("A ratchet that can be quietly wound back is not a ratchet. Pass --force and "
                  "say why in the commit message.", file=sys.stderr)
            return 1
        write_ratchet(a.ratchet, counts,
                      a.today or dt.datetime.now(dt.timezone.utc).date().isoformat())
        print()
        print(f"ratchet updated: journeys {before['journeys']} -> {counts['journeys']}, "
              f"surfaces {before['surfaces']} -> {counts['surfaces']}")
        return 0

    try:
        floor = read_ratchet(a.ratchet)
    except JourneyError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1

    fell = {k: floor[k] - counts[k] for k in counts if counts[k] < floor[k]}
    if fell:
        print()
        for key, drop in sorted(fell.items()):
            print(f"REFUSED: {key} fell from {floor[key]} to {counts[key]} (down {drop}).",
                  file=sys.stderr)
        print("Restore the journeys, or lower the floor deliberately with "
              "`--update --force` and a reason.", file=sys.stderr)
        return 1

    print(f"floor held: journeys {counts['journeys']} >= {floor['journeys']}, "
          f"surfaces {counts['surfaces']} >= {floor['surfaces']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
