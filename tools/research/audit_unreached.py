#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Find public names in the package that nothing in the package reaches.

THE SIBLING OF `audit_flags.py`, and the same defect from the other end. That one finds a flag the
parser promises and nothing keeps. This one finds a function that exists, passes its own tests,
and is reachable from no command line at all.

`tool_call_validity_block` had that shape on the day it was written: four measures of whether a
model's tool calls were well formed, each with its own denominator, each tested, and no caller
anywhere. The same shape has caught this project three times before. `--capability-eval` read a
flag that did not exist, so the in-search capability gate, the only thing in the tool that
PREVENTS damage rather than measuring it, had never once executed. A mutation pass found five
fixes wired to call sites nobody checked, and one where deleting the compass print broke no test
at all.

Coverage does not see any of it. A measure with a thorough unit test and no caller is covered and
unreachable at the same time, and the suite reports the first fact.

WHAT COUNTS AS REACHING A NAME

A Name or Attribute load anywhere in `src/` or `tools/`, other than the `def` or `class` statement
that introduces it.

Deliberately NOT a string constant. Every one of these names appears in some docstring explaining
what it is for, and a docstring is not a caller. Counting strings made the first version of this
script report `refuse_if_untrustworthy` as reached on the strength of a comment that says, in so
many words, that nothing calls it yet.

In the test tree, string constants DO count, because `monkeypatch.setattr(mod, "name", ...)`
reaches a name by spelling it.

WHAT THIS CANNOT TELL YOU, and it is the whole reason the output is a list to read rather than a
gate that fails

It cannot tell a dead knob from a piece built ahead of its caller on purpose. Both look identical
from here. `chattemplate.refuse_if_untrustworthy` is written, tested and deliberately not wired,
with the reason recorded beside the function that reports instead of refusing, and that is a
decision rather than an oversight. So every name this prints needs a person to say which it is,
and the useful artefact is the per-name answer, not the count.

VALIDATED BEFORE IT WAS BELIEVED, because an absence is not evidence until the instrument has been
shown to find a known case. Run against the commit before the tool-call wiring landed it reports
`tool_call_validity_block`; run against the commit after, it does not.

    python3 tools/research/audit_unreached.py .
"""
import ast
import pathlib
import sys

#: A name introduced by a `def` or `class` at the top level of its module. Nested definitions are
#: skipped: a closure is reached by the function that holds it, which is a different question.
TOP_LEVEL = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def public_definitions(src):
    """Every top-level public name in the package, as {name: ["file.py:line", ...]}."""
    out = {}
    for path in sorted(src.glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, TOP_LEVEL) and not node.name.startswith("_"):
                out.setdefault(node.name, []).append(f"{path.name}:{node.lineno}")
    return out


def references(paths, *, strings):
    """Every name referenced in `paths`, as {name: {file.py, ...}}.

    A definition's own `def` line is not a reference to itself, which is checked by line number
    rather than by name, so a module that both defines and calls something is reported correctly.
    """
    seen = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as e:
            # Loud, because a file this cannot read is a file it cannot vouch for, and a silent
            # skip would turn an unparseable module into a clean bill of health for every name in
            # it.
            print(f"  WARNING: {path} could not be read ({e}), so nothing in it was counted as a "
                  f"reference. Every finding below may be a false positive for that reason.",
                  file=sys.stderr)
            continue
        introduced = {(n.name, n.lineno) for n in ast.walk(tree) if isinstance(n, TOP_LEVEL)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif strings and isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = node.value
            else:
                continue
            if (name, getattr(node, "lineno", -1)) in introduced:
                continue
            seen.setdefault(name, set()).add(path.name)
    return seen


def unreached(root):
    """(name, where defined, how many test files touch it) for every unreached public name."""
    root = pathlib.Path(root)
    src = root / "src" / "senbonzakura"
    if not src.is_dir():
        raise SystemExit(f"{root} does not look like the package: no src/senbonzakura/ under it.")
    defined = public_definitions(src)
    if not defined:
        raise SystemExit(
            f"{src} yielded no public definitions, which cannot be right and means this read the "
            f"wrong tree rather than that the package is empty.")
    product = references(sorted(src.glob("*.py")) + sorted((root / "tools").rglob("*.py")),
                         strings=False)
    tested = references(sorted((root / "tests").rglob("*.py")), strings=True)
    rows = [(name, where, len(tested.get(name, ())))
            for name, where in sorted(defined.items()) if name not in product]
    return defined, rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = argv[0] if argv else "."
    defined, rows = unreached(root)
    print(f"{len(defined)} public names in the package; {len(rows)} referenced nowhere in src/ "
          f"or tools/\n")
    for name, where, n in rows:
        # A name with tests is the worse of the two, not the better: somebody wrote a measure,
        # proved it works, and it reaches no user. A name with neither was probably never
        # finished, which is easier to see.
        state = "TESTED-BUT-UNREACHABLE" if n else "UNREACHED-AND-UNTESTED"
        print(f"{state}  {name}  ({', '.join(where)})  test files: {n}")
    if rows:
        print("\nEach one needs a person to say which it is: a dead knob, or a piece built ahead "
              "of its caller on purpose. This script cannot tell, and a reason invented here "
              "would be worse than the finding.")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
