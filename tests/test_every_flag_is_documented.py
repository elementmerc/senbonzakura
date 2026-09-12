# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""No flag ships without help text.

THE ASSUMPTION THIS ENFORCES: neither a human nor an AI is going to read the source. A flag whose
only description is its own name is discoverable by grepping the code and nowhere else, which in
practice means undiscoverable.

Thirty-three flags were in that state when this was written. The worst of them was `--gen-tokens`,
the generation budget: the single setting most able to make a refusal rate read low, declared as
`add_argument("--gen-tokens", type=int, default=48)` and nothing more, while a whole module
(`lengthsweep.py`) existed to explain why 48 is dangerous. Eleven more were on `validate`,
including its primary input `--track` and `--experiment`, whose choices are the bare tokens
e1, e2, e3, e4, transfer and reach.

This walks the AST rather than the parsers because several parsers import torch, and the point is
to check every declaration in the package, including ones behind a subcommand a test would have to
construct.
"""
import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "senbonzakura"

#: `--version` is argparse's own action: it prints the version and exits, and the flag name is the
#: entire description. Anything else earns a sentence.
EXEMPT = {"--version"}


def _declarations():
    """(file, line, flags) for every add_argument call in the package."""
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            names = tuple(a.value for a in node.args
                          if isinstance(a, ast.Constant) and isinstance(a.value, str))
            if not names:
                continue
            has_help = "help" in {kw.arg for kw in node.keywords}
            yield path.name, node.lineno, names, has_help


UNDOCUMENTED = [(f, line, names) for f, line, names, has_help in _declarations()
                if not has_help and not set(names) & (EXEMPT | {"-h", "--help"})]


def test_no_flag_is_declared_without_help_text():
    listing = "\n".join(f"  {f}:{line}  {' '.join(names)}" for f, line, names in UNDOCUMENTED)
    assert not UNDOCUMENTED, (
        f"{len(UNDOCUMENTED)} flag(s) ship with no help text, so `--help` lists a name and says "
        f"nothing about it:\n{listing}")


def test_the_scan_actually_finds_declarations():
    """A guard against the gate above passing because it walked nothing.

    An AST scan that matches no nodes reports a clean tree, which is indistinguishable from a
    clean tree that was really checked. This project has shipped that exact failure (a prompt
    gate whose depth cap returned an empty set) more than once.
    """
    found = list(_declarations())
    assert len(found) > 200, f"the scan found only {len(found)} declarations; it is not walking"
    assert any(has_help for *_rest, has_help in found), "no declaration had help at all"
