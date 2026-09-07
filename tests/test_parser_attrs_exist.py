# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""No command may read an argument its own parser does not define.

WHY THIS FILE EXISTS

`capability` read `a.hf_token` and its parser never declared `--hf-token`, because that flag lives
in each command's own `build_parser` rather than in the shared `loader_parser` they all inherit.
The failure is an AttributeError on the first real invocation, after the arguments have parsed
cleanly, and in this case after the dataset had been located and immediately before a model would
have been loaded. On a rented card that is a download spent to reach a typo.

Nothing caught it. The unit tests exercised the functions directly with their own inputs, and the
parser tests checked that the flags declared were accepted. Neither asks the question that
matters: does the code READ anything the parser does not PROVIDE?

This is the mirror of `test_message_flags_exist.py`, which asks whether a flag a message
RECOMMENDS exists. Same defect class, opposite direction, and both were found the same way: on
first contact with a real invocation rather than by any test.
"""
import ast
import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "senbonzakura"

#: Commands whose parser is `build_parser` and whose entry point reads the parsed namespace.
COMMANDS = ["capability", "score", "margin", "coherence", "drift", "convert", "quantise"]

#: Names read off a namespace that no parser defines because something else sets them. Each is an
#: exemption with a reason, so a reviewer can tell one from an oversight.
SET_ELSEWHERE = {
    "bankai": "set by entry.split_mode before the namespace reaches the run",
    "method": "may be absent on the forward-only paths, and is read with getattr and a default",
}


def _parser_dests(module):
    mod = importlib.import_module(f"senbonzakura.{module}")
    build = getattr(mod, "build_parser", None)
    if build is None:
        pytest.skip(f"{module} builds its parser inline")
    return {a.dest for a in build()._actions}


def _attributes_read(path, names):
    """Every `<name>.<attr>` read in the file, for the local names a parsed namespace is bound to.

    Attribute ACCESS only. A `getattr(a, "x", default)` is deliberately not counted, because that
    form is how the shared helpers cope with namespaces that legitimately lack a flag, and it
    cannot raise.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in names and isinstance(node.ctx, ast.Load)):
            found.add(node.attr)
    return found


@pytest.mark.parametrize("module", COMMANDS)
def test_a_command_never_reads_an_argument_its_parser_does_not_define(module):
    """THE REGRESSION THIS FILE IS NAMED FOR."""
    dests = _parser_dests(module)
    read = _attributes_read(SRC / f"{module}.py", {"a", "args"})
    missing = sorted(
        name for name in read
        if name not in dests and name not in SET_ELSEWHERE and not name.startswith("_"))
    assert not missing, (
        f"{module}.py reads {missing} off its parsed arguments and its parser defines no such "
        f"flag, so the command fails with an AttributeError on a real invocation after parsing "
        f"has already succeeded")


def test_the_check_would_have_caught_the_defect_it_was_written_for():
    """A gate only ever seen passing has not been shown to work.

    Reconstructs the shape that shipped: a module reading `a.hf_token` whose parser has no such
    dest, and requires the detector to flag it.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.py"
        p.write_text("def main(argv=None):\n"
                     "    a = build_parser().parse_args(argv)\n"
                     "    return resolve(a.eval, token=a.hf_token or None)\n", encoding="utf-8")
        read = _attributes_read(p, {"a"})
    assert "hf_token" in read
    assert "eval" in read
    assert "hf_token" not in {"eval", "model", "out"}, (
        "if hf_token is now a shared dest, rewrite this around a flag that is still absent")


def test_a_defaulted_getattr_is_not_counted():
    """`getattr(a, "x", default)` is how the shared helpers cope with namespaces that legitimately
    lack a flag, and it cannot raise. Counting it would make the check noisy enough to ignore."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.py"
        p.write_text('def f(a):\n    return getattr(a, "method", "searched")\n', encoding="utf-8")
        assert _attributes_read(p, {"a"}) == set()


def test_every_exemption_carries_a_reason():
    assert all(isinstance(v, str) and v.strip() for v in SET_ELSEWHERE.values())
