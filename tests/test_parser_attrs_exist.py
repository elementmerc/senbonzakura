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
    lack a flag, and it cannot raise. Counting it would make the check noisy enough to ignore.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.py"
        p.write_text('def f(a):\n    return getattr(a, "method", "searched")\n', encoding="utf-8")
        assert _attributes_read(p, {"a"}) == set()


def test_every_exemption_carries_a_reason():
    assert all(isinstance(v, str) and v.strip() for v in SET_ELSEWHERE.values())


# ── the abliterator, which the check above never looked at ─────────────────────────
#
# ADDED 2026-09-07 after `_capability_score` read `self.args.batch_size` from the day it was
# written. The flag is `--gen-batch`, dest `gen_batch`; `batch_size` has never existed. So
# `--capability-eval` raised AttributeError the moment it was asked to score anything, and the
# in-search capability gate, the one thing in this tool that PREVENTS damage rather than measuring
# it, had never run once.
#
# The check above did not see it because it walks the sub-command modules and binds only the local
# names `a` and `args`. The abliterator reads its flags as `self.args.X` from inside a class in
# `cli.py`, which nothing was looking at.

#: Read off `self.args` in cli.py without a parser flag behind them, each with the reason. An
#: entry here is a decision a reviewer can weigh; its absence is how a typo ships.
CLI_SET_ELSEWHERE = {
    "bankai": "set by entry.split_mode before the namespace reaches the run",
}


def _self_args_reads(path):
    """Every `self.args.<attr>` read in a file. Attribute access only, same rule as above."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)):
            continue
        inner = node.value
        if (isinstance(inner, ast.Attribute) and inner.attr == "args"
                and isinstance(inner.value, ast.Name) and inner.value.id == "self"):
            found.add(node.attr)
    return found


def test_the_abliterator_never_reads_a_flag_its_parser_does_not_define():
    """THE ONE THAT WOULD HAVE CAUGHT IT.

    An AttributeError here does not fail at parse time. It fails after the model is loaded, the
    directions are extracted and the baseline is measured, which on a rented card is an hour spent
    to reach a typo, and in the case that prompted this it was a feature nobody had ever run.
    """
    from senbonzakura.parser import build_parser

    dests = {a.dest for a in build_parser()._actions}
    read = _self_args_reads(SRC / "cli.py")
    missing = sorted(n for n in read
                     if n not in dests and n not in CLI_SET_ELSEWHERE and not n.startswith("_"))
    assert not missing, (
        f"cli.py reads {missing} off self.args and the abliterate parser defines no such flag. "
        f"That is an AttributeError after the model is loaded, not at parse time.")


def test_the_new_check_would_have_caught_the_defect_it_was_written_for():
    """A gate only ever seen passing has not been shown to work."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.py"
        p.write_text("class A:\n"
                     "    def f(self):\n"
                     "        return g(batch=int(self.args.batch_size), n=self.args.gen_batch)\n",
                     encoding="utf-8")
        read = _self_args_reads(p)
    assert read == {"batch_size", "gen_batch"}

    from senbonzakura.parser import build_parser
    dests = {a.dest for a in build_parser()._actions}
    assert "gen_batch" in dests, "the flag that exists"
    assert "batch_size" not in dests, "the flag that never did, and shipped for weeks"


def test_a_plain_args_read_is_not_mistaken_for_a_self_args_read():
    """`args.x` inside a module-level function is a different binding and is covered elsewhere;
    counting it here would make this check duplicate the other one and drift from it.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.py"
        p.write_text("def f(args):\n    return args.nope\n", encoding="utf-8")
        assert _self_args_reads(p) == set()
