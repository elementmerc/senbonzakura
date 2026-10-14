# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The gate that would have caught `--chat-template` shipping dead.

That flag was accepted by the abliterate parser, described in its help, and named in an error
message telling the operator to supply it, while the value never reached the loader. `score` and
`compass` forwarded it correctly, so a package-wide "is this ever read?" check passed it clean.

Two halves here, and the second is the load-bearing one:

  1. the real package is clean, which is the gate;
  2. the checker demonstrably catches the shapes it claims to, which is what stops it decaying
     into a green line that means nothing.

The first version of the checker failed exactly that second test: it counted `tok.chat_template`,
the tokenizer's own field, as a read of the flag with the same name, so it passed a tree with the
real bug reinstated. A name standing in for the thing. `test_a_field_of_the_same_name_...` below
is that failure, pinned.
"""
import ast
import pathlib
import sys
import textwrap

import pytest

# Anchored to this file, not to the working directory: a suite run from anywhere else would
# otherwise audit a tree that is not there and pass on the empty result.
ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = str(ROOT / "src" / "senbonzakura")

sys.path.insert(0, str(ROOT / "tools" / "research"))
import audit_flags  # noqa: E402


def analyse_source(tmp_path, **modules):
    """Write `name=source` modules to a tree and run the audit over it.

    A name may carry a directory (`a__cli` is written as `a/cli.py` when the tree is built with
    `a/cli`), so two modules can share a leaf name in different packages. Double underscores stand in
    for the slash because a keyword argument cannot contain one.
    """
    for name, src in modules.items():
        path = tmp_path / f"{name.replace('__', '/')}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(src), encoding="utf-8")
    declared, read, _splat, _n = audit_flags.analyse(tmp_path)
    return audit_flags.findings(declared, read)


# ── the gate ─────────────────────────────────────────────────────────────────────
def test_the_package_declares_no_flag_it_fails_to_read():
    declared, read, _splat, parsed = audit_flags.analyse(PACKAGE)
    dead, _inspected, unkept = audit_flags.findings(declared, read)
    assert parsed > 0 and declared, "the audit looked at nothing, which is not a pass"
    assert not dead, f"flags declared and never read: {dead}"
    assert not unkept, f"flags declared by a module that never reads them: {unkept}"


def test_the_gate_is_actually_looking_at_the_package():
    """A guard on the guard: if the tree moves, the test above passes over an empty result."""
    declared, _read, _splat, parsed = audit_flags.analyse(PACKAGE)
    assert parsed >= 20, f"only {parsed} modules parsed; the package tree moved"
    assert len(declared) >= 50, f"only {len(declared)} flags found; add_argument parsing broke"


# ── it catches what it claims to ─────────────────────────────────────────────────
def test_a_flag_nothing_reads_is_caught(tmp_path):
    dead, _inspected, _unkept = analyse_source(tmp_path, one="""
        def build(ap):
            ap.add_argument("--never-used", default="")
    """)
    assert [d for d, _ in dead] == ["never_used"]


def test_the_chat_template_shape_is_caught(tmp_path):
    """The real bug, in miniature: a RUNNER declares two flags, keeps one and drops the other.

    The declaring module has to read something, because that is what makes it a runner rather
    than a parser module. `parser.py` declares the whole surface and reads none of it, and a
    checker that cannot tell those apart either flags the split architecture or misses the bug.
    """
    _dead, _inspected, unkept = analyse_source(
        tmp_path,
        declarer="""
            def build(ap):
                ap.add_argument("--chat-template", dest="chat_template", default="")
                ap.add_argument("--model", default="")

            def run(args):
                return load(args.model)          # --model kept, --chat-template dropped
        """,
        reader="""
            def go(args):
                return args.chat_template
        """)
    assert [(d, m) for d, m, _l, _f, _mods in unkept] == [("chat_template", "declarer.py")]


def test_a_pure_parser_module_is_not_blamed_for_flags_it_only_declares(tmp_path):
    """`parser.py` exists so `--help` costs nothing. It declares the surface and runs nothing.

    Blaming it would make the split architecture unrepresentable; exempting a module that reads
    NOTHING is safe, because a module that reads no flag cannot be the one dropping it.
    """
    dead, _inspected, unkept = analyse_source(
        tmp_path,
        parser="""
            def build_parser():
                ap = make()
                ap.add_argument("--thing", default="")
                return ap
        """,
        runner="""
            from .parser import build_parser

            def main(argv=None):
                args = build_parser().parse_args(argv)
                return args.thing
        """)
    assert not dead and not unkept


def test_a_module_that_imports_a_parser_inherits_its_promises(tmp_path):
    """Otherwise the split would silence the check: declare in a module with no execution path,
    read in the modules that matter, and the original bug sails through.
    """
    _dead, _inspected, unkept = analyse_source(
        tmp_path,
        parser="""
            def build_parser():
                ap = make()
                ap.add_argument("--thing", default="")
                ap.add_argument("--other", default="")
                return ap
        """,
        runner="""
            from .parser import build_parser

            def main(argv=None):
                args = build_parser().parse_args(argv)
                return args.other          # --thing inherited and dropped
        """,
        elsewhere="""
            def go(args):
                return args.thing
        """)
    assert [(d, m) for d, m, _l, _f, _mods in unkept] == [("thing", "runner.py")]


def test_two_modules_sharing_a_leaf_name_do_not_share_their_parsers(tmp_path):
    """The defect the auditor was keying into, in miniature.

    `a/cli.py` and `b/cli.py` are different modules that each define `build_parser`. Keyed by leaf
    name, both collapsed to `cli.py`, so `c/main.py`, which imports only `a`'s parser, inherited
    `b`'s `--beta` as well and was reported for dropping a flag it was never handed. This is the
    shape the real tree produced: one module's flags reported against an unrelated importer.
    """
    dead, _inspected, unkept = analyse_source(
        tmp_path,
        a__cli="""
            def build_parser():
                ap = make()
                ap.add_argument("--alpha", default="")
                return ap

            def run(args):
                return args.alpha
        """,
        b__cli="""
            def build_parser():
                ap = make()
                ap.add_argument("--beta", default="")
                return ap

            def run(args):
                return args.beta
        """,
        c__main="""
            from ..a.cli import build_parser

            def main(argv=None):
                args = build_parser().parse_args(argv)
                return args.alpha
        """)
    assert not dead, dead
    assert not unkept, f"c/main.py imports only a's parser, so it may not answer for b's: {unkept}"


def test_a_flag_nothing_reads_is_still_caught_when_another_module_shares_its_leaf_name(tmp_path):
    """The other half of that fix: distinguishing the two modules must not hide a genuinely dead flag."""
    dead, _inspected, _unkept = analyse_source(
        tmp_path,
        a__cli="""
            def build_parser():
                ap = make()
                ap.add_argument("--gone", default="")
                ap.add_argument("--kept", default="")
                return ap

            def run(args):
                return args.kept
        """,
        b__cli="""
            def run(args):
                return args.other
        """)
    assert [d for d, _ in dead] == ["gone"]


def test_an_import_from_another_distribution_inherits_nothing_from_a_same_named_module(tmp_path):
    """The real defect, reduced. `check.py` imports `senbonzakura_check.cli`, a module in another
    distribution that this tree does not audit, and the audit matched it by its leaf name `cli.py` to
    the first-party `cli.py` and inherited that parser's flags. It is answerable for none of them.
    """
    _dead, _inspected, unkept = analyse_source(
        tmp_path,
        cli="""
            def build_parser():
                ap = make()
                ap.add_argument("--root", default="")
                ap.add_argument("--mode", default="")
                return ap

            def run(args):
                return args.root, args.mode
        """,
        check="""
            from senbonzakura_check.cli import build_parser

            def main(argv=None):
                args = build_parser().parse_args(argv)
                return args.mode
        """)
    assert not unkept, f"check.py imports another distribution's parser, not this one: {unkept}"


def test_an_absolute_import_of_this_package_inherits_exactly_as_a_relative_one_does(tmp_path):
    """Absolute imports name the package the tree is, and must resolve to the same module a
    relative import would. The real package uses both spellings.
    """
    _dead, _inspected, unkept = analyse_source(
        tmp_path,
        parser="""
            def build_parser():
                ap = make()
                ap.add_argument("--thing", default="")
                ap.add_argument("--other", default="")
                return ap
        """,
        runner=f"""
            from {tmp_path.name}.parser import build_parser

            def main(argv=None):
                args = build_parser().parse_args(argv)
                return args.other          # --thing inherited and dropped
        """,
        elsewhere="""
            def go(args):
                return args.thing
        """)
    assert [(d, m) for d, m, _l, _f, _mods in unkept] == [("thing", "runner.py")]


def test_a_field_of_the_same_name_is_not_mistaken_for_a_read(tmp_path):
    """The regression that made the first version of this checker useless.

    `tok.chat_template` is the tokenizer's own attribute. It has nothing to do with the flag and
    must not count as reading it, or the checker passes the exact tree it exists to fail.
    """
    dead, _inspected, _unkept = analyse_source(tmp_path, one="""
        def build(ap):
            ap.add_argument("--chat-template", dest="chat_template", default="")

        def unrelated(tok):
            return tok.chat_template
    """)
    assert [d for d, _ in dead] == ["chat_template"]


def test_getattr_on_something_that_is_not_the_namespace_is_not_a_read(tmp_path):
    dead, _inspected, _unkept = analyse_source(tmp_path, one="""
        def build(ap):
            ap.add_argument("--thing", default="")

        def unrelated(tok):
            return getattr(tok, "thing", None)
    """)
    assert [d for d, _ in dead] == ["thing"]


# ── it does not cry wolf ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("owner", ["args", "a", "own", "opts", "ns", "parsed", "self.args"])
def test_every_name_the_namespace_travels_under_counts_as_a_read(tmp_path, owner):
    """`own` is the one that matters: commands separate their own flags from the shared loader's
    `args`, and a checker blind to that reports all of them dead.
    """
    dead, _inspected, _unkept = analyse_source(tmp_path, one=f"""
        def build(ap):
            ap.add_argument("--thing", default="")

        def go({owner.split('.')[0]}):
            return {owner}.thing
    """)
    assert not dead, f"a read via {owner!r} was not recognised"


def test_getattr_on_the_namespace_counts_as_a_read(tmp_path):
    dead, _inspected, _unkept = analyse_source(tmp_path, one="""
        def build(ap):
            ap.add_argument("--thing", default="")

        def go(args):
            return getattr(args, "thing", None)
    """)
    assert not dead


@pytest.mark.parametrize("action", ["version", "help"])
def test_argparse_handles_some_actions_itself_and_they_are_not_defects(tmp_path, action):
    dead, _inspected, _unkept = analyse_source(tmp_path, one=f"""
        def build(ap):
            ap.add_argument("--x", action="{action}")
    """)
    assert not dead


def test_a_short_main_that_reads_its_own_flag_is_not_a_finding(tmp_path):
    """Pass B is a note, never a failure: a small main() declaring and using a flag is normal."""
    dead, inspected, unkept = analyse_source(tmp_path, one="""
        def main(argv=None):
            ap = None
            ap.add_argument("--deep", action="store_true")
            args = ap.parse_args(argv)
            return args.deep
    """)
    assert not dead and not unkept
    assert [d for d, _ in inspected] == ["deep"]


# ── dest derivation matches argparse ─────────────────────────────────────────────
@pytest.mark.parametrize(("call", "expected"), [
    ('ap.add_argument("--max-directions")', "max_directions"),
    ('ap.add_argument("-o", "--out")', "out"),
    ('ap.add_argument("model")', "model"),
    ('ap.add_argument("--thing", dest="other")', "other"),
])
def test_dest_is_derived_the_way_argparse_derives_it(call, expected):
    node = ast.parse(call).body[0].value
    assert audit_flags.dest_of(node) == expected


# ── the runner refuses to report green over nothing ──────────────────────────────
def test_an_empty_tree_fails_rather_than_passing(tmp_path, capsys):
    """A checker that passes when it looked at nothing answers a question it never asked."""
    assert audit_flags.main([str(tmp_path)]) == 1
    assert "nothing to check" in capsys.readouterr().err


def test_the_runner_exits_non_zero_on_a_finding(tmp_path, capsys):
    (tmp_path / "one.py").write_text('def b(ap):\n    ap.add_argument("--x")\n', encoding="utf-8")
    assert audit_flags.main([str(tmp_path)]) == 1
    assert "flag audit FAILED" in capsys.readouterr().err


def test_the_runner_exits_zero_on_the_real_package(capsys):
    assert audit_flags.main([PACKAGE]) == 0
    assert "all reached" in capsys.readouterr().out


def test_a_pure_declarer_keeps_its_exemption_when_it_touches_a_non_flag_attribute(tmp_path):
    """Reading `action.dest` is not reading a flag, and used to cost parser.py its exemption.

    WHAT HAPPENED, 2026-09-27

    The exemption in `analyse` says a module that reads no flag at all cannot drop one, which is
    what makes `parser.py` answerable for nothing: it declares the surface and runs nothing.

    `read` is keyed by ATTRIBUTE NAME and the walker records every attribute access, not only those
    naming a flag. So the set the exemption was computed from held names like `dest` and
    `option_strings`. When `parser.py` gained a loop over `ap._actions` to COUNT its flags, rather
    than leave a hardcoded total in the help text, the audit recorded it as reading two flags,
    dropped its exemption, and reported all seventy flags it declares as dropped by the module that
    declares them. Nothing was dropped.

    The failure mode is the expensive kind: a gate going red for a reason unrelated to the defect it
    guards, which is how a gate earns a bypass. The fix counted the exemption over declared flags
    only, which is what its own comment already claimed it did.
    """
    dead, _inspected, unkept = analyse_source(
        tmp_path,
        parser="""
        def build_parser():
            ap.add_argument("--trials", dest="trials")
            ap.add_argument("--device", dest="device")
            # Counting its own flags, touching attributes that are not flags.
            flags = [a for a in ap._actions if a.option_strings]
            return len([a for a in flags if a.dest])
        """,
        runner="""
        def run(args):
            return args.trials, args.device
        """)
    assert not dead, f"both flags are read in runner.py, so neither is dead: {dead}"
    assert not unkept, (
        "parser.py declares these and reads no FLAG, so it keeps its exemption. It touched "
        f"`option_strings` and `dest`, which are attributes and not flags: {unkept}")


def test_a_module_that_reads_one_real_flag_is_still_answerable(tmp_path):
    """The other half, so the fix above cannot have bought quiet by exempting everybody.

    A declaring module that genuinely reads a flag stays on the hook for every flag it declares,
    which is the property that caught `--chat-template` shipping dead.
    """
    _dead, _inspected, unkept = analyse_source(
        tmp_path,
        parser="""
        def build_parser():
            ap.add_argument("--trials", dest="trials")
            ap.add_argument("--device", dest="device")

        def resolve(args):
            return args.trials
        """,
        runner="""
        def run(args):
            return args.device
        """)
    dropped = {dest for dest, module, *_rest in unkept if module == "parser.py"}
    assert "device" in dropped, (
        "parser.py reads `trials`, so it is not a pure declarer and must answer for `device`, "
        f"which it declares and never reads. Got: {unkept}")


# ── a parsed namespace is recognised by what bound it, not by what it is called ──────────────────
def test_a_name_bound_to_a_parse_is_a_namespace_whatever_it_is_called(tmp_path):
    """The binding is the evidence, not the spelling. `zzq` is on no list anywhere, and the read of
    `--thing` through it is still a read. A name list here would fix this spelling and leave the next
    author's `chosen` or `opts` reported dead.
    """
    dead, _inspected, unkept = analyse_source(tmp_path, one="""
        def build(ap):
            ap.add_argument("--thing", default="")

        def run(argv):
            zzq = build_parser().parse_args(argv)
            return zzq.thing
    """)
    assert not dead and not unkept, (dead, unkept)


def test_a_name_rebound_to_something_else_stops_being_a_namespace(tmp_path):
    """A parse result that is rebound to something else is no longer the namespace, so a read of
    `--thing` through the new value is not a read of the flag.
    """
    dead, _inspected, _unkept = analyse_source(tmp_path, one="""
        def build(ap):
            ap.add_argument("--thing", default="")

        def run(argv):
            zzq = build_parser().parse_args(argv)
            zzq = load_config()
            return zzq.thing
    """)
    assert [d for d, _ in dead] == ["thing"]


def test_a_parse_result_is_not_a_namespace_in_another_function(tmp_path):
    """Scopes are lexical. `zzq` is bound from a parse in one function and is an unrelated name in
    the next, so reading `.thing` there is not a read of the flag.
    """
    dead, _inspected, _unkept = analyse_source(tmp_path, one="""
        def build(ap):
            ap.add_argument("--thing", default="")

        def first(argv):
            zzq = build_parser().parse_args(argv)
            return zzq.kept

        def second(tok):
            return zzq.thing
    """)
    assert "thing" in [d for d, _ in dead]


def test_a_parameter_shadows_an_outer_parse_result(tmp_path):
    """A parameter is a new name in its function, even when an enclosing scope bound the same name
    to a parse. Reads through it are not reads through the outer namespace.
    """
    dead, _inspected, _unkept = analyse_source(tmp_path, one="""
        def build(ap):
            ap.add_argument("--thing", default="")

        def outer(argv):
            n = build_parser().parse_args(argv)

            def inner(n):
                return n.thing
            return inner
    """)
    assert [d for d, _ in dead] == ["thing"]
