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

sys.path.insert(0, str(ROOT / "tools"))
import audit_flags  # noqa: E402


def analyse_source(tmp_path, **modules):
    """Write `name=source` modules to a tree and run the audit over it."""
    for name, src in modules.items():
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(src), encoding="utf-8")
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
