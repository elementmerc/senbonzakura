# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every number a parser takes is checked where it is read, so no command can forget to ask.

WHAT PROMPTED IT, 2026-09-28

`argresolve.whole_number` was written because `score --skip -5` passed every check it met and came
back having measured the LAST five prompts, stamped with a partition boundary that does not exist.
It was applied to `score` and `capability` and to nothing else, and two reviewers found what that
left behind:

  - `--min-directions 0` and `--min-directions -4` parsed, and `Abliterator.__init__` did
    `min(max(1, args.min_directions), self.KMAX)`, so the operator's 0 became 1 with no warning and
    `abliteration.json` honestly recorded 1. Setting `--min-directions` and `--max-directions` equal
    is how this project pins K, so a pinned-K experiment could silently run at a different K, and
    this project has already withdrawn one result for exactly that.
  - `--capability-n -5` parsed, and the probe reads `n = int(... or 0)` and returns nothing on
    `n <= 0`, so a mistyped minus silently switched off the only instrument that can see reasoning
    loss and the run completed reporting no capability figure.
  - `--capability-max-new 0` parsed and was priced at about zero seconds by the slow-probe refusal,
    so generation produced nothing, every item graded indeterminate, and hours of search were spent
    on a probe the command line had already made impossible.

The same commands refused all three when typed at `senbonzakura capability`, which is the worse half
of the finding: two sibling commands disagreed about whether a negative sample size is a thing, and
the one that disagreed silently is the one people script.

WHY THIS IS A SWEEP AND NOT THREE TESTS

Three tests would have closed the three flags the panel happened to look at. The defect is the
class, and the class has eighteen members in one parser. So this walks every numeric action in every
parser in the package and fails on one that takes a bare `int`, which means the next flag somebody
adds inherits the rule instead of having to remember it.
"""
from __future__ import annotations

import argparse

import pytest

from senbonzakura import argresolve, parser

#: Numeric flags that are NOT counts or offsets, each with the reason a floor of zero would be
#: wrong. These are decisions rather than debt: a fraction, a signed index and a strength all have
#: legitimate values that `whole_number` would refuse.
EXEMPT: dict[str, str] = {
    "--kl-scale": "a weight, and a fraction",
    "--sparsity": "a fraction of the layer",
    "--gpu-min-free-frac": "a fraction of the card",
    "--max-kl": "a divergence ceiling, and a float",
    "--max-pause": "seconds, and a float",
    "--layer-lo": "a layer index, and negative indexes from the end",
    "--layer-hi": "a layer index, and negative indexes from the end",
    "--inspect": "takes LAYER and STRENGTH, and a strength may be negative",
    # MOVED OUT OF `NOT_YET_BOUNDED` ON 2026-10-02, and the move is the point: it was recorded as
    # debt and is a decision. `quantise.run` refuses a negative worker count AND one past its own
    # ceiling, in its own words, naming both bounds and what zero means. A floor at parse time
    # would reach the negative half first with a sentence about slicing from the end of a prompt
    # set, which is the wrong explanation for a thread count, and would take the ceiling half's
    # two tests with it. The flag is bounded; it is bounded there.
    "--threads": "quantise refuses both sides itself, naming the ceiling as well as the floor",
    # SAME REASONING, SAME DAY. `fetch._preflight_expectations` refuses a negative byte count and
    # says the refusal cost nothing, which is the sentence that matters to somebody who mistyped a
    # size on a multi-gigabyte download. A parse-time floor would reach it first with a worse
    # explanation and would take that test's assertion on the message with it.
    "--expect-size": "fetch refuses it itself, and says the refusal arrived before the download",
}

#: Numeric flags that ARE counts or offsets and are still unbounded. This is debt, stated as debt,
#: and it is deliberately a separate list from `EXEMPT` so the two cannot be confused: an exemption
#: says the rule does not apply, and a name here says the rule applies and has not been enforced yet.
#:
#: The sweep that produced this list ran on 2026-09-28 and found the class is twenty six flags wide.
#: The abliterate parser and `compass` were fixed the same night, because those carried demonstrated
#: harm: a pinned-K run that silently ran at a different K, a mistyped minus that switched the
#: capability probe off, and `--bootstrap -5`, which reaches an IndexError AFTER both arms' forward
#: passes and before the artefact is written, so a GPU run pays in full and writes nothing.
#:
#: The rest need a per-flag decision about whether zero means something, and getting that wrong
#: refuses a working invocation, which this project's own reasoning calls the worse failure because
#: it teaches people to reach for whatever turns the checking off. Recorded in DEFERRED.md.
#: WORKED THROUGH ON 2026-10-02, and the per-flag decisions the note above asks for are recorded
#: at each declaration site. Ten of the eleven names that were here are now bounded with a floor
#: chosen per flag: zero is kept where it is documented to mean something (`--capability-n` off,
#: `--eval-refusal-final` off, `--gpu-layers` none, `--fit`/`--search` an empty partition) and a
#: floor of one is set where zero makes a rate a division or a batch a `range(0, n, 0)`. The
#: eleventh, `--threads`, moved to `EXEMPT` because the command already refuses it better.
#:
#: Measured before the change, on this tree: `measure --n -5`, `measure --capability-n -5`,
#: `track --fit -5 --search -3`, `head-to-head run --trials -3 --batch -1 --max-new -2
#: --skip-harmful -5`, `imatrix --chunks -4`, `drift --batch -1`, `fetch --expect-size -1`,
#: `capability --seed -1` and five count flags on `validate` were all accepted.
NOT_YET_BOUNDED: dict[str, str] = {
    "--min-applied": "check, where zero may be a legitimate absence of a floor",
}


def _numeric_actions(ap):
    """Every action whose `type` converts a string to a number, with its flag and its converter."""
    for action in ap._actions:
        if not action.option_strings or action.type is None:
            continue
        yield action.option_strings[0], action


def _every_parser():
    """The full abliterate parser, plus every delegated command's own."""
    import importlib

    from senbonzakura.entry import DELEGATED

    yield "abliterate", parser.build_parser(full=True)
    for name, (module_name, *_rest) in sorted(DELEGATED.items()):
        module = importlib.import_module(f"senbonzakura.{module_name}")
        build = getattr(module, "build_parser", None)
        if build is None:
            continue
        try:
            yield name, build()
        except TypeError:
            continue


def test_there_are_parsers_and_numbers_to_check():
    """Without this the sweep below passes on an empty list, which is how a guard reports clean."""
    parsers = dict(_every_parser())
    assert len(parsers) > 5, f"only found {sorted(parsers)}"
    total = sum(len(list(_numeric_actions(ap))) for ap in parsers.values())
    assert total > 30, f"only {total} numeric flags found across {len(parsers)} parsers"


def test_no_numeric_flag_takes_a_bare_int():
    offenders = []
    for command, ap in _every_parser():
        for flag, action in _numeric_actions(ap):
            if action.type in (int, float) and flag not in EXEMPT | NOT_YET_BOUNDED:
                offenders.append(f"{command} {flag}")
    assert not offenders, (
        "these take a bare number, so a negative count or a zero batch is accepted at the command "
        f"line and absorbed somewhere downstream: {sorted(offenders)}\n"
        "  Use argresolve.whole_number('<flag>', minimum=N), which refuses at parse time and names "
        "the flag. If zero or a negative is genuinely meaningful, record it in EXEMPT with the "
        "reason rather than leaving the flag unbounded.")


def test_every_exemption_names_a_flag_that_exists():
    """An exemption for a flag nobody has is an exemption covering the next flag of that name."""
    known = {flag for _c, ap in _every_parser() for flag, _a in _numeric_actions(ap)}
    for table, name in ((EXEMPT, "EXEMPT"), (NOT_YET_BOUNDED, "NOT_YET_BOUNDED")):
        unknown = sorted(set(table) - known)
        assert not unknown, f"{name} names flags no parser declares: {unknown}"


def test_the_debt_list_only_shrinks():
    """A ratchet. Adding a name here is how the class quietly reopens, so the count is pinned.

    Lower it when a flag is bounded. Raising it means a new unbounded count was added, which is the
    thing the sweep exists to stop, and the fix is the bound rather than the number.
    """
    assert len(NOT_YET_BOUNDED) <= 1, (
        f"NOT_YET_BOUNDED has grown to {len(NOT_YET_BOUNDED)}: a new numeric flag was added without "
        f"a floor. Use argresolve.whole_number rather than recording it here.")


#: (flag, a value that must be refused). The three the panel found, plus the shapes around them.
REFUSED = [
    ("--min-directions", "0"), ("--min-directions", "-4"),
    ("--max-directions", "0"),
    ("--capability-n", "-5"),
    ("--capability-max-new", "0"),
    ("--gen-batch", "0"), ("--gen-batch", "-1"),
    ("--inspect-n", "-1"),
    ("--trials", "0"),
    ("--dir-prompts", "0"),
    ("--seed", "-1"),
]


@pytest.mark.parametrize(("flag", "value"), REFUSED)
def test_the_parser_refuses_a_value_that_means_nothing(flag, value, capsys):
    ap = parser.build_parser(full=True)
    with pytest.raises(SystemExit):
        ap.parse_args([flag, value, "Qwen/Qwen3-0.6B"])
    said = capsys.readouterr().err
    assert flag in said, (
        f"the refusal for {flag} {value} does not name the flag, so a reader has to work out which "
        f"of forty arguments it is about: {said!r}")


#: (flag, a value that must still be accepted). Zero is documented as meaning something for each of
#: these, so a floor of one here would refuse a working invocation, which is the worse failure: it
#: teaches people to reach for whatever turns the checking off.
ACCEPTED = [
    ("--capability-n", "0"),          # "0 = off"
    ("--patience", "0"),              # "0 = run all --trials"
    ("--eval-refusal-final", "0"),    # "0 = off"
    ("--ablation-rounds", "0"),       # the default
    ("--seed", "0"),
    ("--inspect-n", "0"),
]


@pytest.mark.parametrize(("flag", "value"), ACCEPTED)
def test_the_parser_still_accepts_a_zero_that_means_something(flag, value):
    parser.build_parser(full=True).parse_args([flag, value, "Qwen/Qwen3-0.6B"])


def test_the_help_for_every_flag_that_accepts_zero_says_so():
    """If zero is allowed and undocumented, nobody knows the probe can be turned off.

    `--seed 0` and `--inspect-n 0` are not in here: a seed of zero needs no explaining and printing
    nothing is the obvious reading of zero prompts.
    """
    ap = parser.build_parser(full=True)
    helps = {flag: (action.help or "") for flag, action in _numeric_actions(ap)}
    for flag, _value in ACCEPTED:
        if flag in ("--seed", "--inspect-n", "--ablation-rounds"):
            continue
        assert "0 = " in helps[flag] or "zero" in helps[flag].lower(), (
            f"{flag} accepts 0 and its help does not say what 0 does: {helps[flag]!r}")


def test_the_validator_refuses_below_its_floor_and_names_the_flag():
    """The shared piece, driven directly, so a change to it fails here rather than in eighteen
    places at once.
    """
    parse = argresolve.whole_number("--example", minimum=1)
    assert parse("3") == 3
    for bad in ("0", "-1", "one", "1.5", ""):
        with pytest.raises(argparse.ArgumentTypeError) as e:
            parse(bad)
        assert "--example" in str(e.value), f"the message for {bad!r} does not name the flag"


# ── the sibling commands, worked through on 2026-10-02 ───────────────────────────────
#
# The sweep above is syntactic: it proves no parser declares a bare `int`. These drive the real
# parsers, because "declares a bounded type" and "refuses the value a user would type" are two
# different claims and only the second one is the property. The one that was fixed before this
# existed, `--min-directions`, was fixed on a parser whose siblings then went on accepting it for
# four days, so the sibling commands get their own table rather than being covered by implication.

#: (command, argv that must be refused, the flag the refusal has to name).
SIBLING_REFUSED = [
    ("measure", ["m", "--out", "o", "--n", "-5"], "--n"),
    ("measure", ["m", "--out", "o", "--n", "0"], "--n"),
    ("measure", ["m", "--out", "o", "--capability-n", "-5"], "--capability-n"),
    ("track", ["--harmful", "h", "--harmless", "g", "--out", "o", "--fit", "-5"], "--fit"),
    ("track", ["--harmful", "h", "--harmless", "g", "--out", "o", "--search", "-3"], "--search"),
    ("imatrix", ["m", "--corpus", "neutral", "--chunks", "-4"], "--chunks"),
    ("imatrix", ["m", "--corpus", "neutral", "--chunks", "0"], "--chunks"),
    ("imatrix", ["m", "--corpus", "neutral", "--gpu-layers", "-1"], "--gpu-layers"),
    ("capability", ["m", "--seed", "-1"], "--seed"),
]

#: The same, for the head-to-head family, which needs more required flags to reach its own.
H2H_REFUSED = [
    (["--trials", "-3"], "--trials"),
    (["--trials", "0"], "--trials"),
    (["--batch", "0"], "--batch"),
    (["--max-new", "-2"], "--max-new"),
    (["--skip-harmful", "-5"], "--skip-harmful"),
]

#: Values that must STILL be accepted, because zero is documented to mean something on each.
SIBLING_ACCEPTED = [
    ("measure", ["m", "--out", "o", "--capability-n", "0"]),
    ("track", ["--harmful", "h", "--harmless", "g", "--out", "o", "--fit", "0", "--search", "0"]),
    ("imatrix", ["m", "--corpus", "neutral", "--gpu-layers", "0"]),
    ("capability", ["m", "--seed", "0"]),
]


def _sibling_parser(command):
    import importlib
    module = importlib.import_module(f"senbonzakura.{command}")
    return module.build_parser()


@pytest.mark.parametrize(("command", "argv", "flag"), SIBLING_REFUSED)
def test_a_sibling_command_refuses_a_count_that_means_nothing(command, argv, flag, capsys):
    ap = _sibling_parser(command)
    with pytest.raises(SystemExit):
        ap.parse_args(argv)
    said = capsys.readouterr().err
    assert flag in said, (
        f"`{command} {' '.join(argv)}` was refused without naming {flag}, so a reader has to work "
        f"out which argument it was about: {said!r}")


@pytest.mark.parametrize(("command", "argv"), SIBLING_ACCEPTED)
def test_a_sibling_command_still_accepts_a_zero_that_means_something(command, argv):
    """The failure that teaches people to turn checking off is refusing a working invocation."""
    _sibling_parser(command).parse_args(argv)


@pytest.mark.parametrize(("argv", "flag"), H2H_REFUSED)
def test_the_head_to_head_refuses_a_count_that_means_nothing(argv, flag, capsys):
    ap = _sibling_parser("headtohead")
    with pytest.raises(SystemExit):
        ap.parse_args(["run", "--model", "m", "--out", "o", "--track", "t", *argv])
    assert flag in capsys.readouterr().err


def test_a_tolerance_cannot_be_negative_or_infinite():
    """`real_number`, the float companion, and the two values that make a comparison meaningless.

    A negative tolerance is a bar nothing clears, so every comparison fails and the run reports a
    mismatch it manufactured. An infinite one is a bar everything clears, so the instrument says
    yes to anything, which for a parity check is the whole question.
    """
    parse = argresolve.real_number("--tolerance", minimum=0.0)
    assert parse("0") == 0.0
    assert parse("1e-6") == pytest.approx(1e-6)
    for bad in ("-1e-9", "inf", "-inf", "nan", "wide", ""):
        with pytest.raises(argparse.ArgumentTypeError) as e:
            parse(bad)
        assert "--tolerance" in str(e.value), f"the message for {bad!r} does not name the flag"


def test_the_real_number_bound_holds_on_both_sides():
    """Mutation test: a one-sided check passes a table where only one side is ever exercised."""
    parse = argresolve.real_number("--fraction", minimum=0.0, maximum=1.0)
    assert parse("0.5") == pytest.approx(0.5)
    for bad in ("-0.1", "1.1"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse(bad)


# ── the backstop, because the sweep above reaches twenty parsers and the package has more ───
#
# FOUND 2026-10-02 WHILE WORKING THE DEBT LIST, and it is this project's most recurring defect
# shape rather than a new one: a check answering a narrower question than the one asked reports
# clean. `_every_parser` walks `entry.DELEGATED` and calls each module's `build_parser`, so it
# sees twenty parsers and 54 numeric flags, which reads like the whole package. It is not.
#
#   - `validate` is in DELEGATED and builds its parser inside a function, so there is no
#     `build_parser` to call and the module is skipped. Its five count flags (`--max-directions`,
#     `--direction-clusters`, `--dir-prompts`, `--eval-refusal`, `--eval-kl`) took a bare `int`
#     and the sweep above was green throughout.
#   - `subspace` and `headtohead_stage` are not in DELEGATED at all: one is a research entry
#     point, the other the stage half of the head-to-head. Five more flags between them.
#
# Every one of those was found by grepping the source for `type=int`, which is the measurement
# this backstop makes into a test. It reads the tree instead of importing it, so it also runs on
# a machine without torch, which is where the sweep above loses three of its own tests.

import ast  # noqa: E402
import pathlib  # noqa: E402

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "src" / "senbonzakura"

#: The two `type=` factories that refuse an out-of-range value and name the flag while doing it.
BOUNDED_FACTORIES = ("whole_number", "real_number")

#: Declarations the backstop lets through that the import-time sweep never sees. Vendored code is
#: upstream's and not ours to restyle (baseline section 6).
BACKSTOP_EXEMPT = {
    "--split-max-tensors": "vendored llama.cpp converter, fetched verbatim at the pinned tag",
}


def _package_sources():
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def _declared_flag(call):
    """The long flag an `add_argument` call declares, or its first string positional."""
    strings = [a.value for a in call.args
               if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    for value in strings:
        if value.startswith("--"):
            return value
    return strings[0] if strings else None


def _bare_numeric_declarations(tree):
    """`[(flag, "int"|"float")]` for every `add_argument` whose `type=` is an unbounded number."""
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg != "type":
                continue
            if isinstance(kw.value, ast.Name) and kw.value.id in ("int", "float"):
                found.append((_declared_flag(node), kw.value.id))
    return found


def test_the_backstop_reads_the_whole_package():
    """A walk over an empty file list passes, which is how a check goes quiet after a move."""
    calls = 0
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls += sum(1 for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "add_argument")
    assert calls >= 100, (
        f"only {calls} `add_argument` calls were found in the package, so this walk has stopped "
        f"reading the tree rather than the tree having stopped declaring flags")


def test_no_declaration_anywhere_in_the_package_takes_a_bare_number():
    """The wider question: every declaration, not only the ones a `build_parser` exposes."""
    allowed = set(EXEMPT) | set(NOT_YET_BOUNDED) | set(BACKSTOP_EXEMPT)
    offenders = {}
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for flag, kind in _bare_numeric_declarations(tree):
            if flag in allowed:
                continue
            offenders.setdefault(path.relative_to(PACKAGE.parent.parent).as_posix(),
                                 []).append((flag, kind))
    assert not offenders, (
        "these declarations take any number at all. The sweep above cannot see them, because the "
        "parser is built inside a function or the module is not in `entry.DELEGATED`:\n  "
        + "\n  ".join(f"{rel}: {flags}" for rel, flags in sorted(offenders.items()))
        + "\n  Use argresolve.whole_number or argresolve.real_number, or name the flag in EXEMPT.")


def test_the_backstop_catches_a_planted_declaration():
    """Mutation test. A walk matching nothing would pass on a tree full of the defect."""
    assert _bare_numeric_declarations(
        ast.parse("ap.add_argument('--trials', type=int, default=8)")) == [("--trials", "int")]
    assert _bare_numeric_declarations(
        ast.parse("p.add_argument('--tolerance', type=float)")) == [("--tolerance", "float")]


def test_the_backstop_accepts_every_bounded_spelling():
    """The other side of the same question: the fix has to read as a fix."""
    for src in (
        "ap.add_argument('--n', type=argresolve.whole_number('--n', minimum=1))",
        "ap.add_argument('--tol', type=argresolve.real_number('--tol', minimum=0.0))",
        "ap.add_argument('--n', type=whole_number('--n', minimum=1))",
    ):
        assert _bare_numeric_declarations(ast.parse(src)) == [], src
    assert BOUNDED_FACTORIES == ("whole_number", "real_number"), (
        "the factory names moved and the two tests above name them as literals")


def test_no_backstop_exemption_names_a_flag_the_package_no_longer_declares():
    """A stale exemption is a hole held open for the next flag of that name."""
    declared = set()
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "add_argument":
                flag = _declared_flag(node)
                if flag:
                    declared.add(flag)
    stale = sorted(set(BACKSTOP_EXEMPT) - declared)
    assert not stale, f"BACKSTOP_EXEMPT names flags no declaration has: {stale}"
