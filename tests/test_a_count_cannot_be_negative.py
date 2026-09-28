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
NOT_YET_BOUNDED: dict[str, str] = {
    "--seed": "capability, and the abliterate and compass copies are bounded",
    "--min-applied": "check, where zero may be a legitimate absence of a floor",
    "--batch": "drift",
    "--expect-size": "fetch",
    "--chunks": "imatrix",
    "--gpu-layers": "imatrix",
    "--n": "measure",
    "--capability-n": "measure, where zero is off as it is on the abliterate path",
    "--threads": "quantise",
    "--fit": "track",
    "--search": "track",
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
    assert len(NOT_YET_BOUNDED) <= 11, (
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
