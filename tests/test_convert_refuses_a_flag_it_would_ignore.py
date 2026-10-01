# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`convert --imatrix` without `--quantise` is refused rather than accepted and dropped.

WHAT PROMPTED IT, 2026-10-01

A CLI surface sweep item: "`convert` accepts `--imatrix` and `--keep-intermediate` without
`--quantise`." Both flags say "with --quantise" in their own help text, and both were accepted
and then ignored. So

    senbonzakura convert ./edited --imatrix cal.imatrix

converted the checkpoint, reported DONE, and applied no importance matrix. Not a crash: a number
produced by a run that was not the run requested, which is the failure shape this project spends
its time finding in other people's tools.

`--keep-intermediate` is the quieter half and the nastier one. Without `--quantise` there is no
intermediate at all, so the flag does not merely get ignored, it describes a file that does not
exist. Somebody passing both and finding one GGUF cannot tell whether the quantisation ran and
the intermediate was dropped or neither happened.

WHY THE WORKED COMMAND IS ASSERTED, NOT JUST THE REFUSAL

The first draft of the fix built that line from the flag names alone and offered
`--quantise Q4_K_M --imatrix` with no FILE, which is an argparse error if a reader pastes it. A
refusal offering a command that does not run is worse than one offering none, so the value the
user typed is carried through and the test holds it.
"""
from __future__ import annotations

import pytest

from senbonzakura import convert, say

#: Every combination of the two dependent flags, as a user would type them.
COMBINATIONS = {
    "the imatrix alone": ["--imatrix", "cal.imatrix"],
    "keep-intermediate alone": ["--keep-intermediate"],
    "both together": ["--imatrix", "cal.imatrix", "--keep-intermediate"],
}


def _refusal(flags, model="./edited"):
    """What `convert` exits with, driven through `run` rather than through the checker function.

    Through `run` deliberately. The check is only worth anything if it sits ahead of `preflight`,
    and a test calling `refuse_dependent_flags` directly would pass with the call placed anywhere
    in the function, including after the conversion. `./edited` does not exist, so a run that got
    past the refusal would die in `preflight` with a different message, which is what the ordering
    test below reads.
    """
    with pytest.raises(SystemExit) as exit_info:
        convert.run([model, *flags])
    return str(exit_info.value)


@pytest.mark.parametrize("case", sorted(COMBINATIONS))
def test_a_dependent_flag_without_quantise_is_refused(case):
    said = _refusal(COMBINATIONS[case])
    assert "--quantise" in said, said
    assert "was not given" in said, said


@pytest.mark.parametrize("case", sorted(COMBINATIONS))
def test_the_refusal_names_every_flag_that_was_given(case):
    """Naming one of two would send the reader to fix half the command line."""
    said = _refusal(COMBINATIONS[case])
    for flag in COMBINATIONS[case]:
        if flag.startswith("--"):
            assert flag in said, f"{flag} was given and is unnamed in: {said}"


def test_the_refusal_names_no_flag_that_was_not_given():
    """Advice about a flag nobody typed is noise, and noise is what gets skimmed past."""
    said = _refusal(COMBINATIONS["keep-intermediate alone"])
    assert "--imatrix" not in said, said


def test_the_worked_command_carries_the_value_the_user_typed():
    """The defect in the fix's own first draft, held as a property.

    A bare `--imatrix` in the suggested command is an argparse error, so the suggestion has to
    round-trip. Asserted on the pasteable text rather than on the flag's presence, because the
    flag was present in the broken version too.
    """
    said = _refusal(COMBINATIONS["the imatrix alone"])
    assert "--quantise Q4_K_M --imatrix cal.imatrix" in said, said


def test_the_refusal_says_what_the_ignored_flag_would_have_done():
    """Without this the reader is told a flag is wrong and not why they wanted it."""
    said = _refusal(COMBINATIONS["both together"])
    assert "importance matrix" in said
    assert "full-precision GGUF" in said


def test_it_refuses_before_the_model_is_even_looked_at():
    """The property that makes this a pre-flight rather than a tidier failure.

    `./nowhere-at-all` does not exist, so a run reaching `preflight` says "no model directory
    at ...". Meeting that message instead would mean the check is downstream of the work.

    MUTATION-CHECKED, 2026-10-01, and one mutant survives on purpose. Moving the call below
    `default_output` is not caught, because `default_output` only resolves a path and touches no
    disk, so that move changes nothing a user could observe. Moving it below `preflight` is caught
    here and by eight other tests. Recorded rather than papered over: a test that fails on a
    behaviour-preserving edit is a test that will be deleted by the next person who refactors.
    """
    said = _refusal(COMBINATIONS["the imatrix alone"], model="./nowhere-at-all")
    assert "no model directory" not in said, (
        f"the flag check now runs after the model is read: {said}")


def test_quantise_given_is_not_refused(tmp_path):
    """The guard must not fire on the combination the flags were written for.

    Reaching `preflight`'s refusal about the missing model is the whole claim: it proves the
    command got past the dependent-flag check rather than being stopped by it.
    """
    said = _refusal(["--quantise", "Q4_K_M", "--imatrix", "cal.imatrix", "--keep-intermediate"],
                    model=str(tmp_path / "nowhere"))
    assert "no model directory" in said, said
    assert "--quantise" not in said, f"the dependent-flag refusal fired with --quantise given: {said}"


def test_neither_flag_given_is_not_refused(tmp_path):
    said = _refusal([], model=str(tmp_path / "nowhere"))
    assert "no model directory" in said, said


@pytest.mark.parametrize("case", sorted(COMBINATIONS))
@pytest.mark.parametrize("columns", [80, 120, 40])
def test_the_refusal_fits_the_terminal(case, columns, monkeypatch):
    """Prose wraps, the worked command does not, and `say` owns which is which.

    The indented command line is exempt by `say.is_verbatim`'s rule rather than by this test's
    opinion: a wrapped command cannot be pasted, which destroys the most useful thing in the
    message in the name of tidiness.
    """
    monkeypatch.setenv("COLUMNS", str(columns))
    said = _refusal(COMBINATIONS[case])
    for line in said.splitlines():
        if say.is_verbatim(line):
            continue
        assert len(line) <= say.CEILING, f"{len(line)} columns at COLUMNS={columns}: {line!r}"


def test_the_table_and_the_parser_agree_on_every_dependent_flag():
    """A row here with no flag there, or the reverse, is the drift this table exists to prevent.

    Read off the parser rather than listed, so adding a third dependent flag to `_NEEDS_QUANTISE`
    without declaring it, or renaming the declared one, fails here instead of producing a refusal
    that names a flag the command does not have.
    """
    parser = convert.build_parser()
    declared = {action.dest: action for action in parser._actions}
    for attr, (spelling, _what, takes_value) in convert._NEEDS_QUANTISE.items():
        assert attr in declared, f"_NEEDS_QUANTISE names {attr}, which the parser does not declare"
        action = declared[attr]
        assert spelling in action.option_strings, (
            f"_NEEDS_QUANTISE spells {attr} as {spelling}, the parser as {action.option_strings}")
        assert bool(action.nargs != 0 and action.metavar) == takes_value, (
            f"_NEEDS_QUANTISE says {spelling} "
            f"{'takes' if takes_value else 'does not take'} a value, and the parser disagrees")
        assert "with --quantise" in (action.help or ""), (
            f"{spelling}'s help text no longer says it depends on --quantise, so either the "
            f"dependency changed or the help drifted from the refusal")
