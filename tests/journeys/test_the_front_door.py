# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The first minute: typing the name, asking for help, and asking what this install can do.

Every one of these is somebody's first contact with the tool, and none of them should end in a
wall of flags, a traceback, or a refusal.
"""
import pytest


@pytest.mark.journey("command:senbonzakura")
def test_typing_the_name_answers_rather_than_refusing(session):
    """It printed a 27 line usage block listing every flag and then refused for want of a model.

    The first thing the tool ever said to anybody was a wall of flags followed by an error, and the
    guided mode, which is what a newcomer wants, appeared nowhere on it.
    """
    result = session()
    result.exited(0).carried_no_traceback()
    result.says("senbonzakura -i", "START HERE")
    result.never_says("--trials", "--kl-scale")
    result.fits_the_window()
    assert len(result.lines) < 25, f"the front door is {len(result.lines)} lines"


@pytest.mark.journey("command:senbonzakura", "state:narrow-terminal")
def test_the_front_door_in_a_half_width_window(session):
    session(columns=52).exited(0).fits_the_window().says("senbonzakura -i")


@pytest.mark.journey("command:doctor")
def test_asking_what_this_install_can_do(session):
    """`doctor` runs on an install too broken to do anything else, so it is never allowed to fail
    for the reasons the thing it is diagnosing fails for.
    """
    result = session("doctor")
    result.carried_no_traceback().fits_the_window()
    # `doctor`'s own scale, read off its own closing line: 0 nothing to report, 1 advisories,
    # 2 something failed. A journey that invented a third number would be asserting about a tool
    # it had not read.
    assert result.status in (0, 1), (
        f"doctor exited {result.status}. 0 is a clean install and 1 is advisories; 2 means "
        f"something failed on the machine running this, which is a finding about the machine.")
    result.says("Bundled corpora")


@pytest.mark.journey("command:doctor", "state:narrow-terminal")
def test_the_doctor_in_a_half_width_window(session):
    session("doctor", columns=52).fits_the_window().carried_no_traceback()


@pytest.mark.journey("command:convert")
def test_a_command_asked_for_help_says_what_it_is(session):
    """Asking what a command does must be answerable without the conditions for running it."""
    result = session("convert", "--help")
    result.exited(0).carried_no_traceback().says("usage")


@pytest.mark.journey("command:interactive")
def test_the_long_spelling_of_the_guided_mode_says_what_it_is(session):
    """`-i` and `senbonzakura interactive` are the same path under two spellings, and both are
    documented, so both are journeyed. Asking what a command does has to be answerable without
    the conditions for running it, which for this one means without a terminal.
    """
    result = session("interactive", "--help")
    result.exited(0).carried_no_traceback().says("usage: senbonzakura interactive")
    result.fits_the_window()


@pytest.mark.journey("command:setup")
def test_asking_what_would_be_changed_changes_nothing(session):
    """`setup` fits the install to the machine, and its whole safety property is that it reports
    before it acts: it changes nothing unless `--apply` is given.
    """
    result = session("setup")
    result.carried_no_traceback().fits_the_window()
    assert result.status in (0, 1, 3), f"setup exited {result.status}"
    result.never_says("Traceback")
