# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The advisories this repository has decided to live with, held to what they claim.

An `audit_ignore` entry silences a real finding. Written once and never read again, it is how a
known hole becomes a permanent one: the person who assessed it moves on, the pin it was assessed
against moves under it, and the entry keeps saying "we looked at this" about a version nobody
looked at.

So every accepted advisory carries the version it was assessed against, and these tests fail the
moment that version moves. Whoever bumps the dependency has to re-read the assessment rather than
inherit it.
"""
import pathlib
import re

import pytest
from tomlread import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _accepted():
    cfg = _pyproject()
    return cfg["tool"]["senbonzakura"]["security"]["accepted_advisories"]


def _pinned(package):
    text = (ROOT / "constraints.txt").read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(package)}==(\S+)", text, flags=re.MULTILINE)
    return m.group(1) if m else None


def test_every_silenced_advisory_has_a_written_assessment():
    """The two lists are separate files' worth of intent and they must name the same advisories.

    `audit_ignore` is what the gate reads; `accepted_advisories` is what a person reads. An id in
    the first and not the second is a finding silenced with no recorded reason, which is exactly
    the move this file exists to prevent.
    """
    ignored = set(_pyproject()["tool"]["invariant"].get("audit_ignore", []))
    assessed = {a["id"] for a in _accepted()}
    unexplained = sorted(ignored - assessed)
    assert not unexplained, (
        f"{unexplained} is silenced in audit_ignore with no entry in accepted_advisories, so the "
        f"gate is quiet about it and nothing says why")
    stale = sorted(assessed - ignored)
    assert not stale, (
        f"{stale} is recorded as accepted and is no longer silenced, so the note is describing a "
        f"decision that is not in force")


@pytest.mark.parametrize("advisory", _accepted(), ids=lambda a: a["id"])
def test_an_assessment_names_what_it_assessed(advisory):
    for field in ("id", "package", "assessed_version", "mitigation", "revisit"):
        assert advisory.get(field), f"{advisory.get('id')} carries no {field}"


@pytest.mark.parametrize("advisory", _accepted(), ids=lambda a: a["id"])
def test_the_assessed_version_is_still_the_pinned_one(advisory):
    """THE EXPIRY. This is the whole mechanism.

    An accepted advisory is a statement about one version of one package. When the pin moves, the
    statement is about a version nobody assessed, and the correct outcome is a failing test that
    sends somebody back to the note rather than a silence that carries forward.

    If this fails because the dependency was upgraded: check whether the advisory is fixed in the
    new version. If it is, delete both the audit_ignore entry and the assessment. If it is not,
    re-read the mitigation against the new code and update assessed_version.
    """
    pinned = _pinned(advisory["package"])
    assert pinned is not None, (
        f"{advisory['package']} is no longer pinned in constraints.txt, so there is nothing to "
        f"hold the assessment of {advisory['id']} to")
    assert pinned == advisory["assessed_version"], (
        f"{advisory['id']} was assessed against {advisory['package']} "
        f"{advisory['assessed_version']} and the pin is now {pinned}. The silence in audit_ignore "
        f"is carrying forward to a version nobody looked at. Re-read the assessment in "
        f"pyproject.toml, then either drop the entry because the advisory is fixed, or update "
        f"assessed_version because it is not")


# ── the other accepted-gap list, which had no guard at all until 2026-10-12 ──────────────────
#
# `unauditable_packages` is the second way this repository decides to live with something: not a
# known finding silenced, but a package the advisory service cannot look up, counted as clean.
# `tools/ci/audit_env.py` refuses an undeclared skip and its error tells the reader to record the
# package "with a reason and a condition for revisiting", which was a promise about a shape that
# NOTHING CHECKED. An entry carrying a bare package name would have satisfied the gate, silenced
# a package, and explained nothing, which is precisely what the list above is guarded against.
#
# Found while adding our own two distributions to the list during the 0.5.0 release, for the
# honest reason that an external feed cannot describe a working tree. Adding to an ungoverned
# accepted-gap list is the moment to govern it.

def _unauditable():
    return _pyproject()["tool"]["senbonzakura"]["security"]["unauditable_packages"]


@pytest.mark.parametrize("entry", _unauditable(), ids=lambda e: e["package"])
def test_an_unauditable_package_names_why_and_when_to_look_again(entry):
    for field in ("package", "reason", "revisit"):
        assert entry.get(field), (
            f"{entry.get('package')} is declared unauditable with no {field}. The gate counts it "
            f"as clean, so this text is the only thing standing between a skipped package and a "
            f"hole nobody revisits")


def test_the_unauditable_list_is_the_one_the_gate_actually_reads():
    """Two readers of one list, held to the same answer.

    These tests read `pyproject.toml` directly; the gate reads it through `audit_env`, which
    canonicalises names. If those ever diverge, this file would be policing entries that silence
    nothing while the gate silences packages this file never saw.
    """
    import sys
    sys.path.insert(0, str(ROOT / "tools" / "ci"))
    import audit_env

    assert set(audit_env.declared_unauditable()) == {
        audit_env._canonical(e["package"]) for e in _unauditable()}


def test_the_mitigation_named_for_the_accelerate_advisory_actually_exists():
    """A mitigation described in prose and absent from the tree is the worst of both: the finding
    is silenced AND nothing is guarding it. So the claim is held to a module that must import and
    a function that must be there.
    """
    accelerate = [a for a in _accepted() if a["package"] == "accelerate"]
    if not accelerate:
        pytest.skip("no accepted advisory for accelerate")
    from senbonzakura import checkpoint
    assert callable(checkpoint.refuse_unsafe_index)
    assert callable(checkpoint.refuse_unsafe_hub_index)
