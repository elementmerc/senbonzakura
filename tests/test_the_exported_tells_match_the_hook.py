# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The CI copy of the tell patterns has to stay the hook's list, not a fork of it.

`check_prose_tells.py` parses the pre-commit hook rather than carrying a copy, so the gate and the
hook can never disagree. CI cannot do that, because the hooks are deliberately kept out of this
repository, so it reads a generated export instead. That trade buys a working gate and takes on
exactly one risk: the export going stale while the hook moves on, which would leave CI enforcing
yesterday's list while looking like it enforces today's.

These tests are where that risk is paid for. The drift comparison can only run where the hook
exists, which is every machine where the hook can be changed, so the check lives where the change
happens rather than where the file is consumed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-commit"
EXPORT = ROOT / "tools" / "ci" / "prose-tells.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _checker():
    spec = importlib.util.spec_from_file_location(
        "cpt_under_test", ROOT / "tools" / "ci" / "check_prose_tells.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patterns(path):
    return [raw for raw, _ in _checker().read_patterns(path)]


def test_the_export_exists_and_parses():
    """Whatever else is true, CI has a list to read and it is not a stub."""
    assert EXPORT.is_file(), (
        f"{EXPORT.relative_to(ROOT)} is missing, so the CI prose gate has no pattern list and "
        f"every test job fails. Regenerate it with: python tools/dev/export_prose_tells.py")
    checker = _checker()
    assert len(_patterns(EXPORT)) >= checker.MIN_PATTERNS


def test_the_export_says_it_is_generated():
    """A hand edit to a generated file is lost on the next export, so it has to say so."""
    text = EXPORT.read_text(encoding="utf-8")
    assert "GENERATED FILE" in text[:100], "the warning has to be the first thing a reader meets"
    assert "export_prose_tells.py" in text, "a generated file has to name what regenerates it"


@pytest.mark.skipif(
    not HOOK.is_file(),
    reason="the fleet pre-commit hook is not installed here, so there is nothing to compare "
           "against. Drift can only be introduced where the hook exists, and that machine runs "
           "this test.")
def test_the_export_has_not_drifted_from_the_hook():
    """The whole justification for the copy: a test catches it going stale."""
    assert _patterns(EXPORT) == _patterns(HOOK), (
        "the exported pattern list no longer matches the hook, so CI is enforcing a different "
        "set of tells than a commit is. Regenerate with: "
        "python tools/dev/export_prose_tells.py")


@pytest.mark.skipif(not HOOK.is_file(), reason="needs the hook to export from")
def test_the_exporter_reports_staleness_rather_than_fixing_it_silently():
    """`--check` is what a gate can call; it must fail rather than quietly rewrite."""
    before = EXPORT.read_text(encoding="utf-8")
    clean = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "dev" / "export_prose_tells.py"), "--check"],
        capture_output=True, text=True, timeout=60, check=False)
    assert clean.returncode == 0, clean.stderr

    EXPORT.write_text(before.replace("GENERATED FILE", "GENERATED FILE "), encoding="utf-8")
    try:
        stale = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "dev" / "export_prose_tells.py"), "--check"],
            capture_output=True, text=True, timeout=60, check=False)
        assert stale.returncode == 1
        assert "stale" in stale.stderr
    finally:
        EXPORT.write_text(before, encoding="utf-8")


def test_the_workflow_names_the_export_rather_than_defaulting_to_the_hook():
    """The defect this file exists for: CI defaulting to a file a public checkout cannot have."""
    text = WORKFLOW.read_text(encoding="utf-8")
    calls = [ln for ln in text.splitlines() if "check_prose_tells.py" in ln and "run:" in ln]
    assert calls, "the workflow no longer runs the prose gate at all"
    for line in calls:
        assert "--hook tools/ci/prose-tells.sh" in line, (
            f"this workflow step reads the pattern list from the pre-commit hook, which a public "
            f"checkout never has, so the job fails with exit 2 rather than scanning anything: "
            f"{line.strip()}")
