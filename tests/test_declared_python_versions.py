# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every Python version the package claims is a version CI actually runs.

WHY THIS EXISTS

`08-v1.0-joss-ready.md` carries the exit-gate line *"Either a Python version test matrix, or
classifiers narrowed to what CI actually tests."* A trove classifier is not decoration: it is what
a package index shows a user deciding whether this will work for them, and pip does not check it.
So a claimed version that nobody runs is a promise made to strangers and kept by luck.

Measured 2026-09-21: the package claims 3.10, 3.11, 3.12, 3.13 and 3.14. CI sweeps 3.10, 3.12 and
3.14 on Linux and one representative version elsewhere. **3.11 and 3.13 are claimed and never
run.** This project has already shipped the shape of that defect twice: `tests/tomlread.py` exists
solely because six test files imported `tomllib` unconditionally, which cannot be collected on the
DECLARED MINIMUM interpreter, and it was caught only when CI's 3.10 job went red, on a machine
where every interpreter was 3.14.

THE RULE IS DELIBERATELY ONE-DIRECTIONAL. A version CI runs but does not claim is fine: testing
more than you promise is a virtue, and the matrix legitimately holds representative rows on
Windows and macOS. A version claimed but not run is the failure. So this asserts a subset, not an
equality.

The plan's own statement of this gap said the package "tests one". It tests three. That line was
written when it was true and nothing marked it when it stopped being, which is the same rot this
file exists to prevent one layer down.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from tomlread import tomllib

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"

_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (\d+\.\d+)$")

#: Any `python-version: "3.x"` or a matrix row's `python: "3.x"`. Read from the text rather than
#: by parsing the workflow as YAML, because YAML is a dependency this project does not carry and
#: the question here is which version strings appear, which is a textual one.
#:
#: THE NEGATIVE LOOKAHEAD AND THE COMMENT STRIPPING ARE BOTH LOAD-BEARING, and the first version
#: of this file had neither. It reported 3.13 as tested. The only occurrence of "3.13" in the
#: workflow is the word `python:3.13-slim` inside a PROSE COMMENT about a docker image, and the
#: pattern matched it because `python:` is also the matrix key and `\s*` permits zero spaces. So
#: the guard written to catch an untested version claim silently excused one, which is precisely
#: the defect class it exists for. Caught before it was committed, by checking where the match
#: came from instead of trusting that the test had failed on the right line.
#: THE TRAILING LOOKAHEAD HAS TO FORBID WORD CHARACTERS, not just a hyphen, and the version that
#: forbade only a hyphen still matched. Against `python:3.13-slim` the engine backtracked the
#: version group down to `3.1`, found the next character was `3` rather than `-`, and reported
#: "3.1" as a tested version. A lookahead does not stop a regex trying a shorter match; it only
#: rejects the one it is looking at. Caught by the test written for the first bug, which is the
#: argument for writing the test rather than reasoning about the fix.
_CI_VERSION = re.compile(r'python(?:-version)?:[ \t]*["\']?(\d+\.\d+)["\']?(?![\w.-])')


def _strip_comments(text):
    """Every `#` comment removed, so prose about Python versions is not read as configuration.

    Crude on purpose: a `#` inside a quoted YAML string would be stripped too. That errs towards
    reporting a version as UNTESTED, which fails loudly and is fixed by looking, rather than
    towards reporting an untested version as covered, which passes silently and is never fixed.
    """
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _claimed():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    out = set()
    for c in data["project"].get("classifiers") or []:
        m = _CLASSIFIER.match(c.strip())
        if m:
            out.add(m.group(1))
    return out


def _tested():
    if not CI.is_file():
        return None
    return set(_CI_VERSION.findall(_strip_comments(CI.read_text(encoding="utf-8"))))


def test_the_package_claims_at_least_one_python_version():
    """A guard whose input is empty passes silently, which is the way guards die here."""
    assert _claimed(), (
        "no `Programming Language :: Python :: X.Y` classifiers in pyproject.toml, so the check "
        "below would compare an empty set against anything and pass. That is the shape of every "
        "check this project has watched go green having measured nothing.")


def test_the_ci_workflow_is_readable_and_names_versions():
    """The other half of the same trap: an unreadable CI file must not read as full coverage."""
    tested = _tested()
    assert tested is not None, f"{CI} does not exist, so nothing here knows what CI runs."
    assert tested, (
        f"no python version strings found in {CI.name}. Either the workflow stopped pinning "
        f"versions or the pattern this test uses has gone stale, and both mean the assertion "
        f"below is vacuous.")


@pytest.mark.parametrize("version", sorted(_claimed()))
def test_every_claimed_python_version_is_one_ci_actually_runs(version):
    """THE ASSERTION. A classifier is a promise to a stranger reading a package index.

    Parametrised so the failure names the offending version rather than a set difference, because
    the fix differs per version: add a matrix row, or drop the claim.
    """
    tested = _tested() or set()
    assert version in tested, (
        f"pyproject.toml claims support for Python {version} and no job in "
        f"{CI.name} runs it. CI runs {sorted(tested)}. A trove classifier is what a package index "
        f"shows somebody deciding whether this will work on their machine, and pip does not "
        f"verify it, so an untested claim is kept by luck. Either add {version} to the matrix, or "
        f"remove the classifier. The v1.0 exit gate asks for exactly one of those two.")


def test_the_declared_floor_is_among_the_claimed_versions():
    """`requires-python` and the classifiers are two statements of the same fact.

    They drift in opposite directions: the floor is enforced by pip and the classifiers are not,
    so a floor raised without touching the classifiers leaves the index advertising a version the
    installer refuses.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    spec = str(data["project"].get("requires-python") or "")
    m = re.search(r">=\s*(\d+\.\d+)", spec)
    assert m, f"requires-python is {spec!r} with no `>=` floor to compare against."
    floor = m.group(1)
    claimed = _claimed()
    assert floor in claimed, (
        f"requires-python declares a floor of {floor} and the classifiers do not list it "
        f"({sorted(claimed)}). pip enforces the floor and ignores the classifiers, so the index "
        f"is advertising a set of versions that does not include the oldest one supported.")


# ── the trap this guard fell into before it worked ───────────────────────────────
def test_a_version_named_only_in_a_comment_is_not_a_tested_version():
    """THE MUTATION THAT WOULD PUT THE ORIGINAL BUG BACK.

    The workflow contains the words `python:3.13-slim` in a comment about a docker image, and
    nothing runs 3.13. The first version of this file read that as a matrix entry and excused a
    claim nobody tests. Both defences are asserted here, because either alone lets it back in.
    """
    sample = (
        '        # that happened twice inside python:3.13-slim, where the failure\n'
        '          - {os: ubuntu-latest, python: "3.10", floor: "95"}\n'
        '          python-version: "3.14"\n')
    got = set(_CI_VERSION.findall(_strip_comments(sample)))
    assert got == {"3.10", "3.14"}, (
        f"parsed {sorted(got)} from a sample whose only real entries are 3.10 and 3.14. "
        f"3.13 appears solely in a comment about a container image.")


def test_stripping_comments_alone_is_not_the_whole_defence():
    """An image tag on a real configuration line must still not read as a version.

    Kept separate from the test above so that removing either the comment strip or the lookahead
    fails a test that names the one that went, rather than one failure covering both.
    """
    assert not _CI_VERSION.findall("        image: python:3.13-slim\n")
