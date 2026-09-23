# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`CITATION.cff`, and the community files JOSS checks for at intake.

These are cheap to get wrong and expensive to get wrong late: a submission is bounced at intake
for a missing file, not reviewed and then bounced, so the cost lands after the work is done.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _citation():
    return (ROOT / "CITATION.cff").read_text(encoding="utf-8")


def test_the_cited_version_is_the_version_this_tree_builds():
    """A citation naming a version nobody can install is worse than one naming none: it sends a
    reader looking for something that does not exist, and it looks authoritative while doing it.
    """
    declared = re.search(r'^version:\s*"([^"]+)"', _citation(), flags=re.MULTILINE)
    assert declared, "CITATION.cff carries no version field"
    src = (ROOT / "src" / "senbonzakura" / "_version.py").read_text(encoding="utf-8")
    actual = re.search(r'__version__\s*=\s*"([^"]+)"', src)
    assert actual, "_version.py no longer declares __version__"
    assert declared.group(1) == actual.group(1), (
        f"CITATION.cff cites {declared.group(1)} and this tree is {actual.group(1)}. The citation "
        f"is the one artefact a reader is invited to copy, so it has to name a real version")


@pytest.mark.parametrize("name", ["CODE_OF_CONDUCT.md", "SUPPORT.md", "SECURITY.md",
                                  "CONTRIBUTING.md", "CITATION.cff"])
def test_the_community_files_exist(name):
    """JOSS asks for three things: how to contribute, how to report issues, how to seek support.
    A CONTRIBUTING.md covering only the first does not satisfy the checklist, and a support route
    that exists only inside GitHub's new-issue chooser is not discoverable by somebody reading
    the repository.
    """
    assert (ROOT / name).is_file(), f"{name} is missing"


@pytest.mark.parametrize("name", ["CODE_OF_CONDUCT.md", "SUPPORT.md", "SECURITY.md"])
def test_the_readme_points_at_them(name):
    """A file nobody is sent to is a file nobody reads. These were all reachable only by knowing
    they existed until 2026-09-22.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert name in readme, f"README.md does not link {name}"


def test_the_security_contact_resolves_to_an_address():
    """CONTRIBUTING.md said "email the address in pyproject.toml" and pyproject.toml carried no
    address, so the one instruction a security reporter follows led nowhere. Found 2026-09-22.
    """
    from tomlread import tomllib
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    emails = [a.get("email") for a in cfg["project"].get("authors", [])]
    assert any(emails), "pyproject.toml names no author email"
    for name in ("SECURITY.md", "CONTRIBUTING.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert any(e and e in text for e in emails), (
            f"{name} does not carry an address a reporter can actually use")


def test_the_readme_asserts_the_projects_own_copyright():
    """The only copyright line in the distributed code used to be upstream's. Every module under
    src/ carries ours now, and the licence section a reader actually reads did not.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Copyright (C) 2026 Daniel Iwugo" in readme
