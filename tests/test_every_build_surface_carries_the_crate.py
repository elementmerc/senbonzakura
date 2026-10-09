# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The extension is required, so every surface that builds this package has to carry its source.

Making the Rust extension required was one change in one place, and it broke four build surfaces
at once because each one keeps its own list of what to carry. CI found them one job at a time:

    the test environment    No module named 'setuptools_rust'   (7 jobs)
    the sdist               can't find manifest for Rust extension at path `rust/Cargo.toml`
    Dockerfile.cuda         the same, inside a container build
    Dockerfile              no failure at all, which is the worst of the four

The last one is why this file is a test rather than a note. The CPU image copied no `setup.py`,
and `setup.py` is where the extension is declared. With it absent setuptools builds the package
as pure Python, the build SUCCEEDS, and the image ships without `senbonzakura._native`. The
failure then waits for a user. A build that fails is a build somebody fixes; a build that quietly
drops a compiled module is a release.

Each check below is one manifest that had to be told, and the test exists so the next required
piece of native code is told everywhere at once.
"""

from __future__ import annotations

import pathlib
import re

from tomlread import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust" / "Cargo.toml"
DOCKERFILES = ("Dockerfile", "Dockerfile.cuda")


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_crate_is_actually_there():
    """Everything below is about carrying this file, so it is worth asserting it exists."""
    assert CRATE.is_file(), f"{CRATE.relative_to(ROOT)} is missing and setup.py declares it"
    assert (ROOT / "rust" / "Cargo.lock").is_file(), (
        "Cargo.lock is not committed, so a source build resolves its own dependency versions and "
        "does not reproduce. Section 5 commits lockfiles for libraries as well as binaries")


def test_the_sdist_carries_the_crate():
    """The sdist shipped without it, so a wheel could not be built from the tarball at all.

    That is every source install: a platform with no published wheel, and anybody who passes
    `--no-binary`. `rust/` sits outside `src/`, so setuptools never picks it up on its own.
    """
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include rust/Cargo.toml" in manifest, (
        "MANIFEST.in does not carry the crate manifest, so `python -m build --sdist` produces a "
        "tarball that cannot build a wheel")
    assert "include rust/Cargo.lock" in manifest, "the sdist needs the lockfile to reproduce"
    assert re.search(r"recursive-include\s+rust/src\s+\*\.rs", manifest), (
        "MANIFEST.in does not carry the crate's sources")


def test_the_test_environment_installs_the_build_backend():
    """`[build-system] requires` exists during a build and not in the environment the suite runs
    in. The wheel tests import `setup.py`, which imports `setuptools_rust`, so it is a test
    dependency too. It was in one list and not the other, and every test job failed on it while
    passing locally on a machine that had it left over from building a wheel.
    """
    cfg = _pyproject()
    dev = cfg["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("setuptools-rust") for d in dev), (
        "the dev extra does not install setuptools-rust, so importing setup.py raises "
        "ModuleNotFoundError in any environment that has not just built a wheel")


def test_the_two_pins_of_the_build_backend_agree():
    """Two places name a version, and the tests read the same code the build runs."""
    cfg = _pyproject()
    build = [r for r in cfg["build-system"]["requires"] if r.startswith("setuptools-rust")]
    dev = [d for d in cfg["project"]["optional-dependencies"]["dev"]
           if d.startswith("setuptools-rust")]
    assert build and dev, "both lists must pin it for them to be comparable"
    assert build[0] == dev[0], (
        f"the build requires {build[0]} and the tests install {dev[0]}, so the suite is reading "
        f"setup.py through a different version of setuptools_rust than a release is built with")


def test_every_container_copies_the_crate_and_the_setup_script():
    """The check that would have caught the silent one.

    A Dockerfile that copies the crate but not `setup.py` builds a pure wheel and ships an image
    with no native module, with no error anywhere. Both lines are required, and a missing
    `setup.py` is the more dangerous absence of the two.
    """
    for name in DOCKERFILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert re.search(r"^COPY\s+rust/\s", text, re.MULTILINE), (
            f"{name} does not copy rust/, so the image build fails with \"can't find manifest "
            f'for Rust extension"')
        assert re.search(r"^COPY\s[^\n]*\bsetup\.py\b", text, re.MULTILINE), (
            f"{name} does not copy setup.py, which is where the extension is declared. Without "
            f"it this image builds SUCCESSFULLY as pure Python and ships with no "
            f"senbonzakura._native, so the first command that needs it fails in a user's hands "
            f"rather than in CI")
