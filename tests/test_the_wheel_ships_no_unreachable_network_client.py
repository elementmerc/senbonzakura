# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Nothing in the shipped package reads a provider credential that nothing can reach.

WHY THIS EXISTS

`rented.py` was 520 lines of RunPod REST client. It read `RUNPOD_API_KEY` from the environment and
made HTTPS calls, it was imported by nothing except its own test, and it shipped in every wheel
from the release it was written in. Nobody could use it, and everybody installed it.

It is good code written for a real planned feature, and that is beside the point. An unreachable
network client that handles a credential is attack surface bought for no benefit, and the benefit
only arrives on the day something calls it.

It now lives in `unshipped/`, which is outside `src/` so that no packaging rule has to remember to
exclude it. `setuptools` discovers packages under `src/`; a file that cannot be discovered cannot
be shipped by accident. This file is the other half: the directory stops it being shipped, and
these tests stop it being moved back without being wired up.

WHAT IS ASSERTED

That the source tree holds no module which both reads a credential and is reachable from nothing.
Asked of the tree rather than of a built wheel, because a test that needs a build step is a test
that gets skipped in the environment where it matters.

`test_the_built_wheel_agrees` does check a real wheel when one is lying about, which is the only
way to be certain, and skips rather than failing when there is not one.
"""
import ast
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "senbonzakura"
UNSHIPPED = ROOT / "unshipped"

#: Environment variables that name somebody's credential. Deliberately a pattern rather than a
#: list: the next provider will not be RunPod, and a list would pass for it on the day it lands.
CREDENTIAL = re.compile(r"\b[A-Z][A-Z0-9_]*(API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)[A-Z0-9_]*\b")

#: Reading one of these is not holding a credential. `HF_TOKEN` is read by code the user reaches
#: through documented flags, which is the whole difference this file is about.
REACHABLE_ANYWAY = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"}


def _modules():
    for p in sorted(PACKAGE.rglob("*.py")):
        if "vendor" in p.parts:
            continue        # somebody else's code, vendored at a pin, not ours to restructure
        yield p


def _imported_names(tree):
    """Every module name this file imports, however it imports it."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[-1])
            out.update(a.name for a in node.names)
    return out


def _reachable_from_the_package():
    """Module stems that something else in the package imports."""
    seen = set()
    for p in _modules():
        seen |= _imported_names(ast.parse(p.read_text(encoding="utf-8")))
    return seen


def test_no_shipped_module_holds_a_credential_that_nothing_can_reach():
    reachable = _reachable_from_the_package()
    offenders = []
    for p in _modules():
        if p.stem in reachable or p.stem in ("__init__", "__main__"):
            continue
        found = {m.group(0) for m in CREDENTIAL.finditer(p.read_text(encoding="utf-8"))}
        found -= REACHABLE_ANYWAY
        if found:
            offenders.append(f"{p.relative_to(ROOT)} reads {sorted(found)} and nothing imports it")
    assert not offenders, (
        "these ship in the wheel, handle a credential, and are reachable from nothing:\n  "
        + "\n  ".join(offenders)
        + "\n\nWire it to a command, or move it to unshipped/. See unshipped/README.md.")


def test_the_runpod_client_is_not_in_the_package():
    """The specific one, by name, so a move back is loud rather than quiet."""
    assert not (PACKAGE / "rented.py").exists(), \
        "rented.py is back in the package. If it is wired up now, delete this test and say so."
    assert (UNSHIPPED / "rented.py").exists(), "unshipped/rented.py is gone; where did it go?"


def test_unshipped_is_outside_the_package_root():
    """The mechanism, not just the outcome. If this moves under src/, setuptools ships it."""
    assert UNSHIPPED.is_dir()
    assert (ROOT / "src") not in UNSHIPPED.parents, \
        "unshipped/ is under src/, where setuptools discovers packages. That is how it got shipped."
    assert not (UNSHIPPED / "__init__.py").exists(), \
        "an __init__.py makes unshipped/ a package, which is what makes a thing discoverable"


def test_unshipped_says_why_it_exists():
    """A directory whose rule is not written down is a directory somebody tidies away."""
    readme = (UNSHIPPED / "README.md").read_text(encoding="utf-8")
    assert "RUNPOD_API_KEY" in readme
    assert "attack surface" in readme


def test_the_module_still_works_where_it_now_lives():
    """Moved, not abandoned. Its own suite covers the behaviour; this covers the import."""
    sys.path.insert(0, str(UNSHIPPED))
    try:
        import rented
    finally:
        sys.path.remove(str(UNSHIPPED))
    # The names the module actually exports, read off it rather than guessed: the first version of
    # this asserted `BASE`, which is an attribute of `RunPodProvider` and not of the module.
    for name in ("RunPodProvider", "FakeProvider", "Rental", "PodSpec"):
        assert hasattr(rented, name), f"rented.{name} is gone; the move broke something"


def test_the_built_wheel_agrees():
    """The only certain check, run when a wheel happens to be lying about.

    Skipped rather than failed without one: requiring a build step would make this the test that
    never runs in the environment where it would catch something.
    """
    wheels = sorted(ROOT.glob("dist/*.whl"))
    if not wheels:
        pytest.skip("no wheel built in this checkout")
    # A WHEEL OLDER THAN THE SOURCE PROVES NOTHING ABOUT THE SOURCE. `dist/` is untracked build
    # output that nobody clears, so without this the suite fails for everybody who built a wheel
    # before the module moved, which is a test that gets deleted rather than one that gets heeded.
    # A current wheel is still the only certain check, so it is still made when there is one.
    newest = max(p.stat().st_mtime for p in PACKAGE.rglob("*.py"))
    if wheels[-1].stat().st_mtime < newest:
        pytest.skip(f"{wheels[-1].name} predates the source; rebuild to check the wheel itself")
    with zipfile.ZipFile(wheels[-1]) as z:
        names = z.namelist()
        assert not [n for n in names if "rented" in n], f"rented is in {wheels[-1].name}"
        for name in names:
            if not name.endswith(".py") or "/vendor/" in name:
                continue
            body = z.read(name).decode("utf-8", "replace")
            assert "rest.runpod.io" not in body, f"{name} in the wheel still names the provider"
