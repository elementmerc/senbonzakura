# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The declared floors and the file that tests them have to agree.

WHY THIS EXISTS

`constraints/floors.txt` opens with the instruction "when a floor in pyproject changes,
change it here in the same commit. A mismatch means the job is testing a version nothing
declares." That instruction has been enforced by somebody remembering it, which is the shape
of gate this project keeps watching fail: a rule written in a comment beside the thing it
governs, with nothing that fires when it is broken.

The failure it prevents is quiet rather than loud. If pyproject says `>=14` and floors.txt
pins `==8`, the `dependency-floor` CI job goes green having installed and tested a version
nobody is allowed to have; if floors.txt has no pin at all, the job resolves to the newest
release and tests the ceiling while its name says floor. Neither shows up as a failure. Both
mean the floor claim in the package metadata is decoration.

Written the day `datasets>=2.15` was replaced by `pyarrow>=14`, which is exactly the kind of
edit that gets made in pyproject and forgotten in the constraints file.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
FLOORS = ROOT / "constraints" / "floors.txt"

#: Dependencies deliberately absent from floors.txt, each with the reason it is absent.
#: An entry here is a decision on the record, not a way to quieten the test: adding one
#: without a reason is the failure this file exists to make visible.
EXEMPT = {
    # Documented in floors.txt itself: torch's CPU wheels are not on PyPI and its floor
    # varies by Python version, so the CI job installs it by hand from the torch index.
    "torch": "installed from the torch index by the job, floor varies by Python version",
    # Documented in floors.txt: the `quant` extra is not installed by `[dev]`, so the
    # dependency-floor job never has it and a pin here would constrain nothing.
    "bitsandbytes": "the quant extra is not installed by the floors job",
    # Not pinned, and that is a gap rather than a decision. Recorded in DEFERRED.md.
    "gguf": "no floor pin yet; see DEFERRED.md",
    "sentencepiece": "no floor pin yet; see DEFERRED.md",
}


def _release(version: str) -> tuple[int, ...]:
    """A version as a comparable release tuple, so `>=7` and `==7.0.0` are the same version.

    PEP 440 says they are, and comparing the strings says they are not. Padding with zeros is
    what makes this gate assert equality of VERSIONS rather than of typing habits, which is
    the difference between a check that binds and one that everybody learns to ignore.
    """
    parts = re.split(r"[.+!-]", version.split("*", maxsplit=1)[0].strip(".")) if version else []
    return tuple(int(p) for p in parts if p.isdigit()) or (0,)


def _same_version(a: str, b: str) -> bool:
    ra, rb = _release(a), _release(b)
    width = max(len(ra), len(rb))
    return ra + (0,) * (width - len(ra)) == rb + (0,) * (width - len(rb))


def _requirements() -> dict[str, str]:
    """Every direct dependency with a `>=` floor, as name to floor, across deps and extras."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    specs = list(project.get("dependencies") or [])
    for extra in (project.get("optional-dependencies") or {}).values():
        specs.extend(extra)
    out = {}
    for spec in specs:
        m = re.match(r"^([A-Za-z0-9._-]+)\s*>=\s*([0-9][0-9A-Za-z.*+!-]*)", spec.split(";")[0].strip())
        if m:
            out.setdefault(m.group(1).lower().replace("_", "-"), m.group(2))
    return out


def _pins() -> dict[str, str]:
    """Every `==` pin in floors.txt, as name to version."""
    out = {}
    for raw in FLOORS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9._-]+)\s*==\s*(.+)$", raw.split("#", maxsplit=1)[0].strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2).strip()
    return out


def test_every_declared_floor_is_pinned_and_tested():
    """A floor with no pin means the CI job resolves to the newest release and proves nothing."""
    missing = sorted(set(_requirements()) - set(_pins()) - set(EXEMPT))
    assert not missing, (
        f"{missing} declare a floor in pyproject.toml and have no `==` pin in "
        f"constraints/floors.txt, so the dependency-floor job installs the newest release and "
        f"tests the ceiling under the name of the floor. Pin them, or add them to EXEMPT here "
        f"with the reason.")


def test_every_pin_matches_the_floor_it_claims_to_test():
    """The louder half: a pin that disagrees tests a version the package forbids."""
    reqs, pins = _requirements(), _pins()
    wrong = {name: (floor, pins[name]) for name, floor in reqs.items()
             if name in pins and not _same_version(pins[name], floor)}
    assert not wrong, (
        "constraints/floors.txt pins versions that are not the declared floors: "
        + "; ".join(f"{n}: pyproject says >={f}, floors.txt pins =={p}" for n, (f, p) in sorted(wrong.items())))


def test_nothing_is_pinned_that_nothing_declares():
    """A pin for a dependency that no longer exists holds an old version in place silently.

    `datasets` is the case that made this worth asserting: it stopped being a runtime
    dependency and stayed a `dev` one, so its pin is still correct. Had it left entirely,
    the pin would have quietly constrained a transitive dependency instead.
    """
    stray = sorted(set(_pins()) - set(_requirements()))
    assert not stray, (
        f"constraints/floors.txt pins {stray}, which pyproject.toml no longer declares with a "
        f"floor. Remove the pin, or restore the declaration.")


@pytest.mark.parametrize("name", sorted(EXEMPT))
def test_each_exemption_is_still_a_real_dependency(name):
    """An exemption for something we no longer depend on is dead text that hides the next one."""
    assert name in _requirements(), (
        f"{name} is exempt from the floor gate and is no longer a declared dependency. "
        f"Remove it from EXEMPT.")
