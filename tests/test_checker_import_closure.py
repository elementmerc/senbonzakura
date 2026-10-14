# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The checker's import closure is the standard library and the checker itself, and nothing else.

WHY THIS IS NOT A WIDER BLOCKLIST

`test_torch_free.py` blocks a fixed list of names, and its bare fixture blocks the names the big
distribution declares. Both are enumerations, and an enumeration has a hole exactly the size of
the things nobody thought to list. A planted `import numpy` in the checker passes both: numpy is on
neither list, and on any machine where the big distribution is installed it imports successfully,
so the defect is invisible until somebody installs `senbonzakura-check` on its own and the checker
falls over. This file states the claim the distribution actually makes, `dependencies = []`, as an
allowlist: anything outside the standard library and `senbonzakura_check` is refused.

Two halves, because each misses something the other catches. The static scan reads every import in
every file, including ones inside a function body that only run on some command, so it cannot be
dodged by reachability. The runtime half imports every module under the allowlist and runs the CLI,
which catches anything reached by name at run time, such as a string handed to `importlib`.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tomlread import tomllib

ROOT = Path(__file__).resolve().parent.parent
CHECKER_SRC = ROOT / "checker" / "src"
CHECKER_PACKAGE = CHECKER_SRC / "senbonzakura_check"
CHECKER_PYPROJECT = ROOT / "checker" / "pyproject.toml"
RECOVERED_FINDING = ROOT / "tests" / "fixtures" / "recovered" / (
    "2026-07-30-compass-qwen3-1.7b-as-first-published.json")

STDLIB = frozenset(sys.stdlib_module_names)
OWN_NAME = "senbonzakura_check"


def _offending_imports(source, filename="<snippet>"):
    """Every absolute import in `source` whose top-level package is neither stdlib nor the checker.

    Relative imports are the checker importing itself and are always allowed. The walk visits every
    node, so an import inside a function body or a conditional is found the same as one at the top.
    """
    tree = ast.parse(source, filename=filename)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.partition(".")[0]
            if top and top not in STDLIB and top != OWN_NAME:
                offenders.append((node.lineno, name))
    return offenders


def test_the_checker_declares_no_runtime_dependencies():
    """The claim the distribution makes, read from the file that makes it."""
    data = tomllib.loads(CHECKER_PYPROJECT.read_text(encoding="utf-8"))
    assert data["project"]["dependencies"] == [], (
        "senbonzakura-check declares a runtime dependency. Adding one is a decision that costs the "
        "property this distribution exists to hold, and the v0.8 plan says a check that needs one "
        "does not ship.")


def test_every_import_in_the_checker_source_is_stdlib_or_the_checker_itself():
    """The static half. Reads every file, so reachability cannot hide an import."""
    found = {}
    for path in sorted(CHECKER_PACKAGE.rglob("*.py")):
        offenders = _offending_imports(path.read_text(encoding="utf-8"), str(path))
        if offenders:
            found[str(path.relative_to(ROOT))] = offenders
    assert not found, (
        "the checker imports something outside the standard library and itself, which breaks "
        f"`dependencies = []`: {found}")


def test_the_scanner_refuses_a_third_party_import_hidden_in_a_function():
    """The scanner is tested before its silence is trusted.

    A scan that reports nothing on a file it could not parse, or that ignores imports inside a
    function body, reads as a clean checker. Each case below is one way that would happen.
    """
    hidden = "def load():\n    import numpy\n    return numpy\n"
    assert _offending_imports(hidden) == [(2, "numpy")]

    dotted = "from yaml.loader import SafeLoader\n"
    assert _offending_imports(dotted) == [(1, "yaml.loader")]

    upward = "import senbonzakura\n"
    assert _offending_imports(upward) == [(1, "senbonzakura")], (
        "an upward import from the checker into the big distribution would put torch back "
        "into the torch-free one")


def test_the_scanner_stays_quiet_on_what_the_checker_is_allowed_to_do():
    """The other half of the control. A scanner that refuses everything proves nothing."""
    allowed = (
        "import json\n"
        "from __future__ import annotations\n"
        "from pathlib import Path\n"
        "import xml.etree.ElementTree\n"
        "from . import registry\n"
        "from .registry import load_checks\n"
        "from senbonzakura_check.cli import main\n"
    )
    assert _offending_imports(allowed) == []


def test_the_scanner_reports_a_file_it_cannot_parse_rather_than_passing_it():
    """A file that fails to parse must not read as a clean one. The syntax error is the finding."""
    with pytest.raises(SyntaxError):
        _offending_imports("def broken(:\n")


_PROBE = r"""
import importlib.abc
import pkgutil
import sys

ALLOWED = frozenset(sys.stdlib_module_names) | {"senbonzakura_check"}


class _Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] not in ALLOWED:
            raise ImportError(f"refused by the stdlib-only gate: {name}", name=name)
        return None


sys.meta_path.insert(0, _Refuse())

import senbonzakura_check

if not senbonzakura_check.__file__.startswith(sys.argv[1]):
    raise SystemExit(f"loaded the wrong checker: {senbonzakura_check.__file__}")
for info in pkgutil.walk_packages(senbonzakura_check.__path__, "senbonzakura_check."):
    __import__(info.name)

from senbonzakura_check.cli import main

raise SystemExit(main([sys.argv[2]]))
"""


def test_the_checker_imports_and_runs_with_nothing_but_the_standard_library(tmp_path):
    """The runtime half, and the claim a stranger's install depends on.

    The probe imports every module of the checker under the allowlist and then runs the CLI over a
    real defective artefact. It must reach a finding (exit 1), not a traceback, and it must be the
    checker in this checkout that answered, which the probe asserts before anything else runs.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CHECKER_SRC)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(CHECKER_SRC), str(RECOVERED_FINDING)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=300, check=False)
    assert "Traceback" not in proc.stderr, proc.stderr
    assert proc.returncode == 1, (
        f"the defective artefact should produce a finding (exit 1) with nothing but the standard "
        f"library, got {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    assert "a-verdict-that-answers-the-same-to-every-item" in proc.stdout
