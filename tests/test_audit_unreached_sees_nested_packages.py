# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The unreached-names audit has to look inside the sub-packages it is meant to audit.

WHAT WAS WRONG. Both halves of the audit read the package with a non-recursive glob, so a module
inside a sub-package was invisible in two ways. A public name defined there was never listed, which
hides a dead function. A public name used only there was counted as referenced nowhere, which turns
a live function into a false positive. The workbench lives in `senbonzakura/workbench/`, so every
public name it calls from the top level read as unreached.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "research"))
import audit_unreached  # noqa: E402


def _package(tmp_path, **files):
    """A minimal checkout: `src/senbonzakura/` with the given files, and empty tests and tools."""
    src = tmp_path / "src" / "senbonzakura"
    for relative, text in files.items():
        path = src / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tools").mkdir()
    return tmp_path


def test_a_name_used_only_inside_a_sub_package_is_reached(tmp_path):
    root = _package(tmp_path, **{
        "__init__.py": "",
        "helper.py": "def helper_fn():\n    return 1\n",
        "sub/__init__.py": "",
        "sub/panel.py": "from senbonzakura.helper import helper_fn\n\n\n"
                        "def use():\n    return helper_fn()\n",
    })
    _, rows = audit_unreached.unreached(root)
    names = [name for name, _, _ in rows]
    assert "helper_fn" not in names, (
        "helper_fn is called from senbonzakura/sub/panel.py, a module the audit never opened, "
        "so it was reported as unreached when it is reached")


def test_a_public_name_defined_inside_a_sub_package_is_listed(tmp_path):
    root = _package(tmp_path, **{
        "__init__.py": "",
        "sub/__init__.py": "",
        "sub/orphan.py": "def nested_orphan():\n    return 2\n",
    })
    defined, rows = audit_unreached.unreached(root)
    assert "nested_orphan" in defined, (
        "a public function in a sub-package was never listed, so a dead one there was invisible")
    assert "nested_orphan" in [name for name, _, _ in rows], (
        "nothing calls nested_orphan, and the audit did not say so")
