# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The documented first run is chosen by a marker, and a README that moves cannot change the choice.

WHAT PROMPTED IT, 2026-10-13

The documented-first-run CI job selected the install command by content: the one ```sh block that
contained `pip install senbonzakura`. A virtual environment route added a second block containing
`bin/pip install senbonzakura`, and the job stopped with "found 2". Stopping was right. The easy
repair, taking the first match, would have passed silently after a reorder and checked a different
route than the one the job names.

These tests pin the marker rule against the real documents and against fixtures that move, repeat
or drop the marker, so a future README change cannot change which commands run without a failure.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

HARNESS = pathlib.Path(__file__).resolve().parent.parent / "tools" / "ci" / "first_run_commands.py"


def _load():
    spec = importlib.util.spec_from_file_location("first_run_commands", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


first_run = _load()

INSTALL = "<!-- documented-first-run: install -->"
QUICK = "<!-- documented-first-run: quickstart -->"
VENV_ROUTE = "```sh\npython3 -m venv ~/senbonzakura\n~/senbonzakura/bin/pip install senbonzakura\n```\n"
INSTALL_BLOCK = "```sh\npip install senbonzakura\nsenbonzakura setup\n```\n"
QUICK_BLOCK = "```sh\nsenbonzakura corpora\nsenbonzakura track build --out corpus\n```\n"


def _tree(tmp_path, readme, quickstart=None):
    root = tmp_path / "repo"
    (root / "docs" / "guide").mkdir(parents=True)
    (root / "README.md").write_text(readme, encoding="utf-8")
    (root / "docs" / "guide" / "quickstart.md").write_text(
        f"{QUICK}\n{QUICK_BLOCK}" if quickstart is None else quickstart, encoding="utf-8")
    return root


def test_the_real_documents_select_the_marked_install_and_quickstart(tmp_path):
    install, quickstart = first_run.build(first_run.ROOT, tmp_path / "out")
    assert "pip install ./checker ." in install
    assert "senbonzakura setup" in install
    assert "bin/pip" not in install, "the venv route is a different path and must not be run"
    assert "senbonzakura track build" in quickstart
    assert (tmp_path / "out" / "install.sh").read_text(encoding="utf-8") == install


def test_the_marker_decides_which_block_runs_not_its_position(tmp_path):
    """The venv route comes first here. A first-match rule would run it, and this must not."""
    root = _tree(tmp_path, VENV_ROUTE + "\n" + INSTALL + "\n" + INSTALL_BLOCK)
    install, _ = first_run.build(root, tmp_path / "out")
    assert "senbonzakura setup" in install
    assert "bin/pip" not in install


def test_a_missing_marker_refuses_rather_than_guessing(tmp_path):
    root = _tree(tmp_path, INSTALL_BLOCK + "\n" + VENV_ROUTE)
    with pytest.raises(first_run.FirstRunError, match=r"has 0 .* markers and needs exactly one"):
        first_run.build(root, tmp_path / "out")
    assert not (tmp_path / "out").exists(), "nothing is written when the selection is refused"


def test_a_repeated_marker_refuses_rather_than_guessing(tmp_path):
    root = _tree(tmp_path, INSTALL + "\n" + INSTALL_BLOCK + "\n" + INSTALL + "\n" + VENV_ROUTE)
    with pytest.raises(first_run.FirstRunError, match=r"has 2 .* markers"):
        first_run.build(root, tmp_path / "out")


def test_a_marker_not_directly_above_a_block_refuses(tmp_path):
    """A paragraph between the marker and the fence means the marker names nothing."""
    root = _tree(tmp_path, INSTALL + "\n\nSome prose that a reader would see.\n\n" + INSTALL_BLOCK)
    with pytest.raises(first_run.FirstRunError, match=r"not directly followed by a ```sh block"):
        first_run.build(root, tmp_path / "out")


def test_a_marked_block_the_substitution_cannot_reach_refuses(tmp_path):
    root = _tree(tmp_path, INSTALL + "\n```sh\nuv tool install senbonzakura\n```\n")
    with pytest.raises(first_run.FirstRunError, match=r"no longer contains 'pip install senbonzakura'"):
        first_run.build(root, tmp_path / "out")


def test_a_quickstart_marker_is_checked_as_strictly_as_the_install(tmp_path):
    quickstart = f"{QUICK}\n{QUICK_BLOCK}\n{QUICK}\n{QUICK_BLOCK}"
    root = _tree(tmp_path, INSTALL + "\n" + INSTALL_BLOCK, quickstart=quickstart)
    with pytest.raises(first_run.FirstRunError, match=r"docs/guide/quickstart.md has 2"):
        first_run.build(root, tmp_path / "out")


def test_the_cli_exits_nonzero_and_says_why(tmp_path, capsys):
    root = _tree(tmp_path, INSTALL_BLOCK)
    assert first_run.main(["--root", str(root), "--out", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("first-run: "), err
    assert "documented-first-run: install" in err


def test_the_cli_prints_what_it_will_run_on_success(tmp_path, capsys):
    assert first_run.main(["--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "--- install ---" in out and "--- quickstart ---" in out
    assert "pip install ./checker ." in out
