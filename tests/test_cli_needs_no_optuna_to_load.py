# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The command module loads without optuna, because only the search uses it.

A top-level `import optuna` made every command need it, so `score` refused to start on a rented
pod whose environment had no optuna, and a whole measurement run lost its refusal score.
"""
import subprocess
import sys

BLOCK = (
    "import importlib.abc, sys\n"
    "class Block(importlib.abc.MetaPathFinder):\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'optuna' or name.startswith('optuna.'):\n"
    "            raise ImportError('optuna is blocked for this test')\n"
    "        return None\n"
    "sys.meta_path.insert(0, Block())\n"
)


def test_the_command_module_imports_with_optuna_unavailable():
    code = BLOCK + "import senbonzakura.cli as c\nprint('loaded', hasattr(c, 'main'))\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "loaded True" in result.stdout


def test_the_search_still_asks_for_optuna_when_it_runs():
    import inspect

    from senbonzakura import cli
    source = inspect.getsource(cli.Abliterator._run_search)
    assert "import optuna" in source
