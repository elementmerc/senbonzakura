# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The front door must not import a deep-learning stack to print a page of text.

`cli.py` imports torch, optuna and transformers at module level. Every command used to reach the
tool through it, so `senbonzakura --help` cost 2.8 seconds and `doctor` — whose entire job is to
tell you an install is incomplete — could not run at all on a machine missing torch. That is the
exact machine it exists to diagnose.

`entry.py` dispatches before importing `cli`, and `parser.py` holds the argument surface, which
uses nothing heavy. These tests hold that arrangement down. They assert the IMPORT GRAPH rather
than a wall-clock number, because a timing threshold on a shared machine is a flaky test and this
project does not keep those.
"""
import subprocess
import sys

import pytest

LIGHT = ("senbonzakura.entry", "senbonzakura.parser", "senbonzakura._version",
         # `parser` reads its flag choices from `separation`, so `separation` inherits the ban.
         "senbonzakura.separation")
HEAVY = ("torch", "optuna", "transformers")


def _imports_after(code):
    """Which heavy modules are in sys.modules after running `code` in a fresh interpreter."""
    probe = (code + "\nimport sys, json\n"
             + f"print(json.dumps({{m: (m in sys.modules) for m in {HEAVY!r}}}), file=sys.stderr)")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 0, r.stderr[-500:]
    import json
    return json.loads(r.stderr.strip().splitlines()[-1])


@pytest.mark.parametrize("module", LIGHT)
def test_the_light_modules_import_nothing_heavy(module):
    """The whole arrangement rests on these three costing nothing."""
    got = _imports_after(f"import {module}")
    assert not any(got.values()), f"{module} pulled in {[k for k, v in got.items() if v]}"


def test_help_does_not_import_torch():
    """`--help` is a page of text. It used to cost 2.8 seconds and a gigabyte of resident memory."""
    got = _imports_after(
        "import sys\n"
        "sys.argv = ['senbonzakura', '--help']\n"
        "from senbonzakura import entry\n"
        "try:\n"
        "    entry.main()\n"
        "except SystemExit:\n"
        "    pass\n"
    )
    assert not any(got.values()), f"--help pulled in {[k for k, v in got.items() if v]}"


def test_an_argument_error_does_not_import_torch():
    """A mistyped flag should be reported before a deep-learning stack loads to report it."""
    got = _imports_after(
        "import sys, os\n"
        "sys.argv = ['senbonzakura', '--not-a-real-flag']\n"
        "sys.stderr = open(os.devnull, 'w')\n"
        "from senbonzakura import entry\n"
        "try:\n"
        "    entry.main()\n"
        "except SystemExit:\n"
        "    pass\n"
        "sys.stderr = sys.__stderr__\n"
    )
    assert not any(got.values()), f"an argument error pulled in {[k for k, v in got.items() if v]}"


def test_a_delegated_command_does_not_import_cli():
    """`doctor` reaching the tool through `cli` is what made it unable to diagnose a missing torch.

    It may import torch itself, because reporting on torch is one of its checks; what it must not
    do is import `cli`, whose module-level imports are unconditional.
    """
    r = subprocess.run(
        [sys.executable, "-c", ("import sys;"
                                "from senbonzakura import entry;"
                                "entry.dispatch('doctor');"
                                "print('senbonzakura.cli' in sys.modules)")],
        capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip() == "False", "dispatching doctor imported cli, which imports torch"


def test_every_delegated_command_resolves():
    """A dispatch table that names a module it cannot import is a command that 404s at runtime."""
    from senbonzakura import entry
    for name in entry.DELEGATED:
        assert callable(entry.dispatch(name)), f"{name} does not resolve to something callable"


def test_the_parser_is_the_same_one_cli_uses():
    """Two parsers that have to agree forever is the failure this split could have introduced."""
    from senbonzakura import cli, parser
    assert cli.build_parser is parser.build_parser
    assert cli.loader_parser is parser.loader_parser
