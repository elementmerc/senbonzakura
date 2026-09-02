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


# ── a partial install gets a sentence, not a traceback ──────────────────────────────
def _dispatch_with(monkeypatch, exc):
    """Make importing the delegated module raise `exc`, and return what dispatch does."""
    from senbonzakura import entry
    monkeypatch.setattr(entry.importlib, "import_module",
                        lambda *a, **k: (_ for _ in ()).throw(exc))
    with pytest.raises(SystemExit) as caught:
        entry.dispatch("compass")
    return str(caught.value)


def test_a_missing_dependency_names_itself_and_what_to_type(monkeypatch):
    """The traceback this replaces pointed at margin.py's `import torch`.

    That describes OUR file to someone whose problem is their environment, and it arrived through
    eleven frames of importlib with the one useful line last.
    """
    message = _dispatch_with(monkeypatch,
                             ModuleNotFoundError("No module named 'torch'", name="torch"))
    assert "cannot run 'compass'" in message
    assert "torch" in message
    assert "pip install" in message
    assert "doctor" in message, "the one command that works without the missing piece"
    assert "Traceback" not in message


def test_a_missing_piece_of_our_own_package_is_a_different_fault(monkeypatch):
    """A damaged install and an incomplete one need different advice, so they get it."""
    message = _dispatch_with(monkeypatch,
                             ModuleNotFoundError("nope", name="senbonzakura.margin"))
    assert "damaged rather than incomplete" in message
    assert "force-reinstall" in message


def test_an_unknown_dependency_still_gets_a_usable_line(monkeypatch):
    # No hint recorded for this one, so it must fall back rather than produce "None".
    message = _dispatch_with(monkeypatch, ImportError("boom", name="some_new_dep"))
    assert "some_new_dep" in message
    assert "pip install senbonzakura" in message
    assert "None" not in message


def test_an_import_error_carrying_no_name_does_not_say_none(monkeypatch):
    message = _dispatch_with(monkeypatch, ImportError("something went wrong"))
    assert "a required package" in message
    assert "None" not in message


def test_a_module_without_its_entry_point_is_named_a_build_fault(monkeypatch):
    """No user action causes this, so it must not be dressed up as a missing dependency."""
    from senbonzakura import entry
    monkeypatch.setattr(entry.importlib, "import_module", lambda *a, **k: object())
    with pytest.raises(SystemExit) as caught:
        entry.dispatch("compass")
    assert "packaging fault in this build" in str(caught.value)
    assert "pip install" not in str(caught.value), "telling them to reinstall would not help"


def test_every_delegated_command_survives_a_missing_dependency(monkeypatch):
    """All of them, not just the one that happened to be tested."""
    from senbonzakura import entry
    monkeypatch.setattr(entry.importlib, "import_module",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ModuleNotFoundError("no", name="torch")))
    for command in entry.DELEGATED:
        with pytest.raises(SystemExit) as caught:
            entry.dispatch(command)
        assert f"cannot run '{command}'" in str(caught.value)
