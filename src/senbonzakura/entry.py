# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The front door, which must not cost four seconds to open.

WHAT WAS WRONG

`cli.py` imports torch, optuna and transformers at module level, and every command reached the
tool through it. So `senbonzakura --help` took 3.9 seconds and loaded a deep-learning stack to
print a page of text, and `senbonzakura doctor` could not run at all on a machine where torch was
missing. That is the exact machine `doctor` exists to diagnose: the command whose whole job is to
tell you an install is incomplete could not start on an incomplete install.

Measured on the development machine before the change:

    import senbonzakura        0.01s     (the package's lazy surface was already fine)
    senbonzakura --help        3.93s     torch, optuna and transformers all imported

WHY THE FIX IS A DISPATCHER RATHER THAN MOVED IMPORTS

Nothing at `cli.py`'s module scope uses torch or optuna; they are imported there only for the
function bodies below. They could therefore be pushed into those functions, but there are dozens
of sites and the result would be a large diff through the surgery code for a startup-time win.

Dispatching first is smaller and stronger. A delegated command never imports `cli` at all, so
`doctor`, `convert`, `quantise`, `imatrix` and `fetch` start in milliseconds and work with no
torch installed. Only abliteration, which genuinely cannot proceed without torch, pays for it.

WHAT THIS MODULE MAY IMPORT

Nothing heavy, ever. `banner` (os, random, shutil, unicodedata) and `_version` are the whole
budget. A future edit that adds `import torch` here silently restores the defect, so the test
suite asserts the cost rather than trusting the comment.
"""
from __future__ import annotations

import importlib
import sys

from ._version import __version__

#: command name -> (module, attribute). The command is what a person types, so it may carry a
#: hyphen; the module is a Python name and may not.
DELEGATED: dict[str, tuple[str, str]] = {
    "head-to-head": ("headtohead", "main"),
    "compass": ("margin", "main"),
    "score": ("score", "main"),
    "coherence": ("coherence", "main"),
    "drift": ("drift", "main"),
    "track": ("track", "main"),
    "validate": ("validate", "main"),
    "capability": ("capability", "main"),
    "report": ("modelcard", "main"),
    "judge": ("judge", "main"),
    "interactive": ("interactive", "run"),
    "quantise": ("quantise", "main"),
    "convert": ("convert", "main"),
    "doctor": ("doctor", "main"),
    "imatrix": ("imatrix", "main"),
    "fetch": ("fetch", "main"),
}


#: Which optional install brings each heavy dependency in, so a failure can say what to type.
#: Only the ones a partial install actually loses; anything absent from here gets the generic line.
_INSTALL_HINT = {
    "torch": "pip install 'senbonzakura[cuda]'   (or the CPU build: pip install torch)",
    "transformers": "pip install senbonzakura",
    "optuna": "pip install senbonzakura",
    "datasets": "pip install senbonzakura",
    "bitsandbytes": "pip install 'senbonzakura[quant]'",
    "shtab": "pip install 'senbonzakura[completion]'",
}


def _cannot_run(command, module_name, error):
    """Turn a failed import of a delegated module into a sentence about the install.

    The traceback this replaces named `margin.py` and the line number of its `import torch`, which
    describes OUR file to someone whose actual problem is that their environment is missing a
    dependency. It also arrived through eleven frames of importlib, so the one useful line was the
    last one.

    A missing package of ours is a different fault from a missing dependency and says so: the first
    means the install is damaged and should be reinstalled, the second means it is incomplete and
    names what to add.
    """
    missing = getattr(error, "name", None) or "a required package"
    if missing.split(".")[0] == __package__:
        return SystemExit(
            f"senbonzakura: cannot run '{command}': part of senbonzakura itself is missing "
            f"({missing}).\n"
            f"  The installation is damaged rather than incomplete. Reinstall it:\n"
            f"    pip install --force-reinstall senbonzakura\n"
            f"  Then run 'senbonzakura doctor' to confirm.")
    hint = _INSTALL_HINT.get(missing.split(".")[0], "pip install senbonzakura")
    return SystemExit(
        f"senbonzakura: cannot run '{command}': it needs {missing}, which is not installed.\n"
        f"  Install it with:\n"
        f"    {hint}\n"
        f"  'senbonzakura doctor' lists everything this install is missing in one go, and it "
        f"runs without any of it.")


def dispatch(name):
    """Import just the module that serves `name` and return its entry point."""
    module_name, attr = DELEGATED[name]
    try:
        module = importlib.import_module(f".{module_name}", __package__)
        return getattr(module, attr)
    except ImportError as e:
        raise _cannot_run(name, module_name, e) from e
    except AttributeError as e:
        # The module imported and does not carry its entry point, which no user action causes.
        # Named as a build fault rather than dressed up as a missing dependency.
        raise SystemExit(
            f"senbonzakura: cannot run '{name}': the '{module_name}' module is present but has no "
            f"'{attr}'. That is a packaging fault in this build, not something your environment "
            f"caused. Please report it with the output of 'senbonzakura doctor'.") from e


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Decoration only, and structurally unable to reach a result: it prints nothing unless stdout
    # is a terminal, so a redirected run, a spec's captured log and every `stdout-contains` check
    # see exactly what they saw before this existed.
    from . import banner
    banner.emit(__version__, sys.stdout)

    if argv and argv[0] in DELEGATED:
        return dispatch(argv[0])(argv[1:])

    # PARSE FIRST, and against the light parser. `--help`, `--version` and every argument error
    # are resolved here, before torch exists in this process: argparse raises SystemExit for all
    # three, so they never reach the import below. Someone who mistyped a flag finds out in a
    # hundredth of a second rather than after a deep-learning stack has loaded to tell them.
    from .parser import build_parser, split_mode
    bankai, rest = split_mode(argv)
    args = build_parser().parse_args(rest)

    # Only an actual abliteration pays for torch, which is the one case where the cost buys
    # something. One parse, handed straight in: parsing again inside `cli` would be two parsers
    # that have to agree forever.
    from .cli import run_parsed
    return run_parsed(args, bankai, rest)
