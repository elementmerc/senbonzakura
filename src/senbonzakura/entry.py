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
    "setup": ("envsetup", "main"),
    "imatrix": ("imatrix", "main"),
    "fetch": ("fetch", "main"),
    # THE ONE COMMAND THAT NEEDS NOTHING. It reads result files and does arithmetic: no torch,
    # no model, no corpus, no network. That is the whole point of it (roadmap, property 3), and
    # `tests/test_check_registry.py` walks its import graph to keep it true.
    "check": ("check.cli", "main"),
}


#: Which optional install brings each heavy dependency in, so a failure can say what to type.
#: Only the ones a partial install actually loses; anything absent gets the generic line.
#:
#: `torch` was advertised here as `senbonzakura[cuda]`, an extra that has never existed in any
#: version of this package, so the one line whose whole job is to tell somebody what to type
#: named something they could not type. A test now walks every hint against the metadata.
#: Since Q-27 the editor's dependencies are BASE dependencies, so missing one of them no longer
#: means "you skipped an extra"; it means the install is damaged or was made with `--no-deps`.
#: Telling that person to install an extra sends them to a command that changes nothing, which is
#: the same failure the note above records, one release later.
_INSTALL_HINT = {
    "torch": "pip install --force-reinstall senbonzakura"
             "   (then: senbonzakura setup, to get the build your hardware can use)",
    "transformers": "pip install --force-reinstall senbonzakura",
    "optuna": "pip install --force-reinstall senbonzakura",
    "accelerate": "pip install --force-reinstall senbonzakura",
    "gguf": "pip install --force-reinstall senbonzakura",
    "sentencepiece": "pip install --force-reinstall senbonzakura",
    "pyarrow": "pip install --force-reinstall senbonzakura",
    # Only the Hub reader and a couple of local shapes need it; local tracks do not.
    "datasets": "pip install 'senbonzakura[hub]'",
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


def exit_status(value):
    """What a command's return value means to a shell.

    THE DEFECT THIS FIXES, which is the same one twice from opposite ends. `__main__` used to
    discard what `main` returned, so `doctor` printed "this install cannot do what it claims"
    over nine failed checks and exited 0. That was fixed with `sys.exit(main())`, and the fix
    broke the other half of the surface: `score`, `compass`, `drift`, `coherence` and `track`
    return their RESULT rather than a status, and `sys.exit` on a non-integer prints it to
    stderr and exits 1. So every successful run of five commands reported failure, with a raw
    Python dict where an error message goes, and any script gating on one saw a corpus it had
    just built correctly written off.

    The two conventions both stay, because both are right where they are: a command whose
    caller wants the numbers returns the numbers, and a command whose whole job is a verdict
    returns the verdict. This is the one place that knows it is talking to a shell, so this is
    where the difference is resolved.

    `True` and `False` are refused rather than mapped. `sys.exit(True)` exits 1, which reads
    exactly backwards, and a command returning a bare boolean has not decided which convention
    it is following.
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        raise TypeError(
            f"a command returned {value!r}. Return an int for a status or the result object "
            f"for the numbers; a bool means neither and exits backwards.")
    if isinstance(value, int):
        # Clamped, because `sys.exit(256)` exits 0 on POSIX: the shell keeps the low byte, so a
        # status that overflows reports success. No command here returns one today, and a
        # verdict that silently inverts is not a thing to leave to nobody doing it later.
        return value if 0 <= value < 256 else 1
    # A result object: the command ran and produced something. Its own failures are raised.
    return 0


def module_entry(fn, argv=None):
    """`python -m senbonzakura.<module>` reports what the console script reports.

    Eight modules had no `__main__` guard at all, so running them that way executed nothing and
    exited 0. `doctor` is the one that matters: it exists to be run before a long job, it is
    invoked in this form in four places in the documentation, and it printed nothing and passed.
    That is the original defect `exit_status` was written for, still live on the documented path,
    found by a review pass that asked what the two invocation forms actually do rather than
    assuming they agree.
    """
    sys.exit(exit_status(fn(argv)))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Decoration only, and structurally unable to reach a result: it prints nothing unless stdout
    # is a terminal, so a redirected run, a spec's captured log and every `stdout-contains` check
    # see exactly what they saw before this existed.
    from . import banner
    banner.emit(__version__, sys.stdout)

    if argv and argv[0] in DELEGATED:
        name = argv[0]
        run = dispatch(name)
        try:
            return exit_status(run(argv[1:]))
        except ImportError as e:
            # `dispatch` only covers imports that happen while the module is being LOADED, and
            # several commands defer their heavy imports into `main` on purpose so that
            # `--help` stays fast. A missing dependency then escaped as an eleven-frame
            # importlib traceback ending at a line number inside one of our files, which
            # describes our code to somebody whose actual problem is their install. Found by
            # running the commands with the deep-learning stack made unimportable.
            raise _cannot_run(name, DELEGATED[name][0], e) from e

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
    # The refusal a 0.3.0 user meets after the dependency split, so it gets the same handler as
    # every other command rather than an eleven-frame traceback ending inside `cli.py`. Caught
    # by installing the built wheel into a clean environment and typing the command, which is
    # the only place the difference is visible: in a developer checkout the stack is always
    # there and this branch never runs.
    try:
        from .cli import run_parsed
    except ImportError as e:
        # `bankai` is a flag, not a name: naming the command after it printed "cannot run
        # 'True'". The word the user typed is the one they can act on.
        raise _cannot_run("kageyoshi" if bankai else "abliterate", "cli", e) from e
    return exit_status(run_parsed(args, bankai, rest))
