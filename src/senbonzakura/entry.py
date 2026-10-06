# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
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
    # Single-turn jailbreak success on a bundled attack set, PAIRED with the over-refusal rate on
    # a benign one. Registered as its own command rather than a flag on `score` because the thing
    # it adds is that both arms run together: a flag could be left off, and a resistance figure
    # with the benign arm left off is the narrower answer this project keeps catching itself
    # reporting as the wider one.
    "jailbreak": ("jailbreak", "main"),
    # The same question over several turns, which is a different property: a refusal that holds
    # once and gives way when the request is simply put again costs an attacker one keystroke.
    # The command word carries a hyphen and the module may not.
    "multi-turn": ("multiturn", "main"),
    # Whether the edit survives a safety-recovery finetune, with the two controls that decide
    # whether the number is about safety at all. The only command here that takes a gradient.
    "tamper": ("tamper", "main"),
    "coherence": ("coherence", "main"),
    "drift": ("drift", "main"),
    "track": ("track", "main"),
    # Fetches and packs the refusal corpora. A release wheel already carries them, so this is
    # for an install made straight from the repository, where they are generated rather than
    # committed. It lived under `tools/`, which ships in no wheel, so exactly the people who
    # needed it were the people who did not have it.
    "corpora": ("corporabuild", "main"),
    # Runs the five instruments against one model and prints one table. It measures nothing
    # itself: each stage is the command that already owns that number, given the command line a
    # person would have typed, so there is no second implementation to drift.
    "measure": ("measure", "main"),
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
    "check": ("check", "main"),
    # THE SECOND COMMAND THAT NEEDS NOTHING, and for the same reason as `check`. A gate whose
    # whole value is that it runs on every change has to be affordable on whatever a CI runner
    # has, so it reads two JSON files and does arithmetic.
    "gate": ("gate", "main"),
    # THE THIRD, and it is what gives the second an input. `gate` was registered and documented
    # while nothing in this repository could write a file it would accept, so the regression gate
    # applied to no property this tool measures. Same weight class as the other two: it reads one
    # JSON file and writes another.
    "baseline": ("baseline", "main"),
    # THE FOURTH THAT NEEDS NOTHING, and it was shipping unreachable. `prereg.py` implements this
    # project's pre-registration format, with 32 tests, and until 2026-09-27 nothing imported it:
    # a complete library, inside the wheel, that no user could reach and no command could call.
    #
    # It is registered here rather than left as a library because the format's whole argument is
    # that a pre-registration should be checkable by somebody other than its author. A checker
    # nobody can run is the same claim made by assertion, which is what the format exists to
    # replace.
    "prereg": ("prereg", "main"),
    # THE FIFTH THAT NEEDS NOTHING, and the one that answers a question people ask before they
    # install anything: will this model fit on my machine, and how long would it take? It reads
    # the checkpoint's safetensors headers, measures the card, the memory, the disk and whether
    # the laptop is plugged in, and refuses a run that cannot finish. Headers only, so it costs
    # about a second on a 61 GB model and never holds a weight.
    "budget": ("streaming", "main"),
}


#: Plain-English second names for commands whose first name says nothing to somebody who has not
#: read the project. `auto` for `kageyoshi` was the first of these and is handled in `split_mode`,
#: because that one is a mode rather than a delegated command.
#:
#: ALIASES RATHER THAN RENAMES, deliberately. Every run spec on record, every documented example
#: and every script anybody has written uses the existing names, and a tool that renames its own
#: commands breaks the record of what was already run. So the old name stays canonical, is what
#: every artefact records, and the alias is a door rather than a replacement.
#:
#: Each entry earns its place by being a word somebody would guess. A second name for a command
#: that was already plain is surface without a reader, and this list is short on purpose.
ALIASES = {
    # `compass` is a metaphor that has to be explained before it means anything. What the command
    # answers is whether the model still tells a harmful request from a harmless one.
    "harm-recognition": "compass",
}

#: Words that were commands here once, or that this project's own documents have used as one, and
#: what they are now. Kept rather than deleted, because a renamed command does not stop being typed
#: the moment it is renamed: it lives on in run specs, in shell history, and in prose.
#:
#: `bench` is the reason this exists. `REPRODUCING.md` told readers to run `senbonzakura bench
#: --help` for months. There is no such command, so `bench` was taken as the model positional,
#: argparse fell through to the abliterator, and it printed a plausible help screen and EXITED 0.
#: The reader checking the benchmark concluded the harness was not shipped. Worse without `--help`:
#: `senbonzakura bench` would have gone to the Hub for a model called `bench` and started editing
#: weights. Found twice, independently, by the 2026-09-25 panel.
RETIRED = {
    "bench": "head-to-head",
    "benchmark": "head-to-head",
    "harm_recognition": "compass",
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
    # Only `tamper --method lora` needs it; `--method full` trains without it.
    "peft": "pip install 'senbonzakura[finetune]'",
}


def _is_installed(name):
    """Is this top-level package present on this interpreter's path?

    `find_spec` LOCATES a module without executing it, which is exactly the distinction needed
    here: a package whose `__init__` raises still has a spec, so "present but broken" and "absent"
    stop looking the same. A broken parent can make the lookup itself raise, and that is answered
    as "cannot tell" rather than allowed to replace one bad message with a crash.
    """
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _cannot_run(command, module_name, error):
    """Turn a failed import of a delegated module into a sentence about the install.

    The traceback this replaces named `margin.py` and the line number of its `import torch`, which
    describes OUR file to someone whose actual problem is that their environment is missing a
    dependency. It also arrived through eleven frames of importlib, so the one useful line was the
    last one.

    A missing package of ours is a different fault from a missing dependency and says so: the first
    means the install is damaged and should be reinstalled, the second means it is incomplete and
    names what to add.

    AND A PRESENT PACKAGE IS A THIRD FAULT, which this asserted its way past until 2026-09-28.
    `error.name` is the package the failed import was reaching INTO, not a package that is absent:

        ImportError: cannot import name 'Cache' from 'transformers'   ->   name == "transformers"

    on an install where transformers is right there. The tool then told the operator it was not
    installed and prescribed an install that changes nothing, over a version mismatch that the
    same pin reproduces exactly.

    TWO SIGNALS DECIDE IT, and they have to agree before absence is claimed:

      * the exception TYPE. `ModuleNotFoundError` is what the import machinery raises when it
        could not find a module; a plain `ImportError` means the module was found, ran, and did
        not hand over what was asked for. Only the first is evidence of absence.
      * `find_spec`, which LOCATES a package without executing it, so a package whose `__init__`
        raises still has a spec. That is the measurement, rather than the assumption.

    An error carrying no package name at all (an `OSError` from a library that will not load,
    which is what `doctor` has always handled beside `ImportError`) is reported as what it is
    rather than dressed up as a missing dependency, because nothing here knows what to install.
    """
    missing = getattr(error, "name", None) or ""
    top = missing.split(".")[0]
    found_it = bool(top) and not isinstance(error, ModuleNotFoundError) and _is_installed(top)
    if found_it and top != __package__:
        return SystemExit(
            f"senbonzakura: cannot run '{command}': {top} is installed here, and importing it "
            f"failed anyway.\n"
            f"    {type(error).__name__}: {error}\n"
            f"  The package is PRESENT, so installing it again will not help on its own. The "
            f"usual causes are a version that does not match what senbonzakura expects and an "
            f"upgrade that did not finish.\n"
            f"  'senbonzakura doctor' reports what this install can and cannot do, and it runs "
            f"without any of this.")
    if not top:
        return SystemExit(
            f"senbonzakura: cannot run '{command}': loading the '{module_name}' module failed, and "
            f"the error names no package, so nothing here can say which one is missing.\n"
            f"    {type(error).__name__}: {error}\n"
            f"  'senbonzakura doctor' reports what this install can and cannot do, and it runs "
            f"without any of this. If it reports a required package absent, that is the one to "
            f"install.")
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
    # OSError beside ImportError, which is what `doctor` has always done for this class: a
    # dependency that is present and will not LOAD raises OSError (a DLL that will not map, a
    # shared object built for another CUDA), and that escaped here as a raw traceback while the
    # same fault one module away got a sentence.
    except (ImportError, OSError) as e:
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
    #
    # EXCEPT WHEN THE RESULT SAYS ITS OWN NUMBER IS NOT A MEASUREMENT, which is a third case and
    # neither of the two above. A command can finish cleanly, write a complete artefact, and know
    # that the figure inside it means nothing: the compass does exactly this when the model's
    # verdict was not at the position being scored, and prints THE AUC ABOVE IS NOT A MEASUREMENT
    # OF HARM DISCRIMINATION in those words.
    #
    # `python -m senbonzakura.margin` already exited 1 on that condition and `senbonzakura compass`
    # did not, because the check lived in one module's `__main__` guard rather than here. A novice
    # in the 2026-09-26 user pass came through the console command, was told in capitals that the
    # number was invalid, got exit 0, and said the run "told me in capitals that its own number is
    # meaningless, and exited 0". Two doors into one command disagreeing about whether it succeeded
    # is this project's most-repeated defect shape, and `exit_status`'s own docstring says this is
    # the one place that knows it is talking to a shell. So the rule lives here and covers every
    # command, including the ones that grow this condition later.
    #
    # A GENERIC MARKER RATHER THAN A CHECK FOR THE COMPASS. `exit_status` must not learn what a
    # readout is; any command that can invalidate its own figure sets `self_invalidated` on the
    # result and gets the behaviour. Status 1, not 2: the run happened and the artefact is real,
    # which is a different thing from a refusal.
    if isinstance(value, dict) and value.get("self_invalidated"):
        return 1
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


def _not_a_command(word):
    """A refusal when the first word is a command somebody meant, or None to carry on.

    THE FALLTHROUGH THIS CLOSES. The model is a positional, so ANY unrecognised first word is a
    valid model id as far as argparse is concerned. `senbonzakura bench --help` therefore printed
    the abliterator's help and exited 0, and `senbonzakura bnech Qwen/Qwen3-1.7B` would go to the
    Hub for a repo called `bnech`. A tool that answers a typo with a plausible screen teaches its
    reader that the documentation is unreliable, which is what happened.

    DELIBERATELY NARROW, because a bare word is a legitimate model id: `gpt2` has no slash and is
    not a path. Refusing every unknown bare word would break real invocations to catch typos. So
    this fires on exactly two things it can be sure about:

      - a word this project has itself used as a command and no longer has (`RETIRED`);
      - a word that is a near miss for a command that exists, when it cannot be a model reference.

    A model id or a local path is never either: a Hub id carries a slash, and a local checkpoint is
    a path that exists. Both are checked before the near-miss test, and the refusal still names the
    way through for the one person who really does have a model called `scoer`.
    """
    import difflib
    import os

    if not word or word.startswith("-"):
        return None
    from .parser import MODES

    # MODES, not the literal "abliterate". Beyond the same drift the notice below had, `known` is
    # also what a typo is matched against, so leaving `kageyoshi` and `auto` out meant the near miss
    # suggester could never propose the two words `--help` presents as the way to run this: a reader
    # who typed `kageyosi` got no suggestion at all and a fetch attempt for a model of that name.
    known = sorted({*DELEGATED, *ALIASES, *MODES})
    if word in known:
        return None
    if word in RETIRED:
        return (f"senbonzakura: `{word}` was renamed to `{RETIRED[word]}`.\n"
                f"  Run `senbonzakura {RETIRED[word]}` instead.\n"
                f"  It is refused rather than ignored because the model is a positional argument, "
                f"so `{word}` would otherwise be read as a model to edit.")
    # A Hub id or a path on disk is a model, never a mistyped command.
    if "/" in word or os.path.exists(word):
        return None
    near = difflib.get_close_matches(word, known, n=3, cutoff=0.8)
    if not near:
        return None
    return (f"senbonzakura: `{word}` is not a command. Did you mean "
            f"{' or '.join('`' + n + '`' for n in near)}?\n"
            f"  If you meant a model called `{word}`, pass it as `--model {word}`, which cannot "
            f"be mistaken for a command.")


def _read_as_a_model(word):
    """A notice that a bare word is being taken as a model id, or None when nothing needs saying.

    WHY THIS IS A NOTICE AND NOT A REFUSAL, 2026-09-27

    `_not_a_command` above fires on a near miss and on a retired name, and deliberately lets every
    other bare word through, because a Hub id is a positional and refusing unknown words to catch
    typos would break real invocations. That reasoning is sound and its consequence was not:
    `senbonzakura frobnicate` was SILENTLY read as `--model frobnicate` and then failed on something
    unrelated, so a surface audit saw it exit 1 complaining about CUDA. On a machine with a card it
    goes on to try to fetch a model called `frobnicate`.

    Refusing it was considered and rejected: canonical Hub ids with no organisation are real, so
    `senbonzakura gpt2` and `senbonzakura bert-base-uncased` must keep working, and a rule about
    slashes would break them. The fix is therefore to say out loud what interpretation was chosen,
    before anything expensive happens, so the one word of feedback the user needed is there.

    The operator chose this over refusing, on 2026-09-27.
    """
    import os

    from .parser import MODES

    if not word or word.startswith("-"):
        return None
    # MODES RATHER THAN THE LITERAL "abliterate", which is what this line held until 2026-09-28 and
    # is why two of the three mode words were wrong. `senbonzakura kageyoshi <model>` is what
    # `--help` calls the recommended way to run this, and every such invocation printed "reading
    # `kageyoshi` as a model id, since it is not a command" to stderr, which is false: `split_mode`
    # peels it off as a mode word a few lines later. On `kageyoshi --help` it was worse, because the
    # next thing printed is a help page rather than the download the notice promises, so a reader
    # meeting the tool for the first time was told it had misunderstood them.
    #
    # Reading the tuple is the fix rather than adding two more literals. A list of three spellings
    # kept in two places drifts, and this is what that drift looks like.
    if word in DELEGATED or word in ALIASES or word in RETIRED or word in MODES:
        return None
    # A slash or an existing path is unambiguous: nobody mistypes a command into either.
    if "/" in word or os.sep in word or os.path.exists(word):
        return None
    from . import say
    return say.refusal_text(
        f"senbonzakura: reading `{word}` as a model id, since it is not a command.",
        "If you meant a command, `senbonzakura --help` lists them. If you meant a model, this is "
        "right and the next line will be the download.")


#: What bare `senbonzakura` answers with: two labelled groups, and the ways in under each.
#:
#: WHY THIS EXISTS. Typing a tool's name is how people ask what it is. This one answered with a
#: 27-line argparse usage block listing every flag, which is not short help by any reading, and
#: then, because no model was given, refused. So the first thing the tool ever said to anybody was
#: a wall of flags followed by an error, and the guided mode, which is the thing a newcomer
#: actually wants, was not mentioned anywhere on it.
#:
#: TWO LABELLED GROUPS, because "start here" against "or go direct" is the actual decision
#: somebody is making, and naming it is what lets them stop reading after the first group.
#:
#: A TABLE RATHER THAN A BLOCK OF TEXT, because the layout depends on the window. See `short_help`.
#: The commands named here are checked against the real command tables by the test suite rather
#: than trusted, since a second list of command names is a second thing to keep true.
WAYS_IN = (
    ("START HERE", (
        ("senbonzakura -i", "the guided way in"),
    )),
    ("Or go direct", (
        ("senbonzakura --model <id>", "abliterate a model"),
        ("senbonzakura convert <dir>", "turn edited weights into a GGUF"),
        ("senbonzakura validate --model <id>", "check for refusal directions"),
        ("senbonzakura doctor", "is this install okay?"),
        ("senbonzakura --help", "every command, and the core flags"),
    )),
)

#: Below this the descriptions stop sharing a column and go under their commands instead. The
#: longest command is 34 characters and a description needs room to be a phrase rather than two
#: words per line, so a shared column stops being readable well before the window gets small.
_STACK_BELOW = 72


def short_help(columns=None):
    """The page bare `senbonzakura` prints: a masthead, then the ways in.

    ONE DESCRIPTION COLUMN, at a fixed depth, WHERE THERE IS ROOM FOR ONE. The earlier draft
    padded each line to its own command's width, so three description columns started at three
    different depths and the eye had no single edge to run down. In a half width window there is
    no room for a second column at all, and holding the layout there means running off the edge,
    so the description drops under its command instead. Found by a journey driving the real
    program in a fifty two column terminal.

    The masthead is here rather than in the banner because the banner only ever draws to a
    terminal, and somebody piping this page into a bug report still needs to know which version
    answered them.
    """
    from . import say

    columns = say.width() if columns is None else columns
    depth = max(len(command) for _group, rows in WAYS_IN for command, _note in rows) + 4
    stacked = columns < _STACK_BELOW

    out = [f"    senbonzakura {__version__}"]
    out.extend(f"    {line}" for line in
               say.lines("Precision uncensoring, with receipts.", columns=columns - 4))
    out.extend(["", ""])
    for group, rows in WAYS_IN:
        out.append(f"  {group}")
        out.append("")
        for command, note in rows:
            if stacked:
                out.append(f"    {command}")
                out.extend(f"        {line}" for line in say.lines(note, columns=columns - 8))
            else:
                out.append(f"    {command:<{depth}}{note}")
            out.append("")
        out.append("")
    # One trailing blank rather than the three the loop leaves.
    while len(out) > 1 and out[-1] == "" and out[-2] == "":
        out.pop()
    return "\n".join(out)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Decoration only, and structurally unable to reach a result: it prints nothing unless stdout
    # is a terminal, so a redirected run, a spec's captured log and every `stdout-contains` check
    # see exactly what they saw before this existed.
    from . import banner
    banner.emit(__version__, sys.stdout)

    # BARE `senbonzakura` IS A QUESTION, NOT A MISTAKE, so it is answered rather than refused.
    # Exit 0 for the same reason: the tool was asked what it is and it said.
    if not argv:
        print(short_help())
        return 0

    # `-i` is the guided mode, and it is the one entry point this page advertises. Handled here
    # rather than added to ALIASES because everything in that table is a word: a flag spelling
    # living beside `harm-recognition` would reach `_not_a_command`, the near-miss search and the
    # documentation guards, none of which are written for flags.
    if argv[0] == "-i":
        argv[0] = "interactive"

    # A plain-English second name resolves to the command it stands for, before anything else
    # looks at it, so every downstream error message names the real command.
    if argv and argv[0] in ALIASES:
        # AND `--help` SAYS SO, in one line, before the page. `harm-recognition --help` printed
        # `compass --help` byte for byte: a page whose usage line, examples and artefact names are
        # all `compass`, handed to somebody who typed a different word. The resolution is the right
        # behaviour and it was silent, which leaves the reader to work out whether they ran the
        # thing they asked for.
        #
        # Only on the help path. Every other invocation stays byte-identical, so a redirected run,
        # a captured log and every `stdout-contains` check see what they saw before.
        if {"-h", "--help"} & set(argv[1:]):
            print(f"`{argv[0]}` is another name for `{ALIASES[argv[0]]}`. "
                  f"Its own help follows,\nand every artefact it writes says "
                  f"`{ALIASES[argv[0]]}`.\n")
        argv[0] = ALIASES[argv[0]]

    if argv:
        mistake = _not_a_command(argv[0])
        if mistake:
            print(mistake, file=sys.stderr)
            return 2
        # Said BEFORE the model loads, on stderr so a captured stdout is unchanged. The interpretation
        # was always this; the only new thing is that the user is told which one was chosen.
        notice = _read_as_a_model(argv[0])
        if notice:
            print(notice, file=sys.stderr)

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
    except (ImportError, OSError) as e:
        # `bankai` is a flag, not a name: naming the command after it printed "cannot run
        # 'True'". The word the user typed is the one they can act on.
        raise _cannot_run("kageyoshi" if bankai else "abliterate", "cli", e) from e
    return exit_status(run_parsed(args, bankai, rest))
