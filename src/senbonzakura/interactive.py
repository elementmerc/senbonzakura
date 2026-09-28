# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A guided front end for people who have not read the flag list yet.

WHAT IT IS FOR

`senbonzakura kageyoshi` has 43 flags. Most of them have sensible defaults and three of them
decide whether the run means anything. A newcomer cannot tell which three from `--help`, and the
usual result is a run that finishes, produces numbers, and answered a different question.

So this asks the handful of questions that matter, in order, with a default on every one.

THE RULE THIS FOLLOWS, AND WHY

**It never runs anything without first printing the exact non-interactive command it is
equivalent to.** That is the whole design constraint, and it is not decoration:

- A run somebody cannot reproduce is not a measurement, and "I clicked through a menu" is not a
  method section. The printed command is what goes in a lab notebook.
- It teaches the CLI instead of replacing it. The second time, the user types the command.
- It makes the interactive path auditable. A guided mode that quietly passed a different flag
  than it displayed would be a very good way to hide a mistake, and printing the command first
  makes that impossible to do without it showing.

WHAT IT WILL NOT DO

It refuses to run when input is not a terminal, rather than blocking forever on a read that will
never return. A script that pipes into this wanted the flags, not the menu, and being told so
beats a job that hangs in CI until something kills it.
"""
import sys
from pathlib import Path

from . import bundled, runrecord, say

#: Datasets offered by name that are FETCHED FROM THE HUB. What ships inside the package is
#: generated from `corpora.CORPORA` by `_bundled_entries` below and offered first.
#:
#: WHAT THIS TABLE USED TO GET WRONG. `advbench` was listed here, pointed at
#: `walledai/AdvBench::train`, and told the reader it is "GATED on the Hub" and needs
#: `hf auth login` before it will fetch. The package bundles AdvBench's 520 prompts, along with
#: HarmBench, StrongREJECT and XSTest, and `--harmful advbench` reads them off the disk with the
#: network unplugged. So the menu sent a newcomer to authenticate against a service for a file
#: they had already installed, and a third of them would have given up there.
#:
#: The licence column is the reason this list is short. Every entry has to be one whose position
#: has actually been checked, and a corpus nobody has traced does not go on a menu.
#:
#: `side` IS LOAD-BEARING, AND ITS ABSENCE WAS THE DEFECT. There was one menu and one question, and whatever
#: came back went straight to `--track`, which takes a directory holding `bad_ds`, `good_ds` and
#: `bad_eval_ds`. Three of the four corpora offered are a single Hub split with one side of the
#: contrast in it, so picking any of them printed a command that dies in the pre-flight with
#: "nothing exists at that path". Only the bundled entry ever worked. A corpus is now offered for
#: the side it actually holds, and a pair of sides is turned into a track by the command that
#: exists for it.
KNOWN_DATASETS = [
    {
        "key": "default",
        "spec": "default",
        "side": "track",
        "title": "The Senbonzakura evaluation track (bundled)",
        "licence": "CC BY-NC 4.0",
        "note": "9,877 prompts, three-way split, ships inside the package. Attribution "
                "required, non-commercial only.",
    },
    {
        "key": "harmful-behaviors",
        "spec": "mlabonne/harmful_behaviors::train",
        "side": "harmful",
        "title": "mlabonne/harmful_behaviors",
        "licence": "undeclared upstream",
        "note": "416 requests, and what Heretic's defaults fit on, so choose it to compare like "
                "with like. Declares no licence; see the dataset card.",
    },
    {
        "key": "harmless-alpaca",
        "spec": "mlabonne/harmless_alpaca::train",
        "side": "harmless",
        "title": "mlabonne/harmless_alpaca",
        "licence": "undeclared upstream, probably CC BY-NC 4.0",
        "note": "The harmless arm most abliteration tutorials use.",
    },
]


#: The bundled corpora this menu offers, in the order the design put them. A curated subset rather
#: than everything in `corpora.CORPORA`: the two halves of XSTest and HarmBench's copyright set are
#: real and answer narrower questions, and a menu that lists six near-identical rows is how a
#: person stops reading menus. The ones left off are still reachable, because `Something of my own`
#: takes a bundled name as readily as a path.
_BUNDLED_ON_THE_MENU = ("advbench", "strongreject", "harmbench", "xstest-safe")

#: `corpora.arm` to the side of the contrast this file asks about.
_ARM_SIDE = {"harmful": "harmful", "benign": "harmless"}


def _bundled_entries(side):
    """Menu rows for the corpora inside the package, generated from the table that defines them.

    GENERATED, NOT TYPED. The row counts and licences on this menu are the ones the licences
    themselves depend on, and this file already carried a copy of them that had gone wrong in the
    worst available way: it named a bundled corpus as a gated Hub download. One source, so a
    corpus repinned upstream cannot leave a stale number on a menu.
    """
    from . import corpora
    rows = []
    for key in _BUNDLED_ON_THE_MENU:
        c = corpora.CORPORA[key]
        if _ARM_SIDE.get(c.arm) != side:
            continue
        rows.append({
            "key": key, "spec": key, "side": side, "licence": c.licence,
            "title": c.name,
            # FIRST SENTENCE ONLY. `corpora` writes for `doctor`, which has room for a paragraph;
            # a menu row that runs to four lines is how a person stops reading the menu.
            "note": f"{c.rows:,} prompts, inside the package. {c.note.split('. ')[0]}",
        })
    return rows


def by_side(side):
    """The corpora that hold `side`, plus a way to name one that is not on the list.

    Bundled first, because a set that is already on the disk cannot be gated, cannot need an
    account, and works with the network unplugged.
    """
    entries = _bundled_entries(side) + [e for e in KNOWN_DATASETS if e["side"] == side]
    entries.append({
        "key": "own", "spec": None, "side": side, "licence": "yours",
        "title": "Something of my own",
        "note": "a text file (one prompt per line), a CSV/JSON with a prompt column, a Hub id, "
                "or the name of another bundled corpus",
    })
    return entries

DEVICES = [
    ("cuda", "an NVIDIA card. Needed to edit a model in any reasonable time"),
    ("cpu", "no card. Fine for scoring and for trying the plumbing, slow for editing"),
    ("mps", "Apple silicon"),
]

#: Where cpu sits in `DEVICES`. Looked up rather than typed, so reordering the menu cannot silently
#: point the fallback at a card. See `pick_device`.
CPU_INDEX = next(i for i, (name, _note) in enumerate(DEVICES) if name == "cpu")


class AbandonedError(Exception):
    """The user chose to stop.

    Named with the suffix the linter wants, but it is not an error and is never printed as one:
    Ctrl+C at a prompt is a decision, and answering it with a traceback would be rude.
    """


def is_tty(stream=None):
    stream = stream or sys.stdin
    return bool(getattr(stream, "isatty", lambda: False)())


#: Characters that need no quoting anywhere. `\` is here for Windows and only for Windows: see
#: `quote`. On POSIX a backslash is an escape character and must be quoted.
_SAFE = "-_./:=[]@,"


def quote(value):
    r"""Shell-quote a value only when it needs it, so the printed command stays readable.

    THE QUOTE CHARACTER FOLLOWS THE SHELL THE READER WILL PASTE INTO. This line exists to be
    copied out of the terminal and run, so quoting it for the wrong shell does not make it ugly,
    it makes it wrong.

    On POSIX that is single quotes, with the usual `'\''` dance for an embedded one. On Windows
    every path contains backslashes, so POSIX rules quoted every single path, and it quoted them
    as `'C:\Users\...'`: `cmd.exe` has no single-quote syntax at all and passes those quotes
    through as part of the path, so the command reliably failed for the one reader it was printed
    for. Double quotes are what both `cmd.exe` and PowerShell understand, and a backslash needs no
    quoting there, so an ordinary Windows path prints bare.
    """
    text = str(value)
    if sys.platform.startswith("win"):
        if text and all(c.isalnum() or c in _SAFE + "\\" for c in text):
            return text
        # `cmd.exe` cannot represent a double quote inside a quoted argument at all. Doubling it
        # is what PowerShell reads, and it is the closest thing to a convention that exists.
        return '"' + text.replace('"', '""') + '"'
    if text and all(c.isalnum() or c in _SAFE for c in text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


def render_command(command, options):
    """The exact non-interactive line that reproduces this run.

    Options whose value is None or False are dropped; True becomes a bare flag. Order is the
    order the questions were asked, so the printed command reads like the conversation did.
    """
    parts = ["senbonzakura", command]
    for flag, value in options.items():
        if value is None or value is False:
            continue
        if value is True:
            # A bare flag, or a positional passed as a key with no value. Quoted either way,
            # because a path with a space in it is a real thing and an unquoted one silently
            # becomes two arguments.
            parts.append(quote(flag) if not flag.startswith("-") else flag)
        else:
            parts.append(f"{flag} {quote(value)}")
    return " ".join(parts)


def _wrap(text, *, indent="", reserve=0, log=print):
    """Print all but the last line of `text`, wrapped, and return the last line.

    WHY THE WHOLE FILE NEEDED THIS. Every question here was a hand-written literal handed straight
    to `input`, so a question longer than the terminal was re-wrapped by the terminal itself, at
    whatever character happened to land on the edge. `doctor` has gone through `say` for months
    and reads correctly at 60 columns; the guided mode, which is the screen written for somebody
    who has never used the tool, was the one that did not.

    The last line comes back rather than being printed, because a question ends in the prompt the
    person types on, and `[default]: ` has to stay on it.
    """
    lead = len(text) - len(text.lstrip("\n"))
    for _ in range(lead):
        log("")
    body = say.lines(text.lstrip("\n"), indent=indent,
                     columns=max(40, say.width() - reserve)) or [""]
    for line in body[:-1]:
        log(line)
    return body[-1]


def _say(text, log=print, indent=""):
    """One paragraph, wrapped to this terminal.

    The literals in this file were hand-wrapped at 79 columns, which reads correctly on a wide
    terminal and raggedly on a narrow one, and the guided mode is the screen most likely to be
    open in a half-width window beside something else. `say` already leaves machine markers and
    indented commands alone, so a command a person has to paste is never folded.
    """
    for line in say.lines(text, indent=indent) or [""]:
        log(line)


def ask(question, *, default=None, ask_fn=input, log=print):
    """One free-text question with an optional default."""
    suffix = f" [{default}]" if default is not None else ""
    question = _wrap(question, reserve=len(suffix) + 2, log=log)
    while True:
        try:
            answer = ask_fn(f"{question}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise AbandonedError from None
        if answer:
            return answer
        if default is not None:
            return default
        log("  That one has no default, so it needs an answer.")


def choose(question, options, *, default=0, ask_fn=input, log=print):
    """Pick one of `options`, a list of (label, description). Returns the index."""
    log("")
    log(_wrap(question, log=log))
    for i, (label, description) in enumerate(options, 1):
        marker = "*" if i - 1 == default else " "
        log(f"  {marker} {i}. {label}")
        if description:
            for line in say.lines(description, indent="       ",
                                  columns=max(40, say.width())):
                log(line)
    while True:
        try:
            answer = ask_fn(f"Choose 1 to {len(options)} [{default + 1}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise AbandonedError from None
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        log(f"  '{answer}' is not one of 1 to {len(options)}.")


def confirm(question, *, default=True, ask_fn=input, log=print):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            answer = ask_fn(f"{question} {suffix}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise AbandonedError from None
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        log("  Please answer y or n.")


def warn_if_unbundled(log=print):
    """The bundled track is packed at release time. Returns True when there is none here.

    TWO CAUSES, TWO REMEDIES, AND IT USED TO PRINT ONE SENTENCE FOR BOTH. A source checkout with no
    packed track is normal and one command fixes it. An installed wheel with no packed track is
    defective, and `tools/packaging/pack_track.py` is not in that install, so naming it sends the
    reader looking for a file that was never shipped. `bundled.running_from_a_checkout` exists to
    tell those apart and this was the one caller not asking it.

    It also promised "choose another option" while the caller returned the bundled track anyway, so
    the only way out of the screen was Ctrl+C at the confirm. The offer is the caller's job now, and
    the return value is what lets it make one.
    """
    if bundled.is_available():
        return False
    log("")
    log("  NOTE: no bundled track is installed here.")
    if bundled.running_from_a_checkout():
        log("        It is packed at release time, so a source checkout carries none until")
        log("        `python tools/packaging/pack_track.py --track <dir>` has been run.")
    else:
        log("        A wheel is supposed to ship one, so its absence in an installed copy is a")
        log("        packaging fault rather than something you can fix here. Please report it.")
    return True


def pick_side(side, question, ask_fn=input, log=print):
    """One side of the contrast, as (spec, licence). Licences are shown before the choice."""
    entries = by_side(side)
    options = [(f"{e['title']}  ({e['licence']})", e["note"]) for e in entries]
    entry = entries[choose(question, options, default=0, ask_fn=ask_fn, log=log)]
    if entry["spec"] is None:
        return ask("  Path, or Hub id (add ::split[:N] to slice it)",
                   ask_fn=ask_fn, log=log), "yours"
    return entry["spec"], entry["licence"]


#: What makes a directory a model this can edit. Asked of the files rather than of the cache
#: index, which is the distinction that decides this whole screen: see `cached_models`.
WEIGHT_SUFFIXES = (".safetensors", ".bin")


def cached_models():
    """Hub models on this machine that actually hold weights, largest first.

    IT IS NOT "IS IT CACHED", IT IS "DOES THE SNAPSHOT HOLD WEIGHTS". A repository lands in the
    cache the moment anything reads its config, so a cache listing is full of entries that are one
    `config.json` and nothing else. The cache on the development machine holds two such repos, and
    a picker built on the index would have offered a 17 GB model as ready to go and been wrong
    about both words: it is not 17 GB here and it is not ready.

    Every failure is answered with an empty list. This screen is a convenience on top of typing a
    model id, and a cache that cannot be read is a reason to ask the question rather than a reason
    to stop.
    """
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
    except Exception:       # see the docstring: no cache is not a failure here
        return []
    found = []
    for repo in getattr(info, "repos", ()):
        if getattr(repo, "repo_type", None) != "model":
            continue
        for revision in getattr(repo, "revisions", ()):
            if any(str(f.file_name).endswith(WEIGHT_SUFFIXES) for f in revision.files):
                found.append({"id": repo.repo_id, "bytes": repo.size_on_disk, "note": ""})
                break
    return sorted(found, key=lambda m: -m["bytes"])


def edited_models(root="."):
    """Directories under `root` holding a model, which on this machine means one you made.

    One level down, for the reason `resumable_runs` gives: a recursive walk of somebody's home
    directory to populate a menu is slow and lists runs from projects they are not in.
    """
    found = []
    try:
        entries = sorted(Path(root).iterdir())
    except OSError:
        return found
    for d in entries:
        if not d.is_dir() or not (d / "config.json").is_file():
            continue
        weights = [p for p in d.iterdir() if p.suffix in WEIGHT_SUFFIXES]
        if not weights:
            continue
        found.append({"id": str(d), "bytes": sum(p.stat().st_size for p in weights),
                      "note": "yours, already edited"})
    return found


def _size(n):
    return f"{n / 1e9:.2f} GB" if n else "—"


def models_on_this_machine(root="."):
    """Everything a run could start from without downloading anything. Yours first."""
    return edited_models(root) + cached_models()


def pick_model(ask_fn=input, log=print, root="."):
    """Which model to edit.

    WHY THIS IS A LIST OF WHAT IS HERE AND NOT A LIST OF SUGGESTIONS. The question used to be one
    free-text line with `Qwen/Qwen3-1.7B` as the default, so pressing Enter through the walk
    started a download of a model nobody had chosen. A nominated model is a recommendation, and
    this project has no measurement that would justify recommending one over another; a fixed list
    also ages into naming whatever was fashionable the year it shipped.

    What is on this machine is a fact rather than a recommendation, and it is the case where
    nothing has to be downloaded and nothing can be gated. When there is nothing here, the screen
    says so and asks, which is a better first screen than five names nobody chose.
    """
    found = models_on_this_machine(root)
    if not found:
        log("")
        _say("Nothing on this machine holds weights this can edit, so this one has to be "
             "fetched.", log=log)
        return ask("Which model? (a Hub id, or a local directory)", ask_fn=ask_fn, log=log)

    # Trimmed, because this is a menu rather than an inventory. `doctor` is the place that lists
    # everything; a screen asking one question offers the plausible answers and a way to type any
    # other. The cut is by size order, so the largest models on the machine are the ones shown.
    shown = found[:6]
    options = [(f"{m['id']}   {_size(m['bytes'])}", m["note"]) for m in shown]
    options.append(("Something else", "a Hub id, or a folder on disk"))
    log("")
    _say("These are already on this machine, so nothing is downloaded and nothing can be gated.",
         log=log)
    index = choose("Which model?", options, default=0, ask_fn=ask_fn, log=log)
    if index == len(shown):
        return ask("  A Hub id, or a local directory", ask_fn=ask_fn, log=log)
    return shown[index]["id"]


def device_available(name):
    """Whether this machine can actually use a device, or None when it cannot be asked.

    None rather than False when torch is absent: the guided mode runs on a base install, and
    "cannot tell" must not be presented to a newcomer as "no".
    """
    try:
        import torch
    except ImportError:
        return None
    if name == "cuda":
        return bool(torch.cuda.is_available())
    if name == "mps":
        backend = getattr(torch.backends, "mps", None)
        return bool(getattr(backend, "is_available", lambda: False)())
    return True


def pick_device(ask_fn=input, log=print):
    """Offer every device, but default to one this machine can actually use.

    FOUND BY ADVERSARIAL USER TESTING, 2026-09-16. `DEVICES` was a static list with cuda first and
    no detection, so on a machine with no card the guided mode walked a newcomer through every
    default and handed them

        senbonzakura kageyoshi ... --device cuda

    which cannot run there. This is the one surface built so that newcomers do not have to know
    things, steering them into the exact failure the device pre-flight now refuses. A sentence is
    better than the traceback it used to be, but proposing a broken command at all is the defect.

    Unavailable devices are still LISTED, because somebody may be composing a command to run
    elsewhere, and a menu that hides options teaches a wrong model of the tool. They are marked,
    and the default moves to the first one that works.
    """
    marked, first_usable = [], None
    for position, (name, note) in enumerate(DEVICES):
        usable = device_available(name)
        if usable and first_usable is None:
            first_usable = position
        suffix = "" if usable is None else ("" if usable else "  [not available on this machine]")
        marked.append((name + suffix, note))
    if first_usable is None:
        # THE HOLE IN THE DETECTION, and it was in the one case this function was written for.
        # `device_available` answers None when torch is absent, None is falsy, so nothing ever set
        # `first_usable` and the default fell back to index 0, which is cuda, with no marker beside
        # it. A base install is exactly where torch is missing, so the walk that exists to stop a
        # newcomer being handed `--device cuda` handed it to them by default there.
        #
        # cpu rather than the first entry, because it is the one that cannot be wrong about the
        # hardware, and said out loud rather than chosen quietly: "cannot tell" is not "no", and a
        # person composing a command for a machine with a card must still be able to say so.
        first_usable = CPU_INDEX
        log("")
        # `pip install senbonzakura`, AND NOT AN EXTRA. This line said
        # `pip install senbonzakura[abliterate]` until 2026-09-28, which was right before Q-27
        # moved torch into the base install and has been advice that changes nothing ever since:
        # the extra still resolves, to the package itself, so somebody following it watched pip
        # do nothing and was no closer to a working card.
        _say("NOTE: PyTorch is not installed here, so this cannot check what the machine has. "
             "cpu is the default for that reason alone. Choose cuda if you know there is a card. "
             "`pip install senbonzakura` brings torch, and `senbonzakura setup` fits it to this "
             "machine.",
             log=log, indent="        ")
    index = choose("Where should it run?", marked, default=first_usable,
                   ask_fn=ask_fn, log=log)
    return DEVICES[index][0]


def pick_track(ask_fn=input, log=print):
    """Which track the search fits and scores on, as (spec, licence, build step or None).

    A track is a directory with three partitions in it, and the three ways to get one are the
    three options here. Building one is a real command with its own flags, so it is returned as a
    step to run first rather than folded into the abliterate line: the rule this file keeps is
    that what it prints is what it runs.
    """
    options = [
        ("The Senbonzakura evaluation track (bundled)  (CC BY-NC 4.0)",
         ("9,877 prompts, three-way split, ships inside the package. Attribution required, "
          "non-commercial only.")),
        ("Build a track from a harmful set and a harmless set",
         ("pick one of each; `senbonzakura track` splits them into fit, search and measure "
          "partitions and checks the three do not overlap")),
        ("A track directory I already built",
         "one holding bad_ds, good_ds and bad_eval_ds"),
    ]
    while True:
        index = choose("Which prompts should it learn refusal from?", options,
                       default=0, ask_fn=ask_fn, log=log)
        if index != 0 or not warn_if_unbundled(log=log):
            break
        # The way out the note used to name and not provide. Defaulting to no, because the run it
        # would start dies in the track pre-flight and the other two options both work here.
        if not confirm("  Ask for it anyway?", default=False, ask_fn=ask_fn, log=log):
            continue
        break

    if index == 0:
        return "default", "CC BY-NC 4.0", None
    if index == 2:
        return ask("  Path to the track directory", ask_fn=ask_fn, log=log), "yours", None

    harmful, harmful_licence = pick_side(
        "harmful", "Which harmful prompts?", ask_fn=ask_fn, log=log)
    harmless, harmless_licence = pick_side(
        "harmless", "Which harmless prompts to contrast them against?", ask_fn=ask_fn, log=log)
    out = ask("\n  Where should the built track go?", default="track", ask_fn=ask_fn, log=log)
    licence = harmful_licence if harmful_licence == harmless_licence else \
        f"{harmful_licence} (harmful) and {harmless_licence} (harmless)"
    build = {"command": "track",
             "options": {"--harmful": harmful, "--harmless": harmless, "--out": out}}
    return out, licence, build


def pick_eval(ask_fn=input, log=print):
    """The prompts a `score` run is measured on, as (spec, licence).

    Scoring needs ONE prompt set, not a three-way split, so this asks for one rather than sending
    somebody to build a track they will use a third of. The bundled entry names its held-out
    partition directly, which is the partition a published number should come from.
    """
    entries = by_side("harmful")
    options = [("The bundled track's held-out harmful prompts  (CC BY-NC 4.0)",
                ("4,636 prompts nothing was fitted or searched on, which is where a number worth "
                 "publishing comes from"))]
    options += [(f"{e['title']}  ({e['licence']})", e["note"]) for e in entries]
    while True:
        index = choose("Which prompts should it be measured on?", options,
                       default=0, ask_fn=ask_fn, log=log)
        if index != 0 or not warn_if_unbundled(log=log):
            break
        if not confirm("  Ask for it anyway?", default=False, ask_fn=ask_fn, log=log):
            continue
        break
    if index == 0:
        return "default/bad_eval_ds", "CC BY-NC 4.0"
    entry = entries[index - 1]
    if entry["spec"] is None:
        return ask("  Path, or Hub id (add ::split[:N] to slice it)",
                   ask_fn=ask_fn, log=log), "yours"
    return entry["spec"], entry["licence"]


#: What makes a directory something you can carry on with, rather than merely occupied. The study
#: is the expensive part: it holds every completed trial. `best-config.json` is the cheaper but
#: more valuable one, because it turns a crashed run from hours of re-searching into minutes of
#: re-baking.
STUDY_DB = "senbon-study.db"
BAKEABLE = "best-config.json"


#: Scene 2's menu. Every entry here has a path behind it, and that is the whole point of the list
#: being this short (critique finding 2): the design offered five recipes and walked one, and the
#: two with nothing behind them were worse than absent, because a person commits to the path before
#: it stops making sense. "Measure a model I already have" was the sharp case, since the screens
#: that follow the abliterate recipe ask where the edited model goes and there is no edited model.
RECIPES = [
    ("abliterate", "Abliterate a model", "the usual job"),
    ("brain", "Build a local brain", "abliterate, then convert for llama.cpp"),
    ("measure", "Measure a model I already have", "no editing, no output model"),
    ("flags", "Everything by hand", "prints the flag list and stops"),
]


def missing_conversion_tools():
    """What this install lacks before it could write a GGUF, as display phrases. Empty when it can.

    ASKED BEFORE THE RUN, WHICH IS THE WHOLE POINT. The "Build a local brain" recipe appends a
    convert-and-quantise step to a plan whose first command downloads a model and spends GPU hours,
    and both halves of that step are build-time artefacts: the converter is a Python script fetched
    by the vendoring tool, and `llama-quantize` is a compiled binary. A source checkout has neither
    until that tool has run, so the recipe could search for hours, save a model, and only then say
    it cannot do the half the person picked the recipe for. Every bit of this verdict is knowable
    before anything is downloaded, which is the reason the rest of the tool pre-flights at all.
    """
    from .vendored import VendorError, find_binary, find_script
    missing = []
    try:
        find_script("convert_hf_to_gguf.py")
    except VendorError:
        missing.append("the vendored converter script")
    try:
        find_binary("llama-quantize", log=lambda _m: None)
    except VendorError:
        missing.append("llama-quantize, which is the only thing that can write a Q4_K_M")
    return missing


def resumable_runs(root="."):
    """Runs under `root` that can be carried on with, as (path, what is there) pairs.

    THE WAY BACK IN (critique finding 3). The guided mode could CREATE a paused run and could not
    FIND one: scene 9 writes the marker and tells you to resume with a flag, and scenes 1 and 2
    never mention it, so somebody who paused yesterday is sent back to the flag list, which is the
    one outcome this mode exists to avoid.

    Only one level down, deliberately. A recursive walk of somebody's home directory to populate a
    menu is slow, surprising, and would list runs from projects they are not in.
    """
    out = []
    try:
        entries = sorted(Path(root).iterdir())
    except OSError:
        return out
    for d in entries:
        if not d.is_dir():
            continue
        has_study = (d / STUDY_DB).is_file()
        has_config = (d / BAKEABLE).is_file()
        if has_study and has_config:
            what = "a search in progress, and a winning config already picked"
        elif has_study:
            what = "a search in progress, with its completed trials kept"
        elif has_config:
            what = "a winning config, so it re-bakes in minutes rather than re-searching"
        else:
            continue
        # `read_quiet`: a damaged record here costs a line of decoration on a menu, not a
        # decision. The strict `read` belongs where the answer gates a run.
        record = runrecord.read_quiet(d)
        if record and record.get("model"):
            what += f"; it was editing {record['model']}"
        out.append((str(d), what, record))
    return out


def licence_for_track(track):
    """The licence a recorded track implies, in the words the fresh path uses for the same corpus.

    Only the bundled alias is knowable from a string: any other value is a directory on somebody's
    disk, whose terms are theirs. The resume path used to hard-code "yours" for every run alike, so
    resuming a run recorded against `default` dropped the CC BY-NC notice that the fresh walk shows
    for the same 9,877 prompts. A licence obligation that appears on one route and not the other is
    worse than one that appears on neither, because it reads as a considered decision.
    """
    return "CC BY-NC 4.0" if str(track) == "default" else "yours"


def resume_plan(path, record, ask_fn=input, log=print):
    """The command that carries on the run in `path`, filled in from what that run recorded.

    THE SCREEN THAT PRINTED A COMMAND NOBODY COULD RUN. This offered
    `senbonzakura kageyoshi --out <dir> --resume`, and `--model` is required, so the one screen
    written to save somebody hours produced an argparse error. Adding `--model` alone would have
    been worse: `--track` has a default, so a resume that omits it carries on one corpus's trials
    while scoring new ones against another, and no artefact says the run changed corpus halfway.

    THE SAME MISTAKE THREE MORE TIMES, found 2026-09-28. `run.json` also records the device, the
    trial budget and the search strategy, and this read none of them, so the parser default won
    every time:

    - `--device` fell back to cuda, and a run resumed on the machine it was copied to died in the
      device pre-flight. That is the one the operator actually met.
    - `--trials` fell back to 60 against a study with 150 trials already in it, so the remaining
      budget computed to zero and the run announced "the trial budget is already spent" and baked.
      The model came out of a search the user had asked to be 200 trials, and every artefact said
      otherwise, with nothing on screen asking them.
    - `--search` is pinned in `runrecord.PINNED`, so a study built with `--search scalar` resumed
      at the default pareto is refused outright, over a flag this menu has never mentioned.

    Anything recorded is carried, and anything carried that could surprise the reader is said out
    loud before the confirm rather than left in the printed line for them to notice.

    A run from a build that predates the record cannot be reconstructed, so its inputs are asked
    for rather than guessed, and the same guard in the abliterator refuses the pair if they
    disagree with the study anyway.
    """
    record = record or {}
    options = {}
    if record.get("model"):
        options["--model"] = record["model"]
    else:
        log("")
        log(f"  {path} does not say which model it was editing, so it predates the record runs")
        log("  now keep. Resuming with the wrong one would continue this search against a")
        log("  different model, so it has to be named.")
        options["--model"] = ask("  Which model was it?", ask_fn=ask_fn, log=log)
    if record.get("track"):
        options["--track"] = record["track"]
    else:
        options["--track"] = ask("  And which track was it scored on?", default="default",
                                 ask_fn=ask_fn, log=log)
    options["--out"] = path

    # THE FLAG THAT ACTUALLY RE-BAKES. A directory holding `best-config.json` and no study is
    # offered as "a winning config, so it re-bakes in minutes rather than re-searching", and this
    # emitted `--resume`, which finds no study, creates one, and starts a fresh search from trial
    # zero under a log line saying "(resuming)". The promise was minutes and the cost was the whole
    # search again. `--bake-config` is the flag that does what the menu row says, and the guided
    # mode had never emitted it.
    root = Path(path)
    baking = (root / BAKEABLE).is_file() and not (root / STUDY_DB).is_file()

    notes = []
    device = record.get("device")
    if device:
        options["--device"] = device
        if device_available(device) is False:
            notes.append(f"--device {device}, which is what it ran on. This machine cannot use "
                         f"that device, so change it on the line above or the pre-flight refuses.")
        else:
            notes.append(f"--device {device}, the device the first leg ran on.")
    trials = record.get("trials")
    if trials and not baking:
        options["--trials"] = trials
        notes.append(f"--trials {trials}, the budget the first leg was given. Without it the "
                     f"default of 60 applies, and a study already past 60 trials would be "
                     f"declared finished and baked early.")
    search = record.get("search")
    # A resumed run meets the same refusal as a fresh one, so it gets the same sizing. The note
    # joins the list this screen already keeps, rather than being printed over it.
    probe_note = size_the_probe_for(device, options)
    if probe_note:
        notes.append("--capability-n 40 and --capability-max-new 256. " + probe_note)
    if search:
        options["--search"] = search
        notes.append(f"--search {search}, the strategy the completed trials were searched with. "
                     f"The study is named after it, so a different one starts from trial zero.")

    if baking:
        options["--bake-config"] = str(root / BAKEABLE)
    else:
        options["--resume"] = True

    log("")
    if baking:
        log(f"  {path} holds a winning configuration and no study, so this bakes that")
        log("  configuration straight out rather than searching for it again.")
    else:
        log("  The completed trials in the study are kept and the search carries on from them.")
    # HONESTY ABOUT WHAT IS WRITTEN OVER. This screen used to open with "Nothing there is
    # overwritten", which is false of every resume: the run record, the winning config and the
    # saved weights are all rewritten in that directory as the run goes on. What survives is the
    # completed trials, which is the thing worth saying, and saying it accurately costs nothing.
    log("  The run record, the winning config and any saved weights in that directory are")
    log("  written over as it goes; the completed trials are what is preserved.")
    if notes:
        log("")
        log("  Taken from what that run recorded, so this leg matches the last one:")
        for note in notes:
            log(f"    {note}")

    # WHAT A RESUME CANNOT KNOW. `run.json` records the inputs of the abliteration and nothing
    # about the recipe around it, so a person who chose "Build a local brain" and crashed gets the
    # model baked and no GGUF, with nothing saying a step is missing. It cannot be added silently
    # and there is no recorded answer to read, so the command is named instead.
    log("")
    log("  This edits and saves the model; it converts nothing. If this run was headed for a")
    log("  GGUF, that is a second command once it finishes:")
    log("")
    log(f"    {render_command('convert', {path: True, '--quantise': 'Q4_K_M'})}")

    command = "abliterate" if record.get("bankai") is False else "kageyoshi"
    return {"command": command, "options": options,
            "licence": licence_for_track(options["--track"]), "recipe": "resume"}


def offer_resume(root=".", ask_fn=input, log=print):
    """Offer to carry on with a run found under `root`. Returns a plan, or None to start fresh.

    Shown only when there is something to show. A menu row that is empty most of the time trains
    people to skip the first question, and the first question is the one that saves them hours.
    """
    found = resumable_runs(root)
    if not found:
        return None
    options = [(f"Carry on with {path}", what) for path, what, _record in found]
    options.append(("Start a new run", "leaves everything above untouched"))
    index = choose("There is a run here you can pick up. What would you like to do?",
                   options, default=0, ask_fn=ask_fn, log=log)
    if index == len(found):
        return None
    path, _what, record = found[index]
    log("")
    log(f"  Carrying on with {path}.")
    return resume_plan(path, record, ask_fn=ask_fn, log=log)


def ask_output(ask_fn=input, log=print, *, chosen=None):
    """Where the edited model goes, and what to do when something is already there.

    Returns (path, resume). `chosen` is the answers already given, keyed as `runrecord.PINNED` is.

    THE LAST QUESTION USED TO INVALIDATE THE FIRST TWO IN SILENCE. The walk asks for the model, the
    track, the device and then the output, and "Continue that run" here adds `--resume`, which
    makes `refuse_across_inputs` pin the model, the track and the search against the `run.json`
    already sitting in that directory. So answers given four screens earlier could be contradicted
    by the last one, and the person found out from a refusal at the end, after a track may already
    have been rebuilt on disk from an answer that no longer applies.

    The record is right there, and this module already reads it to decorate the menu, so the clash
    is shown at the moment the choice is made and the person picks which of the two wins.

    A CHOICE RATHER THAN A WARNING, and that is the point of it (critique finding 5). A warning is
    what a person scrolls past on the way to the next question; this project has lost three
    separate results to a previous run's output sitting in the directory while something reported
    the work already done. The non-interactive path now refuses outright. Here, where somebody is
    being walked through it, refusing would be a dead end, so the three ways out are offered
    directly and the loop repeats until one of them is taken.
    """
    while True:
        out = ask("\nWhere should the edited model go?", default="abliterated",
                  ask_fn=ask_fn, log=log)
        # `runrecord`, not `cli`. This used to reach through `cli`, which imports torch, optuna
        # and transformers at module scope, so on a base install the guided mode asked four
        # questions and then died on an eleven-frame ImportError at this line, having told the
        # person nothing about what to install. The function itself only lists files.
        found = runrecord.occupied_by(out)
        if not found:
            return out, False
        log(f"\n  {out} already holds a previous run:")
        for name in found:
            log(f"    {name}")
        log("  Writing over it would replace some of those files and leave others, and the")
        log("  artefact would then describe two runs with nothing saying so.")
        pick = choose("What should happen?", [
            ("Choose a different directory", "keeps both runs"),
            ("Continue that run", "resumes the search where it stopped, adds --resume"),
        ], default=0, ask_fn=ask_fn, log=log)
        if pick == 1:
            clash = runrecord.mismatches(runrecord.read_quiet(out), chosen or {})
            if not clash:
                return out, True
            log("")
            log(f"  {out} records a run that was given different answers:")
            for field, was, now in clash:
                log(f"    --{field}: it used {was!r}, and you chose {now!r}"
                    f" ({runrecord.PINNED[field]}).")
            log("  Carrying on its completed trials under your answers would score one corpus's")
            log("  trials against another, so the tool refuses that outright. One of the two")
            log("  has to give, and it is your choice which.")
            keep = choose("Which should this run use?", [
                (f"What {out} recorded", ("the answers above change to match it, and the trials "
                                          "already in that study are kept")),
                ("Choose a different directory", "keeps the answers you gave, and keeps both runs"),
            ], default=0, ask_fn=ask_fn, log=log)
            if keep == 0:
                return out, True
            log("  Nothing has been changed. Pick another path.")
            continue
        # Deliberately no "overwrite" option. Deleting somebody's previous result on their behalf,
        # inside a guided flow they are still learning, is not a choice this should offer; the
        # person can remove the directory themselves and come back.
        log("  Nothing has been changed. Pick another path.")


#: What the capability probe costs on a processor, and what the tool's own refusal recommends
#: instead. 200 items at 512 new tokens is minutes on a card and about four hours on a CPU.
CPU_PROBE = {"--capability-n": "40", "--capability-max-new": "256"}


def size_the_probe_for(device, options):
    """Fit the capability probe to the device, and return the sentence that says so, or None.

    THE WALK PRODUCED A COMMAND ITS OWN PRE-FLIGHT REFUSED. The probe defaults are sized for a
    card, `refuse_a_slow_probe` stops a four-hour one before it starts, and this walk never asked
    about either, so somebody on a laptop answered six questions and was handed a command that
    could not run, over flags they had no reason to know existed.

    Applied rather than asked, because a seventh question about a probe budget is worse than a
    sensible default, and NOT applied quietly: it changes what the capability figure means, from
    200 paired items to 40. It goes on the printed command, so the line somebody copies is the
    line that ran and the smaller probe is visible in it.
    """
    if not str(device or "").startswith("cpu"):
        return None
    options.update(CPU_PROBE)
    return ("The capability probe is sized for a processor: 40 items rather than 200, which is "
            "minutes instead of about four hours. It is on the command below, so the same run on "
            "a card is that line without those two flags.")


def ask_trials(ask_fn=input, log=print):
    """The trial budget, checked here rather than five screens later.

    TWO DEFECTS IN ONE PROMPT. It was free text handed straight to the plan, so "abc" and "0" were
    accepted, printed into the command, confirmed by the person, and only then refused by
    `--trials`, which wants a whole number of at least 1. Every other question in this walk
    validates what it takes, and this was the one that did not.

    And the sentence was wrong. "200 is the usual" matches nothing in the tool: the flat default is
    60, and the kageyoshi preset picks 100, 80 or 64 by model size. A number invented for a prompt,
    described as the convention, is how a reader ends up believing the tool has a convention it has
    never had.
    """
    while True:
        # THE PRESETS ARE IN THE PROMPT RATHER THAN IN A MENU, and this is the design's four-row
        # screen collapsed to one line. Its estimate column reads `—` on every row, because
        # nothing here can price a trial before a trial has run, and a menu of three numbers with
        # no numbers beside them carries nothing the numbers do not. A typed answer also needs no
        # `Custom…` row, which is the row that screen existed to justify.
        #
        # None of the three is called usual. The flat default is 60 and the preset picks 100, 80
        # or 64 by model size, so naming one of these as the convention would invent a convention
        # the tool does not have, which is the defect this prompt already carried once.
        answer = ask("How many search trials? More is better and slower: 40 is a quick look, "
                     "200 explores the frontier, 500 is thorough. Left alone, the auto preset "
                     "picks 60 to 100 by model size",
                     default="200", ask_fn=ask_fn, log=log)
        try:
            count = int(str(answer).strip())
        except ValueError:
            count = 0
        if count >= 1:
            return str(count)
        log(f"  '{answer}' is not a trial budget. It has to be a whole number, 1 or more.")


def plan_abliteration(ask_fn=input, log=print):
    """Walk the questions that decide whether an abliteration run means anything."""
    log("")
    _say("Senbonzakura, guided mode. Ctrl+C stops at any point and changes nothing. Every "
         "answer has a default, shown in brackets; press Enter to take it.", log=log)

    # Before any of the questions, because the cheapest run is the one already half done.
    carry_on = offer_resume(ask_fn=ask_fn, log=log)
    if carry_on is not None:
        return carry_on

    index = choose("What would you like to do?",
                   [(label, note) for _key, label, note in RECIPES],
                   default=0, ask_fn=ask_fn, log=log)
    recipe = RECIPES[index][0]
    if recipe == "flags":
        # Not a dead end and not a pretend screen: the person asked for the flags, so they get
        # them, and the guided mode gets out of the way rather than wrapping `--help` in a menu.
        #
        # `--help-all`, NOT `--help`. The menu row promises the flag list and `--help` prints the
        # core flags with the rest suppressed, so the one entry written for somebody who wants to
        # see everything was the one that showed them the short page, and never named the flag that
        # does show everything.
        #
        # `prints_only` because the confirm this plan reaches defaults to no, for the good reason
        # that the other three recipes spend GPU hours on it. This one prints a help page, so the
        # bare Enter that is safe everywhere else in the walk turned the one free entry into
        # "Nothing was run." and an exit.
        return {"command": "--help-all", "options": {}, "licence": None, "recipe": recipe,
                "prints_only": True}

    if recipe == "brain":
        # BEFORE THE MODEL DOWNLOADS, because it is knowable before the model downloads.
        missing = missing_conversion_tools()
        if missing:
            log("")
            log("  This install cannot finish that recipe: it is missing " + " and ".join(missing)
                + ".")
            log("  Both are fetched at build time by `python tools/packaging/vendor_llama.py`, so")
            log("  a source checkout has neither until that has run. An installed copy missing")
            log("  them is a packaging fault worth reporting. `senbonzakura doctor` says which.")
            log("  The abliteration itself needs neither and would still work.")
            if choose("What would you like to do?", [
                ("Abliterate anyway, and convert later",
                 "the edit runs now; the GGUF is a second command once the tools are there"),
                ("Stop here", "nothing is run, and nothing is downloaded"),
            ], default=0, ask_fn=ask_fn, log=log) == 1:
                raise AbandonedError
            recipe = "abliterate"

    model = pick_model(ask_fn=ask_fn, log=log)

    if recipe == "measure":
        # Stops after the questions this recipe's command actually takes. The remaining abliterate
        # questions are where the edited model goes and how many search trials to run, and this
        # recipe edits nothing and searches for nothing. Asking them would be the exact defect
        # finding 2 names: screens built for a different job, reached after the person has already
        # committed to the path. `score` also takes ONE prompt set rather than a three-way split,
        # which is why it asks a different corpus question: the old one appended `/bad_eval_ds` to
        # whatever came back, which made a valid path out of the bundled alias and nonsense out of
        # every other answer.
        evalset, licence = pick_eval(ask_fn=ask_fn, log=log)
        device = pick_device(ask_fn=ask_fn, log=log)
        results = ask("\nWhere should the results go?", default="scores.json",
                      ask_fn=ask_fn, log=log)
        return {"command": "score",
                "options": {"--model": model, "--eval": evalset,
                            "--device": device, "--out": results},
                "licence": licence, "recipe": recipe}

    track, licence, build = pick_track(ask_fn=ask_fn, log=log)
    device = pick_device(ask_fn=ask_fn, log=log)
    out, resume = ask_output(ask_fn=ask_fn, log=log,
                             chosen={"model": model, "track": track})
    search = None
    if resume:
        # The answers this walk never asks for, and the ones it asked for before the directory was
        # named. `ask_output` has already shown any disagreement and taken the person's decision;
        # applying it here is what makes that decision reach the printed command.
        record = runrecord.read_quiet(out) or {}
        model = record.get("model") or model
        search = record.get("search")
        recorded_track = record.get("track")
        if recorded_track and recorded_track != track:
            track, licence = recorded_track, licence_for_track(recorded_track)
            if build:
                # Dropped rather than left to run. It would build a corpus this run cannot use,
                # and it would do it on disk, before the abliteration was refused for using it.
                build = None
                log("")
                log("  The track build is dropped with it: the completed trials were scored on")
                log(f"  {track}, so building another corpus would produce one this run cannot use.")
    trials = ask_trials(ask_fn=ask_fn, log=log)

    options = {
        "--model": model,
        "--track": track,
        "--out": out,
        "--device": device,
        "--trials": trials,
        # THE GUIDED MODE TAKES THE SCREEN, AND SAYS SO ON THE LINE. Decision Q-42 D4: a person who
        # chose to be led gets the dashboard, and a script gets the compact panel beside its log.
        # Passed as the flag rather than as a hidden mode for the reason the `--resume` comment
        # below gives: the command printed here is the command that runs, so somebody who copies
        # it gets the same screen, and somebody who does not want it can delete four words.
        "--panel": "full",
    }
    note = size_the_probe_for(device, options)
    if note:
        log("")
        _say(note, log=log)
    if search:
        options["--search"] = search
    if resume:
        # A flag, not a hidden mode. The whole contract of this file is that the command it prints
        # is the command it runs, so a decision taken in the walkthrough has to appear on the line.
        options["--resume"] = True
    plan = {"command": "kageyoshi", "options": options, "licence": licence, "recipe": recipe}
    if build:
        # Built first, because `--track` is checked before the model is downloaded and a track
        # that does not exist yet fails that check. Shown as its own command for the same reason
        # `then` is: it is a separate program with its own flags.
        plan["first"] = build
    if recipe == "brain":
        # Two commands, shown as two. The convert step is a separate program with its own flags,
        # and pretending one line does both would break the rule this file exists to keep.
        plan["then"] = {"command": "convert",
                        "options": {out: True, "--quantise": "Q4_K_M"}}
    return plan


#: Named counts, because "These are the 2 commands" reads like a machine wrote it.
_HOW_MANY = {2: "two", 3: "three"}


def steps(plan):
    """Every command the plan will run, in the order it will run them.

    `first` builds something the main command needs (a track), `then` does something with what it
    produced (a conversion). Each stays a separate command with its own flags, because collapsing
    them into one line would print something nobody could type, and printing what cannot be typed
    is the one thing this file is built not to do.
    """
    ordered = []
    if plan.get("first"):
        ordered.append(plan["first"])
    ordered.append({"command": plan["command"], "options": plan["options"]})
    if plan.get("then"):
        ordered.append(plan["then"])
    return ordered


#: The commands whose pre-flights this board can run. The other recipes print a help page or score
#: a model, and neither has an abliteration's pre-flight surface.
_EDITS_A_MODEL = ("kageyoshi", "abliterate", "auto")

#: What each mark means, and the order they sort in when the board has to lead with the worst.
PASS, ADVISORY, FAIL = "✓", "!", "✗"


def _machine_rows(plan):
    """What is true of this machine, as (mark, label, detail)."""
    import shutil

    rows = []
    # NO CARD ROW HERE, and that is the second version of this function. The first asked
    # `device_available` and printed a card row beside the `_preflight_device` row below, so a
    # machine with no card reported the same fact twice, in two wordings, from two mechanisms,
    # and the counts line said "2 failing" over one problem. The check owns the question; this
    # group carries what the check does not.
    device = str(plan["options"].get("--device", "") or "").strip().lower()
    if device:
        rows.append((PASS, "device", device))

    out = plan["options"].get("--out")
    if out:
        try:
            probe = Path(out)
            probe = probe if probe.exists() else (probe.parent or Path("."))
            free = shutil.disk_usage(probe).free
        except OSError:
            free = None
        if free is not None:
            # No threshold, because nothing here knows the model's size yet. A number the reader
            # can judge beats a verdict this cannot support: `crashsafe` refuses on the real
            # figure once the weights are resident, and that is where the arithmetic belongs.
            rows.append((PASS, "disk", f"{free / 1e9:.1f} GB free where the output goes"))
    return rows


def _probe_budget(args, log=print):
    """`capability.refuse_a_slow_probe`, given what `run_parsed` gives it."""
    from . import capability

    capability.refuse_a_slow_probe(
        getattr(args, "device", "cpu"),
        getattr(args, "capability_n", 0),
        getattr(args, "capability_max_new", 512),
        spec=getattr(args, "capability_eval", ""),
        allowed=getattr(args, "slow_probe_ok", False),
        log=log)


def _checked_rows(plan):
    """The tool's own pre-flights, run BEFORE the confirm instead of after it.

    WHY THIS IS THE REAL FIX AND THE BOARD IS THE DECORATION. Every check below already existed and
    every one of them ran inside `run_parsed`, which the guided mode reaches only once the person
    has answered "Run it? y". So the walk asked for a commitment and then went to find out whether
    the run was possible, and a refusal that was decidable from the command line alone arrived
    after the decision it should have informed.

    The checks are CALLED rather than reimplemented, and their refusals are caught rather than
    parsed. A second copy of "is this device usable" that agreed with the first until one of them
    was edited is this project's most repeated defect; here the board is a different presentation
    of the same function, so it cannot drift from what the run will do.

    An empty list when `cli` will not import: the guided mode runs on a base install where torch
    may be absent, and a board that cannot be built is a reason to say so rather than to stop.
    """
    from .parser import build_parser, split_mode

    try:
        from . import cli
    except Exception:       # a base install with no torch. `_board` says so rather than lying.
        return None

    argv = _argv_for({"command": plan["command"], "options": plan["options"]})
    try:
        _bankai, rest = split_mode(argv)
        args = build_parser().parse_args(rest)
        cli.resolve_model(args)
        cli.resolve_track(args, log=lambda *_a, **_k: None)
    except SystemExit as e:
        return [(FAIL, "the command", str(e))]

    #: (label, check). Ordered cheapest and most local first, which is the order `run_parsed`
    #: itself uses and the order a reader wants: a fault in what they typed before a fault in
    #: what is on the disk.
    checks = (
        ("the numbers", cli._preflight_numbers),                    # noqa: SLF001
        ("the model", cli._preflight_model),                        # noqa: SLF001
        ("how much it generates", cli._preflight_generation_budget),  # noqa: SLF001
        ("the device", cli._preflight_device),                      # noqa: SLF001
        ("resuming", cli._preflight_recovery),                      # noqa: SLF001
        ("every flag does something", cli._preflight_dead_knobs),   # noqa: SLF001
        ("where it goes", cli._preflight_output),                   # noqa: SLF001
        ("the prompts", cli.refuse_without_a_track),
        ("the prompt files", cli._preflight_datasets),              # noqa: SLF001
        # LAST, exactly as `run_parsed` orders it. Everything above reports a run that CANNOT
        # work; this one reports a run that WOULD work and take an afternoon, and told about
        # both, a person wants the impossible one first.
        #
        # IT WAS MISSING UNTIL A REAL RUN FOUND IT, 2026-09-28. A guided walk on a CPU printed
        # "14 checks, all clear", the operator confirmed, and thirty seconds later the run
        # refused because the capability probe would take four hours. A board that says clear
        # about a run the tool is about to refuse is worse than no board, because the reader has
        # now been told twice and believed the wrong one.
        ("the probe budget", _probe_budget),
    )
    rows = []
    for label, check in checks:
        said = []
        try:
            # Several of these take a log and use it for advisories rather than refusals. Those
            # lines are the check's own words about a run that will work, so they become the row's
            # detail rather than being printed over the board.
            try:
                check(args, log=said.append)
            except TypeError:
                check(args)
        except SystemExit as e:
            rows.append((FAIL, label, str(e.code if isinstance(e.code, str) else e)))
            continue
        except Exception as e:      # a check that crashes is a finding, not a silent pass
            rows.append((ADVISORY, label,
                         f"this check could not run: {type(e).__name__}: {e}"))
            continue
        detail = " ".join(s.strip() for s in said if str(s).strip())
        rows.append((ADVISORY if detail else PASS, label, detail))
    return rows


def _board(plan, log=print):
    """Print the pre-flight, and answer whether anything is wrong with the run.

    COLLAPSED WHEN IT IS CLEAN, and this is the whole argument for the hybrid. A board that prints
    nine ticks before every run is nine lines of ceremony, and a reader learns to skip it; the run
    it is protecting is the one where the tenth line says something. So a clean board is one line
    and a board with something to say is drawn in full.
    """
    if plan["command"] not in _EDITS_A_MODEL:
        return True

    chosen = [(PASS, label, str(value)) for label, value in (
        ("model", plan["options"].get("--model")),
        ("prompts", plan["options"].get("--track")),
        ("output", plan["options"].get("--out")),
        ("trials", plan["options"].get("--trials")),
    ) if value not in (None, "")]
    machine = _machine_rows(plan)
    checked = _checked_rows(plan)

    groups = [("WHAT YOU CHOSE", chosen), ("YOUR MACHINE", machine)]
    if checked is None:
        groups.append(("WHAT WE CHECKED",
                       [(ADVISORY, "not run",
                         ("the deep-learning stack is not installed here, so these cannot be "
                          "checked until it is"))]))
    else:
        groups.append(("WHAT WE CHECKED", checked))

    rows = [r for _name, group in groups for r in group]
    bad = [r for r in rows if r[0] == FAIL]
    advisory = [r for r in rows if r[0] == ADVISORY]

    if not bad and not advisory:
        log("")
        log(f"Pre-flight: {len(rows)} checks, all clear.")
        return True

    # MEASURED, NOT TYPED. A fixed 14 was fine until a label ran to sixteen characters, at which
    # point that row's detail started four columns right of every other row's and the board lost
    # the single edge it is laid out around.
    width = max(len(label) for _mark, label, _detail in rows)
    log("")
    log(f"  PRE-FLIGHT · {plan['options'].get('--model', '')}")
    for name, group in groups:
        if not group:
            continue
        log("")
        log(f"  {name}")
        log("")
        for mark, label, detail in group:
            # THE FIRST LINE ONLY, on the row. A refusal from a pre-flight is a formatted block
            # with a "what to do" list and indented commands in it, and `say` leaves indented
            # lines alone deliberately, so feeding the whole thing through here produced a
            # hundred-column line inside a board. The rest of it is printed below, intact,
            # because it is the part somebody acts on.
            head = str(detail).strip().splitlines()[0] if str(detail).strip() else ""
            first, *rest = say.lines(head, columns=max(40, say.width() - width - 8)) or [""]
            log(f"    {mark}  {label:<{width}} {first}".rstrip())
            for line in rest:
                log(f"       {'':<{width}} {line}")
    log("")
    log(f"    {len(rows)} checks · {len(rows) - len(bad) - len(advisory)} pass · "
        f"{len(advisory)} advisory · {len(bad)} failing")
    for _mark, label, detail in bad:
        body = str(detail).strip().splitlines()
        if len(body) > 1:
            log("")
            log(f"  {label}, in full:")
            log("")
            for line in body:
                log(f"    {line}")
    if bad:
        log("")
        _say("Nothing has run. A failing check is a run that cannot work, so the command below "
             "is printed for the record and starting it would waste the time it asks for.",
             log=log, indent="  ")
    return not bad


def present(plan, *, ask_fn=input, log=print):
    """Show the commands, the licence, and ask. Returns the first command line, or None if declined."""
    ordered = steps(plan)
    lines = [render_command(s["command"], s["options"]) for s in ordered]
    # BEFORE THE COMMAND AND BEFORE THE CONFIRM. The board answers whether the run can work, and
    # an answer that arrives after the person has committed is not an answer, it is a receipt.
    clean = _board(plan, log=log)
    log("")
    if len(ordered) > 1:
        count = _HOW_MANY.get(len(ordered), str(len(ordered)))
        _say(f"These are the {count} commands that will run, in order. They are also what goes "
             f"in a method section, and what to type next time:", log=log)
    else:
        _say("This is the command that will run. It is also the one to put in a method section, "
             "and the one to type next time:", log=log)
    log("")
    for line in lines:
        log(f"    {line}")
    log("")
    if plan.get("licence") and plan["licence"] != "yours":
        _say(f"The prompts are under {plan['licence']}. Attribution is required, and if that "
             f"includes a non-commercial term then commercial use of the corpus is not "
             f"permitted.", log=log)
        log("")
    # `[y/N]`, AND IT IS THE ONLY PROMPT IN THE WALK THAT DEFAULTS TO NO.
    #
    # Every question before this one is safe to press Enter through, and four of them in a row
    # train exactly that reflex: "press Enter to take the default" is printed under each. Then the
    # same keystroke starts an abliteration, which spends GPU hours and, on a rented card, money.
    # The walk taught a habit and then charged for it. A peer session found this by writing a
    # script that said in its own comments it would not confirm the run, sending bare newlines,
    # and starting a run on a real card.
    #
    # `[Y/n]` is right for a command somebody typed deliberately with all its flags. It is wrong
    # at the end of a menu whose whole design is that Enter is safe. The asymmetry is the same one
    # the refusals follow: one extra keystroke from a person who has just read the command and
    # wants it, against an irreversible outcome at the end of a sequence of reversible ones.
    #
    # `prints_only` is the one exception, and it is the exception for the same reason: a plan that
    # prints a help page and stops costs nothing, so the safe-default reasoning above does not
    # apply to it, and applying it anyway made the one free entry on the recipe menu a dead end.
    # A bare Enter there answered "Nothing was run." to somebody who had asked to see the flags.
    # A SOFT GATE, in the words the rest of the tool uses. A failing check means the run cannot
    # work and the person is still the one who decides, so the question changes rather than
    # disappearing: "anyway" is the whole of the warning a second time, in one word.
    question = "Run it?" if clean else "Run it anyway?"
    if not confirm(question, default=bool(plan.get("prints_only")), ask_fn=ask_fn, log=log):
        log("Nothing was run. The commands above still work if you want them later.")
        return None
    return lines[0]


def _duration(seconds):
    """`23m 04s`, or None when nothing timed it."""
    if seconds is None:
        return None
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def _what_it_cost(out):
    """The run's own numbers, read back out of the artefact it wrote, as printable lines.

    READ, NOT REMEMBERED. The figures are in `abliteration.json`, which is the file a reader is
    told to check and the one every other surface quotes. Recomputing them here, or carrying them
    out of the run in a variable, would be a second account of the same measurement.
    """
    import json

    try:
        with open(Path(out) / "abliteration.json", encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return []

    from .modelcard import _number

    lines = []
    before, after = record.get("baseline_refusals"), record.get("post_bake_refusals")
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        lines.append(f"      refusals     {_number(before, 'rate')}  →  {_number(after, 'rate')}")
    kl = record.get("post_bake_kl")
    if isinstance(kl, (int, float)):
        lines.append(f"      drift        {_number(kl, 'kl'):<12} "
                     f"the cost of the edit, on held-out prompts")
    return lines


#: What can follow a finished abliteration, as (label, command, options builder). Commands rather
#: than a menu of verbs, because the rule this file keeps is that anything it runs is printed
#: first, and a follow-on is no exception.
def _next_steps(plan, out):
    steps_after = []
    if not missing_conversion_tools():
        steps_after.append(("Convert it for llama.cpp", "a single GGUF file, runs on CPU",
                            {"command": "convert",
                             "options": {out: True, "--quantise": "Q4_K_M"}}))
    steps_after.append(("Check it against a second set",
                        "does it hold up outside the prompts the search saw",
                        {"command": "score",
                         "options": {"--model": out, "--eval": "default/bad_eval_ds",
                                     "--out": "scores.json"}}))
    return steps_after


def finished(plan, *, seconds=None, ask_fn=input, log=print):
    """The screen at the end of a successful run. Returns a follow-on step, or None.

    WHAT WAS THERE BEFORE: nothing. A guided run that worked printed the tool's own last line and
    exited, so somebody who had waited twenty-three minutes for a thing they had never made before
    got no statement of what they now had, where it was, or what it cost.

    THE ONE RESTRAINT, and it is the reason the numbers are laid out the way they are. The
    celebration belongs to finishing, never to the figures. The refusal pair is measured on this
    run's own prompts; the drift line carries "on held-out prompts" precisely so a reader can see
    which of the two survives contact with anything else. A screen that cheered the refusal number
    would be teaching people to quote it.

    `Nothing, I am done` is the highlighted default, against the design, which highlighted the
    conversion. Somebody who reaches this screen has what they came for, and a menu whose default
    keystroke starts another job is the same trap the confirm before the run was rewritten to
    avoid.
    """
    out = plan["options"].get("--out")
    if plan["command"] not in _EDITS_A_MODEL or not out:
        return None

    log("")
    log("  Your model is ready.")
    log("")
    trials = plan["options"].get("--trials")
    took = _duration(seconds)
    detail = ", ".join(x for x in (f"{trials} trials" if trials else None, took) if x)
    log(f"      {out}/{'    ' + detail if detail else ''}".rstrip())
    cost = _what_it_cost(out)
    if cost:
        log("")
        for line in cost:
            log(line)

    options = _next_steps(plan, out)
    if not options:
        return None
    rows = [("Nothing, I am done", "")]
    rows += [(label, f"{note}:  {render_command(s['command'], s['options'])}")
             for label, note, s in options]
    index = choose("What next?", rows, default=0, ask_fn=ask_fn, log=log)
    return None if index == 0 else options[index - 1][2]


#: What a resumable search leaves in `--out`, and the label for each. Checked on disk rather than
#: asserted: see `log_failure`.
RECOVERABLE = (
    ("senbon-study.db", "the persisted study, so completed trials are not lost"),
    ("best-config.json", "best-config.json, the winning configuration"),
)


def log_failure(plan, reason, *, step=None, log=print, crashed=False):
    """What a person needs when a guided run dies partway: what is kept, and the way back in.

    THE FAILURE SCREEN, in the form this codebase can deliver today (critique finding 1, ranked
    first of thirteen). The design draws ten scenes and every one of them succeeds; the thing that
    actually loses hours is the screen nobody drew. `cli.py` records that a traceback out of
    `_save_weights` has, twice, meant hours of card time producing nothing an operator could use.

    Deliberately not a summary of the error. The tool's own message and the traceback are better
    than anything reconstructable here and they are still on the way out; this adds the sentence
    they do not carry, which is that the search is on disk and one command resumes it.

    `step` says WHICH command died, because the advice is only true of one of them, and WHERE that
    step sits decides which advice. A plan can build a track before the search and convert a model
    after it. Telling somebody whose track build failed that their completed trials are safe on
    disk would be a comforting sentence about a search that never started; telling somebody whose
    conversion failed that there is no partial run to recover writes off a model that finished.

    Matched against the plan's own `first` and `then` for exactly that reason. The test used to be
    "this is not the main command", which is true of both of them, so the conversion step got the
    sentence written for the track build and every word of it was wrong.
    """
    log("── the run stopped ───────────────────────────────────────────")
    if reason:
        log(f"  {reason}")
    log("")
    named = (step or {}).get("command")
    out = plan.get("options", {}).get("--out")
    if step is not None and named == (plan.get("first") or {}).get("command"):
        log(f"  It was the '{named}' step that failed, before the search began, so")
        log("  there is no partial run to recover. Once the cause is fixed, this is the command:")
        log("")
        log(f"    {render_command(step['command'], step['options'])}")
        _log_bug_line(log, crashed)
        log("──────────────────────────────────────────────────────────────")
        return
    if step is not None and named == (plan.get("then") or {}).get("command"):
        # THE BRANCH THAT USED TO GIVE THE OPPOSITE ADVICE. The test was "this is not the main
        # command", which is true of the conversion step as well as the track build, and those two
        # sit on opposite sides of the expensive part. A conversion runs after the abliteration has
        # finished and saved, so telling that person there is no partial run to recover writes off
        # a model that is on their disk and invites them to run the whole search again.
        log(f"  It was the '{named}' step that failed, and that runs after the abliteration, so")
        log("  the edit itself finished and the model was saved. Only the conversion is missing.")
        if out:
            log(f"  The edited model is in {out}.")
        log("")
        log("  Once the cause is fixed, this is the command, and nothing before it repeats:")
        log("")
        log(f"    {render_command(step['command'], step['options'])}")
        _log_bug_line(log, crashed)
        log("──────────────────────────────────────────────────────────────")
        return
    kept = _what_survived(out)
    if kept:
        log(f"  What is on disk, in {out}:")
        for line in kept:
            log(f"    {line}")
        log("")
        log("  To pick up where it stopped:")
        log("")
        log(f"    {render_command(plan['command'], {**plan['options'], '--resume': True})}")
        log("")
        log("  That continues the search rather than starting it again, and if the search had")
        log("  already finished it goes straight to baking and saving.")
    elif out:
        # THE CASE THAT USED TO BE TOLD A COMFORTING LIE. A refusal before the search starts, a
        # device pre-flight being the one actually seen, left `--out` empty; the old text still
        # announced a persisted study and offered `--resume`, which would refuse identically and
        # cost the user a second go at nothing. Now the directory is read.
        log(f"  Nothing recoverable was written to {out}, so the run stopped before the search")
        log("  had anything to save. Fix the cause above and run the same command again;")
        log("  --resume would have nothing to resume.")
    else:
        log("  Nothing was written, so there is nothing to recover.")
    _log_bug_line(log, crashed)
    log("──────────────────────────────────────────────────────────────")


def _what_survived(out):
    """The recoverable artefacts that are ACTUALLY in `out`, as display lines.

    Read from the filesystem, never assumed. The old version asserted a persisted study and a
    best-config.json on every failure alike, including the ones that happened before the search
    began, which is a claim about somebody's disk made without looking at it.
    """
    if not out:
        return []
    import pathlib as _pathlib

    root = _pathlib.Path(out)
    return [label for name, label in RECOVERABLE if (root / name).exists()]


def _log_bug_line(log, crashed):
    """Point at a traceback only when one was actually printed.

    A refusal the tool phrased itself raises SystemExit and prints no traceback, and this line used
    to be printed unconditionally. Telling somebody the traceback above is the useful part of a bug
    report, when the screen above holds no traceback, sends them looking for something that was
    never there and makes the whole block read as a crash it was not.
    """
    log("")
    if crashed:
        log("  If this looks like a bug, the traceback above is the useful part of a report.")
    else:
        log("  This was a refusal, not a crash, so the reason above is the whole story.")


def _argv_for(plan):
    """The plan as an argument list, matching the line `render_command` printed for it.

    One function so the printed command and the executed one cannot drift. A guided mode that ran
    something other than what it displayed would be a very good way to hide a mistake, which is
    the reason this file prints the command at all.
    """
    argv = [plan["command"]]
    for flag, value in plan["options"].items():
        if value is True:
            argv.append(flag)
        elif value not in (None, False):
            argv += [flag, str(value)]
    return argv


#: What `senbonzakura interactive --help` prints. This command takes no flags, so there is no
#: argparse parser here to generate it from, and for a long time that meant it answered `--help`
#: by starting the interview instead: the one command written for somebody who has not read the
#: flag list was the only one that would not say what it was. In a pipe it was worse, refusing
#: with the not-a-terminal message, so `senbonzakura interactive --help | less` told a reader
#: nothing at all.
#: Hand-wrapped at 79 columns, because it is printed rather than formatted by argparse. It used to
#: run to 93, which re-wraps into ragged half-lines on an 80-column terminal. The usage line names
#: `[-h]` because this command accepts it, and a usage line that omits a flag the command takes is
#: wrong in the one place a reader trusts it.
HELP = """usage: senbonzakura interactive [-h]

A guided walk through the few choices that decide whether a run means
anything.

`senbonzakura kageyoshi` has dozens of flags. Most have sensible defaults; a
handful decide whether the numbers answer the question you meant to ask, and
`--help` cannot tell you which. This asks those, in order, with a default on
every one.

It never runs anything without first printing the exact command it is
equivalent to, so the run is reproducible and the second time you can type
that command instead.

It takes no options beyond -h, and it needs a terminal, because it reads
answers. A script wants the flags rather than the menu; `senbonzakura --help`
lists them."""


def run(argv=None, *, ask_fn=input, log=print, stdin=None):
    """Entry point for `senbonzakura interactive`."""
    # Before the terminal check on purpose. Asking what a command does is the one question that
    # must be answerable without the conditions for running it, and a reader piping into `less`
    # has no terminal on stdin.
    if argv and ("-h" in argv or "--help" in argv):
        log(HELP)
        return 0
    if not is_tty(stdin or sys.stdin):
        log("senbonzakura: guided mode needs a terminal, and this input is not one.")
        _say("A script wants the flags rather than the menu. `senbonzakura --help` lists them, "
             "and `senbonzakura interactive` on a terminal prints the command for any run.",
             log=log, indent="  ")
        return 2
    try:
        plan = plan_abliteration(ask_fn=ask_fn, log=log)
        line = present(plan, ask_fn=ask_fn, log=log)
    except AbandonedError:
        log("\nStopped. Nothing was changed.")
        return 130
    if line is None:
        return 0
    import time

    from .cli import main as cli_main
    ordered = steps(plan)
    step = ordered[0]
    started = time.time()
    try:
        code = 0
        for i, step in enumerate(ordered):
            if i:
                log("")
                log(f"Now: {step['command']}.")
            # THROUGH `exit_status`, NEVER RAW. `cli.main` returns a delegated command's own
            # return value unchanged, and `track`, `score`, `compass`, `drift` and `coherence`
            # return their RESULT on success rather than a status. A non-empty dict is truthy, so
            # `if code` read every successful run of those five as a failure: a track that built
            # correctly printed TRACK_BUILT and was then followed by "the run stopped ... it was
            # the 'track' step that failed", and the abliteration the user had confirmed never ran.
            #
            # `entry.exit_status` is the one place that knows how to read both conventions, and its
            # own docstring records this defect being fixed at the `__main__` boundary. Calling
            # `cli.main` directly walked straight back into it.
            from .entry import exit_status
            code = exit_status(cli_main(_argv_for(step)))
            # Stop at the first failure. The steps are ordered because each needs what the one
            # before it produced, so carrying on would abliterate against a track that was not
            # built, or convert a model that was never saved.
            if code:
                break
    except KeyboardInterrupt:
        log("")
        log_failure(plan, "You stopped it.", step=step, log=log)
        return 130
    except SystemExit as e:
        # A refusal the tool phrased itself. Its message has already been printed and is better
        # than anything this could add, so the only thing worth appending is the way back in.
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code:
            log("")
            log_failure(plan, None, step=step, log=log)
        raise
    except Exception as e:
        # Deliberately broad, and deliberately not swallowing. The traceback is what a bug report
        # needs and it still goes to stderr; what a person needs on top of it is the sentence
        # saying their GPU hours are not gone. `cli.py` records that a traceback out of
        # `_save_weights` has twice meant hours of card time producing nothing usable.
        log("")
        log_failure(plan, f"{type(e).__name__}: {e}", step=step, log=log, crashed=True)
        raise
    else:
        # A non-zero status is a failure the command reported without raising, and it used to
        # return silently: the guided mode printed the tool's error and then nothing, with the
        # recovery line reserved for exceptions. The two paths now say the same thing.
        if code:
            log("")
            log_failure(plan, None, step=step, log=log)
            return code
        # THE RUN WORKED, AND THE WALK USED TO END HERE IN SILENCE. See `finished`.
        try:
            follow_on = finished(plan, seconds=time.time() - started, ask_fn=ask_fn, log=log)
        except AbandonedError:
            return 0
        if follow_on is None:
            return 0
        log("")
        log("This is the command, and it is the one to type next time:")
        log("")
        log(f"    {render_command(follow_on['command'], follow_on['options'])}")
        log("")
        from .entry import exit_status as _exit_status
        return _exit_status(cli_main(_argv_for(follow_on)))


if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(run)
