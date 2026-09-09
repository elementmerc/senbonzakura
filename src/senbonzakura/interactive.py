# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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

from . import bundled, runrecord

#: Datasets offered by name. Nothing here ships with the package except `default`: the rest are
#: fetched on demand from the Hub, under the licence shown, so a user chooses with the terms in
#: front of them rather than discovering them later.
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
        "key": "advbench",
        "spec": "walledai/AdvBench::train",
        "side": "harmful",
        "title": "AdvBench (Zou et al. 2023)",
        "licence": "MIT",
        # OFFERED SECOND, AND SAYING SO, because it is gated. It was the first harmful choice and
        # therefore the default, and taking the default gave a newcomer with no Hugging Face
        # account an authentication error from a menu that had promised them a yardstick.
        "note": "520 harmful behaviours, the field's common yardstick. GATED on the Hub: it "
                "needs a Hugging Face account and `huggingface-cli login` before it will fetch.",
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


def by_side(side):
    """The corpora that hold `side`, plus a way to name one that is not on the list."""
    entries = [e for e in KNOWN_DATASETS if e["side"] == side]
    entries.append({
        "key": "own", "spec": None, "side": side, "licence": "yours",
        "title": "Something of my own",
        "note": "a text file (one prompt per line), a CSV/JSON with a prompt column, or a Hub id",
    })
    return entries

DEVICES = [
    ("cuda", "an NVIDIA card. Needed to edit a model in any reasonable time"),
    ("cpu", "no card. Fine for scoring and for trying the plumbing, slow for editing"),
    ("mps", "Apple silicon"),
]


class AbandonedError(Exception):
    """The user chose to stop.

    Named with the suffix the linter wants, but it is not an error and is never printed as one:
    Ctrl+C at a prompt is a decision, and answering it with a traceback would be rude.
    """


def is_tty(stream=None):
    stream = stream or sys.stdin
    return bool(getattr(stream, "isatty", lambda: False)())


def quote(value):
    """Shell-quote a value only when it needs it, so the printed command stays readable."""
    text = str(value)
    if text and all(c.isalnum() or c in "-_./:=[]@," for c in text):
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


def ask(question, *, default=None, ask_fn=input, log=print):
    """One free-text question with an optional default."""
    suffix = f" [{default}]" if default is not None else ""
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
    log(question)
    for i, (label, description) in enumerate(options, 1):
        marker = "*" if i - 1 == default else " "
        log(f"  {marker} {i}. {label}")
        if description:
            log(f"       {description}")
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
    """The bundled track is packed at release time, so a source clone has none."""
    if bundled.is_available():
        return
    log("")
    log("  NOTE: no bundled track is installed in this checkout. It is packed at release")
    log("        time, so a wheel has one and a source clone does not. Run")
    log("        `python tools/pack_track.py --track <dir>`, or choose another option.")


def pick_side(side, question, ask_fn=input, log=print):
    """One side of the contrast, as (spec, licence). Licences are shown before the choice."""
    entries = by_side(side)
    options = [(f"{e['title']}  ({e['licence']})", e["note"]) for e in entries]
    entry = entries[choose(question, options, default=0, ask_fn=ask_fn, log=log)]
    if entry["spec"] is None:
        return ask("  Path, or Hub id (add ::split[:N] to slice it)",
                   ask_fn=ask_fn, log=log), "yours"
    return entry["spec"], entry["licence"]


def pick_device(ask_fn=input, log=print):
    index = choose("Where should it run?", [(name, note) for name, note in DEVICES],
                   default=0, ask_fn=ask_fn, log=log)
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
    index = choose("Which prompts should it learn refusal from?", options,
                   default=0, ask_fn=ask_fn, log=log)

    if index == 0:
        warn_if_unbundled(log=log)
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
    index = choose("Which prompts should it be measured on?", options,
                   default=0, ask_fn=ask_fn, log=log)
    if index == 0:
        warn_if_unbundled(log=log)
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
        record = runrecord.read(d)
        if record and record.get("model"):
            what += f"; it was editing {record['model']}"
        out.append((str(d), what, record))
    return out


def resume_plan(path, record, ask_fn=input, log=print):
    """The command that carries on the run in `path`, filled in from what that run recorded.

    THE SCREEN THAT PRINTED A COMMAND NOBODY COULD RUN. This offered
    `senbonzakura kageyoshi --out <dir> --resume`, and `--model` is required, so the one screen
    written to save somebody hours produced an argparse error. Adding `--model` alone would have
    been worse: `--track` has a default, so a resume that omits it carries on one corpus's trials
    while scoring new ones against another, and no artefact says the run changed corpus halfway.

    Both come off the record the run writes before it starts searching. A run from a build that
    predates that record cannot be reconstructed, so its inputs are asked for rather than guessed,
    and the same guard in the abliterator refuses the pair if they disagree with the study anyway.
    """
    options = {}
    if record and record.get("model"):
        options["--model"] = record["model"]
    else:
        log("")
        log(f"  {path} does not say which model it was editing, so it predates the record runs")
        log("  now keep. Resuming with the wrong one would continue this search against a")
        log("  different model, so it has to be named.")
        options["--model"] = ask("  Which model was it?", ask_fn=ask_fn, log=log)
    if record and record.get("track"):
        options["--track"] = record["track"]
    else:
        options["--track"] = ask("  And which track was it scored on?", default="default",
                                 ask_fn=ask_fn, log=log)
    options["--out"] = path
    options["--resume"] = True
    command = "abliterate" if record and record.get("bankai") is False else "kageyoshi"
    return {"command": command, "options": options, "licence": "yours", "recipe": "resume"}


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
    log(f"  Carrying on with {path}. Nothing there is overwritten: the search continues from the")
    log("  trials it already has, and if it had already finished it goes straight to baking.")
    return resume_plan(path, record, ask_fn=ask_fn, log=log)


def ask_output(ask_fn=input, log=print):
    """Where the edited model goes, and what to do when something is already there.

    Returns (path, resume).

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
            return out, True
        # Deliberately no "overwrite" option. Deleting somebody's previous result on their behalf,
        # inside a guided flow they are still learning, is not a choice this should offer; the
        # person can remove the directory themselves and come back.
        log("  Nothing has been changed. Pick another path.")


def plan_abliteration(ask_fn=input, log=print):
    """Walk the questions that decide whether an abliteration run means anything."""
    log("")
    log("Senbonzakura, guided mode. Ctrl+C stops at any point and changes nothing.")
    log("Every answer has a default, shown in brackets; press Enter to take it.")

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
        return {"command": "--help", "options": {}, "licence": None, "recipe": recipe}

    model = ask("\nWhich model? (a Hub id, or a local directory)",
                default="Qwen/Qwen3-1.7B", ask_fn=ask_fn, log=log)

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
    out, resume = ask_output(ask_fn=ask_fn, log=log)
    trials = ask("How many search trials? More is better and slower; 200 is the usual",
                 default="200", ask_fn=ask_fn, log=log)

    options = {
        "--model": model,
        "--track": track,
        "--out": out,
        "--device": device,
        "--trials": trials,
    }
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


def present(plan, *, ask_fn=input, log=print):
    """Show the commands, the licence, and ask. Returns the first command line, or None if declined."""
    ordered = steps(plan)
    lines = [render_command(s["command"], s["options"]) for s in ordered]
    log("")
    if len(ordered) > 1:
        count = _HOW_MANY.get(len(ordered), str(len(ordered)))
        log(f"These are the {count} commands that will run, in order. They are also what goes in a")
        log("method section, and what to type next time:")
    else:
        log("This is the command that will run. It is also the one to put in a method section,")
        log("and the one to type next time:")
    log("")
    for line in lines:
        log(f"    {line}")
    log("")
    if plan.get("licence") and plan["licence"] != "yours":
        log(f"The prompts are under {plan['licence']}. Attribution is required, and if that")
        log("includes a non-commercial term then commercial use of the corpus is not permitted.")
        log("")
    if not confirm("Run it?", default=True, ask_fn=ask_fn, log=log):
        log("Nothing was run. The commands above still work if you want them later.")
        return None
    return lines[0]


def log_failure(plan, reason, *, step=None, log=print):
    """What a person needs when a guided run dies partway: what is kept, and the way back in.

    THE FAILURE SCREEN, in the form this codebase can deliver today (critique finding 1, ranked
    first of thirteen). The design draws ten scenes and every one of them succeeds; the thing that
    actually loses hours is the screen nobody drew. `cli.py` records that a traceback out of
    `_save_weights` has, twice, meant hours of card time producing nothing an operator could use.

    Deliberately not a summary of the error. The tool's own message and the traceback are better
    than anything reconstructable here and they are still on the way out; this adds the sentence
    they do not carry, which is that the search is on disk and one command resumes it.

    `step` says WHICH command died, because the advice is only true of one of them. A plan can
    build a track first and convert a model afterwards, and telling somebody whose track build
    failed that their completed trials are safe on disk would be a comforting sentence about a
    search that never started.
    """
    log("── the run stopped ───────────────────────────────────────────")
    if reason:
        log(f"  {reason}")
    log("")
    if step is not None and step.get("command") != plan.get("command"):
        log(f"  It was the '{step['command']}' step that failed, before the search began, so")
        log("  there is no partial run to recover. Once the cause is fixed, this is the command:")
        log("")
        log(f"    {render_command(step['command'], step['options'])}")
        log("")
        log("  If this looks like a bug, the traceback above is the useful part of a report.")
        log("──────────────────────────────────────────────────────────────")
        return
    out = plan.get("options", {}).get("--out")
    if out:
        log(f"  What is on disk, in {out}:")
        log("    the persisted study, so completed trials are not lost")
        log("    best-config.json, if the search got as far as picking a winner")
        log("")
        log("  To pick up where it stopped:")
        log("")
        log(f"    {render_command(plan['command'], {**plan['options'], '--resume': True})}")
        log("")
        log("  That continues the search rather than starting it again, and if the search had")
        log("  already finished it goes straight to baking and saving.")
    else:
        log("  Nothing was written, so there is nothing to recover.")
    log("")
    log("  If this looks like a bug, the traceback above is the useful part of a report.")
    log("──────────────────────────────────────────────────────────────")


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


def run(argv=None, *, ask_fn=input, log=print, stdin=None):
    """Entry point for `senbonzakura interactive`."""
    if not is_tty(stdin or sys.stdin):
        log("senbonzakura: guided mode needs a terminal, and this input is not one.")
        log("  A script wants the flags rather than the menu. `senbonzakura --help` lists them,")
        log("  and `senbonzakura interactive` on a terminal prints the command for any run.")
        return 2
    try:
        plan = plan_abliteration(ask_fn=ask_fn, log=log)
        line = present(plan, ask_fn=ask_fn, log=log)
    except AbandonedError:
        log("\nStopped. Nothing was changed.")
        return 130
    if line is None:
        return 0
    from .cli import main as cli_main
    ordered = steps(plan)
    step = ordered[0]
    try:
        code = 0
        for i, step in enumerate(ordered):
            if i:
                log("")
                log(f"Now: {step['command']}.")
            code = cli_main(_argv_for(step))
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
        log_failure(plan, f"{type(e).__name__}: {e}", step=step, log=log)
        raise
    else:
        # A non-zero status is a failure the command reported without raising, and it used to
        # return silently: the guided mode printed the tool's error and then nothing, with the
        # recovery line reserved for exceptions. The two paths now say the same thing.
        if code:
            log("")
            log_failure(plan, None, step=step, log=log)
        return code


if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry
    module_entry(run)
