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

from . import bundled

#: Datasets offered by name. Nothing here ships with the package except `default`: the rest are
#: fetched on demand from the Hub, under the licence shown, so a user chooses with the terms in
#: front of them rather than discovering them later.
#:
#: The licence column is the reason this list is short. Every entry has to be one whose position
#: has actually been checked, and a corpus nobody has traced does not go on a menu.
KNOWN_DATASETS = [
    {
        "key": "default",
        "spec": "default",
        "title": "The Senbonzakura evaluation track (bundled)",
        "licence": "CC BY-NC 4.0",
        "note": "9,877 prompts, three-way split, ships inside the package. Attribution "
                "required, non-commercial only.",
    },
    {
        "key": "advbench",
        "spec": "walledai/AdvBench::train",
        "title": "AdvBench (Zou et al. 2023)",
        "licence": "MIT",
        "note": "520 harmful behaviours. The field's common yardstick, and small: it has no "
                "harmless arm, so you will need one.",
    },
    {
        "key": "harmful-behaviors",
        "spec": "mlabonne/harmful_behaviors::train",
        "title": "mlabonne/harmful_behaviors",
        "licence": "undeclared upstream",
        "note": "What Heretic's defaults fit on, so choose it to compare like with like. "
                "Declares no licence; see the dataset card.",
    },
    {
        "key": "harmless-alpaca",
        "spec": "mlabonne/harmless_alpaca::train",
        "title": "mlabonne/harmless_alpaca (harmless side)",
        "licence": "undeclared upstream, probably CC BY-NC 4.0",
        "note": "The harmless arm most abliteration tutorials use.",
    },
    {
        "key": "own",
        "spec": None,
        "title": "Something of my own",
        "licence": "yours",
        "note": "A track directory, a text/CSV/JSON file, or a Hub id.",
    },
]

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
            parts.append(flag)
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


def pick_dataset(ask_fn=input, log=print):
    """Which corpus, with its licence shown before the choice rather than after."""
    options = []
    for entry in KNOWN_DATASETS:
        label = f"{entry['title']}  ({entry['licence']})"
        options.append((label, entry["note"]))
    index = choose("Which prompts should it learn refusal from?", options,
                   default=0, ask_fn=ask_fn, log=log)
    entry = KNOWN_DATASETS[index]
    if entry["spec"] is None:
        spec = ask("  Path, or Hub id (add ::split[:N] to slice it)", ask_fn=ask_fn, log=log)
        return spec, "yours"
    if entry["key"] == "default" and not bundled.is_available():
        log("")
        log("  NOTE: no bundled track is installed in this checkout. It is packed at release")
        log("        time, so a wheel has one and a source clone does not. Run")
        log("        `python tools/pack_track.py --track <dir>`, or choose another option.")
    return entry["spec"], entry["licence"]


def plan_abliteration(ask_fn=input, log=print):
    """Walk the questions that decide whether an abliteration run means anything."""
    log("")
    log("Senbonzakura, guided mode. Ctrl+C stops at any point and changes nothing.")
    log("Every answer has a default, shown in brackets; press Enter to take it.")

    model = ask("\nWhich model? (a Hub id, or a local directory)",
                default="Qwen/Qwen3-1.7B", ask_fn=ask_fn, log=log)
    track, licence = pick_dataset(ask_fn=ask_fn, log=log)
    device_index = choose("Where should it run?",
                          [(name, note) for name, note in DEVICES],
                          default=0, ask_fn=ask_fn, log=log)
    device = DEVICES[device_index][0]
    out = ask("\nWhere should the edited model go?", default="abliterated",
              ask_fn=ask_fn, log=log)
    trials = ask("How many search trials? More is better and slower; 200 is the usual",
                 default="200", ask_fn=ask_fn, log=log)

    options = {
        "--model": model,
        "--track": track,
        "--out": out,
        "--device": device,
        "--trials": trials,
    }
    return {"command": "kageyoshi", "options": options, "licence": licence}


def present(plan, *, ask_fn=input, log=print):
    """Show the command, the licence, and ask. Returns the command line, or None if declined."""
    line = render_command(plan["command"], plan["options"])
    log("")
    log("This is the command that will run. It is also the one to put in a method section,")
    log("and the one to type next time:")
    log("")
    log(f"    {line}")
    log("")
    if plan.get("licence") and plan["licence"] != "yours":
        log(f"The prompts are under {plan['licence']}. Attribution is required, and if that")
        log("includes a non-commercial term then commercial use of the corpus is not permitted.")
        log("")
    if not confirm("Run it?", default=True, ask_fn=ask_fn, log=log):
        log("Nothing was run. The command above still works if you want it later.")
        return None
    return line


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
    argv = [plan["command"]]
    for flag, value in plan["options"].items():
        if value is True:
            argv.append(flag)
        elif value not in (None, False):
            argv += [flag, str(value)]
    return cli_main(argv)
