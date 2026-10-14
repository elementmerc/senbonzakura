# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The instruments this project ships, served across the plugin contract.

WHY THIS EXISTS, AND IT IS NOT CONVENIENCE

`plugins.py` says our own instruments are the first plugins and that there is no privileged core.
A contract nobody on this side actually travels is a documented fiction: the out-of-process path
would be exercised only by people outside the project, which means it would be exercised only after
it had already broken for them. So one of the DEFAULT plugins has to run at arm's length, in CI, on
every change.

This is that. `python -m senbonzakura.pluginhost gate` is a conforming plugin, written against the
same protocol a stranger would write against, reached through the same `Subprocess` transport, and
validated by the same validator.

WHAT AN INSTRUMENT IS GIVEN, AND WHY IT IS AN ARGV

The inputs are the command line the instrument would have been given by hand, plus the path of the
file it will write. That is not a shortcut around designing a schema: it is the vocabulary this
project already uses for exactly this. `measure.stage_argv` is documented as "the command line this
stage would have been given by hand", built rather than executed so it can be printed and tested
without a card, and every instrument's own `--help` is the schema for it.

A second, modelled schema beside that would have to grow a field whenever an instrument grew a
flag, and the two would drift in the usual direction: the new flag would work from the command line
and be unreachable through the contract, which is the dead-flag defect wearing a different hat.

    inputs = {"argv": ["--baseline", "before.json", "--current", "after.json"],
              "out": "run/gate.json"}

    result = {"status": 0, "document": {...}}   # the file the instrument wrote, or null

WHAT IT REFUSES

Only an instrument this package registers in `entry.DELEGATED`, so a request cannot name an
arbitrary module. A bounded argv, because it arrives from another process. And an `out` path that
is read back only if the instrument wrote one: an absent file is reported as absent rather than as
an empty result, because those are different claims and a pipeline that cannot tell them apart will
quote the second.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import sys
from pathlib import Path

from . import plugins

#: How many arguments one request may carry. Far above any instrument's real command line, and
#: finite, because the list arrives from another process and is held in memory here.
MAX_ARGV = 512

#: The longest a single argument may be. A Hub identifier or a path, generously.
MAX_ARG = 4096

#: The largest result document this reads back off disk. Separate from
#: `plugins.MAX_RESPONSE_BYTES` and smaller, because what is read here is one instrument's result
#: file and the reply then has to carry it.
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024

#: One line per instrument, for the manifest. Only the instruments that answer a question about a
#: model are here: `doctor`, `setup` and `fetch` are tools rather than measurements, and serving
#: them across a measurement contract would make the kind field a lie.
#:
#: NOT the authority on which instruments exist. `entry.DELEGATED` is, and `served()` reads it, so
#: an instrument missing a line here is still servable and simply describes itself plainly. A
#: second list that could silently disagree with the first is the defect `plugins.py` was written
#: to end.
TITLES = {
    "score": "how often the model still refuses, on a track's held-out rows",
    "compass": "whether the model still tells a harmful request from a harmless one",
    "coherence": "how surprised the model is by a fixed neutral passage",
    "capability": "what the edit cost on questions with a right answer",
    "drift": "divergence from a baseline model on one ruler",
    "jailbreak": "single-turn jailbreak success, paired with the over-refusal rate",
    "multi-turn": "whether a refusal that holds once gives way when the request is put again",
    "tamper": "whether the edit survives a safety-recovery finetune",
    "leak": "whether the refusal direction actually left the residual stream",
    "diff": "what changed between a checkpoint and the base it claims to come from",
    "gate": "whether a measured property moved outside its interval",
    "baseline": "a measurement kept as the thing later runs are compared against",
    "prereg": "whether a pre-registration and a run agree about what was promised",
    "budget": "whether this machine can finish this run, and roughly how long it would take",
    "check": "known failure modes in an evaluation setup or result",
    "validate": "whether the extracted direction carries refusal at all",
}


class HostError(Exception):
    """The request named something this host will not serve, with the reason."""


def served():
    """Instrument name -> the module that owns it, read off the registry that already says so."""
    from .entry import DELEGATED

    return {name: module for name, (module, _attr) in DELEGATED.items()}


def _argv(raw):
    """The instrument's command line, validated where it enters.

    Bounded in both directions, and every element has to be a string. An integer arriving here
    would reach `argparse` as a non-string and raise from inside the instrument, which turns a
    bad request into what reads like a broken instrument.
    """
    if isinstance(raw, (str, bytes)):
        raise HostError(
            "argv is a string. It must be a list of arguments, even when there is one of them: a "
            "string would be read one character at a time and every character taken as an "
            "argument.")
    try:
        items = list(raw)
    except TypeError as e:
        raise HostError(
            f"argv is {type(raw).__name__}, which cannot be read as a list of arguments") from e
    if len(items) > MAX_ARGV:
        raise HostError(
            f"argv carries {len(items)} arguments, above the {MAX_ARGV} this reads")
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise HostError(
                f"argv[{i}] is {type(item).__name__} rather than a string. Every argument is text "
                f"by the time a command line exists, and converting one here would hide a caller "
                f"that is building the wrong request.")
        if len(item) > MAX_ARG:
            raise HostError(
                f"argv[{i}] is {len(item)} characters, above the {MAX_ARG} this reads")
        out.append(item)
    return out


def _out_path(value):
    """The path an instrument was told to write, or None when it was told nothing."""
    if value is None or isinstance(value, str):
        return value
    raise HostError(f"out is {type(value).__name__} rather than a path")


def _document(path):
    """The file an instrument wrote, or None when it wrote none, bounded on the way in.

    An absent file is None and an unreadable one raises. That split is the point: "the instrument
    produced no document" is a legitimate outcome for `gate`, which answers with an exit status,
    while "the document is there and is not JSON" is a fault somebody has to fix, and reporting
    both as None would let the second travel as the first.
    """
    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        return None
    size = target.stat().st_size
    if size > MAX_DOCUMENT_BYTES:
        raise HostError(
            f"the instrument wrote {size} bytes to {target}, above the {MAX_DOCUMENT_BYTES} this "
            f"reads back. The reply has to carry this document, so an unbounded read here is an "
            f"unbounded reply there.")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HostError(
            f"the instrument wrote {target} and it could not be read as JSON: {e}. Reported rather "
            f"than treated as no document, because a corrupt result and an absent one are "
            f"different facts.") from e


class Instrument:
    """One of this project's instruments, as a plugin provider.

    The same object serves both transports. In process it is handed to `plugins.InProcess`; out of
    process `__main__` below hands it to `plugins.serve`. That is deliberate rather than tidy: the
    two paths run the same code, so a divergence between them would have to be in the transport,
    which is the half that is tested.
    """

    def __init__(self, name):
        self.name = plugins.name_of(name, "the instrument to serve")
        owners = served()
        if self.name not in owners:
            raise HostError(
                f"'{self.name}' is not an instrument this package registers. Servable: "
                f"{', '.join(sorted(owners))}.")
        self.module = owners[self.name]

    def describe(self):
        return {"protocol": plugins.PROTOCOL, "name": self.name, "kind": "instrument",
                "title": TITLES.get(self.name, f"senbonzakura {self.name}"),
                # NO PRODUCES CLAIM. Every instrument stamps its metric through
                # `measurement.stamp` into the document it writes, and reading that mapping here
                # would mean importing the instrument, which is the cost this host exists to keep
                # out of a describe call. An empty list is the honest answer: the contract says
                # produces is what the plugin declares, and this one declares nothing it has not
                # read.
                "produces": []}

    def run(self, inputs):
        if not hasattr(inputs, "get"):
            return plugins.reply(self.name,
                                 reason=f"inputs is {type(inputs).__name__} rather than a mapping")
        try:
            argv = _argv(inputs.get("argv", []))
            out = _out_path(inputs.get("out"))
            status = self._call(argv)
            return plugins.reply(self.name,
                                 result={"status": status, "document": _document(out)})
        except HostError as e:
            return plugins.reply(self.name, reason=str(e))

    def _call(self, argv):
        """Run the instrument and return its exit status, the way `measure.run_stage` does.

        `SystemExit` is a refusal, which is how several of these say no, and it is caught for the
        same reason it is caught there: an instrument declining an input must not end the process
        that asked. Anything else is left to the caller, which out of process is `plugins.serve`
        and which turns it into a failure reply rather than silence.

        THE INSTRUMENT'S OWN STDOUT GOES TO STDERR WHILE IT RUNS, and that line is the whole of
        why this method exists rather than a bare call. Every instrument here prints its verdict
        for a person to read, and out of process stdout is the protocol: `gate` printing
        "gate FAIL: refusal_rate: REGRESSED" lands in front of the JSON reply and the caller then
        refuses the whole exchange as not JSON. Found by running it, on the first out-of-process
        test written.

        `events.py` already argues this the other way round, about its own stream: a run spec's
        success check reads stdout, so one JSON line in the middle of it is a corrupted record. The
        same rule applies here with the roles swapped, and the resolution is the same one. The
        human log is not discarded: stderr is where a log belongs when something else owns stdout,
        and `plugins.Subprocess` carries that stderr into its error messages, so a plugin that dies
        is still diagnosable.
        """
        module = importlib.import_module(f".{self.module}", __package__)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                returned = module.main(argv)
        except SystemExit as e:
            code = e.code
            if isinstance(code, int):
                return code
            # A string exit code is a message, and argparse writes it to stderr. The status is 2
            # by argparse's own convention, and saying so is better than reporting 0, which is
            # what `int(None)` style handling would have produced.
            raise HostError(f"the instrument refused: {code}") from e
        return 0 if returned is None else int(returned)


def provider_for(name):
    """An `Instrument`, as the one way one is built."""
    return Instrument(name)


def in_process(name):
    """The instrument on the in-process transport, for a caller that trusts it."""
    return plugins.InProcess(name, provider_for(name))


def out_of_process(name, *, executable=None):
    """The instrument on the subprocess transport, which is what CI drives.

    `executable` defaults to the interpreter running now rather than to the string "python", so a
    virtual environment's interpreter serves the plugin rather than whatever happens to be on PATH.
    That is the difference between a test that passes here and one that passes everywhere.
    """
    exe = executable or sys.executable
    return plugins.Subprocess(name, [exe, "-m", f"{__package__}.pluginhost", name])


def main(argv=None):
    """Serve one request for the instrument named on the command line.

    One request per process, which is the protocol: a loop here would be the chatty boundary the
    coarse-grained contract exists to avoid.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        # NOT A PROTOCOL REPLY. Nothing has been read from stdin yet, so there is no instrument to
        # file a failure under, and inventing one would put a reply on stdout that names an
        # instrument nobody asked about.
        print(f"usage: python -m {__package__}.pluginhost <instrument>\n"
              f"Serves one of this project's instruments across the plugin contract "
              f"({plugins.PROTOCOL}). One request on stdin, one reply on stdout.",
              file=sys.stderr)
        return 2
    try:
        provider = provider_for(argv[0])
    except (HostError, plugins.PluginError) as e:
        print(str(e), file=sys.stderr)
        return 2
    return plugins.serve(provider)


if __name__ == "__main__":
    raise SystemExit(main())
