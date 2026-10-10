# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""One contract for every instrument, whoever wrote it and wherever it runs.

WHAT AN INSTRUMENT IS, IN ONE SENTENCE

An instrument answers one question about one model and hands back one result document. `score`
answers "how often does it still refuse", `coherence` answers "how surprised is it by ordinary
prose", and a plugin somebody else writes answers whatever they wrote it to answer.

WHY A CONTRACT AT ALL

Three registries in this tree already do this job informally and none of them knows about the
others. `entry.DELEGATED` maps a command word to the module that runs it. `measure.STAGE_ORDER`
lists the five instruments one command runs together. The checker's `measurement.METRICS` declares
what each number means. Add a sixth instrument and you edit three places, and nothing fails if you
edit two.

So this is the one declaration those three become views of, rather than a fourth list beside them.
Our own instruments are the first plugins; there is no privileged core that gets a shorter path.

THE SHAPE: WHOLE MEASUREMENT IN, WHOLE RESULT OUT

One request goes in, one result comes back, and nothing is said in between. No per-tensor chatter,
no callbacks, no shared handles to a loaded model.

That coarseness is the point rather than a simplification. A plugin that ran out of process but
exchanged thousands of small messages with us about individual weights would be at arm's length on
paper and intimately joined in practice, and "these are two programs" is a claim that has to
survive somebody reading the traffic. A single request and a single reply is a claim anybody can
check by looking at the protocol.

    in process                         out of process
    ----------                         --------------
    us  ->  run(inputs)                us  ->  one JSON line on stdin
    us  <-  one result dict            us  <-  one JSON document on stdout

Both sides go through the same validator. A trusted in-process plugin cannot hand back a shape an
out-of-process one would have been refused for, which is what keeps the two transports one contract
rather than two.

THE TRANSPORT IS A TRUST DECISION, NOT A TIER

In process means code we already trust, so we skip the serialisation and keep a traceback as a
traceback. It does NOT mean code allowed to reach things an out-of-process plugin cannot. If the
only tier that can reach a model is the one we ship, nobody outside can build anything worth
building, and "the community can write their own" becomes a sentence in a README rather than
something that happens.

WHAT THIS MODULE DELIBERATELY DOES NOT DO

It does not enforce anything about who may run what, for the same reason `presets.py` says it
cannot: Python resolves imports while the program is already running, and anything one module can
check another can satisfy. An entry point is a request to import somebody else's code, so a plugin
is trusted exactly as much as everything else installed in the environment and no further. What is
validated is what it hands BACK, at the boundary, which is the half that can be made to hold.

It also does not import torch, transformers or any instrument's module at import time. This module
has to be readable by `entry.py`'s dispatcher and by the checker, neither of which may pay for a
deep-learning stack to print a list of plugin names.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Bumped when a plugin written against the old version would break. A new OPTIONAL field does not
#: bump it; a renamed field, a removed one, or a changed meaning does. Spelled the same way as
#: `events.SCHEMA` because they are the same kind of promise to the same kind of consumer.
PROTOCOL = "senbonzakura-plugin/1"

#: The entry point group a third-party plugin registers itself under. Part of the published
#: contract, so it is a constant rather than a literal typed in two files.
GROUP = "senbonzakura.plugins"

#: The kinds of plugin this version knows how to drive. An `instrument` answers a question about a
#: model and returns a result document. Declared as a closed set so a plugin built for a later
#: version of this contract is refused with a message rather than driven as something it is not.
KINDS = ("instrument",)

#: How long a plugin gets, in seconds, when the caller names no deadline. Generous, because an
#: instrument may hold a model and generate for a long time, and still finite, because baseline
#: section 2.1 has no exception for a subprocess that never comes back.
DEFAULT_DEADLINE_S = 3600.0

#: The longest any deadline may be. A plugin that asks for a week is refused at the boundary rather
#: than granted, because an unbounded wait reached through a config file is still an unbounded wait.
MAX_DEADLINE_S = 86_400.0

#: How long a plugin gets to exit after being asked politely, before it is killed. The same figure
#: and the same reasoning as `ggufrun.SHUTDOWN_GRACE_S`.
SHUTDOWN_GRACE_S = 10.0

#: The largest reply this reads from an out-of-process plugin. A result document is a handful of
#: numbers and a stamp; a gigabyte of it is a plugin that has gone wrong, or one that is trying to
#: exhaust this process, and in both cases the right answer is a refusal rather than a read.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

#: The longest a plugin name may be, and the characters it may use. Narrow on purpose: the name
#: reaches log lines, filenames and the checker's vocabulary, so a name carrying a path separator
#: or a newline is refused where it enters rather than sanitised at every use.
MAX_NAME = 64

#: How many metric names one plugin may declare it produces. Bounded for the same reason
#: `presets.MAX_CLAIMS` is: nothing else here lets an outside source decide how long one of our
#: lists is.
MAX_PRODUCES = 256


class PluginError(Exception):
    """Anything wrong with a plugin or with driving one."""


class PluginContractError(PluginError):
    """The plugin did not keep the contract. Always a fault in the plugin, never the user's."""


class PluginUnavailableError(PluginError):
    """No plugin here answers to that name, with the reason spelled out."""


class PluginTimeoutError(PluginError):
    """The plugin did not finish inside its deadline and was stopped."""


class PluginFailedError(PluginError):
    """The plugin ran, kept the contract, and reported that it could not produce a number.

    A DIFFERENT THING FROM `PluginContractError`, and the distinction is the one `measure.py`
    already draws: an instrument that says "this model is too large for this card" has behaved
    correctly and produced no number, which is worth reporting beside the four that did. A plugin
    that returns a string where a document belongs is broken, and that is worth reporting
    differently because the person who can fix it is not the person running it.
    """


def _name(value, what):
    """A plugin or instrument name, validated where it enters.

    Refused rather than sanitised. A name that has had its slashes stripped is a different name
    from the one the plugin declared, and two plugins whose names differ only in the characters we
    remove would then collide under a name neither of them chose.
    """
    if not isinstance(value, str):
        raise PluginContractError(f"{what} must be a string, and is {type(value).__name__}")
    text = value.strip()
    if not text:
        raise PluginContractError(f"{what} is empty")
    if len(text) > MAX_NAME:
        raise PluginContractError(
            f"{what} is {len(text)} characters, above the {MAX_NAME} this reads")
    if not all(c.isalnum() or c in "-_." for c in text):
        raise PluginContractError(
            f"{what} is {text!r}. A name may hold letters, digits, '-', '_' and '.' and nothing "
            f"else: it reaches log lines, filenames and the checker's metric vocabulary, so a "
            f"path separator or a newline in one is refused here rather than stripped at every "
            f"use.")
    return text


def name_of(value, what="a plugin name"):
    """`_name`, for callers outside this module.

    A thin public face rather than letting other modules reach the underscore name: the rule it
    enforces is part of the contract, so something that is not this file has to be able to apply it
    without the next reader wondering whether it was meant to.
    """
    return _name(value, what)


def deadline_of(value):
    """A deadline in seconds, bounded at both ends, or a refusal naming the bound it broke."""
    if value is None:
        return DEFAULT_DEADLINE_S
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PluginContractError(
            f"a deadline must be a number of seconds, and is {value!r}")
    seconds = float(value)
    if seconds <= 0:
        raise PluginContractError(
            f"a deadline must be above 0 seconds, and is {seconds}. A plugin given no time cannot "
            f"produce a result, and a deadline of 0 reads as 'no deadline' to somebody skimming.")
    if seconds > MAX_DEADLINE_S:
        raise PluginContractError(
            f"a deadline of {seconds} seconds is above the {MAX_DEADLINE_S} this grants. An "
            f"effectively unbounded wait reached through a config file is still an unbounded wait.")
    return seconds


@dataclass(frozen=True)
class Manifest:
    """What a plugin says about itself, validated.

    Deliberately small. Everything here is either needed to drive the plugin or needed to describe
    its number honestly in a report, and nothing is here because it might be useful later: an
    optional field nobody reads is the dead-flag defect arriving from outside the repository, where
    no audit of ours can see it.
    """

    #: The plugin's own name, which is also the key it is registered under.
    name: str
    #: The contract version the plugin was written against.
    protocol: str
    #: What it is. Only `instrument` exists in this version.
    kind: str
    #: One line a person reads in a list of plugins.
    title: str
    #: The metric names it stamps into its result, so a report can say what it will carry before
    #: running it. Sorted, so two installs describe one plugin identically.
    produces: tuple = ()
    #: `in-process` or `subprocess`. Set by whichever transport loaded the plugin rather than by
    #: the plugin, because how it is reached is our decision and not its claim.
    transport: str = ""


_MANIFEST_KEYS = {"name", "protocol", "kind", "title", "produces"}


def validate_manifest(raw, *, plugin, transport):
    """Turn whatever a plugin said about itself into a `Manifest`, or refuse it and say which key.

    This is the boundary for the describe half of the contract, and it is picky about type as well
    as about value for the same reason `presets.validate` is: these strings end up in a published
    report, and a number labelled with whatever a dict happened to hold is worse than no label.
    """
    if not hasattr(raw, "get") or not hasattr(raw, "items"):
        raise PluginContractError(
            f"the plugin '{plugin}' described itself as {type(raw).__name__}, and a manifest must "
            f"be a mapping of keys to values.")
    unknown = sorted(set(raw) - _MANIFEST_KEYS)
    if unknown:
        raise PluginContractError(
            f"the plugin '{plugin}' returned a manifest carrying keys this version does not read: "
            f"{', '.join(unknown)}. Known keys: {', '.join(sorted(_MANIFEST_KEYS))}. Refused "
            f"rather than ignored, because a key nobody reads is something the plugin believes it "
            f"declared. The usual cause is a plugin built against a newer contract than this one.")
    protocol = raw.get("protocol")
    if protocol != PROTOCOL:
        raise PluginContractError(
            f"the plugin '{plugin}' is written against protocol {protocol!r} and this install "
            f"speaks {PROTOCOL!r}. Refused rather than driven anyway: the version is bumped only "
            f"when a field was renamed, removed or given a different meaning, so carrying on would "
            f"mean reading one of those fields wrongly without anything saying so.")
    declared = _name(raw.get("name"), f"the plugin name '{plugin}' declared")
    if declared != plugin:
        raise PluginContractError(
            f"the plugin registered as '{plugin}' calls itself {declared!r}. Refused: a result "
            f"attributed to a name nobody can look up is a result nobody can re-run.")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise PluginContractError(
            f"the plugin '{plugin}' declares kind {kind!r}, and this version drives "
            f"{', '.join(map(repr, KINDS))}. A plugin of an unknown kind is refused rather than "
            f"driven as an instrument, because what would be done with its result is undefined.")
    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        raise PluginContractError(
            f"the plugin '{plugin}' returned no usable 'title'. One line, for the list a person "
            f"reads when they are deciding what to run.")
    produces = _produces(raw.get("produces", ()), plugin=plugin)
    return Manifest(name=declared, protocol=PROTOCOL, kind=kind, title=title.strip(),
                    produces=produces, transport=transport)


def _produces(raw, *, plugin):
    """The metric names a plugin claims, bounded, de-duplicated and sorted."""
    if isinstance(raw, (str, bytes)):
        # A single string is iterable, so without this a plugin declaring `produces: "refusal"`
        # would be read as seven one-character metric names and accepted.
        raise PluginContractError(
            f"the plugin '{plugin}' declares produces as a string. It must be a list of metric "
            f"names, even when there is one of them.")
    out = set()
    read = 0
    try:
        for item in raw:
            # COUNTED ON WHAT WAS READ, NOT ON WHAT WAS KEPT. The same defect `presets._catalogue`
            # records: a generator yielding two names for ever hits de-duplication rather than the
            # cap and spins. The cap is for a source that misbehaves, so it has to bound the read.
            read += 1
            if read > MAX_PRODUCES:
                raise PluginContractError(
                    f"the plugin '{plugin}' declares more than {MAX_PRODUCES} metrics. Reading "
                    f"further is refused: this list is held in memory here and nothing else lets "
                    f"an outside source decide how long one of our lists is.")
            out.add(_name(item, f"a metric name from the plugin '{plugin}'"))
    except TypeError as e:
        raise PluginContractError(
            f"the plugin '{plugin}' declares produces as {type(raw).__name__}, which cannot be "
            f"read as a list of metric names.") from e
    return tuple(sorted(out))


@dataclass(frozen=True)
class Call:
    """One whole measurement, on its way to a plugin.

    `inputs` is passed through rather than modelled. What an instrument needs is the instrument's
    business, and a schema here would have to grow a field every time somebody wrote a plugin that
    asks a question we had not thought of, which is how a contract becomes a bottleneck. What IS
    this module's business is that the envelope around it is well formed and that the reply is
    validated, and both of those are checked.
    """

    instrument: str
    inputs: dict = field(default_factory=dict)
    deadline: float = DEFAULT_DEADLINE_S

    def envelope(self):
        """The request document, which is identical on both transports.

        One shape rather than two is what makes the in-process path a genuine exercise of the
        out-of-process one. A transport that built its own request would let a field rot on the
        side nobody runs in CI.
        """
        return {"protocol": PROTOCOL, "call": "run", "instrument": self.instrument,
                "inputs": dict(self.inputs), "deadline": self.deadline}


def call_for(instrument, inputs=None, *, deadline=None):
    """A validated `Call`, which is how one is built everywhere rather than by hand."""
    if inputs is None:
        inputs = {}
    if not hasattr(inputs, "items"):
        raise PluginContractError(
            f"inputs for {instrument!r} is {type(inputs).__name__} rather than a mapping")
    return Call(instrument=_name(instrument, "the instrument asked for"),
                inputs=dict(inputs), deadline=deadline_of(deadline))


def decode_request(text):
    """A request document back into a `Call`, for the plugin side of the boundary.

    Lives here, in the package the plugin author has installed anyway, so a plugin written in
    Python does not have to re-implement the envelope and drift from it. A plugin written in
    something else reads the four keys from the protocol document instead; they are stable, which
    is the whole of what a version number is for.
    """
    try:
        raw = json.loads(text)
    except (TypeError, ValueError) as e:
        raise PluginContractError(f"the request is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise PluginContractError(
            f"the request is {type(raw).__name__} rather than a JSON object")
    if raw.get("protocol") != PROTOCOL:
        raise PluginContractError(
            f"the request names protocol {raw.get('protocol')!r} and this plugin speaks "
            f"{PROTOCOL!r}")
    if raw.get("call") != "run":
        raise PluginContractError(
            f"the request asks for call {raw.get('call')!r}, and 'run' is the only one that "
            f"carries inputs in this version")
    return call_for(raw.get("instrument"), raw.get("inputs") or {},
                    deadline=raw.get("deadline"))


_RESPONSE_KEYS = {"protocol", "instrument", "ok", "result", "reason"}


def validate_response(raw, *, plugin, instrument):
    """The result document out of a reply, or a refusal naming what was wrong with the reply.

    The one place both transports meet, so a trusted in-process plugin is held to exactly what an
    out-of-process one is held to. Skipping this for code we wrote would mean the path nobody
    outside takes is the only one that is checked, which is the inversion of what a contract is
    for.
    """
    if not hasattr(raw, "get") or not hasattr(raw, "items"):
        raise PluginContractError(
            f"the plugin '{plugin}' replied with {type(raw).__name__}, and a reply must be a "
            f"mapping of keys to values.")
    unknown = sorted(set(raw) - _RESPONSE_KEYS)
    if unknown:
        raise PluginContractError(
            f"the plugin '{plugin}' replied with keys this version does not read: "
            f"{', '.join(unknown)}. Known keys: {', '.join(sorted(_RESPONSE_KEYS))}.")
    if raw.get("protocol") != PROTOCOL:
        raise PluginContractError(
            f"the plugin '{plugin}' replied under protocol {raw.get('protocol')!r} and this "
            f"install speaks {PROTOCOL!r}.")
    answered = raw.get("instrument")
    if answered != instrument:
        raise PluginContractError(
            f"the plugin '{plugin}' was asked for {instrument!r} and replied about "
            f"{answered!r}. Refused: a number filed under the wrong instrument is the defect "
            f"this project has had to withdraw a figure over.")
    ok = raw.get("ok")
    if not isinstance(ok, bool):
        raise PluginContractError(
            f"the plugin '{plugin}' replied with ok={ok!r}, and it must be true or false. A "
            f"truthy string would read as success, which is how a failed run gets quoted.")
    if not ok:
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise PluginContractError(
                f"the plugin '{plugin}' reported a failure for {instrument!r} and gave no "
                f"'reason'. A failure with no sentence beside it is a failure nobody can act on.")
        raise PluginFailedError(
            f"the plugin '{plugin}' could not produce {instrument!r}: {reason.strip()}")
    result = raw.get("result")
    if not hasattr(result, "get") or not hasattr(result, "items"):
        raise PluginContractError(
            f"the plugin '{plugin}' reported success for {instrument!r} and its 'result' is "
            f"{type(result).__name__} rather than a result document. Success with nothing in it "
            f"is the one outcome worse than a refusal, because it reads as a measurement.")
    return dict(result)


def reply(instrument, result=None, *, reason=None):
    """A well formed reply, built by this module so a plugin author never spells one by hand.

    Either a result or a reason, never both and never neither, which is checked here rather than
    left to the caller: a plugin that returns both has not decided whether it succeeded, and this
    is the side of the boundary where that can still be a message instead of a published number.
    """
    name = _name(instrument, "the instrument being replied about")
    if (result is None) == (reason is None):
        raise PluginContractError(
            "a reply carries either a result or a reason, and this call gave "
            + ("both" if result is not None else "neither"))
    if reason is not None:
        return {"protocol": PROTOCOL, "instrument": name, "ok": False, "reason": str(reason)}
    if not hasattr(result, "items"):
        raise PluginContractError(
            f"the result for {name!r} is {type(result).__name__} rather than a mapping")
    return {"protocol": PROTOCOL, "instrument": name, "ok": True, "result": dict(result)}


@dataclass(frozen=True)
class Outcome:
    """What came back, with enough beside it to say where the number was produced.

    The transport is carried because it is a fact about how the figure was taken, in the same way
    the hardware string is. Two runs of one plugin across two transports should agree, and the only
    way anybody can check that is if each result says which one it came through.
    """

    instrument: str
    plugin: str
    transport: str
    result: dict


class Transport:
    """How one plugin is reached. Two of these exist and both answer the same two questions."""

    #: What this transport is called in a `Manifest` and in an `Outcome`.
    label = ""

    def __init__(self, name):
        self.name = _name(name, "the plugin name")

    def describe(self):  # pragma: no cover - the two concrete transports are what is tested
        raise NotImplementedError

    def run(self, call):  # pragma: no cover - as above
        raise NotImplementedError


class InProcess(Transport):
    """A plugin whose code we already trust, called directly.

    No serialisation, and a traceback stays a traceback, which is the whole benefit. The reply
    still goes through `validate_response`, so what this path accepts is exactly what the
    subprocess path accepts.
    """

    label = "in-process"

    def __init__(self, name, provider):
        super().__init__(name)
        missing = [attr for attr in ("describe", "run")
                   if not callable(getattr(provider, attr, None))]
        if missing:
            raise PluginContractError(
                f"the plugin '{self.name}' has no callable "
                f"{' and no callable '.join(missing)}. An in-process plugin exposes describe() "
                f"returning its manifest and run(inputs) returning its reply.")
        self._provider = provider

    def describe(self):
        try:
            raw = self._provider.describe()
        # Deliberately broad. This calls somebody else's code, and every way it can fail means one
        # thing to the caller: the plugin is installed and cannot say what it is.
        except Exception as e:
            raise PluginContractError(
                f"the plugin '{self.name}' raised when asked to describe itself.\n"
                f"    {type(e).__name__}: {e}") from e
        return validate_manifest(raw, plugin=self.name, transport=self.label)

    def run(self, call):
        try:
            raw = self._provider.run(dict(call.inputs))
        except PluginError:
            # A plugin using this module's own vocabulary has already said something useful, and
            # wrapping it would bury that sentence under ours.
            raise
        except Exception as e:
            raise PluginContractError(
                f"the plugin '{self.name}' raised while producing {call.instrument!r}.\n"
                f"    {type(e).__name__}: {e}") from e
        result = validate_response(raw, plugin=self.name, instrument=call.instrument)
        return Outcome(instrument=call.instrument, plugin=self.name, transport=self.label,
                       result=result)


class Subprocess(Transport):
    """A plugin at arm's length: one JSON request on stdin, one JSON document on stdout.

    THE DEADLINE IS NOT OPTIONAL. Baseline section 2.1 has no exception for a subprocess that never
    comes back, and this one is reached by whatever somebody installed, so the wait has to be
    bounded by us rather than by its good behaviour. On a timeout it is asked to stop, given
    `SHUTDOWN_GRACE_S`, and then killed.

    THE REPLY IS BOUNDED. `communicate` reads until the pipe closes, so a plugin that writes for
    ever would otherwise be an out-of-memory in this process. The cap is checked on what was read.

    STDERR IS NOT THE PROTOCOL. A plugin is free to log there and it is carried into the error
    message when something goes wrong, which is what makes a broken plugin diagnosable. Only stdout
    is read as the reply, so logging cannot corrupt a result.
    """

    label = "subprocess"

    def __init__(self, name, argv, *, env=None, cwd=None):
        super().__init__(name)
        argv = [str(a) for a in argv]
        if not argv:
            raise PluginContractError(
                f"the plugin '{self.name}' supplied an empty command line, so there is nothing to "
                f"run.")
        self.argv = argv
        self._env = dict(env) if env is not None else None
        self._cwd = str(cwd) if cwd is not None else None

    def preflight(self):
        """That the command exists, before a caller waits on it.

        Baseline section 2.1 asks for dependency presence to be checked before a long job starts
        rather than halfway through, and the honest message for a missing plugin names the plugin
        and the file, not `FileNotFoundError`.
        """
        exe = self.argv[0]
        if shutil.which(exe) is None and not _looks_like_a_path(exe):
            raise PluginUnavailableError(
                f"the plugin '{self.name}' runs {exe!r}, which is not on PATH and is not a path to "
                f"a file that exists. Install it, or correct the command the plugin registered.")

    def describe(self):
        raw = self._exchange({"protocol": PROTOCOL, "call": "describe"},
                             DEFAULT_DEADLINE_S, what="describe itself")
        return validate_manifest(raw, plugin=self.name, transport=self.label)

    def run(self, call):
        raw = self._exchange(call.envelope(), call.deadline,
                             what=f"produce {call.instrument!r}")
        result = validate_response(raw, plugin=self.name, instrument=call.instrument)
        return Outcome(instrument=call.instrument, plugin=self.name, transport=self.label,
                       result=result)

    def _exchange(self, request, deadline, *, what):
        """One request, one reply, bounded in time and in size."""
        self.preflight()
        try:
            proc = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=self._env, cwd=self._cwd)
        except OSError as e:
            raise PluginUnavailableError(
                f"the plugin '{self.name}' could not be started: {e}") from e
        payload = json.dumps(request)
        try:
            out, err = proc.communicate(payload.encode("utf-8"), timeout=deadline)
        except subprocess.TimeoutExpired as e:
            self._stop(proc)
            raise PluginTimeoutError(
                f"the plugin '{self.name}' was given {deadline:g} seconds to {what} and did not "
                f"reply, so it was stopped. Nothing it may have written is read, because a "
                f"half-written reply is indistinguishable from a complete one that happens to be "
                f"short.") from e
        except OSError as e:
            self._stop(proc)
            raise PluginContractError(
                f"writing the request to the plugin '{self.name}' failed: {e}") from e
        if len(out) > MAX_RESPONSE_BYTES:
            raise PluginContractError(
                f"the plugin '{self.name}' replied with {len(out)} bytes, above the "
                f"{MAX_RESPONSE_BYTES} this reads. A result document is a handful of numbers and "
                f"a stamp, so this is a plugin that has gone wrong rather than a large result.")
        text = out.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0 and not text:
            raise PluginContractError(
                f"the plugin '{self.name}' exited {proc.returncode} without replying when asked "
                f"to {what}." + _stderr_tail(err))
        if not text:
            raise PluginContractError(
                f"the plugin '{self.name}' exited 0 and wrote nothing to stdout when asked to "
                f"{what}. Silence and success are different claims and this one has to be the "
                f"second." + _stderr_tail(err))
        try:
            return json.loads(text)
        except ValueError as e:
            raise PluginContractError(
                f"the plugin '{self.name}' replied with something that is not JSON when asked to "
                f"{what}: {e}." + _stderr_tail(err)) from e

    def _stop(self, proc):
        """Ask, wait, then kill. A plugin left running would hold whatever it had opened."""
        try:
            proc.terminate()
            try:
                proc.wait(timeout=SHUTDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=SHUTDOWN_GRACE_S)
        except OSError:
            # It has already gone. Reported nowhere because there is nothing to report: the caller
            # is on its way to raising about the thing that actually went wrong.
            pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


#: How much of a plugin's stderr is carried into an error message. Enough to hold a traceback's
#: last frames, bounded because this reaches a log and a bug report.
STDERR_TAIL_BYTES = 4000


def _stderr_tail(err):
    """The tail of a plugin's stderr, as a sentence to append, or nothing when it said nothing."""
    if not err:
        return ""
    text = err.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    if len(text) > STDERR_TAIL_BYTES:
        text = "..." + text[-STDERR_TAIL_BYTES:]
    return f"\n  Its own output on stderr:\n{text}"


def _looks_like_a_path(exe):
    """Whether a command names a file rather than something to find on PATH.

    `shutil.which` already resolves a path that exists, so this only has to answer for one that
    does not: a bare name is a PATH lookup that failed, and a name with a separator in it is a
    file that is not there. Both are refusals; they get different sentences.
    """
    return Path(exe).exists()


def _entry_points():
    from importlib.metadata import entry_points

    # Sorted by name, so two plugins are read in one order on every machine and a clash between
    # them is reported the same way twice rather than following whatever order the filesystem gave.
    return sorted(entry_points(group=GROUP), key=lambda ep: (ep.name, ep.value))


def _transport_for(name, provider):
    """Which transport a plugin asked for, read off what it exposes rather than off a flag.

    A plugin that exposes `argv()` wants to be run at arm's length and tells us the command line;
    one that exposes `run()` wants to be called directly. Both are the same contract and the choice
    is the plugin author's, because they are the one who knows whether their code is ours.
    """
    if callable(getattr(provider, "argv", None)):
        try:
            argv = provider.argv()
        except Exception as e:
            raise PluginContractError(
                f"the plugin '{name}' raised when asked for its command line.\n"
                f"    {type(e).__name__}: {e}") from e
        return Subprocess(name, argv)
    if callable(getattr(provider, "run", None)):
        return InProcess(name, provider)
    raise PluginContractError(
        f"the plugin '{name}' exposes neither argv() nor run(), so there is no way to reach it. "
        f"A plugin declares its transport by which of the two it offers: argv() to be run as a "
        f"separate process, run(inputs) to be called in this one.")


def installed():
    """Every plugin registered here, as `{name: Transport}`, refusing a broken registration.

    A FAILED LOAD IS LOUD, for the reason `presets.packs` gives: the only way a plugin gets
    installed is that somebody chose to install it, so swallowing its import error would mean a run
    that silently measured four things while its operator believed it measured five, with no line
    anywhere saying otherwise.
    """
    found = {}
    for ep in _entry_points():
        name = _name(ep.name, "an installed plugin's registered name")
        if name in found:
            raise PluginContractError(
                f"two plugins are installed under the same name '{name}', so which one produced a "
                f"number would depend on install order. Uninstall one.")
        try:
            provider = ep.load()
        # Deliberately broad: `load()` imports somebody else's module, so the failure can be
        # anything that import raises, and all of them mean the plugin is present and unusable.
        except Exception as e:
            raise PluginContractError(
                f"the plugin '{name}' is installed and could not be loaded, so nothing it measures "
                f"is available.\n"
                f"    {type(e).__name__}: {e}\n"
                f"  It registers {ep.value!r} under the '{GROUP}' entry point group. Reported "
                f"rather than ignored, because a run that quietly left an instrument out would "
                f"look exactly like a run that had never been asked for it.") from e
        found[name] = _transport_for(name, provider)
    return found


def default_instruments():
    """The instruments this project ships, read off the registry that already lists them.

    DERIVED RATHER THAN TYPED, which is the point of the row this module implements. A fourth list
    of instrument names would drift from the three that exist, and the drift would be silent: the
    tool would still run, with one instrument missing from one view of it.

    `entry.DELEGATED` is the registry that says which module owns a command, and it is the lightest
    of the three, importing nothing heavier than `banner`. The names are the command words a person
    types, which is also what they are called everywhere else a user meets them.
    """
    from .entry import DELEGATED

    return tuple(sorted(DELEGATED))


def resolve(name, *, extra=None):
    """The transport for one plugin name, or a refusal that lists what is here.

    `extra` lets a caller add plugins that are not installed, which is how a test drives one and
    how the console will eventually drive one from a path. It is a mapping of name to transport and
    it takes precedence over an installed plugin of the same name, deliberately: a caller that
    named a transport explicitly meant that one.
    """
    wanted = _name(name, "the plugin asked for")
    extra = dict(extra or {})
    if wanted in extra:
        return extra[wanted]
    here = installed()
    if wanted in here:
        return here[wanted]
    known = sorted({*here, *extra})
    if known:
        raise PluginUnavailableError(
            f"no plugin called '{wanted}' is available here. Installed: {', '.join(known)}.")
    raise PluginUnavailableError(
        f"no plugin called '{wanted}' is available here, and none is installed. A plugin "
        f"registers itself under the '{GROUP}' entry point group; this project's own instruments "
        f"are reached as commands, with `senbonzakura --help`.")


def describe_all(*, extra=None):
    """Every available plugin's manifest, sorted, with a broken one's reason in place of it.

    One plugin that cannot describe itself must not hide the others, because the list is what a
    person reads to find out what they can run. So the failure is carried as the entry's value
    rather than raised: the caller renders it, and `resolve` is still loud when somebody actually
    asks for that plugin.
    """
    def described(transport):
        try:
            return transport.describe()
        except PluginError as e:
            return e

    return {name: described(transport)
            for name, transport in sorted({**installed(), **dict(extra or {})}.items())}


def run(name, inputs=None, *, deadline=None, extra=None):
    """Drive one plugin, whichever transport it is on, and hand back its result.

    The one call a caller needs. Which transport ran it is in the `Outcome` rather than in the
    caller's hands, which is what makes "the contract is identical across the tiers" a property of
    the code rather than a claim in a document.
    """
    transport = resolve(name, extra=extra)
    call = call_for(name, inputs, deadline=deadline)
    return transport.run(call)


def protocol_document():
    """The contract, as the data a plugin author needs rather than as prose.

    Kept beside the implementation and returned rather than written into a markdown file that would
    then have to be remembered: `docs/reference/plugins.md` quotes this, and a test holds the page
    against it, so the documented protocol cannot drift from the one that runs.
    """
    return {
        "protocol": PROTOCOL,
        "group": GROUP,
        "kinds": list(KINDS),
        "request": {"protocol": PROTOCOL, "call": "run | describe",
                    "instrument": "the name being asked for, on a run",
                    "inputs": "an object the instrument defines",
                    "deadline": "seconds, on a run"},
        "reply_ok": {"protocol": PROTOCOL, "instrument": "the name asked for",
                     "ok": True, "result": "an object this project does not model"},
        "reply_failed": {"protocol": PROTOCOL, "instrument": "the name asked for",
                         "ok": False, "reason": "one sentence a person can act on"},
        "manifest": {"protocol": PROTOCOL, "name": "the registered name", "kind": KINDS[0],
                     "title": "one line", "produces": ["metric names this stamps"]},
        "limits": {"max_response_bytes": MAX_RESPONSE_BYTES,
                   "max_deadline_seconds": MAX_DEADLINE_S,
                   "default_deadline_seconds": DEFAULT_DEADLINE_S,
                   "max_name_characters": MAX_NAME,
                   "max_produces": MAX_PRODUCES},
    }


def serve(provider, *, stdin=None, stdout=None):
    """Run one request against a provider and write the reply. The plugin side's whole main().

    Here rather than in every plugin, so an author writes the measurement and nothing else. It
    reads one request, answers it, and returns an exit status; it never loops, because the protocol
    is one request per process and a loop would be the chatty boundary this contract exists to
    avoid.

    IT ALWAYS WRITES A REPLY. A plugin that raises and exits non-zero with nothing on stdout is
    diagnosable only from its stderr, and the caller then has to decide whether the silence meant
    a crash or a refusal. Turning the raise into a well formed failure reply makes that one
    outcome instead of two.
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    # READ BOUNDED, for the same reason the caller bounds what it reads from us: the request
    # arrives from another process and nothing about it is ours to trust.
    text = stdin.read(MAX_RESPONSE_BYTES + 1)
    if len(text) > MAX_RESPONSE_BYTES:
        json.dump({"protocol": PROTOCOL, "instrument": "unknown", "ok": False,
                   "reason": f"the request is above the {MAX_RESPONSE_BYTES} bytes this reads"},
                  stdout)
        return 2
    instrument = "unknown"
    try:
        raw = json.loads(text) if text.strip() else {}
        if raw.get("call") == "describe":
            json.dump(provider.describe(), stdout)
        else:
            call = decode_request(text)
            instrument = call.instrument
            json.dump(provider.run(dict(call.inputs)), stdout)
    # Deliberately broad, and the docstring above says why: the alternative is a caller that cannot
    # tell a crash from a refusal.
    except Exception as e:
        json.dump(reply(instrument, reason=f"{type(e).__name__}: {e}"), stdout)
        return 1
    return 0
