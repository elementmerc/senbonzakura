# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A behavioural probe somebody else wrote, in a shape this tool can check before it runs it.

WHY A FORMAT AND NOT JUST A FILE

The strategy this project is built on says a domain is owned by whoever owns the workflow, not by
whoever writes the most content. Metasploit did not write the most exploits; it defined the module
format and the console, and other people wrote the modules. The parts of that here are a routing
point, a console, distribution, and a contribution format. This is the contribution format, and it
was the one part that did not exist.

A probe is the unit somebody else can contribute: a set of items, a rule for grading them, and
enough provenance that the number it produces can be argued with. It is deliberately small. If
contributing needs a pull request against our source, almost nobody will, and the format exists
precisely so that the answer to "can I add a behaviour to measure" is a file rather than a patch.

WHAT A PROBE IS

    my-probe/
        probe.toml     what it measures, how to grade it, where it came from
        items.jsonl    the items themselves

The manifest names one of this tool's grading rules rather than shipping code. That is a
deliberate limit: a contributed probe cannot execute anything, so running somebody's probe is
reading their data, not running their program. A format that accepted a grading function would be
a plugin system, and a plugin system in a tool people download to edit model weights is a
different security posture than this project wants.

WHAT IS REFUSED, AND WHY THAT IS THE LOAD-BEARING PART

A contribution format on a tool like this one is an obvious way to post a harmful corpus and call
it a benchmark. So a probe declares its content class, and a probe carrying prompt-shaped field
names is refused whatever it declares. This is the same boundary the prompt-artefact gate enforces
on the repository, arriving at the other end: that gate stops harmful material leaving, and this
stops it arriving dressed as a measurement.

The refusal is on the SHAPE, not on a judgement about the text. Nothing here reads the items and
decides whether they are harmful, because that would be a classifier nobody validated, and this
project does not ship instruments it has not measured.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import tomllib

#: The manifest's filename. One name, so a directory either is a probe or is not.
MANIFEST = "probe.toml"

#: Field names a probe's items may not use, because they are how prompts and model outputs are
#: spelled everywhere in this ecosystem. A probe naming its column `prompt` is either careless or
#: is a corpus in a costume, and neither should load.
#:
#: DELIBERATELY A SEPARATE LIST FROM `tools/ci/check_prompt_artefacts.py`, with a test that pins
#: it as a superset of that one. The two run at opposite ends: that gate scans a repository on its
#: way out, this reads a stranger's file on its way in, and the CI tool is not importable from an
#: installed package. One list in two places drifts, which this project has already paid for, so
#: the agreement is asserted in a test rather than hoped for in a comment.
PROMPT_SHAPED_KEYS = frozenset({
    "prompt", "prompts", "generation", "generations", "text", "texts",
    "completion", "completions", "response", "responses", "output", "outputs",
    "input", "inputs", "instruction", "instructions", "question", "answer", "answers",
    "request", "requests", "messages", "conversation", "conversations",
})

#: What a probe may say it contains. `benign` is the only one that loads. The other values exist
#: so that a refusal can say what was declared rather than only that something was wrong, and so
#: that anybody tempted to contribute a harmful set has to write the word down first.
CONTENT_CLASSES = ("benign", "harmful", "undeclared")

#: The item columns a probe uses. Chosen to sit outside `PROMPT_SHAPED_KEYS`, which is why they
#: are not the obvious `question` and `answer`.
ITEM_KEY = "problem"
REFERENCE_KEY = "reference"


class ProbeError(Exception):
    """A probe this tool will not load, with every reason at once rather than the first."""


@dataclass
class Probe:
    """A loaded probe: what it claims, how to grade it, and the items."""

    name: str
    measures: str
    task: str
    items: list = field(default_factory=list)
    source: dict = field(default_factory=dict)
    version: str = "1"
    path: pathlib.Path | None = None

    def pairs(self, limit=None):
        """(item, reference) in file order, which is what the capability scorer takes."""
        out = [(row[ITEM_KEY], row[REFERENCE_KEY]) for row in self.items]
        return out if limit is None else out[:limit]


def is_probe(path):
    """Whether this directory declares itself a probe. Cheap, and used before a load is attempted."""
    return pathlib.Path(path).is_dir() and (pathlib.Path(path) / MANIFEST).is_file()


def _read_manifest(directory, problems):
    manifest = directory / MANIFEST
    try:
        with open(manifest, "rb") as fh:
            return tomllib.load(fh)
    except OSError as exc:
        problems.append(f"{MANIFEST} cannot be read: {exc}")
    except tomllib.TOMLDecodeError as exc:
        problems.append(f"{MANIFEST} is not valid TOML: {exc}")
    return None


def _check_declaration(doc, problems):
    """The manifest's own claims, before anything is read from disk on their say-so."""
    probe = doc.get("probe") or {}
    for required in ("name", "measures", "task", "items"):
        if not probe.get(required):
            problems.append(
                f"[probe] is missing {required!r}. A probe that does not say what it measures "
                f"produces a number nobody can interpret, which is the failure this format exists "
                f"to prevent")
    from . import capability
    task = probe.get("task")
    if task and task not in capability.TASK_CHOICES:
        problems.append(
            f"[probe] task {task!r} is not a grading rule this tool has. Available: "
            f"{', '.join(capability.TASK_CHOICES)}. A probe names a rule rather than shipping "
            f"code, so that running somebody's probe reads their data rather than runs their "
            f"program")
    return probe


def _check_content_class(doc, problems):
    declared = (doc.get("content") or {}).get("class")
    if declared is None:
        problems.append(
            "[content] class is not declared. It is required rather than defaulted, because the "
            f"default anybody would want is 'benign' and this format is an obvious way to post a "
            f"harmful corpus as a benchmark. Declare one of: {', '.join(CONTENT_CLASSES)}")
    elif declared not in CONTENT_CLASSES:
        problems.append(f"[content] class {declared!r} is not one of {', '.join(CONTENT_CLASSES)}")
    elif declared != "benign":
        problems.append(
            f"[content] class is {declared!r}. This tool loads benign probes only. Harmful "
            f"material belongs in the gated evaluation track, which has a licence and an access "
            f"decision attached to it, not in a contributed measurement")
    return declared


def _check_source(doc, problems):
    """Provenance, so a number this probe produces can be traced rather than trusted."""
    source = doc.get("source") or {}
    if not source.get("licence") and not source.get("license"):
        problems.append(
            "[source] licence is not declared. A probe travels inside other people's runs and "
            "their published results, so its terms have to travel with it")
    if not source.get("citation") and not source.get("dataset") and not source.get("author"):
        problems.append(
            "[source] names neither a dataset, a citation, nor an author, so there is no way to "
            "ask where these items came from")
    if source.get("dataset") and not source.get("revision"):
        problems.append(
            f"[source] dataset {source['dataset']!r} is named without a revision. An unpinned "
            f"dataset is a different dataset on a different day, and the probe's meaning travels "
            f"with the rows")
    return source


def _check_items(directory, probe_decl, problems):
    """The items themselves: they parse, they use our columns, and they can actually be graded."""
    name = probe_decl.get("items")
    if not name:
        return []
    path = (directory / name).resolve()
    # A manifest must not reach outside its own directory. A probe is a thing somebody hands you.
    if not str(path).startswith(str(directory.resolve())):
        problems.append(f"[probe] items {name!r} points outside the probe directory")
        return []
    if not path.is_file():
        problems.append(f"[probe] items {name!r} is not in the probe directory")
        return []

    rows, bad_keys = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            problems.append(f"{name}:{number} is not valid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            problems.append(f"{name}:{number} is not an object")
            continue
        bad_keys |= {k for k in row if str(k).lower() in PROMPT_SHAPED_KEYS}
        missing = [k for k in (ITEM_KEY, REFERENCE_KEY) if k not in row]
        if missing:
            problems.append(f"{name}:{number} has no {' or '.join(missing)}")
            continue
        rows.append(row)

    if bad_keys:
        # Whatever the manifest declared. The shape is the check, because reading the items and
        # judging them would be a classifier nobody here has validated.
        problems.append(
            f"{name} uses the field name(s) {', '.join(sorted(bad_keys))}, which is how prompts "
            f"and model outputs are spelled. Probe items are {ITEM_KEY!r} and {REFERENCE_KEY!r}. "
            f"This is refused on the shape whatever [content] class says")
    if not rows and not problems:
        problems.append(f"{name} holds no items, and an empty probe scores every model the same")
    return rows


def _check_gradeable(rows, task_name, problems):
    """Every reference must be readable by the rule that will grade against it.

    A row whose reference cannot be extracted does not fail, it grades as indeterminate for ever,
    which shrinks the probe silently and reports a rate with a denominator nobody was told about.
    """
    if not rows or not task_name:
        return
    from . import capability
    try:
        task = capability.get_task(task_name)
    except KeyError:
        return                      # already reported by the declaration check
    ungradeable = [i for i, row in enumerate(rows, start=1)
                   if task.reference(row[REFERENCE_KEY]) is None]
    if ungradeable:
        shown = ", ".join(str(i) for i in ungradeable[:5])
        problems.append(
            f"{len(ungradeable)} item(s) have a reference the {task_name!r} rule cannot read "
            f"(first: line {shown}). They would score indeterminate however the model answered, "
            f"so the probe is smaller than it says it is")


def problems_with(path):
    """Every reason this probe will not load, rather than the first one.

    All of them, because a contributor fixing one should already know about the rest. A format
    that reports one problem per attempt teaches people that contributing is tedious, and the
    whole point of having a format is that contributing is not.
    """
    directory = pathlib.Path(path)
    problems = []
    if not directory.is_dir():
        return [(f"{path} is not a directory. A probe is a directory holding {MANIFEST} "
                 f"and its items")]
    if not (directory / MANIFEST).is_file():
        return [f"{path} has no {MANIFEST}, so it does not declare itself a probe"]

    doc = _read_manifest(directory, problems)
    if doc is None:
        return problems
    probe_decl = _check_declaration(doc, problems)
    _check_content_class(doc, problems)
    _check_source(doc, problems)
    rows = _check_items(directory, probe_decl, problems)
    _check_gradeable(rows, probe_decl.get("task"), problems)
    return problems


def load(path):
    """A probe, or a refusal naming every reason it was refused."""
    problems = problems_with(path)
    if problems:
        raise ProbeError(
            f"{path} is not a probe this tool will run:\n"
            + "\n".join(f"  - {p}" for p in problems))
    directory = pathlib.Path(path)
    with open(directory / MANIFEST, "rb") as fh:
        doc = tomllib.load(fh)
    decl = doc["probe"]
    rows = _check_items(directory, decl, [])
    return Probe(name=decl["name"], measures=decl["measures"], task=decl["task"],
                 items=rows, source=doc.get("source") or {},
                 version=str(decl.get("version", "1")), path=directory)
