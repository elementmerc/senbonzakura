# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What a half-finished run says about itself, written at the start rather than at the end.

WHY THIS EXISTS

`--resume` continues the persisted study and takes everything else from the command line, so a
resumed run reads whatever `--model` and `--track` it is given this time. The study does not
remember, and `abliteration.json` (which does record the model) is only written once a run
finishes, which is the one case nobody needs to resume.

Two things went wrong because of that, one on each surface:

- The guided mode's "carry on with a run you already have" screen printed
  `senbonzakura kageyoshi --out <dir> --resume`, and `--model` is required, so the one screen
  written to save somebody hours produced a command argparse rejects outright.
- On the non-interactive path the failure is quieter and worse. `--track` defaults to `track`,
  so a resume that omits it carries on a study whose trials were scored on one corpus using
  another, and nothing in the artefact says the run changed corpus halfway.

So a run writes a small record of its own inputs before it starts searching. Plain JSON, read
without optuna or torch, because the guided mode runs on a base install where neither exists.
"""
import json
from pathlib import Path

from .crashsafe import atomic_write

#: Beside the study it describes. Deliberately not inside `abliteration.json`: that file is the
#: record of a FINISHED run and is written last, and this one has to survive the run being killed.
NAME = "run.json"

#: The inputs a resumed run must not change, and what each one decides. Anything else (trials,
#: device, seed) can legitimately differ on a second leg; these two decide what was measured.
PINNED = {
    "model": "the model the completed trials were scored on",
    "track": "the corpus the completed trials were scored on",
    # ADDED after panel finding S10. The Optuna study is named `senbon-<search>` and the database
    # it lives in is NOT named for the search, so two strategies share one file. Resuming with a
    # different `--search` therefore asks for a study name that does not exist yet, and
    # `load_if_exists` CREATES it: a full fresh search, from trial zero, under a log line saying
    # "(resuming)". Every other guard declines correctly by its own terms, because the build guard
    # is gated on the study having trials and this record pinned only the model and the corpus.
    "search": "the search strategy whose trials are in the study being resumed",
}


#: What a finished or half-finished run leaves in `--out`. Presence of any of these means the
#: directory is not empty in the way that matters: something already used it as a destination.
RUN_ARTEFACTS = ("abliteration.json", "best-config.json", "trials.json", "model.safetensors",
                 "model.safetensors.index.json", "config.json")


def occupied_by(out):
    """The artefacts of a previous run sitting in `--out`, sorted. Empty when it is safe to write.

    THE MISTAKE THIS PROJECT HAS LOST THREE RESULTS TO. On 2026-08-12 three separate defects were
    traced to one mechanism: a previous run's output sitting in the directory while something
    reported the work already done. The failure is quiet by construction. A resumed study skips
    straight to bake and save; a re-run overwrites some files and leaves others; and the artefact
    that describes the run ends up describing two runs, with no field that says so.

    Deliberately a list of what was found rather than a boolean, because the message a person can
    act on names the files. "The directory is not empty" sends them to look; "these four artefacts
    are a previous run" tells them what they are about to lose.

    `run.json` is deliberately NOT on the list. It is written by this module before the search
    starts, so counting it would make every run report its own directory as occupied by a previous
    one.
    """
    d = Path(out)
    if not d.is_dir():
        return []
    found = [name for name in RUN_ARTEFACTS if (d / name).exists()]
    # Sharded weights are named per shard, so the exact filename is not knowable in advance.
    if any(d.glob("model-*-of-*.safetensors")):
        found.append("model-*.safetensors")
    return sorted(found)


def write(out, *, model, track, **extra):
    """Record the inputs of the run now starting in `out`. Overwrites any earlier record.

    Overwriting rather than refusing is deliberate: `--resume` is guarded separately by
    `refuse_across_inputs`, and a fresh run into a directory that has been cleared out should not
    be blocked by a file describing a run that no longer exists.
    """
    record = {"model": model, "track": track, **extra}
    with atomic_write(str(Path(out) / NAME)) as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


class UnreadableError(Exception):
    """A record is there and cannot be read. Distinct from there being none.

    THE DEFECT THIS SPLITS APART, found by the 2026-09-09 panel in code written that morning.
    `read` returned None for four different conditions: no file, no permission, truncated JSON,
    and JSON that parsed to something other than an object. `refuse_across_inputs` returned
    silently on None, so "there is no record, which is fine" and "the record is damaged, which is
    not" were the same value and produced the same behaviour. The guard turned itself off on the
    one input it could not vouch for, and the next line overwrote the damaged file with the new
    run's inputs, so the evidence was gone as well.

    A truncated `run.json`, which is what an interrupted sync off a GPU box leaves, therefore
    meant: the guard declines to check, the record is destroyed, and the study carries on
    optimising one corpus's completed trials against another corpus's prompts. That is the exact
    consequence `refuse_across_inputs` exists to prevent, reached through the guard rather than
    around it.
    """


def read(out):
    """The record in `out`, or None when there is none.

    Raises `Unreadable` when a file is present and cannot be read as a record. Absence is a fact
    about a directory; damage is a fact about a file, and a caller that cannot tell them apart
    cannot decide correctly. `read_quiet` is for the callers that genuinely only want to decorate
    a menu.
    """
    path = Path(out) / NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise UnreadableError(f"{path} exists and could not be read: {e}") from e
    try:
        record = json.loads(text)
    except json.JSONDecodeError as e:
        raise UnreadableError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(record, dict):
        raise UnreadableError(f"{path} holds {type(record).__name__}, not an object")
    return record


def read_quiet(out):
    """The record, or None for absent OR damaged. Only for callers with nothing at stake.

    The guided mode's menu uses this to write "it was editing X" beside a resumable run. A damaged
    record there costs a decoration, not a decision, and refusing to draw a menu because one entry
    is unreadable would be worse than drawing it without that line.
    """
    try:
        return read(out)
    except UnreadableError:
        return None


def mismatches(record, values):
    """Which pinned inputs disagree, as (field, recorded, given) triples.

    A field the record does not carry is skipped rather than treated as a change: records written
    by an older build are incomplete, not wrong, and refusing on their absence would turn every
    resume of an existing run into a failure at exactly the moment resuming matters.
    """
    out = []
    for field in PINNED:
        was = (record or {}).get(field)
        now = values.get(field)
        if was is not None and now is not None and str(was) != str(now):
            out.append((field, was, now))
    return out


def refuse_across_inputs(out, values):
    """Stop a `--resume` that would carry on one run's trials with another run's inputs.

    Raises SystemExit with the two commands the person can actually choose between, because the
    fix is never "try again": it is either point at what the study was built on, or start a fresh
    directory. Silence here means a study whose trials were scored on two different corpora,
    reported as one number, with nothing in the artefact saying so.
    """
    try:
        record = read(out)
    except UnreadableError as e:
        # REFUSED, not skipped. This is the branch that used to be silent, and the caller
        # overwrites the record on the next line, so declining here destroyed the only evidence of
        # what the completed trials were scored on.
        raise SystemExit(
            f"--resume refuses this directory: {e}\n"
            f"  {NAME} records the model and the track the completed trials were scored on, and "
            f"this copy cannot be read, so there is no way to tell whether this run's inputs "
            f"match them. Resuming anyway would risk scoring one corpus's trials against "
            f"another and reporting the two as one number.\n"
            f"\n"
            f"  If you know what this study was built on, delete {NAME} and pass --model and "
            f"--track explicitly; the run writes a fresh record and the study is untouched.\n"
            f"  If you do not, start a fresh run in a directory of its own, which keeps both.") from e
    if record is None:
        return None
    bad = mismatches(record, values)
    if not bad:
        return record
    lines = [f"--resume refuses this directory: {out} holds a run that used different inputs."]
    for field, was, now in bad:
        lines.append(f"  --{field}: the completed trials used {was!r}, and this run was given "
                     f"{now!r} ({PINNED[field]}).")
    lines.append("")
    lines.append("  Resuming would score one corpus's trials against another and report the two")
    lines.append("  as one result. Either carry on with what the study was built on:")
    lines.append("")
    lines.append("    " + " ".join(f"--{field} {record[field]}" for field in PINNED
                                   if record.get(field) is not None))
    lines.append("")
    lines.append("  or start a fresh run in a directory of its own, which keeps both.")
    raise SystemExit("\n".join(lines))
