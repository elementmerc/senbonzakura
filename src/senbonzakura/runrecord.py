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


def read(out):
    """The record in `out`, or None when there is not a readable one.

    None for every failure, including a truncated or hand-edited file. A record is a convenience
    for reconstructing a command; a malformed one must not stop a run that would otherwise work,
    and the guard below treats a missing record as "cannot check" rather than as "no mismatch".
    """
    try:
        text = (Path(out) / NAME).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


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
    record = read(out)
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
