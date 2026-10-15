# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Several behaviours' figures in one score file, without any of them destroying the others.

THE DEFECT THIS EXISTS BECAUSE OF

`score` ends with `atomic_write(a.out)` and `json.dump(res, f)`, which replaces the whole file.
That is correct for one behaviour and fatal for four: the first workbench release draws refusal,
sycophancy, honesty and entity knowledge together on one card, and the card reads them out of one
`score.json`. Behaviours do not share a generation pass, so four behaviours are four runs, and with
one writer that replaces the file, run four leaves a file holding only behaviour four. The card
would then draw one figure and three "not measured" boxes, with nothing anywhere saying that the
three had in fact been measured and then deleted. A measurement silently destroyed by the next one
is the worst shape of failure this project has: it costs GPU hours and leaves no trace.

WHAT A SET IS, AND WHY THE TOP-LEVEL `metrics` BLOCK IS DERIVED

A behaviour set holds one entry per behaviour, each entry being that run's own result document
verbatim, with its own `n`, its own corpus, its own provenance and its own stamps. The top-level
`metrics` block is then rebuilt from those entries on every write, never edited in place.

That is what makes the writer idempotent by construction rather than by care. Re-running a
behaviour replaces its entry and the whole `metrics` block is recomposed, so the second run of
behaviour two cannot leave the first run's slot behind; and the same set of entries always
composes to the same bytes, whatever order they arrived in, because the maps are built in sorted
key order.

WHAT MAY DIFFER BETWEEN TWO BEHAVIOURS IN ONE SET, AND WHAT MAY NOT

`baseline.PINNED` names the fields that decide whether two measurements are comparable. Within one
set, three of them legitimately differ and the rest may not:

    input_digest   each behaviour has its own corpus. Sycophancy prompts are not refusal prompts
    partition      each corpus has its own held-out boundary
    seeds          each method draws its interval its own way, or draws no samples at all

    precision      the same weights read at bf16 and at nf4 are two readings, and a card that put
    prompt_format  them side by side would be comparing the quantiser with the behaviour
    tool_version   the edit and the scorer both live here, so two versions are two instruments
    model          weights are the one thing the whole card is about

The first three are not a comparison the card makes, so they are recorded as legitimately varying
and left alone. The last four are refused on mismatch, loudly, naming both values. Merging across
them would produce a card that reads as one measurement of one model and is not.

WHY A SELF-INVALIDATED RESULT IS REFUSED RATHER THAN MERGED

`workbench/core/artefacts.py` checks `self_invalidated` at the DOCUMENT level and withholds every
figure underneath it, deliberately: a `tamper` run that disowned its own measurement still wrote a
complete, tidy-looking per-metric block, and a panel reading `metrics.<name>.value` would have
drawn it. That rule is right and it is why a self-invalidated behaviour cannot join a set. One
run's invalidation would condemn the other three honest figures sitting beside it, and nothing in
the file would say which. So the merge refuses and names the fix, which is to keep that run in its
own file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from senbonzakura_check.measurement import METRICS_KEY

from .crashsafe import atomic_write

#: The field an artefact declares its own shape in, as `workbench/core/artefacts.read_json` reads it.
SCHEMA_FIELD = "schema"

#: What a behaviour set declares itself to be. Versioned, so a viewer built against this shape and
#: handed a later one refuses with the version it got rather than reading half the fields.
SCHEMA = "senbonzakura-behaviour-set/1"

#: Where the per-behaviour entries live.
BEHAVIOURS_KEY = "behaviours"

#: The keys of the entries, sorted, so a reader has the roster without walking the map.
ROSTER_KEY = "behaviour_keys"

#: The identity fields every behaviour in one set shares, recorded once.
SHARED_KEY = "pinned_shared"

#: THE DOCUMENT-LEVEL INVALIDATION FIELD, as `workbench/core/artefacts.INVALIDATION_FIELD` spells
#: it. A COPY, and the copy is deliberate: the reader lives in the viewer package and the writers
#: are in the core, so importing it here would run the dependency backwards, from the thing that
#: produces artefacts to the thing that draws them. `tests/test_a_behaviour_set_merges.py` fails if
#: the two spellings ever disagree, which is the same enforcement `measurement.SEPARATION_ESTIMATORS`
#: uses for the same reason.
INVALIDATION_FIELD = "self_invalidated"

#: A behaviour key travels in a query string, a picker and a filename, so it is held to a small safe
#: alphabet. ONE SOURCE OF TRUTH: `workbench/card/behaviours.KEY_PATTERN` imports this rather than
#: spelling its own, because a key the card accepts and the writer refuses is a behaviour that can be
#: registered and never measured.
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,39}$")

#: The pinned fields that legitimately take a different value for each behaviour in a set.
MAY_DIFFER = ("input_digest", "partition", "seeds")

#: The pinned fields every behaviour in a set must agree on, read off the stamps themselves.
MUST_AGREE = ("precision", "prompt_format", "tool_version")

#: Why some pinned fields differ inside one set, in the words that go into the artefact.
MAY_DIFFER_DOC = (
    "the pinned fields on which this file's behaviours legitimately differ. Behaviours do not "
    "share a generation pass: each one has its own corpus, its own held-out boundary and its own "
    "way of drawing an interval, so one value for any of these would be a claim about the others "
    "that is not true. Every figure carries its own value in its own metrics slot. The fields NOT "
    "listed here are checked to agree before a behaviour is allowed to join, because a card that "
    "drew two precisions or two tool versions as one measurement would be comparing the instrument "
    "with the model."
)

#: The size this writer refuses a set above, so a malformed or runaway file is not parsed. Far above
#: anything real: a four-behaviour set is a few tens of kilobytes, and the viewer's own cap is 64 MB.
MAX_SET_BYTES = 8 * 1024 * 1024


class BehaviourSetError(Exception):
    """A behaviour could not join a set without making a claim the measurements do not support.

    Raised rather than warned, and raised BEFORE anything is written. A warning printed over a
    destructive write is read by nobody: the file outlives the terminal, and the whole purpose here
    is that no run's result is lost to the next run's write.
    """


def _require_mapping(value, *, where):
    if not isinstance(value, dict):
        raise BehaviourSetError(
            f"{where} is {type(value).__name__}, and a behaviour set is built from objects of named "
            f"fields. Something other than a senbonzakura run produced it.")
    return value


def check_key(behaviour):
    """The behaviour key, or a refusal naming what a key looks like."""
    if not isinstance(behaviour, str) or not KEY_PATTERN.match(behaviour):
        raise BehaviourSetError(
            f"{behaviour!r} is not a usable behaviour key. A key is lowercase letters, digits and "
            f"hyphens, starting with a letter, at most 40 characters, because it appears in the "
            f"address a card is requested from and in the picker a reader ticks.")
    return behaviour


def metric_slots(doc, *, where):
    """The stamped metric slots of one result, or a refusal saying there are none to draw.

    A BEHAVIOUR WITH NO STAMPED METRIC CONTRIBUTES NO FIGURE, and is refused rather than stored.
    The card reads `metrics.<metric>.<estimator>` and nothing else, so an unstamped result would
    join a set, take up a row, and draw as "not measured" with no reason a reader could act on.
    """
    block = _require_mapping(doc, where=where).get(METRICS_KEY)
    if not isinstance(block, dict) or not block:
        raise BehaviourSetError(
            f"{where} carries no stamped metrics block, so it holds no figure a card can read and "
            f"nothing would be drawn for it. A measuring command records its figures through "
            f"`measurement.stamp`; this result did not, so there is nothing to add to the set.")
    unstamped = sorted(slot for slot, entry in block.items() if not isinstance(entry, dict))
    if unstamped:
        raise BehaviourSetError(
            f"{where} has the metric slots {unstamped} holding something other than a stamp block, "
            f"so their provenance cannot be read and they must not be merged.")
    return block


def shared_identity(slots, *, where):
    """The identity `MUST_AGREE` fields of one result, or a refusal when its own slots disagree.

    An absent field is refused rather than read as a wildcard, which is the same direction
    `baseline.comparability` takes: a field not recorded on one side cannot be shown to match, and
    treating unknown as agreement is how two instruments come to be drawn as one.
    """
    identity = {}
    for field in MUST_AGREE:
        seen = sorted({json.dumps(block.get(field), sort_keys=True) for block in slots.values()})
        if len(seen) > 1:
            raise BehaviourSetError(
                f"{where} stamps more than one {field} across its own metric slots ({', '.join(seen)}), "
                f"so there is no single value for the set to agree with. One result is one reading.")
        value = json.loads(seen[0])
        if value is None or value == "":
            raise BehaviourSetError(
                f"{where} records no {field}, so whether it matches the behaviours already in the "
                f"set cannot be shown, and unknown is not the same as agreeing. The measuring "
                f"command fills this through `stamps.pinned`.")
        identity[field] = value
    return identity


def model_of(doc, *, where):
    """The model string a result was taken on, or a refusal. The one thing a whole card is about."""
    model = _require_mapping(doc, where=where).get("model")
    if not isinstance(model, str) or not model.strip():
        raise BehaviourSetError(
            f"{where} names no model, so a card built from it could not say which weights any of "
            f"its figures describe. A measuring command records the checkpoint it was pointed at.")
    return model.strip()


def _compose_metrics(entries):
    """The set's `metrics` block, rebuilt from the entries, or a refusal on a slot two behaviours claim.

    Built in sorted key order at both levels, so the same entries always serialise to the same
    bytes: iteration order of a map must not reach a result file.
    """
    metrics, owner = {}, {}
    for key in sorted(entries):
        for slot, block in sorted(metric_slots(entries[key], where=f"the {key!r} entry").items()):
            if slot in metrics:
                raise BehaviourSetError(
                    f"{owner[slot]!r} and {key!r} both stamp the metric slot {slot!r}, so a reader "
                    f"could not tell whose figure it is and one would silently stand for the other. "
                    f"Two behaviours measuring one quantity need two estimators, stamped "
                    f"`by_estimator=True`, or two files.")
            metrics[slot], owner[slot] = block, key
    return metrics


def _refuse_invalidated(doc, *, where):
    invalidation = doc.get(INVALIDATION_FIELD)
    if invalidation:
        raise BehaviourSetError(
            f"{where} disowned its own measurement: {invalidation} A set is read at the document "
            f"level, so joining it would withdraw every other behaviour's figure in the same file "
            f"and nothing in the file would say which of them was the broken one. Keep this run in "
            f"its own output path, fix what it reports, and run it again.")


def merge(existing, result, *, behaviour):
    """The set that holds `result` as `behaviour`, beside whatever `existing` already held.

    `existing` is None for a file that is not there yet. Pure: it touches no disk, so every refusal
    below is testable without a model, a corpus or a generation pass.
    """
    check_key(behaviour)
    where = f"the {behaviour!r} result"
    _require_mapping(result, where=where)
    _refuse_invalidated(result, where=where)
    slots = metric_slots(result, where=where)
    identity = shared_identity(slots, where=where)
    model = model_of(result, where=where)

    entries = {behaviour: result}
    if existing is not None and existing != {}:
        _require_mapping(existing, where="the file already at this path")
        declared = existing.get(SCHEMA_FIELD)
        if declared != SCHEMA:
            raise BehaviourSetError(
                f"the file already at this path declares the schema {declared!r} rather than "
                f"{SCHEMA!r}, so it is not a behaviour set and merging into it would guess which "
                f"behaviour its figures belong to. A single-behaviour result file cannot be adopted, "
                f"because nothing in it says which behaviour it holds. Write the set to a path of "
                f"its own, or re-run the earlier pass into that path with its own behaviour named.")
        _refuse_invalidated(existing, where="the set already at this path")
        held = _require_mapping(existing.get(BEHAVIOURS_KEY), where="the set's behaviours block")
        for key, entry in held.items():
            check_key(key)
            _require_mapping(entry, where=f"the set's {key!r} entry")
        was = existing.get(SHARED_KEY)
        if held and not isinstance(was, dict):
            raise BehaviourSetError(
                "the set already at this path records no shared identity, so whether this "
                f"behaviour was measured on the same model cannot be shown. Delete the file and "
                f"measure the behaviours again, or write this one to a path of its own.")
        for key, entry in held.items():
            if key == behaviour:
                # THE IDEMPOTENT RE-RUN, and the only entry a merge is allowed to replace. Its old
                # metric slots go with it, because the block below is rebuilt from the entries
                # rather than edited, so a renamed estimator cannot leave the previous slot behind.
                continue
            entries[key] = entry
        if held:
            kept = model_of(existing, where="the set already at this path")
            if kept != model:
                raise BehaviourSetError(
                    f"the set already at this path holds behaviours measured on {kept!r} and this "
                    f"result was measured on {model!r}. One card names one model, so merging these "
                    f"would draw two models' figures as one measurement. Use a separate output path "
                    f"per model.")
            for field in MUST_AGREE:
                before = was.get(field)
                if before != identity[field]:
                    raise BehaviourSetError(
                        f"the set already at this path was measured with {field} {before!r} and "
                        f"this result with {identity[field]!r}. Those are two instruments, not two "
                        f"behaviours of one model, and a card drawing them together would report "
                        f"the difference between them as a property of the model. Re-measure the "
                        f"other behaviours under the same conditions, or use a separate path.")

    return {
        SCHEMA_FIELD: SCHEMA,
        "model": model,
        ROSTER_KEY: sorted(entries),
        SHARED_KEY: dict(sorted(identity.items())),
        "pinned_may_differ": list(MAY_DIFFER),
        "pinned_may_differ_doc": MAY_DIFFER_DOC,
        METRICS_KEY: _compose_metrics(entries),
        BEHAVIOURS_KEY: {key: entries[key] for key in sorted(entries)},
    }


def read(path):
    """The set at `path`, None when there is nothing there, or a refusal naming what is wrong.

    AN UNREADABLE FILE IS A REFUSAL AND NOT AN EMPTY SET. Reading it as empty is exactly the
    destructive write this module exists to prevent: a file that will not parse is far more likely
    to be a run killed partway through its write than a file nobody wants.
    """
    target = Path(path)
    try:
        size = target.stat().st_size
    except FileNotFoundError:
        return None
    except OSError as e:
        raise BehaviourSetError(
            f"{target.name} is there and cannot be examined ({type(e).__name__}: {e}), so whether "
            f"it holds other behaviours' figures is unknown and it will not be overwritten.") from e
    if size > MAX_SET_BYTES:
        raise BehaviourSetError(
            f"{target.name} is {size:,} bytes and a behaviour set is read up to {MAX_SET_BYTES:,}. "
            f"A four-behaviour set is tens of kilobytes, so this is not one, and it is refused "
            f"before parsing rather than handed to a parser.")
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise BehaviourSetError(
            f"{target.name} could not be read as UTF-8 text ({type(e).__name__}: {e}), so it will "
            f"not be overwritten. A run killed partway through its write leaves a file like this.") from e
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError as e:
        raise BehaviourSetError(
            f"{target.name} is not valid JSON ({e}), so whether it holds other behaviours' figures "
            f"is unknown and it will not be overwritten. Look at the file, then delete it if it is "
            f"the remains of a killed run.") from e


def write(path, result, *, behaviour):
    """Add `result` to the set at `path` as `behaviour`, and return the set that was written.

    Atomic through `crashsafe.atomic_write`, so a reader sees either the previous set or the whole
    new one. The read, the merge and every refusal happen before a single byte is written.
    """
    merged = merge(read(path), result, behaviour=check_key(behaviour))
    with atomic_write(path) as f:
        json.dump(merged, f, indent=2)
    return merged
