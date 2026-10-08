# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The recorded measurement a later one is judged against, and the rules for judging.

WHY THIS EXISTS, and it is not "because CI should run more"

A number gets checked when somebody wonders about it. That is the difference between a tool used
occasionally and one used constantly, and it is the property this project most clearly lacks. A
gate closes it: a change that moves a measured property outside its interval fails a build, before
anything is promoted, without anyone having to wonder.

THE FAILURE THIS MODULE EXISTS TO PREVENT IS NOT A REGRESSION. It is a gate that compares two
numbers which were never comparable, and gives the result a green tick. This project has already
published a table where exactly that happened: three copies of the prompt renderer drifted, and a
configuration selected under one format was reported under another. Automating that comparison
would convert a subtle error into a passing build, which is worse than having no gate, because a
passing build is evidence and a missing gate is not.

So the refusal is the feature. A baseline records what it was measured under, and a comparison
against a baseline that does not match REFUSES rather than comparing anyway. The same discipline
already exists one layer down in `runrecord`, which stops a `--resume` carrying one run's trials
into another run's inputs, and the shape is deliberately the same: a small set of pinned fields,
a mismatch report, and a refusal that names the two things the reader can actually choose between.

THE GATE FIRES ON THE INTERVAL, NOT THE POINT ESTIMATE, and that is what keeps it switched on. A
gate that fails on noise is disabled within two weeks and never re-enabled, so "the mean moved" is
not a finding here; "the intervals no longer overlap" is. This is also why this work could not come
before the intervals existed.

Torch-free and import-light on purpose: the gate runs on whatever hardware a CI runner has.
"""
import hashlib
import json
import sys
from pathlib import Path

from . import argresolve, say
from .crashsafe import atomic_write

#: Bumped only for a change that makes an older file mean something different. Adding a field is
#: not that: readers use `.get`, and an absent field is reported as unknown rather than as a match.
SCHEMA = "senbonzakura-baseline/1"

#: WHERE A MEASUREMENT CAN BE FOUND, which is a different question from what it was measured on.
#:
#: NAMED FOR WHAT IT ADDRESSES RATHER THAN FOR WHAT IT HASHES, because this file already carries
#: `input_digest` and two similarly named digests with different jobs in one module is a drift
#: this project has paid for once: the other field was called `track_digest` until 2026-09-21,
#: which read as the digest of a track and so only fitted a corpus-scored run, while
#: `coherence.py` invented `passage_digest` for the same slot and the two never met.
#:
#: `input_digest` answers "which input was this number taken on". This answers "which measurement
#: is this", and the two are never interchangeable: a thousand measurements of one corpus share an
#: `input_digest` and have a thousand addresses.
#:
#: WHY IT IS WORTH HAVING AT ALL. A team gating many adapters a day re-measures a baseline it
#: already holds unless there is a name under which the old one can be found, and the cost of
#: re-measuring is what makes a gate too expensive to run on every change, which is this gate's
#: own revisit trigger. An address is also how a baseline can be quoted in a build log and later
#: fetched rather than described.
ADDRESS_FIELD = "artefact_address"

#: Characters of the digest used for each directory level in an addressed store.
#:
#: The layout is `hash[0:2]/hash[2:4]/hash`, per baseline section 12.3, so the same tree works
#: unchanged on a local disk and on object storage, and no directory holds more entries than a
#: filesystem is comfortable with.
_FANOUT = 2

#: What `partition` says for a measurement taken on a fixed input rather than on rows of a corpus.
#:
#: A NAMED SENTINEL, AND DELIBERATELY NOT AN EXEMPTION. The obvious alternative was to let
#: `partition` be absent for probe-shaped metrics, and it is the wrong one: the whole value of
#: this field is that it is never absent, because absence is what `comparability` has to treat as
#: unknown and unknown is what the gate refuses on. Making absence legal for one class of metric
#: makes it legal to forget, and a forgotten partition is how a figure scored on the selection
#: rows got published as a held-out one.
#:
#: So the probe says out loud that it read one fixed passage. The reader meets a value they have
#: to learn once; what they never meet is a hole.
FIXED_PASSAGE = "fixed-passage"

#: What a writer is claiming when it stamps `deterministic=True`, and why the claim is the writer's
#: to make rather than the gate's to infer.
#:
#: THE HOLE THIS FILLS (operator decision, 2026-09-25, option A of three). `from_artefact` demanded
#: an interval from every metric, and `coherence` is one deterministic forward pass over one fixed
#: passage: it has no run-to-run spread for an interval to describe. So the one writer that carried
#: all eight pinned fields was refused on the field it could not honestly supply, and `baseline`
#: accepted nothing this repository could produce, which is the exact condition the module header
#: says it was written to end.
#:
#: The two rejected alternatives are both worse in the same way. Recording a TOKEN-LEVEL interval
#: would fill the field with variation across the passage's tokens, which is a different quantity
#: wearing the interval's name, and this project withdraws numbers over exactly that. Reporting
#: over SEVERAL passages would produce a real interval and change what the metric means, breaking
#: comparability with every coherence figure already published.
#:
#: THE CLAIM: under the conditions `PINNED` fixes, re-running this measurement returns the same
#: number. Not "the spread is small"; the same number. A metric that is merely stable is not
#: deterministic and must record its interval like everything else.
#:
#: THE RESIDUAL, named rather than papered over: `PINNED` does not fix the accelerator, and a
#: bfloat16 forward pass can differ in its last bits between two devices. A deterministic baseline
#: compared across machines can therefore read as a regression on a difference that is hardware
#: rather than model. If that turns up in practice the answer is to pin the device in `PINNED`, not
#: to add a tolerance: a tolerance is an interval that nobody measured.
DETERMINISTIC_MEANS = (
    "declared deterministic: under the pinned conditions this measurement returns the same "
    "number, so it has no interval and is compared exactly")

#: What must agree for two measurements to be comparable, and what each one decides. This is the
#: whole safety argument of the module, so each entry says why it is here rather than only what it
#: is called.
PINNED = {
    "model": "the weights the measurement was taken on",
    "metric": "which property was measured, by identity rather than by display name",
    # NAMED FOR THE SLOT AND NOT FOR THE CORPUS, since 2026-09-21. It was `track_digest`, which
    # reads as "the digest of a track" and so only fits a measurement scored on a track.
    # `coherence.py` measures a fixed passage, invented `passage_digest` for the same slot, and
    # the two never met: `comparability` reports a pinned field absent on BOTH sides as a
    # mismatch, so a coherence figure and a baseline could never be compared and nothing said
    # why. One neutral name for "which input this number was taken on" is what makes them
    # comparable at all.
    "input_digest": "which input the measurement was taken on: the corpus for a scored run, the "
                    "passage for a probe. Neutral because the slot is the same question",
    "partition": "which rows of that corpus, so a measure-partition figure is never compared "
                 f"against a search-partition one. A probe that reads no corpus says "
                 f"{FIXED_PASSAGE!r} rather than leaving this absent",
    "prompt_format": "how the prompt was rendered. Three copies of the renderer drifted once and "
                     "a table was published across the gap",
    "tool_version": "the code that produced it, because the edit and the scorer both live here",
    # ADDED 2026-09-21, by two panel reviewers who reached the same hole from opposite sides.
    #
    # THE ESTIMATOR. `measurement.py` opens by arguing that a metric is not its name, it is its
    # name plus the procedure that produced it, and that two procedures under one name is what
    # withdrew four claims on 2026-08-05. This module was written beside it and pinned the name
    # alone. `refusal_rate` has two declared estimators, one ours and one Heretic's; `kl` has two,
    # one of which the registry says is NOT a KL divergence. Without this field a baseline taken
    # with one ruler and a run taken with the other compare clean and the gate reports a
    # regression that is entirely a change of instrument. `tool_version` is not a substitute:
    # both estimators ship in the same build.
    "estimator": "the procedure that produced the number, not just what it is called. Two "
                 "estimators of one metric are two different numbers",
    # THE PRECISION. The loader is bfloat16 by default and nf4 double-quantised under
    # `--load-in-4bit`, and `device_map='auto'` will split a model across VRAM, host RAM and disk
    # on a small card. A baseline taken on a rented card in bf16 and a candidate taken in 4-bit
    # because that is the only way it fits are not the same measurement, and the nf4 round-trip
    # alone moves per-token likelihood by an amount comparable to what an ablation costs. The
    # abliteration record already captures the device; the gate simply never consulted it.
    "precision": "the numerical precision the measurement was computed at, because a 4-bit "
                 "reading and a bfloat16 one of the same model are different measurements",
}

#: A metric where a LARGER number is better (capability accuracy), against one where a SMALLER
#: number is better (refusal rate, KL divergence). Stored per baseline rather than inferred from
#: the name, because inferring it from a name is how "noncompliance" and "compliance" end up
#: sharing a direction.
HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"
DIRECTIONS = (HIGHER_IS_BETTER, LOWER_IS_BETTER)


class BaselineError(Exception):
    """A baseline that cannot be read, or one that cannot be compared against."""


def record(*, model, metric, direction, point, interval, input_digest, partition,
           prompt_format, tool_version, estimator, precision, seeds, n, deterministic=False,
           extra=None):
    """Build a baseline artefact. Every field is required except `extra`, deliberately.

    A baseline with a hole in it is the thing this module exists to refuse, so there is no way to
    build one by forgetting an argument: a missing field would be silently absent later, and
    absence is exactly what the comparability check has to treat as unknown.

    `deterministic=True` is the ONE case where `interval` may be None, and it is a claim rather
    than an exemption: see DETERMINISTIC_MEANS for what the writer is asserting by setting it.
    """
    if direction not in DIRECTIONS:
        raise BaselineError(
            f"direction must be one of {DIRECTIONS}, not {direction!r}. Whether a number moving up "
            f"is better or worse is not inferable from the metric's name, and guessing it wrong "
            f"turns a regression into a pass.")
    if deterministic:
        if interval is not None:
            raise BaselineError(
                f"this baseline is declared deterministic and also carries an interval "
                f"{interval}. A number that does not vary between runs has nothing for an "
                f"interval to describe, so one of the two claims is wrong, and a gate cannot "
                f"tell which. Drop the interval, or drop the deterministic flag.")
    else:
        if interval is None:
            raise BaselineError(
                "this baseline has no interval and is not declared deterministic, so there is "
                "nothing for the gate to fire on: it compares intervals rather than point "
                "estimates, because a gate that fails on noise is switched off within a "
                "fortnight. Record the interval the measurement reports, or declare the metric "
                "deterministic if it genuinely does not vary between runs.")
        lo, hi = interval
        if lo > hi:
            raise BaselineError(
                f"interval {interval} is inverted: its low bound exceeds its high one.")
        if not (lo <= point <= hi):
            raise BaselineError(
                f"point estimate {point} lies outside its own interval {interval}. One of the two "
                f"was computed on different data from the other, and a gate built on it would "
                f"compare a number to an interval that never described it.")
    if n <= 0:
        raise BaselineError(f"a baseline measured on {n} observations is not a measurement.")
    built = {
        "schema": SCHEMA,
        "model": str(model),
        "metric": str(metric),
        "direction": direction,
        "point": float(point),
        "interval": None if deterministic else [float(interval[0]), float(interval[1])],
        "deterministic": bool(deterministic),
        "n": int(n),
        "input_digest": str(input_digest),
        "partition": str(partition),
        "prompt_format": str(prompt_format),
        "tool_version": str(tool_version),
        "estimator": str(estimator),
        "precision": str(precision),
        # Sorted at the boundary so two runs that chose the same seeds in a different order
        # produce byte-identical baselines.
        "seeds": sorted(int(s) for s in seeds),
        "extra": dict(extra or {}),
    }
    built[ADDRESS_FIELD] = address(built)
    return built


def address(doc):
    """Where this measurement can be found: a digest of everything else it says.

    Two measurements that say the same thing have the same address, which is what makes a baseline
    fetchable rather than re-measurable. The address field itself is excluded, so computing the
    address of a document that already carries one returns the same value rather than a new one:
    without that, stamping would change the content and the content would change the stamp.

    The serialisation is canonical (sorted keys, no spacing, ASCII escapes) because an address
    that depends on how a dictionary happened to be printed is not an address.
    """
    body = {k: v for k, v in doc.items() if k != ADDRESS_FIELD}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_address(spec):
    """Whether this string is an address rather than a path. Deliberately strict.

    A loose test would claim a filename, and resolving a path as an address would send a reader
    looking in a store for a file that is sitting in front of them.
    """
    spec = str(spec)
    if not spec.startswith("sha256:"):
        return False
    rest = spec[len("sha256:"):]
    return len(rest) == 64 and all(c in "0123456789abcdef" for c in rest)


def addressed_path(store, spec):
    """Where an address lives under `store`, as `hash[0:2]/hash[2:4]/hash.json`."""
    if not is_address(spec):
        raise BaselineError(
            f"{spec!r} is not a measurement address. An address is `sha256:` followed by 64 "
            f"lowercase hexadecimal characters, which is what `senbonzakura baseline` prints when "
            f"it records one.")
    digest = str(spec)[len("sha256:"):]
    return (Path(store) / digest[:_FANOUT] / digest[_FANOUT:2 * _FANOUT]
            / f"{digest}.json")


def resolve(spec, store=None):
    """Turn either a path or an address into the path to read, and say why if it cannot.

    The two are told apart by shape rather than by a flag, because a reader who has an address
    pastes the address, and asking them to also say which kind of string they pasted is a question
    the string already answers.
    """
    if not is_address(spec):
        return Path(spec)
    if store is None:
        raise BaselineError(
            f"{spec} is a measurement address and no store was given, so there is nowhere to look "
            f"for it. Pass the directory the measurement was recorded into, or pass its path.")
    return addressed_path(store, spec)


def write_addressed(store, baseline):
    """Record a measurement under its own address, and never twice.

    IDEMPOTENT BY CONSTRUCTION rather than by a check bolted on: the path is derived from the
    content, so re-running after an interruption writes the same bytes to the same place. A file
    that is already there and already says the same thing is left exactly as it is, because the
    first recording's mtime is part of the history this store exists to keep.

    A file already there saying something DIFFERENT under the same address would mean two
    documents sharing a sha256, so it is reported rather than resolved. Nobody will meet it; the
    alternative is silently preferring one of two measurements that both claim to be the same.
    """
    want = dict(baseline)
    want[ADDRESS_FIELD] = address(want)
    path = addressed_path(store, want[ADDRESS_FIELD])
    if path.exists():
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError, OSError) as e:
            raise BaselineError(
                f"{path} already holds this measurement's address and cannot be read ({e}), so "
                f"this run cannot tell whether the recording it would make is already there. The "
                f"file is left alone: a store that overwrites what it cannot read is not a "
                f"history.") from e
        if held != want:
            raise BaselineError(
                f"{path} already holds a DIFFERENT measurement under the same address, which "
                f"would mean two documents sharing a sha256. Nothing was written.")
        return path
    with atomic_write(path) as f:
        json.dump(want, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write(path, baseline):
    """Write a baseline atomically. Never in place over an existing one.

    A new baseline is a NEW artefact and the old one stays, because "what did we compare against
    in March" has to remain answerable in June. Overwriting would make the history of a gate
    unauditable at exactly the moment somebody disputes a verdict.
    """
    path = Path(path)
    if path.exists():
        raise BaselineError(
            f"{path} already exists, and baselines are never overwritten in place. A new baseline "
            f"is a new file: name it for what changed, and keep the old one so the comparison it "
            f"was used for stays reproducible.")
    with atomic_write(path) as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


#: What a compass result carried before the canonical `metrics` block existed, as committed under
#: `evidence/compass-2026-07-30/`. Matched on THREE of the four rather than on any single key:
#: `model` and `seed` are on nearly everything this project writes, and a bare `auc` appears inside
#: other artefacts' nested blocks, so a one-key test would claim files it cannot describe. A guard
#: that covers one spelling of a shape reports confidently on the others.
_PRE_METRICS_COMPASS_KEYS = ("auc", "controls", "n_harmful", "n_harmless")


#: What `senbonzakura measure` writes: a summary of several stages, each with its own artefact
#: beside it. Matched on the `stages` mapping rather than on a filename, because a renamed file is
#: still the same document.
_MEASURE_STAGE_KEYS = ("stages", "measured", "table")


#: What a `measure` summary declares itself to be. The literal, not an import of
#: `measure.SUMMARY_KIND`: this module is on the path of every `baseline` invocation and `measure`
#: pulls in the whole stage machinery, so importing it here to read one string would make reading
#: a baseline pay for a command it is not running. The two are held together by a test instead.
_SUMMARY_KIND = "summary"


def _is_measure_summary(doc):
    """Whether this is a `measure` summary rather than a single measurement.

    `measure` runs several instruments and writes one document describing all of them, plus a
    per-stage artefact for each. It carries no `metrics` block, because it is not itself a
    measurement of anything: it is an index of several.

    THE DECLARATION FIRST, THE SHAPE SECOND. Since Q-116 the document says `kind: summary`, and a
    stated kind beats a guess read off a shape: "it has a stages dict and no metrics block" is an
    inference about intent, and an artefact that happened to carry both would be misread with
    nothing for a reader to check. The shape check is kept behind it, unchanged, because summaries
    written before the declaration existed are still on people's disks and still deserve the
    refusal that points at the per-stage files.
    """
    if doc.get("kind") == _SUMMARY_KIND:
        stages = doc.get("stages")
        if isinstance(stages, dict) and stages:
            files = sorted(v for v in stages.values() if isinstance(v, str) and v.endswith(".json"))
            return "kind", files or sorted(f"{n}.json" for n in stages)
        # DECLARED A SUMMARY AND LISTING NOTHING is still a summary, and still not a measurement.
        # Falling through to the shape check here would report it as an old unstamped artefact,
        # which is the false advice this whole branch exists to have stopped giving.
        return "kind", []
    for key in _MEASURE_STAGE_KEYS:
        value = doc.get(key)
        if isinstance(value, dict) and value:
            # `measure` writes {stage: filename}, so the values ARE the files to point at. Derived
            # from the document rather than rebuilt as f"{name}.json", because a reader is being
            # told which file to open and that has to be the name the run actually wrote.
            files = sorted(v for v in value.values() if isinstance(v, str) and v.endswith(".json"))
            return key, files or sorted(f"{n}.json" for n in value)
    return None


def _pre_metrics_compass(doc):
    """Whether this is a compass result from before the metrics block, by its own field names."""
    return sum(k in doc for k in _PRE_METRICS_COMPASS_KEYS) >= 3


def read(path):
    """Load a baseline, or say which of the several different problems it is."""
    path = Path(path)
    if not path.is_file():
        # AND WHAT TO DO ABOUT IT. "no baseline at X." is true, and a reader meeting it on a first
        # run has no way to know that a baseline is a file this tool writes rather than one they
        # were supposed to have, or which command writes it.
        what = "is a directory" if path.is_dir() else "is not there"
        raise BaselineError(
            f"no baseline at {path}: it {what}.\n"
            f"  A baseline is a file this tool writes, from a measurement you have already taken:\n"
            f"    senbonzakura baseline <a result artefact> --seeds <n> --out {path}\n"
            f"  then gate a later run against it.")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError) as e:
        raise BaselineError(f"{path} is not readable JSON ({e}).") from e
    if not isinstance(loaded, dict):
        raise BaselineError(f"{path} holds {type(loaded).__name__}, not a baseline object.")
    got = loaded.get("schema")
    if got is None and ("metrics" in loaded or "measurement" in loaded):
        # THE COMMONEST WAY TO GET HERE IS NOT A SCHEMA CHANGE, it is handing this a raw run output.
        #
        # A reader followed the hint `baseline` prints on success, passed the new run's result
        # artefact, and got "declares schema None" (2026-09-26). That sentence is about versioning
        # and their file had no version because it was never a baseline; nothing told them the file
        # was the wrong KIND, or that one command turns the one they have into the one wanted.
        #
        # Detected on `metrics`, which is the block every result artefact in this project stamps and
        # no baseline carries, so this cannot fire on a genuinely old baseline.
        raise BaselineError(
            f"{path} looks like a result artefact rather than a recorded measurement: it carries a "
            f"`metrics` block and no baseline `schema`. This compares two RECORDED measurements, so "
            f"record this one first:\n"
            f"    senbonzakura baseline {path} --seeds <n> --out recorded.json\n"
            f"then pass `recorded.json`. The command prints the flag to use next when it succeeds.")
    if got is None and _pre_metrics_compass(loaded):
        # THE SAME DEFECT AS THE BLOCK ABOVE, ONE GENERATION OF ARTEFACT EARLIER. That one was
        # shipped knowing this case was open: it detects a result artefact by its `metrics` block,
        # and compass only started stamping one later, so this project's OWN published evidence
        # under `evidence/compass-2026-07-30/` still got "declares schema None". A checker that
        # cannot read its author's published figures is the exact shape this repository criticises
        # in other tools.
        #
        # It does NOT tell the reader to run `senbonzakura baseline` on it, because that command
        # reads the `metrics` block too and would refuse with a second, different sentence. A
        # correct diagnostic wired to a next step that also fails is worse than no hint at all.
        raise BaselineError(
            f"{path} is a compass result from before this tool stamped a `metrics` block, so "
            f"there is no figure in it this gate knows how to read.\n"
            f"  It holds an AUC of {loaded.get('auc')} and its own controls, which you can read "
            f"directly, but nothing records which metric that figure IS.\n"
            f"  To gate on a compass figure, re-run `senbonzakura compass` with this build: it "
            f"stamps the block, and the run is then recordable with `senbonzakura baseline`.\n"
            f"  Archived evidence from before the block cannot be converted, only re-measured.")
    if got is None:
        # NOT "declares schema None", WHICH READS AS A VERSION PROBLEM. The two branches above catch
        # the shapes this tool recognises; anything left carrying no `schema` at all was most likely
        # never a baseline, and telling somebody their file declares None sends them looking for a
        # migration that does not exist. Same defect as the one those branches were written for, on
        # the case they do not match.
        raise BaselineError(
            f"{path} carries no `schema` field, so nothing in it says what kind of file it is. A "
            f"baseline written by this tool always declares {SCHEMA!r}.\n"
            f"  If this is a run's result artefact, record it first:\n"
            f"    senbonzakura baseline {path} --seeds <n> --out recorded.json\n"
            f"  If it is something else, this gate cannot read it: it compares two recorded "
            f"measurements and nothing else.")
    if got != SCHEMA:
        raise BaselineError(
            f"{path} declares schema {got!r} and this build understands {SCHEMA!r}. A schema "
            f"change means an older file means something different, so it is refused rather than "
            f"read on the assumption that the fields still line up.")
    # THE ADDRESS IS CHECKED WHEN IT IS THERE, AND NOT DEMANDED WHEN IT IS NOT. A baseline
    # recorded before this field existed is an older file that means the same thing, which is
    # the case the comment on `SCHEMA` reserves `.get` for. A baseline that carries an address
    # that does not match its contents is a different matter: somebody edited the number and
    # left the stamp, and a gate that read it would compare against a measurement nobody took.
    stamped = loaded.get(ADDRESS_FIELD)
    if stamped is not None and stamped != address(loaded):
        raise BaselineError(
            f"{path} carries the address {stamped} and its contents address to "
            f"{address(loaded)}, so the file has been changed since it was recorded. A baseline is "
            f"never edited: record a new one, name it for what changed, and keep the old one.")
    return loaded


def comparability(baseline, now):
    """Which pinned fields disagree, as (field, recorded, given, why) tuples. Empty means usable.

    A field that either side does not carry is reported as a MISMATCH rather than skipped, which is
    the opposite of what `runrecord.mismatches` does, and the difference is deliberate. There, an
    older record with a missing field still describes a run somebody is trying to finish, and
    refusing would strand them. Here, an unknown field means the gate cannot tell whether the two
    measurements are comparable, and "cannot tell" has to fail: the entire value of this module is
    that it does not compare things it cannot vouch for.
    """
    out = []
    for field, why in PINNED.items():
        was = (baseline or {}).get(field)
        current = (now or {}).get(field)
        if was is None or current is None:
            out.append((field, was, current,
                        f"not recorded on one side, so it cannot be shown to match: {why}"))
        elif str(was) != str(current):
            out.append((field, was, current, why))
    return out


#: How much wider this run's interval may be than the baseline's before the comparison is refused.
#:
#: THE HOLE THIS CLOSES. The gate fires on intervals rather than point estimates, which is what
#: keeps it switched on: a gate that fails on noise is disabled within a fortnight. But the width
#: of an interval is under the measurer's control and nothing constrained it, so measuring badly
#: always passed. A baseline of 0.094 [0.06, 0.13] on n=200 against a run of 0.55 [0.05, 0.95] on
#: n=4 overlapped, and the gate printed "there is no evidence the property moved" over a refusal
#: rate that had gone from 9.4% to 55%. That sentence is true and reads as reassurance.
#:
#: A REFUSAL RATHER THAN A FAILURE, which is the whole reason it lives here and not in `verdict`.
#: "The property regressed" and "this run could not have seen it regress" are different findings,
#: and the module argues at length that conflating them teaches a reader to ignore the difference.
#: Exit 2, not exit 1.
#:
#: Two is a convention and is named as one. It is loose on purpose: an interval twice as wide is
#: unmistakably a blunter instrument, while a rule tight enough to catch a 20% widening would fire
#: on ordinary seed-to-seed variation and be switched off, which is the failure this whole module
#: is shaped around.
MAX_WIDTH_RATIO = 2.0


def interval_of(doc):
    """The interval a baseline or a stamped measurement carries, or None if it is deterministic.

    One place that knows the field may legitimately be absent, so no caller has to remember it.
    `tuple(doc["interval"])` at three call sites was how a deterministic measurement turned into
    a TypeError, and this command reports a TypeError as a refusal, which reads as "not
    comparable" for a measurement that was perfectly comparable.
    """
    if doc.get("deterministic"):
        return None
    interval = doc.get("interval")
    return None if interval is None else (float(interval[0]), float(interval[1]))


def refuse_if_too_blunt(baseline, now_interval):
    """Stop a comparison whose current measurement is too imprecise to have seen anything.

    Raises BaselineError, which the gate reports as REFUSED rather than as a regression.

    SKIPPED FOR A DETERMINISTIC BASELINE, per `DETERMINISTIC_MEANS`. The guard asks whether this
    run's interval is too wide to have seen the property move; a measurement that returns the same
    number every time has no width, and there is nothing for it to have failed to see.
    """
    if baseline.get("deterministic") or now_interval is None:
        return
    lo, hi = float(now_interval[0]), float(now_interval[1])
    base_lo, base_hi = float(baseline["interval"][0]), float(baseline["interval"][1])
    base_width, width = base_hi - base_lo, hi - lo
    if base_width <= 0 or width <= base_width * MAX_WIDTH_RATIO:
        return
    raise BaselineError(
        f"this run's interval [{lo:.4f}, {hi:.4f}] is {width / base_width:.1f} times wider than "
        f"the baseline's [{base_lo:.4f}, {base_hi:.4f}], so an overlap between them is not "
        f"evidence that the property held: it is evidence that this run could not have seen it "
        f"move. Measure at the precision the baseline was measured at, or record a new baseline "
        f"at this precision and say in its filename what changed.\n"
        f"  The baseline was taken on n={baseline.get('n')} with seeds {baseline.get('seeds')}.")


def refuse_if_incomparable(baseline, now):
    """Stop a comparison that would put a green tick on two numbers that never matched.

    Raises BaselineError naming every disagreement at once, because a reader fixing them one
    error at a time is a reader who re-runs the gate six times.
    """
    bad = comparability(baseline, now)
    if not bad:
        return
    lines = [f"  {field}: baseline {was!r}, this run {current!r}  ({why})"
             for field, was, current, why in bad]
    raise BaselineError(
        "this measurement is not comparable to that baseline, so it was not compared:\n"
        + "\n".join(lines)
        + "\n\nEither measure under the conditions the baseline records, or record a new baseline "
          "and say in its filename what changed. Comparing across a mismatch is how a drifted "
          "prompt renderer got a published table and a passing build.")


def overlaps(a, b):
    """Do two closed intervals share any point? Touching at an endpoint counts as overlapping.

    Touching counts because the boundary case is noise, not evidence, and a gate that fires on
    `hi == lo` fires on rounding.
    """
    return a[0] <= b[1] and b[0] <= a[1]


def _deterministic_verdict(baseline, now_point, now_interval, metric, direction, moved):
    """Compare two readings of a metric that claims to return the same number every time.

    EXACTLY, and that is the point rather than an oversight. `DETERMINISTIC_MEANS` records what
    the writer asserted: under the pinned conditions this measurement does not vary. If it varied,
    either the model changed or the claim was false, and both deserve a reader's attention. A
    tolerance here would be an interval nobody measured, which is the thing the whole module
    refuses to compare against.
    """
    if now_interval is not None:
        raise BaselineError(
            f"the baseline for {metric} is deterministic and this measurement carries an "
            f"interval {list(now_interval)}, so the two were not produced by the same "
            f"instrument. Compare a deterministic reading against a deterministic baseline.")
    base = float(baseline["point"])
    if moved == 0:
        return True, f"{metric}: unchanged", (
            f"baseline {base:.4f} and this run {float(now_point):.4f} are the same number. "
            f"{DETERMINISTIC_MEANS.capitalize()}, so an exact match is what a pass looks like "
            f"and any difference at all would have been reported.")
    worse = moved < 0 if direction == HIGHER_IS_BETTER else moved > 0
    where = "below" if moved < 0 else "above"
    if worse:
        return False, f"{metric}: REGRESSED", (
            f"baseline {base:.4f}, this run {float(now_point):.4f}, {abs(moved):.4f} {where} it "
            f"and in the worse direction for a metric where {direction.replace('_', ' ')}. This "
            f"metric is {DETERMINISTIC_MEANS}, so there is no run-to-run noise for the move to "
            f"be: either the model changed or the determinism claim was wrong.")
    return True, f"{metric}: improved", (
        f"baseline {base:.4f}, this run {float(now_point):.4f}, {abs(moved):.4f} {where} it and "
        f"in the better direction. This metric is {DETERMINISTIC_MEANS}, so the move is real "
        f"rather than noise and is worth knowing about even though it passes.")


def verdict(baseline, now_point, now_interval):
    """Compare, and return (ok, headline, detail). Refuses first; never compares blind.

    A regression requires BOTH that the intervals are disjoint and that the move is in the
    direction that is worse. Disjoint-and-better is a pass, and it still says so out loud, because
    a measurement that moved a long way is worth a reader's attention whichever way it went.
    """
    metric, direction = baseline["metric"], baseline["direction"]
    moved = float(now_point) - float(baseline["point"])

    if baseline.get("deterministic"):
        return _deterministic_verdict(baseline, now_point, now_interval, metric, direction, moved)

    if now_interval is None:
        raise BaselineError(
            f"the baseline for {metric} carries an interval and this measurement does not, so "
            f"there is nothing to compare it against. A metric that was measured with a spread "
            f"and is now declared deterministic changed instrument between the two readings, "
            f"which is what `estimator` and `tool_version` are pinned to catch.")
    lo, hi = float(now_interval[0]), float(now_interval[1])
    if lo > hi:
        raise BaselineError(f"interval [{lo}, {hi}] is inverted.")
    base_iv = (float(baseline["interval"][0]), float(baseline["interval"][1]))

    if overlaps(base_iv, (lo, hi)):
        # THE INTERVAL IS PRINTED ON A PASS, not only on a failure. A gate whose threshold is so
        # wide that nothing trips it reads as safety and is decoration, and the only way a reader
        # can tell the difference is by seeing how much room there was.
        return True, f"{metric}: within interval", (
            f"baseline {baseline['point']:.4f} {list(base_iv)} on n={baseline['n']}, "
            f"this run {float(now_point):.4f} [{lo:.4f}, {hi:.4f}]. The intervals overlap, so "
            f"there is no evidence the property moved. Moved {moved:+.4f} on the point estimate, "
            f"which on its own is not a finding.")

    worse = moved < 0 if direction == HIGHER_IS_BETTER else moved > 0
    where = "below" if moved < 0 else "above"
    if worse:
        return False, f"{metric}: REGRESSED", (
            f"baseline {baseline['point']:.4f} {list(base_iv)} on n={baseline['n']}, "
            f"this run {float(now_point):.4f} [{lo:.4f}, {hi:.4f}]. The intervals do not overlap "
            f"and the value moved {where} it by {abs(moved):.4f}, which is the worse direction for "
            f"a metric where {direction.replace('_', ' ')}.")
    return True, f"{metric}: improved beyond its interval", (
        f"baseline {baseline['point']:.4f} {list(base_iv)} on n={baseline['n']}, "
        f"this run {float(now_point):.4f} [{lo:.4f}, {hi:.4f}]. The intervals do not overlap and "
        f"the move is in the better direction, so this passes. It is still a change worth "
        f"recording: something altered the measurement by {abs(moved):.4f}.")


#: What a green verdict does NOT say, printed with every one of them. Somebody will eventually read a
#: green tick as "the model is safe", and the only defence is that the artefact says otherwise in
#: the same breath as the verdict.
VERDICT_CAVEAT = (
    "A pass means one measured property did not move outside its interval, on one track, under "
    "the conditions the baseline records. It is not a statement that the model is safe, that "
    "other properties held, or that the track represents anything beyond itself.")

#: The same sentence turned the other way up, for the verdict it actually follows.
#:
#: `VERDICT_CAVEAT` was printed under a failure too, so a reader who had just been told a property
#: regressed then read a paragraph about what a pass does not mean. The caveat matters as much
#: here and says a different thing: one property moved, and the ones nobody gated are not known
#: to have held.
FAILURE_CAVEAT = (
    "A failure means one measured property moved outside its interval, on one track, under the "
    "conditions the baseline records. Nothing here says the other properties held: they were not "
    "gated, and a single failing gate is a floor on what moved rather than the whole of it.")


# ── the producer, without which none of the above has an input ───────────────────────────────
#
# WHY THIS EXISTS, and it is the largest thing the 2026-09-21 panel found. Two reviewers noticed
# independently that `record()` had no call site outside its own test, that `gate` was registered
# in the dispatch table and documented in the CLI reference while nothing in the repository could
# produce a file it would accept, and that the module docstring's claim - "a change that moves a
# measured property outside its interval fails a build" - was therefore not true of any property
# this tool measures. Read the commit subjects alone and you would believe otherwise.
#
# The gap was real rather than cosmetic: every writer stamps its figures through
# `measurement.stamp`, which puts the identity INSIDE the metrics block, and `comparability` reads
# the pinned fields from the TOP level. The two halves were built a fortnight apart and never met.
# This is the adapter between them, and it refuses rather than guesses.

#: Where a pinned field may be found in a stamped artefact, in order of preference.
#:
#: `model` lives at the top of every artefact this project writes; everything else lives in the
#: metric's own block, because it describes that measurement rather than the file. `metric` is
#: taken from the block too, not from the key, since a key may be `metric.estimator`.
_TOP_LEVEL = ("model",)


def _pinned_from(block, doc, metric_key):
    """Every pinned field for one metric, or a refusal naming all the ones that are missing.

    NAMES THEM ALL AT ONCE, deliberately. A reader fixing one field per run is a reader who runs
    this six times, and each run costs a re-measurement rather than a re-read.
    """
    found, missing = {}, []
    for field in PINNED:
        value = doc.get(field) if field in _TOP_LEVEL else block.get(field)
        if value is None and field == "metric":
            value = metric_key.split(".", 1)[0]
        if value is None:
            missing.append(field)
        else:
            found[field] = value
    if missing:
        raise BaselineError(
            f"{metric_key} cannot become a baseline: it records no "
            f"{', '.join(repr(m) for m in missing)}.\n"
            f"  Each of those decides whether a later measurement may be compared with this one, "
            f"and a baseline with a hole in it is what this module exists to refuse. They are "
            f"written by `measurement.stamp`, so the fix is in whatever produced the artefact "
            f"rather than here.\n"
            f"  What it does carry: {', '.join(sorted(k for k in block if block[k] is not None))}")
    return found


def from_artefact(doc, metric_key, *, seeds):
    """Build a baseline from a stamped measurement artefact.

    `seeds` is passed rather than read, because a single artefact is one run and a baseline that
    claims a spread it does not have is worse than no baseline. The caller states which seeds the
    figure rests on and that claim lands in the file where a reader can check it.
    """
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        # A `measure` SUMMARY IS NOT AN OLD ARTEFACT, and saying so was this refusal's worst
        # property. A surface audit fed it a `measure.json` written ninety seconds earlier by the
        # same build and was told the file predated a stamp introduced in September: advice that is
        # false and cannot be followed, about the headline command's own output, in the next step of
        # the documented CI chain. The reader then has to work out unaided that the per-stage files
        # beside it do work.
        found = _is_measure_summary(doc)
        if found:
            key, stages = found
            listed = ", ".join(stages) or "the per-stage files beside it"
            # BLANK LINES BETWEEN PARAGRAPHS, and FOUR SPACES before the command. `say.reflow`
            # joins consecutive two-space lines into one paragraph, which is what a continuation
            # indent means, and passes four-space lines through verbatim so a command stays
            # pasteable. Written with a two-space indent throughout, this whole refusal reflowed
            # into one block with the example command wrapped in the middle of it.
            raise BaselineError(
                f"this looks like a `senbonzakura measure` summary rather than a single "
                f"measurement: it carries `{key}` and no `metrics` block.\n"
                f"\n"
                f"That is correct for what it is. `measure` runs several instruments and writes one "
                f"document describing all of them, so there is no single figure in it to record. "
                f"Each instrument's own artefact is written beside it, and each of those does "
                f"carry a stamped figure.\n"
                f"\n"
                f"Record one of those instead: {listed}\n"
                f"\n"
                f"    senbonzakura baseline {stages[0] if stages else '<stage>.json'} "
                f"--seeds <n> --out b.json")
        raise BaselineError(
            "this artefact carries no `metrics` block, so there is no stamped figure to build a "
            "baseline from. Artefacts written before 2026-09-12 predate the stamp; re-run the "
            "measurement with a current build rather than hand-writing one.")
    block = metrics.get(metric_key)
    if not isinstance(block, dict):
        raise BaselineError(
            f"no metric {metric_key!r} in this artefact. It carries: "
            f"{', '.join(sorted(metrics))}.")

    deterministic = bool(block.get("deterministic"))
    interval = block.get("interval")
    if deterministic and interval is not None:
        raise BaselineError(
            f"{metric_key} is stamped deterministic and also carries an interval {interval}. "
            f"A number that does not vary between runs has nothing for an interval to describe, "
            f"so one of the two claims is wrong and nothing here can tell which.")
    if not deterministic and not (isinstance(interval, (list, tuple)) and len(interval) == 2):
        raise BaselineError(
            f"{metric_key} has no interval, and this gate fires on intervals rather than on point "
            f"estimates: a gate that fails on noise is switched off within a fortnight. Measure it "
            f"with the interval its command reports, stamp it `deterministic=True` if it genuinely "
            f"returns the same number every run, or record the baseline by hand and say in "
            f"the filename what it rests on.")
    n = block.get("n")
    if not isinstance(n, int) or n <= 0:
        raise BaselineError(f"{metric_key} records n={n!r}, which is not a sample size.")

    pinned = _pinned_from(block, doc, metric_key)
    higher = block.get("higher_is_better")
    if higher is None:
        raise BaselineError(
            f"{metric_key} does not say which direction is better, so a gate reading it could not "
            f"tell a regression from an improvement.")
    return record(
        direction=HIGHER_IS_BETTER if higher else LOWER_IS_BETTER,
        point=block["value"], interval=None if deterministic else tuple(interval),
        deterministic=deterministic, seeds=seeds, n=n,
        extra={"from_metric_key": metric_key, "units": block.get("units")},
        **pinned)


def build_parser():
    p = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog="senbonzakura baseline",
        description="Record a measurement as the baseline a later run is gated against.")
    # THE ARTEFACT WITHOUT A FLAG. It is the one thing this command cannot work out, so it is the
    # positional; `--measurement` still works and is what every recorded invocation passes.
    p.add_argument("measurement_positional", nargs="?", default=None, metavar="MEASUREMENT",
                   help="a result artefact carrying a stamped `metrics` block, given without a "
                        "flag. Equivalent to --measurement.")
    p.add_argument("--measurement", default=None,
                   help="a result artefact carrying a stamped `metrics` block")
    p.add_argument("--metric", default=None,
                   help="which key inside that block to record, e.g. `coherence` or "
                        "`refusal_rate.senbonzakura-ruler`. Left out, it is read from the "
                        "artefact when the artefact stamped exactly one metric, and refused "
                        "naming every candidate when it stamped several")
    # NOT `required=True`, so the refusal can name a next step. argparse's own
    # "the following arguments are required: --seeds" says what is missing and nothing about what to
    # type, and a reader who has not met this command does not know what shape a seed list takes.
    p.add_argument("--seeds", default=None,
                   help="the seeds this figure rests on, comma separated. Stated rather than "
                        "inferred: one artefact is one run, and a baseline claiming a spread it "
                        "does not have is worse than none")
    p.add_argument("--out", default=None,
                   help="where to write it (default: ./baselines/<metric>.json). Never "
                        "overwritten: a new baseline is a new file")
    return p


def only_metric(doc):
    """The one metric this artefact stamped, or a refusal naming every candidate.

    Inferred rather than demanded ONLY when there is nothing to infer between. A command that
    guessed among several would pick one silently, and a baseline recording a metric nobody chose
    gates a later run on a property nobody meant to protect.
    """
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        # A `measure` SUMMARY IS NOT AN OLD ARTEFACT, and saying so was this refusal's worst
        # property. A surface audit fed it a `measure.json` written ninety seconds earlier by the
        # same build and was told the file predated a stamp introduced in September: advice that is
        # false and cannot be followed, about the headline command's own output, in the next step of
        # the documented CI chain. The reader then has to work out unaided that the per-stage files
        # beside it do work.
        found = _is_measure_summary(doc)
        if found:
            key, stages = found
            listed = ", ".join(stages) or "the per-stage files beside it"
            # BLANK LINES BETWEEN PARAGRAPHS, and FOUR SPACES before the command. `say.reflow`
            # joins consecutive two-space lines into one paragraph, which is what a continuation
            # indent means, and passes four-space lines through verbatim so a command stays
            # pasteable. Written with a two-space indent throughout, this whole refusal reflowed
            # into one block with the example command wrapped in the middle of it.
            raise BaselineError(
                f"this looks like a `senbonzakura measure` summary rather than a single "
                f"measurement: it carries `{key}` and no `metrics` block.\n"
                f"\n"
                f"That is correct for what it is. `measure` runs several instruments and writes one "
                f"document describing all of them, so there is no single figure in it to record. "
                f"Each instrument's own artefact is written beside it, and each of those does "
                f"carry a stamped figure.\n"
                f"\n"
                f"Record one of those instead: {listed}\n"
                f"\n"
                f"    senbonzakura baseline {stages[0] if stages else '<stage>.json'} "
                f"--seeds <n> --out b.json")
        raise BaselineError(
            "this artefact carries no `metrics` block, so there is nothing to record and nothing "
            "to name with --metric either. Artefacts written before 2026-09-12 predate the stamp; "
            "re-run the measurement with a current build.")
    if len(metrics) > 1:
        raise BaselineError(
            f"this artefact stamped {len(metrics)} metrics, so --metric has to say which one: "
            f"{', '.join(sorted(metrics))}. Guessing between them would gate a later run on a "
            f"property nobody chose.")
    return next(iter(metrics))


def default_out(metric):
    """Where a baseline goes when nobody said: one directory, one file per metric.

    BUILT AS A STRING WITH FORWARD SLASHES, not through `Path`. This value is documented in
    `--help` as `./baselines/<metric>.json` and it goes into the log and into a baseline file that
    people compare across machines. `Path` renders it with a backslash separator on Windows, so
    the documented default and the real one disagreed there; CI's windows row caught it. Windows
    accepts a forward slash in every path API, so nothing is lost by pinning the spelling.
    """
    return f"baselines/{metric.replace('/', '_')}.json"


def main(argv=None):
    a = build_parser().parse_args(argv)
    measurement = a.measurement or a.measurement_positional
    if not a.seeds:
        given = measurement or "<result.json>"
        print(f"--seeds is missing. It records which seeds this figure rests on, and a baseline "
              f"that claims a spread it has not got is worse than no baseline.\n"
              f"  For a single run:\n"
              f"    senbonzakura baseline {given} --seeds 42\n"
              f"  For several, comma separated:\n"
              f"    senbonzakura baseline {given} --seeds 42,43,44", file=sys.stderr)
        return 2
    if a.measurement and a.measurement_positional and a.measurement != a.measurement_positional:
        print(f"two different artefacts were given: {a.measurement_positional!r} as a positional "
              f"and {a.measurement!r} with --measurement. Pass one.", file=sys.stderr)
        return 2
    if not measurement:
        print("no measurement given. Pass the artefact as the first argument, or with "
              "--measurement.", file=sys.stderr)
        return 2
    a.measurement = measurement
    try:
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    except ValueError:
        print(f"--seeds must be integers, got {a.seeds!r}", file=sys.stderr)
        return 2
    if not seeds:
        print("--seeds is empty, so nothing says what this figure rests on", file=sys.stderr)
        return 2
    try:
        doc = json.loads(Path(a.measurement).read_text(encoding="utf-8"))
        metric = a.metric or only_metric(doc)
        if not a.metric:
            print(f"--metric was left out and this artefact stamped only {metric!r}, so that is "
                  f"what is being recorded")
        if a.out is None:
            a.out = default_out(metric)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        written = from_artefact(doc, metric, seeds=seeds)
        write(a.out, written)
    except (OSError, json.JSONDecodeError) as e:
        # A DIRECTORY IS THE COMMON CASE and it used to arrive as a bare errno. Somebody passes the
        # run's output directory, which is what every other command here takes, and "[Errno 21] Is a
        # directory" names the mistake in a dialect nobody thinks in.
        where = Path(a.measurement)
        if where.is_dir():
            print("\n".join(say.reflow(
                f"{where} is a directory. This takes the result artefact itself, which is the json "
                f"file a measurement wrote, not the directory holding it. Look for a file such as "
                f"coherence.json or compass.json inside it.")), file=sys.stderr)
        else:
            print(f"cannot read {a.measurement}: {e}", file=sys.stderr)
        return 2
    except BaselineError as e:
        # Wrapped: these refusals are three paragraphs and were printed as three long lines.
        print("\n".join(["refused:", *say.reflow(str(e))]), file=sys.stderr)
        return 2
    # THE UNITS AND THE MISSING INTERVAL, BOTH SAID RATHER THAN PRINTED RAW. This line used to read
    # "at 2.7486 None on n=1": `interval` is legitimately None on a metric stamped deterministic,
    # and interpolating it put the word None where a reader looks for a spread. The units were in
    # the artefact the whole time, one field away, so the summary of a measurement was the one
    # place its units did not appear.
    units = (written.get("extra") or {}).get("units")
    spread = ("no interval, because the metric is stamped deterministic"
              if written["interval"] is None else f"interval {written['interval']}")
    print(f"baseline written to {a.out}: {written['metric']} at {written['point']:.4f}"
          f"{f' {units}' if units else ''} ({spread}) on n={written['n']}, "
          f"seeds {written['seeds']}")
    # THE HINT USED TO BE UNFOLLOWABLE. It read `--measurement <new>`, and a reader passed the new
    # run's result artefact, because that is what `--measurement` means on THIS command. `gate`
    # wanted a recorded measurement, so it exited 2 on a schema message. Both steps are named now,
    # and the flag is the one `gate` actually documents.
    print("  gate a later run with, once you have recorded it the same way:")
    print("    senbonzakura baseline <the later run's result> --seeds <n> --out later.json")
    print(f"    senbonzakura gate --baseline {a.out} --current later.json")
    return 0


if __name__ == "__main__":
    # The guard eight modules once lacked, so `python -m senbonzakura.<module>` executed nothing
    # and exited 0 while the documentation said otherwise.
    sys.exit(main())
