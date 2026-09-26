# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""One command for the question everybody actually asks: did I break my model?

WHAT WAS WRONG

Answering it took five commands and five output files, each with its own required flags, and then
a sixth to assemble two of them into a card. Every one of those five is the right shape for what
it does, and none of them is the question. Somebody who has just abliterated a model wants a
verdict; they should not have to know that refusal lives in `score`, harm recognition in
`compass`, fluency in `coherence` and reasoning in `capability`, nor that four of the five want a
partition boundary read out of a track manifest.

So this runs them, on one model, from one track, into one directory, and prints one table.

WHAT IT IS NOT

It is not a new measurement. Every number here comes from the command that already produced it,
called with the arguments that command already takes, so a figure from `senbonzakura measure` and
a figure from `senbonzakura score` are the same figure and there is no second implementation to
drift. The five commands stay, because a single axis is a legitimate thing to want and because
they take knobs this deliberately does not expose.

It is also not a pass or a fail. This project has withdrawn published numbers before, and a
green tick over four instruments would be exactly the artefact that invites somebody to quote a
result they have not read. The table reports what each instrument said and what it cannot say;
the reading is the user's.

HOW A FAILING STAGE IS HANDLED

Each stage is independent and is allowed to fail on its own. A model too large for the card will
fail `capability` and still produce a refusal rate, and reporting three numbers and a named
failure is more use than reporting nothing. The exit status is non-zero if any stage failed, so a
script still knows, and the table says which one and why.

Stages are also skipped when their output file is already there, so an interrupted run resumes
rather than repeating an hour of generation. `--force` re-runs everything.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple

from . import stamps
from .crashsafe import atomic_write

#: Stage name -> (module, output filename). The module is looked up through `entry.DELEGATED`
#: where it appears there, so a command that moves keeps being run rather than quietly dropping
#: out of this list.
STAGE_ORDER = ("score", "compass", "coherence", "capability", "drift")

#: What each stage's result file is called inside `--out`. Named after the command rather than
#: after the quantity, so somebody who wants to re-run one by hand can see which it was.
OUTPUTS = {name: f"{name}.json" for name in STAGE_ORDER}


class StageError(Exception):
    """One instrument did not produce a number, with the reason phrased for a person."""


def _track_spec(track, split):
    """A dataset spec for one split of a track, whether that track is a directory or the bundle.

    `default/bad_eval_ds` is a real spec that `dataset.resolve` understands, so the bundled track
    needs no special case anywhere downstream; it only needs to be spelled that way here.
    """
    return f"{track.rstrip('/')}/{split}"


#: The stages whose parsers declare `--hf-token`. `margin`, `coherence` and `drift` do not, and
#: handing a flag to a parser that does not know it makes argparse print the flag AND ITS VALUE.
#: Kept as data beside the builder rather than as a `try` around the run, because the failure is
#: at argument-construction time and the fix has to be too.
TAKES_HF_TOKEN = frozenset({"score", "capability"})


def stage_argv(name, args, out_dir):
    """The command line this stage would have been given by hand.

    Built rather than executed here so it can be printed, tested without a GPU, and quoted in the
    table. A reader who wants to re-run one stage with different knobs can copy the line.
    """
    common = ["--model", args.model, "--device", args.device]
    if args.hf_token and name in TAKES_HF_TOKEN:
        # The VALUE is never printed: `_shown` below replaces it. It is here because the stage
        # needs it, and a gated model fails four ways without it.
        #
        # GUARDED BY THE STAGE, because three of the five do not declare the flag and argparse
        # prints an unrecognised argument WITH ITS VALUE. `--hf-token hf_live_token` therefore
        # went to stderr verbatim, in the error text of three stages, under a comment on this
        # very line promising the value is never printed. Found 2026-09-25.
        common += ["--hf-token", args.hf_token]
    if args.trust_remote_code:
        common += ["--trust-remote-code"]
    out = str(out_dir / OUTPUTS[name])

    if name == "score":
        # `--skip` IS NOT OPTIONAL HERE. `bad_eval_ds` is the search rows followed by the measure
        # rows, and `track.json` records `skip_harmful` so a caller can land on the second half.
        # Without it `score` takes prompts from the head of the file, which is the partition the
        # search chose the configuration on, and the table below labels the figure "measured on
        # this track's held-out rows and on no others".
        #
        # `compass` already did this correctly through `margin.resolve_skips`, so one table was
        # reading two different partitions and saying so about neither. This is the project's own
        # documented incident class: `checker/.../a-rate-with-no-partition-beside-it.json` records
        # a 0.0% that travelled as a measured refusal rate having been scored on the selection
        # partition.
        #
        # HANDED TO `score` RATHER THAN RESOLVED HERE, since 2026-09-25, and the difference is the
        # whole finding. This used to call a local `_harmful_skip` that re-implemented the manifest
        # read as `Path(track) / "track.json"`. On the DEFAULT track that path does not exist, the
        # OSError was caught, and it returned 0, so the front door of this tool scored the
        # selection rows under a caption promising held-out ones. `track.read_manifest` has a
        # bundled-alias branch for exactly this and returns 132. Two readers of one manifest, and
        # the one used here was the one that had never been told about the bundled track.
        #
        # There is now one resolver. `score --track --track-arm harmful` reads the boundary, says
        # whether a manifest confirmed it, and stamps the partition accordingly, which also means
        # a `measure` figure can finally be compared against a `score` baseline: passing `--skip`
        # alone stamped `rows-from-132`, which never equals a verified `measure`.
        return [*common, "--eval", _track_spec(args.track, "bad_eval_ds"),
                "--track", args.track, "--track-arm", "harmful",
                "--out", out, "--n", str(args.n)]
    if name == "compass":
        return [*common,
                "--harmful", _track_spec(args.track, "bad_eval_ds"),
                "--harmless", _track_spec(args.track, "good_ds"),
                "--track", args.track, "--out", out, "--n", str(args.n),
                # NO PER-PROMPT ROWS. `compass` writes the prompt text and the model's reply
                # beside each margin, and those prompts are harmful. A command somebody runs to
                # find out whether their model is broken should not leave that on their disk
                # without them having asked for it; `senbonzakura compass` still will.
                "--no-margins"]
    if name == "coherence":
        return [*common, "--out", out]
    if name == "capability":
        return [*common, "--out", out, "--n", str(args.capability_n)]
    if name == "drift":
        return [*common, "--base", args.baseline,
                "--prompts", _track_spec(args.track, "good_ds"), "--out", out]
    raise KeyError(name)


def _shown(argv, token):
    """The same command line with the token replaced, for printing and for the artefact."""
    if not token:
        return list(argv)
    return ["***" if a == token else a for a in argv]


def run_stage(name, argv, *, log=print):
    """Call the command that owns this measurement, in this process, and let it write its file.

    In process rather than as a subprocess so that a model already resolved in the Hub cache is
    not re-resolved, and so a traceback stays a traceback. The cost is that a stage raising
    `SystemExit` would end the run, which is exactly what several of them do on a bad input, so
    that is caught here and turned into one failed stage.
    """
    import importlib

    from .entry import DELEGATED

    module_name = DELEGATED[name][0]
    module = importlib.import_module(f".{module_name}", __package__)
    log(f"  senbonzakura {name} {' '.join(argv)}")
    try:
        return module.main(argv)
    except SystemExit as e:
        # A refusal, which is a stage's normal way of saying no.
        #
        # `e.code` IS THE MESSAGE OR THE STATUS, and telling them apart matters. `SystemExit(3)`
        # stringifies to "3", which is truthy, so a naive `str(e) or ...` reports a bare "3" as
        # the reason a stage produced no number, and the table then carries a status code where a
        # sentence belongs. An int is a status; anything else is something a person wrote.
        code = e.code
        reason = f"exited {code}" if isinstance(code, int) else (str(code) if code else "exited 1")
        raise StageError(reason) from e
    except Exception as e:
        # DELIBERATELY BROAD. Whatever one instrument does wrong, the other four still have
        # numbers to report, and a stage is allowed to fail on its own; narrowing this would make
        # the list of exceptions five commands can raise something this module has to track.
        raise StageError(f"{type(e).__name__}: {e}") from e


def read_result(path):
    """One stage's result file, or None when it did not write one."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


#: Which STAMPED metric to read out of each stage's result, and what it means in one line. The
#: caveat is not decoration: every one of these is a figure this project has had to qualify in
#: public.
#:
#: READ OUT OF THE `metrics` BLOCK, not off the top of the file. Every command stamps its figure
#: there through `measurement.stamp`, which exists precisely so that a reader, or the checker in
#: `senbonzakura/check/`, can find a number without knowing which command wrote the file. The
#: first version of this read plain top-level keys, and three of the five happened to have one:
#: `capability` keeps its accuracy under `summary`, so the table printed "not reported" for a
#: model that had just scored 29 of 39 in the log two lines above. A guard that checked the key
#: appeared anywhere in the writing module passed it, which is this project's own through-line:
#: a check that covers one spelling of a defect reports clean on the others.
READINGS = {
    # THE PARTITION IS NOT IN THIS SENTENCE, deliberately, since 2026-09-25. It used to assert
    # "Measured on this track's held-out rows and on no others", which is a claim about the run
    # rather than a description of the metric, and it was false for every default-track run. What
    # the rows were is read from the artefact by `partition_note` and appended per run.
    "score": ("refusal_rate.senbonzakura-ruler",
              "hard refusals left, as a fraction"),
    "compass": ("compass_auc.margin-past-preamble",
                ("harm recognition. 0.5 is a coin toss, which means the model no longer tells a "
                 "harmful request from a harmless one")),
    # NOT the perplexity. This command stamps the negative log likelihood, and the perplexity
    # printed in its own log is that number exponentiated. The first version of this row said
    # "perplexity" over a value of 2.6111 while the stage had printed `ppl=13.61`, which is a
    # mislabelled number and worse than the blank row it replaced. The units below now come out
    # of the stamp, so a row cannot describe itself as something the file disagrees with.
    "coherence": ("coherence",
                  ("how surprised the model is by a fixed neutral passage. Its exponential is "
                   "the perplexity, which the stage's own result file carries. Comparable only "
                   "against the same passage on the same tokeniser")),
    "capability": ("capability",
                   ("what the edit cost, on questions the model either gets right or does not. "
                    "Read `accuracy_reportable` in the file before quoting it")),
    "drift": ("kl",
              ("divergence from the baseline on one ruler, which is the comparable cost. Read "
               "`precision_ok` before quoting it")),
}


def read_figure(result, metric):
    """One stage's stamped figure, or (None, why not) so the table can say which.

    Returns (value, why_not, units). The units come from the stamp rather than from a sentence
    in this module, because a row that names its own units can be wrong about them: the first
    version called the coherence figure a perplexity, over a value that was the log likelihood.

    Why-not is separate from value because "this stage wrote no figure" and "this stage wrote a
    figure I could not find" are different faults and only one of them is the user's problem.
    """
    if not isinstance(result, dict):
        return None, "the result file did not parse as an object", None
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return None, "the result carries no stamped `metrics` block", None
    block = metrics.get(metric)
    if not isinstance(block, dict):
        return None, (f"no stamped metric {metric!r}; the file carries "
                      f"{', '.join(sorted(metrics)) or 'none'}"), None
    value = block.get("value")
    if value is None:
        return None, f"{metric} was stamped without a value", None
    return value, None, block.get("units")


def partition_note(result, metric):
    """What the stamp says this figure was measured on, as a sentence, or "" when it says nothing.

    THE CAPTION MUST NOT OUTRUN THE ARTEFACT. The `score` row's note used to assert "Measured on
    this track's held-out rows and on no others" unconditionally, while the argv builder was
    silently landing on the selection rows for the default track. A caption is a claim; this one
    was the wrong claim in exactly the case a reader was least able to check it. So the claim now
    comes out of the same stamp the number does, and a run whose boundary nothing confirmed says
    so rather than borrowing the verified wording.
    """
    if not isinstance(result, dict):
        return ""
    block = (result.get("metrics") or {}).get(metric)
    if not isinstance(block, dict):
        return ""
    partition = block.get("partition")
    if partition == stamps.MEASURE:
        return "Measured on this track's held-out rows and on no others."
    if partition == stamps.ALL_ROWS:
        return ("Measured on EVERY row of the set, which includes the rows the search selected "
                "on, so this is in-sample and not comparable with a held-out figure.")
    if isinstance(partition, str) and partition.startswith(stamps.UNVERIFIED_PREFIX):
        skipped = partition[len(stamps.UNVERIFIED_PREFIX):]
        return (f"Measured after skipping {skipped} rows, a boundary no manifest confirmed, so "
                f"this is not established to be the held-out partition.")
    if isinstance(partition, str) and partition:
        return f"Measured on the {partition!r} partition."
    return ("The artefact records no partition, so which rows produced this number cannot be "
            "established from it.")


def _figure(value):
    """One number, at a width a column can hold.

    A real run printed `13.613728595914115` in a table beside `0.2083`, which is fifteen decimal
    places of a perplexity nobody can use and a column that no longer lines up. Four places is
    past the precision any of these instruments claims; the full value is in the stage's own
    result file, which is what a reader quotes from.

    NO RENDERING MAY REACH A BOUNDARY THE VALUE DID NOT, since 2026-09-25. `.4f` collapsed
    anything under 5e-05 to the string `0` and `-2e-05` to `-0`. A KL divergence of 3e-05 is an
    ordinary reading for a light edit, and the row's own caption calls it "the comparable cost",
    so the table said an edit had changed nothing when it had changed a little. A refusal rate
    rendered as a flat zero is the specific claim this project has already had to withdraw once.

    Plain `.4g` was tried and rejected: it prints an AUC of 0.99996 as `1`, which is the same
    defect at the other end of the range, and it cuts `13.6137` to `13.61`, losing precision the
    old format kept. So the rule is stated as a property rather than as a format. Four decimal
    places by default, and precision grows only while the rendered text would round to a boundary
    the measurement did not reach: never to zero from a nonzero reading, and never to one from a
    rate or an AUC below it. Both boundaries carry a meaning a reader acts on, which is why they
    are the two worth the extra digits.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    # Exact zero is a measurement, not a rounding, and is the one case that may print as "0".
    if value == 0:
        return "0"

    def _misleads(text):
        try:
            back = float(text)
        except ValueError:                                  # pragma: no cover - defensive
            return False
        if back == 0:                                       # a nonzero reading shown as none
            return True
        return abs(back) == 1 and abs(value) < 1            # below a ceiling, shown as at it

    for places in range(4, 13):
        rendered = f"{value:.{places}f}".rstrip("0").rstrip(".")
        if not _misleads(rendered):
            return rendered
    # Past twelve places a fixed-point rendering is longer than the column and no clearer; the
    # exponent form says "very small" in the space a table has.
    return f"{value:.4g}"


def _interval_of(result, metric):
    """A stage's stamped interval as a printable string, or a word saying why there is none.

    `deterministic` is a claim the writer makes, not an absence: a coherence reading is one forward
    pass over one fixed passage and returns the same number every time, so "deterministic" is the
    honest cell rather than a blank that reads as an oversight.
    """
    if not isinstance(result, dict):
        return ""
    block = (result.get("metrics") or {}).get(metric)
    if not isinstance(block, dict):
        return ""
    if block.get("deterministic"):
        return "deterministic"
    interval = block.get("interval")
    if not (isinstance(interval, (list, tuple)) and len(interval) == 2):
        return "no interval"
    lo, hi = interval
    return f"[{_figure(float(lo))}, {_figure(float(hi))}]"


class Row(NamedTuple):
    """One line of the table, addressed by name rather than by counting.

    A NAMED TUPLE BECAUSE ADDING A COLUMN BROKE EVERY READER, on 2026-09-25. These were bare
    four-tuples and callers reached into them positionally, so `rows[0][3]` meant "the note".
    Inserting the interval column moved the note to index 4, and every one of those readers
    silently began asserting against a different field. Nothing about that is visible at the call
    site, which is the whole problem: a positional tuple makes a change of shape look like a
    change of nothing.

    `note` stays at index 3 so readers written before the column existed still mean what they
    meant, and the new field goes last. That ordering is load-bearing rather than tidy.
    """

    stage: str
    figure: str
    units: str
    note: str
    interval: str = ""


def verdict_rows(results):
    """A `Row` per stage, with a named absence where one produced nothing."""
    rows = []
    for name in STAGE_ORDER:
        if name not in results:
            continue
        r = results[name]
        metric, note = READINGS[name]
        if isinstance(r, str):                  # a failure, carried as its message
            rows.append(Row(name, "not measured", "", r))
            continue
        value, why_not, units = read_figure(r, metric)
        if value is None:
            # The REASON, not a blank. A row saying only "not reported" beside a stage that
            # printed its number two lines earlier tells the reader nothing they can act on.
            rows.append(Row(name, "not reported", "", why_not))
            continue
        where = partition_note(r, metric)
        rows.append(Row(name, _figure(value), units or "",
                        f"{note}. {where}" if where else note,
                        _interval_of(r, metric)))
    return rows


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura measure",
        description="Run every instrument against one model and print one table. The same "
                    "commands, the same numbers, one command line.")
    ap.add_argument("model", help="the model to measure: a Hub id or a local directory")
    ap.add_argument("--out", default="measurements",
                    help="directory for the result files, one per instrument (default: "
                         "./measurements)")
    ap.add_argument("--track", default="default",
                    help="the evaluation track the prompts come from. 'default' is the one "
                         "bundled in this install; otherwise a directory `senbonzakura track` "
                         "built")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu")
    ap.add_argument("--baseline", default=None,
                    help="the model this one was edited from. Given, the coherence cost is also "
                         "measured against it on one ruler, which is the comparable number; "
                         "without it that stage is skipped rather than guessed at")
    ap.add_argument("--n", type=int, default=200,
                    help="prompts per instrument for refusal and harm recognition (default: 200)")
    ap.add_argument("--capability-n", type=int, default=40, dest="capability_n",
                    help="graded questions for the capability probe (default: 40). It generates "
                         "long answers, so this is the stage that decides how long the run takes")
    ap.add_argument("--hf-token", default=None, dest="hf_token",
                    help="a Hugging Face token for a gated model. Never printed, and never "
                         "written into the summary")
    ap.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code",
                    help="allow models that ship custom modelling code; off by default")
    ap.add_argument("--only", action="append", choices=STAGE_ORDER, default=None,
                    help="run only this instrument. Repeatable")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages whose result file is already there. Without it an "
                         "interrupted run resumes instead of repeating what it already did")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="print the command line each stage would be given and stop. No model is "
                         "loaded and nothing is written")
    return ap


def wanted_stages(args):
    """Which instruments this invocation runs, and why one is left out.

    `drift` needs a second model, so it is in only when `--baseline` names one. Asked for
    explicitly without one, it is a refusal rather than a silent omission: somebody who typed
    `--only drift` has said what they want and should be told why they are not getting it.
    """
    chosen = list(args.only) if args.only else [n for n in STAGE_ORDER if n != "drift"]
    if "drift" in chosen and not args.baseline:
        if args.only:
            raise SystemExit(
                "senbonzakura measure: --only drift needs --baseline, because drift is the "
                "coherence cost AGAINST the model this one was edited from. There is nothing to "
                "compare a single model with.")
        chosen.remove("drift")
    if args.baseline and not args.only and "drift" not in chosen:
        chosen.append("drift")
    # A FLAG THAT CHANGES NOTHING IS REFUSED, not accepted quietly. `--only score --baseline X`
    # used to run one stage, never mention the baseline, and exit 0, so the operator got a table
    # that looked like the one they asked for and was missing the comparison they named a model
    # for. Every other unusable combination in this command refuses; this one was silent, which is
    # the shape the whole 2026-09-25 panel kept finding.
    if args.baseline and "drift" not in chosen:
        raise SystemExit(
            f"senbonzakura measure: --baseline {args.baseline} is only read by the drift stage, "
            f"and --only {' '.join(args.only)} does not include it, so the baseline would be "
            f"accepted and ignored. Add drift to --only, or drop --baseline.")
    return [n for n in STAGE_ORDER if n in chosen]


def run(args, *, log=print):
    """Every stage in order, each allowed to fail on its own. Returns (results, failures)."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = wanted_stages(args)

    results = {}
    failures = []
    for name in stages:
        argv = stage_argv(name, args, out_dir)
        path = out_dir / OUTPUTS[name]
        if path.is_file() and not args.force:
            log(f"{name}: already measured, at {path}. --force re-runs it")
            # A RESUMED STAGE WHOSE FILE CANNOT BE READ IS A FAILURE, like any other.
            #
            # This set the message and did not append to `failures`, so a truncated or corrupt
            # result from an earlier run printed "not measured" in the table and the command
            # exited 0, against a module docstring promising a non-zero status if any stage
            # failed. The resume path is exactly where a half-written file turns up, which makes
            # it the worst place to treat one as merely absent.
            got = read_result(path)
            if got is None:
                log(f"  FAILED: {path} exists and could not be read")
                results[name] = "the result file could not be read"
                failures.append(name)
            else:
                results[name] = got
            continue
        log(f"{name}:")
        started = time.monotonic()
        try:
            run_stage(name, argv, log=log)
        except StageError as e:
            log(f"  FAILED after {time.monotonic() - started:.0f}s: {e}")
            results[name] = str(e)
            failures.append(name)
            continue
        log(f"  done in {time.monotonic() - started:.0f}s")
        got = read_result(path)
        results[name] = got if got is not None else "the stage reported success and wrote no file"
        if got is None:
            failures.append(name)
    return results, failures


def format_table(rows):
    """The table, with an interval column.

    THE ONE ARTEFACT BUILT TO BE QUOTED WAS THE ONE THAT STRIPPED THE UNCERTAINTY. Every stage
    stamps an interval where it has one, `compass` as `auc_ci`, `capability` as `accuracy_ci`,
    `drift` as `kl_ci`, and `score` since 2026-09-25 as a Wilson interval, and this table read
    `value` and threw the rest away. At this command's own defaults `--capability-n 40` carries an
    interval of roughly fifteen percentage points, and the table printed the point estimate alone.

    `METHOD.md` says a figure with no interval cannot be gated on and asks "is there an interval?"
    in its own checklist. The table a reader quotes from was failing that check, which is how the
    incident it recounts happened: a rate that moved from under a tenth to over half because n had
    fallen from 200 to 4, with nothing beside it to say so.
    """
    if not rows:
        return ["nothing was measured"]
    name_w = max(len(r.stage) for r in rows)
    unit_w = max(len(r.units) for r in rows)
    iv_w = max(len(r.interval) for r in rows)
    # UNITS BESIDE THE NUMBER, with the interval after. The first version of this column put
    # the interval between them, which separates a figure from the thing that says what it
    # measures: "2.6111  nats-per-token" became "2.6111  no interval  nats-per-token". A
    # reader scanning the table reads the figure and its units as one phrase.
    return [f"  {r.stage:<{name_w}}  {r.figure:>12}  {r.units:<{unit_w}}  "
            f"{r.interval:<{iv_w}}  {r.note}"
            for r in rows]


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)

    if args.dry_run:
        for name in wanted_stages(args):
            shown = _shown(stage_argv(name, args, out_dir), args.hf_token)
            print(f"senbonzakura {name} {' '.join(shown)}")
        return 0

    print(f"measuring {args.model} on track {args.track}, into {out_dir}/")
    results, failures = run(args, log=print)

    print()
    print("what the instruments said:")
    for line in format_table(verdict_rows(results)):
        print(line)
    print()
    print("None of this is a pass or a fail. Each figure is comparable only against the same "
          "instrument on the same track; see docs/guide/what-we-know before quoting any of it.")

    # ATOMIC, like every other result this project writes. `write_text` truncates and then writes,
    # so a kill during it leaves a file that exists, is not empty, and is not a result. This is the
    # file `senbonzakura report` reads, and it is written at the end of a run that may have taken
    # hours, which is exactly when somebody reaches for Ctrl+C. Every other writer here goes
    # through `crashsafe.atomic_write`; this one was missed.
    summary = out_dir / "measure.json"
    with atomic_write(summary) as fh:
        json.dump({
            "schema": "senbonzakura-measure/1",
            "model": args.model,
            "track": args.track,
            "baseline": args.baseline,
            "stages": {n: OUTPUTS[n] for n in results},
            "failed": failures,
            # The command lines, with the token removed, so a reader can re-run any single stage.
            "commands": {n: _shown(stage_argv(n, args, out_dir), args.hf_token) for n in results},
        }, fh, indent=2)
        fh.write("\n")
    print(f"wrote {summary}")

    if failures:
        print(f"\n{len(failures)} instrument(s) did not produce a number: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry

    module_entry(main)
