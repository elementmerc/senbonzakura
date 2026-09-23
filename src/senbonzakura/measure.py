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


def stage_argv(name, args, out_dir):
    """The command line this stage would have been given by hand.

    Built rather than executed here so it can be printed, tested without a GPU, and quoted in the
    table. A reader who wants to re-run one stage with different knobs can copy the line.
    """
    common = ["--model", args.model, "--device", args.device]
    if args.hf_token:
        # The VALUE is never printed: `_shown` below replaces it. It is here because the stage
        # needs it, and a gated model fails four ways without it.
        common += ["--hf-token", args.hf_token]
    if args.trust_remote_code:
        common += ["--trust-remote-code"]
    out = str(out_dir / OUTPUTS[name])

    if name == "score":
        return [*common, "--eval", _track_spec(args.track, "bad_eval_ds"),
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
        # A refusal, which is a stage's normal way of saying no. `str(e)` is the message where
        # there is one and the status where there is not.
        raise StageError(str(e) or f"exited {e.code}") from e
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
    "score": ("refusal_rate.senbonzakura-ruler",
              ("hard refusals left, as a fraction. Measured on this track's held-out rows and on "
               "no others")),
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


def _figure(value):
    """One number, at a width a column can hold.

    A real run printed `13.613728595914115` in a table beside `0.2083`, which is fifteen decimal
    places of a perplexity nobody can use and a column that no longer lines up. Four places is
    past the precision any of these instruments claims; the full value is in the stage's own
    result file, which is what a reader quotes from.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def verdict_rows(results):
    """(stage, figure, units, note) per stage, with a named absence where one produced nothing."""
    rows = []
    for name in STAGE_ORDER:
        if name not in results:
            continue
        r = results[name]
        metric, note = READINGS[name]
        if isinstance(r, str):                  # a failure, carried as its message
            rows.append((name, "not measured", r))
            continue
        value, why_not, units = read_figure(r, metric)
        if value is None:
            # The REASON, not a blank. A row saying only "not reported" beside a stage that
            # printed its number two lines earlier tells the reader nothing they can act on.
            rows.append((name, "not reported", "", why_not))
            continue
        rows.append((name, _figure(value), units or "", note))
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
            results[name] = read_result(path) or "the result file could not be read"
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
    if not rows:
        return ["nothing was measured"]
    name_w = max(len(name) for name, _, _, _ in rows)
    unit_w = max(len(u) for _, _, u, _ in rows)
    return [f"  {name:<{name_w}}  {figure:>12}  {units:<{unit_w}}  {note}"
            for name, figure, units, note in rows]


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

    summary = out_dir / "measure.json"
    summary.write_text(json.dumps({
        "schema": "senbonzakura-measure/1",
        "model": args.model,
        "track": args.track,
        "baseline": args.baseline,
        "stages": {n: OUTPUTS[n] for n in results},
        "failed": failures,
        # The command lines, with the token removed, so a reader can re-run any single stage.
        "commands": {n: _shown(stage_argv(n, args, out_dir), args.hf_token) for n in results},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary}")

    if failures:
        print(f"\n{len(failures)} instrument(s) did not produce a number: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":   # pragma: no cover
    from .entry import module_entry

    module_entry(main)
