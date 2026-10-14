# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura attribute`: which blocks of a model carry refusal, as a measurement.

This is the orchestration and the artefact. The measurement is `attribution`, the real model
behind it is `attribution_hf`, and neither reads anything the abliteration tool wrote.

TWO READOUT ARMS, REPORTED TOGETHER. The disjoint arm was declared before any real model was run;
the purity arm was added after the fit partition had been inspected. The artefact says which is
which, and where they disagree NEITHER wins: the top level status is `arms_disagree` and no block
may be tinted. A second arm that gives the nicer answer is not thereby the better instrument.

WHAT IS WRITTEN. One JSON file of numbers, digests and decoded single opener tokens. No prompt, no
reply and no prompt set ever reaches it, which `tools/ci/check_prompt_artefacts.py` relies on.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

from . import attribution as A
from ._version import __version__

SCHEMA = "senbonzakura-attribution/1"
ARMS_DISAGREE = "arms_disagree"
SELFTEST_FAILED = "instrument_failed_selftest"
STANDIN_LIMIT = (
    "The behaviour removed control is an in-memory stand-in (the direction removed from every "
    "block by hook), not a checkpoint the abliteration tool produced.")


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} cannot be written to the artefact")


def combine(arms, selftest):
    """The top level status from the arms and the self test. Pure, so it can be tested alone.

    Neither arm wins a disagreement. A card reads `status` and `tint_allowed` and nothing else.
    """
    results = list(arms.values())
    statuses = {r["status"] for r in results}
    if not selftest.get("passed"):
        return SELFTEST_FAILED, ("the planted models did not all get their planted status, so "
                                 "this instrument is not currently trustworthy"), False
    if len(statuses) > 1:
        detail = "; ".join(f"{name} arm: {r['status']}" for name, r in arms.items())
        return ARMS_DISAGREE, (f"the two readout arms disagree ({detail}). Neither is preferred; "
                               f"both are reported."), False
    status = results[0]["status"]
    if status == A.LOCALISED:
        sets = {tuple(b["index"] for b in r["blocks"] if b["exceeds_null"]) for r in results}
        if len(sets) > 1:
            return ARMS_DISAGREE, ("both arms say localised but name different blocks: "
                                   + "; ".join(f"{n}: {[b['index'] for b in r['blocks'] if b['exceeds_null']]}"
                                               for n, r in arms.items())), False
    return (status, results[0]["status_reason"],
            all(r.get("tint_allowed") for r in results) and status == A.LOCALISED)


def model_specific_limits(arms):
    """Limits that only a run can know, drawn from what it measured."""
    out = []
    for name, result in arms.items():
        detail = (result.get("controls") or {}).get("readout_agreement_detail") or {}
        sets = detail.get("token_sets") or {}
        share = sets.get("refusal_set_covers_fit_refusals")
        if share is not None and share < 0.5:
            out.append(f"{name} arm: the refusal tokens open only {share:.0%} of this model's fit "
                       f"refusals, so on this model the readout is mostly measuring the absence "
                       f"of a compliance opener rather than the presence of a refusal.")
        cover = detail.get("onset_coverage")
        if cover is not None and cover < 0.5:
            out.append(f"{name} arm: only {cover:.0%} of search replies open with a token in "
                       f"either set; the rest are predicted through logit mass.")
        style = (result.get("controls") or {}).get("style_direction") or {}
        if style.get("flagged"):
            out.append(f"{name} arm: removing a prompt length direction moves the readout about "
                       f"as much as the behaviour direction does, so the behaviour direction may "
                       f"be partly reading style.")
    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog="senbonzakura attribute",
        description="Say which blocks of a model carry refusal, as a measurement with its own "
                    "controls. It measures the unedited model; where an edit wrote is not "
                    "evidence of where the behaviour lived, and nothing here reads it.")
    p.add_argument("model", help="Hugging Face model id or local path (unedited)")
    p.add_argument("--track", required=True,
                   help="a track directory; its recorded fit, search and measure boundaries are used")
    p.add_argument("--out", required=True, help="directory to write attribution.json into")
    p.add_argument("--device", default="cuda", help="cuda or cpu (default: cuda)")
    p.add_argument("--revision", default=None, help="model revision to pin")
    p.add_argument("--n-measure", type=int, default=400,
                   help="requests per arm to measure effects on (default: 400)")
    p.add_argument("--n-confirm", type=int, default=150,
                   help="fresh requests per arm for the generation check (default: 150)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for every resample, shuffle and random choice (default: 0)")
    p.add_argument("--batch", type=int, default=16,
                   help="prompts per forward pass; results are reproducible only at a fixed "
                        "batch size (default: 16)")
    p.add_argument("--budget-minutes", type=float, default=110.0,
                   help="refuse to start if the readout work alone is projected to exceed this")
    p.add_argument("--force", action="store_true",
                   help="redo arms that already have a result for these exact inputs")
    return p


def _stage(log, name):
    class _S:
        def __enter__(self):
            self.t = time.monotonic()
            log(f"[{name}] start")

        def __exit__(self, kind, *_):
            log(f"[{name}] {'failed' if kind else 'done'} in {time.monotonic() - self.t:.0f}s")
    return _S()


def run(args, *, log=print):
    import torch

    from . import attribution_hf as H
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    conv = A.DEFAULT

    with _stage(log, "load"):
        model, tok, identity = H.load(args.model, device=args.device, revision=args.revision)
    with _stage(log, "partitions"):
        parts, digests = H.build_partitions(args.track, n_measure=args.n_measure,
                                            n_confirm=args.n_confirm, seed=args.seed)

    nb = H.HFSubject(model, tok).n_blocks
    fingerprint = {"model": identity, "digests": digests, "seed": args.seed,
                   "instrument": A.INSTRUMENT_VERSION, "conventions": conv.__dict__}
    _project(model, tok, parts, nb, conv, args, log)

    arms = {}
    todo = []
    for arm_name in (H.ARM_DISJOINT, H.ARM_PURITY):
        cache = out / f"arm-{arm_name}.json"
        if cache.exists() and not args.force:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == json.loads(json.dumps(fingerprint, default=_jsonable)):
                log(f"[{arm_name}] reusing {cache} (same inputs)")
                arms[arm_name] = saved["result"]
                continue
        todo.append(arm_name)
    subjects = {n: H.HFSubject(model, tok, batch=args.batch, arm=n, log=log) for n in todo}
    with _stage(log, "readout tokens"):
        for subject in subjects.values():
            subject.prepare(parts)
    shared = H.share_readouts(subjects.values())
    for arm_name, subject in subjects.items():
        with _stage(log, f"arm {arm_name}"):
            result = A.attribute(
                subject, parts, seed=args.seed, conv=conv, log=log,
                predictive=lambda s, p, e, d, r: H.predictive_check(s, p, e, d, r, seed=args.seed,
                                                                    conv=conv))
        result["readout_arm"] = {"name": arm_name, **H.ARM_DECLARATIONS[arm_name]}
        arms[arm_name] = result
        _write(out / f"arm-{arm_name}.json", {"fingerprint": fingerprint, "result": result})
    if shared is not None:
        log(f"shared forward passes: {shared.misses} computed, {shared.hits} reused by the other arm")

    with _stage(log, "selftest"):
        selftest = A.selftest()
    with _stage(log, "behaviour removed stand-in"):
        removed = _behaviour_removed(model, tok, parts, args, conv, log)

    status, reason, tint = combine(arms, selftest)
    artefact = {
        "schema": SCHEMA, "created": stamp,
        "instrument": {"name": A.INSTRUMENT, "version": A.INSTRUMENT_VERSION,
                       "estimator": A.ESTIMATOR, "units": A.UNITS, "senbonzakura": __version__},
        "model": identity, "n_blocks": nb,
        "behaviour": {
            "name": "refusal",
            "contrast": "harmful against harmless requests from the track",
            "readout": "first-token refusal margin, validated against generated replies",
            "track": Path(args.track).name, "partitions": digests,
            "one_prompt_per_request": True,
        },
        "sampling": {"seed": args.seed, "n_measure": args.n_measure, "n_confirm": args.n_confirm,
                     "batch": args.batch, "torch": torch.__version__},
        "status": status, "status_reason": reason, "tint_allowed": tint,
        "arms_agree": status != ARMS_DISAGREE,
        "arms": arms,
        "controls": {"selftest": selftest, "behaviour_removed_standin": removed},
        "limits": [*A.LIMITS, STANDIN_LIMIT, *model_specific_limits(arms)],
        "comparable_with": {"schema": SCHEMA, "instrument_version": A.INSTRUMENT_VERSION,
                            "estimator": A.ESTIMATOR, "precision": identity["precision"],
                            "model_revision": identity["revision"]},
    }
    _write(out / "attribution.json", artefact)
    log(f"status: {status}  tint_allowed: {tint}\n  {reason}\n  written to {out / 'attribution.json'}")
    return artefact


def _write(path, doc):
    from .crashsafe import atomic_write
    with atomic_write(str(path)) as handle:
        handle.write(json.dumps(doc, indent=1, sort_keys=True, default=_jsonable))


def _project(model, tok, parts, nb, conv, args, log):
    """Time a real pass and refuse before the long work if it will not fit the budget."""
    from . import attribution_hf as H
    probe = H.HFSubject(model, tok, batch=args.batch, log=log)
    sample = list(parts["measure"]["harmful"].prompts)[:64]
    import torch
    direction = torch.zeros(model.config.hidden_size)
    direction[0] = 1.0
    probe.refusal_ids = torch.tensor([0], device=model.device)
    probe.comply_ids = torch.tensor([1], device=model.device)
    # The first pass pays for CUDA start up, kernel selection and tokenising, none of which recur,
    # and timing it projected 1,061 minutes for a run that is an order of magnitude cheaper. So one
    # untimed pass warms everything and the timed pass is the steady state.
    probe.readout(sample, direction, range(nb))
    t = time.monotonic()
    probe.readout(sample, direction, range(nb))
    per_prompt = (time.monotonic() - t) / len(sample)
    prompts = len(parts["measure"]["harmful"].prompts) + len(parts["measure"]["harmless"].prompts)
    passes = (2 + 3 * nb) + (conv.shuffled_label_replicates + conv.random_direction_replicates) \
        * (nb + 2)
    minutes = passes * prompts * per_prompt / 60.0
    log(f"projection: {per_prompt * 1000:.1f} ms per prompt with every block hooked; "
        f"{passes} passes x {prompts} prompts, shared by both arms, is about {minutes:.0f} minutes "
        f"of readout work (a projection from {len(sample)} prompts, generation not included)")
    if minutes > args.budget_minutes:
        raise SystemExit(
            f"projected {minutes:.0f} minutes of readout work is over the {args.budget_minutes:.0f} "
            f"minute budget. Anything over two GPU hours is a conversation with the operator "
            f"first; lower --n-measure or raise --budget-minutes knowingly.")


def _behaviour_removed(model, tok, parts, args, conv, log):
    """Run the readout stage on the model with the direction already removed from every block.

    An instrument that has never been seen to return nothing has not been validated. With the
    behaviour removed the readout cannot be built or the total effect collapses; either way it
    must NOT report a localised behaviour.
    """
    import dataclasses

    from . import attribution_hf as H
    base = H.HFSubject(model, tok, batch=args.batch, log=log)
    base.prepare(parts)
    if base.refusal_ids is None:
        return {"ran": False, "why": base.readout_why_not}
    res = {s: base.residuals(list(parts["fit"][s].prompts)) for s in ("harmful", "harmless")}
    srch = {s: base.residuals(list(parts["search"][s].prompts)) for s in ("harmful", "harmless")}
    direction, _, _ = A.fit_direction(res["harmful"], res["harmless"], srch["harmful"],
                                      srch["harmless"])
    edited = H.HFSubject(model, tok, batch=args.batch, log=log)
    edited.edit_standing(direction, range(edited.n_blocks))
    fast = dataclasses.replace(conv, shuffled_label_replicates=6, random_direction_replicates=6,
                               resamples=300)
    result = A.attribute(edited, parts, seed=args.seed, conv=fast)
    return {"ran": True, "status": result["status"], "status_reason": result["status_reason"],
            "expected_not": A.LOCALISED, "ok": result["status"] != A.LOCALISED,
            "conventions_note": "replicate counts reduced; this control stops early when it can"}


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_measure < A.DEFAULT.min_prompts or args.n_confirm < 0:
        raise SystemExit(f"--n-measure must be at least {A.DEFAULT.min_prompts}; below that a "
                         f"rate is not reported.")
    try:
        run(args)
    except A.AttributionError as e:
        print(f"attribution stopped: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
