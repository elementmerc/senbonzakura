# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Two follow-ups to the first Qwen2.5-0.5B run, declared before either was run.

WHY THEY EXIST. That run said `diffuse` with 91% of the single block effect in blocks 12 to 21 and
the largest effects at blocks 17 to 20. Two things in it need testing and neither can be read off
the data that produced it:

  BAND. Blocks 12 to 21 were picked by looking at the measure partition's own results, so any claim
  about them is exploratory. The 150 fresh requests per arm that the run set aside were never used
  to rank anything, so they can test it.

  POSITION. The direction was fitted at position 21 of 24 (the residual after block 20) and the
  largest single block effect is block 20. A profile that peaks at the block just before the
  fitting position is exactly what an artefact of the fitting position would look like.

EVERYTHING BELOW WAS WRITTEN BEFORE EITHER FOLLOW-UP WAS RUN. The band, the positions, the
predictions and the criteria that would falsify them are fixed here; the judging functions are the
criteria as code. `PREDECLARATION_SHA256` goes into the artefact so a reader can check the text was
not changed afterwards. A negative result is kept and reported.

THESE ARE EXPLORATORY AND SAY SO. The band follow-up is a hypothesis generated from one run's data
and tested on held-out requests; a supported result is "survived one held-out test", not a discovery.
"""
from __future__ import annotations

import hashlib

#: Blocks 12 to 21 inclusive, as 0 based decoder block indices. Fixed.
BAND = tuple(range(12, 22))
#: Fitting positions to try (position p is the residual after block p - 1). 21 is the original.
POSITIONS = (15, 18, 24)
REFERENCE_POSITION = 21
#: The share of the positive single block effect that must sit in the band for the profile to be
#: called stable at a position. A declared convention.
STABLE_BAND_SHARE = 0.6
#: A peak within this many blocks of the block just before the fitting position counts as sitting
#: at the fitting position; within this many of the reference peak counts as staying put.
PEAK_TOLERANCE = 1

PREDECLARATION = f"""\
FOLLOW-UP 1, THE BAND (exploratory, generated from the first run's data).
Prediction: removing the direction from blocks {BAND[0]} to {BAND[-1]} while the model writes its
replies lowers the harmful refusal rate on the 150 fresh requests by more than removing it from the
same number of blocks drawn at random from the remaining blocks, with the interval on the
difference (requests resampled whole) entirely above zero.
Falsified if: the lower end of that interval is at or below zero. A falsified band is reported as
not supported and kept in the artefact.

FOLLOW-UP 2, THE FITTING POSITION.
The direction is refitted at positions {list(POSITIONS)} (the original is {REFERENCE_POSITION}) on the
same fit partition, and the single block effects are recomputed on the same measure requests.
The profile is called STABLE if the largest block sits within {PEAK_TOLERANCE} of the reference peak
at two or more of the three positions AND the band holds at least {STABLE_BAND_SHARE:.0%} of the
positive effect at all three.
It is called TRACKING THE POSITION if the largest block sits within {PEAK_TOLERANCE} of the block just
before the fitting position (position - 1) at two or more of the three positions, and the peak
moves in the same order as the positions.
Anything else is MIXED and reported as mixed. No threshold of the original run is changed.
"""
PREDECLARATION_SHA256 = hashlib.sha256(PREDECLARATION.encode("utf-8")).hexdigest()


def positive_share(effects, blocks):
    """The share of the summed positive effect that falls in `blocks`."""
    positive = [max(float(e), 0.0) for e in effects]
    total = sum(positive)
    return sum(positive[k] for k in blocks if k < len(positive)) / total if total > 0 else None


def peak(effects):
    return max(range(len(effects)), key=lambda k: (float(effects[k]), -k))


def profile_stats(effects, position, reference_peak):
    top = peak(effects)
    return {
        "position": position, "peak_block": top, "peak_effect": float(effects[top]),
        "top3": sorted(range(len(effects)), key=lambda k: -float(effects[k]))[:3],
        "band_share": positive_share(effects, BAND),
        "peak_near_reference_peak": abs(top - reference_peak) <= PEAK_TOLERANCE,
        "peak_near_fitting_position": abs(top - (position - 1)) <= PEAK_TOLERANCE,
    }


def judge_positions(profiles, reference_effects):
    """The verdict on the position follow-up, from the declared criteria alone.

    `profiles` maps each position in `POSITIONS` to its list of single block effects.
    """
    missing = [p for p in POSITIONS if p not in profiles]
    if missing:
        raise ValueError(f"positions {missing} have no profile, so the declared test cannot be "
                         f"judged")
    reference_peak = peak(reference_effects)
    rows = {p: profile_stats(profiles[p], p, reference_peak) for p in POSITIONS}
    stays = sum(r["peak_near_reference_peak"] for r in rows.values())
    sits = sum(r["peak_near_fitting_position"] for r in rows.values())
    shares_ok = all((r["band_share"] or 0.0) >= STABLE_BAND_SHARE for r in rows.values())
    peaks_in_order = [rows[p]["peak_block"] for p in POSITIONS]
    moves_with_position = peaks_in_order == sorted(peaks_in_order) and len(set(peaks_in_order)) > 1
    if stays >= 2 and shares_ok:
        verdict = "stable"
    elif sits >= 2 and moves_with_position:
        verdict = "tracks_position"
    else:
        verdict = "mixed"
    return {"verdict": verdict, "rows": rows, "reference_peak": reference_peak,
            "peaks_staying_put": stays, "peaks_at_fitting_position": sits,
            "band_share_ok_everywhere": shares_ok, "peak_moves_with_position": moves_with_position}


def judge_band(check):
    """The verdict on the band follow-up from the generation check's own result."""
    if not check.get("ran"):
        return {"verdict": "not_run", "why": check.get("why", "the check did not run")}
    low = check["contrast"]["ci"][0]
    return {"verdict": "supported" if low > 0.0 else "not_supported",
            "interval_lower_end": low,
            "framing": ("exploratory: the band came from the first run's own data and survived one "
                        "held-out test" if low > 0.0 else
                        "the band did not beat random blocks of the same number on held-out "
                        "requests; the negative result is kept")}


# --------------------------------------------------------------------------- where it starts

START_POSITIONS = (15, 18, 21, 24)
#: Blocks 0 to 9 inclusive, the "early" set.
EARLY = tuple(range(10))
#: The first block of the comparison set, which runs to the last block of the model.
LATE_FROM = 10
#: Declared with the test. The largest share of the position follow-up's total effect that blocks 0
#: to 9 held, in the tables, was about 14% (at position 15), and 0.15 is that rounded up in
#: percentage points of the harmful refusal rate. A convention chosen AFTER seeing the tables.
EARLY_DROP_CEILING = 0.15
#: How much more the comparison set must lower refusals than the early set, for the test to be
#: able to see a difference at all.
CONTRAST_FLOOR = 0.15

START_DECLARATION = f"""\
FOLLOW-UP 3, WHERE IT STARTS. Declared before it was run, and before anything further was looked at.

PROVENANCE OF THE IDEA, stated so nobody downstream presents it as predeclared. It came from READING
the four per block tables of follow-up 2 (computed on the measure partition), in which blocks 0 to 9
sit near zero at every fitting position. It was NOT declared before those tables existed. The two
numbers below were chosen after seeing them. The test itself runs on 150 requests that no earlier
step used.

Claim: the direction is not written by blocks 0 to 9, whichever of the positions
{list(START_POSITIONS)} it is fitted at.
For each of those positions the direction is refitted on the fit partition, and while the model
writes its replies it is removed from (a) blocks 0 to 9 and (b) blocks {LATE_FROM} to the last block.
Compared with no ablation, on the fresh requests, requests resampled whole:
  1. the early set lowers the harmful refusal rate by at most {EARLY_DROP_CEILING:.2f}: the upper end
     of the 95% interval on its drop is no greater than {EARLY_DROP_CEILING:.2f};
  2. the later set lowers it by more than the early set, with the lower end of the interval on the
     difference above {CONTRAST_FLOOR:.2f}. This is a positive control: if the later set cannot be
     seen to matter more, the test could not have shown a difference and says nothing.
FALSIFIED if criterion 1 fails at ANY of the four positions: the early blocks do carry the direction
at that position, and there is no stable lower edge.
UNINFORMATIVE if criterion 1 holds everywhere but criterion 2 fails at any position.
SUPPORTED only if both hold at all four. A supported result is "survived one held out test" and
remains exploratory. The upper edge is not addressed by this test at all.
"""
START_SHA256 = hashlib.sha256(START_DECLARATION.encode("utf-8")).hexdigest()


def judge_start(rows):
    """The verdict from the four positions' results, by the declared criteria and nothing else.

    `rows` maps each position to `{"early_drop_ci": [lo, hi], "contrast_ci": [lo, hi]}`.
    """
    missing = [p for p in START_POSITIONS if p not in rows]
    if missing:
        raise ValueError(f"positions {missing} have no result, so the declared test cannot be "
                         f"judged")
    broken = [p for p in START_POSITIONS if rows[p]["early_drop_ci"][1] > EARLY_DROP_CEILING]
    blind = [p for p in START_POSITIONS if rows[p]["contrast_ci"][0] <= CONTRAST_FLOOR]
    if broken:
        verdict = "falsified"
    elif blind:
        verdict = "uninformative"
    else:
        verdict = "supported"
    return {"verdict": verdict, "positions_where_early_blocks_matter": broken,
            "positions_where_the_test_could_not_see_a_difference": blind,
            "framing": ("exploratory: the idea came from reading tables, and a supported result is "
                        "one held out test" if verdict == "supported" else
                        "kept as it came out; see the positions listed")}


# ------------------------------------------------------------------------------ the follow-up run

SCHEMA = "senbonzakura-attribution-followups/1"
START_SCHEMA = "senbonzakura-attribution-start/1"


def build_parser():
    from . import argresolve
    from . import attribution as A
    p = argresolve.ParserThatNamesUnknownFlags(
        allow_abbrev=False,
        prog="python -m senbonzakura.attribution_followups",
        description="Run the declared follow-ups against a finished attribution run. Prefix the "
                    "arguments with `start` for the where-it-starts test instead.")
    p.add_argument("model", help="the model the original run measured, as it named it")
    p.add_argument("--track", required=True, help="the track directory the original run used")
    p.add_argument("--artefact", required=True,
                   help="the attribution.json the original run produced; the inputs must match it")
    p.add_argument("--out", required=True, help="directory to write the result file into")
    p.add_argument("--n-measure", type=argresolve.whole_number(
        "--n-measure", minimum=A.DEFAULT.min_prompts), default=400,
        help="requests per arm the original run measured on (default: 400)")
    p.add_argument("--n-confirm", type=argresolve.whole_number("--n-confirm"), default=150,
                   help="fresh requests per arm the original run set aside (default: 150)")
    p.add_argument("--seed", type=argresolve.whole_number("--seed"), default=0,
                   help="the seed the original run used (default: 0)")
    p.add_argument("--batch", type=argresolve.whole_number("--batch", minimum=1), default=16,
                   help="prompts per forward pass; use the original run's value (default: 16)")
    p.add_argument("--device", default="cuda", help="cuda or cpu (default: cuda)")
    from .parser import add_format_flags
    add_format_flags(p)
    p.add_argument("--n-fresh", type=argresolve.whole_number("--n-fresh", minimum=1), default=150,
                   help="requests no earlier step used, for the where-it-starts test "
                        "(default: 150)")
    return p


def _loading_options(args):
    """The loader flags, read here because this module is the one that declares them."""
    return {"load_in_4bit": args.load_in_4bit, "load_in_8bit": args.load_in_8bit,
            "gguf_file": args.gguf_file or None, "trust_remote_code": args.trust_remote_code}


def run(args, *, log=print):
    import json
    import time
    from pathlib import Path

    from . import attribution as A
    from . import attribution_hf as H
    from . import attribution_run as R

    original = json.loads(Path(args.artefact).read_text(encoding="utf-8"))
    t0 = time.monotonic()
    model, tok, identity = H.load(args.model, device=args.device,
                                  revision=original["model"]["revision"],
                                  **_loading_options(args))
    H.refuse_other_loading_form(identity, original["model"])
    parts, digests = H.build_partitions(args.track, n_measure=args.n_measure,
                                        n_confirm=args.n_confirm, seed=args.seed)
    wanted = original["behaviour"]["partitions"]
    got = json.loads(json.dumps(digests))
    mismatched = [k for k in wanted if wanted[k] != got.get(k)]
    if mismatched or identity["revision"] != original["model"]["revision"]:
        differing = mismatched or "model revision"
        raise A.AttributionError(
            f"these are not the inputs the original run used (differing: {differing}), so a "
            f"follow-up on them would answer a different question.")

    subjects = {n: H.HFSubject(model, tok, batch=args.batch, arm=n, log=log)
                for n in (H.ARM_DISJOINT, H.ARM_PURITY)}
    for s in subjects.values():
        s.prepare(parts)
    H.share_readouts(subjects.values())
    first = subjects[H.ARM_DISJOINT]
    nb = first.n_blocks

    res = {k: {s: first.residuals(list(parts[k][s].prompts)) for s in ("harmful", "harmless")}
           for k in ("fit", "search")}
    direction, position, aucs = A.fit_direction(
        res["fit"]["harmful"], res["fit"]["harmless"], res["search"]["harmful"],
        res["search"]["harmless"])
    orig_dir = original["arms"][H.ARM_DISJOINT]["direction"]
    if position != orig_dir["position"] or abs(aucs[position] - orig_dir["held_out_auc"]) > 1e-9:
        raise A.AttributionError(
            f"the direction refitted here (position {position}, held out AUC {aucs[position]:.6f}) "
            f"is not the one the original run used ({orig_dir}), so the run is not reproducing it.")
    log(f"direction reproduced: position {position}, held out AUC {aucs[position]:.6f}")

    out = {"schema": SCHEMA, "predeclaration": PREDECLARATION,
           "predeclaration_sha256": PREDECLARATION_SHA256,
           "inputs": {"artefact": Path(args.artefact).name, "model": identity,
                      "partitions": digests, "seed": args.seed, "batch": args.batch},
           "exploratory": True, "arms": {}}

    log("[band] generation check on the fresh requests")
    ranking = list(range(nb))
    band_check = H.predictive_check(first, parts, list(BAND), direction, ranking, seed=args.seed)
    out["band"] = {"blocks": list(BAND), "check": band_check, "judgement": judge_band(band_check)}

    for arm_name, subject in subjects.items():
        log(f"[positions] {arm_name} arm")
        profiles, details = {}, {}
        for p in (REFERENCE_POSITION, *POSITIONS):
            d_p, held_out = A.fit_direction_at(res["fit"]["harmful"], res["fit"]["harmless"],
                                               res["search"]["harmful"], res["search"]["harmless"],
                                               p)
            prof = A.profile_with_intervals(subject, parts["measure"], d_p, nb, seed=args.seed)
            profiles[p] = prof["effects"]
            details[str(p)] = {"held_out_auc": held_out, **prof}
        stored = [b["effect"] for b in original["arms"][arm_name]["blocks"]]
        drift = max(abs(a - b) for a, b in zip(profiles[REFERENCE_POSITION], stored, strict=True))
        out["arms"][arm_name] = {
            "positions": details,
            "judgement": judge_positions({p: profiles[p] for p in POSITIONS},
                                         profiles[REFERENCE_POSITION]),
            "reference_reproduces_original_to": drift}
    verdicts = {n: a["judgement"]["verdict"] for n, a in out["arms"].items()}
    out["position_verdict_by_arm"] = verdicts
    out["position_arms_agree"] = len(set(verdicts.values())) == 1
    out["seconds"] = round(time.monotonic() - t0)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    R.write_json(Path(args.out) / "followups.json", out)
    log(f"band: {out['band']['judgement']['verdict']}  positions: {verdicts}")
    return out


def run_start(args, *, log=print):
    """Run the declared where-it-starts test on requests no earlier step used."""
    import json
    import time
    from pathlib import Path

    import torch

    from . import attribution as A
    from . import attribution_hf as H
    from . import attribution_run as R

    original = json.loads(Path(args.artefact).read_text(encoding="utf-8"))
    t0 = time.monotonic()
    model, tok, identity = H.load(args.model, device=args.device,
                                  revision=original["model"]["revision"],
                                  **_loading_options(args))
    H.refuse_other_loading_form(identity, original["model"])
    parts, digests = H.build_partitions(args.track, n_measure=args.n_measure,
                                        n_confirm=args.n_confirm, seed=args.seed)
    if (json.loads(json.dumps(digests)) != original["behaviour"]["partitions"]
            or identity["revision"] != original["model"]["revision"]):
        raise A.AttributionError("these are not the inputs the original run used, so a test on "
                                 "them would answer a different question.")
    used = set(parts["measure"]["harmful"].groups) | set(parts["confirm"]["harmful"].groups)
    fresh, fresh_digest = H.fresh_requests(args.track, used, n=args.n_fresh, seed=args.seed)

    subject = H.HFSubject(model, tok, batch=args.batch, log=log)
    nb = subject.n_blocks
    res = {k: {s: subject.residuals(list(parts[k][s].prompts)) for s in ("harmful", "harmless")}
           for k in ("fit", "search")}
    conv = A.DEFAULT
    late = tuple(range(LATE_FROM, nb))

    def flags(direction, blocks):
        got = subject.refusal_flags(list(fresh.prompts), direction, blocks)
        return torch.tensor(got, dtype=torch.float64)

    log("[start] baseline replies")
    base = flags(None, ())
    rows, details = {}, {}
    for p in START_POSITIONS:
        d_p, held_out = A.fit_direction_at(res["fit"]["harmful"], res["fit"]["harmless"],
                                           res["search"]["harmful"], res["search"]["harmless"], p)
        log(f"[start] position {p}")
        early, later = flags(d_p, EARLY), flags(d_p, late)
        drop_early, drop_late = base - early, base - later
        zeros = torch.zeros_like(base)[None]
        gen = torch.Generator().manual_seed(args.seed + p)
        pe, be = A.bootstrap_effects(drop_early[None], zeros, fresh.groups, fresh.groups,
                                     conv.resamples, gen)
        pc, bc = A.bootstrap_effects((drop_late - drop_early)[None], zeros, fresh.groups,
                                     fresh.groups, conv.resamples, gen)
        elo, ehi = A.interval(be, conv.alpha)
        clo, chi = A.interval(bc, conv.alpha)
        rows[p] = {"early_drop_ci": [float(elo[0]), float(ehi[0])],
                   "contrast_ci": [float(clo[0]), float(chi[0])]}
        details[str(p)] = {
            "held_out_auc": held_out, "early_drop": float(pe[0]),
            "late_drop_minus_early_drop": float(pc[0]), **rows[p],
            "harmful_refusal_rate": {"none": float(base.mean()), "early": float(early.mean()),
                                     "later": float(later.mean())}}
    out = {"schema": START_SCHEMA, "declaration": START_DECLARATION,
           "declaration_sha256": START_SHA256, "exploratory": True,
           "inputs": {"artefact": Path(args.artefact).name, "model": identity,
                      "fresh_requests": fresh_digest, "early_blocks": list(EARLY),
                      "later_blocks": [LATE_FROM, nb - 1], "seed": args.seed},
           "positions": details, "judgement": judge_start(rows),
           "seconds": round(time.monotonic() - t0)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    R.write_json(Path(args.out) / "start.json", out)
    log(f"where it starts: {out['judgement']['verdict']}")
    return out


def main(argv=None):
    import sys

    from . import attribution as A
    argv = list(sys.argv[1:] if argv is None else argv)
    runner = run
    if argv and argv[0] == "start":
        runner, argv = run_start, argv[1:]
    args = build_parser().parse_args(argv)
    try:
        runner(args)
    except A.AttributionError as e:
        print(f"follow-up stopped: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
