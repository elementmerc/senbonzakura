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


# ------------------------------------------------------------------------------ the follow-up run

SCHEMA = "senbonzakura-attribution-followups/1"


def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m senbonzakura.attribution_followups",
        description="Run the two declared follow-ups against a finished attribution run.")
    p.add_argument("model")
    p.add_argument("--track", required=True)
    p.add_argument("--artefact", required=True, help="the attribution.json the run produced")
    p.add_argument("--out", required=True, help="directory to write followups.json into")
    p.add_argument("--n-measure", type=int, default=400)
    p.add_argument("--n-confirm", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    return p


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
                                  revision=original["model"]["revision"])
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


def main(argv=None):
    import sys

    from . import attribution as A
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except A.AttributionError as e:
        print(f"follow-up stopped: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
