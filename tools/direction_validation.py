#!/usr/bin/env python3
r"""Do the extra refusal directions carry refusal, or do they carry topic?

Senbonzakura's clustered extractor finds several directions per layer. That it finds them is not
evidence that they are refusal directions: a cluster of harmful prompts about one subject differs
from harmless prompts partly BECAUSE of the subject, and the separation filter that was supposed
to tell those apart currently accepts every candidate it is shown.

The published work has the same hole. The clustering approach here was arrived at independently
in "Exploring the multi-dimensional refusal subspace" (LessWrong), which validates with refusal
rates and MMLU and states plainly that it has no random-direction baseline and no cross-topic
generalisation test. "On the Failure of Topic-Matched Contrast Baselines in Multi-Directional
Refusal Abliteration" (arXiv:2603.22061) argues that topic-matched contrasts cannot separate the
two, and recommends precisely the two controls missing above.

This runs three experiments:

  E1  leave-one-cluster-out. Fit directions with one cluster excluded, then measure whether they
      still separate that cluster's prompts from harmless ones. A topic direction cannot separate
      a topic it never saw; a refusal direction can. Compared against random directions as a
      floor and against the global difference-of-means as a ceiling. No generation, minutes.

  E2  random-direction control. Ablate K directions where the extras are random and orthogonal
      rather than fitted. If fitted extras do not beat random extras at matched K, the extra
      directions are not carrying signal and the gain is just "cut more".

  E3  K sweep from ONE fixed configuration. Bake the same window at K = 1..8 and measure refusal,
      the harmless arm, KL and coherence. Varying only K is the point: the withdrawn five-seed
      comparison failed because its two arms ran different searches.

E1 answers whether the directions mean anything. E2 says whether the extras beat noise. E3 says
whether more of them is better. Run in that order; E1 is cheap and can invalidate the other two.

Usage:
  python tools/direction_validation.py --model <hf-id> --track <dir> --experiment e1|e2|e3|all \\
      --out results.json [--device cuda] [--max-directions 8] [--seed 42]
"""
import argparse
import json
import sys

import torch

from senbonzakura import cli


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", default="track")
    ap.add_argument("--experiment", default="all", choices=["e1", "e2", "e3", "e4", "all"])
    ap.add_argument("--strengths", default="0.3,0.5,0.7,0.85,1.0",
                    help="ablation strengths for the E4 grid. The weakest must leave refusal "
                         "partly standing or the grid has no room for K to show an effect, "
                         "which is how the first run of this sweep measured nothing.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-directions", type=int, default=8)
    ap.add_argument("--direction-clusters", type=int, default=8)
    ap.add_argument("--dir-prompts", type=int, default=128)
    ap.add_argument("--eval-refusal", type=int, default=64)
    ap.add_argument("--eval-kl", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--directions-from", default=None,
                    help="load a [NL+1, K, H] direction set (as written by tools/rdo.py) instead "
                         "of extracting cluster directions. The rest of the harness is unchanged, "
                         "so two direction sources are scored with one measuring stick.")
    own = ap.parse_args(argv)

    args = cli.build_parser().parse_args([
        "--model", own.model, "--track", own.track, "--device", own.device,
        "--max-directions", str(own.max_directions),
        "--direction-clusters", str(own.direction_clusters),
        "--dir-prompts", str(own.dir_prompts),
        "--eval-refusal", str(own.eval_refusal), "--eval-kl", str(own.eval_kl),
        "--seed", str(own.seed)])
    return own, args


# ── E1: leave one cluster out ─────────────────────────────────────────────────────────
def experiment_1(a, log):
    """Fit without a cluster, then ask whether the fitted directions separate it anyway.

    The comparison that matters is fitted-extra against RANDOM at the same layer. A random unit
    direction orthogonal to the same basis is the honest floor: it shares every property of a
    fitted direction except having been fitted.
    """
    args = a.args
    bad = a.load(f"{args.track}/bad_ds", args.dir_prompts)
    good = a.load(args.good_ds or f"{args.track}/good_ds", args.dir_prompts)
    Rb, Rg = a.collect_resid(bad), a.collect_resid(good)
    NL = a.NL
    g = torch.Generator().manual_seed(args.seed)

    # Mid-stack layers only. The extraction window is where directions are actually used, and the
    # embedding layer carries no prompt information at the last token at all.
    layers = [int(NL * f) for f in (0.3, 0.45, 0.6, 0.75)]
    n_clusters = max(2, args.direction_clusters)
    folds = []

    for li in layers:
        labels = cli._kmeans_labels(Rb[li], n_clusters, args.seed + li)
        mg = Rg[li].mean(0)
        gd = mg / mg.norm().clamp_min(1e-8)

        for c in labels.unique():
            held = labels == c
            if int(held.sum()) < cli.MIN_CLUSTER_ROWS or int((~held).sum()) < cli.MIN_CLUSTER_ROWS:
                continue
            train, test = Rb[li][~held], Rb[li][held]

            # Everything below is fitted on `train` only. `test` is never seen.
            d0 = cli._orth_to(train.mean(0) - mg, [gd])
            d0 = d0 / d0.norm().clamp_min(1e-8)
            basis = [gd, d0]

            sub = cli._kmeans_labels(train, max(2, n_clusters - 1), args.seed + li + 1)
            extras = []
            for sc in sub.unique():
                rows = train[sub == sc]
                if rows.size(0) < cli.MIN_CLUSTER_ROWS:
                    continue
                v = cli._orth_to(rows.mean(0) - mg, basis)
                if v.norm() < 1e-6:
                    continue
                v = v / v.norm()
                extras.append((cli._axis_separation(rows, Rg[li], v), v))
            extras.sort(key=lambda e: -e[0])

            # The floor: a random unit direction orthogonalised against the same basis.
            rnd = []
            for _ in range(8):
                r = torch.randn(a.H, generator=g)
                r = cli._orth_to(r, basis)
                if r.norm() > 1e-6:
                    rnd.append(cli._axis_separation(test, Rg[li], r / r.norm()))

            folds.append({
                "layer": li,
                "held_out_cluster": int(c),
                "held_out_rows": int(held.sum()),
                # Ceiling: the global refusal direction, fitted without the held-out cluster.
                "d0_on_held_out": round(cli._axis_separation(test, Rg[li], d0), 4),
                # The question: does the best EXTRA direction, never fitted on this cluster,
                # separate it?
                "best_extra_on_held_out": round(
                    cli._axis_separation(test, Rg[li], extras[0][1]), 4) if extras else None,
                "extra_on_train": round(extras[0][0], 4) if extras else None,
                "random_mean": round(sum(rnd) / len(rnd), 4) if rnd else None,
                "random_max": round(max(rnd), 4) if rnd else None,
            })
            log(f"  layer {li} fold {int(c)}: d0={folds[-1]['d0_on_held_out']} "
                f"extra={folds[-1]['best_extra_on_held_out']} "
                f"random={folds[-1]['random_mean']}")

    usable = [f for f in folds if f["best_extra_on_held_out"] is not None
              and f["random_mean"] is not None]
    verdict = "no usable folds"
    summary = {}
    if usable:
        extra = [f["best_extra_on_held_out"] for f in usable]
        rand = [f["random_mean"] for f in usable]
        d0s = [f["d0_on_held_out"] for f in usable]
        wins = sum(1 for f in usable if f["best_extra_on_held_out"] > f["random_max"])
        summary = {
            "folds": len(usable),
            "mean_extra": round(sum(extra) / len(extra), 4),
            "mean_random": round(sum(rand) / len(rand), 4),
            "mean_d0": round(sum(d0s) / len(d0s), 4),
            "folds_where_extra_beats_random_max": wins,
        }
        # A fitted direction that cannot beat the best of eight random ones on unseen prompts is
        # not carrying anything transferable, whatever its score on the data it was fitted to.
        if wins > len(usable) * 0.75:
            verdict = ("the extra directions generalise to prompts they were never fitted on and "
                       "beat the random floor in most folds, which is what a refusal direction "
                       "should do and a topic direction should not")
        elif wins < len(usable) * 0.25:
            verdict = ("the extra directions do NOT generalise: on held-out clusters they score "
                       "no better than random directions. They are fitting the clusters they came "
                       "from, which is what a topic direction looks like")
        else:
            verdict = ("mixed: the extra directions beat random on some held-out clusters and not "
                       "others, so they carry something transferable but not reliably")

    return {"folds": folds, "summary": summary, "verdict": verdict}


# ── E2 / E3: bake and measure ─────────────────────────────────────────────────────────
def _measure(a, label, K, log, randomise_extras=False, seed=0, strength=1.0):
    """Bake one window at K directions and one strength, and report what it cost and bought."""
    NL = a.NL
    saved = None
    if randomise_extras and K > 1:
        # Replace directions 1..K-1 with random ones orthogonal to the primary and to each other.
        saved = a.dirs_multi.clone()
        g = torch.Generator().manual_seed(seed)
        for li in range(NL + 1):
            keep = a.dirs_multi[li, 0].float()
            if keep.norm() < 1e-6:
                continue
            basis = [keep / keep.norm()]
            for j in range(1, K):
                r = torch.randn(a.H, generator=g)
                r = cli._orth_to(r, basis)
                if r.norm() < 1e-6:
                    continue
                r = r / r.norm()
                a.dirs_multi[li, j] = r.to(a.dirs_multi.dtype)
                basis.append(r)

    a.bake(int(NL * 0.6), strength, 0.0, max(2, NL // 4), K=K)
    row = {
        "arm": label,
        "K": K,
        "strength": round(strength, 3),
        "random_extras": randomise_extras,
        "harmful_refusal": round(a.refusal_rate(a.bad_eval), 4),
        # The harmless arm is not optional: a refusal rate alone cannot tell "the abliteration
        # worked" from "the model is too damaged to refuse anything".
        "harmless_refusal": round(a.refusal_rate(a.kl_eval), 4),
        "kl": round(a.kl_vs_orig(a.kl_eval), 4),
    }
    a.restore_weights()
    if saved is not None:
        a.dirs_multi = saved
    log(f"  {label}: s={strength:.2f} harmful={row['harmful_refusal']} "
        f"harmless={row['harmless_refusal']} kl={row['kl']}")
    return row


def matched_refusal_table(rows, tolerance=0.05):
    """Compare K at MATCHED refusal removal, which is the only comparison that means anything.

    Reading KL off arms that removed different amounts of refusal compares nothing: more ablation
    always costs more KL, so whichever arm cut harder "loses" regardless of whether its directions
    were better chosen. The papers this replicates (Wollschlaeger, Piras) compare at matched
    attack success for the same reason.

    For each K, take the arm with the LOWEST KL among those reaching the target refusal removal.
    A K that never reaches the target has no entry rather than a flattering one.
    """
    out = {}
    fitted = [r for r in rows if not r["random_extras"]]
    if not fitted:
        return {"targets": out, "note": "no fitted arms"}

    # Prefer the true unablated anchor (strength 0) when it exists. Falling back to the
    # least-ablated ARM was a real distortion in the first run: its weakest arm had already
    # stripped most of the refusal, so "90% removed" meant 90% below an already-gutted 21% and
    # every percentage read stronger than it was. The K ranking was unaffected, since every arm
    # is measured against the same anchor, but the labels were not honest.
    anchors = [r for r in fitted if r["strength"] == 0.0]
    baseline = (anchors[0]["harmful_refusal"] if anchors
                else max(r["harmful_refusal"] for r in fitted))
    for target_frac in (0.5, 0.75, 0.9):
        target = baseline * (1.0 - target_frac)
        best = {}
        for r in fitted:
            if r["harmful_refusal"] <= target + tolerance * baseline:
                cur = best.get(r["K"])
                if cur is None or r["kl"] < cur["kl"]:
                    best[r["K"]] = r
        if best:
            out[f"{int(target_frac * 100)}%_removed"] = {
                str(K): {"kl": r["kl"], "strength": r["strength"],
                         "harmful_refusal": r["harmful_refusal"]}
                for K, r in sorted(best.items())}
    return {"targets": out, "baseline_refusal": baseline,
            "baseline_is_unablated": bool(anchors)}


def degenerate_reason(rows):
    """Say why a grid cannot be read, rather than letting a flat table look like a result.

    The first run of this sweep put every arm at exactly 0.0% refusal because the default window
    already removed everything at K=1. Eleven arms, one number, and the table looked like data.
    A sweep with no spread measures nothing about what it varied, and it has to say so in the
    output rather than in someone's head.
    """
    fitted = [r for r in rows if not r["random_extras"] and r["strength"] > 0.0]
    if len(fitted) < 2:
        return "fewer than two fitted arms"
    vals = {r["harmful_refusal"] for r in fitted}
    if len(vals) == 1:
        v = next(iter(vals))
        floor = "floor (every arm removed all refusal)" if v == 0.0 else \
                "ceiling (no arm removed any refusal)" if v >= 1.0 else \
                f"a single value ({v})"
        return (f"every fitted arm landed on {floor}, so this grid has no dynamic range and "
                f"says nothing about K. Widen the strength range until the weakest arm leaves "
                f"refusal partly standing.")
    return None


def experiment_4(a, log, strengths):
    """The K sweep with headroom: every K at every strength, then compared at matched refusal."""
    rows = []
    kmax = max(a.dirs_per_layer)
    ks = sorted({1, 2, 3, min(5, kmax), min(8, kmax)} & set(range(1, kmax + 1))) or [1]
    # The unablated anchor. Without it the "percent removed" labels are relative to whichever arm
    # happened to ablate least, which in the first run was an arm that had already removed most of
    # the refusal. One extra evaluation buys percentages that mean what they say.
    log("E4: measuring the unablated anchor first")
    rows.append(_measure(a, "unablated-K1-s0", 1, log, strength=0.0))
    log(f"E4: {len(ks)} direction counts x {len(strengths)} strengths")
    rows.extend(_measure(a, f"fitted-K{K}-s{s:g}", K, log, strength=s)
                for K in ks for s in strengths)
    # The random control at the same strengths, so "fitted beats random" is judged with headroom.
    rows.extend(_measure(a, f"random-K{K}-s{s:g}", K, log,
                         randomise_extras=True, seed=a.args.seed + K, strength=s)
                for K in [k for k in ks if k > 1][:2] for s in strengths)

    bad = degenerate_reason(rows)
    if bad:
        log(f"  DEGENERATE: {bad}")
    return {"rows": rows, "max_k_available": kmax, "strengths": list(strengths),
            "degenerate_reason": bad, "matched_refusal": matched_refusal_table(rows)}


def load_directions(a, path, log):
    """Install an externally optimised direction set in place of the extracted one.

    Validated rather than trusted: a set with the wrong shape, or one whose rows are not
    orthonormal, would be scored under a geometry the bake does not implement and the comparison
    would be measuring the mistake instead of the method.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    dirs = blob["dirs_multi"] if isinstance(blob, dict) else blob
    if dirs.shape[0] != a.NL + 1 or dirs.shape[2] != a.H:
        raise SystemExit(
            f"{path} holds directions of shape {tuple(dirs.shape)}, which does not fit this model "
            f"(expected [{a.NL + 1}, K, {a.H}]). A direction set from another model cannot be "
            f"scored here.")
    for li in range(dirs.shape[0]):
        M = dirs[li].float()
        live = M[M.norm(dim=-1) > 1e-6]
        if live.shape[0] > 1:
            gram = live @ live.T
            if not torch.allclose(gram, torch.eye(live.shape[0]), atol=1e-3):
                raise SystemExit(
                    f"{path}: the directions at layer {li} are not orthonormal. The bake computes "
                    f"R^T(RW), which is a projection only for an orthonormal basis; correlated "
                    f"rows over-subtract along what they share, which is extra ablation strength "
                    f"disguised as an extra direction.")
    a.dirs_multi = dirs.to(torch.bfloat16)
    a.dirs_per_layer = [int((a.dirs_multi[li].float().norm(dim=-1) > 1e-6).sum())
                        for li in range(a.NL + 1)]
    a.KMAX = max(1, int(dirs.shape[1]))
    log(f"loaded directions from {path}: K up to {max(a.dirs_per_layer)} per layer")


def experiments_2_and_3(a, log, which, directions_from=None):
    args = a.args
    TR = args.track
    if directions_from:
        load_directions(a, directions_from, log)
    else:
        a.extract_directions(f"{TR}/bad_ds", args.good_ds or f"{TR}/good_ds", args.hedge_ds,
                             args.clean_ds or args.good_ds or f"{TR}/good_ds")
    a.bad_eval = a.load(f"{TR}/bad_eval_ds", args.eval_refusal)
    _kl_all = a.load(args.good_ds or f"{TR}/good_ds", args.dir_prompts + args.eval_kl)
    a.kl_eval = _kl_all[args.dir_prompts:args.dir_prompts + args.eval_kl] or _kl_all[:args.eval_kl]
    a.orig_lp = a.first_token_logprobs(a.kl_eval)
    a.snapshot_weights()

    rows = []
    kmax = max(a.dirs_per_layer)
    if which in ("e3", "all"):
        log("E3: the same window baked at each K")
        rows.extend(_measure(a, f"fitted-K{K}", K, log)
                    for K in range(1, min(a.KMAX, max(1, kmax)) + 1))
    if which in ("e2", "all"):
        log("E2: the same K with random extras instead of fitted ones")
        rows.extend(_measure(a, f"random-extras-K{K}", K, log,
                             randomise_extras=True, seed=args.seed + K)
                    for K in (2, 3, 5) if kmax >= K)
    return {"rows": rows, "max_k_available": kmax}


def main(argv=None):
    own, args = build_args(argv)
    log = lambda m: print(m, flush=True)   # noqa: E731
    a = cli.Abliterator(args, log)

    record = {"model": args.model, "track": args.track, "seed": args.seed,
              "experiment": own.experiment, "directions_from": own.directions_from,
              "axis_separation_threshold": cli.MIN_AXIS_SEPARATION}

    if own.experiment in ("e1", "all"):
        log("E1: leave one cluster out")
        record["e1"] = experiment_1(a, log)
        log(f"\n  E1 verdict: {record['e1']['verdict']}.\n")

    if own.experiment in ("e2", "e3", "all"):
        record["e2_e3"] = experiments_2_and_3(a, log, own.experiment, own.directions_from)

    if own.experiment in ("e4", "all"):
        strengths = [float(x) for x in own.strengths.split(",") if x.strip()]
        if own.experiment == "e4":
            # e4 alone still needs the directions and the eval sets that "all" builds above.
            args_ = a.args
            TR = args_.track
            if own.directions_from:
                load_directions(a, own.directions_from, log)
            else:
                a.extract_directions(f"{TR}/bad_ds", args_.good_ds or f"{TR}/good_ds",
                                     args_.hedge_ds,
                                     args_.clean_ds or args_.good_ds or f"{TR}/good_ds")
            a.bad_eval = a.load(f"{TR}/bad_eval_ds", args_.eval_refusal)
            _kl = a.load(args_.good_ds or f"{TR}/good_ds", args_.dir_prompts + args_.eval_kl)
            a.kl_eval = _kl[args_.dir_prompts:args_.dir_prompts + args_.eval_kl] or _kl[:args_.eval_kl]
            a.orig_lp = a.first_token_logprobs(a.kl_eval)
            a.snapshot_weights()
        record["e4"] = experiment_4(a, log, strengths)
        bad = record["e4"]["degenerate_reason"]
        log(f"\n  E4: {'UNREADABLE — ' + bad if bad else 'grid has spread; see matched_refusal'}\n")

    with cli.atomic_write(own.out) as f:
        json.dump(record, f, indent=2)
    log(f"written to {own.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
