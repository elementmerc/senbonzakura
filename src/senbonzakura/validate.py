# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
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
import contextlib
import json
import math

import torch

from . import cli


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", default="track")
    ap.add_argument("--experiment", default="all",
                    choices=["e1", "e2", "e3", "e4", "transfer", "reach", "all"])
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

            # Are these clusters "kinds of refusal" or "amounts of refusal"?
            #
            # The published methods cluster different things. The LessWrong work clusters prompt
            # TEXT into semantic categories; Piras trains a SOM on harmful representations. This
            # clusters raw residuals, and the dominant variance in a harmful residual cloud at a
            # refusal-relevant layer may simply be how hard refusal is firing. If so, every
            # cluster's difference-of-means points along d0, orthogonalising leaves noise, and the
            # extras are topic by construction. Cosine against d0 measures exactly that: near 1
            # means the cluster differs from harmless in the same DIRECTION as everything else and
            # only in degree.
            cos_to_d0 = []
            for sc in sub.unique():
                rows_c = train[sub == sc]
                if rows_c.size(0) < cli.MIN_CLUSTER_ROWS:
                    continue
                # Orthogonalised against the harmless direction FIRST, exactly as the extractor
                # does before it builds a candidate. Without that step the harmless mean's own
                # offset dominates the cosine and a set of perfectly aligned clusters reads as
                # 0.81 rather than 1.0, which is the diagnostic measuring its own arithmetic.
                diff = cli._orth_to(rows_c.mean(0) - mg, [gd])
                n_diff = diff.norm()
                if n_diff > 1e-6:
                    cos_to_d0.append(round(float((diff / n_diff) @ d0), 4))

            folds.append({
                "cluster_cos_to_d0": cos_to_d0,
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

    # Across every fold: how aligned are the cluster directions with the global one?
    all_cos = [abs(c) for f in folds for c in f.get("cluster_cos_to_d0", [])]
    alignment = {}
    if all_cos:
        all_cos.sort()
        med = all_cos[len(all_cos) // 2]
        # What fraction of a cluster's own direction SURVIVES removing the global one. The cosine
        # alone does not separate the cases: clusters that differ only in degree score 0.999 and
        # clusters carrying a real second component score 0.93, which is a hair apart on a cosine
        # and 0.04 against 0.37 once expressed as the surviving part. That surviving part is also
        # the thing the extractor actually keeps, so it is the honest quantity to report.
        residual = (max(0.0, 1.0 - med ** 2)) ** 0.5
        alignment = {
            "clusters_measured": len(all_cos),
            "median_abs_cos_to_d0": round(med, 4),
            "max_abs_cos_to_d0": round(all_cos[-1], 4),
            "median_residual_fraction": round(residual, 4),
            "fraction_above_0_99": round(sum(1 for c in all_cos if c > 0.99) / len(all_cos), 3),
        }
        if residual < 0.1:
            alignment["reading"] = (
                "the clusters differ from harmless in the same DIRECTION as the global mean "
                "difference and only in degree, so they are amounts of refusal rather than kinds "
                "of it. Orthogonalising against d0 then leaves almost nothing, which is why the "
                "extra directions cannot generalise. Clustering prompt TEXT into semantic "
                "categories, as the published work does, is the change this points at.")
        else:
            alignment["reading"] = (
                "the clusters point in genuinely different directions from the global mean "
                "difference, not merely further along it, so their failure to generalise is NOT "
                "explained by alignment. Different is not the same as shared: a direction unique "
                "to one cluster is topic structure, and only the held-out folds above can tell "
                "the two apart. Read this beside them rather than on its own")

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

    return {"folds": folds, "summary": summary, "verdict": verdict,
            "cluster_alignment": alignment}


# ── the transfer gap: the operation optimised against, versus the one applied ─────────
@contextlib.contextmanager
def residual_ablation(a, K, P, wmax, wmin, D):
    """Ablate by forward hook on the residual stream, using the bake's own layer profile.

    RDO optimises a HOOK, which removes the direction from the accumulated residual stream. The
    evaluation BAKES, which removes it from the weights that write into that stream. Those are
    related and not identical: the hook also removes whatever earlier layers and the embedding
    put there, while the bake only stops new writes. A direction tuned against one may transfer
    poorly to the other, and RDO's directions remove far less refusal than a plain
    difference-of-means, so the gap is the leading suspect.

    Sharing `layer_weight` with the bake is what makes this a measurement of the OPERATION. Give
    the hook its own profile and the comparison measures the profile instead.
    """
    handles = []

    def make(idx):
        w = cli.layer_weight(idx, P, wmax, wmin, D)
        M = a.dirs_multi[idx + 1][:K].float()
        live = M[M.norm(dim=-1) > 1e-6]

        def hook(_module, _inp, out):
            if w <= 0.0 or live.shape[0] == 0:
                return out
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            R = live.to(hf.device)
            hf = hf - w * ((hf @ R.T) @ R)
            hf = hf.to(h.dtype)
            return (hf, *out[1:]) if isinstance(out, tuple) else hf
        return hook

    try:
        for i, layer in enumerate(a.layers):
            handles.append(layer.register_forward_hook(make(i)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def transfer_gap(a, log, K, strength=1.0):
    """Score the SAME directions under the hook and under the bake, and report the difference."""
    NL = a.NL
    P, D = int(NL * 0.6), max(2, NL // 4)

    with residual_ablation(a, K, P, strength, 0.0, D):
        hooked = {"harmful_refusal": round(a.refusal_rate(a.bad_eval), 4),
                  "harmless_refusal": round(a.refusal_rate(a.kl_eval), 4),
                  "kl": round(a.kl_vs_orig(a.kl_eval), 4)}

    a.bake(P, strength, 0.0, D, K=K)
    baked = {"harmful_refusal": round(a.refusal_rate(a.bad_eval), 4),
             "harmless_refusal": round(a.refusal_rate(a.kl_eval), 4),
             "kl": round(a.kl_vs_orig(a.kl_eval), 4)}
    a.restore_weights()

    d_ref = baked["harmful_refusal"] - hooked["harmful_refusal"]
    log(f"  K={K} s={strength}: hook harmful={hooked['harmful_refusal']} kl={hooked['kl']} | "
        f"bake harmful={baked['harmful_refusal']} kl={baked['kl']} | delta={d_ref:+.4f}")
    return {"K": K, "strength": strength, "hook": hooked, "bake": baked,
            "refusal_delta_bake_minus_hook": round(d_ref, 4)}


def projection_magnitude(a, log, K):
    """How much activation actually lies along each kept direction.

    arXiv:2603.22061 found topic-matched contrasts producing geometrically purer directions that
    do nothing, and named the mechanism: matching "reduces the magnitude of the extracted
    direction below the threshold at which weight-matrix projection perturbs the residual
    stream". Purity is not the same as effect. A direction can separate the classes perfectly and
    still lie along an axis the activations barely occupy, and removing it then changes nothing.

    Our extra directions are orthogonalised against d0, which removes the largest shared
    component BY CONSTRUCTION, so they are exactly the shape that paper warns about and we have
    never measured their magnitude. Reported as a fraction of the primary direction's magnitude,
    because the absolute scale of a residual stream is not comparable across models or layers.
    """
    args = a.args
    bad = a.load(f"{args.track}/bad_ds", args.dir_prompts)
    Rb = a.collect_resid(bad)
    NL = a.NL
    P, D = int(NL * 0.6), max(2, NL // 4)
    layers = [li for li in range(NL + 1) if cli.layer_weight(li - 1, P, 1.0, 0.0, D) > 0]

    per_k = {j: [] for j in range(K)}
    for li in layers:
        M = a.dirs_multi[li][:K].float()
        for j in range(min(K, M.shape[0])):
            v = M[j]
            if v.norm() < 1e-6:
                continue
            per_k[j].append(float((Rb[li] @ v).abs().mean()))

    primary = sum(per_k[0]) / len(per_k[0]) if per_k[0] else 0.0
    out = {}
    for j, vals in per_k.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        out[str(j)] = {"mean_abs_projection": round(mean, 4),
                       "fraction_of_primary": round(mean / primary, 4) if primary else None}
    weakest = min((v["fraction_of_primary"] for v in out.values()
                   if v["fraction_of_primary"] is not None), default=None)
    reading = ("no directions measured" if weakest is None else
               (f"the weakest kept direction carries {weakest:.1%} of the primary's activation "
                f"magnitude, so removing it perturbs the residual stream by that much less. A "
                f"direction this small cannot do much whatever its separation score says, which "
                f"is the failure arXiv:2603.22061 describes")
               if weakest < 0.25 else
               (f"every kept direction carries at least {weakest:.1%} of the primary's activation "
                f"magnitude, so they are large enough to act and their weakness is not explained "
                f"by magnitude"))
    log(f"  projection magnitudes (fraction of primary): "
        f"{ {k: v['fraction_of_primary'] for k, v in out.items()} }")
    return {"per_direction": out, "weakest_fraction_of_primary": weakest, "reading": reading}


#: Below this, a layer's residual component along the direction did not meaningfully move, and the
#: edit did not reach that layer's residual stream. Chosen as a floor rather than a target: the
#: gemma failure showed a whole architecture at essentially zero while a working one halved.
MIN_REACH = 0.10


def bake_reach(a, log, K=1, strength=1.0):
    """Does the weight edit actually reach the residual stream, AT EACH LAYER, by layer type?

    WHY THIS IS NOT ANSWERED BY `transfer_gap`

    That compares hook and bake as whole-model refusal rates, which detects a model where the edit
    lands nowhere. It cannot see a model where the edit lands on SOME layers and not others, and
    that is exactly the open question on a hybrid: LFM2 writes its residual stream through a short
    convolution on most of its layers, and the bake edits `conv.out_proj` because it is
    dimensionally identical to an attention `o_proj` and sits in the same position. That is correct
    linear algebra and, until this ran, an untested claim about behaviour. On LFM2.5-8B-A1B it
    governs 18 of 24 layers.

    HOW IT IS MEASURED

    Take the residual stream at every layer on the harmful prompts, project it onto the refusal
    direction, and record the magnitude. Bake. Take it again. A layer whose edit landed has a
    smaller component along the direction; a layer whose edit did nothing has the same component it
    started with. Grouping by layer type turns that into the question actually being asked.

    This is the gemma check localised. Gemma's whole-model disagreement was 0.578 against Qwen3's
    0.016, and the cause was a normalisation between the edited weight and the stream. Per layer,
    the same shape of failure is visible before it costs a run.
    """
    from . import cli as _cli

    args = a.args
    bad = a.load(f"{args.track}/bad_ds", args.dir_prompts)
    if not bad:
        raise ValueError("no harmful prompts, so there is nothing to project.")

    NL = a.NL
    P, D = int(NL * 0.6), max(2, NL // 4)

    def component(R):
        # Mean absolute projection of each layer's residual onto the kept directions.
        out = []
        for li in range(NL + 1):
            M = a.dirs_multi[li][:K].float()
            live = M[M.norm(dim=-1) > 1e-6]
            if live.shape[0] == 0:
                out.append(0.0)
                continue
            H = R[li].float()
            out.append(float((H @ live.T).abs().mean()))
        return out

    before = component(a.collect_resid(bad))
    a.bake(P, strength, 0.0, D, K=K)
    after = component(a.collect_resid(bad))
    a.restore_weights()

    # Which layers the profile actually edits. A layer the weight profile skipped is not evidence
    # about anything, and including it would dilute the very fraction being measured.
    edited = [li for li in range(NL) if _cli.layer_weight(li, P, strength, 0.0, D) > 0]

    rows, by_kind = [], {}
    for li in edited:
        b, af = before[li + 1], after[li + 1]
        drop = 0.0 if b <= 0 else (b - af) / b
        layer = a.layers[li]
        kind = ("conv" if _cli._conv_outproj(layer) is not None and not _cli._has_attention(layer)
                else "attention")
        rows.append({"layer": li, "kind": kind, "before": round(b, 5), "after": round(af, 5),
                     "reduction": round(drop, 4)})
        by_kind.setdefault(kind, []).append(drop)

    summary = {k: {"layers": len(v), "mean_reduction": round(sum(v) / len(v), 4),
                   "min_reduction": round(min(v), 4),
                   "below_floor": sum(1 for x in v if x < MIN_REACH)}
               for k, v in sorted(by_kind.items())}
    for kind, st in summary.items():
        log(f"  {kind}: {st['layers']} layer(s), mean reduction {st['mean_reduction']:.3f}, "
            f"worst {st['min_reduction']:.3f}"
            + (f", {st['below_floor']} BELOW THE FLOOR" if st["below_floor"] else ""))

    # A MEAN over layer types is not the check. Measured on LFM2.5-350M on 2026-08-18, conv layers
    # averaged 0.495 against attention's 0.759, which reads as healthy, while ONE conv layer had
    # moved by 0.076: below the floor, and invisible in its own average. A summary that hides a
    # per-item failure is the defect class this whole command exists to catch, so individual
    # layers are counted as well as averaged.
    failed = [k for k, st in summary.items()
              if st["mean_reduction"] < MIN_REACH or st["below_floor"]]
    stragglers = [r for r in rows if r["reduction"] < MIN_REACH]
    if failed:
        dead = [k for k in failed if summary[k]["mean_reduction"] < MIN_REACH]
        if dead:
            reading = (f"the edit does NOT reach the residual stream on {', '.join(dead)} layers "
                       f"(mean reduction below {MIN_REACH}). An edit that does not land is a "
                       f"partial abliteration that looks like a whole one, which is the gemma "
                       f"failure. Do not publish a number from this architecture until it is "
                       f"understood.")
        else:
            where = ", ".join(f"layer {r['layer']} ({r['kind']}, {r['reduction']:.3f})"
                              for r in stragglers[:5])
            reading = (f"the edit lands on average and NOT everywhere: "
                       f"{len(stragglers)} layer(s) moved less than {MIN_REACH} along the "
                       f"direction, at {where}. The averages look healthy, so this is only "
                       f"visible per layer. Worth understanding before a published run leans on "
                       f"those layers, and not on its own a reason to withhold a result.")
    elif len(summary) > 1:
        kinds = sorted(summary, key=lambda k: summary[k]["mean_reduction"])
        lo, hi = summary[kinds[0]]["mean_reduction"], summary[kinds[-1]]["mean_reduction"]
        ratio = (lo / hi) if hi > 0 else 0.0
        reading = (f"the edit reaches every layer type. {kinds[0]} layers reduce by {lo:.3f} and "
                   f"{kinds[-1]} by {hi:.3f}, a ratio of {ratio:.2f}"
                   + ("; comparable, so the bake treats both positions alike."
                      if ratio >= 0.5 else
                      "; the weaker type lands less than half as hard, which is worth "
                      "understanding before a published run leans on it."))
    else:
        only = next(iter(summary))
        reading = (f"the edit reaches the residual stream on all {only} layers "
                   f"(mean reduction {summary[only]['mean_reduction']:.3f}). This architecture has "
                   f"one residual-writing position, so there is no cross-type comparison to make.")
    log(f"  reading: {reading}")
    return {"per_layer": rows, "by_kind": summary, "min_reach": MIN_REACH,
            "layers_below_floor": [r["layer"] for r in stragglers],
            "architectures_failing": failed, "reading": reading}


def experiment_reach(a, log, ks, strengths):
    """Per-architecture bake validation: does the edit land, and does it land evenly?"""
    log("R: does the weight edit reach the residual stream, per layer type")
    K = min(ks) if ks else 1
    s = max(strengths) if strengths else 1.0
    return bake_reach(a, log, K=K, strength=s)


def experiment_transfer(a, log, ks, strengths):
    """The whole point: if these two disagree, RDO is optimising the wrong operation."""
    log("T: the same directions ablated by hook and by weight bake")
    rows = [transfer_gap(a, log, K, s) for K in ks for s in strengths]
    deltas = [abs(r["refusal_delta_bake_minus_hook"]) for r in rows]
    worst = max(deltas) if deltas else 0.0
    mean = sum(deltas) / len(deltas) if deltas else 0.0
    if worst < 0.05:
        reading = ("hook and bake agree closely, so a direction optimised against the hook is "
                   "optimised against what the bake does and the transfer is not the problem")
    else:
        reading = (f"hook and bake disagree by up to {worst:.3f} in refusal rate, so RDO is "
                   f"optimising an operation the evaluation does not perform. The fix is to "
                   f"optimise the bake directly rather than a residual-stream hook")
    return {"rows": rows, "mean_abs_delta": round(mean, 4),
            "max_abs_delta": round(worst, 4), "reading": reading}


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


def matched_refusal_table(rows, tolerance=0.05, n_eval=None):
    """Compare K at MATCHED refusal removal, which is the only comparison that means anything.

    Reading KL off arms that removed different amounts of refusal compares nothing: more ablation
    always costs more KL, so whichever arm cut harder "loses" regardless of whether its directions
    were better chosen. The papers this replicates (Wollschlaeger, Piras) compare at matched
    attack success for the same reason.

    For each K, take the arm with the LOWEST KL among those reaching the target refusal removal.
    A K that never reaches the target has no entry rather than a flattering one.
    """
    out, ranking = {}, {}
    fitted = [r for r in rows if not r["random_extras"]]
    if not fitted:
        # The same key set as the normal return, so a reader of one shape does not KeyError on
        # the other. An empty table is a legitimate outcome, not a different kind of object.
        return {"targets": out, "ranking": ranking, "note": "no fitted arms",
                "baseline_refusal": None, "baseline_is_unablated": False, "n_eval": n_eval,
                "degenerate_reason": "no fitted arms", "target_tolerance": tolerance}

    # A grid with no dynamic range says nothing about K, and `rank_band` cannot see that on its
    # own: when every arm sits at the same refusal the spread is zero, the matched-level test
    # passes trivially, and a confident winner is published for a grid that measured nothing.
    degenerate = degenerate_reason(rows)

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
        # The arm CLOSEST to the target, not the cheapest arm that cleared it. Those are
        # different selections and the difference decided a verdict: on one grid, K=1 reached
        # 14.8% refusal for KL 0.019 and K=2 reached 17.2% for KL 0.018, and picking the
        # cheapest-that-cleared crowned K=2 for doing less work. An arm that overshoots by 0.21
        # and one that overshoots by 0.23 are not at the same refusal level, so their KL is not
        # comparable, which is the whole point of matching. Ties on distance go to lower KL.
        #
        # Membership is TWO-SIDED. It used to be `refusal <= target + tolerance*baseline`, open
        # downward, so an arm that removed everything was a member of the 50% band: a grid whose
        # arms sat at 0.02 and 0.00 refusal published a verdict under the heading "50%_removed"
        # when both arms had removed essentially all of it. Open-ended membership also made the
        # spread unbounded, which is why the comparability test below could not describe its own
        # threshold honestly. An arm that blew past this band belongs to a deeper one.
        lo, hi = target - tolerance * baseline, target + tolerance * baseline
        best = {}
        for r in fitted:
            if lo <= r["harmful_refusal"] <= hi:
                key = (abs(r["harmful_refusal"] - target), r["kl"])
                cur = best.get(r["K"])
                if cur is None or key < (abs(cur["harmful_refusal"] - target), cur["kl"]):
                    best[r["K"]] = r
        if best:
            entries = {
                str(K): {"kl": r["kl"], "strength": r["strength"],
                         "harmful_refusal": r["harmful_refusal"],
                         # How far past the target this arm landed. Two arms with very different
                         # overshoots are not a matched comparison and their KL must not be read
                         # as one; the report says so rather than leaving it to be noticed.
                         "overshoot": round(target - r["harmful_refusal"], 4)}
                for K, r in sorted(best.items())}
            band = f"{int(target_frac * 100)}%_removed"
            out[band] = entries
            # Deliberately a SIBLING of `targets` rather than extra keys inside a band. Mixing
            # verdict keys in with the per-K arms means every consumer that iterates a band trips
            # over them, which is a worse defect than the one being fixed.
            ranking[band] = rank_band(entries, baseline, tolerance, n_eval=n_eval,
                                      degenerate=degenerate)
    return {"targets": out, "ranking": ranking, "baseline_refusal": baseline,
            "baseline_is_unablated": bool(anchors), "n_eval": n_eval,
            "degenerate_reason": degenerate,
            "target_tolerance": tolerance}


# Two arms whose KL differs by less than this fraction are a tie, not a winner. On 2026-08-04 a
# report crowned "MULTI WINS" from KL 0.0088 against 0.0089, a difference of about one percent
# that is well inside the run-to-run noise of a 64-prompt KL estimate.
TIE_FRACTION = 0.05


def refusal_match_tolerance(refusals, n_eval, baseline, tolerance):
    """How far apart two refusal rates may sit and still count as the same level.

    A fixed fraction of the baseline was the wrong quantity and was wrong in both directions at
    once. Refusal is a proportion measured on `n_eval` prompts, so the precision of a DIFFERENCE
    between two arms is set by the binomial standard error, not by the baseline. At p = 0.15 on
    64 prompts that standard error is about 0.065, so a threshold of 0.05 * baseline = 0.038
    refused a ranking on more than half of all pairs whose true refusal was IDENTICAL, while
    still admitting genuinely mismatched pairs at larger p.

    Two standard errors of the difference, which is what this returns when `n_eval` is known, is
    the quantity that actually answers "could these two arms be at the same level". Without
    `n_eval` there is nothing better than the old constant, so it falls back and says so by
    returning the same number.
    """
    if not n_eval or n_eval <= 0:
        return tolerance * baseline
    p = sum(refusals) / len(refusals)
    p = min(max(p, 1.0 / n_eval), 1.0 - 1.0 / n_eval)   # keep the variance from collapsing at 0/1
    return 2.0 * math.sqrt(2.0 * p * (1.0 - p) / n_eval)


def rank_band(entries, baseline, tolerance, n_eval=None, degenerate=None):
    """Decide whether a band's arms may be ranked at all, and by how much the winner won.

    Everything here exists because a report printed a confident verdict it had not earned:

    1. A grid with no dynamic range was ranked. When every arm sits at the same refusal the
       spread is zero, so the matched-level test passes TRIVIALLY and a winner is announced for a
       grid that measured nothing. `degenerate` carries that judgement in from the caller, since
       a band cannot see the grid it came from.
    2. Arms at different refusal levels were ranked. More ablation always costs more KL, so that
       compares strengths rather than directions.
    3. A one-percent KL difference was reported as a win.
    4. A zero or negative KL produced a division and a reason string asserting that -0.001 and
       0.5 are within five percent of each other.
    """
    arms = dict(entries)
    if not arms:
        return {"comparable": False, "spread": 0.0,
                "reason": "no arm reached this band"}
    if len(arms) < 2:
        return {"comparable": False, "spread": 0.0,
                "reason": "only one arm reached this band, so there is nothing to compare it "
                          "against"}
    if degenerate:
        return {"comparable": False, "spread": 0.0,
                "reason": f"the grid itself says nothing about K ({degenerate}), so no ranking "
                          f"drawn from it can mean anything"}

    kls = [e.get("kl") for e in arms.values()]
    if any(not isinstance(k, (int, float)) or isinstance(k, bool) for k in kls):
        return {"comparable": False, "spread": 0.0,
                "reason": "at least one arm has a non-numeric KL, so the arms cannot be ordered"}

    refusals = [e["harmful_refusal"] for e in arms.values()]
    spread = max(refusals) - min(refusals)
    allowed = refusal_match_tolerance(refusals, n_eval, baseline, tolerance)
    if spread > allowed:
        return {"comparable": False, "spread": round(spread, 4), "allowed": round(allowed, 4),
                "reason": (f"the arms span {spread:.4f} in refusal against a matching tolerance "
                           f"of {allowed:.4f}, so they are not at the same level and their KL "
                           f"cannot be ranked")}

    order = sorted(arms.items(), key=lambda kv: kv[1]["kl"])
    (best_k, best_e), (next_k, next_e) = order[0], order[1]
    if best_e["kl"] <= 0 or next_e["kl"] <= 0:
        return {"comparable": False, "spread": round(spread, 4), "allowed": round(allowed, 4),
                "reason": (f"K={best_k} reports a KL of {best_e['kl']}, which is not a positive "
                           f"divergence, so no ratio between arms is meaningful")}
    if (next_e["kl"] - best_e["kl"]) <= TIE_FRACTION * next_e["kl"]:
        return {"comparable": True, "spread": round(spread, 4), "allowed": round(allowed, 4),
                "cheapest": None,
                "reason": (f"K={best_k} and K={next_k} are within {TIE_FRACTION:.0%} on KL "
                           f"({best_e['kl']} against {next_e['kl']}), which is a tie rather than "
                           f"a win")}
    return {"comparable": True, "spread": round(spread, 4), "allowed": round(allowed, 4),
            "cheapest": int(best_k), "margin": round(next_e["kl"] / best_e["kl"], 3),
            "reason": (f"K={best_k} is cheapest at a matched refusal level, "
                       f"{next_e['kl'] / best_e['kl']:.2f}x below the next best")}


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


def _directions_meta(directions_from):
    """The sidecar `tools/rdo.py` writes beside a direction file, or None.

    `directions_from` is only a path, and two arms of the same ladder differ by what is INSIDE
    that file (the score form, the induction weight, the initialisation) rather than by its name.
    Carrying the sidecar into the result means one artefact answers "which directions were these",
    instead of the answer living in a second file that no report ever opened.
    """
    if not directions_from:
        return None
    try:
        with open(f"{directions_from}.json", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {"error": f"no readable sidecar beside {directions_from}"}
    # The optimisation history is long and is already in the sidecar; the configuration is what a
    # reader needs here.
    return {k: v for k, v in meta.items() if k != "history"} if isinstance(meta, dict) else None


def floor_direction_counts(ks):
    """Which direction counts get a random-direction control arm, bounded in cost.

    Always the largest. It used to be the first two above 1 and nothing else, which left the
    HIGHEST K unfloored: on 2026-08-04 the headline arm was K=4 while the only random controls
    were K=2 and K=3, so the one number a writeup would quote had to be compared against a floor
    at a different direction count. The largest K is exactly the arm a claim gets made about, so
    it is the one that can least afford to go without.
    """
    # sorted() on entry: `multi[:2] + multi[-1:]` assumed an ordering the caller was
    # never required to provide, and on an unsorted list the largest K went unfloored,
    # which is the exact defect this function exists to prevent.
    multi = sorted({k for k in ks if k > 1})
    return sorted(set(multi[:2] + multi[-1:]))


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
    # The largest K is always covered. It used to be the first two K above 1 and nothing else,
    # which left the HIGHEST K without a floor: on 2026-08-04 the headline arm was K=4 and the
    # only random arms were K=2 and K=3, so the one number the writeup would quote had to be
    # compared against a random control at a different direction count. The largest K is precisely
    # the arm a claim gets made about, so it is the one that can least afford to go unfloored.
    rows.extend(_measure(a, f"random-K{K}-s{s:g}", K, log,
                         randomise_extras=True, seed=a.args.seed + K, strength=s)
                for K in floor_direction_counts(ks) for s in strengths)

    bad = degenerate_reason(rows)
    if bad:
        log(f"  DEGENERATE: {bad}")
    return {"rows": rows, "max_k_available": kmax, "strengths": list(strengths),
            "degenerate_reason": bad,
            "matched_refusal": matched_refusal_table(rows, n_eval=a.args.eval_refusal)}


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


def prepare_for_bakes(a, log, directions_from=None):
    """Install the directions, build the eval sets, and snapshot the pristine weights.

    One copy, called by every experiment that bakes. It was two copies for an afternoon, one in
    each caller, which is how this project ended up with three drifted renderings of the same
    chat prompt and a compass reading its verdict off the wrong token for nine days. The KL set
    is the harmless slice AFTER the prompts the directions were fitted on, so coherence is
    measured on rows the directions never saw.
    """
    args = a.args
    TR = args.track
    if directions_from:
        load_directions(a, directions_from, log)
    else:
        a.extract_directions(f"{TR}/bad_ds", args.good_ds or f"{TR}/good_ds", args.hedge_ds,
                             args.clean_ds or args.good_ds or f"{TR}/good_ds")
    a.bad_eval = a.load(f"{TR}/bad_eval_ds", args.eval_refusal)
    _kl_all = a.load(args.good_ds or f"{TR}/good_ds", args.dir_prompts + args.eval_kl)
    # Falls back to the head of the harmless set when it is too small to spare a disjoint slice,
    # which is a real degradation and is why the fallback is visible rather than silent.
    disjoint = _kl_all[args.dir_prompts:args.dir_prompts + args.eval_kl]
    if not disjoint:
        log(f"  NOTE: the harmless set holds {len(_kl_all)} prompts, too few to spare a slice "
            f"disjoint from the {args.dir_prompts} the directions were fitted on. Coherence is "
            f"being measured on prompts the directions saw, which flatters it.")
    a.kl_eval = disjoint or _kl_all[:args.eval_kl]
    a.orig_lp = a.first_token_logprobs(a.kl_eval)
    a.snapshot_weights()


def experiments_2_and_3(a, log, which, directions_from=None):
    args = a.args
    prepare_for_bakes(a, log, directions_from)

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

    # Everything that changes what this record MEANS, so a later reader (or a resume guard) can
    # tell two runs apart. It used to be six fields, of which only `directions_from` differed
    # between the arms of a four-arm ladder, and that field is a path: swap the flags, keep the
    # filenames, and nothing in the artefact recorded the difference. The direction file's own
    # sidecar is folded in for the same reason, since "which directions" is the largest single
    # thing that varies between arms and it lived in a separate file nothing read.
    record = {"model": args.model, "track": args.track, "seed": args.seed,
              "experiment": own.experiment, "directions_from": own.directions_from,
              "axis_separation_threshold": cli.MIN_AXIS_SEPARATION,
              "max_directions": args.max_directions, "direction_clusters": args.direction_clusters,
              "dir_prompts": args.dir_prompts, "eval_refusal": args.eval_refusal,
              "eval_kl": args.eval_kl, "strengths": own.strengths,
              "code_version": cli.code_version(),
              "directions_meta": _directions_meta(own.directions_from)}

    if own.experiment in ("e1", "all"):
        log("E1: leave one cluster out")
        record["e1"] = experiment_1(a, log)
        log(f"\n  E1 verdict: {record['e1']['verdict']}.\n")

    if own.experiment in ("e2", "e3", "all"):
        record["e2_e3"] = experiments_2_and_3(a, log, own.experiment, own.directions_from)

    if own.experiment == "transfer":
        prepare_for_bakes(a, log, own.directions_from)
        kmax = max(a.dirs_per_layer)
        ks = sorted({1, 2, min(4, kmax)} & set(range(1, kmax + 1))) or [1]
        record["transfer"] = experiment_transfer(a, log, ks, [0.5, 1.0])
        log(f"\n  T: {record['transfer']['reading']}\n")
        record["projection_magnitude"] = projection_magnitude(a, log, max(ks))
        log(f"\n  M: {record['projection_magnitude']['reading']}\n")

    if own.experiment in ("reach", "all"):
        # Cheap next to the grids, and it answers a question that invalidates them if the answer
        # is wrong: an edit that never reaches the stream makes every number below it meaningless.
        if own.experiment == "reach":
            prepare_for_bakes(a, log, own.directions_from)
        strengths_r = [float(x) for x in own.strengths.split(",") if x.strip()]
        record["reach"] = experiment_reach(a, log, [1], strengths_r)
        if record["reach"]["architectures_failing"]:
            log(f"\n  R: {record['reach']['reading']}\n")

    if own.experiment in ("e4", "all"):
        strengths = [float(x) for x in own.strengths.split(",") if x.strip()]
        if own.experiment == "e4":
            # e4 alone still needs what the e2/e3 branch above would have built.
            prepare_for_bakes(a, log, own.directions_from)
        record["e4"] = experiment_4(a, log, strengths)
        bad = record["e4"]["degenerate_reason"]
        log(f"\n  E4: {'UNREADABLE — ' + bad if bad else 'grid has spread; see matched_refusal'}\n")

    with cli.atomic_write(own.out) as f:
        json.dump(record, f, indent=2)
    log(f"written to {own.out}")
    return 0
