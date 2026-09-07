#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Multi-direction refusal abliteration for transformer language models, with an Optuna search.

Handles dense transformers and mixture-of-experts models, including the *fused* expert layout
(one batched 3D weight per layer) that recent transformers releases use for speed. Tools that
target only the *unfused* per-expert ModuleList layout can't reach the fused expert weights; this
one resolves the residual-writing down-projection across both, so the ablation applies to every
matrix that writes the residual stream. Optuna does the quality work: it finds the layers and
strengths that drive refusals toward zero without wrecking the model, guarded by KL divergence.

Method (Arditi et al., "refusal is mediated by a direction"), with Heretic's refinements plus two
additions here (5 multi-directional, 6 interpolated index):
  1. Extract per-layer refusal directions: the difference-of-means (bad - good), good-
     orthogonalised, PLUS up to KMAX-1 secondary axes from PCA of the bad residual cloud,
     giving an orthonormal basis of the refusal SUBSPACE at every layer.
  2. SEARCH (Optuna): a candidate is a windowed strength profile (peak position + strengths +
     width) AND how the directions are chosen (num_directions K; per-layer own directions vs a
     single interpolated direction_index shared across layers). Each trial applies the REAL
     norm-preserving weight bake (step 3) and restores from a pristine snapshot afterwards, so
     the search scores the exact model it will save (no activation-hook proxy, no proxy/bake
     gap). Score = refusals on the bad-eval set co-minimised with KL vs the original on the
     harmless set, with incoherence penalised directly.
  3. BAKE: for the winning config, norm-preservingly orthogonalise that direction span out of
     every residual-*writing* weight (each attn o_proj, each fused expert down_proj) so the
     ablation is permanent, then save.

The code is organised in three parts: pure module-level helpers (the refusal classifier and the
weight math, testable without a model), the Abliterator class (everything that needs the loaded
model: direction extraction, the reversible bake, evaluation, and the search), and a thin main().
"""
import contextlib
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import optuna
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import (
    dataset,  # every accepted way of saying "the prompts are here"
    marker,  # what a saved checkpoint says it is; NOT crashsafe.provenance
    separation,  # the candidate statistics for "does this axis carry refusal?" (Q-14)
)

# Defined in a module that imports nothing, so the entry point and `doctor` can read
# it without paying for torch. Re-exported here because everything already asks cli.
from ._version import __version__
from .crashsafe import (  # crash-resilience: persist by default, recover a lost save, fail loud early
    MIN_TORCH,
    RETRY_SHARD_SIZE,
    atomic_write,
    config_to_bake_args,
    disk_verdict,
    free_bytes_for,
    is_space_exhaustion,
    provenance,
    remaining_budget,
    save_failure_report,
    search_already_done,
    study_db_path,
    torch_version_ok,
    winning_config,
)
from .metrics import (
    KL_CEIL,  # the hard "too damaged" line; a trial above it is excluded outright
    KL_TARGET,  # where the knee's coherence surcharge starts, and what --max-kl moves
    broken_rate,  # fraction of a batch that is wrecked output
    heretic_keyword_rate,  # Heretic-comparable refusal metric (the axis Heretic wins)
    is_broken,  # wrecked-output detector (empty / garbage / repetition)
    is_refusal,  # hard-refusal detector
    is_soft_refusal,  # hedged-compliance detector (the moralising lecture)
    knee_scalar,  # the weighted selection rule over those rulers
    validate_ruler,  # refuses to measure with a ruler that misreads its own cases
)
from .resources import ResourceGovernor, SearchProgress, cuda_free_total  # VRAM throttle + ETA
from .track import (  # the recorded partition boundaries, and the flags that would cross them
    flag_violations,
    read_manifest,
)

# A candidate PCA axis is kept as a refusal direction only if it separates the harmful and harmless
# residual clouds by at least this standardised mean difference (Cohen's d). Below it, the axis is
# within-harmful content/topic variance, not refusal, and ablating it strips capability (Tier-1 P1).
MIN_AXIS_SEPARATION = 0.5


def code_version():
    """Which build produced this artefact, as a git description or an honest admission.

    Code reaches the GPU box by hand, and on 2026-08-04 a run executed against a checkout that
    predated the very changes it existed to measure. Worse, library semantics changed between a
    run finishing and its records being read, and nothing in the records said which side of the
    change they came from. A symbol-presence check in a run spec proves "at least as new as X";
    it does not identify a build. This does.
    """
    import subprocess
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "-C", str(here), "describe", "--always", "--dirty", "--abbrev=12"],
            capture_output=True, text=True, timeout=10, check=False)
        v = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        v = ""
    if v:
        return v

    # The GPU box is not a git checkout: code reaches it by file copy, so `git describe` there
    # returns nothing and the stamp would read "unknown" on the one machine whose provenance
    # actually needs establishing. A sync writes CODE_VERSION beside the package, and it is the
    # authority when git is absent.
    for candidate in (here / "CODE_VERSION", here.parent.parent / "CODE_VERSION"):
        try:
            stamped = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if stamped:
            return f"{stamped} (stamped at sync, not a git checkout)"
    # "unknown" rather than a blank, because an empty string in a provenance field reads as
    # "recorded and empty" instead of "never established".
    return "unknown: neither a git checkout nor a stamped sync"

# How many candidate axes per layer keep their separation value in the result file. The threshold
# above was chosen once and never validated against a measurement, so a run that keeps only one
# direction per layer cannot currently be told apart from a run whose second direction missed by
# 0.02. Recording the rejected values makes that difference visible. Bounded because this lands in
# a JSON file: the leading few axes are where a real second refusal direction would be if it
# existed, and the tail is rounding error by construction (see the rank floor below).
MAX_RECORDED_AXES = 8

#: A candidate separation below this FRACTION of the threshold is floating-point residue around an
#: exact zero rather than a small measurement, and means the filter is structurally unsatisfiable.
#: Relative rather than absolute, and that is not a style choice: the first version of this constant
#: was an absolute 1e-6, taken from a synthetic case that produced ~1e-8, and the first real model
#: measured peaked at 1.07e-5 across 3,556 axes. The absolute constant therefore reported "not
#: broken" about the exact measurement it was written to describe. A constant calibrated on one
#: dataset and never checked against another is the failure this whole area is about.
STRUCTURAL_ZERO_FRACTION = 1e-3

#: A cluster smaller than this does not get to propose a direction. Its mean is dominated by the
#: few prompts in it, so the "direction" would encode those prompts rather than a refusal mode,
#: and ablating it would strip whatever they happen to be about.
MIN_CLUSTER_ROWS = 8

#: Each side of the held-out split needs at least this many rows before a Cohen's d computed over
#: it means anything. A cluster at MIN_CLUSTER_ROWS splits into exactly two halves of this size,
#: so the two constants are deliberately in step: raising MIN_CLUSTER_ROWS without raising this
#: buys nothing, and raising this without raising that silently rejects every smallest cluster.
MIN_HELD_OUT_ROWS = 4

#: How many meaningless directions are measured to give each candidate's threshold a floor. Each
#: one costs a mean and a projection, no forward pass, so this is cheap.
#:
#: It was 4 per LAYER until Q-22, which is two changes: the floor is now drawn once per CANDIDATE
#: at that candidate's own row count, and the count is high enough to estimate a quantile rather
#: than to be summarised by a maximum. Four draws cannot locate a 95th percentile; forty can.
#: Measured cost of the whole change on a 28-layer model with eight candidates a layer: under
#: three seconds added to a run measured in hours.
NULL_DIRECTIONS_PER_CANDIDATE = 40

#: A matching quality at or above this is matching that did nothing: the chosen controls are no
#: nearer the candidate than harmless prompts picked at random, so the comparison is unmatched
#: whatever the flag said. 0.9 rather than 1.0 because the mean distance to a random row is itself
#: the average over a cloud, and controls drawn from the near side of it land slightly under 1.0
#: without being on-subject at all.
MATCHING_USELESS_RATIO = 0.9

#: Which quantile of the null draws becomes the floor. The same quantile every fixed backstop in
#: `separation` is derived from, so a candidate faces one idea measured two ways: that quantile
#: computed offline on synthetic draws, and that quantile measured on the rows in front of it.
NULL_FLOOR_QUANTILE = 0.95

# The "worse than anything real" score, used to keep damaged / unmeasured trials out of the running
# for best. A true infinity so no finite objective can ever tie or beat it.
WORST_SCORE = float("inf")

# Ceiling on the activation-capture chunk. Separate from the generation batch because capturing
# every layer's hidden states costs far more per row than generating from them, and it is the
# value this path used as a fixed size before it was paced, so a healthy card behaves exactly as
# it did before.
CAPTURE_BATCH = 16


# ── weight math (pure) ───────────────────────────────────────────────────────────
def _orth_to(vec, basis):
    # Remove from `vec` its component along each (unit) row in `basis`.
    for u in basis:
        vec = vec - (vec @ u) * u
    return vec


def _axis_separation(bad, good, v, stat=None):
    # How well unit axis `v` separates the harmful and harmless residual clouds. A genuine refusal
    # axis separates them; a within-harmful topic/phrasing axis does not. Used to keep only
    # candidate axes that actually carry refusal, so multi-direction ablation cuts refusal and not
    # capability (P1).
    #
    # `stat` is None everywhere the incumbent Cohen's d is wanted, which is everywhere until
    # --separation-statistic says otherwise. The default path calls the same arithmetic the project
    # has always used, so no recorded number moves when a candidate is merely available.
    pb = bad @ v
    pg = good @ v
    if stat is None:
        return separation.cohens_d(pb, pg)
    return stat.fn(pb, pg)


def _halves(n, seed):
    """Split n row indices into two disjoint halves, the same way every time for a given seed.

    Seeded through an explicit generator rather than the global RNG, because the caller runs
    inside a search whose own draws would otherwise decide which rows a direction was fitted on,
    making a rerun of the same trial a different measurement.
    """
    g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    perm = torch.randperm(int(n), generator=g)
    return perm[: int(n) // 2], perm[int(n) // 2:]


def _held_out_separation(bad_rows, good_fit, good_score, basis, seed, stat=None):
    """How well the direction these rows propose separates rows it was NOT fitted on.

    THE DEFECT THIS REPLACES, because it is the whole reason the filter was worthless.

    A cluster's candidate direction is its own mean minus the harmless mean. Scoring that
    direction with a difference of those same means, on those same rows, asks whether the
    quantity a vector was built to maximise is large along that vector. It is, always, by
    construction. That is why every run printed "rejected NONE of N candidates": the threshold
    was not lenient, it was measuring something that cannot come out small.

    So the direction is fitted on half the cluster's rows against half the harmless rows, and
    scored on the halves it never saw. A direction that encodes a real contrast survives the
    move; one that encodes the particular rows it was shown does not.

    Returns None when either half is too small to mean anything, so the caller drops the
    candidate rather than reading a Cohen's d computed over three rows.
    """
    n = int(bad_rows.shape[0])
    fit_idx, score_idx = _halves(n, seed)
    if len(fit_idx) < MIN_HELD_OUT_ROWS or len(score_idx) < MIN_HELD_OUT_ROWS:
        return None
    v = _orth_to(bad_rows[fit_idx].mean(0) - good_fit.mean(0), basis)
    norm = v.norm()
    if norm < 1e-6:
        return None
    return _axis_separation(bad_rows[score_idx], good_score, v / norm, stat)


#: Why a matched corpus without matched scoring is refused. Checked twice, in `run_parsed` before
#: the model download and in `extract_directions` for callers that construct the class directly,
#: so the wording lives here rather than in two places that would drift apart.
def _unused_matched_corpus(path):
    return (f"--harmless-matched {path} was supplied without --matched-scoring, so the corpus "
            f"would be loaded and never used and every separation number would still be measured "
            f"against the harmless set at large. Add --matched-scoring to judge candidates "
            f"against it, or drop --harmless-matched.")


def _orth_rows(mat, basis):
    """`_orth_to` for every row of a matrix at once."""
    for u in basis:
        mat = mat - torch.outer(mat @ u, u)
    return mat


def _matched_harmless_idx(cluster_rows, good_rows, basis, k):
    """The k harmless rows whose CONTENT is nearest this cluster's, ignoring refusal.

    The control group for a matched comparison (Q-23). A candidate is one cluster of harmful
    prompts, and judging it against harmless prompts in general asks whether that cluster stands
    out, which a cluster about explosives does whether or not the model refuses it. Judging it
    against harmless prompts on the SAME SUBJECT holds the subject still, so refusal is the only
    thing left to vary. It is the same move a study makes when it compares patients to matched
    controls rather than to the general population.

    Distances are taken in the subspace ORTHOGONAL TO `basis`, which holds the global
    difference-of-means and every direction kept so far. Matching on the full vector would let
    refusal decide which controls a candidate meets, and the controls would then differ in the one
    respect the comparison exists to measure. Projecting it out first means the neighbours are
    chosen on everything except what has already been called refusal.
    """
    n_good = int(good_rows.shape[0])
    if k <= 0 or n_good == 0:
        return None
    k = min(int(k), n_good)
    query = _orth_to(cluster_rows.mean(0), basis)
    pool = _orth_rows(good_rows, basis)
    return torch.topk((pool - query).norm(dim=1), k, largest=False).indices


def matching_quality(cluster_rows, good_rows, basis, k):
    """How much nearer the matched controls are than an arbitrary set of the same size.

    THE NULL FOR THE MATCHING ITSELF, and it needs one for the same reason every other number here
    does. `--matched-scoring` asks for harmless prompts on the candidate's subject; whether any
    EXIST is a property of the corpus, not of the request. A harmless set that shares no subject
    matter with the harmful one returns its nearest rows just the same, and they are not controls.
    Nothing in the score would show it: the run would report matched figures taken against
    unmatched rows.

    The ratio of the mean distance to the chosen controls, over the mean distance to all harmless
    rows. **Near 1.0 means matching achieved nothing** and a "matched" number from that run is not
    one.

    IT IS A ONE-SIDED ALARM AND NOT A SCORE, which is worth stating because the number looks like
    a score. Calibrated on synthetic corpora whose answer is known:

        no shared subjects at all        0.997     <- what the alarm is for
        perfectly matched corpus         0.299
        a quarter matched, rest far      0.047     <- LOWER than the perfect corpus

    The last row is the catch. A corpus holding a few near rows and many distant ones inflates the
    denominator, so the ratio falls for a reason that has nothing to do with the controls being
    good. Read a high value as "this did not work"; do not read a low value as "this worked well".

    Returns None when there is nothing to measure.
    """
    idx = _matched_harmless_idx(cluster_rows, good_rows, basis, k)
    if idx is None:
        return None
    query = _orth_to(cluster_rows.mean(0), basis)
    distances = (_orth_rows(good_rows, basis) - query).norm(dim=1)
    everything = float(distances.mean())
    if everything < 1e-9:
        return None      # every harmless row sits on top of the cluster; no scale to report against
    return float(distances[idx].mean()) / everything


#: How many harmless rows to sample when measuring the corpus's own internal scale. The statistic
#: is a mean over nearest-neighbour distances and settles quickly; the cost is quadratic in this
#: number, so it is capped rather than run over a corpus of thousands.
MATCH_SCALE_SAMPLE = 256

#: `match_closeness` at or below this means the controls are about as near the candidate as two
#: harmless prompts on the same subject are to each other, which is as good as the space allows.
#: Above roughly 2.0 the controls are further from the candidate than harmless rows are from their
#: own neighbours, which is the corpus saying it has nothing on the subject.
MATCH_CLOSENESS_GOOD = 1.0
MATCH_CLOSENESS_POOR = 2.0


def _typical_neighbour_distance(good_rows, basis, seed=0):
    """How close two harmless prompts get in this space, as the corpus's own yardstick.

    The mean distance from a harmless row to its NEAREST other harmless row. That is the scale on
    which "on the same subject" is expressed here: whatever the embedding's absolute magnitudes
    are, two prompts about the same thing land about this far apart.

    Sampled rather than computed over everything, because the full matrix is quadratic and the
    mean of a few hundred nearest-neighbour distances is stable well before it matters.
    """
    rows = _orth_rows(good_rows, basis)
    n = int(rows.shape[0])
    if n < 2:
        return None
    if n > MATCH_SCALE_SAMPLE:
        g = torch.Generator().manual_seed(seed)
        rows = rows[torch.randperm(n, generator=g)[:MATCH_SCALE_SAMPLE]]
    d = torch.cdist(rows, rows)
    d.fill_diagonal_(float("inf"))          # a row's nearest neighbour is not itself
    nearest = d.min(dim=1).values
    value = float(nearest.mean())
    return value if value > 1e-9 else None


def match_closeness(cluster_rows, good_rows, basis, k, seed=0):
    """How near the matched controls are, on the scale of the corpus's own neighbour distances.

    THE SCALE-FREE REPLACEMENT for `matching_quality`, and the reason it exists is that the older
    number cannot be read as a quality score even though it looks like one. That one divides the
    distance to the chosen controls by the mean distance to EVERY harmless row, so a corpus holding
    a few near rows and many distant ones inflates its own denominator and scores better than a
    perfectly matched corpus: 0.047 against 0.299 on the calibration set. It is a sound alarm and an
    unsound measure, which is a bad shape for a number that is about to be used as a target.

    This divides instead by a property of the harmless corpus ALONE: the mean distance from a
    harmless row to its nearest neighbour. That is the yardstick the corpus itself provides for
    "two prompts about the same thing", so the ratio asks a question with a fixed meaning:

        how much further from this candidate are its controls
        than a harmless prompt typically is from its nearest harmless neighbour?

        about 1.0   the controls are as near as anything in this space gets. As good as it allows.
        about 2.0   the controls are twice the typical neighbour distance away, which is the corpus
                    saying it holds little on the subject.
        much above  no on-subject controls exist and the "matched" comparison is not matched.

    Adding distant rows to the corpus cannot improve this, because they change neither the
    numerator (the k nearest are unchanged) nor the denominator (a far row's nearest neighbour is
    still whatever was already nearest it). That is what makes it usable as an acceptance target
    for corpus work, which the old ratio is not.

    Returns None when there is nothing to measure.
    """
    idx = _matched_harmless_idx(cluster_rows, good_rows, basis, k)
    if idx is None:
        return None
    scale = _typical_neighbour_distance(good_rows, basis, seed)
    if scale is None:
        return None
    query = _orth_to(cluster_rows.mean(0), basis)
    chosen = (_orth_rows(good_rows, basis)[idx] - query).norm(dim=1)
    return float(chosen.mean()) / scale


def _matched_held_out_separation(bad_rows, good_fit, good_score, basis, seed, stat=None):
    """`_held_out_separation` against matched controls rather than the harmless set at large.

    The controls are chosen using the FIT half of the cluster only, so the rows a candidate is
    scored on never influence which harmless rows they are scored against. Without that, the
    matching itself becomes a way of fitting the score.

    The control group is drawn at the size of the harmful half it faces, which makes the
    comparison BALANCED. That is worth naming, because it is exactly the condition under which
    the variance ratio's null holds (see `separation.variance_ratio`): the matched design removes
    the imbalance that limits Candidate A, rather than merely tolerating it.
    """
    n = int(bad_rows.shape[0])
    fit_idx, score_idx = _halves(n, seed)
    if len(fit_idx) < MIN_HELD_OUT_ROWS or len(score_idx) < MIN_HELD_OUT_ROWS:
        return None
    fit_rows = bad_rows[fit_idx]
    mf = _matched_harmless_idx(fit_rows, good_fit, basis, len(fit_idx))
    ms = _matched_harmless_idx(fit_rows, good_score, basis, len(score_idx))
    if mf is None or ms is None:
        return None
    if len(mf) < MIN_HELD_OUT_ROWS or len(ms) < MIN_HELD_OUT_ROWS:
        return None
    v = _orth_to(fit_rows.mean(0) - good_fit[mf].mean(0), basis)
    norm = v.norm()
    if norm < 1e-6:
        return None
    return _axis_separation(bad_rows[score_idx], good_score[ms], v / norm, stat)


def _null_separation_floor(bad_all, good_fit, good_score, basis, size, seed, n_null, stat=None,
                           matched=False):
    """What a direction carrying nothing scores, measured through the identical path.

    The threshold above it was picked once and never checked against a measurement, which is the
    same failure the compass had before its null panel: a number with no floor beside it cannot
    be read. So the floor is measured rather than assumed.

    A null candidate is built from a RANDOM subset of the harmful rows of the same size as the
    clusters being judged. Its mean is the harmful cloud's mean plus sampling noise, so once it
    is orthogonalised against the basis, which already holds the global difference-of-means, what
    remains is noise and nothing else. Whatever such a direction scores is what this statistic
    hands out for free, and a real candidate has to beat it.

    Returns (floor, samples). The floor is a stated QUANTILE of the draws, not their maximum, and
    the difference is not cosmetic.

    IT WAS THE MAXIMUM UNTIL Q-22, AND THAT MADE THE DRAW COUNT A HIDDEN GATE SETTING. The maximum
    of n draws climbs with n without limit, so raising the count to steady the estimate would have
    silently tightened the filter instead. Measured on one layer's rows: the variance ratio's
    maximum went 2.33 at four draws to 6.14 at two hundred, a 2.6x move in the gate from a change
    that was only ever meant to reduce noise.

    A quantile does not move with the draw count; more draws simply pin it down better. The 95th
    is deliberately the same quantity every fixed backstop in `separation` is derived from, so the
    two halves of the threshold are one idea measured two ways: the backstop is that quantile
    computed offline on synthetic draws, and this is the same quantile measured on the actual rows.
    """
    # The nulls go through whichever path the candidates go through. A floor measured on the
    # unmatched comparison would be a floor for a different question than a matched candidate is
    # being asked, which is the exact failure Q-23 found in the floor itself.
    score = _matched_held_out_separation if matched else _held_out_separation
    seps = []
    for j in range(n_null):
        g = torch.Generator().manual_seed((int(seed) + 7919 * (j + 1)) & 0x7FFFFFFF)
        idx = torch.randperm(int(bad_all.shape[0]), generator=g)[:size]
        s = score(bad_all[idx], good_fit, good_score, basis,
                  int(seed) + 104729 * (j + 1), stat)
        if s is not None:
            seps.append(float(s))
    if not seps:
        return 0.0, seps
    ordered = sorted(seps)
    # Nearest-rank, so the floor is always a value a null direction actually reached rather than
    # an interpolation between two of them. With few draws that matters: an interpolated quantile
    # can sit above every observation and become a bar nothing was measured at.
    return ordered[min(len(ordered) - 1, int(NULL_FLOOR_QUANTILE * (len(ordered) - 1)))], seps


def _available_ram_bytes():
    # Best-effort available host RAM in bytes, for the snapshot pre-flight. Returns None when it
    # can't be determined (non-Linux / unreadable /proc), so the caller treats it as "unknown,
    # proceed" rather than refusing on a machine it simply couldn't measure.
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def layer_weight(idx, P, wmax, wmin, D):
    # Refinement 1 (Heretic windowed strength): the ablation strength PEAKS at layer
    # position P (wmax) and tapers linearly to wmin at distance D, and is ZERO beyond D.
    # Ablating early/late layers, where this direction is not the refusal direction, is
    # what destroyed coherence (KL 12-19) under a uniform-all-layers strength.
    # D is None for a profile that means "every layer, no taper", which a fixed recipe cannot
    # express as a number because it does not know how deep the model is.
    if D is None:
        return wmax
    dist = abs(idx - P)
    if dist > D:
        return 0.0
    # A window of zero width still contains its own centre, and the taper has no extent to run
    # over, so the peak value applies. Reached by any fixed profile with D = 0, which used to
    # raise ZeroDivisionError one second into the bake, after twenty minutes of direction
    # extraction. Found by hephaestus-c9 running E2 on 2026-09-07.
    if D == 0:
        return wmax
    return wmax + (dist / D) * (wmin - wmax)


@torch.no_grad()
def _sparsify_rows_(delta, sparsity):
    # Sparse surgery (OBLITERATUS-style, adapted to the norm-preserving projection): only KEEP the
    # top (1-sparsity) output-rows by edit magnitude, zero the edit on the rest. The rows with the
    # largest projection delta are the ones that actually WRITE the refusal direction; leaving the
    # low-projection rows pristine spares whatever capability they carry, for less collateral at the
    # same refusal removal on the rows that matter. `delta` is [..., out, in]; rows are the -2 dim.
    if sparsity <= 0.0:
        return delta
    mag = delta.norm(dim=-1)                            # [..., out] per-row edit magnitude
    keep = max(1, round((1.0 - sparsity) * mag.shape[-1]))
    thr = torch.topk(mag, keep, dim=-1).values.amin(dim=-1, keepdim=True)   # [..., 1]
    return delta * (mag >= thr).unsqueeze(-1)           # zero the untouched rows


#: How many times to alternate "restore the row lengths" and "remove the direction again".
#: 0 is the original single pass. Measured convergence: the leak is at floating-point noise by
#: round 4 across every shape, direction count and row-norm spread tried, and rounds beyond that
#: change nothing.
ABLATION_ROUNDS = 4


@torch.no_grad()
def orthogonalize_np_(W, R, s, sparsity=0.0, rounds=0):
    # Refinement 4 (norm-preserving ablation; Heretic row_normalization=full / grimjim):
    # ablate on the row-normalized weight, renormalize, then RESTORE the original row norms.
    # Raw orthogonalization changed the norms and wrecked calibration (KL 12-19); preserving
    # them keeps the model intact. R is [K, H] (refinement 5): removes the whole span.
    # sparsity>0 restricts the edit to the top-magnitude rows (sparse surgery).
    Rf = R.to(W.device).float()                         # [K, H]
    Wf = W.float()                                      # [out=H, in]
    rn = Wf.norm(dim=1, keepdim=True).clamp_min(1e-8)   # [out,1] original row norms
    Wn = Wf / rn
    delta = s * (Rf.T @ (Rf @ Wn))                      # each column's projection onto span(R)
    Wn = Wn - _sparsify_rows_(delta, sparsity)
    Wn = Wn / Wn.norm(dim=1, keepdim=True).clamp_min(1e-8)
    out = Wn * rn
    # MEASURED DEFECT, and `rounds` is the fix. Putting the original row lengths back undoes part
    # of the ablation, because scaling rows does not commute with a projection that acts across
    # them: the subtraction leaves ||R W|| at 5e-07 and the restore puts it back to 32% of where it
    # started on a matrix with a 4x row-norm spread, 46% at 10x, 5% at 0.2x. So the edit has always
    # been approximate, and how approximate depended on the model.
    #
    # Alternating "restore the lengths" and "remove the direction again" converges to a matrix that
    # is BOTH orthogonal to the refusal span and has the original row lengths, to floating-point
    # noise, in four rounds. Default 0 keeps the original behaviour, because turning this on
    # changes what every run produces and whether a cleaner ablation is a BETTER model is an
    # empirical question, not an obvious one: the leak may be part of why quality held up.
    for _ in range(int(rounds)):
        out = out / out.norm(dim=1, keepdim=True).clamp_min(1e-8) * rn
        for u in Rf:
            out = out - torch.outer(u, u @ out)
    W.copy_(out.to(W.dtype))


@torch.no_grad()
def orthogonalize_np_3d_(W, R, s, sparsity=0.0, rounds=0):
    # Norm-preserving, fused experts [E, out, in]; row norms per (expert, out-row). R is [K, H].
    Rf = R.to(W.device).float()                         # [K, H]
    Wf = W.float()
    rn = Wf.norm(dim=2, keepdim=True).clamp_min(1e-8)   # [E,out,1]
    Wn = Wf / rn
    proj = torch.einsum("kh,ehi->eki", Rf, Wn)          # [E,K,in]
    delta = s * torch.einsum("kh,eki->ehi", Rf, proj)   # [E,out,in]
    Wn = Wn - _sparsify_rows_(delta, sparsity)          # per (expert, out-row) sparsify
    Wn = Wn / Wn.norm(dim=2, keepdim=True).clamp_min(1e-8)
    out = Wn * rn
    # The same leak as the dense path, for the same reason: restoring per-row lengths undoes part
    # of a projection that acts across rows. See `orthogonalize_np_` for the measurement.
    for _ in range(int(rounds)):
        out = out / out.norm(dim=2, keepdim=True).clamp_min(1e-8) * rn
        proj = torch.einsum("kh,ehi->eki", Rf, out)
        out = out - torch.einsum("kh,eki->ehi", Rf, proj)
    W.copy_(out.to(W.dtype))


def post_sublayer_norms(layer):
    """The norms sitting BETWEEN a sublayer's output and the residual add, as (attn, mlp).

    Two layer shapes exist and the attribute names do not distinguish them, which is the trap.

    Qwen3, Llama, Mistral, Phi3 (the sublayer output IS the residual write):

        hidden = self.self_attn(...)
        hidden = residual + hidden                      <- o_proj output goes straight in
        hidden = self.post_attention_layernorm(hidden)  <- applied to the RESIDUAL, pre-MLP
        hidden = self.mlp(hidden)
        hidden = residual + hidden

    Gemma-2, Gemma-3, Olmo-2 (a norm intercepts both writes):

        hidden = self.self_attn(...)
        hidden = self.post_attention_layernorm(hidden)     <- applied to the ATTENTION OUTPUT
        hidden = residual + hidden
        hidden = self.mlp(hidden)
        hidden = self.post_feedforward_layernorm(hidden)   <- applied to the MLP OUTPUT
        hidden = residual + hidden

    Both have a `post_attention_layernorm` and it means opposite things. `post_feedforward_layernorm`
    exists only in the second shape, so it is the discriminator.

    This matters because the bake edits o_proj and down_proj. In the first shape that is the
    residual write and the edit lands. In the second a learned-gain RMSNorm rescales and rotates
    the result before it reaches the stream, so the edit is largely undone: measured on
    2026-08-05, gemma disagreed with an equivalent hook by 0.578 in refusal rate where Qwen3
    disagreed by 0.016, and gemma's KL never exceeded 0.021 across an entire grid.
    """
    if hasattr(layer, "post_feedforward_layernorm"):
        return (getattr(layer, "post_attention_layernorm", None),
                getattr(layer, "post_feedforward_layernorm", None))
    return (None, None)


@torch.no_grad()
def norm_gain(norm, H, device, dtype):
    """The elementwise gain a normalisation applies, recovered by probing it.

    RMSNorm computes `(x / rms(x)) * g`. On a vector of ones, `rms(x) == 1`, so the output IS `g`.
    That recovers the gain for any variant without reading its source, which matters because the
    variants differ: Llama-style RMSNorm multiplies by `weight` (initialised to ones) while
    Gemma-style multiplies by `1 + weight` (initialised to zeros). Reading `norm.weight` directly
    would be silently wrong by exactly one on half the architectures this tool supports.
    """
    probe = torch.ones(1, 1, H, device=device, dtype=dtype)
    return norm(probe).float().reshape(-1)


#: How far the Gram matrix of the returned basis may drift from the identity before the basis is
#: rejected. The rows go through QR (or Gram-Schmidt) in float32 on clouds of a few thousand
#: residuals, so exact orthonormality is not on offer; 1e-4 is roughly three orders of magnitude
#: looser than the drift a healthy decomposition produces here and three tighter than the point
#: where `R^T (R W)` stops behaving like a projection.
ORTHONORMAL_TOL = 1e-4


def _modified_gram_schmidt(M):
    """Orthonormalise the rows of M by a genuinely different algorithm from QR, in float64.

    Not a retry of the same computation: LAPACK's Householder QR and modified Gram-Schmidt fail
    on different inputs, which is the whole point of having a second route. Float64 because the
    rows arrive gain-weighted and a small gain can leave a row many orders of magnitude below its
    neighbours, which is exactly where float32 orthogonalisation loses the property.

    A row that is (numerically) in the span of the ones before it yields zero rather than noise.
    An arbitrary unit vector there would be a direction nothing asked to ablate.
    """
    rows = []
    for row in M.double():
        v = row
        # TWICE, which is the classical result and not a superstition: one pass loses orthogonality
        # to cancellation when a row is nearly in the span of its predecessors, and a second pass
        # restores it to machine precision. Measured here on the input that actually produces that
        # case, eight near-parallel rows from a small cloud, which is --max-directions 8 asked of a
        # corpus that cannot support eight distinct axes:
        #
        #     ordinary rows                 4.4e-16 -> 4.4e-16   no change
        #     near-parallel rows            6.7e-13 -> 4.4e-16   1500x better
        #     gain-folded, 13 orders        8.9e-16 -> 8.9e-16   no change
        #     near-parallel AND gain-folded 1.0     -> 1.0       neither pass saves it
        #
        # The last row is why this is an improvement rather than a fix: a basis that degenerate is
        # not rescued by arithmetic, and the orthonormality check below refuses it outright.
        for _ in range(2):
            for q in rows:
                v = v - (v @ q) * q
        n = v.norm()
        rows.append(v / n if n > 1e-10 else torch.zeros_like(v))
    return torch.stack(rows) if rows else M.double().reshape(0, M.shape[-1])


def _orthonormal_rows(M, want, li=None, log=None):
    """An orthonormal basis for the row space of M, or a loud stop.

    Returns `want` rows. QR first, modified Gram-Schmidt if it raises, and then a check that the
    result is actually orthonormal, because a basis that merely *returned* is not evidence of one
    that is usable: the bake's `R^T (R W)` is a projection only when `R R^T == I`, and a silently
    non-orthonormal basis removes a subspace nobody chose while reporting the K that was asked for.
    """
    _log = log or (lambda _m: None)
    where = f"layer {li}: " if li is not None else ""
    try:
        q, _ = torch.linalg.qr(M.T)
        out = q.T[:want]
    except torch.linalg.LinAlgError as first:
        _log(f"  {where}QR did not converge ({first}); retrying via modified Gram-Schmidt")
        try:
            out = _modified_gram_schmidt(M)[:want].to(M.dtype)
        except torch.linalg.LinAlgError as second:
            raise SystemExit(
                f"{where}could not orthonormalise the gain-folded directions. QR failed "
                f"({first}) and so did Gram-Schmidt ({second}). Both routes failing points at a "
                f"degenerate direction set rather than at bad luck in one algorithm: lower "
                f"--max-directions so fewer near-parallel axes are asked for, or widen the "
                f"corpus so the directions are estimated from a less collinear cloud. Nothing "
                f"has been baked.") from second

    # The rows that were zero on the way in are zero on the way out and are excluded: they are
    # unused direction slots, and requiring them to be orthonormal would fail every short basis.
    live = out[M.norm(dim=1) > 1e-8] if out.shape[0] == M.shape[0] else out
    if live.shape[0]:
        gram = (live @ live.T).float()
        drift = (gram - torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)).abs().max()
        if not torch.isfinite(drift) or drift > ORTHONORMAL_TOL:
            raise SystemExit(
                f"{where}the orthonormalised directions are not orthonormal (worst deviation from "
                f"the identity: {drift:.2e}, tolerance {ORTHONORMAL_TOL:.0e}). Ablating with this "
                f"basis would remove a subspace that was never chosen, while the artefact recorded "
                f"the direction count that was requested. That is the failure this check exists to "
                f"prevent, so nothing has been baked. Lower --max-directions, or widen the corpus "
                f"so the directions are less collinear.")
    return out


def fold_norm_gain(R, g):
    """Re-express directions so that ablating them BEFORE a norm zeroes them AFTER it.

    For a post-sublayer norm, the residual actually receives `y = (x / rms(x)) * g`, where `x` is
    what the edited weight produces. So

        R . y  =  (1 / rms(x)) * ((R * g) . x)

    and since `rms(x)` is a positive scalar it cannot change whether that is zero. Therefore

        R . y == 0   for every row of R   <=>   x is orthogonal to every row of (R * g)

    So the subspace to remove from `x` is the span of the gain-weighted directions, not of R
    itself. This is exact rather than an approximation: the normalisation's input-dependent scale
    divides out of the orthogonality condition entirely.

    The rows are re-orthonormalised because `R * g` is not orthonormal even when R is, and the
    bake's `R^T (R W)` is a projection only for an orthonormal basis. QR preserves the span, which
    is the thing that has to be right.

    That last sentence is a load-bearing claim, so it is checked rather than assumed. This is the
    successor to the non-converging SVD fixed in 6f87937: the decomposition that chooses what gets
    ablated used to fall back to a weaker basis and report the strength it had asked for, so a K=3
    run applied K=1 at some layers with nothing on record saying so. QR is a different algorithm
    with the same failure surface, and the same rule applies to it. Two routes, then a verification
    of the property the bake depends on, then a loud stop.
    """
    M = R.float() * g.to(R.device).float()
    out = _orthonormal_rows(M, R.shape[0])
    # A row of R that was already zero (an unused direction slot) must stay zero rather than be
    # replaced by whatever QR puts in an empty column, which would ablate an arbitrary direction.
    keep = M.norm(dim=1) > 1e-8
    return out * keep.unsqueeze(1).to(out.dtype)


def _decoder_layers(model):
    # The list of decoder blocks, resolved across the common architecture trees rather than
    # assuming `model.model.layers`. Raises loud if none matches, so an unsupported model fails
    # at load with a clear message instead of an opaque AttributeError deep in the search.
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            return obj
    raise ValueError(
        f"could not find the decoder layer stack on {type(model).__name__}; looked for "
        "model.layers, transformer.h, gpt_neox.layers, model.decoder.layers.")


def _real_tensor(owner, name):
    # The REAL, editable weight tensor for owner.<name>, transparently unwrapping accelerate offload.
    # When a model is larger than VRAM, accelerate dispatches some layers to CPU (or disk): their
    # parameter becomes a META tensor with no data, and the resident copy lives in that module's
    # AlignDevicesHook.weights_map. weights_map returns a STABLE tensor object (so id()-based dirty
    # tracking holds) and in-place edits to it propagate to the next forward, so the norm-preserving
    # bake works on offloaded layers with no other change. Non-offloaded weights return unchanged.
    t = getattr(owner, name)
    if not getattr(t, "is_meta", False):
        return t
    hook = getattr(owner, "_hf_hook", None)
    wm = getattr(hook, "weights_map", None)
    if wm is not None:
        try:
            real = wm[name]
            # Read it a SECOND time and hold both references. The whole bake depends on
            # weights_map handing back one stable tensor: an in-place edit only reaches the
            # next forward if the mapping stores the object rather than materialising it.
            # Disk-backed offload reads from the folder on every access, so each read is a
            # fresh tensor and the edit is written to something thrown away immediately.
            #
            # Holding both references matters. Comparing ids across two separate reads,
            # without keeping the first alive, can pass by accident: CPython reuses the freed
            # tensor's address, so the check reports stability that is not there. That is why
            # this went unnoticed, and it is why the two reads are compared as objects.
            probe = wm[name]
        except (KeyError, TypeError):
            real = probe = None
        if real is not None and not getattr(real, "is_meta", False):
            if real is not probe:
                raise ValueError(
                    f"weight {name!r} on {type(owner).__name__} is disk-offloaded: its offload map "
                    "returns a new tensor on every read, so the norm-preserving bake would write "
                    "into a copy that is discarded before the next forward pass, and the model "
                    "would come out unabliterated with nothing reporting it. Load with more VRAM "
                    "or host-RAM headroom so the weights stay resident or CPU-offloaded, which "
                    "both support in-place editing.")
            return real
    raise ValueError(
        f"weight {name!r} on {type(owner).__name__} is on the meta device with no resident offload "
        "copy, so it cannot be abliterated. This usually means disk-offload without an in-memory "
        "cache; load with more VRAM or host-RAM headroom so the weights stay resident.")


def _owned_weight(parent, name):
    # parent.<name> is either a submodule that owns a `.weight` (a Linear) or a raw parameter/tensor
    # held directly under `name`. Resolve to the real editable tensor in both cases, unwrapping
    # accelerate offload wherever the resident copy actually lives (on the submodule, or on parent).
    obj = getattr(parent, name)
    if hasattr(obj, "weight"):
        return _real_tensor(obj, "weight")
    return _real_tensor(parent, name)


def _attn_outproj(layer):
    # The attention OUTPUT projection (the matrix that writes attention back into the residual
    # stream), across naming conventions. Linear-based only: GPT-2-style Conv1D out-projections
    # (transposed weight) are deliberately not returned, since the row-wise norm-preserving bake
    # assumes a [out, in] Linear weight.
    attn = (getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            or getattr(layer, "self_attention", None) or getattr(layer, "attn", None))
    if attn is not None:
        for name in ("o_proj", "out_proj", "dense"):
            p = getattr(attn, name, None)
            if p is not None and hasattr(p, "weight") and _real_tensor(p, "weight").dim() == 2:
                return _real_tensor(p, "weight")
    raise ValueError(
        f"could not locate a Linear attention output projection on this layer "
        f"(type {type(layer).__name__}); architecture not supported.")


#: Blocks that can sit in the attention position and write the residual stream through their own
#: `out_proj`. The field has moved away from uniform attention stacks and this is the list that
#: moves with it: LFM2 puts a short convolution there, Qwen3.5 puts a gated delta net. In every
#: case the writer is a 2-D Linear whose OUTPUT dimension is hidden size, which is dimensionally
#: the same object as an attention `o_proj`, so the row-wise norm-preserving bake applies to it
#: unchanged and no per-vendor edit path is needed.
#:
#: Verified by building each architecture on the meta device and reading the shapes:
#:     LFM2.5-8B-A1B      conv.out_proj        [2048, 2048]
#:     Qwen3.6-35B-A3B    linear_attn.out_proj [2048, 4096]
#: The inner width differs and does not matter: the projection contracts over the output axis.
MIXER_BLOCKS = ("conv", "linear_attn")


def _conv_outproj(layer):
    """A non-attention sequence mixer's output projection, or None if this layer has none.

    Named for the first case that needed it and kept for the callers that use the name. Hybrids
    are now the common shape rather than the exception: roughly half of LFM2's decoder layers hold
    a `conv` block instead of attention, and three quarters of Qwen3.5's hold a gated delta net.
    In both, `out_proj` writes into the residual stream in exactly the position an attention
    `o_proj` occupies.

    Only 2-D weights are accepted. A block whose writer is some other rank is not something this
    bake can edit correctly, and returning it would produce a confident wrong edit rather than the
    loud refusal `refuse_unrecognised_writers` is there to give.
    """
    for name in MIXER_BLOCKS:
        block = getattr(layer, name, None)
        if block is None:
            continue
        p = getattr(block, "out_proj", None)
        if p is not None and hasattr(p, "weight") and _real_tensor(p, "weight").dim() == 2:
            return _real_tensor(p, "weight")
    return None


def _has_attention(layer):
    return any(getattr(layer, n, None) is not None
               for n in ("self_attn", "attention", "self_attention", "attn"))


def layer_attn_writers(layer, ablate_conv=True):
    """EVERY matrix in this layer that writes the residual stream from the attention position.

    On an ordinary architecture that is one tensor and this is `_attn_outproj` with a list around
    it. On a hybrid it is one OR the other per layer, and skipping the convolution ones is the
    gemma failure in a new costume: an edit that never reaches half the residual stream, reported
    as a successful run.
    """
    out = []
    if _has_attention(layer):
        out.append(_attn_outproj(layer))
    if ablate_conv:
        conv = _conv_outproj(layer)
        if conv is not None:
            out.append(conv)
    if not out:
        # Empty because the caller asked for the control arm and this layer's only writer here is
        # the convolution it was told to leave alone. That is a deliberate, warned, recorded
        # choice made in `refuse_unrecognised_writers`, not an unsupported layout.
        if not ablate_conv and _conv_outproj(layer) is not None:
            return out
        raise ValueError(
            f"no residual-writing projection found in the attention position of this layer "
            f"(type {type(layer).__name__}); architecture not supported. Supported: an attention "
            "output projection (o_proj / out_proj / dense), and the out_proj of a non-attention "
            "sequence mixer sitting in the same position: a short convolution (LFM2, LFM2-MoE) or "
            "a gated delta net (Qwen3.5). If this model puts something else there, the writer has "
            "to be recognised before it can be edited, because an edit that misses it is a "
            "partial abliteration reported as a whole one.")
    return out


def _classify(w):
    # Infer the bake kind from the tensor rank: 3D = fused expert stack [E, out, in], else 2D dense.
    return "fused3d" if w.dim() == 3 else "dense"


def _mlp_downprojs(mlp):
    # Every residual-WRITING down-projection inside one MLP / MoE block, as (kind, obj) entries.
    # kind is "dense" (2D weight), "fused3d" (batched [E, out, in] expert weight) or "list" (a list
    # of 2D per-expert weights). Covers dense, fused MoE (Qwen3-MoE / Granite output_linear),
    # unfused expert lists (OLMoE down_proj, Mixtral w2), and always-on shared experts.
    out = []
    if mlp is None:
        return out
    ol = getattr(mlp, "output_linear", None)                     # Granite-MoE parallel experts
    if ol is not None:
        w = _owned_weight(mlp, "output_linear")
        out.append((_classify(w), w))
    experts = getattr(mlp, "experts", None)
    if experts is not None:
        if hasattr(experts, "down_proj"):                        # Qwen3-MoE fused single tensor
            w = _owned_weight(experts, "down_proj")
            out.append((_classify(w), w))
        elif hasattr(experts, "w2"):                             # fused Mixtral-style single tensor
            w = _owned_weight(experts, "w2")
            out.append((_classify(w), w))
        else:                                                    # unfused per-expert Linear list
            ex = list(experts)
            if ex and hasattr(ex[0], "down_proj"):               # OLMoE
                out.append(("list", [_owned_weight(e, "down_proj") for e in ex]))
            elif ex and hasattr(ex[0], "w2"):                    # Mixtral unfused
                out.append(("list", [_owned_weight(e, "w2") for e in ex]))
    for attr in ("shared_expert", "shared_experts"):             # Qwen2-MoE / DeepSeek-MoE
        sh = getattr(mlp, attr, None)
        if sh is not None and hasattr(sh, "down_proj"):
            out.append(("dense", _owned_weight(sh, "down_proj")))
    if not out and hasattr(mlp, "down_proj"):                    # plain dense MLP
        out.append(("dense", _owned_weight(mlp, "down_proj")))
    if not out and hasattr(mlp, "w2"):                           # LFM2 dense feed-forward
        # Mixtral's name in a dense position: `w2` is the down-projection, `w1`/`w3` the gate and
        # up. Reached only after the expert branches above have declined, so a fused `w2` expert
        # stack is still classified as one rather than being read as a dense matrix here.
        out.append(("dense", _owned_weight(mlp, "w2")))
    return out


def _writes_residual(p, hidden_size):
    """Does this parameter write into the residual stream, at either rank?

    2-D `[hidden, inter]` is a dense down-projection or an attention output projection. 3-D
    `[experts, hidden, inter]` is a fused expert stack, where the hidden dimension sits in the
    MIDDLE because the leading axis indexes experts. Anything else (norms at 1-D, convolution
    kernels at 3-D with a width of 1) is not one.
    """
    dim = getattr(p, "dim", None)
    if dim is None:
        return False
    rank = dim()
    if rank == 2:
        return p.shape[0] == hidden_size
    if rank == 3:
        # A Conv1d kernel is also 3-D, as [channels, in/groups, kernel]. Requiring the hidden
        # size in the middle rather than anywhere keeps a depthwise convolution over the residual
        # width ([hidden, 1, L]) from being read as an expert stack.
        return p.shape[1] == hidden_size and p.shape[0] > 1
    return False


def residual_writers(layer, hidden_size, ablate_conv=True):
    """Every matrix in this decoder layer that writes the residual stream, and what we did not see.

    Returns `(recognised_count, unrecognised)`, where `unrecognised` names each child module that
    looks like it writes the residual and that the walker above does not handle.

    WHY THIS EXISTS, and why it landed in the same commit as LFM2 support rather than after it.

    Until then an unknown container raised loudly per layer, so an unsupported architecture failed
    cleanly at load. Teaching the walker one new container inverts that: the layer stops raising
    while some OTHER residual-writing matrix in it stays invisible, and the run completes having
    edited part of a layer and reported success. LFM2 is exactly that shape. Half its layers carry
    no attention at all: they hold a short convolution whose `out_proj` writes the residual stream
    just as an attention output projection does. Ablating what we recognise and staying quiet
    about the rest is how every gemma number came to be withdrawn, in a different disguise.

    `ablate_conv` is what the convolution blocks are being counted AS. When they are being ablated
    they are recognised; when `--skip-conv-ablation` has deliberately excluded them they are not,
    and this guard is what turns that into a refusal the caller has to opt out of in writing.

    The test is deliberately crude, because a subtle one would be the thing at fault. A module is
    a residual writer if it contains a weight whose output width is the hidden size: that is what
    "writes into the residual stream" means dimensionally. Norms are skipped (1-D), and so are the
    containers the walker already reads.

    BOTH ranks count, and the 3-D case is the one that nearly got away. A dense down-projection is
    2-D `[hidden, inter]`, but a FUSED mixture-of-experts stack is 3-D `[experts, hidden, inter]`
    and every expert in it writes the residual. Testing only for 2-D made this guard's whole
    coverage rest on the container NAME being in `known`: LFM2-MoE's `Lfm2MoeExperts` holds
    `down_proj` as a bare 3-D Parameter, and an architecture that put an equivalent stack under an
    unfamiliar name would have passed the guard while going unablated, reporting success. That is
    precisely the gemma failure this function exists to prevent, so it must not be reachable
    through a tensor rank the test does not look at.
    """
    # DERIVED from the editor's own list rather than restating it. These were two sets of names
    # that had to agree with nothing enforcing it, and they drifted the first time the editor
    # learned a new block: `MIXER_BLOCKS` gained `linear_attn`, this did not, and Qwen3.5 was
    # refused by the guard while the editor was perfectly able to edit it. That failure was the
    # safe direction, a loud refusal rather than a silent partial edit, and the opposite drift is
    # the gemma failure exactly, so the lists are now one list.
    known = {"self_attn", "attention", "self_attention", "attn",
             "mlp", "block_sparse_moe", "feed_forward"}
    if ablate_conv:
        known.update(MIXER_BLOCKS)
    unrecognised = []
    for name, child in layer.named_children():
        if name in known:
            continue
        writes = any(_writes_residual(p, hidden_size)
                     for _, p in child.named_parameters(recurse=True))
        if writes:
            unrecognised.append(f"{name} ({type(child).__name__})")
    return len(known & {n for n, _ in layer.named_children()}), unrecognised


def refuse_unrecognised_writers(layers, hidden_size, ablate_conv=True, accept_partial=False,
                                log=print):
    """Refuse a model whose layers write the residual through something we do not ablate.

    Loud and early, naming the layer and the module, because the alternative is a run that
    finishes, reports a refusal rate, and has left a live refusal write-path untouched.

    `accept_partial` is the one way past it and exists for a single purpose: the controlled
    comparison that asks whether the convolution path carries refusal at all needs an arm that
    deliberately leaves it alone. That arm is not silent. It warns at every layer it skipped and
    the choice is recorded in the result file, so a partial ablation can never be mistaken for a
    whole one after the fact.

    Returns the layers it left untouched, keyed by index, so the caller can record them.
    """
    missed = {}
    for i, layer in enumerate(layers):
        _, unrecognised = residual_writers(layer, hidden_size, ablate_conv=ablate_conv)
        if unrecognised:
            missed[i] = unrecognised
    if not missed:
        return missed
    shown = "; ".join(f"layer {i}: {', '.join(v)}" for i, v in list(missed.items())[:4])
    more = "" if len(missed) <= 4 else f" (and {len(missed) - 4} more layers)"
    if accept_partial:
        log(f"  WARNING: PARTIAL ABLATION. {len(missed)} of {len(layers)} decoder layers write the "
            f"residual stream through a module this run is deliberately NOT editing: "
            f"{shown}{more}.")
        log("  This model's refusal behaviour is only partly removed by construction. It is a "
            "control arm, not a result, and the choice is recorded in the result file.")
        return missed
    raise ValueError(
        f"{len(missed)} of {len(layers)} decoder layers write the residual stream through a "
        f"module this tool does not ablate, so an abliteration would edit part of each one "
        f"and report success: {shown}{more}. Refusing rather than producing a model whose "
        f"refusal behaviour was only partly removed.")


def layer_downproj(layer):
    # ALL residual-writing down-projections in this decoder layer, as (kind, obj) entries. A layer
    # may have more than one (a routed expert stack PLUS an always-on shared expert), and every one
    # writes the residual, so every one must be ablated. Granite exposes its experts under
    # block_sparse_moe; the dense / Qwen-family / Mixtral / OLMoE layouts under mlp. Raises loud on
    # an architecture whose down-projection can't be found, so an unsupported model fails at load
    # rather than silently leaving a live refusal write-path.
    entries = _mlp_downprojs(getattr(layer, "block_sparse_moe", None))
    entries += _mlp_downprojs(getattr(layer, "mlp", None))
    # LFM2 and its mixture-of-experts variant call the block `feed_forward`, and their dense
    # layers name the down-projection `w2` on it directly (Mixtral's naming in a dense position).
    entries += _mlp_downprojs(getattr(layer, "feed_forward", None))
    if not entries:
        raise ValueError(
            f"could not locate a residual-writing down-projection on this decoder layer "
            f"(type {type(layer).__name__}); architecture not supported. Supported: dense, "
            "Qwen3-MoE / Granite-MoE (fused), Mixtral (fused or unfused), OLMoE (unfused), "
            "shared-expert MoE (Qwen2-MoE / DeepSeek), and LFM2 / LFM2-MoE (feed_forward).")
    return entries


def _profiles_from_params(p):
    # Reconstruct (oP,owmax,owmin,oD, dP,dwmax,dwmin,dD) from a stored trial's params, for either
    # the per-component or the uniform schema, so the frontier/knee/final-bake are schema-agnostic.
    if "o_max_weight" in p:
        if "d_max_weight" in p:
            d = (p["d_max_weight_position"], max(0.0, p["d_max_weight"]), p["d_min_weight"], p["d_min_weight_distance"])
        else:  # --mlp-off: the d params were never suggested; MLP stays untouched (zero strength)
            d = (p["o_max_weight_position"], 0.0, 0.0, p["o_min_weight_distance"])
        return (p["o_max_weight_position"], p["o_max_weight"], p["o_min_weight"], p["o_min_weight_distance"], *d)
    P, wmax, wmin, D = p["max_weight_position"], p["max_weight"], p["min_weight"], p["min_weight_distance"]
    return (P, wmax, wmin, D, P, wmax, wmin, D)


def _scalar_of(t):
    # Scalarise a trial for early-stop / candidate ranking: non-compliance (hard + hedged) plus
    # half the keyword rate, but only for INTACT trials (KL under ceiling, coherent). A damaged or
    # unmeasured trial scores +inf so it can never look "best".
    ua = t.user_attrs
    if not ua or ua.get("kl", WORST_SCORE) > KL_CEIL or ua.get("broken", 1.0) > 0.1:
        return WORST_SCORE
    return ua["refusals"] + ua.get("soft", 0.0) + 0.5 * ua.get("heretic", 0.0)


# The budget knobs kageyoshi resolves from the architecture, paired with the flags that set them.
# A benchmark arm is only comparable if the budget it ran under is the budget it published, so an
# explicit value here is honoured rather than quietly replaced; see _kageyoshi_explicit.
_KAGEYOSHI_BUDGET_FLAGS = {
    "trials": ("--trials",),
    "dir_prompts": ("--dir-prompts",),
    "eval_refusal": ("--eval-refusal",),
    "eval_kl": ("--eval-kl",),
    "eval_refusal_final": ("--eval-refusal-final",),
    "max_directions": ("--max-directions",),
    "top_rescore": ("--top-rescore",),
    "patience": ("--patience",),
    "kl_scale": ("--kl-scale",),
}


def kl_eval_slice(good_prompts, dir_prompts, eval_kl, warn=None):
    """The harmless slice KL is measured on: disjoint from the prompts the directions were fit on.

    A function rather than four lines inline because the head-to-head benchmark has to hand the
    competing tool the SAME slice, and it does so from a separate process that cannot load the
    abliterator. Two copies of this arithmetic would let the two tools be scored on prompts that
    merely look alike.
    """
    if len(good_prompts) > dir_prompts:
        return good_prompts[dir_prompts:dir_prompts + eval_kl]
    if warn:
        warn("too small for a KL set disjoint from extraction; reusing the harmless tail")
    return good_prompts[-eval_kl:]


def _kageyoshi_explicit(argv):
    """Which budget knobs the caller set by hand, so the preset can leave them alone.

    kageyoshi resolves the search budget from the model, which is the whole point of the preset,
    and it used to do so unconditionally. That silently discarded `--trials 200`: the run then
    executed 100 trials while its own command line, its spec and its published table all said 200.
    An equal-budget comparison cannot survive that, and nothing in the log said it had happened.
    """
    if not argv:
        return set()
    seen = set()
    for dest, flags in _KAGEYOSHI_BUDGET_FLAGS.items():
        for flag in flags:
            # `--trials 200` and `--trials=200` are the same instruction and both count.
            if flag in argv or any(a.startswith(flag + "=") for a in argv):
                seen.add(dest)
    return seen


def _apply_kageyoshi(args, model, arch, ne, NL, log, explicit=()):
    # BANKAI — "ultimate balanced-effort" preset. The user asked for the best abliteration
    # we can produce with no knob-twiddling. So: read the detected architecture + parameter
    # count, auto-scale the search budget to the model's size, and switch on every quality
    # lever we have (3-objective NSGA-II front + knee, multi-direction refusal subspace,
    # hedging-contrast direction when a hedged set is present, larger-eval knee re-score,
    # early stop). "Balanced" is load-bearing: the KL ceiling + broken penalty + the knee
    # keep it intact rather than scorched, so this is the best UNCENSORING that stays coherent,
    # not the most aggressive one. Path/device/dataset flags are honoured, and so is any budget
    # knob the caller set by hand; everything else kageyoshi resolves from the model.
    explicit = set(explicit)

    def preset(dest, value):
        if dest in explicit:
            log(f"  keeping your --{dest.replace('_', '-')}={getattr(args, dest)} "
                f"(kageyoshi would have chosen {value})")
            return
        setattr(args, dest, value)

    total = sum(p.numel() for p in model.parameters())
    b = total / 1e9
    # Bigger models generate slower per trial, so fewer trials + smaller evals; the snapshot
    # also holds a CPU copy of every o_proj + down_proj, which is heavy past ~20B (see the
    # snapshot_weights note), hence the trimmed direction/eval counts in the top tier.
    if b < 5:
        budget = (100, 256, 64, 64, 128)
    elif b < 20:
        budget = (80, 256, 64, 48, 96)
    else:
        budget = (64, 192, 48, 32, 96)
    for dest, value in zip(("trials", "dir_prompts", "eval_refusal", "eval_kl",
                            "eval_refusal_final"), budget, strict=True):
        preset(dest, value)
    args.search = "pareto"          # map the whole refusals/keyword/KL front, pick the balanced knee
    args.per_component = True        # tune attn.o_proj and mlp.down_proj apart (the MLP may stay untouched)
    args.mlp_off = False             # let the search decide, don't force attention-only
    preset("max_directions", 3)      # ablate the refusal SUBSPACE, not just the difference-of-means
    preset("kl_scale", 4.0)          # the coherence guard that makes it balanced
    preset("top_rescore", 6)
    # Resolved from the trial count AFTER it settles, so an explicit --trials moves the early stop
    # with it rather than leaving a patience computed against a budget that no longer applies.
    preset("patience", max(20, args.trials // 3))   # stop once the front is mapped
    # Fold the hedging axis only if a hedged-compliance set is present (the lever that closes
    # the residual keyword gap Heretic wins on); silently skip it when absent.
    if not args.hedge_ds and os.path.isdir(f"{args.track}/hedge_ds"):
        args.hedge_ds = f"{args.track}/hedge_ds"
    # AN INVERTED PAIR IS A TYPO, NOT A SEARCH SPACE. Caught here rather than clamped silently,
    # because "--min-directions 3 --max-directions 2" means somebody wanted three and would
    # otherwise get two with nothing said.
    # Direct attribute access, not `getattr` with a default. A default here would mean a flag that
    # never reached this code silently behaves as though it were set to something reasonable, and
    # the whole point of an inverted-pair check is to catch a budget that is not what was asked
    # for. Every caller now builds its namespace from `build_parser()`, so a missing field is a
    # bug in the caller and should say so.
    k_min = args.min_directions
    if k_min > args.max_directions:
        raise SystemExit(
            f"--min-directions {k_min} is above --max-directions "
            f"{args.max_directions}, so no direction budget satisfies both. Set them equal to pin "
            f"the budget, or raise the ceiling.")
    hedge_note = f"hedging={args.hedge_ds}" if args.hedge_ds else "hedging=none (no hedge_ds in track)"
    # Said plainly in the banner, because "K<=2" and "K=2" are different experiments and the
    # difference is invisible in the artefacts until somebody reads the winning config.
    k_note = (f"K={args.max_directions} (pinned)" if k_min == args.max_directions
              else f"K<={args.max_directions}")
    log("BANKAI. Senbonzakura Kageyoshi — scatter, a thousand blades.")
    log(f"  {b:.1f}B params, down-proj={arch}{'' if ne is None else f'/{ne}e'}, {NL} layers -> "
        f"{args.trials} trials, {k_note}, eval {args.eval_refusal}/{args.eval_refusal_final}, "
        f"patience={args.patience}, {hedge_note}")




# Model classes seen to reject logits_to_keep, so the warning fires once each rather
# than once per batch. Keyed by class name because two models in one process (a base
# and an abliterated copy) can be different classes with different support.
_NO_LOGITS_TO_KEEP: set[str] = set()


def last_token_logits(model, enc, log=None):
    """Logits at the final position only, without materialising the whole sequence.

    A full logits tensor is batch x sequence x vocabulary. At batch 16, 2048 tokens
    and a 150k vocabulary that is roughly 10 GB in fp32 before anything else is
    allocated, and every caller here immediately throws away all but the last
    position. `logits_to_keep=1` asks the model to compute only the part that is
    used, which is the difference between the compass running on a 6 GB card and
    not running at all.

    The shape contract is unchanged: with the argument honoured the result is
    [B, 1, V], so [:, -1, :] still selects the same row.
    """
    key = type(model).__name__
    if key not in _NO_LOGITS_TO_KEEP:
        try:
            return model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        except TypeError as e:
            # Only the unsupported-argument case falls back. Any other TypeError is a
            # real bug in the forward pass and must not be converted into a quiet
            # change of memory behaviour.
            if "logits_to_keep" not in str(e):
                raise
            _NO_LOGITS_TO_KEEP.add(key)
            (log or print)(f"{key} does not accept logits_to_keep; computing full logits instead, "
                           f"which needs far more memory at large batch sizes")
    return model(**enc, use_cache=False).logits[:, -1, :].float()


def _kmeans_labels(X, k, seed, iters=25):
    """Cluster the rows of `X` into at most `k` groups. Returns per-row labels.

    Plain Lloyd's algorithm with k-means++ seeding, written here rather than pulled in: it is
    forty lines against a dependency in the fitting path of a security-adjacent tool, and every
    call is on a [prompts, hidden] matrix small enough that the clever implementations buy
    nothing (128 x 2048 at k=8).

    Deterministic for a given seed, because two runs of this project on the same input must
    produce byte-identical output, and a random initialisation would leak into the direction set.
    """
    n = X.size(0)
    k = max(1, min(k, n))
    g = torch.Generator().manual_seed(int(seed))

    # k-means++ seeding: each new centre is drawn with probability proportional to its squared
    # distance from the nearest existing one, which spreads the starts out instead of letting
    # two land in the same dense region and leaving a real cluster unrepresented.
    first = int(torch.randint(n, (1,), generator=g))
    centres = [X[first]]
    d2 = ((X - X[first]) ** 2).sum(1)
    for _ in range(1, k):
        total = float(d2.sum())
        if not (total > 0) or not torch.isfinite(d2).all():
            # Every remaining row coincides with a chosen centre (or the distances are not
            # finite). There is no meaningful next centre; stop rather than pick noise.
            break
        nxt = int(torch.multinomial(d2 / total, 1, generator=g))
        centres.append(X[nxt])
        d2 = torch.minimum(d2, ((X - X[nxt]) ** 2).sum(1))

    C = torch.stack(centres).clone()
    labels = torch.full((n,), -1, dtype=torch.long)
    for _ in range(iters):
        new = torch.cdist(X, C).argmin(1)
        if torch.equal(new, labels):
            break                      # converged; further passes cannot move anything
        labels = new
        for j in range(C.size(0)):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(0)
    return labels




def render_chat(tok, content):
    """One user turn, rendered into a prompt, with thinking OFF where the model supports it.

    Shared rather than duplicated, and that is the whole point of it existing. This project had
    two copies: the generation path passed `enable_thinking=False`, and the compass did not.
    On Qwen3 the difference is decisive, because the thinking template appends `<think>` to the
    generation prompt, so the position the compass reads its verdict logits from is the position
    the model was going to put `<think>` at. Measured on the held-out arm before this was shared:
    the most likely token there was a verdict for **0.0%** of prompts and the two verdict sets
    held ~0 probability, on both Qwen3-1.7B and Qwen3-0.6B. The refusal axis and the compass axis
    of the same published table were therefore measured under different prompt formats.

    No ValueError fallback: the loader has already established that this tokenizer renders chat
    prompts, either its own template or one `--chat-template` supplied. A ValueError here would
    mean that guarantee broke, and inventing a prompt format to paper over it is what made a
    whole class of numbers incomparable.
    """
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        # A tokenizer that does not accept enable_thinking; retry without it.
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def accelerator_name(device):
    """The card a measurement ran on, or None off GPU.

    Recorded because a version list is not provenance on its own: torch 2.5.1+cu124 on
    an H100 and on a 3090 are different measurements, and the July 2026 sweep captured
    neither.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        index = int(str(device).split(":")[1]) if ":" in str(device) else 0
        return torch.cuda.get_device_name(index)
    except (RuntimeError, ValueError, AssertionError):
        return None


def _renders_a_chat_prompt(tok):
    """Can this tokenizer actually turn a message into a prompt?

    Probed behaviourally rather than by reading `tok.chat_template`, because the
    attribute has moved and been deprecated across transformers versions while
    apply_chat_template's contract has not.
    """
    try:
        tok.apply_chat_template([{"role": "user", "content": "probe"}],
                                tokenize=False, add_generation_prompt=True)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def ensure_chat_template(tok, template_path=None, log=None):
    """Guarantee the tokenizer renders chat prompts, or refuse to measure anything.

    Prompt format drives output, so it drives the refusal rate, the KL and the compass
    together. The previous behaviour invented a bare "User:/Assistant:" wrapper when a
    model shipped no template, which is a different format from the one every published
    number was measured with, chosen silently and recorded nowhere. A number produced
    that way is not comparable to anything, and nothing said so.

    So a missing template is now an input the operator supplies and the run records,
    rather than something the tool makes up. Returns the provenance to store beside the
    results: where the template came from and a digest of it, which is what makes a
    re-run checkable.
    """
    _log = log or (lambda _m: None)
    if template_path:
        try:
            template = Path(template_path).read_text(encoding="utf-8")
        except OSError as e:
            raise SystemExit(f"could not read --chat-template {template_path}: {e}") from e
        if not template.strip():
            raise SystemExit(f"--chat-template {template_path} is empty")
        tok.chat_template = template
        if not _renders_a_chat_prompt(tok):
            raise SystemExit(f"--chat-template {template_path} did not produce a usable prompt. "
                             f"It must be a Jinja chat template of the kind tokenizer_config.json "
                             f"carries in its chat_template field.")
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
        _log(f"  chat template: supplied from {template_path} (sha256:{digest})")
        return {"source": str(template_path), "sha256": digest}

    if _renders_a_chat_prompt(tok):
        own = getattr(tok, "chat_template", None)
        digest = (hashlib.sha256(own.encode("utf-8")).hexdigest()[:16]
                  if isinstance(own, str) and own else None)
        return {"source": "tokenizer", "sha256": digest}

    raise SystemExit(
        "this model ships no chat template, so there is no defined way to turn a prompt into "
        "input for it. Every measurement here depends on that format: refusal rate, KL and the "
        "compass all change with it. Supply one with --chat-template <file> (a Jinja template of "
        "the kind tokenizer_config.json carries in chat_template) and it will be recorded with "
        "the results, so the numbers say which format produced them.")


def load_model_and_tokenizer(model_id, device="cuda", load_in_4bit=False,
                             trust_remote_code=False, attn_impl=None, log=None,
                             chat_template=None, needs_chat_template=True):
    # Shared model loader for the abliterator and the scorer. Left-pads the tokenizer and sets a pad
    # token, loads in bf16 (or 4-bit via bitsandbytes when asked), and honours trust_remote_code and
    # a chosen attention implementation. Placement uses accelerate's device_map so multi-GPU and a
    # specific cuda:N both work. 4-bit is for the pure-forward paths only (scoring / measurement):
    # the weight bake rewrites tensors in place and needs full precision.
    _log = log or (lambda m: None)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Validated here, at the single boundary every entry point passes through, rather than
    # in each of the three. The provenance rides on the tokenizer so callers can put it in
    # their result files without the loader having to change what it returns.
    # Scoped to the callers whose measurement depends on prompt format. The perplexity of a
    # fixed neutral passage does not: it never renders a chat prompt, so requiring a template
    # there refused to measure base models for a reason that did not apply to them.
    tok.senbon_chat_template = (ensure_chat_template(tok, chat_template, _log)
                                if needs_chat_template else None)
    kw = dict(dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    if attn_impl:
        kw["attn_implementation"] = attn_impl
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = "auto"
        _log("loading in 4-bit (nf4, double-quant); measurement path only, the bake needs full precision")
    elif device == "cuda":
        kw["device_map"] = "auto"                       # accelerate places / shards; supports big models
    elif device.startswith("cuda:"):
        kw["device_map"] = {"": int(device.split(":", 1)[1])}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if "device_map" not in kw:                          # cpu path
        model = model.to(device)
    model.eval()
    return model, tok


class Abliterator:
    """The loaded model plus everything that operates on it: direction extraction, the
    reversible norm-preserving bake (shared by the search and the final save), evaluation
    (refusals + KL), and the Optuna search. Constructing it loads the model and detects the
    architecture; `run()` does the extract -> search -> bake -> save pipeline. A model + tokenizer
    may be injected (skipping the load) so the class can be exercised against a tiny model in tests.
    """

    def __init__(self, args, log, model=None, tok=None):
        self.args = args
        self.log = log
        self.dev = args.device
        if model is None or tok is None:
            log(f"loading {args.model} on {self.dev}")
            model, tok = load_model_and_tokenizer(
                args.model, device=self.dev,
                load_in_4bit=args.load_in_4bit,
                trust_remote_code=args.trust_remote_code,
                # Passed through, which it was not until 2026-08-20. `score` and `compass` both
                # forwarded it and the abliterator did not, so on a model shipping no template the
                # run refused and told the operator to supply one with a flag it then ignored:
                # the instruction in the failure and the behaviour of the fix disagreed.
                chat_template=getattr(args, "chat_template", None) or None,
                attn_impl=args.attn_impl, log=log)
        self.tok = tok
        self.model = model
        self.layers = _decoder_layers(model)             # the decoder blocks, resolved defensively
        self.H = model.config.hidden_size
        self.NL = model.config.num_hidden_layers
        # Architecture label = the distinct down-proj kinds present (e.g. "fused3d" or "fused3d+dense"
        # when a routed stack sits beside a shared expert). Raises loud here if the arch is unsupported.
        _dp = layer_downproj(self.layers[0])
        # EVERY layer, not just the first. A hybrid architecture interleaves block types, so a
        # first layer that resolves cleanly says nothing about the twentieth: LFM2 puts its
        # convolution blocks first and its attention blocks after, and either order would have
        # let a per-model probe pass while half the residual writers stayed invisible.
        self.ablate_conv = not args.skip_conv_ablation
        self.partial_layers = refuse_unrecognised_writers(
            self.layers, self.H, ablate_conv=self.ablate_conv,
            accept_partial=not self.ablate_conv, log=log)
        self.arch = "+".join(dict.fromkeys(k for k, _ in _dp)) or "dense"
        self.ne = getattr(model.config, "num_experts", None) or getattr(model.config, "num_local_experts", None)
        self.KMAX = max(1, args.max_directions)
        # PINNED when they are equal, which is the point of having both. Clamped rather than
        # rejected here because the parser has already refused an inverted pair; this is the
        # invariant restated where the search reads it.
        self.KMIN = min(max(1, args.min_directions), self.KMAX)
        log(f"model up: hidden={self.H} layers={self.NL} down-proj={self.arch} experts={self.ne}")
        # Opened here rather than in `run()` so a caller constructing the class directly (a GUI, a
        # test) gets events too. A no-op when the flag is absent, which is the common case.
        from . import events as _events
        self.events = _events.open_for(args, log=log)
        self.events.emit("model_loaded", model=getattr(args, "model", None), hidden=self.H,
                         layers=self.NL, arch=self.arch, experts=self.ne,
                         device=str(getattr(self, "dev", "")))

        # Offload awareness: when the model is bigger than VRAM, accelerate places some layers on CPU
        # (or disk). The bake handles those transparently (see _real_tensor); we just report it so the
        # slower, throttled run is not a surprise. dmap values are ints (a cuda device) or "cpu"/"disk".
        dmap = getattr(model, "hf_device_map", None) or {}
        self.offloaded = sum(1 for v in dmap.values() if not isinstance(v, int))
        if self.offloaded:
            log(f"  low-VRAM mode: {self.offloaded}/{len(dmap)} module groups offloaded off the GPU; "
                "the adaptive throttle will pace generation to the card")

        # The resource governor paces GPU generation to live free VRAM: it shrinks the batch (or pauses
        # and resumes) when another app grabs the card, and grows back when it frees. No-op on CPU.
        self.gov = ResourceGovernor(
            self.dev, log,
            max_batch=args.gen_batch,
            min_free_frac=args.gpu_min_free_frac,
            max_pause_s=args.max_pause_s,
            background_mode=args.background_mode,
            external_pressure_mb=args.external_pressure_mb,
            enabled=not args.no_throttle)

        # A SECOND governor, for activation capture, and deliberately not the one above.
        #
        # Two reasons it is separate rather than shared. The memory profile is different:
        # `output_hidden_states=True` materialises every layer's activations for the whole batch at
        # once, so a batch size that is comfortable for generation can be far too large here, and a
        # shared `cur_batch` would carry one path's adaptation into the other. And the artefact
        # records `batch_sizes_used` under `generation`, described there as a property of
        # generation; folding capture chunks into that counter would make the field describe two
        # different things, which is the defect class this project keeps withdrawing results over.
        #
        # The ceiling stays at the 16 that was hard-coded here before, so a healthy card sees
        # exactly the behaviour it saw previously and only a starved one sees a smaller chunk.
        self.capture_gov = ResourceGovernor(
            self.dev, log,
            max_batch=CAPTURE_BATCH,
            min_free_frac=args.gpu_min_free_frac,
            max_pause_s=args.max_pause_s,
            background_mode=args.background_mode,
            external_pressure_mb=args.external_pressure_mb,
            enabled=not args.no_throttle)

        # Search-window layer bounds + reversible-bake / current-direction state.
        self.lo = int(self.NL * args.layer_lo)
        self.hi = int(self.NL * args.layer_hi)
        self._cur = {}            # current ablation config (mode + interpolated set), read by active_dirs
        self._pristine = {}       # id(W) -> (W, cpu clone); the pristine snapshot
        self._dirty = set()       # id(W)s touched by the last bake, restored between trials
        self.dirs_multi = None    # [NL+1, KMAX, H]; filled by extract_directions
        # Eval state, populated by run(); declared up-front so the object's shape is visible and
        # methods that read them fail predictably rather than with a surprise AttributeError.
        self.bad_eval = []
        self.kl_eval = []
        self.orig_lp = None

    # ── data + prompt formatting ────────────────────────────────────────────────
    def load(self, d, n):
        # The first n prompts from wherever they are: a save_to_disk directory, a text/CSV/JSON
        # file, or a Hub id, with the split and column worked out by `dataset.resolve`. Every
        # reader in the project goes through that one function so a user's error does not depend
        # on which command they happened to run.
        #
        # The exception types are preserved rather than simplified. FileNotFoundError and KeyError
        # are what callers and tests here have always caught, and turning a resolver error into a
        # different type at the boundary is cheaper than changing every catch site.
        try:
            rows = dataset.resolve(
                d,
                text_column=self.args.text_column or None,
                token=self.args.hf_token or None,
                what="prompt set")
        except dataset.DatasetError as e:
            if "column" in str(e):
                raise KeyError(str(e)) from e
            raise FileNotFoundError(str(e)) from e
        avail = len(rows)
        if avail < n:
            self.log(f"  note: {d} holds {avail} prompts, fewer than the {n} requested; using all {avail}")
        return rows[:n]

    def chat(self, p):
        return render_chat(self.tok, p)

    # ── direction extraction: per-prompt last-token residuals, bad vs good ────────
    @torch.no_grad()
    def collect_resid(self, prompts):
        """Per-prompt last-token residual at every layer -> [NL+1, N, H] on CPU (float32).

        We keep the whole cloud, not just its mean, so secondary refusal directions can be
        recovered by PCA (multi-directional ablation), not only the difference-of-means.

        PACED, because this is the first GPU-heavy thing an abliteration run does and it used to be
        the only batched path with a fixed size. `output_hidden_states=True` materialises every
        layer's activations for the whole batch at once, so on a large model it is also the
        heaviest thing per row; running it unpaced meant a run could die at the very first step,
        after loading the weights and before a single trial, on a card another process was already
        using. The governor shrinks the chunk on out-of-memory and waits when the card is too full
        for even one prompt, which is machinery this file already had and this path alone did not
        use.

        Memory here is bounded by the caller's prompt count rather than streamed: the whole cloud
        is the point, since PCA needs it. For the models this tool targets that is small (25 layers
        x 256 prompts x 2048 wide x 4 bytes is 52 MB), and the arithmetic is written down in the
        deferred ledger rather than left to be rediscovered.
        """
        def _capture(chunk):
            ch = [self.chat(p) for p in chunk]
            enc = self.tok(ch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            out = self.model(**enc, output_hidden_states=True, use_cache=False)
            # The tokenizer is LEFT-padded (see __init__), so the real prompt always ends at the
            # final column and the last real token is at index -1 for every row. Indexing by
            # attention_mask.sum()-1 would point into the pad region for every prompt shorter than
            # the batch max, averaging pad-token activations into the refusal subspace (the old bug).
            per = torch.stack([h[:, -1, :].float().cpu() for h in out.hidden_states], 0)  # [NL+1, b, H]
            # One entry per PROMPT, not one per chunk: the governor's contract is a list the same
            # length as the items it was handed, which is also what lets it re-run a shrunken chunk
            # without the caller having to know the chunk boundaries moved.
            return [per[:, j, :] for j in range(per.shape[1])]

        rows = self.capture_gov.run(_capture, list(prompts))
        if not rows:
            raise ValueError("no activations captured: the prompt list was empty.")
        return torch.stack(rows, 1)  # [NL+1, N, H]

    def extract_directions(self, bad_dir, good_dir_path, hedge_ds, clean_src):
        # Build up to KMAX ORTHONORMAL refusal directions per layer into self.dirs_multi.
        args, NL, H, KMAX, log = self.args, self.NL, self.H, self.KMAX, self.log
        log("extracting refusal directions (primary + secondary)")
        self.events.emit("phase", phase="phase_directions")
        bad = self.load(bad_dir, args.dir_prompts)
        good = self.load(good_dir_path, args.dir_prompts)
        if not bad or not good:
            raise ValueError(
                f"empty contrast set (bad={len(bad)} good={len(good)} prompts): need non-empty "
                f"datasets at {bad_dir} and {good_dir_path}. Direction extraction needs both.")
        Rb = self.collect_resid(bad)                         # [NL+1, Nb, H] cpu float32
        Rg = self.collect_resid(good)                        # [NL+1, Ng, H]
        mb = Rb.mean(1); mg = Rg.mean(1)                     # [NL+1, H]
        # The pool matched scoring draws its controls from. It is `Rg` unless a set written on the
        # harmful side's own subjects was supplied, and the distinction matters because matching
        # can only ever find what the pool contains: asking for the nearest harmless rows to a
        # cluster about explosives returns rows either way, and they are controls only if
        # something on the subject is in there to find.
        #
        # `mg` and `good_dir` deliberately stay on `Rg` whatever this is. They exist to keep the
        # ablation from tearing out good behaviour in general, which is a different job from
        # holding the subject still while refusal varies, and pointing them at a small on-subject
        # corpus would narrow what the run protects.
        Rc = Rg
        matched_src = getattr(args, "harmless_matched", "") or ""
        # Resolved here rather than at its point of use so the corpus below can only be loaded on
        # the path that reads it. `run_parsed` already refuses this combination before the model
        # is downloaded; this guard is for a caller constructing the class directly, and shares
        # one wording with it so the two cannot drift.
        matched_scoring = bool(getattr(args, "matched_scoring", False))
        if matched_src and not matched_scoring:
            raise SystemExit(_unused_matched_corpus(matched_src))
        if matched_src:
            matched_rows = self.load(matched_src, args.dir_prompts)
            if not matched_rows:
                raise ValueError(
                    f"--harmless-matched {matched_src} holds no prompts. It is the pool matched "
                    f"scoring draws its controls from, so an empty one would silently fall back "
                    f"to the ordinary harmless set and report matched figures taken against it.")
            Rc = self.collect_resid(matched_rows)            # [NL+1, Nc, H]
            log(f"matched control pool: {len(matched_rows)} prompts from {matched_src}")
            if len(matched_rows) < 2 * MIN_HELD_OUT_ROWS:
                log(f"  NOTE: {len(matched_rows)} matched prompts cannot be split into two halves "
                    f"of {MIN_HELD_OUT_ROWS}, so candidates fall back to in-sample scoring and "
                    f"the separation filter is not evidence about them.")
        # Refinement 3 (Heretic `orthogonalize_direction`): keep only the component ORTHOGONAL to
        # the good direction, so ablation does not tear out good behaviour itself (a major cause of
        # our early high harmless KL). clamp_min guards a degenerate (near-zero) mean from producing
        # NaN directions that would then run the whole search on garbage.
        good_dir = mg / mg.norm(dim=-1, keepdim=True).clamp_min(1e-8)   # unit good direction, per layer

        # Optional hedging contrast (lever 2): the difference-of-means captures HARD refusal, not the
        # disclaimer/"I must warn you" hedging that the keyword metric flags on complying answers. If a
        # hedged-compliance set is supplied, extract mean(hedged) - mean(clean) at each layer and fold it
        # in as a GUARANTEED ablated direction, so the search can strip the hedging axis the hard-refusal
        # direction never sees. Any datasets.save_to_disk directory with a 'text' column will do;
        # `python -m senbonzakura.track` builds the main three, and a hedge set is the same shape.
        hedge_md = None
        if hedge_ds:
            hedged = self.load(hedge_ds, args.dir_prompts)
            cleanc = self.load(clean_src, args.dir_prompts)
            hedge_md = self.collect_resid(hedged).mean(1) - self.collect_resid(cleanc).mean(1)   # [NL+1, H]
            log(f"hedging contrast: {len(hedged)} hedged vs {len(cleanc)} clean (folded as a guaranteed direction)")

        # Refinement 5 (multi-directional): d0 is the difference-of-means (the canonical Arditi
        # direction), good-orthogonalized; d1.. are the top principal axes of the bad residual cloud
        # AFTER projecting out good_dir and the earlier directions, so the set spans the refusal
        # SUBSPACE (refusal is not always a single direction). The search picks how many (num_directions)
        # to actually ablate.
        dirs_multi = torch.zeros(NL + 1, KMAX, H)
        # Per layer, the separation of every candidate PCA axis that was actually measured, kept or
        # not. Without this a layer that got one direction is indistinguishable from a layer whose
        # second candidate missed the threshold by a hair, and that difference is the whole
        # question of whether refusal here is one direction or one direction plus a constant.
        axis_seps = [[] for _ in range(NL + 1)]
        # `axis_seps` is a bounded SAMPLE (the leading MAX_RECORDED_AXES per layer). These two are
        # exact totals over every axis actually measured, which is far more: one real layer measured
        # 127 candidates against the 8 that were kept in the record. A verdict about whether any
        # axis can clear the threshold has to come from all of them, not from the first few.
        axes_measured_total = 0
        axes_rejected_total = 0
        # Of those rejections, the ones that cleared the fixed constant and still lost to a
        # direction carrying nothing. Recorded apart because it is the only number that says the
        # null floor did any work; a total alone cannot distinguish a filter with a measured floor
        # from a filter with a lucky constant.
        axes_rejected_by_null = 0
        layer_null_floors = [None] * (NL + 1)
        hedge_applied = [False] * (NL + 1)
        max_sep_seen = 0.0
        # Which statistic decides whether a candidate axis carries refusal. Resolved once here
        # rather than per layer, so every layer of a run is judged on one ruler and the artefact
        # can name it. `getattr` because the forward-only commands share this class's helpers
        # through argument namespaces that never had the flag.
        sep_stat = separation.get(getattr(args, "separation_statistic",
                                          separation.DEFAULT_STATISTIC))
        # `matched_scoring` (whether a candidate is judged against matched controls or against the
        # harmless set at large, Q-23) is resolved above, before the control pool is loaded, so the
        # corpus can only be read on the path that uses it. One comparison per run, named in the
        # artefact, rather than a property a reader has to infer from the command line.
        # How well the matching actually worked, gathered across every candidate at every layer.
        # Whether on-subject harmless prompts exist is a property of the corpus, and a run that
        # asked for matched controls and got arbitrary ones must say so rather than publish
        # "matched" numbers taken against rows that match nothing.
        matching_ratios = []
        closeness_values = []
        # Never propose more clusters than there are prompts to fill them at MIN_CLUSTER_ROWS
        # each. Below two, there is no second refusal mode to look for and the run says so rather
        # than quietly measuring nothing: a contrast set this small cannot support the claim, and
        # silently returning one direction is how the previous defect stayed invisible.
        n_clusters = min(int(args.direction_clusters), len(bad) // MIN_CLUSTER_ROWS)
        if KMAX > 1 and n_clusters < 2:
            log(f"  NOTE: {len(bad)} harmful prompts cannot fill two clusters of "
                f"{MIN_CLUSTER_ROWS}, so no second refusal mode can be looked for and this run "
                f"applies a SINGLE direction per layer despite --max-directions {KMAX}. Raise "
                f"--dir-prompts to at least {2 * MIN_CLUSTER_ROWS} to search for more. The "
                f"per-layer counts are recorded in abliteration.json under directions_per_layer.")
        for li in range(NL + 1):
            gd = good_dir[li]
            if args.no_good_orth:
                # Ablation study: raw difference-of-means, NOT orthogonalised to the harmless mean,
                # and the harmless direction is left out of the basis so the PCA axes are not
                # good-orthogonalised either. This is the toggle that isolates Refinement 3 (the
                # projection grimjim proposes) so its effect on the search can be measured.
                d0 = mb[li] - mg[li]
                d0 = d0 / d0.norm().clamp_min(1e-8)
                basis = [d0]; kept = [d0]
            else:
                d0 = _orth_to(mb[li] - mg[li], [gd])
                d0 = d0 / d0.norm().clamp_min(1e-8)
                basis = [gd, d0]; kept = [d0]
            # At most H mutually orthonormal vectors exist in an H-dimensional space, and everything
            # kept has to stay orthonormal to the rest of `basis`, which on the default path also
            # holds the harmless direction (and does not under --no-good-orth). So the ceiling on
            # kept directions is H minus whatever sits in the basis without being kept, not H.
            # Asking for more used to hand the PCA loop pure numerical noise to normalise: with H=8
            # the post-projection residual of a linearly dependent axis still cleared the 1e-6 norm
            # guard in float32, so it was scaled to unit length and ablated as though it carried
            # refusal. Cap what is kept; KMAX itself stays as-is because it sets the tensor width.
            kmax_eff = min(KMAX, H - (len(basis) - len(kept)))
            # Guaranteed hedging direction (lever 2), good- and d0-orthogonalised, before the
            # cluster candidates fill the rest.
            #
            # It is gated on there being room, which means K=1 SILENTLY LOSES IT: at K=1 the
            # primary direction already fills the budget. That turns any K=1 against K>1
            # comparison into a comparison of "more directions AND a supervised hedging contrast
            # aimed at the very metric being reported", which is two changes wearing one name.
            # Recorded per layer and announced below rather than left to be discovered.
            if hedge_md is not None and len(kept) < kmax_eff:
                hv = _orth_to(hedge_md[li], basis)
                n = hv.norm()
                if n > 1e-6:
                    hv = hv / n
                    kept.append(hv); basis.append(hv)
                    hedge_applied[li] = True
            if kmax_eff > len(kept) and n_clusters >= 2:
                # Candidates are per-CLUSTER difference-of-means, not principal axes of the
                # harmful cloud. The distinction is the whole reason this code was rewritten on
                # 2026-08-03, and it is not a matter of taste:
                #
                # A principal axis describes how the harmful cloud VARIES, which is a different
                # question from what separates harmful from harmless. Worse, every candidate had
                # to be orthogonalised against a basis spanning both class means, and the filter
                # that judged it was a difference of those means, so the score was exactly zero
                # by construction and no axis could ever be kept. See private/research/
                # 2026-08-03-the-separation-filter-can-never-pass.md.
                #
                # Clustering fixes the cause rather than the symptom. Refusal is not one
                # behaviour: a model refuses a weapons request differently from a self-harm one.
                # Each cluster's own mean, measured against the harmless mean, is a direction that
                # separates BY CONSTRUCTION, and what survives orthogonalisation against d0 is
                # precisely the part of that cluster's refusal the global mean difference misses.
                # That residue is what a second direction is supposed to be.
                # The harmless rows are split once per layer and the halves are used for every
                # candidate at that layer, so two candidates are judged against the same rows and
                # their scores are comparable. Splitting per candidate would make each score a
                # measurement on a different exam.
                # Split the pool the scoring actually uses. Under matched scoring with a supplied
                # matched set that is `Rc`, so the halves the controls are drawn from and the
                # halves the null floor is measured through are the same rows the comparison runs
                # on. Splitting `Rg` here while drawing controls from `Rc` would have measured the
                # floor on one corpus and the candidates on another.
                ctl = Rc[li] if matched_scoring else Rg[li]
                gi_fit, gi_score = _halves(int(ctl.shape[0]), args.seed)
                good_fit, good_score = ctl[gi_fit], ctl[gi_score]
                held_out_usable = (len(gi_fit) >= MIN_HELD_OUT_ROWS
                                   and len(gi_score) >= MIN_HELD_OUT_ROWS)
                if not held_out_usable and li == self.lo:
                    log(f"  NOTE: {int(ctl.shape[0])} harmless prompts cannot be split into "
                        f"two halves of {MIN_HELD_OUT_ROWS}, so candidate directions are scored "
                        f"IN SAMPLE and the refusal-separation filter is not evidence about "
                        f"them. Raise --dir-prompts to at least {2 * MIN_HELD_OUT_ROWS}.")
                labels = _kmeans_labels(Rb[li], n_clusters, args.seed + li)
                # Largest clusters first, ties broken by cluster id, so the candidate ORDER does
                # not depend on dictionary iteration or on how many directions were requested.
                sizes = [(int((labels == c).sum()), int(c)) for c in labels.unique()]
                sizes.sort(key=lambda s: (-s[0], s[1]))

                # Every candidate is scored before any is kept, then the best are taken. Ranking
                # rather than first-past-the-post is what makes a K comparison honest: the
                # candidate SET is identical whatever K is, so K is a budget and nothing else.
                # The withdrawn five-seed comparison failed for exactly the opposite reason.
                eligible = [(size, c) for size, c in sizes if size >= MIN_CLUSTER_ROWS]
                # The floor a candidate has to clear, measured rather than assumed, AND MEASURED AT
                # THAT CANDIDATE'S OWN ROW COUNT (Q-22). It used to be drawn once per layer at the
                # median eligible cluster size, so every candidate away from that median was judged
                # against a bar built for a different cluster. That matters because a null's value
                # depends on the group size for every statistic here: the incumbent keeps a
                # meaningless axis 21.2% of the time at 8 rows and 0.0% at 256 against one constant.
                # Cached by size, because two clusters of the same size share a floor exactly and
                # the draws are the expensive part.
                floors_by_size = {}

                scored, dropped, dropped_by_null = [], 0, 0
                for _size, c in eligible:
                    rows = Rb[li][labels == c]
                    cand_size = int(rows.shape[0])
                    # `basis` is read here and appended to only after this loop finishes, so every
                    # candidate at this layer is scored against the same basis and their floors
                    # are comparable.
                    if held_out_usable and cand_size not in floors_by_size:
                        value, _null_samples = _null_separation_floor(
                            Rb[li], good_fit, good_score, basis, cand_size,
                            args.seed + li, NULL_DIRECTIONS_PER_CANDIDATE, sep_stat,
                            matched=matched_scoring)
                        floors_by_size[cand_size] = float(value)
                    null_floor = floors_by_size.get(cand_size, 0.0)
                    threshold = max(sep_stat.threshold, null_floor)
                    # Judged out of sample, then fitted on everything. The decision has to be made
                    # on rows the candidate never saw or it is not a decision; the direction that
                    # is actually applied should still use every row available to estimate it.
                    # The anchor the candidate direction is measured FROM. Under matched scoring it
                    # must be the same controls the score was taken against: selecting a direction
                    # on one comparison and then ablating a direction built from another would
                    # keep an axis for carrying refusal and cut an axis carrying topic.
                    anchor = mg[li]
                    if matched_scoring:
                        midx = _matched_harmless_idx(rows, ctl, basis, cand_size)
                        if midx is None or len(midx) < MIN_HELD_OUT_ROWS:
                            continue
                        anchor = ctl[midx].mean(0)
                        q = matching_quality(rows, ctl, basis, cand_size)
                        if q is not None:
                            matching_ratios.append(q)
                        # The scale-free companion (Q-25 D2). Recorded beside the alarm rather
                        # than replacing it: the alarm is what the artefact has always carried,
                        # and this is the number corpus work is measured against.
                        closeness = match_closeness(rows, ctl, basis, cand_size, args.seed + li)
                        if closeness is not None:
                            closeness_values.append(closeness)
                    if held_out_usable:
                        score_fn = (_matched_held_out_separation if matched_scoring
                                    else _held_out_separation)
                        sep = score_fn(rows, good_fit, good_score, basis,
                                       args.seed + li + int(c), sep_stat)
                        if sep is None:
                            continue
                    else:
                        # No usable harmless split. Scored in sample, which the note above already
                        # said is not evidence; kept rather than skipped so a small-corpus run
                        # still produces directions instead of silently becoming a K=1 run.
                        v_probe = _orth_to(rows.mean(0) - anchor, basis)
                        if v_probe.norm() < 1e-6:
                            continue
                        sep = _axis_separation(rows, ctl, v_probe / v_probe.norm(), sep_stat)
                    v = _orth_to(rows.mean(0) - anchor, basis)
                    n = v.norm()
                    if n < 1e-6:
                        continue      # this cluster's refusal is entirely inside what d0 already cuts
                    v = v / n
                    axes_measured_total += 1
                    max_sep_seen = max(max_sep_seen, abs(float(sep)))
                    if len(axis_seps[li]) < MAX_RECORDED_AXES:
                        axis_seps[li].append(round(float(sep), 4))
                    if sep < threshold:
                        dropped += 1
                        axes_rejected_total += 1
                        if sep >= sep_stat.threshold:
                            # Cleared the fixed constant and lost to a direction carrying nothing.
                            # Counted apart because it is the case the null was added to catch.
                            dropped_by_null += 1
                            axes_rejected_by_null += 1
                        continue
                    scored.append((float(sep), int(c), v))

                scored.sort(key=lambda s: (-s[0], s[1]))
                for _sep, _c, cand in scored:
                    if len(kept) >= kmax_eff:
                        break
                    # Re-orthogonalise against what has been kept since this candidate was scored.
                    # Two clusters can carry overlapping refusal, and keeping both unmodified
                    # would put a near-duplicate row in a basis the bake assumes is orthonormal.
                    w = _orth_to(cand, basis)
                    n = w.norm()
                    if n < 1e-6:
                        continue
                    w = w / n
                    kept.append(w); basis.append(w)
                # There is now a floor per candidate size rather than one per layer, so the record
                # keeps the STRICTEST bar any candidate here had to clear. A single number can no
                # longer describe them all, and the strictest is the one that decided the closest
                # call; the spread across sizes is what `floors_by_size` would show if the field
                # were widened, which it is not, because the artefact is read by people.
                if floors_by_size:
                    layer_null_floors[li] = round(max(floors_by_size.values()), 4)
                if dropped and li == self.lo:   # one representative log line, not NL of them
                    floor_lo, floor_hi = min(floors_by_size.values()), max(floors_by_size.values())
                    log(f"  layer {li}: dropped {dropped} cluster direction(s) below the "
                        f"refusal-separation threshold ({sep_stat.name}, the larger of the fixed "
                        f"{sep_stat.threshold} and a null floor measured at each candidate's own "
                        f"row count, which ranged {floor_lo:.4f} to {floor_hi:.4f} across "
                        f"{len(floors_by_size)} distinct cluster size(s)); {dropped_by_null} of "
                        f"them cleared the constant and lost to a direction carrying nothing")
            for j, v in enumerate(kept):
                dirs_multi[li, j] = v
        self.dirs_multi = dirs_multi.to(torch.bfloat16)      # [NL+1, KMAX, H]; unused rows stay 0 (ablate nothing)
        # How many directions each layer ACTUALLY got. A layer can fall short of KMAX for
        # several legitimate reasons (the separation filter, the rank floor, a degenerate
        # cloud) and the request is not evidence of the result, so record the achieved
        # count per layer and say so when it differs. Without this, a run reports the K it
        # asked for and nothing anywhere states the K it applied.
        # INDEXED BY RESIDUAL-STREAM POSITION, NOT BY LAYER, and the difference is load-bearing.
        # `dirs_multi` has NL+1 entries: position 0 is the embedding output and position i+1 is
        # what decoder layer i writes into, which is why `active_dirs` reads `dirs_multi[idx+1]`.
        # Position 0 is never ablated, because `embed_tokens` is deliberately left alone.
        self.dirs_per_position = [int((self.dirs_multi[li].float().norm(dim=-1) > 1e-6).sum())
                                  for li in range(NL + 1)]
        # The LAYER view, which is what every message and every reader actually wants. Dropping
        # position 0 is what makes the name true.
        #
        # It was one list called `dirs_per_layer` carrying positions, and that cost a real
        # investigation: the record began `[0, 1, 1, ...]` on every model, and a reader concluded
        # layer 0 received no direction. It always received one, from position 1. The same
        # mislabelling also made `min()` of the list ALWAYS zero, so every run with a shortfall
        # announced "the counts run 0 to K" and named one layer too many, and it sliced the search
        # window with layer indices into a position-indexed list, reporting the window one layer
        # off. One wrong noun, three wrong statements.
        self.dirs_per_layer = self.dirs_per_position[1:]
        # Report the WHOLE model, and name the window separately. Scoping this line to the
        # search window cost most of a day on 2026-08-03: it said "within the search window
        # layers got 1 to 1 directions", which left open, and wrongly, that layers outside the
        # window had more. They did not, and the comparison the run existed to make had been
        # measuring nothing for hours before anyone read the per-layer record.
        #
        # The loudest case is the one that matters: when NO layer anywhere got more than one
        # direction, a multi-direction run is a single-direction run and must say so in those
        # words, because that is the sentence a reader needs and "1 to 1" is not it.
        # The separations themselves, and the one number that says whether the single-direction
        # result is a property of the model or of the threshold: the largest separation any
        # REJECTED axis reached. A best rejected value just under the threshold means the
        # constant decided the outcome; one far below it means the second direction is not there.
        self.axis_separations = axis_seps
        # WHICH RULER produced every number above. Without it two runs' separations are on
        # different scales and nothing in the record says so, which is the confound this project
        # has met three times: a figure that describes the rows, the tool or the threshold it was
        # taken under, with the record silent about which.
        self.separation_statistic = sep_stat.name
        self.separation_null = sep_stat.null
        # WHAT THE CANDIDATE WAS COMPARED AGAINST, which decides what its score can mean at all.
        # An unmatched score says a cluster stands out from harmless prompts; a matched one says
        # it stands out from harmless prompts on the same subject. Only the second is evidence
        # about refusal, and a record that does not say which is not a record.
        self.matched_scoring = matched_scoring
        # Which corpus the controls came from. "matched against what" is not answerable from the
        # ratio alone, and a run that used a dedicated on-subject set and one that used the
        # nearest rows of the ordinary harmless set are different measurements.
        self.matched_source = matched_src or None
        self.matching_quality = (round(sorted(matching_ratios)[len(matching_ratios) // 2], 4)
                                 if matching_ratios else None)
        # How near the controls are on the corpus's own scale. Unlike the ratio above this can be
        # read as a quality and used as a target, because padding the corpus cannot improve it.
        self.match_closeness = (round(sorted(closeness_values)[len(closeness_values) // 2], 4)
                                if closeness_values else None)
        if self.matching_quality is not None and self.matching_quality > MATCHING_USELESS_RATIO:
            log(f"  MATCHING ACHIEVED NOTHING: the harmless prompts chosen as controls sit "
                f"{self.matching_quality:.2f} times the average distance from their candidate, "
                f"where 1.00 means they are no nearer than any harmless prompt taken at random. "
                f"The corpus has no harmless prompts on these subjects, so --matched-scoring "
                f"asked for controls and received arbitrary rows. Every separation number from "
                f"this run is an UNMATCHED number wearing a matched label; treat it as such and "
                f"build a topic-matched harmless set before reading it as evidence about refusal.")
        self.axes_measured_total = axes_measured_total
        self.max_axis_separation = max_sep_seen if axes_measured_total else None
        measured = [d for layer in axis_seps for d in layer]
        rejected = [d for d in measured if d < sep_stat.threshold]
        self.best_rejected_separation = max(rejected) if rejected else None
        self.axes_rejected_total = axes_rejected_total
        self.axes_rejected_by_null = axes_rejected_by_null
        self.layer_null_floors = layer_null_floors
        floors = [f for f in layer_null_floors if f is not None]
        self.null_separation_floor = max(floors) if floors else None
        self.hedge_applied_layers = int(sum(hedge_applied))
        if hedge_md is not None and self.hedge_applied_layers < NL + 1:
            log(f"  NOTE: the hedging direction was applied at {self.hedge_applied_layers} of "
                f"{NL + 1} layers. It needs a free direction slot, so a run at "
                f"--max-directions 1 loses it entirely. A comparison against a larger K is then "
                f"a comparison of two things at once, the direction count and the hedging "
                f"contrast, and the hedging contrast targets the metric being reported.")

        # A guard that accepts everything discriminates exactly as much as one that rejects
        # everything: not at all. Both extremes are alarms and both have now happened here, the
        # first for the project's whole history and the second within an hour of fixing it. The
        # rejection rate is therefore reported on every run rather than inspected when something
        # already looks wrong, because "the filter exists" was taken for "the filter works" twice.
        # The floor the threshold was actually held to, printed on every run whatever the outcome.
        # Until 2026-08-16 the threshold was a constant nobody had ever compared against a
        # measurement, and the filter it governed rejected nothing on every run for the project's
        # whole history. A floor that is never shown is a floor nobody checks.
        if self.null_separation_floor is not None:
            log(f"  null-direction floor: a direction built from a random subset of the harmful "
                f"rows, carrying nothing, scores up to {self.null_separation_floor:.4f} through "
                f"the same held-out path. Candidates were held to "
                f"max({sep_stat.threshold}, that), and {axes_rejected_by_null} cleared the "
                f"constant but not the floor.")
        if axes_measured_total:
            reject_rate = axes_rejected_total / axes_measured_total
            if reject_rate == 0.0:
                log(f"  NOTE: the refusal-separation filter rejected NONE of "
                    f"{axes_measured_total} candidate directions. Every one beat both the fixed "
                    f"{sep_stat.threshold} and the measured null floor on rows it was not "
                    f"fitted on, which is a stronger statement than this note used to carry, but "
                    f"a filter that rejects nothing still discriminates nothing. It is evidence "
                    f"the candidates separate held-out harmful from held-out harmless prompts; "
                    f"it is NOT evidence they carry refusal rather than topic. To ask that "
                    f"question, re-run with --matched-scoring, which judges each candidate "
                    f"against the harmless prompts nearest it in content rather than against the "
                    f"set at large. A subject-matched CORPUS alone does not answer it: measured "
                    f"2026-09-02, scoring a matched corpus against the pool still could not tell "
                    f"a world containing refusal from one containing none.")
            elif reject_rate == 1.0:
                log(f"  NOTE: the refusal-separation filter rejected ALL "
                    f"{axes_measured_total} candidate directions, so no candidate could be kept "
                    f"and the direction count is a property of the filter rather than the model.")
        # An exact zero across every axis is not a small measurement. The candidates are
        # orthogonalised against a basis spanning both class means, and Cohen's d is a difference
        # of class means, so the numerator vanishes by construction and no positive threshold can
        # be cleared. Measured 2026-08-03 on two models and three corpora: every one of 224 axes
        # returned ~1e-8. This is louder than the shortfall note below because a shortfall is a
        # result and this is a broken instrument.
        structural_zero = sep_stat.threshold * STRUCTURAL_ZERO_FRACTION
        self.filter_is_unsatisfiable = bool(axes_measured_total) and max_sep_seen < structural_zero
        if self.filter_is_unsatisfiable:
            log(f"  BROKEN FILTER: all {axes_measured_total} candidate axes scored a refusal separation "
                f"of ~0, which the geometry forces rather than the data: the axes are "
                f"orthogonalised against a basis spanning both class means, and the separation "
                f"statistic is a difference of class means. No axis can clear "
                f"{sep_stat.threshold}, or any positive value, so --max-directions above 1 "
                f"cannot take effect. See private/research/"
                f"2026-08-03-the-separation-filter-can-never-pass.md.")
        elif self.best_rejected_separation is not None:
            near = self.best_rejected_separation >= sep_stat.threshold * 0.8
            log(f"  rejected-axis separations: {len(rejected)} axis/axes measured below the "
                f"threshold, best {self.best_rejected_separation:.4f} against "
                f"{sep_stat.threshold}"
                + (". That is close enough to the threshold that the cut-off, not the model, "
                   "decided the direction count; treat the single-direction reading as a "
                   "property of the threshold until it is varied." if near else
                   ". Well clear of the threshold, so lowering it would not add a direction."))

        window = self.dirs_per_layer[self.lo:self.hi + 1] or self.dirs_per_layer
        whole = self.dirs_per_layer or [0]
        if KMAX > 1 and max(whole) <= 1:
            log(f"  NOTE: no layer anywhere got more than one direction, so this run ablates a "
                f"SINGLE direction per layer despite --max-directions {KMAX}. Nothing here is "
                f"evidence about multiple directions. The per-layer counts are in "
                f"abliteration.json under directions_per_layer.")
        elif whole and min(whole) < KMAX:
            log(f"  note: across all {len(whole)} layers the counts run {min(whole)} to "
                f"{max(whole)}, not the {KMAX} requested (search window layers "
                f"{self.lo} to {self.hi}: {min(window)} to {max(window)}); the applied K is "
                f"recorded per layer in abliteration.json under directions_per_layer.")
        log(f"directions ready: {tuple(self.dirs_multi.shape)} (<= {KMAX}/layer, orthonormal, "
            f"good-orthogonalized, refusal-separation filtered)")

    def _interp_multi(self, fidx):
        # Refinement 6 (Heretic float `direction_index`): linearly interpolate the K-direction set
        # between the two nearest residual-stack indices and re-orthonormalize. A float index reaches
        # refusal directions that are not aligned to any single layer.
        NL = self.NL
        lo_i = int(fidx); hi_i = min(lo_i + 1, NL); frac = fidx - lo_i
        M = (1 - frac) * self.dirs_multi[lo_i].float() + frac * self.dirs_multi[hi_i].float()  # [KMAX, H]
        out = torch.zeros_like(M)
        for j in range(M.size(0)):                        # Gram-Schmidt to restore orthonormality
            v = M[j].clone()
            for i in range(j):
                v = v - (v @ out[i]) * out[i]
            n = v.norm()
            out[j] = v / n if n > 1e-6 else v * 0.0
        return out.to(torch.bfloat16)                     # [KMAX, H]

    def _fold_for(self, R, norm):
        """R, or the gain-folded equivalent when a norm intercepts the write.

        Cached per (norm, R version) would be premature: the gain probe is one forward pass
        through a single RMSNorm on a [1,1,H] tensor, which is negligible beside the matmuls this
        sits between, and caching it would need invalidating whenever the direction set changes.
        """
        if norm is None:
            return R
        g = norm_gain(norm, self.H, self.dev, torch.float32)
        return fold_norm_gain(R, g)

    def active_dirs(self, idx, K):
        # The K unit directions to ablate at layer `idx` under the current dir_mode:
        #   per_layer -> that layer's own set (dirs_multi[idx+1])
        #   single    -> one interpolated set shared across all layers (Heretic-style)
        M = self._cur["single_set"] if self._cur.get("mode") == "single" and self._cur.get("single_set") is not None \
            else self.dirs_multi[idx + 1]
        return M[:K]   # [K, H]

    # ── reversible norm-preserving bake (used by BOTH the search and the final save) ──
    def snapshot_weights(self):
        # Snapshot the pristine residual-writing weights ONCE so the search can apply the real
        # bake, score it, then restore exactly between trials. This unifies the search and the
        # save: the search optimises the same norm-preserving surgery we ultimately keep, so there
        # is no proxy/bake gap (the old activation-hook proxy over-estimated damage, e.g. proxy
        # KL 4.5 vs baked KL 0.30 on Qwen3-1.7B). NOTE: holds one CPU copy of every o_proj +
        # down_proj; trivial for small models, heavy for the 30B, hence the RAM pre-flight below.
        self._pristine.clear(); self._dirty.clear()
        targets = []
        for layer in self.layers:
            targets += layer_attn_writers(layer, ablate_conv=self.ablate_conv)
            for kind, obj in layer_downproj(layer):
                targets += obj if kind == "list" else [obj]
        need = sum(W.numel() * W.element_size() for W in targets)
        avail = _available_ram_bytes()
        if avail is None:
            # Both probes are POSIX: /proc/meminfo is Linux-only and SC_AVPHYS_PAGES is
            # absent on Windows and unreliable on macOS. Proceeding is right, because an
            # unmeasurable machine is not a small one, but proceeding QUIETLY is not: the
            # operator would believe a guard ran when none did, and the failure it guards
            # against is an out-of-memory kill partway through a rented GPU run.
            self.log(f"  snapshot: cannot measure host RAM on this platform, so the "
                     f"{need/1e9:.1f} GB pre-flight was skipped, not passed")
        elif need > 0.9 * avail:
            raise MemoryError(
                f"the reversible search needs {need/1e9:.1f} GB of host RAM to hold the pristine copy "
                f"of every o_proj + down_proj, but only {avail/1e9:.1f} GB is available. Free memory, "
                f"pick a smaller model, or run on a box with more RAM.")
        if need > 8e9:
            self.log(f"  snapshot: holding {need/1e9:.1f} GB of pristine weights in host RAM")
        for W in targets:
            self._pristine[id(W)] = (W, W.detach().clone().to("cpu"))

    def _mark_dirty(self, W):
        self._dirty.add(id(W))

    @torch.no_grad()
    def restore_weights(self):
        # Copy the pristine values back into just the weights the last bake touched.
        for i in list(self._dirty):
            W, c = self._pristine[i]
            W.copy_(c.to(W.device))
        self._dirty.clear()

    @torch.no_grad()
    def bake_pc(self, oP, owmax, owmin, oD, dP, dwmax, dwmin, dD, K=1, mode="per_layer", didx=None):
        # PER-COMPONENT windowed ablation (Heretic decouples attn.o_proj from mlp.down_proj):
        # attn.o_proj follows the o-profile, mlp.down_proj the d-profile. The d-profile may be
        # all-zero (dwmax==0) to leave the MLP entirely untouched, which Heretic's issue #202 finds
        # often preserves intelligence (ablating the MLP hurts more than it helps). Windowed per
        # layer (ref 1), directions per config (ref 2/5/6), norm-preserving (ref 4). embed_tokens is
        # deliberately left alone (Heretic notes the benefit is unclear + a norm-preserving embed edit
        # is awkward).
        self._cur["mode"] = mode
        self._cur["single_set"] = self._interp_multi(didx) if (mode == "single" and didx is not None) else None
        sp = float(self.args.sparsity)         # sparse surgery: 0 = edit every row
        # 0 keeps the historical single pass, which leaves part of the direction in place; see
        # `orthogonalize_np_`. getattr because the forward-only paths share this class through
        # namespaces that never had the flag.
        rounds = int(getattr(self.args, "ablation_rounds", 0))
        for idx, layer in enumerate(self.layers):
            wo = layer_weight(idx, oP, owmax, owmin, oD)
            wd = layer_weight(idx, dP, dwmax, dwmin, dD)
            if wo == 0.0 and wd == 0.0:
                continue
            R = self.active_dirs(idx, K).to(self.dev).float()   # [K, H]
            R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-8)  # renormalize (interp/GS drift); zero rows stay ~0
            # On Gemma-2/3 and Olmo-2 a learned-gain norm sits between these weights and the
            # residual stream, so removing span(R) here is not removing it from the stream. Fold
            # the gain in so the post-norm write is what ends up orthogonal to R. On every other
            # architecture both norms are None and R is used unchanged.
            attn_norm, mlp_norm = post_sublayer_norms(layer)
            R_attn = self._fold_for(R, attn_norm)
            R_mlp = self._fold_for(R, mlp_norm)
            if wo > 0.0:
                # Every residual writer in the attention position, which on a hybrid is the
                # convolution's out_proj on the layers that have no attention at all.
                for op in layer_attn_writers(layer, ablate_conv=self.ablate_conv):
                    self._mark_dirty(op); orthogonalize_np_(op, R_attn, wo, sp, rounds=rounds)
            if wd > 0.0:
                for kind, obj in layer_downproj(layer):         # every residual-writing down-proj
                    if kind == "fused3d":
                        self._mark_dirty(obj); orthogonalize_np_3d_(obj, R_mlp, wd, sp, rounds=rounds)
                    elif kind == "list":
                        for W in obj:
                            self._mark_dirty(W); orthogonalize_np_(W, R_mlp, wd, sp, rounds=rounds)
                    else:  # dense
                        self._mark_dirty(obj); orthogonalize_np_(obj, R_mlp, wd, sp, rounds=rounds)

    @torch.no_grad()
    def bake(self, P, wmax, wmin, D, K=1, mode="per_layer", didx=None):
        # Uniform wrapper: the same profile on both components (the pre-decouple behaviour). Used by
        # --inspect / --bench-only and the --uniform search path.
        self.bake_pc(P, wmax, wmin, D, P, wmax, wmin, D, K, mode, didx)

    # ── eval: refusals (generation) + KL vs original (first-token) ───────────────
    @torch.no_grad()
    def gen_batch(self, prompts):
        # The governor drives the chunking: it picks the batch size from live free VRAM, shrinks on an
        # out-of-memory instead of crashing, and pauses/resumes when another app takes the card.
        def _do(chunk):
            ch = [self.chat(p) for p in chunk]
            enc = self.tok(ch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            g = self.model.generate(**enc, max_new_tokens=self.args.gen_tokens, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
            return [self.tok.decode(g[j, enc.input_ids.size(1):], skip_special_tokens=True)
                    for j in range(g.size(0))]
        return self.gov.run(_do, list(prompts))

    @torch.no_grad()
    def refusal_rate(self, prompts):
        return sum(is_refusal(t) for t in self.gen_batch(prompts)) / max(1, len(prompts))

    @torch.no_grad()
    def first_token_logprobs(self, prompts):
        # Same governor-paced chunking as gen_batch; each chunk yields a list of per-row [V] log-prob
        # tensors, which the governor concatenates in order for a final stack.
        def _do(chunk):
            ch = [self.chat(p) for p in chunk]
            enc = self.tok(ch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            lo = last_token_logits(self.model, enc, self.log)
            return list(F.log_softmax(lo, dim=-1).cpu())   # list of [V] tensors, one per prompt
        rows = self.gov.run(_do, list(prompts))
        return torch.stack(rows, 0)  # [N, V]

    @torch.no_grad()
    def kl_vs_orig(self, prompts):
        lp = self.first_token_logprobs(prompts)
        p = self.orig_lp.exp()
        return (p * (self.orig_lp - lp)).sum(-1).mean().item()  # KL(orig || ablated)

    # ── the search ───────────────────────────────────────────────────────────────
    def _suggest_profiles(self, trial):
        # Return (oP,owmax,owmin,oD, dP,dwmax,dwmin,dD). Per-component (default) tunes attn.o_proj
        # and mlp.down_proj separately; the MLP max_weight lower bound is NEGATIVE (clamped to 0) so
        # "leave the MLP alone" is a reachable, probable point in the search (Heretic's asymmetry).
        args, lo, hi, NL = self.args, self.lo, self.hi, self.NL
        if args.per_component:
            oP = trial.suggest_int("o_max_weight_position", lo, hi)
            owmax = trial.suggest_float("o_max_weight", 0.2, 1.4)
            owmin = trial.suggest_float("o_min_weight", 0.0, 0.6)
            oD = trial.suggest_int("o_min_weight_distance", 2, max(3, NL // 3))
            if args.mlp_off:
                # Attention-only: leave mlp.down_proj untouched and don't spend search
                # dimensions on it (lever 3).
                return (oP, owmax, owmin, oD, oP, 0.0, 0.0, oD)
            dP = trial.suggest_int("d_max_weight_position", lo, hi)
            dwmax = max(0.0, trial.suggest_float("d_max_weight", -0.25, 1.4))  # <=0 -> MLP untouched
            dwmin = trial.suggest_float("d_min_weight", 0.0, 0.6)
            dD = trial.suggest_int("d_min_weight_distance", 2, max(3, NL // 3))
            return (oP, owmax, owmin, oD, dP, dwmax, dwmin, dD)
        P = trial.suggest_int("max_weight_position", lo, hi)
        wmax = trial.suggest_float("max_weight", 0.2, 1.4)
        wmin = trial.suggest_float("min_weight", 0.0, 0.6)
        D = trial.suggest_int("min_weight_distance", 2, max(3, NL // 3))
        return (P, wmax, wmin, D, P, wmax, wmin, D)

    def objective(self, trial):
        # Search the windowed profile PER COMPONENT (peak position + peak/edge strengths + window
        # width for attn.o_proj and mlp.down_proj independently), not a single (layer, strength).
        args, log = self.args, self.log
        K = trial.suggest_int("num_directions", self.KMIN, self.KMAX)  # refinement 5: how many
        mode = trial.suggest_categorical("dir_mode", ["per_layer", "single"])
        didx = trial.suggest_float("direction_index", self.lo, self.hi) if mode == "single" else None  # refinement 6
        pr = self._suggest_profiles(trial)
        self.restore_weights()                             # undo the previous trial's bake
        self.bake_pc(*pr, K, mode, didx)                   # apply the REAL norm-preserving bake
        gens = self.gen_batch(self.bad_eval)
        ref = sum(is_refusal(t) for t in gens) / max(1, len(gens))
        soft = sum(is_soft_refusal(t) for t in gens) / max(1, len(gens))   # hedged compliance (lever 1/2)
        # Heretic-comparable rate (the axis Heretic wins)
        hk = heretic_keyword_rate(gens)
        brk = broken_rate(gens)                                         # coherence on the harmful set
        hbrk = broken_rate(self.gen_batch(self.kl_eval[:min(16, len(self.kl_eval))]))  # coherence on harmless
        kl = self.kl_vs_orig(self.kl_eval)
        broken = max(brk, hbrk)
        noncomp = ref + soft                                            # strict non-compliance: hard refusal + hedging
        trial.set_user_attr("refusals", ref); trial.set_user_attr("soft", soft); trial.set_user_attr("kl", kl)
        trial.set_user_attr("broken", broken); trial.set_user_attr("heretic", hk)
        # Drive non-compliance (hard AND hedged) to 0 while keeping the model intact: penalise KL past
        # the ceiling AND penalise incoherence directly, so a wrecked config (which scores LOW refusals
        # since broken != refusal) can never win. The keyword rate rides as a separate Pareto axis.
        scal = noncomp + 0.5 * hk + args.kl_scale * max(0.0, kl - KL_CEIL) + 0.3 * kl + 2.0 * broken
        log(f"  trial {trial.number}: o(P={pr[0]},wmax={pr[1]:.2f}) d(P={pr[4]},wmax={pr[5]:.2f}) K={K} {mode}"
            f"{'' if didx is None else f' di={didx:.1f}'} -> refusals={ref*100:.1f}% soft={soft*100:.1f}% "
            f"heretic={hk*100:.1f}% broken={broken*100:.0f}% KL={kl:.4f} obj={scal:.4f}")
        self.events.emit("trial", number=trial.number, k=K, mode=mode,
                         refusals=round(ref, 6), soft=round(soft, 6), heretic=round(hk, 6),
                         broken=round(broken, 6), kl=round(kl, 6), objective=round(scal, 6))
        if args.search == "pareto":
            # THREE objectives (lever 1): strict non-compliance (hard+hedge, with a broken penalty),
            # the Heretic keyword rate (its own axis so the search actually targets what Heretic wins),
            # and KL. NSGA-II maps the frontier; we pick the knee afterwards.
            return noncomp + 2.0 * broken, hk, kl
        return scal

    # ── the full pipeline ─────────────────────────────────────────────────────────
    def model_bytes(self):
        """Bytes the weights will occupy on disk, which is what save_pretrained writes."""
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def preflight_disk(self):
        """Refuse to start work whose only possible ending is a half-written save."""
        need = self.model_bytes()
        ok, message = disk_verdict(need, free_bytes_for(self.args.out))
        if not ok:
            raise SystemExit(f"not enough disk to save the result: {message}")
        self.log(message)

    def preflight_writable(self):
        """Prove every path this run writes to can be written to, before it spends an hour.

        Three artefacts used to be written into `--track`, which is an input. A read-only corpus
        (what the benchmark container mounts, and what anyone sharing a dataset would want) then
        killed the run one artefact at a time: the study at the first trial, the trial table after
        the last one, the winning config after the bake. Each failure arrived as a SQLAlchemy or
        OSError naming a `.part` file, an hour apart, with the GPU work already spent.

        Those writes now go to `--out`. This checks the paths anyway, because the next one added
        in the wrong place should cost a second at startup rather than an hour of card time.
        """
        db = study_db_path(self.args.study_db, self.args.no_persist_study,
                           self.args.track, self.args.out)
        targets = [("--out", self.args.out)]
        if not self.args.no_persist_study:
            targets.append(("--study-db", os.path.dirname(db) or "."))
        for flag, path in targets:
            self._prove_writable(flag, path)

    @staticmethod
    def _prove_writable(flag, path):
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".senbonzakura-write-probe")
            with open(probe, "w") as f:
                f.write("")
            os.unlink(probe)
        except OSError as e:
            raise SystemExit(
                f"{flag} points at {path}, which this run cannot write to: {e}. Every artefact "
                f"the run produces goes there, so it would fail partway through with the GPU work "
                f"already spent. Point it somewhere writable.") from e

    def free_before_save(self):
        """Give the save every resource it can have, because it is the crash-prone step.

        Two measured failures motivate this. First, `save_pretrained` on a fused MoE needs
        extra VRAM, because transformers 5.x reshapes the fused expert tensors on the way
        out; on a full 80 GB card that came up 384 MiB short. Second, the pristine snapshot
        holds a host-RAM copy of every residual-writing weight, and by this point it has
        done its job: the search is over and the winner is baked.

        Dropping the snapshot means restore_weights() no longer works, which is why this
        runs after the post-bake measurement and immediately before the write.
        """
        log = self.log
        held = len(self._pristine)
        self._pristine.clear()
        self._dirty.clear()
        gc.collect()
        if held:
            log(f"  save prep: released the pristine snapshot ({held} tensors) from host RAM")

        # A model accelerate dispatched across devices cannot be moved with .to(); moving a
        # resident one to the host sidesteps both the save-time VRAM spike and the fused-expert
        # revert path that causes it.
        if getattr(self.model, "hf_device_map", None):
            log("  save prep: model is dispatched across devices, leaving its placement alone")
        elif str(self.dev).startswith("cuda"):
            self.model.to("cpu")
            log("  save prep: moved the weights to host RAM for the write")
        if str(self.dev).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def eval_provenance(self):
        """Which rows the run's refusal figures came from, and what they may therefore be used for.

        The track splits harmful prompts into fit / search / measure and records where the
        boundaries fell, precisely so "held out" is a property of the files rather than an
        arithmetic convention. The abliterator reads the head of `bad_eval_ds`, which is the search
        partition followed by the measure partition, and `flag_violations` stops it reaching past
        the first. All of that is correct. What was missing is any statement of it in the output:
        the numbers were selection-set numbers and looked exactly like published ones.
        """
        args = self.args
        manifest = read_manifest(args.track) if args.track else None
        counts = ((manifest or {}).get("counts") or {}).get("harmful") or {}
        search_rows = counts.get("search")
        n = len(self.bad_eval)
        return {
            "track": str(args.track) if args.track else None,
            "dataset": "bad_eval_ds",
            "rows_scored": n,
            # None when there is no manifest to check against, which is honest: a bare prompt file
            # has no partition and claiming one would be worse than admitting we cannot tell.
            "partition": None if search_rows is None else ("search" if n <= search_rows else
                                                           "search+measure"),
            "search_partition_rows": search_rows,
            "held_out": False if search_rows is not None else None,
            "note": ("These are SELECTION-SET figures: the search chose its winner by scoring these "
                     "same rows, so they are the maximum of N draws rather than a measurement. For "
                     "a publishable number, score the measure partition with "
                     "`senbonzakura score --skip <track.json skip_harmful>`, or "
                     "`senbonzakura margin --skip-harmful <same>` for the harm-discrimination "
                     "figure."),
        }

    def _save_weights(self):
        """Write the baked model, and survive the one failure mode that costs the most.

        The save is the crash-prone step and it runs when every expensive thing is already
        done: the search, the bake and the post-bake measurement. A traceback here has, twice,
        meant hours of GPU time producing nothing an operator could use, because the run died
        without saying that `best-config.json` had already been written and makes the whole
        thing re-bakeable in minutes.

        So: one retry at smaller shards when the failure looks like running out of room, and
        an exit that names the recovery either way. The retry is narrow on purpose. Peak disk
        during a write is the finished shards plus the one being assembled, so smaller shards
        genuinely lower the high-water mark, and the same holds for the host-RAM buffer; a
        retry at the SAME size would just fail again more slowly.
        """
        args, log = self.args, self.log

        def _write(shard_size):
            # 4 GB rather than the 5 GB default even on the first attempt: a failed write loses
            # less, and nothing downstream cares how many shards there are.
            self.model.save_pretrained(args.out, safe_serialization=True, max_shard_size=shard_size)
            self.tok.save_pretrained(args.out)

        try:
            _write("4GB")
        except Exception as first:
            free = free_bytes_for(args.out)
            # cuda_free_total returns None off a cuda device or when the query fails; it is
            # context for the report, so an unmeasurable card is reported as unmeasured
            # rather than allowed to mask the real error.
            vram = cuda_free_total(str(self.dev))
            cuda_free = vram[0] if vram else None
            log(f"  SAVE FAILED: {first}")
            if not is_space_exhaustion(first):
                # Not a space problem, so a smaller shard changes nothing and retrying would
                # only delay the report.
                raise SystemExit(save_failure_report(
                    first, args.out, free_bytes=free, cuda_free=cuda_free)) from first

            log(f"  the failure looks like exhausted space, so retrying once at "
                f"{RETRY_SHARD_SIZE} shards")
            # Everything the first attempt allocated is still held until this runs, and a
            # partial shard from the failed write is dead weight the retry does not need.
            gc.collect()
            if str(self.dev).startswith("cuda") and torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
            try:
                _write(RETRY_SHARD_SIZE)
            except Exception as second:
                raise SystemExit(save_failure_report(
                    second, args.out, free_bytes=free_bytes_for(args.out),
                    cuda_free=cuda_free, retried=True)) from second
            log(f"  the retry at {RETRY_SHARD_SIZE} shards succeeded")
        else:
            return

    @contextlib.contextmanager
    def _study_storage_scope(self):
        """Hold the Optuna study storage for one run, and release its pool on the way out.

        Optuna's RDBStorage keeps a SQLAlchemy connection pool that nothing disposes, so a
        sqlite-backed study leaves roughly two open database handles behind. Today that
        costs a warning and nothing else: one study per process, and the handles go when
        the process does. It stops being free the moment anything copies or uploads the
        study database inside the same process, which is what --upload-to will do.

        `engine.dispose()` is the whole fix, measured. `remove_session()` looks like it
        should help and changes nothing, so a fix written around it would read as correct
        and leak exactly as before.
        """
        self._study_storage = None
        try:
            yield
        finally:
            storage, self._study_storage = self._study_storage, None
            if storage is not None:
                storage.engine.dispose()

    def run(self):
        # A scope rather than a try/finally around the search: the study is read as far as
        # the knee selection, 170 lines later, so a `with` there would reindent the whole
        # search. Wrapping the call keeps disposal deterministic on every path, including
        # the SystemExit an all-trials-failed search raises.
        with self._study_storage_scope():
            return self._run()

    def _run(self):
        """The pipeline, as four named phases with three diagnostic exits between the first two.

        It was one 290-line method, and the cost was specific rather than aesthetic: the phases
        share mutable state through `self`, so reading the knee selection meant scrolling back
        170 lines to learn what `self.bad_eval` held and whether the weights were pristine at
        that point. Each phase is named now and what crosses a boundary is an argument or a
        return value.

        The exits sit after preparation because each one wants the directions and the eval sets
        and no search: `--inspect` shows what an ablation does to real generations, `--bench-only`
        probes one fixed window, and `--bake-config` re-bakes a saved winner, which turns a
        crashed save into minutes of work rather than a repeat of the whole search.
        """
        args = self.args
        TR = args.track
        base_ref = self._prepare_evals(TR)
        if args.inspect is not None:
            return self._inspect()
        if args.bench_only:
            return self._bench_only()
        if args.bake_config:
            return self._bake_saved_config(base_ref)
        fixed = self._method_profile()
        if fixed is not None:
            # A recipe that pins the ablation has nothing to search for. Going through the search
            # anyway and then ignoring its answer would burn the GPU time the recipe exists to
            # save, and would put a study on disk describing a run that did not use it.
            from .crashsafe import config_to_bake_args
            bpr, b_K, b_mode, b_di = config_to_bake_args(fixed)
            self.log(f"method {args.method}: baking a fixed profile, no search")
            return self._bake_and_save(bpr, b_K, b_mode, b_di, base_ref)
        study, db = self._run_search(TR)
        bpr, b_K, b_mode, b_di = self._select_knee(study, db, TR)
        return self._bake_and_save(bpr, b_K, b_mode, b_di, base_ref)

    def _prepare_evals(self, TR):
        """Extract the directions, build the eval sets, and prove the run can finish.

        Returns the baseline refusal rate, which every later phase reports against. Leaves
        `self.bad_eval`, `self.kl_eval` and `self.orig_lp` set and the weights snapshotted, so
        the search can restore a pristine model between trials.
        """
        args, log = self.args, self.log
        # Before the search spends a GPU on it. Every trial's objective is scored with
        # this ruler, so one that misreads does not fail, it optimises toward the wrong
        # configuration and reports a confident number for it.
        log(f"ruler self-check: {validate_ruler()} cases pass")
        # A track records where its partitions end; every dataset here is read as a head of
        # N rows, so a flag larger than a partition walks straight into the next one, and the
        # failure is silent: the run succeeds and reports a number selected on the rows it
        # claims to have held out.
        #
        # Here rather than in main(), for a reason worth keeping: `kageyoshi` resolves its own
        # budget from the loaded model's size and OVERWRITES eval_refusal_final, so a check
        # sitting before that would inspect the parser default (0) and pass every time. The
        # model is loaded by this point, which is later than a preflight should be, but it is
        # the first moment the values being checked are the values that will be used.
        manifest = read_manifest(TR)
        if manifest:
            bad_flags = flag_violations(
                manifest, eval_refusal=args.eval_refusal,
                eval_refusal_final=args.eval_refusal_final,
                dir_prompts=args.dir_prompts, eval_kl=args.eval_kl)
            if bad_flags:
                raise SystemExit("these flags would read past the boundaries "
                                 f"{TR}/track.json records:\n"
                                 + "\n".join(f"  {b}" for b in bad_flags))
            log(f"track boundaries: {manifest['counts']}")
        GOOD_DS = args.good_ds or f"{TR}/good_ds"
        clean_src = args.clean_ds or GOOD_DS
        self.extract_directions(f"{TR}/bad_ds", GOOD_DS, args.hedge_ds, clean_src)

        # eval sets: refusals (generation) on the bad-eval set, KL vs original on a harmless set that
        # is DISJOINT from the direction-extraction prompts (the first dir_prompts of good), so
        # coherence is measured on prompts the directions were not fit on. Falls back to the harmless
        # tail (with a warning) only when the dataset is too small to spare a disjoint slice.
        self.bad_eval = self.load(f"{TR}/bad_eval_ds", args.eval_refusal)
        self.kl_eval = kl_eval_slice(
            self.load(GOOD_DS, args.dir_prompts + args.eval_kl),
            args.dir_prompts, args.eval_kl,
            lambda m: log(f"  note: {GOOD_DS} {m}"))

        log("caching original first-token distribution (KL reference) + baseline refusals")
        self.orig_lp = self.first_token_logprobs(self.kl_eval)
        base_ref = self.refusal_rate(self.bad_eval)
        log(f"BASELINE refusals: {base_ref*100:.1f}%  on {len(self.bad_eval)} bad-eval prompts")

        # Disk, before the search rather than after it. The save is the last thing a run does
        # and the most expensive thing to lose: a 57 GB base plus a 61 GB output on a 120 GB
        # volume died partway through writing shards, hours in. Checking here costs one syscall.
        self.preflight_disk()
        self.preflight_writable()

        # pristine copy taken now, on the untouched model; enables reversible search/inspect/bench
        self.snapshot_weights()
        return base_ref

    def _inspect(self):
        """`--inspect`: what one ablation window does to real generations, before and after.

        A KL number cannot tell "wrecked" from "a few benign first tokens flipped", so this
        prints the text and reverts. Nothing is searched and nothing is saved.
        """
        args, log, NL = self.args, self.log, self.NL
        # Eyeball what the ablation actually does to real generations: harmful (should
        # comply after) and harmless (should stay coherent). The KL number alone hides
        # whether high KL = "wrecked" or just "a few benign first-tokens flipped".
        ilayer = int(args.inspect[0]); istr = float(args.inspect[1]); n = args.inspect_n
        iD = max(2, NL // 4)
        hprompts = self.bad_eval[:n]; gprompts = self.kl_eval[:n]
        log(f"INSPECT window P={ilayer} wmax={istr} wmin=0 D={iD}: {n} harmful + {n} harmless, pre vs post")
        pre_h = self.gen_batch(hprompts); pre_g = self.gen_batch(gprompts)
        self.bake(ilayer, istr, 0.0, iD)
        post_h = self.gen_batch(hprompts); post_g = self.gen_batch(gprompts)
        kl = self.kl_vs_orig(self.kl_eval); self.restore_weights()
        def show(tag, prompts, pre, post):
            for p, a, b in zip(prompts, pre, post, strict=True):
                print(f"\n### {tag}: {p[:110].strip()}")
                print(f"  PRE : {a[:220].strip()!r}")
                print(f"  POST: {b[:220].strip()!r}")
        show("HARMFUL", hprompts, pre_h, post_h)
        show("HARMLESS", gprompts, pre_g, post_g)
        def pct(xs, f):
            return 100 * sum(f(t) for t in xs) / max(1, len(xs))
        log(f"INSPECT harmful:   refusals {pct(pre_h,is_refusal):.0f}%->{pct(post_h,is_refusal):.0f}%  "
            f"broken {pct(pre_h,is_broken):.0f}%->{pct(post_h,is_broken):.0f}%")
        log(f"INSPECT harmless:  refusals {pct(pre_g,is_refusal):.0f}%->{pct(post_g,is_refusal):.0f}%  "
            f"broken {pct(pre_g,is_broken):.0f}%->{pct(post_g,is_broken):.0f}%  first-token KL={kl:.4f}")
        log("INSPECT verdict: want harmful refusals DOWN with harmless broken≈0 and KL low")

    def _bench_only(self):
        """`--bench-only`: one fixed default window, measured then reverted. A probe, not a search."""
        log, NL = self.log, self.NL
        self.bake(int(NL*0.6), 1.0, 0.0, max(2, NL//4), K=self.KMAX)
        r = self.refusal_rate(self.bad_eval)
        k = self.kl_vs_orig(self.kl_eval)
        self.restore_weights()
        log(f"BENCH-ONLY default window (P={int(NL*0.6)}, wmax=1.0, K={self.KMAX}): "
            f"refusals={r*100:.1f}% KL={k:.4f}")

    def _capability_baseline(self):
        """The unedited model's score on the capability probe, and the items it was scored on.

        Returns (accuracy or None, items). Empty items means no probe was asked for, which is the
        default: this needs a graded benchmark the operator supplies, and inventing one would be
        worse than not measuring.

        Taken with the weights PRISTINE. The whole quantity of interest is a drop, and a drop
        needs a before; measuring the baseline after any bake would compare a damaged model to
        itself and report zero.
        """
        args = self.args
        spec = getattr(args, "capability_eval", "") or ""
        n = int(getattr(args, "capability_n", 0) or 0)
        if not spec or n <= 0:
            return None, []
        from . import dataset
        try:
            questions, answers = dataset.resolve_pairs(spec, token=args.hf_token or None)
        except dataset.DatasetError as e:
            raise SystemExit(
                f"--capability-eval {spec} cannot be read: {e}. A capability probe with no "
                f"benchmark behind it would silently score every candidate the same.") from e
        items = list(zip(questions[:n], answers[:n], strict=True))
        self.restore_weights()          # measure the model as it arrived, not as a trial left it
        return self._capability_score(items), items

    def _capability_score(self, items):
        """Accuracy of whatever is currently baked, on the probe items, or None if nothing graded.

        None rather than 0.0 when nothing could be graded, for the same reason it is None
        everywhere else: a run that measured nothing must not read as a run that measured a zero,
        and here that difference would move a selection.
        """
        if not items:
            return None
        from . import capability
        task = capability.get_task(getattr(self.args, "capability_task", "numeric"))
        prompts = [task.prompt.format(q) for q, _a in items]
        gens, truncated = capability.generate_with_truncation(
            self.model, self.tok, prompts, self.dev,
            batch=max(1, int(self.args.batch_size)),
            max_new=int(getattr(self.args, "capability_max_new", 320)))
        verdicts = capability.grade(gens, [a for _q, a in items], truncated,
                                    task=getattr(self.args, "capability_task", "numeric"))
        s = capability.summarise(verdicts)
        return s["accuracy"]

    def _capability_drop(self, baseline, items):
        """How much accuracy the currently baked config cost, or None when it cannot be said.

        None when there was no probe, no baseline, or nothing gradeable now. A candidate that
        cannot be scored must not be scored as unharmed: that would let a config which destroyed
        the model's ability to answer at all win on a drop of zero, which is the brokenness defect
        in a new costume.
        """
        if baseline is None or not items:
            return None
        now = self._capability_score(items)
        if now is None:
            # It answered nothing gradeable. That is the largest drop available, not an absent
            # measurement, because the baseline proved these items ARE gradeable on this model.
            return baseline
        return baseline - now

    def _method_profile(self):
        """The bake profile this run's method pins, or None when the method searches.

        Read here rather than in the parser so the recipe is applied at the one place that decides
        between searching and baking, and cannot be half-applied by a caller that built its own
        argument namespace.
        """
        from . import methods
        name = getattr(self.args, "method", methods.DEFAULT_METHOD)
        return methods.get(name).settings.get("bake_profile")

    def _bake_saved_config(self, base_ref):
        """`--bake-config`: bake a saved winner without searching for it again.

        The directions and eval sets are already prepared, so this reproduces the searched
        result. It exists because a crashed save used to cost the whole search a second time.
        """
        args, log = self.args, self.log
        with open(args.bake_config, encoding="utf-8") as f:
            cfg = json.load(f)
        bpr, b_K, b_mode, b_di = config_to_bake_args(cfg)
        log(f"direct bake from {args.bake_config} (skipping the search)")
        return self._bake_and_save(bpr, b_K, b_mode, b_di, base_ref)

    def _run_search(self, TR):
        """Run the Optuna search, and return the study with the path it persisted to.

        The db path comes back as well as the study because the failure message when no trial
        produced a measurement needs to tell the operator where the completed work is.
        """
        args, log = self.args, self.log
        lo, hi = self.lo, self.hi
        log(f"searching {args.trials} trials over layers [{lo},{hi}] ({args.search})")
        # NSGA-II needs a population to exert selection pressure; Optuna's default of 50 is far larger
        # than the small trial budgets here (a 60-trial run would be barely one generation), so pin the
        # population to a fraction of the budget so evolution actually happens instead of degenerating
        # to random sampling.
        pop = max(4, min(50, args.trials // 4))
        storage = None
        study_name = f"senbon-{args.search}"
        db = study_db_path(args.study_db, args.no_persist_study, TR, args.out)
        if db:
            # Built as an object rather than passed as a URL string so its connection pool
            # can be released; see _study_storage_scope.
            storage = optuna.storages.RDBStorage(f"sqlite:///{db}")
            self._study_storage = storage
            log(f"persistent study at {db} ({'resuming' if args.resume else 'fresh'}); "
                f"a killed run resumes with --resume (or --no-persist-study to disable)")
        if args.search == "pareto":
            study = optuna.create_study(
                directions=["minimize", "minimize", "minimize"],
                sampler=optuna.samplers.NSGAIISampler(seed=args.seed, population_size=pop),
                storage=storage, study_name=study_name, load_if_exists=args.resume)
        else:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=12),
                storage=storage, study_name=study_name, load_if_exists=args.resume)

        # --resume on a study that already finished its search (ran the budget or early-stopped)
        # must skip straight to bake+save, not re-search it. Without this, resume re-runs the whole
        # search on a completed study (the trap that wasted ~34 min of GPU on the first large run).
        skip_search = args.resume and search_already_done(study.user_attrs)
        if skip_search:
            log("resumed study already finished its search; skipping to bake + save")

        # `--trials` is a BUDGET, not a per-invocation quota. Optuna's `n_trials` counts only the
        # trials THIS call runs, so a resumed search that had already spent 168 of its 200 ran 200
        # more and finished at 368 while every artefact still said 200. In a comparison whose whole
        # claim is a matched budget, that silently hands one arm most of an extra run, so the
        # remainder is worked out here. Trials that failed still spent their time, so they count.
        trial_budget = args.trials
        if args.resume and not skip_search:
            # Only trials that FINISHED count against the budget. A study interrupted mid-trial
            # leaves that trial RUNNING for ever (nothing ever revisits it), and charging the arm
            # for a trial that produced no result would leave it a trial short of the tool it is
            # being compared against. A failed trial did produce a result, so it counts.
            spent = len(study.get_trials(deepcopy=False, states=(
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.FAIL,
                optuna.trial.TrialState.PRUNED)))
            trial_budget = remaining_budget(args.trials, spent, args.resume)
            if spent:
                log(f"resume: {spent} of {args.trials} trials already spent; running {trial_budget} more")
                if trial_budget == 0:
                    log("the trial budget is already spent; skipping to bake + save")
                    skip_search = True

        # Early stop (lever 4): stop once the best scalarised score (non-compliance + 0.5*keyword under
        # the KL/broken guards) hasn't improved for --patience consecutive trials, on the theory that the
        # frontier is mapped and more sampling of the same space won't help.
        _stall = {"best": WORST_SCORE, "since": 0}
        def _patience_cb(study, trial):
            if args.patience <= 0:
                return
            s = _scalar_of(trial)
            if s < _stall["best"] - 1e-4:
                _stall["best"] = s; _stall["since"] = 0
            else:
                _stall["since"] += 1
                if _stall["since"] >= args.patience:
                    log(f"early stop: no improvement in {args.patience} trials (best scalar {_stall['best']:.4f})")
                    study.stop()

        # Per-trial progress with an ETA that discounts any time the governor spent paused for VRAM,
        # so the estimate stays honest even when a game is opened mid-run.
        progress = SearchProgress(trial_budget, log, governor=self.gov)
        def _progress_cb(study, trial):
            progress.tick()

        # A single flaky trial (a transient CUDA OOM on an unlucky batch, say) must not kill the whole
        # search; catch it so Optuna marks that trial failed and moves on. Whatever the outcome, the
        # last trial's bake is left applied, so restore to pristine before any downstream read/mutate.
        if not skip_search:
            if args.warm_start and not study.get_trials(deepcopy=False):
                # Warm-start: enqueue one known-decent config (mid-late window, full projection, single
                # direction) so the search starts from a good point rather than cold random sampling.
                # Optuna samples any params we leave unspecified. Fresh study only (a --resume already
                # has trials); wrapped so a param-range mismatch degrades to a cold search, not a crash.
                pos = min(max(int(self.NL * 0.6), self.lo), self.hi)
                dist = max(2, self.NL // 4)
                # The warm start must be INSIDE the search space. With the budget pinned above 1
                # a seed of 1 is out of range, and Optuna's mismatch handling degrades the whole
                # search to cold random sampling: a quiet loss of the good starting point rather
                # than an error.
                seed = {"num_directions": self.KMIN, "dir_mode": "per_layer"}
                if args.per_component:
                    seed.update({"o_max_weight_position": pos, "o_max_weight": 1.0,
                                 "o_min_weight": 0.0, "o_min_weight_distance": dist,
                                 "d_max_weight_position": pos, "d_max_weight": 1.0,
                                 "d_min_weight": 0.0, "d_min_weight_distance": dist})
                else:
                    seed.update({"max_weight_position": pos, "max_weight": 1.0,
                                 "min_weight": 0.0, "min_weight_distance": dist})
                try:
                    study.enqueue_trial(seed, skip_if_exists=True)
                    # THE VALUE, not a constant. This line hardcoded "K=1" while the seed itself
                    # used the pinned budget, so on a K=2 arm it reported a warm start outside the
                    # search space that had in fact been seeded correctly. It is the exact line
                    # somebody reads to check that very thing, which makes a wrong one worse than
                    # none: found in the rehearsal of 2026-08-12, one arm after the ceiling bug it
                    # was written to guard against.
                    log(f"warm-start: seeded the search with a diff-of-means config "
                        f"(P={pos}, wmax=1.0, K={seed['num_directions']})")
                except Exception as e:
                    log(f"warm-start seed skipped ({e}); searching cold")
            try:
                study.optimize(self.objective, n_trials=trial_budget, callbacks=[_patience_cb, _progress_cb],
                               catch=(RuntimeError,))
            finally:
                self.restore_weights()
            # Mark the search finished (budget spent or early-stopped) so a later --resume skips it
            # instead of re-searching. Persisted with the study, so it survives a crash after the search.
            study.set_user_attr("search_done", True)
        self.trials_ran = len(study.get_trials(deepcopy=False, states=(
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED)))
        return study, db

    def _select_knee(self, study, db, TR):
        """Pick the most-uncensored configuration that is still intact, and return its profile.

        Reads the MEASURED user attributes rather than Optuna's own best trial, which is
        undefined for a multi-objective study, so one code path serves both search modes.
        """
        args, log = self.args, self.log

        def _row(t):
            pr = _profiles_from_params(t.params)
            return {"oP": pr[0], "owmax": round(pr[1], 3), "oD": pr[3],
                    "dP": pr[4], "dwmax": round(pr[5], 3), "dD": pr[7],
                    "K": t.params.get("num_directions", 1), "mode": t.params.get("dir_mode", "per_layer"),
                    "di": (round(t.params["direction_index"], 2) if "direction_index" in t.params else None),
                    "refusals": round(t.user_attrs["refusals"], 4),
                    "heretic": round(t.user_attrs.get("heretic", 0.0), 4),
                    "kl": round(t.user_attrs["kl"], 4), "broken": round(t.user_attrs.get("broken", 0.0), 4)}

        rows = sorted([_row(t) for t in study.trials if t.user_attrs], key=lambda r: (r["refusals"], r["kl"]))
        # Beside the run's other artefacts, not inside the corpus. The trial table is something
        # this run produced; the track is something it was given. Writing it into the track ruled
        # out a read-only corpus, which the benchmark mounts, and let two runs over one track
        # overwrite each other's table with nothing said.
        with atomic_write(f"{self.args.out}/trials.json") as f:
            json.dump(rows, f, indent=2)
        _ceil_for_log = args.max_kl if args.max_kl is not None else KL_CEIL
        log(f"frontier (lowest refusals first, intact = KL <= {_ceil_for_log} AND broken≈0):")
        for r in [x for x in rows if x["kl"] <= _ceil_for_log and x["broken"] <= 0.1][:8]:
            di_s = "" if r["di"] is None else f" di={r['di']}"
            log(f"   o(P={r['oP']},wmax={r['owmax']},D={r['oD']}) d(P={r['dP']},wmax={r['dwmax']},D={r['dD']}) "
                f"K={r['K']} {r['mode']}{di_s} refusals={r['refusals']*100:.1f}% heretic={r['heretic']*100:.1f}% "
                f"broken={r['broken']*100:.0f}% KL={r['kl']:.4f}")

        if args.search == "pareto":
            log(f"Pareto front ({len(study.best_trials)} non-dominated configs, refusals vs KL):")
            for t in sorted(study.best_trials, key=lambda t: t.user_attrs["refusals"])[:10]:
                log(f"   refusals={t.user_attrs['refusals']*100:.1f}% broken={t.user_attrs.get('broken',0)*100:.0f}% "
                    f"KL={t.user_attrs['kl']:.4f}")

        # Pick the KNEE: the most-uncensored config that is still intact (KL under ceiling, coherent).
        # Reads the MEASURED user_attrs, so it works for both search modes (and for multi-objective,
        # where study.best_trial is undefined). Fall back to the whole set if nothing is under ceiling.
        # Preference now weighs hedging (soft) and the Heretic keyword rate, not hard refusals alone.
        _cand = [t for t in study.trials if t.user_attrs]
        if not _cand:
            # min() on an empty sequence raises "min() arg is an empty sequence", which after
            # several hours of rented GPU tells the operator nothing about what went wrong.
            failed = sum(1 for t in study.trials if str(getattr(t, "state", "")).endswith("FAIL"))
            where = f" The study is at {db}." if db else ""
            raise SystemExit(
                f"the search produced no usable trial: {len(study.trials)} ran, {failed} failed, and "
                f"none recorded measurements, so there is no configuration to bake. Every trial "
                f"raised before it could be scored; the usual causes are running out of VRAM during "
                f"generation, a model that will not generate at all, or an eval set that loaded "
                f"empty.{where} Fix the cause and re-run with --resume to keep the completed work, "
                f"or use --bake-config to save a known configuration without searching.")
        # THE CEILING IS THE CALLER'S IF THEY SET ONE. `--max-kl` is not advisory: a run asked to
        # stay under a drift and then handed a model above it has answered a different question.
        ceiling = args.max_kl if args.max_kl is not None else KL_CEIL
        surcharge_from = args.max_kl if args.max_kl is not None else KL_TARGET
        _intact = [t for t in _cand
                   if t.user_attrs["kl"] <= ceiling and t.user_attrs.get("broken", 0.0) <= 0.1]
        if args.max_kl is not None and not _intact:
            best_kl = min(t.user_attrs["kl"] for t in _cand)
            raise SystemExit(
                f"no configuration met --max-kl {args.max_kl}: of {len(_cand)} scored trials the "
                f"least drift any of them achieved was {best_kl:.4f}, and every trial under the "
                f"limit was incoherent. Nothing is baked, because a model above the ceiling you "
                f"asked for is not the thing you asked for. Either raise --max-kl (a value near "
                f"{best_kl:.3f} is reachable on this model), search longer with more --trials, or "
                f"re-run with --resume to keep this study and widen the limit.")

        # Re-score the top candidates on a LARGER bad-eval before choosing, so the knee isn't overfit to
        # the small search eval (lever 5). Held in a local dict; the trials themselves aren't mutated.
        _final = {}
        _pool = _intact or _cand
        if not _intact:
            # Reachable only without --max-kl, since the branch above exits when one was asked for.
            # It used to happen in silence, and a silent fallback here is the worst kind: every
            # remaining candidate is over the drift ceiling or wrecked, and the run still prints a
            # BEST line that reads exactly like a healthy one. The knee scalar now carries a
            # brokenness term so the pick among them is at least the least damaged, but the honest
            # statement is that there was nothing intact to choose from.
            worst = min(t.user_attrs["kl"] for t in _cand) if _cand else float("nan")
            log(f"  WARNING: none of {len(_cand)} scored trials was intact (KL <= {ceiling} AND "
                f"broken <= 0.1), so the winner is chosen from DAMAGED configurations. The least "
                f"drift any trial achieved was {worst:.4f}. Treat the saved model as a search "
                f"artefact, not a result, and re-run with more --trials or a wider --max-kl.")
        # The capability probe, and where it sits is the whole design decision. Running it per
        # TRIAL would multiply a 200-trial search by a graded benchmark; running it here costs
        # --top-rescore generations, on exactly the candidates that could still win. The
        # catastrophic case (a model that stopped answering) is already caught per trial by the
        # brokenness term; this catches the quieter one, a model that answers fluently and can no
        # longer reason, which nothing in the objective could see.
        cap_baseline, cap_items = self._capability_baseline()
        if cap_items:
            log(f"capability probe: {len(cap_items)} items, baseline accuracy "
                f"{cap_baseline if cap_baseline is not None else 'not gradeable'}")
        if args.eval_refusal_final and args.eval_refusal_final > len(self.bad_eval):
            big = self.load(f"{TR}/bad_eval_ds", args.eval_refusal_final)
            ranked = sorted(_pool, key=_scalar_of)[:max(1, args.top_rescore)]
            log(f"re-scoring top {len(ranked)} candidates on {len(big)} bad-eval prompts (lever 5)")
            for t in ranked:
                pr = _profiles_from_params(t.params)
                K = t.params.get("num_directions", 1); mode = t.params.get("dir_mode", "per_layer")
                di = t.params.get("direction_index")
                self.restore_weights(); self.bake_pc(*pr, K, mode, di)
                g = self.gen_batch(big)
                r = sum(is_refusal(x) for x in g) / max(1, len(g))
                s = sum(is_soft_refusal(x) for x in g) / max(1, len(g))
                h = heretic_keyword_rate(g)
                # Re-measured for the same reason as the other three: this pass exists because the
                # knee should not be overfit to the small search eval, and until 2026-09-06 it
                # refreshed every axis EXCEPT this one. A config whose output only falls apart on
                # the larger eval kept its stale, low brokenness and could win on a refusal rate
                # that had dropped because the model had stopped answering rather than started
                # complying. The harmless side is not re-measured here (it is not what --eval-
                # refusal-final grows) so the trial's value is kept as a floor.
                b = max(broken_rate(g), t.user_attrs.get("broken", 0.0))
                drop = self._capability_drop(cap_baseline, cap_items)
                _final[t.number] = {"refusals": r, "soft": s, "heretic": h, "broken": b,
                                    "capability_drop": drop}
                cap_note = "" if drop is None else f" capability={-drop*100:+.1f}pp"
                log(f"   trial {t.number}: refusals={r*100:.1f}% soft={s*100:.1f}% "
                    f"heretic={h*100:.1f}% broken={b*100:.0f}% KL={t.user_attrs['kl']:.4f}"
                    f"{cap_note}")
            self.restore_weights()
            _pool = ranked

        def _knee_key(t):
            # P2 fix: minimise a WEIGHTED scalar of the three objectives so the Heretic keyword /
            # hedging axis actually STEERS the pick, instead of the old lexicographic tuple where it
            # was a mere tiebreaker behind hard+soft refusal and so almost never decided the winner
            # (the whole point of making it a first-class search objective was defeated at selection
            # time). KL only surcharges above the "comfortably intact" target.
            f = _final.get(t.number)
            ua = t.user_attrs
            ref = f["refusals"] if f else ua["refusals"]
            soft = f["soft"] if f else ua.get("soft", 0.0)
            her = f["heretic"] if f else ua.get("heretic", 0.0)
            brk = f["broken"] if f else ua.get("broken", 0.0)
            # None means nobody measured it, which must score as no penalty rather than as no
            # loss: those are the same number here and only one of them is a claim.
            drop = (f or {}).get("capability_drop") or 0.0
            return knee_scalar(ref, soft, her, ua["kl"], surcharge_from, broken=brk,
                               capability_drop=drop)

        best = min(_pool, key=_knee_key)
        bp = best.params
        b_K = bp.get("num_directions", 1); b_mode = bp.get("dir_mode", "per_layer")
        b_di = bp.get("direction_index")
        bpr = _profiles_from_params(bp)
        log(f"BEST: o(P={bpr[0]},wmax={bpr[1]:.3f},wmin={bpr[2]:.3f},D={bpr[3]}) "
            f"d(P={bpr[4]},wmax={bpr[5]:.3f},wmin={bpr[6]:.3f},D={bpr[7]}) K={b_K} mode={b_mode}"
            f"{'' if b_di is None else f' di={b_di:.2f}'} refusals={best.user_attrs['refusals']*100:.1f}% "
            f"heretic={best.user_attrs.get('heretic',0)*100:.1f}% "
            f"broken={best.user_attrs.get('broken',0)*100:.0f}% KL={best.user_attrs['kl']:.4f}")

        return bpr, b_K, b_mode, b_di

    def _bake_and_save(self, bpr, b_K, b_mode, b_di, base_ref):
        """Bake the winning config into the weights and save. Extracted from run() so a
        --bake-config recovery can save a known config WITHOUT re-searching. Writes
        best-config.json BEFORE the (crash-prone) save, so even a failed save leaves a
        re-bakeable artefact and a lost save becomes a minutes-long re-bake, not a re-search.
        """
        args, log = self.args, self.log
        # Beside the model it describes, not inside the corpus. `--bake-config` takes an explicit
        # path so nothing depended on the old location, and the old location made a run over a
        # read-only track impossible while overwriting the previous run's winner in place.
        with atomic_write(f"{args.out}/best-config.json") as f:
            json.dump(winning_config(bpr, b_K, b_mode, b_di), f, indent=2)
        log(f"wrote winning config to {args.out}/best-config.json (re-bakeable with --bake-config)")

        # ── BAKE the winner into the weights + save ──────────────────────────────────
        # The search left the LAST trial's bake applied; restore to pristine, then bake the winner.
        # Because the search scored this exact operation, POST-BAKE should reproduce the best trial's
        # numbers (that equality is the check that search and bake are unified, no proxy gap).
        self.restore_weights()
        log("baking best config (per-component windowed, per-config directions, norm-preserving)")
        self.bake_pc(*bpr, b_K, b_mode, b_di)
        # POST-BAKE re-measure on the actual weights; should match the best trial (same operation).
        post_gens = self.gen_batch(self.bad_eval)
        post_ref = sum(is_refusal(t) for t in post_gens) / max(1, len(post_gens))
        post_heretic = heretic_keyword_rate(post_gens)
        post_kl = self.kl_vs_orig(self.kl_eval)
        post_brk = broken_rate(self.gen_batch(self.kl_eval[:min(16, len(self.kl_eval))]))
        log(f"POST-BAKE (weights, no hooks): refusals={post_ref*100:.1f}% heretic={post_heretic*100:.1f}% "
            f"broken={post_brk*100:.0f}% KL={post_kl:.4f}")

        self.free_before_save()
        log(f"saving to {args.out}")
        self.events.emit("phase", phase="phase_saving")
        self._save_weights()
        # WHAT THIS CHECKPOINT IS, written where copying one file out of the directory cannot
        # shed it. Strict for a partial ablation and best-effort for a whole one: the first is a
        # model that must never pass for the second, and the second losing a provenance line is
        # not worth killing a save whose GPU work is already spent.
        marker.stamp(
            args.out,
            marker.fields(
                version=__version__, ablate_conv=self.ablate_conv,
                partial_layers=self.partial_layers, num_directions=b_K, dir_mode=b_mode,
                seed=args.seed,
                base_model=getattr(getattr(self.model, "config", None), "_name_or_path", None)),
            required=bool(self.partial_layers), log=log)
        with atomic_write(f"{args.out}/abliteration.json") as f:
            json.dump({"per_component": args.per_component,
                       "o_profile": {"max_weight_position": bpr[0], "max_weight": bpr[1],
                                     "min_weight": bpr[2], "min_weight_distance": bpr[3]},
                       "d_profile": {"max_weight_position": bpr[4], "max_weight": bpr[5],
                                     "min_weight": bpr[6], "min_weight_distance": bpr[7]},
                       "num_directions": b_K, "dir_mode": b_mode, "direction_index": b_di,
                       "max_directions": self.KMAX,
                       # WHETHER THIS MODEL IS A WHOLE ABLITERATION OR A CONTROL ARM. On a hybrid
                       # architecture the convolution blocks write the residual stream too, and a
                       # run told to leave them alone produces a model whose refusal behaviour is
                       # only partly removed. That is a legitimate arm of one experiment and an
                       # indefensible thing to publish unlabelled, so the flag and the layers it
                       # skipped are both recorded rather than inferred from the log.
                       "ablate_conv": self.ablate_conv,
                       "partially_ablated_layers": sorted(self.partial_layers),
                       "baseline_refusals": base_ref, "post_bake_refusals": post_ref,
                       "post_bake_heretic": post_heretic, "post_bake_broken": post_brk, "post_bake_kl": post_kl,
                       # WHAT THOSE REFUSAL FIGURES ARE, which the artefact could not previously say.
                       # They come from `bad_eval_ds`, whose head is the track's SELECTION partition:
                       # the rows the search scored 200 trials against. That is the correct set to
                       # search on and the wrong set to publish from, because the winner is the best
                       # of N draws over exactly these prompts. A run's 0.0% travelled as a measured
                       # refusal rate on 2026-08-16 because nothing here said otherwise.
                       "refusal_eval": self.eval_provenance(),
                       # HOW THEY WERE PRODUCED. Generation is greedy, so there is no sampling noise,
                       # but the batch floats with free VRAM and left-padding makes batch composition
                       # part of the numerics. A reader comparing two runs needs to know whether the
                       # machinery was pinned; `--no-throttle` pins it.
                       "generation": {"greedy": True, "max_new_tokens": args.gen_tokens,
                                      **self.gov.report()},
                       # The same disclosure for the capture pass, which the generation block does
                       # not cover and which has a stronger claim to it: these activations ARE the
                       # refusal direction. Batch composition enters their numerics exactly as it
                       # enters generation's, so a reader comparing two runs' directions needs to
                       # know whether the chunking was pinned here too.
                       "direction_capture": {"prompts_per_side": args.dir_prompts,
                                             **self.capture_gov.report()},
                       "sparsity": float(args.sparsity),
                       # Provenance: a score without the seed that produced it cannot be
                       # re-run, and cannot be told apart from a re-sample of the same config.
                       # WHAT THIS RESULT IS ABOUT. The record carried the seed, the search, the
                       # trial count and the whole dependency tree, and never once said which
                       # model it had edited. Every resume guard in the run specs asks
                       # `model=...` of this file and can only ever get a miss, so the "already
                       # complete" branch they all carry has never fired. `model` is the path as
                       # given, which inside a sealed container is a mount point rather than an
                       # identity; `model_id` is what the checkpoint calls itself, which is the
                       # part a stranger can look up.
                       "model": args.model,
                       "model_id": getattr(getattr(self.model, "config", None), "_name_or_path", None),
                       "seed": args.seed, "search": args.search, "trials": args.trials,
                       # `trials` is what was ASKED for; this is what the study actually holds.
                       # They came apart on 2026-08-06, when a resumed arm ran its full budget a
                       # second time and every artefact it wrote still reported the budget. An
                       # equal-budget claim that cannot be checked against the artefact is not a
                       # claim, so the count that settles it is recorded beside the request.
                       "trials_ran": getattr(self, "trials_ran", None),
                       "warm_start": args.warm_start, "good_orth": not args.no_good_orth,
                       "chat_template": getattr(self.tok, "senbon_chat_template", None),
                       # The K actually applied at each layer, which is not always the K asked
                       # for: the separation filter, the rank floor and a degenerate cloud can
                       # each reduce it, and num_directions alone cannot show that.
                       # Both views, because they answer different questions and the reader
                       # cannot reconstruct one from the other without knowing the convention.
                       "directions_per_layer": getattr(self, "dirs_per_layer", None),
                       "directions_per_position": getattr(self, "dirs_per_position", None),
                       "directions_index_note": (
                           "directions_per_layer[i] is decoder layer i. "
                           "directions_per_position[i] is the residual-stream position: 0 is the "
                           "embedding output, which is never ablated, and position i+1 is what "
                           "layer i writes into. A leading 0 in the position list is that choice, "
                           "not a layer that missed out. `axis_separations` uses the same "
                           "position indexing, so its entry i pairs with "
                           "directions_per_position[i]."),
                       # Why each layer got the count it did. A rejected axis whose separation
                       # sits just under the threshold means the constant chose the direction
                       # count; one far under it means the second direction is genuinely absent.
                       # The count alone cannot tell those apart, which is why it is recorded.
                       # A bounded sample per layer, plus the two exact totals over every axis
                       # measured. The sample is the evidence; the totals are the count it came from.
                       "axis_separations": getattr(self, "axis_separations", None),
                       # The statistic, its null and its threshold travel together. A threshold
                       # with no statistic beside it is unreadable the moment there is more than
                       # one, and a null is what turns the score into evidence rather than a
                       # number: 2.1 means nothing until something says what nothing scores.
                       "separation_statistic": getattr(self, "separation_statistic",
                                                       separation.DEFAULT_STATISTIC),
                       "separation_null": getattr(self, "separation_null",
                                                  separation.get(
                                                      separation.DEFAULT_STATISTIC).null),
                       "matched_scoring": getattr(self, "matched_scoring", False),
                       # null when the controls were drawn from the ordinary harmless set.
                       "matched_source": getattr(self, "matched_source", None),
                       # Which recipe produced this. Without it two arms of a comparison are two
                       # runs whose settings a reader has to diff by eye, which is not a
                       # comparison, and the artefact is the only place that survives the session.
                       "method": getattr(args, "method", "searched"),
                       # Whether the matching found anything. Near 1.0 means it did not, and a
                       # matched_scoring:true run with a quality near 1.0 produced unmatched
                       # numbers under a matched label.
                       "matching_quality": getattr(self, "matching_quality", None),
                       # About 1.0 is as near as the space allows; above about 2.0 the corpus
                       # holds little on the candidate's subject. The acceptance target for
                       # corpus work, where matching_quality is only an alarm.
                       "match_closeness": getattr(self, "match_closeness", None),
                       "axis_separation_threshold": separation.get(
                           getattr(self, "separation_statistic",
                                   separation.DEFAULT_STATISTIC)).threshold,
                       "axes_measured_total": getattr(self, "axes_measured_total", None),
                       "max_axis_separation": getattr(self, "max_axis_separation", None),
                       "best_rejected_separation": getattr(self, "best_rejected_separation", None),
                       "axes_rejected_total": getattr(self, "axes_rejected_total", None),
                       # The floor the threshold was measured against, and how much work it did.
                       # `axis_separation_threshold` alone records a constant that was chosen
                       # once and never checked; these record what a direction carrying nothing
                       # scored on the same rows, which is the only thing that makes the constant
                       # readable. Per layer as well as overall, because a floor that varies by
                       # layer and a floor that does not are different findings.
                       "null_separation_floor": getattr(self, "null_separation_floor", None),
                       "null_separation_floor_per_layer": getattr(self, "layer_null_floors", None),
                       "axes_rejected_by_null": getattr(self, "axes_rejected_by_null", None),
                       # Candidate directions are fitted on half the rows and scored on the other
                       # half. Recorded because every separation figure written before this was
                       # in-sample and therefore could not come out small.
                       "separation_held_out": True,
                       # How many layers actually got the hedging direction. It is gated on a
                       # free slot, so K=1 gets none of it and a K comparison would be confounded.
                       "hedge_applied_layers": getattr(self, "hedge_applied_layers", None),
                       "filter_is_unsatisfiable": getattr(self, "filter_is_unsatisfiable", None),
                       "provenance": provenance(device=self.dev,
                                                accelerator=accelerator_name(self.dev))},
                      f, indent=2)
        self.events.emit("done", out=str(args.out))
        self.events.close()
        log("DONE")


# The commands that live in sibling modules. Dispatched by name, and imported only when one is
# actually asked for: `margin` imports this module, so a module-level import here is circular.
from .entry import DELEGATED  # noqa: E402
from .entry import dispatch as _delegate  # noqa: E402

# The parser lives in a module that imports nothing heavy, so `--help`, `--version`
# and every argument error cost a hundredth of a second instead of 2.8. Re-exported
# because the tests, the help test and every caller already ask `cli` for it.
# RE-EXPORTED, and the noqa is load-bearing. `loader_parser` is not called in this module, so
# ruff's autofix removes it as unused, and `coherence`, `drift`, `score` and `margin` all do
# `from .cli import loader_parser`. Removing it breaks four commands at import time.
from .parser import build_parser, loader_parser, split_mode  # noqa: E402, F401


def main(argv=None, *, emit_banner=True):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Decoration only, and structurally unable to reach a result: it prints nothing unless
    # stdout is a terminal, so a redirected run, a spec's captured log and every `stdout-contains`
    # check see exactly what they saw before this existed. Imported here rather than at module
    # level to keep the package's import surface small.
    if emit_banner:
        from . import banner
        banner.emit(__version__, sys.stdout)

    # Subcommands, with abliteration as the default. `senbonzakura --model X --out Y` keeps
    # working exactly as before, because every run spec on record and every README example is
    # written that way, and a tool that renames its own entry point breaks the records of what
    # was already run.
    if argv and argv[0] in DELEGATED:
        return _delegate(argv[0])(argv[1:])

    # `kageyoshi` is a real subcommand now rather than an argv[0] trick: it runs the abliterator
    # with the auto-scaled best-effort preset, resolved after the model loads once the
    # architecture and parameter count are known. `abliterate` names the default explicitly.
    bankai, argv = split_mode(argv)
    args = build_parser().parse_args(argv)
    return run_parsed(args, bankai, argv)


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
    """
    d = Path(out)
    if not d.is_dir():
        return []
    found = [name for name in RUN_ARTEFACTS if (d / name).exists()]
    # Sharded weights are named per shard, so the exact filename is not knowable in advance.
    if any(d.glob("model-*-of-*.safetensors")):
        found.append("model-*.safetensors")
    return sorted(found)


def _preflight_output(args):
    """Refuse to write into a directory a previous run already used, unless told which to do.

    Checked here rather than at save time for the usual reason: save time is after the search, so
    discovering it there costs the whole run. `--resume` is the one flag that makes an occupied
    directory expected rather than a mistake, and `--bake-config` reads a config the previous run
    left, so both pass through.
    """
    if getattr(args, "resume", False) or getattr(args, "bake_config", None):
        return
    out = getattr(args, "out", None)
    if not out:
        return
    found = occupied_by(out)
    if not found:
        return
    raise SystemExit(
        f"--out {out} already holds a previous run:\n"
        + "\n".join(f"  {name}" for name in found)
        + f"\n\nThis is refused rather than warned about, because a run that writes over another "
          f"one fails quietly: some files are replaced, some are left, and the artefact ends up "
          f"describing two runs with no field that says so. This project has lost three results "
          f"to exactly that.\n\nPick one:\n"
          f"  --out <somewhere else>   keep both\n"
          f"  --resume                 continue the run that is already there\n"
          f"  delete {out} yourself    if you meant to start again")


def _preflight_datasets(args):
    """Prove every dataset the run needs can be read, BEFORE the model is downloaded.

    The run used to discover a missing track in `extract_directions`, which is after the weights
    are resident. On a rented pod that is a 61 GB download and a full load thrown away for a
    mistake that was visible from the command line: `--track` defaults to the relative directory
    `track`, so a run started anywhere else, or with `HF_HUB_OFFLINE=1` and no flag, dies on
    `could not fetch the Hub dataset 'track/bad_ds'` having already spent the expensive part.

    Each source is resolved with a read budget of one row rather than merely stat-ed, so this also
    catches a directory that exists and holds the wrong thing: an unreadable format, or a text
    column that is not there. Reading one row of a track is nothing beside loading a model.

    Reported ALL AT ONCE. A pre-flight that stops at the first fault turns one wasted start into
    three, which on rented hardware is the whole point of having one.
    """
    from . import dataset as _dataset

    track = getattr(args, "track", "track")
    good = getattr(args, "good_ds", None) or f"{track}/good_ds"
    # `clean_ds` defaults to whatever `good_ds` resolves to, so it is not listed separately unless
    # it was given: checking the same source twice would report one fault as two.
    required = [(f"{track}/bad_ds", "the harmful prompts directions are extracted from"),
                (good, "the harmless prompts directions are contrasted against"),
                (f"{track}/bad_eval_ds", "the held-out harmful prompts refusal is scored on")]
    optional = [(getattr(args, "hedge_ds", ""), "the hedged-compliance contrast (--hedge-ds)"),
                (getattr(args, "clean_ds", ""), "the clean contrast (--clean-ds)"),
                (getattr(args, "harmless_matched", ""),
                 "the matched control pool (--harmless-matched)")]
    def fault(spec, what):
        """The reason this source cannot be read, or None. A function rather than a try inside the
        loop so every source is attempted and the loop stays a plain comprehension.
        """
        try:
            _dataset.resolve(spec, text_column=getattr(args, "text_column", None) or None,
                             token=getattr(args, "hf_token", None) or None, limit=1,
                             what="prompt set")
        except _dataset.DatasetError as e:
            return f"  {spec}\n      ({what})\n      {e}"
        return None

    checked = required + [(s, w) for s, w in optional if s]
    faults = [f for f in (fault(s, w) for s, w in checked) if f]
    if not faults:
        return
    hint = ""
    if any(f.lstrip().startswith(track + "/") for f in faults):
        # The single most common cause, and the fix is one flag. The bundled track ships inside
        # the wheel, so it needs no network and no files on disk.
        hint = ("\n\nIf you have not built a track, the one bundled in this install needs neither "
                "a network nor a download:\n  --track default")
    raise SystemExit(
        f"{len(faults)} of the datasets this run needs cannot be read, and this is checked before "
        f"the model is downloaded so that finding out costs nothing:\n"
        + "\n".join(faults) + hint)


def _apply_method_args(args, log):
    """Pin the argument values this run's method fixes, and refuse to fight the command line.

    A recipe that only sets a bake profile can be applied where the bake happens. One that fixes
    how directions are EXTRACTED has to be applied before extraction, and the danger is different:
    the flag it pins is one the operator can also type.

    So a conflict is refused rather than resolved. Silently overriding an explicit
    `--max-directions 4` would produce an artefact recording a method whose defining property the
    operator believed they had turned off, which is the kind of result that gets withdrawn. A
    value that MATCHES what the method pins is not a conflict and passes without comment.
    """
    from . import methods

    pinned = methods.get(getattr(args, "method", methods.DEFAULT_METHOD)).settings.get("args")
    if not pinned:
        return
    defaults = build_parser()
    for name, value in sorted(pinned.items()):
        typed = getattr(args, name)
        if typed != value and typed != defaults.get_default(name):
            flag = "--" + name.replace("_", "-")
            raise SystemExit(
                f"--method {args.method} fixes {flag} at {value}, and this command line asks for "
                f"{typed}. That is the property the method is defined by, so it is refused rather "
                f"than overridden: a run recording one method and behaving as another is how a "
                f"published number gets withdrawn. Drop the method or drop the flag.")
        if typed != value:
            log(f"method {args.method} pins {name.replace('_', '-')} = {value}")
        setattr(args, name, value)


def run_parsed(args, bankai, argv):
    """Everything after parsing. Split out so the entry point can parse without importing torch.

    `entry.main` does the parse against the light parser and calls straight in here, so a person
    who typed a bad flag has already been told so before this module is imported at all.
    """
    if args.load_in_4bit:
        # The abliterator rewrites weights in place (the norm-preserving bake), which needs full
        # precision; 4-bit Params4bit can't be orthogonalised. Reject early with a clear pointer
        # rather than failing cryptically at bake time.
        raise SystemExit(
            "senbonzakura abliterates by rewriting weights, which needs full precision, so "
            "--load-in-4bit is not supported here. Use it with the scorer to measure a model on "
            "low VRAM: python -m senbonzakura.score --load-in-4bit --model <dir> --eval <ds> --out r.json")

    # A flag combination needs no model to check, so it is checked here rather than where it is
    # used. `extract_directions` runs after the weights are resident and after two residual passes,
    # and discovering an unusable command line there costs a download and a load on a rented card
    # for a mistake that was visible before either.
    if getattr(args, "harmless_matched", "") and not getattr(args, "matched_scoring", False):
        # Refused rather than ignored, and rather than silently switching matched scoring on. A
        # supplied corpus that goes unused is the failure where a reader believes the run answered
        # the matched question because they passed the matched set; enabling it for them would be
        # the mirror image, changing what every separation number in the artefact means on the
        # strength of an inferred intention.
        raise SystemExit(_unused_matched_corpus(args.harmless_matched))

    # Fail loud on an incompatible torch BEFORE the model download (a fused-MoE class imports
    # torch.distributed.tensor.DTensor, torch >= 2.5); otherwise the run dies only after pulling
    # tens of GB. The runpod pytorch 2.4 image tripped exactly this.
    if not torch_version_ok(torch.__version__):
        raise SystemExit(
            f"senbonzakura needs torch >= {MIN_TORCH[0]}.{MIN_TORCH[1]} (the transformers MoE path "
            f"imports torch.distributed.tensor.DTensor); found {torch.__version__}. "
            f"Install a compatible build, e.g. torch==2.5.1.")

    # After the torch check and before the model. The torch check is instant and local, and an
    # unusable interpreter makes every other fault moot, so it goes first; this one may touch the
    # network for a Hub track and is still nothing beside pulling a model.
    _preflight_output(args)
    _preflight_datasets(args)

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    # Before the model, because the conflict it can raise is visible from the command line alone
    # and finding it after a download costs a rented card an hour.
    _apply_method_args(args, log)

    abl = Abliterator(args, log)
    if bankai:
        _apply_kageyoshi(args, abl.model, abl.arch, abl.ne, abl.NL, log,
                         explicit=_kageyoshi_explicit(argv))
    return abl.run()


if __name__ == "__main__":
    main()
