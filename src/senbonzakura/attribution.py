# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Which blocks of a model carry a behaviour, measured by taking each one's contribution away.

THE QUANTITY. Take the UNEDITED model. For one block, remove from that block's write to the
residual stream the part that lies along a fixed direction `d`, at every token position, and let
every later block run freely. The effect of the block is how far the behaviour readout falls on
prompts that show the behaviour, minus how far it falls on prompts that don't (so damage from
tampering with a block is not mistaken for the behaviour). The same is done to every block at once,
which gives the total `T`, and to growing prefixes and suffixes of blocks.

THE DISTINCTION THIS MODULE EXISTS TO KEEP. Where the abliteration tool WROTE is a record of its own
choices and is not evidence of where the behaviour lived. Nothing here reads the edit's footprint:
no import from the abliterator, no layer profile, no search result. `tests/test_attribution_firewall.py`
fails if that changes. The direction `d` is the vehicle of the intervention; the readout is the
behaviour. They are different objects and a result is stated about the readout.

A NEGATIVE ANSWER IS AN ANSWER. `classify` returns one of seven statuses and the per block numbers
only travel inside one. `tint_allowed` is computed in exactly one place, `tint_allowed`, and is true
only for `localised` with a predictive check that held up. `direction_inert` (the direction cannot
see the behaviour) is deliberately not `diffuse` (the behaviour is spread out): conflating them
would let a decoy or a many dimensional behaviour read as a humble "spread out".

WHAT IT DOES NOT CLAIM. It does not claim the behaviour is one dimensional (it is relative to `d`).
It does not claim sufficiency (a block that is needed is not thereby a block that is enough). It
cannot see redundant circuits through single block ablation, and says `redundant` when the sum of
the single effects falls well short of the total. It is whole decoder blocks only.

THE DECISION CONVENTIONS are in `Conventions`, are declared before any result and are recorded in the
artefact. They are conventions and not measured thresholds, and the output says so.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass

INSTRUMENT = "block-attribution"
INSTRUMENT_VERSION = "1"
ESTIMATOR = "ablate-direction-from-block-write"
UNITS = "readout units; share of the total effect where stated"

LOCALISED, DIFFUSE, REDUNDANT = "localised", "diffuse", "redundant"
ABSENT, DIRECTION_INERT, READOUT_INVALID, UNRESOLVED = (
    "absent", "direction_inert", "readout_invalid", "unresolved")
STATUSES = (ABSENT, READOUT_INVALID, UNRESOLVED, DIRECTION_INERT, REDUNDANT, DIFFUSE, LOCALISED)

#: What this instrument does not do, written into every artefact so a reader meets it.
LIMITS = (
    (
        "Whole decoder blocks only. Sub-block attribution (attention against MLP against convolution) "
        "is deferred, because the normalisation between a sub-layer and the residual stream makes it "
        "easy to measure the wrong thing."
    ),
    (
        "Relative to one extracted direction. A behaviour that runs through other directions reports "
        "direction_inert, which says nothing about whether it is localised."
    ),
    "A block that is needed is not thereby a block that is enough.",
    (
        "A first token readout can end up measuring whether the model opens with a compliance "
        "opener. It then predicts refusal well because refusal is mostly not complying, which is "
        "detecting the absence of compliance and a weaker claim than detecting refusal. Whether "
        "that happened on this run is in controls.readout_agreement_detail (the share of fit "
        "refusals the refusal set covers, and the openers that fall in neither set)."
    ),
    "Single block ablation cannot identify blocks that back one another up; redundant says so.",
    (
        "The readout is a first token proxy for the behaviour, validated against generation on a "
        "separate partition; it is not generation."
    ),
    (
        "One model, one precision, one prompt format. Artefacts that differ in any of these are not "
        "comparable."
    ),
    (
        "The planted test models switch off block to block dependence on purpose, and real models "
        "have it. This instrument measures the TOTAL effect of removing the direction from a "
        "block's write, indirect routes through later blocks included, so a block can score high "
        "because later blocks read what it wrote, not because it holds the behaviour itself."
    ),
    (
        "direction_inert depends on the null being sound. It is returned when the total effect "
        "does not clear what shuffled label and random directions do, so a null that is too "
        "strong or too weak moves that verdict, and it is the status hardest to verify by eye."
    ),
    "No measurement above 3B parameters stands for this project.",
)


class AttributionError(ValueError):
    """The measurement cannot be taken as asked, for a reason stated in the message."""


@dataclass(frozen=True)
class Conventions:
    """Declared before any result. Conventions, not measured thresholds."""

    min_prompts: int = 30            # metrics.MIN_REPORTABLE_N: below this a rate is not reported
    absent_auc_floor: float = 0.6    # lower end of the baseline AUC interval must clear this
    readout_agreement_floor: float = 0.75
    redundant_ratio_ceiling: float = 0.6   # sum of single effects over total, upper interval end
    effective_blocks_fraction: float = 0.5  # localised needs n_eff at most this share of blocks
    reliability_floor: float = 0.5   # median Jaccard of the exceeding set across random halves
    null_quantile: float = 0.95
    resamples: int = 1000
    shuffled_label_replicates: int = 20
    random_direction_replicates: int = 20
    random_direction_pcs: int = 8
    split_half_repeats: int = 200
    alpha: float = 0.05


DEFAULT = Conventions()


# ----------------------------------------------------------------------------- the intervention

def unwrap(out):
    import torch  # local: this module is import-light so `--help` does not pay for torch
    if isinstance(out, tuple):
        out = out[0] if out else None
    if not isinstance(out, torch.Tensor):
        raise AttributionError(
            f"a decoder block returned {type(out).__name__} rather than a tensor or a tuple "
            f"starting with one, so its write to the residual stream cannot be read. This "
            f"architecture has not been met; the run stops rather than guess.")
    return out


@contextlib.contextmanager
def ablating(layers, direction, blocks):
    """Remove `direction` from the write of each block in `blocks`, at every position.

    A block's write is what it adds: its output minus its input. Hooking the whole block rather
    than a sub-module is deliberate, because a norm applied after the sub-layer (the Gemma
    arrangement) sits between the weights and the stream and would hide the write from a hook on
    the weights. Blocks outside the list are untouched, so an empty list is the unedited model.
    """
    import torch  # local: this module is import-light so `--help` does not pay for torch
    blocks = tuple(blocks)
    n = len(layers)
    bad = [k for k in blocks if not 0 <= k < n]
    if bad:
        raise AttributionError(f"block(s) {bad} do not exist: this model has blocks 0 to {n - 1}.")
    handles = []
    if blocks:
        d = direction.detach().float()
        norm = float(d.norm())
        if not math.isfinite(norm) or norm <= 0.0:
            raise AttributionError(
                f"the direction has norm {norm}, so it points nowhere and nothing can be removed "
                f"along it.")
        d = d / norm

        def make():
            def hook(_module, args, kwargs, out):
                before = args[0] if args else kwargs.get("hidden_states")
                if not isinstance(before, torch.Tensor):
                    raise AttributionError(
                        "could not find the hidden states a decoder block was given, so its write "
                        "cannot be separated from its input.")
                after = unwrap(out)
                write = (after - before).float()
                kept = after.float() - (write @ d.to(write.device)).unsqueeze(-1) * d.to(write.device)
                kept = kept.to(after.dtype)
                return (kept, *out[1:]) if isinstance(out, tuple) else kept
            return hook
        handles.extend(layers[k].register_forward_hook(make(), with_kwargs=True) for k in blocks)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


# ------------------------------------------------------------------------------ the statistics

def auc(positive, negative):
    """P(a positive outranks a negative), ties counting a half, from tie averaged ranks."""
    import torch  # local: this module is import-light so `--help` does not pay for torch
    pos = torch.as_tensor(positive, dtype=torch.float64).flatten()
    neg = torch.as_tensor(negative, dtype=torch.float64).flatten()
    if pos.numel() == 0 or neg.numel() == 0:
        raise AttributionError("an AUC needs at least one value on each side.")
    x = torch.cat([pos, neg])
    _, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    last = torch.cumsum(counts, 0).to(torch.float64)
    ranks = (last - (counts.to(torch.float64) - 1) / 2)[inverse]
    n, m = pos.numel(), neg.numel()
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


@dataclass(frozen=True)
class Arm:
    """Prompts on one side of a contrast, with the request each one came from.

    Template variants of one request share a group and are resampled together, because treating
    them as independent inflates the effective sample (seven to one in the shipped track) and makes
    every interval too narrow.
    """

    prompts: tuple
    groups: tuple

    def __post_init__(self):
        if len(self.prompts) != len(self.groups):
            raise AttributionError(
                f"{len(self.prompts)} prompts but {len(self.groups)} group labels; every prompt "
                f"needs the request it came from.")


def arm(prompts, groups=None):
    prompts = tuple(prompts)
    return Arm(prompts, tuple(range(len(prompts))) if groups is None else tuple(groups))


def _group_index(groups):
    import torch  # local: this module is import-light so `--help` does not pay for torch
    order = {}
    for g in groups:
        order.setdefault(g, len(order))
    return torch.tensor([order[g] for g in groups]), len(order)


def _resample_weights(groups, draws, gen):
    """Per prompt multiplicities `[draws, n]` for resampling whole groups with replacement."""
    import torch  # local: this module is import-light so `--help` does not pay for torch
    index, n_groups = _group_index(groups)
    picks = torch.randint(n_groups, (draws, n_groups), generator=gen)
    counts = torch.zeros(draws, n_groups, dtype=torch.float64).scatter_add_(
        1, picks, torch.ones(draws, n_groups, dtype=torch.float64))
    return counts[:, index]


def bootstrap_effects(drop_h, drop_l, groups_h, groups_l, draws, gen):
    """Point effects `[C]` and resampled effects `[draws, C]` of mean(drop_h) - mean(drop_l)."""
    point = drop_h.mean(1) - drop_l.mean(1)
    wh = _resample_weights(groups_h, draws, gen)
    wl = _resample_weights(groups_l, draws, gen)
    boot = ((wh @ drop_h.T) / wh.sum(1, keepdim=True)
            - (wl @ drop_l.T) / wl.sum(1, keepdim=True))
    return point, boot


def interval(values, alpha):
    import torch  # local: this module is import-light so `--help` does not pay for torch
    lo, hi = torch.quantile(values, torch.tensor([alpha / 2, 1 - alpha / 2], dtype=values.dtype),
                            dim=0)
    return lo, hi


def effective_blocks(effects):
    """The number of blocks that would carry the same effect if it were shared equally.

    1 over the sum of squared shares, over the positive effects. A single block carrying
    everything gives 1; twelve equal blocks give 12. A negative effect is noise or a block that
    works against the behaviour and is not counted as carrying any.
    """
    positive = effects.clamp(min=0)
    total = float(positive.sum())
    if total <= 0.0:
        return float(len(effects))
    shares = positive / total
    return float(1.0 / (shares * shares).sum())


# ------------------------------------------------------------------------------- the direction

def fit_direction(fit_h, fit_l, search_h, search_l):
    """The mean difference direction at the position where it separates held-out prompts best.

    `fit_*` and `search_*` are `[positions, n, width]`. Fitted on one partition and scored on
    another, so the position is chosen by held-out AUC and not by the raw length of the difference,
    which grows with depth and always picks the last position. Returns the unit direction, the
    chosen position and the held-out AUC at every position.
    """
    if fit_h.shape[0] != search_h.shape[0]:
        raise AttributionError("the fit and search residuals disagree about how many positions "
                               "there are, so they were not taken from the same model.")
    aucs, dirs = [], []
    for p in range(fit_h.shape[0]):
        diff = fit_h[p].double().mean(0) - fit_l[p].double().mean(0)
        norm = float(diff.norm())
        if not math.isfinite(norm) or norm <= 0.0:
            aucs.append(0.5)
            dirs.append(None)
            continue
        unit = diff / norm
        dirs.append(unit)
        aucs.append(auc(search_h[p].double() @ unit, search_l[p].double() @ unit))
    live = [p for p, d in enumerate(dirs) if d is not None]
    if not live:
        raise AttributionError("the two fit arms have identical mean residuals at every position, "
                               "so there is no direction to extract.")
    # The embedding position (0) is excluded: no block writes it, so a direction chosen there
    # could never be ablated from anything.
    pool = [p for p in live if p > 0] or live
    best = max(pool, key=lambda p: (aucs[p], -p))
    return dirs[best].float(), best, aucs


def fit_direction_at(fit_h, fit_l, search_h, search_l, position):
    """The mean difference direction at a position chosen by the CALLER, and its held-out AUC.

    For asking what a result looks like if the direction had been taken somewhere else. The
    position is an input here and not a search, so nothing about it was selected on the data.
    """
    if not 0 <= position < fit_h.shape[0]:
        raise AttributionError(f"position {position} does not exist: this model has positions 0 "
                               f"to {fit_h.shape[0] - 1}.")
    diff = fit_h[position].double().mean(0) - fit_l[position].double().mean(0)
    norm = float(diff.norm())
    if not math.isfinite(norm) or norm <= 0.0:
        raise AttributionError(f"the two fit arms have identical mean residuals at position "
                               f"{position}, so there is no direction to extract there.")
    unit = diff / norm
    held_out = auc(search_h[position].double() @ unit, search_l[position].double() @ unit)
    return unit.float(), held_out


def profile_with_intervals(subject, measure, direction, n_blocks, conv=DEFAULT, seed=0):
    """Total and single block effects with intervals, for one direction, and nothing else.

    No nulls and no status: this is for comparing profiles under different directions, where the
    question is whether the shape moved.
    """
    import torch  # local: this module is import-light so `--help` does not pay for torch
    conds = {"none": (), "all": tuple(range(n_blocks)),
             **{f"single:{k}": (k,) for k in range(n_blocks)}}
    dh, dl, _, _ = _effects_for(subject, measure, direction, conds)
    point, boot = bootstrap_effects(dh, dl, measure["harmful"].groups, measure["harmless"].groups,
                                    conv.resamples, torch.Generator().manual_seed(seed))
    lo, hi = interval(boot, conv.alpha)
    return {"total": float(point[1]), "total_ci": [float(lo[1]), float(hi[1])],
            "effects": [float(point[2 + k]) for k in range(n_blocks)],
            "effects_ci": [[float(lo[2 + k]), float(hi[2 + k])] for k in range(n_blocks)]}


# ------------------------------------------------------------------------------------- the run

def _conditions(n_blocks):
    """Name -> blocks ablated. Order is part of the artefact."""
    conds = {"none": (), "all": tuple(range(n_blocks))}
    for k in range(n_blocks):
        conds[f"single:{k}"] = (k,)
    for k in range(n_blocks):
        conds[f"front:{k}"] = tuple(range(k + 1))
    for k in range(n_blocks):
        conds[f"back:{k}"] = tuple(range(k, n_blocks))
    return conds


def _readouts(subject, arm_, direction, conds):
    import torch  # local: this module is import-light so `--help` does not pay for torch
    rows = []
    for blocks in conds.values():
        out = torch.as_tensor(subject.readout(list(arm_.prompts), direction, blocks),
                              dtype=torch.float64)
        if out.shape != (len(arm_.prompts),) or not bool(torch.isfinite(out).all()):
            raise AttributionError(
                f"the readout returned {tuple(out.shape)} with non-finite values where "
                f"({len(arm_.prompts)},) finite values were expected, so the run stops here.")
        rows.append(out)
    return torch.stack(rows)


def _effects_for(subject, measure, direction, conds):
    rh = _readouts(subject, measure["harmful"], direction, conds)
    rl = _readouts(subject, measure["harmless"], direction, conds)
    return rh[0:1] - rh, rl[0:1] - rl, rh[0], rl[0]


def _validate(parts, conv):
    for name in ("fit", "search", "measure"):
        if name not in parts:
            raise AttributionError(f"the {name} partition is missing.")
        for side in ("harmful", "harmless"):
            if side not in parts[name] or len(parts[name][side].prompts) == 0:
                raise AttributionError(f"the {name} partition has no {side} prompts.")


def attribute(subject, parts, *, seed=0, conv=DEFAULT, predictive=None, log=None):
    """The whole measurement. `parts` maps fit, search and measure to `{harmful, harmless}` arms."""
    import torch  # local: this module is import-light so `--help` does not pay for torch
    log = log or (lambda *_: None)
    _validate(parts, conv)
    nb = subject.n_blocks
    gen = torch.Generator().manual_seed(seed)
    measure = parts["measure"]
    n_h, n_l = len(measure["harmful"].prompts), len(measure["harmless"].prompts)

    if min(n_h, n_l) < conv.min_prompts:
        return _finish(UNRESOLVED, (
            f"{min(n_h, n_l)} measurement prompts in the smaller arm is below the floor of "
            f"{conv.min_prompts}, under which a rate is not reported."), conv, nb, seed)

    conds = _conditions(nb)
    names = list(conds)
    log(f"attribution: {len(names)} conditions over {n_h} + {n_l} measurement prompts")

    if hasattr(subject, "prepare"):
        subject.prepare(parts)

    # The readout is validated BEFORE anything is ranked with it, and a readout that fails is the
    # whole answer: with no valid readout there is no way to tell an absent behaviour from one the
    # proxy cannot see, so neither is claimed.
    agreement = subject.readout_agreement(parts["search"]["harmful"].prompts,
                                          parts["search"]["harmless"].prompts)
    agreement = None if agreement is None or not math.isfinite(agreement) else float(agreement)
    problem = _readout_problem(agreement, conv)
    if problem:
        out = _finish(READOUT_INVALID, problem, conv, nb, seed)
        out["controls"] = {"readout_agreement": agreement,
                           "readout_agreement_detail": getattr(subject, "agreement_detail", None)}
        return out

    res = {k: {s: subject.residuals(list(parts[k][s].prompts)) for s in ("harmful", "harmless")}
           for k in ("fit", "search")}
    direction, position, position_aucs = fit_direction(
        res["fit"]["harmful"], res["fit"]["harmless"],
        res["search"]["harmful"], res["search"]["harmless"])

    drop_h, drop_l, base_h, base_l = _effects_for(subject, measure, direction, conds)
    gh, gl = measure["harmful"].groups, measure["harmless"].groups
    point, boot = bootstrap_effects(drop_h, drop_l, gh, gl, conv.resamples, gen)
    lo, hi = interval(boot, conv.alpha)

    # -- is there a behaviour at all
    base_auc = auc(base_h, base_l)
    auc_lo = _auc_interval(base_h, base_l, gh, gl, conv, gen)[0]
    idx = {n: i for i, n in enumerate(names)}
    singles = [idx[f"single:{k}"] for k in range(nb)]
    e_single, e_lo = point[singles], lo[singles]
    total, total_lo, total_hi = point[idx["all"]], lo[idx["all"]], hi[idx["all"]]

    nulls = _null_replicates(subject, parts, res, measure, direction, seed, conv)
    tau = float(torch.quantile(nulls["max_block"], conv.null_quantile))
    t_null = float(torch.quantile(nulls["abs_total"], conv.null_quantile))

    ratio = (boot[:, singles].sum(1) / boot[:, idx["all"]].clamp(min=1e-9))
    ratio_hi = float(torch.quantile(ratio, 1 - conv.alpha / 2))
    exceeding = [k for k in range(nb) if float(e_single[k]) > tau and float(e_lo[k]) > 0.0]
    n_eff = effective_blocks(e_single)
    reliability = _split_half(drop_h[singles], drop_l[singles], gh, gl, tau, conv, gen,
                              reference=set(exceeding))

    evidence = {
        "n_smaller_arm": min(n_h, n_l), "baseline_auc_lo": auc_lo,
        "agreement": agreement, "total": float(total), "total_lo": float(total_lo),
        "t_null": t_null, "ratio_hi": ratio_hi, "exceeding": exceeding, "n_eff": n_eff,
        "reliability": reliability["median_jaccard"],
    }
    status, reason = classify(evidence, conv, nb)

    blocks = []
    for k in range(nb):
        i = idx[f"single:{k}"]
        blocks.append({
            "index": k,
            "effect": float(point[i]), "effect_ci": [float(lo[i]), float(hi[i])],
            "share_of_total": float(point[i] / total) if float(total) > 0 else None,
            "harmless_drop": float(drop_l[i].mean()), "harmful_drop": float(drop_h[i].mean()),
            "exceeds_null": k in exceeding,
        })
    out = _finish(status, reason, conv, nb, seed)
    out.update({
        "evidence": evidence,
        "direction": {"position": int(position), "held_out_auc": float(position_aucs[position])},
        "totals": {"effect": float(total), "ci": [float(total_lo), float(total_hi)],
                   "baseline_gap": float(base_h.mean() - base_l.mean())},
        "blocks": blocks,
        "concentration": {"effective_blocks": n_eff, "n_blocks": nb},
        "curves": {
            "front": [float(point[idx[f"front:{k}"]]) for k in range(nb)],
            "back": [float(point[idx[f"back:{k}"]]) for k in range(nb)],
        },
        "reliability": reliability,
        "sum_of_single_effects_over_total": {"upper": ratio_hi},
        "controls": {
            "baseline": {"auc": base_auc, "auc_lower": auc_lo},
            "readout_agreement": agreement,
            "readout_agreement_detail": getattr(subject, "agreement_detail", None),
            "shuffled_label": nulls["shuffled"],
            "random_direction": nulls["random"],
            "style_direction": style_direction(subject, parts, res, position, float(total), nb),
            "family_wise_threshold": tau,
            "total_threshold": t_null,
        },
    })
    if status == LOCALISED and predictive is not None:
        ranking = sorted(range(nb), key=lambda k: -float(e_single[k]))
        out["predictive_check"] = predictive(subject, parts, exceeding, direction, ranking)
    elif status == LOCALISED:
        out["predictive_check"] = {"ran": False,
                                   "why": "no predictive check was supplied for this run"}
    out["tint_allowed"] = tint_allowed(out)
    return out


#: Declared with the instrument and reported only, never used to change a status: a direction that
#: separates long prompts from short ones, removed from every block, is flagged when it does at
#: least this share of what the behaviour direction does.
STYLE_FLAG_RATIO = 0.5


def style_direction(subject, parts, res, position, total, nb):
    """What removing a prompt LENGTH direction does to the same readout, beside the real total.

    Harmful prompts tend to be longer or shorter than harmless ones, so a direction fitted to the
    class contrast can be partly a direction for length. This fits one for length alone (long
    against short within each class, so the classes are balanced) and removes it from every block.
    If that moves the readout about as much as the behaviour direction does, the behaviour
    direction may be reading style. Reported beside the result; it changes no status.
    """
    import torch  # local: this module is import-light so `--help` does not pay for torch
    first = parts["fit"]["harmful"].prompts[:1]
    if not (first and isinstance(first[0], str)):
        return {"ran": False, "why": "prompts are not text, so there is no length to split on"}
    long_rows, short_rows = [], []
    for side in ("harmful", "harmless"):
        prompts = parts["fit"][side].prompts
        lengths = sorted(len(p) for p in prompts)
        cut = lengths[len(lengths) // 2]
        resid = res["fit"][side][position].double()
        for i, p in enumerate(prompts):
            (long_rows if len(p) > cut else short_rows).append(resid[i])
    if not long_rows or not short_rows:
        return {"ran": False, "why": "every fit prompt has the same length"}
    diff = torch.stack(long_rows).mean(0) - torch.stack(short_rows).mean(0)
    if not float(diff.norm()) > 0.0:
        return {"ran": False, "why": "long and short prompts have identical mean residuals"}
    t_style, singles = total_and_singles(subject, parts["measure"], (diff / diff.norm()).float(), nb)
    return {"ran": True, "total": t_style, "max_single": float(singles.max()),
            "ratio_to_behaviour_total": t_style / total if total > 0 else None,
            "flag_ratio": STYLE_FLAG_RATIO,
            "flagged": bool(total > 0 and t_style / total >= STYLE_FLAG_RATIO)}


def selftest(*, conv=DEFAULT):
    """Run every planted model and report whether each got its planted status.

    An instrument that has never been seen to return nothing has not been validated, so every
    artefact carries this beside its result. It is cheap and deterministic.
    """
    from . import attribution_fixtures as fx
    cases = {}
    for name, expected in fx.EXPECTED.items():
        got = attribute(fx.ToySubject(name), fx.partitions(name), seed=0, conv=conv)["status"]
        cases[name] = {"expected": expected, "got": got, "ok": got == expected}
    return {"passed": all(c["ok"] for c in cases.values()), "instrument_version": INSTRUMENT_VERSION,
            "cases": cases}


def _auc_interval(base_h, base_l, gh, gl, conv, gen):
    import torch  # local: this module is import-light so `--help` does not pay for torch
    ih, il = _group_index(gh)[0], _group_index(gl)[0]
    n_gh, n_gl = int(ih.max()) + 1, int(il.max()) + 1
    values = []
    for _ in range(min(conv.resamples, 400)):
        ph = torch.randint(n_gh, (n_gh,), generator=gen)
        pl = torch.randint(n_gl, (n_gl,), generator=gen)
        sel_h = torch.cat([(ih == g).nonzero().flatten() for g in ph])
        sel_l = torch.cat([(il == g).nonzero().flatten() for g in pl])
        values.append(auc(base_h[sel_h], base_l[sel_l]))
    v = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(v, conv.alpha / 2)), float(torch.quantile(v, 1 - conv.alpha / 2))


def _readout_problem(agreement, conv):
    """Why the readout cannot stand in for the behaviour, or None when it can."""
    if agreement is None:
        return ("the readout could not be validated against generated behaviour (see "
                "controls.readout_agreement_detail), so it cannot stand in for the behaviour.")
    if agreement < conv.readout_agreement_floor:
        return (f"the readout agrees with generated behaviour at {agreement:.3f}, below "
                f"{conv.readout_agreement_floor}, so it cannot stand in for the behaviour.")
    return None


def classify(ev, conv, n_blocks):
    """The status and the reason, from the evidence alone. Pure, so a broken copy can be compared."""
    if ev["n_smaller_arm"] < conv.min_prompts:
        return UNRESOLVED, (f"{ev['n_smaller_arm']} measurement prompts is below the floor of "
                            f"{conv.min_prompts}.")
    problem = _readout_problem(ev["agreement"], conv)
    if problem:
        return READOUT_INVALID, problem
    if ev["baseline_auc_lo"] <= conv.absent_auc_floor:
        return ABSENT, (f"the baseline readout does not separate harmful from harmless prompts "
                        f"(AUC interval starts at {ev['baseline_auc_lo']:.3f}, floor "
                        f"{conv.absent_auc_floor}), so there is no behaviour to attribute.")
    if ev["total_lo"] <= 0.0 or ev["total"] <= ev["t_null"]:
        return DIRECTION_INERT, (
            f"removing the direction from every block moved the readout by {ev['total']:.3f}, "
            f"which does not clear what shuffled label and random directions do "
            f"({ev['t_null']:.3f}). The direction cannot see this behaviour; that says nothing "
            f"about whether it is localised.")
    if ev["ratio_hi"] < conv.redundant_ratio_ceiling:
        return REDUNDANT, (
            f"the single block effects sum to under {conv.redundant_ratio_ceiling} of the total "
            f"(upper end {ev['ratio_hi']:.3f}), so blocks are covering for one another and a "
            f"single ablation cannot identify them.")
    if not ev["exceeding"] or ev["n_eff"] > conv.effective_blocks_fraction * n_blocks:
        return DIFFUSE, (
            f"the effect is real but no block stands out: {len(ev['exceeding'])} clear the "
            f"family wise null and the effect is shared as if by {ev['n_eff']:.1f} of "
            f"{n_blocks} blocks.")
    if ev["reliability"] < conv.reliability_floor:
        return UNRESOLVED, (f"{len(ev['exceeding'])} block(s) clear the null but the set is not "
                            f"reproduced between random halves (median Jaccard "
                            f"{ev['reliability']:.2f}, floor {conv.reliability_floor}).")
    return LOCALISED, (f"{len(ev['exceeding'])} block(s) clear the family wise null with the "
                       f"effect shared as if by {ev['n_eff']:.1f} of {n_blocks} blocks.")


def tint_allowed(result):
    """The one place a card is permitted to colour individual blocks.

    True only for `localised` whose predictive check ran and was supported. Everything else,
    including a localised result nobody checked against generation, is false.
    """
    if result.get("status") != LOCALISED:
        return False
    check = result.get("predictive_check") or {}
    return bool(check.get("ran")) and check.get("supported") is True


def _finish(status, reason, conv, n_blocks, seed):
    if status not in STATUSES:
        raise AttributionError(f"{status!r} is not a status this instrument can return.")
    return {
        "instrument": {"name": INSTRUMENT, "version": INSTRUMENT_VERSION,
                       "estimator": ESTIMATOR, "units": UNITS},
        "status": status, "status_reason": reason, "tint_allowed": False,
        "n_blocks": n_blocks,
        "sampling": {"seed": seed, "resamples": conv.resamples,
                     "shuffled_label_replicates": conv.shuffled_label_replicates,
                     "random_direction_replicates": conv.random_direction_replicates},
        "conventions": asdict(conv),
        "limits": list(LIMITS),
    }


# ----------------------------------------------------------------------------------- the nulls

def total_and_singles(subject, measure, direction, nb):
    """The total effect and every single block effect of removing `direction`, on the measure arms."""
    sub = {"none": (), "all": tuple(range(nb)), **{f"single:{k}": (k,) for k in range(nb)}}
    dh, dl, _, _ = _effects_for(subject, measure, direction, sub)
    eff = dh.mean(1) - dl.mean(1)
    return float(eff[1]), eff[2:]


def _null_replicates(subject, parts, res, measure, direction, seed, conv):
    """Shuffled label and random direction replicates of the whole measurement.

    A shuffled label replicate permutes which REQUESTS count as harmful in the fit and search
    partitions and runs the identical path, position choice included; the measurement arms keep
    their true labels. It answers "what does this pipeline report when the direction came out of
    noise". The random direction replicates
    answer "what does removing an arbitrary direction from where the blocks write do". Each is drawn
    from the top principal directions of what the blocks write (label free) and is NOT matched to
    the real direction's energy: the first draft picked the best of 64 candidates by energy, which
    selected the directions most aligned with the dominant write, so on a behaviour spread across
    every block the null was as large as the effect and the instrument called it direction_inert on
    4 of 6 seeds. The energy ratio is reported instead.
    Both feed the family wise threshold, which is taken over the LARGEST block effect
    per replicate, because the best of many blocks manufactures a winner from nothing.
    """
    import torch  # local: this module is import-light so `--help` does not pay for torch
    nb = subject.n_blocks
    gen = torch.Generator().manual_seed(seed + 7919)
    def stats(d):
        return total_and_singles(subject, measure, d, nb)

    shuffled, random_ = [], []
    pooled = {k: torch.cat([res[k]["harmful"], res[k]["harmless"]], 1) for k in ("fit", "search")}
    sizes = {k: (res[k]["harmful"].shape[1], res[k]["harmless"].shape[1]) for k in pooled}
    # Qualified by side, because two arms may number their requests from the same start.
    groups = {k: tuple(("h", g) for g in parts[k]["harmful"].groups)
              + tuple(("l", g) for g in parts[k]["harmless"].groups) for k in pooled}
    for _ in range(conv.shuffled_label_replicates):
        labels = {}
        for k in pooled:
            index, n_groups = _group_index(groups[k])
            harmful_groups = torch.randperm(n_groups, generator=gen)[
                :max(1, round(n_groups * sizes[k][0] / sum(sizes[k])))]
            is_h = torch.isin(index, harmful_groups)
            if bool(is_h.all()) or not bool(is_h.any()):
                is_h[0] = not bool(is_h[0])
            labels[k] = is_h
        d, _, _ = fit_direction(pooled["fit"][:, labels["fit"]], pooled["fit"][:, ~labels["fit"]],
                                pooled["search"][:, labels["search"]],
                                pooled["search"][:, ~labels["search"]])
        shuffled.append(stats(d))

    pooled_fit = torch.cat([res["fit"]["harmful"], res["fit"]["harmless"]], 1)
    writes = (pooled_fit[1:] - pooled_fit[:-1]).double()  # [blocks, n, H], label free
    flat = writes.reshape(-1, writes.shape[-1])
    flat = flat - flat.mean(0)
    _, _, vt = torch.linalg.svd(flat, full_matrices=False)
    basis = vt[:conv.random_direction_pcs].float()
    target = float((writes @ direction.double()).abs().mean())
    energy_ratio = []
    for _ in range(conv.random_direction_replicates):
        coef = torch.randn(basis.shape[0], generator=gen)
        pick = (coef / coef.norm()) @ basis
        energy_ratio.append(float((writes.float() @ pick).abs().mean()) / target
                            if target > 0 else float("nan"))
        random_.append(stats(pick))

    every = shuffled + random_
    return {
        "max_block": torch.tensor([float(s.max()) for _, s in every], dtype=torch.float64),
        "abs_total": torch.tensor([abs(t) for t, _ in every], dtype=torch.float64),
        "shuffled": _null_summary(shuffled),
        "random": {**_null_summary(random_),
                   "energy_ratio_median": float(torch.tensor(energy_ratio).median())},
    }


def _null_summary(reps):
    import torch  # local: this module is import-light so `--help` does not pay for torch
    totals = torch.tensor([t for t, _ in reps], dtype=torch.float64)
    maxima = torch.tensor([float(s.max()) for _, s in reps], dtype=torch.float64)
    return {"replicates": len(reps), "total_abs_max": float(totals.abs().max()),
            "max_block_median": float(maxima.median()), "max_block_max": float(maxima.max())}


def _split_half(drop_h, drop_l, gh, gl, tau, conv, gen, *, reference):
    """How often two random halves of the requests agree on which blocks clear the threshold."""
    import torch  # local: this module is import-light so `--help` does not pay for torch
    ih, nh = _group_index(gh)
    il, nl = _group_index(gl)
    scores = []
    for _ in range(conv.split_half_repeats):
        side_h = torch.randperm(nh, generator=gen) % 2
        side_l = torch.randperm(nl, generator=gen) % 2
        sets = []
        for half in (0, 1):
            mh, ml = side_h[ih] == half, side_l[il] == half
            if not bool(mh.any()) or not bool(ml.any()):
                sets.append(set())
                continue
            eff = drop_h[:, mh].mean(1) - drop_l[:, ml].mean(1)
            sets.append({k for k in range(len(eff)) if float(eff[k]) > tau})
        union = sets[0] | sets[1]
        scores.append(len(sets[0] & sets[1]) / len(union) if union else 0.0)
    scores_t = torch.tensor(scores, dtype=torch.float64)
    return {"median_jaccard": float(scores_t.median()), "repeats": conv.split_half_repeats,
            "reference_set_size": len(reference)}
