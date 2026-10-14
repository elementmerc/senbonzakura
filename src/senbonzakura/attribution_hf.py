# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A real language model, presented to the attribution engine.

THE READOUT. The behaviour readout for refusal is a first token margin: at the position where the
reply begins, the log odds between the tokens that open this model's refusals and the tokens that
open its compliant replies. One forward pass, no generation, deterministic at a fixed batch size.

IT IS A PROXY AND IT IS CHECKED, NOT TRUSTED. The two token sets are learned from the model's own
greedy replies on the FIT partition, labelled by the project's refusal ruler, so nothing is
hand listed. Whether the margin then predicts the reply the model really gives is measured on the
SEARCH partition and reported as `agreement`. A model that warms up before it refuses puts the
decision past the first token, and the margin then predicts nothing; the engine reports
`readout_invalid` for that and does not rank blocks on a readout that cannot see the behaviour.

NOTHING HERE READS THE EDIT. This module loads a model, renders prompts through the project's one
renderer, and hooks decoder blocks. The footprint firewall test covers it.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import time

import torch

from . import attribution as A
from . import metrics, modelmap
from .firsttoken import last_token_logits, render_chat

#: Greedy reply length used to label a prompt as refused or not. Long enough that a refusal after
#: a short preamble is seen, which `metrics.is_refusal` scans the whole reply for.
REPLY_TOKENS = 128
#: Fewer than this many replies in a class and the agreement is not a measurement.
MIN_PER_CLASS = 5
#: A token must open at least this many replies to join a set, so one stray reply cannot define it.
#: Fixed when the module was written, before any real model had been run through it, and recorded
#: in the artefact. The sensitivity of the sets to it is reported as information and never used to
#: choose it.
MIN_TOKEN_COUNT = 2


#: THE TWO READOUT ARMS. The disjoint arm was declared before any real model was run through this
#: code. The purity arm was added AFTER the fit partition had been inspected, once the disjoint
#: rule was seen to exclude the commonest refusal opener. Both are reported and neither wins on
#: the day: the second arm is honest only because it is fixed in this file before its output was
#: seen, and the artefact says which arm is which so a reader can weigh them.
ARM_DISJOINT, ARM_PURITY = "disjoint", "purity"
#: Fixed now and not to be changed to suit a result. A token joins a set when at least this share
#: of the fit replies it opens are of that kind.
PURITY_THRESHOLD = 0.9
ARM_DECLARATIONS = {
    ARM_DISJOINT: {
        "declared": "before any real model was run through this code",
        "rule": ("a token joins the refusal set when it opens at least MIN_TOKEN_COUNT fit replies "
                 "and none of them is a compliant reply, and the compliance set the other way "
                 "round; a token that opens both kinds is dropped"),
    },
    ARM_PURITY: {
        "declared": "after the fit partition had been inspected",
        "rule": ("a token joins the refusal set when it opens at least MIN_TOKEN_COUNT fit replies "
                 f"and at least {PURITY_THRESHOLD} of them are refusals, and the compliance set "
                 f"the other way round; a token below {PURITY_THRESHOLD} either way is dropped"),
    },
}


def token_sets(refused, complied, arm):
    """The refusal tokens, the compliance tokens and the count dropped, by the arm's exact rule."""
    if arm not in ARM_DECLARATIONS:
        raise A.AttributionError(f"{arm!r} is not a readout arm; the arms are "
                                 f"{sorted(ARM_DECLARATIONS)}.")
    ref, com, dropped = [], [], 0
    for token in sorted(set(refused) | set(complied)):
        r, c = refused.get(token, 0), complied.get(token, 0)
        if r + c < MIN_TOKEN_COUNT:
            continue
        if arm == ARM_DISJOINT:
            pure_r, pure_c = c == 0, r == 0
        else:
            pure_r, pure_c = r / (r + c) >= PURITY_THRESHOLD, c / (r + c) >= PURITY_THRESHOLD
        if pure_r:
            ref.append(token)
        elif pure_c:
            com.append(token)
        else:
            dropped += 1
    return ref, com, dropped


def load(model_id, *, device="cuda", dtype=torch.bfloat16, revision=None):
    """The model, its tokenizer and the identity a result must carry."""
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, revision=revision)
    model = model.to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    identity = {
        "id": model_id,
        "revision": getattr(model.config, "_commit_hash", None) or revision,
        "precision": str(dtype).removeprefix("torch."),
        "transformers": transformers.__version__, "torch": torch.__version__,
        "device": str(device),
    }
    return model, tok, identity


def digest(prompts):
    return hashlib.sha256("\n".join(sorted(prompts)).encode("utf-8")).hexdigest()[:16]


def build_partitions(track_dir, *, n_measure=200, n_confirm=0, seed=0):
    """Fit, search and measure arms from a track, one prompt per request on the measure side.

    Template variants of one request are not independent, so a request is represented by a single
    randomly chosen variant in every partition and the group labels are the requests themselves.
    The partition boundaries are the ones the track recorded; nothing here re-slices them.
    Returns the arms and the digests that identify the exact rows, never the rows.
    """
    from pathlib import Path

    from . import track
    harmful, harmless = track.load_partitions(Path(track_dir))
    gen = torch.Generator().manual_seed(seed)
    parts, digests = {}, {}
    for name in ("fit", "search", "measure"):
        parts.setdefault(name, {})
        digests.setdefault(name, {})
        for side, rows in (("harmful", harmful[name]), ("harmless", harmless[name])):
            groups = track.group_by_request(rows)
            keys = sorted(groups)
            order = torch.randperm(len(keys), generator=gen).tolist()
            cap = n_measure if name == "measure" else len(keys)
            sliced = {name: order[:cap]}
            if name == "measure" and n_confirm:
                # Fresh requests the ranking never saw, so the predictive check is not graded on
                # the rows it was chosen with.
                sliced["confirm"] = order[cap:cap + n_confirm]
            for label, indices in sliced.items():
                chosen = [keys[i] for i in indices]
                picks = [groups[k][int(torch.randint(len(groups[k]), (1,), generator=gen))]
                         for k in chosen]
                parts.setdefault(label, {})[side] = A.arm(picks, chosen)
                digests.setdefault(label, {})[side] = {
                    "n": len(picks), "requests": len(groups), "rows_in_partition": len(rows),
                    "digest": digest(picks)}
    return parts, digests


class HFSubject:
    def __init__(self, model, tok, *, batch=16, arm=ARM_DISJOINT, log=print):
        if arm not in ARM_DECLARATIONS:
            raise A.AttributionError(f"{arm!r} is not a readout arm; the arms are "
                                     f"{sorted(ARM_DECLARATIONS)}.")
        self.model, self.tok, self.batch, self.log, self.arm = model, tok, batch, log, arm
        self._standing = []
        self._encoded = {}
        self.layers = modelmap.stack_or_raise(model)
        self.n_blocks = len(self.layers)
        self.refusal_ids = self.comply_ids = None
        self.readout_why_not = None
        self.agreement_detail = {}
        self._replies = {}

    # ------------------------------------------------------------------ encoding and capture
    def _encode(self, prompts):
        key = tuple(prompts)
        if key not in self._encoded:
            text = [render_chat(self.tok, p) for p in prompts]
            self._encoded[key] = self.tok(text, return_tensors="pt", padding=True,
                                          add_special_tokens=False).to(self.model.device)
        return self._encoded[key]

    def edit_standing(self, direction, blocks):
        """Keep `direction` removed from `blocks` for every later call, as an edited model would be.

        An in-memory stand-in for a checkpoint that has the behaviour removed, used as a control.
        It is a hook and not a weight edit, so it says nothing about what the real bake does.
        """
        self._standing = [(direction, tuple(blocks))]
        self._replies.clear()

    def _under_standing_edit(self):
        stack = contextlib.ExitStack()
        for direction, blocks in self._standing:
            stack.enter_context(A.ablating(self.layers, direction, blocks))
        return stack

    def _batches(self, prompts):
        prompts = list(prompts)
        for i in range(0, len(prompts), self.batch):
            yield prompts[i:i + self.batch]

    @torch.no_grad()
    def residuals(self, prompts):
        """Last position residual at the input and after every block: `[blocks + 1, n, width]`.

        Taken from hooks on the blocks themselves and NOT from `hidden_states`, whose final entry
        is after the model's last norm and so is a different space from the one blocks write to.
        """
        rows = []
        seen = {}
        for chunk in self._batches(prompts):
            seen.clear()

            handles = [self.layers[0].register_forward_pre_hook(
                lambda _m, args, kwargs: seen.__setitem__(0, (args[0] if args else
                                                              kwargs["hidden_states"])[:, -1, :]
                                                          .float().cpu()), with_kwargs=True)]
            for k, layer in enumerate(self.layers):
                handles.append(layer.register_forward_hook(
                    lambda _m, _a, out, k=k: seen.__setitem__(
                        k + 1, A.unwrap(out)[:, -1, :].float().cpu())))
            try:
                with self._under_standing_edit():
                    self.model(**self._encode(chunk), use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            rows.append(torch.stack([seen[k] for k in range(self.n_blocks + 1)]))
        return torch.cat(rows, 1)

    # ------------------------------------------------------------------- the readout's tokens
    @torch.no_grad()
    def _reply(self, chunk, direction=None, blocks=()):
        enc = self._encode(chunk)
        with self._under_standing_edit(), A.ablating(self.layers, direction, blocks):
            out = self.model.generate(**enc, max_new_tokens=REPLY_TOKENS, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        return [(int(new[i, 0]), metrics.is_refusal(texts[i])) for i in range(len(chunk))]

    def _label(self, prompts):
        todo = [p for p in prompts if p not in self._replies]
        beat = time.monotonic()
        for n, chunk in enumerate(self._batches(todo)):
            for p, got in zip(chunk, self._reply(chunk), strict=True):
                self._replies[p] = got
            if time.monotonic() - beat > 30:                     # pragma: no cover - timing
                beat = time.monotonic()
                self.log(f"  replies: {min((n + 1) * self.batch, len(todo))} of {len(todo)}")
        return [self._replies[p] for p in prompts]

    def prepare(self, parts):
        """Learn which tokens open a refusal and which open a compliant reply, from FIT only."""
        self.refusal_ids = self.comply_ids = None
        self.agreement_detail = {}
        prompts = list(parts["fit"]["harmful"].prompts) + list(parts["fit"]["harmless"].prompts)
        refused, complied = {}, {}
        for first, was_refusal in self._label(prompts):
            bucket = refused if was_refusal else complied
            bucket[first] = bucket.get(first, 0) + 1
        both = set(refused) & set(complied)
        sensitivity = {}
        for floor in (1, 2, 3):
            r_, c_, _ = token_sets(
                {t: n for t, n in refused.items() if n >= floor},
                {t: n for t, n in complied.items() if n >= floor}, self.arm)
            sensitivity[str(floor)] = {"refusal_tokens": len(r_), "compliance_tokens": len(c_)}
        ref, com, dropped = token_sets(refused, complied, self.arm)
        self._fit_counts = (refused, complied)
        self.readout_why_not = None
        if not ref or not com:
            self.readout_why_not = (
                f"no first token cleanly opens only refusals ({len(ref)} found) or only compliant "
                f"replies ({len(com)} found) on the fit partition, so there is no readout to build")
            return
        self.refusal_ids = torch.tensor(ref, device=self.model.device)
        self.comply_ids = torch.tensor(com, device=self.model.device)
        self.agreement_detail["token_sets"] = {
            "refusal_tokens": len(ref), "compliance_tokens": len(com),
            "shared_tokens_dropped": len(both), "dropped_by_the_arm_rule": dropped,
            "arm": self.arm, **ARM_DECLARATIONS[self.arm],
            "purity_threshold": PURITY_THRESHOLD,
            "refusal_set_covers_fit_refusals": (sum(refused.get(t, 0) for t in ref)
                                                / max(sum(refused.values()), 1)),
            "min_token_count": MIN_TOKEN_COUNT, "sensitivity_to_min_token_count": sensitivity,
            "refusal_openers_by_count": {self.tok.convert_ids_to_tokens([t])[0]: c
                                         for t, c in sorted(refused.items(), key=lambda x: -x[1])[:8]},
            "compliance_openers_by_count": {self.tok.convert_ids_to_tokens([t])[0]: c
                                            for t, c in sorted(complied.items(),
                                                               key=lambda x: -x[1])[:8]},
            "refusal_token_text": self.tok.convert_ids_to_tokens(ref),
            "compliance_token_text": self.tok.convert_ids_to_tokens(com[:16]),
            "fit_refused": sum(refused.values()), "fit_complied": sum(complied.values())}

    def refusal_flags(self, prompts, direction=None, blocks=()):
        """Whether each prompt's greedy reply is a refusal, with `blocks` ablated while it is written.

        Not cached: the point is to see what a different condition does to the same prompts.
        """
        out = []
        for chunk in self._batches(prompts):
            out.extend(r for _, r in self._reply(chunk, direction, blocks))
        return out

    # ---------------------------------------------------------------------------- the readout
    @torch.no_grad()
    def readout(self, prompts, direction=None, blocks=()):
        if self.refusal_ids is None:
            raise A.AttributionError(self.readout_why_not or "prepare() has not been called")
        out = []
        for chunk in self._batches(prompts):
            with self._under_standing_edit(), A.ablating(self.layers, direction, blocks):
                logits = last_token_logits(self.model, self._encode(chunk), log=self.log)
            out.append(torch.logsumexp(logits[:, self.refusal_ids], 1)
                       - torch.logsumexp(logits[:, self.comply_ids], 1))
        return torch.cat(out).double().cpu()

    def _opener_counts(self, ids, top=12):
        """Decoded opener tokens with their counts, most common first. Single tokens only."""
        counts = {}
        for t in ids:
            counts[t] = counts.get(t, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: -x[1])[:top]
        return {self.tok.convert_ids_to_tokens([t])[0]: n for t, n in ranked}

    @staticmethod
    def _split_agreement(margins, labelled, covered, want):
        """AUC of the margin for 'was a refusal' over the replies whose opener is (not) in a set."""
        refused = [m for m, (_, r), c in zip(margins, labelled, covered, strict=True)
                   if c == want and r]
        complied = [m for m, (_, r), c in zip(margins, labelled, covered, strict=True)
                    if c == want and not r]
        ok = len(refused) >= MIN_PER_CLASS and len(complied) >= MIN_PER_CLASS
        return {"refused": len(refused), "complied": len(complied),
                "auc": A.auc(refused, complied) if ok else None}

    def readout_agreement(self, harmful, harmless):
        """How well the margin predicts the reply the model actually gives, on SEARCH prompts.

        AUC of the margin for 'this reply was a refusal', over both arms pooled. None when the
        readout could not be built or one class has too few replies to say, which the engine
        reports as `readout_invalid` with the reason.
        """
        if self.refusal_ids is None:
            self.agreement_detail["why_not"] = self.readout_why_not
            return None
        prompts = list(harmful) + list(harmless)
        labelled = self._label(prompts)
        margins = self.readout(prompts).tolist()
        refused = [m for m, (_, r) in zip(margins, labelled, strict=True) if r]
        complied = [m for m, (_, r) in zip(margins, labelled, strict=True) if not r]
        known = set(self.refusal_ids.tolist()) | set(self.comply_ids.tolist())
        covered = [f in known for f, _ in labelled]
        self.agreement_detail.update({
            "search_refused": len(refused), "search_complied": len(complied),
            "onset_coverage": sum(covered) / len(labelled),
            "uncovered_openers": self._opener_counts([f for f, c in zip(
                (f for f, _ in labelled), covered, strict=True) if not c]),
            "by_coverage": {
                "covered": self._split_agreement(margins, labelled, covered, True),
                "uncovered": self._split_agreement(margins, labelled, covered, False)}})
        if len(refused) < MIN_PER_CLASS or len(complied) < MIN_PER_CLASS:
            self.agreement_detail["why_not"] = (
                f"only {len(refused)} refusals and {len(complied)} compliant replies on the "
                f"search partition, fewer than {MIN_PER_CLASS} in a class")
            return None
        value = A.auc(refused, complied)
        h_labels = labelled[:len(harmful)]
        h_ref = [m for m, (_, r) in zip(margins[:len(harmful)], h_labels, strict=True) if r]
        h_com = [m for m, (_, r) in zip(margins[:len(harmful)], h_labels, strict=True) if not r]
        self.agreement_detail["auc_pooled"] = value
        self.agreement_detail["auc_within_harmful"] = (
            A.auc(h_ref, h_com) if len(h_ref) >= MIN_PER_CLASS and len(h_com) >= MIN_PER_CLASS
            else None)
        return value if math.isfinite(value) else None


#: Random block sets of the same size as the claimed one, to say what removing ANY blocks of that
#: size does to generated refusals.
PREDICTIVE_RANDOM_SETS = 5


def predictive_check(subject, parts, exceeding, direction, ranking, *, seed=0, conv=A.DEFAULT):
    """Does the ranking predict what happens to the replies the model actually writes?

    The first token margin is a proxy, so this is where the claim meets the behaviour. Remove the
    direction from the claimed blocks while the model GENERATES, on requests the ranking never saw,
    and compare the fall in refusals with removing it from blocks of the same number chosen at
    random from the rest. Supported only if the claimed blocks do better than those at the 95%
    level, requests resampled whole. A failure is a finding and is reported as one.
    """
    nb = subject.n_blocks
    if "confirm" not in parts:
        return {"ran": False, "why": "no fresh requests were set aside for the check"}
    claimed = sorted(exceeding)
    others = [k for k in range(nb) if k not in claimed]
    if not claimed or len(others) < len(claimed):
        return {"ran": False, "why": f"{len(claimed)} claimed block(s) leave too few others of "
                                     f"the same number to compare against"}
    gen = torch.Generator().manual_seed(seed + 104729)
    sets = [sorted(others[i] for i in torch.randperm(len(others), generator=gen)[:len(claimed)])
            for _ in range(PREDICTIVE_RANDOM_SETS)]
    harmful, harmless = parts["confirm"]["harmful"], parts["confirm"]["harmless"]

    def flags(arm_, blocks):
        return torch.tensor(subject.refusal_flags(list(arm_.prompts), direction, blocks),
                            dtype=torch.float64)
    base = flags(harmful, ())
    top = flags(harmful, claimed)
    random_ = [flags(harmful, b) for b in sets]
    everything = flags(harmful, range(nb))
    drop_top = base - top
    drop_random = torch.stack([base - r for r in random_]).mean(0)
    contrast = drop_top - drop_random
    point, boot = A.bootstrap_effects(contrast[None], torch.zeros_like(contrast)[None],
                                      harmful.groups, harmful.groups, conv.resamples,
                                      torch.Generator().manual_seed(seed + 1))
    lo, hi = A.interval(boot, conv.alpha)
    side = {"none": float(flags(harmless, ()).mean()),
            "claimed": float(flags(harmless, claimed).mean()),
            "all_blocks": float(flags(harmless, range(nb)).mean())}
    return {
        "ran": True, "supported": bool(float(lo[0]) > 0.0), "blocks": claimed,
        "random_sets": sets, "n_requests": len(harmful.prompts),
        "harmful_refusal_rate": {"none": float(base.mean()), "claimed": float(top.mean()),
                                 "random_same_size_mean": float(torch.stack(random_).mean()),
                                 "all_blocks": float(everything.mean())},
        "contrast": {"drop_claimed_minus_drop_random": float(point[0]),
                     "ci": [float(lo[0]), float(hi[0])]},
        "harmless_refusal_rate": side,
        "rule": "supported when the lower end of the interval on (drop with the claimed blocks "
                "minus the mean drop with random blocks of the same number) is above zero",
    }
