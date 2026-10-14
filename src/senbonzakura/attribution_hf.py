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


def build_partitions(track_dir, *, n_measure=200, seed=0):
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
        parts[name], digests[name] = {}, {}
        for side, rows in (("harmful", harmful[name]), ("harmless", harmless[name])):
            groups = track.group_by_request(rows)
            keys = sorted(groups)
            order = torch.randperm(len(keys), generator=gen).tolist()
            cap = n_measure if name == "measure" else len(keys)
            chosen = [keys[i] for i in order[:cap]]
            picks = [groups[k][int(torch.randint(len(groups[k]), (1,), generator=gen))]
                     for k in chosen]
            parts[name][side] = A.arm(picks, chosen)
            digests[name][side] = {"n": len(picks), "requests": len(groups),
                                   "rows_in_partition": len(rows), "digest": digest(picks)}
    return parts, digests


class HFSubject:
    def __init__(self, model, tok, *, batch=16, log=print):
        self.model, self.tok, self.batch, self.log = model, tok, batch, log
        self.layers = modelmap.stack_or_raise(model)
        self.n_blocks = len(self.layers)
        self.refusal_ids = self.comply_ids = None
        self.readout_why_not = None
        self.agreement_detail = {}
        self._replies = {}

    # ------------------------------------------------------------------ encoding and capture
    def _encode(self, prompts):
        text = [render_chat(self.tok, p) for p in prompts]
        return self.tok(text, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(self.model.device)

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
                self.model(**self._encode(chunk), use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            rows.append(torch.stack([seen[k] for k in range(self.n_blocks + 1)]))
        return torch.cat(rows, 1)

    # ------------------------------------------------------------------- the readout's tokens
    @torch.no_grad()
    def _reply(self, chunk):
        enc = self._encode(chunk)
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
        prompts = list(parts["fit"]["harmful"].prompts) + list(parts["fit"]["harmless"].prompts)
        refused, complied = {}, {}
        for first, was_refusal in self._label(prompts):
            bucket = refused if was_refusal else complied
            bucket[first] = bucket.get(first, 0) + 1
        both = set(refused) & set(complied)
        sensitivity = {}
        for floor in (1, 2, 3):
            sensitivity[str(floor)] = {
                "refusal_tokens": sum(1 for t, c in refused.items() if c >= floor and t not in both),
                "compliance_tokens": sum(1 for t, c in complied.items()
                                         if c >= floor and t not in both)}
        ref = sorted(t for t, c in refused.items() if c >= MIN_TOKEN_COUNT and t not in both)
        com = sorted(t for t, c in complied.items() if c >= MIN_TOKEN_COUNT and t not in both)
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
            "shared_tokens_dropped": len(both),
            "min_token_count": MIN_TOKEN_COUNT, "sensitivity_to_min_token_count": sensitivity,
            "refusal_openers_by_count": {self.tok.convert_ids_to_tokens([t])[0]: c
                                         for t, c in sorted(refused.items(), key=lambda x: -x[1])[:8]},
            "compliance_openers_by_count": {self.tok.convert_ids_to_tokens([t])[0]: c
                                            for t, c in sorted(complied.items(),
                                                               key=lambda x: -x[1])[:8]},
            "refusal_token_text": self.tok.convert_ids_to_tokens(ref),
            "compliance_token_text": self.tok.convert_ids_to_tokens(com[:16]),
            "fit_refused": sum(refused.values()), "fit_complied": sum(complied.values())}

    # ---------------------------------------------------------------------------- the readout
    @torch.no_grad()
    def readout(self, prompts, direction=None, blocks=()):
        if self.refusal_ids is None:
            raise A.AttributionError(self.readout_why_not or "prepare() has not been called")
        out = []
        for chunk in self._batches(prompts):
            with A.ablating(self.layers, direction, blocks):
                logits = last_token_logits(self.model, self._encode(chunk), log=self.log)
            out.append(torch.logsumexp(logits[:, self.refusal_ids], 1)
                       - torch.logsumexp(logits[:, self.comply_ids], 1))
        return torch.cat(out).double().cpu()

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
