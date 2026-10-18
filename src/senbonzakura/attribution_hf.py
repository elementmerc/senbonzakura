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
from . import hubmessage, loadreport, metrics, modelmap
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

#: THE PREFIX CACHE. A condition whose lowest edited block is k changes nothing below k, so the
#: blocks below it would compute exactly what they computed for the unedited pass. Their outputs
#: are saved once per batch (on the processor, not the graphics card) and replayed, and only blocks
#: k and up run again. About a third fewer block passes over a full sweep, with the same numbers.
#: Past this many saved bytes a batch is simply computed in full, which is slower and identical.
PREFIX_CACHE_BYTES = 2 * 1024 ** 3
#: The replay is checked against a full pass the first time it is used from each block, because an
#: architecture that reads a block's output anywhere but the next block would make it wrong, and
#: only for some starting blocks. One extra batch per block, against tens of conditions a sweep.
#: Logits further apart than this switch the cache off for that subject, and the run says so.
PREFIX_CHECK_ATOL = 1e-3


class _Replay(torch.nn.Module):
    """Stands in for a block below the first edited one, handing on what the blocks below wrote."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, *_args, **_kwargs):
        return self.value


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


class SharedReadouts:
    """Last position logits at a set of token ids, kept per condition and shared between arms.

    The two readout arms differ only in which tokens they read, and the forward pass that produces
    those logits is identical, so computing it once per condition halves the run without changing a
    single number. The key is the exact prompts, the exact direction and the exact blocks, so an
    arm asking something the other did not simply misses and computes it.
    """

    def __init__(self, ids):
        self.ids = torch.as_tensor(sorted({int(t) for t in ids}))
        self._store = {}
        self.hits = self.misses = 0

    def fetch(self, prompts, direction, blocks, compute):
        # Order matters (the readouts come back in prompt order), and a hash that can collide would
        # hand one question another's answer, so the key is a cryptographic digest of the exact
        # prompts in order.
        key = (hashlib.sha256("\x00".join(prompts).encode("utf-8")).hexdigest(),
               None if direction is None else hashlib.sha256(
                   direction.detach().float().cpu().numpy().tobytes()).hexdigest(),
               tuple(blocks))
        if key in self._store:
            self.hits += 1
        else:
            self.misses += 1
            self._store[key] = compute()
        return self._store[key]


def share_readouts(subjects):
    """Let subjects that have all been prepared share their forward passes. Returns the cache."""
    ready = [s for s in subjects if s.refusal_ids is not None]
    shared = SharedReadouts([t for s in ready for t in
                             (*s.refusal_ids.tolist(), *s.comply_ids.tolist())]) if ready else None
    for s in ready:
        s.use_shared(shared)
    return shared


def load(model_id, *, device="cuda", dtype=torch.bfloat16, revision=None, chat_template=None, log=None):
    """The model, its tokenizer and the identity a result must carry.

    THE PROMPT FORMAT IS CHECKED AND RECORDED HERE, BY THE SAME CODE `score` USES. A model's own template, or the one
    `chat_template` names (a path, or a bundled name), must render a chat prompt, and the provenance (its source and a
    digest of it) rides in the identity, so the attribution artefact says which format produced its numbers exactly
    as `score.json` does. A model with no template and none supplied is refused here.
    """
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # A missing or malformed file fails as OSError or ValueError deep inside transformers, which used
    # to reach the person as a traceback; it stops the run in a sentence instead.
    try:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    except (OSError, ValueError) as e:
        detail = " ".join(hubmessage.useful_lines(e)) or type(e).__name__
        raise A.AttributionError(f"the model '{model_id}' could not be loaded: {detail}") from e
    # Before the weights are read: a model that cannot be measured at all should say so in seconds, not after a load.
    from .cli import ensure_chat_template
    try:
        tok.senbon_chat_template = ensure_chat_template(tok, chat_template, log)
    except SystemExit as e:
        raise A.AttributionError(str(e.code)) from e
    try:
        model, info = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, revision=revision,
                                                           output_loading_info=True)
    except (OSError, ValueError) as e:
        detail = " ".join(hubmessage.useful_lines(e)) or type(e).__name__
        raise A.AttributionError(f"the model '{model_id}' could not be loaded: {detail}") from e
    refusal = loadreport.missing_weights(model_id, info)
    if refusal:
        raise A.AttributionError(refusal)
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
        "chat_template": tok.senbon_chat_template,
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


def fresh_requests(track_dir, used_groups, *, n, seed):
    """`n` harmful measure requests that no earlier step used, one prompt each, and their digest.

    Drawn by its own generator, so asking for them cannot disturb the draws that the original
    partitions came from. Returns the arm and a description that never contains a prompt.
    """
    from pathlib import Path

    from . import track
    harmful, _ = track.load_partitions(Path(track_dir))
    groups = track.group_by_request(harmful["measure"])
    keys = sorted(k for k in groups if k not in set(used_groups))
    if len(keys) < n:
        raise A.AttributionError(f"only {len(keys)} unused requests remain in the measure "
                                 f"partition, fewer than the {n} asked for.")
    gen = torch.Generator().manual_seed(seed + 9001)
    chosen = [keys[i] for i in torch.randperm(len(keys), generator=gen).tolist()[:n]]
    picks = [groups[k][int(torch.randint(len(groups[k]), (1,), generator=gen))] for k in chosen]
    return A.arm(picks, chosen), {"n": n, "unused_requests_available": len(keys),
                                  "digest": digest(picks)}


class HFSubject:
    """A model presented to the attribution engine, for one behaviour at a time.

    `label` is what decides, from a greedy reply, whether that reply SHOWS the behaviour. It is
    the one behaviour-specific thing in this class and everything else is general: the token sets
    are learned from whatever `label` says, the readout is the log odds between them, and the
    agreement is measured against the same labels. It defaults to the refusal ruler, which is the
    behaviour this module was written for, and the toxicity channel passes `toxicity.is_rude`.

    IT IS A PARAMETER RATHER THAN A SUBCLASS because the alternative was a copy of this file with
    one function swapped, and a second copy of the readout machinery is a second place for a
    readout defect to be fixed in only one of them. The engine already treats the readout as a
    proxy to be validated rather than trusted, so a second behaviour clears the same bar as the
    first by construction: a label function whose readout cannot predict generation reports
    `readout_invalid` exactly as refusal would.
    """

    def __init__(self, model, tok, *, batch=16, arm=ARM_DISJOINT, log=print, label=None,
                 prefix_cache_bytes=None):
        if arm not in ARM_DECLARATIONS:
            raise A.AttributionError(f"{arm!r} is not a readout arm; the arms are "
                                     f"{sorted(ARM_DECLARATIONS)}.")
        self.model, self.tok, self.batch, self.log, self.arm = model, tok, batch, log, arm
        self.label = metrics.is_refusal if label is None else label
        if not callable(self.label):
            raise A.AttributionError(
                f"the behaviour label is {self.label!r}, which is not callable. It must be a "
                f"function from one reply's text to True when the reply shows the behaviour.")
        self._standing = []
        self._encoded = {}
        self._shared = None
        self.layers = modelmap.stack_or_raise(model)
        self.n_blocks = len(self.layers)
        self.refusal_ids = self.comply_ids = None
        self.readout_why_not = None
        self.agreement_detail = {}
        self._replies = {}
        self._prefix_cap = PREFIX_CACHE_BYTES if prefix_cache_bytes is None else prefix_cache_bytes
        self._prefix = {}
        self._prefix_bytes = 0
        self._prefix_full = False
        self._prefix_checked = set()
        #: None while the cache may be used, else why it was switched off for this subject.
        self.prefix_off = None if self._prefix_cap > 0 else "switched off by the caller"
        self.prefix_detail = {"replayed_batches": 0, "full_batches": 0, "check_max_abs_diff": None}

    # ------------------------------------------------------------------ encoding and capture
    def _encode(self, prompts):
        key = tuple(prompts)
        if key not in self._encoded:
            text = [render_chat(self.tok, p) for p in prompts]
            self._encoded[key] = self.tok(text, return_tensors="pt", padding=True,
                                          add_special_tokens=False).to(self.model.device)
        return self._encoded[key]

    def padded_tokens(self, prompts):
        """How many token positions these prompts take once rendered and padded together."""
        return int(self._encode(prompts)["attention_mask"].shape[1])

    def edit_standing(self, direction, blocks):
        """Keep `direction` removed from `blocks` for every later call, as an edited model would be.

        An in-memory stand-in for a checkpoint that has the behaviour removed, used as a control.
        It is a hook and not a weight edit, so it says nothing about what the real bake does.
        """
        self._standing = [(direction, tuple(blocks))]
        self._replies.clear()
        # The saved outputs were made under the old edit, so none of them holds any longer.
        self._prefix.clear()
        self._prefix_bytes = 0
        self._prefix_full = False

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
        return [(int(new[i, 0]), bool(self.label(texts[i]))) for i in range(len(chunk))]

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

    @torch.no_grad()
    def replies(self, prompts, direction=None, blocks=()):
        """The greedy reply text for each prompt, with `blocks` ablated while it is written.

        The flag methods throw the text away, which is right for them and wrong for a judge that
        has to be validated: an agreement figure needs the replies a human can read beside the
        verdicts. Not cached, for the same reason `refusal_flags` is not: the point is to see what
        a different condition does to the same prompts.
        """
        out = []
        for chunk in self._batches(prompts):
            enc = self._encode(chunk)
            with self._under_standing_edit(), A.ablating(self.layers, direction, blocks):
                got = self.model.generate(**enc, max_new_tokens=REPLY_TOKENS, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            new = got[:, enc["input_ids"].shape[1]:]
            out.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return out

    def refusal_flags(self, prompts, direction=None, blocks=()):
        """Whether each prompt's greedy reply shows the behaviour, with `blocks` ablated as it is written.

        "Shows the behaviour" is whatever this subject's `label` says, which is the refusal ruler
        unless a caller passed another one. The name is refusal's because refusal is what the
        method was written for and what every caller in the package asks it for.

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
        prompts = list(prompts)
        union, ref_pos, com_pos = self._union()
        got = (self._shared.fetch(prompts, direction, blocks,
                                  lambda: self._logits_at(prompts, direction, blocks, union))
               if self._shared is not None else self._logits_at(prompts, direction, blocks, union))
        return (torch.logsumexp(got[:, ref_pos], 1)
                - torch.logsumexp(got[:, com_pos], 1)).double()

    def use_shared(self, shared):
        self._shared = shared

    def _union(self):
        """The token ids to read, and where each arm's tokens sit among them."""
        ids = (self._shared.ids if self._shared is not None
               else torch.cat([self.refusal_ids, self.comply_ids]).cpu())
        where = {int(t): i for i, t in enumerate(ids.tolist())}
        return (ids, torch.tensor([where[int(t)] for t in self.refusal_ids.tolist()]),
                torch.tensor([where[int(t)] for t in self.comply_ids.tolist()]))

    @torch.no_grad()
    def _logits_at(self, prompts, direction, blocks, ids):
        out = []
        for chunk in self._batches(prompts):
            logits = self._chunk_logits(chunk, direction, tuple(blocks))
            out.append(logits[:, ids.to(logits.device)].cpu())
        return torch.cat(out)

    def _full(self, enc, direction, blocks):
        with self._under_standing_edit(), A.ablating(self.layers, direction, blocks):
            return last_token_logits(self.model, enc, log=self.log)

    def _chunk_logits(self, chunk, direction, blocks):
        """One batch's last position logits under the condition, from the prefix cache where it can."""
        enc = self._encode(chunk)
        if self.prefix_off is not None or (blocks and min(blocks) == 0):
            self.prefix_detail["full_batches"] += 1
            return self._full(enc, direction, blocks)
        key = tuple(chunk)
        entry = self._prefix.get(key)
        if entry is None:
            if self._prefix_full:
                self.prefix_detail["full_batches"] += 1
                return self._full(enc, direction, blocks)
            logits, outputs = self._fill(enc)
            entry = self._keep(key, logits, outputs)
            if entry is None:
                self.prefix_detail["full_batches"] += 1
                return logits if not blocks else self._full(enc, direction, blocks)
        if not blocks:
            return entry["logits"]
        got = self._replayed(entry, enc, direction, blocks)
        if min(blocks) not in self._prefix_checked:
            full = self._full(enc, direction, blocks)
            diff = float((got.float().cpu() - full.float().cpu()).abs().max())
            seen = self.prefix_detail["check_max_abs_diff"]
            self.prefix_detail["check_max_abs_diff"] = diff if seen is None else max(seen, diff)
            if not diff <= PREFIX_CHECK_ATOL:
                self._switch_prefix_off(
                    f"the replayed blocks gave logits {diff:.3g} away from a full pass, so this "
                    f"architecture uses a block's output somewhere besides the next block")
                self.prefix_detail["full_batches"] += 1
                return full
            self._prefix_checked.add(min(blocks))
        self.prefix_detail["replayed_batches"] += 1
        return got

    def _fill(self, enc):
        """The unedited-condition logits, and what every block wrote, from one pass."""
        written = {}
        # Hooks run in the order they were attached, so these go on after the standing edit's: what
        # they save is each block's output with the edit already applied, as a full pass would see it.
        with self._under_standing_edit():
            handles = [layer.register_forward_hook(
                lambda _m, _a, out, k=k: written.__setitem__(k, out)) for k, layer in enumerate(self.layers)]
            try:
                logits = last_token_logits(self.model, enc, log=self.log)
            finally:
                for handle in handles:
                    handle.remove()
        return logits, [written.get(k) for k in range(self.n_blocks)]

    def _keep(self, key, logits, outputs):
        """Save one batch's block outputs if they fit under the cap; None when they don't."""
        saved = []
        size = 0
        for out in outputs:
            if out is None:
                self._switch_prefix_off("a block did not run in the saved pass")
                return None
            tail = out[1:] if isinstance(out, tuple) else None
            if tail is not None and any(isinstance(t, torch.Tensor) for t in tail):
                self._switch_prefix_off("this architecture's blocks hand on more than the stream")
                return None
            stream = A.unwrap(out).detach().to("cpu", copy=True)
            size += stream.numel() * stream.element_size()
            saved.append((stream, tail))
        if self._prefix_bytes + size > self._prefix_cap:
            self.log(f"  prefix cache full at {len(self._prefix)} batch(es) "
                     f"({self._prefix_bytes / 2 ** 20:.0f} MiB); later batches run in full")
            self._prefix_full = True
            return None
        self._prefix_bytes += size
        self._prefix[key] = {"logits": logits.detach().cpu(), "outputs": saved}
        return self._prefix[key]

    def _replayed(self, entry, enc, direction, blocks):
        """The condition's logits with the blocks below its lowest edited block replayed, not run.

        The standing edit's hooks are attached first, to the real blocks, and only then are the
        lower blocks swapped out: its effect on them is already in what was saved, and a hook on a
        stand-in would read a write from an input the stand-in never used.
        """
        k = min(blocks)
        stream, tail = entry["outputs"][k - 1]
        stream = stream.to(self.model.device)
        value = (stream, *tail) if tail is not None else stream
        originals = [self.layers[j] for j in range(k)]
        with self._under_standing_edit():
            try:
                for j in range(k):
                    self.layers[j] = _Replay(value)
                with A.ablating(self.layers, direction, blocks):
                    return last_token_logits(self.model, enc, log=self.log)
            finally:
                for j, layer in enumerate(originals):
                    self.layers[j] = layer

    def _switch_prefix_off(self, why):
        self.prefix_off = why
        self._prefix.clear()
        self._prefix_bytes = 0
        self.log(f"  prefix cache off for this model: {why}; every condition runs in full")

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
