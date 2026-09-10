#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""One coherence ruler, applied to any model after the fact.

WHY THIS EXISTS

Both this project and Heretic report a KL divergence, both compute it the same way, and the two
numbers still cannot be put in one column. On the 2026-08-12 head-to-head Heretic reported 0.0014
to 0.0032 and we reported 0.157 to 0.212, a hundredfold apart, and no honest sentence could be
written across them: each tool measured its own model, on its own slice of prompts, during its own
search. Reading those as a comparison is the mistake four claims were withdrawn for on 2026-08-05.

`senbonzakura compass` already solved this for harm recognition. One instrument, run by us, after
the fact, on held-out prompts, over every model whoever produced it. This is the same idea for
coherence, and it is the missing third axis of the benchmark.

WHAT IT MEASURES

The model's first-token probability distribution on ordinary harmless prompts, against the same
distribution from the unmodified base model:

    KL(base || candidate), summed over the vocabulary, averaged over prompts

Zero means the edited model predicts exactly what the original did. It rises as the edit changes
the model's behaviour on prompts that have nothing to do with refusal, which is the definition of
collateral damage.

It is deliberately the SAME arithmetic as the search's own `kl_vs_orig`, importing the same chat
renderer and the same logits helper rather than restating either, because two copies of this
formula is how two tools come to disagree while both claiming to measure coherence.

WHY THE BASE LOGPROBS ARE CACHED

Scoring ten models means the base model would otherwise be loaded and re-run ten times for an
answer that cannot change. The cache records which base, which prompts and which chat template it
was built from, and is refused if any of those differ: a cache keyed on nothing is how you score
ten models against the wrong reference and never find out.
"""
import argparse
import hashlib
import json
import os

import torch

from .cli import load_model_and_tokenizer, load_tokenizer, loader_parser
from .crashsafe import atomic_write

# The measurement itself, from the torch-only module the sealed best-of-N pass also uses, so the
# published drift figure and the figure that selects an arm cannot drift apart.
from .firsttoken import first_token_logprobs, kl, kl_per_prompt

CACHE_SCHEMA = "senbonzakura-drift-base/1"


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura.drift",
        description="Measure how far an edited model's predictions have drifted from its base.",
        parents=[loader_parser()])
    ap.add_argument("--base", required=True,
                    help="the unmodified model the candidate was edited from")
    ap.add_argument("--prompts", required=True,
                    help="one harmless prompt per line; the same file for every model compared")
    ap.add_argument("--out", required=True, help="results json path")
    ap.add_argument("--label", default="")
    ap.add_argument("--batch", type=int, default=16,
                    help="prompts per forward pass. Reduction order depends on it, so a "
                         "comparison must hold it fixed across models")
    ap.add_argument("--base-cache", default=None,
                    help="where to keep the base model's distributions, so scoring N models "
                         "loads the base once rather than N times")
    return ap


def read_prompts(path):
    with open(path, encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        raise SystemExit(f"drift: {path} holds no prompts, so there is nothing to measure on.")
    return prompts


def fingerprint(base, prompts, template):
    """What the cached base distributions are only valid for.

    The template is in here because the same model gives different first-token distributions under
    different chat formats, and this project has already published numbers where a configuration
    was selected under one prompt format and reported under another.
    """
    h = hashlib.sha256()
    h.update(f"{base}\n".encode())
    h.update(f"{template}\n".encode())
    for p in prompts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


#: Below this, a KL computed from bf16 logits is not precise enough to quote.
#:
#: MEASURED, not assumed (2026-08-17). The same edit scored in float32 and in bfloat16, on matched
#: inputs, at four edit strengths:
#:
#:     KL 0.00074   bf16 differs by 0.7%
#:     KL 0.00290   bf16 differs by 0.2%
#:     KL 0.01039   bf16 differs by 0.0%
#:     KL 0.02935   bf16 differs by 0.0%
#:
#: So above roughly 1e-3 the dtype is irrelevant and the published figures (~0.06) are unaffected.
#: Below it the gap grows as the KL shrinks, because bf16 carries eight bits of mantissa and the
#: quantity being summed gets smaller than its resolution. A drift of 0.0005 quoted from a bf16 run
#: is a number about the arithmetic rather than about the model.
#:
#: This exists because a second implementation of this measurement turned up (`kl_llama.py`) which
#: used float32 throughout. Its formula was identical, so the two agree wherever the figure is large
#: enough to matter, and that agreement is only knowable because it was checked.
BF16_KL_FLOOR = 1e-3

#: Bootstrap replicates for the drift interval. The same count the compass uses, so the two
#: intervals in one report are built the same way and a reader can compare their widths.
BOOTSTRAP_DRAWS = 2000


def logits_dtype_of(model):
    """The dtype this model computes in, or "unknown".

    Read off the model rather than assumed, because the loader's choice is what decides whether a
    small figure is a measurement or an artefact of the arithmetic. "unknown" is a real answer and
    is recorded as one: a model shape this cannot read is not evidence of reduced precision, and
    flagging on ignorance would fire on every stub while telling nobody anything.
    """
    dt = getattr(model, "dtype", None)
    if dt is None:
        try:
            dt = next(model.parameters()).dtype
        except (AttributeError, StopIteration, TypeError):
            return "unknown"
    return str(dt).replace("torch.", "")


def precision_verdict(value, dtype_name, *, floor=BF16_KL_FLOOR):
    """Is this KL large enough for the precision it was computed in? Returns (ok, note).

    Pure, so the boundary is testable without a model. `ok` False does not mean the number is
    wrong; it means it is smaller than the arithmetic that produced it can resolve, and quoting it
    to four decimal places would claim a precision nothing supports.
    """
    low = str(dtype_name).lower()
    reduced = any(t in low for t in ("bfloat16", "bf16", "float16", "fp16"))
    if not reduced or value >= floor:
        return True, None
    return False, (
        f"this drift of {value:.2e} was computed from {dtype_name} logits, and below {floor:.0e} "
        f"that arithmetic cannot resolve the quantity being summed: the same edit measured in "
        f"float32 and bfloat16 diverges by under 0.1% at 0.01 and by 0.7% at 0.0007, growing as "
        f"the figure shrinks. Treat it as 'below {floor:.0e}' rather than as a value, or re-run "
        f"the pair in float32 if the exact number matters.")


def kl_interval(base_lp, cand_lp, *, seed=0, draws=BOOTSTRAP_DRAWS):
    """A seeded bootstrap interval over PROMPT sampling, for the drift figure.

    THE AXIS EVERY HEADLINE COMPARISON TURNS ON, AND IT HAD NO UNCERTAINTY AT ALL. The K=1
    against K=2 result, the head-to-head's "half the collateral damage" claim and `validate`'s
    matched-refusal ranking are all decided on drift, and it was reported as a bare number while
    the compass and the capability score both shipped seeded prompt-level bootstraps. `validate`
    even asserts that a five-point gap is "outside the run-to-run noise of a 64-prompt KL
    estimate", a noise level nothing had ever measured.

    The per-prompt values are already in hand, so this costs no GPU: it resamples the prompts,
    which is the uncertainty that actually dominates, exactly as the compass does. Re-running on
    the same machine would measure floating-point reduction order, which is a reassuring number
    about the wrong thing.
    """
    per = kl_per_prompt(base_lp, cand_lp)
    n = per.shape[0]
    if n < 2:
        return None, None
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    flat = per.detach().to("cpu").float()
    means = torch.stack([flat[torch.randint(n, (n,), generator=g)].mean() for _ in range(draws)])
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975]))
    return float(lo), float(hi)


def load_base_logprobs(a, prompts, fp, log=print):
    """The base model's distributions, from cache when the cache is genuinely for this question."""
    cache = a.base_cache
    if cache and os.path.isfile(cache):
        # weights_only=True: the payload is {str: str|int|Tensor}, which it handles, and the
        # path comes from a CLI flag, so the permissive mode was an arbitrary-code-execution
        # surface bought for nothing.
        doc = torch.load(cache, map_location="cpu", weights_only=True)
        if (doc.get("schema") == CACHE_SCHEMA and doc.get("fingerprint") == fp
                and int(doc.get("batch", -1)) == int(a.batch)):
            log(f"drift: reusing the base distributions cached at {cache}")
            return doc["logprobs"]
        # LOUD, not silent. A stale cache here means every model gets compared to the wrong
        # reference and the whole table is wrong in the same direction, which is undetectable
        # from the output.
        log(f"drift: the cache at {cache} was built for a different base, prompt set, chat "
            f"template or batch size. Recomputing.")

    base_model, base_tok = load_model_and_tokenizer(
        a.base, device=a.device, load_in_4bit=a.load_in_4bit,
        trust_remote_code=a.trust_remote_code)
    lp = first_token_logprobs(base_model, base_tok, prompts, batch=a.batch)
    del base_model
    if torch.cuda.is_available():       # pragma: no cover - depends on the machine
        torch.cuda.empty_cache()
    if cache:
        with atomic_write(cache, binary=True) as f:
            torch.save({"schema": CACHE_SCHEMA, "fingerprint": fp,
                        "batch": int(a.batch), "logprobs": lp}, f)
        log(f"drift: cached the base distributions at {cache}")
    return lp


def main(argv=None):
    a = build_parser().parse_args(argv)
    prompts = read_prompts(a.prompts)

    # ORDER IS LOAD-BEARING, and it used to be the other way round. The candidate was loaded
    # first and `load_base_logprobs` then loaded the base while the candidate was still on the
    # device, so the peak was two full bf16 copies. On the 6 GB card this project is sized for
    # that does not fit, and `head-to-head/best_of_n_heretic.py` states the same constraint and
    # designs around it. A warm --base-cache hid it, so only the FIRST run against a given base
    # was ever cold, and that is the run the published coherence figure comes from.
    #
    # Only the tokenizer is needed to fingerprint the question, so the base is measured and freed
    # before the candidate arrives, and the peak is one model.
    tok = load_tokenizer(a.model, trust_remote_code=a.trust_remote_code)
    template = getattr(tok, "chat_template", None) or ""
    fp = fingerprint(a.base, prompts, template)

    base_lp = load_base_logprobs(a, prompts, fp)
    if base_lp.shape[0] != len(prompts):
        raise SystemExit(
            f"drift: the base distributions cover {base_lp.shape[0]} prompts and this run has "
            f"{len(prompts)}. Refusing to compare rows that are not the same prompts.")

    cand, tok = load_model_and_tokenizer(
        a.model, device=a.device, load_in_4bit=a.load_in_4bit,
        trust_remote_code=a.trust_remote_code)
    cand_lp = first_token_logprobs(cand, tok, prompts, batch=a.batch)
    value = kl(base_lp, cand_lp)
    kl_lo, kl_hi = kl_interval(base_lp, cand_lp, seed=int(getattr(a, "seed", 0) or 0))

    dtype_name = logits_dtype_of(cand)
    precise, note = precision_verdict(value, dtype_name)

    res = {
        "label": a.label,
        "model": a.model,
        "base": a.base,
        # Stated in the artefact, not left to the reader: the whole reason this module exists is
        # that two KL figures on different slices were put in one column.
        "prompts": a.prompts,
        "n_prompts": len(prompts),
        "batch": int(a.batch),
        "fingerprint": fp,
        "kl": value,
        # WHAT PRECISION SUPPORTS THAT FIGURE. Recording the dtype and the verdict means a reader
        # does not have to know about mantissa widths to avoid quoting a number this run cannot
        # support. `kl_llama.py` computed the same quantity in float32; the two agree above the
        # floor and diverge below it.
        "logits_dtype": dtype_name,
        "precision_ok": precise,
        "precision_floor": BF16_KL_FLOOR,
        "precision_note": note,
        "instrument": "senbonzakura.drift, KL(base||candidate) on first-token distributions",
        # An interval over prompt sampling, from the per-prompt values this already computed and
        # then averaged away. None when there is one prompt, because an interval from one
        # observation is a decoration.
        "kl_ci": None if kl_lo is None else [kl_lo, kl_hi],
        "kl_ci_method": f"seeded percentile bootstrap over prompts, {BOOTSTRAP_DRAWS} draws",
    }
    with atomic_write(a.out) as f:
        json.dump(res, f, indent=2)
    if not precise:
        # In the same breath as the number, because this is exactly the caveat that gets lost
        # between an artefact and a table.
        print(f"DRIFT_BELOW_PRECISION {a.label}: {note}")
    span = "" if kl_lo is None else f" [{kl_lo:.4f}, {kl_hi:.4f}]"
    print(f"DRIFT_DONE {a.label} kl={value:.4f}{span} n={len(prompts)} batch={a.batch} "
          f"dtype={dtype_name}{'' if precise else ' PRECISION-LIMITED'}")
    return res


if __name__ == "__main__":   # pragma: no cover
    # Through `exit_status` so `python -m senbonzakura.<module>` reports what the console
    # script reports. A bare `main()` discards the return, which is how `doctor` printed
    # nine failed checks and exited 0; `sys.exit(main())` alone breaks the other way for
    # the commands that return their result rather than a status.
    import sys

    from .entry import exit_status
    sys.exit(exit_status(main()))
