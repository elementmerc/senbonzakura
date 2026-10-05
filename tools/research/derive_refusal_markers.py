#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Derive a lexical refusal marker list from our own generations, clean of Heretic.

WHY THIS EXISTS. `metrics.HERETIC_MARKERS` is copied verbatim from Heretic under AGPL, so it is
the one thing standing between this project and a licence it can offer commercially. The point of
this script is to produce a replacement that was MEASURED rather than transcribed.

CLEAN ROOM, AND HOW IT IS KEPT. This file never reads `HERETIC_MARKERS`, never imports it, and
never uses `_heretic_norm`. It calls `heretic_keyword_rate` as an opaque function at the very end,
to compare two numbers, which is a measurement of behaviour rather than a reading of source. Its
own normalisation is lowercase plus whitespace collapse, which is the obvious thing and not
anybody's invention.

THE CONTROL, AND RUN IT WHENEVER THE DERIVATION CHANGES:

    tools/research/derive_refusal_markers.py <generations.jsonl> --control shuffled

That permutes the refusal labels and derives against the permutation, so the markers are selected
against noise. **It is not a formality.** With the labels shuffled this method still reaches an
in-sample kappa of 0.5923, because a greedy search over every n-gram in 259 replies can find
phrases that separate any labelling at all. So the real run's in-sample 0.9920 is not a result: the
floor for pure noise is 0.59, and only a HELD-OUT comparison between the real and shuffled runs
says whether anything was measured.

GROUND TRUTH IS OUR OWN SEMANTIC METRIC, `metrics.is_refusal`, not a human label. That is a real
limitation and it is stated here rather than hidden: a list distilled from `is_refusal` can at best
approximate `is_refusal` cheaply, and cannot be independent evidence that `is_refusal` is right.
What it CAN be is an honest lexical axis whose disagreement with the semantic metric is measured
and published, which is exactly what the keyword term in `knee_scalar` is for.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from senbonzakura import metrics

WORD = re.compile(r"[a-z']+")


def norm(text: str) -> str:
    """Lowercase and collapse whitespace. Deliberately the dullest possible normalisation."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def ngrams(text: str, lo: int = 2, hi: int = 6) -> set[str]:
    words = WORD.findall(norm(text))
    out = set()
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            out.add(" ".join(words[i:i + n]))
    return out


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        # `strip_reasoning` exists because a model that emits a think block states its verdict
        # inside it, and reading the marker there measures the deliberation and not the answer.
        r["reply"] = metrics.strip_reasoning(r["generation"])
        r["is_refusal"] = metrics.is_refusal(r["reply"])
    return rows


def candidates(rows: list[dict], *, head_chars: int, min_support: int) -> Counter:
    """Count how many REFUSING replies contain each n-gram, within the head window.

    HEAD WINDOW, NOT THE WHOLE REPLY. This project measured its own markers at a median of
    character 306 on defended replies, and a 2,200 character reply has room for a phrase like
    "I cannot" to appear deep inside a perfectly compliant answer. Matching only the opening
    buys precision for free and matches what a refusal actually is: the model declining up front.
    """
    c = Counter()
    for r in rows:
        if r["is_refusal"]:
            c.update(ngrams(r["reply"][:head_chars]))
    return Counter({k: v for k, v in c.items() if v >= min_support})


def score_marker(rows: list[dict], marker: str, head_chars: int) -> tuple[int, int]:
    """(refusals hit, compliant replies wrongly hit)."""
    hit_ref = hit_ok = 0
    for r in rows:
        if marker in norm(r["reply"][:head_chars]):
            if r["is_refusal"]:
                hit_ref += 1
            else:
                hit_ok += 1
    return hit_ref, hit_ok


def kappa(rows: list[dict], markers: list[str], head_chars: int) -> dict:
    """Cohen's kappa of the marker list against our semantic metric, plus the raw cells.

    KAPPA RATHER THAN ACCURACY, because this project refuses a grader that cannot show it beats
    chance, and a list that fires on everything scores well on accuracy alone.
    """
    tp = fp = fn = tn = 0
    for r in rows:
        head = norm(r["reply"][:head_chars])
        fired = any(m in head for m in markers)
        if r["is_refusal"] and fired:
            tp += 1
        elif r["is_refusal"]:
            fn += 1
        elif fired:
            fp += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "kappa": k,
            "precision": prec, "recall": rec, "rate": (tp + fp) / n}


def select(rows: list[dict], *, head_chars: int, min_support: int,
           max_fp_per_marker: int, limit: int) -> list[str]:
    """Greedy: take the marker that adds the most uncaught refusals per false positive.

    GREEDY AND NOT EXHAUSTIVE, on purpose. The search space is every n-gram in every refusal and
    the objective is not convex, so an exhaustive answer would be a different project. Greedy with
    a stated rule is reproducible and inspectable, which matters more here than optimal.
    """
    pool = candidates(rows, head_chars=head_chars, min_support=min_support)
    chosen: list[str] = []
    uncaught = {i for i, r in enumerate(rows) if r["is_refusal"]}
    while len(chosen) < limit and uncaught:
        best = None
        for m in pool:
            if m in chosen:
                continue
            _, hit_ok = score_marker(rows, m, head_chars)
            if hit_ok > max_fp_per_marker:
                continue
            gained = {i for i in uncaught
                      if m in norm(rows[i]["reply"][:head_chars])}
            if not gained:
                continue
            # Prefer more refusals caught, then fewer false positives, then the shorter phrase:
            # a short marker generalises to replies this corpus does not contain.
            key = (len(gained), -hit_ok, -len(m))
            if best is None or key > best[0]:
                best = (key, m, gained)
        if best is None:
            break
        _, m, gained = best
        chosen.append(m)
        uncaught -= gained
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("generations", type=Path, help="the JSONL written by score --save-generations")
    ap.add_argument("--head-chars", type=int, default=400,
                    help="match only the first N characters of a reply (default: 400)")
    ap.add_argument("--min-support", type=int, default=4,
                    help="an n-gram must appear in at least N refusals to be considered")
    ap.add_argument("--max-fp", type=int, default=1,
                    help="reject a marker that fires on more than N compliant replies")
    ap.add_argument("--limit", type=int, default=40, help="most markers to keep")
    ap.add_argument("--out", type=Path, help="write the chosen list here as JSON")
    ap.add_argument("--control", choices=("shuffled",),
                    help="BREAK THE COMPARISON ON PURPOSE. 'shuffled' permutes the refusal labels "
                         "before deriving, so the markers are selected against noise. A kappa that "
                         "survives that is not measuring refusal, it is measuring the method's "
                         "ability to fit any labelling, and the real run's number would mean "
                         "nothing. Run this whenever the derivation changes.")
    a = ap.parse_args(argv)

    rows = load(a.generations)
    if a.control == "shuffled":
        # Permute the LABELS and leave the replies alone, so the refusal rate is identical and the
        # only thing destroyed is which reply carries which label. Seeded, because a control whose
        # result moves run to run cannot be compared with the run it is controlling.
        import random as _random
        labels = [r["is_refusal"] for r in rows]

        # it MUST be seeded and reproducible, which is the opposite of what a CSPRNG gives.
        _random.Random(20261005).shuffle(labels)  # noqa: S311
        for r, lab in zip(rows, labels, strict=True):
            r["is_refusal"] = lab
        print("CONTROL: refusal labels shuffled. A high kappa here means the method fits noise.\n")
    n_ref = sum(r["is_refusal"] for r in rows)
    print(f"replies: {len(rows)}   our metric calls {n_ref} of them refusals "
          f"({n_ref / len(rows):.1%})")
    if n_ref < 20:
        print("REFUSING TO DERIVE: fewer than 20 refusals is not enough to measure a marker on.",
              file=sys.stderr)
        return 2

    markers = select(rows, head_chars=a.head_chars, min_support=a.min_support,
                     max_fp_per_marker=a.max_fp, limit=a.limit)
    print(f"\nderived {len(markers)} markers from the head {a.head_chars} characters:")
    for m in markers:
        hr, ho = score_marker(rows, m, a.head_chars)
        print(f"   {hr:4d} refusals  {ho:2d} false  {m!r}")

    ours = kappa(rows, markers, a.head_chars)
    print("\nTHE DERIVED LIST against " +
          ("SHUFFLED labels (this is the control, not a result):" if a.control
           else "our own semantic metric:"))
    print(f"   kappa {ours['kappa']:.4f}   precision {ours['precision']:.4f}   "
          f"recall {ours['recall']:.4f}   rate {ours['rate']:.4f}")
    print(f"   tp {ours['tp']}  fp {ours['fp']}  fn {ours['fn']}  tn {ours['tn']}")

    # The only contact with Heretic in this file, and it is a number rather than a list.
    inherited = metrics.heretic_keyword_rate([r["reply"] for r in rows])
    print("\nFOR COMPARISON, as rates only:")
    print(f"   our semantic metric   {n_ref / len(rows):.4f}")
    print(f"   the derived list      {ours['rate']:.4f}")
    print(f"   the inherited list    {inherited:.4f}")

    if a.out:
        a.out.write_text(json.dumps({
            "markers": markers,
            "head_chars": a.head_chars,
            "derived_from": str(a.generations),
            "replies": len(rows),
            "refusals_by_our_metric": n_ref,
            "against_our_metric": ours,
            "inherited_list_rate_for_comparison": inherited,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
