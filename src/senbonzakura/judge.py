# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Before a judge grades anything, prove the judge works.

WHY THIS EXISTS BEFORE ANY JUDGE DOES

Every task in `capability` today is graded by code against a reference answer, and each one records
why that needs no judge. That runs out. Tool use, instruction following with open-ended output, and
anything agentic cannot be marked by string equality, and the usual answer is to ask a model. The
moment that happens, every number downstream inherits the judge's reliability, and this project has
already withdrawn results over an unvalidated grader once.

So the harness comes first, and it is deliberately hostile to the judge.

WHY AGREEMENT IS NOT THE NUMBER TO REPORT

A judge that answers "correct" to everything agrees 90% of the time with a set that is 90% correct.
That is the single most common way a judge looks good and is useless, and raw agreement cannot see
it. What sees it is per-class agreement, and Cohen's kappa, which asks how much better than chance
the judge did given how often each answer appears.

    judge says "correct" always, on a 90% correct set
        agreement  0.90     looks excellent
        kappa      0.00     it learned nothing
        recall on the wrong items  0.00     it never catches an error

A judge certified on agreement alone would pass. This refuses it.
"""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import MIN_REPORTABLE_N, wilson_interval

#: Agreement above chance, below which a judge is not usable for grading. 0.6 is the conventional
#: "substantial" mark and it is a floor rather than a target: a judge at 0.61 is one whose errors
#: still have to be reported beside anything it grades.
MIN_KAPPA = 0.6

#: A judge must catch at least this share of each class. Stated per class because the failure this
#: guards is a judge that is excellent on the common answer and blind to the rare one, which is
#: exactly the direction that flatters a model under test.
MIN_PER_CLASS_RECALL = 0.5


def cohens_kappa(judge, reference):
    """Agreement above what chance would give, on the same items.

    Returns None when there is nothing to compute, and 1.0 for perfect agreement. Can go negative,
    which means the judge is worse than guessing and is worth seeing rather than clamping.

    The correction is the whole point: two graders that both say "correct" nine times in ten agree
    81% of the time by accident, and a raw figure of 0.81 would read as agreement.
    """
    pairs = [(j, r) for j, r in zip(judge, reference, strict=True)
             if j is not None and r is not None]
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for j, r in pairs if j == r) / n
    labels = {j for j, _ in pairs} | {r for _, r in pairs}
    expected = sum(
        (sum(1 for j, _ in pairs if j == lab) / n) * (sum(1 for _, r in pairs if r == lab) / n)
        for lab in labels)
    if expected >= 1.0:
        # Every item carries the same label on both sides, so chance already explains everything
        # and kappa is undefined. Reporting 1.0 here would certify a judge on a set that could not
        # have caught it being wrong.
        return None
    return (observed - expected) / (1 - expected)


def per_class(judge, reference):
    """For each reference label, how often the judge found it. The imbalance check.

    A judge blind to one class is the failure mode raw agreement cannot see, and it is worse than
    random when the class it misses is the one that would have caught the model out.
    """
    out = {}
    labels = sorted({r for r in reference if r is not None})
    for lab in labels:
        idx = [i for i, r in enumerate(reference) if r == lab]
        hit = sum(1 for i in idx if judge[i] == lab)
        lo, hi = wilson_interval(hit, len(idx))
        out[lab] = {
            "n": len(idx),
            "found": hit,
            "recall": (hit / len(idx)) if idx else None,
            "ci": (round(lo, 4), round(hi, 4)),
        }
    return out


def validate(judge, reference, *, min_kappa=MIN_KAPPA, min_recall=MIN_PER_CLASS_RECALL,
             floor=MIN_REPORTABLE_N):
    """Whether a judge may be used, and every reason it may not.

    Returns a verdict dict. `certified` is the only field a caller should branch on, and the
    reasons are what a person needs when it is False.

    Deliberately reports agreement as well, clearly labelled as the misleading one, because
    somebody will look for it and it is better to show it beside kappa than to have it computed
    elsewhere without the correction.
    """
    if len(judge) != len(reference):
        return {
            "certified": False,
            "n": 0,
            "reasons": [("the judge and the reference cover different numbers of items, so they "
                         "cannot be compared item by item")],
        }
    pairs = [(j, r) for j, r in zip(judge, reference, strict=True)
             if j is not None and r is not None]
    n = len(pairs)
    agreement = (sum(1 for j, r in pairs if j == r) / n) if n else None
    kappa = cohens_kappa(judge, reference)
    classes = per_class(judge, reference)

    reasons = []
    if n < floor:
        reasons.append(
            f"only {n} items could be compared, below the floor of {floor}. A judge validated on "
            f"this few has not been validated.")
    if kappa is None:
        reasons.append(
            "agreement above chance could not be computed, usually because every item carries the "
            "same label. A set that could not have caught the judge being wrong cannot certify it.")
    elif kappa < min_kappa:
        reasons.append(
            f"agreement above chance is {kappa:.2f}, below {min_kappa}. Raw agreement of "
            f"{agreement:.2f} is not the number that matters: a judge answering the common label "
            f"every time scores well on it and learns nothing.")
    for lab, stats in classes.items():
        if stats["recall"] is not None and stats["recall"] < min_recall:
            reasons.append(
                f"it found only {stats['found']} of {stats['n']} items whose true label is "
                f"{lab!r} ({stats['recall']:.0%}), below {min_recall:.0%}. A judge blind to one "
                f"class flatters whatever it is grading in that direction.")
    return {
        "certified": not reasons,
        "n": n,
        # Reported and labelled, not hidden: the misleading number is the one people look for.
        "agreement": round(agreement, 4) if agreement is not None else None,
        "agreement_is_misleading_because": (
            "it counts a judge that always answers the common label as correct that often"),
        "kappa": round(kappa, 4) if kappa is not None else None,
        "per_class": classes,
        "reasons": reasons,
    }


def report(v):
    """The verdict in the words a person needs before trusting a number the judge produced."""
    lines = [f"judge validation on {v['n']} items"]
    if v.get("kappa") is not None:
        lines.append(f"  agreement above chance (kappa): {v['kappa']}")
    if v.get("agreement") is not None:
        lines.append(f"  raw agreement: {v['agreement']}  <- NOT the number that matters")
    for lab, s in (v.get("per_class") or {}).items():
        lines.append(f"  found {s['found']}/{s['n']} of the {lab!r} items  95% CI {s['ci']}")
    if v["certified"]:
        lines.append("  CERTIFIED: this judge may grade, and its error rate belongs beside "
                     "anything it grades.")
    else:
        lines.append("  NOT CERTIFIED. Nothing this judge grades should be published:")
        lines += [f"    - {r}" for r in v["reasons"]]
    return lines


def build_parser():
    import argparse

    ap = argparse.ArgumentParser(
        prog="senbonzakura judge",
        description="Check a judge against reference labels before letting it grade anything.")
    ap.add_argument("--judge", required=True,
                    help="a jsonl of the judge's verdicts, one object per line, with a 'verdict' "
                         "field")
    ap.add_argument("--reference", required=True,
                    help="the same items with the labels believed correct, in the same order")
    ap.add_argument("--field", default="verdict", help="which field holds the label")
    ap.add_argument("--out", default="", help="write the verdict as json")
    return ap


def _labels(path, field):
    rows = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            rows.append(json.loads(line).get(field))
    return rows


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        judge = _labels(a.judge, a.field)
        reference = _labels(a.reference, a.field)
    except (OSError, ValueError) as e:
        raise SystemExit(f"could not read the verdicts: {e}") from e
    v = validate(judge, reference)
    for line in report(v):
        print(line)
    if a.out:
        Path(a.out).write_text(json.dumps(v, indent=2), encoding="utf-8")
    # Non-zero when not certified, so a pipeline cannot proceed to grade with it by accident.
    return 0 if v["certified"] else 1
