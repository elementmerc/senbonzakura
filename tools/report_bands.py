#!/usr/bin/env python3
"""Print a matched-refusal comparison, using the harness's verdict rather than a fresh argmin.

Every run spec in this repo grew its own report as a Python heredoc, and every one of them ended
with some spelling of:

    best = min(kls, key=kls.get)
    print(f"cheapest K={best}  MULTI WINS")

`matched_refusal_table` already decides whether a band may be ranked at all: it refuses when the
grid has no dynamic range, when the arms are not at the same refusal level, and when the margin is
inside the noise. None of that reached the screen, because the screen was drawn by a different
piece of code that only knew how to take a minimum. On 2026-08-04 that gap printed "MULTI WINS"
three times, once from KL 0.0088 against 0.0089.

So the verdict is read, never recomputed. If a band is not rankable this says why, and the number
that would have been the winner is simply not printed.

    python tools/report_bands.py A0=path/a0.json A1=path/a1.json ...

Exit 0 only when every named file was readable and carried a grid. A report that found nothing
must not be a green job: the run specs match on a needle in stdout, so a report that prints
"MISSING" six times and exits 0 is indistinguishable from one that worked.
"""
import argparse
import json
import sys


def load(path):
    """The document, or a string saying why not. Never raises."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "does not exist"
    except (OSError, ValueError) as e:
        return None, f"unreadable ({e})"


def band_lines(table, band):
    """One line per arm, plus the verdict line, for a single band."""
    entries = table.get("targets", {}).get(band, {})
    lines = []
    arms = "  ".join(
        f"K={K}:{e['kl']:.4f}(at {e['harmful_refusal']:.4f})"
        for K, e in sorted(entries.items(), key=lambda kv: int(kv[0])))
    lines.append(f"    {band:<14} {arms}")
    verdict = table.get("ranking", {}).get(band)
    if not verdict:
        lines.append("      no verdict recorded for this band")
    elif verdict.get("cheapest") is not None:
        lines.append(f"      cheapest K={verdict['cheapest']}, "
                     f"{verdict.get('margin')}x below the next best")
    else:
        lines.append(f"      NO RANKING: {verdict.get('reason')}")
    return lines


def describe(label, doc):
    """Everything worth printing about one result file. Returns (lines, ok)."""
    lines = [f"--- {label}"]
    e4 = (doc or {}).get("e4")
    if not e4:
        lines.append("    no e4 grid in this file")
        return lines, False

    # Provenance first: two files that differ only by a path in `directions_from` are two
    # different experiments, and which one this is has to be visible above the numbers.
    prov = {k: doc.get(k) for k in ("model", "seed", "code_version") if doc.get(k) is not None}
    dm = doc.get("directions_meta") or {}
    for k in ("init", "score", "induce", "start_layer", "steps"):
        if k in dm:
            prov[k] = dm[k]
    if prov:
        lines.append("    " + "  ".join(f"{k}={v}" for k, v in prov.items()))

    if e4.get("degenerate_reason"):
        lines.append(f"    UNREADABLE GRID: {e4['degenerate_reason']}")
        return lines, True

    table = e4.get("matched_refusal") or {}
    if not table.get("baseline_is_unablated"):
        lines.append("    NO UNABLATED ANCHOR, so every 'percent removed' label is relative to "
                     "whichever arm ablated least; bands not shown")
        return lines, True

    lines.append(f"    baseline refusal {table.get('baseline_refusal')} (unablated), "
                 f"n_eval={table.get('n_eval')}")
    if not table.get("targets"):
        lines.append("    no arm reached any band")
        return lines, True
    for band in table["targets"]:
        lines.extend(band_lines(table, band))
    return lines, True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", metavar="LABEL=path.json")
    ap.add_argument("--allow-missing", action="store_true",
                    help="exit 0 even when some arms are unreadable. Off by default, because a "
                         "report that found nothing must not look like a report that worked.")
    a = ap.parse_args(argv)

    pairs = []
    for spec in a.arms:
        if "=" not in spec:
            print(f"report_bands: expected LABEL=path, got {spec!r}", file=sys.stderr)
            return 2
        label, _, path = spec.partition("=")
        pairs.append((label, path))

    missing = []
    print("\n=== KL at matched refusal removal ===")
    print("Lower KL at the SAME refusal level is better. A band with no ranking is one the")
    print("harness refused to rank, and the reason is printed instead of a winner.\n")
    for label, path in pairs:
        doc, err = load(path)
        if err:
            print(f"--- {label}\n    MISSING: {err}\n")
            missing.append(f"{label}: {err}")
            continue
        lines, ok = describe(label, doc)
        print("\n".join(lines) + "\n")
        if not ok:
            missing.append(f"{label}: no grid")

    if missing:
        print(f"INCOMPLETE: {len(missing)} of {len(pairs)} arms produced nothing readable:")
        for m in missing:
            print(f"  {m}")
        if not a.allow_missing:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
