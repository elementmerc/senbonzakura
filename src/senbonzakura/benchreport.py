"""Read a finished head-to-head and say what it found, without saying more than it found.

WHY THIS IS NOT `report_bands.py`

`report_bands.py` reads the K-comparison grid that `senbonzakura validate` writes: one model,
several direction budgets, ranked within a matched-refusal band. A head-to-head is a different
shape. Two tools, five seeds each, one scorer applied to every resulting model afterwards. Pointed
at compass scores, `report_bands.py` correctly reports that it can see no grid.

WHAT IS ACTUALLY COMPARABLE, AND WHAT ONLY LOOKS IT

This is the whole design of this file, so it is stated before any number is printed.

**The compass is comparable.** One instrument, run by us, over every model from both tools, on the
same held-out prompts, after the fact. Whatever it says about tool A and tool B is a comparison of
the models, because nothing else differs.

**The two tools' own KL figures are NOT comparable to each other**, and reporting them side by
side in one column would be the error this project withdrew four claims for on 2026-08-05.
senbonzakura's `post_bake_kl` comes from our estimator on our coherence slice; Heretic's comes from
its estimator on its own evaluation, and even pointed at the same prompts those are two different
measurements wearing one name. They are printed, because hiding them would be worse, but they are
printed in separate rows labelled with whose estimator produced them, and no gap between them is
ever called a win.

**The refusal figures in Heretic's `best_of_n.json` ARE ours**, measured by our rulers on the
shared re-score slice during the selection pass. Those are comparable with each other across
Heretic's seeds; they are not the same slice our own arm reports its `post_bake_refusals` on, so
that pairing is labelled too rather than tabulated as one column.

THE TIE RULE

The exit gate is explicit: a gap smaller than the spread is reported as a tie. So the verdict
takes the two tools' means and compares the gap against the pooled spread, and prints "tie" when
the gap does not clear it. "Still suggestive, now with a spread" is an acceptable and publishable
outcome; a winner declared inside the noise is not.

With fewer than three seeds a spread is not an estimate of anything, so the verdict says so and
declines rather than dividing by a number it does not have.

WHY IT LIVES IN THE PACKAGE

It was a script under `tools/`, which meant it did not ship in the wheel, so the only thing that
could read a published head-to-head was a checkout of this repository. That is the same defect as
the compass shipping in no released artefact, and it matters more here: a reader's whole recourse
against a table they doubt is being able to re-derive it.

Usage:
    senbonzakura bench report <run-directory>
"""
import argparse
import json
import math
import os
import re
import sys

#: Recognises `senbon-seed42` / `heretic-seed42` and the `-own-pick` variant the selection pass
#: writes when its choice differed from Heretic's own first offer.
ARM = re.compile(r"^scored-(?P<tool>senbon|heretic)-seed(?P<seed>\d+)(?P<variant>-own-pick)?$")

#: Below this many seeds a spread is not an estimate. Three gives a variance with two degrees of
#: freedom, whose interval already runs from about half to six times the point estimate; the gate
#: asks for five. Two is arithmetic dressed as statistics.
MIN_SEEDS_FOR_A_SPREAD = 3

#: A spread this small is zero, and the distinction matters because it is not reachable by `== 0`.
#: `stdev([0.95, 0.95, 0.95])` is 2.7e-16, not 0.0: the mean of three identical floats is not
#: exactly that float, so the deviations are tiny rather than absent. An exact comparison silently
#: took the "we have a spread" branch and printed "pooled spread of 0.0000" beside a verdict it
#: had no business drawing. AUC lives in [0,1] and is reported to four decimals, so anything below
#: this is numerically indistinguishable from no variation at all.
SPREAD_IS_ZERO = 1e-9


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    """Sample standard deviation. Zero for a single point, which callers must not divide by."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def collect(run_dir):
    """Every scored arm in the directory, keyed by tool, with its own artefact beside it."""
    arms = []
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".json"):
            continue
        m = ARM.match(name[: -len(".json")])
        if not m:
            continue
        scored = load_json(os.path.join(run_dir, name))
        if not scored or "auc" not in scored:
            arms.append({"name": name, "tool": m["tool"], "seed": int(m["seed"]),
                         "variant": m["variant"] or "", "unreadable": True})
            continue
        arm = {
            "name": name[: -len(".json")],
            "tool": m["tool"],
            "seed": int(m["seed"]),
            "variant": m["variant"] or "",
            "auc": scored["auc"],
            "auc_ci": scored.get("auc_ci"),
            "length_only_auc": (scored.get("controls") or {}).get("length_only_auc"),
            "unreadable": False,
        }
        arm.update(own_numbers(run_dir, m["tool"], m["seed"]))
        arm.update(one_ruler_drift(run_dir, m["tool"], m["seed"], m["variant"]))
        arm.update(one_ruler_refusal(run_dir, m["tool"], m["seed"], m["variant"]))
        arms.append(arm)
    return arms


def one_ruler_refusal(run_dir, tool, seed, variant):
    """Refusals counted by us, on one slice, for every model.

    The axis the tool exists to move, and the last of the three to become comparable. `own_refusals`
    below is each tool's self-report: same rulers, different prompts, so not a column.
    """
    label = f"{tool}-seed{seed}{variant or ''}"
    d = load_json(os.path.join(run_dir, f"refusal-{label}.json")) or {}
    return {
        "one_refusal": d.get("refusal"),
        "one_noncompliant": d.get("noncompliant"),
        "one_keyword": d.get("heretic"),
        "one_broken": d.get("broken"),
        "one_refusal_n": d.get("n"),
    }


def one_ruler_drift(run_dir, tool, seed, variant):
    """Drift from the base, measured by US, on one slice, for every model.

    The counterpart of the compass and the reason this report can finally say something about
    coherence. `own_kl` below is each tool's self-report and is not comparable across rows; this
    one is, because every row came from the same instrument, the same base, the same prompts and
    the same batch size.
    """
    label = f"{tool}-seed{seed}{variant or ''}"
    d = load_json(os.path.join(run_dir, f"drift-{label}.json")) or {}
    return {
        "drift_kl": d.get("kl"),
        "drift_prompts": d.get("prompts"),
        "drift_n": d.get("n_prompts"),
    }


def own_numbers(run_dir, tool, seed):
    """What the arm itself recorded, tagged with whose estimator produced it.

    Deliberately kept apart from the compass fields: these two tools measured these numbers with
    their own code, and the tag is what stops a reader treating one column as one measurement.
    """
    if tool == "senbon":
        d = load_json(os.path.join(run_dir, f"senbon-seed{seed}", "abliteration.json")) or {}
        return {
            "own_refusals": d.get("post_bake_refusals"),
            "own_kl": d.get("post_bake_kl"),
            "own_kl_estimator": "senbonzakura, our coherence slice (held back from fitting)",
            "own_refusal_estimator": "senbonzakura rulers, our search eval",
        }
    d = load_json(os.path.join(run_dir, f"heretic-seed{seed}", "best_of_n.json")) or {}
    winner = d.get("winner") or {}
    return {
        "own_refusals": winner.get("refusals"),
        "own_kl": winner.get("kl"),
        # Named precisely, because this is the pairing that must never be read as one column.
        "own_kl_estimator": "Heretic, its own evaluation (its full harmless set)",
        "own_refusal_estimator": "senbonzakura rulers, shared re-score slice",
    }


def verdict(by_tool):
    """Compare the two tools on the compass, and refuse to call anything inside the noise."""
    tools = sorted(by_tool)
    if len(tools) != 2:
        return f"NO VERDICT: {len(tools)} tool(s) present; a head-to-head needs two."
    a, b = tools
    xa = [arm["auc"] for arm in by_tool[a]]
    xb = [arm["auc"] for arm in by_tool[b]]
    if min(len(xa), len(xb)) < MIN_SEEDS_FOR_A_SPREAD:
        return (f"NO VERDICT: {a} has {len(xa)} seed(s) and {b} has {len(xb)}. Below "
                f"{MIN_SEEDS_FOR_A_SPREAD} a spread is not an estimate of anything, so no gap "
                f"can be judged against it.")
    ma, mb = mean(xa), mean(xb)
    # Pooled, because the question is whether the gap clears the noise in BOTH arms rather than
    # the noise in whichever one happens to be quieter.
    spread = math.sqrt((stdev(xa) ** 2 + stdev(xb) ** 2) / 2)
    gap = abs(ma - mb)
    if spread < SPREAD_IS_ZERO:
        # Every seed of both tools landing on the same number is not a clean result, it is a
        # measurement to check: identical scores usually mean the seed did not reach the thing it
        # was supposed to vary. With no gap either there is nothing to say; with a gap, the gap is
        # reported and the implausibility is reported beside it rather than quietly ignored.
        if gap < SPREAD_IS_ZERO:
            return ("NO VERDICT: both tools scored identically on every seed. That is a "
                    "measurement to investigate rather than a result to publish.")
        winner = a if ma > mb else b
        return (f"{winner} scores higher ({max(ma, mb):.4f} against {min(ma, mb):.4f}), but "
                f"EVERY seed of both tools returned an identical score, so the spread is exactly "
                f"zero. Check that the seed reaches the search before reading the gap: a spread "
                f"of zero across five seeds is not a tight measurement, it is usually a seed that "
                f"never varied anything.")
    if gap < spread:
        return (f"TIE on harm recognition: {a} {ma:.4f}, {b} {mb:.4f}, gap {gap:.4f}, pooled "
                f"spread {spread:.4f}. The gap is smaller than the spread, and the gate says that "
                f"is a tie.")
    winner, loser = (a, b) if ma > mb else (b, a)
    return (f"{winner} scores higher on harm recognition than {loser}: {max(ma, mb):.4f} against "
            f"{min(ma, mb):.4f}, gap {gap:.4f} against a pooled spread of {spread:.4f}.")


def render(arms):
    lines = []
    unreadable = [a for a in arms if a["unreadable"]]
    readable = [a for a in arms if not a["unreadable"]]

    lines.append("=== Harm recognition, one instrument over every model ===")
    lines.append("Our compass, run after the fact on the same held-out prompts for both tools.")
    lines.append("This is the axis where a comparison means something.")
    lines.append("")
    lines.append(f"{'arm':<28} {'AUC':>8} {'95% CI':>18} {'length-only':>12}")
    for a in sorted(readable, key=lambda a: (a["tool"], a["seed"], a["variant"])):
        ci = a["auc_ci"]
        ci_s = f"[{ci[0]:.4f},{ci[1]:.4f}]" if isinstance(ci, list) and len(ci) == 2 else "-"
        lo = a["length_only_auc"]
        lines.append(f"{a['name']:<28} {a['auc']:>8.4f} {ci_s:>18} "
                     f"{(f'{lo:.4f}' if lo is not None else '-'):>12}")

    by_tool = {}
    # The own-pick rows are Heretic-as-shipped rather than the arm the equal budget defines, so
    # they are shown above and kept out of the verdict.
    for a in readable:
        if a["variant"]:
            continue
        by_tool.setdefault(a["tool"], []).append(a)

    lines.append("")
    lines.append("=== Per tool, across seeds ===")
    for tool in sorted(by_tool):
        xs = [a["auc"] for a in by_tool[tool]]
        lines.append(f"  {tool:<10} n={len(xs)}  mean AUC {mean(xs):.4f}  spread {stdev(xs):.4f}")

    lines.append("")
    lines.append("=== Verdict ===")
    lines.append(verdict(by_tool))

    lines.append("")
    lines.append("=== Refusals removed, one ruler over every model ===")
    lines.append("Counted by us on the same held-out prompts for both tools, so this is the")
    lines.append("comparable version of the axis the tool exists to move. `noncompliant` is hard")
    lines.append("refusal plus hedging; `broken` is output that is not English, and a low refusal")
    lines.append("rate beside a high broken rate is a wrecked model, not a good result.")
    lines.append("")
    lines.append(f"{'arm':<28} {'refusal':>8} {'noncomp':>8} {'keyword':>8} {'broken':>7} {'n':>5}")
    have_ref = False
    for a in sorted(readable, key=lambda a: (a["tool"], a["seed"], a["variant"])):
        r = a.get("one_refusal")
        if r is not None:
            have_ref = True
        def _pc(v):
            return f"{v*100:.1f}%" if v is not None else "-"
        lines.append(f"{a['name']:<28} {_pc(r):>8} {_pc(a.get('one_noncompliant')):>8} "
                     f"{_pc(a.get('one_keyword')):>8} {_pc(a.get('one_broken')):>7} "
                     f"{(a.get('one_refusal_n') or '-'):>5}")
    if not have_ref:
        lines.append("")
        lines.append("  No refusal files found. This axis is still two self-reports on different")
        lines.append("  prompts, so no sentence may be written across the tools about it.")
    else:
        by_tool_ref = {}
        for a in readable:
            if a["variant"] or a.get("one_refusal") is None:
                continue
            by_tool_ref.setdefault(a["tool"], []).append(a["one_refusal"])
        lines.append("")
        for tool in sorted(by_tool_ref):
            xs = by_tool_ref[tool]
            spread = f"{stdev(xs)*100:.1f}%" if len(xs) > 1 else "-"
            lines.append(f"  {tool:<10} n={len(xs)}  mean refusal {mean(xs)*100:.1f}%  "
                         f"spread {spread}")

    lines.append("")
    lines.append("=== Coherence drift, one instrument over every model ===")
    lines.append("KL(base || edited) on first-token distributions, same prompts and same batch")
    lines.append("size for every model, measured by us afterwards. THIS column is comparable.")
    lines.append("Lower is less collateral damage. Read it beside the refusal rates below: a tool")
    lines.append("that left refusals standing has paid less for its coherence, so a lower number")
    lines.append("here is only a better result at a matched refusal rate.")
    lines.append("")
    lines.append(f"{'arm':<28} {'drift KL':>10} {'prompts':>9}")
    have_drift = False
    for a in sorted(readable, key=lambda a: (a["tool"], a["seed"], a["variant"])):
        dk, dn = a.get("drift_kl"), a.get("drift_n")
        if dk is not None:
            have_drift = True
        lines.append(f"{a['name']:<28} {(f'{dk:.4f}' if dk is not None else '-'):>10} "
                     f"{(dn if dn is not None else '-'):>9}")
    if not have_drift:
        lines.append("")
        lines.append("  No drift files found. Run `senbonzakura bench head-to-head` with scoring")
        lines.append("  enabled, or `senbonzakura drift` per model, to fill this in.")
    else:
        by_tool_drift = {}
        for a in readable:
            if a["variant"] or a.get("drift_kl") is None:
                continue
            by_tool_drift.setdefault(a["tool"], []).append(a["drift_kl"])
        lines.append("")
        for tool in sorted(by_tool_drift):
            xs = by_tool_drift[tool]
            spread = f"{stdev(xs):.4f}" if len(xs) > 1 else "-"
            lines.append(f"  {tool:<10} n={len(xs)}  mean drift {mean(xs):.4f}  spread {spread}")

    lines.append("")
    lines.append("=== What each tool reported about itself ===")
    lines.append("NOT a comparison. The two KL figures come from different estimators on different")
    lines.append("slices, and putting them in one column is the error four claims were withdrawn")
    lines.append("for on 2026-08-05. Read each row against its own label, never across rows.")
    lines.append("The comparable coherence number is the drift column above, not these.")
    lines.append("")
    lines.append("The slices differ in a way that matters and does not favour us: ours is measured")
    lines.append("on harmless prompts HELD BACK from direction fitting, while Heretic's is measured")
    lines.append("on the harmless prompts it was given, its directions included. Ours is therefore")
    lines.append("the harder ground, which is one more reason these two numbers are not a column.")
    for a in sorted(readable, key=lambda a: (a["tool"], a["seed"], a["variant"])):
        if a["variant"]:
            continue
        r, k = a.get("own_refusals"), a.get("own_kl")
        lines.append(f"  {a['name']:<28} refusals "
                     f"{(f'{r*100:.1f}%' if r is not None else '-'):>7}  "
                     f"KL {(f'{k:.4f}' if k is not None else '-'):>8}")
        lines.append(f"    {'':<26} refusal estimator: {a.get('own_refusal_estimator', '?')}")
        lines.append(f"    {'':<26} KL estimator:      {a.get('own_kl_estimator', '?')}")

    if unreadable:
        lines.append("")
        lines.append(f"=== {len(unreadable)} arm(s) produced nothing readable ===")
        lines.extend(f"  {a['name']}" for a in unreadable)
    return "\n".join(lines), bool(unreadable), len(readable)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the directory holding scored-*.json and the arm directories")
    ap.add_argument("--allow-unreadable", action="store_true",
                    help="report on the arms that could be read instead of failing. Off by "
                         "default: a table missing an arm nobody mentioned is how a partial run "
                         "gets published as a whole one.")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.run_dir):
        raise SystemExit(f"bench report: no directory at {a.run_dir}")

    arms = collect(a.run_dir)
    if not arms:
        raise SystemExit(
            f"bench report: no scored arms under {a.run_dir}. Expected files named "
            f"scored-<tool>-seed<N>.json, which is what the score job writes.")

    text, had_unreadable, n_readable = render(arms)
    print(text)
    if had_unreadable and not a.allow_unreadable:
        print("\nFAILED: some arms produced nothing readable; pass --allow-unreadable to report "
              "on the rest anyway.", file=sys.stderr)
        return 1
    if n_readable == 0:
        print("\nFAILED: nothing readable at all.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
