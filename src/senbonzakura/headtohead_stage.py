# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Write out the exact prompt slices both tools are scored on, as plain text files.

WHY THIS EXISTS

Heretic's two scorers each own an evaluation prompt set, and both default to a Hugging Face
dataset ID: `mlabonne/harmful_behaviors` `test[:100]` for the keyword rate, `mlabonne/harmless_alpaca`
`test[:100]` for the KL divergence. The sealed benchmark box has no network, so an arm pointed only
at our corpus (which is what `[good_prompts]` and `[bad_prompts]` control) dies at scorer
initialisation before it runs a single trial.

Pointing those scorers at our corpus is not merely a workaround for the missing network. It is what
makes the head-to-head a comparison of two tools rather than a comparison of two evaluation sets:
each tool's search is steered by whatever its scorers measure, so two tools optimising against
different prompts have not been given the same problem.

WHY TEXT FILES RATHER THAN A SLICE STRING

Heretic accepts a dataset path plus a `split` slice, so in principle we could hand it
`train[256:320]` and be done. In practice the slice senbonzakura uses is computed at run time from
the resolved budget: the KL prompts are the harmless set MINUS the prompts the directions were
fitted on, with a documented fallback when the set is too small to spare a disjoint slice. Writing
a slice string by hand means reimplementing that arithmetic in a second place, where it can drift
without anything failing. So this calls the same function the abliterator calls, and writes the
result out verbatim. What Heretic reads is then the same list of strings, not a slice that ought to
produce the same list.

Heretic reads a plain text file as one prompt per line, so the only constraint is that no prompt
may contain a newline. That is checked rather than assumed, because a corpus row with an embedded
newline would silently become two prompts and shift every subsequent one.

Run on the host, in senbonzakura's environment, before the container arms start:
`senbonzakura head-to-head stage --track <track> --out <slices>`.

It moved out of `headtohead/` and into the package on 2026-08-06 for the same reason the report did:
a script beside the repository does not ship in the wheel, so nothing an outsider installs could
stage the inputs the benchmark needs, and the benchmark was therefore unrunnable by anyone but us.
"""
import argparse
import sys
from pathlib import Path

from senbonzakura.cli import kl_eval_slice, rescore_eval_slice


def load_texts(directory, n, *, text_column=None, token=None):
    """The first n prompts from a dataset, in whatever shape it arrives in."""
    from senbonzakura import dataset
    try:
        rows = dataset.resolve(directory, text_column=text_column, token=token,
                               strip=False, what="headtohead stage input")
    except dataset.DatasetError as e:
        raise SystemExit(f"headtohead stage: {e}") from e
    if len(rows) < n:
        print(f"headtohead stage: NOTE {directory} holds {len(rows)} rows, fewer than the {n} "
              f"requested; using all {len(rows)}", file=sys.stderr)
    return rows[:n]


def write_slice(path, prompts, label):
    """One prompt per line, which is Heretic's plain-text format.

    A prompt containing a newline would become two prompts on the other side of this file and
    shift every prompt after it, so the two tools would be scored on lists that differ from the
    second offending row onward. Refuse rather than write a file that reads as valid.
    """
    bad = [i for i, p in enumerate(prompts) if "\n" in p or "\r" in p]
    if bad:
        raise SystemExit(
            f"headtohead stage: {len(bad)} prompt(s) in the {label} slice contain a line break "
            f"(first at index {bad[0]}). Heretic reads one prompt per line, so writing these would "
            f"silently split them and shift every prompt after. Fix the corpus rows first.")
    blank = [i for i, p in enumerate(prompts) if not p.strip()]
    if blank:
        raise SystemExit(
            f"headtohead stage: {len(blank)} blank prompt(s) in the {label} slice (first at index "
            f"{blank[0]}). Heretic ignores empty lines, so the slice it reads would be shorter than "
            f"the one senbonzakura is scored on.")
    # Heretic strips each line as it reads it and senbonzakura does not, so a prompt carrying
    # surrounding whitespace reaches the two tools as two different strings.
    padded = [i for i, p in enumerate(prompts) if p != p.strip()]
    if padded:
        raise SystemExit(
            f"headtohead stage: {len(padded)} prompt(s) in the {label} slice carry leading or "
            f"trailing whitespace (first at index {padded[0]}). Heretic strips it on read and "
            f"senbonzakura does not, so the two tools would be scored on different strings. Clean "
            f"the corpus rows rather than the file written here.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"headtohead stage: {len(prompts):>4} prompts -> {path}  ({label})")
    return len(prompts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, help="the track holding bad_ds / good_ds / bad_eval_ds")
    ap.add_argument("--out", required=True, help="directory to write the slices into")
    # These four default to the values kageyoshi resolves for a model under 5B, which is the size
    # class the first head-to-head runs at (Qwen3-1.7B). They are flags rather than constants
    # because a larger model gets a smaller budget and the slices must follow it.
    ap.add_argument("--dir-prompts", type=int, default=256)
    ap.add_argument("--eval-refusal", type=int, default=64)
    ap.add_argument("--eval-refusal-final", type=int, default=128)
    ap.add_argument("--eval-kl", type=int, default=64)
    a = ap.parse_args(argv)

    track, out = Path(a.track), Path(a.out)
    if not track.is_dir():
        raise SystemExit(f"headtohead stage: no track at {track}")

    if a.eval_refusal_final < a.eval_refusal:
        raise SystemExit(
            f"headtohead stage: --eval-refusal-final ({a.eval_refusal_final}) is smaller than "
            f"--eval-refusal ({a.eval_refusal}). The final slice is the LARGER one the best-of-N "
            f"pass re-scores on; with it smaller, the selection would see less evidence than the "
            f"search did and the pass would be worse than not running it. The two are disjoint, "
            f"so this asks for {a.eval_refusal + a.eval_refusal_final} rows of bad_eval_ds.")

    counts = {}
    # The search-time refusal slice: what steers each tool's own optimisation.
    counts["keyword"] = write_slice(
        out / "keyword_prompts.txt", load_texts(track / "bad_eval_ds", a.eval_refusal),
        "search-time refusal eval, Heretic's KeywordRate scorer")
    # The larger slice the best-of-N selection re-scores on, for both tools, cut from PAST the
    # search-time rows above rather than from the head. Taken from the head it contained the
    # search slice whole, so half the evidence the selection saw was evidence the search had
    # already optimised against. Same helper the abliterator uses, so the two cannot drift.
    counts["final"] = write_slice(
        out / "final_prompts.txt",
        rescore_eval_slice(
            load_texts(track / "bad_eval_ds", a.eval_refusal + a.eval_refusal_final),
            a.eval_refusal, a.eval_refusal_final,
            lambda m: print(f"headtohead stage: WARNING {m}", file=sys.stderr)),
        "best-of-N re-score, both tools, held out from the search-time slice")
    # The coherence slice, disjoint from the extraction prompts by the same rule the abliterator
    # applies, computed by the same function rather than a second copy of the arithmetic.
    counts["kl"] = write_slice(
        out / "kl_prompts.txt",
        kl_eval_slice(load_texts(track / "good_ds", a.dir_prompts + a.eval_kl),
                      a.dir_prompts, a.eval_kl,
                      lambda m: print(f"headtohead stage: NOTE good_ds {m}", file=sys.stderr)),
        "KL divergence, disjoint from direction extraction")

    # The prompts each tool FITS its directions on. Heretic takes these as files; senbonzakura
    # reads them from the track directly. Same rows either way, which is the point: a tool given a
    # different fitting set is solving a different problem, and the difference would never show up
    # in either tool's own reporting.
    counts["fit_bad"] = write_slice(
        out / "bad.txt", load_texts(track / "bad_ds", a.dir_prompts),
        "harmful prompts the directions are fitted on")
    counts["fit_good"] = write_slice(
        out / "good.txt", load_texts(track / "good_ds", a.dir_prompts),
        "harmless prompts the directions are fitted on")

    # WHICH CORPUS THESE CAME FROM, recorded beside them. Once one tool reads the track directly
    # and another reads files cut from it, "both tools read the same corpus" stops being visible in
    # either command line and becomes an invariant nothing is checking. Slices cut from corpus A
    # beside a run pointed at corpus B would be silent: every arm finishes, every artefact is
    # present, and the table means nothing.
    from .headtohead import write_slice_provenance
    print(f"headtohead stage: recorded the source corpus in {write_slice_provenance(out, track)}")

    # The final slice is a superset of the search slice by construction (both are the head of
    # bad_eval_ds). Stated as a check because a corpus shorter than the final count would silently
    # make them equal, and the re-score would then add nothing while the table said it had.
    if counts["final"] <= counts["keyword"]:
        raise SystemExit(
            f"headtohead stage: the best-of-N slice ({counts['final']}) is not larger than the "
            f"search slice ({counts['keyword']}), so re-scoring on it would repeat the search's own "
            f"measurement rather than test it on fresh evidence. bad_eval_ds is too small; either "
            f"grow it or lower --eval-refusal.")

    print("STAGE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
