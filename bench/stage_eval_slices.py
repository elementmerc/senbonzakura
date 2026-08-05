#!/usr/bin/env python3
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

Run on the host, in senbonzakura's environment, before the container arms start.
"""
import argparse
import sys
from pathlib import Path

from datasets import load_from_disk

from senbonzakura.cli import kl_eval_slice


def load_texts(directory, n):
    ds = load_from_disk(directory)
    if "text" not in getattr(ds, "column_names", []):
        raise SystemExit(f"stage_eval_slices: {directory} has no 'text' column "
                         f"(columns: {getattr(ds, 'column_names', '?')})")
    take = min(n, len(ds))
    if take < n:
        print(f"stage_eval_slices: NOTE {directory} holds {len(ds)} rows, fewer than the {n} "
              f"requested; using all {len(ds)}", file=sys.stderr)
    return [ds[i]["text"] for i in range(take)]


def write_slice(path, prompts, label):
    """One prompt per line, which is Heretic's plain-text format.

    A prompt containing a newline would become two prompts on the other side of this file and
    shift every prompt after it, so the two tools would be scored on lists that differ from the
    second offending row onward. Refuse rather than write a file that reads as valid.
    """
    bad = [i for i, p in enumerate(prompts) if "\n" in p or "\r" in p]
    if bad:
        raise SystemExit(
            f"stage_eval_slices: {len(bad)} prompt(s) in the {label} slice contain a line break "
            f"(first at index {bad[0]}). Heretic reads one prompt per line, so writing these would "
            f"silently split them and shift every prompt after. Fix the corpus rows first.")
    blank = [i for i, p in enumerate(prompts) if not p.strip()]
    if blank:
        raise SystemExit(
            f"stage_eval_slices: {len(blank)} blank prompt(s) in the {label} slice (first at index "
            f"{blank[0]}). Heretic ignores empty lines, so the slice it reads would be shorter than "
            f"the one senbonzakura is scored on.")
    # Heretic strips each line as it reads it and senbonzakura does not, so a prompt carrying
    # surrounding whitespace reaches the two tools as two different strings.
    padded = [i for i, p in enumerate(prompts) if p != p.strip()]
    if padded:
        raise SystemExit(
            f"stage_eval_slices: {len(padded)} prompt(s) in the {label} slice carry leading or "
            f"trailing whitespace (first at index {padded[0]}). Heretic strips it on read and "
            f"senbonzakura does not, so the two tools would be scored on different strings. Clean "
            f"the corpus rows rather than the file written here.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"stage_eval_slices: {len(prompts):>4} prompts -> {path}  ({label})")
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
        raise SystemExit(f"stage_eval_slices: no track at {track}")

    if a.eval_refusal_final < a.eval_refusal:
        raise SystemExit(
            f"stage_eval_slices: --eval-refusal-final ({a.eval_refusal_final}) is smaller than "
            f"--eval-refusal ({a.eval_refusal}). The final slice is the LARGER one the best-of-N "
            f"pass re-scores on; with it smaller, the selection would see less evidence than the "
            f"search did and the pass would be worse than not running it.")

    counts = {}
    # The search-time refusal slice: what steers each tool's own optimisation.
    counts["keyword"] = write_slice(
        out / "keyword_prompts.txt", load_texts(track / "bad_eval_ds", a.eval_refusal),
        "search-time refusal eval, Heretic's KeywordRate scorer")
    # The larger slice the best-of-N selection re-scores on, for both tools.
    counts["final"] = write_slice(
        out / "final_prompts.txt", load_texts(track / "bad_eval_ds", a.eval_refusal_final),
        "best-of-N re-score, both tools")
    # The coherence slice, disjoint from the extraction prompts by the same rule the abliterator
    # applies, computed by the same function rather than a second copy of the arithmetic.
    counts["kl"] = write_slice(
        out / "kl_prompts.txt",
        kl_eval_slice(load_texts(track / "good_ds", a.dir_prompts + a.eval_kl),
                      a.dir_prompts, a.eval_kl,
                      lambda m: print(f"stage_eval_slices: NOTE good_ds {m}", file=sys.stderr)),
        "KL divergence, disjoint from direction extraction")

    # The final slice is a superset of the search slice by construction (both are the head of
    # bad_eval_ds). Stated as a check because a corpus shorter than the final count would silently
    # make them equal, and the re-score would then add nothing while the table said it had.
    if counts["final"] <= counts["keyword"]:
        raise SystemExit(
            f"stage_eval_slices: the best-of-N slice ({counts['final']}) is not larger than the "
            f"search slice ({counts['keyword']}), so re-scoring on it would repeat the search's own "
            f"measurement rather than test it on fresh evidence. bad_eval_ds is too small; either "
            f"grow it or lower --eval-refusal.")

    print("STAGE_SLICES_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
