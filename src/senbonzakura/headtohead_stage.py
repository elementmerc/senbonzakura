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
from senbonzakura.metrics import MIN_REPORTABLE_N
from senbonzakura.track import flag_violations, read_manifest


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
            f"and the re-score keeps only the {a.eval_refusal_final - a.eval_refusal} of them "
            f"the search did not see.")

    # THE GUARD THAT EXISTS FOR EXACTLY THIS, and whose only caller was the abliterate path.
    #
    # This command takes the same five counts as free integers and reads each dataset by
    # head-count, so on a track whose harmful search partition is 96 rows the default
    # --eval-refusal-final 128 cuts 32 rows out of the MEASURE partition. Both tools are then
    # selected by best-of-N on rows they are later scored on, every artefact is present, every arm
    # completes, and the published refusal comparison describes the rows the winner was chosen on.
    #
    # That is the C4 defect from the September panel reappearing through the one path that builds
    # the slices BOTH tools share. The bundled track happens to survive it today; nothing enforced
    # that. Checked before anything is written, because a slice already on disk is a slice
    # something can pick up.
    manifest = read_manifest(track)
    if manifest:
        bad_flags = flag_violations(
            manifest, eval_refusal=a.eval_refusal, eval_refusal_final=a.eval_refusal_final,
            dir_prompts=a.dir_prompts, eval_kl=a.eval_kl)
        if bad_flags:
            raise SystemExit(
                f"headtohead stage: these counts would read past the boundaries "
                f"{track}/track.json records, and both tools would be selected on rows they are "
                f"later scored on:\n" + "\n".join(f"  {b}" for b in bad_flags))
    else:
        print(f"headtohead stage: {track} carries no track.json, so the partition boundaries "
              f"cannot be checked. The slices below are cut by head-count and NOTHING here can "
              f"tell you whether they cross a held-out boundary.", file=sys.stderr)

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
            load_texts(track / "bad_eval_ds", a.eval_refusal_final),
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
    # THE SELECTION'S OWN COHERENCE SLICE, and it must not be the file above.
    #
    # `kl_prompts.txt` is handed to Heretic during its search as the set its own KL is computed
    # on, so ranking best-of-N candidates by a KL measured there asks each tool how it did on
    # prompts one of them tuned against. `drift_prompt_slice` already refuses exactly this
    # confound for the PUBLISHED coherence figure and says so in as many words; the argument was
    # never carried across to the selection, where it decides which candidate wins.
    #
    # Bounded rather than fatal: it biases which of six candidates is chosen, not the headline,
    # and it is roughly symmetric because our own search also optimises KL on its own slice. Cut
    # from past `--dir-prompts + --eval-kl`, by the same rule that makes `final_prompts` disjoint.
    counts["kl_select"] = write_slice(
        out / "bestofn_kl_prompts.txt",
        kl_eval_slice(load_texts(track / "good_ds", a.dir_prompts + a.eval_kl + a.eval_kl),
                      a.dir_prompts + a.eval_kl, a.eval_kl,
                      lambda m: print(f"headtohead stage: WARNING the selection's coherence slice "
                                      f"is {m}, so it overlaps the one Heretic tunes against",
                                      file=sys.stderr)),
        "best-of-N coherence, held out from the slice Heretic's search optimises against")

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

    # THIS CHECK'S PREMISE INVERTED WHEN THE SLICES BECAME DISJOINT, so its question changed with
    # it. It used to require the final slice to be strictly LARGER than the search slice, because
    # both were the head of `bad_eval_ds` and equal sizes meant the same rows twice, a re-score
    # that added nothing while the table said it had. Now they share no rows at all, so size is no
    # longer a proxy for freshness and "larger" is not the property to demand.
    #
    # What still has to hold is that the re-score has enough evidence to be worth the generations
    # it costs. Below the floor at which this project will state a rate, it is not.
    if counts["final"] < MIN_REPORTABLE_N:
        raise SystemExit(
            f"headtohead stage: the best-of-N slice holds {counts['final']} prompts, below the "
            f"{MIN_REPORTABLE_N} this project will state a rate over. It is cut from the rows "
            f"between --eval-refusal ({a.eval_refusal}) and --eval-refusal-final "
            f"({a.eval_refusal_final}), so widen that gap, or rebuild the track with a larger "
            f"--search partition. Selecting a published arm on fewer than {MIN_REPORTABLE_N} "
            f"prompts is a coin toss with a decimal point.")

    print("STAGE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
