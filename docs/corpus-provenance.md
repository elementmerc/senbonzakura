# Where the evaluation corpus came from

This page answers one question: for every prompt in the bundled evaluation track, where did it
come from and what is known about it?

**This page is about origins. The [evaluation track card](./evaluation-track-card.md) is about
limitations.** If you want to know what is wrong with the corpus, read the card, which lists every
known defect. If you want to know where a row came from, read this. Neither page repeats the
other, deliberately: two documents describing one corpus is how a corpus ends up with two
different official stories.

## Why this page exists

Four of the five known defects in this corpus were found by somebody reading it, not by anything
checking it. That is a bad way to find out that your ruler is bent. So the corpus now carries its
own provenance record, and every check described here is something a reader can run.

## The three sources

| Source | Side | Rows it contributes | Declared licence | Revision read |
|---|---|---|---|---|
| `Bahushruth/abliteration-harmful-enriched` | harmful | most of the harmful side | apache-2.0, in `cardData` and tagged | `f29c0b77` |
| `mlabonne/harmful_behaviors`, reached through the above | harmful | about 430 | **none declared** | `02c6a92c`-era, unchanged since 2024-05-30 |
| `mlabonne/harmless_alpaca` | harmless | the harmless side | **none declared** | `02c6a92c` |

Two of the three declare no licence at all. The card explains what we believe the true position
is for each, and why that is a belief rather than a fact. The short version: the 520 rows in
`mlabonne/harmful_behaviors` are, by row count and by inspection, AdvBench's
`harmful_behaviors.csv` from the `llm-attacks` repository, which is MIT licensed. That is an
inference from the contents, and neither dataset card says so.

We distribute the track under **CC BY-NC 4.0**, the most restrictive licence in the chain.

## How the rows are laid out, and why that matters

The track holds three datasets, and **they are not three partitions.** Three partitions are
packed into them:

```
  bad_ds        harmful  fit
  bad_eval_ds   harmful  search  ++  measure      (in that order)
  good_ds       harmless fit  ++  search  ++  measure   (in that order)
```

So the boundaries between partitions are **recorded in the manifest, not derivable from the
files.** A consumer that reads the first N rows of `good_ds` is reading across the fit boundary
once N exceeds the fit count, and nothing about the file says so.

That is a real hazard and the tool guards against it. `senbonzakura.track.flag_violations` reads
the manifest and refuses any combination of flags that would cross a recorded boundary. It runs
on the abliteration path, on the head-to-head stager, and on `senbonzakura validate`.

**A track with no manifest is unguarded.** The check needs recorded boundaries to check against,
so a track built before manifests existed skips it silently. If you hold such a track, rebuild it
or audit it:

```sh
python -m senbonzakura.track --out <track> --audit
```

## The counts

| Dataset | Rows |
|---|---|
| `bad_ds` | 259 |
| `bad_eval_ds` | 4,636 |
| `good_ds` | 4,982 |

Two figures worth knowing if you are configuring a run, measured 2026-09-11: the harmless
`fit + search` partitions hold **385** rows between them and the harmful ones hold **391**. Some
other tools default to asking for 400 prompts a side, which is more than this track keeps aside
for fitting and selection, so a run at that default would be refused rather than quietly reading
into the measurement rows.

## What is not recorded, and the size of the gap

**The harmless top-ups cannot be reproduced.** Rows were added to the harmless side by hand in
August 2026 and the addition was not recorded: not the source, not the count, not the revision.
So `senbonzakura track build` reproduces a corpus of the same shape from the same upstreams, and
it does not reproduce these rows.

**The size of that gap has not been measured.** Nobody has counted how many rows of the shipped
`good_ds` have no recorded source. Until that count exists, treat the harmless side as partly
unreproducible and do not assume a rebuild will give you the same corpus.

Every track built after that point carries a manifest recording its sources and revisions, so
this cannot happen again. It did happen once, and it happened to the corpus behind the published
figures.

## Checking the corpus against a public benchmark

AdvBench is inside this track, arriving through `mlabonne/harmful_behaviors`. If any of those rows
sit in the `fit` or `search` partitions, then a model tuned with this track was tuned on requests
AdvBench would later mark it against, and any AdvBench score from it is in-sample rather than held
out.

The same applies to any other public set assembled from these upstreams, which is most of them.
Check before you rely on a figure:

```sh
python -m senbonzakura.track --out <track> --contamination advbench.txt \
    --contamination-name AdvBench
```

Add `--fail-on-contamination` to make it an error rather than a report, and
`--contamination-report <path>` to keep the result as an artefact.

## Citing

Cite AdvBench (Zou, Wang, Carlini, Nasr, Kolter and Fredrikson, *Universal and Transferable
Adversarial Attacks on Aligned Language Models*, 2023, MIT licence), Alpaca, the two HuggingFace
datasets the builder fetches, and this repository. If you publish a number from the harmful side,
publish the harmless arm beside it.
