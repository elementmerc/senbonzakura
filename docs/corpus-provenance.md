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
| `mlabonne/harmful_behaviors`, reached through the above | harmful | 176, measured 2026-10-01 | **none declared** | `01cead01` |
| `mlabonne/harmless_alpaca` | harmless | the harmless side | **none declared** | `02c6a92c` |

Two of the three declare no licence at all. The card explains what we believe the true position
is for each, and why that is a belief rather than a fact. The short version: the 520 rows in
`mlabonne/harmful_behaviors` are, by row count and by inspection, AdvBench's
`harmful_behaviors.csv` from the `llm-attacks` repository, which is MIT licensed. That is an
inference from the contents, and neither dataset card says so.

Of those 520 upstream rows, **176 rows of this track's harmful side carry one of their requests**,
measured template-aware on 2026-10-01. An earlier estimate of "about 430" circulated in this
project's docs and was not supported by measurement; it has been corrected everywhere it appeared.

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

**Part of the harmless side cannot be reproduced.** Rows were added to the harmless side after the
first build and the addition was not recorded: not the source, not the count, not the revision. So
`senbonzakura track build` reproduces a corpus of the same shape from the same upstreams, and it
does not reproduce these rows.

**The gap was measured on 2026-10-01 and it is large.** Of the 4,982 rows in the shipped `good_ds`,
**2,036 (40.87%) have no request present in `mlabonne/harmless_alpaca`**, the only harmless
upstream this track declares. They are spread through all three partitions rather than appended as
a tail:

| Partition | Rows with no recorded source | Of |
|---|---|---|
| `fit` | 106 | 257 |
| `search` | 58 | 128 |
| `measure` | 1,872 | 4,597 |
| **total** | **2,036** | **4,982** |

**Those 2,036 rows are 386 distinct requests, not 2,036 independent ones.** Each appears about
five times under a different template. The 2,946 rows that do trace to `harmless_alpaca` are 2,946
distinct requests, one row each, with no template expansion. So the unreproducible part of the
harmless side is a second source of 386 requests that was template-expanded and then mixed in
before partitioning, which is why it is distributed evenly rather than sitting at the end.

Two things follow, and the second is the one that matters for anybody planning a rebuild. The gap
to close is 386 requests rather than 2,036 rows, which is a tractable recovery job rather than a
hopeless one. And the harmless side is 41% template-expanded against a harmful side that is
expanded throughout, so the two sides of this track do not have the same internal structure. Treat
the harmless side as partly unreproducible and do not assume a rebuild will give you the same
corpus.

How to reproduce this measurement: read `good_ds` and `mlabonne/harmless_alpaca`, key both with
`senbonzakura.track.request_key` over templates discovered across the union of the two, and count
`good_ds` rows whose key is absent upstream. Matching by raw string instead gives a different and
wrong answer, for the reason the contamination checker documents.

**What the count does not cover, stated so it does not read as cleared.** It tests against the one
harmless upstream this track declares. If those 386 requests came from an undeclared second
dataset rather than being written by hand, they have a source that was simply never recorded, which
is a recoverable situation rather than a permanent one, and this count cannot tell the two apart.
`wangzhang/abliterix-datasets` is the obvious candidate to check next. It was **not** checked and
not cleared: the copy available when this was measured held only a ref with no data, so there was
nothing to compare against.

Every track built after that point carries a manifest recording its sources and revisions, so
this cannot happen again. It did happen once, and it happened to the corpus behind the published
figures.

## Checking the corpus against a public benchmark

AdvBench is inside this track, arriving through `mlabonne/harmful_behaviors`. If any of those rows
sit in the `fit` or `search` partitions, then a model tuned with this track was tuned on requests
AdvBench would later mark it against, and any AdvBench score from it is in-sample rather than held
out.

**That check has now been run, and the answer is that this track is contaminated.** Measured
2026-10-01: of AdvBench's 508 distinct requests, 8 are in `fit`, 3 are in `search`, 159 are in
`measure` and 338 are absent. So 11 requests were fitted or searched on, and an AdvBench figure
over the whole benchmark from this track is in-sample. A clean figure is still available over the
497 requests that are either in `measure` or were never held, and a figure reported that way has to
say so in those words. The card carries the table and the caveats on how it was measured.

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
