# Where the evaluation corpus came from

For every prompt in the bundled evaluation track, this page says where it came from and what is known about it.

**This page is about origins. The [evaluation track card](./evaluation-track-card.md) is about limitations.** The card lists every known defect. This page says where each row came from. Neither repeats the other, so the corpus cannot end up with two official stories.

## Why this page exists

Four of the five known defects were found by somebody reading the corpus, not by anything checking it. So the corpus now carries its own provenance record, and every check below is one a reader can run.

## The three sources

| Source | Side | Rows it contributes | Declared licence | Revision read |
|---|---|---|---|---|
| `Bahushruth/abliteration-harmful-enriched` | harmful | most of the harmful side | apache-2.0, in `cardData` and tagged | `f29c0b77` |
| `mlabonne/harmful_behaviors`, reached through the above | harmful | 176, measured 2026-10-01 | **none declared** | `01cead01` |
| `mlabonne/harmless_alpaca` | harmless | the harmless side | **none declared** | `02c6a92c` |

Two of the three declare no licence. The card explains what we believe the position is, and why that is a belief. The short version: the 520 rows in `mlabonne/harmful_behaviors` are, by row count and by inspection, AdvBench's `harmful_behaviors.csv` from the `llm-attacks` repository, which is MIT licensed. That is an inference from the contents. Neither dataset card says so.

Of those 520 upstream rows, **176 rows of this track's harmful side carry one of their requests**, measured template-aware on 2026-10-01. An earlier estimate of "about 430" was not supported by measurement and has been corrected everywhere it appeared.

We distribute the track under **CC BY-NC 4.0**, the most restrictive licence in the chain.

## How the rows are laid out

The track holds three datasets. **They are not three partitions.** Three partitions are packed into them:

```
  bad_ds        harmful  fit
  bad_eval_ds   harmful  search  ++  measure      (in that order)
  good_ds       harmless fit  ++  search  ++  measure   (in that order)
```

The partition boundaries are **recorded in the manifest, not derivable from the files.** A consumer that reads the first N rows of `good_ds` crosses the fit boundary once N exceeds the fit count, and nothing in the file says so.

`senbonzakura.track.flag_violations` reads the manifest and refuses any flag combination that crosses a recorded boundary. It runs on the abliteration path, on the head-to-head stager, and on `senbonzakura validate`.

**A track with no manifest is unguarded.** The check needs recorded boundaries, so a track built before manifests existed skips it silently. If you hold such a track, rebuild it or audit it:

```sh
python -m senbonzakura.track --out <track> --audit
```

## The counts

| Dataset | Rows |
|---|---|
| `bad_ds` | 259 |
| `bad_eval_ds` | 4,636 |
| `good_ds` | 4,982 |

Measured 2026-09-11: the harmless `fit + search` partitions hold **385** rows and the harmful ones hold **391**. Some tools default to 400 prompts a side, which is more than this track keeps aside for fitting and selection. A run at that default is refused rather than quietly reading into the measurement rows.

## What is not recorded

**Part of the harmless side cannot be reproduced.** Rows were added to the harmless side after the first build, and the addition was not recorded: not the source, the count, or the revision. So `senbonzakura track build` reproduces a corpus of the same shape from the same upstreams, but not these rows.

**The gap was measured on 2026-10-01, and it is large.** Of the 4,982 rows in the shipped `good_ds`, **2,036 (40.87%) have no request present in `mlabonne/harmless_alpaca`**, the only harmless upstream this track declares. They are spread through all three partitions, not appended as a tail:

| Partition | Rows with no recorded source | Of |
|---|---|---|
| `fit` | 106 | 257 |
| `search` | 58 | 128 |
| `measure` | 1,872 | 4,597 |
| **total** | **2,036** | **4,982** |

**Those 2,036 rows are 386 distinct requests, not 2,036 independent ones.** Each appears about five times under a different template. The 2,946 rows that trace to `harmless_alpaca` are 2,946 distinct requests, one row each, with no template expansion. So the unreproducible part is a second source of 386 requests, template-expanded and then mixed in before partitioning. That is why it is spread evenly.

Two things follow. The gap is 386 requests, not 2,036 rows, which is a tractable recovery job. And the harmless side is 41% template-expanded, against a harmful side that is expanded throughout, so the two sides do not share internal structure. Treat the harmless side as partly unreproducible. Do not assume a rebuild gives the same corpus.

To reproduce this measurement, read `good_ds` and `mlabonne/harmless_alpaca`, key both with `senbonzakura.track.request_key` over templates discovered across the union of the two, and count `good_ds` rows whose key is absent upstream. Matching by raw string gives a different and wrong answer, for the reason the contamination checker documents.

**What the count does not cover.** It tests against the one harmless upstream this track declares. If the 386 requests came from an undeclared second dataset, their source was never recorded. That is recoverable, but this count cannot tell the two cases apart. `wangzhang/abliterix-datasets` is the obvious candidate to check next. It was **not** checked and not cleared: the copy available when this was measured held only a ref with no data.

Every track built since carries a manifest recording its sources and revisions, so this cannot happen again. It did happen once, to the corpus behind the published figures.

## Checking the corpus against a public benchmark

AdvBench is inside this track, arriving through `mlabonne/harmful_behaviors`. If any of those rows sit in the `fit` or `search` partitions, a model tuned with this track was tuned on requests AdvBench would later test it against. Any AdvBench score from it is in-sample rather than held out.

**That check has been run, and this track is contaminated.** Measured 2026-10-01: of AdvBench's 508 distinct requests, 8 are in `fit`, 3 in `search`, 159 in `measure`, and 338 are absent. So 11 requests were fitted or searched on, and an AdvBench figure over the whole benchmark from this track is in-sample. A clean figure is still available over the 497 requests that are in `measure` or were never held. A figure reported that way has to say so. The card carries the table and the caveats on how it was measured.

The same applies to any other public set assembled from these upstreams, which is most of them. Check before you rely on a figure:

```sh
python -m senbonzakura.track --out <track> --contamination advbench.txt \
    --contamination-name AdvBench
```

Add `--fail-on-contamination` to make it an error rather than a report, and `--contamination-report <path>` to keep the result as an artefact.

## Citing

Cite AdvBench (Zou, Wang, Carlini, Nasr, Kolter and Fredrikson, *Universal and Transferable Adversarial Attacks on Aligned Language Models*, 2023, MIT licence), Alpaca, the two HuggingFace datasets the builder fetches, and this repository. If you publish a number from the harmful side, publish the harmless arm beside it.
