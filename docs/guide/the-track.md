# The track

A **track** is the set of prompts the tool learns from and is judged on: harmful prompts, harmless prompts, and a split that
decides which rows a published number may come from.

## Why the split matters

If you fit the directions on some prompts and measure on the same prompts, you've marked your own homework.

::: warning This project did exactly that, for months
The original corpus had all 200 harmful evaluation prompts inside the 4,918-row fitting set. On the harmless side it was 196 out of
197. Every number measured on it was in-sample and flattering.
:::

So a track is split three ways, and the tool enforces it:

| Partition | Harmful | Harmless | What touches it |
|---|--:|--:|---|
| `fit` | 259 | 257 | Finding the directions |
| `search` | 132 | 128 | Scoring attempts, picking a winner |
| `measure` | 4,504 | 4,597 | **Every published number, and nothing else** |

## Checking a benchmark for contamination

Before you quote a score on a public benchmark such as AdvBench, check whether its requests are already inside your fit or search rows.
If they are, the number doesn't mean what it says.

```sh
senbonzakura track --out mytrack \
    --contamination advbench.txt --contamination-name AdvBench \
    --contamination-report contamination-advbench.json
```

Against our own bundled track it printed:

```
contamination: AdvBench against this track
  520 rows, 508 distinct requests (templates discovered: 131)
  in fit 8, in search 3, in measure 159, absent 338
  CONTAMINATED: 11 of its requests were fitted or searched on.
  publishable today: 159 requests from the measure partition, plus 338 this track has never held.
```

::: warning Public corpora overlap more than you'd expect
**176 rows** of this project's harmful corpus **are AdvBench**, arriving through `mlabonne/harmful_behaviors` without anybody choosing
them. The check on our own bundled track, run on 2026-10-01, came back contaminated: 11 of AdvBench's requests were in the partitions
we fit and search on.
:::

- **It matches requests, not strings.** The tool strips the wrappers corpora put around the same request before comparing.
- **`measure` is not contamination.** That slice is held out by design. Overlap with `fit` or `search` is the fatal kind.
- **It reports counts, never a prompt**, so the output is safe to paste anywhere.

`--fail-on-contamination` makes it exit non-zero, which suits a CI job. Without the flag it reports and exits cleanly.

The track build ends by printing the flags that point the compass at the held-out rows:

```
TRACK_BUILT mytrack  harmful {'fit': 256, 'search': 128, 'measure': 616}
  measure with: --skip-harmful 128 --skip-harmless 384 --n 616
```

Copy the second line into your compass command. [The compass](/guide/compass) explains what to do with it.

## Getting the prompts

```sh
senbonzakura track build --out corpus
```

It fetches the public sources at **pinned revisions**, writes `harmful.txt` and `harmless.txt`, and records what it fetched in
`sources.json`. It refuses to run if an upstream licence has changed since the recipe was written.

The sources return about four times as many harmless prompts, so the builder balances the two sides with a seeded sample.
`sources.json` records the seed and the rows dropped. `--no-balance` keeps the raw pools.

::: warning The track is published under CC BY-NC 4.0
Every row came from somewhere else, so the track carries the strictest term in its chain: attribution to every source, and
non-commercial use only. Access is gated, so you accept the acceptable-use terms first. The rebuilt pool has the same shape as
the original, not the same rows. The [dataset card](/evaluation-track-card) covers provenance.
:::

::: danger If you hold a copy of this track from before 2026-07-30, check it
An earlier export, `ops-malware/senbon-track-35axis-clean`, was labelled "clean" and wasn't. Measured on 2026-08-16, **189 of its 200
harmful evaluation rows, and all 196 harmless ones, were inside its own fitting set**. The published track,
[`ops-malware/senbonzakura-dataset`](https://huggingface.co/datasets/ops-malware/senbonzakura-dataset), is the repaired one, with zero
overlap on both sides. Check any copy you have:

```sh
senbonzakura track --out mytrack --audit
```
:::

## Build your own, and have it checked

```sh
senbonzakura track \
    --harmful harmful.txt --harmless harmless.txt \
    --out mytrack --fit 256 --search 128
```

One prompt per line. It deduplicates on a case- and punctuation-insensitive key, splits each side into `fit`, `search` and `measure`
by index, and **writes nothing unless every check passes**. It refuses a track if anything in `measure` also appears in `fit` or
`search`, if a prompt is labelled both harmful and harmless, if a partition is empty, or if the sides differ by more than 10%.

Writing the directories by hand skips every check, and the checks are what make the split worth anything. If you do it anyway, each
partition is a Hugging Face dataset with one column, `text`, in the directories `bad_ds`, `good_ds` and `bad_eval_ds`.

### Why the check is on requests, not wording

Corpora are usually built by crossing a few phrasings with a list of requests. "Explain in detail how to X" and "Write a step by step
guide to X" are two strings and one question. Compare whole prompts and you get a clean split. On this project's own corpus, comparing
the underlying requests changes **zero overlap** into **60% of the evaluation set**. The builder works the wrappers out from your data
and keeps every wording of one request on the same side of the split.

## `track.json`: where the boundaries are

Nothing in a directory of prompts says where one partition ends and the next begins, so the builder writes it down:

```json
{
  "schema": "senbonzakura-track/1",
  "sources": { "harmful": "harmful.txt", "harmless": "harmless.txt" },
  "counts": {
    "harmful":  { "fit": 256, "search": 128, "measure": 616 },
    "harmless": { "fit": 256, "search": 128, "measure": 616 }
  },
  "skip_harmful": 128,
  "skip_harmless": 384,
  "n_harmful": 616,
  "n_harmless": 616
}
```

Three things read this file. The audit (`--audit`) refuses a track whose rows disagree with its counts. The boundary check refuses a run
whose flags would cross a partition. The measure flags are read from it. A track without `track.json` still runs, but none of those
checks apply.

If you have a `label<TAB>prompt` file, add `--labels` to `--audit`. That checks whether every category reaches the `measure` slice. A
category that doesn't makes your score narrower than it looks.

Rebuilding backs the old track up once, to `<track>.pre-build`, and never overwrites that backup.

## Optional directories and a toy track

`good_matched_ds` (harmless prompts matched to the harmful ones by topic, read by `--harmless-matched`) and `hedge_ds` (hedged
compliances) are optional. Neither is built by `senbonzakura track` today.

`examples/toy-track/` is committed and runs from a clone. Its prompts are synthetic placeholders, so it shows the plumbing works and
measures nothing.

## Where next

- [The compass](/guide/compass), for what to point at the `measure` slice once you have one.
