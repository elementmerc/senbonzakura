# The track

A **track** is the pile of prompts the tool learns from and is judged on. Harmful ones, harmless
ones, and a split that decides which of them any published number is allowed to come from.

It sounds like plumbing. It is the single easiest place to accidentally lie to yourself, so it
gets a page.

## Why the split is the whole thing

Suppose you fit your refusal directions on 500 harmful prompts, then measure how well the
abliteration worked using those same 500 prompts.

You'll get a spectacular result. You have also learned nothing, because you tested the model on
the exact questions it was tuned against. It's revising from the answer sheet.

::: warning This project did exactly that, for months
The original corpus had **all 200 harmful evaluation prompts sitting inside the 4,918-row fitting
set**. On the harmless side it was 196 out of 197.

Every number measured on it was in-sample and flattering, and nothing in the tooling noticed,
because "the file parsed and the numbers came out" is not a check.
:::

So a track is split three ways, and the tool enforces it:

| Partition | Harmful | Harmless | What touches it |
|---|--:|--:|---|
| `fit` | 259 | 257 | Finding the directions |
| `search` | 132 | 128 | Scoring attempts, picking a winner |
| `measure` | 4,504 | 4,597 | **Every published number, and nothing else** |

Note the sizes. The part that does the work is small. The part that's protected is 92% of the
corpus, and the abliteration never sees it.

## Build one, and have it checked

Doing it by hand is how the mistake happens, so there is a builder:

```sh
python -m senbonzakura.track \
    --harmful harmful.txt --harmless harmless.txt \
    --out mytrack --fit 256 --search 128
```

One prompt per line. It deduplicates on a case- and punctuation-insensitive key,
splits each side into three parts by index, and **writes nothing at all unless every
check passes**:

- `fit` — the directions are extracted from these
- `search` — the search scores trials on these
- `measure` — the published number comes from these, and nothing else touches them

The checks are the point. A track is refused if anything in `measure` also appears in
`fit` or `search`, if a prompt is labelled both harmful and harmless, if a partition is
empty, or if the two sides differ in size by more than 10%. It reports counts only and
never prints a prompt, so its output is safe to paste anywhere.

### What it writes, if you want to build one yourself

Each partition is a HuggingFace dataset saved with `save_to_disk`, holding exactly one
column named `text`, one prompt per row:

```python
from datasets import Dataset
Dataset.from_dict({"text": ["first prompt", "second prompt"]}).save_to_disk("mytrack/bad_ds")
```

That is the whole schema. Anything the tool reads, it reads from a `text` column, so a
dataset built any other way loads and then scores nothing. The directories it expects are
`bad_ds`, `good_ds` and `bad_eval_ds`, with `good_eval_ds` and `good_matched_ds` optional
(see below).

Use the builder anyway if you can. Writing the directories by hand skips every check
above, and the checks are the reason the split is worth anything.

## The check that isn't obvious

Here's the one that catches people, including me.

Corpora are usually built by crossing a handful of phrasings with a list of requests. So *"Explain
in detail how to X"* and *"Write a step by step guide to X"* are two different strings and one
question.

Compare whole prompts and you get a clean split. Compare the underlying **requests** and you find
your evaluation set is stuffed with training questions wearing a different hat.

On this project's own corpus, that's the difference between **"zero overlap"** and **60% of the
evaluation set**. The first number is what a naive check reported. The second is the truth.

So the builder works the phrasings out from your corpus, strips them, and keeps every wording of
one request on the same side of the split. You don't configure it; it's derived from your data, so
it works on somebody else's corpus too and can't go stale against ours.

::: warning If you take one thing from this page
**The split isn't bookkeeping.** It's the difference between a measurement and a compliment
you paid yourself.
:::

## `track.json`: where the boundaries are recorded

The three partitions are stored end to end inside each dataset, so `bad_ds` is the fit rows
followed by nothing else, while `bad_eval_ds` is the search rows followed by the measure
rows. **Nothing about a directory of prompts says where one partition ends and the next
begins**, so the builder writes it down:

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

| Field | What it is |
|---|---|
| `schema` | The format version. A manifest declaring a version this build does not know is **refused**, rather than read with the fields it happens to recognise. A manifest with no `schema` at all is read as version 1, because hand-written ones predate the field. |
| `sources` | Where the prompts came from, so a track can be traced to its input. |
| `counts` | Rows in each partition, per side. The record the boundary checks are made against. |
| `skip_harmful`, `skip_harmless` | What to pass to `--skip-harmful` and `--skip-harmless` to land on the measure rows. |
| `n_harmful`, `n_harmless` | What to pass to `--n`. |

**This file is load-bearing, not documentation.** Three things read it:

- **The audit** (`--audit`) re-checks an existing track: leakage, empty partitions, whether
  the rows on disk still match the counts recorded here. A `track.json` that disagrees with
  the rows beside it is refused, because every slice taken from it would silently read the
  wrong rows.
- **The boundary check** refuses a run whose flags would cross a partition. Asking the search
  to score more prompts than the `search` partition holds means selecting a configuration on
  the rows the published number comes from, and the tool stops rather than doing it.
- **The measure flags above** are read from it rather than remembered.

A track with no `track.json` still runs. It simply gets none of the above, and every
boundary becomes something you have to keep right by hand.

## The optional fourth and fifth directories

| Directory | What it holds | What reads it |
|---|---|---|
| `good_matched_ds` | Harmless prompts matched to the harmful ones by topic | `--harmless-matched`, the compass's construct-validity control |
| `hedge_ds` | Hedged compliances, neither refusal nor clean answer | The hedging direction, when one exists |

Both are optional and neither is built by `senbonzakura.track` today. `good_matched_ds`
answers "is the compass reading refusal, or is it reading topic?" by holding the subject
matter still. `hedge_ds` has no public dataset behind it anywhere, which is why the
single-versus-multi comparison is two arms rather than three.

## Re-checking a track you already have

Including one somebody assembled by hand, or one of yours from six months ago:

```sh
python -m senbonzakura.track --out mytrack --audit
```

![The track audit running against the committed toy track](/media/track-audit.gif)

That's the real thing, recorded against `examples/toy-track` which ships in the repository, so
you can run the exact command above and get the exact output above.

If you've got a `label<TAB>prompt` file for the corpus, pass `--labels` as well. It buys
you the one check that can't be run without it: whether every category in the track
actually reaches the `measure` slice. A category that doesn't is the mirror image of
leakage. Leakage makes your number better than it should be; this makes it **narrower than
it looks**, a score on eight topics being reported as a score on twelve.

Rebuilding backs the old track up once, to `<track>.pre-build`, and never overwrites that
backup. So a rebuild you regret costs you nothing, and a second rebuild you regret can't
quietly eat the good copy.

## A toy track to try it on

`examples/toy-track/` is committed and runs straight from a clone. Its prompts are
synthetic placeholders rather than real harmful text, because it exists to show the
plumbing works, not to measure anything. Point the tool at it, break something on purpose,
watch the audit complain.

## Where next

- [Contamination](/guide/contamination) to check whether the benchmark you want to quote is
  already sitting inside your fit rows.
- [The compass](/guide/compass) for what to point at the `measure` slice once you have one.
