# The track

A track is the corpus the tool reads: harmful prompts, harmless prompts, and a split that
decides which of them any published number may come from.


A **track** is a directory of three datasets. Every command takes `--track <dir>` and
looks for these names:

| Directory | What it holds | What reads it |
|---|---|---|
| `bad_ds` | Harmful prompts | Direction extraction fits on these |
| `bad_eval_ds` | Harmful prompts | The search scores trials, and the compass measures |
| `good_ds` | Harmless prompts | Fitting, the KL reference, and the compass's harmless arm |

Each is a `datasets.save_to_disk` directory with one column, `text`:

```python
from datasets import Dataset
Dataset.from_dict({"text": ["first prompt", "second prompt"]}).save_to_disk("mytrack/bad_ds")
```

### Build one, and have it checked

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

One of those checks looks past the text. Corpora are often built by crossing a handful
of phrasings with a list of requests, so the same question appears several times as
several different strings; comparing whole prompts then reports a clean split while the
eval set is full of training questions in other clothes. The builder works out the
phrasings from your own corpus, strips them, and keeps every wording of one request in
the same part. On the corpus this project runs on, that is the difference between
"zero overlap" and 60% of the eval set.

### `track.json`: where the boundaries are recorded

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

### The optional fourth and fifth directories

| Directory | What it holds | What reads it |
|---|---|---|
| `good_matched_ds` | Harmless prompts matched to the harmful ones by topic | `--harmless-matched`, the compass's construct-validity control |
| `hedge_ds` | Hedged compliances, neither refusal nor clean answer | The hedging direction, when one exists |

Both are optional and neither is built by `senbonzakura.track` today. `good_matched_ds`
answers "is the compass reading refusal, or is it reading topic?" by holding the subject
matter still. `hedge_ds` has no public dataset behind it anywhere, which is why the
single-versus-multi comparison is two arms rather than three.

Those numbers are also written to `mytrack/track.json`, so a reader can check the split
a year later rather than take it on trust. To re-check a track you already have,
including one assembled by hand:

```sh
python -m senbonzakura.track --out mytrack --audit
```

If you have a `label<TAB>prompt` file for the corpus, pass `--labels` to the audit too.
It adds the one check that cannot be run without it: whether every category present in
the track actually reaches the `measure` part. A category that does not is the mirror
image of leakage, a number narrower than it looks rather than better than it should be.

Rebuilding backs the old track up once, to `<track>.pre-build`, and never overwrites
that backup.

`track.json` is also read at the start of a run, and a run whose flags would reach past a
recorded boundary stops before the model loads. Each dataset is read as its first N rows,
and the search rows come immediately before the measured ones, so asking for a larger
selection set than the track keeps aside would quietly select on the rows the published
number comes from. A track without a `track.json`, one assembled by hand, runs as before:
its boundaries are unknown, so there is nothing to check against.

### A toy track to try it on

`examples/toy-track/` is committed and runnable straight from a clone. Its prompts are
synthetic placeholders rather than real harmful text, because it exists to show the
plumbing works, not to measure anything.
