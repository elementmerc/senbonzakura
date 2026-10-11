# Compass, Qwen3-1.7B, 2026-10-11

The run behind the README's "One command, on your own machine" example. It exists because that
example quoted figures that were in no committed artefact, and because the command it quoted them
from was scoring the wrong rows.

## What was measured

| | |
|---|---|
| Model | `Qwen/Qwen3-1.7B`, unedited |
| Track | the bundled evaluation track, `default` |
| Rows scored | 60 per arm |
| Harmful arm | rows 132 to 191 of `bad_eval_ds`, after the 132 search rows |
| Harmless arm | rows 385 to 444 of `good_ds`, after the 257 fit rows and the 128 search rows |
| Boundary source | `track`, read from the track's own manifest, not guessed |
| Device | cpu, 16 cores |
| Wall clock | 3 minutes 44 seconds |
| Seed | 42 |
| senbonzakura | `023869d` |
| Model revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Corpus revision | `416bdf8ac34e85be3cb834d808f59ee5022fc929494bb86bb3b0d6a14657a602`, the packed track's `sha256_of_tar` |

The exact command, which is the one in the README:

```sh
senbonzakura compass \
    --model Qwen/Qwen3-1.7B \
    --track default \
    --harmful default/bad_eval_ds \
    --harmless default/good_ds \
    --n 60 \
    --out compass-toy.json --device cpu
```

## The numbers

| Figure | Value |
|---|---|
| Compass AUC | **0.9872**, 95% interval [0.9722, 0.9972] |
| Length-only ruler | **0.6853** |
| Strongest null of five | word count, 0.7283 |
| Characters null | 0.6933 |
| Verdict read-out | 100% of prompts put a verdict token where the compass reads one, both arms |

The compass clears every control by a wide margin, and the read-out is clean, so the AUC is a
measurement of harm discrimination rather than of sentence length or of a misread position.

## Why `--track default` is in the command, and what it is worth

**The flag is the measurement.** Without it the run has no manifest, so an explicit
`--skip-harmful 0 --skip-harmless 0` is honoured without comment and the scored rows are the ones
at the top of each file. On this track those are:

```
bad_eval_ds  4636 rows = 132 search + 4504 measure
good_ds      4982 rows = 257 fit    + 128 search + 4597 measure
```

So the harmful arm would score the rows the search selected its winning trial from, and the
harmless arm would score the rows the abliteration directions were fitted on. The run exits 0 and
reports a plausible number either way, which is what makes it worth writing down.

Measured both ways on the same model and the same sample size:

| | compass AUC | length-only ruler |
|---|---|---|
| Scoring from row 0 (fit and search rows) | 0.9851 | 0.8339 |
| Scoring from the recorded boundary | **0.9872** | **0.6853** |

The compass barely moves. The control moves a lot, from 0.8339 to 0.6853, because the opening rows
of this track are separable by sentence length in a way the held-out rows are not. A reader who
took the first row of that table would have a figure flattered by its partition and no way to tell.
Rebalancing the opening rows is `v0.9` row 9.11.

## Reproducing it

Run the command above. Two things decide whether you land on the same figures:

- **The seed.** 42 is the default and is recorded in the artefact. The bootstrap interval moves
  with it; the AUC does not, because the row selection is deterministic given the skips.
- **The environment.** `provenance.packages` in the artefact pins torch, transformers, pyarrow and
  the rest. `torch==2.14.0+cu130` running on cpu is the measurement here.

Three independent runs of this configuration on this machine, at three different commits of the
tool, all gave AUC 0.9872 and a length-only ruler of 0.6853. None of the three intervening changes
touched the scoring path, so that is a check on the harness rather than on the measurement; what it
rules out is a figure that moves when nothing relevant moved.

## What is not here, and why

**The per-prompt margins.** `margins_path` names `margins/base-Qwen3-1.7B.jsonl` and that file is
not in this directory. Those rows hold the prompts, so they live in an ignored `results/` tree, as
`evidence/README.md` sets out and as `tools/ci/check_prompt_artefacts.py` enforces. The pointer is
kept because it says what the artefact was written beside.

**A pin for a local checkpoint.** This artefact is of a Hub model, so `model_revision` is the
resolved commit. A checkpoint on disk has no upstream revision, and the honest pin for one is a
digest of its weights, which this project does not have yet. `stamps.model_revision_of` therefore
records nothing for a local path rather than a placeholder, and the provenance guard is expected to
refuse such an artefact until the digest exists. The design question is in `DEFERRED.md`.

**A `dirty` flag that says `false`.** The stamp reads `"dirty": null`, meaning unknown, and a
consumer is expected to treat unknown as dirty. The tree WAS clean at `d7aac58`: it was committed,
then synced, then built with `SENBON_BUILD_COMMIT` set, which is the path `setup.py` documents for
a build box that has no `.git`. What that path cannot currently carry is the cleanliness of the
tree it came from, because the far side has no git to ask. That gap is in `DEFERRED.md`; it is
recorded here rather than argued away, because the honest reading of this artefact's stamp is
"commit d7aac58, cleanliness unproven".
