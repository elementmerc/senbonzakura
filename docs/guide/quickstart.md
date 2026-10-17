# Quickstart

::: danger What you are about to make
A model that will answer requests the original refused, including harmful ones. The change is permanent in
the weights, the base model's licence still governs it, and you should not put one in front of other people
without saying what it is. [What it is](/guide/what-it-is) has the rest.
:::

Most of the time is spent waiting. The nearest measured expectation is **108 minutes from the command to
`DONE`** for **Qwen3-0.6B** on a 6 GB card at the defaults, with the weights cached. That's a different model
from the one below, so read it as an order of magnitude.

You need a GPU with 6 GB or more, and Python 3.10 or newer. No card? Jump to
[without a graphics card](#without-a-graphics-card).

## Install

```sh
pip install senbonzakura
```

**The prompts come with it.** The wheel carries six research corpora and the packed evaluation track, so
`--track default` works with nothing else to fetch. That's roughly 6,200 harmful prompts on your disk.

If you install from a clone instead, the track isn't in the install. Build the one you want, and pass it by name:

<!-- documented-first-run: quickstart -->
```sh
senbonzakura corpora                   # the refusal corpora
senbonzakura track build --out corpus  # prompts, from public sources
senbonzakura track --harmful corpus/harmful.txt \
                   --harmless corpus/harmless.txt --out track
```

::: warning `senbonzakura corpora` needs the GitHub CLI
It fetches through `gh`, so no credential is handled by this project's own code. Without `gh` the command exits
1 and says so. Install it from <https://cli.github.com>, run `gh auth login`, then run this again.
:::

```sh
senbonzakura setup
```

The install brings torch, transformers, accelerate and optuna: 70 packages, 5.9 GB. [Install](/guide/install)
has the breakdown and the date it was measured.

`setup` exists because **pip picks by platform, not by hardware**:

| Your machine | What pip alone gives you |
|---|---|
| Linux with a GPU | the CUDA build. Correct |
| Linux with no GPU | the CUDA build anyway, and 15 CUDA packages you cannot use |
| macOS | a Metal build. Correct |
| Windows with a GPU | **a CPU-only build. Your card sits idle** |

`setup` reads the machine, says what it found, and prints the fix. It changes nothing unless you add `--apply`.

## Edit a model

```sh
senbonzakura Qwen/Qwen2.5-0.5B-Instruct
```

The model is the only thing the command can't guess. It writes to `./abliterated`, and uses `./track` if you
built one, otherwise the bundled track. The log says which.

Then go and make a cup of tea.

::: tip The bundled track is about 6,200 harmful prompts in your site-packages
[The install page](/guide/install#what-comes-in-the-box-including-the-part-people-don-t-expect) says what is in it.
:::

Every setting is still a flag:

```sh
senbonzakura Qwen/Qwen2.5-0.5B-Instruct --track mytrack --out my-model --device cuda
```

## What you get

A model in `./abliterated`, with `run.json` (what was done) and `abliteration.json` (what it cost). While it runs
you'll see lines like this:

```
trial 47: o(P=18,wmax=0.62) d(P=14,wmax=0.31) K=2 per_layer -> refusals=3.1% soft=5.2% heretic=12.5% broken=0% KL=0.1900 obj=0.2431
```

`refusals` is the one you came for. `KL` is what it cost you: lower is less damage.

::: warning What you are now holding
The model will answer things the original declined, and loading it differently doesn't undo that. **The base
model's licence still governs it.** If you publish it, `senbonzakura report` writes the card that should travel
beside the weights.

The refusal rates in `abliteration.json` come from the rows the search chose its winner by. They are the best of
those attempts, not a reading of the model. Score the held-back rows with `senbonzakura score`, and pass that file
to `senbonzakura report --refusal`. The card then publishes the figure with its sample size and interval.

`abliteration.json` holds settings and numbers, no prompts and no replies. The command that keeps per-prompt rows
is `senbonzakura compass`.
:::

## Without a graphics card {#without-a-graphics-card}

You can't edit a model on a processor in any useful time, but the instruments run. This scores a stock model on
the track bundled in your install:

```sh
senbonzakura compass \
    --model Qwen/Qwen3-1.7B \
    --track default \
    --harmful default/bad_eval_ds \
    --harmless default/good_ds \
    --n 60 \
    --out compass-toy.json --device cpu
```

**Measured: 3 minutes 44 seconds on sixteen cores**, with the compass at 0.9872 (95% interval 0.9722 to 0.9972)
against a length-only ruler's 0.6853. The artefact and environment are in
[evidence/compass-2026-10-11](https://github.com/elementmerc/senbonzakura/blob/dev/evidence/compass-2026-10-11/README.md).
Fewer or slower cores take proportionally longer. The model is **about 3.8 GB** the first time.

::: tip Why not a smaller model
Two smaller models decline to report a figure. That's the tool working, but a run that exits 1 looks like a broken
install, so they make poor first commands.

- `HuggingFaceTB/SmolLM2-135M-Instruct` puts no verdict token where the compass reads one, so it prints
  `MARGIN_READOUT_SUSPECT` and declines to call its AUC a measurement.
- `Qwen/Qwen3-0.6B` answers with a verdict on every prompt, then exits 1 with `MARGIN_NULLS_DOMINATE`. A control
  that reads only surface text separates the two sets as well as the model does. It does not exit 0.

| Rows | Compass AUC | The control that beat it | Wall clock |
|---|---|---|---|
| 12 | 0.7292, interval [0.5174, 0.9097] | characters, 1.0000 | 27s |
| 60 | 0.7047, interval [0.6126, 0.7940] | punctuation density, 0.2590, which is informative in reverse and further from chance than the compass | 83s |

At 0.6B the signal is too weak on this track for any sample size worth running on a processor.
:::

`default/<partition>` means the evaluation track inside the install, so there's no corpus to fetch. The **model**
is still downloaded from the Hub the first time. If you cloned the repository, `examples/toy-track/` also works, and
it is smaller.

`--track default` is what makes the answer mean anything. It scores rows from 132 and 385 on, the prompts nothing
was fitted or selected on. Without it the run scores the opening rows, the search and fit rows, and reports a
plausible number anyway. [The compass](/guide/compass) explains how to read the result.

## Where next

- [Your first run](/guide/first-run): the same command in detail, plus manual mode.
- [What it is](/guide/what-it-is): what this does, and what it destroys.
- [What is and is not established](/guide/what-we-know), if you plan to publish anything.
