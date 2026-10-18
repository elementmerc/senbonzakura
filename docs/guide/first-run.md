# Your first run

::: danger Before the first command
What comes out of this will answer requests the original refused, including harmful ones, and the change is
permanent in the weights. Read [What it is](/guide/what-it-is) first. This page is the mechanics.
:::

You need a GPU, a model, and a track. If you don't have a track, see [the track](/guide/the-track).

## The one command

```sh
senbonzakura kageyoshi \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --track mytrack \
    --out my-abliterated-model \
    --device cuda
```

Plan for a couple of hours. The nearest measured expectation is **108 minutes from the command to `DONE`** for
**Qwen3-0.6B** on a 6 GB card at the defaults, with the weights cached. The model in the command has not been timed
at the default budget.

::: tip Why this model
It refuses 58.6% of the bundled evaluation set (n=128), so you can watch the change. The tool declines to edit a
model below 5% refusal.
:::

## What it does while you wait

It runs a few hundred attempts at uncensoring, each with different settings, and scores each on three things: refusals
left, drift from the original, and whether the output has turned to mush. Then it applies the best trade-off and saves
the result.

```
trial 47: o(P=18,wmax=0.62) d(P=14,wmax=0.31) K=2 -> refusals=3.1% broken=0% KL=0.19
```

`refusals` is the one you came for. `KL` is what it cost you.

### The phases

::: warning The ETA is the search's ETA, not the run's
The `progress` line counts search trials. When its ETA reaches zero, the run has finished **step 3 of 7** and is not
done. On the measured 0.6B run the search reached its last trial at 3166 seconds, and `DONE` printed at 6507 seconds.
:::

| # | Phase | What it costs |
|---|---|---|
| 1 | Load the model, cache the original first-token distribution (the KL reference), measure the baseline refusal rate | Minutes on a GPU |
| 2 | Baseline capability probe on the unedited model | 200 graded items |
| 3 | **The search.** `--trials` attempts, each baked, scored and rolled back | The part the progress line and ETA describe |
| 4 | Re-score the top `--top-rescore` candidates (6) on 128 prompts held out from the search | Six full generation passes |
| 5 | Bake the winner into the weights | Fast |
| 6 | Post-bake measurement: refusals, KL, capability probe again | Another 200-item probe |
| 7 | Save the model and write `abliteration.json` | Disk-bound |

Phases 4 and 6 matter. Phase 4 stops the winner being overfit to the in-search eval. Phase 6 is the only measurement
taken on the weights you'll ship. To shorten the tail, use `--capability-n` or `--top-rescore`. Both measure less, so
lower them knowingly.

On CPU, phase 1 can sit for several minutes with no output. It isn't stuck. It works through its prompts a batch at
a time, and that loop doesn't report progress.

## The knobs

`kageyoshi` owns the search settings. If you pass `--trials` or `--max-directions` with it, they're ignored, and it
says so. For control, use the manual mode.

## Doing it by hand

```sh
senbonzakura abliterate \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --track mytrack \
    --out my-abliterated-model \
    --device cuda \
    --trials 200 \
    --max-directions 3
```

[Flags worth knowing](/reference/flags) lists the ones that change what a run means.

## When it finishes

You get a model directory you can load with `transformers`, plus `abliteration.json`: the winning configuration, the
seed, how many trials ran, the package versions and the commit.

The directory holds the weights and `abliteration.json`, with no prompt text or replies. The command that keeps
per-prompt rows is `senbonzakura compass`. It keeps them because a rate with no rows behind it can't be checked. Its
output then holds harmful prompts and replies, so don't push it to a public repository. `--no-margins` turns them off.

## Check whether you broke it

Your model has stopped refusing. Has it also stopped *understanding* that some requests are dangerous? Those are
different, and only one is a problem.

```sh
senbonzakura compass --model my-abliterated-model \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds \
    --out compass.json
```

[The compass](/guide/compass) explains what the number means, and what it doesn't.

## So what?

One command gets you an edited model. The second tells you what it cost. Running the first without the second is how
people publish a refusal rate for a model that has quietly been damaged.
