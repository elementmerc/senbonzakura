# Benchmarking against another tool

There's no results table on this page yet, and that's not an oversight.

A head-to-head against [Heretic](https://github.com/p-e-w/heretic), the other open-source
abliteration tool with an automated search, is built and tested and hasn't been run to
completion. When it has, the table goes here. Until then this page is the recipe, so you
can run it yourself and not have to take our word for anything.

::: tip Why the delay is the honest option
Several correctness fixes landed recently in direction extraction, the multi-direction
basis and knee selection, and they moved our own numbers noticeably in our favour. That is
precisely the moment to slow down. A comparison run on code you've just improved, against a
competitor you haven't touched, is not a comparison. So it gets re-run under the corrected
code, on identical ground, and then published whichever way it falls.
:::

## Run it yourself

The comparison is a command, not a private script we keep in a drawer. It runs on one
machine, needs no orchestrator, and produces the arms, the scores and the report:

```sh
# 1. Cut the prompt slices every tool is scored on, from one corpus.
python -m senbonzakura.bench stage --track mytrack --out slices

# 2. Run every tool over every seed, score every model, print the verdict.
python -m senbonzakura.bench head-to-head \
    --tools senbon,heretic --seeds 42,43,44,45,46 --trials 200 \
    --model Qwen/Qwen3-1.7B --track mytrack --eval-slices slices \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds \
    --out results/h2h

# 3. Read a finished run again later, without re-running anything.
python -m senbonzakura.bench report results/h2h
```

::: tip New word: arm
One tool, one seed, one run. Five seeds and two tools is ten arms. The word comes from
clinical trials, where an arm is one group of patients getting one treatment, and it's the
right word here for the same reason: the whole design rests on the arms being identical
except for the one thing you're testing.
:::

## What makes it a comparison rather than two runs

This is the part that's easy to get wrong and hard to notice you got wrong.

Both tools get the same corpus, the same trial budget and the same prompt slices. Then
every model either tool produces is scored afterwards **by one instrument**: our compass, on
held-out prompts, run by us. Nobody grades their own homework.

The slices record which corpus they were cut from, and the run refuses to start if that
doesn't match the corpus you passed. That guard exists because of a hole found while writing
the tests: once one tool reads the corpus directly and the other reads slices cut from it,
"both tools read the same corpus" stops being visible anywhere on either command line. Every
arm would have finished, every artefact would have been present, and the table would have
meant nothing at all.

Each tool's own reported numbers are printed too, in separate rows labelled with whose
estimator produced them. **No gap between those rows is ever called a win.** Two tools
reporting different refusal rates might mean one is better, or it might mean they count
refusals differently, and from the outside those look identical.

## Sandbox anything you didn't write

```sh
--isolate docker --image senbon=IMAGE --image heretic=IMAGE
```

Each arm then runs with no network, read-only inputs, no capabilities and no credentials.

Without it you get a warning, because otherwise a third-party abliteration tool is running
on your machine with your network and your keys, on the strength of you having been curious
about a benchmark. We use it on our own arms too, which is less about trusting ourselves and
more that an isolated arm and a comfortable one aren't the same measurement.

## Stopping and starting is safe

An arm is skipped only when a manifest agrees with this run's tool, seed, model and budget
**and** every artefact it declared is present.

An arm that exits cleanly having produced nothing counts as a failure and leaves no
manifest, so the next run retries it rather than inheriting the silence. This is a rule
learned the hard way: on 2026-08-05 a rehearsal came back with five jobs done having
measured absolutely nothing, because the scoring step printed success unconditionally. A
green tick that can't go red isn't a check.

## Fewer than three seeds gets no verdict

A spread calculated from two points is arithmetic dressed up as statistics, so the report
declines to name a winner and says why. And a gap smaller than the spread is reported as a
tie, not as a narrow victory.

::: tip New word: seed
The number that fixes all the random choices in a run. Same seed, same choices, same result.
Different seeds tell you how much of your result was the method and how much was luck, which
is why one seed is an anecdote.
:::

## Adding another tool

An adapter, and it's data rather than logic: how to invoke the tool, what output proves it
actually ran, where it leaves its model, and how to read its own reported figures.

See `ADAPTERS` in `senbonzakura/bench.py`. A pull request adding one is genuinely welcome,
including from the author of the tool you're adding.

## Where next

- [The compass](/guide/compass), which is the instrument doing the scoring here.
- [Contamination](/guide/contamination), because a comparison on prompts both tools trained
  on isn't one.
