# Quickstart

::: danger What you are about to make
A model that will answer requests the original refused, including harmful ones. It is permanent in
the weights, the base model's licence still governs it, and you should not put one in front of
other people without saying what it is. [What it is](/guide/what-it-is) has the rest.
:::

One command to install, one to run, and most of the time is spent waiting. The nearest thing to a
measured expectation: **108 minutes from the command to `DONE`**, end to end, for **Qwen3-0.6B** on
a 6 GB card at the defaults with the weights already cached. That is a different model from the one
below, of about the same size, so treat it as the right order of magnitude rather than a promise.
"About an hour" is roughly what the *search* alone costs, and about half the wall clock falls
**after** the search's progress line reaches `ETA 0s`; [your first run](/guide/first-run) breaks
that down.

You need a GPU with 6 GB or more, and Python 3.10 or newer. No card? Jump to
[without a graphics card](#without-a-graphics-card).

## Install

```sh
pip install senbonzakura
```

That pulls `senbonzakura-check` with it, because the uncensoring tool depends on it by name.

**The prompts come with it.** The wheel carries six research corpora and the packed evaluation
track, so `--track default` works with nothing else to fetch or build. That is roughly 6,200
harmful prompts written to your disk, which is worth knowing before you install rather than
after.

**What follows builds a track you pass by name, and does NOT restore `--track default`.** That
one needs the packed blob, which is written by `tools/packaging/pack_track.py`, and `tools/`
ships in no wheel, so it is reachable from a clone and not from a `pip install git+...`. Use
`--track track` with what you build here; that is the working arrangement, not a consolation
prize. This box said "build them" without saying which of the two you get, which sent a reader
who followed it exactly back to the failure it was written to prevent.

<!-- documented-first-run: quickstart -->
```sh
senbonzakura corpora                   # the refusal corpora
senbonzakura track build --out corpus  # prompts, from public sources
senbonzakura track --harmful corpus/harmful.txt \
                   --harmless corpus/harmless.txt --out track
```

::: warning `senbonzakura corpora` needs the GitHub CLI
It fetches from public sources through `gh`, deliberately, so that no credential is ever handled
by this project's own code. Without it the command exits 1 and says so. Install `gh` from
<https://cli.github.com>, run `gh auth login`, then run this again.
:::

Or bring your own; [the track page](/guide/the-track) has both routes. A released wheel carries
all of it and none of this is needed.

```sh
senbonzakura setup
```

The install brought torch, transformers, accelerate and optuna: 70 packages, 5.9 GB, nothing to
choose.
[Install](/guide/install) has the breakdown and the date it was measured.

`setup` exists because **pip picks by platform, not by hardware**:

| Your machine | What pip alone gives you |
|---|---|
| Linux with a GPU | the CUDA build. Correct |
| Linux with no GPU | the CUDA build anyway, and 15 CUDA packages you cannot use |
| macOS | a Metal build. Correct |
| Windows with a GPU | **a CPU-only build. Your card sits idle** |

`setup` reads the machine, says what it found, and prints the command that fixes it. It changes
nothing unless you add `--apply`. The Windows row is bold because a gaming laptop with a 3060 in it
installs a torch that cannot see the card, nothing warns you, and the search then takes a day
instead of an hour.

## Edit a model

```sh
senbonzakura Qwen/Qwen2.5-0.5B-Instruct
```

That is the whole command. The model is the only thing it cannot guess.

It writes to `./abliterated`, and picks a track: `./track` if you have built one, otherwise the
evaluation track bundled in the install. It says which in the log, because a default that quietly
depends on your working directory is how two runs of the same command stop being comparable.

Then go and make a cup of tea.

::: tip The bundled track is about 6,200 harmful prompts in your site-packages
Worth knowing before you put this on a shared machine.
[The install page](/guide/install#what-comes-in-the-box-including-the-part-people-don-t-expect)
says what is in it and how to build a wheel without it.
:::

Everything is still a flag when you want it:

```sh
senbonzakura Qwen/Qwen2.5-0.5B-Instruct --track mytrack --out my-model --device cuda
```

## What you get

A model in `./abliterated`, with `run.json` recording what was done and `abliteration.json`
recording what it cost: refusals left, how far the model drifted, and whether the output turned to
mush.

While it runs you will see lines like this:

```
trial 47: o(P=18,wmax=0.62) d(P=14,wmax=0.31) K=2 per_layer -> refusals=3.1% soft=5.2% heretic=12.5% broken=0% KL=0.1900 obj=0.2431
```

`refusals` is the one you came for. `KL` is what it cost you, lower is less damage. Briefly:
`soft` counts hedging too, `heretic` is the other tool's keyword metric so the two are comparable,
`broken` is output that stopped being English, `obj` is the number the search minimises.

::: warning What you are now holding
The model will answer things the original declined to, and loading it differently does not undo
that: the behaviour is gone from the weights. **The base model's licence still governs it**, and
this tool cannot loosen those terms. If you publish it, `senbonzakura report` writes the card that
should travel beside the weights.

The refusal rates in `abliteration.json` come from the rows the search chose its winner by
scoring, so they are the best of all those attempts rather than a reading of the model. Score the
rows it held back with `senbonzakura score` and pass that file to `senbonzakura report --refusal`:
the card then publishes the figure with its sample size and its interval, and says which rows it
came from.

`abliteration.json` records settings and numbers, no prompts and no replies. The command that
keeps per-prompt rows is `senbonzakura compass`, and it takes `--no-margins` if you would rather
it did not.
:::

## Without a graphics card {#without-a-graphics-card}

You cannot edit a model on CPU in any useful time, but the instruments run fine. This scores a
stock model on the track bundled in your install:

```sh
senbonzakura compass \
    --model Qwen/Qwen3-1.7B \
    --track default \
    --harmful default/bad_eval_ds \
    --harmless default/good_ds \
    --n 60 \
    --out compass-toy.json --device cpu
```

**Measured: 3 minutes 44 seconds on sixteen cores**, with the compass at 0.9872 (95% interval
0.9722 to 0.9972) against a length-only ruler's 0.6853. The artefact and the exact environment are
in [evidence/compass-2026-10-11](https://github.com/elementmerc/senbonzakura/blob/dev/evidence/compass-2026-10-11/README.md).
Fewer or slower cores take proportionally longer, and nobody has timed it on an older processor. The
model is **about 3.8 GB** the first time, which is the real cost of this command.

::: tip Why this model, and why not a cheaper one. Measured 2026-10-11.
Two smaller models were tried here and both refuse to report a figure, for two different reasons.
That is the tool working, and it makes them poor first commands, because a run that exits 1 reads
as a broken install.

`HuggingFaceTB/SmolLM2-135M-Instruct`, until 2026-09-26: at 135M the model puts no verdict token
where the compass reads one, so the run prints `MARGIN_READOUT_SUSPECT` and declines to call its
AUC a measurement.

`Qwen/Qwen3-0.6B`, until 2026-10-11, and this page recommended it while claiming it "exits 0". It
does not. It answers with a verdict on 100% of prompts, which is the part that was true, and then
exits 1 with `MARGIN_NULLS_DOMINATE`, because a control that reads nothing but surface text
separates the two sets as well as the model does. Both sample sizes were measured:

| Rows | Compass AUC | The control that beat it | Wall clock |
|---|---|---|---|
| 12 | 0.7292, interval [0.5174, 0.9097] | characters, 1.0000 | 27s |
| 60 | 0.7047, interval [0.6126, 0.7940] | punctuation density, 0.2590, which is informative in reverse and further from chance than the compass | 83s |

So the fault was never only the flags this command used to carry. At 0.6B the signal is too weak on
this track for any sample size worth running on a processor, and twelve rows could not have
supported a claim regardless.
:::

`default/<partition>` means the evaluation track inside the install, so there is no corpus to
fetch and nothing to assemble. The **model** is still downloaded from the Hub the first time, so
this one does need a network. It used to say `examples/toy-track/...`, which exists in a clone and in no
install, so the one recipe written for readers without a card was the one they could not run. If you
did clone the repository, `examples/toy-track/` still works and is smaller.

It asks whether the model still recognises a harmful request when it sees one. `--track default`
is what makes the answer mean anything: it reads the track's recorded boundary and scores rows 132
and 385 onward, so the figure is taken on prompts nothing was fitted or selected on. Without it the
run would score the opening rows of each file, which are the search and fit rows, and report a
plausible number anyway. [The compass](/guide/compass) explains how to read the result.

## Where next

- [Your first run](/guide/first-run) walks the same command in detail, plus manual mode.
- [What it is](/guide/what-it-is): what this does, and what it destroys.
- [Read this before quoting a number](/guide/what-we-know), if you plan to publish anything.
