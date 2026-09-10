# Quickstart

Two commands and about an hour, with nothing to download but the model.

You need a GPU with 6 GB or more, and Python 3.10 or newer. No GPU? Skip to
[try it without a graphics card](#try-it-without-a-graphics-card) below.

## Install

```sh
pip install 'senbonzakura[abliterate]'
```

That is 68 packages, because `abliterate` brings torch, transformers, accelerate and optuna. On
Linux and Windows, 15 of them are CUDA libraries: **pip resolves by platform, not by hardware**, so
you get the CUDA build of torch whether or not there is a card in the machine.

If you have no GPU, take torch from the CPU index first and the rest will fit around it. That is
10 packages instead of 68, and none of them CUDA:

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install 'senbonzakura[abliterate]'
```

Plain `pip install senbonzakura`, with no extra, is two packages and no torch at all. It gets you
the commands that check things: build a track, audit one, look for contamination, run `doctor`.
None of those need a model or a card, so none of them make you download one.

## Edit a model

```sh
senbonzakura kageyoshi \
    --model Qwen/Qwen3-1.7B \
    --track default \
    --out my-abliterated-model \
    --device cuda
```

Then go and make a cup of tea. On a 6 GB card, a 1.7B model takes about an hour.

`--track default` is the evaluation track bundled inside the wheel. It needs no network and no
files on disk, so there is nothing to assemble before your first run. When you want to measure
your own model on your own prompts, [build a track](/guide/the-track); until then this is the one
to use.

## What you get

A model in `my-abliterated-model`, and beside it a `run.json` recording what was done, and an
`abliteration.json` recording what it cost: how many refusals are left, how far the model drifted
from the original, and whether the output turned to mush.

While it runs you will see lines like this:

```
trial 47: o(P=18,wmax=0.62) d(P=14,wmax=0.31) K=2 -> refusals=3.1% heretic=12.5% broken=0% KL=0.19
```

`refusals` is the one you came for. `KL` is what it cost you.

## Try it without a graphics card {#try-it-without-a-graphics-card}

You cannot edit a model on CPU in any useful time, but you can run the instruments. This scores
a stock model on the toy track committed in the repository, on CPU, in a couple of minutes:

```sh
senbonzakura compass \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --harmful examples/toy-track/bad_eval_ds \
    --harmless examples/toy-track/good_ds \
    --skip-harmful 0 --skip-harmless 0 --n 12 \
    --out compass-toy.json --device cpu
```

It asks whether the model still recognises a harmful request when it sees one. See
[the compass](/guide/compass) for how to read what comes back, which is worth doing before you
trust any number this tool prints.

## Where next

- [Your first run](/guide/first-run) walks through the same command in detail, and the manual mode.
- [What it is](/guide/what-it-is) is the background, if you would rather have it before the buttons.
- [Read this before quoting a number](/guide/what-we-know) matters if you plan to publish anything
  you measured with this.
