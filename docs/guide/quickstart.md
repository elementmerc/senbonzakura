# Quickstart

Two commands and about an hour. The install is 5.8 GB and takes about two minutes; after that there is nothing to
download but the model.

You need a GPU with 6 GB or more, and Python 3.10 or newer. No GPU? Skip to
[try it without a graphics card](#try-it-without-a-graphics-card) below.

## Install

```sh
pip install senbonzakura
senbonzakura setup
```

The first command brings everything: torch, transformers, accelerate, optuna. It is 68 packages
and 5.8 GB on disk, measured 2026-09-11, and there is nothing else to choose. (`pip list` will
say 69, because it counts pip itself.)

The second command exists because **pip picks by platform, not by hardware**. There is no way for
a package to say "install the CUDA build if there is a card", so what you get depends on which
operating system you are on rather than on what is in the machine:

| Your machine | What pip alone gives you |
|---|---|
| Linux with a GPU | the CUDA build. Correct |
| Linux with no GPU | the CUDA build anyway, and about 15 CUDA packages you cannot use |
| macOS | a build that uses Metal. Correct |
| Windows with a GPU | **a CPU-only build. Your card will sit idle** |

`senbonzakura setup` looks at the machine, says what it found, and prints the one command that
fixes it. It changes nothing unless you add `--apply`.

::: tip Why the Windows row is in bold
PyPI's Windows torch is 124 MB; the CUDA one is not on PyPI at all. So a gaming laptop with a
3060 in it installs a torch that cannot see the card, and nothing warns you. The search then runs
on CPU and takes a day instead of an hour.
:::

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
files on disk, so there is nothing to assemble before your first run. It is also **roughly 6,500
harmful prompts sitting inside your site-packages**, which is worth knowing before you put this on
a shared machine; [the install page](/guide/install#what-comes-in-the-box-including-the-part-people-don-t-expect) says what is in it and
how to build a wheel without it. When you want to measure
your own model on your own prompts, [build a track](/guide/the-track); until then this is the one
to use.

## What you get

A model in `my-abliterated-model`, and beside it a `run.json` recording what was done, and an
`abliteration.json` recording what it cost: how many refusals are left, how far the model drifted
from the original, and whether the output turned to mush.

::: warning What you are now holding
The model will answer things the original declined to answer, and that is not reversible by
loading it differently: the refusal behaviour has been removed from the weights. **The base
model's licence still governs it**, unchanged, and this tool cannot loosen those terms. If you
publish it, run `senbonzakura report` to generate the card that should travel beside the weights.

The tool also writes a `margins` file next to your output by default, and those rows contain the
prompts and what the model said to them. Pass `--no-margins` if you would rather it did not.
:::

While it runs you will see lines like this:

```
trial 47: o(P=18,wmax=0.62) d(P=14,wmax=0.31) K=2 per_layer -> refusals=3.1% soft=5.2% heretic=12.5% broken=0% KL=0.1900 obj=0.2431
```

`refusals` is the one you came for: hard refusals left, as a percentage. `KL` is what it cost
you, and lower is less damage. The rest, briefly: `soft` counts hedging as well as refusal,
`heretic` is the other tool's keyword metric reported so the two are comparable, `broken` is
output that stopped being English, and `obj` is the single number the search is minimising.

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
