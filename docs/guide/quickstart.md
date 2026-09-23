# Quickstart

::: danger What you are about to make
A model that will answer requests the original refused, including harmful ones. It is permanent in
the weights, the base model's licence still governs it, and you should not put one in front of
other people without saying what it is. [What it is](/guide/what-it-is) has the rest.
:::

One command to install, one to run. About an hour on a 6 GB card, most of it waiting.

You need a GPU with 6 GB or more, and Python 3.10 or newer. No card? Jump to
[without a graphics card](#without-a-graphics-card).

## Install

::: warning Not from PyPI, not yet
PyPI serves 0.3.0 from July 2026. Its numbers are withdrawn, and `senbonzakura-check` is not on
PyPI yet, so the current version cannot resolve from the index at all. Install from the repository
instead:

```sh
pip install "git+https://github.com/elementmerc/senbonzakura@dev#subdirectory=checker" \
            "git+https://github.com/elementmerc/senbonzakura@dev"
```

Both URLs, one command: given separately, the big package goes looking for the small one on PyPI
and does not find it. `@dev` is load-bearing too, or pip takes the default branch and hands you
something close to the version this box is warning you about.

**That install has no prompts in it.** The corpora and the bundled track are generated artefacts
kept out of git, because they are harmful prompts and a public repository is not where those
belong. So `--track default` will not work from it. Build them, in the install you just made:

```sh
senbonzakura corpora                   # the refusal corpora
senbonzakura track build --out corpus  # prompts, from public sources
senbonzakura track --harmful corpus/harmful.txt \
                   --harmless corpus/harmless.txt --out track
```

Or bring your own; [the track page](/guide/the-track) has both routes. A released wheel carries
all of it and none of this is needed.
:::

```sh
senbonzakura setup
```

That brings torch, transformers, accelerate and optuna: 68 packages, 5.8 GB, nothing to choose.

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
senbonzakura Qwen/Qwen3-1.7B
```

That is the whole command. The model is the only thing it cannot guess.

It writes to `./abliterated`, and picks a track: `./track` if you have built one, otherwise the
evaluation track bundled in the install. It says which in the log, because a default that quietly
depends on your working directory is how two runs of the same command stop being comparable.

Then go and make a cup of tea.

::: tip The bundled track is about 6,500 harmful prompts in your site-packages
Worth knowing before you put this on a shared machine.
[The install page](/guide/install#what-comes-in-the-box-including-the-part-people-don-t-expect)
says what is in it and how to build a wheel without it.
:::

Everything is still a flag when you want it:

```sh
senbonzakura Qwen/Qwen3-1.7B --track mytrack --out my-model --device cuda
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

`abliteration.json` records settings and numbers, no prompts and no replies. The command that
keeps per-prompt rows is `senbonzakura compass`, and it takes `--no-margins` if you would rather
it did not.
:::

## Without a graphics card {#without-a-graphics-card}

You cannot edit a model on CPU in any useful time, but the instruments run fine. This scores a
stock model on the toy track in the repository, in a couple of minutes:

```sh
senbonzakura compass \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --harmful examples/toy-track/bad_eval_ds \
    --harmless examples/toy-track/good_ds \
    --skip-harmful 0 --skip-harmless 0 --n 12 \
    --out compass-toy.json --device cpu
```

It asks whether the model still recognises a harmful request when it sees one. Twelve rows
measures nothing, and the tool makes you pass those three flags rather than pretending otherwise.
[The compass](/guide/compass) explains how to read it.

## Where next

- [Your first run](/guide/first-run) walks the same command in detail, plus manual mode.
- [What it is](/guide/what-it-is): what this does, and what it destroys.
- [Read this before quoting a number](/guide/what-we-know), if you plan to publish anything.
