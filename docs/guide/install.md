# Install

Two lines, and then you can go and read the interesting pages.

```sh
pip install .          # or: uv pip install .
senbonzakura --help
```

That pulls in torch, transformers, accelerate, datasets and optuna, which is most of a
gigabyte and takes a few minutes on a decent connection. If you'd rather type
`python -m senbonzakura` than `senbonzakura`, both work and they're the same thing.

A man page goes to `share/man/man1/senbonzakura.1`, so `man senbonzakura` works if your
system picks up manuals from wherever pip put them. Mine doesn't. Yours might.

## Do you need a GPU?

For editing a model, yes, realistically. For everything else, no.

| What you're doing | What it needs |
|---|---|
| Abliterating a model | A CUDA card. 6 GB gets you to about 2B parameters |
| Scoring a model that already exists | A card, or a lot of patience on CPU |
| Running the test suite | Nothing. CPU, no downloads, no card |
| Building the corpus, checking contamination | Nothing |

The 6 GB figure isn't a recommendation, it's a confession: this whole project is built
around one 6 GB laptop card, which is exactly why every model it's ever been run on is
under 3B parameters. [Limits](/guide/limits) is blunt about what that means for the
numbers.

## The two optional extras

Neither is needed for a normal run, and the tool works without both.

**`pip install ".[quant]"`** adds 4-bit loading through bitsandbytes, so you can *score* a
model that's too big to sit on your card in full precision:

```sh
python -m senbonzakura.score --load-in-4bit
```

Note the word "score". This is a measurement option, not an abliteration one, and you can't
use it for the editing half. The reason is a bit lovely: abliteration works by rewriting
weights so they no longer point along the refusal direction, and that rewrite is a
multiplication done in place. A 4-bit tensor isn't a grid of numbers you can multiply, it's
a compressed sketch of one, and you can't do surgery on a sketch. So the abliterator loads
in full precision and there's no flag to talk it out of that.

**`pip install ".[completion]"`** gets you tab completion for bash, zsh and tcsh. It's
generated on demand, the same one-time dance `pip`, `gh` and `poetry` all use:

```sh
senbonzakura --print-completion bash | sudo tee /etc/bash_completion.d/senbonzakura
```

Without the extra, the `--print-completion` flag simply isn't there. Nothing else changes.

## Running the tests

You don't need a model, a download or a graphics card. The whole suite runs on CPU against
small hand-built fixtures, which is deliberate: a test suite you can only run on the
machine that has the card is a test suite that gets run once a week.

```sh
python -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest              # add --cov for coverage
```

That first torch line pulls the CPU build, which is about a fifth of the size of the CUDA
one. If you already have a CUDA torch installed, skip it; the tests don't care.

## If you're going to commit

Wire the local gates once per clone:

```sh
bash tools/install-local-hooks.sh
```

That hooks `tools/check_prompt_artefacts.py` into your pre-commit path, and it exists
because of a specific hazard rather than as general tidiness.

::: warning What the gate is actually stopping
The tool keeps per-prompt margins and generations by default, because every single scoring
bug in this project's history was invisible in the percentages and completely obvious in the
text. Those rows contain harmful prompts and the replies a model gave to them. They're the
most useful debugging artefact here and the last thing you want to push to a public
repository at half past one in the morning.

The gate refuses any staged JSON or JSONL carrying a `prompt` or `generation` field. CI runs
the same check across the whole tree, but by the time CI sees it, the commit exists.
:::

## Which models it can actually open

- Dense transformers: Llama, Qwen, Mistral, Gemma, Phi and the rest of that shape.
- Fused-expert mixture-of-experts: Qwen3-MoE, Granite-MoE.
- Mixtral, fused or unfused.
- OLMoE.
- Shared-expert MoE: Qwen2-MoE, DeepSeek-MoE.
- LFM2, including its MoE variant. These are **hybrids**: some of their layers hold a short
  convolution where other models hold attention, and that convolution writes into the model's
  running state exactly as attention does. On LFM2.5-350M it is 10 layers out of 16. Those are
  edited too, because editing the other six and reporting success would be an abliteration that
  never reached most of the model.

::: tip New words: dense and mixture-of-experts
A **dense** model runs every one of its weights on every token. A **mixture-of-experts**
model keeps a pile of specialist sub-networks and routes each token to a couple of them, so
it's big on disk and cheap to run. It matters here because the two store their weights in
different shapes, and abliteration is weight surgery: you have to know which drawer things
are in.
:::

An architecture it doesn't recognise **fails loudly at load** with the layer type named. It
would be easy to make it shrug and carry on, and the result would be a model that came back
looking abliterated and wasn't, because the edit never reached the layers that mattered.
That isn't a hypothetical. It's precisely what happened on Gemma for months, and it cost
this project every Gemma number it had ever published.

## Where next

- [Your first run](/guide/first-run) for the shortest path to an edited model.
- [The method](/guide/how-it-works) if you'd like to know what it's about to do to your
  weights before you let it.
