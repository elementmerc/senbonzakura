# Install

Two lines, and then you can go and read the interesting pages.

```sh
pip install .          # or: uv pip install .
senbonzakura --help
```

That pulls in pyarrow (and numpy, which pyarrow needs), landing at about 210 MB installed on
Linux with Python 3.14. It gets you the
commands that *check* things: build an evaluation split, audit one, check a benchmark for
contamination, check what your install can do. None of those need a model or a graphics card,
so none of them should make you download one.

To edit a model, add the extra:

```sh
pip install '.[abliterate]'
```

That's torch, transformers, accelerate and optuna, taking the install to about 1.5 GB with the
CPU-only torch wheel and a few minutes on a decent connection. The default CUDA wheel and its
`nvidia-*` dependencies are larger again. Read both figures as the size of the difference rather
than as what you will see. If you type
`senbonzakura abliterate` without the extra, the tool tells you and prints the line above
rather than showing you a traceback.

If you'd rather type `python -m senbonzakura` than `senbonzakura`, both work and they're the
same thing.

**Coming from 0.3.0?** Everything used to arrive in one install. `abliterate`, `convert` and
the scoring commands now live behind `[abliterate]`; nothing else changed.

## Reading datasets straight off the HuggingFace Hub

```sh
pip install '.[hub]'
```

The tool doesn't need the `datasets` package for ordinary work. It reads and writes tracks
with pyarrow, and every file format it accepts (`.txt`, `.csv`, `.json`, `.jsonl`,
`.parquet`) works without it. What needs the extra is pointing a flag at a Hub id like
`--good-ds tatsu-lab/alpaca`, which has to go and fetch it. Ask for one without the extra
and the tool says so and names what to install.

If you're packaging this for a distribution, that's the reason for the split: `datasets`
isn't in Debian at all, and pyarrow is.

A man page goes to `share/man/man1/senbonzakura.1`, so `man senbonzakura` works if your
system picks up manuals from wherever pip put them. Mine doesn't. Yours might.

## If you're checking our numbers rather than using the tool

Install against the pinned set instead:

```sh
pip install . -c constraints.txt
```

`constraints.txt` is the exact version of every dependency the published numbers were measured
with, read off the machine that produced them rather than resolved fresh. The ordinary install
above uses version *ranges*, which tell you what a run could have used; this tells you what it
did.

One thing it deliberately doesn't pin is torch's CUDA build suffix, so the file installs on a
machine without a GPU. The version is the part that changes arithmetic; the suffix changes how
many seconds it takes.

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

## Converting and quantising needs one system library

If you install a platform wheel (one whose filename ends in something other than `py3-none-any`),
it carries the pinned `llama.cpp` binaries so that `convert`, `quantise` and `imatrix` work
without a source checkout.

Those binaries are built against OpenMP, and a Python wheel has no way to ask your system for a
system library. On most desktop Linux installs it is already there. On a slim container image, a
minimal server, or a fresh CI runner it often is not, and the binaries cannot start.

`senbonzakura doctor` tells you which library is missing and what to install:

```
✗  llama-quantize   present at .../llama-quantize and cannot start: a shared library is
                    missing (libgomp.so.1)
                    -> install the library it names. On Debian and Ubuntu the usual one is
                       libgomp1: apt-get install -y libgomp1
```

Re-installing the package will not help, because the binary is correct and the system is missing
a dependency of it.

**If you would rather not think about this**, use the container image, which carries the library
and is checked at build time:

```
docker run --rm -v "$PWD:/work" senbonzakura doctor
```

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

::: warning Quantised uploads can't be abliterated, and the popular ones are quantised
Abliteration is weight surgery: it rewrites real matrices in place. A 4-bit or 8-bit upload
doesn't store those matrices in a form that can be rewritten, so the tool refuses it rather than
pretending. That includes the GGUF files most local runners use, and it includes the
`bnb-4bit` uploads that repackage popular models at half the size.

This catches people out because those uploads have a well-earned reputation for being smaller at
no cost to quality, so they're the natural thing to reach for. Start from the original
full-precision repository instead. You can quantise afterwards; you can't abliterate a
quantisation.
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
