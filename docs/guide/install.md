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

## What comes in the box, including the part people don't expect

The wheel carries the evaluation track and six public research corpora, so `--track default`
and `--good-ds advbench` work with no network. That is roughly 6,500 harmful prompts sitting
inside your site-packages.

They're wrapped rather than plaintext, which stops a scraper finding them by accident and
stops nothing else: the key ships beside them and `src/senbonzakura/bundled.py` says so in as
many words. Every one of them is a public research dataset. The tool prints the attribution
and the licence the first time it loads one, because several of them require it.

If you'd rather not have them, build the wheel yourself without running
`tools/build_corpora.py` and `tools/pack_track.py`; everything except the bundled defaults
still works, and the tool tells you what to run if you ask for one.

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

## Where next

- [Quickstart](/guide/quickstart) is two commands and an edited model.
- [Which models it can open](/reference/models) if you want to check yours is supported.
- [Your first run](/guide/first-run) for what those commands are actually doing.
- [Running the tests and contributing](/contributing) if you are here to work on the tool
  rather than to use it.
