# Install

One command, and then you can go and read the other pages.

```sh
pip install senbonzakura
senbonzakura setup
```

`senbonzakura` depends on `senbonzakura-check`, so pip fetches both. The check package is torch-free. Install it alone
with `pip install senbonzakura-check` if you only want to check somebody's result file.

**You get the evaluation track with it.** The wheel carries the packed track and six research corpora, so `--track
default` works straight away. That's roughly 6,200 harmful prompts on your disk, and the
[README](https://github.com/elementmerc/senbonzakura#what-this-repository-does-not-contain) has the detail.

**On Linux you also get the quantiser.** With glibc 2.35 or newer, pip picks a wheel carrying llama.cpp's binaries, so
`convert` and `quantise` work straight away. Elsewhere the universal wheel installs without them. `senbonzakura doctor`
says which one you got.

:::: warning A wheel built from a clone does not carry `--track default`
The two `.bin` blobs in `src/senbonzakura/data/` are generated, not committed, because they hold harmful prompts. A wheel
built from a clone installs, imports and answers `--help`, then fails on `--track default`, and nothing warns you first.
Build the blobs before building the wheel:

```sh
senbonzakura corpora          # public corpora, pinned commits
python tools/packaging/pack_track.py --track <your-track>
```

::: warning `senbonzakura corpora` needs the GitHub CLI
It fetches through `gh`, so no credential is handled by this project's own code. Without `gh` it exits 1 and says so.
Install it from <https://cli.github.com>, run `gh auth login`, then run this again.
:::

The first step needs only the network. The second needs a track: [one you built](/guide/the-track), or the gated dataset.
::::

The install brings everything (torch, transformers, accelerate, optuna and the rest). Nothing is behind an extra.

It is **70 packages and 5.9 GB**, measured 2026-09-28 in an empty virtualenv on Python 3.14. The count is what `pip list`
prints:

| | |
|---|---|
| Rows in `pip list` | **69** |
| On disk | **5.9 GB** |
| Of those rows, `nvidia-*` wheels | 15 |
| Plus `triton` | 1 |
| Rows that are pip and setuptools rather than dependencies | 2 |
| Rows that are this project (`senbonzakura`, `senbonzakura-check`) | 2 |

Sixteen CUDA and Triton wheels account for almost all of the 5.9 GB. Different interpreters ship different defaults, so
this count decays without anyone editing it. That's why the interpreter and date are given, and why a test checks every
page that quotes these numbers.

## Choosing torch for your machine

pip picks by platform, not by hardware. The marker system knows the interpreter, OS and architecture, and nothing about
whether a GPU is present:

| Your machine | What pip alone gives you |
|---|---|
| Linux with a GPU | the CUDA build. Correct |
| Linux with no GPU | the CUDA build anyway, and about 15 CUDA packages you cannot use |
| macOS on Apple silicon | a build that uses Metal. Correct |
| Windows with a GPU | **a CPU-only build. Your card will sit idle** |

::: tip Why the Windows row is in bold
PyPI's Windows torch is 124 MB and CPU-only, and the CUDA build is not on PyPI at all. A gaming laptop with a 3060 installs
a torch that cannot see the card, and nothing warns you. The search then runs on CPU and takes a day instead of an hour.
:::

`senbonzakura setup` asks the driver (torch would report only what the installed build can reach), says what it found, and
prints the one command that fixes it. It changes nothing unless you add `--apply`.

### No graphics card? Install a CPU torch first

To avoid the 15 CUDA packages, install torch from the CPU index before the tool:

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

pip then leaves that torch alone. This is what the Docker image does, which is why the image is **2.11 GB rather than about
6 GB**. That figure is measured. The pip route's total is not, so treat the saving as the same shape, not the same number.

You give up the card, not the instruments. `compass`, `drift`, `score`, `check`, `doctor`, `convert` and `quantise` all run
on a processor. Abliteration runs too, slowly. See [the quickstart](/guide/quickstart#without-a-graphics-card).

### Disk tight? Move the model cache

Model weights go to Hugging Face's cache, `~/.cache/huggingface` by default, and they are the largest download. Qwen3-1.7B
is about 3.8 GB. To use a bigger disk:

```sh
export HF_HOME=/mnt/big-disk/huggingface
```

`senbonzakura doctor` reports the track cache, not the model cache, so it won't confirm the move. `hf env` prints
`HF_HUB_CACHE`, the directory the weights actually land in.

## Reading datasets from the Hub

```sh
pip install 'senbonzakura[hub]'
```

Every file format the tool accepts (`.txt`, `.csv`, `.json`, `.jsonl`, `.parquet`) works without the extra. It's needed only
for a Hub id such as `--good-ds tatsu-lab/alpaca`, and the tool names what to install if you ask without it.

## What comes in the box, including the part people don't expect

The wheel carries the evaluation track and six public research corpora, so `--track default` and `--good-ds advbench` work
with no network. That is roughly 6,200 harmful prompts inside your site-packages.

They're wrapped rather than plaintext. That stops a scraper finding them by accident and nothing else, because the key ships
beside them (`src/senbonzakura/bundled.py` says so). Each one is a public research dataset, and the tool prints the attribution
and licence the first time it loads one, because several require it.

To leave them out, build the wheel without running `senbonzakura corpora` and `tools/packaging/pack_track.py`. Everything except
the bundled defaults still works.

### The man page

The man page installs to `share/man/man1/senbonzakura.1`. `man senbonzakura` finds it when the environment you installed into is
the one you're using, so activating the virtualenv is enough.

| Route | Works? | Why |
|---|---|---|
| Virtualenv, activated | Yes | The venv's `bin` is on `PATH`, so its `share/man` is on the manual path |
| `pip install --user` | Yes, if `~/.local/bin` is on `PATH` | Same rule, applied to `~/.local` |
| `pipx install` | No | The executable is linked into `~/.local/bin`, but the page stays in pipx's own virtualenv |

For pipx, point `man` at the file directly:

```sh
man "$(python -c 'import sysconfig,pathlib; print(pathlib.Path(sysconfig.get_path("data"))/"share/man/man1/senbonzakura.1")')"
```

## If you're checking our numbers

Install against the pinned set:

```sh
pip install . -c constraints.txt
```

**This one needs Python 3.12 or newer**, even though the tool runs on 3.10. The numpy release pinned in `constraints.txt`
dropped 3.11. On 3.10 or 3.11 the command fails, and pip's message names only the package it couldn't find, which looks like
a broken pin.

`constraints.txt` records the exact version of every dependency the published numbers were measured with. The ordinary install
uses ranges, which say what a run *could* have used. It deliberately leaves out torch's CUDA suffix, so the file installs on a
machine without a GPU.

## Do you need a GPU?

For editing a model, yes, realistically. For everything else, no.

| What you're doing | What it needs |
|---|---|
| Uncensoring a model | A CUDA card. 6 GB has been measured at 1.7B, and 3B wants 8 to 12 GB |
| Scoring a model that already exists | A card, or a lot of patience on CPU |
| Running the test suite | Nothing. CPU, no downloads, no card |
| Building the corpus, checking contamination | Nothing |

The 6 GB figure is a limit, not a recommendation. This project was built around one 6 GB laptop card, which is why every model
it has run is under 3B parameters. [Limits](/guide/limits) explains what that means for the numbers. 1.7B is the only row anyone
has run. The rest is arithmetic.

## What size card {#what-size-card}

Loading the whole model sets the floor by arithmetic: 16-bit weights are two bytes per parameter, plus headroom for activations
and for the reference copy the divergence measurement compares against. Streaming needs far less, at the price of speed. See
[the way out](#the-way-out) below.

| Model size | Weights at 16-bit | Card that will do it, loading the whole model |
|---|---|---|
| 1.7B | 3.4 GB | 6 GB. Measured |
| 3B | 6 GB | 8 to 12 GB |
| 8B | 16 GB | 24 GB |
| 30B | 60 GB | 80 GB |

Only the first row is measured. The rest is arithmetic plus headroom.

::: warning A mixture-of-experts model needs room for every expert, not the active ones
A model described as "30B total, 3B active" runs like a 3B model, but uncensoring edits the weights, and it has to edit **all**
of them. Size the card, or the disk when streaming, for the 30B total.
:::

**`--load-in-4bit` doesn't help the editor, and the tool refuses it on the editing path.** A 4-bit tensor can't be rewritten in
place. It's accepted on the measuring commands, where nothing is rewritten.

### Which model formats the measuring commands load

Every command that measures a model (`score`, `drift`, `coherence`, `attribute` and the rest) loads the same formats and takes the
same flags. Plain safetensors needs nothing extra. For the others, the command tells you in one sentence which package is
missing and how to install it.

| The model is | What you pass | Extra to install |
|---|---|---|
| Ordinary safetensors | nothing | none |
| Any model, squeezed to 4 or 8 bits as it loads | `--load-in-4bit` or `--load-in-8bit` | `quant` |
| Saved already in 4 or 8 bit bitsandbytes form | nothing, it's read from the folder | `quant` |
| GPTQ | nothing | `gptq` |
| AWQ | nothing | `awq` |
| A GGUF file, or a folder or hub id holding one | the file path, or `--gguf-file NAME` | none (already included) |
| A model that ships its own code | `--trust-remote-code`, only if you trust the repository | none |

A GGUF is expanded back to full precision to be measured, so the figure is for those expanded weights and not for the file as
llama.cpp runs it. The result records the quantisation it came from. The editor refuses all of these, because it rewrites weights
and needs the full precision originals.

### Models trained at very low precision cannot be edited

Some models are **trained** at around 1.6 bits per weight, not squeezed afterwards. The training compensates for the squeezing.
Our edit rotates a sheet of those weights slightly, and snapping the rotated values back to the allowed levels undoes that
compensation. The result is worse in ways nobody can predict, so the edit destroys the thing it is editing. This gap is not
closed.

Measuring still works. Running a low-precision model is what it was built for, so every instrument here can check what somebody
else's edit did to one.

**On a card too small.** `accelerate` places what fits and pushes the rest to host RAM or disk. Host RAM is slow and works. Disk
**does not work**. A disk-offloaded tensor hands back a fresh copy on each read, so an edit would be written to something discarded,
and the model would come out unedited with nothing reporting it. The tool refuses, names the weight, and says what to do. The refusal
fires after the download and load, so do this arithmetic first.

### The way out {#the-way-out}

That limit is about writing, not reading. A forward pass only reads, so disk offload is sound for anything that measures. Streaming
takes the directions off a checkpoint larger than your card and RAM together, one group of layers at a time: `senbonzakura
stream-extract`, then `senbonzakura stream-bake` to apply them. `senbonzakura budget --model <a local checkpoint>` estimates what a
model will ask of the machine. See [the command reference](/reference/cli) for the flags.

## Converting and quantising needs one system library

A platform wheel carries the pinned `llama.cpp` binaries, so `convert`, `quantise` and `imatrix` work without a source checkout. The
binaries are built against OpenMP. A Python wheel can't ask the system for that library, so on a slim container or a fresh CI runner
they can fail to start. Measured on 2026-09-11, the same wheel fails inside `python:3.13-slim` and runs on Ubuntu under WSL, which has
`libgomp1`. The fault belongs to the image, not the wheel.

`senbonzakura doctor` names the missing library:

```
✗  llama-quantize   present at .../llama-quantize and cannot start: a shared library is
                    missing (libgomp.so.1)
                    -> install the library it names. On Debian and Ubuntu the usual one is
                       libgomp1: apt-get install -y libgomp1
```

Reinstalling won't help. The binary is correct and the system is missing a dependency. The container image carries the library:

```
docker run --rm -v "$PWD:/work" ghcr.io/elementmerc/senbonzakura:v0.5.1 doctor
```

## The optional extras

None is needed for a normal run.

| Extra | What it adds |
|---|---|
| `quant` | 4-bit loading through bitsandbytes, for *scoring* a model too big for your card |
| `completion` | tab completion for bash, zsh and tcsh |
| `hub` | reading datasets from the Hub (see above) |
| `finetune` | the adapter path of `tamper --method lora` (see [does the edit hold up](/guide/robustness)) |
| `abliterate` | nothing. It is an alias kept so older instructions still resolve |
| `all` | every one of the above |

**`pip install "senbonzakura[quant]"`** lets you *score* a model too big for your card:

```sh
senbonzakura score --load-in-4bit
```

This is a measurement option, not an uncensoring one. Uncensoring rewrites weights with an in-place multiplication, and a 4-bit
tensor is a compressed sketch of those weights. The editor loads in full precision, and there's no flag to override that.

**`pip install "senbonzakura[completion]"`** gives you tab completion:

```sh
senbonzakura --print-completion bash | sudo tee /etc/bash_completion.d/senbonzakura
```

## Where next

- [Quickstart](/guide/quickstart): two commands and an edited model.
- [Which models it can open](/reference/models), if you want to check yours is supported.
- [Your first run](/guide/first-run), for what those commands do.
