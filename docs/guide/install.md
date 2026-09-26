# Install

One command, and then you can go and read the interesting pages.

```sh
pip install senbonzakura
senbonzakura setup
```

`senbonzakura` names `senbonzakura-check` as a dependency, so pip fetches both. The small one is
torch-free and installs in seconds; you can also install it on its own with
`pip install senbonzakura-check` if all you want is to check somebody's result file.

::: tip What changed
Until 0.4.0 this page said "not from PyPI, not yet" and gave a pair of `git+` URLs instead. PyPI
served 0.3.0, whose numbers are withdrawn, and `senbonzakura-check` was not published at all, so
`pip install senbonzakura` could not resolve. Both packages are on PyPI from 0.4.0 and the command
above is the whole install.
:::

**You get the evaluation track with it.** The wheel carries the packed track and six research
corpora, so `--track default` works immediately. That is roughly 6,200 harmful prompts on your
disk; the README's "what this repository does not contain" section has the detail.

**On Linux you get the quantiser too.** pip picks the most specific wheel that fits, so glibc 2.35
or newer gets one carrying llama.cpp's binaries and `convert` and `quantise` work straight away.
Everywhere else the universal wheel installs and works without them. `senbonzakura doctor` says
which one you received.

::: warning One thing that install does not give you: `--track default`
The two `.bin` blobs in `src/senbonzakura/data/` are generated rather than committed, because they
hold harmful prompts, so a build from a plain clone carries neither. The tool installs, imports
and answers `--help` exactly as normal, and then fails on `--track default`. That is not a subtle
failure once you meet it, but nothing warns you beforehand, which is why it is here.

Build them from a clone before building the wheel:

```sh
senbonzakura corpora          # public corpora, pinned commits
python tools/packaging/pack_track.py --track <your-track>
```

::: warning `senbonzakura corpora` needs the GitHub CLI
It fetches from public sources through `gh`, deliberately, so that no credential is ever handled
by this project's own code. Without it the command exits 1 and says so. Install `gh` from
<https://cli.github.com>, run `gh auth login`, then run this again.
:::

The first needs only the network. The second needs a track, which is either
[one you built](/guide/the-track) or the gated dataset.
:::

The first brings everything: torch, transformers, accelerate, optuna, and the rest. It is 68
packages and 5.8 GB on disk, measured on a clean Python 3.12 environment on
2026-09-11, and there is nothing else to choose. Nineteen of the 68 are NVIDIA CUDA and Triton
wheels, which is where almost all of it goes. (`pip list` will say 69: it counts pip itself.)
Nothing is behind an extra.

The second exists because **pip picks by platform, not by hardware**. PEP 508 environment
markers describe the interpreter, the operating system and the architecture, and there is no
marker for "has an NVIDIA GPU". A wheel runs no code at install time, so it cannot look. So what
you get depends on your operating system rather than on what is in the machine:

| Your machine | What pip alone gives you |
|---|---|
| Linux with a GPU | the CUDA build. Correct |
| Linux with no GPU | the CUDA build anyway, and about 15 CUDA packages you cannot use |
| macOS on Apple silicon | a build that uses Metal. Correct |
| Windows with a GPU | **a CPU-only build. Your card will sit idle** |

::: tip Why the Windows row is in bold
PyPI's Windows torch is 124 MB and CPU-only; the CUDA build is not on PyPI at all. So a gaming
laptop with a 3060 in it installs a torch that cannot see the card, and nothing warns you. The
search then runs on CPU and takes a day instead of an hour.
:::

`senbonzakura setup` asks the driver (not torch, which would report what the installed build can
reach rather than what the machine has), says what it found, and prints the one command that
fixes it. It changes nothing unless you add `--apply`.

If you'd rather type `python -m senbonzakura` than `senbonzakura`, both work and they're the
same thing.

**Coming from 0.3.x?** Nothing about your install changes: 0.3.0 also installed torch by
default. If you tracked `dev` in early September you were told to add an `[abliterate]` extra;
that extra still resolves, and now installs exactly what a bare install does.

## Reading datasets straight off the HuggingFace Hub

```sh
pip install 'senbonzakura[hub]'
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
and `--good-ds advbench` work with no network. That is roughly 6,200 harmful prompts sitting
inside your site-packages.

They're wrapped rather than plaintext, which stops a scraper finding them by accident and
stops nothing else: the key ships beside them and `src/senbonzakura/bundled.py` says so in as
many words. Every one of them is a public research dataset. The tool prints the attribution
and the licence the first time it loads one, because several of them require it.

If you'd rather not have them, build the wheel yourself without running
`senbonzakura corpora` and `tools/packaging/pack_track.py`; everything except the bundled defaults
still works, and the tool tells you what to run if you ask for one.

A man page goes to `share/man/man1/senbonzakura.1`, and `man senbonzakura` finds it
whenever the environment you installed into is the one you're using. `man` builds its
search path from `PATH`: for every `.../bin` on it, `man` also looks in the sibling
`.../share/man`. So activating the virtualenv is all it takes.

```sh
man senbonzakura        # with the virtualenv active
```

If it says `No manual entry`, the install route is what decides it.

| Route | Works? | Why |
|---|---|---|
| Virtualenv, activated | Yes | The venv's `bin` is on `PATH`, so its `share/man` is on the manual path |
| `pip install --user` | Yes, if `~/.local/bin` is on `PATH` | Same rule, applied to `~/.local` |
| `pipx install` | No | The executable is linked into `~/.local/bin`, but the page stays in pipx's own virtualenv, which never reaches `PATH` |

For the pipx case, or any other time you want the page without activating anything,
point `man` straight at the file:

```sh
man "$(python -c 'import sysconfig,pathlib; print(pathlib.Path(sysconfig.get_path("data"))/"share/man/man1/senbonzakura.1")')"
```

Measured on Debian's `man-db` 2026-09-26. **This page used to say the page probably
wouldn't be found, and that was wrong**: the machine it was written on had a broken
`man` binary, and a local fault got written down as a property of the tool.

## If you're checking our numbers rather than using the tool

Install against the pinned set instead:

```sh
pip install . -c constraints.txt
```

**This one needs Python 3.12 or newer**, even though the tool itself runs on 3.10. The numpy
release pinned in `constraints.txt` dropped support for 3.11, so on 3.10 or 3.11 the command
above fails to resolve and pip's message names only the package it could not find, which reads
like a broken pin rather than the wrong interpreter. Measured 2026-09-22, on a dependency
scanner that had quietly built itself a 3.11 environment and reported our pins as broken.

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

**It is a property of the image, not of the wheel**, and the same wheel goes both ways. Measured
on 2026-09-11: inside `python:3.13-slim` the quantiser will not start; on Ubuntu under WSL, from
the identical wheel, `doctor` reports "llama-quantize vendored, runs", because Ubuntu carries
`libgomp1` already. So a report of this from one machine says nothing about another, and
re-vendoring fixes neither.

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
docker run --rm -v "$PWD:/work" ghcr.io/elementmerc/senbonzakura:dev doctor
```

## The optional extras

None of these is needed for a normal run, and the tool works without all of them.

| Extra | What it adds |
|---|---|
| `quant` | 4-bit loading through bitsandbytes, for *scoring* a model too big to sit on your card |
| `completion` | tab completion for bash, zsh and tcsh |
| `hub` | reading datasets straight off the HuggingFace Hub (see above) |
| `abliterate` | nothing. It is an alias kept so 0.3.x instructions still resolve |
| `all` | every one of the above |

**`pip install "senbonzakura[quant]"`** adds 4-bit loading through bitsandbytes, so you can *score* a
model that's too big to sit on your card in full precision:

```sh
senbonzakura score --load-in-4bit
```

Note the word "score". This is a measurement option, not an abliteration one, and you can't
use it for the editing half. The reason is a bit lovely: abliteration works by rewriting
weights so they no longer point along the refusal direction, and that rewrite is a
multiplication done in place. A 4-bit tensor isn't a grid of numbers you can multiply, it's
a compressed sketch of one, and you can't do surgery on a sketch. So the abliterator loads
in full precision and there's no flag to talk it out of that.

**`pip install "senbonzakura[completion]"`** gets you tab completion for bash, zsh and tcsh. It's
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
