<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/elementmerc/senbonzakura/b02383ab124f8d38f2886323d39b8a933883c6ce/assets/brand/readme-banner-dark.png">
  <img src="https://raw.githubusercontent.com/elementmerc/senbonzakura/b02383ab124f8d38f2886323d39b8a933883c6ce/assets/brand/readme-banner.png" width="820"
       alt="Senbonzakura">
</picture>

**The world's top open-weight AI model workshop.**

Measure, uncensor and audit.

<p>
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/elementmerc/senbonzakura/ci.yml?branch=main&amp;logo=githubactions&amp;logoColor=white&amp;label=CI&amp;labelColor=24292f" alt="CI on the released branch" /></a>
  <a href="https://pypi.org/project/senbonzakura/"><img src="https://img.shields.io/badge/python-3.10%20to%203.14-3776AB?logo=python&amp;logoColor=white&amp;labelColor=24292f" alt="Python 3.10 to 3.14" /></a>
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-4C566A?labelColor=24292f" alt="Platform: Linux, macOS and Windows" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-AGPL--3.0--or--later-A42E2B?logo=gnu&amp;logoColor=white&amp;labelColor=24292f" alt="Licence: AGPL-3.0-or-later" /></a>
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/tests-10100-0A9EDC?logo=pytest&amp;logoColor=white&amp;labelColor=24292f" alt="10100 tests" /></a>
  <a href="https://app.codecov.io/gh/elementmerc/senbonzakura/tree/dev"><img src="https://img.shields.io/codecov/c/github/elementmerc/senbonzakura/dev?logo=codecov&amp;logoColor=white&amp;label=coverage%20dev&amp;labelColor=24292f" alt="Coverage on the development branch, measured in CI" /></a>
</p>

<a href="https://elementmerc.github.io/senbonzakura"><strong>Documentation</strong></a>
&nbsp;·&nbsp;
<a href="CHANGELOG.md">Changelog</a>
&nbsp;·&nbsp;
<a href="https://elementmerc.github.io/senbonzakura/guide/what-we-know">What is and is not established</a>

</div>

---

Senbonzakura removes the refusal behaviour from an open-weight language model, and measures what
that removal cost.

The technique's name in the literature is abliteration, and this page calls it uncensoring
because that's what people looking for it call it.

Removing it is the easy half. Any uncensoring tool can stop a model saying "I can't help with
that". The hard part is knowing whether you also took out its reasoning, and a tool that cannot
tell the difference will report a lobotomy as a success. So every number here arrives with its
conditions: what it was measured on, on rows nothing was fitted or selected on, with an
interval, and with a control that would expose it if it were measuring the wrong thing.

Named for Byakuya Kuchiki's zanpakutō, the sword that scatters into a thousand blades.

> **What it makes.** A model that will answer requests the original refused, including harmful
> ones. That is the point, and it is permanent in the weights. The base model's licence still
> governs the result. Don't put one in front of other people without saying what it is. The
> package carries roughly 6,200 harmful prompts so it runs offline.
>
> [The detail](#what-this-repository-does-not-contain), and `ACCEPTABLE-USE.md` ships in the
> package.

## Try it without installing anything

[**Open the notebook in Colab**](https://colab.research.google.com/github/elementmerc/senbonzakura/blob/v0.5.0/notebooks/senbonzakura_colab.ipynb).
Free GPU, nothing on your machine. It measures a model, edits it, then measures what that cost.
Nobody has timed it on Colab's hardware, so it gives you no duration to hold us to.

Or, if you have Docker:

```sh
docker run --rm ghcr.io/elementmerc/senbonzakura:v0.5.0 doctor
```

That image is **2.11 GB** (measured), because its torch comes from the CPU index. It cannot use a
graphics card even with `--gpus all`, and `doctor` says so rather than pretending.

`doctor` reports what your install can and cannot do, which on a first run is more useful than it
sounds.

## Install

```sh
pip install senbonzakura
senbonzakura setup
```

That brings torch, transformers, accelerate and optuna with it: **70 packages, 5.9 GB**, fifteen
of them CUDA wheels.

**If you have no graphics card, install torch from the CPU index first and those fifteen never
arrive:**

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Then install the tool as above. pip finds torch already satisfied and leaves it alone. This is the route the Docker image
takes, which is **2.11 GB rather than about 6 GB** (measured, and the only measured figure here:
nobody has timed or sized the pip route separately). Everything that doesn't need a card still
works, which is every instrument: `compass`, `drift`, `score`, `check`, `doctor`, `convert` and
`quantise`. Editing a model is the part that really wants a card.

> This paragraph used to tell you *not* to run that command, and it was right to until 0.4.0 went
> out. [What we got wrong](https://elementmerc.github.io/senbonzakura/guide/what-we-got-wrong) has
> that one and every other correction, kept rather than deleted.

**`--track default` works from it.** The wheel carries the packed evaluation track and six
research corpora, so the commands below run straight after installing, with no corpus to fetch
and nothing to build. That is also roughly 6,200 harmful prompts written to your disk, which is
the thing to know before you install rather than after;
[the detail](#what-this-repository-does-not-contain) is below.

**On Linux you get the quantiser too.** pip picks the most specific wheel that fits, so a glibc
2.35 or newer machine receives one carrying llama.cpp's binaries and `convert` and `quantise`
work immediately. Everywhere else, and on older Linux, the universal wheel installs and works
without them. `senbonzakura doctor` says which you got.

**If pip refuses with "externally-managed-environment", make a virtual environment.** Debian,
Ubuntu and Fedora mark the system Python as managed by the package manager, so `pip install` into
it is blocked, and so is installing into your home directory. A stranger trying this on a fresh
Ubuntu hit that and
gave up, which is the only install note here written from somebody failing rather than from the
documentation:

```sh
python3 -m venv ~/senbonzakura
~/senbonzakura/bin/pip install senbonzakura
~/senbonzakura/bin/senbonzakura doctor
```

Everything after that uses `~/senbonzakura/bin/senbonzakura`, or you activate it once with
`source ~/senbonzakura/bin/activate` and type `senbonzakura`. Do not reach for
`--break-system-packages`: it means what it says, and this package brings torch with it.

**A stale copy on your PATH will shadow a new install.** If `senbonzakura --version` disagrees
with what you just installed, an older one is earlier in the PATH; `which -a senbonzakura` lists
every copy.

`senbonzakura setup` exists because **pip picks by platform, not by hardware**. On Windows, PyPI's
torch is CPU-only, so **a card there sits idle** and nothing warns you. It says what it found and
prints the command that fixes it, changing nothing unless you add `--apply`.
[Install](https://elementmerc.github.io/senbonzakura/guide/install) has the full table.

Editing a model wants a CUDA card, and **how much of one scales with the model**: the weights have
to be resident in full precision, because the uncensoring edit rewrites them in place. 6 GB is
enough for the sizes this project has actually measured, all under 3B.
[What size card](https://elementmerc.github.io/senbonzakura/guide/install#what-size-card) does the
arithmetic, including the trap that a mixture-of-experts model needs room for *all* its experts and
not just the active ones. The measuring commands run on CPU.

## One command, on your own machine

No corpus, no GPU, no model to edit first. This reads the evaluation track bundled in the install,
so it works straight after `pip install`:

```sh
senbonzakura compass \
    --model Qwen/Qwen3-1.7B \
    --track default \
    --harmful default/bad_eval_ds \
    --harmless default/good_ds \
    --n 60 \
    --out compass-toy.json --device cpu
```

**`--track default` is the load-bearing flag, not the model.** The bundled track records where its
own partition boundaries fell, and passing it is the only way the run skips exactly the rows the
edit was fitted on and the search selected from, rather than a guess at them. Leave it out and the
run scores whichever rows happen to come first, which on this track are the fitting rows: a number
that looks fine and is marking its own homework.

The compass also has to beat a control rather than just score well. A ruler that reads nothing but
sentence length is run over the same prompts, and if that separates them as well as the compass
does, the run says so and refuses its own figure.

**Measured: 3 minutes 44 seconds on 16 processor cores**, with the compass at 0.9872 (95% interval
0.9722 to 0.9972) against the length ruler's 0.6853 and a clean verdict read-out on every prompt.
The first run also spends longer than that fetching 3.8 GB of weights. The artefact, the exact
command and the environment it was measured in are in
[evidence/compass-2026-10-11](evidence/compass-2026-10-11/README.md).

**That is the no-graphics-card case, deliberately**, because it is the one worth promising: the
command above ends in `--device cpu` and a run that needs no card is a run anyone can repeat. A
card is faster and **we have published no measured figure for this command on one**, so there is
nothing here to compare it against. Do not read it against the 108 minutes quoted for an
abliteration run either. That figure is from a 6 GB card, it is a different command doing a much
larger job, and the two numbers share neither the work nor the hardware.

> **This example has now been wrong three times, and the third time is the instructive one.** It
> used `HuggingFaceTB/SmolLM2-135M-Instruct` until 2026-09-26: at 135M the model does not put a
> verdict token where the compass reads one, so the run said `MARGIN_READOUT_SUSPECT` and refused
> its own AUC. It was changed to `Qwen3-0.6B --n 12`, which traded one refusal for another and
> exits 1 with `MARGIN_NULLS_DOMINATE`.
>
> On 2026-10-11 it was changed again, to `Qwen3-1.7B --n 60` with `--skip-harmful 0
> --skip-harmless 0`, and a figure was quoted for it. **That was the worst of the three.** Explicit
> zero skips mean the harmful arm scores the 132 search rows and the harmless arm scores the 257
> rows the directions were fitted on, so the run exits 0 and reports a number this README's own
> opening promises it will not: one taken on rows something was fitted on. The previous version at
> least had the decency to refuse. A journey persona found it the same day by reading the skip
> flags' own help text against the arithmetic of the track.
>
> The figures that were here could not be traced to anything committed, and the weights they were
> taken on were not on the machine afterwards, so nobody could reproduce or even locate them. The
> ones above replace them and come from a committed artefact. The configuration was run three
> times, at three different commits, and all three gave the same AUC and the same control figure.
> `evidence/` is where a figure has to live before this file repeats it, and a test now enforces
> that.

This used to read `examples/toy-track/...`, which is in the repository and in **no install**, so the
first command under a heading saying "on your own machine" could not be run by anyone who had
installed the tool rather than cloned it. Two readers hit it on 2026-09-26 and neither could get
past it without guessing. From a clone, `examples/toy-track/` still works and is smaller.

**This writes two files, and one of them holds harmful prompts in the clear.** Beside
`compass-toy.json` you get `compass-toy.margins.jsonl`, the per-prompt rows the AUC was computed
from, and twelve of those rows are the harmful prompts themselves, unwrapped, in whatever directory
you ran the command in. They are there so the number can be checked rather than believed, which is
the whole argument of this project, and they are the same prompts the wheel deliberately keeps
obfuscated. **Do not `git add` that file.** Add `--no-margins` to the command above if you would
rather not have it at all.

Said here because this is the first command most people run, and until 2026-09-26 it was said only
on two pages this section does not link, so the reader most likely to be surprised was the one least
likely to have been told.

**Do not quote what it says.** Twelve rows measures nothing, and the tool makes you pass those
three flags rather than pretending otherwise. The run also prints its own null controls, and on
twelve rows one of those will often beat the instrument; if it does, the number is not a weak
measurement, it is not one at all.

## The real thing

```sh
# A corpus with a split that stops you marking your own homework.
senbonzakura track --harmful harmful.txt --harmless harmless.txt --out mytrack

# Search for a configuration and apply it. Timed at 108 minutes end to end on a 6 GB card, at
# the defaults, with the weights already cached, for Qwen3-0.6B: a different model of about the
# same size. The model below has not been timed at the default budget.
senbonzakura kageyoshi --model Qwen/Qwen2.5-0.5B-Instruct --track mytrack --out abliterated --device cuda

# Ask what the edit cost.
senbonzakura compass --model abliterated \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds --out compass.json
```

`harmful.txt` and `harmless.txt` are yours to supply, deliberately.
[The track page](https://elementmerc.github.io/senbonzakura/guide/the-track) covers building an
equivalent from public sources. The split matters more than the prompts.

## What it measures

Each is a command, and each measurement carries an interval rather than a bare number:

| | The question it answers |
|---|---|
| `score` | Are the refusals actually gone, on rows the search never saw? |
| `compass` | Does the model still **recognise** harm, as opposed to still refusing it? |
| `capability` | What did the edit cost on tasks the model either gets right or does not? Refusal rates and divergence cannot see reasoning loss. Runs by default. |
| `drift` | How far did the **first-token** distribution move, on a ruler that can be pointed at a model edited by any tool? |
| `validate` | Does a direction set carry refusal, or carry topic? |
| `check` | Reads result files from **other** tools and reports how their numbers could be wrong. No GPU, no model, no network. |
| `report` | Assembles the above into the card that should travel beside the weights. |

Two exceptions to the interval, named here rather than discovered later. `score` stamps a Wilson
interval and the raw counts into its result file, and its finishing line on the terminal still
prints a bare percentage. `coherence` prints no interval at all, because one forward pass over one
fixed passage returns the same number every time, so there's no run-to-run spread for an interval
to describe. `check` and `report` read other numbers rather than producing one.

**What `drift` measures, precisely.** One position per prompt: the next-token distribution at the
end of the rendered prompt, before the model has written anything. That's cheap, it's the same
quantity on both sides of a comparison, and it's blind to a shift that only appears later in a
long answer. A reasoning model that still opens identically and then reasons differently will look
untouched on this ruler. Treat `drift` as a named estimator rather than as the whole distance
between two models, and read [the benchmark page](https://elementmerc.github.io/senbonzakura/guide/benchmark)
for what it does and doesn't license you to say.

Two habits run through all of them. **The split is three-way**, so the rows a configuration is
selected on are never the rows it is reported on. And **every figure arrives with a control**,
usually a ruler reading nothing but prompt length: if that separates the arms as well as the real
instrument does, the real instrument is measuring sentence length.

**The honest ceiling.** No measurement above 3B parameters still stands: Qwen3-4B was run in
July 2026 and withdrawn, and nothing has been re-run above 3B since. The instruments also weaken
as the model does. Qwen3-1.7B's compass scores 0.9887 against a length-only ruler's 0.6564.
Qwen3-0.6B scores **0.6616 against that same 0.6564**, which is not a measurement of anything.

- [What is and is not established](https://elementmerc.github.io/senbonzakura/guide/what-we-know)
  — the full account, including the results that went against us.
- [REPRODUCING.md](REPRODUCING.md) — every figure above, mapped to the file it came from and the
  command that makes it. No GPU or corpus needed to check them.
- [METHOD.md](METHOD.md) — how to measure a behavioural property of a model defensibly. Not about
  this tool.
- [probes/](probes/) — add a behaviour of your own for the tool to measure.

## Documentation

**[elementmerc.github.io/senbonzakura](https://elementmerc.github.io/senbonzakura)**

## What this repository does not contain

By design, this is methods and results, not a loaded weapon:

- **No model weights in this git tree.** Uncensored checkpoints are published separately, on the
  Hub, and they are public and ungated: each one under its base model's own licence, which travels
  with the weights and is not ours to loosen. **The evaluation numbers on most of those cards are
  withdrawn**, for the reasons in the changelog, so read the card before trusting a figure on it.
  (This bullet twice said the opposite of the truth; see
  [what we got wrong](https://elementmerc.github.io/senbonzakura/guide/what-we-got-wrong).)
- **No harmful prompt sets in this git tree, and this is the bullet that needs the most care.**
  The evaluation track is published separately as a
  **[gated dataset](https://huggingface.co/datasets/ops-malware/senbonzakura-dataset)** under
  CC BY-NC 4.0. It holds prompts only: no completions, no answers.

  **A released wheel is a different matter.** It carries roughly 6,200 harmful prompts, wrapped so
  a scraper does not find them in plaintext. The wrapping is a speed bump, not protection: the key
  ships beside them and `bundled.py` says so. **To find out whether they are on your disk, run
  `senbonzakura doctor`**, which lists every bundled corpus it can decode, with its licence.

  This paragraph used to end "neither install you can run today carries them", and explained that
  0.3.0 on PyPI predated the bundled track and that a `git+...` build generates the blobs rather
  than committing them. That was true when it was written and stopped being true at this release:
  the wheel carries both blobs, so a plain `pip install` now puts those prompts on your disk. A
  sentence telling a reader which installs hold harmful content is a sentence that goes stale
  silently, and the wrong direction for it to go stale is reassuring. Asking the tool cannot go
  stale. [The dataset card](docs/evaluation-track-card.md) has the channel-by-channel table.

  Where that number comes from, since a figure nobody can derive is a figure nobody can check:
  4,895 in the packed evaluation track (259 fitting rows plus 4,636 evaluation rows) and 1,333
  across the five harmful research corpora (advbench 520, harmbench 200, harmbench-copyright 100,
  strongreject 313, xstest-unsafe 200). A sixth corpus, xstest-safe, holds 250 BENIGN prompts
  used as controls and is not counted here. This said "roughly 6,500" until 2026-09-25, which was
  only reachable by counting those 250 benign controls as harmful.
- **No harmful outputs.**

Uncensoring removes safety guardrails wholesale. That is both the point and the danger. Use it
accordingly.

**Disclaimer.** This software is provided without warranty of any kind, on its correctness, its
fitness for any purpose, or the accuracy of any number it produces. Responsibility for what is
done with it, and with any model modified or measured using it, sits with whoever does it. A model
with its refusals removed will answer things a deployed model should not; putting one in front of
other people is a decision with consequences that belong to whoever makes it.

Note on licences, and there are three separate ones in play:

- **The code** is **AGPL-3.0-or-later** (it embeds a keyword metric copied from Heretic, which is
  AGPL). **If you run a modified version of this code as a network service, the AGPL's section 13
  obliges you to offer your modified source to the people using it over that network.** That is
  the clause that distinguishes the AGPL from the GPL, and it is the one that bites an
  uncensoring-as-a-service business rather than an internal user: running it privately, however
  commercially, triggers nothing. Two readers with no knowledge of this project went looking for
  this in 2026-09 and found it stated nowhere but the licence file itself, which is the wrong place
  for the one licence fact a commercial reader most needs.
- **The bundled evaluation track** under `senbonzakura/data/` is a separate work aggregated into
  the same wheel, and it is **CC BY-NC 4.0: non-commercial**. The package metadata carries one
  licence expression and that expression describes the code, so if you are using this
  commercially, supply your own corpus with `--track` rather than using `--track default`. Full
  attribution is in `THIRD-PARTY-NOTICES.md`, which is installed beside the package.
- **A model you uncensor** keeps the **base model's** licence and use restrictions:
  redistributing an uncensored checkpoint is governed by that upstream licence (Qwen, Llama,
  Gemma and so on), not by this repository's.

## Credit

- Arditi, Obeso, et al. [*Refusal in Language Models Is Mediated by a Single
  Direction*](https://arxiv.org/abs/2406.11717) (2024). The direction method this builds on.
- [Heretic](https://github.com/p-e-w/heretic) by p-e-w (Philipp Emanuel Weidmann),
  AGPL-3.0. The automated, KL-guarded search this builds on, and the keyword metric
  reported here for comparison (its marker list copied verbatim and its normalisation
  adapted, which is why this project is AGPL; see
  [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)).
- Maxime Labonne. [*Uncensor any LLM with abliteration*](https://huggingface.co/blog/mlabonne/abliteration).
  The tutorial that popularised the technique.

## Getting help

Ask in [Discussions](https://github.com/elementmerc/senbonzakura/discussions), report a defect in
[Issues](https://github.com/elementmerc/senbonzakura/issues), and report a security problem by
email rather than in public: [SECURITY.md](SECURITY.md) has the address and what is in scope.
[SUPPORT.md](SUPPORT.md) says what makes a question easy to answer, and what not to paste into a
public thread.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests need a test, a commit message that says
why, and one line added to [CONTRIBUTORS.md](CONTRIBUTORS.md) agreeing to the
[CLA](CLA.md). The project is AGPL and stays AGPL; you keep the copyright in what you write.
Everyone taking part is held to the [code of conduct](CODE_OF_CONDUCT.md).

## Licence

Copyright (C) 2026 Daniel Iwugo.

**AGPL-3.0-or-later.** See [LICENSE](LICENSE).

**Senbonzakura is a modified work based in part on Heretic, and it is not Heretic.** Modified by
Daniel Iwugo; first included 2026-07-14, most recently modified 2026-10-07. Only the keyword rate
is shared code: its marker list is kept byte-identical and its normalisation is adapted from
upstream, so only that number is a like-for-like comparison with Heretic; everything else here is
measured by our own instrument. The full statement, and what
was and was not changed, is in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
