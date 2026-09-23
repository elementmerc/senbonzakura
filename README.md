<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/elementmerc/senbonzakura/b02383ab124f8d38f2886323d39b8a933883c6ce/assets/brand/readme-banner-dark.png">
  <img src="https://raw.githubusercontent.com/elementmerc/senbonzakura/b02383ab124f8d38f2886323d39b8a933883c6ce/assets/brand/readme-banner.png" width="820"
       alt="Senbonzakura">
</picture>

**Precision abliteration, with receipts.**

<p>
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/elementmerc/senbonzakura/ci.yml?branch=dev&amp;logo=githubactions&amp;logoColor=white&amp;label=CI&amp;labelColor=24292f" alt="CI" /></a>
  <a href="https://pypi.org/project/senbonzakura/"><img src="https://img.shields.io/badge/python-3.10%20to%203.14-3776AB?logo=python&amp;logoColor=white&amp;labelColor=24292f" alt="Python 3.10 to 3.14" /></a>
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-4C566A?labelColor=24292f" alt="Platform: Linux, macOS and Windows" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-AGPL--3.0--or--later-A42E2B?logo=gnu&amp;logoColor=white&amp;labelColor=24292f" alt="Licence: AGPL-3.0-or-later" /></a>
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/tests-5128-0A9EDC?logo=pytest&amp;logoColor=white&amp;labelColor=24292f" alt="5128 tests" /></a>
  <a href="https://app.codecov.io/gh/elementmerc/senbonzakura/tree/dev"><img src="https://img.shields.io/codecov/c/github/elementmerc/senbonzakura/dev?logo=codecov&amp;logoColor=white&amp;label=coverage&amp;labelColor=24292f" alt="Coverage, measured in CI" /></a>
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

Removing it is the easy half. Any abliterator can stop a model saying "I can't help with that".
The hard part is knowing whether you also took out its reasoning, and a tool that cannot tell the
difference will report a lobotomy as a success. So every number here arrives with its conditions:
what it was measured on, on rows nothing was fitted or selected on, with an interval, and with a
control that would expose it if it were measuring the wrong thing.

Named for Byakuya Kuchiki's zanpakutō, the sword that scatters into a thousand blades.

> **What it makes.** A model that will answer requests the original refused, including harmful
> ones. That is the point, and it is permanent in the weights. The base model's licence still
> governs the result. Don't put one in front of other people without saying what it is. The
> package carries roughly 6,500 harmful prompts so it runs offline.
>
> [The detail](#what-this-repository-does-not-contain), and `ACCEPTABLE-USE.md` ships in the
> package.

## Try it without installing anything

[**Open the notebook in Colab**](https://colab.research.google.com/github/elementmerc/senbonzakura/blob/dev/notebooks/senbonzakura_colab.ipynb).
Free GPU, nothing on your machine, about fifteen minutes. It measures a model, edits it, then
measures what that cost.

Or, if you have Docker:

```sh
docker run --rm ghcr.io/elementmerc/senbonzakura:dev doctor
```

`doctor` reports what your install can and cannot do, which on a first run is more useful than it
sounds.

## Install

```sh
pip install senbonzakura
senbonzakura setup
```

> **PyPI serves 0.3.0, from July 2026, and you do not want it.** Its numbers are withdrawn, and
> `senbonzakura-check` is not on PyPI yet, so this version cannot resolve from the index.
> Until it is published, install from the repository, checker first:
>
> ```sh
> pip install "git+https://github.com/elementmerc/senbonzakura#subdirectory=checker"
> pip install "git+https://github.com/elementmerc/senbonzakura"
> ```
>
> **`--track default` will not work from that install.** The bundled corpora are generated
> artefacts kept out of git, because they are harmful prompts. Build them from a clone
> (`tools/packaging/build_corpora.py`, public sources, pinned commits) or bring your own
> corpus. [The track page](https://elementmerc.github.io/senbonzakura/guide/the-track) has both
> routes.

That brings torch, transformers, accelerate and optuna: **68 packages, 5.8 GB**, nineteen of them
CUDA wheels. There is nothing to choose.

`senbonzakura setup` exists because **pip picks by platform, not by hardware**. On Windows, PyPI's
torch is CPU-only, so **a card there sits idle** and nothing warns you. It says what it found and
prints the command that fixes it, changing nothing unless you add `--apply`.
[Install](https://elementmerc.github.io/senbonzakura/guide/install) has the full table.

Editing a model wants a CUDA card with 6 GB. The measuring commands run on CPU.

## One command, on your own machine

No corpus, no GPU, no model to edit first. The toy track is committed here:

```sh
senbonzakura compass \
    --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --harmful examples/toy-track/bad_eval_ds \
    --harmless examples/toy-track/good_ds \
    --skip-harmful 0 --skip-harmless 0 --n 12 \
    --out compass-toy.json --device cpu
```

**Do not quote what it says.** Twelve rows measures nothing, and the tool makes you pass those
three flags rather than pretending otherwise.

## The real thing

```sh
# A corpus with a split that stops you marking your own homework.
senbonzakura track --harmful harmful.txt --harmless harmless.txt --out mytrack

# Search for a configuration and apply it. About an hour for a 1.7B on a 6 GB card.
senbonzakura kageyoshi --model Qwen/Qwen3-1.7B --track mytrack --out abliterated --device cuda

# Ask what the edit cost.
senbonzakura compass --model abliterated \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds --out compass.json
```

`harmful.txt` and `harmless.txt` are yours to supply, deliberately.
[The track page](https://elementmerc.github.io/senbonzakura/guide/the-track) covers building an
equivalent from public sources. The split matters more than the prompts.

## What it measures

Each is a command, and each reports an interval rather than a bare number:

| | The question it answers |
|---|---|
| `score` | Are the refusals actually gone, on rows the search never saw? |
| `compass` | Does the model still **recognise** harm, as opposed to still refusing it? |
| `capability` | What did the edit cost on tasks the model either gets right or does not? Refusal rates and divergence cannot see reasoning loss. Runs by default. |
| `drift` | How far did the output distribution move, on a ruler that can be pointed at a model edited by any tool? |
| `validate` | Does a direction set carry refusal, or carry topic? |
| `check` | Reads result files from **other** tools and reports how their numbers could be wrong. No GPU, no model, no network. |
| `report` | Assembles the above into the card that should travel beside the weights. |

Two habits run through all of them. **The split is three-way**, so the rows a configuration is
selected on are never the rows it is reported on. And **every figure arrives with a control**,
usually a ruler reading nothing but prompt length: if that separates the arms as well as the real
instrument does, the real instrument is measuring sentence length.

**The honest ceiling.** Nothing has been measured above 3B parameters, and the instruments weaken
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

- **No model weights in this git tree.** Abliterated checkpoints from this work are published
  separately on HuggingFace, each under its base model's own licence, which travels with the
  weights and is not ours to loosen.
- **No harmful prompt sets in this git tree, and this is the bullet that needs the most care.**
  The evaluation track is published separately as a
  **[gated dataset](https://huggingface.co/datasets/ops-malware/senbonzakura-dataset)** under
  CC BY-NC 4.0. It holds prompts only: no completions, no answers.

  **The installed wheel is a different matter.** It carries roughly 6,500 harmful prompts,
  wrapped so a scraper does not find them in plaintext. The wrapping is a speed bump, not
  protection: the key ships beside them and `bundled.py` says so. An install is not prompt-free.
- **No harmful outputs.**

Abliteration removes safety guardrails wholesale. That is both the point and the danger. Use it
accordingly.

**Disclaimer.** This software is provided without warranty of any kind, on its correctness, its
fitness for any purpose, or the accuracy of any number it produces. Responsibility for what is
done with it, and with any model modified or measured using it, sits with whoever does it. A model
with its refusals removed will answer things a deployed model should not; putting one in front of
other people is a decision with consequences that belong to whoever makes it.

Note on licences, and there are three separate ones in play:

- **The code** is **AGPL-3.0-or-later** (it embeds a keyword metric copied from Heretic, which is
  AGPL).
- **The bundled evaluation track** under `senbonzakura/data/` is a separate work aggregated into
  the same wheel, and it is **CC BY-NC 4.0: non-commercial**. The package metadata carries one
  licence expression and that expression describes the code, so if you are using this
  commercially, supply your own corpus with `--track` rather than using `--track default`. Full
  attribution is in `THIRD-PARTY-NOTICES.md`, which is installed beside the package.
- **A model you abliterate** keeps the **base model's** licence and use restrictions:
  redistributing an abliterated checkpoint is governed by that upstream licence (Qwen, Llama,
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
Daniel Iwugo; first included 2026-07-14, most recently modified 2026-09-23. Only the keyword rate
is shared code: its marker list is kept byte-identical and its normalisation is adapted from
upstream, so only that number is a like-for-like comparison with Heretic; everything else here is
measured by our own instrument. The full statement, and what
was and was not changed, is in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
