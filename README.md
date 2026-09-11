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
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/tests-3626-0A9EDC?logo=pytest&amp;logoColor=white&amp;labelColor=24292f" alt="3626 tests" /></a>
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

The removal is the easy half. Any abliterator can stop a model saying "I can't help with that";
the hard part is knowing whether you also took out its judgement, its reasoning, or its grip on
the language, and a tool that cannot tell those apart will report a lobotomy as a success. So
every number this prints arrives with the conditions attached: what it was measured on, on rows
nothing was fitted or selected on, with an interval, and with the controls that would expose it if
it were measuring the wrong thing.

Named for Byakuya Kuchiki's zanpakutō, the sword that scatters into a thousand blades.

## Install

```sh
pip install senbonzakura
senbonzakura setup
```

The first command brings everything needed to edit a model: torch, transformers, accelerate,
optuna. That is 68 packages and a few gigabytes, and there is nothing else to choose.

The second exists because **pip picks by platform, not by hardware**. There is no way for a
package to declare "install the CUDA build if there is a card", so what you get depends on your
operating system rather than on what is in the machine:

| Your machine | What pip alone gives you |
|---|---|
| Linux with a GPU | the CUDA build. Correct |
| Linux with no GPU | the CUDA build anyway, and about 15 CUDA packages you cannot use |
| macOS on Apple silicon | a build that uses Metal. Correct |
| Windows with a GPU | **a CPU-only build. Your card will sit idle** |

That last row is the one that costs you a day. PyPI's Windows torch is 124 MB and CPU-only; the
CUDA build is not on PyPI at all. So a gaming laptop with a 3060 in it installs a torch that
cannot see the card, nothing warns you, and the search runs on CPU.

`senbonzakura setup` looks at the machine, says what it found, and prints the one command that
fixes it. It changes nothing unless you add `--apply`.

Editing a model wants a CUDA card with 6 GB on it. The measuring commands run on CPU.

## One worked example

```sh
# Build a corpus with a split that stops you marking your own homework.
senbonzakura track --harmful harmful.txt --harmless harmless.txt --out mytrack

# Search for a configuration and apply it. About an hour for a 1.7B on a 6 GB card.
senbonzakura kageyoshi --model Qwen/Qwen3-1.7B --track mytrack --out abliterated --device cuda

# Ask what the edit cost.
senbonzakura compass --model abliterated \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds --out compass.json
```

Nothing to hand? The instruments run on a toy track committed to this repository, with no corpus
to build and no model to edit first, in
[one command](https://elementmerc.github.io/senbonzakura/guide/compass).

## What it measures

Each of these is a command, and each reports an interval rather than a bare number:

| | The question it answers |
|---|---|
| `score` | Are the refusals actually gone, on rows the search never saw? |
| `compass` | Does the model still **recognise** harm, as opposed to still refusing it? |
| `capability` | What did the edit cost on tasks the model either gets right or does not? Refusal rates and KL cannot see reasoning loss. |
| `drift` | How far did the output distribution move from the original, on one ruler that can be pointed at a model edited by any tool? |
| `validate` | Does a direction set carry refusal, or carry topic? |
| `report` | Assembles the above into the card that should travel beside the weights. |

Two habits run through all of them. **The evaluation split is three-way**, so the rows a
configuration is selected on are never the rows it is reported on. And **every figure arrives with
the control that would expose it**: the most useful is a ruler reading nothing but prompt length,
because if that separates the two arms as well as the real instrument does, the real instrument is
measuring sentence length.

**The honest ceiling.** Nothing has been measured above 3B parameters, and most of the set is
under 2B. The instruments also get weaker as the model does: on the two measurements committed
here, Qwen3-1.7B's compass scores 0.9887 against a length-only ruler's 0.6564, and Qwen3-0.6B
scores **0.6616 against that same 0.6564**, a gap of five thousandths, which is not a measurement
of anything.

[Read this before quoting any number](https://elementmerc.github.io/senbonzakura/guide/what-we-know)
is the full account, including the results that did not go our way.

## Documentation

**[elementmerc.github.io/senbonzakura](https://elementmerc.github.io/senbonzakura)**

The guide covers the method, the corpus and its split, the instruments, benchmarking against
another tool, and every condition attached to every number this project has published.

## What this repository does not contain

By design, this is methods and results, not a loaded weapon:

- **No model weights in this git tree.** Abliterated checkpoints from this work are published
  separately on HuggingFace, each under its base model's own licence, which travels with the
  weights and is not ours to loosen.
- **No harmful prompt sets in this git tree, and this is the bullet that needs the most care.**
  The evaluation track is published separately as a
  **[gated dataset](https://huggingface.co/datasets/ops-malware/senbonzakura-dataset)** under
  CC BY-NC 4.0. It holds prompts only: no completions, no answers.

  **The wheel you install from PyPI is a different matter and you should know it.** It carries
  the evaluation track and six public research corpora, roughly 6,500 harmful prompts, wrapped so
  a scraper does not find them in plaintext. That wrapping is a speed bump and not protection: the
  key ships beside them, and anyone who reads `src/senbonzakura/bundled.py` can recover them in
  five lines. The gate on the HuggingFace copy does not apply to them. Saying only the first half
  of this would leave the impression that a `pip install` is prompt-free, and it is not.
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
  reported here for comparison (copied verbatim, which is why this project is
  AGPL; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)).
- Maxime Labonne. [*Uncensor any LLM with abliteration*](https://huggingface.co/blog/mlabonne/abliteration).
  The tutorial that popularised the technique.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests need a test, a commit message that says
why, and one line added to [CONTRIBUTORS.md](CONTRIBUTORS.md) agreeing to the
[CLA](CLA.md). The project is AGPL and stays AGPL; you keep the copyright in what you write.

## Licence

**AGPL-3.0-or-later.** See [LICENSE](LICENSE).

**Senbonzakura is a modified work based in part on Heretic, and it is not Heretic.** Modified by
Daniel Iwugo; first included 2026-07-14, most recently modified 2026-09-08. Only the keyword rate
is shared code and it is kept byte-identical, so only that number is a like-for-like comparison
with Heretic; everything else here is measured by our own instrument. The full statement, and what
was and was not changed, is in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
