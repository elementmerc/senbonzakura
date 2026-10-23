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
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/tests-11734-0A9EDC?logo=pytest&amp;logoColor=white&amp;labelColor=24292f" alt="11734 tests" /></a>
  <a href="https://app.codecov.io/gh/elementmerc/senbonzakura/tree/dev"><img src="https://img.shields.io/codecov/c/github/elementmerc/senbonzakura/dev?logo=codecov&amp;logoColor=white&amp;label=coverage%20dev&amp;labelColor=24292f" alt="Coverage on the development branch, measured in CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-AGPL--3.0--or--later-A42E2B?logo=gnu&amp;logoColor=white&amp;labelColor=24292f" alt="Licence: AGPL-3.0-or-later" /></a>
</p>

</div>

Senbonzakura is the open research tool behind the Senbonzakura app, which you can download from [senbon.dev](https://senbon.dev). This repository holds the free command line tool, which removes refusal behaviour from an open-weight language model and measures what that removal cost.

> **What it makes.** A model that will answer requests the original refused, including harmful ones. That is the point, and it is permanent in the weights. The base model's licence still governs the result, so don't put one in front of other people without saying what it is. The package carries roughly 6,200 harmful prompts so it runs offline: 4,895 in the packed evaluation track and 1,333 across five harmful research corpora. The [guide](https://elementmerc.github.io/senbonzakura/guide/what-it-is) explains the rest.

## Install

<!-- documented-first-run: install -->
```sh
pip install senbonzakura
senbonzakura setup
```

Needs Python 3.10 to 3.14. The install is large, because it brings torch with it.

On Windows, PyPI's torch is CPU-only, so a graphics card there sits idle and nothing warns you. `senbonzakura setup` says what it found and prints the command that fixes it. It changes nothing unless you add `--apply`. The [install guide](https://elementmerc.github.io/senbonzakura/guide/install) covers the other platforms and the virtual environment route.

## Try it

One command, no graphics card, no corpus to fetch. It reads the evaluation track bundled in the install:

```sh
senbonzakura compass \
    --model Qwen/Qwen3-1.7B \
    --track default \
    --harmful default/bad_eval_ds \
    --harmless default/good_ds \
    --n 60 \
    --out compass-toy.json --device cpu
```

On the bundled track, the compass scores Qwen3-1.7B at 0.9887 against a length-only ruler's 0.6564. Qwen3-0.6B scores 0.6616 against that same 0.6564, which is not a measurement of anything. The artefact is in [evidence/compass-2026-10-11](evidence/compass-2026-10-11/README.md).

`--track default` is the flag that matters. It keeps the run off the rows the edit was fitted on. The run also writes `compass-toy.margins.jsonl`, which holds harmful prompts in plain text, so don't `git add` it. Add `--no-margins` to skip it. The [compass page](https://elementmerc.github.io/senbonzakura/guide/compass) explains the result.

From a clone of the repository, the same flags work with the smaller track in `examples/toy-track/`: `--harmful examples/toy-track/bad_eval_ds --harmless examples/toy-track/good_ds`.

To use your own prompts, supply two files and build a track: `senbonzakura track --harmful harmful.txt --harmless harmless.txt --out mytrack`. The split matters more than the prompts. The [track page](https://elementmerc.github.io/senbonzakura/guide/the-track) explains it.

Prefer a notebook? [Open it in Colab](https://colab.research.google.com/github/elementmerc/senbonzakura/blob/v0.5.1/notebooks/senbonzakura_colab.ipynb).

## Documentation

- [The docs site](https://elementmerc.github.io/senbonzakura): the guide and the command reference.
- [What is and is not established](https://elementmerc.github.io/senbonzakura/guide/what-we-know): the claims this project stands behind, and the ones it withdrew.
- [REPRODUCING.md](REPRODUCING.md): how to check every published figure.
- [METHOD.md](METHOD.md): how to measure a behaviour in a model defensibly.
- [CHANGELOG.md](CHANGELOG.md): what changed in each release.

## What this repository does not contain

There are no model weights and no harmful outputs in this repository. The harmful evaluation track is published separately as a [gated dataset](https://huggingface.co/datasets/ops-malware/senbonzakura-dataset) under CC BY-NC 4.0. Run `senbonzakura doctor` to see which bundled corpora your install can decode, with their licences. The [dataset card](docs/evaluation-track-card.md) has the full breakdown.

Uncensoring removes safety guardrails wholesale. That is both the point and the danger, so use it accordingly. The software comes without warranty. Whoever runs it, or puts a modified model in front of anyone, owns the consequences.

## Getting help

Ask in [Discussions](https://github.com/elementmerc/senbonzakura/discussions) and report defects in [Issues](https://github.com/elementmerc/senbonzakura/issues). Report security problems by email, not in public: [SECURITY.md](SECURITY.md) has the address. [SUPPORT.md](SUPPORT.md) says what makes a question easy to answer.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Add one line to [CONTRIBUTORS.md](CONTRIBUTORS.md) agreeing to the [CLA](CLA.md). Everyone taking part is held to the [code of conduct](CODE_OF_CONDUCT.md).

## Cite

Use the [CITATION.cff](CITATION.cff) file, or GitHub's "Cite this repository" button.

## Credit

- Arditi, Obeso, et al. [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717) (2024). The direction method this builds on.
- [Heretic](https://github.com/p-e-w/heretic) by p-e-w (Philipp Emanuel Weidmann), AGPL-3.0. The automated, KL-guarded search this builds on.
- Maxime Labonne. [*Uncensor any LLM with abliteration*](https://huggingface.co/blog/mlabonne/abliteration). The tutorial that popularised the technique.

## Licence

Copyright (C) 2026 Daniel Iwugo.

**AGPL-3.0-or-later.** See [LICENSE](LICENSE). If you run a modified version of this code as a network service, section 13 of the AGPL requires you to offer your modified source to the people using it. The bundled evaluation track is CC BY-NC 4.0 (non-commercial), so for commercial use supply your own corpus with `--track`. A model you uncensor keeps its base model's licence.

