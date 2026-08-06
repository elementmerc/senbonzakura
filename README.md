# Senbonzakura

**Multi-direction refusal abliteration for transformer language models.**

<p align="center">
  <img src="https://raw.githubusercontent.com/elementmerc/senbonzakura/main/docs/senbonzakura-kageyoshi.png" width="760" alt="Senbonzakura Kageyoshi: a thousand blades in formation">
</p>

<p align="center">
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/elementmerc/senbonzakura/ci.yml?branch=main&style=flat-square&label=CI&labelColor=2b3038&color=3d4d6b"></a>
  <a href="https://pypi.org/project/senbonzakura/"><img alt="PyPI" src="https://img.shields.io/pypi/v/senbonzakura?style=flat-square&labelColor=2b3038&color=3d4d6b"></a>
  <a href="https://pypi.org/project/senbonzakura/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-3d4d6b?style=flat-square&labelColor=2b3038"></a>
  <a href="LICENSE"><img alt="Licence" src="https://img.shields.io/badge/licence-AGPL--3.0--or--later-c77b5a?style=flat-square&labelColor=2b3038"></a>
  <a href="https://elementmerc.github.io/senbonzakura"><img alt="Documentation" src="https://img.shields.io/badge/docs-elementmerc.github.io-3d4d6b?style=flat-square&labelColor=2b3038"></a>
</p>

Senbonzakura removes the refusal behaviour from an open-weight language model by
finding the *directions* in its activation space that carry "I can't help with
that" and orthogonalising them out of the weights. It builds on the
single-direction method of Arditi et al. and the automated search of Heretic, and
adds the one thing that moved the needle in my own runs: cutting in **several
directions at once**, not just one.

Named for Byakuya Kuchiki's zanpakutō, the sword that scatters into a thousand
blades. Refusal is not one blade. It's many.

## Why multi-direction

## Why multi-direction

Refusal in a language model is *mostly* one direction in its activation space
([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)). Mostly. The last stubborn few percent
lives in a small handful of nearby directions the single-arrow method never sees. Senbonzakura
searches for the whole subspace and orthogonalises all of it out of the weights.

> **Read this before quoting anything.** The multi-direction feature had never worked until
> 2026-08-03: the check deciding whether a candidate direction carried refusal could not accept
> any direction, on any model, at any setting. Every run before that applied exactly one
> direction. It has been rewritten and now finds up to eight per layer, but **finding more
> directions is not the same as showing they carry refusal**, and that has not been shown. The
> headline claim is currently unsupported by this project's own evidence.
>
> The full account is in the documentation:
> [what is and is not established](https://elementmerc.github.io/senbonzakura/guide/what-we-know).

The other half of the project is the measurement. Removing a model's refusals is easy; knowing
whether you also removed its ability to *recognise* harm is not, and that is what the compass is
for.

## Install

```sh
pip install senbonzakura
senbonzakura --help
```

## One worked example

```sh
# Build a corpus with a split that stops you marking your own homework.
senbonzakura track --harmful harmful.txt --harmless harmless.txt --out mytrack

# Search for a configuration and apply it. About an hour for a 1.7B on a 6 GB card.
senbonzakura kageyoshi --model Qwen/Qwen3-1.7B --track mytrack --out abliterated --device cuda

# Ask whether the edited model still knows which requests are dangerous.
senbonzakura compass --model abliterated \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds --out compass.json
```

## Documentation

**[elementmerc.github.io/senbonzakura](https://elementmerc.github.io/senbonzakura)**

The guide covers the method, the corpus and its split, the compass, benchmarking against another
tool, and every condition attached to every number this project has published.

## What this repository does not contain

By design, this is methods and results, not a loaded weapon:

- **No pre-abliterated model weights.** Run the tool yourself.
- **No harmful prompt sets.** The contrast and evaluation data you supply are your
  own; none ship here.
- **No harmful outputs.**

Abliteration removes safety guardrails wholesale. That is both the point and the
danger. Use it accordingly.

Note on licences: this tool is **AGPL-3.0-or-later** (it embeds a keyword metric copied from
Heretic, which is AGPL). Separately, a model
you abliterate keeps the **base model's** licence and use restrictions: redistributing an
abliterated checkpoint is governed by that upstream licence (Qwen, Llama, Gemma and so on), not by
this repository's.

## Credit

- Arditi, Obeso, et al. [*Refusal in Language Models Is Mediated by a Single
  Direction*](https://arxiv.org/abs/2406.11717) (2024). The direction method this builds on.
- [Heretic](https://github.com/p-e-w/heretic) by p-e-w (Philipp Emanuel Weidmann),
  AGPL-3.0. The automated, KL-guarded search this refines, and the keyword metric
  reported here for comparison (copied verbatim, which is why this project is
  AGPL; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)).
- Maxime Labonne. [*Uncensor any LLM with abliteration*](https://huggingface.co/blog/mlabonne/abliteration). The tutorial that
  popularised the technique.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests need a test, a commit message that says
why, and one line added to [CONTRIBUTORS.md](CONTRIBUTORS.md) agreeing to the
[CLA](CLA.md). The project is AGPL and stays AGPL; you keep the copyright in what you write.

## Licence

**AGPL-3.0-or-later.** See [LICENSE](LICENSE). Senbonzakura is copyleft because it embeds a
keyword metric copied verbatim from [Heretic](https://github.com/p-e-w/heretic) (AGPL-3.0); see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

**Senbonzakura is a modified work based in part on Heretic, and it is not Heretic.** Modified by
Daniel Iwugo; first included 2026-07-14, most recently modified 2026-07-29. Only the keyword rate
is shared code and it is kept byte-identical, so only that number is a like-for-like comparison
with Heretic; everything else here is measured by our own instrument. The full statement, and what
was and was not changed, is in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
