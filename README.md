<div align="center">

<img src="icon.svg" alt="Senbonzakura logo" width="96" height="96">

# Senbonzakura

**Multi-direction refusal abliteration, and the instruments to tell whether it worked**

<p>
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/senbonzakura/"><img src="https://img.shields.io/pypi/v/senbonzakura?labelColor=2b3038&color=3d4d6b" alt="PyPI" /></a>
  <a href="https://pypi.org/project/senbonzakura/"><img src="https://img.shields.io/badge/python-3.10%20to%203.14-3d4d6b?labelColor=2b3038" alt="Python 3.10 to 3.14" /></a>
  <a href="https://github.com/elementmerc/senbonzakura"><img src="https://img.shields.io/badge/tested%20on-Linux%20%7C%20macOS%20%7C%20Windows-3d4d6b?labelColor=2b3038" alt="Tested on Linux, macOS and Windows" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-AGPL--3.0--or--later-c77b5a?labelColor=2b3038" alt="AGPL-3.0-or-later" /></a>
</p>
<p>
  <a href="https://elementmerc.github.io/senbonzakura/guide/compass"><img src="https://img.shields.io/badge/every%20figure-has%20an%20interval-3d4d6b?labelColor=2b3038" alt="Every published figure carries a confidence interval" /></a>
  <a href="https://elementmerc.github.io/senbonzakura/guide/compass"><img src="https://img.shields.io/badge/comparison-KL%20at%20matched%20refusal-c77b5a?labelColor=2b3038" alt="Compared at matched refusal removal" /></a>
  <a href="https://elementmerc.github.io/senbonzakura/guide/the-track"><img src="https://img.shields.io/badge/evaluation-held%20out%2C%20three--way%20split-3d4d6b?labelColor=2b3038" alt="Held-out evaluation, three-way split" /></a>
  <a href="https://elementmerc.github.io/senbonzakura/guide/contamination"><img src="https://img.shields.io/badge/benchmarks-contamination%20checked-3d4d6b?labelColor=2b3038" alt="Contamination checked against public benchmarks" /></a>
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/tests-1090-3d4d6b?labelColor=2b3038" alt="1090 tests" /></a>
  <a href="https://github.com/elementmerc/senbonzakura/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/coverage-95.4%25-3d4d6b?labelColor=2b3038" alt="95.4% branch coverage, gated in CI" /></a>
</p>

<a href="https://elementmerc.github.io/senbonzakura"><strong>Documentation</strong></a>
&nbsp;·&nbsp;
<a href="CHANGELOG.md">Changelog</a>
&nbsp;·&nbsp;
<a href="https://elementmerc.github.io/senbonzakura/guide/what-we-know">What is and is not established</a>

<img src="https://raw.githubusercontent.com/elementmerc/senbonzakura/main/docs/senbonzakura-kageyoshi.png" width="760" alt="Senbonzakura Kageyoshi: a thousand blades in formation">

</div>

---

Senbonzakura removes the refusal behaviour from an open-weight language model by
finding the *directions* in its activation space that carry "I can't help with
that" and orthogonalising them out of the weights. It builds on the
single-direction method of Arditi et al. and the automated search of Heretic, and
can cut in **several directions at once** rather than one.

Named for Byakuya Kuchiki's zanpakutō, the sword that scatters into a thousand
blades.

## Why multi-direction, and what happened when we tested it

Refusal in a language model is *mostly* one direction in its activation space
([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)). Mostly. The idea this project was
built on is that the last stubborn few percent lives in a small handful of nearby directions the
single-arrow method never sees, so it searches for the whole subspace and orthogonalises all of
it out of the weights.

> **Read this before quoting anything. On the one model where we can measure it properly, that
> idea does not hold.** One direction against two, five seeds each, everything else held still,
> every model scored afterwards by one instrument on prompts nothing was fitted or selected on:
> both budgets removed hard refusal, and two directions did about **twice the collateral damage**
> for no refusal benefit (drift 0.093 against 0.050). More directions cost more and bought
> nothing.
>
> That is one model, Qwen3-1.7B, and it isn't the last word for every architecture. It is enough
> to say the headline idea is **unsupported by this project's own evidence**, and we would rather
> you heard it here than found it out yourself.
>
> The feature also had never worked until 2026-08-03: the check deciding whether a candidate
> direction carried refusal could not accept any direction, on any model, at any setting, so every
> run before that applied exactly one direction whatever it was asked for.
>
> **What this tool is actually good at is the measurement**, and that part survived the same
> scrutiny: [the head-to-head](https://elementmerc.github.io/senbonzakura/guide/benchmark).
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

Nothing to hand? The compass runs on a toy track committed to this repository, with no corpus to
build and no model to edit first, in
[one command](https://elementmerc.github.io/senbonzakura/guide/compass). Its answer is a perfect
score you should not believe, and the controls printed beside it say why.

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
