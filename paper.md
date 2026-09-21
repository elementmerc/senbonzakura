---
title: "Senbonzakura: measured refusal abliteration for transformer language models"
tags:
  - Python
  - machine learning
  - large language models
  - mechanistic interpretability
  - model evaluation
  - AI safety
authors:
  - name: Daniel Iwugo
    affiliation: 1
affiliations:
  - name: Independent researcher, United Kingdom
    index: 1
date: 21 September 2026
bibliography: paper.bib
---

# Summary

Abliteration means removing a language model's refusal behaviour by editing its weights rather
than by talking it into cooperating. It rests on a result from @arditi2024: refusal is carried by
a direction in activation space, and projecting that direction out of the weights that write to
the residual stream removes the behaviour for good. Several tools do this, and do it well.

Senbonzakura does it too, and so do the others: Heretic runs eleven benchmarks from the Language
Model Evaluation Harness, abliterix reports an original-against-edited table. Measuring the cost
is not the gap. The gap is that nothing validates the instrument doing the measuring.

Taking refusal out is the easy part. The hard part is knowing whether you also took out the
model's judgement, its reasoning, or its grip on the language, because a tool that only reports
the refusal rate cannot tell those apart. A lobotomy and a success look the same on that chart.

So the package ships the instruments next to the edit. It scores refusal with two rulers, its own
and a competitor's, measures capability on tasks graded by code rather than by another model,
measures coherence as the perplexity of a fixed passage, measures the distributional cost as KL
divergence on harmless prompts, and measures whether the model still *recognises* harm even though
it will now discuss it. Each figure carries the conditions it was taken under: which rows it was
scored on, the instrument that produced it, and a null control beside it. Intervals are on the
compass and the capability score; the refusal rate does not have one yet, and coherence cannot
have one, because it reads a single passage and one observation admits no interval.

That last one matters more than it sounds. A control is the whole reason a number means anything:
this project's harm-recognition compass was once matched, on the same exam, by a control that
reads nothing but sentence length, and nothing else would have caught it.

# Statement of need

Work on abliteration reports that it works, and the better tools report what else changed. What
none of them report is whether the thing doing the reporting can be trusted, and the field has no
shared way of asking. Three things make that harder than it looks, and I ran
into all three.

**A refusal rate is a property of the corpus, the prompt format and the token budget, not just of
the model.** Score the same weights on different prompts and the number moves; score them with a
budget shorter than the model's refusals and the rate reads low, because a refusal cut off before
it is emitted counts as compliance. Both have happened to figures this project published, and both
were caught after the numbers had travelled.

**Reporting a configuration on the rows you picked it on is not a measurement.** A search over a
few hundred trials finds whatever does best on what you showed it, so that score is the best of N
draws. Senbonzakura splits its corpus into fitting rows, selection rows, and rows nothing has
seen, and refuses to cross the boundary.

**Two numbers are comparable only if they were made the same way.** If the scorer changed between
runs, you have two instruments, not two measurements. I have withdrawn figures for exactly that,
so each one now carries its identity: metric, estimator, units, the input it was taken on, the
precision it was computed at, and the version of the code that took it.

A companion package, `senbonzakura-check`, reads those records back and checks them. It also reads
other harnesses' result files [@gao2024; @inspect2024], so it can be pointed at somebody else's
numbers rather than only at mine, and it installs without PyTorch.

The ceiling is low enough to state here rather than leave to the documentation.

**Nothing has been measured above 3B parameters**, and the instruments weaken as the model
shrinks: Qwen3-1.7B's harm-recognition compass scores 0.9887 against a length-only control at
0.6564, while Qwen3-0.6B scores 0.6616 against that same control, which is not a measurement of
anything. **Every Gemma figure is withdrawn**, the edit having been found not to reach the
residual stream there. The documentation carries a page saying which claims the repository's
evidence supports and which it does not. I would rather ship something that argues with its own
author than something that does not.

# State of the field

`Heretic` [@heretic2025] is the direct influence on the search here. It automates the choice of
ablation parameters under a KL constraint, which is a real advance on hand-tuning. Senbonzakura
uses its keyword metric verbatim and reproduces its search parameterisation, which is why this
package is AGPL-3.0-or-later, and runs that metric as one arm of its own comparisons so both tools
can be read on one ruler. `abliterix` [@abliterix2025] is the closest direct competitor and is
ahead of this project on method breadth, on mixture-of-experts handling and on prebuilt
configurations.

The multi-direction idea is neither mine nor new: Wollschläger et al. [@wollschlager2025] show
refusal is a cone and report attack success rising with the number of ablated directions, and
Piras et al. [@piras2025] report large gains over a single direction. My five-seed comparison on
Qwen3-1.7B did not reproduce that advantage, at roughly twice the divergence for no measurable
refusal gain. Two papers say one thing and one measurement here says another, on one model with
one search: a disagreement to resolve rather than a refutation.

What this package adds is narrower than "it measures the cost", and is stated narrowly on
purpose: it validates the instrument before reporting the number, and refuses a judge that does
not agree with itself. It also publishes divergence at a matched refusal rate, for the figures
published here; generalising that to an arbitrary pair of runs needs each tool's frontier
materialised rather than one configuration per seed.

# Availability and licence

Senbonzakura is Python, builds on `transformers` [@wolf2020] and PyTorch, uses `Optuna`
[@akiba2019] for the search, runs on Linux, macOS and Windows, and is AGPL-3.0-or-later.

# Ethics and dual use

The models this produces will answer requests the original refused, including harmful ones. That
is what the tool is for, and it is permanent in the weights. No abliterated weights are published
with this software and the evaluation corpus is gated; the base model's licence and any
acceptable-use policy still govern anything you make with it, and the documentation is explicit
that you should not put one in front of other people without saying what it is.

# Acknowledgements

Thanks to Philipp Emanuel Weidmann, whose work on `Heretic` shaped the search design and whose
keyword metric is used verbatim here, and to the authors of @arditi2024.

# References
