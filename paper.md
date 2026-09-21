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

Abliteration means removing a language model's refusal behaviour by editing its weights, rather
than by talking it into cooperating. It works because of a result from @arditi2024: refusal in an
instruction-tuned model is carried by a direction in activation space, and if you project that
direction out of the weights that write to the residual stream, the behaviour goes and it stays
gone. Several tools do this, and they do it well.

Senbonzakura does it too. The difference is that it then measures what the edit cost, and that
second half is the reason I built it.

Taking refusal out is the easy part. The hard part is knowing whether you also took out the
model's judgement, its reasoning, or its grip on the language, because a tool that only reports
the refusal rate cannot tell those apart. A lobotomy and a success look the same on that chart.

So the package ships the instruments next to the edit. It scores refusal with two independent
rulers, measures capability on tasks that are graded by code rather than by another model,
measures coherence as the perplexity of a fixed passage, measures the distributional cost as KL
divergence on harmless prompts, and measures whether the model still *recognises* harm even though
it will now discuss it. Every figure comes with the conditions attached: which rows it was scored
on, an interval instead of a single number, and a null control beside it.

That last one matters more than it sounds. A control is the whole reason a number means anything,
and this project has twice found an instrument measuring the wrong thing only because the control
caught it.

# Statement of need

Work on abliteration reports that it works. What it mostly does not report is what else changed,
and the field has no shared way of asking. Three things make that harder than it looks, and I ran
into all three.

**A refusal rate is a property of the corpus, the prompt format and the token budget, not just of
the model.** Score the same weights on a different set of prompts and the number moves. Score them
with a generation budget shorter than the model's refusals and the rate reads low, because a
refusal that gets cut off before it is emitted is counted as compliance. Both of those have
happened to figures this project published, and both were caught after the numbers had already
travelled.

**Picking a configuration and reporting it on the same rows you picked it on is not a
measurement.** An automated search over a few hundred trials will find whatever does best on what
you showed it. Reporting that score on those same rows reports the best of N draws. Senbonzakura
splits its corpus into rows the direction is fitted on, rows the search selects on, and rows
nothing has ever seen, and it refuses to cross the boundary.

**Two numbers are only comparable if they were made the same way.** If the scorer changed between
two runs, you have two numbers on two instruments. I have had to withdraw figures for exactly that
reason, so every measurement now carries its own identity: the metric, the estimator that produced
it, the units, the input it was taken on, and the version of the code that took it.

A companion package, `senbonzakura-check`, reads those records back and checks them. It also reads
result files from other harnesses [@gao2024; @inspect2024], so you can point it at somebody else's
numbers instead of only at mine. It installs without PyTorch, because a tool that reads JSON has no
business pulling down a deep learning stack.

The project is deliberate about what it has *not* shown. I started it on the hypothesis that
removing several refusal directions beats removing one. My own five-seed comparison did not support
that, so the result was withdrawn rather than quietly dropped, and the documentation carries a page
saying which claims the evidence in the repository actually supports and which it does not. I would
rather ship something that argues with its own author than something that does not.

# State of the field

`Heretic` [@heretic2025] is the closest neighbour and the direct influence on the search here. It
automates the choice of ablation parameters under a KL constraint, which is a real advance on
hand-tuning. Senbonzakura uses its keyword metric verbatim, which is why this package is
AGPL-3.0-or-later, and runs that metric as one arm of its own comparisons so both tools can be read
on one ruler.

The difference is what happens after the edit. Existing tools optimise against a refusal score
under a divergence constraint. Senbonzakura also reports capability and harm recognition on
held-out rows, publishes the divergence at a *matched* refusal rate so that two tools which removed
different amounts of refusal are not compared as though they removed the same amount, and writes
into the artefact what would make each figure wrong.

# Availability and licence

Senbonzakura is Python, builds on `transformers` [@wolf2020] and PyTorch, and uses `Optuna`
[@akiba2019] for the search. It runs on Linux, macOS and Windows, on CPU or on a consumer GPU, and
it is licensed AGPL-3.0-or-later.

One thing should be said plainly. The models it produces will answer requests the original refused,
including harmful ones. That is what the tool is for, and it is permanent in the weights. The base
model's licence still governs the result, and the documentation is explicit that you should not put
one in front of other people without saying what it is.

# Acknowledgements

Thanks to Philipp Emanuel Weidmann, whose work on `Heretic` shaped the search design here, and to
the authors of @arditi2024, whose result the whole approach rests on.

# References
