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
date: 26 September 2026
bibliography: paper.bib
---

# Summary

Abliteration removes a language model's refusal behaviour by editing its weights rather than by
talking it into cooperating. It rests on @arditi2024: refusal is carried by a direction in
activation space, and projecting it out of the weights that write to the residual stream removes
the behaviour for good. Several tools do this well, and the better ones report what else changed.
What none of them do is validate the instrument.

Taking refusal out is easy. The hard part is knowing whether you also took out the model's
judgement, its reasoning or its grip on the language, and a tool reporting only the refusal rate
cannot tell success from damage.

So Senbonzakura ships the instruments beside the edit: refusal on two rulers, its own and a
competitor's; capability graded by code rather than by another model; coherence as perplexity on a
fixed passage; distributional cost as KL divergence; and whether the model still *recognises* harm
although it will now discuss it. Every figure carries its conditions and a null control. The
control is why a number means anything: this compass was once matched, on the same exam, by a
ruler reading only sentence length.

# Statement of need

Three things make such a number hard to trust, and this project met all three.

A refusal rate is a property of the corpus, the prompt format and the token budget, not only of
the model: a refusal cut off before it's emitted counts as compliance. Reporting a configuration
on the rows it was selected on is not a measurement, since a search returns the best of N draws,
so the corpus splits into fitting, selection, and unseen rows. Two numbers are comparable only if
made the same way, so each figure records its metric, estimator, units, input, precision and code
version. Figures here have been withdrawn for each fault, after they had travelled.

A companion package, `senbonzakura-check`, reads those records back and checks them, and reads
other harnesses' result files [@gao2024; @inspect2024] too, so it can be pointed at somebody
else's numbers without PyTorch.

The ceiling belongs here rather than in the documentation. **No standing measurement exists above
3B parameters**, and the instruments weaken as the model shrinks: Qwen3-1.7B's compass scores
0.9887 against a length-only control at 0.6564, while Qwen3-0.6B scores 0.6616 against that same
control, which measures nothing. **Every Gemma figure is withdrawn**: the edit never reached the
residual stream.

# State of the field

`Heretic` [@heretic2025] is the direct influence on the search, automating ablation parameters
under a KL constraint. Senbonzakura takes its keyword marker list verbatim and adapts its
normalisation, which is why this package is AGPL-3.0-or-later, and runs that metric beside its own
so both tools read on one ruler. `abliterix` [@abliterix2025] is ahead on method breadth,
mixture-of-experts handling and prebuilt configurations.

Wollschläger et al. [@wollschlager2025] show refusal is a cone, and Piras et al.
[@piras2025] report large gains over one direction. Five seeds per
arm on Qwen3-1.7B went the other way, at matched hard refusal: drift 0.0497 against 0.0932,
*p* = 0.016 two-sided exact permutation, 0.048 dropping the outlying seed.
These arms predate the held-out selection and took whatever an unselective filter accepted, so
they price an *arbitrary* second direction where both papers choose theirs deliberately, on one
family at one size.

What this package adds is narrow: it validates the instrument before reporting the number.

# Ethics and dual use

The models this produces will answer requests the original refused, including harmful ones. That's
the point, and it's permanent in the weights. No abliterated weights ship with it; the base
model's licence still governs anything made with it, and the documentation says plainly not to put
one in front of others without saying what it is.

**A release is not prompt-free.** The held-out track is gated, but a released wheel carries
roughly 6,200 harmful prompts, obfuscated with the key beside them: a speed bump. A corpus-free
build is documented.

# Availability

`pip install senbonzakura`, on Linux, macOS and Windows, on `transformers` [@wolf2020], PyTorch
and `Optuna` [@akiba2019]. AGPL-3.0-or-later. Thanks to Philipp Emanuel Weidmann, whose `Heretic` shaped this
search and whose keyword markers are used here, and to the authors of @arditi2024.

# References
