# How this compares to the other abliteration tools

Read from the source of all three on 2026-09-08, against **Heretic at `7675b90`** and
**abliterix at `v1.12.1`**. Where a claim below points at a file and a line, that is a line
somebody can go and read. Where the answer is "nobody does this", it means a search of both
source trees turned up nothing, and the search terms are named so you can repeat it.

The honest summary first, because two things I previously believed about this comparison
turned out to be wrong when I actually went and read the code, and it would be poor form to
bury that: **both of the other tools measure capability, and both already support the
architectures I thought we led on.** What survives is narrower and, I think, more interesting.

## The three-line version

| | Heretic | abliterix | Senbonzakura |
|---|---|---|---|
| Measures what the edit cost | Yes, 11 lm-eval benchmarks | Yes, lm-eval plus a GSM8K "capability tax" | Yes, 5 graded tasks |
| Validates the thing doing the grading | No | No | **Yes, and it refuses a bad one** |
| Puts an interval on the number | No | No, filtered out before display | **Yes, on every figure** |

## What everyone does, including us

**Everyone measures capability.** I said in an earlier draft that nothing in the field did,
and that was simply false.

Heretic ships eleven benchmarks from the Language Model Evaluation Harness as defaults, in
`src/heretic/config.py`: AGIEval, BIG-Bench Hard, CommonsenseQA, EQ-Bench, GSM8K, HellaSwag,
IFEval, MMLU, MMLU-Pro, PIQA and WinoGrande. It benchmarks the original model alongside the
edited one, which is the part that matters: a score with nothing to compare it against says
very little.

abliterix has the same idea with a different shape, an lm-eval table of Original, Abliterated
and Delta, plus a GSM8K helper it calls a capability tax
(`src/abliterix/external_eval.py`).

**Everyone covers the hybrid architectures.** I thought we led here and we did not. Heretic
handles `linear_attn.out_proj` and `conv.out_proj` (`src/heretic/model.py:405` and `:423`);
abliterix handles those plus `mixer.out_proj` and `mamba.out_proj`
(`src/abliterix/core/engine.py:1074`, `:1114`, `:1126`, `:1128`). We were level, not ahead,
and we have since added five families all three of us can now handle.

I also thought abliterix covered an attention variant we could not, and that was my mistake
too. The thing an abliteration edit has to reach is whichever weight writes back into the
residual stream, and for that variant it is an ordinary `o_proj` like any other. Their extra
work on the projections feeding into attention is a different technique with a different
purpose, not coverage we lacked.

**Everyone has the norm-restore problem.** All three tools preserve each weight row's
original length after removing a direction, which is what keeps the edited model coherent.
Projection acts across rows and rescaling acts per row, and those two operations do not
commute, so part of the direction survives. Measured on real Qwen3-1.7B weights, the median
survival is about 27%. This is inherited from the published method rather than invented by
any of us, and none of the three tools reports it. We at least now measure ours.

## What is actually ours

Three things, and they are all about whether a number deserves to be believed rather than
about the edit itself.

**Nobody validates the judge.** Both other tools grade refusal with something, and neither
checks that the something is any good. abliterix uses an external model as a judge
(`google/gemini-3.1-flash-lite-preview` by default, `src/abliterix/settings.py:1405`).
Searching both source trees for *kappa*, *Cohen*, *inter-rater* and *agreement* returns
nothing at all. That is the whole reason `senbonzakura judge` exists: it scores a candidate
judge against a labelled set and reports Cohen's kappa alongside raw agreement, and it will
refuse a judge that agrees with the labels 90% of the time at a kappa of 0.00. A judge can hit
90% by saying "not a refusal" to everything if 90% of the set is not a refusal. Raw agreement
cannot tell those two judges apart. Kappa can, and a tool that never computes it has no way to
know which one it is using.

**Nobody puts an interval on the number.** Searching both trees for *confidence interval*,
*Wilson* and *bootstrap* finds one unrelated hit. abliterix goes further than not computing
one: lm-eval hands it standard errors and it explicitly strips them before display
(`src/abliterix/interactive.py:277` and `:283`). So a user sees an MMLU percentage with no
way to tell whether the two points it moved are a finding or a coin flip.

Every figure this tool prints carries an interval, and the comparisons that matter are paired
rather than compared side by side, which is a meaningfully tighter test on the same data.

**We publish what our own numbers cost.** This is less a feature than a habit, and it is the
one I would most like to be copied. When a measurement here turns out to have been made
wrongly, the number is withdrawn in the CHANGELOG under its own heading with the size of the
error, rather than quietly corrected. Several of them are: an entire model family's results,
a direction-count comparison, a refusal rate scored on the wrong partition.

## What we are worse at

**Adoption cost.** Installing this pulls torch and expects a GPU, same as the others.
Nothing in this document changes that, and it is the largest single thing standing between the
tool and the people who would use it.

**Breadth of benchmark.** Eleven standard benchmarks with a harness everybody already trusts
beats five tasks graded by our own code, for the specific purpose of convincing a stranger.
Ours are chosen so they can be graded without a judge at all, which is a different trade, but
it is a trade and not a win.

**Single-model evidence.** Most of the numbers here come from Qwen3-1.7B, because that is
what fits on the card this was built on. A claim measured on one model is a claim about one
model.

## Repeating any of this

Every line reference above is a public repository at a pinned commit. Nothing here was run to
produce it, which is deliberate: reading the source answers "what does this tool measure"
without either project's benchmark harness getting a say in the answer.

If something below is out of date, or if I have read one of these wrongly, please open an
issue. Two of the claims in the first draft of this page were wrong in the other tools'
favour, and I would rather find the third one that way than not at all.
