# How this compares to the other abliteration tools

Read from the source of all three on 2026-09-08, against **Heretic at `7675b90`** and
**abliterix at `v1.12.1`**. Where a claim below points at a file and a line, that is a line
somebody can go and read. Where the answer is "nobody does this", the search terms are named
so you can repeat it over the WHOLE repository, which is a correction: an earlier draft
searched only `src/` and got a headline claim badly wrong because of it.

Four things this page first said were wrong, all of them in our favour, and every one is
corrected below rather than quietly dropped: both other tools measure capability; both already
supported the architectures we thought we led on; abliterix computes confidence intervals and
paired bootstraps in its A/B scripts; and a norm-restore figure quoted here as measured on real
weights was not measured by anything in this repository.

If that reads like a lot of retractions for one page, it is, and it is also the reason to
trust what is left. What survives is narrower than the first draft claimed.

## Which tools this page read, and which it didn't

The title says "the other abliteration tools" and the page surveys two of them. That's a scope,
so it belongs in writing rather than in the reader's inference.

| Tool | Read | How |
|---|---|---|
| [Heretic](https://github.com/p-e-w/heretic) | Yes, at `7675b90` on 2026-09-08 | Source, line by line |
| [abliterix](https://github.com/wuwangzhang1216/abliterix) | Yes, at `v1.12.1` on 2026-09-08 | Source, line by line |
| [apostate](https://github.com/heterodoxin/apostate) | Partly, at `main` on 2026-10-01 | Source, through a summarising reader rather than a clone. Enough to correct the claims below, not enough to give it a column |
| Anything else | No | Not searched for systematically |

Where a sentence below says "both other tools", or counts three tools, it means Heretic,
abliterix and this one, and it's scoped to the dates above. The apostate read is recent and
partial, and two independent passes disagreed about which grader its main pipeline uses, so it
corrects the claims it touches and gets no column until somebody has run it.

## The three-line version

| | Heretic | abliterix | Senbonzakura |
|---|---|---|---|
| Measures what the edit cost | Yes, 11 lm-eval benchmarks | Yes, lm-eval plus a GSM8K "capability tax" | Yes, but on one bundled dataset: a 256 row GSM8K subset, read by whichever of five grading rules fits it |
| Reports its grader's agreement above chance | No | No | **Yes, and it refuses a grader that fails** |
| Puts an interval on the number | No | Yes in its A/B scripts, no in the interactive table | Yes on every measurement, and the refusal rate's lives in the result file rather than on the terminal line |
| Measures whether the edit survives an attack | No | Scoring loops for single-turn and multi-turn, not wired into its run loop, and no prompts shipped | Yes: single-turn, multi-turn and a safety-recovery finetune, each with a control arm and an interval, with the prompt sets in the wheel |
| Runs the safety-recovery finetune itself | No | No, it provides the formula and the caller provides the finetune | Yes, trained end to end, scored on a held-out split, against a neutral-finetune control |

Read the interval row carefully, because an earlier version of this page got it wrong in our
favour and it was the page's headline claim. Four of these five rows are a clear lead, and the
last two became true on 2026-10-05 rather than having been true all along.

## What everyone does, including us

**Everyone measures capability.** I said in an earlier draft that nothing in the field did,
and that was simply false.

Heretic ships eleven benchmarks from the Language Model Evaluation Harness as defaults, in
`src/heretic/config.py`: AGIEval, BIG-Bench Hard, CommonsenseQA, EQ-Bench, GSM8K, HellaSwag,
IFEval, MMLU, MMLU-Pro, PIQA and WinoGrande. It offers to benchmark the original model alongside the
edited one, which is the part that matters: a score with nothing to compare it against says
very little. It is a prompt rather than a default (`src/heretic/main.py:1359`), because it
doubles the time, so a user in a hurry gets the one-model run.

abliterix has the same idea with a different shape: an lm-eval table of Original, Abliterated
and Delta (`src/abliterix/interactive.py:288`), plus library helpers for a GSM8K capability tax
and several jailbreak harnesses (`src/abliterix/external_eval.py`), which that file notes are
not wired into its CLI run loop.

**Hybrid architectures: we were not ahead, and abliterix is ahead of Heretic.** I thought we
led here and we did not. Heretic handles `linear_attn.out_proj` and `conv.out_proj`
(`src/heretic/model.py:405` and `:423`) and has no state-space path at all, so on Mamba-2,
Jamba and Nemotron it reaches no residual writer. abliterix handles those plus
`mixer.out_proj`, `mamba.out_proj` and NemotronH's `mixer.o_proj`
(`src/abliterix/core/engine.py:1074`, `:1114`, `:1126`, `:1128`, `:1132`), which is a wider
list than this page first credited it with. We have since added five families, so on this axis
we are level with abliterix and both of us are ahead of Heretic.

I also thought abliterix covered an attention variant we could not, and that was my mistake
too. The thing an abliteration edit has to reach is whichever weight writes back into the
residual stream, and for that variant it is an ordinary `o_proj` like any other. Their extra
work on the projections feeding into attention is a different technique with a different
purpose, not coverage we lacked.

**All three tools here have the norm-restore problem, and it isn't a law of the field.** All
three preserve each weight row's
original length after removing a direction, which is what keeps the edited model coherent.
Projection acts across rows and rescaling acts per row, and those two operations do not
commute, so part of the direction survives.

How much depends on how uneven the row lengths are, and the honest answer is a range rather
than a number. On synthetic matrices, median survival runs from about **6%** where rows are
unusually even (a 0.2x spread) to **34 to 36%** at 4x and **41 to 44%** at 10x, the inner ranges
being the spread across one, two and four ablated directions. Real checkpoints sit inside that
range; the tool measures a model's row-length spread (`tools/research/leak_sweep.py --model`) so you can
place your own weights on the curve, and it does not turn that into a single figure, because the
instrument cannot produce one.

Every figure in that sentence comes from `tools/research/leak_sweep.py`, which runs in four seconds and is
deterministic, and a test compares this page against it on every CI run. It has to, because the
sentence used to read "about 5% ... 32% at 4x and 46% at 10x", and those three were not one
measurement: 32% was near the BOTTOM of the range at 4x and 46% was near the TOP at 10x, which
made the curve look steeper than the script has ever produced. Nothing was checking, so nobody
could tell.

An earlier draft of this page quoted "about 27% on real Qwen3-1.7B weights". That number was
not measured by anything in this repository and has been withdrawn.

This is inherited from the published method rather than invented by any of us, and no tool read
here reports **how much of the direction survives**. Two of them do report whether the
restoration is switched on, which is a different fact: OBLITERATUS records it per run as
`method_config.norm_preserve`, and Heretic's row-normalisation setting is readable from its own
configuration. Knowing the restoration ran tells you the leak exists. It does not tell you how
big it is, and the size is the part that decides whether a published refusal rate means what it
says.

It was also stated here as a property of the field, and it isn't one. A fourth tool, apostate,
bakes its projection without any per-row norm restoration, so it doesn't have this problem to
inherit. That's a different trade rather than a free win, since the restoration is what the
published method uses to keep the edited model coherent, and the question of what dropping it
costs is open. The honest shape of the claim: three tools do this and none of them measures the
residue, and skipping the restoration is a route out that at least one tool has taken.

## What is actually ours

Three things, and they are all about whether a number deserves to be believed rather than
about the edit itself.

**Nobody computes agreement above chance.** That's the defensible version, and it's narrower
than what this page said first, which was "nobody validates the judge". `grep -ri kappa` returns
zero across all three other trees, including apostate's, so no tool in the field reports whether
its grader beats guessing. But apostate's delivery grader is HarmBench, an externally validated
published classifier pinned by revision, which is a real answer to "is your grader any good" even
though it isn't this one. "Nobody validates the judge" invites the reply "we use HarmBench,
pinned", and that reply would be fair.

Both other tools read here grade refusal with something, and neither
checks that the something is any good. abliterix uses an external model as a judge
(`google/gemini-3.1-flash-lite-preview` by default, `src/abliterix/settings.py:1405`);
Heretic's grader is a fixed keyword list. `grep -ri kappa` returns zero in both trees.

Credit where it is due: abliterix names the gap itself. Its
`docs/benchmarks/2026-05-pod-validation.md:132` records "Single judge model (Gemini Flash
Lite). Cross-judge calibration is on the roadmap" and cites the literature for it. They have
not done it; they have written down that it needs doing, which is more than most.

That gap is the whole reason `senbonzakura judge` exists: it scores a candidate
judge against a labelled set and reports Cohen's kappa alongside raw agreement, and it will
refuse a judge that agrees with the labels 90% of the time at a kappa of 0.00. A judge can hit
90% by saying "not a refusal" to everything if 90% of the set is not a refusal. Raw agreement
cannot tell those two judges apart. Kappa can, and a tool that never computes it has no way to
know which one it is using.

**Intervals: a genuine gap, and a much narrower one than this page first claimed.**

The first version of this page said nobody computes an interval, on the strength of a search
that only covered `src/`. abliterix keeps its statistics in `scripts/`, and they are real work:
`wilson_rate_interval` and `paired_bootstrap_delta_ci` at `scripts/ab_test_qwen35.py:483` and
`:394`, a cluster bootstrap at `:434`, decision gates written on the interval bound rather than
on the point estimate, and `ci95` in their published benchmark JSON. That is a better statistical
practice than this page originally credited them with, and correcting it is the point of
publishing the page at all.

What is true is narrower. **Heretic computes none at all.** And abliterix's interactive lm-eval
table, the surface most users actually read, is handed standard errors by lm-eval and strips
them before display (`src/abliterix/interactive.py:277` and `:283`), so a user comparing two
models there sees bare percentages.

**And we don't manage it everywhere either, though less of this is outstanding than the page
said.** The compass and the capability score carry intervals on every figure. So does `drift`,
which prints a seeded bootstrap interval beside its KL on the same line
(`src/senbonzakura/drift.py`, `kl_interval`), and so does the refusal rate, which stamps a Wilson
interval and the raw counts into its result file (`src/senbonzakura/score.py`, `_stamp_refusal`).
This page previously said all three printed bare numbers. Two of those three were fixed on
2026-09-25 and the page wasn't re-read.

What's actually left is one line of terminal output and one measurement that doesn't want an
interval. `score` prints its finishing percentage bare even though the artefact beside it carries
the interval, which is a display gap rather than a measurement gap and is still a gap, because the
line on the terminal is what gets pasted into a message. `coherence` carries no interval and is
not going to: one forward pass over one fixed passage under fixed precision returns the same
number every time, so there's no run-to-run spread to describe, and a token-level interval would
put variation across the passage into a field that means variation across runs.

**We publish what our own numbers cost.** This is less a feature than a habit, and it is the
one I would most like to be copied. When a measurement here turns out to have been made
wrongly, the number is withdrawn in the CHANGELOG under its own heading with the size of the
error, rather than quietly corrected. Several of them are: an entire model family's results,
a direction-count comparison, a refusal rate scored on the wrong partition.

## What we are worse at

**Adoption cost.** Editing a model here pulls torch and expects a GPU, same as the others.
The checking commands no longer do, which helps, and it does not change the fact that the thing
the tool is named for needs a card.

**Breadth of benchmark.** Eleven standard benchmarks with a harness everybody already trusts
beats one dataset graded by our own code, for the specific purpose of convincing a stranger.

Worth being exact about, because the row above reads better than it is. "Five graded tasks" is
five **grading rules**, not five datasets: `numeric`, `multiple-choice`, `tool-call`, `exact` and
`constraints` are five ways to read an answer, and a run uses whichever one fits the dataset it
was given. What ships in the package is a single dataset, a 256 row subset of GSM8K, which is
the same benchmark the table credits abliterix with under another name. The other four rules
exist for a benchmark you bring yourself. So the honest comparison is eleven datasets against
one, and what we have instead is that ours can be graded without a judge at all. That is a
different trade, and it is a trade rather than a win.

**Breadth of attack, which this paragraph got wrong twice in one week and is worth reading as a
record of both corrections.** It first said abliterix "carries helpers for JALMBench, MTJ-Bench and
Crescendo, and TamperBench". **Reading the clone said otherwise.** It then said we had nothing
equivalent to the attack loops abliterix does have, and **that stopped being true on 2026-10-05**,
when `senbonzakura jailbreak`, `senbonzakura multi-turn` and `senbonzakura tamper` landed.

What is real on their side: `evaluate_jailbreak` and `evaluate_multi_turn` in
`src/abliterix/external_eval.py` are **working attack-scoring loops**. They take a caller-supplied
generator, run prompts through it, and classify the replies, and the multi-turn one walks a
dialogue history turn by turn.

What is real on ours, and the difference is the uncertainty rather than the loop: all three of our
commands report a **Wilson interval on every rate**, including the terminal summary lines, and
each ships the control that tells a result from an artefact of the method. `jailbreak` runs a
**paired over-refusal arm** on harmless prompts, because a model that complies with everything
scores perfectly on attack prompts and is useless; it offers two estimators, and `--baseline`
**refuses to subtract** a run that was not the same measurement. `multi-turn` records the
escalation by name and digest, runs a **plain-repetition control** by default to show the gain came
from the escalation rather than from asking twice, and reports **"cannot tell"** when the two
intervals overlap rather than quoting a difference. `tamper` runs **a real safety-recovery finetune
end to end**, with a neutral-finetune control arm and a base-model ceiling arm, on a positional
split so the held-out prompts were never trained on.

So the row inverts: **the loops are no longer the gap, the intervals and the controls are the
difference**, and that is a claim a reader can check by running the commands.

What is not real: its `pyproject.toml` **depends on none of JALMBench, MTJ-Bench, TamperBench or
StrongREJECT**, there is no vendored copy and no optional extra, and `grep -RIn tamperbench` over
the whole clone returns four hits, three of them docstrings in that one file and one a test title.
The file's own docstring says these are "**library helpers, they are not auto-wired into the CLI run
loop**". And **it ships no attack prompts**: the docstring says "prompts in whatever format the
generator expects", so the corpus is the caller's problem. A harness without its corpus is less than
a benchmark integration and more than nothing.

**On tamper resistance specifically, this was the row we were losing and is now the row we are
clearly ahead on.** `compute_tamper_resistance` is three lines of arithmetic over two floats the
caller must already hold: it computes `1.0 - (post - pre) / (1.0 - pre)`, clipped, and its own
docstring calls it an "Abliterix-specific normalized recovery metric (inspired by recovery
evaluations such as TamperBench)". It generates nothing, loads nothing and trains nothing, so **a
user must run the safety-recovery finetune themselves by means abliterix does not provide**. The
formula is the easy half; the finetune is the measurement.

`senbonzakura tamper` **runs that finetune**, which is why this row moved. It trains the recovery
arm, scores held-out prompts the training never saw, and reports the recovered fraction with an
interval. It runs a **neutral finetune as a control arm**, so a recovery number cannot be the
result of finetuning on anything at all, and a **base-model arm** for the ceiling. The run is
**pre-registered before it executes**, and `prereg --run` checks the executed run against the
registration by machine rather than by trust.

So the honest statement of this row: **we have the corpora and now the loops as well**, since six
prompt sets ship inside our wheel under MIT and CC-BY. The open question is no longer whether
anybody measures repair survival; it is how far the numbers generalise, and ours so far come from
a small number of models on modest hardware.

**Single-model evidence.** Most of the numbers here come from Qwen3-1.7B, because that is
what fits on the card this was built on. A claim measured on one model is a claim about one
model.

## Repeating any of this

Every line reference above is a public repository at a pinned commit. Nothing here was run to
produce it, which is deliberate: reading the source answers "what does this tool measure"
without either project's benchmark harness getting a say in the answer.

If something here is out of date, or if I have read one of these wrongly, please open an
issue. Four claims in the first draft of this page were wrong in our own favour and were found by
someone reading it adversarially rather than by me. I would rather find the fifth
that way than not at all.
