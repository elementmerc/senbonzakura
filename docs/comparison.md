# How this compares to the other abliteration tools

Read from source on 2026-09-08: **Heretic at `7675b90`** and **abliterix at `v1.12.1`**. File and line references point at code you can read yourself. Where a claim is that nobody does something, the search terms are named so you can repeat the search over the whole repository. An earlier draft searched only `src/` and got a headline claim badly wrong.

Four things this page first said were wrong, all in our favour, are corrected below: both other tools measure capability; both already supported the architectures we thought we led on; abliterix computes confidence intervals in its A/B scripts; and a figure of about 27% on real Qwen3-1.7B weights was not measured by anything in this repository. That figure is withdrawn.

## Which tools this page read

| Tool | Read | How |
|---|---|---|
| [Heretic](https://github.com/p-e-w/heretic) | Yes, at `7675b90` on 2026-09-08 | Source, line by line |
| [abliterix](https://github.com/wuwangzhang1216/abliterix) | Yes, at `v1.12.1` on 2026-09-08 | Source, line by line |
| [apostate](https://github.com/heterodoxin/apostate) | Partly, at `main` on 2026-10-01 | Source, through a summarising reader rather than a clone. Enough to correct some claims below, not enough for a column |
| Anything else | No | Not searched for systematically |

"Both other tools" means Heretic and abliterix. The apostate read is recent and partial, and two passes disagreed about which grader its pipeline uses, so it gets no column until somebody has run it.

## The short version

| | Heretic | abliterix | Senbonzakura |
|---|---|---|---|
| Measures what the edit cost | Yes, 11 lm-eval benchmarks | Yes, lm-eval plus a GSM8K "capability tax" | Yes, but on one bundled dataset: a 256 row GSM8K subset, read by whichever of five grading rules fits it |
| Reports its grader's agreement above chance | No | No | **Yes, and it refuses a grader that fails** |
| Puts an interval on the number | No | Yes in its A/B scripts, no in the interactive table | Yes on every measurement; the refusal rate's interval is in the result file |
| Measures whether the edit survives an attack | No | Scoring loops for single-turn and multi-turn, not wired into its run loop, no prompts shipped | Yes: single-turn, multi-turn and a safety-recovery finetune, each with a control arm and an interval, prompt sets in the wheel |
| Runs the safety-recovery finetune itself | No | No, it provides the formula and the caller provides the finetune | Yes, end to end, scored on a held-out split, against a neutral-finetune control |

The interval row is the one an earlier version got wrong in our favour. The last two rows became true on 2026-10-05, not before.

## Where we are level or behind

**Everyone measures capability.** Heretic ships eleven lm-eval benchmarks as defaults (`src/heretic/config.py`) and can benchmark the original model beside the edited one (`src/heretic/main.py:1359`). It is a prompt, not a default, because it doubles the run time. abliterix has an lm-eval table of Original, Abliterated and Delta (`src/abliterix/interactive.py:288`), plus a GSM8K helper and jailbreak harnesses in `src/abliterix/external_eval.py`, which that file says are not wired into its CLI run loop.

**Hybrid architectures: abliterix is ahead of Heretic, and we are now level with abliterix.** Heretic handles `linear_attn.out_proj` and `conv.out_proj` (`src/heretic/model.py:405` and `:423`) and has no state-space path, so on Mamba-2, Jamba and Nemotron it reaches no residual writer. abliterix handles those plus `mixer.out_proj`, `mamba.out_proj` and NemotronH's `mixer.o_proj` (`src/abliterix/core/engine.py:1074`, `:1114`, `:1126`, `:1128`, `:1132`). We have since added five families. We did not lead here, as an earlier draft claimed.

abliterix also seemed to cover an attention variant we could not. It does not: for that variant the residual writer is an ordinary `o_proj`. Its extra work on the projections feeding attention is a different technique, not coverage we lacked.

**Norm restoration is a shared leak, not a field-wide law.** All three tools here keep each weight row's original length after removing a direction. Projection acts across rows and rescaling acts per row, and the two do not commute, so part of the direction survives. How much depends on how uneven the row lengths are. On synthetic matrices, median survival runs from about **6%** where rows are unusually even (a 0.2x spread) to **34 to 36%** at 4x and **41 to 44%** at 10x. The inner ranges are the spread across one, two and four ablated directions. Real checkpoints sit inside that range. The tool measures a model's row-length spread (`tools/research/leak_sweep.py --model`) so you can place your own weights on the curve.

Every figure in that sentence comes from `tools/research/leak_sweep.py`, which runs in four seconds and is deterministic, and a test compares this page against it on every CI run. The old sentence read "about 5% ... 32% at 4x and 46% at 10x". Those were not one measurement: 32% was near the bottom of the 4x range and 46% near the top of the 10x range, which made the curve look steeper than the script ever produced.

No tool read here reports how much of the direction survives. OBLITERATUS records whether restoration ran (`method_config.norm_preserve`), and Heretic's row-normalisation setting is readable from its config. That tells you the leak exists, not how big it is. apostate bakes its projection without per-row restoration, so it does not have this leak. That is a different trade, and what dropping restoration costs is still open.

## What is ours

**Agreement above chance.** No other tool computes it. `grep -ri kappa` returns zero across all three other trees, apostate's included. apostate's delivery grader is HarmBench, an externally validated classifier pinned by revision, so "nobody validates the judge" is not the defensible claim. The defensible claim is narrower: nobody reports whether a grader beats guessing.

abliterix uses an external model as its judge (`google/gemini-3.1-flash-lite-preview` by default, `src/abliterix/settings.py:1405`). Heretic uses a fixed keyword list. Neither checks that its grader is any good. abliterix does name the gap in `docs/benchmarks/2026-05-pod-validation.md:132`: "Cross-judge calibration is on the roadmap". They have written down that it needs doing, which is more than most.

`senbonzakura judge` fills the gap. It scores a candidate judge against a labelled set and reports Cohen's kappa beside raw agreement, and it refuses a judge that scores 90% agreement at a kappa of 0.00. A judge that says "not a refusal" to everything scores 90% on a set that is 90% non-refusals. Raw agreement cannot tell that judge from a good one. Kappa can.

**Intervals.** Heretic computes none. abliterix keeps real statistics in `scripts/`: a Wilson rate interval, a paired bootstrap, a cluster bootstrap, and decision gates written on the interval bound (`scripts/ab_test_qwen35.py:483`, `:394`, `:434`). Its interactive lm-eval table gets standard errors from lm-eval and strips them before display (`src/abliterix/interactive.py:277` and `:283`), so a user comparing two models there sees bare percentages.

We are not complete either. The compass and the capability score carry intervals on every figure. `drift` prints a seeded bootstrap interval beside its KL (`src/senbonzakura/drift.py`, `kl_interval`). The refusal rate stamps a Wilson interval and the raw counts into its result file (`src/senbonzakura/score.py`, `_stamp_refusal`). Two of those three were fixed on 2026-09-25, and an earlier version of this page still said all three were bare.

Two gaps remain. `score` prints its finishing percentage bare on the terminal, even though the artefact beside it carries the interval. `coherence` has no interval and will not get one: one forward pass over one fixed passage returns the same number every time, so there is no run-to-run spread to describe.

**Withdrawals.** When a measurement here turns out to have been made wrongly, the number is withdrawn in the CHANGELOG under its own heading, with the size of the error. Several have been: an entire model family's results, a direction-count comparison, and a refusal rate scored on the wrong partition.

## What we are worse at

**Adoption cost.** Editing a model here pulls in torch and needs a GPU, as the others do. The checking commands no longer need one.

**Breadth of benchmark.** Eleven standard benchmarks with a harness people already trust beat one dataset graded by our own code, if the goal is convincing a stranger. "Five graded tasks" means five **grading rules** (`numeric`, `multiple-choice`, `tool-call`, `exact` and `constraints`), not five datasets. A run uses whichever rule fits the dataset it was given. The package ships one dataset, a 256 row GSM8K subset, which is the benchmark the table credits abliterix with too. The honest comparison is eleven datasets against one. What we have instead is that our data can be graded without a judge. That is a trade, not a win.

**Breadth of attack.** This paragraph got wrong twice in one week. It first said abliterix carried helpers for JALMBench, MTJ-Bench, Crescendo and TamperBench. Reading the clone showed otherwise. It then said we had nothing equivalent to abliterix's attack loops. That stopped being true on 2026-10-05, when `senbonzakura jailbreak`, `senbonzakura multi-turn` and `senbonzakura tamper` landed.

abliterix has `evaluate_jailbreak` and `evaluate_multi_turn` in `src/abliterix/external_eval.py`. They are working attack-scoring loops, and they take a caller-supplied generator. Its `pyproject.toml` depends on none of JALMBench, MTJ-Bench, TamperBench or StrongREJECT, and it ships no attack prompts. The file's own docstring says these are "library helpers, they are not auto-wired into the CLI run loop".

Ours have the uncertainty and the controls. All three commands report a Wilson interval on every rate, including the terminal summary. `jailbreak` runs a paired over-refusal arm on harmless prompts and refuses to subtract a baseline that was not the same measurement. `multi-turn` runs a plain-repetition control by default and reports "cannot tell" when intervals overlap. `tamper` runs a real safety-recovery finetune with a neutral-finetune control and a base-model ceiling arm, on a positional split so held-out prompts were never trained on. The difference is the uncertainty and the controls, not the loop, and a reader can check that by running the commands.

**Tamper resistance.** abliterix's `compute_tamper_resistance` is three lines of arithmetic over two floats the caller already holds: `1.0 - (post - pre) / (1.0 - pre)`, clipped. It generates nothing, loads nothing and trains nothing, so the user runs the finetune by means abliterix does not provide. `senbonzakura tamper` runs that finetune, scores held-out prompts, and reports the recovered fraction with an interval. The run is pre-registered before it executes, and `prereg --run` checks the executed run against the registration by machine.

Six prompt sets ship inside our wheel under MIT and CC-BY. The open question is no longer whether anybody measures repair survival. It is how far the numbers generalise, and ours come from a small number of models on modest hardware.

**Single-model evidence.** Most of our numbers come from Qwen3-1.7B, because that is what fits the card this was built on. A claim measured on one model is a claim about one model.

## Repeating this

Every line reference points at a public repository at a pinned commit. Nothing here was run to produce the read. That is deliberate: reading the source answers "what does this tool measure" without either project's benchmark harness getting a say.

If something here is out of date, or if we have misread one of these tools, please open an issue. Four claims in the first draft were wrong in our favour, and an outside reader found them, not us. We would rather find the fifth that way than not at all.
