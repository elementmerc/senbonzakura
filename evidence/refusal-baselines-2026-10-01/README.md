# Baseline refusal rates, eight models

How often each un-edited model refuses the bundled evaluation track, measured on the
held-out rows nothing was fitted on. These are the numbers a reader needs before any
claim about how far an edit moved a model: without a baseline, a post-edit refusal rate
is a figure with no scale behind it.

Two rows were produced on 2026-09-27 and never harvested, which is why this directory
exists; the research notes said no artefact held these figures, and they had been sitting
on the GPU machine the whole time. The rest were measured on 2026-10-01.

## The numbers

Both columns are the same 128 held-out prompts, scored twice by two different rulers.

| Model | This project's ruler | Heretic's keyword rule | Measured |
|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | 80.5% | 81.2% | 09-27 and 10-01 |
| allenai/OLMo-2-0425-1B-Instruct | 68.8% | 71.9% | 10-01 |
| LiquidAI/LFM2-1.2B | 60.9% | 65.6% | 10-01 |
| Qwen/Qwen3-1.7B | 37.5% | 65.6% | 09-27 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 25.0% | 36.7% | 09-27 and 10-01 |
| HuggingFaceTB/SmolLM2-135M-Instruct | 6.3% | 16.4% | 09-27 |
| Qwen/Qwen3-0.6B | 4.7% | 38.3% | 09-27 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 0.8% | 7.8% | 09-27 |

Every row is n = 128 with a Wilson score interval in the JSON beside the point estimate.
Read the interval, not the digit: at n = 128 a rate near 50% carries roughly plus or
minus 9 percentage points.

## Two things worth taking from this table

**The two rulers disagree, and the gap widens as the rate falls.** At the top of the
table they agree within a point. At the bottom they differ by a factor of eight. Qwen3-0.6B
is the sharp case: this project's ruler reads 4.7% and Heretic's keyword rule reads 38.3%,
which puts the same model on opposite sides of the 5% floor below which this tool refuses
to edit at all. The two Qwen3 rows are both thinking models, which emit a reasoning block
before the answer, and a keyword rule that scans the whole output sees refusal language in
reasoning that the model then declines to act on. Which ruler a comparison table reports is
therefore a decision, not a detail, and it is one the head-to-head contract does not yet
record.

**Two models were measured twice, four days apart, on separate runs, and both reproduced
to the digit.** Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct return identical point
estimates and identical Wilson intervals on 09-27 and 10-01. In each pair the two files
differ in exactly two fields, and neither is a result: the later run records torch 2.14.1
and transformers 5.18.0 where the earlier one records 2.14.0 and 5.17.0. So the scorer
held steady across a torch patch bump and a transformers minor bump, which is a harder
test than a repeat run on a frozen environment. That is the reproducibility claim this
project makes about its own scorer, tested rather than asserted.

## Reading the provenance

Each file carries a `provenance` block: the tool version, Python, platform, device, the
accelerator, and the version of every package that could move a number. All ten were taken
on an RTX 3060 Laptop GPU under WSL2, with senbonzakura 0.4.0, torch 2.14.1+cu130 and
transformers 5.18.0.

**`"git": null` in every file, and that is a real limitation.** These runs used an
installed wheel rather than a source checkout, so the tool could record its version but
not the commit it was built from. A version string pins the release; it does not pin a
working tree. Treat these as 0.4.0-release numbers rather than as traceable to a specific
commit, and prefer a source-checkout run for anything that has to be bisected later.

## Files

`baseline-*.json` are the 2026-09-27 sweep. The rest are the 2026-10-01 screen. Where a
model appears in both, the two files are kept rather than deduplicated, because the pair
is the reproducibility evidence.
