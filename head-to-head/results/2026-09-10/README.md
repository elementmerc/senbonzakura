# The 2026-09-10 head-to-head, as it was measured

Thirty artefacts: five seeds, two tools, three instruments, plus the run's own summary.
Qwen3-1.7B, both tools driven on one machine, every model scored afterwards by our instruments on
prompts neither tool was fitted or selected on.

## Why these are here

`CONTRACT.md` §4 requires a record of every run, and for a month there was none for this one. The
coherence figures on [the benchmark page](https://elementmerc.github.io/senbonzakura/guide/benchmark)
came from these files, and until now those files existed on a single laptop and nowhere a reader
could reach.

That had a cost worth stating. The 2026-09-10 review panel could not check the figures, and the
first correction to that page consequently carried **p = 0.405** and a worst seed of **0.0605**,
both of which came from a synthetic run directory built to exercise the report rather than from
this run. The real values are `p = 0.238` and `0.0936`. A published measurement whose artefacts
are unreachable is a measurement nobody can dispute, which is not a virtue.

`tests/test_published_head_to_head.py` recomputes every figure on the benchmark page from the
files in this directory, so the page and the run cannot drift apart again without CI failing.

## The figures

| | senbonzakura | Heretic |
|---|---|---|
| drift, per seed (42 to 46) | 0.0363, 0.0406, 0.0510, 0.0936, 0.0512 | 0.1075, 0.3591, 0.0439, 0.0573, 0.0456 |
| drift, mean | 0.0545 | 0.1227 |
| drift, median | 0.0510 | 0.0573 |
| hard refusal | 0.0% ×4, 0.5% ×1 | 0.0% ×5 |

Exact permutation test over the ten seeds: **p = 0.238**. There is no detectable coherence
difference on this model at this size. One Heretic seed (43, at 0.3591 against siblings of 0.0439
to 0.1075) carries the whole mean gap; without it Heretic's mean is 0.0636 against our 0.0545.

Heretic is **marginally ahead** on refusal removal, not behind. Both differences are negligible.

This is the run where **both tools were given the same best-of-N selection pass**, which is what
makes it readable and the 2026-08-12 run not.

## What was changed on the way in, and what is not here

Absolute paths were reduced to their basenames. They recorded which machine the run happened on
rather than anything about the measurement, and shipping a build machine's filesystem is a defect
this project found in its own published wheel the same day.

One key was renamed: `prompts`, which holds the NAME of the prompt file and never its contents,
became `prompts_file`. This repository's pre-commit gate refuses any committed JSON carrying a
`prompts` key at all, regardless of what is under it, and that bluntness is deliberate: a leak
gate that reasons about values is a gate with a hole. The data moved rather than the check.

No number, and no rounding, was touched.

Not here, and excluded by `.gitignore` on purpose: the edited models (36 GB), and the per-prompt
margin rows, which are `.jsonl` and carry prompt text and the replies to it. Everything in this
directory is aggregate.

## Read this beside the run's own limits

They are on the benchmark page and they are not small. The equal-budget matching is asymmetric:
trials were matched **up** to Heretic's 200 on the stated principle that starving a comparison
decides it, and direction-fitting prompts were matched **down** to our 256 where Heretic's shipped
default is 400 per side. Both moves went away from the other tool's larger default, and this one
lands on the estimator its method depends on. Re-running at 400 per side is queued work.
