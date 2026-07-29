# Measured environment: the July 2026 compass sweep

**Status: INCOMPLETE, and deliberately not reconstructed.**

This is the environment behind the seven-model AUC table in
`docs/writeups/2026-07-23-the-model-still-knows.md`. It is a placeholder rather
than a record, because the record was never taken.

## What is actually known

| Item | Value | Source |
|---|---|---|
| torch | `2.5.1` | pinned in `scripts/runpod/pod-bootstrap.sh` and the `holst/` specs |
| Accelerator | rented RTX 3090 (compass sweep), rented 4090 (unabliterated baselines) | session records |
| Evaluation track | `ops-malware/senbon-track-35axis-clean`, revision unrecorded | `holst/*.toml` |
| senbonzakura commit | unrecorded | — |
| Everything else | **unknown** | — |

transformers, accelerate, datasets, optuna, numpy, safetensors and tokenizers
were resolved to whatever was current on the pod on the day, and no `pip freeze`
was captured. The dataset revision was not pinned either, so even the inputs are
identified by name rather than by content.

## Why this file exists in this state

Writing a plausible-looking version list would be worse than admitting the gap:
it would read as provenance and be a guess. The honest record is that the July
numbers cannot be reproduced exactly, only approximately, and anything published
from them has to say so.

This is also the argument for the discipline going forward, in one page: every
result published from now on gets a completed `measured-YYYY-MM-DD` file written
at the time, because it costs one command then and cannot be recovered later.

## What replaces it

The sprint re-runs these measurements under corrected code with `--seed`
recorded in `abliteration.json`. When those land, a complete measured file goes
beside them and this one stays as the reason the practice exists.
