# Architecture

A high-level map of how Senbonzakura works and where each piece lives. For the
why behind multi-direction abliteration, see the README; this doc is for someone
who wants to read or extend the code.

## What abliteration is here

An aligned model carries a "refusal" behaviour: for prompts it judges harmful, it
steers its own activations toward an "I can't help with that" response. Arditi et
al. showed that steering is mostly a single *direction* in the model's activation
space. Find that direction and subtract it out of the weights, and the model stops
refusing without being retrained.

Senbonzakura's one difference from the single-direction method: refusal is not one
direction but a small *subspace*. It finds several refusal directions at once and
orthogonalises all of them out, which clears the stubborn residual the single
arrow leaves behind. Everything below is in service of finding the right subspace
and cutting it without breaking the model.

## The pipeline

```
  load + detect architecture
          |
          v
  capture activations   <- run harmful and harmless prompts, record hidden states
          |
          v
  compute refusal directions   <- difference of means, per layer, as a subspace
          |
          v
  search   <- NSGA-II over (refusal, keyword rate, KL); pick the knee
          |
          v
  bake   <- orthogonalise the chosen directions out of the weights, save
          |
          v
  score   <- refusal metrics + coherence on a held-out eval
```

Each stage is bounded and measured. The search stage is what makes the result
balanced rather than maximally aggressive: it trades refusal removal against KL
divergence (how far the model drifts from the original) and a coherence penalty,
and picks the knee of that trade-off, not the extreme.

## Module map

| Module | Responsibility |
|---|---|
| `cli.py` | The command surface and the orchestration of the pipeline above. Architecture detection (dense, fused MoE, expert-list), the extract-search-bake flow, `kageyoshi` one-shot preset, and all flags. |
| `score.py` | Refusal scoring against an eval set: hard refusal, the strict (keyword) rate, and broken-output detection. |
| `metrics.py` | The keyword metric itself (the list copied from Heretic, hence the AGPL licence) and the normalisation around it. |
| `coherence.py` | The neutral-passage perplexity probe, the check that a cut model still reads coherently. Reuses the shared model loader so its flags match the scorer exactly. |
| `resources.py` | The resource governor: VRAM-pressure-adaptive batch sizing, accelerate offload for models larger than VRAM, and the background mode that yields the GPU to a foreground game. |
| `__main__.py` | `python -m senbonzakura` entry point. |

## The search objectives

The search optimises three axes at once (three-objective NSGA-II), because
optimising refusal alone produces an incoherent or over-drifted model:

- **Strict non-compliance** — hard refusal plus hedging, the thing we want to drive
  down.
- **Keyword rate** — the Heretic keyword metric, kept as its own axis so the search
  targets it directly rather than hoping it falls out of the refusal axis.
- **KL divergence** — how far the cut model drifts from the original on harmless
  prompts, the guard against wrecking general behaviour.

The knee of the resulting Pareto front is the shipped config: the most uncensored
point that still holds coherence and low drift.

## Extension points

- **New model families.** Architecture detection lives in `cli.py`; adding a family
  means teaching the detector its layer and MLP structure. The surgery itself
  generalises once the structure is known.
- **New objectives.** The search takes a list of objectives; a new axis (say, a
  toxicity guard) slots into the NSGA-II front alongside the existing three.
- **New metrics.** `score.py` and `metrics.py` are separable from the search; a new
  refusal ruler can be added without touching the pipeline.

## Licence note

The keyword metric in `metrics.py` is copied verbatim from Heretic, which is
AGPL-3.0. That is why the whole project is AGPL-3.0-or-later rather than a
permissive licence. See `THIRD-PARTY-NOTICES.md` for the derivation.
