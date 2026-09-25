# Architecture

A high-level map of how Senbonzakura works and where each piece lives. For the
why behind the method, see the README; this doc is for someone
who wants to read or extend the code.

## What abliteration is here

An aligned model carries a "refusal" behaviour: for prompts it judges harmful, it
steers its own activations toward an "I can't help with that" response. Arditi et
al. showed that steering is mostly a single *direction* in the model's activation
space. Find that direction and subtract it out of the weights, and the model stops
refusing without being retrained.

Senbonzakura's one difference from the single-direction method is that it can treat
refusal as a small *subspace* rather than one direction, finding several at once and
orthogonalising all of them out. **Whether that helps is a question this project has
now answered, against itself:** on Qwen3-1.7B, two directions cost 1.5 to 1.9 times the
coherence of one at the same refusal rate and bought nothing. The capability stays
because the question is open on architectures we cannot yet measure, but the default
posture is that one direction is what works. See
[what is and is not established](/guide/what-we-know).

Everything below is in service of finding directions and cutting them without
breaking the model, whether that is one of them or several.

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

There are 56 modules under `src/senbonzakura/`. This lists the ones somebody reading or extending
the code will meet, grouped by what they are for. It listed six until 2026-09-25, none of them
from the measurement suite, which is the half of the project the README calls the interesting one.

**The edit**

| Module | Responsibility |
|---|---|
| `cli.py` | The command surface and the orchestration of the pipeline above. Architecture detection (dense, fused MoE, expert-list), and the extract, search and bake flow. It is the largest module here by a wide margin. |
| `parser.py` | **Every flag.** The line above used to say `cli.py` held them; it does not, and the split is deliberate: `parser.py` imports no deep-learning stack, so `--help` does not load torch. |
| `resources.py` | The resource governor: VRAM-pressure-adaptive batch sizing, accelerate offload for models larger than VRAM, and the background mode that yields the GPU to a foreground game. |
| `crashsafe.py` | Reversible snapshots, resume markers, and the provenance block every result carries. |
| `streaming.py` | The shard-at-a-time path for models larger than memory. |

**The instruments, which is what the release is about**

| Module | Responsibility |
|---|---|
| `score.py` | Refusal scoring against an eval set: hard refusal, the strict (keyword) rate, and broken-output detection. |
| `margin.py` | The compass. Does the model still recognise harm, scored by logit margin rather than by counting verdicts. |
| `capability.py` | What the edit cost, on five code-graded tasks. |
| `coherence.py` | The neutral-passage perplexity probe, the check that a cut model still reads coherently. |
| `drift.py` | Distributional cost as KL divergence against the unedited model. |
| `measure.py` | Runs all five of the above into one table. It measures nothing itself; each stage is the command that owns that number. |
| `metrics.py` | The keyword metric itself (the list copied from Heretic, hence the AGPL licence) and the normalisation around it. |
| `lengthsweep.py` | The token-budget sweep behind the "a refusal cut off before it is emitted counts as compliance" caveat. |

**The evidence**

| Module | Responsibility |
|---|---|
| `track.py`, `trackbuild.py`, `trackio.py` | Building, splitting and reading the evaluation track, and the contamination audit. |
| `bundled.py`, `corpora.py` | The packed track and corpora that ship inside the wheel, and the key that unwraps them. |
| `baseline.py` | Turning a measurement into a baseline a later run can be gated against. |
| `validate.py` | The six experiments behind the multi-direction claim. |
| `modelcard.py` | The model card for a finished run. |

**Comparison, conversion and the environment**

| Module | Responsibility |
|---|---|
| `headtohead.py`, `headtohead_report.py` | Running this tool and another one on the same model under one ruler, and reporting it. |
| `convert.py`, `quantise.py`, `gguf_io.py` | GGUF conversion, quantisation, and reading a GGUF header without loading the model. |
| `doctor.py`, `envsetup.py` | Whether this install can do what it claims, and putting the right torch on the machine. |
| `interactive.py` | The guided mode for people who would rather not read the flag list. |
| `fetch.py`, `dataset.py` | Resumable model fetching, and reading the shapes of dataset a track can come from. |
| `__main__.py` | `python -m senbonzakura` entry point. |

The separate `senbonzakura-check` distribution lives under `checker/` and imports none of this;
the dependency runs big to small and only that way.

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

The keyword metric in `metrics.py` takes its marker list verbatim from Heretic
and adapts that project's normalisation, and Heretic is AGPL-3.0. That is why the whole project is AGPL-3.0-or-later rather than a
permissive licence. See `THIRD-PARTY-NOTICES.md` for the derivation.
