# Third-party notices

Senbonzakura includes third-party code, which determines the licence of the
whole project.

## Heretic

`src/senbonzakura/metrics.py` contains a keyword-marker list (`HERETIC_MARKERS`)
and a normalisation function (`_heretic_norm`) copied verbatim from Heretic:

- Project: https://github.com/p-e-w/heretic
- Copyright (C) 2025-2026 Philipp Emanuel Weidmann and contributors
- Licence: AGPL-3.0-or-later

They are reproduced so that Senbonzakura's residual-refusal figures stay directly
comparable to Heretic's own keyword metric. Because this AGPL-licensed code is
included and distributed, Senbonzakura as a whole is distributed under
AGPL-3.0-or-later (see [LICENSE](LICENSE)).

### Statement of modification (AGPL-3.0 section 5(a))

**Senbonzakura is a modified work based in part on Heretic, and it is not
Heretic.** Section 5(a) of the licence requires a work like this one to say so
prominently and to give a date, so that nobody mistakes our behaviour for
upstream's.

- **Modified by:** Daniel Iwugo.
- **First included:** 2026-07-14, in the initial package.
- **Relicensed to AGPL-3.0-or-later for this inclusion:** 2026-07-17.
- **Most recent modification to the file carrying it:** 2026-09-08 (this is
  the date the README's credit section must agree with; the two drifted by six
  weeks, which is the one field section 5(a) is actually about).

What was and was not changed, because the distinction is the whole point of the
notice:

- `HERETIC_MARKERS` and `_heretic_norm` are **byte-identical to upstream** and are
  deliberately kept that way. If they ever drift, the comparability that is the
  only reason for copying them is gone, and this notice becomes wrong.
- **Everything around them is ours**, including a stricter `is_refusal`, a soft
  refusal detector, and the surrounding module. Senbonzakura reports the Heretic
  keyword rate *alongside* its own metric rather than as its metric.
- **The search reuses Heretic's PARAMETERISATION, and the notice used to deny
  it.** It said the search was independent work that did not reimplement
  Heretic's; an audit on 2026-09-10 put the two search spaces side by side and
  they match parameter for parameter: `direction_scope` against `dir_mode`,
  `direction_index`, and the per-component `max_weight`, `max_weight_position`,
  `min_weight` and `min_weight_distance` profile, including the `-0.25` lower
  bound whose purpose is to let the MLP profile fall to zero and leave that
  component untouched. That bound and its reason are a specific numeric design
  choice of Heretic's, reproduced exactly. Senbonzakura's own CLI help already
  conceded the point, describing the per-component split as "Heretic-style",
  while this file, which is the one that is legally load-bearing, denied it.

  What is ours around that parameterisation: the direction extraction, every
  measurement, a different sampler (NSGA-II rather than TPE), a different
  objective, different bounds, and the direction-count parameter that has no
  counterpart upstream. Both works are AGPL-3.0-or-later, so nothing about the
  licensing turns on this; the claim was simply false and is withdrawn.
- **One further piece of Heretic IS copied, in the benchmark harness.**
  `head-to-head/best_of_n_heretic.py` reproduces Heretic's direction block,
  including its projected-abliteration step, and imports `heretic.config`,
  `heretic.model` and `heretic.utils` directly. It exists so that the
  equal-budget selection pass this project applies to its own runs can be
  applied to Heretic's in the same way, from outside Heretic's code. That file
  now carries the upstream copyright line and a dated statement of modification
  directly above the copied functions; until 2026-09-10 it stated only functional
  provenance, which is a weaker thing than authorship, and this notice described
  that as the file saying so about itself. Both works are AGPL-3.0-or-later, so
  the licensing is unaffected; the notice was wrong, and it was found by an audit
  reading the tree against this file rather than reading this file alone.

A reader comparing a Senbonzakura number to a Heretic number should know that
only the keyword rate is a like-for-like comparison, and that every other figure
is measured by our own instrument.

## ggml and llama.cpp

The wheel for a platform carries pinned binaries and a vendored Python tree,
which together are by far the largest third-party component in the artefact:

- `src/senbonzakura/vendor/bin/<platform>/` — `libllama.so`, the `libggml*`
  family and the `llama-quantize` and `llama-imatrix` executables, roughly
  35 MB.
- `src/senbonzakura/vendor/src/gguf-py/gguf/` — llama.cpp's `gguf-py` package.
- `src/senbonzakura/vendor/src/conversion/` — llama.cpp's
  `convert_hf_to_gguf.py`, **restructured by us** from one file into per
  architecture modules so that a broken architecture can be detected on import
  rather than at use. The behaviour is upstream's; the arrangement is ours.

- Project: https://github.com/ggml-org/llama.cpp
- Copyright (c) 2023-2026 The ggml authors
- Licence: MIT

The MIT text ships beside both trees, at `vendor/src/LICENSE` and at
`vendor/bin/<platform>/LICENSE`. The second of those was missing until
2026-09-08: the source tree carried the notice and the object code beside it did
not, which is what MIT's "all copies or substantial portions" is about. The
vendoring tool now places it, so a re-vendor cannot drop it again.

Which release is pinned, and its hashes, are in
`src/senbonzakura/vendor/pins.json`, and `senbonzakura doctor` reports them.

## Bundled corpora

Six prompt sets ship inside the wheel so the tool runs offline. Each is
redistributed under its own licence, several of which require attribution.
Those notices are in [THIRD-PARTY-CORPORA.md](THIRD-PARTY-CORPORA.md), which
ships inside the package, and the tool prints the relevant one the first time a
bundled corpus is loaded in a process.

## The bundled evaluation track

**A seventh thing ships in the wheel and was attributed nowhere inside it.**
`src/senbonzakura/data/default-track.bin` is the corpus behind `--track default`,
and `tools/check_wheel.py` requires every release wheel to carry it. Its licence
chain was documented only in `docs/evaluation-track-card.md`, and `docs/` is in
neither `package-data` nor `license-files`, so it does not ship. The runtime
notice printed an attribution and then pointed the reader at a file their install
did not contain. Found on 2026-09-10; it is the same defect shape already fixed
once for THIRD-PARTY-CORPORA.md, recurring one level down.

The track is distributed under **CC BY-NC 4.0**, the most restrictive licence in
its chain. Attribution is required and **commercial use is not permitted**; a
commercial user should supply their own corpus with `--track` rather than using
the bundled one.

| Source | What it contributes | Its licence |
|---|---|---|
| `Bahushruth/abliteration-harmful-enriched` | most of the harmful side | Apache-2.0 (see [APACHE-2.0.txt](APACHE-2.0.txt)) |
| AdvBench, Zou et al. 2023 | the harmful root of the above | MIT |
| `mlabonne/harmless_alpaca` | the harmless side | declares nothing |
| `tatsu-lab/alpaca`, the probable root of the above | the harmless side, indirectly | CC BY-NC 4.0 |

Citations:

- AdvBench: Zou, Wang, Carlini, Nasr, Kolter and Fredrikson, *Universal and
  Transferable Adversarial Attacks on Aligned Language Models* (2023).
- Alpaca: Taori, Gulrajani, Zhang, Guestrin, Liang and Hashimoto, *Stanford
  Alpaca: An Instruction-following LLaMA Model* (2023), CC BY-NC 4.0.

Licence texts: <https://creativecommons.org/licenses/by-nc/4.0/legalcode> for
CC BY-NC 4.0; Apache-2.0 is reproduced in full beside this file.

The full provenance, including the per-source commit hashes and the reasoning
behind the inferred licence on the harmless side, is in the dataset card at
<https://elementmerc.github.io/senbonzakura/evaluation-track-card>.
