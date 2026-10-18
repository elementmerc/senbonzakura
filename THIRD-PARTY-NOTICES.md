# Third-party notices

Senbonzakura includes or builds on the third-party work below.

## Heretic (credit; no code included)

Heretic (https://github.com/p-e-w/heretic, Copyright (C) 2025-2026 Philipp
Emanuel Weidmann and contributors, AGPL-3.0-or-later) shaped this project's
search, and is credited here for that.

- **No Heretic code is included.** Until 2026-10-18 `src/senbonzakura/metrics.py`
  carried Heretic's keyword marker list, copied verbatim, and a normalisation
  function adapted from it, so that a residual refusal figure could be read on
  Heretic's ruler beside our own. Both were removed on 2026-10-18, together with
  the benchmark harness that ran Heretic, which had reproduced part of Heretic's
  direction code. Every refusal figure the package now reports is measured by its
  own rulers. Committed artefacts from before that date may still carry a
  `heretic` field; `senbonzakura check` reads it where it is present.
- **The search follows Heretic's parameterisation.** An audit on 2026-09-10 put
  the two search spaces side by side and they match parameter for parameter:
  the direction scope, the interpolated direction index, and the per-component
  `max_weight`, `max_weight_position`, `min_weight` and `min_weight_distance`
  profile, including the `-0.25` lower bound that lets the MLP profile fall to
  zero. That is Heretic's design, and it is credited as such. What is ours around
  it: the direction extraction, every measurement, a different sampler (NSGA-II
  rather than TPE), a different objective, different bounds, and the
  direction-count parameter that has no counterpart upstream.

## ggml and llama.cpp

The wheel for a platform carries pinned binaries and a vendored Python tree,
which together are by far the largest third-party component in the artefact:

- `src/senbonzakura/vendor/bin/<platform>/` — `libllama.so`, the `libggml*`
  family and the `llama-quantize` and `llama-imatrix` executables, roughly
  35 MB.
- `src/senbonzakura/vendor/src/gguf-py/gguf/` — llama.cpp's `gguf-py` package.
- `src/senbonzakura/vendor/src/conversion/` — llama.cpp's own `conversion`
  package, fetched unmodified at the pinned tag. Upstream split
  `convert_hf_to_gguf.py` into per architecture modules; that file is now a
  312 line entry point that imports from this package. **The split and the
  layout are upstream's**, and this entry claimed they were ours until
  2026-09-28. We add nothing to the tree and change nothing in it.

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
and `tools/ci/check_wheel.py` requires every release wheel to carry it. Its licence
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

**These terms were accepted on 2026-10-01, reviewed by Daniel Iwugo.** Two of the
direct upstreams declare no licence at all, which the card records and which the
acceptance does not change; the track ships under the strictest term in the chain
so that the gap costs a reader nothing.

The full provenance, including the per-source commit hashes and the reasoning
behind the inferred licence on the harmless side, is in the dataset card at
<https://elementmerc.github.io/senbonzakura/evaluation-track-card>. Where the rows
came from, and which of them cannot be reproduced, is in
<https://elementmerc.github.io/senbonzakura/corpus-provenance>.

## The bundled capability probe

**An eighth thing ships in the wheel and was attributed nowhere, and this is the
third time that sentence has had to be written in this file.** The section above
records it happening to the evaluation track, which records it happening to
THIRD-PARTY-CORPORA.md before that. Found on 2026-09-26 by a reader who was
trying to work out whom to credit for a capability number and could not: the
string `gsm8k` appeared in no notice, no licence file and no user-facing
document, while the tool told them six times per run that *"every bundled corpus
and its terms are listed in THIRD-PARTY-CORPORA.md"*.

`src/senbonzakura/data/capability-gsm8k.jsonl` is what `--capability-eval
bundled` reads, which is the **default**, so any capability figure this tool
reports was measured on it unless the user supplied their own benchmark. At
152 KB it is the largest data file in the package, larger than either prompt
blob.

| | |
|---|---|
| Source | `openai/gsm8k`, the `main` config, `test` split |
| Pinned revision | `740312add88f781978c0658806c59bc2815b9866` |
| Licence | **MIT**, which requires this notice to travel with the distribution |
| What ships | 256 of the split's 1,319 rows, problem and reference answer only |
| Built by | `tools/packaging/build_capability_probe.py`, which verifies the count and the revision |

Citation: Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek, Hilton,
Nakano, Hesse and Schulman, *Training Verifiers to Solve Math Word Problems*
(2021).

It is grade-school arithmetic rather than corpus material, which is why it is
committed to the repository while the prompt corpora are not. That distinction
explains where the file lives; it says nothing about whether the licence applies,
and MIT applies to 256 rows exactly as it applies to 1,319.

## The residual leak metric

**Something was taken and attributed only in a source comment**, which is not
the same as attributed. Found on 2026-10-08, the day the metric gained its first
caller: until then nothing in the tool could produce the figure, so the
gap had never had a user to matter to. The moment `abliterate --leak-report`
exists, the number reaches an artefact, a model card and whoever reads them, and
the credit has to travel with it rather than sitting in a docstring that only a
contributor opens.

`src/senbonzakura/residualleak.py` computes, per residual-stream position, the
mean over probe prompts of the component of the residual along the refusal
direction divided by the residual norm, taken at the last token.

| | |
|---|---|
| Source of the metric | `orcabonsai-27B-uncensored`, which verifies its own runtime ablation this way |
| The published claim it is read from | that the residual component along the direction falls to about 1e-6 of the residual norm across depth |
| Licence | **Apache License 2.0**, which permits the reuse and asks for this notice |
| What was taken | the quantity and its arithmetic. No code, no weights, no data |
| What is ours | the per-position reporting, the basis correction at the output, and the refusal to report a figure when more than one direction was applied |

The basis correction is worth naming as a difference rather than a borrowing.
A learned per-dimension weight in the final normalisation does not preserve
orthogonality to a fixed vector, so a direction that has genuinely gone still
reads as present when measured at the output against the original direction. The
figure this tool reports there is taken in the basis the normalisation maps the
direction into, and the naive reading is deliberately unreachable through the
API.

## The architecture gallery, as the seed list for the architecture corpus

Recorded 2026-10-08, the day `tools/research/fetch_architecture_corpus.py` first ran.
Nothing from this one ships in the wheel, and the entry exists anyway: the section
below says a notice obligation attaches to what we *redistribute*, and a research
input that shapes a published claim deserves credit on its own terms rather than on
the terms of a licence test it was never going to trigger.

The corpus answers a question the local model cache cannot: given about a hundred
published architectures rather than the six that happen to be downloaded here, could
one reader read all of them. The gallery is where the hundred came from.

| | |
|---|---|
| Source | `rasbt/llm-architecture-gallery`, the `models.yml` seed list |
| Pinned revision | `589d213a12188abc4b554a0c32ca735ceda75616` |
| Licence | **Apache License 2.0**, read from the repository's own LICENSE file |
| What was taken | the list of models and, per entry, the Hub repository, the licence, the date, the scale, the declared decoder type, attention description, layer mix and KV-cache figure |
| What was not taken | the gallery author's summaries, highlights, images and benchmark scores. His prose is his |
| What is ours | every fetched `config.json`, tensor inventory and prompt-format file, the normalisation into one shape, the observed-vocabulary counts and the failure map |
| Where it lives | `private/research/architecture-corpus-2026-10-08/`, unshipped and unpublished |

**The licence field was not the authority.** GitHub's reported licence for this
project has been wrong before, so the LICENSE file was fetched and read: 11,357
bytes of genuine Apache text, with the README stating nothing that contradicts it.

**The model metadata underneath is a separate question and is handled separately.**
Each record carries the licence the gallery states for that model, plus the licence
its Hub model card declares, so no record can be reused without its terms arriving
with it. The corpus holds public metadata for our own analysis; it republishes
nobody's model card and no weights were fetched by any route.

## Hanken Grotesk, vendored for the brand cards

**What it is and where it sits.** Two static cuts of Hanken Grotesk, at weights 500 and 700, in
`assets/brand/fonts/`, with the licence beside them in `assets/brand/fonts/OFL.txt`.

**Licence: SIL Open Font License 1.1.** The OFL permits redistribution provided the licence
travels with the font, which is why the text sits in the same directory rather than being linked
to. Copyright 2021 The Hanken Grotesk Project Authors.

**Why it is vendored at all**, since this project vendors as little as it can. The brand cards are
set in the face the website uses, so that a reader who clicks a link preview lands on a page that
looks like the card. The website gets that face from Google Fonts at build time as `woff2`, which
the card generator cannot read and which only exists after a web build has run. Vendoring two
small files means a clean clone can regenerate the cards with no network and no dependency on the
site having been built.

**Instantiated, not synthesised.** The upstream font is variable, with a `wght` axis from 100 to
900. The two cuts here were instantiated from that axis at 500 and 700 rather than produced by
asking a renderer to embolden the regular, which smears outlines rather than selecting a drawn
weight.

**Not in the wheel.** These two files are inputs to the card generator in a source checkout. No
package data names them, so a user who installs from PyPI does not receive them.

## Licence reach, and why this file stops where it does

Recorded 2026-09-22, because the v1.0 gate asks for it and because "these notices look complete"
is not an answer anybody can check.

**The test is redistribution, not use.** A notice obligation attaches to code this project
*ships*. Everything above is shipped: Heretic's marker list inside `metrics.py`, the vendored
`llama.cpp` and `gguf-py` trees, the bundled corpora, the bundled evaluation track. They travel
inside the wheel, so their terms travel with them.

**The declared runtime dependencies are not shipped.** `torch`, `transformers`, `accelerate`,
`optuna`, `pyarrow`, `huggingface_hub`, `gguf`, `sentencepiece` and `datasets` are resolved by pip
from their own publishers. We distribute a name and a version range; the user receives the package
from PyPI under its own licence, and nothing of theirs is inside our artefact. So no notice is
owed and none is given, and that is a decision rather than an oversight.

**Two checks keep it honest.** `tools/ci/check_wheel.py` asserts every file in `license-files`
reaches `dist-info/licenses/` in the built artefact, and `tests/test_platform_wheel.py` ties that
list to the one `pyproject.toml` declares, so neither can grow without the other.

**What would change this.** Vendoring any dependency, bundling a model, or shipping a dataset we
did not build. Each moves a package from the second list to the first, and each needs this file
extended in the same change.

**What is not settled here.** Whether AGPL section 13, the network clause, reaches somebody
serving a model this tool produced. See `private/decisions.md` Q-39: the documentation makes no
claim either way, deliberately.
