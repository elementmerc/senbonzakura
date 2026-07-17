# Senbonzakura

**Multi-direction refusal abliteration for transformer language models.**

<p align="center">
  <img src="https://raw.githubusercontent.com/elementmerc/senbonzakura/main/docs/senbonzakura-kageyoshi.png" width="760" alt="Senbonzakura Kageyoshi: a thousand blades in formation">
</p>

Senbonzakura removes the refusal behaviour from an open-weight language model by
finding the *directions* in its activation space that carry "I can't help with
that" and orthogonalising them out of the weights. It builds on the
single-direction method of Arditi et al. and the automated search of Heretic, and
adds the one thing that moved the needle in my own runs: cutting in **several
directions at once**, not just one.

Named for Byakuya Kuchiki's zanpakutō, the sword that scatters into a thousand
blades. Refusal is not one blade. It's many.

## Why multi-direction

The original finding ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)) is that refusal is *mostly* one
direction. Mostly. The last stubborn few percent lives in a small handful of
nearby directions the single-arrow method never sees. Account for a refusal
*subspace* instead of a single vector and the residual refusals fall the rest of
the way, without the model losing its coherence.

The clearest measurement is on Qwen3-4B, over a 290-prompt evaluation scored with
the same ruler:

| Configuration | Hard refusal | Strict (Heretic keyword) | Broken | Coherence (PPL, base 12.97) |
|---|--:|--:|--:|--:|
| Stock, uncut | 7.9% | 63.8% | 0.0% | 12.97 |
| Single direction | 2.1% | 36.6% | 0.0% | 12.97 |
| **Senbonzakura (multi-direction)** | **0.0%** | **20.0%** | **0.0%** | 13.29 |

Single-direction leaves better than a third of the strict count standing.
Multi-direction cuts it to a fifth and drives hard refusal to zero, with no broken
output. Coherence stays essentially level with the base model: single-direction is
identical to stock, multi about two percent higher, both inside run-to-run noise.

That advantage may grow with model size, but two points is not a trend I'd bet on.
On the smaller Qwen3-1.7B, under the same corrected code, single and multi roughly
tie (both clear the strict count to around eight percent); the extra directions buy
little there, and clearly separate only at 4B. Two model sizes at one seed each is
suggestive, not an established scaling law: it could be a real trend (a bigger model
spreading refusal across more directions) or per-model variance, and more sizes and
repeated seeds would be needed to tell.

## How it works

1. **Extract the refusal subspace.** For a few hundred harmful and harmless
   prompts, record the last-token residual at every layer. The difference of
   means (harmful minus harmless), good-orthogonalised, is the primary refusal
   direction; up to K-1 further axes come from a PCA of the harmful residual
   cloud. Together they span the refusal subspace at each layer.
2. **Search (Optuna).** An NSGA-II search over which layers to cut, how strongly,
   how many directions, and per-layer directions versus a single interpolated one,
   maps the whole refusals-versus-coherence frontier. Each trial applies the
   *real* norm-preserving weight bake and restores from a pristine snapshot, so
   the search scores the exact model it will save, with no proxy gap. Coherence is
   protected directly by a KL-divergence term against the original model on
   harmless prompts.
3. **Bake.** For the winning configuration, orthogonalise the refusal span out of
   every residual-writing weight (each attention output projection and each MLP
   down-projection, including the fused expert tensors of mixture-of-experts
   models) so the change is permanent. Save.

The result is a model that has lost the machinery of refusal, not one that has
been told to ignore it.

## Install

```sh
pip install .          # or:  uv pip install .   (torch, transformers, accelerate, datasets, optuna)
senbonzakura --help    # console command; equivalently: python -m senbonzakura --help
```

To score a large model on a low-VRAM card, `pip install ".[quant]"` adds 4-bit
(bitsandbytes) loading **for the scorer**: `python -m senbonzakura.score --load-in-4bit`.
The abliterator itself runs in full precision, because it rewrites weights and 4-bit
tensors can't be orthogonalised in place, so `--load-in-4bit` is a measurement option, not
an abliteration one. A man page is installed to `share/man/man1/senbonzakura.1`.

Shell completion for bash, zsh, and tcsh is generated on demand (the same one time step
`pip`, `gh`, and `poetry` use): `senbonzakura --print-completion bash | sudo tee
/etc/bash_completion.d/senbonzakura`, or the zsh/tcsh equivalent for your shell.

Supported architectures: dense transformers (Llama, Qwen, Mistral, Gemma, Phi and the
like), fused-expert MoE (Qwen3-MoE, Granite-MoE), Mixtral (fused or unfused), OLMoE, and
shared-expert MoE (Qwen2-MoE, DeepSeek-MoE). An unsupported layout fails loudly at load
rather than silently under-ablating.

## Usage

The fast path, if you just want the best result and no knob-twiddling:

```sh
senbonzakura kageyoshi --model <hf-model-or-path> --out <dir> --track <dir> --device cuda
```

`kageyoshi` is the ultimate balanced-effort mode. It detects the
architecture (dense, fused MoE, or expert-list) and parameter count, scales the
search budget accordingly, and switches on every quality lever, so you supply only
the paths. "Balanced" is the point here. It picks the most uncensored config that stays
coherent (the KL ceiling and coherence penalty guard it), not the most aggressive
one. It owns the search knobs; manual `--trials` / `--max-directions` and friends
are ignored in this mode. If a `hedge_ds/` sits in your track directory it folds the
hedging axis in automatically.

For full manual control:

```sh
senbonzakura --model <hf-model-or-path> --out <dir> \
    --track <dir-holding-bad_ds-good_ds-bad_eval_ds> \
    --search pareto --max-directions 6 --trials 200 --device cuda \
    --eval-refusal-final 256 --patience 40
```

Score any model on a held-out evaluation set with the same ruler:

```sh
python -m senbonzakura.score --model <dir> --eval <eval-dataset> --out results.json --label mymodel
```

Measure its coherence on the same loader, the other half of the ruler (the
perplexity the model assigns to one fixed neutral passage, lower is better):

```sh
python -m senbonzakura.coherence --model <dir> --out coherence.json --label mymodel --load-in-4bit
```

`senbonzakura.metrics` is that shared ruler: hard refusal, soft refusal (the "I
can't, but here's a lecture" hedge), broken output, and the Heretic keyword rate
(copied verbatim, so the numbers are directly comparable to Heretic's published
figures).

### Notable flags (`senbonzakura --help` for all)

- `--max-directions K` — size of the refusal subspace to ablate (1 = single-direction).
- `--mlp-off` — attention-only ablation (leave `mlp.down_proj` untouched); tests the
  "attention carries refusal, MLP carries capability" hypothesis.
- `--hedge-ds DIR` — fold a hedged-vs-clean contrast direction into the basis, so the
  search can strip the disclaimer/hedging axis the difference-of-means direction misses.
- `--patience N` — stop the search once the frontier stalls for N trials.
- `--eval-refusal-final N` — re-score the top frontier candidates on a larger eval before
  picking the knee, so the choice isn't overfit to the small search eval.
- `--inspect LAYER STRENGTH` — print real generations before and after a cut.

The search minimises **three** objectives at once: strict non-compliance
(hard refusal plus hedging), the Heretic keyword rate as its own axis, and KL
divergence (coherence). Earlier versions optimised only hard-refusal-versus-KL and
left the keyword/hedging axis to chance.

## Benchmark

A matched comparison against [Heretic](https://github.com/p-e-w/heretic) on
gemma-3-12b-it, the model Heretic reports in its own README, is planned so the two
methods can be read side by side on identical ground: same base model, same
evaluation, same keyword ruler. It will be added to this section when run.

### Reproducibility and status

- **Single run, seed 42.** The Optuna sampler is seeded, but GPU kernels (matmul reductions,
  SVD) are not bit-deterministic, so a re-run can differ by a percent or two. No error bars are
  reported; treat small differences as noise.
- **The [Why multi-direction](#why-multi-direction) table is freshly measured** on Qwen3-4B with
  the current code, and every cell is produced by the shipped tools:
  `python -m senbonzakura.score --load-in-4bit` on the 290-prompt eval for the refusal columns, and
  `python -m senbonzakura.coherence --load-in-4bit` on the fixed neutral passage for the perplexity,
  both through the same 4-bit loader.
- **No head-to-head against Heretic is published yet** (see [Benchmark](#benchmark)). Recent
  correctness fixes to direction extraction, the multi-direction basis, and knee selection moved the
  numbers substantially in Senbonzakura's favour on the keyword axis, so any comparison is run under
  the corrected code, on identical ground, before it goes in.

## What this repository does not contain

By design, this is methods and results, not a loaded weapon:

- **No pre-abliterated model weights.** Run the tool yourself.
- **No harmful prompt sets.** The contrast and evaluation data you supply are your
  own; none ship here.
- **No harmful outputs.**

Abliteration removes safety guardrails wholesale. That is both the point and the
danger. Use it accordingly.

Note on licences: this tool is **AGPL-3.0-or-later** (it embeds a keyword metric copied from
Heretic, which is AGPL). Separately, a model
you abliterate keeps the **base model's** licence and use restrictions: redistributing an
abliterated checkpoint is governed by that upstream licence (Qwen, Llama, Gemma and so on), not by
this repository's.

## Credit

- Arditi, Obeso, et al. [*Refusal in Language Models Is Mediated by a Single
  Direction*](https://arxiv.org/abs/2406.11717) (2024). The direction method this builds on.
- [Heretic](https://github.com/p-e-w/heretic) by p-e-w (Philipp Emanuel Weidmann),
  AGPL-3.0. The automated, KL-guarded search this refines, and the keyword metric
  reported here for comparison (copied verbatim, which is why this project is
  AGPL; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)).
- Maxime Labonne. [*Uncensor any LLM with abliteration*](https://huggingface.co/blog/mlabonne/abliteration). The tutorial that
  popularised the technique.

## Licence

**AGPL-3.0-or-later.** See [LICENSE](LICENSE). Senbonzakura is copyleft because it embeds a
keyword metric copied verbatim from [Heretic](https://github.com/p-e-w/heretic) (AGPL-3.0); see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
