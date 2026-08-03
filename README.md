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
| Search pinned to one direction | 2.1% | 36.6% | 0.0% | 12.97 |
| **Search allowed up to three** | **0.0%** | **20.0%** | **0.0%** | 13.29 |

**Both rows applied one direction per layer.** The bottom row is named for the budget the
search was given, not for the number of directions it used, and until 2026-08-03 those were
the same words for a reason that turned out to be a bug. The difference between the two rows
is real and is a difference between two searches. Read the caveats below before quoting it.

> ### Read this table with the caveats attached, not 300 lines below it
>
> **These numbers were measured on 2026-07-14 and two scoring bugs have been fixed since.**
> They are kept here because deleting a published claim is not the same as correcting one,
> and because the qualitative shape (multi below single below stock) is what the rest of this
> section argues. Do not treat the figures as current.
>
> - **The refusal scanner read only the first 240 characters** of a reply until 2026-07-21
>   (`71cc119`). Fifty-one of fifty-six refusal markers land past that point, so refusals were
>   undercounted.
> - **The prompt renderer had drifted into three copies** until 2026-07-30 (`d5a16e0`), and the
>   scorer's copy left thinking enabled. Qwen3-4B is a thinking model, so the replies being
>   scored were not the replies the search had selected on.
> - **This is a 290-prompt evaluation that is not held out.** Some of these prompts are the
>   ones the winning configuration was chosen on, which flatters any tuned method, this one
>   included.
> - **One seed, no intervals.** A single run is a sample.
>
> **Why it has not been re-measured:** Qwen3-4B does not fit the 6 GB card this project is
> built around, so a corrected run needs hardware we do not own. That is the honest reason,
> not an oversight.
>
> ### The multi-direction claim has not been tested in a way that isolates it
>
> Added 2026-08-03, and it is the more important caveat of the two.
>
> A comparison was run on Qwen3-1.7B specifically to test whether removing several directions
> beats removing one: two arms, five seeds each, held out, everything identical but the
> direction budget. **Both arms turned out to have removed exactly one direction per layer.**
> The extra capacity existed in the data structure and was never filled, because no second
> axis cleared the tool's own refusal-separation threshold at any of the model's 29 layers, so
> the surgery in the two arms was identical and the comparison measured something else.
>
> **Corrected the same day, and it is worse than the paragraph above said.** The reason no
> second axis ever cleared the threshold is not a fact about Qwen3-1.7B. The check that decides
> whether a candidate direction carries refusal compares the average harmful reply against the
> average harmless one, measured along a direction that is built to sit at right angles to both
> of those averages. The difference it looks for is zero every time, by construction. **No
> direction could pass that check, on any model, at any setting.**
>
> Measured across two model families and three prompt sets: 13,970 candidate directions, every
> one rejected, none close. So **every run this project made before 2026-08-03 applied exactly
> one direction, whatever it was asked for**, and the bottom row of the table above is a second
> search configuration rather than a second direction.
>
> ### What has been fixed, and what has not
>
> The extractor was rewritten the same day. Candidate directions now come from clustering the
> harmful prompts and taking each cluster's own average against the harmless average, so a
> candidate separates the two groups by construction rather than being built unable to. On the
> same four probes it now finds and applies **up to eight directions per layer** where it
> previously found one.
>
> **That means the feature does something. It does not yet mean the claim is true.** Two things
> are still unshown, and the README will not say otherwise until they are measured:
>
> - **Whether the extra directions carry refusal rather than topic.** The threshold that was
>   supposed to answer this now rejects nothing at all: all 678 candidates scored between 0.90
>   and 9.02 against a bar of 0.5. It has swapped failure modes, not started working. A cluster
>   of harmful prompts about one subject separates from harmless prompts partly *because* of the
>   subject, and cutting that removes capability rather than refusal.
> - **Whether removing several directions beats removing one.** The comparison that was meant to
>   show this is withdrawn, and re-running it needs the above settled first.
>
> So: **nobody should believe the multi-direction claim on this project's evidence**, us
> included. What changed today is that the question can now be asked at all.

Read the next three paragraphs against the caveat above: "multi-direction" names the
configuration those runs requested, not the number of directions they applied, which was one.

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
   direction. Further directions come from grouping the harmful prompts and taking
   each group's own average against the harmless average: a model refuses a weapons
   request differently from a self-harm one, and a single average over both is the
   average of two things that are not the same thing. Each is then orthogonalised
   against the directions already kept, and the best-separating are taken up to K.

   Earlier versions took those further directions from a PCA of the harmful residual
   cloud, which asks how the harmful prompts VARY rather than what separates them from
   harmless ones. That method could never keep a single extra direction, for a reason
   in the geometry rather than the data; see the caveat under the table above.
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

Shell completion for bash, zsh, and tcsh needs `pip install ".[completion]"`, then is
generated on demand (the same one time step `pip`, `gh`, and `poetry` use):
`senbonzakura --print-completion bash | sudo tee /etc/bash_completion.d/senbonzakura`, or the
zsh/tcsh equivalent for your shell. Without that extra the flag is simply absent; nothing
else changes.

### Running the tests

```sh
python -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU build, no GPU needed
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                                                     # add --cov for coverage
```

The suite runs entirely on CPU against small hand-built fixtures, so it needs no model
download and no GPU.

If you plan to commit, wire the local gates once per clone:

```sh
bash tools/install-local-hooks.sh
```

That connects `tools/check_prompt_artefacts.py` to your pre-commit path. The tool keeps
per-prompt margins and generations by default, because every scoring bug in this project's
history was invisible in the percentages and obvious in the text, and those rows hold
harmful prompts and the replies a model gave to them. The gate refuses any staged JSON or
JSONL carrying a `prompt` or `generation` field. CI runs the same check over the tree, but
by the time CI sees it the commit exists.

Supported architectures: dense transformers (Llama, Qwen, Mistral, Gemma, Phi and the
like), fused-expert MoE (Qwen3-MoE, Granite-MoE), Mixtral (fused or unfused), OLMoE, and
shared-expert MoE (Qwen2-MoE, DeepSeek-MoE). An unsupported layout fails loudly at load
rather than silently under-ablating.

## The track: what the tool reads

A **track** is a directory of three datasets. Every command takes `--track <dir>` and
looks for these names:

| Directory | What it holds | What reads it |
|---|---|---|
| `bad_ds` | Harmful prompts | Direction extraction fits on these |
| `bad_eval_ds` | Harmful prompts | The search scores trials, and the compass measures |
| `good_ds` | Harmless prompts | Fitting, the KL reference, and the compass's harmless arm |

Each is a `datasets.save_to_disk` directory with one column, `text`:

```python
from datasets import Dataset
Dataset.from_dict({"text": ["first prompt", "second prompt"]}).save_to_disk("mytrack/bad_ds")
```

### Build one, and have it checked

Doing it by hand is how the mistake happens, so there is a builder:

```sh
python -m senbonzakura.track \
    --harmful harmful.txt --harmless harmless.txt \
    --out mytrack --fit 256 --search 128
```

One prompt per line. It deduplicates on a case- and punctuation-insensitive key,
splits each side into three parts by index, and **writes nothing at all unless every
check passes**:

- `fit` — the directions are extracted from these
- `search` — the search scores trials on these
- `measure` — the published number comes from these, and nothing else touches them

The checks are the point. A track is refused if anything in `measure` also appears in
`fit` or `search`, if a prompt is labelled both harmful and harmless, if a partition is
empty, or if the two sides differ in size by more than 10%. It reports counts only and
never prints a prompt, so its output is safe to paste anywhere.

One of those checks looks past the text. Corpora are often built by crossing a handful
of phrasings with a list of requests, so the same question appears several times as
several different strings; comparing whole prompts then reports a clean split while the
eval set is full of training questions in other clothes. The builder works out the
phrasings from your own corpus, strips them, and keeps every wording of one request in
the same part. On the corpus this project runs on, that is the difference between
"zero overlap" and 60% of the eval set.

It finishes by telling you the flags that put the compass on the held-out rows:

```
TRACK_BUILT mytrack  harmful {'fit': 256, 'search': 128, 'measure': 616}
  measure with: --skip-harmful 128 --skip-harmless 384 --n 616
```

### `track.json`: where the boundaries are recorded

The three partitions are stored end to end inside each dataset, so `bad_ds` is the fit rows
followed by nothing else, while `bad_eval_ds` is the search rows followed by the measure
rows. **Nothing about a directory of prompts says where one partition ends and the next
begins**, so the builder writes it down:

```json
{
  "schema": "senbonzakura-track/1",
  "sources": { "harmful": "harmful.txt", "harmless": "harmless.txt" },
  "counts": {
    "harmful":  { "fit": 256, "search": 128, "measure": 616 },
    "harmless": { "fit": 256, "search": 128, "measure": 616 }
  },
  "skip_harmful": 128,
  "skip_harmless": 384,
  "n_harmful": 616,
  "n_harmless": 616
}
```

| Field | What it is |
|---|---|
| `schema` | The format version. A manifest declaring a version this build does not know is **refused**, rather than read with the fields it happens to recognise. A manifest with no `schema` at all is read as version 1, because hand-written ones predate the field. |
| `sources` | Where the prompts came from, so a track can be traced to its input. |
| `counts` | Rows in each partition, per side. The record the boundary checks are made against. |
| `skip_harmful`, `skip_harmless` | What to pass to `--skip-harmful` and `--skip-harmless` to land on the measure rows. |
| `n_harmful`, `n_harmless` | What to pass to `--n`. |

**This file is load-bearing, not documentation.** Three things read it:

- **The audit** (`--audit`) re-checks an existing track: leakage, empty partitions, whether
  the rows on disk still match the counts recorded here. A `track.json` that disagrees with
  the rows beside it is refused, because every slice taken from it would silently read the
  wrong rows.
- **The boundary check** refuses a run whose flags would cross a partition. Asking the search
  to score more prompts than the `search` partition holds means selecting a configuration on
  the rows the published number comes from, and the tool stops rather than doing it.
- **The measure flags above** are read from it rather than remembered.

A track with no `track.json` still runs. It simply gets none of the above, and every
boundary becomes something you have to keep right by hand.

### The optional fourth and fifth directories

| Directory | What it holds | What reads it |
|---|---|---|
| `good_matched_ds` | Harmless prompts matched to the harmful ones by topic | `--harmless-matched`, the compass's construct-validity control |
| `hedge_ds` | Hedged compliances, neither refusal nor clean answer | The hedging direction, when one exists |

Both are optional and neither is built by `senbonzakura.track` today. `good_matched_ds`
answers "is the compass reading refusal, or is it reading topic?" by holding the subject
matter still. `hedge_ds` has no public dataset behind it anywhere, which is why the
single-versus-multi comparison is two arms rather than three.

Those numbers are also written to `mytrack/track.json`, so a reader can check the split
a year later rather than take it on trust. To re-check a track you already have,
including one assembled by hand:

```sh
python -m senbonzakura.track --out mytrack --audit
```

If you have a `label<TAB>prompt` file for the corpus, pass `--labels` to the audit too.
It adds the one check that cannot be run without it: whether every category present in
the track actually reaches the `measure` part. A category that does not is the mirror
image of leakage, a number narrower than it looks rather than better than it should be.

Rebuilding backs the old track up once, to `<track>.pre-build`, and never overwrites
that backup.

`track.json` is also read at the start of a run, and a run whose flags would reach past a
recorded boundary stops before the model loads. Each dataset is read as its first N rows,
and the search rows come immediately before the measured ones, so asking for a larger
selection set than the track keeps aside would quietly select on the rows the published
number comes from. A track without a `track.json`, one assembled by hand, runs as before:
its boundaries are unknown, so there is nothing to check against.

### A toy track to try it on

`examples/toy-track/` is committed and runnable straight from a clone. Its prompts are
synthetic placeholders rather than real harmful text, because it exists to show the
plumbing works, not to measure anything.

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

## Checking whether the directions are refusal directions

A method that finds several directions has not thereby shown that the extra ones remove
refusal. They might be removing subject matter. A set of harmful prompts about one topic
differs from harmless prompts partly because of the topic, and cutting that costs the
model knowledge rather than caution.

`senbonzakura validate` is the check. It is separate from the abliterator on purpose:
the tool that produces a direction set should not be the only thing that grades it.

```sh
senbonzakura validate --model <hf-id> --track mytrack --experiment all --out result.json
```

Three questions, and each has a control that can fail:

- **Do the directions work on prompts they were never fitted on?** The harmful prompts are
  grouped, one group is held out, directions are fitted on the rest, and the held-out group
  is scored. A direction tied to a subject cannot separate a subject it never saw. The
  score is reported next to a **random direction** measured the same way, because a number
  without a floor beside it cannot be read.
- **Do the fitted extra directions beat random ones?** The same run repeats with the extra
  directions replaced by random ones. If cutting random directions does as well, the
  fitted ones were not carrying anything.
- **Is removing several better than removing one?** Direction count is swept against
  ablation strength, and the results are compared **at matched refusal removal**. This
  matters more than it sounds: cutting harder always costs more coherence, so comparing
  coherence between runs that removed different amounts of refusal compares nothing.

The run reports an unreadable grid as unreadable. If every setting lands on the same
refusal rate, there is no room for direction count to show an effect, and it says so
instead of printing a table that looks like data.

**We publish our own results from this, including when they are unflattering.** See the
caveat under the table above.

## Benchmark

A matched comparison against [Heretic](https://github.com/p-e-w/heretic) on
gemma-3-12b-it, the model Heretic reports in its own README, is planned so the two
methods can be read side by side on identical ground: same base model, same
evaluation, same keyword ruler. It will be added to this section when run.

### Reproducibility and status

- **The [Why multi-direction](#why-multi-direction) table is NOT current.** It was measured on
  2026-07-14, before two scoring fixes, on an evaluation that is not held out. The caveats are
  printed beside the table itself rather than here, because a qualification three hundred lines
  from the thing it qualifies is not a qualification. Every cell was produced by the shipped
  tools (`python -m senbonzakura.score --load-in-4bit` for the refusal columns,
  `python -m senbonzakura.coherence --load-in-4bit` for perplexity), which is what makes it
  reproducible; it is comparability with today's code that it lacks.
- **Every figure this project measured before 2026-07-30 is affected.** Two bugs: the refusal
  scanner read a 240-character window (`71cc119`, 2026-07-21), and the prompt renderer existed
  in three copies that had drifted, so a configuration selected under one prompt format was
  reported under another (`d5a16e0`, 2026-07-30). Numbers either side of those dates are not
  comparable, and this project would rather say so than quietly restate them.
- **Determinism, measured rather than assumed.** On a fixed batch size the forward-only paths
  are reproducible: each compass command was run three times on the same card at batch 16 and
  produced AUCs identical to four decimal places. That says nothing about generation, which is
  sampled, and nothing about a different batch size, which changes reduction order.
- **The model-size ceiling.** Every model this tool has been run on is **under 3B, and six of
  the seven are under 2B**; gemma-2-2b-it is the largest at 2.61B and carries the strongest
  result. Nothing here is evidence about how the method behaves at 7B, 30B or beyond. The
  streaming work exists to make those sizes reachable on hardware we own.
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests need a test, a commit message that says
why, and one line added to [CONTRIBUTORS.md](CONTRIBUTORS.md) agreeing to the
[CLA](CLA.md). The project is AGPL and stays AGPL; you keep the copyright in what you write.

## Licence

**AGPL-3.0-or-later.** See [LICENSE](LICENSE). Senbonzakura is copyleft because it embeds a
keyword metric copied verbatim from [Heretic](https://github.com/p-e-w/heretic) (AGPL-3.0); see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

**Senbonzakura is a modified work based in part on Heretic, and it is not Heretic.** Modified by
Daniel Iwugo; first included 2026-07-14, most recently modified 2026-07-29. Only the keyword rate
is shared code and it is kept byte-identical, so only that number is a like-for-like comparison
with Heretic; everything else here is measured by our own instrument. The full statement, and what
was and was not changed, is in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
