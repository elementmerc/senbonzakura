# The Senbonzakura evaluation track: dataset card

This card covers the evaluation track that every published number in this project was measured on. It answers three questions: what is in the track, where each row came from, and what you may do with it.

**This card covers what is in the track and what is wrong with it.** For where each row came from, which revision it was fetched at, and how the partitions are packed into the files, see [where the evaluation corpus came from](./corpus-provenance.md).

**Nothing here is generated content.** The track is a partitioned collection of prompts drawn from existing public datasets. It holds no model outputs, no completions and no answers to any harmful request. It is a measuring instrument, not a corpus of harm.

**The rows are published as a gated dataset, not inside this repository:** [`ops-malware/senbonzakura-dataset`](https://huggingface.co/datasets/ops-malware/senbonzakura-dataset). The track is distributed under **CC BY-NC 4.0**, the most restrictive licence in its chain, with attribution to every source named below. The gate makes taking the rows a deliberate act. A reader accepts the acceptable-use terms first. The gate collects nothing beyond what HuggingFace needs to run it.

**A released wheel is different, and you should know it.** The same rows ship inside the released package as `--track default`. Installing a release puts roughly 6,200 harmful prompts in your site-packages, with no gate and no terms accepted. They are obfuscated, not protected: the key ships beside them. The HuggingFace gate does not apply to them.

**Which channel carries what.** Read this before running `pip install`:

| Channel | Carries the 6,200 rows? |
|---|---|
| `pip install senbonzakura` (PyPI, 0.4.0) | **Yes**, as `--track default`, with no gate |
| A build from a clone, or `pip install git+...` | No. The two `.bin` blobs under `src/senbonzakura/data/` are generated rather than committed, so a source build carries neither |
| The HuggingFace dataset | Yes, gated, under CC BY-NC 4.0 |

::: warning This table was the other way round until 2026-09-26
It said "neither obtainable install carries the rows". That was true while PyPI served only 0.3.0, from July 2026. 0.4.0 is on PyPI and its wheel carries both blobs, so the sentence became a reassurance and pointed at the wrong artefact. If you installed before reading this, `senbonzakura doctor` lists what your install holds.
:::

`senbonzakura track build` rebuilds a pool of the same shape from the same upstreams, for anyone who would rather fetch the sources themselves.

## What it is

A three-way split of harmful and harmless prompts. The track fits refusal directions, searches for a configuration, and measures the result. The partitions are disjoint by recorded row index, not by convention.

| Partition | Harmful rows | Harmless rows | Used for |
|---|--:|--:|---|
| fit | 259 | 257 | extracting refusal directions |
| search | 132 | 128 | selecting a configuration |
| measure | 4,504 | 4,597 | the published numbers |

The boundaries live in `track.json`. Every run reads them at start, and a run whose flags would reach past one is refused. Audit a track with:

```sh
python -m senbonzakura.track --out mytrack --audit
```

The split is by request, not by string. The same question wears many templates. Comparing whole prompts once reported a clean split while 60% of the measured set was training questions in other clothes.

## Where the rows came from, and what each licence says

**Two of the three direct sources declare no licence**, and the probable root of the harmless side is non-commercial. That combination sets the terms this track is distributed under.

Every position below was read from the HuggingFace dataset API on **2026-08-16**. The revision is recorded so a later reader can tell a changed card from a wrong one.

| Source | Side | Declared licence | Revision read |
|---|---|---|---|
| `Bahushruth/abliteration-harmful-enriched` | most of the harmful side | **apache-2.0**, in `cardData` and tagged | `f29c0b77` |
| `mlabonne/harmful_behaviors`, reached through the above | 176 harmful rows, measured | **None.** No `license` field, no `cardData` licence, no tag | `01cead01` |
| `mlabonne/harmless_alpaca` | the harmless side | **None.** No `license` field, no `cardData` licence, no tag | `02c6a92c` |
| `tatsu-lab/alpaca`, the probable root of the above | the harmless side, indirectly | **cc-by-nc-4.0**, declared and tagged | `dce01c9b` |
| The harmless top-ups | 2,036 rows, being 386 requests, measured 2026-10-01 | **Authored for this project**, stated by the operator 2026-10-05. No copyright claim, see below | no upstream, so nothing to pin |

### The two inferred links

**Harmful side.** `mlabonne/harmful_behaviors` holds 520 rows (416 train, 104 test). By row count and inspection, its contents are AdvBench's `harmful_behaviors.csv` from Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models* (the `llm-attacks` repository, MIT licence). We believe that slice is MIT with attribution, and we attribute it below.

**Harmless side.** `mlabonne/harmless_alpaca` holds 31,323 rows and declares nothing. Its name, and the abliteration tutorial that introduced it, both point at Stanford's `tatsu-lab/alpaca`. Alpaca declares **CC BY-NC 4.0**: attribution, and **non-commercial use only**. If that inference is right, the harmless side carries a restriction stronger than anything on the harmful side.

Both beliefs are inferences from names, row counts and content shape. No dataset card says either. If you are making a licensing decision that depends on them, verify it yourself. If an upstream maintainer tells us we are wrong, we will correct the card.

### The decision on the licence gap

**Accepted, 2026-10-01, reviewed by Daniel Iwugo.** The terms above are accepted as the position this project redistributes under. It is a recorded acceptance, so nobody needs to re-derive it each time they read the card.

The evidence supports it. `mlabonne/harmful_behaviors` holds the 520 rows of AdvBench's CSV and matches it on inspection, and `mlabonne/harmless_alpaca` points at `tatsu-lab/alpaca` by name, origin and shape. The track is distributed under the strictest term in the chain, so the acceptance costs a reader nothing even where an inference is wrong.

The acceptance does **not** claim that `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca` declare a licence. They still declare none, and that stays on this card. Asking the upstream maintainer for a statement is worth doing, but it no longer blocks anything here.

### What this track is distributed under, and why

Every licence in the chain permits redistribution. They differ in what they attach to it. Applying the most restrictive term to the whole is the only position that satisfies all of them at once. So the track is **CC BY-NC 4.0**: attribution required, non-commercial use only.

| Slice | Upstream terms | What redistribution requires |
|---|---|---|
| Most of the harmful side | Apache-2.0 | Carry the notice |
| 176 harmful rows | MIT, inferred | Attribute Zou et al. |
| The harmless side | CC BY-NC 4.0, inferred | Attribute, and non-commercial |

The harmless top-ups had no answer at all until 2026-10-05. That day the operator recorded that he generated them himself, with a proprietary assistant he holds, and that it carries no Llama lineage. That is recorded as his statement, because it is the only form the answer can take. There is no dataset to fetch and no seed to replay, so nobody can check it or regenerate those 386 requests. **Saying so is the point.** A reader can confirm the rows are present and hash them, but cannot rebuild them. That is a weaker property than the rest of the track has, and it cannot be fixed after the fact.

**Commercial use of the harmless side rests on a permission, not on the public licence.** The 386 requests are the minority. The other 2,946 harmless rows trace to Alpaca, which declares CC BY-NC 4.0. The owner of this project records that he obtained Alpaca's approval to use the material commercially and without attribution (2026-10-05). **So a reader who checks only the public tags will conclude the track is non-commercial, and will be reading the tags correctly.** The permission is specific to this project and does not travel to anyone who copies the track. If you want to use this corpus commercially, seek your own.

Attribution stays here regardless of the waiver. Withdrawing a citation would make the chain harder to audit, and this page exists to be audited.

Much of this corpus is machine-generated: Bahushruth's card describes synthetic prompts, and Alpaca was model-generated. Content with no human author attracts thin or no copyright in many jurisdictions, so some of it may carry no restriction at all. That argument is not relied on here. Attributing everything and shipping under the strictest term settles the question without needing it.

### Attribution

- AdvBench, from Zou, Wang, Carlini, Nasr, Kolter and Fredrikson, *Universal and Transferable Adversarial Attacks on Aligned Language Models* (2023), `llm-attacks`, MIT licence.
- Alpaca, from Taori, Gulrajani, Zhang, Dubois, Li, Guestrin, Liang and Hashimoto, *Stanford Alpaca: An Instruction-following LLaMA Model* (2023), CC BY-NC 4.0.
- `Bahushruth/abliteration-harmful-enriched`, Apache-2.0.
- `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`, both undeclared.

## What it is for, and what it is not for

**For:** measuring whether a refusal-removal method worked, and whether the resulting model still recognises harm. A refusal rate reported without a harmless arm beside it cannot tell "the abliteration worked" apart from "the model is too damaged to refuse anything". The harmless partition is not optional.

**Not for:** training a model to comply with the harmful prompts. The rows are requests, not instructions with answers, so there is nothing here to imitate.

## Known limitations

These are recorded because an instrument with undisclosed limits is worse than none.

- **The pool's assembly is not recorded, though its provenance is.** `track.json` records where every boundary fell, and the split can be rebuilt from the pool exactly. What was never written down is how the pool was assembled, specifically the harmless top-ups added after the first build: their source, their count, and the revision. So `senbonzakura track build` reconstructs a pool of the same shape from the same named upstreams, not the identical one. Anything built since carries a manifest, so this cannot recur.

  **Measured 2026-10-01: 2,036 of the 4,982 harmless rows (40.87%) are the gap, being 386 distinct requests expanded about five times each.** They are spread across all three partitions (106 of 257 in `fit`, 58 of 128 in `search`, 1,872 of 4,597 in `measure`), not appended as a tail. That says a second source was mixed in before partitioning. `docs/corpus-provenance.md` carries the table and the method.
- **A copy distributed before 2026-07-30 leaks badly, and was labelled "clean".** Measured 2026-08-16: in that export, 189 of 200 harmful evaluation rows and 196 of 196 harmless ones sat inside the fitting set. The repair landed on 2026-07-30 and the export was never refreshed, so its name promised the opposite of its contents for six weeks. The track published here is the repaired one, at 0 overlap on both sides by exact match and by request key. If you hold an earlier copy, check it. `python -m senbonzakura.track --out <track> --audit` is the check.
- **The corpus is not axis-balanced**, despite the "35axis" name. Later top-ups were not spread evenly across categories, so per-category rates are not comparable.
- **Some harmful prompts may not be harmful.** The pool was assembled from upstream labels and has not been re-adjudicated row by row.
- **A ruler reading only prompt length beats one of the measured models.** That control is published with the results. Treat any AUC near the length-only baseline as a null.
- **The topic-matched harmless set is retired, as of 2026-10-01.** It was never partitioned, so results using it were not held out in the same sense as the rest. Rather than partition it, we measured whether it was worth keeping. On Qwen3-1.7B its subject-matching statistic read 0.722, against 0.718 for the general corpus. A synthetically perfect match reads 0.299. So the set matched no better than the corpus it was meant to improve on. Partitioning it would have bought 0.004 of subject matching for a schema change and a repack.

  **Retired by documentation, not by deletion.** The set stays on disk with that measurement beside it. If you hold a figure computed against it, the lack of partitioning is the caveat. It attaches only to abliterated models: on a base model nothing is fitted on any rows, so there is nothing to hold out from.

  This gets revisited if matching at scoring time, rather than in the dataset, turns out to change a result. The synthetic evidence so far says a matched dataset is necessary but not sufficient, which is why the dataset was the cheaper thing to give up.
- **AdvBench is inside this track, and 11 of its requests reached the fitting side.** Measured 2026-10-01, the verdict is **contaminated**. AdvBench arrives through `mlabonne/harmful_behaviors`. Of its 508 distinct requests, 8 sit in the harmful `fit` partition and 3 in `search`, so a model tuned with this track was tuned on 11 requests AdvBench would later test it against. **Any AdvBench figure computed over the whole benchmark from this track is in-sample for those 11 requests and must not be reported as held out.**

  | Where | Distinct AdvBench requests |
  |---|---|
  | `fit` | 8 |
  | `search` | 3 |
  | `measure` | 159 |
  | never held by this track | 338 |

  The clean figure a reader can still compute covers **497 requests**: the 159 in `measure` plus the 338 this track has never held. Report that subset and say which subset it is. Reproduce the measurement with the track's own checker:

  ```sh
  python -m senbonzakura.track --out <this track> --contamination advbench.txt \
      --contamination-name AdvBench
  ```

  Two caveats. The benchmark used is AdvBench **as it arrives in this track**: the 520 rows of `mlabonne/harmful_behaviors` at revision `01cead01`, whose row count and contents match AdvBench's `harmful_behaviors.csv`. It was not a fresh fetch of the `llm-attacks` CSV. And matching is by request key over discovered templates, not by string. A whole-prompt comparison on this corpus once reported zero overlap while the same requests sat on both sides in different templates.

  The same applies to any other public set assembled from the same upstreams, which is most of them.
- **The "about 430 rows are AdvBench" figure this card used to carry is not supported by measurement.** The measured count is **176 of 4,895 harmful rows (3.6%)** whose request appears in `mlabonne/harmful_behaviors`, counted template-aware on 2026-10-01. The 430 was an estimate, roughly two and a half times too high. The corrected figure is used throughout this card.

## Citing

If you build a track with this recipe, cite AdvBench and Alpaca (above), the two HuggingFace datasets the builder fetches, and this repository. If you publish numbers from it, publish the harmless arm beside them.

## Disclaimer

This track is measurement material. It exists so that a claim about refusal removal can be checked by somebody who does not trust the person making it.

**What is supplied is prompts, not answers.** Every row is a request. There are no completions, no harmful outputs, and nothing a model could be trained to imitate. The harmful partition is useful for asking a model a question and seeing what it does. It is not a corpus of harmful knowledge.

**No warranty is given**, on the data's accuracy, labelling, fitness for any purpose, or the correctness of any number measured with it. The known limitations above are the ones that have been found, not the ones that exist.

**Responsibility for use sits with the user.** Taking this track means accepting what attaches to it: CC BY-NC 4.0, attribution to every source named above, the acceptable-use terms of any model it is pointed at, and the law where you are. Neither the author of this repository nor the upstream authors is responsible for what anyone does with it, or with any model measured or modified using it.

**Abliteration removes safety behaviour wholesale.** That is what it is for, and it is the reason to be careful. A model with its refusals removed will answer things a deployed model should not. Putting one in front of users is a decision whose consequences belong to whoever makes it.
