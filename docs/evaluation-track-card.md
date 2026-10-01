# The Senbonzakura evaluation track: dataset card

This is the card for the evaluation track every published number in this project was measured
on. It exists so a reader can answer three questions without asking us: what is in it, where
every row came from, and what they are allowed to do with it.

**This card covers what is in the track and what is wrong with it.** For where each row came
from, which revision it was fetched at, and how the partitions are packed into the files, see
[where the evaluation corpus came from](./corpus-provenance.md). The two pages are kept separate
on purpose, so there is one place to look for each question rather than two accounts of both.

**Nothing here is generated content.** The track is a partitioned collection of prompts drawn
from existing public datasets. It contains no model outputs, no completions, and no answers to
any harmful request. It is a measuring instrument, not a corpus of harm.

**The rows are published as a gated dataset, not inside this repository:**
[`ops-malware/senbonzakura-dataset`](https://huggingface.co/datasets/ops-malware/senbonzakura-dataset). Every row came from
somewhere else, so the track carries the upstream terms forward: it is distributed under
**CC BY-NC 4.0**, the most restrictive licence in its chain, with attribution to every source
named below. Access is gated so that taking it is a deliberate act: a reader accepts the
acceptable-use terms before downloading, and an automated scraper does not get it by accident.
The gate collects nothing about you beyond what HuggingFace needs to operate it; it exists to
make the agreement intentional, not to build a list.

**A released wheel is a different matter, and you should know it.** The same rows ship inside the
released package as `--track default`, so installing a release puts roughly 6,200 harmful prompts
in your site-packages with no gate and no terms accepted. They are obfuscated rather than
protected: the key ships beside them, and anyone who wants them can have them in an afternoon. The
gate on the HuggingFace copy does not apply to them. Saying only the paragraph above would leave
the impression that every install is prompt-free, and a release is not.

**Which channel carries what, today.** The ordinary install is now one of the channels that
carries them, so read this table before running `pip install`:

| Channel | Carries the 6,200 rows? |
|---|---|
| `pip install senbonzakura` (PyPI, 0.4.0) | **Yes**, as `--track default`, with no gate |
| A build from a clone, or `pip install git+...` | No. The two `.bin` blobs under `src/senbonzakura/data/` are generated rather than committed, so a build from source carries neither |
| The HuggingFace dataset | Yes, gated, under CC BY-NC 4.0 |

::: warning This table was the other way round until 2026-09-26
It said "neither obtainable install carries the rows", which was true while PyPI served only
0.3.0, from July 2026, and the only other route was a build from source. 0.4.0 is on PyPI and the
wheel carries both blobs, so the sentence that had been a disclosure became a reassurance, and
pointed at the wrong artefact. If you installed before reading this, `senbonzakura doctor` lists
what your install holds.
:::

`senbonzakura track build` rebuilds a pool of the same shape from the same upstreams for anyone who
would rather fetch the sources themselves.

## What it is

A three-way split of harmful and harmless prompts, used to fit refusal directions, to search
for a configuration, and to measure the result. The three partitions are disjoint by recorded
row index rather than by convention:

| Partition | Harmful rows | Harmless rows | Used for |
|---|--:|--:|---|
| fit | 259 | 257 | extracting refusal directions |
| search | 132 | 128 | selecting a configuration |
| measure | 4,504 | 4,597 | the published numbers |

The boundaries live in `track.json` and are read at the start of every run, so a run whose
flags would reach past one is refused rather than quietly allowed. Rebuild and audit it with:

```sh
python -m senbonzakura.track --out mytrack --audit
```

The split is by request rather than by string. The same question wears many templates, and
comparing whole prompts once reported a clean split while 60% of the measured set was training
questions in other clothes.

## Where the rows came from, and what is known about each licence

This is the part worth reading slowly. **Two of the three direct sources declare no licence at
all**, and the probable root of the harmless side is non-commercial. That combination is what
sets the terms this track is distributed under; see "What this track is distributed under" below.

Every position in the table was read from the HuggingFace dataset API on **2026-08-16**, and the
revision each was read at is recorded so a later reader can tell a changed card from a wrong one.

| Source | Side | Declared licence | Revision read |
|---|---|---|---|
| `Bahushruth/abliteration-harmful-enriched` | most of the harmful side | **apache-2.0**, declared in `cardData` and tagged | `f29c0b77` |
| `mlabonne/harmful_behaviors`, reached through the above | 176 harmful rows, measured | **None.** No `license` field, no `cardData` licence, no tag | `01cead01` |
| `mlabonne/harmless_alpaca` | the harmless side | **None.** No `license` field, no `cardData` licence, no tag | `02c6a92c` |
| `tatsu-lab/alpaca`, the probable root of the above | the harmless side, indirectly | **cc-by-nc-4.0**, declared and tagged | `dce01c9b` |
| The harmless top-ups | 2,036 rows, being 386 requests, measured 2026-10-01 | **Unrecorded.** See the reproducibility limitation below | not recorded |

### The two inferred links, stated plainly

**Harmful side.** `mlabonne/harmful_behaviors` holds 520 rows (416 train, 104 test). That is
exactly the row count and, on inspection, the contents of AdvBench's `harmful_behaviors.csv`
from Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models*
(the `llm-attacks` repository, MIT licence). We therefore believe the true position for that
slice is MIT with attribution, and we attribute it below.

**Harmless side.** `mlabonne/harmless_alpaca` holds 31,323 rows and declares nothing. Its name,
and the abliteration tutorial that introduced it, both point at Stanford's `tatsu-lab/alpaca` as
its source. Alpaca declares **CC BY-NC 4.0**: attribution required, and **non-commercial use
only**. If that inference is right, the harmless side of this track carries a non-commercial
restriction, which is a stronger constraint than anything on the harmful side.

**Both beliefs are inferences from names, row counts and content shape, not statements any of
the dataset cards make.** They are recorded here rather than quietly relied on. If you are making
a licensing decision that depends on either, verify it yourself rather than taking this card's
word for it. If an upstream maintainer tells us this is wrong, we will correct the card.

### The decision on the licence gap

**Accepted, 2026-10-01, reviewed by Daniel Iwugo.** The licence terms above are accepted as the
position this project redistributes under. This is a recorded acceptance rather than a line of
reasoning, so it does not need re-deriving every time somebody reads the card.

What supports it is the evidence in the two subsections above, which stays evidence rather than
becoming the claim: `mlabonne/harmful_behaviors` holds exactly the 520 rows of AdvBench's
`harmful_behaviors.csv` and matches it on inspection, and `mlabonne/harmless_alpaca` points at
`tatsu-lab/alpaca` by name, by origin and by shape. The track is distributed under the strictest
term in the chain, so the acceptance costs a reader nothing even where an inference is wrong.

What the acceptance does **not** claim: `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`
still declare no licence at all, and that fact is not resolved by anybody accepting anything. It
stays on this card because a reader doing their own diligence needs it. Asking the upstream
maintainer for a statement remains worth doing as a courtesy and would let the table read through
to a declared term rather than an inferred one; it is no longer a blocker on anything here.

### What this track is distributed under, and why

Every licence in the chain permits redistribution; they differ in what they attach to it. Taking
the most restrictive of them and applying it to the whole is the only position that satisfies all
three at once, so the track is distributed under **CC BY-NC 4.0**: attribution required,
non-commercial use only.

| Slice | Upstream terms | What redistribution requires |
|---|---|---|
| Most of the harmful side | Apache-2.0 | Carry the notice |
| 176 harmful rows | MIT, inferred | Attribute Zou et al. |
| The harmless side | CC BY-NC 4.0, inferred | Attribute, and non-commercial |

Two of the direct upstreams declare nothing at all, which is why the table reads through to the
roots rather than stopping at them. Most of this corpus is also machine-generated (Bahushruth's
card describes synthetic prompts; Alpaca was model-generated), and content with no human author
attracts thin or no copyright in many jurisdictions, so a good deal of it may carry no
restriction whatsoever. That argument is not relied on here. Attributing everything and shipping
under the strictest term costs nothing and settles the question without needing it.

For anyone who would rather fetch the sources themselves, `senbonzakura track build` assembles a pool
of the same shape from the same upstreams at pinned revisions.

### Attribution

- AdvBench, from Zou, Wang, Carlini, Nasr, Kolter and Fredrikson, *Universal and Transferable
  Adversarial Attacks on Aligned Language Models* (2023), `llm-attacks`, MIT licence.
- Alpaca, from Taori, Gulrajani, Zhang, Dubois, Li, Guestrin, Liang and Hashimoto,
  *Stanford Alpaca: An Instruction-following LLaMA Model* (2023), CC BY-NC 4.0.
- `Bahushruth/abliteration-harmful-enriched`, Apache-2.0.
- `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`, both undeclared.

## What it is for, and what it is not for

**For:** measuring whether a refusal-removal method worked, and whether the model that came out
still recognises harm. A refusal rate reported without a harmless arm beside it cannot tell
"the abliteration worked" apart from "the model is too damaged to refuse anything", so the
harmless partition is not optional.

**Not for:** training a model to comply with the harmful prompts. The rows are requests, not
instructions with answers, and there is nothing here to imitate.

## Known limitations

These are recorded because a measuring instrument with undisclosed limits is worse than none.

- **The prompt pool's provenance is recorded; the pool's own assembly is not.** The track itself
  is fully described: `track.json` records where every boundary fell, and the split can be
  rebuilt from the pool exactly. What was never written down is how the pool was assembled from
  its upstreams, specifically the harmless "top-ups" added after the first build: not their
  source, not their count, not the revision they came from. So `senbonzakura track build` reconstructs
  a pool of the same shape from the same named upstreams rather than the identical one. Anything
  built from here on carries a manifest, so this cannot recur.

  **The difference was measured on 2026-10-01 and it is 2,036 of the 4,982 harmless rows (40.87%),
  being 386 distinct requests expanded about five times each.** It is spread across all three
  partitions (106 of 257 in `fit`, 58 of 128 in `search`, 1,872 of 4,597 in `measure`) rather than
  appended as a tail, which says a second source was mixed in before partitioning rather than added
  at the end. `docs/corpus-provenance.md` carries the table and the method.
- **A copy of this track distributed before 2026-07-30 leaks, badly, and was labelled "clean".**
  Measured 2026-08-16: in that earlier export, 189 of 200 harmful evaluation rows and 196 of 196
  harmless ones were sitting inside the fitting set. The repair landed on 2026-07-30 and the
  export was never refreshed, so its name promised the opposite of its contents for six weeks.
  The track published here is the repaired one, verified at 0 overlap on both sides by exact
  match and by request key. If you hold an earlier copy, check it rather than trusting its name;
  `python -m senbonzakura.track --out <track> --audit` is the check.
- **The corpus is not axis-balanced** despite the "35axis" name. Later top-ups were not
  distributed evenly across categories, so per-category rates are not comparable to each other.
- **Some prompts in the harmful pool may not be harmful.** The pool was assembled from upstream
  labels and has not been re-adjudicated row by row.
- **A ruler reading only prompt length beats one of the measured models.** That control is
  published alongside the results for exactly this reason; treat any AUC near the length-only
  baseline as a null.
- **The topic-matched harmless set is retired, as of 2026-10-01.** It was never partitioned, so
  results using it were not held out in the same sense as the rest. Rather than partition it, we
  measured whether it was worth keeping: on Qwen3-1.7B its subject-matching statistic read 0.722
  against 0.718 for the general corpus, where a synthetically perfect match reads 0.299. So the
  set we called topic-matched matched no better than the corpus it was meant to improve on, and
  partitioning it would have bought 0.004 of subject matching for a schema change and a repack.

  **Retired by documentation, not by deletion.** The set stays on disk with that measurement
  beside it, so anybody who reads the trade differently can pick it back up. If you hold a figure
  computed against it, the lack of partitioning is the caveat that attaches, and it attaches only
  to abliterated models: on a base model nothing is fitted on any rows, so there is nothing to
  hold out from.

  This gets revisited if matching at scoring time, rather than in the dataset, turns out to change
  a result. The synthetic evidence so far says a matched dataset is necessary and not sufficient,
  which is why the dataset was the cheaper thing to give up.
- **AdvBench is inside this track, and 11 of its requests reached the fitting side.** Measured
  2026-10-01, and the verdict is **contaminated**. AdvBench arrives through
  `mlabonne/harmful_behaviors` as described above. Of its 508 distinct requests, 8 sit in the
  harmful `fit` partition and 3 in `search`, so a model tuned with this track was tuned on 11
  requests AdvBench would later mark it against. **Any AdvBench figure computed over the whole
  benchmark from this track is in-sample for those 11 requests and must not be reported as held
  out.**

  | Where | Distinct AdvBench requests |
  |---|---|
  | `fit` | 8 |
  | `search` | 3 |
  | `measure` | 159 |
  | never held by this track | 338 |

  What that leaves is a clean figure a reader can still compute: **497 requests**, being the 159
  in `measure` plus the 338 this track has never held. Report that subset and say which subset it
  is. Reproduce the measurement with the track's own checker:

  ```sh
  python -m senbonzakura.track --out <this track> --contamination advbench.txt \
      --contamination-name AdvBench
  ```

  Two caveats on the number, stated rather than buried. The benchmark it was measured against is
  AdvBench **as it arrives in this track**, that is the 520 rows of `mlabonne/harmful_behaviors`
  at revision `01cead01`, whose row count and contents match AdvBench's `harmful_behaviors.csv`;
  it was not a fresh fetch of the `llm-attacks` CSV. And matching is by request key over
  discovered templates, not by string, because a whole-prompt comparison on this corpus once
  reported zero overlap while the same requests sat on both sides wearing different templates.

  The same applies to any other public set assembled from the same upstream sources, which is
  most of them.
- **The "about 430 rows are AdvBench" figure this card used to carry is not supported by
  measurement.** The measured count is **176 of 4,895 harmful rows (3.6%)** whose request appears
  in `mlabonne/harmful_behaviors`, counted template-aware on 2026-10-01. The 430 was an estimate
  and it was roughly two and a half times too high. The corrected figure is used throughout this
  card.

## Citing

If you build a track with this recipe, cite AdvBench and Alpaca (above), the two HuggingFace
datasets the builder fetches, and this repository. If you publish numbers from it, publish the
harmless arm beside them.

## Disclaimer

This track is measurement material. It exists so that a claim about refusal removal can be
checked by somebody who does not trust the person making it, which is the only kind of checking
worth having.

**What is supplied is prompts, not answers.** Every row is a request. There are no completions,
no harmful outputs, and nothing here that a model could be trained to imitate. The harmful
partition is useful for asking a model a question and seeing what it does; it is not a corpus of
harmful knowledge.

**No warranty of any kind is given**, on the data's accuracy, its labelling, its fitness for any
purpose, or the correctness of any number measured with it. The known limitations above are the
ones that have been found, not the ones that exist.

**Responsibility for use sits with the user.** Taking this track means accepting what attaches to
it: CC BY-NC 4.0, attribution to every source named above, the acceptable-use terms of any model
it is pointed at, and the law where you are. Neither the author of this repository nor the authors
of the upstream sources are responsible for what anyone does with it, or with any model measured
or modified using it.

**Abliteration removes safety behaviour wholesale.** That is what it is for and it is the reason
to be careful. A model with its refusals removed will answer things a deployed model should not,
and putting one in front of users is a decision with consequences that belong to whoever makes
it.
