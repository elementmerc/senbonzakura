# The Senbonzakura evaluation track: dataset card

This is the card for the evaluation track every published number in this project was measured
on. It exists so a reader can answer three questions without asking us: what is in it, where
every row came from, and what they are allowed to do with it.

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
make the agreement intentional, not to build a list. `tools/build_track.py` rebuilds a pool of
the same shape from the same upstreams for anyone who would rather fetch the sources themselves.

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
| `mlabonne/harmful_behaviors`, reached through the above | about 430 harmful rows | **None.** No `license` field, no `cardData` licence, no tag | `02c6a92c`-era, unchanged since 2024-05-30 |
| `mlabonne/harmless_alpaca` | the harmless side | **None.** No `license` field, no `cardData` licence, no tag | `02c6a92c` |
| `tatsu-lab/alpaca`, the probable root of the above | the harmless side, indirectly | **cc-by-nc-4.0**, declared and tagged | `dce01c9b` |
| The harmless top-ups | part of the harmless side | **Unrecorded.** See the reproducibility limitation below | not recorded |

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

### What this track is distributed under, and why

Every licence in the chain permits redistribution; they differ in what they attach to it. Taking
the most restrictive of them and applying it to the whole is the only position that satisfies all
three at once, so the track is distributed under **CC BY-NC 4.0**: attribution required,
non-commercial use only.

| Slice | Upstream terms | What redistribution requires |
|---|---|---|
| Most of the harmful side | Apache-2.0 | Carry the notice |
| About 430 harmful rows | MIT, inferred | Attribute Zou et al. |
| The harmless side | CC BY-NC 4.0, inferred | Attribute, and non-commercial |

Two of the direct upstreams declare nothing at all, which is why the table reads through to the
roots rather than stopping at them. Most of this corpus is also machine-generated (Bahushruth's
card describes synthetic prompts; Alpaca was model-generated), and content with no human author
attracts thin or no copyright in many jurisdictions, so a good deal of it may carry no
restriction whatsoever. That argument is not relied on here. Attributing everything and shipping
under the strictest term costs nothing and settles the question without needing it.

For anyone who would rather fetch the sources themselves, `tools/build_track.py` assembles a pool
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
  source, not their count, not the revision they came from. So `tools/build_track.py` reconstructs
  a pool of the same shape from the same named upstreams rather than the identical one, and the
  difference is unquantified. Anything built from here on carries a manifest, so this cannot
  recur.
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
- **The topic-matched harmless set is not partitioned**, so results using it are not held out
  in the same sense as the rest.
- **AdvBench is inside this track, and how much of it reached the fitting side has not been
  measured.** About 430 rows of the harmful side are AdvBench, arriving through
  `mlabonne/harmful_behaviors` as described above. If any of those rows sit in the `fit` or
  `search` partitions, then a model tuned with this track was tuned on requests AdvBench would
  later mark it against, and any AdvBench figure from it is in-sample rather than held out.
  Nobody has run that check yet, so this card does not tell you the answer; it tells you the
  question is open. Run it yourself before relying on any such figure:

  ```sh
  python -m senbonzakura.track --out <this track> --contamination advbench.txt \
      --contamination-name AdvBench
  ```

  The same applies to any other public set assembled from the same upstream sources, which is
  most of them.

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
