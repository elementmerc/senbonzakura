# The Senbonzakura evaluation track: dataset card

This is the card that ships with the published evaluation track. It exists so a reader can
answer three questions without asking us: what is in it, where every row came from, and what
they are allowed to do with it.

**Nothing here is generated content.** The track is a partitioned collection of prompts drawn
from existing public datasets. It contains no model outputs, no completions, and no answers to
any harmful request. It is a measuring instrument, not a corpus of harm.

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

This is the part worth reading slowly, because one link in the chain is inferred rather than
stated.

| Source | Rows | Licence position |
|---|--:|---|
| `Bahushruth/abliteration-harmful-enriched` | most of the harmful side | **Declared** apache-2.0 on the dataset card |
| `mlabonne/harmful_behaviors`, reached through the above | about 430 | **Declares nothing.** Empty README, no `license` field, no `cardData` tag |
| `mlabonne/harmless_alpaca` and the harmless top-ups | the harmless side | As declared upstream |

**The inferred link, stated plainly.** `mlabonne/harmful_behaviors` holds 520 rows (416 train,
104 test). That is exactly the row count and, on inspection, the contents of AdvBench's
`harmful_behaviors.csv` from Zou et al., *Universal and Transferable Adversarial Attacks on
Aligned Language Models* (the `llm-attacks` repository, MIT licence). We therefore believe the
true position for that slice is MIT with attribution, and we attribute it below.

**That belief is an inference from a row count and a content match, not a statement either
dataset card makes.** We are recording it here rather than quietly relying on it. If you are
making a licensing decision that depends on those roughly 430 rows, verify it yourself rather
than taking this card's word for it. If an upstream maintainer tells us this is wrong, we will
correct the card and re-cut the release.

### Attribution

- AdvBench, from Zou, Wang, Carlini, Nasr, Kolter and Fredrikson, *Universal and Transferable
  Adversarial Attacks on Aligned Language Models* (2023), `llm-attacks`, MIT licence.
- `Bahushruth/abliteration-harmful-enriched`, Apache-2.0.
- `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`.

## What it is for, and what it is not for

**For:** measuring whether a refusal-removal method worked, and whether the model that came out
still recognises harm. A refusal rate reported without a harmless arm beside it cannot tell
"the abliteration worked" apart from "the model is too damaged to refuse anything", so the
harmless partition is not optional.

**Not for:** training a model to comply with the harmful prompts. The rows are requests, not
instructions with answers, and there is nothing here to imitate.

## Known limitations

These are recorded because a measuring instrument with undisclosed limits is worse than none.

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

If you use the track, cite AdvBench (above) and this repository. If you publish numbers from
it, publish the harmless arm beside them.

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

**Responsibility for use sits with the user.** Downloading this track means taking on whatever
obligations attach to it: the upstream licences recorded above, the acceptable-use terms of any
model it is pointed at, and the law where the user is. Neither the author of this repository nor
the authors of the upstream sources are responsible for what anyone does with it, or with any
model measured or modified using it.

**Abliteration removes safety behaviour wholesale.** That is what it is for and it is the reason
to be careful. A model with its refusals removed will answer things a deployed model should not,
and putting one in front of users is a decision with consequences that belong to whoever makes
it.
