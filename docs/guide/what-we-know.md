# What is and is not established

This is the page about the claim on the tin: that removing several refusal directions beats
removing one.

Here's the summary, and I'd rather you got it from me in the first paragraph than worked it
out from the appendix.

::: warning The short version
**We tested it, and on the one model we can measure properly the claim is false.** Two
directions cost roughly 1.5 to 1.9 times the collateral damage of one at the same refusal rate, across five
seeds a side. The feature had also never worked at all until it was rewritten on 2026-08-03,
which is why the question could not be asked before then.
:::

The rest of this page is how we got there, because the failures are more instructive than
the result would have been.

## The feature that never once worked

Start with the comparison that was supposed to settle it.

A run on Qwen3-1.7B, built specifically to test whether removing several directions beats
removing one: two arms, five seeds each, held out, everything identical except the
direction budget. Clean design. It finished, it had a p-value, it looked great.

**Both arms had removed exactly one direction per layer.**

The extra capacity existed in the data structure and simply never got filled. No second
axis cleared the tool's own refusal-separation threshold at any of the model's 29 layers,
so the surgery in the two arms was byte-for-byte identical. The comparison compared two
searches, not two direction budgets, and it got withdrawn the same day.

::: tip New words: refusal-separation threshold
The check that decides whether a candidate direction actually carries refusal, rather than
carrying something else that happens to be nearby. It's the gate between "I found a
direction" and "I found a refusal direction", and the whole multi-direction idea rests on
it.
:::

### And then it got worse

The obvious next question is why no second axis ever cleared the threshold. The comfortable
answer would have been "refusal in Qwen3-1.7B just is one direction", which is a
respectable scientific finding and would have been a nice paper.

It's not that.

The check compares the average harmful reply against the average harmless one, measured
along a direction that was **constructed to sit at right angles to both of those
averages**. Measure a difference along an axis that's perpendicular to the difference and
you get zero. Every time. By construction.

So it wasn't strict. It was unsatisfiable. **No direction could pass that check, on any
model, at any setting, ever.**

If you want the physical version: it's a metal detector wired so the coil can never
complete a circuit. It sweeps the beach beautifully. It has never once beeped. And for
months you conclude, reasonably, that this is a beach with no metal on it.

Measured across two model families and three prompt sets: **13,970 candidate directions,
every single one rejected, none of them close.**

**So what:** every run this project made before 2026-08-03 applied exactly one direction,
whatever it was asked for. The tool named after a sword that splits into a thousand blades
had been using one blade since the day it was written. 🥲

## What's been fixed

The extractor was rewritten the same day.

Candidate directions now come from clustering the harmful prompts and taking each cluster's
own average against the harmless average. The important word is *own*: a candidate now
separates the two groups **by construction**, which is the exact opposite of the previous
arrangement, where it was built unable to.

On the same four probes it now finds and applies **up to eight directions per layer** where
it previously found one.

## What hasn't

The feature now does something. That is not the same as the claim being true, and the
README won't say otherwise until two things are measured.

**Do the extra directions carry refusal, or do they carry topic?** Still open, but for a
narrower reason than before.

The threshold meant to answer this rejected nothing at all: all 678 candidates scored between
0.90 and 9.02 against a bar of 0.5. On 2026-08-16 we found out why, and it wasn't leniency. A
candidate direction is a cluster's mean minus the harmless mean, and the score judging it was a
difference of those same means, computed on those same rows. It was asking whether the quantity
a vector was built to maximise is large along that vector. It cannot come out small. The bar was
never doing anything, on any run, ever.

That's fixed. Each candidate is now fitted on half its rows and scored on the half it never saw,
and the threshold has a measured floor beside it: directions built from random subsets of the
harmful prompts, carrying nothing, scored through the identical path. A candidate has to beat
the best of them.

**What's still open is the part the fix doesn't reach.** A held-out score says a direction
separates harmful prompts from harmless ones. It doesn't say *why*. Cluster your harmful prompts
and one cluster will be about, say, explosives. That cluster separates from the harmless prompts
partly because it's about explosives, not because of anything to do with refusal, and it
separates just as well on rows it never saw. Cut along it and you haven't removed the model's
reluctance, you've removed its chemistry.

Telling those apart needs the subject matter held still, so refusal is the only thing left to
vary. It's the move a medical study makes when it compares patients against matched controls
rather than against the general population.

On 2026-09-02 we measured how badly the unmatched comparison fails, on synthetic data where we
knew the answer in advance. Two worlds, identical except that one contains refusal and the other
contains none at all. Every candidate in both should be rejected in the second world.

| how a candidate is judged | world with no refusal | world with refusal |
|---|---|---|
| against harmless prompts in general | 3.547 | 3.703 |
| against harmless prompts on the same subject | **0.319** | **4.369** |

Read the top row twice. **The two worlds score the same**, so no threshold anywhere on that scale
could separate them, and no choice of statistic changes that. We tried four. The bottom row is the
same data judged against matched controls: the world with no refusal lands on the scale's own
"nothing here" value, and the world with refusal stands clear.

Two things follow, and the second was a surprise.

The first is that the tool now has `--matched-scoring`, which picks each candidate's controls from
the harmless prompts nearest it in content. It's off by default, because it changes what every
separation number means and we haven't yet measured it on a real model.

The second is that **a topic-matched harmless corpus would not have been enough on its own**. In
the numbers above the harmless side already had the same mix of subjects as the harmful side, and
the unmatched comparison still couldn't tell the two worlds apart. The matching has to happen when
each candidate is scored, not only when the corpus is built. This page previously said the dataset
was the missing piece; that was half the answer.

One honest caveat: those figures are synthetic. Whether subject matter swamps refusal by that much
inside a real model is still unmeasured, and until it is, this question stays open.

**Does removing several directions beat removing one?** We ran it, and the answer on the one
model we can measure properly is **no**. See the next section.

## The answer: run on 2026-08-13, noncompliance column scored later

One direction against two. Same tool, same corpus, same search budget, same five seeds, one
parameter apart, with the direction budget pinned at both ends so neither arm could drift into
the other's territory. Every resulting model was then scored by one instrument on 200 prompts
that nothing was fitted or selected on.

| Direction budget | Coherence drift | Hard refusal | Noncompliance |
|---|---|---|---|
| One | **0.0497** (spread 0.0177) | 0.1% | ~~3.8%~~ |
| Two | 0.0932 (spread 0.0482) | 0.3% | ~~4.2%~~ |

::: warning The noncompliance column is withdrawn, 2026-09-09
The arms ran on 2026-08-13; the noncompliance column was scored separately on
2026-09-07, which is why two dates appear on this page. The next day the hedging detector was found to count
statements of fact about legality or danger as hedging, so a complete, compliant technical answer
that mentioned an activity is illegal was scored as a soft refusal. Both figures came from it.

The conclusion of this section does not rest on them: it is argued on coherence drift at matched
hard refusal, and hard refusal is measured by a different ruler that this defect did not touch.
The column is struck through rather than deleted because it was published.
:::

Both budgets removed hard refusal, which is what makes the rest readable: this is a comparison
at matched refusal, where the only thing left to differ is the price paid for it. Two directions
did roughly **1.5 to 1.9 times the collateral damage and bought nothing**, and they were less
predictable seed to seed (the spread is nearly three times as wide).

The gap is 0.0435, and an exact permutation test over all 252 ways of splitting these ten
seeds puts it at **p = 0.016**. That is a real difference, and stronger than the way this
paragraph used to describe it.

::: tip You can recompute this yourself
The ten per-seed drift values are committed at
[`evidence/k-sweep-2026-08-13/drift-per-seed.json`](https://github.com/elementmerc/senbonzakura/blob/main/evidence/k-sweep-2026-08-13/drift-per-seed.json),
and `tests/test_published_k_sweep.py` recomputes both figures on this page from that file through
the project's own estimator. Until 2026-09-10 those numbers lived only in a working note that is
not in this repository, so the most-quoted figure here traced to nothing a reader could open.
:::

**A correction, 2026-09-09.** This used to read "clears the pooled spread by about a fifth,
real but slim", and then said that dropping the worst two-direction seed made the result
*stronger*. Both sentences came from comparing a gap to a pooled spread, which this project's
own code now says in as many words is not a test, because it ignores how many observations
there are. Under the permutation test the second sentence is **wrong**: dropping that seed
halves the gap, from 0.0435 to 0.0222, and moves p from 0.016 to **0.048**. The result does
not rest on one bad run, but it is meaningfully weaker without it, and the opposite was
published here for a month.

**What it doesn't say.** This is Qwen3-1.7B alone. Refusal geometry may differ on larger models
or on other families, and the honest position is that the question is open elsewhere and closed
here. It also doesn't rescue the threshold problem above: these arms ran before the held-out
score existed, so their directions were selected by a filter that could not reject anything at
all. "Two directions" means "two directions this tool chose", not "two directions known to carry
refusal". A re-run under the fixed filter may select different directions, and this table is the
old filter's answer until it does.

**Why publish a negative result about your own headline feature?** Because the alternative is
waiting for a friendlier model, and a project whose whole argument is that other people's
numbers deserve scrutiny doesn't get to make an exception for its own.

## So where does that leave the project

Honestly: with a working single-direction abliterator, an unusually good
[measurement suite](/guide/compass), and a headline claim that its own measurements do not
support on the model it can measure best.

That's a less exciting sentence than the README of most tools in this space, and it's the
true one. The tool keeps the multi-direction capability because the question is genuinely open
on architectures we can't yet reach, and because being able to *test* the idea is worth more
than believing it. By default the search chooses its own budget between one and three
directions, and left to itself on this model it picks one more often than not.

## Where next

- [Limits and known defects](/guide/limits) for every other condition attached to every
  other number here.
- [The compass](/guide/compass) for the half of the project that does work.
