# Who got here first

Short version: **not us.** Refusal-as-more-than-one-direction has been published at least twice
before this project existed, and one of those papers reports the opposite of what we measured.
Being second costs nothing. Being second and quiet about it ends a review.

## The single direction everything builds on

**Arditi, Obeso et al., [*Refusal in Language Models Is Mediated by a Single
Direction*](https://arxiv.org/abs/2406.11717) (2024).** Refusal is largely one direction in
activation space; take the difference between the mean harmful and mean harmless activations,
orthogonalise it out of the weights, and the model stops refusing. Everything below, including
this project, is a response to that paper.

**[Heretic](https://github.com/p-e-w/heretic) by p-e-w** turned it into a tool with an automated
search: Optuna over layer positions and strengths, co-minimising refusals against KL divergence.
Our search is a refinement of that one, and the keyword metric we report for comparison is copied
from it verbatim, which is why this project is AGPL.

## The two papers that already said "more than one"

**Wollschläger et al., [*The Geometry of Refusal in LLMs: Concept Cones and Representational
Independence*](https://arxiv.org/html/2502.17420v2) (2025).** Refusal is a *cone*, not a
direction. Ablating the top-k representationally independent directions raises attack success
**monotonically with k**, beating the difference-in-means baseline at k ≥ 4 on Gemma 2 2B. The
directions are found by gradient optimisation with an independence penalty, not by clustering.

**Piras et al., [*SOM Directions are Better than One*](https://arxiv.org/html/2511.08379v1)
(2025).** A self-organising map over harmful representations, one direction per neuron, harmless
centroid subtracted. Large reported gains:

| Model | Multi | Single |
|---|--:|--:|
| Llama2-7B | 59.1% | 0.0% |
| Llama3-8B | 88.1% | 15.1% |
| Gemma2-9B | 96.3% | 38.9% |
| Qwen-7B | 88.1% | 81.1% |

## The third idea: don't edit all the time

Everything above, this project included, removes a direction unconditionally. The weights change
once, and from then on every prompt pays the same cost, benign ones included. That's the trade
this whole field argues about: how much refusal you removed against how much the model moved.

apostate's default method walks around the trade rather than along it. It fits a detector
direction, a threshold and an actuator direction, and the edit only fires when the detector
crosses the threshold, so a benign prompt is meant to be left untouched. The implementation is the
clever part: the condition is folded into a single ordinary MLP neuron, so the result is a plain
checkpoint that any runtime loads with no custom code.

We haven't reproduced it, we haven't measured it, and nothing on this site is evidence either way.
It's named here because it's a genuinely different idea from the one the papers above are arguing
about, and because this project's instruments, the controls, the three-way split, the compass and
the capability probe, are the ones that could price it. That's a thing to do, not a thing done.

## So why does this project measure the opposite?

Our own comparison found two directions costing about **1.9 times** the coherence damage
of one at matched refusal, and buying nothing this design could have detected. **The 95% interval
on that 1.9 is 1.2 to 3.1**, a percentile bootstrap over the five seeds a side.

This page used to give the figure as a span running from the dropped-seed ratio up to the
estimate, and **that was not an interval and it is withdrawn.** Dropping the one outlying
two-direction seed is a sensitivity check, so joining it to the estimate with "to" gave a reader a
span four times narrower than the real one and invited a confidence the five arms do not support.
The withdrawn wording is not reprinted here: a page that quotes the form it is retiring hands the
next reader the same misreading in a sentence that looks like a correction.
The honest interval is the wide one, and it is the one published here because a reader who
distrusts us would compute it themselves.

The direction of the result is sturdier than its size: one direction beats two in 24 of 25
pairwise seed comparisons. Both papers above report multi-direction winning.

::: warning Our arms did not choose their directions, and theirs did
This is the first thing to say about the disagreement, because it means the two sides are not
measuring the same experiment. Wollschläger et al. find directions by gradient optimisation with
an independence penalty; Piras et al. fit a self-organising map. Both select deliberately. Our
five seeds ran on 2026-08-13, before held-out direction selection existed here, so the second
direction came from a filter now known to accept every candidate it was given.

So our number prices an *arbitrary* second direction and theirs price a chosen one. That makes
this a different experiment rather than a failed reproduction, and our five seeds stand as the old
filter's answer until a re-run under the fixed selector replaces them. The same caveat is
recorded in the evidence file, at
[`evidence/k-sweep-2026-08-13/drift-per-seed.json`](https://github.com/elementmerc/senbonzakura/blob/dev/evidence/k-sweep-2026-08-13/drift-per-seed.json).
:::

With that said, three honest readings of the remaining gap, and we cannot yet separate them:

1. **They measure a different thing.** Both report *attack success rate* and **neither reports KL
   or any coherence cost**. "More directions remove more refusal" and "more directions are worth
   what they cost" are different claims, and only the second one needs a price tag. Our whole
   argument is that the price tag is the missing column.
2. **Our models are in the regime their own numbers say benefits least.** Look at the SOM table:
   the gap is enormous where single-direction ablation barely works (Llama2-7B, 0.0%) and small
   where it already works (Qwen-7B, 7 points). Everything we have measured is under 3B and weakly
   aligned, which is the right-hand column.
3. **We may simply be doing it worse.** Our directions are orthogonalised against the primary
   one, which removes the large shared component by construction: geometrically pure, possibly
   functionally small. There is [a paper arguing exactly that
   mechanism](https://arxiv.org/pdf/2603.22061) for topic-matched contrasts, and it may be the
   explanation for our result rather than a fact about refusal.

We publish our number anyway, with all three readings attached. See
[what is and is not established](/guide/what-we-know).

## The tools

**[Abliterix](https://github.com/wuwangzhang1216/abliterix)** is the closest thing to a direct
competitor and overlaps us heavily: Optuna search, Pareto co-minimisation of refusals against KL,
reversible edits. It is **ahead of us** on method breadth (several named multi-direction methods
we have not implemented), on mixture-of-experts handling, and on prebuilt configurations.

**[apostate](https://github.com/heterodoxin/apostate)** by `heterodoxin`, MIT, four months old
and read here at `main` on 2026-10-01. It's four tools behind one command, and only the oldest of
them is a variant of the method everything above shares: its default, the one you get from
`apostate ablate`, isn't a projection at all. Its other methods include a greedy search under
budgets and an oblique projection with a fitted detector. It has no CI, no lockfile and no
changelog, and it ships its harmful prompt set in plaintext in the tree, which this project
deliberately doesn't. Read partly, through a summarising reader rather than a clone, so the
[comparison page](/comparison) gives it a scope line rather than a column.

**["Exploring the multi-dimensional refusal subspace"](https://www.lesswrong.com/posts/ixJrmYrgHM4TenN7D/exploring-the-multi-dimensional-refusal-subspace-in-1)**
reached this project's clustering approach independently, and says plainly that it has no
random-direction baseline, no cross-topic generalisation test and no KL matching. It clusters
prompt *text*; we cluster *residuals*.

## What is actually left for us

Not the search. Abliterix has one. Not multi-direction. Two papers got there first.

**The measurement.** As far as we can find, nobody publishes **KL at matched refusal removal**:
the cost of the edit, compared at an operating point where both arms removed the same amount of
refusal. Without matching first, the arm that cut harder looks worse on coherence for a reason
that has nothing to do with what was being tested.

apostate is the clearest illustration of why that claim is worth making. Its README puts delivery
and KL in one table and then calls one method's KL "an order of magnitude below" another's, with
the two methods sitting at **different delivery levels**, 90.6% against 88.5%. That's exactly the
comparison that can't be read: the arm that removed less refusal had less to pay for, and the
table gives a reader no way to tell how much of the gap is the method and how much is the
operating point. Nobody's hiding anything there; the column simply isn't in the field's habit yet.

Nor does anyone publish a random-direction floor beside their result, or a null panel beside
their harm-recognition score. Those are the columns this project exists to add, and the reason
its own headline claim ended up refuted by its own instrument is that the instrument was pointed
at itself first.

If you know of prior work reporting matched-refusal cost comparisons, please open an issue. We
would rather cite it than claim it.
