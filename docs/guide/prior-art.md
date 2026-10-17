# Who got here first

Short version: **not us.** Refusal as more than one direction has been published at least twice before this
project existed, and one of those papers reports the opposite of what we measured. Being second costs nothing.
Being second and quiet about it ends a review.

## The single direction everything builds on

**Arditi, Obeso et al., [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717)
(2024).** Refusal is largely one direction in activation space. Take the difference between the mean harmful and
mean harmless activations, remove it from the weights, and the model stops refusing. Everything below, including this
project, responds to that paper.

**[Heretic](https://github.com/p-e-w/heretic) by p-e-w** turned it into a tool with an automated search: Optuna over
layer positions and strengths, minimising refusals against KL divergence together. Our search refines that one, and
the keyword metric we report for comparison is copied from it, which is why this project is AGPL.

## The two papers that already said "more than one"

**Wollschläger et al., [*The Geometry of Refusal in LLMs: Concept Cones and Representational Independence*](https://arxiv.org/html/2502.17420v2)
(2025).** Refusal is a *cone*, not a direction. Ablating the top-k representationally independent directions raises
attack success **monotonically with k**, beating the difference-in-means baseline at k ≥ 4 on Gemma 2 2B. The directions
are found by gradient optimisation with an independence penalty, not by clustering.

**Piras et al., [*SOM Directions are Better than One*](https://arxiv.org/html/2511.08379v1) (2025).** A self-organising map
over harmful representations, one direction per neuron, harmless centroid subtracted. The reported gains are large:

| Model | Multi | Single |
|---|--:|--:|
| Llama2-7B | 59.1% | 0.0% |
| Llama3-8B | 88.1% | 15.1% |
| Gemma2-9B | 96.3% | 38.9% |
| Qwen-7B | 88.1% | 81.1% |

## The third idea: don't edit all the time

Everything above, this project included, removes a direction unconditionally. The weights change once, and every prompt
pays the same cost, benign ones included. That trade is what the field argues about: how much refusal you removed
against how much the model moved.

apostate's default method walks around the trade. It fits a detector direction, a threshold and an actuator direction,
and the edit fires only when the detector crosses the threshold, so a benign prompt is meant to be left alone. The
condition is folded into one ordinary MLP neuron, so the result is a plain checkpoint that any runtime loads.

We haven't reproduced it or measured it, and nothing on this site is evidence either way. It's named here because it's a
different idea from the one the papers above argue about, and because this project's instruments are the ones that could
price it. That's a thing to do, not a thing done.

## Why this project measures the opposite

Our own comparison found two directions costing about **1.9 times** the coherence damage of one at matched refusal, and
buying nothing this design could have detected. **The 95% interval on that 1.9 is 1.2 to 3.1**, a percentile bootstrap over
the five seeds a side. The span from the dropped-seed ratio up to the estimate is not an interval, and it is withdrawn.
The withdrawn wording is not reprinted here. See [what is and is not established](/guide/what-we-know) for the figures.

The direction of the result is sturdier than its size: one direction beats two in 24 of 25 pairwise seed comparisons. Both
papers above report multi-direction winning.

::: warning Our arms did not choose their directions, and theirs did
The two sides are not measuring the same experiment. Wollschläger et al. find directions by gradient optimisation with an
independence penalty. Piras et al. fit a self-organising map. Both select deliberately. Our five seeds ran on 2026-08-13,
before held-out direction selection existed here, so the second direction came from a filter now known to accept every
candidate it was given.

So our number prices an *arbitrary* second direction, and theirs price a chosen one. That makes this a different experiment,
not a failed reproduction. Our five seeds stand as the old filter's answer until a re-run under the fixed selector replaces
them. The same caveat is recorded in the evidence file, at
[`evidence/k-sweep-2026-08-13/drift-per-seed.json`](https://github.com/elementmerc/senbonzakura/blob/dev/evidence/k-sweep-2026-08-13/drift-per-seed.json).
:::

Three honest readings of the gap remain, and we can't yet separate them:

1. **They measure a different thing.** Both report *attack success rate*, and **neither reports KL or any coherence cost**.
   "More directions remove more refusal" and "more directions are worth what they cost" are different claims. Only the second
   needs a price tag, and that price tag is the missing column.
2. **Our models are in the regime their own numbers say benefits least.** In the SOM table the gap is enormous where
   single-direction ablation barely works (Llama2-7B, 0.0%) and small where it already works (Qwen-7B). Everything we have
   measured is under 3B and weakly aligned, which is the right-hand column.
3. **We may simply be doing it worse.** Our directions are orthogonalised against the primary one, which removes the large
   shared component by construction: geometrically pure, possibly functionally small. [A paper argues exactly that mechanism](https://arxiv.org/pdf/2603.22061)
   for topic-matched contrasts, and it may explain our result rather than a fact about refusal.

We publish our number with all three readings attached. See [what is and is not established](/guide/what-we-know).

## The tools

**[Abliterix](https://github.com/wuwangzhang1216/abliterix)** is the closest direct competitor. It overlaps us heavily: Optuna
search, Pareto co-minimisation of refusals against KL, reversible edits. It is **ahead of us** on method breadth (several named
multi-direction methods we have not implemented), on mixture-of-experts handling, and on prebuilt configurations.

**[apostate](https://github.com/heterodoxin/apostate)** by `heterodoxin`, MIT, read here at `main` on 2026-10-01. It is four tools
behind one command, and only the oldest is a variant of the method everything above shares. Its default, from `apostate ablate`,
isn't a projection at all. It has no CI, no lockfile and no changelog, and it ships its harmful prompt set in plaintext in the
tree, which this project deliberately doesn't do. This read was partly through a summarising reader rather than a clone, so the
[comparison page](/comparison) gives it a scope line rather than a column.

**["Exploring the multi-dimensional refusal subspace"](https://www.lesswrong.com/posts/ixJrmYrgHM4TenN7D/exploring-the-multi-dimensional-refusal-subspace-in-1)**
reached this project's clustering approach independently. It says plainly that it has no random-direction baseline, no
cross-topic generalisation test and no KL matching. It clusters prompt *text*. We cluster *residuals*.

## What is left for us

Not the search: Abliterix has one. Not multi-direction: two papers got there first.

**The measurement.** As far as we can find, nobody publishes **KL at matched refusal removal**: the cost of the edit, compared
at an operating point where both arms removed the same amount of refusal. Without matching first, the arm that cut harder looks
worse on coherence for a reason unrelated to what was tested.

apostate shows why that claim is worth making. Its README puts delivery and KL in one table, then calls one method's KL "an order
of magnitude below" another's, with the two methods at **different delivery levels**, 90.6% against 88.5%. The arm that removed
less refusal had less to pay for, and the table gives no way to tell how much of the gap is the method and how much is the
operating point. Nobody is hiding anything. The column simply isn't in the field's habit yet.

Nor does anyone publish a random-direction floor beside a result, or a null panel beside a harm-recognition score. Those are the
columns this project exists to add.

If you know of prior work reporting matched-refusal cost comparisons, please open an issue. We would rather cite it than claim it.
