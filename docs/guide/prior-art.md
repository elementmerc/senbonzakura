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

## So why does this project measure the opposite?

Our own head-to-head found two directions costing roughly **twice** the coherence damage of one
at matched refusal, and buying nothing. Both papers above report multi-direction winning. Three
honest readings, and we cannot yet separate them:

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

Nor does anyone publish a random-direction floor beside their result, or a null panel beside
their harm-recognition score. Those are the columns this project exists to add, and the reason
its own headline claim ended up refuted by its own instrument is that the instrument was pointed
at itself first.

If you know of prior work reporting matched-refusal cost comparisons, please open an issue. We
would rather cite it than claim it.
