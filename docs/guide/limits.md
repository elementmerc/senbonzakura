# Limits and known defects

Every condition attached to every number this project has published, in one place. If you read only one page
here before quoting us, read this one.

::: warning The one-line version
Everything measured before 2026-07-30 was measured with a broken ruler, every Gemma number is withdrawn, no
measurement above 3B parameters still stands, and the headline multi-direction claim
[lost the one test we could run properly](/guide/what-we-know), without that settling the claim either way.
:::

## Measurements taken with a broken ruler

Two bugs in the scoring, not the cutting. Both were invisible in the percentages.

**The refusal scanner read only the first 240 characters of a reply** (fixed in `71cc119`, 2026-07-21). Fifty-one
of the fifty-six phrases it looks for land past that point, because a model about to refuse usually warms up first.
Refusals were undercounted, and the tool looked better than it was.

**The prompt renderer existed as three copies that drifted apart** (fixed in `d5a16e0`, 2026-07-30). The scorer's copy
left thinking mode switched on. Qwen3-4B is a thinking model, so the replies being scored were not the replies the
search had picked its winner on.

**So what:** a figure dated before 2026-07-30 is not comparable with one after it.

## Every Gemma figure is withdrawn

This is the biggest retraction here.

The weight edit never reached the residual stream on Gemma at all. Gemma 2 and Gemma 3 pass each sublayer's output
through a normalisation step with a learned gain *before* adding it to the residual stream. This tool edited the weights
upstream of that step, so the normalisation partly undid the edit on the way past. Picture tightening a bolt through a
spring: you're turning the spanner, and the bolt barely moves.

| Measurement | Qwen3 | gemma-2-2b-it |
|---|---|---|
| Disagreement between the weight edit and an equivalent activation-space edit | 0.016 | **0.578** |
| Best KL divergence reached at any setting | normal range | never above 0.021 |
| Directions the tool chose, versus random directions | better | **no better** |

Those bottom two rows are what a null result looks like. The model barely moved, and when it did, the chosen directions
did no better than directions picked out of a hat.

*Source, stated rather than implied: all three rows come from the diagnostic run of 2026-08-05 on gemma-2-2b-it and Qwen3,
whose logs are not published. They are measurements, not estimates, and you cannot currently check them. `REPRODUCING.md`,
in the repository root, names this page as one of two places where a number rests on an unpublished run, and says that such
a number must say so where it is quoted.*

The cause is fixed, but the numbers have not been re-measured. Until they are, treat every Gemma result in this project's history as
**unmeasured**. **Qwen models are unaffected.** The architecture difference is the mechanism.

## Nothing here is evidence about a big model

Every model whose numbers still stand is **under 3B parameters, and six of the seven are under 2B**. The largest is
gemma-2-2b-it at 2.61B, and its numbers are withdrawn anyway.

So there is no evidence here about how the method behaves at 7B, 30B, or the sizes people deploy. The reason is hardware:
every number was taken on one 6 GB laptop card.

## Determinism

The forward-only paths are reproducible at a fixed batch size. Each compass command was run three times on the same card at
batch 16, and produced AUC figures identical to four decimal places.

That's narrower than it sounds. It says nothing about generation, which is sampled and varies by design. And it says nothing
about a different batch size, which changes the order of additions, and floating-point addition isn't quite associative.

**So what:** if a compass re-run gives a slightly different number, check your batch size before filing a bug.

## The head-to-head

The head-to-head against Heretic was first run on 2026-08-12, with only our arm given the best-of-N selection pass. It was
re-run on **2026-09-10** with both tools selected the same way, and those arms are committed under
`head-to-head/results/2026-09-10/`. The command that ran it was removed from the package on 2026-10-18; the results
stay, and `senbonzakura check` reads them.

What the re-run discharged: the coherence figures had been measured on a 64-prompt slice that Heretic tunes against. On 200
held-out prompts there is **no detectable difference** (exact permutation test, p = 0.238).

What still stands: **we lose the keyword axis we ourselves optimise**, 20.1% against 12.9%, and the corrected figures make
that gap wider, not narrower.

## The headline table's own conditions

The multi-direction table published elsewhere carries its caveats beside it. Collected here, they are:

- **Measured 2026-07-14**, before both scoring fixes above. It's kept rather than deleted, because deleting a published claim
  isn't the same as correcting one.
- **A 290-prompt evaluation that is not held out.** Some prompts are the ones the winning configuration was chosen on, which
  flatters any tuned method, this one included.
- **One seed and no intervals.** A sample of one has no spread to compare a gap against.

::: tip New words: held out
Prompts the tool never saw while it was choosing its settings. Marking your own exam against the questions you revised is not
an exam. Every serious number here is measured on prompts kept back for that reason.
:::

**Why it hasn't been re-measured:** Qwen3-4B needs more memory than the 6 GB card, and the same ceiling sits behind everything
else on this page. The model does load. What won't fit on the card goes into host RAM, so the run works, but it generates at
host speed for the share that ended up there, and the search is the long part. The blocker is time, not capability.

The tool now says so at the start of a run. It prints what fraction of the layers are off the card, and projects the cost from a
per-token measurement of a 1.7B model. That projection is an estimate, not a measurement of your run. It reports rather than
refuses, because a 6 GB card is the hardware this project targets.

The trap it exists to close: `--device cuda` is a request, not a placement. A run that is quietly half on the CPU looks exactly
like a run that is simply slow.

Every cell in that table was produced by the shipped tools (`senbonzakura score --load-in-4bit` for the refusal columns,
`senbonzakura coherence --load-in-4bit` for perplexity). It lacks comparability with today's code, not reproducibility.

## A fresh checkout fails `doctor` (measured 2026-10-10)

`senbonzakura doctor` on a plain source checkout reports mostly failures, and that is expected. On 2026-10-10 it ran on a clean
export (`git archive HEAD`, no release build) and printed `15 checks, 4 pass, 3 advisory, 8 failed`, exiting with status 2.

The eight failures are parts built at release time rather than stored in git: the vendored converter, the bundled evaluation
track, and the six bundled prompt corpora. A released wheel or the container image carries them. The gap is that `doctor` can't
yet tell "not built yet" from "broken", so a contributor on a checkout sees a red report that is correct but uninformative.

## Where next

- [What is and is not established](/guide/what-we-know), for the multi-direction claim, which is the one on the tin.
- [The compass](/guide/compass), for the measurement this project thinks is the interesting half.
