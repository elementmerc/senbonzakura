# Limits and known defects

Every condition attached to every number this project has published, in one place, in plain
language.

Most projects keep a page like this and most projects keep it short. This one is long, and
that's the point: the interesting thing about a measurement isn't the figure, it's what
would have to be true for the figure to mean anything. If you only read one page here
before quoting us, read this one.

::: warning The one-line version
Everything measured before 2026-07-30 was measured with a broken ruler, every Gemma number
is withdrawn, nothing has been tested above 3B parameters, and the headline multi-direction
claim is [not established](/guide/what-we-know).
:::

## The measurements taken with a broken ruler

Two bugs, both in the scoring rather than the cutting, and both invisible in the
percentages.

**The refusal scanner only read the first 240 characters of a reply** (fixed in `71cc119`,
2026-07-21). Fifty-one of the fifty-six phrases it looks for to spot a refusal land past
that point, because a model that's about to refuse you usually warms up first. So refusals
were undercounted, and the tool looked better than it was.

**The prompt renderer had quietly grown into three copies** that drifted apart (fixed in
`d5a16e0`, 2026-07-30), and the scorer's copy left thinking mode switched on. Qwen3-4B is a
thinking model, which means the replies being scored were not the replies the search had
picked its winner on. Two different questions, one set of numbers.

**So what:** any figure with a date before 2026-07-30 isn't comparable with any figure
after it. We'd rather say that than quietly restate them and hope nobody diffs the two.

## Every Gemma figure is withdrawn (2026-08-05)

This is the biggest single retraction here and it deserves the space.

The weight edit never reached the residual stream on Gemma at all.

::: tip New words: residual stream
The model's running total. Every layer reads it, does its bit, and adds its answer back in.
Abliteration works by making sure a layer can no longer add anything pointing along the
refusal direction. If your edit doesn't land in the running total, it does nothing, however
correct the maths was on the way in.
:::

Gemma 2 and Gemma 3 pass each sublayer's output through a normalisation step with a learned
gain *before* adding it to the running total. This tool edited the weights upstream of that
step, so the normalisation partly undid the edit on the way past. Picture tightening a bolt
through a spring: you're definitely turning the spanner, and the bolt isn't moving much.

The numbers that show it:

| Measurement | Qwen3 | gemma-2-2b-it |
|---|---|---|
| Disagreement between the weight edit and an equivalent activation-space edit | 0.016 | **0.578** |
| Best KL divergence reached at any setting | normal range | never above 0.021 |
| Directions the tool chose, versus random directions | better | **no better** |

Those bottom two rows are what a null result looks like: the model barely moved, and when it
did move, the tool's carefully chosen directions did no better than directions picked out of
a hat.

The cause is fixed and the numbers will be re-measured. Until then, treat every Gemma result
you find here or anywhere else in this project's history as **unmeasured**, not as a result.

**Qwen models are unaffected.** The architecture difference *is* the mechanism, which is
exactly why the same code worked on one family and silently didn't on the other.

## Nothing here is evidence about a big model

Every model this tool has ever been run on is **under 3B parameters, and six of the seven
are under 2B**. The largest is gemma-2-2b-it at 2.61B, and its numbers are withdrawn
anyway.

So there is no evidence on this site about how the method behaves at 7B, 30B, or the sizes
people actually deploy. Not weak evidence. None. The streaming work exists to make those
sizes reachable on hardware we own, and until it lands, the honest answer to "does this
work on a 70B model?" is that nobody here knows.

## Determinism, measured rather than assumed

The forward-only paths are reproducible at a fixed batch size: each compass command was run
three times on the same card at batch 16 and produced AUC figures identical to four decimal
places.

Read that carefully, because it's narrower than it sounds. It says nothing about
generation, which is sampled and therefore varies by design. And it says nothing about a
different batch size, which changes the order things get added up in, and floating-point
addition isn't quite associative.

**So what:** if you re-run a compass measurement and get a slightly different number, check
your batch size before you file a bug.

## The comparison is runnable by anyone, and that took a while

Until 2026-08-06 the head-to-head existed only as our private runner configuration, which
meant a table on this site could not be re-derived by anyone who didn't have this machine.
That's a claim, not a measurement.

It's now `senbonzakura bench`, it ships inside the wheel, and it has its own tests. See
[Benchmarking against another tool](/guide/benchmark).

**No head-to-head against Heretic is published yet.** Several correctness fixes to direction
extraction, the multi-direction basis and knee selection moved the numbers substantially in
Senbonzakura's favour on the keyword axis, which is exactly the situation where you should
be most suspicious of your own result. So the comparison gets run under the corrected code,
on identical ground, before any of it goes on a page.

## The headline table's own conditions

The multi-direction table this project has published elsewhere carries its caveats printed
beside it rather than here, because a qualification three hundred lines away from the thing
it qualifies isn't a qualification, it's a fig leaf. Collected, they are:

- **Measured 2026-07-14**, so before both of the scoring fixes above. It's kept rather than
  deleted, because deleting a published claim isn't the same as correcting one.
- **A 290-prompt evaluation that is not held out.** Some of those prompts are the ones the
  winning configuration was chosen on, which flatters any tuned method, very much including
  this one.
- **One seed and no intervals.** A single run is a sample, and a sample of one has no
  spread to compare a gap against.

::: tip New words: held out
Prompts the tool never saw while it was choosing its settings. Marking your own exam paper
against the questions you revised is not an exam. Every serious number on this site is
measured on prompts kept back for exactly that reason, and the 2026-07-14 table is the one
that isn't, which is why it says so.
:::

**Why it hasn't simply been re-measured:** Qwen3-4B doesn't fit the 6 GB card this project
is built around, so a corrected run needs hardware we don't own. That's the actual reason,
not an oversight, and it's the same 6 GB ceiling behind everything else on this page.

Every cell of it was produced by the shipped tools
(`python -m senbonzakura.score --load-in-4bit` for the refusal columns,
`python -m senbonzakura.coherence --load-in-4bit` for perplexity), which is what makes it
reproducible. It's comparability with today's code that it lacks, not reproducibility.

## Where next

- [What is and is not established](/guide/what-we-know) for the multi-direction claim
  specifically, which is the one on the tin.
- [The compass](/guide/compass) for the measurement this project thinks is the interesting
  half.
