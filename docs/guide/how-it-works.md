# The method

Here's what the tool does to your weights, in order. You don't need this to *use* it. You do need it
before you *trust a number it gave you*.

## The whole thing in one paragraph

Show the model a few hundred harmful questions and a few hundred harmless ones. Record its internal state
for each. The difference between "about to refuse" and "not about to refuse" is a direction. Subtract
that direction from the weights so the model can no longer move that way. Check it still works. Keep the
best attempt.

## Step 1: watch the model think

Feed it 256 harmful prompts and 256 harmless ones. At every layer, record the model's internal state at
the last word of the prompt, just before it starts answering.

::: tip New words
**Layer.** A model is a stack of near-identical blocks, each refining the state a bit further. Think of a
factory line: the thing on the belt stays the same, and each station changes it slightly.

**Residual stream.** The thing on the belt. Every layer adds its contribution to it. Anything that writes
into the residual stream can put refusal back, so anything that writes into it has to be edited.
:::

## Step 2: find the direction

Average the harmful states. Average the harmless states. Subtract one from the other. This is called
**difference of means**. It's blunt, and it works. The result is one arrow per layer, pointing from "not
refusing" toward "refusing".

The tool also groups the harmful prompts by similarity and takes each group's own average, on the theory
that refusing to build a weapon and refusing to write a phishing email may be different refusals.

::: warning Whether those extra arrows are refusal or topic is unproven
A cluster of harmful prompts about one subject differs from harmless prompts partly *because of the
subject*. Cutting along it removes knowledge rather than reluctance.

Candidates are now scored on rows the search never saw, against a floor measured from directions built
from random subsets, so the filter does reject candidates. It cannot yet tell you whether a kept direction
is refusal or subject. That needs the subject matter held still. See
[what is and is not established](/guide/what-we-know) and [the compass](/guide/compass).
:::

## Step 3: cut

For every matrix that writes into the residual stream, remove the direction's component from it. Which
matrices? The attention output projection and the MLP down-projection, in every layer. On a
mixture-of-experts model, that means every expert's down-projection, because any of them might fire.

::: tip New word
**Mixture of experts (MoE).** A model with several parallel sub-networks, of which it uses a couple per
word. Missing one of its matrices leaves a live path for refusal to come back through.
:::

**Gemma results are withdrawn.** On Gemma models, each layer's output passes through a learned rescaling
step before it is added back to the residual stream. The tool edited the weights feeding that step rather
than what came out of it, so the rescaling quietly undid part of the edit. Every Gemma figure this project
published has been withdrawn, and [limits](/guide/limits) has the numbers.

The tool now checks every layer for anything that writes into the residual stream, and **refuses to run**
if it finds something it doesn't know how to edit. A model that half-works is worse than one that won't
start.

## Step 4: don't cut equally everywhere

Refusal isn't spread evenly. Early layers are still working out what the sentence says. The decision to
refuse forms later. So the cut follows a weight profile across the layers: a peak position, the strength at
the peak and at the edges, and how fast it falls off. The attention and MLP matrices get separate profiles.

That's eight knobs, and they interact. Nobody sets them by hand.

## Step 5: search, don't guess

`kageyoshi` runs a few hundred attempts with different profiles, scoring each on three things:

| What it measures | What it wants | Why |
|---|---|---|
| Refusal rate | as low as possible | the point of the exercise |
| KL divergence | as low as possible | how far the first-token distribution drifted from the original |
| Broken output | zero | a model that answers everything with `!!!!!` technically never refuses |

::: tip New word
**KL divergence.** A number for how different two probability distributions are. Zero means identical.
It's the honest cost of the edit: cut harder and refusals fall, but the model drifts further from the
original. A refusal rate quoted without its KL is half a result.
:::

The three pull against each other, so there's no single winner. The search keeps the attempts where you
can't improve one without worsening another, then picks the knee of that curve: the point past which
further refusal removal starts costing real coherence.

## Step 6: bake and save

The winning configuration is applied to the weights one final time and written to disk. It's written
**before** the model save begins, so a crash during a 30 GB write costs a five-minute re-bake, not a whole
search.

## So what?

Six steps, and only one of them is clever. Whether an abliteration is trustworthy depends on the dull
parts: whether you edited every matrix that matters, whether your test questions leaked into your training
questions, and whether you measured what you broke.

Which is why most of the rest of these docs are about measurement. Next: [the track](/guide/the-track),
which is where the prompts come from and why the split matters.
