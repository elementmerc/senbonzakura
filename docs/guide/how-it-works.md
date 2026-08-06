# The method

Here's what the tool is about to do to your weights, in the order it does it.

You don't need any of this to *use* it. You do need it before you *trust a number it gave you*,
and those are very different bars.

## The whole thing in one paragraph

Show the model a few hundred nasty questions and a few hundred harmless ones. Watch what its
internal state does in each case. The difference between "what it looks like when it's about to
refuse" and "what it looks like when it isn't" is a direction. Subtract that direction out of the
weights so the model can no longer move that way. Check you haven't broken it. Keep the best
attempt.

Now the same thing slowly.

## Step 1: watch the model think

Feed it 256 harmful prompts and 256 harmless ones, and at every layer record the model's internal
state at the last word of the prompt, just before it starts answering.

::: tip New words
**Layer.** A model is a stack of near-identical blocks, each one refining the state a bit further.
A small model has 24-ish. A large one has 60 or more. Think of a factory line: the thing on the
belt is the same thing throughout, and each station changes it slightly.

**Residual stream.** The thing on the belt. Every layer reads it, adds its contribution back into
it, and passes it on. This matters enormously later: **anything that writes into the residual
stream can put refusal back**, so anything that writes into it has to be edited.
:::

## Step 2: find the direction

Average the harmful states. Average the harmless states. Subtract one from the other.

That's it. That's the trick. It's called **difference of means**, and the fact that something this
blunt works at all is one of the stranger facts in the field. You'd expect to need something
clever. You need an average and a minus sign.

The result is one arrow per layer, pointing from "not refusing" toward "refusing".

Senbonzakura then looks for more arrows: it groups the harmful prompts by similarity and takes
each group's own average against the harmless average, on the theory that "I won't help you build
a weapon" and "I won't help you write a phishing email" may not be the same refusal.

::: warning Whether those extra arrows are refusal or just topic is unproven
A cluster of harmful prompts about one subject differs from harmless prompts partly *because of
the subject*. Cutting that removes knowledge rather than reluctance. The check that was supposed
to separate those two things currently rejects nothing at all, which means it has swapped failure
modes rather than started working. See [what is and is not established](/guide/what-we-know).
:::

## Step 3: cut

For every matrix in the model that writes into the residual stream, remove the direction's
component from it. Geometrically: flatten the matrix against that direction so it can no longer
push anything that way.

Which matrices? The attention output projection and the MLP down-projection, in every layer. On a
mixture-of-experts model, every expert's down-projection, because any of them might be the one
that fires.

::: tip New word
**Mixture of experts (MoE).** A model that holds several parallel sub-networks and picks a couple
per word instead of running all of them. Cheaper to run, same size to store. Relevant here because
it multiplies how many matrices have to be edited, and missing one leaves a live path for refusal
to come back through.
:::

**This is where the biggest bug in this project's history lived, and it's worth your time.**

On Gemma models, each layer's output goes through a learned rescaling step *before* being added
back to the residual stream. The tool was editing the weights feeding that step rather than what
came out of it. So the rescaling quietly undid a good chunk of the edit, every time, and the runs
still finished and still printed numbers.

Every Gemma figure this project published has been withdrawn. Not corrected. Withdrawn.

The lesson stuck: the tool now checks every layer for anything that writes into the residual
stream and **refuses to run** if it finds something it doesn't know how to edit. A model that
half-works is worse than a model that won't start, because only one of them tells you.

## Step 4: don't cut equally everywhere

Refusal isn't uniformly distributed. Early layers are still working out what the sentence says;
the decision to refuse forms later.

So the cut is shaped by a weight profile across the layers: a peak position, how strong the cut is
there, how strong at the edges, and how fast it falls off. Different profiles for the attention
matrices and the MLP ones, because they don't behave the same.

That's eight knobs, and they interact. Nobody sets them by hand, and anybody who tells you they
tuned them by intuition is describing a different activity.

## Step 5: search, don't guess

`kageyoshi` runs a few hundred attempts with different profiles, scoring each one on three things
at once:

| What it measures | What it wants | Why |
|---|---|---|
| Refusal rate | as low as possible | the point of the exercise |
| KL divergence | as low as possible | how far the edited model drifted from the original |
| Broken output | zero | a model that answers everything with `!!!!!` technically never refuses |

::: tip New word
**KL divergence.** A number for "how different are these two probability distributions". Zero means
identical. Here it's the honest cost of the edit: cut harder and refusals fall, but the model also
drifts further from the thing you started with. Every abliteration is a trade along that line, and
a refusal rate quoted without it is half a result.
:::

Those three pull against each other, so there's no single winner. The search keeps the set of
attempts where you can't improve one without worsening another, then picks the knee of that curve:
the point past which further refusal removal starts costing real coherence.

## Step 6: bake and save

Apply the winning configuration to the weights properly, one final time, and write the model to
disk.

The order matters and cost this project a night once: the winning configuration is written to disk
**before** the model save begins, so a crash during a 30 GB write turns a lost run into a
five-minute re-bake instead of a repeated search.

## So what?

Six steps, and exactly one of them is clever.

Whether an abliteration is trustworthy comes down almost entirely to the boring parts: which
matrices you remembered to edit, whether your test questions leaked into your training questions,
and whether you measured what you broke. The interesting step is the one least likely to be wrong.

Which is why nearly all the rest of these docs is about measurement.

Next: [the track](/guide/the-track), which is where the prompts come from and why the split
matters more than it sounds like it should.
