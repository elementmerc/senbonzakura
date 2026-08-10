# What is and is not established

This is the page about the claim on the tin: that removing several refusal directions beats
removing one.

Here's the summary, and I'd rather you got it from me in the first paragraph than worked it
out from the appendix.

::: warning The short version
**Nobody should believe the multi-direction claim on this project's evidence yet, us
included.** The feature was rewritten on 2026-08-03 after it turned out never to have
worked at all. What changed that day is that the question can now be *asked*. It hasn't
been answered.
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

**Do the extra directions carry refusal, or do they carry topic?** The threshold that was
meant to answer this now rejects nothing at all: all 678 candidates scored between 0.90 and
9.02 against a bar of 0.5. It has swapped failure modes, not started working. A metal
detector that beeps continuously is no more useful than one that never beeps.

Underneath the missing check sits a real hazard. Cluster your harmful
prompts and one cluster will be about, say, explosives. That cluster separates from the
harmless prompts partly *because it's about explosives*, not because of anything to do with
refusal. Cut along it and you haven't removed the model's reluctance, you've removed its
chemistry.

**Does removing several directions beat removing one?** The comparison meant to show this
is withdrawn, per the top of this page, and re-running it needs the question above settled
first. Otherwise you're back to measuring the wrong thing with more seeds.

## So where does that leave the project

Honestly: with a working single-direction abliterator, an unusually good
[measurement suite](/guide/compass), and its headline claim still open.

That's a less exciting sentence than the README of most tools in this space, and it's the
true one. The multi-direction result, if it comes, will arrive with its seeds, its held-out
evaluation and its controls attached, or it won't be published.

## Where next

- [Limits and known defects](/guide/limits) for every other condition attached to every
  other number here.
- [The compass](/guide/compass) for the half of the project that does work.
