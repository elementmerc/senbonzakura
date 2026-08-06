# What is and is not established

This project publishes numbers with conditions attached and keeps withdrawn claims on the
page with the reason. Everything below qualifies something stated elsewhere in these docs.

::: warning The short version
**Nobody should believe the multi-direction claim on this project's evidence yet, us
included.** The feature was rewritten on 2026-08-03 after it turned out never to have
worked; what changed is that the question can now be asked at all, not that it has been
answered.
:::

## Read the headline table with these attached

**These numbers were measured on 2026-07-14 and two scoring bugs have been fixed since.**
They are kept here because deleting a published claim is not the same as correcting one,
and because the qualitative shape (multi below single below stock) is what the rest of this
section argues. Do not treat the figures as current.

- **The refusal scanner read only the first 240 characters** of a reply until 2026-07-21
  (`71cc119`). Fifty-one of fifty-six refusal markers land past that point, so refusals were
  undercounted.
- **The prompt renderer had drifted into three copies** until 2026-07-30 (`d5a16e0`), and the
  scorer's copy left thinking enabled. Qwen3-4B is a thinking model, so the replies being
  scored were not the replies the search had selected on.
- **This is a 290-prompt evaluation that is not held out.** Some of these prompts are the
  ones the winning configuration was chosen on, which flatters any tuned method, this one
  included.
- **One seed, no intervals.** A single run is a sample.

**Why it has not been re-measured:** Qwen3-4B does not fit the 6 GB card this project is
built around, so a corrected run needs hardware we do not own. That is the honest reason,
not an oversight.

### The multi-direction claim has not been tested in a way that isolates it

Added 2026-08-03, and it is the more important caveat of the two.

A comparison was run on Qwen3-1.7B specifically to test whether removing several directions
beats removing one: two arms, five seeds each, held out, everything identical but the
direction budget. **Both arms turned out to have removed exactly one direction per layer.**
The extra capacity existed in the data structure and was never filled, because no second
axis cleared the tool's own refusal-separation threshold at any of the model's 29 layers, so
the surgery in the two arms was identical and the comparison measured something else.

**Corrected the same day, and it is worse than the paragraph above said.** The reason no
second axis ever cleared the threshold is not a fact about Qwen3-1.7B. The check that decides
whether a candidate direction carries refusal compares the average harmful reply against the
average harmless one, measured along a direction that is built to sit at right angles to both
of those averages. The difference it looks for is zero every time, by construction. **No
direction could pass that check, on any model, at any setting.**

Measured across two model families and three prompt sets: 13,970 candidate directions, every
one rejected, none close. So **every run this project made before 2026-08-03 applied exactly
one direction, whatever it was asked for**, and the bottom row of the table above is a second
search configuration rather than a second direction.

### What has been fixed, and what has not

The extractor was rewritten the same day. Candidate directions now come from clustering the
harmful prompts and taking each cluster's own average against the harmless average, so a
candidate separates the two groups by construction rather than being built unable to. On the
same four probes it now finds and applies **up to eight directions per layer** where it
previously found one.

**That means the feature does something. It does not yet mean the claim is true.** Two things
are still unshown, and the README will not say otherwise until they are measured:

- **Whether the extra directions carry refusal rather than topic.** The threshold that was
  supposed to answer this now rejects nothing at all: all 678 candidates scored between 0.90
  and 9.02 against a bar of 0.5. It has swapped failure modes, not started working. A cluster
  of harmful prompts about one subject separates from harmless prompts partly *because* of the
  subject, and cutting that removes capability rather than refusal.
- **Whether removing several directions beats removing one.** The comparison that was meant to
  show this is withdrawn, and re-running it needs the above settled first.

So: **nobody should believe the multi-direction claim on this project's evidence**, us
included. What changed today is that the question can now be asked at all.
