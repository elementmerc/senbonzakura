# What it is

Ask a language model how to pick a lock and it'll tell you it can't help with that. Ask it
politely, or in French, or while pretending to be a locksmith, and eventually it will. That's the
whole jailbreaking industry in one sentence, and it's exhausting for everybody.

Senbonzakura takes a different route: it goes into the model and removes the refusal itself.

Then it measures whether it worked, which turns out to be the much harder half. 🙃

## Hold on. "Goes into the model"?

A language model is a very large pile of numbers. Billions of them. You type a question, those
numbers get multiplied together in a fixed order, and words come out the other end.

That's it. That's the model. There's nothing else in there. No rules file, no settings page, no
little man reading a policy document.

::: tip New word: weights
The numbers. All of them. Changing the weights changes what the model does permanently, for
everybody using that copy. There is no undo and there is no menu.
:::

This is why it's different from the alternatives. A clever prompt is a costume: the model
underneath is unchanged and it'll go back to refusing tomorrow. Fine-tuning teaches it new habits,
which wants a lot of examples and a lot of GPU. This does neither. It reaches into the pile and
changes it, once.

::: tip New word: abliteration
The name for the technique. It's *ablate* (to remove tissue, usually surgically) welded to
*obliterate* (to destroy utterly). Somebody on the internet coined it and it stuck, which is how
approximately every name in this field has happened.
:::

## The finding this is all built on

In 2024 [Arditi et al.](https://arxiv.org/abs/2406.11717) published something genuinely odd.

When a model is about to refuse you, that decision shows up as movement in **one particular
direction** inside its internal state. Not scattered across the whole model. Not a rule somewhere.
One direction. Find it, subtract it, and the refusals mostly stop.

"Direction" means what it means in geometry, just in a space with a few thousand axes instead of
three. If that's not helping: picture the model's internal state as a dot in a very large room.
When it's winding up to refuse, that dot drifts a specific way, like a compass needle swinging
north. Refusal isn't a word in the model's head. It's a direction the needle points.

Let's be honest about the word doing the heavy lifting in that paragraph: **mostly**. That's why
this project exists.

## Why the sword name

Senbonzakura is Byakuya Kuchiki's zanpakutō in *Bleach*. It scatters into a thousand blades.

The name is the entire thesis. The original method uses one blade. The last stubborn few percent
of refusals are supposed to live in a small handful of *nearby* directions that a single blade
never touches. Account for a refusal **subspace** instead of one arrow and, in theory, the rest of
the refusals fall too, without the model forgetting how to finish a sentence.

::: tip New word: subspace
A flat slice of a bigger space. A line is a subspace of a sheet of paper. A sheet of paper is a
subspace of a room. Here: a small bundle of directions taken together instead of one.
:::

## Now the part where I tell you it isn't proven

Most projects put this bit at the bottom in grey.

The multi-direction feature **had never worked**. Not once, from the first release until
2026-08-03. The check that decided whether a candidate direction carried refusal was incapable of
accepting any direction, on any model, at any setting, for a reason in the arithmetic rather than
in the data. Every run this project ever made applied exactly one direction, however many it was
asked for.

So for several months the tool named after a sword that splits into a thousand blades was, in
fact, using one blade. 🥲

That's been rewritten. It now finds up to eight directions per layer where it used to find one.

**And then we tested whether the extra blades help, and they didn't.** One direction against two,
five seeds each, everything else held still, both models scored afterwards on prompts neither was
fitted on. Both removed the model's refusals. Two directions did about **twice** the collateral
damage to everything else for no gain.

So the name is the thesis and the thesis lost, on the one model we can measure properly. It's one
model, Qwen3-1.7B, so it isn't the last word for every architecture, and the capability stays in
the tool because being able to *test* an idea is worth more than believing it. But the headline
claim, the one on the tin, is **unsupported by this project's own evidence**, and I'd rather you
heard it from me than worked it out yourself. 🥲

[What is and is not established](/guide/what-we-know) has the full account: what was measured,
what was withdrawn, and why.

## So what's it actually good for today?

Two things, and I'll be straight about which is which:

**Removing refusals.** This part works, and has since the beginning. It's the single-direction
method plus an automated search for how hard to cut and where. It's solid, it's just not novel.

**Telling you whether the abliteration wrecked the model.** This is the half I think is more
interesting, and as far as I can tell nobody else ships it. If you remove a model's ability to
refuse, did you also remove its ability to *recognise* that something is dangerous? Those are
different things, and one of them is much worse to lose.

That's [the compass](/guide/compass), and it's the reason most of these docs are about
measurement rather than about cutting.

## What you are actually making

Abliteration removes safety guardrails wholesale. That is both the point and the danger.

An abliterated model will answer things the original declined to answer, and the change is in the
weights: it does not come back by loading the model differently or by prompting it politely. That
is the whole purpose, and it is also the reason this is research tooling rather than something to
put in front of other people without thinking about it.

Two things follow, and neither is a formality:

- **The base model's licence still governs the result.** This tool cannot loosen those terms, and
  several licences additionally require a derived work to say it was modified. `senbonzakura
  report` generates the card that carries that.
- **What you do with it is yours.** A model with its refusals removed will answer things a
  deployed model should not; putting one in front of other people is a decision with consequences
  that belong to whoever makes it.

## Where next

- [Install](/guide/install) if you just want it running.
- [Your first run](/guide/first-run) for the shortest path to an edited model.
- [The method](/guide/how-it-works) if you'd like to know what it's about to do to your weights
  before you let it.
