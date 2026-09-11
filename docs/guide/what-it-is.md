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

The name came from an idea this project started with: that the last stubborn few percent of
refusals live in a small bundle of *nearby* directions a single blade never touches, so cutting in
several at once would take the rest without damaging the model further.

We built that, and then we measured it, and it did not hold. Cutting in more directions cost more
and bought nothing on the one model we can measure properly. The capability stays in the tool,
because being able to test an idea is worth more than believing it, and the name stayed too.
[What is and is not established](/guide/what-we-know) has the measurement, the conditions on it,
and the honest limits of the negative result.

That is the short version of why this project is the way it is. An idea that survives its own
measurement is worth something; one that does not is worth saying so about.

## So what is it for?

**The best-measured abliteration we can manage.** That is the aim, and everything else here is in
service of it.

Removing refusals is the part that has worked since the beginning: the single-direction method of
Arditi et al., plus an automated search for how hard to cut and where, building on Heretic's. It
is solid and it is not novel.

What is meant to be worth your attention is that every number this prints can be checked. The
evaluation split is three-way, so the rows a configuration is chosen on are never the rows it is
reported on. Every figure carries an interval. Every figure arrives beside a control that would
expose it if it were measuring something else, the sharpest being a ruler that reads nothing but
sentence length.

The instruments that do that are features rather than the point:

- [the compass](/guide/compass), which asks whether the edited model still *recognises* harm as
  opposed to still refusing it, because those are different things and one is much worse to lose;
- `capability`, for what the edit cost on tasks with a right answer, which refusal rates and KL
  cannot see;
- `drift`, one coherence ruler that can be pointed at a model edited by any tool, including
  somebody else's;
- `validate`, which asks whether a direction set carries refusal or carries topic.

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
