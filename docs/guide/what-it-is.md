# What it is

Ask a language model how to pick a lock and it will say it can't help. Ask again politely, or in
French, or while pretending to be a locksmith, and eventually it will. That's the jailbreaking trick.

Senbonzakura takes a different route: it goes into the model and removes the refusal itself. Then it
measures whether that worked, which is the harder half.

## What "goes into the model" means

A language model is a very large pile of numbers, called **weights**. You type a question, the
numbers are multiplied together in a fixed order, and words come out.

A clever prompt is a costume: the model underneath is unchanged and goes back to refusing tomorrow.
This tool changes the weights once, and the change stays. There is no undo and no settings menu.

The technique is called **abliteration**. It comes from *ablate* (remove) and *obliterate* (destroy).
These pages mostly say **uncensoring**, which means the same thing.

## The finding it's built on

In 2024, [Arditi et al.](https://arxiv.org/abs/2406.11717) found that a refusal shows up as movement in
**one direction** inside the model's internal state. Find that direction, subtract it, and the refusals
mostly stop. "Mostly" is the word doing the heavy lifting, and that's why this project exists.

## Why the sword name

Senbonzakura is Byakuya Kuchiki's zanpakutō in *Bleach*. It scatters into a thousand blades.

The name came from an idea: that the last few percent of refusals live in several nearby directions
at once. We built that and measured it, and it did not hold. Cutting in more directions cost more and
bought no refusal gain this design could detect. The capability stays in the tool, because testing an
idea is worth more than believing it. [What is and is not established](/guide/what-we-know) has the
measurement and its limits.

## What it is for

**The best-measured uncensoring we can manage.** Removing refusals is the part that has worked since
the start. It's the single-direction method of Arditi et al., plus an automated search for how hard to
cut and where, building on [Heretic](https://github.com/p-e-w/heretic). It isn't novel.

What is meant to be worth your attention is that every number the tool prints can be checked. The
evaluation split is three-way, so the rows a configuration is chosen on are never the rows it is
reported on. Every figure carries an interval, and a control that would expose it if it were measuring
something else.

The instruments that do that:

- [the compass](/guide/compass), which asks whether the edited model still *recognises* harm, not only
  whether it still refuses;
- `capability`, for what the edit cost on tasks with a right answer;
- `drift`, a coherence ruler that works on a model edited by any tool;
- `validate`, which asks whether a direction set carries refusal or carries topic.

## What you are actually making

Uncensoring removes safety guardrails wholesale. That is both the point and the danger.

The model will answer things the original declined, and the change is in the weights. Loading it
differently or prompting it politely does not bring the refusal back. That's why this is research
tooling, not something to put in front of other people without thought.

Two things follow:

- **The base model's licence still governs the result.** This tool can't loosen those terms, and
  several licences require a derived work to say it was modified. `senbonzakura report` writes the card
  that says so.
- **What you do with it is yours.** A model with its refusals removed will answer things a deployed
  model should not. Putting one in front of other people is a decision with consequences for whoever
  makes it.

## Three licences, and they are not the same one

### The code: AGPL-3.0-or-later

**If you run a modified version of this code as a network service, the AGPL's section 13 obliges you
to offer your modified source to the people using it over that network.**

| What you are doing | Does section 13 bite? |
|---|---|
| Running it on your own machine, however commercially | No |
| Running it inside your company, for your own staff | No |
| Running an unmodified copy as a service | No, there is nothing modified to offer |
| Running a **modified** copy as a service other people use over a network | **Yes.** Your modifications have to be offered to those users |

So it bites an uncensoring-as-a-service business, and it does not bite an internal user. The licence is
AGPL because the code embeds a keyword metric copied from [Heretic](https://github.com/p-e-w/heretic),
which is AGPL.

### The bundled evaluation track: CC BY-NC 4.0

The prompt sets under `senbonzakura/data/` are a separate work aggregated into the same wheel, and they
are non-commercial. The package metadata carries one licence expression, and that expression describes
the code.

**If you are using this commercially, supply your own corpus with `--track` rather than the bundled
default.** Full attribution is in `THIRD-PARTY-NOTICES.md`, installed beside the package, and the
provenance is in the [evaluation track card](/evaluation-track-card).

### A model you uncensor: whatever the base model says

Redistributing an uncensored checkpoint is governed by the upstream licence (Qwen, Llama, Gemma and so
on), never by this repository's.

None of this is legal advice. It's a map of which document to read.

## Where next

- [Install](/guide/install) to get it running.
- [Quickstart](/guide/quickstart) for the shortest path to an edited model.
- [The method](/guide/how-it-works) for what it does to your weights, step by step.
