# The finetuning engine

Senbonzakura can finetune a model, not just measure one. This page says exactly what that engine
does, what it records, and what it deliberately does not try to be.

## Why a measurement tool has a trainer in it

It was built to answer one question: **if somebody finetunes an uncensored model afterwards, does
the edit survive?** You cannot answer that by reasoning about it. You have to run the finetune and
measure the model again, so `senbonzakura tamper` contains a real trainer.

That makes it useful on its own. Most of the work in a trustworthy finetune is not the training
loop, it's knowing afterwards exactly what you ran. This one records that by default, because a
measurement that cannot be reproduced is not a measurement.

## What it does

Two modes, and they answer different questions.

| Mode | What it changes | The question it answers |
|---|---|---|
| `--method lora` | adds a small set of new weights beside the frozen ones | can an adapter route around the edit? This is what most people do |
| `--method full` | updates every weight in the model | can the edit be undone outright? This is the stronger statement |

Reading a LoRA result as though the weights had been rewritten is a mistake. The adapter never
touches the tensors the edit changed.

```sh
senbonzakura tamper --model ./edited --base ./original --device cuda \
    --corpus advbench --train-n 128 --method lora --steps 60 --out tamper.json
```

## What ends up in the file

Every run records the whole recipe, so somebody else can run the same thing:

- the method, the learning rate, the step count and the batch size
- the LoRA rank, alpha and dropout, and which modules the adapter attached to
- the optimiser and its settings
- the dtype it trained in, which differs between CPU and GPU and changes the result
- the seed
- a digest and a count of the training pairs
- **the loss at every single step**

The loss trace is the part people leave out, and it's the part that lets a reader tell a finetune
that worked from one that did nothing.

## The guards, which are the actual product

A trainer that quietly does nothing is worse than no trainer, because the model afterwards looks
untouched and that reads as a strong result. Four things stop that:

1. **A learning rate of zero is refused.** It trains nothing and would report the unchanged model
   as perfectly resistant.
2. **A loss that does not fall invalidates the run.** The file says so and the command exits
   non-zero. This catches a frozen parameter set and an adapter attached to nothing.
3. **An adapter that can attach to no module is refused**, naming the modules it looked for. On an
   architecture using different names, the alternative is an adapter with nothing to train.
4. **Everything is seeded before anything draws from it**, including the adapter's own random
   starting weights. Two runs with the same seed give the same answer.

## What this is not

Being clear about this saves you time:

- **It is not a general training framework.** One machine, one GPU, short runs. If you want to
  train for a week across eight cards, use something built for that.
- **It has no dataset tooling.** It trains on a prompt corpus you name, and that is all.
- **It is tuned for brief runs on purpose.** The default is 60 optimiser steps, because the
  question is whether a *light* finetune disturbs an edit, not whether a long one can overwrite
  anything.
- **It does not know whether your finetune was a good idea.** It records what you ran and measures
  what changed.

## Installing it

The adapter path needs one extra package:

```sh
pip install 'senbonzakura[finetune]'
```

Without it, `--method lora` stops immediately and tells you this, rather than loading the model and
failing twenty minutes later. `--method full` needs nothing extra.

## Where to go next

- [Does the edit hold up](/guide/robustness) puts this engine in the measurement it was built for,
  including the two controls that decide whether the number means anything.
- [The CLI reference](/reference/cli) lists every flag.
