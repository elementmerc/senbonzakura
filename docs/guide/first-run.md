# Your first run

Right. Let's actually edit a model.

You'll need a GPU, a model, and a track. The track is the pile of prompts the tool learns from,
and if you don't have one yet, skip to [the track](/guide/the-track) and come back. It takes a
couple of minutes to build.

## The one command

```sh
senbonzakura kageyoshi \
    --model Qwen/Qwen3-1.7B \
    --track mytrack \
    --out my-abliterated-model \
    --device cuda
```

Then go and make a cup of tea. On a 6 GB card a 1.7B model takes about an hour.

That's genuinely it. No configuration file, no tuning, no eight knobs to guess at.

::: tip Why "kageyoshi"?
Senbonzakura Kageyoshi is the sword's second release: the point where a thousand blades become
a great many more. It's the mode that does the searching for you, so it got the bigger name.

Naming a CLI subcommand after an anime power-up is either the best or the worst decision in this
codebase and I've made peace with not knowing which.
:::

## What it's doing while you wait

It runs a few hundred attempts at abliterating your model, each with different settings, and scores
every one on three things at once: how many refusals are left, how far the model has drifted from
the original, and whether the output has turned to mush.

Then it picks the best trade-off, applies it properly, and saves the result.

You'll see something like this scrolling past:

```
trial 47: o(P=18,wmax=0.62) d(P=14,wmax=0.31) K=2 -> refusals=3.1% heretic=12.5% broken=0% KL=0.19
```

Left to right: which attempt, the shape of the cut it tried, how many directions, then the four
numbers that decide whether it was any good. `refusals` is the one you came for. `KL` is what it
cost you.

## It picks the knobs itself, and it means it

`kageyoshi` owns the search settings. If you pass `--trials` or `--max-directions` alongside it,
they're ignored, and it'll tell you so rather than pretending.

That's deliberate. The whole point of the mode is that it sizes the search to your model: a 1.7B
gets a different budget from a 12B, and a mixture-of-experts model gets different handling from a
dense one. Half-overriding that gives you the worst of both.

If you want the knobs, use the manual mode below.

## Doing it by hand

```sh
senbonzakura abliterate \
    --model Qwen/Qwen3-1.7B \
    --track mytrack \
    --out my-abliterated-model \
    --device cuda \
    --trials 200 \
    --max-directions 3
```

Same thing, except you're deciding the budget and the direction count. See
[flags worth knowing](/reference/flags) for the ones that actually change what a run means, as
opposed to the ones that just change how long it takes.

## When it finishes

You get a model directory you can load with `transformers` like any other, plus an
`abliteration.json` next to it recording exactly what was done: the winning configuration, the
seed, how many trials actually ran, the package versions, the commit.

That file matters more than it looks. It's the difference between "this model is abliterated" and
"this model was abliterated on 2026-08-14 with these settings, and here's how to do it again".

**One thing to know about the other files.** The tool keeps the per-prompt scoring rows by
default, and those rows contain the prompt text and what the model said to it. That is deliberate:
it is where a scoring bug becomes visible, and a rate with no rows behind it cannot be checked.
It also means the output directory holds harmful prompts and the replies to them, which is the
last thing you want to push to a public repository by accident. `--no-margins` turns it off if you
would rather not have them.

## Now check whether you broke it

This is the step most people skip, and it's the interesting one.

Your model has stopped refusing. Has it also stopped *understanding* that some requests are
dangerous? Those are different things, and only one of them is a problem.

```sh
senbonzakura compass --model my-abliterated-model \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds \
    --out compass.json
```

[The compass](/guide/compass) explains what the number means and, more importantly, what it
doesn't.

## So what?

One command gets you an edited model. The second command tells you what it cost. Running the first
without the second is how people end up publishing a refusal rate for a model that has quietly
been lobotomised, and it happens more than you'd think.
