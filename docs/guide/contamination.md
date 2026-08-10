# Contamination

You want to report your model's score on AdvBench, the benchmark everyone in this field quotes.
Sensible. It's how a reader places you against the literature.

One problem: is AdvBench already *inside* the corpus you fitted on?

If it is, you tuned the model on the exam paper. The number is still a number, it just doesn't
mean what it says.

::: warning This is not a hypothetical for us either
About **430 rows** of this project's harmful corpus **are AdvBench**, arriving through
`mlabonne/harmful_behaviors` without anybody choosing to put them there.

Public corpora are assembled from each other. A benchmark can be inside yours and nothing will
mention it.
:::

## Ask the tool

Point it at the benchmark you're about to quote. One prompt per line, same format as the
corpus builder takes. It reads your track and writes nothing to it, so there's no way to
make things worse by asking.

```sh
python -m senbonzakura.track --out mytrack \
    --contamination advbench.txt --contamination-name AdvBench \
    --contamination-report contamination-advbench.json
```

```
contamination: AdvBench against this track
  520 rows, 520 distinct requests (templates discovered: 7)
  in fit 2, in search 3, in measure 7, absent 508
  CONTAMINATED: 5 of its requests were fitted or searched on.
```

Five rows out of 520. Small enough to shrug at, and it's still the difference between a
number you can publish and one you can't.

## Three things to know about how it counts

**It matches requests, not strings.** A benchmark that asks the same question in different
words still counts as overlap, and it should: your model didn't learn the punctuation, it
learned the request. Comparing raw text would have come back clean here and been wrong.

::: tip New word: template
Corpora love a wrapper. "How do I X" and "Write a guide explaining how to X" are one
request in two costumes. The tool works out the wrappers your corpus uses and strips them
before comparing, which is why the output above mentions seven of them.
:::

**`measure` is not contamination.** That slice is held out by design, so a benchmark row
found only there is fine to report, and the tool tells you how many rows qualify. Overlap
with `fit` or `search` is the fatal kind.

**It reports counts and never a prompt.** The output is safe to paste into an issue, a
paper or a group chat without accidentally publishing a list of harmful requests.

## Making it a gate rather than a note

```sh
--fail-on-contamination
```

Exits non-zero instead of merely reporting, which is what you want inside a script or a CI
job. Without the flag it always exits cleanly, on purpose: a check that refuses by default
is a check people stop running, and then you find out nothing at all.

## What it leaves you with

It finishes by handing you the flags that point the compass at the held-out rows, so you
don't have to work the arithmetic out yourself:

```
TRACK_BUILT mytrack  harmful {'fit': 256, 'search': 128, 'measure': 616}
  measure with: --skip-harmful 128 --skip-harmless 384 --n 616
```

**So what:** copy that second line into your compass command and whatever it reports is
measured on prompts the model was never fitted or searched on. That's the whole game.

## Where next

- [The track](/guide/the-track) for how the three slices get cut in the first place.
- [The compass](/guide/compass) for what to do with the held-out rows.

