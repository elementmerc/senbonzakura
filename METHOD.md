# Measuring a behavioural property of a language model

This is not documentation for a piece of software. It is what we have learned about measuring a
behavioural property of a model so that the number survives somebody else reading it carefully.
It is written to be used against any tool, including this one, and every rule in it is here
because we broke it first and had to withdraw something.

A behavioural property is anything you can only observe by running the model and judging what
comes out: refusal, hedging, helpfulness, whether an answer is correct, whether the text is
coherent. These are harder to measure than they look, because the measurement has as many moving
parts as the thing being measured, and almost all of them are invisible in the final number.

The short version: **the number is not the measurement.** The measurement is the number plus the
conditions it was taken under, and a number reported without them cannot be checked, cannot be
compared, and cannot be withdrawn cleanly when it turns out to be wrong.

---

## 1. Measure the instrument before you trust the result

The single most useful thing in this document. Before reporting that your model scores X on some
property, run a **null control**: a deliberately stupid measurement that should score much worse.
If it does not, your instrument is measuring something other than what you named it.

The control has to be plausible enough to be embarrassing. A good one for text is a ruler that
reads nothing but sentence length. If a length-only ruler separates your two groups as well as
your carefully built instrument does, then your instrument is a length detector wearing a
different name.

This is not hypothetical. Our own harm-recognition measurement scores 0.9887 on one model against
a length-only control at 0.6564, which is a real gap. On a smaller model of the same family it
scores 0.6616 against that same control at 0.6564. The second pair is not a weak result; it is
**not a measurement of anything**, and nothing except the control would have told us.

**What to do:** pick a control before you look at the result, report it beside every figure, and
treat "the control did nearly as well" as a failed measurement rather than a caveat.

## 2. A rate is a property of the setup, not of the model

A refusal rate, a hedging rate, a helpfulness score: none of these belong to the model alone. Each
belongs to the model, the prompts, the prompt format, the generation budget, the decoding
settings, and the rule that decides what counts. Change any of those and the number moves, usually
without anybody noticing that it has.

Three ways this goes wrong in practice:

**The budget is shorter than the behaviour.** If a model refuses at the end of a long preamble and
you stop generating at 48 tokens, the refusal is never emitted, so it is scored as compliance. The
model got safer by being cut off. We measured this: a defended model whose refusals sit past the
300th character scored as almost fully compliant under a short scan, and correctly under a full
one.

**The rule is looking in the wrong place.** A scan of the first part of a reply measures where the
behaviour sits, not whether it is there.

**The rule fires on ordinary language.** Our hedging detector counted phrases like "is illegal"
and "security risk", which appear constantly in complete, compliant technical answers. It scored
those answers as refusals, and it carried full weight in the rule that chose which model to ship.
So the search was being steered toward models that do not caveat, which is a tone preference
wearing the name of a compliance measurement.

**What to do:** record the corpus, the prompt format, the budget and the decision rule next to the
number. If you cannot say what budget produced a rate, you do not have a rate.

## 3. Never report a figure on the rows you chose it with

If you search over configurations and report the score of the winner, that score is the best of N
draws, not an estimate of anything. This is the most common way a good-faith measurement becomes
an overstatement, and it does not feel like cheating while you are doing it.

Split the data three ways and keep the boundary:

| Partition | What touches it |
|---|---|
| Fitting | Whatever the method derives from data: directions, thresholds, calibration |
| Selection | The search, the hyperparameters, anything you compare and pick between |
| Measurement | Nothing, until the final number. Once you look, it is spent |

Two-way splits are not enough when there is a search, because the search consumes the second
partition and leaves nothing clean to report on.

We have published a figure scored on a selection partition, with a sample of 64, and had to
withdraw it. The model was fine. The number described rows the configuration had been chosen on,
which is a different claim from the one the sentence around it made.

**What to do:** make the split a property of the data rather than a habit, so that crossing it
requires an explicit act. Check the split in the code that slices, not in the process that is
supposed to remember.

## 4. A sample too small to carry a rate should not be printed as one

A proportion needs a denominator large enough to mean something, and the failure is quiet: the
arithmetic works on four samples and produces a number with a decimal point in it.

One of our own regression gates once passed a refusal rate that moved from under a tenth to over
half, because the sample behind it had fallen from 200 to 4. Both numbers were computed correctly.

**What to do:** set a floor below which a rate is reported as counts rather than a percentage, and
attach an interval to anything you do report. A figure with no interval cannot be gated on,
because a gate that fires on noise is a gate that gets switched off.

## 5. Two numbers are comparable only if they were made the same way

You cannot compare a figure to one from last month unless nothing that produces it has changed. In
practice something always has: the scorer, the corpus, the precision, the prompt format, the
tokeniser. If the instrument changed, you have two instruments and one comparison that means
nothing.

The fix is to make every measurement carry its own identity, so that the comparison can be refused
rather than made wrongly:

- **What** was measured: the metric name.
- **How**: the estimator, by name, and its units.
- **On what**: a digest of the exact input, and the partition it came from.
- **At what precision**: a number computed in 4-bit and one computed in bfloat16 are not the same
  measurement of the same thing.
- **How many**: the sample size and the seeds.
- **With what**: the version of the code that took it.

Then a comparison between two records with different estimators, or different precisions, is
**refused**. A green tick on two numbers that were never comparable is worse than no check,
because it teaches whoever sees it that the check means something.

## 6. Validate the judge before letting it grade

If a model grades your outputs, it is an instrument and it gets measured like one, before it is
used and not after.

Measure **agreement above chance**, not raw agreement. A judge that answers the majority label
every time agrees 90% of the time on a set that is 90% one label, and has learned nothing. We
certified a judge once at 0.90 agreement whose agreement above chance was 0.00.

Also measure how much of each class it caught. A judge blind to the rare class flatters whatever
it grades, and the rare class is usually the one you care about.

Prefer grading by code where the task allows it. A task with a checkable answer needs no judge,
and an ungradeable answer should count as indeterminate rather than wrong.

## 7. Check that the intervention did what you think

A measurement pipeline will happily report the effect of an intervention that never happened. We
edited one model family for weeks and published the results before discovering that the edit was
not reaching the part of the network we thought it was: the numbers were real, the mechanism
described in the prose was not, and every figure for that family was withdrawn.

**What to do:** measure the intervention directly, not only its consequences. If you claim to have
changed something inside the model, show that thing changed, on the model you are reporting.

## 8. Publish the results that went against you

Two things follow from everything above, and they are the part that takes character rather than
engineering.

**Say where the instrument stops working.** Ours has never been run above 3B parameters, and it
degrades on small models until it measures nothing. That sentence belongs next to the results, not
in a limitations section at the end that readers skip.

**Withdraw properly.** When a figure turns out to be wrong, say so where the figure was, say what
the corrected value is, and do not quietly restate the old one elsewhere. We have published a
comparison table that was drawing on a superseded run on one axis while using the corrected run on
the others, which flattered us on the axis where we lose. The check for this is mechanical: when
you correct a number, grep for it.

---

## A checklist

Before a number leaves the building:

1. Is there a null control beside it, and did it score much worse?
2. Can you state the corpus, the prompt format, the generation budget and the decision rule?
3. Was it measured on rows that nothing was fitted or selected on?
4. Is the sample big enough to carry a rate, and is there an interval?
5. Does the record carry metric, estimator, units, input digest, precision, sample size, seeds and
   code version?
6. If a model graded it, was that model certified above chance first?
7. Did you verify the intervention itself, not only its effects?
8. Does the surrounding prose claim exactly what the number supports, and no more?

A no to any of these is not a reason to delete the measurement. It is a reason to say which one,
next to the number.

---

## On this document

Written for anyone measuring behavioural properties of language models, whatever tool they use.
The incidents are from `senbonzakura`, an abliteration tool, and the rules are not specific to
abliteration: every one of them was learned by getting a number wrong in public and having to work
out why.

Corrections are welcome. If you think one of these is wrong, the disagreement is more useful to us
than the agreement.
