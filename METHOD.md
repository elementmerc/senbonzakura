# Measuring a behavioural property of a language model

This is about measuring a behavioural property of a model so that the number survives a careful reader. Refusal, hedging, helpfulness, correctness and coherence all count. The rules apply to any tool, including this one.

These properties only show up when you run the model and judge what it says. The measurement has as many moving parts as the thing being measured, and most of them are hidden inside the final number.

**The number is not the measurement.** The measurement is the number plus the conditions it was taken under. Without those conditions, a number cannot be checked, compared or withdrawn cleanly.

---

## 1. Measure the instrument before you trust the result

Run a null control: a measurement that should score much worse. If it scores nearly as well, your instrument is measuring something other than what you named it. For text, a good control is a ruler that reads only sentence length.

Our harm-recognition measure scores 0.9887 on Qwen3-1.7B against a length-only control at 0.6564. That is a real gap. On Qwen3-0.6B it scores 0.6616 against the same control. That second pair is not a weak result. It measures nothing, and only the control showed it.

**What to do:** pick the control before you look at the result, report it beside every figure, and treat "the control did nearly as well" as a failed measurement, not a caveat.

## 2. A rate belongs to the setup, not the model

A refusal rate belongs to the model, the prompts, the prompt format, the generation budget, the decoding settings and the rule that decides what counts. Change any of them and the number moves, usually without anyone noticing.

- **The budget is shorter than the behaviour.** A refusal that appears after token 48 is never generated, so it scores as compliance. We measured a defended model whose refusals sat past the 300th character. It scored as almost fully compliant under a short scan, and correctly under a full one.
- **The rule looks in the wrong place.** A scan of the first part of a reply shows where the behaviour sits, not whether it is there.
- **The rule fires on ordinary language.** Our hedging detector counted phrases like "is illegal" and "security risk". Those appear in complete, compliant technical answers. The detector carried full weight in the rule that chose which model to ship, so the search favoured models that do not caveat. That was a tone preference under a compliance name.

**What to do:** record the corpus, the prompt format, the budget and the decision rule next to the number. If you cannot say what budget produced a rate, you do not have a rate.

## 3. Never report a figure on the rows you chose it with

If you search over configurations and report the score of the winner, you are reporting the best of N draws, not an estimate. This is the most common way a good-faith measurement becomes an overstatement.

Split the data three ways and keep the boundaries:

| Partition | What touches it |
|---|---|
| Fitting | Whatever the method derives from data: directions, thresholds, calibration |
| Selection | The search, the hyperparameters, anything you compare and pick between |
| Measurement | Nothing, until the final number. Once you look, it is spent |

Two-way splits are not enough when there is a search, because the search uses up the second partition.

We once published a figure scored on a selection partition, with a sample of 64, and withdrew it. The model was fine. The number described rows the configuration had been chosen on, which is a different claim from the one the sentence around it made.

**What to do:** make the split a property of the data, so that crossing it takes an explicit act. Check the split in the code that slices the data, not in a process that is supposed to remember it.

## 4. A sample too small to carry a rate should not be printed as one

The failure is quiet. Arithmetic on four samples still produces a number with a decimal point. One of our regression gates passed a refusal rate that moved from under a tenth to over half, because the sample fell from 200 to 4. Both numbers were computed correctly.

**What to do:** set a floor below which a rate is reported as counts rather than a percentage, and attach an interval to every rate you do report. A gate that fires on noise gets switched off.

## 5. Two numbers are comparable only if they were made the same way

Something always changes between two runs: the scorer, the corpus, the precision, the prompt format or the tokeniser. If the instrument changed, you have two instruments and one comparison that means nothing.

Make every measurement carry its own identity, so a comparison can be refused rather than made wrongly:

- **What** was measured: the metric name.
- **How**: the estimator, by name, and its units.
- **On what**: a digest of the exact input, and the partition it came from.
- **At what precision**: a number computed in 4-bit and one computed in bfloat16 are different measurements.
- **How many**: the sample size and the seeds.
- **With what**: the version of the code that took it.

A comparison between records with different estimators or precisions must be refused. A green tick on two numbers that were never comparable is worse than no check, because it teaches whoever sees it that the check means something.

## 6. Validate the judge before it grades

If a model grades your outputs, it is an instrument. Measure it before you use it.

Measure **agreement above chance**, not raw agreement. A judge that always gives the majority label agrees 90% of the time on a set that is 90% one label, and it has learned nothing. We certified one judge at 0.90 raw agreement whose agreement above chance was 0.00.

Also measure how much of each class it catches. A judge that is blind to the rare class flatters whatever it grades, and the rare class is usually the one you care about.

Grade by code wherever the task allows it. An answer that cannot be graded counts as indeterminate, not wrong.

## 7. Check that the intervention did what you think

A measurement pipeline will happily report the effect of an intervention that never happened. We edited one model family for weeks and published the results before finding that the edit was not reaching the part of the network we thought it was. The numbers were real. The mechanism in the prose was not, and every figure for that family was withdrawn.

**What to do:** measure the intervention directly, not only its consequences. If you claim to have changed something inside the model, show that the thing changed, on the model you are reporting.

## 8. Publish the results that went against you

Say where the instrument stops working, next to the results. Ours has never been run above 3B parameters, and it degrades on small models until it measures nothing.

When a figure turns out to be wrong, say so where the figure was, give the corrected value, and do not restate the old one elsewhere. We once published a comparison table that drew on a superseded run on one axis and the corrected run on the others. That flattered us on the axis where we lose.

**What to do:** when you correct a number, grep for it. The check is mechanical.

---

## A checklist

Before a number leaves the building:

1. Is there a null control beside it, and did it score much worse?
2. Can you state the corpus, the prompt format, the generation budget and the decision rule?
3. Was it measured on rows that nothing was fitted or selected on?
4. Is the sample big enough to carry a rate, and is there an interval?
5. Does the record carry the metric, estimator, units, input digest, precision, sample size, seeds and code version?
6. If a model graded it, was that model certified above chance first?
7. Did you verify the intervention itself, not only its effects?
8. Does the surrounding prose claim exactly what the number supports, and no more?

A no to any of these is not a reason to delete the measurement. It is a reason to say which one, next to the number.

---

## On this document

This is written for anyone measuring behavioural properties of language models, whatever tool they use. The incidents come from `senbonzakura`, an abliteration tool. Each rule is there because we got a number wrong in public and had to work out why.

Corrections are welcome. If you think one of these rules is wrong, the disagreement is more useful to us than agreement.
