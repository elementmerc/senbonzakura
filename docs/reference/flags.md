# Flags worth knowing

There are 77 flags, and `senbonzakura --help` shows the handful a run needs while
`senbonzakura --help-all` lists every one. This page said "forty-odd" until 2026-09-26, which was
off by forty per cent. These are the seven that
change what a run *means* rather than how it's spelled, so if you're going to read about any
of them, read about these.

## The seven

| Flag | What it does | Reach for it when |
|---|---|---|
| `--max-directions K` | A **ceiling**, not a count. Each trial draws its own number of directions between `--min-directions` and this, so the search decides whether a second direction earns its place. Defaults to `3`, and left alone it picks one more often than not | You want to test the whole premise of this project. Pin it by setting `--min-directions` and `--max-directions` to the **same** number: that is what turns a ceiling into an experiment, and setting one alone does not |
| `--mlp-off` | Attention-only: leaves `mlp.down_proj` alone | You suspect the MLP is carrying the model's capability and the attention is carrying its reluctance |
| `--hedge-ds DIR` | Folds a hedged-versus-clean contrast into the basis | The model has stopped refusing and started waffling. See below |
| `--patience N` | Stop once the search hasn't improved for N trials | You're paying for the GPU by the hour |
| `--eval-refusal-final N` | Re-score the best candidates on a bigger evaluation before picking the winner | Always, really. See below |
| `--inspect LAYER STRENGTH` | Print real generations from before and after a cut | You want to look at the actual text instead of a percentage |
| `--capability-eval` | What the run measures the edit's cost against. On by default, using a benchmark that ships with the package | You want to point it at your own graded benchmark, or turn it off. See below |
| `--slow-probe-ok` | Runs the capability probe on a CPU even when the tool has worked out it will take hours | You have no GPU, you have read the estimate, and you mean it. See below |

## The three that deserve a paragraph

**`--hedge-ds`** exists because refusal has a polite cousin. A model that no longer says "I
can't help with that" may instead produce four paragraphs of disclaimers wrapped around a
useless answer, and a keyword scanner counts that as compliance. Hedging sits along its own
axis, and the plain difference-of-means direction doesn't touch it. This flag hands the
search a second contrast, built from hedged replies against clean ones, so it can strip
that axis too.

**`--eval-refusal-final`** is a defence against fooling yourself. The search evaluates
thousands of candidates, so it needs a small, fast evaluation to do it; and if you then
crown the winner on that same small evaluation, you've picked whichever configuration got
luckiest on those particular prompts. This flag re-scores the top handful on a larger set
before choosing. It costs a few minutes and it's the difference between a result and a
coincidence.

::: tip New word: the knee
Ablate harder and refusals fall, but so does coherence. Plot one against the other and you
get a curve with a bend in it: the point where you start paying a lot of coherence for very
little refusal. That bend is the knee, and picking it is what the search is actually for.
:::

## What the search is optimising

Three objectives at once, not two:

1. **Strict non-compliance**: hard refusals plus hedging.
2. **The Heretic keyword rate**, kept as its own separate axis so the comparison against
   that tool is on its own terms rather than ours.
3. **KL divergence**, which stands in for "is the model still any good".

::: tip New words: KL divergence
A number for how far the edited model's predictions have drifted from the original's. Zero
means identical. It's the coherence alarm: if refusals fell to nothing and this number went
through the roof, you didn't remove the refusal, you removed the model.
:::

Earlier versions optimised hard-refusal against KL only and left the keyword and hedging
axis to chance, which is how you end up with a model that scores beautifully and hedges
constantly.

## The one that runs whether you ask for it or not

**`--capability-eval`** is the one that runs whether you ask for it or not, so it is worth
knowing what it does. Every other number a run gives you is about refusal: how often the model
refused, how often it hedged, how far its first-token distribution moved, whether it still forms
sentences. None of those asks the model to do anything hard. A model can come through a bake with
a low divergence and nothing broken, and have lost the ability to work through a problem in
several steps, because nothing in the run ever asked it to.

So every run now answers 256 arithmetic questions before the edit and the same ones after, and
reports the difference. The questions ship inside the package, so this needs no network and no
account. `--capability-n` sets how many are used; the default of 200 is chosen because a smaller
sample does not give you a weaker answer, it gives you none at all, and the run then says "not
gradeable" however well the model did.

Two things it will tell you that are easy to miss. If the answers get cut off before the model
reaches a number, those count as *ungradeable*, never as wrong: a budget set too low would
otherwise look exactly like the model getting worse. And if the unedited model was already bad at
the task, the run says so, because a model that could not do something before the edit cannot be
shown to have lost it.

Point it at your own benchmark with a question column and an answer column, or pass an empty
string to switch it off.

### The rest of the capability family

Four more flags shape that probe. None of them appears in `senbonzakura --help`, which shows the
fifteen flags a run needs; they are in `senbonzakura --help-all` along with the other 59.

| Flag | Default | What it decides |
|---|---|---|
| `--capability-n N` | 200 | How many items the probe uses. 0 turns it off. Before and after are scored on the same items, so the comparison is paired |
| `--capability-task TASK` | `numeric` | How an answer is graded. `senbonzakura capability --help` says what each task measures |
| `--capability-max-new N` | 512 | Token budget per answer. A worked solution is long, and a budget that cuts it off measures the budget rather than the model |
| `--slow-probe-ok` | off | Run the probe on a CPU anyway, when the tool has worked out it will take hours |

### The other override of the same kind

**`--short-budget-ok`** is not a capability flag, and it belongs beside `--slow-probe-ok` because
it is the same sort of thing: an override for a refusal the tool raises about its own measurement
being worthless rather than about the machine.

`--gen-tokens` below the visibility floor is refused without it. The reason is worth understanding
before you reach for the override: **a refusal the model never reaches is not counted.** With a
short budget the model gets cut off before it would have refused, the run scores that as
compliance, and the search then prefers configurations whose refusal lands just after the cutoff.
The number goes in the direction you were hoping for and measures the budget.

So it is for a smoke test and not for a figure you will quote. To find the budget your model
actually needs, run `senbonzakura score --length-sweep` instead of lowering the floor.

**`--capability-task`** is the one to look at if your own benchmark's answers are not numbers. The
default grader reads a number out of the model's answer, which is right for arithmetic and wrong
for a multiple-choice set or a free-text one. Pointing `--capability-eval` at your own data
without changing this is the easy mistake: the model answers correctly, the grader cannot find a
number, and every item comes back ungradeable.

### Grading a tool call against the tools you offered

`senbonzakura capability --task tool-call` compares the model's call against a reference call: did
it name the tool the answer key names, with the same arguments? That is one question. There's a
second one it can't answer, which is whether the call could have been run at all.

`--tool-schema FILE` adds it. The file is JSON, either a list of tool declarations or a provider
request body with a `tools` key, so in most cases it's a file you already have. Two things happen
with it:

1. The tools are written into the prompt, so the model is told what it may call and what each one
   takes. Without that step the next part would be unfair, because "passed an argument the tool
   does not declare" is only a failure if the model was ever told what the tool declares.
2. Every reply is then checked against those declarations by code.

What comes back, each over its own denominator because they aren't over the same set of replies:

| Measure | Over | Reads as |
|---|---|---|
| emitted a call at all | the replies that could be graded | the model produced something call shaped rather than prose |
| mechanically valid | the replies that could be graded | the call could have been executed |
| named a tool that was not offered | the replies that emitted a call | the model invented a tool |
| missing a required argument | the calls naming a real tool | a declared requirement was left out |
| argument of the wrong declared type | the calls naming a real tool | a string where a number was declared, and so on |
| passed an argument the tool does not declare | the calls naming a real tool | the model made up a parameter |

**None of these says the model chose the right tool.** A model that reaches for the wrong tool
every single time, with clean arguments of the right types, scores 100% on every row above. That
is why the measures sit beside the graded accuracy and not instead of it: accuracy says whether
the call matched the answer key, validity says whether it was well formed. A model can score 0% on
the first and 100% on the second, and the two together say something neither says alone.

A rate over too few replies is withheld rather than printed, and the counts are given in its
place. Replies cut off by the token budget are counted as ungradeable, never as failures, for the
same reason the accuracy does it: the answers that run out of budget are the long ones, so they
aren't missing at random.

The flag is refused on any other task, because grading a reply against tools when the task was
arithmetic measures nothing. Offering a different set of tools also changes the exam fingerprint,
so a later `--compare-to` refuses to pair two runs that were offered different toolboxes instead
of reporting the difference as a change in capability.

**`--capability-max-new`** interacts with the ungradeable rule above. Lowering it to save time
makes answers get cut off before the model reaches its conclusion, and those count as
ungradeable rather than wrong. So a budget set too low does not give you a faster measurement, it
gives you fewer measurements and a run that says so.

**`--slow-probe-ok`** exists because the probe's defaults are sized for a GPU, where they take
minutes. On a CPU with a 1.7B model they take about four hours. Rather than looking like a hung
run all afternoon, the tool works out what it is about to cost and stops:

```
senbonzakura: 100% of this model's layers are on the CPU or on disk, not on the GPU, so the
capability probe would generate at host speed and take roughly 4.1 hours (200 items at 512
tokens each).
  The card has less free memory than this model needs, so accelerate put the rest in host RAM.
  That is a working model and a very slow one, and --device cuda does not make it a GPU run.
  What to do:
    free the card, or use one with more memory, or load smaller with --load-in-4bit
    measure less of it:  --n 40 --max-new 256
    if you meant it and will leave it running, add --slow-probe-ok
```

Three answers, and the flag is the last of them. Measuring less of it is usually the right one:
the figure gets a wider error bar and you get it today.

Note that the worked command there says `--n` and `--max-new`, not `--capability-n` and
`--capability-max-new`. That is because the message comes from `senbonzakura capability`, which
can be run on its own against a model you already have, and which spells the same two settings
without the prefix. Inside an `abliterate` run, use the prefixed names.

## How much of the direction is still there? `--leak-report`

Every other number here is about behaviour. The refusal rate, the harm recognition score, the KL,
the capability probe: all of them watch what the model does and answer "did it change". This one
looks inside instead, at the weights rather than at the output.

Picture the model's internal state at each layer as an arrow in a very high dimensional space, and
the refusal direction as one particular axis in that space. The measurement is the length of the
arrow's shadow on that axis, divided by the length of the arrow. A smaller shadow means less of
that direction is present.

### Read it beside the refusal rate, never on its own

**It is not a verdict on whether the edit worked**, and this section used to say it was. On
`Qwen2.5-0.5B-Instruct` we removed refusal behaviour completely, to a rate of zero, and 88 to 96
per cent of the base model's refusal direction was still sitting in the residual stream
afterwards. Two independently produced checkpoints, measured two ways. A high figure beside a
collapsed refusal rate is a real thing that happens, so "the shadow is still there" does not mean
"the edit failed".

The opposite corner happens too, and for a different reason: a model can be built so that the
direction a contrast set finds is a decoy, in which case the shadow goes away and the refusal
stays. Published work describes exactly that defence.

So the figure and the behaviour are two measurements, and neither answers the other's question:

| | refusal collapsed | refusal unchanged |
|---|---|---|
| **shadow small** | the straightforward case | the edit landed on something that was not doing the work, a decoy among them |
| **shadow large** | measured, and real: refusal gone with the direction still present | the edit did not reach the model |

What the measurement is good for is the pair. On its own it tells you about magnitude, which is
worth knowing and is not a result. Why refusal can go while the magnitude stays is an open
question with three live candidates (the norm restore putting length back, the stream carrying the
component by untouched routes, and the extracted direction not being the one that causes refusal);
`private/research/leak-field-2026-10-08/` has the run and the arithmetic.

### What it does not need

It needs no judge, no sampling and not a single generated token, so there is nothing to validate a
grader against and no run-to-run variance worth speaking of. One forward pass per probe prompt,
which is seconds. `--leak-prompts` sets how many prompts it averages over, default 32.

The figure comes out once per residual-stream position: position 0 is what the embedding produced,
and position `i + 1` is what decoder layer `i` wrote. The run also prints a mean across those
positions.

### Two things it says that look like bad news and are not

**The figure at the output is bigger, and that is a ruler problem rather than a model problem.**
There is a normalisation step at the very end of the model which multiplies each dimension by its
own learned number. That is a bit like stretching a sheet of graph paper by a different amount
along each axis: a right angle drawn on it no longer looks like a right angle, even though nothing
on the sheet moved. So a direction that has genuinely gone still casts a visible shadow when
measured against the original axis after that stretch. Measured against the axis the stretch maps
it onto, it is gone again. The run reports the second one and names the basis beside it, and there
is deliberately no way to ask it for the first one on its own.

**Sometimes there is no figure at all, and the run says which recipe stopped it.** The measurement
is defined against one direction. If the search settled on applying several directions per layer,
or a different direction at each layer, then there is no single axis for the shadow to fall on, and
the honest output is to say so. Picking one of them and reporting its shadow under the plain name
would be a true number about a question nobody asked, printed next to a claim it does not support.

The result lands in `abliteration.json` under `residual_leak`, always, in one of three states: you
did not ask, you asked and there is no figure with the reason, or the figure with its basis.

### Where the idea came from

The metric is read from `orcabonsai-27B-uncensored`, which verifies its own runtime ablation this
way. Their licence is Apache 2.0, so the arithmetic is ours to use with attribution, and
`THIRD-PARTY-NOTICES.md` carries it.

## Where next

- [The full CLI reference](/reference/cli).
- [Your first run](/guide/first-run), which uses almost none of this.
