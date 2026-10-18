# Flags worth knowing

There are 77 flags. `senbonzakura --help` shows the ones a run needs, and `senbonzakura --help-all` lists every one.

These are the flags that change what a run *means*, rather than how it is spelled. Read about these first.

## The main ones

| Flag | What it does | Reach for it when |
|---|---|---|
| `--max-directions K` | A **ceiling**, not a count. Each trial draws its own number of directions between `--min-directions` and this. Defaults to `3`. Left alone, the count varies from seed to seed | You want to test the premise of the project. Pin it by setting `--min-directions` and `--max-directions` to the **same** number. Setting one alone does not pin it |
| `--mlp-off` | Attention only: leaves `mlp.down_proj` alone | You suspect the MLP carries capability and the attention carries the reluctance |
| `--hedge-ds DIR` | Adds a hedged-versus-clean contrast to the basis | The model has stopped refusing and started waffling. See below |
| `--patience N` | Stops once the search has not improved for N trials | You are paying for the GPU by the hour |
| `--eval-refusal-final N` | Re-scores the best candidates on a bigger evaluation before picking the winner | Nearly always. See below |
| `--inspect LAYER STRENGTH` | Prints real generations from before and after a cut | You want to read the text, not a percentage |
| `--capability-eval` | Measures the edit's cost against a benchmark. On by default, using a benchmark that ships with the package | You want your own graded benchmark, or you want it off. See below |
| `--slow-probe-ok` | Runs the capability probe on a CPU even when the tool estimates hours | You have no GPU, you have read the estimate, and you mean it. See below |

**`--hedge-ds`** matters because refusal has a polite cousin. A model may stop saying "I can't help with that" and write four paragraphs of disclaimers around a useless answer. A keyword scanner counts that as compliance. Hedging sits on its own axis, and the plain difference-of-means direction does not touch it. This flag gives the search a second contrast, built from hedged replies against clean ones, so it can strip that axis too.

**`--eval-refusal-final`** guards against fooling yourself. The search scores thousands of candidates on a small, fast evaluation. If you then pick the winner on that same small set, you have picked whichever configuration got luckiest on those prompts. This flag re-scores the top few on a larger set before choosing. It costs a few minutes.

::: tip New word: the knee
Ablate harder and refusals fall, but so does coherence. Plot one against the other and you get a curve with a bend. The bend is the knee: the point where you start paying a lot of coherence for very little refusal. Picking it is what the search is for.
:::

## What the search optimises

Two objectives at once:

1. **Strict non-compliance**: hard refusals plus hedging.
2. **KL divergence**, which stands in for "is the model still any good".

::: tip New words: KL divergence
A number for how far the edited model's predictions have drifted from the original's. Zero means identical. It is the coherence alarm. If refusals fell to nothing and this number shot up, you removed the model, not the refusal.
:::

Earlier versions optimised hard refusal against KL only and left the keyword and hedging axis to chance. That produced a model that scores well and hedges constantly.

## The capability probe, which runs by default

**`--capability-eval`** runs whether you ask for it or not, so it is worth knowing what it does. Every other number in a run is about refusal: how often the model refused, hedged, or drifted. None of them asks the model to do anything hard. A model can come through an edit with low divergence and nothing visibly broken, and still have lost the ability to work through a problem in several steps.

So every run answers 256 arithmetic questions before the edit and the same ones after, and reports the difference. The questions ship inside the package, so this needs no network. `--capability-n` sets how many are used. The default of 200 is chosen because a smaller sample gives no weaker answer. It gives none at all, and the run then says "not gradeable" however well the model did.

Two things are easy to miss. If answers are cut off before the model reaches a number, they count as *ungradeable*, never as wrong. A budget set too low would otherwise look exactly like the model getting worse. And if the unedited model was already bad at the task, the run says so. A model that could not do something before the edit cannot be shown to have lost it.

Point it at your own benchmark with a question column and an answer column, or pass an empty string to switch it off.

### The rest of the capability family

| Flag | Default | What it decides |
|---|---|---|
| `--capability-n N` | 200 | How many items the probe uses. 0 turns it off. Before and after are scored on the same items, so the comparison is paired |
| `--capability-task TASK` | `numeric` | How an answer is graded. `senbonzakura capability --help` says what each task measures |
| `--capability-max-new N` | 512 | Token budget per answer. A budget that cuts a worked solution off measures the budget, not the model |
| `--slow-probe-ok` | off | Runs the probe on a CPU anyway, when the tool estimates hours |

**`--capability-task`** matters if your answers are not numbers. The default grader reads a number out of the reply, which is right for arithmetic and wrong for a multiple-choice or free-text set. Pointing `--capability-eval` at your own data without changing this is the easy mistake. The model answers correctly, the grader finds no number, and every item comes back ungradeable.

**`--capability-max-new`** interacts with the ungradeable rule. A low budget cuts answers off before the conclusion, and those count as ungradeable. A low budget therefore gives you fewer measurements and a run that says so, not a faster measurement.

### Grading a tool call

`senbonzakura capability --task tool-call` compares the model's call with a reference call: did it name the tool the answer key names, with the same arguments? That is one question. A second is whether the call could have run at all.

`--tool-schema FILE` adds that second question. The file is JSON: a list of tool declarations, or a provider request body with a `tools` key. Two things happen:

1. The tools are written into the prompt, so the model is told what it may call. Without this step, "passed an argument the tool does not declare" would be unfair, because the model was never told what the tool declares.
2. Every reply is checked against those declarations by code.

Each measure has its own denominator, because the sets of replies differ:

| Measure | Over | Reads as |
|---|---|---|
| emitted a call at all | the replies that could be graded | the model produced something call-shaped, not prose |
| mechanically valid | the replies that could be graded | the call could have been executed |
| named a tool that was not offered | the replies that emitted a call | the model invented a tool |
| missing a required argument | the calls naming a real tool | a declared requirement was left out |
| argument of the wrong declared type | the calls naming a real tool | a string where a number was declared, and so on |
| passed an argument the tool does not declare | the calls naming a real tool | the model made up a parameter |

**None of these says the model chose the right tool.** A model that always reaches for the wrong tool, with clean arguments of the right types, scores 100% on every row above. Accuracy says whether the call matched the answer key. Validity says whether it was well formed. A model can score 0% on the first and 100% on the second.

A rate over too few replies is withheld, and the counts are shown in its place. Replies cut off by the token budget are ungradeable, never failures, for the same reason as above.

The flag is refused on any other task. Offering a different toolbox also changes the exam fingerprint, so a later `--compare-to` refuses to pair two runs offered different tools.

## The override for a worthless measurement

**`--short-budget-ok`** belongs beside `--slow-probe-ok`. It overrides a refusal the tool raises about its own measurement being worthless, rather than about the machine.

`--gen-tokens` below the visibility floor is refused without it. The reason matters before you reach for the override: **a refusal the model never reaches is not counted.** With a short budget the model is cut off before it would have refused. The run scores that as compliance, and the search then prefers configurations whose refusal lands just after the cutoff. The number moves the way you hoped and measures the budget.

So the override is for a smoke test, not a figure you will quote. To find the budget your model needs, run `senbonzakura score --length-sweep` instead of lowering the floor.

**`--slow-probe-ok`** exists because the probe's defaults are sized for a GPU, where they take minutes. On a CPU with a 1.7B model they take about four hours. Rather than looking like a hung run all afternoon, the tool works out the cost and stops:

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

Three answers, and the flag is the last. Measuring less is usually the right one. The figure gets a wider error bar, and you get it today.

The worked command says `--n` and `--max-new`, not `--capability-n` and `--capability-max-new`. The message comes from `senbonzakura capability`, which can run on its own against a model you already have, and which spells the same two settings without the prefix. Inside an `abliterate` run, use the prefixed names.

## How much of the direction is still there? `--leak-report`

Every other number here is about behaviour. This one looks inside the weights instead of at the output.

Picture the model's internal state at each layer as an arrow in a very high dimensional space. The refusal direction is one axis in that space. The measurement is the length of the arrow's shadow on that axis, divided by the arrow's length. A smaller shadow means less of the direction is present.

### Read it beside the refusal rate, never on its own

**It is not a verdict on whether the edit worked.** An earlier version of this section said it was. On `Qwen2.5-0.5B-Instruct` we removed refusal behaviour completely, to a rate of zero, and 88 to 96 per cent of the base model's refusal direction was still in the residual stream afterwards. Two independently produced checkpoints, measured two ways. So "the shadow is still there" does not mean "the edit failed".

The opposite corner happens too. A model can be built so that the direction a contrast set finds is a decoy. The shadow then goes away and the refusal stays. Published work describes exactly that defence.

| | refusal collapsed | refusal unchanged |
|---|---|---|
| **shadow small** | the straightforward case | the edit landed on something that was not doing the work, a decoy among them |
| **shadow large** | measured, and real: refusal gone with the direction still present | the edit did not reach the model |

On its own the figure tells you about magnitude. That is worth knowing, but it is not a result. Why refusal can go while the magnitude stays is open, with three live candidates: the norm restore putting length back, the stream carrying the component by untouched routes, and the extracted direction not being the one that causes refusal.

### What it does not need

It needs no judge, no sampling and no generated tokens, so there is no grader to validate and no run-to-run variance worth speaking of. It is one forward pass per probe prompt, which takes seconds. `--leak-prompts` sets how many prompts it averages over. The default is 32.

The figure comes out once per residual-stream position. Position 0 is what the embedding produced, and position `i + 1` is what decoder layer `i` wrote.

### Which of the two means to quote

An abliteration does not touch every position. The strength peaks at one layer and fades a few layers out. On a typical run more than half the positions are never edited, and they keep the whole refusal direction. Average them in and the number rises for a reason unrelated to how well the edit worked.

So `abliterate --leak-report` prints two means and names the useful one first:

| The mean over | Answers | Quote it when |
|---|---|---|
| the positions the edit reached | Did the edit remove the direction where it acted? | You are talking about the edit |
| every position | Is the direction still anywhere in the stream? | You are talking about the whole model |

Both land in `abliteration.json`, as `mean_over_edited_positions` and `mean`, with the positions listed beside them. The standalone `leak` command prints only the second, and says so. It is pointed at a checkpoint it did not edit, so it cannot know where an edit acted.

On one of our runs, an edit that removed about two thirds of the direction where it acted read 88 to 96 per cent across the whole stack, because fourteen of twenty-five positions were never edited.

### Two readings that look like bad news and are not

**The figure at the output is bigger, and that is a ruler problem, not a model problem.** The model has a normalisation step at the very end that multiplies each dimension by its own learned number. Think of stretching a sheet of graph paper by a different amount along each axis. A right angle on it no longer looks like one, though nothing on the sheet moved. So a direction that has genuinely gone still casts a visible shadow against the original axis. Measured against the axis the stretch maps it onto, it is gone again. The run reports the second and names the basis beside it. There is deliberately no way to ask for the first on its own.

**Sometimes there is no figure, and the run says which recipe stopped it.** The measurement is defined against one direction. If the search applied several directions per layer, or a different direction at each layer, there is no single axis for the shadow to fall on. The honest output is to say so. Picking one and reporting its shadow under the plain name would be a true number about a question nobody asked.

The result lands in `abliteration.json` under `residual_leak`, in one of three states: you did not ask, you asked and there is no figure with the reason, or the figure with its basis.

### Where the idea came from

The metric is read from `orcabonsai-27B-uncensored`, which verifies its own runtime ablation this way. Their licence is Apache 2.0, so the arithmetic is ours to use with attribution. `THIRD-PARTY-NOTICES.md` carries it.

## Where next

- [The full CLI reference](/reference/cli).
- [Your first run](/guide/first-run), which uses almost none of this.
