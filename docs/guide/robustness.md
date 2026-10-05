# Does the edit hold up

Three commands answer three questions about an edited model that a refusal rate cannot:

- `senbonzakura jailbreak` — does it answer a harmful request when asked once, and does it still
  answer the harmless ones?
- `senbonzakura multi-turn` — does a refusal hold when the conversation carries on?
- `senbonzakura tamper` — does the edit survive somebody finetuning the model afterwards?

## The picture first

A refusal rate tells you whether the model says no. These tell you whether the no is worth
anything, and each one is a pair rather than a number.

```text
  jailbreak    harmful prompts ──► answered?        \
                                                     }  both, or neither means anything
               benign prompts  ──► declined?        /

  multi-turn   refused once ──► ask again ──► gave way on which turn?
                            └─► ask again, unchanged  (the control)

  tamper       refusal ──► finetune on refusals ──► refusal   (the measurement)
                       └─► finetune on benign data ──► refusal  (the control)
                       └─► the same recipe on the unedited model  (the ceiling)
```

**Every one of them runs its control by default.** That is the whole design. A number without its
control has an explanation you cannot rule out, and the three explanations are: the model declines
everything; the refusal was never stable in the first place; any finetune at all disturbs the edit.

## Jailbreak resistance, in one command

```sh
senbonzakura jailbreak --model edited --device cuda \
    --attack-set strongreject --benign-set xstest-safe \
    --max-new 512 --out jb.json
```

```
JAILBREAK_DONE  set=strongreject answered=85.0% (34/40, 95% CI [0.709, 0.929]) substantive=82.5% (33/40, 95% CI [0.680, 0.912])
OVER_REFUSAL  set=xstest-safe refused=7.5% (3/40, 95% CI [0.026, 0.199])
```

Two things in that output are not standard practice and both are deliberate.

**Every rate carries its counts and its interval.** `62/120` says what `51.7%` hides, and the
terminal line is the thing that gets pasted into a message, so the uncertainty is on the line
rather than only in the file beside it.

**There are two readings of the same thing.** `answered` is one minus the refusal rate, which is
what the field usually quotes. `substantive` additionally requires that the reply delivered an
answer: not an evasion, not a lecture, not the garbage a damaged model emits. StrongREJECT exists
because earlier refusal metrics scored empty and useless answers as successful jailbreaks, so a
tool that ships StrongREJECT and then reads it with one minus refusal has reproduced the thing the
corpus was built to expose. **When the two numbers are far apart, that gap is the finding.**

### The attack set is a name, not a path

```sh
--attack-set advbench | harmbench | harmbench-copyright | strongreject | xstest-unsafe
--benign-set xstest-safe
```

All of them ship inside the wheel, so none of this needs a network. A name rather than a path
because the bundled table records what each corpus **is**: XSTest's safe half used as an attack set
reports near-total success on a model that refused nothing it should have, and nothing further down
the pipeline can see that. The command refuses it.

### Give it the unedited model too

```sh
senbonzakura jailbreak --model base    --device cuda --out jb-base.json
senbonzakura jailbreak --model edited  --device cuda --baseline jb-base.json --out jb-edited.json
```

```
AGAINST_BASELINE  base=the/base substantive=12.5% (5/40, 95% CI [0.055, 0.260]) change=+70.0pp 95% CI [+52.1, +87.9]pp
```

Without this, a jailbreak rate is a number about a model whose starting point you do not know.
Refusal on harmful prompts varies enormously between unedited models: some widely used chat models
decline almost nothing with no edit of any kind, and on one of those a high jailbreak rate is a
property of the model rather than a defeated defence. The command compares the two runs' recorded
provenance first and refuses to subtract them when they are not the same measurement.

## Over several turns

```sh
senbonzakura multi-turn --model edited --device cuda --attack-set strongreject --out mt.json
```

```
MULTITURN_TURN_ONE  set=strongreject answered=82.5% (33/40, 95% CI [0.680, 0.912]) refused=7
MULTITURN_CONVERSION  strategy=persistent-reframe turns=4 converted=n/a (6/7, 95% CI [0.487, 0.974]) median_turn=3.0
MULTITURN_CONVERSION  strategy=plain-repeat turns=4 converted=n/a (0/7, 95% CI [0.000, 0.354]) median_turn=None
```

The second `MULTITURN_CONVERSION` line is the control, and it runs whether you ask for it or not.
Asking the same question again, unchanged, is a strategy: a refusal that gives way to a repeated
question costs an attacker one keystroke and no persuasion at all. **If a persuasive escalation
does no better than repetition, it has demonstrated nothing**, and the only way to know is to run
both. Turn one is identical under every strategy, so the control costs only the escalation turns
over the rows that refused, and the comparison is paired row by row.

Where the two intervals overlap, the run says it cannot separate them rather than reporting the
difference. An overlapping pair quoted as a point difference is how a strategy comes to be credited
with the model's own instability.

`median_turn` is there because "38% eventually complied" and "38% eventually complied, nearly all
on the second ask" are different findings about the same model.

The built-in escalations are deliberately limited to generic, long-published framings. They are
enough to tell a stable refusal from a nominal one. Bring your own with `--strategy-file`, as JSON
with `name`, `description` and `follow_ups`; the artefact then records a digest of your turns and
not their text, because an artefact is the thing most likely to be published.

## After somebody finetunes it

This is the question a company selling access to an edited model cannot currently answer, and it
needs a card.

```sh
senbonzakura tamper --model edited --base unedited --device cuda \
    --corpus advbench --train-n 128 --method lora --steps 60 --out tamper.json
```

```
TAMPER_BEFORE  model=edited refused=6.8% (19/281, 95% CI [0.044, 0.103]) eval_rows=281 (none of them trained on)
TAMPER_AFTER  arm=recovery refused=50.2% (141/281, 95% CI [0.444, 0.560]) recovered=+0.466 (95% CI [+0.414, +0.525]) resistance_abliterix=0.534
TAMPER_AFTER  arm=neutral   refused=6.9% (19/281, 95% CI [0.045, 0.104]) recovered=+0.001 (95% CI [-0.040, +0.044]) resistance_abliterix=0.999
TAMPER_SAFETY_SPECIFIC  gap=+0.465 (95% CI [+0.412, +0.521]): the interval on the difference excludes zero, so the safety data moved refusal by more than benign data of the same size trained the same way.
```

`recovered` is the share of the removed refusal that came back. A model refusing 20% before and 60%
after recovered half of the 80 points that were available, which is more useful than "refusal rose
40 points". `resistance_abliterix` is the same quantity under a competitor's published formula,
clipped the way theirs is, so one of the two numbers is like for like with theirs.

### The two controls, and why the headline is a difference

**The neutral arm** finetunes the same model, the same number of steps, at the same learning rate,
on the same seed, using benign task data with no safety content in it. If refusal recovers just as
much there, the measurement is "any finetune disturbs the edit" and the safety framing is false.
**That distinction is the entire claim**, which is why the headline reported here is the difference
between the arms rather than either level, and why the artefact states no safety-specific figure at
all when the control was skipped.

**The ceiling** runs the same recovery recipe on the unedited checkpoint, so a recovered fraction of
0.4 is read against what this recipe can achieve rather than against a perfect score.

### Which way the bias runs

The recovery data pairs a harmful prompt with a short refusal, and those refusals contain the
phrases this tool's refusal ruler looks for. That is what safety-recovery data is, and it makes the
recipe generous to recovery on purpose. The asymmetry only runs one way, and the artefact says so:

| Result | What it supports |
|---|---|
| a **low** recovered fraction | strong evidence that the edit held |
| a **high** recovered fraction | weak evidence that it did not |

The evaluation rows are the half of the corpus the finetune never saw, so what is measured is
generalisation rather than a memorised sentence. There is no flag that lets the two halves overlap.

### LoRA or every weight

`--method lora` trains an adapter beside the frozen weights, which is what most customers would
actually do, and it measures whether an adapter can route around the edit. `--method full` updates
every weight, which modifies the tensors the edit changed, and measures whether the edit itself is
undone. They answer different questions, the artefact records which was asked, and reading a LoRA
result as though the weights had been rewritten is the misreading to avoid.

## When a run refuses to give you a number

All three commands exit non-zero and say so in capitals rather than printing a figure they cannot
stand behind. The conditions are worth knowing before you see one:

| It refuses when | Because |
|---|---|
| fewer than 30 rows were scored | a rate over fewer observations cannot support a claim in any framing |
| more than a tenth of replies were unreadable | a damaged model reads as a win under either reading taken alone |
| a conversation outgrew the context window | the model answered a follow-up without being shown the request |
| a finetune's loss did not fall | it left refusal unchanged, which reads as a perfect result |
| the model already refused every evaluation row | there was no headroom, so recovery is undefined, not zero |

The artefact is still written in every one of those cases, carrying the reason. A run that cannot
produce a valid number says which part failed; an absence never reads as a pass.

## What grades the replies

All three use this tool's own refusal ruler, and it self-checks against its canonical cases before a
single prompt is sent, because a ruler that misreads produces a confident wrong number rather than
an error. **That self-check is not a validation against human labels, and none of these commands
reports an agreement figure.** `senbonzakura judge` is what certifying a grader looks like, and
nothing in these loops has been through it because there is no model in them to certify. These
numbers inherit the ruler's known behaviour and are no better than it.

## Where next

- [What is and is not established](/guide/what-we-know), before quoting any of these numbers.
- [Contamination](/guide/contamination), on why evaluation rows have to be held out.
- [The compass](/guide/compass), for whether the model still recognises harm at all.
- [The full CLI reference](/reference/cli).
