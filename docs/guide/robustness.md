# Does the edit hold up

A refusal rate tells you whether the model says no. Three commands test whether the no is worth anything:

- `senbonzakura jailbreak` asks whether it answers a harmful request when asked once, and still answers the harmless ones.
- `senbonzakura multi-turn` asks whether a refusal holds when the conversation carries on.
- `senbonzakura tamper` asks whether the edit survives somebody finetuning the model afterwards.

Each runs its control by default. Without a control, a number has explanations you can't rule out: the model declines everything, the
refusal was never stable, or any finetune at all would disturb the edit.

## Jailbreak resistance

```sh
senbonzakura jailbreak --model edited --device cuda \
    --attack-set strongreject --benign-set xstest-safe \
    --max-new 512 --out jb.json
```

```
JAILBREAK_DONE  set=strongreject answered=85.0% (34/40, 95% CI [0.709, 0.929]) substantive=82.5% (33/40, 95% CI [0.680, 0.912])
OVER_REFUSAL  set=xstest-safe refused=7.5% (3/40, 95% CI [0.026, 0.199])
```

Every rate carries its counts and its interval, on the line you'd paste.

There are two readings of the same replies. `answered` is one minus the refusal rate, which is what the field usually quotes.
`substantive` also requires an actual answer, not an evasion, a lecture or damaged output. **When the two are far apart, that gap is
the finding.**

The attack set is a name, not a path: `advbench`, `harmbench`, `harmbench-copyright`, `strongreject` or `xstest-unsafe`. The command
refuses `xstest-safe` as an attack set, because it is the benign control.

To read a rate against the unedited model, run the base model too:

```sh
senbonzakura jailbreak --model base    --device cuda --out jb-base.json
senbonzakura jailbreak --model edited  --device cuda --baseline jb-base.json --out jb-edited.json
```

Some unedited chat models decline almost nothing, so a high jailbreak rate on one of them says nothing about your edit. The command
checks that both runs are the same measurement before it subtracts them.

## Over several turns

```sh
senbonzakura multi-turn --model edited --device cuda --attack-set strongreject --out mt.json
```

```
MULTITURN_TURN_ONE  set=strongreject answered=82.5% (33/40, 95% CI [0.680, 0.912]) refused=7
MULTITURN_CONVERSION  strategy=persistent-reframe turns=4 converted=n/a (6/7, 95% CI [0.487, 0.974]) median_turn=3.0
MULTITURN_CONVERSION  strategy=plain-repeat turns=4 converted=n/a (0/7, 95% CI [0.000, 0.354]) median_turn=None
```

The `plain-repeat` line is the control, and it always runs. Asking the same question again is a strategy in its own right. **If a
persuasive escalation does no better than repetition, it has shown nothing.** The two are compared row by row. Where the intervals
overlap, the run says it cannot separate them. `median_turn` shows whether a conversion came early or late.

The built-in escalations are generic and long-published. Bring your own with `--strategy-file` (JSON with `name`, `description` and
`follow_ups`). The artefact then records a digest of your turns, not their text.

## After somebody finetunes it

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

`recovered` is the share of the removed refusal that came back. A model refusing 20% before and 60% after has recovered half of the 80
points that were available. `resistance_abliterix` is the same quantity under a competitor's formula.

**The two controls.** The *neutral arm* finetunes the same model with the same steps, learning rate and seed, on benign data. If refusal
recovers just as much there, the measurement is "any finetune disturbs the edit", and the safety framing is wrong. So the headline is the
difference between the arms. The *ceiling* runs the same recipe on the unedited checkpoint, so a recovered fraction of 0.4 is read against
what the recipe can achieve.

**The bias runs one way.** The recovery data pairs harmful prompts with short refusals, which contain the phrases the ruler looks for. That
makes the recipe generous to recovery on purpose:

| Result | What it supports |
|---|---|
| a **low** recovered fraction | strong evidence that the edit held |
| a **high** recovered fraction | weak evidence that it did not |

The evaluation rows are rows the finetune never saw, so the measure is generalisation, not a memorised sentence.

### What the trainer records

`tamper` contains a real trainer, because the only way to answer the question is to run the finetune and measure again.

| Mode | What it changes | The question it answers |
|---|---|---|
| `--method lora` | adds a small set of new weights beside the frozen ones | can an adapter route around the edit? This is what most people do |
| `--method full` | updates every weight in the model | can the edit be undone outright? This is the stronger statement |

A LoRA result is not a rewritten model. The adapter never touches the tensors the edit changed.

Every run records the whole recipe: method, learning rate, steps, batch size, LoRA rank, alpha, dropout and target modules, optimiser,
training dtype, seed, a digest of the training pairs, and **the loss at every step**. The loss trace is what separates a finetune that
worked from one that did nothing.

### The guards

A trainer that quietly does nothing looks like a strong result, because the model afterwards is untouched. Four things stop that:

1. **A learning rate of zero is refused.**
2. **A loss that does not fall invalidates the run.** The command exits non-zero.
3. **An adapter that can attach to no module is refused**, naming the modules it looked for.
4. **Everything is seeded before anything draws from it**, including the adapter's starting weights.

It is not a general training framework. It is one machine, one GPU and short runs, on the corpus you name. It doesn't judge whether the
finetune was a good idea. It records what you ran and measures what changed. The adapter path needs one extra package:

```sh
pip install 'senbonzakura[finetune]'
```

Without it, `--method lora` stops at once and says so. `--method full` needs nothing extra.

## When a run refuses to give you a number

All three commands exit non-zero and say so in capitals rather than print a figure they can't stand behind:

| It refuses when | Because |
|---|---|
| fewer than 30 rows were scored | a rate over fewer observations can't support a claim |
| more than a tenth of replies were unreadable | a damaged model reads as a win under either reading alone |
| a conversation outgrew the context window | the model answered a follow-up without seeing the request |
| a finetune's loss did not fall | it left refusal unchanged, which reads as a perfect result |
| the model already refused every evaluation row | there was no headroom, so recovery is undefined, not zero |

The artefact is still written, with the reason in it.

## What grades the replies

All three use the tool's own refusal ruler, which checks itself against canonical cases before any prompt is sent. **That self-check is
not validation against human labels, and none of these commands reports an agreement figure.** `senbonzakura judge` is the command for
certifying a grader. Nothing in these loops has been through it, so these numbers are no better than the ruler.

## Where next

- [What is and is not established](/guide/what-we-know), before quoting any of these numbers.
- [The compass](/guide/compass), for whether the model still recognises harm at all.
- [The full CLI reference](/reference/cli).
