# Flags worth knowing

There are forty-odd flags. `senbonzakura --help` lists them all. These are the seven that
change what a run *means* rather than how it's spelled, so if you're going to read about any
of them, read about these.

## The seven

| Flag | What it does | Reach for it when |
|---|---|---|
| `--max-directions K` | How many refusal directions to remove per layer. `1` is the classic single-direction method | You want to test the whole premise of this project. It defaults to `3` |
| `--mlp-off` | Attention-only: leaves `mlp.down_proj` alone | You suspect the MLP is carrying the model's capability and the attention is carrying its reluctance |
| `--hedge-ds DIR` | Folds a hedged-versus-clean contrast into the basis | The model has stopped refusing and started waffling. See below |
| `--patience N` | Stop once the search hasn't improved for N trials | You're paying for the GPU by the hour |
| `--eval-refusal-final N` | Re-score the best candidates on a bigger evaluation before picking the winner | Always, really. See below |
| `--inspect LAYER STRENGTH` | Print real generations from before and after a cut | You want to look at the actual text instead of a percentage |
| `--capability-eval` | What the run measures the edit's cost against. On by default, using a benchmark that ships with the package | You want to point it at your own graded benchmark, or turn it off. See below |

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

## Where next

- [The full CLI reference](/reference/cli).
- [Your first run](/guide/first-run), which uses almost none of this.

**`--capability-eval`** is the one that runs whether you ask for it or not, so it is worth
knowing what it does. Every other number a run gives you is about refusal: how often the model
refused, how often it hedged, how far its output distribution moved, whether it still forms
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
