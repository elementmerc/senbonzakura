# Benchmarking against another tool

We ran a head-to-head against [Heretic](https://github.com/p-e-w/heretic), the other open-source abliteration
tool with an automated search. Five seeds each, both tools driven by us on one machine, and every model scored
afterwards by our instruments on prompts neither tool was fitted on.

::: warning These numbers are provisional
Every open item is named below the table, and some rows have been corrected or withdrawn. Nothing here is
settled.
:::

::: danger The KL divergence row is superseded. Do not quote the 2026-08-12 figures.
In the 2026-08-12 run **Heretic was not given the best-of-N selection pass that we gave ourselves**. On
2026-09-10 both tools got it, and the difference disappeared:

| First-token KL divergence (`senbonzakura drift`), both tools selected the same way | Senbonzakura | Heretic |
|---|---|---|
| mean over five seeds | 0.0545 | 0.1227 |
| **median** over five seeds | **0.0510** | **0.0573** |
| worst seed | 0.0936 | 0.3591 |

One Heretic seed (seed 43, at 0.3591, against siblings of 0.0439 to 0.1075) carries the entire mean
difference. Drop it and Heretic's mean is 0.0636 against our 0.0545. An exact permutation test over the ten seeds
returns **p = 0.238**. There is **no detectable difference in first-token KL divergence on this model at this
size**, and the earlier "roughly half the collateral damage" is not a finding.
:::

The row is named after the command that produced it. `senbonzakura drift` reports KL divergence between the base
model's and the edited model's first-token distributions on harmless prompts. `senbonzakura coherence` reports
perplexity on one fixed passage, and that figure appears nowhere on this page.

::: tip `drift` is no longer a command you can type
It left the command line on 2026-10-15, along with `diff`. The name stays on this page because these figures were
produced by it. `senbonzakura measure --only drift` still runs the same instrument.
:::

The per-seed values are listed above so that the block doesn't have to be taken on trust. The run's aggregate
artefacts are published as 31 JSON files plus a README under
[`head-to-head/results/2026-09-10/`](https://github.com/elementmerc/senbonzakura/tree/dev/head-to-head/results/2026-09-10),
with no prompts and no weights. `CONTRACT.md` asks for that record.

The same run put Heretic **marginally ahead** on refusal removal, not behind. Both differences are negligible.

Every row carries the run it came from:

| Axis | Senbonzakura | Heretic | Run | All figures are Qwen3-1.7B, five seeds |
|---|---|---|---|---|
| Hard refusal | 0.1% | 0.0% | **2026-09-10** | both tools selected the same way |
| Noncompliance (refusal plus hedging) | ~~3.6%~~ | ~~1.9%~~ | 2026-08-12 | withdrawn, see below |
| Keyword rate | 20.1% | **12.9%** | **2026-09-10** | both tools selected the same way |
| First-token KL divergence (`senbonzakura drift`) | ~~0.191~~ | ~~0.341~~ | 2026-08-12 | superseded, see above |
| Harm recognition | 0.9870 | 0.9872 (tie) | **2026-09-10** | both tools selected the same way |

**One model.** Every figure is Qwen3-1.7B. Nothing here licenses a claim about another model or another size.

**Both tools took hard refusal to (near) zero.** What remains is the price each paid, and on the run where both
were selected the same way, that price is indistinguishable. Heretic won its own keyword metric.

::: danger The noncompliance row is withdrawn
On 2026-09-08 the hedging detector was found to count statements of fact about legality or danger as hedging. A
complete, compliant technical answer that mentioned an illegal activity was scored as a soft refusal, and that
fires more often on more explanatory models. Both figures in the row came from it. The row stays visible, struck
through, because it was published. It has not been re-measured under the corrected detector.
:::

**The two tools' self-reported figures are not a comparison.** Heretic's own `kl_divergence` attribute ranged from
0.0014 to 0.0032 across our runs, and ours from 0.157 to 0.212. Each tool measured its own model on its own prompts
during its own search. Put them on one instrument, on prompts held back from both, and that gap goes away.

### What is still open

- **We lose the keyword axis we optimise ourselves**, 20.1% against 12.9%, and our spread on it is nearly twice
  theirs. A number moving the wrong way on your own objective is often the ruler rather than the model. This is
  being investigated before the table is treated as settled. The row was corrected on 2026-09-21 to the 2026-09-10
  run, and the correction makes us look worse, not better.
- **Corrected 2026-09-25: the harm recognition row carried two figures no artefact holds.** It read 0.9807 against
  0.9821. Recomputing the mean AUC from the committed arms gives **0.9870 against 0.9872**, which is what the row says
  now. The reading, a tie with a fraction in Heretic's favour, is unchanged.
- **Corrected 2026-09-25, against us: hard refusal was printed as 0.0% on both sides.** Heretic's is 0.0% across all five
  seeds. Ours is **0.1%**: four seeds at zero and seed 46 at 0.5%.
- **The drift figures carry two prompt counts, and only one is current.** The superseded 2026-08-12 row was measured on
  64 prompts. The 2026-09-10 figures use 200 held-out prompts, and `head-to-head` takes its drift slice at 200 as of
  2026-10-10. Cite the 200-prompt figures.
- **Our arms did not pin the direction budget, so the table mixes K.** Three of the five arms recorded the count they
  applied, and it was two, one and two. The other two could not be recovered. This says nothing either way about
  whether more directions help. The clean contrast is the 2026-08-13 K comparison, which pinned the budget at both
  ends. See [what is and is not established](/guide/what-we-know).
- **The equal-budget matching is not symmetric.** Trials were matched up to Heretic's 200. Direction-fitting prompts
  were matched down to our 256, where Heretic's default is 400 per side. Both moves went away from the other tool's
  larger default. Re-running at 400 per side for both has not been done.
- **A narrower contamination point.** Within one run both tools fit on the identical staged files and are scored on
  identical held-out slices, so overlap is symmetric. What survives: Heretic's keyword scorer was developed against
  this dataset family, so the keyword axis is measured by a ruler built on the ground one tool optimises against.

The recipe below is how you run the comparison today. The 2026-08-12 run was scored with an earlier form of the
harness, and no record of that run is kept in this repository. The 2026-09-10 re-run is the one whose arms are
committed. The 2026-08-12 comparison has not been re-run under the current code.

## Run it yourself

The comparison is a command. It runs on one machine and produces the arms, the scores and the report:

```sh
# 1. Cut the prompt slices every tool is scored on, from one corpus.
senbonzakura head-to-head stage --track mytrack --out slices

# 2. Run every tool over every seed, score every model, print the verdict.
senbonzakura head-to-head run \
    --tools senbon,heretic --seeds 42,43,44,45,46 --trials 200 \
    --model Qwen/Qwen3-1.7B --track mytrack --eval-slices slices \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds \
    --out results/h2h

# 3. Read a finished run again later, without re-running anything.
senbonzakura head-to-head report results/h2h
```

::: tip New word: arm
One tool, one seed, one run. Five seeds and two tools is ten arms. The design rests on the arms being identical
except for the one thing being tested.
:::

## What makes it a comparison

Both tools get the same corpus, the same trial budget and the same prompt slices. Every model either tool produces
is then scored **by one instrument**, our compass, on held-out prompts. Nobody grades their own homework.

::: warning "Same trial budget" is not the same as "same budget"
An equal trial count isn't a fair fight. Our search gets three things Heretic's does not:

| What we get | What it does |
|---|---|
| A warm-start trial | One configuration is queued before the search, so trial 1 is an informed guess |
| A patience early-stop | The search stops once the front stops moving, so equal trial counts may not mean equal trials *run* |
| A re-score pass over the top candidates | The winner is picked from a best-of-N selection, which Heretic had no equivalent of in the 2026-08-12 run |

So budget is reported three ways: trials requested, trials actually run, and total generations consumed. The third
survives all three differences, because a generation is the unit of work both tools actually spend.

The harness gave Heretic the same best-of-N selection from 2026-09-01, and the 2026-09-10 run was the first with
both. Where the two tied, this matters less. Where we lead, read it with this in mind.
:::

The slices record which corpus they were cut from, and the run refuses to start if that doesn't match the corpus you
passed. Without that guard, one tool could read the corpus directly while the other read slices, and nothing on the
command line would show the difference.

Each tool's own reported numbers are printed in separate rows, labelled with whose estimator produced them. **No gap
between those rows is ever called a win.** Different refusal rates might mean one tool is better, or that they count
refusals differently, and from outside those look identical.

## Sandbox anything you didn't write

```sh
--isolate docker --image senbon=IMAGE --image heretic=IMAGE
```

Each arm then runs with no network, read-only inputs, no capabilities and no credentials. Without the flag you get a
warning, because a third-party abliteration tool would otherwise run on your machine with your network and keys.

## Stopping and starting is safe

An arm is skipped only when its manifest agrees with this run's tool, seed, model and budget **and** every artefact it
declared is present. An arm that exits cleanly having produced nothing counts as a failure, so the next run retries it.
On 2026-08-05 a rehearsal finished five jobs having measured nothing, because the scoring step printed success
regardless. A green tick that can't go red isn't a check.

## Fewer than three seeds gets no verdict

A spread from two points is arithmetic dressed up as statistics, so the report declines to name a winner. A gap smaller
than the spread is reported as a tie, not as a narrow victory.

## Adding another tool

An adapter is data rather than logic: how to invoke the tool, what output proves it ran, where it leaves its model, and
how to read its own figures. See `ADAPTERS` in `senbonzakura/headtohead.py`. Pull requests adding one are welcome,
including from the author of the tool being added.

## Where next

- [The compass](/guide/compass), the instrument doing the scoring here.
- [The track](/guide/the-track), for checking whether the prompts you compare on are already inside your fit rows.
