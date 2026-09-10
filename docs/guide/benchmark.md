# Benchmarking against another tool

The head-to-head against [Heretic](https://github.com/p-e-w/heretic), the other open-source
abliteration tool with an automated search, has been run: five seeds each, both tools driven by
us on one machine, every model scored afterwards by our instruments on prompts neither tool was
fitted on.

::: warning These numbers are provisional, and several things are still open
They are published here because the alternative is a page that says nothing while we know
something. Every open item is named below the table, and one row has since been withdrawn
outright. Nothing here is hidden and nothing here is settled.
:::

::: danger The coherence row is superseded, 2026-09-10. Do not quote it.
The table below is the 2026-08-12 run, in which **Heretic was not given the best-of-N selection
pass that Senbonzakura gave itself**. When both tools got that pass, on 2026-09-10, the coherence
difference disappeared:

| Coherence drift, both tools selected the same way | Senbonzakura | Heretic |
|---|---|---|
| mean over five seeds | 0.0545 | 0.1227 |
| **median** over five seeds | **0.0510** | **0.0573** |
| worst seed | 0.0936 | 0.3591 |

One Heretic seed (seed 43, at 0.3591 against siblings of 0.0439 to 0.1075) carries the entire mean
difference. Drop it and Heretic's mean is 0.0636 against our 0.0545. An exact permutation test over
the ten seeds returns **p = 0.238**, so there is **no detectable coherence difference on this model
at this size**, and the earlier "roughly half the collateral damage" is not a finding.

The per-seed values are listed above in full precisely so this block does not have to be taken on
trust. Publishing the run's thirty artefacts, which are aggregate JSON with no prompts and no
weights in them, is an open decision rather than an oversight: this repository's ignore rules
exclude every `results/` directory by default, under a heading that begins "No loaded guns", and
widening that is not a change to make casually. `CONTRACT.md` §4 asks for the record; until it is
here, these numbers are a report of what we measured rather than something you can re-derive.

The same run put Heretic **marginally ahead** on refusal removal, not behind. Both differences are
negligible; the direction was not, and it was reported the wrong way round here for a month.

The re-run's arms are not yet committed to this repository, which is a gap named in
[CONTRACT.md](https://github.com/elementmerc/senbonzakura/blob/main/head-to-head/CONTRACT.md) 4
and not yet closed. Until they are, treat every number on this page as provisional in both
directions.
:::

| Axis | Senbonzakura | Heretic |
|---|---|---|
| Hard refusal | 0.0% | 0.0% |
| Noncompliance (refusal plus hedging) | ~~3.6%~~ | ~~1.9%~~ withdrawn, see below |
| Keyword rate | 16.1% | **9.6%** |
| Coherence drift | ~~0.191~~ | ~~0.341~~ superseded, see above |
| Harm recognition | 0.9807 | 0.9821 (tie) |

**Both tools took hard refusal to zero, which is what makes the rest readable.** What that leaves
is the price each paid for it, and on the one run where both tools were selected the same way,
that price is indistinguishable. Heretic won its own keyword metric.

::: danger The noncompliance row is withdrawn, 2026-09-09
This run was scored on 2026-08-12. On 2026-09-08 the hedging detector was found to count
statements of fact about legality or danger as hedging, so a complete, compliant technical answer
that mentioned an activity is illegal was scored as a soft refusal. That fires asymmetrically: a
more explanatory model collects more of them whether or not it complied. Both figures in that row
were produced by it, so neither is a measurement of what its name says, and the row used to bold
the other tool's number as a win.

The row stays visible with a line through it rather than being deleted, because it was published
and people read it. It will be re-measured when the head-to-head is re-run under the corrected
detector, and not before.
:::

**The two tools' self-reported figures are not a comparison, and that is the durable point.**
Heretic self-reports a coherence divergence of 0.0014 to 0.0032 against our 0.157 to 0.212, a
hundredfold apart, because each measured its own model on its own prompts during its own search.
Put them on one instrument, on prompts held back from both, and that hundredfold gap goes away. An
apparent reversal was published here for a month on the strength of a run in which only one tool
got the selection pass.

### What is still open

- **We lose the keyword axis that we ourselves optimise**, 16.1% against 9.6%, and our spread on
  it is three times theirs. A number moving the wrong way on your own objective is usually the
  ruler rather than the model, and that is being investigated before this table is treated as
  settled.
- **The drift figures in the table were measured on 64 prompts** where the other axes use 200,
  and that slice is the one Heretic tunes against. The re-measurement on 200 held-out prompts has
  since been done, and it is the superseding block at the top of this page: no detectable
  difference.
- **The equal-budget matching is not symmetric, and it should be.** Trials were matched UP to
  Heretic's 200, on the stated principle that starving the comparison would decide it for us.
  Direction-fitting prompts were matched DOWN to our 256, where Heretic's shipped default is 400
  per side. Both moves went away from the other tool's larger default, and this one lands on the
  estimator its method depends on. Re-running at 400 per side for both is queued work.
- **A narrower contamination point than this page used to make.** It previously said an unknown
  share of our evaluation was Heretic's training data and that this flattered Heretic. In the
  harness both tools fit on the identical staged `bad.txt` and `good.txt` and are scored on
  identical held-out slices, so within a run the overlap is symmetric and flatters neither. What
  survives is narrower and still worth stating: Heretic's keyword scorer was developed against
  this dataset family, so the keyword axis is measured by a ruler built on the ground one tool
  optimises against.

The recipe below is how you run this comparison today. It is **not**, as this page used to claim,
the same command that produced the table: that run was scored on 2026-08-12, and the harness has
been renamed and repaired since, so the command shown here did not exist in this form when those
figures were measured. No record of the run is kept in this repository either, which the harness's
own contract requires and which this page should have said. Re-running it under the current code
is queued work, and until that happens the figures above are a report of what we saw rather than
something you can reproduce from this page.

## Run it yourself

The comparison is a command, not a private script we keep in a drawer. It runs on one
machine, needs no orchestrator, and produces the arms, the scores and the report:

```sh
# 1. Cut the prompt slices every tool is scored on, from one corpus.
python -m senbonzakura.headtohead stage --track mytrack --out slices

# 2. Run every tool over every seed, score every model, print the verdict.
python -m senbonzakura.headtohead run \
    --tools senbon,heretic --seeds 42,43,44,45,46 --trials 200 \
    --model Qwen/Qwen3-1.7B --track mytrack --eval-slices slices \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds \
    --out results/h2h

# 3. Read a finished run again later, without re-running anything.
python -m senbonzakura.headtohead report results/h2h
```

::: tip New word: arm
One tool, one seed, one run. Five seeds and two tools is ten arms. The word comes from
clinical trials, where an arm is one group of patients getting one treatment, and it's the
right word here for the same reason: the whole design rests on the arms being identical
except for the one thing you're testing.
:::

## What makes it a comparison rather than two runs

This is the part that's easy to get wrong and hard to notice you got wrong.

Both tools get the same corpus, the same trial budget and the same prompt slices. Then
every model either tool produces is scored afterwards **by one instrument**: our compass, on
held-out prompts, run by us. Nobody grades their own homework.

::: warning "Same trial budget" is not the same as "same budget", and we had the advantage
An equal trial count sounds like a fair fight. It isn't quite, because senbonzakura's search
gets three things Heretic's does not, and all three spend effort a trial count doesn't show:

| What we get | What it does |
|---|---|
| A warm-start trial | One configuration is enqueued before the search begins, so trial 1 is an informed guess rather than a random draw |
| A patience early-stop | The search can stop once the front stops moving, so an equal trial count may not be an equal number of trials *run* |
| A re-score pass over the top candidates | The best few are measured again and the winner is picked from that, which is a best-of-N selection Heretic has no equivalent of |

So budget is reported three ways rather than one: **trials requested, trials actually run, and
total generations consumed**. Those are in every arm's artefact, and the third is the one that
survives all three differences above, because a generation is the unit of work both tools
actually spend.

**Heretic was not given an equivalent best-of-N pass in the run above.** We are saying so rather
than adjusting for it: the honest position is that our arm had a selection advantage the other did
not, and any result where we win by less than that advantage is not a result. Where the two tied,
this matters less; where we lead, read it with this paragraph in mind.

That changed on 2026-09-01, after the table above was measured. The harness now gives Heretic the
same best-of-N selection, applied from outside its own code, and `head-to-head/EQUAL-BUDGET.md`
records the reasoning. **It has not yet produced a number**: no Heretic arm has run through it, so
the table above is still the one measured without it. The two paragraphs read as a contradiction
if you meet them without their dates, so here are the dates.
:::

The slices record which corpus they were cut from, and the run refuses to start if that
doesn't match the corpus you passed. That guard exists because of a hole found while writing
the tests: once one tool reads the corpus directly and the other reads slices cut from it,
"both tools read the same corpus" stops being visible anywhere on either command line. Every
arm would have finished, every artefact would have been present, and the table would have
meant nothing at all.

Each tool's own reported numbers are printed too, in separate rows labelled with whose
estimator produced them. **No gap between those rows is ever called a win.** Two tools
reporting different refusal rates might mean one is better, or it might mean they count
refusals differently, and from the outside those look identical.

## Sandbox anything you didn't write

```sh
--isolate docker --image senbon=IMAGE --image heretic=IMAGE
```

Each arm then runs with no network, read-only inputs, no capabilities and no credentials.

Without it you get a warning, because otherwise a third-party abliteration tool is running
on your machine with your network and your keys, on the strength of you having been curious
about a benchmark. We use it on our own arms too, which is less about trusting ourselves and
more that an isolated arm and a comfortable one aren't the same measurement.

## Stopping and starting is safe

An arm is skipped only when a manifest agrees with this run's tool, seed, model and budget
**and** every artefact it declared is present.

An arm that exits cleanly having produced nothing counts as a failure and leaves no
manifest, so the next run retries it rather than inheriting the silence. This is a rule
learned the hard way: on 2026-08-05 a rehearsal came back with five jobs done having
measured absolutely nothing, because the scoring step printed success unconditionally. A
green tick that can't go red isn't a check.

## Fewer than three seeds gets no verdict

A spread calculated from two points is arithmetic dressed up as statistics, so the report
declines to name a winner and says why. And a gap smaller than the spread is reported as a
tie, not as a narrow victory.

::: tip New word: seed
The number that fixes all the random choices in a run. Same seed, same choices, same result.
Different seeds tell you how much of your result was the method and how much was luck, which
is why one seed is an anecdote.
:::

## Adding another tool

An adapter, and it's data rather than logic: how to invoke the tool, what output proves it
actually ran, where it leaves its model, and how to read its own reported figures.

See `ADAPTERS` in `senbonzakura/bench.py`. A pull request adding one is genuinely welcome,
including from the author of the tool you're adding.

## Where next

- [The compass](/guide/compass), which is the instrument doing the scoring here.
- [Contamination](/guide/contamination), because a comparison on prompts both tools trained
  on isn't one.
