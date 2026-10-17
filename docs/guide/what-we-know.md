# What is and is not established

This page covers the claim on the tin: that removing several refusal directions beats removing one.

::: warning The short version
**We tested it, and on the one model we can measure properly, two directions did worse than one.**
Two directions cost about **1.9 times** the collateral damage of one, on the mean over five seeds a side
(95% interval 1.2 to 3.1, percentile bootstrap over the five seeds a side; 1.4 on the mean dropping the one
outlying two-direction seed, which is a sensitivity check rather than the bottom of an interval).

**That is not the same as the claim being false.** Whether removing several directions beats removing one is
**unanswered**. Three things stand between the result and the claim. It is one model. The extractor had never
worked before 2026-08-03, so the arms predate the held-out direction selection the tool now ships. That means
what they price is an *arbitrary* second direction, not a chosen one. And the two arms are comparable on
collateral damage because both landed at the floor on refusal, 1 hard refusal in 1000 against 3 in 1000. That
is a difference no test could call apart (Fisher exact, two-sided, p = 0.62), so the arms weren't tuned to meet
at one rate.
:::

## The feature that never worked

The comparison meant to settle the question was built to test whether removing several directions beats one: two
arms, five seeds each, held out, identical except for the direction budget. It finished and had a p-value.

**Both arms had removed exactly one direction per layer.** No second candidate cleared the refusal-separation
threshold at any of the model's 29 layers, so the surgery in the two arms was identical. The comparison compared
two searches, not two direction budgets, and it was withdrawn the same day.

The reason no second candidate passed is worse than strict. The check compares the average harmful reply against
the average harmless one, measured along a direction constructed to sit at right angles to both averages. A
difference measured along a perpendicular axis is zero, every time, by construction. **No direction could pass that
check, on any model, at any setting.** It was a metal detector wired so the coil could never complete a circuit.

Across two model families and three prompt sets, that check rejected **13,970 candidate directions, every one**.

::: tip New words: refusal-separation threshold
The check that decides whether a candidate direction carries refusal, rather than something nearby. The whole
multi-direction idea rests on it.
:::

**So what:** every run before 2026-08-03 applied exactly one direction, whatever it was asked for.

*Source, stated rather than implied: the 13,970 figure is a count from the run logs of that sweep, which are not
published. It is a measurement, not an estimate, and you cannot currently check it. `REPRODUCING.md`, in the
repository root, says a quoted number with no published source is a defect, so it is named here rather than left
as a number with no provenance.*

## What's been fixed, and what hasn't

Candidate directions now come from clustering the harmful prompts and taking each cluster's own average against the
harmless average. Because each candidate is built from its own group, it separates the two sets by construction. The
tool allows **up to eight directions per layer**. Eight is a ceiling, not a count any run reached, and this page
doesn't record how many a run applied.

The threshold meant to vet candidates rejected nothing at all: all 678 candidates scored between 0.90 and 9.02
against a bar of 0.5. The cause was the same circularity. A candidate's score was computed on the same rows its
direction was built from, so it could not come out small. Each candidate is now fitted on half its rows and scored
on the half it never saw. The threshold also has a measured floor: directions built from random subsets of the
harmful prompts, scored the same way. A candidate has to beat their 95th percentile, taken at nearest rank. That's
a 5% false-keep rate per candidate by design.

*Source, stated rather than implied: the 678 count and the 0.90 to 9.02 range are read from that run's logs, which
are not published. Measured, not estimated, and not checkable by a reader.*

**What is still open is the part the fix doesn't reach.** A held-out score says a direction separates harmful
prompts from harmless ones. It doesn't say *why*. A cluster about explosives separates from the harmless prompts
partly because it's about explosives, and it separates just as well on rows it never saw. Cut along it and you've
removed the model's chemistry, not its reluctance.

Telling those apart needs the subject matter held still, the way a medical study compares patients against matched
controls. On 2026-09-02 we measured how badly the unmatched comparison fails, on synthetic data where the answer was
known. Two worlds, identical except that one contains refusal and the other doesn't. Every candidate should be
rejected in the second.

| how a candidate is judged | world with no refusal | world with refusal |
|---|---|---|
| against harmless prompts in general | 3.547 | 3.703 |
| against harmless prompts on the same subject | **0.319** | **4.369** |

Read the top row twice. **The two worlds score the same**, so no threshold on that scale could separate them, and
we tried four statistics. The bottom row is the same data judged against matched controls: the world with no
refusal lands on the scale's "nothing here" value, and the world with refusal stands clear.

*Source, stated rather than implied: these four values come from a synthetic experiment whose result files are held
outside this repository and are not published, so a reader cannot re-derive them here. The experiment is synthetic by
design, which is the point, but that does not make the numbers checkable.*

Two things follow from this.

The first is `--matched-scoring`, which picks each candidate's controls from the harmless prompts nearest to it in
content. **It is on by default, and it changes what every separation number means.** It has been on since
2026-09-17, and it still hasn't been measured on a real model. Pass `--no-matched-scoring` to reproduce an older run.

That leaves three candidate selection regimes in play. The 2026-08-13 arms, the only multi-direction evidence here,
ran unmatched and in sample against a filter that could not reject anything. What ships today is matched by default.
**The published evidence corresponds to neither**, which limits how far those arms can speak about the installed tool.

The second is that **a topic-matched harmless corpus would not have been enough on its own**. In the numbers above
the harmless side already had the same subject mix, and the unmatched comparison still couldn't tell the two worlds
apart. The matching has to happen when each candidate is scored, not only when the corpus is built.

One honest caveat: those figures are synthetic. Whether subject matter swamps refusal by that much inside a real model
is unmeasured, so the question stays open until it is measured.

**Does removing several directions beat removing one?** We ran it. On the one model we can measure properly, the answer
is **no**. The next section has the run.

## The answer: run on 2026-08-13

One direction against two. Same tool, same corpus, same search budget, same five seeds, one parameter apart, with the
direction budget pinned at both ends. Every resulting model was scored by one instrument on 200 prompts that nothing
was fitted or selected on.

| Direction budget | Coherence drift | Hard refusal | Noncompliance |
|---|---|---|---|
| One | **0.0497** (spread 0.0177) | 0.1% | ~~3.8%~~ |
| Two | 0.0932 (spread 0.0482) | 0.3% | ~~4.2%~~ |

::: warning The noncompliance column is withdrawn, 2026-09-09
The arms ran on 2026-08-13. The noncompliance column was scored separately on 2026-09-07. The next day the hedging
detector was found to count statements of fact about legality or danger as hedging, so a compliant technical answer
was scored as a soft refusal. Both figures came from it. The conclusion does not rest on them. It rests on coherence
drift at matched hard refusal, and hard refusal is measured by a different ruler that this defect did not touch. The
column is struck through because it was published.
:::

Both budgets removed hard refusal, so this is a comparison at matched refusal, and the only thing left to differ is
the price paid. Two directions cost about **1.9 times the collateral damage**, bought nothing this design could have
detected, and were less predictable seed to seed (the spread is nearly three times as wide).

**The cost.** 1.9 is the ratio of the means over five seeds a side. Its 95% interval is 1.2 to 3.1, a percentile
bootstrap over the five seeds a side. The 1.4 figure is a sensitivity check, the ratio with the outlying two-direction
seed dropped. It is not the lower end of the interval. The interval is four times wider than a span joining the two
figures would suggest, and that span is withdrawn. The CHANGELOG's 0.4.0 entry carries the old wording and the
correction.

**The absence of a benefit.** Both arms sit on the floor of the refusal measurement, so this design could not have
shown a gain. One direction already leaves hard refusal at 0.1%, the measurement moves in steps of 0.5 points at 200
prompts an arm, and the largest gain two directions could have shown is 0.1 points, a fifth of one scorable event. So
"bought nothing" is what this run can support, and "bought nothing measurable at this operating point" is what it
means. Whether a second direction buys anything where one leaves refusal standing needs a run where refusal has room
to move.

The gap is 0.0435. A **two-sided exact permutation test on the difference of means**, over all 252 ways of splitting
these ten seeds, puts it at **p = 0.016**. The sidedness is named because the one-sided version gives 0.008, and a
reader who landed there first would conclude we had understated our own negative result.

::: tip You can recompute this yourself
The ten per-seed drift values are committed at
[`evidence/k-sweep-2026-08-13/drift-per-seed.json`](https://github.com/elementmerc/senbonzakura/blob/dev/evidence/k-sweep-2026-08-13/drift-per-seed.json),
along with the ten per-seed hard refusal fractions behind the 0.1% and 0.3% column. `tests/test_published_k_sweep.py`
recomputes every figure on this page from that file through the project's own estimator.
:::

**A correction, 2026-09-09.** An earlier version said that dropping the worst two-direction seed made the result
*stronger*. That came from comparing a gap to a pooled spread, which ignores how many observations there are. Under
the permutation test, dropping that seed halves the gap, from 0.0435 to 0.0222, and moves p from 0.016 to **0.048**.
The result does not rest on one bad run, but it is meaningfully weaker without that seed.

**What it doesn't say.** This is Qwen3-1.7B alone. Refusal geometry may differ on larger models or other families, so
the question is open elsewhere and closed here. It also doesn't rescue the threshold problem above. These arms ran
before the held-out score existed, so their directions were selected by a filter that could not reject anything. "Two
directions" means "two directions this tool chose", not "two directions known to carry refusal". A re-run under the
fixed filter may select different directions, and this table is the old filter's answer until it does.

**Why publish a negative result about your own headline feature?** Because the alternative is waiting for a friendlier
model. A project whose argument is that other people's numbers deserve scrutiny doesn't get an exception for its own.

## So where does that leave the project

With a working single-direction abliterator, an unusually good [measurement suite](/guide/compass), and a headline claim
its own measurements do not support on the model it can measure best.

The tool keeps the multi-direction capability because the question is open on architectures we can't yet reach, and
because testing an idea is worth more than believing it. By default the search chooses its own budget between one and
three directions. On this model the count varies from seed to seed: the 2026-09-10 head-to-head recovered the count for
three of its five arms, and they applied two, one and two.

## Where next

- [Limits and known defects](/guide/limits) for every other condition attached to the numbers here.
- [The compass](/guide/compass) for the half of the project that does the measuring.
