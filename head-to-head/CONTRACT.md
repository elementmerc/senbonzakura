# The abliteration benchmark contract

**What this is.** A fixed way of comparing tools that remove refusal from language models, so that
two numbers from two tools mean the same thing. The contract is the contribution. The results
table is downstream of it, and we publish ours whether or not they flatter us.

**Status: no results in this repository, and a published table that predates them.** Section 4 is
still empty and that is the honest state of this contract: nothing under `results/` records a run.

A run did happen. Five arms of each tool were scored on 2026-08-12 and the figures are published
on the documentation site's benchmark page. No record of it was kept here, so those figures do not
meet the traceability this contract promises in section 4, and the page now says so. The harness
has also been renamed and repaired since, so the commands in this directory are not the ones that
produced them. Re-running under the current code, and keeping the records this time, is what
closes the gap.

**Run it yourself and tell us we are wrong.** Everything needed is in this directory. If a number
here is unfair to a tool you wrote or use, we would rather hear it than defend it: open an issue
with the command you ran and what you got.

---

## 1. The problem this exists to solve

Every abliteration tool reports that it removed most of a model's refusals. Almost none report
what that cost, and the few that report a cost do not report it at a level where the tools can be
compared.

There are three ways a comparison goes wrong, and all three are common:

**Comparing at each tool's defaults.** Defaults encode the author's taste about how much damage is
acceptable. One tool targets a KL of 0.01, another targets a refusal rate, another stops when a
search plateaus. A table built from defaults tells you which author is more cautious, not which
method is better.

**Comparing refusal removal without its cost.** More ablation always removes more refusal. Any
tool can reach zero refusals by cutting hard enough, and arrive at a model that answers everything
with the same sentence. A refusal number with no cost beside it is not a measurement.

**Comparing at unmatched refusal levels.** Two tools that removed different amounts of refusal
cannot have their costs compared, because the one that cut harder pays more whatever its method
is. This is subtle and it is the one that catches careful people. It caught us: four claims in
our own write-up were withdrawn on 2026-08-05 for exactly this.

## 2. The contract

### 2.1 One judge for every tool, and it is not the tool's own

Every model produced by every tool is scored by the same instrument. Tools that ship their own
refusal judge do not use it here. Otherwise the comparison is partly between judges, and a tool
whose judge is lenient about its own output wins for the wrong reason.

We use senbonzakura's local judge, which is a keyword-and-classifier scorer that needs no external
API. That is a choice with a bias we should name: it is our instrument, and we built it. Two
mitigations. First, it reports the verbatim Heretic keyword rate alongside its own metric, so a
reader who trusts that metric more can use it. Second, every generation is kept, so anyone can
rescore the same outputs with a different judge and publish the difference.

### 2.2 Every tool is swept, never run once

Each tool exposes some knob that trades refusal removal against damage: an ablation strength, a KL
target, a number of trials. Each tool is run across **at least five points** on its own knob,
chosen to span from "barely ablated" to "clearly over-ablated".

The result of a tool is therefore not a point. It is a **frontier**.

### 2.3 The comparison is at matched refusal removal

Tools are compared by reading off the frontier at a fixed level of refusal removed: 50%, 75% and
90% of the model's original refusal rate.

A band is only reported when the arms in it are actually at the same level. "The same" is two
standard errors of the difference between two proportions at the eval size used, not a fixed
percentage: at 15% refusal on 128 prompts that is about 6 points, and a comparison of arms further
apart than that is not a comparison. Where no arm reaches a band, the cell says so rather than
borrowing the nearest arm.

**NOT YET DELIVERED, and stated here rather than discovered by a reader.** The first
head-to-head produces, for every arm of both tools, a harm-recognition score from one instrument
applied afterwards, plus each tool's own self-reported refusal and KL. That supports a comparison
on harm recognition, which `tools/report_head_to_head.py` makes with the tie rule above. It does
**not** yet support the matched-refusal band comparison this section describes, because reading
two tools at the same refusal level needs each tool's frontier materialised and re-measured with
one KL estimator, and the arms as run give one configuration per seed rather than a sweep.

The two tools' own KL figures are not a substitute and are not presented as one: ours comes from
our estimator on our coherence slice, Heretic's from its own evaluation, and a column containing
both would be the error this project withdrew four claims for on 2026-08-05.

So this section describes what the benchmark is for, and the first run is a step towards it rather
than the whole of it. The work is recorded in `DEFERRED.md`. Nothing published from the first run
will claim a matched-refusal comparison.

### 2.4 A random-direction floor, where the method admits one

Tools that work by removing directions get a control arm in which some of those directions are
replaced by random ones of the same count. Without it, "removing K directions helped" cannot be
told apart from "removing more of the space helped".

This does not apply to every tool and the table says where it does not.

### 2.5 Both corpora

Every tool is run on both contrast corpora: ours and the abliterix bilingual set (both
AGPL-3.0). A single-corpus benchmark bakes that corpus's defects into every row. Ours has a
measured one: a classifier reading nothing but prompt length separates its two classes 67.4% of
the time, against 53.3% for abliterix. Anything a ruler can separate, a direction-finding method
can pick up instead of refusal.

### 2.6 Equal budget, stated in three units

Search-based tools are given the same budget, defined as **trials and wall-clock and total
generations consumed**, counting every selection or re-scoring stage.

This matters more than it sounds. Senbonzakura's search gets three things Heretic's does not: a
warm-start trial enqueued before the search begins, a patience-based early stop, and a re-scoring
pass over its best candidates. Comparing "200 trials against 200 trials" would hand us three extra
chances and call it equal. So Heretic is given an equivalent best-of-N selection pass, applied
outside its own code, and the budget artefact records exactly what each tool received.

If a tool cannot be equalised, the table says what advantage it had rather than quietly keeping it.

### 2.7 What every row reports

| Column | Why |
|---|---|
| Refusal rate, before and after | The headline everyone already reports |
| **KL at matched refusal removal** | The cost, at a level where costs can be compared |
| Over-refusal on harmless prompts | Ablation that breaks normal use is not a success |
| Degeneration rate | Repetition and collapse, which a refusal metric will not catch |
| Coherence | That the model still forms sentences |
| **Runtime and peak VRAM** | What it costs to actually run. Nobody publishes this and everybody cares |
| Random-direction floor | Where the method admits one |
| Tool version, image digest, corpus, seed | So a disputed number can be traced |

### 2.8 Seeds and intervals

Five seeds per arm. Every reported figure carries an interval. **A gap smaller than the spread is
reported as a tie**, not as a win.

### 2.9 Isolation, and why it is part of the contract

**Every tool runs in a container, ours included**, with no network, inputs mounted read-only, no
credentials of any kind, and a wall-clock bound. See `run-isolated.sh`.

*Amended 2026-08-06.* This used to say "each third-party tool", which was the original intent: the
container exists to protect the card from somebody else's dependency tree, and by that argument
our own arms did not need one. Running them outside it would have put senbonzakura on the host's
torch and the other tool on the image's pinned build, which is two sets of kernels underneath a
table claiming one environment. The run now refuses to start if the two images disagree about
which torch they carry, and our arm is audited by the same self-test as everybody else's.

This is in the contract rather than in a footnote because it constrains the measurement: a tool
that cannot reach the network cannot download anything mid-run, so every input is staged in
advance and the run is reproducible. The security property and the reproducibility property are
the same property.

## 3. Scope, and its honest limits

**Hardware.** One 6 GB laptop GPU. That is the ceiling and it is a real one: it puts models above
about 2B out of reach, and Heretic's own README reports a 12B model while abliterix's benchmark
spec asks for 24 GB dense and 80 GB for mixture-of-experts. **This benchmark says nothing about
how these tools behave at the sizes they are usually used at.** Results at 2B may not survive at
8B, and we would not be surprised either way.

**First version.** Two tools, two models, done properly, rather than four tools measured
carelessly:

| | |
|---|---|
| Tools | senbonzakura, Heretic |
| Models | Qwen3-1.7B, Llama-3.2-1B |
| Corpora | ours, abliterix |
| Seeds | 5 per arm |

Abliterix and OBLITERATUS are next, as additional rows under the same contract.

**Gemma is excluded from the first version, and the reason is ours.** Until 2026-08-05 our own
weight edit did not reach the residual stream on Gemma-family models, because they normalise each
sublayer's output before adding it to the stream and we edited upstream of that step. It is fixed,
and Gemma is excluded until our own numbers there have been re-measured, so that a bug of ours is
not published as a property of a model.

## 4. Results

Nothing here yet, and that is the point rather than an oversight: the 2026-08-12 run was published
without leaving a record under `results/`, so this section stayed empty while a table went up. A
cell with no record behind it is not traceable however carefully it was measured. The table lands
here when the re-run finishes, with every cell traceable to a record under `results/`.

## 5. What would make this benchmark wrong

Stated in advance, because a benchmark that cannot say how it would fail is advocacy:

- **The judge is ours.** If our judge systematically favours the kind of output our tool produces,
  every row is tilted and the effect would be invisible from inside. The generations are kept
  precisely so someone else can check this.
- **Our corpus has a measured length confound**, which is why both corpora are run. If results
  disagree across corpora, the disagreement is the finding.
- **Equalising budget requires a judgement** about what counts as a selection stage. Ours is
  written down in the budget artefact and is arguable.
- **We are not neutral.** We wrote one of the tools. The contract, the harness and the raw
  generations are published so that our conclusions can be checked against our evidence rather
  than taken on trust.
- **2B models on one card.** See scope.

## 6. Licences

Senbonzakura is AGPL-3.0-or-later and a derivative of Heretic. Heretic, abliterix and OBLITERATUS
are AGPL-3.0. The abliterix corpus is AGPL-3.0. No tool's code is copied into this repository;
each is cloned at a pinned ref at run time and the ref is recorded with the result.
