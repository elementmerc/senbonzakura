# Equal budget, defined before the run

**Committed before the head-to-head executes**, so it is checkable by someone other than its
author. The v0.4 exit gate requires this and requires it in three units, because a single unit
hides the advantage.

## Why one number is not enough

"Two hundred trials each" sounds equal and is not. A trial is not a fixed amount of work, and a
search is not the only place a tool spends effort. Senbonzakura spends effort in three places
Heretic does not, and every one of them is invisible in a trial count.

### What senbonzakura gets that Heretic does not

Read from `src/senbonzakura/cli.py`:

| Advantage | Where | What it buys |
|---|---|---|
| **Warm start** | `:1855-1873`, `--warm-start` on by default | The search is seeded with a difference-of-means configuration before any random sampling, so trial one is already a working answer rather than a draw from the prior |
| **Patience early stop** | `:537`, `:1830-1842`, `patience = max(20, trials // 3)` | The search stops once the front stops improving, so its trials are spent where they help. With the default 60 trials that is a patience of 20 |
| **Re-scoring pass** | `:536`, `:1945`, `top_rescore = 6` | The best six candidates are re-scored and the winner picked from that second look. This is a best-of-six selection stage that costs generations and appears in no trial count |

The third is the one that matters most. A best-of-six pass over a noisy objective is worth a
noticeable amount on its own, and a benchmark that ignores it is measuring our selection procedure
and calling it our method.

### What Heretic gets that senbonzakura does not

Stated for symmetry, because a one-sided accounting is not an accounting:

- **200 trials by default** against our 60. On trial count alone Heretic is given more than three
  times the search.

  *Provenance, added 2026-10-01.* The figure is 200 **at tag `v1.4.0`**, which is the ref
  `Dockerfile.heretic` pins and the ref every published arm ran against. **Two places hold it and
  both are named, so a reader checking this lands on it either way**: `config.default.toml` carries
  `n_trials = 200`, and `src/heretic/config.py` carries `n_trials: int = Field(default=200, ...)`.
  Both read at that tag on 2026-10-01. **Upstream's unreleased `master` lowers the same default to
  100**, so this number is version-specific and now says so. **The published comparison used 200**,
  which is the v1.4.0 default, and nothing below changes.
- **Two scorers on a Pareto front** (keyword rate, KL divergence), the same shape as ours. *True at
  `v1.4.0`, where the two are fixed. On `master` the scorers are plugins and the list is
  configurable, with `KeywordRate` and `KLDivergence` still the default pair, so the shape survives
  and the mechanism is no longer fixed.*
- Heretic is the upstream this project derives from, so its defaults have had far more exposure to
  real models than ours have.

## The definition

A run's budget is the triple, all three recorded per arm:

1. **Trials.** Every objective evaluation, including any that early-stopping skipped, reported as
   both "trials configured" and "trials actually run".
2. **Wall clock.** Seconds from tool invocation to final artefact, on the same card, with nothing
   else on it. The GPU lock is held for the whole arm.
3. **Total generations.** Every prompt generated during the entire run, *including the re-scoring
   stage*. This is the unit that catches selection passes, and it is the one a trial count hides.

## How the arms are equalised

**Heretic is given an equivalent best-of-N pass.** The operator chose this over the gate's other
option, which was to state the advantage and leave it in place. Equalising is the stronger result:
if senbonzakura wins with its selection advantages removed, the win is about the method.

**Since 2026-09-01. Exercised on 2026-09-10.**

Appended rather than rewritten, on 2026-09-25. The paragraph below is what this file said between
2026-09-01 and 2026-09-10, kept verbatim because a pre-commitment that gets edited after the run is
not one. What made it worth keeping is that it went stale and nothing noticed:

> **Since 2026-09-01, and not yet exercised.** This describes the harness as it now stands, not the
> comparison published on the documentation site: that table was measured on 2026-08-12, three weeks
> before this pass was wired in, and its own page says Heretic was not given the pass. Both
> statements are true of different moments and neither used to say which, which read as a flat
> contradiction. No Heretic arm has run through this pass yet, so nothing published rests on it, and
> the rule it should apply is itself an open question (see the S1 decision brief).

**What is true now.** The equal-budget pass ran on 2026-09-10, and the published comparison at
`head-to-head/results/2026-09-10/` rests on it: five seeds a side, Heretic's arms selected through
this pass. The documentation site's table was corrected to say so on 2026-09-25.

**Why this matters more than an ordinary stale sentence.** This file is the pre-commitment, and it
is what a sceptic is pointed at to check that the rules were fixed before the run rather than after
it. For a fortnight it told that reader the fair comparison did not exist, while three rows of the
published table rested on it. A reviewer doing the right thing, ignoring the summary page and
opening the pre-registration, would have concluded the published table was the old unfair design
relabelled. Found by the 2026-09-25 review panel.

Concretely:

- Heretic runs its search normally. Its Optuna study retains every trial.
- Its top six candidates are re-scored on the same held-out slice senbonzakura's six are re-scored
  on, with the same generation budget per candidate.
- The winner of that re-score, picked by the same weighted knee scalar, is Heretic's reported arm.

This is implemented **outside Heretic's code** (`head-to-head/best_of_n_heretic.py`), reading its study,
so nothing about the tool is modified and the comparison is still of the tool as its author wrote
it plus a selection pass we added to match our own.

### How each tool's six are chosen

*Amended 2026-08-05, before any arm ran, when the pass was implemented.* The paragraph above
originally said Heretic's six would be picked "by the same scalarisation senbonzakura uses". That
is not implementable at any budget worth spending, and the correction matters enough to state
rather than quietly fix.

Our scalariser reads measurements. Applying it to Heretic's 200 trials (the v1.4.0 default, as
above) would mean rebuilding and
re-scoring all 200 models, which is the entire search over again and would hand Heretic several
times the compute nothing else in the comparison gets. Worse, it would not be the equivalent
procedure: senbonzakura's own six are nominated from the numbers *its* search already measured, not
from a fresh pass over every trial.

So each tool nominates its six the way it ranks its own front, from its own in-search scores. The
pass is equivalent where it has to be: **the same six-candidate depth, the same larger held-out
slice, the same rulers, the same knee scalar, and KL taken from the trial in both cases** (the
re-score refreshes the refusal axes on more evidence and leaves the coherence axis alone, which is
what ours does).

### Both scorers read our slices, and that is not a workaround

Heretic's two scorers each own an evaluation prompt set, separate from the corpus the directions
are fitted on, and both default to a Hugging Face dataset. Inside a box with no network those
defaults cannot load at all. They are pointed instead at the slices senbonzakura is scored on,
written by `head-to-head/stage_eval_slices.py` from the same code that builds them for our own arm.

*Described at `v1.4.0`, added 2026-10-01, where the four prompt sets are fields of `Settings`. On
`master` the fitting sets belong to the `Abliteration` modifier and the evaluation set to the
`KeywordRate` scorer, so the structure holds and the config path does not. That is one of the
integration points a pin move would rewrite, and it changes nothing about the requirement below.*

This is a comparability requirement, not a concession to the isolation. A tool's search is steered
by whatever its scorers measure. Two tools optimising against different prompts have not been given
the same problem, and the resulting table would compare evaluation sets while claiming to compare
tools.

### Both tools run in the same sealed box, on the same torch

*Added 2026-08-05, before any arm ran.* The isolation was built to protect the card from somebody
else's dependency tree, so on that argument our own arms did not need it and would have run on the
host environment. That would have left senbonzakura on torch 2.13.0+cu130 and Heretic on the
pinned 2.5.1+cu124: different kernels, different numerics, and a KL divergence measured against a
slightly different baseline. The table would have said "same card", which is true and which a
reader would reasonably take to mean more than it did.

Both tools now run in containers built from one base, and the run refuses to start if the two
images do not carry the same torch. Our own arm is audited by the same self-test as everybody
else's, which is also the answer to a fair question: why would the isolation apply only to other
people's code.

### Heretic's unaided pick is reported alongside

The pass records which trial Heretic's own menu offers first, saves that model too, and scores it
with the same instrument. A reader can therefore see whether the selection we added is what moved
the number, rather than having to take our word that it was applied fairly.

*"Offers first" is a `v1.4.0` behaviour, stated 2026-10-01.* At that tag the menu's order comes
from upstream's own Pareto walk over completed trials, which `best_of_n_heretic.py` reproduces. On
`master` the order comes from `study.best_trials` instead, so **which trial is offered first is
itself version-specific** and an unaided pick taken from a different version is not the same
quantity. Nothing published is affected, because every arm ran at the pinned tag.

**Trials are matched at 200 for both**, which raises ours from its default of 60. Matching upward
rather than down: capping Heretic at 60 would hand us a result that depends on starving the
comparison, and 200 is Heretic's own default so it is the number its author considers adequate.

*The justification is version-specific and the rule is not, stated 2026-10-01.* "The number its
author considers adequate" was 200 at `v1.4.0`, the pinned tag every arm ran against, and is 100 on
upstream's unreleased `master`. **The matching rule does not change and no arm is re-run:** 200 is
still the larger of the two numbers, so matching upward to it still refuses a result that depends
on starving the comparison, and it is still a number the author shipped. What this note fixes is
that the sentence read as a claim about Heretic in general when it was a claim about one tag. If the
pin ever moves, the rule to apply is the same one, which means matching to whichever default is
larger and saying which version it came from.

**The warm start stays on, and is reported.** It cannot be given to Heretic without modifying its
search, and switching it off would benchmark a configuration nobody ships. So the table states
that senbonzakura's arm was warm-started and Heretic's was not, and `--no-warm-start` exists, so a
second row can be added later if the difference is disputed.

**Patience is switched off for both** (`--patience 0`). An early stop makes wall-clock and trials
disagree, and with trials matched at 200 the point of the stop, which is not wasting a budget, no
longer applies.

## What remains unequal, stated plainly

| | senbonzakura | Heretic |
|---|---|---|
| Trials | 200 | 200 |
| Warm start | **yes** | no |
| Patience early stop | off | off |
| Best-of-N selection | 6 | **6, added by us** |
| Judge | ours | ours |
| Corpus | both | both |
| Direction-fit prompts | 256 per side | 256 per side |
| Card | same | same |
| Container, torch, CUDA | same, by construction | same, by construction |

**The warm start is the one asymmetry left**, and it is disclosed rather than hidden. Everything
else is matched.

**"By construction" is doing real work in that last row, and it used to say only "same".** Both
arms run from the same image by design, and that is what the harness enforces. It is not what the
published artefacts record: the `provenance` block in every committed result file describes the
environment of the machine that SCORED the arms, not the machine that ran them, so nothing on disk
can be used to check the claim. A reader auditing this table would find the row unverifiable and
have to take the design on trust. Naming which of the two it is, an enforced property or a recorded
one, is the difference between a claim and a receipt. Publishing the per-arm environment is a
separate open item (see the panel artefact of 2026-09-25); until it lands, this row is an assertion
about the harness and reads as one.

## The pre-commitment

Written before any arm runs. If the numbers come back badly for senbonzakura, this file does not
change: it is committed first precisely so that it cannot. Any later amendment appears in the git
history of this file, where anyone can see it.

**Amendments so far.** Three, all on 2026-08-05 and all before any arm ran: how each tool's six
candidates are nominated, that both tools read our evaluation slices, and that both now run in the
same container on the same torch. They are recorded in the file rather than only in the history,
because an amendment a reader has to go looking for is an amendment that was half hidden.

Every one of them came from building the thing and then running one short arm end to end. That is
the argument for a dry run: the pre-registration was written carefully and was still wrong in
three places that only running it could show.

**Corrections, which are a different kind of change from an amendment, and the count above stays
three.** An amendment changes what was promised. A correction attaches provenance to a claim
without moving it. This file's own rule is that a pre-commitment edited after the run is not one,
so the distinction is recorded rather than assumed, and every correction below says what it did not
change.

- **2026-10-01, the versionless upstream quotes.** Five claims about Heretic carried no version:
  the `n_trials = 200` default and its `config.default.toml` filename, the two fixed scorers, where
  the four prompt sets live in the config, which trial the unaided menu offers first, and the
  "number its author considers adequate" justification for matching at 200. Every one is true at
  tag `v1.4.0`, which is what `Dockerfile.heretic` pins and what every published arm ran against,
  and **three of the five read differently on upstream's unreleased `master`**, where `n_trials`
  defaults to 100, the scorers became configurable plugins, and the prompt sets moved onto the
  modifier and the scorer. Each claim now carries its version and its read date.

  **Nothing in the budget definition, the equalisation rule or any published number changes.** The
  defect was missing provenance on a number, not the number. It matters here more than it would
  anywhere else, because this is the document a sceptic is pointed at to check that the rules were
  fixed before the run, and an unversioned quote about a moving upstream is the one kind of
  sentence that can go false while nobody edits it. That is the same failure this file already
  carries a 2026-09-25 append about, arriving by a different route: last time a sentence went stale
  because our harness changed, this time because somebody else's did.

  **The rule this sets, so it does not have to be rediscovered:** every quoted upstream default,
  filename or behaviour in this file names the ref it was read at and the date it was read. A
  figure without a ref is a figure that cannot be checked and cannot be noticed going wrong.

- **2026-10-01, which prompt budget each published figure actually used, and the shortfall against
  400.** The table above says 256 direction-fit prompts per side, and that is what every published
  arm in this directory ran at. It needs stating next to the other number a reader will meet,
  because Heretic's shipped default is **400 per side** and this track cannot serve it.

  Measured on the bundled track on 2026-09-11 and re-read from `track.json` on 2026-10-01, the
  `fit + search` partitions hold **385 rows on the harmless side and 391 on the harmful side**. So
  a run asking for 400 a side is refused rather than quietly reading into the measurement rows,
  which is `flag_violations` doing its job. The shortfall against the 400 the pre-registration
  asked for is **15 rows, or 3.75%, on the harmless side** and 9 rows, or 2.25%, on the harmful
  side. The harmless figure is the binding one.

  The decision (D32) is to accept 385 and 391 as the instrument, being the most this track serves,
  and to disclose the 3.75% rather than report the smaller number without saying it is smaller.
  **So there are two budgets in play and they belong to different runs:** every figure published in
  this directory used 256 a side, and any future run on this track uses 385 and 391. Neither is
  400, and no figure here should be read as a 400-prompt result.

  **Nothing in any published number changes.** This correction names the budget behind figures that
  already existed and records the ceiling that binds the next run. It is written here rather than
  only in the decision log because the budget is a property of the instrument, and this is the file
  a sceptic reads to find out what the instrument was.
