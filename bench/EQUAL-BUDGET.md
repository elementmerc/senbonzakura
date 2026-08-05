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

- **200 trials by default** (`config.default.toml: n_trials = 200`) against our 60. On trial count
  alone Heretic is given more than three times the search.
- **Two scorers on a Pareto front** (keyword rate, KL divergence), the same shape as ours.
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

Concretely:

- Heretic runs its search normally. Its Optuna study retains every trial.
- Its top six candidates are re-scored on the same held-out slice senbonzakura's six are re-scored
  on, with the same generation budget per candidate.
- The winner of that re-score, picked by the same weighted knee scalar, is Heretic's reported arm.

This is implemented **outside Heretic's code** (`bench/best_of_n_heretic.py`), reading its study,
so nothing about the tool is modified and the comparison is still of the tool as its author wrote
it plus a selection pass we added to match our own.

### How each tool's six are chosen

*Amended 2026-08-05, before any arm ran, when the pass was implemented.* The paragraph above
originally said Heretic's six would be picked "by the same scalarisation senbonzakura uses". That
is not implementable at any budget worth spending, and the correction matters enough to state
rather than quietly fix.

Our scalariser reads measurements. Applying it to Heretic's 200 trials would mean rebuilding and
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
written by `bench/stage_eval_slices.py` from the same code that builds them for our own arm.

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

**Trials are matched at 200 for both**, which raises ours from its default of 60. Matching upward
rather than down: capping Heretic at 60 would hand us a result that depends on starving the
comparison, and 200 is Heretic's own default so it is the number its author considers adequate.

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
| Container, torch, CUDA | same | same |

**The warm start is the one asymmetry left**, and it is disclosed rather than hidden. Everything
else is matched.

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
