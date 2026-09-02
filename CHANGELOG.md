# Changelog

All notable changes to Senbonzakura are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — TYBW

The release that makes the measurement trustworthy. The date is added at tag time.

**The multi-direction feature had never worked, and was rewritten in this release.** The check
that decided whether a candidate direction carries refusal could not accept any direction, on
any model, at any setting, for a reason in the maths rather than in the data, so every earlier
run applied exactly one direction however many it was asked for. Directions are now found by
grouping the harmful prompts and taking each group's own average, which finds up to eight per
layer where the old method found one.

**Read the next sentence before quoting the one above.** Finding more directions is not the
same as showing they carry refusal, and this release does not show that. Any direction count
from a version before this one is not trustworthy, and the comparison that was meant to prove
several directions beat one is withdrawn.

**The check itself was then found to be measuring nothing, for a second reason.** It scored each
candidate direction on the very rows the direction was built from, so it was asking whether the
quantity a vector was built to maximise is large along that vector. It always is, which is why
the filter accepted every candidate on every run. Candidates are now fitted on half their rows
and scored on the half they never saw, and the threshold has a floor measured from directions
that carry nothing, so a candidate has to beat what the statistic hands out for free. Both
numbers are recorded in every result file. This still does not show a direction carries refusal
rather than topic, and the run says so in those words.

**And when we finally ran that comparison properly, more directions lost.** One direction
against two, five seeds each, everything else held still, every model scored afterwards by one
instrument on 200 prompts nothing was fitted or selected on:

| Direction budget | Coherence drift | Hard refusal |
|---|---|---|
| One direction | 0.0497, spread 0.0177 | 0.1% |
| Two directions | 0.0932, spread 0.0482 | 0.3% |

Both budgets removed hard refusal, so this is a comparison at matched refusal. Two directions
did roughly twice the collateral damage and bought nothing for it, and they were less
predictable run to run. The gap clears the pooled spread by about a fifth, which is real but
slim; dropping the worst two-direction seed makes it wider rather than narrower, so it doesn't
rest on one bad run. This is one model, Qwen3-1.7B, and it does not settle the question for
every architecture. It does mean the project's central idea is unsupported where we can
currently measure it, and we would rather publish that than wait for a friendlier model.

**If you have numbers from an earlier version, re-measure them.** Two scoring bugs were
fixed in this cycle and both moved published figures. Anything measured before 2026-07-30 is
not comparable to what this version produces.

### Measurement

- The compass, a harm-recognition score, is now a first-class command: it asks whether an
  abliterated model still recognises harm rather than only whether it complies.
- Every compass figure now carries a confidence interval. An AUC without one invites belief
  the data cannot support.
- The compass measures on rows nothing was fitted or selected on. Its harmful arm was
  previously scored on the prompts the winning configuration had been chosen from.
- Results ship with the controls that qualify them: what a ruler reading only prompt length
  would score, the same figure over one fixed token pair, and a topic-matched arm.
- A read-out audit reports what the model actually put at the position being scored, which is
  how the largest bug in this release was found.
- `--seed` exists, and every result file records the value it ran at. One run cannot tell a
  finding from a coin toss.
- Every result records what produced it: the code version, the package versions, and the
  commit.

- The compass says when its own read-out is not a verdict. It records what token actually sits
  at the position the margin is read from and how much probability the two verdict spellings
  hold, on both arms, and prints a warning beside the figure when that position holds something
  else. On a thinking model it can hold the reasoning opener, and the margin is then a comparison
  between two tokens the model was never going to produce.
- The per-layer direction counts in the run record are per layer. They were indexed by
  residual-stream position under a name that said layer, so each entry described the layer
  before the one it named, the minimum was always zero, and the search-window report was out by
  one. Both views now ship, with a note saying which is which.

### Data

- `senbonzakura.track` builds an evaluation split and refuses to write one that leaks.
  Nothing that appears in the measured partition may appear in the fitted or searched ones.
- The leak check compares requests, not strings. The same question wears many templates, and
  comparing whole prompts reported a clean split while 60% of the measured set was training
  questions in other clothes.
- Every category present in the corpus reaches the measured partition, so a published figure
  cannot come from a narrower slice than it claims.
- `track.json` records where the partition boundaries fell, and a run whose flags would cross
  one is refused rather than quietly allowed.
- Per-prompt margins and generations are kept, so the next question does not need the GPU
  back, and a guard refuses to commit any file containing prompts.
- `--contamination` reports how much of a public benchmark a track has already been fitted or
  searched on, which is what decides whether a figure on that benchmark would be in-sample.
  It compares requests rather than strings, counts rows and never prints one, and names how
  many rows could still be reported cleanly. Add `--fail-on-contamination` to gate on it.
- The evaluation track's card now records that AdvBench is inside the track, that how much of
  it reached the fitting side is unmeasured, and how to check before quoting such a figure.
- `tools/build_track.py` fetches the public corpora this project measures on, at pinned
  revisions, and writes the two prompt files the track builder splits. No prompt rows ship in
  this repository: two of the three upstream datasets declare no licence at all and the
  probable root of the harmless side is non-commercial, so the recipe is what can honestly be
  published. It refuses to run if an upstream's declared licence has moved since the recipe was
  written.
- The dataset card now traces every source to what it actually declares, checked against the
  Hub rather than remembered, and records which links are inferred. It also records that the
  exact corpus behind the published numbers cannot be rebuilt by anyone: harmless top-ups were
  added by hand and never recorded.

### Benchmark

- `senbonzakura head-to-head` runs a matched comparison between abliteration tools on one machine:
  `run` puts every arm through, scoring every model with one instrument, `stage` cuts the prompt
  slices every tool is scored on, and `report` reads a finished run.
- Both tools get the same corpus, the same budget and the same prompt slices, and the slices
  record which corpus they came from so a mismatched pair is refused rather than run.
- `--isolate docker` runs each arm with no network, read-only inputs and no credentials.
  Without it the run warns, because a third-party tool otherwise runs with yours.
- An arm is skipped only when a manifest agrees with the run's tool, seed, model and budget and
  every artefact it declared is present. An arm that exits cleanly having produced nothing is a
  failure and leaves no manifest, so the next run retries it.
- Adding another tool is an adapter: how to invoke it, what proves it ran, where it leaves a
  model, how to read its own figures.
- Fewer than three seeds gets no verdict, and a gap smaller than the spread is reported as a tie.
  A gap equal to the spread is a tie too, as is one that clears it by less than the report prints,
  because a margin the reader can't see in the figures given is not a margin they can check.
- `senbonzakura drift` measures coherence the way the compass measures harm recognition: one
  instrument, run by us afterwards, on held-out prompts, over every model whoever made it. Two
  tools' self-reported divergences are two measurements wearing one name, and this release found
  they can disagree by a hundredfold and reverse order once put on one ruler.
- `--max-kl` searches for the fewest refusals under a coherence ceiling, which is the question
  other tools ask. It refuses rather than falling back when no configuration meets the ceiling.

### Engine

- The search can be given a fixed prompt format instead of inventing one when a model has no
  chat template.
- Hybrid architectures are supported. On models where some layers hold a short convolution
  instead of attention, that convolution writes the model's running state just as attention
  does, and it is now edited too. Editing only part of a layer and reporting success is how
  every Gemma figure in this project came to be withdrawn.
- Every saved model carries a record of what produced it, in both its configuration file and
  the weight files themselves, so copying a single file out of the folder does not lose it. It
  names the version, the settings, and above all whether the model is a complete abliteration
  or a deliberately partial one.
- Sparse surgery restricts the ablation to the rows that write refusal.
- The search can warm-start from a difference-of-means seed.
- A disk-space check runs before the search rather than during the save, so a long run cannot
  die at the last step.
- Every result file is written atomically, so an interrupted run leaves the previous result
  intact rather than a truncated one.

### CLI

- Real subcommands: `abliterate`, `kageyoshi`, `compass`, `score`, `coherence` and `track`.
  The existing flag form is unchanged, because every run on record is written that way.
- A startup banner on a terminal, from a rotating set of five designs. It prints nothing when
  output is redirected, so captured logs are byte-identical to before.
- One model-loading surface instead of four near-copies. The four had drifted apart, which is
  what put the compass's read-out on the wrong token.

- `bench` is now `head-to-head`, and its operation moved with it: `head-to-head run`, `stage`
  and `report`. The old name was the word a newcomer reaches for when asking whether a model is
  any good on their machine, which is a different question from comparing two abliteration tools.
- `drift` and the head-to-head were both dispatched without being listed in the help.

### Documentation

- The README documents the track layout, the manifest, and the optional datasets.
- The headline comparison table now carries its caveats beside it: which two scoring bugs
  affected it, that its evaluation is not held out, that it is one seed, and why it has not
  been re-measured.
- A contributor licence agreement, a contributing guide, and a contributors file.

### Packaging and CI

- Continuous integration, which this repository had never had, on Linux, Windows and macOS
  across five Python versions.
- Declared dependency floors are tested at the floor, because a floor nothing installs at is
  not a tested floor.
- Importing the package no longer imports the whole command-line interface, so running any
  module with `-m` no longer prints a warning about unpredictable behaviour.
- Container images, so the tool can be run without installing anything. The CPU image converts,
  quantises and builds importance matrices; a separate CUDA image abliterates on a GPU. Both
  carry the pinned llama.cpp binaries, which a wheel cannot.
- Per-platform wheels that carry those binaries, so `pip install` can convert and quantise. A
  wheel is tagged for a platform only when it actually holds one platform's binaries, so a
  package claiming to run anywhere can never contain a Linux executable.
- Converting and quantising from a wheel needs the OpenMP runtime, which a wheel has no way to
  ask a system for. `doctor` names the missing library and what to install; the container images
  carry it already.
- The command line starts in a hundredth of a second instead of just under three, because
  printing help no longer loads a deep-learning stack. `doctor` can now run on a machine that is
  missing the very libraries it exists to report on.
- Every quantisation writes a record beside its output naming the toolchain that produced it:
  the pinned version, the build the binary reports about itself, and the importance matrix used,
  if any.

### Licence

- The AGPL section 5(a) statement of modification, with dates, in the notices file, the
  README, and beside the copied code. A test pins the copied region so it cannot drift from
  upstream without someone deciding to let it.

### Security

- The pre-commit hooks no longer execute the project configuration file.
- The prompt-retention guard reads what is staged rather than what is in the working tree, so
  an artefact staged with prompts and then tidied on disk is still refused.

### Known defects

- **Every Gemma measurement is withdrawn.** The weight edit did not reach the model's running
  state on that architecture. Gemma 2 and Gemma 3 pass each layer's output through a learned
  rescaling step before adding it back, and the tool edited the weights feeding that step rather
  than what comes out of it, so the rescaling partly undid the edit. Measured on gemma-2-2b-it:
  the weight edit and an equivalent direct intervention disagreed by 0.578 in refusal rate,
  against 0.016 on Qwen3, and no Gemma setting moved KL above 0.021. The cause is fixed in this
  release and Gemma numbers will be re-measured; until then they are unmeasured rather than
  wrong. Llama, Mistral, Phi and Qwen are unaffected.
- **Ablating a direction removes most of it, not all of it.** The edit preserves each weight
  row's original length, which is what keeps the model coherent, and that step does not commute
  with the removal, so roughly an eighth of a direction survives on every architecture. This is
  a deliberate trade-off inherited from the upstream method rather than a new fault, but it was
  not written down anywhere and "the direction was ablated" reads stronger than what happens.
- **Finding several directions is not the same as showing they are refusal directions.** The
  check meant to tell a refusal direction from a topic direction now accepts every candidate it
  is shown, where before it rejected every one. It has changed which way it fails rather than
  started working, so nothing in this release establishes that the extra directions remove
  refusal rather than ability. A run reports the rejection rate on every extraction, and both
  "none rejected" and "all rejected" print a warning.
- **A direction count reported before this release cannot be trusted**, including the
  comparison table in the README. Those runs applied one direction whatever they requested.
  This does not mean refusal is a single direction; it means the tool had not measured it.
- **Whether removing several directions beats removing one is unanswered.** The comparison
  intended to settle it is withdrawn, because both of its arms turned out to be removing the
  same single direction.

### Engine

- Refusal directions are found by grouping the harmful prompts and taking each group's own
  average against the harmless average, rather than by looking at how the harmful prompts vary.
  A model refuses a weapons request differently from a self-harm one, and the old method could
  not represent that. `--direction-clusters` sets how many groups to look for.
- A run that asks for several directions and gets them keeps the same first direction it would
  have kept at a budget of one, so two runs that differ only in the budget differ only in the
  budget.
- A contrast set too small to form two groups says so and explains how many prompts it needs,
  rather than quietly returning a single direction.

### Fixed

- A run that requests several refusal directions and applies one now says so in those words.
  The note that reported this was scoped to part of the model and read as though the rest had
  been fine, which is what hid the defect above for most of a day.
- Every candidate direction's score is kept in the result file, whether it was used or not. A
  count on its own cannot say whether a direction was rejected narrowly or was never possible,
  and those are different findings.
- The refusal scanner read only the first 240 characters of a reply, and 51 of 56 refusal
  markers land past that point.
- The prompt renderer existed in three copies that had drifted, so a configuration selected
  under one prompt format was reported under another.
- The compass scored the prompt read back to the model rather than the model's verdict.
- A non-converging matrix decomposition quietly weakened the ablation and reported the
  strength it had been asked for, not the one it applied.
- A layer offloaded to disk was never abliterated, and nothing said so.
- Nine orthonormal directions cannot exist in an eight-dimensional space, and the code
  believed they could.
- Nonsense slice arguments produced a plausible-looking result file instead of an error.

### Other

Bug fixes and improvements.

## [0.3.0] "Pilot" — 2026-07-17

The first public release: on PyPI, on GitHub, AGPL-3.0.

### Added
- `kageyoshi`, a one-shot balanced preset: supply only the paths and it scales
  the search budget to the model's size and turns on every quality lever. The KL
  ceiling and coherence penalty hold the result at the most uncensored config
  that stays coherent.
- A resource governor that paces GPU work under live VRAM pressure and offloads a
  model larger than VRAM, so the tool runs on constrained cards. Includes a
  background mode that yields the GPU to a foreground game and resumes on close.
- Shell completion via shtab: `--print-completion` emits a script for bash, zsh,
  or tcsh, the one-time install pattern pip and gh use.
- A packaged coherence probe (`senbonzakura.coherence`), the neutral-passage
  perplexity check, reusing the shared model loader so its flags match the scorer.

### Changed
- Relicensed to AGPL-3.0-or-later. The keyword metric copies Heretic verbatim and
  Heretic is AGPL, so copyleft reaches the whole work; Apache was never available
  to us. Upstream notice restored on the copied region, THIRD-PARTY-NOTICES added.
- README rebuilt around reproducible Qwen3-4B numbers, with an honest note on how
  the single-versus-multi gap widens with model size.

### Removed
- The superseded Qwen3-1.7B Heretic comparison table. Leading a first-time reader
  with a comparison we have publicly disowned is worse than none; replaced with a
  note that a matched re-run under the corrected code is pending.

### Fixed
- The coherence scorer rejected `--load-in-4bit`, so a bench chain that passed the
  flag logged a false coherence failure and stalled the 4B numbers. The packaged
  module accepts the scorer's full flag set and a parser test guards the regression.

## [0.2.0] — 2026-07-14

### Added
- Installable package: `src/` layout, a `senbonzakura` console command,
  `python -m senbonzakura`, `[quant]` and `[dev]` extras, a man page, `--version`.
- Multi-direction refusal search grounded in the head-to-head data: the keyword
  gap versus Heretic was a search and contrast problem, not weak ablation.
  - Three-objective NSGA-II over strict non-compliance, the keyword rate, and KL,
    so the keyword and hedging axes are optimised, not just reported.
  - `--hedge-ds` folds a hedged-versus-clean contrast direction into the basis.
  - `--mlp-off` pins the MLP ablation to zero to test where refusal lives.
  - `--patience` stops the search once the frontier stalls.
  - `--eval-refusal-final` re-scores the top frontier on a larger eval before the
    knee is chosen, so the pick is not overfit to the small search eval.

## [0.1.0] — earlier

### Added
- The initial abliteration pipeline: activation capture, difference-of-means
  refusal direction, weight orthogonalisation, and refusal/coherence scoring.

### Other
- Bug fixes and improvements.

[0.3.0]: https://github.com/elementmerc/senbonzakura/releases/tag/v0.3.0
