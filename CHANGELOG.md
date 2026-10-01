# Changelog

All notable changes to Senbonzakura are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Other
- Bug fixes and improvements.

## [0.4.1] "TYBW" — 2026-10-01

A patch release. Nothing here changes what the tool measures, so 0.4.0 figures stay comparable.

### Guided mode
- Typing `senbonzakura` on its own now prints a short page naming the ways in, rather than a 27 line flag list and an error.
- `senbonzakura -i` is a short spelling of `senbonzakura interactive`.
- The guided mode offers the models already on this machine, so a first run need not be a download.
- Every pre-flight check runs before you confirm rather than after, so a run that was going to be refused is refused first.
- When a run finishes it says what changed, what it cost and what to do next, each as a command you can paste.
- The capability probe is sized for the machine it will run on, so the command printed is one the pre-flight accepts.
- The front door, the guided walk and `senbonzakura doctor` fold to the terminal width.

### Install
- `brew install` and `scoop install` work. Both manifests shipped with a placeholder URL and no checksum at 0.4.0.
- Card temperature and power on the live dashboard work on a plain `pip install senbonzakura`.

### CLI
- `senbonzakura check` now exits 2 for a directory holding artefacts it could not read, where it exited 0. A CI step pointed at corrupt files goes from green to red on this release.
- `senbonzakura compass` reports its AUC, the interval with its confidence level named, and a plain verdict.
- `convert --verbose` and `quantise --verbose` show every line the vendored converter and quantiser print. A warning, an error or a failure is never summarised away.
- The JSON report carries `applied`, so a reader can tell a file that was examined from one that was only opened.
- No command abbreviates its flags any more, so what is accepted is what the help page lists.
- A mistyped flag is reported even when a required flag is missing too. The second used to hide the first.
- A model id that does not exist is reported as a model id that does not exist, rather than as an authentication failure.
- `senbonzakura gate` names the command that writes a baseline when it refuses for want of one.
- `senbonzakura track` with no arguments answers with a command you can run.
- `senbonzakura corpora --check` no longer needs the GitHub CLI when every source is already on the machine.
- `senbonzakura baseline` keeps the units that were sitting in the artefact beside the value.
- `senbonzakura doctor` names itself before the corpus attribution, and says how many imports it left out of a list.
- `--out` refuses a directory holding a tokeniser, which a save could previously overwrite in silence.
- `--inspect` marks where it cut a prompt or a generation to fit the column.
- `--print-completion` without the completion extra says which install it needs.
- A quantisation receipt records the hash of the files it names rather than of the quantiser alone.
- A result produced by an installed wheel can say which commit built it.

### Documentation
- `pip install senbonzakura` is the documented install everywhere. Several pages still offered `git+` URLs and told readers not to use PyPI.
- The evaluation track card said no install carries the bundled harmful prompts, and the published wheel carries them.
- The first command in the README and the quickstart exited 1, because it measured a model too small to produce the token the compass reads.
- The pages introducing `senbonzakura corpora` say it needs the GitHub CLI.
- The reference page no longer says the GitHub Action and the pre-commit hook cannot work yet.
- `senbonzakura measure` no longer points at a documentation path that no wheel carries.
- The guided mode no longer tells readers to run `huggingface-cli login`, which `hf auth login` replaced.
- The Colab notebook drew its refusal rates from the partition a run selects on and captioned them held out. It now reads the boundary from the track's own manifest.
- The notebook also promised that everything works without a GPU, a line per trial under a method that runs no trials, and an unmeasured time for the measuring cells.

### GitHub Action
- The action can be listed on the GitHub Marketplace. Its description was 185 characters against a limit of 125.

### Other
- Bug fixes and improvements.

## [0.4.0] "TYBW" — 2026-09-26

The release that makes the measurement trustworthy.

**The multi-direction feature had never worked, and was rewritten in this release.** The check
that decided whether a candidate direction carries refusal couldn't accept any direction, on any
model, at any setting, for a reason in the maths rather than in the data, so every earlier run
applied exactly one direction however many it was asked for. Directions are now found by grouping
the harmful prompts and taking each group's own average, which finds up to eight per layer where
the old method found one.

**Read the next sentence before quoting the one above.** Finding more directions isn't the same as
showing they carry refusal, and this release doesn't show that. Any direction count from a version
before this one isn't trustworthy, and the comparison that was meant to prove several directions
beat one is withdrawn.

**The check itself was then found to be measuring nothing, for a second reason.** It scored each
candidate direction on the very rows the direction was built from, so it was asking whether the
quantity a vector was built to maximise is large along that vector. It always is, which is why the
filter accepted every candidate on every run. Candidates are now fitted on half their rows and
scored on the half they never saw, and the threshold has a floor measured from directions that
carry nothing, so a candidate has to beat what the statistic hands out for free. Both numbers are
recorded in every result file. This still doesn't show a direction carries refusal rather than
topic, and the run says so in those words.

**And when we ran that comparison, more directions lost.** One direction against two, five seeds
each, with the same tool, the same corpus, the same search budget and the same direction budget
pinned at both ends, one parameter apart; every model scored afterwards by one instrument on 200
prompts nothing was fitted or selected on:

| Direction budget | Coherence drift | Hard refusal |
|---|---|---|
| One direction | 0.0497, spread 0.0177 | 0.1% |
| Two directions | 0.0932, spread 0.0482 | 0.3% |

Both budgets removed hard refusal, so this is a comparison at matched refusal, and the per-seed
drift and hard refusal values behind both columns are committed at
`evidence/k-sweep-2026-08-13/drift-per-seed.json`. Two directions cost about 1.9 times the
collateral damage on the mean over five seeds, bought nothing measurable for it, and were less
predictable run to run.

**Two figures in the paragraph above were corrected on 2026-09-28, after this release.** It gave
the cost as "roughly 1.4 to 1.9", which reads as an interval and isn't one: 1.9 is the ratio of the
means and 1.4 is the same ratio with the one outlying two-direction seed dropped, which is a
sensitivity check. The 95% interval is 1.2 to 3.1, four times wider. And "bought nothing" is only
true of what this design could see: both arms sit on the floor of the refusal measurement, so the
largest gain two directions could ever have shown here is 0.1 of a percentage point, a fifth of one
scorable event. The figures above are the corrected ones, and this note stays under the release it
corrects so that a reader of these notes meets the correction rather than having to find the guide
page that carries it. One direction beats two in 24 of the 25 pairwise seed comparisons, so the
direction of the result is sturdier than its size. An exact permutation test over all 252 splits of
the ten seeds puts the difference at p = 0.016. Dropping the worst two-direction seed halves the
gap and moves that to p = 0.048, so the finding doesn't rest on one run and is weaker without it.
This is one model, Qwen3-1.7B, and it doesn't settle the question for every architecture.

**Read that table with its limit attached.** These arms ran on 2026-08-13, before the held-out
direction selection described above existed, so the second direction in each two-direction arm was
chosen by the filter now known to accept everything. What the table prices is an *arbitrary* second
direction, which is a weaker claim than the thesis it gets read against: it isn't evidence that two
well-chosen directions cost more than one. The word "properly" stood here until 2026-09-25 and
invited exactly that reading. The table stands as the old filter's answer until a re-run under the
fixed selector replaces it, and the same caveat is recorded in the evidence file itself.

Where we can currently measure it, then, the project's central idea is unsupported, and we'd rather
publish that than wait for a friendlier model.

**If you have numbers from an earlier version, re-measure them.** Two scoring bugs were fixed in
this cycle and both moved published figures. Anything measured before 2026-07-30 isn't comparable
to what this version produces.

**And one measurement definition changed on 2026-09-08, so noncompliance figures moved again.** The
hedging detector counted statements of fact about legality or danger as hedging: a complete,
compliant technical answer that mentioned an activity is illegal was scored as a soft refusal. That
fires asymmetrically, because a more explanatory model collects more of them regardless of whether
it complied, and it carried full weight in the rule that decides which trial is saved as the
finished model, so the search was being steered toward models that don't caveat. Noncompliance
rates from before this date aren't comparable with rates after it.

### Breaking

Read this section before upgrading. Nothing here breaks an install; what breaks is the meaning of
results you already hold.

- **Every number produced by an earlier version is withdrawn, not adjusted.** Three separate faults
  in this cycle each changed what the tool measures: the direction filter accepted every candidate
  it was given, the hedging detector scored compliant answers as soft refusals, and the weight edit
  didn't reach the running state on one family of models. There is no conversion factor between an
  old figure and a new one. Re-measure.
- **`--max-directions` above 1 did nothing before this release, on any model, at any setting.**
  Every earlier run applied exactly one direction however many it was asked for, so a run whose
  records say it used eight used one. This release finds up to eight per layer, and the comparison
  that was meant to show several beat one is withdrawn; when it was finally run properly, more
  directions lost.
- **Gemma, Gemma 3 and Olmo 2 results from earlier versions describe the base model.** Those
  architectures rescale each layer's output before adding it to the model's running state, and the
  edit was applied upstream of that rescaling, so it was largely undone. Fixed in this release, but
  any earlier Gemma checkpoint or figure is about an unedited model.
- **The evaluation track is now required, and a run without one refuses instead of guessing.** An
  install carries a bundled track and `--track default` uses it, so most users see no change; a
  clone has none, and there the refusal names the two commands that build one.
- **The `datasets` package moved to an extra, and naming a dataset by its HuggingFace id needs it.**
  `--good-ds owner/name` now asks for `pip install 'senbonzakura[hub]'`, and shell completion moved
  the same way to `senbonzakura[completion]`; both were part of a plain install in 0.3.0. Local
  tracks, the bundled corpora and every file format the tool reads are unaffected.
- **0.3.0 couldn't run the compass at all**, and neither could 0.3.1, which was a licence patch with
  byte-identical code. Those wheels contain no `margin.py` and no `crashsafe.py`, and this release
  is the first one that has the command.

### Known defects

- **Every Gemma measurement is withdrawn.** The weight edit didn't reach the model's running state
  on that architecture: Gemma 2 and Gemma 3 pass each layer's output through a learned rescaling
  step before adding it back, and the tool edited the weights feeding that step rather than what
  comes out of it, so the rescaling partly undid the edit. Measured on gemma-2-2b-it, the weight
  edit and an equivalent direct intervention disagreed by 0.578 in refusal rate, against 0.016 on
  Qwen3, and no Gemma setting moved KL above 0.021. The cause is fixed in this release and Gemma
  numbers will be re-measured; until then they are unmeasured rather than wrong. Llama, Mistral,
  Phi and Qwen are unaffected.
- **Ablating a direction removes most of it, not all of it.** The edit preserves each weight row's
  original length, which is what keeps the model coherent, and that step doesn't commute with the
  removal, so roughly an eighth of a direction survives on every architecture. This is a deliberate
  trade-off inherited from the upstream method rather than a new fault, but it wasn't written down
  anywhere and "the direction was ablated" reads stronger than what happens.
- **Finding several directions isn't the same as showing they are refusal directions.** The check
  meant to tell a refusal direction from a topic direction does now reject: each candidate is
  fitted on half its rows and scored on the half it never saw, against a threshold with a measured
  floor beneath it. This entry said it "accepts every candidate it is shown" until 2026-09-25,
  which described the state before that fix and contradicted the body of this same release entry.
  **What is still open is the thing that matters**: a filter that can reject is not evidence that
  what it accepted carries refusal rather than topic, so nothing in this release establishes that
  the extra directions remove refusal rather than ability. A run reports the rejection rate on every
  extraction, and both "none rejected" and "all rejected" print a warning.
- **A direction count reported before this release cannot be trusted**, including the comparison
  table in the README. Those runs applied one direction whatever they requested, which doesn't mean
  refusal is a single direction; it means the tool hadn't measured it.
- **Whether removing several directions beats removing one is unanswered.** The comparison intended
  to settle it is withdrawn, because both of its arms turned out to be removing the same single
  direction.

### Measurement

- Every run now measures what the edit cost on a task the model either gets right or doesn't, and
  it does so by default. Refusal rates, drift and brokenness never ask the model to reason, so a
  model could hold a low divergence with nothing broken and have lost multi-step arithmetic.
- The package carries the probe it uses, so this works with no network: 256 grade-school arithmetic
  questions, attributed in THIRD-PARTY-NOTICES.md under "The bundled capability probe". Point
  `--capability-eval` at your own graded benchmark instead, or pass an empty string to turn it off.
- A run that pinned its settings and skipped the search used to measure no capability at all and
  report success, so result files now say whether capability was measured, separately from whether
  it was asked for.
- The compass, a harm-recognition score, is now a first-class command: it asks whether an
  abliterated model still recognises harm rather than only whether it complies.
- Every compass figure carries a confidence interval, and the compass measures on rows nothing was
  fitted or selected on; its harmful arm was previously scored on the prompts the winning
  configuration had been chosen from.
- The compass warns when the position its margin is read from holds something other than a verdict
  token, which on a thinking model can be the reasoning opener, and records what the position
  actually held on both arms.
- Results ship with the controls that qualify them: what a ruler reading only prompt length would
  score, the same figure over one fixed token pair, and a topic-matched arm.
- A read-out audit reports what the model actually put at the position being scored, which is how
  the largest bug in this release was found.
- `--seed` exists, and every result file records the value it ran at alongside the code version,
  the package versions and the commit.
- The per-layer direction counts in the run record are now per layer, where they were indexed by
  residual-stream position under a name that said layer, so each entry described the layer before
  the one it named and the search-window report was out by one.

### Engine

- Refusal directions are found by grouping the harmful prompts and taking each group's own average
  against the harmless average, rather than by looking at how the harmful prompts vary, because a
  model refuses a weapons request differently from a self-harm one. `--direction-clusters` sets how
  many groups to look for.
- A run that asks for several directions keeps the same first direction it would have kept at a
  budget of one, so two runs that differ only in the budget differ only in the budget.
- A contrast set too small to form two groups says so and explains how many prompts it needs.
- Hybrid architectures are supported: where a layer holds a short convolution instead of attention,
  that convolution writes the model's running state just as attention does, and it's now edited too.
- Every saved model carries a record of what produced it in both its configuration file and its
  weight files, naming the version, the settings, and whether the model is a complete abliteration
  or a deliberately partial one.
- Sparse surgery restricts the ablation to the rows that write refusal.
- The search can warm-start from a difference-of-means seed, and can be given a fixed prompt format
  instead of inventing one when a model has no chat template.
- A disk-space check runs before the search rather than during the save, and every result file is
  written atomically, so an interrupted run leaves the previous result intact.

### CLI

- **`senbonzakura measure <model>` answers "did I break my model?" in one command**, where it took
  five. Those five still exist and the numbers are identical, because each row is that command run
  with the arguments you would have typed.
- **`senbonzakura Qwen/Qwen3-1.7B` is now the whole command**, because the output directory and the
  evaluation track are defaulted and the model is the only thing the tool can't guess.
- `senbonzakura quantise ./abliterated` now works on a checkpoint rather than only on a GGUF, and
  converts first with the same pinned converter `convert` uses.
- `--help` shows the flags a run needs and `--help-all` shows all 69, where the default command's
  help page was 470 lines; both forms accept exactly the same arguments.
- Two build steps became commands, `senbonzakura corpora` and `senbonzakura track build`, where
  they were scripts under `tools/` that ship in no wheel.
- Four commands stopped demanding arguments they could work out: `capability` takes the model
  without a flag, `track` defaults its output directory to the one the abliterator looks for, and
  `baseline` takes the artefact as its first argument and derives the rest. `baseline --seeds` is
  still required, because a baseline claiming a spread it doesn't have is worse than none.
- `harm-recognition` is a second name for `compass`, the way `auto` already is for `kageyoshi`, and
  both original names still work and are what every artefact records.
- `bench` is now `head-to-head`, with its operation at `head-to-head run`, `stage` and `report`.
- Real subcommands exist for `abliterate`, `kageyoshi`, `compass`, `score`, `coherence` and
  `track`, and the existing flag form is unchanged.
- Left out, `--track` takes `./track` when that directory exists and the bundled track otherwise,
  and says in the log which it chose.
- The capability probe refuses to start a run that would take hours on a CPU and names four ways
  forward; measured on a 1.7B model, the defaults are minutes on a GPU and about four hours on a
  processor.
- The probe reports progress every 30 seconds, where it printed nothing at all before.
- `drift` and the head-to-head were both dispatched without being listed in the help.
- The guided mode asks the corpus question each command actually takes, because abliterating needs
  a track with three partitions and scoring needs one prompt set.
- Picking two Hub corpora in the guided mode now builds a track from them and shows that as its own
  command, run before the abliteration that needs it.
- A paused run records the model and the corpus it was working on, so the guided mode's offer to
  carry on produces a command that runs.
- `--resume` refuses a directory whose completed trials used a different model or track, and names
  both ways forward.
- A startup banner on a terminal, printing nothing when output is redirected so captured logs are
  byte-identical to before.

### Data

- `senbonzakura.track` builds an evaluation split and refuses to write one that leaks: nothing in
  the measured partition may appear in the fitted or searched ones, every category in the corpus
  reaches the measured partition, and `track.json` records where the boundaries fell so a run whose
  flags would cross one is refused.
- The leak check compares requests rather than strings, because the same question wears many
  templates and comparing whole prompts reported a clean split while 60% of the measured set was
  training questions in other clothes.
- `--contamination` reports how much of a public benchmark a track has already been fitted or
  searched on, which is what decides whether a figure on that benchmark would be in-sample; it
  counts rows and never prints one, and `--fail-on-contamination` gates on it.
- The evaluation track's card records that AdvBench is inside the track, that how much of it
  reached the fitting side is unmeasured, and how to check before quoting such a figure.
- `tools/packaging/build_track.py` fetches the public corpora this project measures on at pinned
  revisions, and refuses to run if an upstream's declared licence has moved since the recipe was
  written. No prompt rows ship in this repository: two of the three upstream datasets declare no
  licence at all and the probable root of the harmless side is non-commercial, so the recipe is
  what can honestly be published.
- The dataset card traces every source to what it actually declares, checked against the Hub rather
  than remembered, records which links are inferred, and records that the exact corpus behind the
  published numbers can't be rebuilt by anyone, because harmless top-ups were added by hand and
  never recorded.
- `senbonzakura track build` balances the two sides, because the sources return roughly four times
  as many harmless prompts as harmful ones and `senbonzakura track` refuses a pair more than 10%
  apart; `sources.json` records the seed and how many rows were dropped, and `--no-balance` gives
  you the raw pools.
- A format for contributing a behavioural probe, with a manifest, a declared content class and
  gates that refuse a corpus posing as a benchmark. The gates check shape only; a person still
  reads what a probe contains, and `probes/README.md` says so to contributor and user alike.
- Per-prompt margins and generations are kept, so the next question doesn't need the GPU back, and
  a guard refuses to commit any file containing prompts.

### Benchmark

- `senbonzakura head-to-head` runs a matched comparison between abliteration tools on one machine:
  `run` puts every arm through with one scoring instrument, `stage` cuts the prompt slices, and
  `report` reads a finished run.
- Both tools get the same corpus, budget and prompt slices, and the slices record which corpus they
  came from so a mismatched pair is refused rather than run.
- `--isolate docker` runs each arm with no network, read-only inputs and no credentials; without it
  the run warns, because a third-party tool otherwise runs with yours.
- Every comparable axis (harm recognition, coherence drift, refusals removed) gets a permutation
  test rather than two means side by side, with the median beside the mean. Where the number of
  seeds makes significance impossible for any arrangement of the data, the report says the
  comparison couldn't have concluded rather than calling it a tie.
- `senbonzakura drift` measures coherence the way the compass measures harm recognition, on
  held-out prompts over every model whoever made it, and this release found two tools' self-reported
  divergences can disagree by a hundredfold and reverse order once put on one ruler.
- `--max-kl` searches for the fewest refusals under a coherence ceiling, and refuses rather than
  falling back when no configuration meets it.
- An arm is skipped only when a manifest agrees with the run's tool, seed, model and budget and
  every artefact it declared is present, so an arm that exits cleanly having produced nothing is
  retried.
- Adding another tool is an adapter: how to invoke it, what proves it ran, where it leaves a model,
  how to read its own figures.

### Install

- `pip install senbonzakura` installs torch, transformers, accelerate and optuna, as 0.3.0 also
  did, and is 69 packages and 5.9 GB. If you tracked `dev` in early September you were told to add
  an `abliterate` extra; it still resolves and now installs exactly what a plain install does.
- `senbonzakura setup` puts the right build of torch on the machine it is run on, because pip picks
  by platform rather than by hardware, and changes nothing unless you pass `--apply`.
- The wheel carries the bundled evaluation track, so `--track default` works with no network; that
  is roughly 6,200 harmful prompts inside your site-packages, and the install page says what is in
  it and how to build a wheel without it. 0.3.0 shipped no corpora at all.

### New since 0.3.0

0.3.0 installed three things you could run: abliterating a model (as `senbonzakura --model ...` or
`senbonzakura kageyoshi`, the only mode word it knew), plus `python -m senbonzakura.score` and
`python -m senbonzakura.coherence`. Every command below is one that release didn't contain at all.

- `senbonzakura compass`: does the model still recognise harm, as opposed to still refusing?
- `senbonzakura capability`: what did the edit cost, on five graded tasks?
- `senbonzakura validate`: the checks behind a published number, including the experiments.
- `senbonzakura judge`: validates a judge before it grades anything, and refuses a bad one.
- `senbonzakura track`: build an evaluation split, and check it for contamination.
- `senbonzakura head-to-head`: run this tool and another one on the same model, one ruler.
- `senbonzakura convert`, `quantise`, `imatrix`: GGUF conversion and quantisation.
- `senbonzakura doctor`: can this install actually do what it claims, before a long run?
- `senbonzakura fetch`: fetch a model, resumable.
- `senbonzakura report`: the model card for a finished run.
- `senbonzakura interactive`: a guided mode for people who don't want to read the flag list.
- `senbonzakura drift`: how far the edited model moved from the original.
- `senbonzakura setup`: put the right build of torch on this machine, which pip can't do itself.

### Packaging and CI

- Continuous integration, which this repository had never had, on Linux, Windows and macOS across
  five platform and interpreter combinations covering three Python versions. This said "five Python
  versions" until 2026-09-25; the matrix has five rows and runs 3.10, 3.12 and 3.14, and 3.11 and
  3.13 were removed from the package classifiers in this same release because claiming an
  interpreter nothing tests is a promise kept by luck.
- A plain `pip install senbonzakura` still installs the deep-learning stack, as 0.3.0 did; during
  development this release briefly split it out behind an `abliterate` extra and that was reversed
  before release, with the extra kept so anything written against it keeps working.
- The torch-free capability itself wasn't withdrawn, only the default, and it's still measured on
  every commit: the checking commands run with torch, transformers, accelerate and optuna all made
  unimportable, so a verifier or a distribution package can still be built without them.
- The `datasets` package is no longer needed for ordinary work, and tracks, the bundled corpora and
  every file format the tool accepts are read and written with pyarrow instead. Nothing about the
  files on disk changed: a track built by an older version reads the same, and one built by this
  version still loads in `datasets`.
- Container images, so the tool can be run without installing anything: a CPU image that converts,
  quantises and builds importance matrices, and a CUDA image that abliterates on a GPU, both
  carrying the pinned llama.cpp binaries a wheel can't.
- Per-platform wheels that carry those binaries, so `pip install` can convert and quantise, tagged
  for a platform only when they actually hold that platform's binaries.
- Converting and quantising from a wheel needs the OpenMP runtime, which a wheel has no way to ask
  a system for, so `doctor` names the missing library and what to install.
- The command line starts in a hundredth of a second instead of just under three, because printing
  help no longer loads a deep-learning stack, and `doctor` can now run on a machine missing the
  libraries it exists to report on.
- Every quantisation writes a record beside its output naming the pinned toolchain version, the
  build the binary reports about itself, and the importance matrix used, if any.
- Declared dependency floors are tested at the floor, and a check enforces that the pinned floors
  and the declared ones are the same versions.
- Importing the package no longer imports the whole command-line interface, so running any module
  with `-m` no longer warns about unpredictable behaviour.

### Documentation

- The install commands now name a branch, where without one pip and git took the repository's
  default branch, which held no checker package at all.
- A Colab notebook that measures a model, edits it, and measures what that cost, on a free GPU with
  nothing installed locally.
- The headline comparison table now carries its caveats beside it: which two scoring bugs affected
  it, that its evaluation isn't held out, that it is one seed, and why it hasn't been re-measured.
- The quickstart now carries the same warning the other install pages do, that the published
  version is withdrawn.
- The README documents the track layout, the manifest, and the optional datasets.
- A contributor licence agreement, a contributing guide, and a contributors file.

### Licence

- The AGPL section 5(a) statement of modification, with dates, in the notices file, the README, and
  beside the copied code. A test pins the copied region so it can't drift from upstream without
  someone deciding to let it.

### Security

- The pre-commit hooks no longer execute the project configuration file.
- The prompt-retention guard reads what is staged rather than what is in the working tree, so an
  artefact staged with prompts and then tidied on disk is still refused.

### Fixed

- **`senbonzakura measure` scored the prompts a configuration had been chosen on, and captioned the
  figure as held out.** The command carried a second copy of the code that reads a track's
  boundaries, and that copy didn't know the bundled track's alias, so on the default track it read
  the boundary as zero where the real one is 132. There is one reader now, and the caption comes
  from what the run recorded, so a figure whose boundary nothing confirmed says exactly that.
- A run could wait for ever for video memory, because the pause that fires under memory pressure
  waited for a level a card filled by our own model can never reach and the wait before the first
  batch had no deadline; a pause now never waits for our own model to stop being resident.
- The first run the documentation describes failed at its second command, because `senbonzakura
  track build` needed a package only the `hub` extra installs; it reads the Hub directly now.
- The refusal scanner read only the first 240 characters of a reply, and 51 of 56 refusal markers
  land past that point.
- The compass scored the prompt read back to the model rather than the model's verdict.
- A layer offloaded to disk was never abliterated, and nothing said so.
- A non-converging matrix decomposition quietly weakened the ablation and reported the strength it
  had been asked for, not the one it applied.
- The prompt renderer existed in three copies that had drifted, so a configuration selected under
  one prompt format was reported under another.
- `senbonzakura score` couldn't say which rows its number came from, so the headline refusal rate
  could never be compared against a baseline; it records the partition it measured and a confidence
  interval for both estimators.
- A run that requests several refusal directions and applies one now says so in those words, where
  the note that reported it was scoped to part of the model and read as though the rest had been
  fine, which is what hid the defect above for most of a day.
- Every candidate direction's score is kept in the result file whether it was used or not, because
  a count alone can't say whether a direction was rejected narrowly or was never possible.
- Nine orthonormal directions can't exist in an eight-dimensional space, and the code believed they
  could.
- Nonsense slice arguments produced a plausible-looking result file instead of an error.
- A mistyped or retired command printed a usage block, and now names the closest real command or
  what replaced it.
- A problem in a probe file reported a line number from inside the tool instead of the line in the
  file the user wrote, and an empty prompt file was read as a run with nothing to refuse rather
  than as a mistake.
- The guided mode's corpus menu offered four corpora and three of them printed a command that
  couldn't run, and its offer to carry on with an existing run printed a command with no model on
  it.
- AdvBench is gated on the Hub and was the first harmful corpus offered, so pressing Enter gave
  anyone without a Hugging Face account an authentication error; it is still offered, second, and
  says what it needs.
- The guided mode read a helper out of a module that imports the deep-learning stack, so on an
  install without it the walkthrough asked four questions and then produced a traceback.
- A command that reported failure by returning a status rather than by raising ended the guided run
  silently; both now print what is on disk and how to carry on.

### Other

- Bug fixes and improvements.

## [0.3.1] "Petals" — 2026-09-09

A licence-compliance patch on 0.3.0, with no code change. Every file in the package is
byte-identical to the one 0.3.0 shipped.

### Licence

- The package now carries the AGPL section 5(a) statement of modification inside the artefact
  itself. Senbonzakura includes code copied verbatim from Heretic, and 0.3.0 was published
  carrying the licence text alone, so the statement saying this is a modified work and when it
  was modified reached nobody who installed it.

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

[0.4.0]: https://github.com/elementmerc/senbonzakura/releases/tag/v0.4.0
[0.3.1]: https://github.com/elementmerc/senbonzakura/releases/tag/v0.3.1
[0.3.0]: https://github.com/elementmerc/senbonzakura/releases/tag/v0.3.0
