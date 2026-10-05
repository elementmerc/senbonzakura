# Changelog

All notable changes to Senbonzakura are recorded here, newest first, grouped by the part of the
tool they touch. Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- A run given both `--hf-token` and `--base-licence` wrote the token into the model card (`README.md`) beside the weights, in the "Reproducing it" command. The card now shows the flag with its value replaced by `***`, and `senbonzakura report --command` does the same. If you published a card made by 0.4.0 or 0.4.1 from a command that used `--hf-token`, revoke that token and replace the card.

### New command: `jailbreak`
- Reports single-turn jailbreak success on one of the bundled attack sets, with a Wilson interval
  and the raw counts on every figure, including the one printed to the terminal
- Reports it beside the over-refusal rate on a benign set, in the same run, because a model that
  declines every request scores perfectly on jailbreak resistance and is useless. Measuring one
  arm without the other is the mistake the command is shaped to prevent, and an artefact missing
  one of them says so
- Reads the attack set by name rather than by path, and refuses a benign corpus in the attack
  slot: the safe half of XSTest used as an attack set reports near-total success on a model that
  refused nothing it should have, and nothing downstream can see it
- Gives two readings of the same thing: one minus the refusal rate, which is what the field
  usually quotes, and the share of replies that actually delivered an answer, which excludes
  evasions and the garbage a damaged model emits. A large gap between them is the finding
- Takes `--baseline`, a previous run against the unedited model, so the figure can be read as a
  statement about the edit. Baseline refusal on harmful prompts varies enormously between models,
  so a jailbreak rate on its own is a number about a starting point the reader does not know
- Refuses to call the run a measurement when more than a tenth of the replies are unreadable,
  because a damaged model reads as a win under either reading taken alone

### New command: `multi-turn`
- Asks whether a refusal holds when the conversation continues, and reports the turn at which it
  first gives way rather than only a rate
- Runs the plain-repetition control over the same refused prompts, so an escalation that does no
  better than asking the same question again is visible as such. Where the two intervals overlap,
  the run says it cannot separate them rather than reporting the difference
- Records the escalation by name and by digest, so a figure can be traced to the exact turns.
  Bring your own with `--strategy-file`, and its text is deliberately kept out of the artefact
- Refuses the run when a conversation outgrew the context window, because the model then answered
  a follow-up without having been shown the original request and the reply scores the same either
  way

### New command: `tamper`
- Measures whether an abliteration survives a brief safety-recovery finetune: refusal before, the
  finetune, refusal after, on the held-out half of the corpus so no evaluated row was trained on
- Runs the neutral-data control in the same command. If refusal recovers as much after training
  on benign task data, the result is about finetuning rather than about safety, and the artefact
  states no safety-specific figure without that arm
- Runs the same recipe on the unedited model as a ceiling, so a recovered fraction is read against
  what the recipe can achieve rather than against a perfect score
- Records the whole recipe: method, adapter shape, learning rate, schedule, optimiser, steps,
  batch size, gradient clip, dtype, seed, a digest of the training pairs and the loss at every
  step. A finetune whose loss did not fall is refused rather than reported, because it leaves
  refusal unchanged and that reads as a perfect result
- Trains either an adapter or every weight, and says which, because they answer different
  questions: one tests whether an adapter can route around the edit and the other whether the edit
  itself is undone
- States which way its own bias runs: the recovery data teaches the model the phrases the refusal
  ruler looks for, so a low recovered fraction is strong evidence and a high one is weak

### New command: `budget`
- Answers "will this model fit on this machine, and roughly how long would a run take" before
  anything is downloaded or loaded, from the checkpoint's headers alone
- Names each thing the run needs, separately: the widest layer, the float32 working set the
  rewrite holds, the key and value cache, the captured activations, the writable copies on disk
  and the per layer markers
- Reports the page-locked memory ceiling and which side of it the host side store falls on,
  because a store above the ceiling loses the overlap between copying and computing and the run
  then looks slow for a reason that is not the streaming
- Says whether the machine is on battery, since a laptop card on battery clocks to about a third
  and turns a five day estimate into a fortnight with nothing in any log to say why
- Prints the write total as a fraction of a typical drive's rated lifetime, so the wear is a
  decision rather than a surprise
- Refuses a run this machine cannot finish, naming the resource and the shortfall, and a resource
  this machine will not report is marked skipped rather than passed, in the verdict as well as in
  the line above it
- Writes the whole budget as JSON with `--out`, through a temporary file and a rename, so a
  truncated file can never read as a complete one

### GitHub Action
- Give the step a baseline and a measurement and it also runs the regression gate, so a property
  that moves outside its interval fails the build
- A pass, a regression and a refusal produce three different annotations, and a refusal fails the
  step with its own message because nothing was shown about the model at all
- Passing one of the two gate inputs without the other stops the run rather than quietly skipping
  the comparison
- Reports `gate-status` as an output, empty when no baseline was given, which is not the same as
  zero
- Inputs reach the step's shell through the environment rather than being pasted into the script,
  so a path containing a quote is a path and not a command

### Regression gate
- `--history` records every run, pass, fail or refusal, with the whole measurement rather than
  just its number, so a regression caught in March is still evidence in June
- Records are named after their own contents, so re-running a comparison records it once and
  nothing is ever overwritten
- A recorded measurement carries the digest of its own contents, so a baseline can be fetched by
  address with `--store` instead of measured again, and one that has been edited since is refused
- The verdict names the property, the partition, the sample size and the estimator it looked at,
  and says what it did not cover
- A history that cannot be written is reported and does not change the verdict

### Checker
- Findings are printed as each file is finished, so a sweep over thousands of artefacts that you
  stop halfway has still told you about the half it did
- An interrupted run has no summary and no exit status, and with `--json` the report does not
  parse, so a partial report is never read as a clean one
- A long sweep says where it has got to every thirty seconds, on standard error, so `--json` stays
  machine readable
- `--pair` checks that it was given two files before reading any of them

### Command line
- A negative or zero count is refused at the flag on every command rather than on some of them,
  so a mistyped minus can no longer measure the last five prompts and report them as the first
- `--inspect` refuses a fractional layer index instead of truncating it, and a negative strength
  instead of adding the refusal direction back
- The governor's free memory fraction, the pause ceiling and the divergence ceiling are checked
  before a run starts, because each of them moves a bar rather than crashing

### Measurement
- A null ruler that separates the two sides perfectly in the opposite direction is named as the
  strongest one, where it used to be scored as a coin flip, so the headline control line and the
  rule that invalidates a run now agree on the same case
- A null panel that could not be scored at all says so instead of printing a control at chance
- The null floor the direction filter holds candidates to is reported as the percentile it is,
  with its false keep rate, rather than as a bar every null draw fails
- `capability`, `drift` and the compass pass of `score` now record inside their own result file
  that a figure is not a measurement, which is the field `measure` and the checker already read,
  so a run that could not measure anything no longer reads like one that measured a zero
- `drift` refuses a divergence that came back as NaN or as infinity, one that came back
  materially below zero, and one averaged over a single prompt, naming the likely cause each time
- `score --harm-recognition` refuses a rate over fewer than thirty replies, which is the floor
  every other rate in the tool is already reported through
- `capability` exits non-zero when the run it is compared against could not grade its own
  answers, a case it has always printed as not quotable and then reported as a success
- A drift below the bfloat16 floor stays a caveat beside the number rather than becoming a
  refusal, because the measured gap to float32 is under one percent at the figures published here
- `measure` counts an instrument that disowned its own figure as one that produced no number, so
  the table, the failure list in `measure.json` and the exit status now agree about the same run

### Documentation
- The page explaining how the tool works now describes the direction filter as it behaves, with
  a held out score and a measured floor, matching the two pages that already did
- The capability probe is described as five grading rules over one bundled dataset, not as five
  datasets, on every page that counts them
- The GPU requirements table and the prose above it gave different answers for a 6 GB card, and
  the prose now quotes the measured row
- The flags page had its closing links in the middle of the body, above a third of the page
- Two pages disagreed about whether our own numbers carry intervals; both are corrected and the
  two real exceptions are named where a reader meets them
- `drift` is named as a **first-token** measurement in the README, the CLI reference and the flags
  page, not just in the benchmark and first-run guides
- The norm-restore sentence claimed a property of the field on a reading of two other tools;
  **withdrawn, not adjusted**, and restated scoped to the tools actually read
- "Nobody validates the judge" is now "nobody computes agreement above chance", which is the
  claim the evidence supports
- The comparison page says which tools it read, which it didn't, and on what date
- A fourth tool and a third idea, the conditional edit, are named in the prior-art page

### Other
- Bug fixes and improvements.

## [0.4.1] "TYBW" — 2026-10-01

A patch release. Nothing here changes what the tool measures, so 0.4.0 figures stay comparable.

### Guided mode
- `senbonzakura -i` drives the whole tool without reading the flag list
- `senbonzakura` on its own prints the ways in, not a flag list and an error
- Models already on the machine are offered, so a first run need not download one
- Pre-flight checks run before you confirm, not after
- A finished run prints what changed, what it cost, and what to do next
- Output folds to the terminal width

### Install
- `brew install` and `scoop install` work (0.4.0 shipped a placeholder URL and no checksum)
- Card temperature and power on the live dashboard work on a plain `pip install`

### CLI
- `senbonzakura check` exits 2 for a directory it could not read (previously 0)
- `senbonzakura compass` reports its AUC, a named interval, and a verdict
- `convert --verbose` and `quantise --verbose` show every line the vendored tools print
- Flags are no longer abbreviated, so what is accepted is what `--help` lists
- A model id that does not exist is reported as one, not as an authentication failure
- Each head-to-head arm records its own wall clock and its peak card memory

### Documentation
- The README bullet on published weights was false in both directions
- `pip install senbonzakura` is the documented install everywhere
- The Colab notebook measured on the selection partition and called it held out
- The first command in the README and the quickstart exited 1
- Four pages corrected: the track card, `corpora`, the Action, and the login command

### GitHub Action
- Listable on the GitHub Marketplace (its description was 185 characters against a cap of 125)

### Other
- Bug fixes and improvements.

## [0.4.0] "TYBW" — 2026-09-26

The release that makes the measurement trustworthy.

### Read this before quoting any number from this project
- Every figure from an earlier version is **withdrawn, not adjusted**. Re-measure
- Three faults each changed what the tool measures:
- the direction filter accepted every candidate it was given
- the hedging detector scored compliant answers as soft refusals
- the weight edit never reached the running state on Gemma
- There is no conversion factor between an old figure and a new one
- Anything measured before 2026-07-30 is not comparable with what this version produces
- Hedging stopped counting statements of fact about legality on 2026-09-08
- Noncompliance rates either side of that date are not comparable

### The multi-direction feature had never worked
- The check deciding whether a candidate carries refusal could not accept any direction
- On any model, at any setting, for a reason in the maths rather than the data
- So every earlier run applied exactly one direction however many it was asked for
- Directions now come from grouping the harmful prompts and averaging each group
- That finds up to eight per layer where the old method found one
- **Finding more directions is not showing they carry refusal, and this release does not show that**
- The comparison meant to prove several beat one is withdrawn
- The check also scored each candidate on the rows it was built from, so it always accepted
- Candidates are now fitted on half their rows and scored on the other half
- The threshold has a floor measured from directions that carry nothing

### And when the comparison was run, more directions lost

One direction against two, five seeds each, same tool, corpus, search budget and direction
budget, one parameter apart, scored on 200 prompts nothing was fitted or selected on.

| Direction budget | Coherence drift | Hard refusal |
|---|---|---|
| One direction | 0.0497, spread 0.0177 | 0.1% |
| Two directions | 0.0932, spread 0.0482 | 0.3% |

- Two directions cost about 1.9 times the collateral damage and bought nothing measurable
- Per-seed values are committed at `evidence/k-sweep-2026-08-13/drift-per-seed.json`
- **Corrected 2026-09-28:** the 95% interval on that 1.9 is 1.2 to 3.1
- An earlier "1.4 to 1.9" was not an interval; 1.4 drops one seed as a sensitivity check
- **"Bought nothing" is only true of what this design could see**
- Both arms sit on the refusal floor, so the largest visible gain was 0.1 of a point
- One direction beats two in 24 of 25 pairwise seed comparisons
- A permutation test over all 252 splits gives p = 0.016, or 0.048 without the worst seed
- This is one model, Qwen3-1.7B, and it does not settle the question for every architecture
- **These arms ran before held-out direction selection existed**
- So each second direction was chosen by the filter now known to accept everything
- The table prices an *arbitrary* second direction, not a well-chosen one

Where we can currently measure it, the project's central idea is unsupported, and we would rather
publish that than wait for a friendlier model.

### Breaking

Nothing here breaks an install. What breaks is the meaning of results you already hold.

- Every number from an earlier version is withdrawn (see above)
- `--max-directions` above 1 did nothing before this release, on any model, at any setting
- Gemma, Gemma 3 and Olmo 2 figures from earlier versions describe the **base** model
- The evaluation track is now required; a run without one refuses instead of guessing
- `--good-ds owner/name` now needs `senbonzakura[hub]`
- Shell completion needs `senbonzakura[completion]`
- 0.3.0 and 0.3.1 could not run the compass at all; this is the first release that has it

### Known defects
- **Every Gemma measurement is withdrawn.** The edit never reached the running state
- Gemma rescales each layer's output before adding it back, and the tool edited the input to that
  step
- The baked edit and a live hook disagreed by 0.578 in refusal rate, against 0.016 on Qwen3
- Fixed here. Earlier Gemma numbers are unmeasured rather than wrong
- **Ablating a direction removes most of it, not all.** Roughly an eighth survives
- Preserving each weight row's length keeps the model coherent and does not commute with the removal
- **A filter that can reject is not evidence that what it accepted carries refusal**
- Nothing here establishes that the extra directions remove refusal rather than ability
- A direction count reported before this release cannot be trusted, including the README's table
- Whether removing several directions beats removing one is unanswered

### Measurement
- Every run measures what the edit cost on a task the model either gets right or does not
- The package carries its own probe, so this works with no network: 256 arithmetic questions
- Attributed in THIRD-PARTY-NOTICES.md. Point `--capability-eval` elsewhere, or pass `""`
- A run that pinned its settings used to measure no capability and report success
- Result files now say whether capability was measured, not only whether it was asked for
- `senbonzakura compass` is a first-class command: does the model still recognise harm?
- Every compass figure carries an interval, measured on rows nothing was fitted or selected on
- The compass warns when the position it reads holds something other than a verdict token
- Results ship with their controls: a length-only ruler, a fixed token pair, a topic-matched arm
- A read-out audit reports what the model put at the scored position
- `--seed` exists, and every result records it with the code version, packages and commit

### Engine
- Directions come from grouping harmful prompts: a weapons refusal differs from a self-harm one
- `--direction-clusters` sets how many groups to look for
- A run asking for several directions keeps the same first direction it would have kept at one
- A contrast set too small to form two groups says so, and how many prompts it needs
- Hybrid architectures are supported: a layer holding a short convolution is now edited too
- Every saved model records what produced it, in its config and its weight files
- Sparse surgery restricts the ablation to the rows that write refusal
- The search can warm-start from a difference-of-means seed, or take a fixed prompt format
- A disk-space check runs before the search, and every result file is written atomically

### CLI
- `senbonzakura measure <model>` answers "did I break my model?" in one command, where it took five
- `senbonzakura Qwen/Qwen3-1.7B` is the whole command; the output directory and track are defaulted
- `senbonzakura quantise ./abliterated` works on a checkpoint, not only on a GGUF
- `--help` shows the flags a run needs and `--help-all` shows all 69 (the default was 470 lines)
- Two build steps became commands: `senbonzakura corpora` and `senbonzakura track build`
- Four commands stopped demanding arguments they could work out
- `harm-recognition` is a second name for `compass`, and `bench` is now `head-to-head`
- Real subcommands for `abliterate`, `kageyoshi`, `compass`, `score`, `coherence` and `track`
- `--track` left out takes `./track` if it exists, else the bundled track, and says which
- The capability probe refuses a run that would take hours on a CPU, and names four ways forward
- The probe reports progress every 30 seconds, where it printed nothing
- A startup banner on a terminal, and nothing when output is redirected

### Data
- `senbonzakura.track` builds an evaluation split and refuses to write one that leaks
- The leak check compares requests rather than strings
- Comparing whole prompts called a split clean while 60% of it was training questions reworded
- `--contamination` reports how much of a benchmark a track was fitted on, and never prints a row
- `--fail-on-contamination` gates on it
- The track card records that AdvBench is inside the track, and that how much reached the fitting
  side is unmeasured
- `build_track.py` fetches the public corpora at pinned revisions
- It refuses if an upstream licence has moved. **No prompt rows ship in this repository**
- The dataset card traces every source to what it declares
- It records that the exact corpus behind the published numbers cannot be rebuilt by anyone
- `senbonzakura track build` balances the two sides, because the sources return four times as many
  harmless prompts as harmful ones
- A format for contributing a behavioural probe, with gates that refuse a corpus posing as a
  benchmark
- The gates check shape only; a person still reads what a probe contains
- Per-prompt margins and generations are kept, and a guard refuses to commit a file with prompts

### Benchmark
- `senbonzakura head-to-head` runs a matched comparison between abliteration tools on one machine
- Both tools get the same corpus, budget and prompt slices. A mismatched pair is refused
- `--isolate docker` runs each arm with no network, read-only inputs and no credentials
- Every comparable axis gets a permutation test rather than two means side by side
- Where the seed count makes significance impossible, the report says so rather than calling a tie
- `senbonzakura drift` measures coherence over every model, whoever made it
- This release found two tools' self-reported divergences differ by a hundredfold and reverse order
- `--max-kl` searches for the fewest refusals under a coherence ceiling, and refuses if none meets
  it
- An arm that exits cleanly having produced nothing is retried rather than skipped
- Adding another tool is an adapter: how to invoke it, what proves it ran, where it leaves a model

### Install
- `pip install senbonzakura` installs torch, transformers, accelerate and optuna, as 0.3.0 did. 69
  packages, 5.9 GB
- `senbonzakura setup` puts the right build of torch on the machine, because pip picks by platform
- It changes nothing without `--apply`
- The wheel carries the bundled track, so `--track default` works with no network
- That is roughly 6,200 harmful prompts inside your site-packages, and the install page says so

### New since 0.3.0

0.3.0 installed three things you could run. Every command below is one it did not contain.

- `compass`: does the model still recognise harm, as opposed to still refusing?
- `capability`: what did the edit cost, on five graded tasks?
- `validate`: the checks behind a published number
- `judge`: validates a judge before it grades anything, and refuses a bad one
- `track`: build an evaluation split, and check it for contamination
- `head-to-head`: run this tool and another on the same model, one ruler
- `convert`, `quantise`, `imatrix`: GGUF conversion and quantisation
- `doctor`: can this install do what it claims, before a long run?
- `fetch`: fetch a model, resumable
- `report`: the model card for a finished run
- `interactive`: a guided mode for people who do not want to read the flag list
- `drift`: how far the edited model moved from the original
- `setup`: put the right build of torch on this machine, which pip cannot do itself

### Packaging and CI
- Continuous integration, which this repository had never had
- Linux, Windows and macOS across five combinations covering Python 3.10, 3.12 and 3.14
- 3.11 and 3.13 were removed from the classifiers, because claiming an interpreter nothing tests is
  a promise kept by luck
- The checking commands are tested with torch, transformers, accelerate and optuna unimportable
- So a verifier can still be built without them
- `datasets` is no longer needed for ordinary work; tracks are read with pyarrow
- Nothing on disk changed, and a track built by an older version reads the same
- Container images, so the tool runs without installing anything
- A CPU image that converts and quantises, and a CUDA image that abliterates
- Per-platform wheels carrying the pinned llama.cpp binaries
- Tagged for a platform only when they actually hold that platform's binaries
- Converting from a wheel needs the OpenMP runtime, which a wheel cannot ask a system for, so
  `doctor` names the missing library
- The command line starts in a hundredth of a second instead of just under three
- Every quantisation writes a record naming the toolchain version and the importance matrix used
- Dependency floors are tested at the floor, and a check enforces that pinned and declared agree

### Documentation
- A Colab notebook that measures a model, edits it, and measures what that cost, on a free GPU
- The headline comparison table now carries its caveats beside it
- The quickstart carries the same warning the other install pages do
- The README documents the track layout, the manifest, and the optional datasets
- A contributor licence agreement, a contributing guide, and a contributors file

### Licence
- The AGPL section 5(a) statement of modification, with dates, in three places
- A test pins the copied region so it cannot drift silently

### Security
- The pre-commit hooks no longer execute the project configuration file
- The prompt-retention guard reads what is staged rather than what is on disk

### Fixed
- **`senbonzakura measure` scored the prompts a configuration was chosen on, and called them held
  out**
- It carried a second copy of the boundary reader, which did not know the bundled track's alias
- So it read the boundary as zero where the real one is 132
- A run could wait for ever for video memory
- The pause firing under memory pressure waited for a level our own model can never leave
- The first run the documentation describes failed at its second command
- The refusal scanner read only the first 240 characters of a reply, and 51 of 56 refusal markers
  land past that point
- The compass scored the prompt read back to the model rather than the model's verdict
- A layer offloaded to disk was never abliterated, and nothing said so
- A non-converging matrix decomposition quietly weakened the ablation and reported full strength
- The prompt renderer existed in three copies that had drifted
- A configuration selected under one prompt format was reported under another
- `senbonzakura score` could not say which rows its number came from
- Every candidate direction's score is kept whether it was used or not
- Nine orthonormal directions cannot exist in an eight-dimensional space, and the code believed they
  could
- Nonsense slice arguments produced a plausible-looking result file instead of an error
- A mistyped command names the closest real command, or what replaced it
- A problem in a probe file reported a line number from inside the tool
- The guided mode's corpus menu offered four corpora and three printed a command that could not run
- AdvBench is gated on the Hub and was offered first, so pressing Enter gave anyone without an
  account an authentication error
- The guided mode read a helper from a module that imports torch
- On an install without it, the walkthrough asked four questions and produced a traceback
- A command reporting failure by return status ended the guided run silently

### Other
- Bug fixes and improvements.

## [0.3.1] "Petals" — 2026-09-09

A licence-compliance patch on 0.3.0, with no code change. Every file is byte-identical to 0.3.0's.

### Licence
- The package carries the AGPL section 5(a) statement of modification inside the artefact
- 0.3.0 shipped the licence text alone, so the statement reached nobody who installed it

### Other
- Bug fixes and improvements.

## [0.3.0] "Pilot" — 2026-07-17

The first public release: on PyPI, on GitHub, AGPL-3.0.

### Engine
- `kageyoshi`, a one-shot balanced preset: supply the paths and it does the rest
- It scales the search budget to the model's size and turns on every quality lever
- A resource governor that paces GPU work under live VRAM pressure
- It offloads a model larger than VRAM, and yields the GPU to a foreground game
- A packaged coherence probe, the neutral-passage perplexity check

### CLI
- Shell completion via shtab: `--print-completion` emits a script for bash, zsh or tcsh
- The coherence scorer rejected `--load-in-4bit`, so a bench chain passing the flag logged a false
  coherence failure

### Licence
- Relicensed to AGPL-3.0-or-later
- The keyword metric copies Heretic verbatim and Heretic is AGPL, so copyleft reaches the whole work

### Documentation
- README rebuilt around reproducible Qwen3-4B numbers
- The superseded Qwen3-1.7B Heretic comparison table is gone
- Leading a first-time reader with a comparison we have disowned is worse than none

### Other
- Bug fixes and improvements.

## [0.2.0] — 2026-07-14

### Install
- Installable package: `src/` layout, a `senbonzakura` console command, extras, a man page

### Engine
- Multi-direction refusal search: three-objective NSGA-II
- Over strict non-compliance, the keyword rate and KL, so both axes are optimised not only reported
- `--hedge-ds` folds a hedged-versus-clean contrast direction into the basis
- `--mlp-off` pins the MLP ablation to zero, to test where refusal lives
- `--patience` stops the search once the frontier stalls
- `--eval-refusal-final` re-scores the top frontier on a larger eval before the knee is chosen

### Other
- Bug fixes and improvements.

## [0.1.0] — earlier

### Engine
- The initial pipeline: activation capture and a difference-of-means refusal direction
- Weight orthogonalisation, and refusal and coherence scoring

### Other
- Bug fixes and improvements.

[0.4.1]: https://github.com/elementmerc/senbonzakura/releases/tag/v0.4.1
[0.4.0]: https://github.com/elementmerc/senbonzakura/releases/tag/v0.4.0
[0.3.1]: https://github.com/elementmerc/senbonzakura/releases/tag/v0.3.1
[0.3.0]: https://github.com/elementmerc/senbonzakura/releases/tag/v0.3.0
