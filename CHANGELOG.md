# Changelog

All notable changes to Senbonzakura are recorded here, newest first, grouped by the part of the
tool they touch. Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Not in any released version yet. Each entry describes behaviour that is in the tree, grouped by the
surface it changed, and will carry a version heading when it ships.

### Budget
- Says what else would fit, not only whether this run does: for each resource it names the largest
  single thing you could still add and how many times this run's own figure that is, so a machine
  with room to spare tells you how much room rather than just "yes"

### Checker
- `--min-applied` refuses a negative count at the flag, like every other count on both commands,
  rather than accepting one and comparing against it

### Command line
- `attribute` pins one date for the whole run when the model's chat template writes today's date (Llama 3's does), so a run
  that crosses midnight, or a repeat on another day, no longer reads different prompts; the artefact records the date, and
  `--template-date YYYY-MM-DD` repeats a run with it
- `attribute` runs the faster sweep by default: the front blocks' forward passes and the greedy replies are shared where
  the numbers are provably the same, and the first batch is checked against a full pass. `--reference-sweep` runs the
  original way, `--fast-sweep` is still accepted and does nothing, and the artefact records which sweep ran
- `attribute` spills the prefix cache into host memory by itself, and only when its projection says the batches won't all
  fit the cache; it still never takes more than half the free memory. `--prefix-cache-spill` forces it on,
  `--no-prefix-cache-spill` forces it off, and the artefact records the choice
- `attribute` marks a readout as fragile (`readout_fragile`, with the tokens behind it) when leaving out two fit replies
  or fewer would change which tokens are in either set, and says so in its output and its limits; the verdict is not changed
- Every command that measures a model now loads the same formats and takes the same flags: plain safetensors, 4-bit and
  8-bit bitsandbytes (on the fly with `--load-in-4bit` or `--load-in-8bit`, or stored that way in the checkpoint), GPTQ,
  AWQ and GGUF (a file, a folder, or a hub id with `--gguf-file`); `attribute` had none of these and now has all of them,
  plus `--trust-remote-code`, which stays off unless you pass it
- `measure` hands the same loading flags to every stage it runs, so a 4-bit, 8-bit or GGUF model can be measured
  end to end; the validate stage rewrites weights and refuses them in one sentence
- A GGUF is expanded to full precision to be measured, and the result records that, along with which quantisation it came
  from; a folder holding only a GGUF with no tokenizer files now loads the tokenizer from the file or says in one sentence
  which package to install or which file to add
- A format whose backend package isn't installed is refused in one sentence naming the package and the install command,
  before any weights are fetched; the new `gptq` and `awq` extras install the GPTQ and AWQ backends
- Results record the loading format, quantisation method and bits, whether the weights were dequantised, and the backend
  package's version; artefacts written before these fields read as plain full precision loads
- `attribute` takes `--chat-template`, checked the same way as in `score`, and its artefact records which template
  produced the figures; a model with no template is now refused up front, before its weights are read
- `score` takes `--revision`, loads that revision of a Hub model, and records the model's revision in its results the way
  `attribute` does, so a score and an attribution of the same model can be tied together by more than a name
- `--help` on `drift`, `score`, `coherence`, `validate` and `stream-extract` answers in about a
  sixteenth of a second instead of over three, because reading a help page no longer loads the
  machine-learning libraries the command would only need once it had work to do
- When a command is missing a required argument it now prints the shortest command line it will
  accept, indented so you can copy it straight back out, on every subcommand rather than on the
  few that had the example written by hand

### Plugins
- A plugin can add a measurement. The contract is one documented protocol with two transports, in
  the same process for something installed beside us and in a subprocess for anything else, with a
  deadline on every call, a cap on the reply and the plugin's own standard error quoted back when
  it fails. Our own instruments are served across it too, so the first thing the contract is held
  to is us

### Report
- `senbonzakura report --overrefusal` takes a run on harmless prompts and adds an over-refusal
  section to the card, with its count and interval. Without it the card says that side was not
  measured, because a refusal rate on its own cannot tell a model that refuses what it should from
  one that refuses everything
- `senbonzakura head-to-head report` takes `--out` and writes the comparison as a machine readable
  summary. It declares itself a summary, stamps no figure of its own, carries each arm's own
  recorded conditions word for word, and lists the conditions the arms differ on, so a reader can
  see what a comparison between them is and isn't entitled to say

### Documentation
- A new reference page documents the plugin protocol: the manifest a plugin returns, the request
  and reply shapes, the limits on each, and how to install one so the tool finds it

### Other
- Bug fixes and improvements.

## [0.5.1] "Kuchiki" — 2026-10-13

A hotfix. No new features: security, and the defects that leave somebody worse off than before
they ran the command.

### Security
- The compiled part of the tool is built on PyO3 0.29.0, which fixes two advisories against 0.27: an
  out of bounds read when iterating a list or tuple, and a missing thread-safety bound on closures.
  The tool's own code does not reach either path, but the library it ships is now the fixed one
- `diff` and `budget` could be ended by a crafted checkpoint header. A file of about 200 KB was
  enough. Both now refuse it and say why
- The wheel no longer carries a RunPod client that read an API key from your environment and could
  be reached from no command

### Fixed
- `attribute` records the split-half reliability as not applicable (null) when no block clears the null, where it used to record 0, which read as a failed reproduction. No status changes, because such a run was never tested for reliability. Older artefacts keep their 0, and their reference set size of 0 says it was not applicable
- `attribute` records the torch and transformers versions it was made with in the artefact's comparability
  block, and its follow-up runs refuse an artefact made under different versions, naming both. An artefact
  from before this change is refused rather than assumed to match, because its versions were not recorded
- `setup --apply` put back the torch it replaced when the new one will not start. It used to leave
  the machine with no working torch at all, after a download of several gigabytes
- `setup` says the download is large, and on WSL says the chosen CUDA build may not run there, both
  before it starts rather than after it fails
- `doctor` fails when a package the install declares is missing. It reported all clear while `score`
  refused on the same missing package and told you to run `doctor` to find it
- `doctor` names the graphics card it can see and why it is unused, instead of reporting only that
  no card is in use
- `tamper` refuses a finetune whose loss stopped being a number, rather than reporting it as
  resistance to tampering. No figure from 0.5.0 is affected: the command shipped the same day
- `tamper` also refuses a finetune whose loss blew up and came back, which it used to report as a
  finetune that did nothing. Every value in such a run is a real number, so the check for a loss
  that stopped being a number cannot see it
- `tamper` describes a loss that rose as having risen. It used to call it a fall of a negative
  amount, which says the opposite to anyone reading quickly
- `diff` reports a tensor holding a NaN or an infinity as exactly that, and says which side it is
  on, instead of claiming it cannot read the dtype
- `diff` holds its memory flat whatever the tensor's size, which is what its own `--help` promised.
  It is also about three times faster
- `diff` warns before a comparison that will take a long time, and says something while it is
  inside one large tensor
- A model path that exists on your disk is no longer answered with advice about authenticating to
  Hugging Face
- A model path that does not exist is answered the same way. Typing a directory wrongly used to
  send you off to check a token that was never involved
- A checkpoint cloned without its large files is named as exactly that, with the files that are
  still placeholders and the command that fetches them. It used to fail inside a JSON parser
- `stream-extract` refuses `--offload-dir` together with `--device cpu` at the point you type them,
  in one sentence saying which to drop. The two cannot work together, and on a machine with no
  usable card the advice given for the missing card led straight into the pair
- A direction set says how many directions it holds rather than how many it had room for. A run
  that found one per position reported three, which is the size of the space they sit in
- A direction file records the number found alongside the number allocated, so the file can be read
  without guessing which one a figure refers to
- A direction set is written whole or not at all, so a run stopped during the write cannot leave a
  file that looks like a direction set and is not one
- Ctrl+C says what happened in a sentence and exits 130. Every command except the guided search
  used to answer it with a traceback through whatever it was doing at the time
- `senbonzakura-check` says a file is not text rather than failing with a decoding error, which is
  what happened when you pointed it at a model or a direction set
- A flag that refuses a number out of range no longer explains it by describing something the flag
  does not do. `--seed -1` was told that a negative seed slices rows from the end of a set, and a
  seed does not index anything. The explanation now appears only on the flags it is true of
- A flag you typed correctly is not suggested back to you. Passing `--steps 0` was answered with
  "Did you mean --steps?" alongside the real complaint
- `diff` and `budget` leave no `.part` file behind when writing `--out` fails

### Breaking
- `--inspect` refuses a negative layer index or a negative ablation strength. This landed in 0.5.0
  and was not written down at the time
- The floor on `senbonzakura-check` moves to 0.5.1. The two distributions share a version line, so
  a pin holding the checker at 0.5.0 will conflict

### Docs
- The README says what to do when pip refuses with "externally-managed-environment", which is the
  default on Debian, Ubuntu and Fedora
- A published AUC comparison in `evidence/compass-2026-10-11` is withdrawn. One row quoted two
  figures that no committed artefact produces and that cannot be recomputed from the record beside
  them. The removal is explained in place rather than made quietly
- The website said 0.4.1 and linked that tag, two releases after it stopped being current

### Other
- Bug fixes and improvements.

## [0.5.0] "Kuchiki" — 2026-10-13

The headline is streaming: a checkpoint too large to load can be measured and edited on one machine. Five new commands ask whether an edit holds up.

### Breaking
- `--device cuda` on a machine with no usable card refuses before the download, instead of running fifty times slower on the processor
- `senbonzakura gate` refuses a point estimate that is not a finite number (exit 2), where a NaN used to pass as "within interval"
- Wheels are platform-specific now, because the package ships a compiled Rust extension, so a platform with no published wheel builds from source and needs a Rust toolchain
- Floors moved: huggingface_hub 1.5.0, transformers 5.10.1, torch 2.7, and senbonzakura-check 0.5.0, so a pin holding the checker at 0.4.x will conflict

### Security
- A run given both `--hf-token` and `--base-licence` wrote the token into the model card; it is `***` there now, and in `report --command`. If you published a card from such a run under 0.4.0 or 0.4.1, revoke that token and replace the card

### Streaming
- `stream-extract` pulls refusal directions out of a checkpoint too large to load, and writes the file `stream-bake` applies
- `--offload-dir` puts the blocks that fit in neither the card nor host RAM on disk, read back as the forward pass reaches them
- The directions a streamed run finds are held identical to a full run's by test

### New commands
- `diff` compares a checkpoint against the base it claims to come from, tensor by tensor, with no model loaded and no network. It is not a safety verdict, and says so in its own output
- `jailbreak` reports single-turn jailbreak success beside the over-refusal rate on a benign set, in the same run, because a model that declines everything scores perfectly on the first figure alone
- `multi-turn` asks whether a refusal holds when the conversation continues, with the plain-repetition control beside it
- `tamper` measures whether an abliteration survives a brief safety-recovery finetune, against a neutral-data control and the unedited model as a ceiling
- `budget` answers "will this model fit on this machine" from the checkpoint's headers, before anything is downloaded

### Measurement
- `abliterate --leak-report` measures how much of the refusal direction is left in the weights that were saved, with no judge and not one generated token
- That figure is the mean over the positions the edit reached, named before the old mean over all of them, which positions never edited had inflated
- `capability --task tool-call --tool-schema <file>` offers the model real tool declarations and checks every reply against them by code, each rate over its own denominator
- Two defects in the tool-call reader, both of which scored a correct call as wrong; no published figure came through it, so nothing moves
- Graded prose measures beside `is_broken`: a repetition rate, a vocabulary ratio, and a shortfall against the generation budget. `is_broken` itself is unchanged, so every published figure resting on it still holds
- A null ruler that separates the two sides perfectly in the wrong direction is named as the strongest one, where it used to be scored as a coin flip
- `drift`, `score --harm-recognition` and `capability` each refuse a figure they cannot stand behind, naming the likely cause
- `quantise --like <reference.gguf>` copies the per-tensor precision schedule out of a finished GGUF; the importance matrix behind such a build is not copied, and every run says so

### Report
- Every rate carries its counts and a 95% interval, so a figure over nine replies no longer reads like one over three thousand
- New "What was measured" and "What this does not cover" sections name the estimator behind each figure and the gaps the run actually has
- The card reads `--refusal`, `--multiturn` and `--tamper` artefacts
- A run that disowned its own figures is quoted nowhere on the card, and is still listed as a gap
- Three sentences an artefact wrote about its own figures were in the file and never reached the page

### Integration
- The Action runs the regression gate when given a baseline and a measurement, and a pass, a regression and a refusal produce three different annotations
- `gate --history` records every run, pass, fail or refusal, named after its own contents, so a regression caught in March is still evidence in June
- A recorded measurement carries its own digest, so `--store` can fetch a baseline by address, and refuses one that has been edited since
- `check` prints findings as each file finishes, and an interrupted sweep has no summary and no exit status, so a partial report is never read as a clean one
- `check --quiet` prints one line per finding and nothing else, for a build log or a grep

### Documentation
- A new page on whether an edit holds up: what each new command measures, the control it runs, and when a run refuses to give you a number at all
- The norm-restore claim about the field is withdrawn, not adjusted, and restated scoped to the tools actually read
- "Nobody validates the judge" is now "nobody computes agreement above chance", which is the claim the evidence supports
- The comparison page says which tools it read, which it did not, and on what date
- Pages that disagreed with each other are corrected: the GPU requirements table for a 6 GB card, the capability probe as five grading rules over one dataset, and `drift` as a first-token measurement

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
