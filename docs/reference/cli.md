# Commands

Every command is `python -m senbonzakura <name>`, or `senbonzakura <name>` after installing.
`--help` on any of them is authoritative; this page is the map.

The default command has 76 flags. `--help` shows the ones a run needs, and `--help-all` shows
every one of them with its full description. Nothing is hidden from the parser: both forms accept
exactly the same arguments.

## Editing a model

| Command | What it does |
|---|---|
| `abliterate` | Remove refusal directions from a model's weights, at a configuration you choose. |
| `kageyoshi` | The same thing with the configuration searched for rather than given. This is the usual entry point. |

## Measuring one

| Command | What it does |
|---|---|
| `measure` | Every instrument against one model, into one directory, as one table: refusal, harm recognition, fluency and capability, plus the coherence cost when `--baseline` names the model it was edited from. It measures nothing itself; each row is the command below it, run with the arguments you would have typed. |
| `compass`<br>(or `harm-recognition`) | Harm recognition: does the edited model still know a harmful request when it sees one? Reports an AUC with a seeded bootstrap interval and its null controls. |
| `score` | Refusal and compliance rates over an evaluation set. |
| `tamper` | Whether the abliteration survives a safety-recovery finetune: refusal before, a brief finetune, refusal after, on prompts the finetune never saw. Runs the neutral-data control that says whether the effect is about safety at all, and the same recipe on the unedited model as a ceiling. |
| `multi-turn` | Whether a refusal holds when the conversation continues, and the turn at which it gives way. Runs a named escalation and the plain-repetition control over the same refused prompts, so an escalation that does no better than asking again is visible as such. |
| `jailbreak` | Single-turn jailbreak success on a bundled attack set, with a Wilson interval and the raw counts, reported beside the over-refusal rate on a benign set. Both arms run in one command, because a model that declines every request scores perfectly on jailbreak resistance. |
| `coherence` | Perplexity against a reference, so a model that stopped refusing because it stopped working is visible as such. |
| `drift` | One coherence ruler applied to any model after the fact, so a model edited by any tool lands on the same scale. First-token KL, one position per prompt. This is the comparable coherence number. |
| `capability` | What the edit COST, on tasks the model either gets right or does not. Refusal rates and KL cannot see reasoning loss. |
| `judge` | Check a grading model against reference labels before letting it grade anything. Reports agreement above chance, and exits non-zero when not certified. |
| `validate` | Compare direction budgets at matched refusal removal, which is the comparison this project exists to make. |
| `report` | Assemble a run's artefacts into the model card that should travel beside the weights. |

## Checking somebody else's result

| Command | What it does |
|---|---|
| `check` | Read an evaluation result file, from this tool or from another one, and report how the number could be wrong. Each finding names the incident behind it, what to do about it, and what would make the finding itself wrong. |
| `baseline` | Turn one measurement into the baseline file `gate` compares against. It carries the conditions the number was measured under, the interval, the sample size, the seeds, the estimator and the precision, and it refuses rather than guessing when any of those is missing, naming all of them at once so you fix them in one pass. |
| `prereg` | Check a pre-registration: read the one fenced `prereg` block out of its Markdown and report what is missing or will not do what it is for. With `--run`, also check whether the run that claims to satisfy it actually did what it promised. Exits 0 when clean, 1 when the document is flawed, and 2 when it is not a pre-registration at all, because those are three different answers. |
| `gate` | Compare a measurement against a recorded baseline and fail the build when a property moved outside its interval. It refuses, rather than comparing, when the two were measured under different conditions: a green tick on two numbers that never matched is worse than no gate at all. Exits 0 when steady, 1 on a regression, and 2 when it refused, because a refusal says nothing about the model. |

`check` is the one command that needs nothing: no model, no corpus, no card, no network. It
reads files and does arithmetic. It understands
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) result files,
[Inspect](https://inspect.aisi.org.uk) eval logs, and this project's own artefacts.

```sh
senbonzakura check results/            # a directory, searched for .json
senbonzakura check run.json --json     # machine-readable, for a CI job
```

Three exit codes, and they mean different things on purpose:

| Code | Meaning |
|---|---|
| `0` | every file was read, and nothing fired |
| `1` | at least one finding |
| `2` | at least one file could not be read at all |

That last one matters more than it looks. A file the checker cannot parse is reported as
**unchecked**, never as clean, because a clean report on something nobody read is
indistinguishable from a clean bill of health. For the same reason the summary says how many
checks did not apply to a file: "nothing found" across checks that could not run is a different
statement from "nothing found".

**A clean report is not a certificate.** It looks for known failure modes. It cannot tell you a
number is right, and it says so in its own output.

Findings are printed as each file is finished rather than collected and shown at the end, so a
sweep over a few thousand artefacts that you stop halfway has still told you about the half it
did. The closing summary is the one thing that waits: a run that was interrupted has no summary
line and no exit status, and that absence is what tells you the report is partial. With `--json`
the array is left unterminated for the same reason, so a consumer parses it and knows the run did
not finish rather than reading a short report as a clean one.

A long sweep says where it has got to every thirty seconds, on standard error, so `--json` on
standard output stays machine readable while it does.

### Running it without choosing to

A checker somebody remembers to run is a checker that runs occasionally. Neither of these pulls
torch into your CI or your commit hook, and they get there by different routes. The Action
installs `senbonzakura-check`, a separate package that declares no dependencies at all, so an
ordinary install resolves to one small wheel. The hook installs nothing: it runs the checker out
of the clone pre-commit already made, on the `python3` already on your PATH.

That is worth saying because the obvious shortcut is wrong. Both of these used to reach for
`--no-deps`, which suppresses dependency resolution, and suppressing it is what once hid a broken
install: the package arrived, the command could not run, and the step reported success.

::: tip These work from v0.4.0, which is out
The `v0.4.0` tag exists, `action.yml` and `.pre-commit-hooks.yaml` are on the default branch that
Actions and pre-commit read, and `senbonzakura-check` is on PyPI. Until 2026-09-26 none of that
was true and this box said so, because the snippets had been documented before any of it was and
had never been run.
:::

**In GitHub Actions:**

```yaml
- uses: elementmerc/senbonzakura@v0.4.0      # pin it
  with:
    path: results/
    version: "==0.4.0"
    fail-on-findings: true                    # false to report without blocking
```

It exposes `checked`, `findings`, `unchecked` and `report` as outputs, so a later step can act on
each count separately. Read `checked` as well as `findings`: zero findings over zero files is
arithmetically identical to a clean sweep, so a path that drifts reports green for ever.
`fail-on-unchecked` is true by default and is worth leaving that way: a gate that ignores a file
it could not read goes green on the day your result format changes.

**Failing the build on a behavioural regression.** Give the same step a baseline and a
measurement and it also runs `gate`, so a model change that moves a property outside its interval
fails before anything is promoted.

```yaml
- uses: elementmerc/senbonzakura@v0.4.0      # pin it
  with:
    path: results/
    baseline: baselines/refusal-rate.json     # written by `senbonzakura baseline`
    measurement: out/this-run.json            # the same shape, from this run
```

It reports what it concluded as `gate-status`: `0` the property stayed inside its interval, `1` it
regressed, `2` the two measurements were never comparable. The last one fails the build too, with
a different message, because a refusal means nothing was shown about the model at all and reading
that as either a pass or a regression teaches you to ignore the difference. Pass one of the two
inputs without the other and the step stops rather than quietly skipping the comparison: a
workflow whose author believes their numbers are gated and whose build has never compared them is
worse than no gate.

**As a pre-commit hook:**

```yaml
repos:
  - repo: https://github.com/elementmerc/senbonzakura
    rev: v0.4.0                               # pin it
    hooks:
      - id: senbonzakura-check
```

It fires only on JSON under `results/`, `evals/`, `logs/` and similar, because a hook that runs
on every commit regardless is a hook that gets skipped with `--no-verify`.

One difference worth knowing between the two surfaces and the command line. **Naming a file is a
claim that it is a result; a pattern match is not.** So `senbonzakura check run.json` on an
unrecognised file exits 2, while the hook passes `--skip-unknown`, because pre-commit chose those
paths with a regex rather than a person choosing them. Without that, a config file living in
`results/` would block the commit.

## Setting up the install

| Command | What it does |
|---|---|
| `setup` | Look at this machine and put the right build of torch on it. pip cannot: there is no environment marker for a GPU, and PyPI cannot depend on PyTorch's index. Prints the command; `--apply` runs it. |
| `doctor` | Check this install can actually do the job, and say so loudly when it cannot. |
| `budget` | Will this model fit on this machine, and roughly how long would a run take? Reads the checkpoint's headers, measures the card, the memory, the free disk and whether a laptop is plugged in, and refuses a run that cannot finish. |

`budget` reads headers and never a weight, so it costs about a second on a 61 GB checkpoint and
needs no card and no network. It names every figure it used, including the ones it could not
measure: a pool this machine will not report is marked skipped rather than passed, and the
verdict says so too, because a check that could not run is not a check that passed.

```
senbonzakura budget --model ./Qwen3-4B --trials 64 --prompts 48 --tokens 48
```

It exits 0 when the run fits, 1 when this machine is short of something, and 2 when the directory
is not a checkpoint it can read. Two of its figures are worth knowing about:

- **The time estimate assumes 1.95 GB/s** of sustained reading, which was measured on one machine
  and one disk. Pass `--read-rate` with your own number rather than inheriting ours.
- **The page-locked ceiling is a design constraint, not a tuning note.** A host side store above
  it loses the overlap between copying and computing, so the run gets much slower for a reason
  that has nothing to do with streaming. The report says which side of the ceiling your store
  falls on, and `--no-pinned-probe` skips the measurement, which is the only part that allocates.

## The corpus

| Command | What it does |
|---|---|
| `corpora` | Fetch and pack the refusal corpora this tool measures with, from public sources at pinned commits. A released wheel already carries them; an install made straight from the repository does not, because they are generated rather than committed. Needs the GitHub CLI. |
| `track build` | Fetch the harmful and harmless prompt pools, at pinned revisions, and write the two text files the next command splits. It refuses to run if any upstream's declared licence has moved since the recipe was written. |
| `track` | Build an evaluation split, audit an existing one, or check it against a public benchmark for contamination. |

## Comparing tools

| Command | What it does |
|---|---|
| `head-to-head stage` | Cut the prompt slices every tool will be scored on, from one corpus. **Run this first.** |
| `head-to-head run` | Run every tool over every seed, score every model with one instrument, print the verdict. |
| `head-to-head report` | Read a finished run again, without re-running anything. |

`stage` before `run`, always. `run --eval-slices` takes the directory `stage` writes; it's an
input you generate, not a directory the run fills in, and `run` refuses to start without it. If
you upgrade and an old slices directory stops working, re-cut it: a release can add a required
slice, and a directory staged before that is short of it rather than broken.

::: warning Check which source you benchmarked
`run` prints the senbonzakura source tree it's about to measure, and its commit, before the first
arm. It defaults to the tree the command itself was run from, so a pip-installed senbonzakura
benchmarks site-packages even if you're sitting in your checkout. Pass `--senbon-src DIR` to say
which one you mean, and read that line rather than assuming.
:::

See [Benchmarking against another tool](/guide/benchmark) for what makes it a comparison rather
than two runs that happened near each other.

## Shipping the result

| Command | What it does |
|---|---|
| `convert` | Turn edited weights into a GGUF with the pinned converter, optionally quantising in the same step. |
| `quantise` | Shrink a model with the pinned `llama-quantize`, then read the output back to confirm it is the quantisation that was asked for. Given a transformers checkpoint rather than a GGUF, it converts first, so `senbonzakura quantise ./abliterated` is the whole route from edited weights to something llama.cpp will serve. |
| `imatrix` | Compute an importance matrix so a quantisation keeps the weights that matter, and record what it was calibrated on. |
| `fetch` | Download a model file and prove it is the one asked for: length, GGUF header, architecture, and the quantisation its name claims. |

## If you would rather be asked than type flags

| Command | What it does |
|---|---|
| `interactive` | A guided walk through the handful of choices that decide whether a run means anything. It prints the exact command before running it, so the second time you can type that instead. |
| `auto` | An alias for `kageyoshi`, for anyone who has not met the name. |

`interactive` needs a terminal, because it reads answers, and it refuses with exit status 2 when
its input is a pipe. It takes `-h` and one other flag:

| Flag | What it does |
|---|---|
| `--no-interactive` | Refuse instead of asking, and exit 2. |

That flag is for one case. Deciding whether there is a person present is a guess, made by asking
the operating system whether the input is a terminal, and the guess is wrong whenever something
hands the command a fake terminal with nobody behind it. `docker run -t`, `expect` and some CI
agents all do that. The command cannot tell them from a person, so it prints the menu and waits
for an answer that never comes. Passing the flag says so outright, and saying beats guessing:

```sh
docker run -t your-image senbonzakura interactive --no-interactive
```

```
senbonzakura: guided mode was asked to run with --no-interactive, so it is not asking anything.
  A script wants the flags rather than the menu. `senbonzakura --help` lists them,
  and `senbonzakura interactive` on a terminal prints the command for any run.
```

If you are writing the script, the flags are what you want in it rather than the flag that turns
the menu off; `senbonzakura --help` lists them. The flag is for the case where something else
decides how your command is launched.
