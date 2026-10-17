# Commands

Every command runs as `senbonzakura <name>`, or `python -m senbonzakura <name>` from a checkout. `--help` on any command is the authority. This page is the map.

The default command has 77 flags. `--help` shows the ones a run needs. `--help-all` shows every one.

## Editing a model

| Command | What it does |
|---|---|
| `abliterate` | Remove refusal directions from a model's weights, at a configuration you choose. |
| `kageyoshi` | The same, with the configuration searched for. The usual entry point. |

## Measuring one

| Command | What it does |
|---|---|
| `measure` | Runs each instrument below against one model and writes one summary table. It measures nothing itself. Its `measure.json` holds each instrument's conditions verbatim, so `baseline` takes one of the per-instrument files, such as `score.json`, rather than the summary. |
| `compass`<br>(or `harm-recognition`) | Does the model still recognise a harmful request? Reports an AUC with a seeded bootstrap interval and its null controls. |
| `score` | Refusal and compliance rates over an evaluation set. |
| `tamper` | Whether the abliteration survives a safety-recovery finetune. Includes a neutral-data control and the same recipe on the unedited model as a ceiling. |
| `multi-turn` | Whether a refusal holds as the conversation continues, and the turn where it gives way. Includes a plain-repetition control. |
| `jailbreak` | Jailbreak success on a bundled attack set, with a Wilson interval, beside the over-refusal rate on a benign set. Both run together, because a model that declines everything scores perfectly on attacks. |
| `coherence` | Perplexity against a reference, so a model that stopped refusing because it stopped working shows up. |
| `drift` | One coherence ruler for any model, however it was edited. First-token KL. |
| `leak` | How much of a refusal direction is still present in a model's residual stream. It is a magnitude, not a verdict on whether an edit worked. Read it beside a refusal rate. We removed refusal completely on a model that kept 88 to 96 per cent of its original refusal direction. No judge and no generated tokens, so it works on any checkpoint. |
| `attribute` | Which blocks of an unedited model carry refusal, by removing one block's share of a direction at a time. Status is one of `localised`, `diffuse`, `redundant`, `direction_inert`, `absent`, `readout_invalid` or `unresolved`. Writes numbers and digests, never a prompt. |
| `capability` | What the edit cost on tasks the model gets right or wrong. Refusal rates and KL cannot see reasoning loss. |
| `judge` | Check a grading model against reference labels. Reports agreement above chance, and exits non-zero when it is not certified. |
| `validate` | Compare direction budgets at matched refusal removal. |
| `report` | Assemble a run's artefacts into the model card that travels beside the weights. |
| `stream-extract` | Extract refusal directions from a checkpoint too large to load. It is a forward pass only, and it writes the file `stream-bake` applies. |
| `stream-bake` | Apply a saved direction set to a checkpoint too large to load. It rewrites one shard at a time, and it never writes to the model directory. |

## Checking somebody else's result

| Command | What it does |
|---|---|
| `check` | Read an evaluation result file, from this tool or another, and report how its number could be wrong. Each finding names what to do about it. |
| `diff` | Compare a checkpoint against the base it claims to come from, tensor by tensor. **It is not a safety verdict and not a backdoor check.** A comparison showing no change means the weights match the base, and nothing more. |
| `baseline` | Turn one measurement into the baseline file `gate` compares against. It refuses when a condition is missing, and names every missing one at once. |
| `prereg` | Check a pre-registration. With `--run`, also check that the run did what it promised. |
| `gate` | Compare a measurement against a recorded baseline. Fails the build when a property moves outside its interval. It refuses when the two were measured under different conditions. Exits 0 when steady, 1 on a regression, and 2 when it refused. |

`check` and `diff` load no model. `check` needs only a result file. `diff` reads two checkpoints a tensor at a time. They are the cheapest commands here, and the only ones that work with no GPU.

```sh
senbonzakura check results/            # a directory, searched for .json
senbonzakura check run.json --json     # machine-readable, for a CI job
```

`check` exits 0 when every file was read and nothing fired, 1 on at least one finding, and 2 when a file could not be read at all. An unreadable file is reported as **unchecked**, never as clean.

**A clean report is not a certificate.** `check` looks for known failure modes. It cannot tell you a number is right.

### In CI and pre-commit

The GitHub Action installs `senbonzakura-check`, which declares no dependencies. The pre-commit hook runs the checker from the clone on your `python3`, and installs nothing.

```yaml
- uses: elementmerc/senbonzakura@v0.4.0      # pin it
  with:
    path: results/
    version: "==0.4.0"
    fail-on-findings: true
```

Read `checked` as well as `findings`. Zero findings over zero files looks identical to a clean sweep. `fail-on-unchecked` is true by default, and it should stay that way.

To fail the build on a regression, pass a baseline and a measurement. The step reports `gate-status` as `0`, `1` or `2`, and it fails on `2` as well.

```yaml
- uses: elementmerc/senbonzakura@v0.4.0      # pin it
  with:
    path: results/
    baseline: baselines/refusal-rate.json     # written by `senbonzakura baseline`
    measurement: out/this-run.json            # the same shape, from this run
```

The pre-commit hook only looks at JSON under `results/`, `evals/` and `logs/`. Naming a file on the command line claims it is a result, so `senbonzakura check run.json` exits 2 on an unrecognised file. The hook passes `--skip-unknown`, because pre-commit chose its paths by pattern.

```yaml
repos:
  - repo: https://github.com/elementmerc/senbonzakura
    rev: v0.4.0                               # pin it
    hooks:
      - id: senbonzakura-check
```

## Setting up the install

| Command | What it does |
|---|---|
| `setup` | Works out the right build of torch for this machine. Prints the command, and `--apply` runs it. |
| `doctor` | Checks that this install can do the job, and says so loudly when it cannot. |
| `budget` | Will this model fit on this machine, and roughly how long would a run take? Reads headers only, so it costs about a second on a 61 GB checkpoint. |

```sh
senbonzakura budget --model ./Qwen3-4B --trials 64 --prompts 48 --tokens 48
```

`budget` exits 0 when the run fits, 1 when the machine is short of something, and 2 when the directory is not a checkpoint it can read. Its time estimate assumes 1.95 GB/s of sustained reading, measured on one machine. Pass `--read-rate` with your own figure.

## The corpus

| Command | What it does |
|---|---|
| `corpora` | Fetch and pack the refusal corpora from public sources at pinned commits. A released wheel already carries them. A build from a clone does not. Needs the GitHub CLI. |
| `track build` | Fetch the harmful and harmless prompt pools at pinned revisions. Refuses to run if an upstream's licence has moved. |
| `track` | Build an evaluation split, audit one, or check one against a public benchmark for contamination. |

## Comparing tools

| Command | What it does |
|---|---|
| `head-to-head stage` | Cut the prompt slices every tool is scored on. **Run this first.** |
| `head-to-head run` | Run every tool over every seed, score every model with one instrument, and print the verdict. |
| `head-to-head report` | Read a finished run again, without re-running anything. |

`run` refuses to start without the slices `stage` writes. If an old slices directory stops working after an upgrade, cut it again.

::: warning Check which source you benchmarked
`run` prints the source tree it is about to measure, and its commit, before the first arm. It defaults to the tree the command was run from. A pip-installed senbonzakura benchmarks site-packages, even from inside a checkout. Pass `--senbon-src DIR` to say which one you mean.
:::

See [Benchmarking against another tool](/guide/benchmark).

## Shipping the result

| Command | What it does |
|---|---|
| `convert` | Turn edited weights into a GGUF with the pinned converter, optionally quantising in the same step. |
| `quantise` | Shrink a model with the pinned `llama-quantize`, then read the output back to confirm the quantisation. Given a transformers checkpoint, it converts first. |
| `imatrix` | Compute an importance matrix so a quantisation keeps the weights that matter, and record what it was calibrated on. |
| `fetch` | Download a model file and check it is the one asked for: length, GGUF header, architecture, and the quantisation its name claims. |

## If you would rather be asked than type flags

| Command | What it does |
|---|---|
| `interactive` | A guided walk through the choices that decide whether a run means anything. It prints the exact command before running it. |
| `auto` | An alias for `kageyoshi`. |

`interactive` needs a terminal and exits 2 when its input is a pipe. Its one other flag is `--no-interactive`, which refuses instead of asking. Use it when something fakes a terminal with nobody behind it, such as `docker run -t`:

```sh
docker run -t your-image senbonzakura interactive --no-interactive
```

A script wants the flags, not the menu. `senbonzakura --help` lists them.
