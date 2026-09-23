# Commands

Every command is `python -m senbonzakura <name>`, or `senbonzakura <name>` after installing.
`--help` on any of them is authoritative; this page is the map.

The default command has 69 flags. `--help` shows the ones a run needs, and `--help-all` shows
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
| `compass` | Harm recognition: does the edited model still know a harmful request when it sees one? Reports an AUC with a seeded bootstrap interval and its null controls. |
| `score` | Refusal and compliance rates over an evaluation set. |
| `coherence` | Perplexity against a reference, so a model that stopped refusing because it stopped working is visible as such. |
| `drift` | One coherence ruler applied to any model after the fact, so a model edited by any tool lands on the same scale. This is the comparable coherence number. |
| `capability` | What the edit COST, on tasks the model either gets right or does not. Refusal rates and KL cannot see reasoning loss. |
| `judge` | Check a grading model against reference labels before letting it grade anything. Reports agreement above chance, and exits non-zero when not certified. |
| `validate` | Compare direction budgets at matched refusal removal, which is the comparison this project exists to make. |
| `report` | Assemble a run's artefacts into the model card that should travel beside the weights. |

## Checking somebody else's result

| Command | What it does |
|---|---|
| `check` | Read an evaluation result file, from this tool or from another one, and report how the number could be wrong. Each finding names the incident behind it, what to do about it, and what would make the finding itself wrong. |
| `baseline` | Turn one measurement into the baseline file `gate` compares against. It carries the conditions the number was measured under, the interval, the sample size, the seeds, the estimator and the precision, and it refuses rather than guessing when any of those is missing, naming all of them at once so you fix them in one pass. |
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

### Running it without choosing to

A checker somebody remembers to run is a checker that runs occasionally. Both of these install
the package with `--no-deps`, so neither pulls torch into your CI or your commit hook.

**In GitHub Actions:**

```yaml
- uses: elementmerc/senbonzakura@v0.4.0      # pin it
  with:
    path: results/
    version: "==0.4.0"
    fail-on-findings: true                    # false to report without blocking
```

It exposes `findings`, `unchecked` and `report` as outputs, so a later step can act on the two
counts separately. `fail-on-unchecked` is true by default and is worth leaving that way: a gate
that ignores a file it could not read goes green on the day your result format changes.

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
| `quantise` | Shrink a GGUF with the pinned `llama-quantize`, then read the output back to confirm it is the quantisation that was asked for. |
| `imatrix` | Compute an importance matrix so a quantisation keeps the weights that matter, and record what it was calibrated on. |
| `fetch` | Download a model file and prove it is the one asked for: length, GGUF header, architecture, and the quantisation its name claims. |

## If you would rather be asked than type flags

| Command | What it does |
|---|---|
| `interactive` | A guided walk through the handful of choices that decide whether a run means anything. It prints the exact command before running it, so the second time you can type that instead. |
| `auto` | An alias for `kageyoshi`, for anyone who has not met the name. |
