# Commands

Every command is `python -m senbonzakura <name>`, or `senbonzakura <name>` after installing.
`--help` on any of them is authoritative; this page is the map.

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

## Setting up the install

| Command | What it does |
|---|---|
| `setup` | Look at this machine and put the right build of torch on it. pip cannot: there is no environment marker for a GPU, and PyPI cannot depend on PyTorch's index. Prints the command; `--apply` runs it. |
| `doctor` | Check this install can actually do the job, and say so loudly when it cannot. |

## The corpus

| Command | What it does |
|---|---|
| `track` | Build an evaluation split, audit an existing one, or check it against a public benchmark for contamination. |

## Comparing tools

| Command | What it does |
|---|---|
| `head-to-head stage` | Cut the prompt slices every tool will be scored on, from one corpus. |
| `head-to-head run` | Run every tool over every seed, score every model with one instrument, print the verdict. |
| `head-to-head report` | Read a finished run again, without re-running anything. |

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
