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
| `validate` | Compare direction budgets at matched refusal removal, which is the comparison this project exists to make. |

## The corpus

| Command | What it does |
|---|---|
| `track` | Build an evaluation split, audit an existing one, or check it against a public benchmark for contamination. |

## Comparing tools

| Command | What it does |
|---|---|
| `bench stage` | Cut the prompt slices every tool will be scored on, from one corpus. |
| `head-to-head run` | Run every tool over every seed, score every model with one instrument, print the verdict. |
| `bench report` | Read a finished run again, without re-running anything. |

See [Benchmarking against another tool](/guide/benchmark) for what makes it a comparison rather
than two runs that happened near each other.
