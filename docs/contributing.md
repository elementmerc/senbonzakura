# Contributing

You are here to work on the tool rather than to use it. Everything below assumes a clone rather
than a `pip install`.

## Running the tests

You don't need a model, a download or a graphics card. The whole suite runs on CPU against
small hand-built fixtures, which is deliberate: a test suite you can only run on the
machine that has the card is a test suite that gets run once a week.

```sh
python -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest              # add --cov for coverage
```

That first torch line pulls the CPU build, which is about a fifth of the size of the CUDA
one. If you already have a CUDA torch installed, skip it; the tests don't care.

## If you're going to commit

Wire the local gates once per clone:

```sh
bash tools/install-local-hooks.sh
```

That hooks `tools/check_prompt_artefacts.py` into your pre-commit path, and it exists
because of a specific hazard rather than as general tidiness.

::: warning What the gate is actually stopping
The tool keeps per-prompt margins and generations by default, because every single scoring
bug in this project's history was invisible in the percentages and completely obvious in the
text. Those rows contain harmful prompts and the replies a model gave to them. They're the
most useful debugging artefact here and the last thing you want to push to a public
repository at half past one in the morning.

The gate refuses any staged JSON or JSONL carrying a `prompt` or `generation` field. CI runs
the same check across the whole tree, but by the time CI sees it, the commit exists.
:::

## Where next

- [Which models it can open](/reference/models) for what the architecture modules cover.
- [Read this before quoting a number](/guide/what-we-know) for the standard any new measurement
  in this repository is held to.
