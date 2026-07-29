# Evidence

The committed home for result artefacts. Everything here is version-controlled on
purpose, which is the whole difference between this directory and `results/`.

| Directory | Version-controlled? | Holds |
|---|---|---|
| `results/` | No, ignored | Raw run output, including the per-prompt margins and any retained generations. The prompts live here and stay here. |
| `evidence/` | **Yes** | Result artefacts with the `prompt` and `generation` fields stripped, plus the provenance needed to read them. |

## The rule

`.gitignore` excludes `*.jsonl` and then re-includes `evidence/**/*.jsonl`
explicitly. The negation is deliberate: without it, committing anything here would
need `git add -f`, and a habit of `git add -f` defeats the ignore rule it bypasses.
So the safe path is the easy path, and the unsafe one needs a flag.

`tools/check_prompt_artefacts.py` enforces the stripping rather than trusting it. It
refuses any staged `.json` or `.jsonl` carrying a `prompt`, `prompts`, `generation`
or `generations` key at any depth, and CI runs it over the whole tree. A finding is
either an unstripped artefact or a file that was never meant to be committed.

## What provenance means here

A number without its environment is not evidence. Each committed result needs, in
the file or beside it:

- the senbonzakura commit it was produced by,
- the model id and revision,
- the dataset and its pinned revision,
- the accelerator (`torch==2.5.1+cu124` on an H100 and on a 3090 are different
  measurements),
- the seeds, the device, and both arm skips,
- a pointer to the matching `constraints/measured-YYYY-MM-DD.md`.

The reason this list is written down: the July 2026 sweep captured none of it, and
`constraints/measured-2026-07-27-compass-sweep.md` is the record of what that costs.
It cannot be reconstructed, only re-run.
