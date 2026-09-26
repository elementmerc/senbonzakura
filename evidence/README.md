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

`tools/ci/check_prompt_artefacts.py` enforces the stripping rather than trusting it. It
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

### What the artefacts in here actually carry

**Corrected 2026-09-25, because the list above was a rule nothing in this directory met.** A review
panel measured it and found that stating a requirement is not the same as enforcing one, which is
the defect this project keeps finding in other people's work.

| Artefact | Missing from the list above |
|---|---|
| `compass-2026-07-30/base-qwen3-*.json` | the model revision, the dataset identity, the dataset revision |
| `k-sweep-2026-08-13/drift-per-seed.json` | the commit, the model revision, the dataset identity, the dataset revision, the accelerator |

So a reader who obtained the gated corpus still could not re-take the compass figures exactly:
nothing records which revision of the dataset or of `Qwen/Qwen3-1.7B` produced them, and both can
move upstream. The k-sweep file is a hand-written summary of a run whose per-arm artefacts were not
kept, which is also why its hard-refusal column had no source until it was recovered from the run's
own job database on 2026-09-25.

**Neither gap is filled by editing the files.** A provenance field nobody measured is worse than an
absent one, because it reads as a receipt. They are superseded by re-measurement, not by annotation.

**The rule is enforced from now on.** `tests/test_evidence_carries_its_provenance.py` checks every
artefact in here against the list above, with those two directories recorded as exceptions that
name what they lack and why. A third exception requires a reason in that file, and a second test
refuses an exception broader than the artefact needs, so the list shrinks as artefacts improve
rather than growing as standards slip.
