# Reproducing the published numbers

Every figure this project states in public, what file it came from, and what you would run to
produce that file yourself. If a number is quoted anywhere in the documentation and is not in the
table below, that is a defect: please open an issue.

The point of this file is narrow and worth stating. A claim that traces to a paragraph is not
checkable. A claim that traces to a committed artefact can be read without trusting us, and a
claim that traces to a command can be re-taken on your own hardware and disagreed with.

## Before anything: what you need

```sh
pip install "git+https://github.com/elementmerc/senbonzakura#subdirectory=checker"
pip install "git+https://github.com/elementmerc/senbonzakura"
```

To stand exactly where the numbers were taken, add the pinned set. It needs Python 3.12 or newer,
for the reason `docs/guide/install.md` gives:

```sh
pip install . -c constraints.txt      # from a clone
```

`requirements.lock` carries the same versions with hashes, if you want the bytes checked as well
as the versions.

A GPU is needed to edit a model. **Nothing in the table below needs one to verify**, because the
artefacts are committed: you can check every published figure against its file with no hardware at
all, and re-take it with hardware if you want to.

## The committed evidence

| Directory | What is in it |
|---|---|
| `evidence/compass-2026-07-30/` | Harm-recognition AUC for two base models, with the null controls beside them |
| `evidence/k-sweep-2026-08-13/` | Per-seed coherence drift for the one-direction against two-direction comparison |
| `head-to-head/results/` | Thirty arms of the comparison against another tool, on one corpus and one budget |

Result files have the `prompt` and `generation` fields stripped. That is not tidying: this
repository does not carry model generations or harmful prompts, and `CONTRIBUTING.md` explains
why that is a firm rule.

## Claim to artefact to command

### The compass figures, quoted in the README and in `paper.md`

| Claim | Value | Artefact |
|---|---|---|
| Qwen3-1.7B harm-recognition AUC | 0.9887 | `evidence/compass-2026-07-30/base-qwen3-1.7b.json` |
| Length-only control, both models | 0.6564 | the same two files |
| Qwen3-0.6B harm-recognition AUC | 0.6616 | `evidence/compass-2026-07-30/base-qwen3-0.6b.json` |

Read them without running anything:

```sh
python -c "import json;d=json.load(open('evidence/compass-2026-07-30/base-qwen3-1.7b.json'));print(d['metrics'])"
```

Re-take them. This needs a GPU and the corpus, and the split is the part that matters:

```sh
senbonzakura track --harmful harmful.txt --harmless harmless.txt --out mytrack
senbonzakura compass --model Qwen/Qwen3-1.7B \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds --out compass.json
```

**The second row is the one to look at**, and it is the reason this project believes what it says
about controls. 0.6616 against a length-only control at 0.6564 is not a weak result, it is a null:
the interval sits astride the control. The same file records that the model answered `HARMFUL` to
all 4,504 harmful prompts and all 4,504 harmless ones.

### The multi-direction result, quoted in `paper.md`

| Claim | Artefact |
|---|---|
| Five seeds, one direction against two, no measurable refusal gain at roughly twice the divergence | `evidence/k-sweep-2026-08-13/drift-per-seed.json` |

That file exists because the p-value this project was quoted on most often traced to a working
note nobody outside could open. It is ten floats and no prompts.

```sh
python -c "import json;d=json.load(open('evidence/k-sweep-2026-08-13/drift-per-seed.json'));print(d['drift_kl'])"
```

### The head-to-head, quoted in `docs/guide/benchmark.md`

Thirty arms, committed under `head-to-head/results/`. The whole comparison runs as one command,
and it takes hardware and hours:

```sh
senbonzakura bench --help
```

The keyword-rate row on that page was corrected on 2026-09-21: it had been quoting a superseded
run on the one axis where we lose. The page carries the correction rather than only the corrected
number.

## Checking our numbers the way we check other people's

The companion package reads result files, from this tool or from others, and reports how a figure
could be wrong: a rate on a sample too small to carry it, a proportion outside zero to one, a
figure reported on the rows it was selected on. It needs no model, no corpus and no network.

```sh
senbonzakura check head-to-head/results/ --min-applied 2
senbonzakura check evidence/
```

`--min-applied` is worth using and worth understanding. A file counts as checked when a single
check applied to it, so a run where almost nothing applied looks exactly like a clean run in the
exit code. CI asks for at least 2 per artefact, which is the floor the committed evidence actually
supports rather than a number chosen to pass.

Point it at somebody else's files too. It understands result files from
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and
[Inspect](https://inspect.aisi.org.uk).

## What you cannot reproduce from this repository, and why

**The corpus.** The harmful prompt set is not here. The evaluation track is published as a gated
dataset, and `docs/guide/the-track.md` explains the split and how to build an equivalent one from
public sources. This is the deliberate limit on reproducibility in this project, and it is a
trade: a fully reproducible harmful corpus in a public repository is a harmful corpus in a public
repository.

**Anything above 3B parameters.** It has never been run. The instruments also weaken as the model
shrinks, which the compass row above shows better than any sentence could.

**The abliterated weights.** No abliterated checkpoint is published with this software.

## If a number does not reproduce

Tell us. A figure that does not reproduce is the most useful issue this project can receive, and
it has happened before: several published numbers have been withdrawn after exactly that. Include
the command, the artefact you got, and `senbonzakura doctor` output.
[What is and is not established](https://elementmerc.github.io/senbonzakura/guide/what-we-know)
is the running record of which claims survived.
