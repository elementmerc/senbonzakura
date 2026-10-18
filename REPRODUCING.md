# Reproducing the published numbers

Every figure this project quotes falls into one of three groups.

1. **It traces to a committed file.** The compass figures, the per-seed drift and the head-to-head. You can read them from a file in this repository with no hardware, and re-take them with hardware.
2. **It traces to a run whose logs are not committed.** The evidence for the Gemma withdrawal ([`docs/guide/limits.md`](docs/guide/limits.md)) and the count of rejected candidate directions in [what is and is not established](https://elementmerc.github.io/senbonzakura/guide/what-we-know). Both pages say so where the number appears.
3. **Neither.** A number with no source beside it is a defect. Please open an issue.

## Setup

```sh
git clone https://github.com/elementmerc/senbonzakura     # the committed artefacts
cd senbonzakura
pip install senbonzakura==0.4.0                           # the tool and the bundled track
```

You need the clone. `evidence/` and `head-to-head/` are not in the package, so the commands below fail with `FileNotFoundError` without it. You need the pip install as well, because the wheel carries the two `.bin` blobs and the bundled track that a source build lacks. The version is pinned because a figure can only be re-taken against the code that produced it.

To stand exactly where the numbers were taken, use the pinned set. It needs Python 3.12 or newer, as [`docs/guide/install.md`](docs/guide/install.md) explains:

```sh
pip install . -c constraints.txt      # from a clone
```

`requirements.lock` carries the same versions with hashes, if you want the bytes checked too.

Reading the committed numbers needs only the clone. Re-taking one needs a GPU.

## The committed evidence

| Directory | What is in it |
|---|---|
| `evidence/compass-2026-07-30/` | Harm-recognition AUC for two base models, with the null controls beside them |
| `evidence/k-sweep-2026-08-13/` | Per-seed coherence drift for one direction against two |
| `head-to-head/results/` | Thirty artefacts: ten arms, three instruments against another tool, on one corpus and one budget |

Result files have the `prompt` and `generation` fields stripped. This repository does not carry model generations or harmful prompts. [`CONTRIBUTING.md`](CONTRIBUTING.md) explains why.

## Compass figures

Quoted in the README and in `paper.md`.

| Claim | Value | Field | Artefact |
|---|---|---|---|
| Qwen3-1.7B harm-recognition AUC | 0.9887 | `auc` | `evidence/compass-2026-07-30/base-qwen3-1.7b.json` |
| Length-only control, both models | 0.6564 | `controls.length_only_auc` | the same two files |
| Qwen3-0.6B harm-recognition AUC | 0.6616 | `auc` | `evidence/compass-2026-07-30/base-qwen3-0.6b.json` |

The control is stored at full precision (`0.6563985752549933`) and quoted to four places.

Read all three without running anything:

```sh
python3 -c "
import json
for name in ('base-qwen3-1.7b', 'base-qwen3-0.6b'):
    d = json.load(open('evidence/compass-2026-07-30/' + name + '.json'))
    print(name, d['auc'], d['auc_ci'], round(d['controls']['length_only_auc'], 4))
"
```

To re-take them, use the recipe in [`evidence/compass-2026-07-30/README.md`](evidence/compass-2026-07-30/README.md). It is copied here word for word so that there is one command, not two that can disagree. `<track>` is what `senbonzakura track` wrote.

```sh
python -m senbonzakura.margin --model Qwen/Qwen3-1.7B \
    --harmful <track>/bad_eval_ds --harmless <track>/good_ds \
    --harmless-matched <matched>/good_matched_ds \
    --skip-harmful 132 --skip-harmless 385 --n 4504 \
    --batch 16 --bootstrap 2000 --seed 42 --device cuda \
    --out base-qwen3-1.7b.json
```

- **The `--harmless-matched` line cannot be followed as written.** Nothing in this repository builds a `good_matched_ds` directory, and `senbonzakura track` does not write one ([the track guide](docs/guide/the-track.md)). The one copy that existed was hand-built, sat outside this repository, and is not carried here. Pointing the flag at a missing path stops the run with a message asking for a directory built by `senbonzakura track`, which that step does not produce. The ledger chose not to build a replacement. On 2026-09-02 the old set scored 0.722 on topic match against 0.718 for the general corpus, where a perfect match scores 0.299. The item is in [`DEFERRED.md`](DEFERRED.md).
- To re-take the numbers without the topic-matched control, delete that line. The file then has no `topic_matched` entry, so the construct-validity check is missing from it.
- **Every other flag is load-bearing.** The skips are the boundaries recorded in the track's `track.json`. Without them the compass falls back to 128 and 320, warns, and still writes a file that looks like the published one.
- **The 0.6616 row is a null.** The interval sits astride the length-only control at 0.6564, so the instrument does not separate the two groups on the smaller model. The file records that the model answered `HARMFUL` to all 4,504 harmful prompts and all 4,504 harmless ones.

## Multi-direction result

Quoted in `paper.md`.

| Claim | Artefact |
|---|---|
| Five seeds, one direction against two: no refusal gain this design could detect, at about 1.9 times the divergence on the five-seed mean (95% interval 1.2 to 3.1, percentile bootstrap over the five seeds a side) | `evidence/k-sweep-2026-08-13/drift-per-seed.json` |
| Both budgets at matched hard refusal, 0.1% against 0.3% | `evidence/k-sweep-2026-08-13/drift-per-seed.json`, `hard_refusal` |

The file holds per-seed divergence and refusal values and no prompts. The refusal values were recovered from the run's database on 2026-09-25, because they had no artefact before then.

The ratio depends on one seed and on the estimator. The mean over all five seeds is 1.8757. Dropping the one outlying two-direction seed gives 1.4472. An earlier version of this file quoted 1.5, which came from a median. The figures here are means.

**Read it with its limit attached.** These arms ran on 2026-08-13, before the held-out direction score existed. Their second directions were chosen by a filter that accepted every candidate it was shown, so they measure that *arbitrary* second directions cost drift. That is a weaker claim than the thesis the table is read against. The artefact's own `caveat` field says so. A re-run under the fixed selector has not been done, so this is a known limit, not a settled answer.

```sh
python3 -c "
import json
d = json.load(open('evidence/k-sweep-2026-08-13/drift-per-seed.json'))
print(d['seeds'])
print(d['drift_kl'])
print(d['hard_refusal'])
print(d['caveat'])
"
```

## Head-to-head

Thirty artefacts under `head-to-head/results/`: ten arms, measured by three instruments against another tool. They stay in the repository and `senbonzakura check` still reads them. The command that produced them was removed from the package on 2026-10-18, so they can be checked here but not re-run with it.

## Checking results

The companion checker reads result files, from this tool or from others. It reports how a figure could be wrong: a rate on too small a sample, a proportion outside zero to one, or a figure reported on the rows it was selected on. It needs no model, corpus or network.

```sh
senbonzakura check head-to-head/results/ --min-applied 2
senbonzakura check evidence/
```

`--min-applied` matters. A file counts as checked when one check applied to it, so a run where almost nothing applied looks clean. CI asks for at least 2 per artefact. The checker also reads result files from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [Inspect](https://inspect.aisi.org.uk).

## What a result file records

Each result file carries a `provenance` block: tool version and commit, Python version, platform and package versions. It is not complete. Checked on 2026-10-10 against `head-to-head/results/2026-09-10/`:

- Python is recorded as a version (`3.12.3`), not as an interpreter path. Record `sys.executable` yourself when you reproduce a run.
- The corpus revision and the model's Hub revision are not recorded. The label `senbon-seed42` is a name, not an identity.
- The drift files and the summary carry no environment at all. Do not assume they came from the same environment as the scored file beside them.

## What you cannot reproduce from this repository

**The corpus.** The harmful prompt set is not here. The evaluation track is a gated dataset. [The track guide](docs/guide/the-track.md) explains the split and how to build an equivalent from public sources. This limit is deliberate: a fully reproducible harmful corpus in a public repository would be a harmful corpus in a public repository.

**Anything above 3B parameters.** No measurement above 3B stands. Qwen3-4B was run in July 2026, and those figures are withdrawn. The scorer had thinking mode on, so it scored replies the search had not picked its winner on. Nothing above 3B has been re-run, because the GPU this project owns cannot hold one. The instruments also weaken as the model shrinks, as the compass table shows.

**The abliterated weights.** No abliterated checkpoint is published with this software.

## If a number does not reproduce

Tell us. A figure that does not reproduce is the most useful issue this project can receive, and several published numbers have been withdrawn after exactly that. Include the command, the artefact you got and the output of `senbonzakura doctor`. [What is and is not established](https://elementmerc.github.io/senbonzakura/guide/what-we-know) is the running record of which claims have survived.
