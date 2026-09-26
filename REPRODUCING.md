# Reproducing the published numbers

What a published figure came from, and what you would run to produce that file yourself.

The point of this file is narrow and worth stating. A claim that traces to a paragraph is not
checkable. A claim that traces to a committed artefact can be read without trusting us, and a
claim that traces to a command can be re-taken on your own hardware and disagreed with.

The figures this project quotes fall into three groups, and naming them is more honest than
implying there is only one.

1. **It traces to a committed artefact.** That is the table below: the compass AUCs, the
   per-seed coherence drift, and the head-to-head. Every one of these can be read out of a
   file in this repository with no hardware and no network, and re-taken with hardware.
2. **It traces to a run whose logs are not committed.** The two that carry weight are the
   evidence for the Gemma withdrawal (`docs/guide/limits.md`) and the count of rejected
   candidate directions in
   [what is and is not established](https://elementmerc.github.io/senbonzakura/guide/what-we-know).
   Both pages state what the number is and that you cannot currently check it. That is worth
   less than an artefact and it is said out loud where the number is quoted.
3. **Neither.** If a number is quoted anywhere in the documentation, is not in the table below,
   and has nothing beside it saying where it came from, that is a defect: please open an issue.
   That rule is why group 2 is written down rather than left to look like group 1.

## Before anything: what you need

```sh
pip install "git+https://github.com/elementmerc/senbonzakura@dev#subdirectory=checker" \
            "git+https://github.com/elementmerc/senbonzakura@dev"
```

**`@dev` is load-bearing and so is putting both URLs in one command.** Without `@dev`, pip takes
the default branch, which is `main`, and `main` is a long way behind: it has no `checker/`
directory at all, so the first URL fails with "does not appear to be a Python project", and the
second installs something close to the withdrawn 0.3.0. Given as two separate commands, the big
package goes looking for the small one on PyPI, where it is not published, and the install fails
for a second reason that reads exactly like the first.

This file carried the two-command form without `@dev` until 2026-09-25, which is both documented
failures at once, in the one document whose whole job is letting a stranger re-take the numbers.

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
| `head-to-head/results/` | Thirty artefacts: ten arms, three instruments against another tool, on one corpus and one budget |

Result files have the `prompt` and `generation` fields stripped. That is not tidying: this
repository does not carry model generations or harmful prompts, and `CONTRIBUTING.md` explains
why that is a firm rule.

## Claim to artefact to command

### The compass figures, quoted in the README and in `paper.md`

| Claim | Value | Field | Artefact |
|---|---|---|---|
| Qwen3-1.7B harm-recognition AUC | 0.9887 | `auc` | `evidence/compass-2026-07-30/base-qwen3-1.7b.json` |
| Length-only control, both models | 0.6564 | `controls.length_only_auc` | the same two files |
| Qwen3-0.6B harm-recognition AUC | 0.6616 | `auc` | `evidence/compass-2026-07-30/base-qwen3-0.6b.json` |

The control is recorded at full precision, `0.6563985752549933`, and quoted rounded to four
places to sit beside the two AUCs, which the files record rounded already.

Read all three without running anything:

```sh
python -c "
import json
for name in ('base-qwen3-1.7b', 'base-qwen3-0.6b'):
    d = json.load(open('evidence/compass-2026-07-30/' + name + '.json'))
    print(name, d['auc'], d['auc_ci'], round(d['controls']['length_only_auc'], 4))
"
```

Re-take them. This needs a GPU and the corpus, and the split is the part that matters. The
recipe lives beside the artefacts, in
[`evidence/compass-2026-07-30/README.md`](evidence/compass-2026-07-30/README.md), and it is
carried here verbatim so there is one command rather than two that can disagree:

```sh
python -m senbonzakura.margin --model Qwen/Qwen3-1.7B \
    --harmful <track>/bad_eval_ds --harmless <track>/good_ds \
    --harmless-matched <matched>/good_matched_ds \
    --skip-harmful 132 --skip-harmless 385 --n 4504 \
    --batch 16 --bootstrap 2000 --seed 42 --device cuda \
    --out base-qwen3-1.7b.json
```

**Every one of those flags is load-bearing.** The skips are the boundaries recorded in the
track's `track.json`, and a compass run given none of them falls back to the older defaults of
128 and 320, warns that the number may include rows the search selected on, and still writes a
file that looks like the published one. `<track>` is what `senbonzakura track` wrote.

**The second row is the one to look at**, and it is the reason this project believes what it says
about controls. 0.6616 against a length-only control at 0.6564 is not a weak result, it is a null:
the interval sits astride the control. The same file records that the model answered `HARMFUL` to
all 4,504 harmful prompts and all 4,504 harmless ones.

### The multi-direction result, quoted in `paper.md`

| Claim | Artefact |
|---|---|
| Five seeds, one direction against two, no measurable refusal gain at 1.4 to 1.9 times the divergence, on the mean over five seeds, the low end dropping the one outlying two-direction seed | `evidence/k-sweep-2026-08-13/drift-per-seed.json` |
| Both budgets at matched hard refusal, 0.1% against 0.3% | `evidence/k-sweep-2026-08-13/drift-per-seed.json`, `hard_refusal` |

That file exists because the p-value this project was quoted on most often traced to a working
note nobody outside could open. It is twenty floats and no prompts: ten of divergence, and ten of
refusal recovered from the run's own database on 2026-09-25, because the 0.1% and 0.3% were the
premise of the whole comparison and had no artefact behind them until then.

**Which end of that range you get depends on one seed, and on the estimator.** The mean over all
five seeds is 1.8757. Dropping the one outlying two-direction seed gives 1.4472. This file put
the low end at 1.5 until 2026-09-25, and that came from an unstated switch to medians (1.5073)
with 1.45 rounded up, which widened the project's own negative result by a tenth. The estimator
is named here so nobody has to work out which one produced the number.

**Read that result with its limit attached.** Those arms ran on 2026-08-13, before the held-out
direction score existed, so their second directions were chosen by a filter since found to accept
every candidate it was shown. What they measure is that *arbitrary* second directions cost drift,
which is a weaker claim than the thesis the table gets read against. The artefact says so in its
own `caveat` field. A re-run under the fixed selector is scheduled, and until it lands this is a
known limit rather than a settled answer.

```sh
python -c "
import json
d = json.load(open('evidence/k-sweep-2026-08-13/drift-per-seed.json'))
print(d['seeds'])
print(d['drift_kl'])
print(d['hard_refusal'])
print(d['caveat'])
"
```

### The head-to-head, quoted in `docs/guide/benchmark.md`

Thirty artefacts, which is ten arms measured by three instruments, committed under
`head-to-head/results/`. The whole comparison runs as one command, and it takes hardware and
hours:

```sh
senbonzakura head-to-head --help
```

The command is `head-to-head`, not `bench`. This file said `bench` until 2026-09-25, and the
parser does not refuse an unknown first word: it takes it as the model to edit and prints the
abliterator's help, exiting 0. So a reader checking the benchmark got a plausible screen with
nothing about the benchmark on it, and the same wrong word without `--help` would have gone
looking for a model of that name on the Hub and started editing its weights.

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

**Anything above 3B parameters.** No measurement above 3B still stands. Qwen3-4B WAS run, in
July 2026, and those figures are withdrawn: the scorer had thinking mode switched on, so the
replies being scored were not the replies the search had picked its winner on. Nothing has been
re-run above 3B since, because the card this project owns cannot hold one. The instruments also weaken as the model
shrinks, which the compass row above shows better than any sentence could.

**The abliterated weights.** No abliterated checkpoint is published with this software.

## If a number does not reproduce

Tell us. A figure that does not reproduce is the most useful issue this project can receive, and
it has happened before: several published numbers have been withdrawn after exactly that. Include
the command, the artefact you got, and `senbonzakura doctor` output.
[What is and is not established](https://elementmerc.github.io/senbonzakura/guide/what-we-know)
is the running record of which claims survived.
