# Failing a build on a regression

`senbonzakura baseline` writes down what a measurement was. `senbonzakura gate` compares a later measurement against it and
exits non-zero when the property moved. Together they stop a change from quietly making a model worse.

Neither command needs a GPU, a model or a network. They read JSON and write JSON.

## A worked example you can run

Everything below was run to produce the output shown. No model is involved. The inputs are result artefacts with a stamped
`metrics` block, which every scoring command here writes.

### 1. Record the baseline

```sh
senbonzakura baseline before.json --seeds 42,43,44,45,46 --out baselines/coherence.json
```

```
--metric was left out and this artefact stamped only 'coherence', so that is what is being recorded
baseline written to baselines/coherence.json: coherence at 3.0100 nats-per-token
(interval [2.95, 3.07]) on n=1, seeds [42, 43, 44, 45, 46]
  gate a later run with, once you have recorded it the same way:
    senbonzakura baseline <the later run's result> --seeds <n> --out later.json
    senbonzakura gate --baseline baselines/coherence.json --current later.json
```

`--seeds` is stated by you and never inferred. A baseline that claims a spread it doesn't have would let the gate accept a change
that moved the property.

If the artefact stamped more than one metric, the command refuses and names them all. Pass `--metric coherence` (or
`--metric refusal_rate.senbonzakura-ruler`) to choose. The output file is never overwritten. A new baseline is a new file.

`measure` writes an index, `measure.json`, not a measurement, so `baseline` has nothing to record from it. Point `baseline` at one
of the instrument files beside it.

### 2. Record the new run, then gate it

```sh
senbonzakura baseline after.json --seeds 42,43,44,45,46 --out later.json
senbonzakura gate --baseline baselines/coherence.json --current later.json
```

A run that did not move the property:

```
gate OK: coherence: within interval
  baseline 3.0100 [2.95, 3.07] on n=1, this run 3.0300 [2.9700, 3.0900]. The intervals overlap,
  so there is no evidence the property moved. Moved +0.0200 on the point estimate, which on its
  own is not a finding.
```

A run that did:

```
gate FAIL: coherence: REGRESSED
  baseline 3.0100 [2.95, 3.07] on n=1, this run 3.4400 [3.3800, 3.5000]. The intervals do not
  overlap and the value moved above it by 0.4300, which is the worse direction for a metric where
  lower is better.
```

In CI, run both commands bare and let the step fail on any non-zero status:

```yaml
- name: The edit did not cost coherence
  run: |
    senbonzakura baseline result.json --seeds 42,43,44,45,46 --out later.json
    senbonzakura gate --baseline baselines/coherence.json --current later.json
```

That's conservative in the right way: it goes red on a regression and on a refusal. To tell them apart, read the status:

```sh
set +e
senbonzakura gate --baseline baselines/coherence.json --current later.json
status=$?
set -e
case "$status" in
  0) echo "coherence held" ;;
  1) echo "coherence REGRESSED"; exit 1 ;;
  2) echo "the comparison never happened: fix the baseline, do not read this as a result"; exit 1 ;;
  *) echo "gate exited $status, which this script does not model"; exit 1 ;;
esac
```

### Two lines not to write

Both look right and are wrong, for the same reason: **they ask "did it fail" of a command that answers three things.**

```sh
senbonzakura gate --baseline b.json --current c.json || echo "coherence regressed"
```

A refused comparison prints "coherence regressed". Nothing regressed. Nothing was compared, and somebody is now hunting for a
change in the model instead of a mismatch in the baseline.

```sh
if senbonzakura gate --baseline b.json --current c.json; then ship; fi
```

A refused comparison doesn't ship, and says nothing about why. The usual fix is re-recording the baseline until it goes green,
which makes the gate meaningless.

The `case` block above is longer than either line, and it's the shortest version that can't lie.

## The three exit codes

A CI step that branches on "non-zero" treats two different answers as one. A regression is a finding about the model. A refusal is
a finding about your pipeline.

| Status | Means | What to do |
|--:|---|---|
| 0 | `gate OK` | Nothing. The property did not move outside its interval |
| 1 | `gate FAIL` | A real regression. The change moved the property in the worse direction |
| 2 | `gate REFUSED` | The two measurements are not comparable. Nothing was compared, so this is **not** a pass and **not** a regression |

`baseline` also exits 2 when it refuses to record: an artefact with more than one stamped metric, or an output path that already
holds a baseline.

## Why it compares intervals and not numbers

Two measurements of the same model differ run to run. A gate that fired on every movement would get switched off, and a disabled
check looks like a passing one on the dashboard. So overlapping intervals mean no evidence the property moved. The direction matters
too: `coherence` is in nats per token, where lower is better, so moving **up** outside the interval is the regression.

## What the gate refuses to compare

A baseline records the conditions it was taken under. A comparison is refused unless all of these match, and the refusal names the
field that disagreed:

| Field | Why a difference makes the comparison meaningless |
|---|---|
| `model` | A figure on one set of weights says nothing about another |
| `metric` | Two properties, by identity rather than by display name |
| `estimator` | Two estimators of one metric are two different numbers |
| `input_digest` | Which corpus or passage the figure was taken on |
| `partition` | So a measure-partition figure is never judged against a search-partition one |
| `prompt_format` | Three copies of the prompt renderer drifted once, and a table was published across the gap |
| `precision` | A 4-bit reading and a bfloat16 reading of the same model are different measurements |
| `tool_version` | The edit and the scorer both live in this package |
| `seeds` | The seeds a sampled figure was drawn over. A method that draws no samples says so (`no-seed`), which is different from a missing field |

What a refusal looks like:

```
gate REFUSED: this measurement is not comparable to that baseline, so it was not compared:
  model: baseline 'Qwen/Qwen3-1.7B', this run 'Qwen/Qwen3-0.6B'  (the weights the measurement
  was taken on)

Either measure under the conditions the baseline records, or record a new baseline and say in
its filename what changed. Comparing across a mismatch is how a drifted prompt renderer got a
published table and a passing build.
```

A refusal is the gate working.

## Keeping the history

Pass `--history` and every run leaves a record, pass or fail:

```sh
senbonzakura gate --baseline baselines/coherence.json --current later.json \
  --history gate-history/
```

Each record holds the comparison, the verdict and the whole measurement, so a disputed verdict can be re-judged. Records are named
after their contents, so a re-run records once and nothing is overwritten. If the directory can't be written, the verdict still
stands. A full disk must not turn a regression into a pass.

## Addresses instead of files

Each recorded measurement carries a `sha256:` digest of its contents. With `--store` pointing at a directory, either file can be given
as an address:

```sh
senbonzakura gate --store baselines/ \
  --baseline sha256:8392ca6c81f978b6...  --current out/this-run.json
```

This saves re-measuring a baseline you already hold. An edited measurement no longer matches its own digest, so the gate refuses it.

## What a pass does not mean

The command prints this every time:

> A pass means one measured property did not move outside its interval, on one track, under the
> conditions the baseline records. It is not a statement that the model is safe, that other
> properties held, or that the track represents anything beyond itself.

A green gate on coherence says nothing about refusal, and the reverse holds too. Gate each property you care about with its own baseline.
The command also prints the slice it checked:

```
  Checked: coherence on the fixed-passage partition, n=1, seeds [42],
           estimator 'senbonzakura-nll' at bfloat16.
  Not checked: every other property, every other partition, and anything this
           metric does not measure.
```

## Where next

- [The compass](/guide/compass), for the measurement whose figure is easiest to quote wrongly.
- [What is and is not established](/guide/what-we-know), before quoting any of these numbers.
- [The full CLI reference](/reference/cli).
