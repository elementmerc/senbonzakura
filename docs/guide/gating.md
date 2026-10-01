# Failing a build on a regression

`senbonzakura baseline` writes down what a measurement was. `senbonzakura gate` compares a later
measurement against it and exits non-zero when the property moved. Together they are how you stop
a change quietly making a model worse.

Neither command needs a GPU, a model or a network. They read JSON and write JSON.

## The picture first

Think of a baseline as a photograph with the camera settings written on the back. A gate is
somebody holding a new photograph next to it and saying whether the thing in the picture moved, or
whether you just changed the lens.

```text
  a run  ──►  result.json  ──►  senbonzakura baseline  ──►  baselines/coherence.json
                                                                      │
  a later run  ──►  result.json  ──►  senbonzakura baseline  ──►  later.json
                                                                      │
                                         senbonzakura gate ◄──────────┘
                                                  │
                                       exit 0  or  exit 1
```

The second `baseline` call is the step people skip. The gate compares two **recorded**
measurements, never a recorded one against a raw result artefact, because a raw artefact does not
say how many seeds the figure rests on and the gate will not guess.

## A worked example you can run

Everything below was run to produce the output shown. No model is involved; the inputs are result
artefacts carrying a stamped `metrics` block, which every scoring command in this tool writes.

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

`--seeds` is stated by you and never inferred. One artefact is one run, and a baseline claiming a
spread it does not have is worse than no baseline: the gate would then accept a change that moved
the property, because it thinks the interval is wider than it is.

If the artefact stamped more than one metric, the command refuses and names every candidate, so
pass `--metric coherence` (or `--metric refusal_rate.senbonzakura-ruler`) to choose.

The output file is never overwritten. A new baseline is a new file, so a baseline cannot be edited
into agreeing with the run it was supposed to judge.

### 2. Record the new run the same way, then gate it

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

Exit status 0.

A run that did:

```
gate FAIL: coherence: REGRESSED
  baseline 3.0100 [2.95, 3.07] on n=1, this run 3.4400 [3.3800, 3.5000]. The intervals do not
  overlap and the value moved above it by 0.4300, which is the worse direction for a metric where
  lower is better.
```

Exit status 1, which is what makes it a gate. In a CI step, run both commands bare and let the
step fail on any non-zero status:

```yaml
- name: The edit did not cost coherence
  run: |
    senbonzakura baseline result.json --seeds 42,43,44,45,46 --out later.json
    senbonzakura gate --baseline baselines/coherence.json --current later.json
```

That is correct and it is conservative: it goes red on a regression and red on a refusal, which is
the right way round for both. Read the status yourself only when you want to tell them apart, and
then read the actual number rather than "did it fail":

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

Both of these look right and are wrong, and they are wrong in the same way: **they ask "did it
fail" of a command that answers three things.**

```sh
senbonzakura gate --baseline b.json --current c.json || echo "coherence regressed"
```

A refused comparison prints "coherence regressed". Nothing regressed. Nothing was measured
against anything, and somebody is now looking for a change in the model instead of a mismatch in
their baseline.

```sh
if senbonzakura gate --baseline b.json --current c.json; then ship; fi
```

A refused comparison does not ship, which sounds safe, and says nothing about why. The build is
red for a reason nobody is told, so the fix people reach for is re-recording the baseline until it
goes green, which is the one repair that makes the gate meaningless.

The `case` block above is longer than either line and it is the shortest version that cannot lie.

## Why it compares intervals and not numbers

Two measurements of the same model differ run to run. A gate that fired whenever the number moved
would fire constantly and get switched off, which is worse than having no gate: a disabled check
reads as a passing one on the dashboard.

So the comparison is between intervals. Overlapping intervals mean there is no evidence the
property moved, and the point estimate's drift is reported but is not the verdict. The direction
matters too: `coherence` is measured in nats per token, where lower is better, so moving **up**
outside the interval is the regression and moving down is not.

## What the gate refuses to do

It refuses to compare two measurements that are not comparable, and it says which field disagreed.
A baseline carries the conditions it was taken under, and all of these have to match:

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

What a refusal looks like:

```
gate REFUSED: this measurement is not comparable to that baseline, so it was not compared:
  model: baseline 'Qwen/Qwen3-1.7B', this run 'Qwen/Qwen3-0.6B'  (the weights the measurement
  was taken on)

Either measure under the conditions the baseline records, or record a new baseline and say in
its filename what changed. Comparing across a mismatch is how a drifted prompt renderer got a
published table and a passing build.
```

A refusal here is the gate working. The alternative is a number that looks like a comparison and
is not one, which is the single defect this whole tool exists to find.

## The three exit codes

A CI step that branches on "non-zero" is treating two different answers as one. They are not the
same, and the difference is worth wiring up: a regression is a finding about the model, and a
refusal is a finding about your pipeline.

| Status | Means | What to do |
|--:|---|---|
| 0 | `gate OK` | Nothing. The property did not move outside its interval |
| 1 | `gate FAIL` | A real regression. The change moved the property in the worse direction |
| 2 | `gate REFUSED` | The two measurements are not comparable. Nothing was compared, so this is **not** a pass and **not** a regression |

Status 2 is the one to be careful with. A step written as `gate || echo "regression"` reports a
mismatched baseline as a regression, and a step written as `if gate; then ok; fi` treats it as a
failure without saying the comparison never happened.

## Recording the baseline itself

`senbonzakura baseline` refuses rather than guessing, in two cases worth knowing before they
interrupt a build. Both exit 2, the same status `gate` uses for "I did not do the comparison".

An artefact with more than one stamped metric:

```
refused:
  this artefact stamped 2 metrics, so --metric has to say which one: coherence,
  refusal_rate. Guessing between them would gate a later run on a property
  nobody chose.
```

An output path that already holds a baseline:

```
refused:
  baselines/coherence.json
  already exists, and baselines are never overwritten in place. A new baseline
  is a new file: name it for what changed, and keep the old one so the
  ...
```

## Keeping the history, so March is still evidence in June

By default the gate reads two files, decides, and writes nothing. Pass `--history` and every run
leaves a record behind, pass or fail.

```sh
senbonzakura gate --baseline baselines/coherence.json --current later.json \
  --history gate-history/
```

Each record holds what was compared, what the verdict was, and the whole measurement rather than
just its number, so somebody disputing the verdict later can re-judge it from the conditions it
was taken under. Records are named after their own contents, which means two things worth
knowing. A re-run of the same comparison records it once, so a job killed halfway is safe to
re-run. And nothing is ever overwritten: a different comparison is a different name.

If the directory cannot be written to, the gate says so and the verdict stands. Whether your disk
is full is a fact about your disk, and a full disk must not turn a regression into a pass or a
pass into a regression.

## Fetching a baseline instead of re-measuring it

Every recorded measurement carries the digest of its own contents, printed as
`sha256:` followed by sixty-four characters. That is its address, and two measurements that say
the same thing share one.

```sh
senbonzakura gate --store baselines/ \
  --baseline sha256:8392ca6c81f978b6...  --current out/this-run.json
```

With `--store` pointing at a directory, either file may be given as an address rather than a
path, and the gate looks it up. The point is cost: a team gating several models a day
re-measures a baseline it already holds unless there is a name under which the old one can be
found, and re-measuring is what makes a gate too expensive to run on every change.

The address is not the same thing as `input_digest`, which says which input the number was taken
on. A thousand measurements of one corpus share an `input_digest` and have a thousand addresses.

One useful side effect: because the address covers the contents, a recorded measurement that has
been edited since no longer matches its own stamp, and the gate refuses it rather than comparing
against a number nobody measured.

## What a pass does not mean

Printed by the command itself, every time, and worth repeating here:

> A pass means one measured property did not move outside its interval, on one track, under the
> conditions the baseline records. It is not a statement that the model is safe, that other
> properties held, or that the track represents anything beyond itself.

A green gate on coherence says nothing about refusal, and a green gate on refusal says nothing
about capability. Gate the properties you care about, each with its own baseline file.

That sentence on its own is not much use, because "one property, on one track" does not say
which. So the command prints the slice beside it, every run:

```
  Checked: coherence on the fixed-passage partition, n=1, seeds [42],
           estimator 'senbonzakura-nll' at bfloat16.
  Not checked: every other property, every other partition, and anything this
           metric does not measure.
```

A reader can then tell a gate over one passage from a gate over a whole corpus without going to
look for the baseline file.

## Where next

- [The compass](/guide/compass), for the measurement whose figure is easiest to quote wrongly.
- [What is and is not established](/guide/what-we-know), before quoting any of these numbers.
- [The full CLI reference](/reference/cli).
