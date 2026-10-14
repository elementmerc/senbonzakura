# Writing a plugin

An **instrument** answers one question about one model and hands back one result. `score` answers
"how often does it still refuse". `coherence` answers "how surprised is it by ordinary prose". A
plugin you write answers whatever you wrote it to answer.

This page is the whole contract. Everything in it is a fact the tool enforces, and
`senbonzakura.plugins.protocol_document()` returns the same values in machine-readable form, so a
disagreement between this page and the tool is this page's fault.

## The shape: one request in, one result out

One request goes in, one result comes back, and nothing is said in between. No per-weight chatter,
no callbacks, no shared handle to a loaded model.

The coarseness is the point rather than a simplification. A plugin exchanging thousands of small
messages about individual tensors would be a separate process on paper and one program in
practice, and "these are two programs" is a claim somebody should be able to check by reading the
traffic. A single request and a single reply is a claim anybody can check.

```
in process                        out of process
----------                        --------------
us  ->  run(inputs)               us  ->  one JSON object on stdin
us  <-  one reply mapping         us  <-  one JSON object on stdout
```

Both go through the same validator. A plugin in the same process cannot hand back a shape a
separate process would have been refused for.

## The protocol version

`senbonzakura-plugin/1`. Every request and every reply carries it, and the tool refuses anything
that names a different one rather than reading it anyway. The number is bumped only when a field
is renamed, removed, or given a different meaning; a new optional field does not bump it. So
refusing is the right answer: carrying on would mean reading one of those fields wrongly with
nothing to say so.

## A request

```json
{
  "protocol": "senbonzakura-plugin/1",
  "call": "run",
  "instrument": "my-metric",
  "inputs": { "model": "example/synthetic-7b", "prompts": ["..."] },
  "deadline": 3600.0
}
```

`inputs` is yours. The tool passes it through without modelling it, because what your instrument
needs is your business, and a schema here would have to grow a field every time somebody wrote a
plugin asking a question nobody had thought of. Your own documentation is the schema for it.

`call` is `run` or `describe`. A `describe` request carries no inputs and no deadline.

## A reply, two shapes

Success:

```json
{
  "protocol": "senbonzakura-plugin/1",
  "instrument": "my-metric",
  "ok": true,
  "result": { "metrics": { "my_metric": 0.42 } }
}
```

Could not produce a number:

```json
{
  "protocol": "senbonzakura-plugin/1",
  "instrument": "my-metric",
  "ok": false,
  "reason": "the card has 6 GB and this model needs 17"
}
```

`ok` has to be `true` or `false` and nothing else. The string `"false"` is truthy in most
languages, so a reply carrying it would read as a success, which is how a failed run gets quoted
as a result. The tool refuses it.

A failure without a `reason` is also refused. A failure nobody can act on is worse than a crash,
because a crash at least says where it happened.

`result` has to be an object. Success with nothing in it is the one outcome worse than a refusal:
it reads as a measurement.

## A manifest

A `describe` request is answered with this, and with nothing else in it:

```json
{
  "protocol": "senbonzakura-plugin/1",
  "name": "my-metric",
  "kind": "instrument",
  "title": "one line a person reads in a list of plugins",
  "produces": ["my_metric"]
}
```

`name` has to match the name you registered under. A result filed under a name nobody can look up
is a result nobody can re-run.

`kind` is `instrument`. It is a closed set, so a plugin built against a later contract is refused
with a message rather than driven as something it is not.

`produces` lists the metric names your result stamps. It is a list even when there is one of them:
a bare string would be read one character at a time.

An unknown key in a manifest is refused rather than ignored, because a key nobody reads is
something you believe you declared.

## How a plugin is found

An ordinary Python distribution that registers itself under the entry point group
`senbonzakura.plugins`:

```toml
[project.entry-points."senbonzakura.plugins"]
my-metric = "my_package"
```

The object it points at declares its own transport by what it exposes:

| Exposes | How it is run | Use it when |
|---|---|---|
| `run(inputs)` and `describe()` | Called directly, in this process | The code is already trusted here |
| `argv()` and `describe()` | Started as a separate process, one request on stdin | Anything else |

**The transport is a trust decision and not a capability tier.** In process means code that is
already trusted, so the serialisation is skipped and a traceback stays a traceback. It does not
mean code allowed to reach things a separate process cannot. If the only tier that could reach a
model were the one this project ships, nobody outside could build anything worth building.

Two plugins registered under one name are refused rather than ordered, because which one produced
a number would then depend on install order. A plugin that cannot be imported is refused loudly:
a run that quietly left an instrument out would look exactly like a run that had never been asked
for it.

## Running one of this project's own instruments as a plugin

Every instrument here is reachable through the same contract, served at arm's length:

```bash
echo '{"protocol":"senbonzakura-plugin/1","call":"describe"}' \
  | python -m senbonzakura.pluginhost gate
```

Its inputs are the command line the instrument would have been given by hand, plus the path of
the file it writes:

```json
{
  "argv": ["--baseline", "before.json", "--measurement", "after.json"],
  "out": "run/gate.json"
}
```

and the result carries the instrument's exit status and the document it wrote, or `null` where it
wrote none.

This is not a convenience. The out-of-process path has to be travelled by this project too, or it
would be exercised only by people outside it, which means it would break for them first and we
would hear about it from them.

## The limits

Every one of these is enforced, and `protocol_document()["limits"]` returns them so a plugin can
read them rather than assume them.

| Limit | Value | Why |
|---|---|---|
| Reply size | 8 MiB | A result document is a handful of numbers and a stamp. A gigabyte of it is a plugin that has gone wrong |
| Default deadline | 3600 s | An instrument may hold a model and generate for a long time |
| Longest deadline | 86400 s | An effectively unbounded wait reached through a config file is still an unbounded wait |
| Name length | 64 characters | A name reaches log lines, filenames and metric vocabularies |
| Name characters | letters, digits, `-`, `_`, `.` | A path separator or a newline in one is refused where it enters, rather than stripped at every use |
| Metrics declared | 256 | This list is held in memory here, and nothing else lets an outside source decide how long one of our lists is |

A plugin that passes its deadline is asked to stop, given 10 seconds, and then killed. Nothing it
may have written is read, because a half-written reply cannot be told apart from a complete one
that happens to be short.

## What is not checked, and will not be

Nothing here enforces who may run what. Python resolves imports while the program is already
running, and anything one module can check another can satisfy, so a resolver pretending otherwise
would be theatre. An entry point is a request to import your code, which means a plugin is trusted
exactly as much as everything else installed in the same environment and no further.

What **is** checked is what a plugin hands back, at the boundary, and that is the half that can be
made to hold.

## Logging

Write it to stderr. Out of process, stdout carries the reply, so a line printed there lands in
front of the JSON and the whole exchange is refused as unparseable. Anything on stderr is carried
into the error message when something goes wrong, which is what makes a broken plugin
diagnosable.
