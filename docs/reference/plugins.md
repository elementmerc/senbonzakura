# Writing a plugin

An **instrument** answers one question about one model and returns one result. `score` answers "how often does it still refuse". A plugin answers whatever you wrote it to answer.

This page is the contract. Everything here is enforced by the tool. `senbonzakura.plugins.protocol_document()` returns the same values in machine-readable form. If this page and the tool disagree, the page is wrong.

## The shape: one request in, one result out

One request goes in and one result comes back. Nothing is said in between: no per-weight messages, no callbacks, no shared handle to a loaded model.

```
in process                        out of process
----------                        --------------
us  ->  run(inputs)               us  ->  one JSON object on stdin
us  <-  one reply mapping         us  <-  one JSON object on stdout
```

Both paths go through the same validator. A plugin in the same process cannot return a shape that a separate process would have been refused for.

## The protocol version

`senbonzakura-plugin/1`. Every request and reply carries it. The tool refuses any other version rather than guessing. The number changes only when a field is renamed, removed, or given a new meaning. A new optional field does not change it.

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

`inputs` is yours. The tool passes it through without modelling it, so your own documentation is its schema.

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

- `ok` must be `true` or `false`, and nothing else. The string `"false"` is truthy in most languages, so it would read as success. The tool refuses it.
- A failure without a `reason` is refused.
- `result` must be an object. A success with nothing in it is worse than a refusal, because it reads as a measurement.

## A manifest

A `describe` request is answered with this, and nothing else:

```json
{
  "protocol": "senbonzakura-plugin/1",
  "name": "my-metric",
  "kind": "instrument",
  "title": "one line a person reads in a list of plugins",
  "produces": ["my_metric"]
}
```

- `name` must match the name you registered under.
- `kind` is `instrument`. It is a closed set, so a plugin built for a later contract is refused with a message.
- `produces` lists the metric names your result stamps. It is a list even for one name, because a bare string would be read one character at a time.
- An unknown key in a manifest is refused.

## How a plugin is found

Register an ordinary Python distribution under the entry point group `senbonzakura.plugins`:

```toml
[project.entry-points."senbonzakura.plugins"]
my-metric = "my_package"
```

The object declares its transport by what it exposes:

| Exposes | How it is run | Use it when |
|---|---|---|
| `run(inputs)` and `describe()` | Called directly, in this process | The code is already trusted here |
| `argv()` and `describe()` | Started as a separate process, one request on stdin | Anything else |

The transport is a trust decision, not a capability tier. In-process code is code you already trust. It is not allowed to reach anything a separate process cannot.

Two plugins under one name are refused, because which one produced a number would depend on install order. A plugin that cannot be imported is refused loudly, so a missing instrument never looks like one that was never asked for.

## This project's own instruments, served the same way

Every instrument here is reachable through the same contract:

```bash
echo '{"protocol":"senbonzakura-plugin/1","call":"describe"}' \
  | python -m senbonzakura.pluginhost gate
```

The inputs are the command line the instrument would take by hand, plus the path of the file it writes:

```json
{
  "argv": ["--baseline", "before.json", "--measurement", "after.json"],
  "out": "run/gate.json"
}
```

The result carries the instrument's exit status and the document it wrote, or `null` if it wrote none. The out-of-process path is exercised by this project too. Otherwise it would break for outside users first.

## The limits

Each limit is enforced. `protocol_document()["limits"]` returns them, so a plugin can read them rather than assume them.

| Limit | Value | Why |
|---|---|---|
| Reply size | 8 MiB | A result is a handful of numbers and a stamp. A bigger one means the plugin has gone wrong |
| Default deadline | 3600 s | An instrument may hold a model and generate for a long time |
| Longest deadline | 86400 s | An unbounded wait set in a config file is still an unbounded wait |
| Name length | 64 characters | A name reaches log lines, filenames and metric vocabularies |
| Name characters | letters, digits, `-`, `_`, `.` | A path separator or a newline is refused where the name enters, not stripped later |
| Metrics declared | 256 | The list is held in memory, and no outside source decides how long it is |

A plugin that passes its deadline is asked to stop, given 10 seconds, and then killed. Nothing it may have written is read, because a half-written reply cannot be told from a complete one that happens to be short.

## What is not checked

The tool does not enforce who may run what. Python resolves imports while the program is running, so a resolver that claimed otherwise would be theatre. An entry point is a request to import your code. A plugin is trusted exactly as far as everything else installed in the same environment, and no further.

What **is** checked is what a plugin returns, at the boundary. That is the half that can be enforced.

## Logging

Write logs to stderr. Out of process, stdout carries the reply, so a line printed there lands in front of the JSON and the whole exchange is refused. Anything on stderr is copied into the error message when something goes wrong.
