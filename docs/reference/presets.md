# Tuned presets, and the pack that supplies them

A run searches for its own settings. It tries layer bands, direction counts and ablation
strengths, scores each trial, and keeps the best one. That search is why a run takes hours on a
card, and it is also why the result is trustworthy: the numbers came from measuring this model
rather than from a table somebody wrote down.

A **preset** is the settings a previous measured run already found for one model, written down so
the next run can start from them. There is nothing else to it. A preset is a shortcut past the
search, not a different method and not a better one.

## What the open tool does

Everything. The search is in the package you installed, every instrument that scores it is in the
package you installed, and no part of either is held back. If no preset is installed, a run looks
for its own settings, which is the path every published number from this project was measured on.

That is worth saying plainly because the usual shape of this arrangement is the opposite: a free
tier that exists to be inadequate. Measuring is the whole product here, and a measurement nobody
can read is worth nothing, so the measuring code stays open. The settings a search arrives at are
a different kind of thing: they are the output of hours of card time, they decide nothing about
how a result is measured, and a published figure stays checkable whether or not you know which
layer band produced it.

## How a preset reaches the tool

A pack is an ordinary Python distribution that registers itself under the entry point group
`senbonzakura.presets`:

```toml
[project.entry-points."senbonzakura.presets"]
my-pack = "my_pack"
```

The tool then finds it by name rather than by path, which matters because the same pack can be a
source checkout on one machine and an installed wheel on another. A fixed path would work in one
of those cases and fail in the other.

A pack exposes two callables:

| Callable | What it returns |
|---|---|
| `catalogue()` | The model identifiers it holds, and nothing heavier. Listing what a pack covers never loads a preset |
| `preset(model)` | One record for that identifier, or nothing |

A record is a mapping:

```python
{
    "model": "example/synthetic-7b",
    "settings": {"layer_lo": 0.2, "layer_hi": 0.7, "max_directions": 4},
    "provenance": "which run produced these, on what date, on what hardware",
    "requires_at_least": "0.4.0",      # optional
    "requires_below": "0.4.1",         # optional
}
```

Every key in `settings` has to be an argument this version of the tool actually reads, and a key
it does not read is refused rather than ignored. A setting nobody applies is the worst outcome
available here: the person who supplied it believes it took effect, and nothing says otherwise.
The same reasoning covers the version keys. A layer band measured against a different search is
not a setting for this one, so a mismatch is a refusal that names both versions.

`provenance` is required. A tuned number with no account of where it came from cannot be
re-derived and nobody can tell whether it still applies.

## Reading it from Python

```python
from senbonzakura import presets

presets.available()            # every model any installed pack covers
presets.find("owner/model")    # the preset, or None if nothing here holds one
presets.require("owner/model") # the preset, or a refusal that explains the absence
```

`find` returning `None` is an answer and not a fault. `require` is for the caller that wants a
message instead, and the message says what was missing, what it would have supplied, and that the
search still runs without it.

## What this is not

It is not a licence check, and it cannot be. Python resolves imports while the program is already
running, so anything one module can check at that point another module can satisfy. The resolver
is a lookup that finds nothing and says so. Nothing about it is a lock.

## What is published today

No pack is published. The resolver and the contract above are in the package so that a pack can
exist without changing the open tool, and the open tool behaves exactly as it does with no pack
installed, because that is the only state anybody is in.

::: tip This page is about settings, not about corpora
The evaluation track and the research prompt corpora that ship inside the wheel are a separate
matter with their own licences and their own page. See
[corpus provenance](/corpus-provenance) and `THIRD-PARTY-CORPORA.md` in the repository for what
the distribution carries and under what terms.
:::
