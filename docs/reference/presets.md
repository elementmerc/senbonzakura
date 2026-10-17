# Tuned presets, and the pack that supplies them

A run searches for its own settings. It tries layer bands, direction counts and ablation strengths, scores each trial, and keeps the best. That search is why a run takes hours on a card, and why the result can be trusted: it came from measuring this model.

A **preset** is the settings a previous measured run found for one model, written down so the next run can start from them. A preset is a shortcut past the search. It is not a different method.

## What the open tool does

The open tool runs the full search, and every instrument that scores it is in the package. If no preset is installed, a run finds its own settings. Every published number from this project was measured that way.

## How a preset reaches the tool

A pack is an ordinary Python distribution that registers itself under the entry point group `senbonzakura.presets`:

```toml
[project.entry-points."senbonzakura.presets"]
my-pack = "my_pack"
```

The tool finds a pack by name, not by path. The same pack can be a source checkout on one machine and an installed wheel on another.

A pack exposes two callables:

| Callable | What it returns |
|---|---|
| `catalogue()` | The model identifiers it holds, and nothing heavier. Listing never loads a preset |
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

Every key in `settings` must be an argument this version of the tool reads. A key it does not read is refused, not ignored. A setting nobody applies is the worst outcome: the person who supplied it believes it took effect, and nothing says otherwise. The version keys work the same way. A layer band measured against a different search is not a setting for this one, so a mismatch is refused, and the refusal names both versions.

`provenance` is required. A tuned number with no account of where it came from cannot be re-derived, and nobody can tell whether it still applies.

## Reading it from Python

```python
from senbonzakura import presets

presets.available()            # every model any installed pack covers
presets.find("owner/model")    # the preset, or None if nothing here holds one
presets.require("owner/model") # the preset, or a refusal that explains the absence
```

`find` returning `None` is an answer, not a fault. `require` is for a caller that wants a message. The message says what was missing, what it would have supplied, and that the search still runs without it.

## What this is not

It is not a licence check, and it cannot be one. Python resolves imports while the program runs, so anything one module can check, another can satisfy. The resolver is a lookup that finds nothing and says so. It is not a lock.

## What is published today

No pack is published. The resolver and the contract above are in the package so a pack can exist without changing the open tool. With no pack installed, the tool behaves exactly as it does with one.

::: tip This page is about settings, not corpora
The evaluation track and the research prompt corpora in the wheel have their own licences and their own page. See [corpus provenance](/corpus-provenance), and `THIRD-PARTY-CORPORA.md` in the repository, for what the distribution carries and under what terms.
:::
