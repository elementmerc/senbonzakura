# Constraints

Three files, because no one file can do three jobs.

| File | Job | Editable? |
|---|---|---|
| `ci.txt` | Installable. Pins the CPU environment, so a dependency release cannot turn a green run red without someone choosing it. Used by the `pinned` CI job. | Yes, deliberately, when a dependency is upgraded. |
| `floors.txt` | Installable. Pins every declared dependency to exactly its floor, so the floors in `pyproject.toml` are a tested claim rather than a guess. Used by the `dependency-floor` CI job. | Yes, in the same commit as the floor it mirrors. |
| `measured-YYYY-MM-DD.md` | Documentary. Records the exact environment a published result was produced in, including the hardware and the index it came from. | **No.** Write one per published result and never touch it again. |

The first split exists because the published numbers came from a rented CUDA pod
and CI runs CPU-only, so a single file would either be uninstallable in CI or
would misdescribe the measurement. The second exists because a floor and a pin
point in opposite directions: `ci.txt` answers "what do we know works", and
`floors.txt` answers "what is the oldest thing we claim works".

## Using them

```sh
pip install -e ".[dev]" -c constraints/ci.txt      # the known-good environment
pip install -e ".[dev]" -c constraints/floors.txt  # the oldest supported one
```

A measured file is not for installing, which is why it is Markdown rather than a
`.txt` pip would accept. It is the answer to "what exactly produced this
number", asked a year later by someone who cannot ask you.

## What goes in a measured file

Everything needed to tell whether a re-run is comparable: the resolved versions
of every dependency, the Python version, the torch build (the `+cpu` or `+cu124`
suffix matters), the index URL, the accelerator, and the commit of this
repository. A version list without the hardware is not provenance, because
`torch==2.5.1+cu124` on an H100 and on a 3090 are different measurements.
