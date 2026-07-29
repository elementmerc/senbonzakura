# Constraints

Two files, because one cannot do both jobs.

| File | Job | Editable? |
|---|---|---|
| `ci.txt` | Installable. Pins the CPU environment CI resolves, so a dependency release cannot turn a green run red without someone choosing it. | Yes, deliberately, when a dependency is upgraded. |
| `measured-YYYY-MM-DD.txt` | Documentary. Records the exact environment a published result was produced in, including the hardware and the index it came from. | **No.** Write one per published result and never touch it again. |

The split exists because the published numbers came from a rented CUDA pod and
CI runs CPU-only, so a single file would either be uninstallable in CI or would
misdescribe the measurement.

## Using them

```sh
pip install -e ".[dev]" -c constraints/ci.txt
```

A measured file is not for installing. It is the answer to "what exactly
produced this number", asked a year later by someone who cannot ask you.

## What goes in a measured file

Everything needed to tell whether a re-run is comparable: the resolved versions
of every dependency, the Python version, the torch build (the `+cpu` or `+cu124`
suffix matters), the index URL, the accelerator, and the commit of this
repository. A version list without the hardware is not provenance, because
`torch==2.5.1+cu124` on an H100 and on a 3090 are different measurements.
