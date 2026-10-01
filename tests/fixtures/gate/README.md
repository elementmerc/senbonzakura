# A planted regression, so the gate has been watched failing

Four baseline-shaped artefacts: one recorded baseline and three measurements judged against it.
They exist for one reason, which is that a gate nobody has watched fail is decoration.

| File | What it is | What the gate does with it |
|---|---|---|
| `baseline-refusal-rate.json` | the recorded baseline | the thing the other three are judged against |
| `measurement-steady.json` | a run inside the baseline's interval | passes, status 0, and prints how much room there was |
| `measurement-regressed.json` | the planted regression | fails, status 1, naming which property moved and by how much |
| `measurement-incomparable.json` | steady, but rendered by a different prompt format | refuses, status 2, naming the field that disagreed |

## These numbers describe no run

Every file carries `extra._provenance` with `kind: invented`, which is the strongest word in
`../README.md`'s vocabulary for "nobody measured this". The point estimates and intervals were
chosen to put one measurement outside the baseline's interval, one inside it, and one beyond the
comparability rules. Nothing here may be quoted as a measured figure, and the provenance block
says so inside each file as well as here, because a file travels and a README does not.

The three outcomes are kept apart on purpose. Status 1 means the property regressed; status 2
means the two measurements were never comparable and nothing was shown about the model at all. A
build that reads those as the same thing teaches its reader to ignore both.

## Why these are committed rather than built in a temporary directory

`tests/test_gate.py` already builds equivalent measurements through `baseline.record`, which
proves the arithmetic. It cannot prove the status a shell sees, because it never leaves the
process. These files are on disk so the command can be driven as a subprocess, with the exit
status read the way a build system reads it, and so the CI job that gates this project's own
numbers has something committed to point at.

## Keeping them honest

`baseline.record` built all four, so a field this project later makes mandatory cannot quietly
stay absent here. If the writer changes shape, regenerate them rather than hand-editing: a
hand-edited baseline is a file that passes the reader and would never have passed the writer.
