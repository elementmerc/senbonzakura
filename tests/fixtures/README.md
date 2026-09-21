# The seeded corpus of known-bad setups

Reconstructions of measurement incidents, used as the checker's ground truth. Every file here
is an artefact somebody could plausibly have on disk, and every one of them is wrong in a way
that was discovered the expensive way.

`tests/test_seeded_incidents.py` runs `senbonzakura check` over each one and asserts the named
check fires. That is the difference between a checker that has been reasoned about and one that
has been watched working.

## The one rule that makes this corpus worth anything

**Every fixture says where it came from.** Each file carries a `_provenance` block, which the
checker ignores, holding:

- `incident` — what happened, and when.
- `documented_at` — where the record of it lives.
- `kind` — `reconstructed` or `invented`, and they are different kinds of evidence.
- `inferred` — what the public record did not say and this file guessed.
- `not_reproduced` — what could not be rebuilt at all.

An artefact taken from a record and an artefact somebody made up to exercise a rule are not
interchangeable, and a corpus that does not say which is which is a corpus that will be quoted
as though it were all the first kind. This project has already published a p-value that came
out of a fixture built to exercise a report.

## Two halves, and the second one is the measurement

**`ours/`** reconstructs incidents from this project's own history. These are the incidents the
checks were written against, so a catch here proves the check works and proves nothing about
whether the check set is any good. It is the easy half and it is still necessary.

**`held-out/`** holds incidents from outside this project, chosen AFTER the checks were frozen
at a commit, by somebody who did not pick them for being coverable. A miss here is a real
result and is reported as one. Nothing in `held-out/` may cause a check to be written or
widened: doing that converts the only honest measurement of the check set into a fitted one,
which is exit-gate loophole 2 of the v0.8 rung and the reason the directory is separate.
