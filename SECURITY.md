# Security policy

## Reporting a vulnerability

Email **ops@themalwarefiles.com**. Please do not open a public issue.

Include what you did, what happened, and which version you were running. `senbonzakura doctor`
prints the version and the pinned tool versions, and pasting its output saves a round trip.

You will get a reply within seven days. This is a project with one maintainer, so that is a
realistic promise rather than a generous one. If a week passes with no reply, send the email
again; it is far more likely to have been missed than ignored.

## What counts as a vulnerability here

This tool removes a model's refusal behaviour. **A model that answers a harmful request after
being abliterated is the tool working as documented, not a vulnerability.** Please do not report
it as one, and please do not include the generation.

What is in scope is anything that harms the person running the tool, or anyone downstream of a
number it produced:

- Code execution, file reads or file writes outside the paths you pointed it at, including
  through a crafted checkpoint, dataset, GGUF or result file. This tool loads other people's
  files, so this is the category that matters most.
- Credentials, tokens or local paths reaching a log, a result artefact, a commit or the terminal.
- A measurement that can be made to report a number that is wrong in a way the report does not
  disclose. A tool whose whole claim is that its figures carry their conditions has to treat a
  silently wrong figure as a security problem, not merely a bug.
- Anything in the published package that is not built from the tagged source.

## What we will do

Confirmed reports get a fix, a CHANGELOG entry describing the problem in plain language, and
credit in that entry unless you ask us not to. If a published figure turns out to be affected,
the figure is withdrawn in the documentation as well as fixed in the code. That has happened
before and it is the policy rather than an exception.

## Dependencies with no fix available

Where a dependency has a known advisory and no fixed release, the advisory is recorded in
`pyproject.toml` under `[tool.senbonzakura.security]`, together with what was done about it and
what is still exposed. A test fails if the assessed version moves, so an accepted risk cannot be
inherited without being re-read. If you think one of those assessments is wrong, that is a report
worth sending.

## Supported versions

The most recent release. This project does not backport.
