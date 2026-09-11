# senbonzakura-check

Read an evaluation result file and report how the number could be wrong.

```sh
pip install senbonzakura-check
senbonzakura-check results/
```

**It installs in seconds and pulls nothing.** No torch, no model, no corpus, no GPU, no network.
It reads JSON and does arithmetic.

It understands result files from
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
[Inspect](https://inspect.aisi.org.uk), and
[senbonzakura](https://github.com/elementmerc/senbonzakura) itself, at the **artefact** level:
it never imports those tools, so their release schedule is not your breakage.

Every finding names four things, because a finding that says only what fired sends you to the
source:

- what it detects
- **the incident that motivated it**, because a finding with no citation is an opinion
- how to fix it
- **what would make the check wrong**, so you can judge it without reading the code

## Three exit codes, and they mean different things

| Code | Meaning |
|---|---|
| `0` | every file was read, and nothing fired |
| `1` | at least one finding |
| `2` | at least one file could not be read at all |

A file the checker cannot parse is reported **unchecked**, never clean. A clean report on
something nobody read is indistinguishable from a clean bill of health, and a CI gate that
treats the two the same goes green the day your result format changes.

For the same reason the summary says how many checks **did not apply** to a file: "nothing
found" across checks that could not run is a different statement from "nothing found".

**A clean report is not a certificate.** It looks for known failure modes. It cannot tell you a
number is right, and it says so in its own output.

## Running it without choosing to

In GitHub Actions:

```yaml
- uses: elementmerc/senbonzakura@v0.4.0      # pin it
  with:
    path: results/
```

As a pre-commit hook:

```yaml
repos:
  - repo: https://github.com/elementmerc/senbonzakura
    rev: v0.4.0                               # pin it
    hooks:
      - id: senbonzakura-check
```

## Adding a check

A check is a JSON file in `checks/`, and the engine never hardcodes one. It carries an id, what
it detects, the incident behind it, the remedy, a confidence, what would make it wrong, and two
controls: a document that makes it fire and one that does not. **A check that cannot demonstrate
firing is not merged**, because a check nobody has watched fail is a check nobody has tested.

## Licence

AGPL-3.0-or-later. This is part of
[senbonzakura](https://github.com/elementmerc/senbonzakura); the abliterator lives in the
`senbonzakura` distribution and brings the deep-learning stack with it. This one does not.
