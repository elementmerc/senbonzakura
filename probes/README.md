# Contributed probes

A probe is a behaviour somebody wants measured: a set of items, a rule for grading them, and
enough provenance that the number it produces can be argued with.

Probes in this directory are distributed with the project. Anyone who installs it gets them.

## Read this before you use one

**We check the shape of a probe. We do not vouch for its contents.**

The checks below are mechanical. They confirm a probe declares what it is, names a grading rule
that exists, carries a licence, and does not use the field names that prompts and model outputs
are spelled with. They cannot tell you whether two hundred innocuous-looking rows measure what
their author says they measure, and nothing here reads the items and judges them, because that
would be a classifier nobody has validated and this project does not ship instruments it has not
measured.

So: **look at a probe before you report a number from it.** It is a small file of plain text and
reading it takes a minute. That is not a disclaimer to skip past; it is the one check that the
machinery cannot do for you.

## Contributing one

Copy `examples/probes/arithmetic-demo`, change the items, keep the shape.

```
your-probe/
    probe.toml     what it measures, how to grade it, where it came from
    items.jsonl    the items, as {"problem": ..., "reference": ...}
```

Open a pull request adding it here. Three things happen, in this order.

1. **The format gate runs automatically.** `tools/ci/check_probes.py` validates every probe in
   this directory on every push. It reports every problem at once rather than the first, because
   a contributor fixing one should already know about the rest.
2. **A human reads the items.** This is required and it is not a formality. The gate can only see
   shape, and the obvious way to abuse a contribution format on a tool like this one is to post a
   harmful corpus as a benchmark. A probe is merged when somebody has read it.
3. **It ships.** From then on it travels inside other people's runs, which is why the licence and
   the provenance are required rather than encouraged.

## What will be refused

- Items using prompt-shaped field names, whatever the manifest declares. Refused on shape, always.
- Anything declaring a content class other than `benign`. Harmful material belongs in the gated
  evaluation track, which has an access decision attached to it.
- A grading rule this tool does not have. A probe names a rule rather than shipping code, so
  running somebody's probe reads their data rather than runs their program.
- A named dataset with no pinned revision. An unpinned dataset is a different dataset on a
  different day.
- Items whose reference the grading rule cannot read. They do not fail, they score indeterminate
  for ever, which shrinks the probe silently.
- No licence.

## Running one

```sh
senbonzakura abliterate --model MODEL --capability-eval probes/your-probe
```
