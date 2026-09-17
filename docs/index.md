---
layout: home
hero:
  name: Senbonzakura
  text: Precision abliteration, with receipts
  tagline: >
    Refusal abliteration for open-weight language models, and the instruments to tell you
    whether it worked. Including the runs where the answer was no.
  image:
    src: /mark.svg
    alt: Senbonzakura
  actions:
    - theme: brand
      text: Quickstart
      link: /guide/quickstart
    - theme: alt
      text: What it is
      link: /guide/what-it-is
    - theme: alt
      text: Check our numbers
      link: /guide/what-we-know
    - theme: alt
      text: Contribute
      link: /contributing

features:
  - title: Publishes the runs that went against it
    details: >
      Ideas this project was built on have been tested and dropped when the measurement said so,
      and the withdrawals are written up rather than quietly deleted. Every figure carries the
      conditions it was measured under.
    link: /guide/what-we-know
    linkText: What is and is not established
  - title: Ships its own instruments
    details: >
      A compass that asks whether the model still recognises harm, held-out arms, seeded
      bootstrap intervals, and null controls printed beside every result.
    link: /guide/compass
    linkText: Measuring the result
  - title: Says what it does not know
    details: >
      Withdrawn claims stay on the page with the reason. Numbers carry the conditions that
      qualify them, next to the number rather than three hundred lines below it.
    link: /guide/what-we-know
    linkText: What is established
---

## What this actually does

A model that has been through this tool will answer requests the original refused, including
harmful ones. That is the point of it, not a side effect, and it is not reversible by loading
the model differently: the refusal behaviour has been removed from the weights.

Everything else on this page is about whether the removal can be measured honestly. That
question only matters if you already understand the first paragraph.

Three things worth knowing before the Quickstart button:

- **The base model's licence still governs the result.** Abliteration is a modification, not a
  relicensing. Several model families carry a use policy that binds derivatives, and some of
  those policies speak directly to removing safety behaviour.
- **The package carries roughly 6,500 harmful prompts** so it runs with no network. They are
  prompts and not answers, and one ordinary run writes them to your cache directory in plain
  form. On a shared machine, consider whether whoever administers it would expect that.
- **Do not put an abliterated model in front of other people without saying what it is.** They
  cannot see the weights and have no way to know the refusals are gone.

[What it is](/guide/what-it-is) covers this properly, and
[Acceptable use](https://github.com/elementmerc/senbonzakura/blob/dev/ACCEPTABLE-USE.md) is the
statement that ships inside the package.
