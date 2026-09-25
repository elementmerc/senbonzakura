# Acceptable use

Senbonzakura removes refusal behaviour from language models and measures what that removal
cost. Read this before you use it. It ships inside the package, so you have it whether or not
you found the repository.

This is a statement of what the project considers acceptable. It does not add terms to the
AGPL-3.0-or-later licence the software is under, and it does not take any away.

## The short version

A model that has been through this tool will answer requests the original refused, including
harmful ones. That is what it does. It is not a bug to be reported, a limitation to be worked
around, or a side effect. Everything below follows from it.

**Do not put an abliterated model in front of other people without saying what it is.** Not on a
website, not in a product, not in a Discord bot, not behind an API. The people using it cannot
see the weights and have no way to know the refusals are gone.

**Do not use this tool to produce material that is illegal where you are.** Removing a model's
refusals does not make the output lawful, and it does not move responsibility for it onto the
model or onto this project.

**Do not use it against a model you are not permitted to modify.** See the licence section.

## What it is for

- Measuring how much capability an abliteration costs, which is the thing the field mostly does
  not measure.
- Comparing abliteration methods on one ruler, including methods that are not ours.
- Safety and interpretability research into refusal behaviour: where it lives in a model, how
  fragile it is, and what it takes to remove it.
- Red teaming a model you own or are authorised to test.
- Auditing a checkpoint somebody else published. `drift`, `score`, `capability` and `compass`
  run against any model and do not require you to abliterate anything.

## What it is not for

- Producing an uncensored service for other people to use without telling them.
- Generating material that is illegal where you are, or that is aimed at a specific real person.
- Removing refusals from a model whose licence forbids it, or whose use policy your derivative
  would breach.
- Stripping a safety classifier or a moderation model of its ability to decline.
- Claiming a measurement this tool did not produce. If you publish numbers from it, publish the
  artefacts too, or say which ones you are withholding and why.

## The licences that follow the weights

Three separate licences are usually in play and they do not merge.

**This software** is AGPL-3.0-or-later. That covers the tool.

**The base model** keeps its own licence, whatever it was. Abliteration is a modification, not a
relicensing: nothing this tool does can loosen the terms the weights arrived under. Several model
families, Gemma and Llama among them, carry a use policy that binds derivatives, and a model you
abliterate is a derivative. Some of those policies speak directly to removing safety behaviour.
Read the licence your base model came with and satisfy it before you publish. `senbonzakura
report --base-licence` writes a card that names it; naming it is not the same as complying with
it, and the card says so.

**The corpora** bundled in the package carry their own terms. The evaluation track is CC BY-NC
4.0: attribution required, non-commercial. The research corpora are variously MIT and CC BY 4.0.
`THIRD-PARTY-CORPORA.md` ships beside this file and lists each one. Attribution travels with any
figure you publish from them.

## What the package contains that you should know about

The wheel carries roughly 6,200 harmful prompts: the evaluation track and six public research
corpora, so the tool runs with no network. They are obfuscated in the package to keep them out of
automated scrapes, and the key ships beside them, so that is a speed bump and not protection. One
ordinary run unpacks them to your cache directory in plain form.

They are prompts, not answers. No model weights, no completions, and no jailbreak techniques are
distributed here.

If you are running this on a shared or institutional machine, that corpus lands on it. Consider
whether whoever administers that machine would expect it to.

## If you publish an abliterated model

- Say it is abliterated, in the model card and in the name.
- Carry the base model's licence and any use policy that came with it.
- Publish the artefacts behind any number you quote, or say which you are withholding.
- Do not claim "minimal degradation" without a capability measurement. That claim is the reason
  this tool exists, and it is unfalsifiable without the numbers behind it.

## Reporting a problem

Security issues, licence problems, and misuse of anything this project publishes: open an issue
at the repository, or contact the maintainer. If the report is sensitive, say so and do not put
the detail in the issue.

## No warranty

This software comes with no warranty of any kind, on its correctness, its fitness for any
purpose, or the accuracy of any number it produces. Responsibility for what is done with it, and
with any model modified or measured using it, sits with whoever does it.
