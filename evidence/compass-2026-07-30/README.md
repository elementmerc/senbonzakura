# Compass, 2026-07-30 — the first held-out measurement

Two base models, measured on rows nothing was fitted on and nothing was selected on. These
are the first result files this repository has carried; every earlier number lived only in a
terminal or on a personal account, which is why the table in the main README has been hard to
check.

**Nothing here is an abliterated model.** These are the "before" side. The paired
before-and-after comparison needs an abliteration run, which is a different spec.

## What the numbers say

| model | AUC (95% interval) | length-only AUC | canonical-token AUC | topic-matched AUC |
|---|---|---|---|---|
| Qwen3-1.7B | 0.9887 (0.9871 to 0.9904) | 0.6564 | 0.9887 | 0.9944 (n=280) |
| Qwen3-0.6B | 0.6616 (0.6504 to 0.6722) | 0.6564 | 0.6616 | 0.7104 (n=280) |

AUC is the chance that a harmful prompt gets a higher HARMFUL-minus-BENIGN margin than a
harmless one. 0.5 is a coin toss.

**Two very different results, and the second one is a null.** Qwen3-1.7B separates the two
arms almost perfectly. Qwen3-0.6B scores 0.6616 where **counting tokens alone scores 0.6564**,
and its interval (0.6504 to 0.6722) sits astride that baseline: on this corpus, by this
measure, the 0.6B does not recognise harm at all. Its actual output says the same thing more
bluntly. It answers `HARMFUL` to all 4,504 harmful prompts *and* all 4,504 harmless ones.

The 1.7B, by contrast, discriminates in what it emits: `HARMFUL` for 98.3% of the harmful arm
and 18.5% of the harmless one.

## An earlier version of this file had different numbers

This artefact was first published with 0.9636 and 0.7263, and those numbers were measured at
the wrong position. Recorded here rather than quietly replaced, because the artefact is meant
to be a record.

Qwen3 is a thinking model. Its chat template appends `<think>` to the generation prompt, so
the position immediately after the prompt, where the compass read its two verdict logits, is
where the model was going to start reasoning. The read-out audit is what caught it:

| | reading `<think>`'s position | reading the verdict's position |
|---|---|---|
| most likely token is a verdict | 0.0% of prompts | 100% |
| probability held by the two verdicts | ~0 | 1.0000 |
| most frequent token there | `<think>` | `H`, then `B` |
| Qwen3-1.7B AUC | 0.9636 | 0.9887 |
| Qwen3-0.6B AUC | 0.7263 | 0.6616 |

The cause was three copies of the same three lines of prompt-rendering code, drifted apart:
the search generated with thinking turned off, while the scorer that produces the published
refusal rate and the compass that produces this AUC both left it on. One renderer now, with a
test asserting all three use it.

**Every senbonzakura number published before 2026-07-30 was measured under that split** and is
not comparable to what this tool now produces.

## Read the three control columns before the first one

**Length-only AUC (0.6564).** Rank the prompts by how many tokens they contain, ignore the
model completely, and you already score 0.6564. That is a fact about the corpus rather than
about any model, which is why it is identical in both rows: same prompts, same tokenizer. It
is the floor a real result has to clear, and the 0.6B does not clear it.

The mean prompt lengths differ by only 1.5 tokens out of 55, so this floor comes from the
shape of the two length distributions and not from one arm simply being longer. It will not be
fixed by trimming a few long prompts.

**Canonical-token AUC.** The headline takes, for each verdict word, the best of several
spellings (`HARMFUL`, ` HARMFUL`, `Harmful`, ` harmful`, and so on), and that choice is made
by looking at the logits. The canonical column removes it and uses one fixed token pair: the
first token of the spelling the judge prompt actually asks for.

**It now agrees with the headline to four decimals on both models.** That is worth stating
plainly, because in the earlier, wrongly-positioned measurement it did not: the gap was 0.15
on the 1.7B and the 0.6B inverted below chance. The free parameter was never the problem. The
position was, and the spelling gap was a symptom of reading a position where no spelling of
the verdict carried any probability.

**Topic-matched AUC.** The main harmless arm is drawn from different subject matter than the
harmful arm, so a model that only recognises topics can score well without recognising harm.
The topic-matched arm holds subject matter still. Both models score *higher* on it than on the
unmatched arm, which is the reassuring direction: the discrimination survives when topic stops
being a clue. Caveat, and it matters: that 280-row set predates the rebuilt track and is not
partitioned, so nothing guarantees its rows were held out. Harmless for a base model; it would
need fixing before the same column is quoted for an abliterated one.

## What the corpus is, which bounds what the number means

These prompts are not a sample of "harm" in general. The measured arm is dominated by two
categories:

| category | measured rows |
|---|---|
| fraud | 810 |
| cyber | 503 |
| everything else | 33 categories, the rest of the 4,504 |

So an AUC from this corpus is weighted toward fraud and cyber requests. A model that
recognises those two well and the other thirty-three poorly would score close to one of the
numbers above, and this table cannot tell that apart from broad competence.

Two smaller caveats in the same direction. The category labels were assigned by a local model
rather than by hand, validated at 89.0% agreement against the subset with ground truth. And
roughly eleven of 1,616 harmful requests came out of the labelling as political or religious
debate topics, which a model may be right to answer; if so, both arms carry a few rows that
inflate any refusal figure measured through them.

## Provenance

Both files carry it inline: package versions, python, platform, the card, the seed, the
recorded skips, and the decoded token ids that were scored. Two things to note about it.

- `git.source` is `"declared"`, not `"git"`. The run executed from a shipped tarball rather
  than a checkout, so git could not answer and the commit (`558a237`) was declared by the run.
  That is a claim rather than a measurement, and `dirty` is `null` because nothing could check
  the tree against it.
- The card is an RTX 3060 Laptop (6 GB). A version list without the hardware is not
  provenance: the same code on a different card is a different measurement.

## How to reproduce

```sh
python -m senbonzakura.margin --model Qwen/Qwen3-1.7B \
    --harmful <track>/bad_eval_ds --harmless <track>/good_ds \
    --harmless-matched <matched>/good_matched_ds \
    --skip-harmful 132 --skip-harmless 385 --n 4504 \
    --batch 16 --bootstrap 2000 --seed 42 --device cuda \
    --out base-qwen3-1.7b.json
```

The skips are not arithmetic conventions; they are the boundaries recorded in the track's
`track.json`, and a run whose flags would cross them stops rather than measuring.

**Determinism, measured rather than assumed.** Each of these two commands ran three times on
this card at batch 16 and produced identical AUCs to four decimal places. That is not a general
guarantee: batching reorders reductions, so a different `--batch` is a different number, which
is why the batch is recorded here.

**Cost.** 2 min 45 s for the 1.7B and 1 min 44 s for the 0.6B, for 9,288 prompts each plus
2,000 bootstrap resamples. One forward pass per prompt and no generation, which is why the
whole held-out arm is affordable rather than something to subsample.

**Check the read-out before trusting any of it.** Each result file carries a `readout` block
per arm: what fraction of prompts have a verdict as their most likely token, how much
probability the two verdict sets hold, and the most frequent tokens at that position with
counts. If `argmax_is_verdict` is not near 1.0, the AUC above it is a comparison of two tokens
the model was not going to emit, and the number should be thrown away rather than interpreted.
That is not hypothetical: it is what happened to the first version of this file.

## What is missing, deliberately

The per-prompt margins are **not** committed. They pair harmful prompts with what a model
did, so they live in the ignored `results/` tree; `tools/check_prompt_artefacts.py` refuses
any staged file carrying a `prompt` or `generation` field, and it is wired into the commit
path rather than only into CI.
