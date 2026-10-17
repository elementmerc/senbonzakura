# What we got wrong

This page records things the documentation said that were not true, and what they say now. The current truth
lives in the pages it corrects. The corrections are kept here, so a returning reader can check that an old
figure was replaced rather than erased.

[What is and is not established](/guide/what-we-know) is the more important sister page. It covers the research
claims, including the headline claim this project is named for, which we tested and could not support. This page
is about the documentation.

::: tip Why keep it at all
A project that quietly edits its own claims asks to be trusted twice: once about the new number, and once about
there having been no old one. A reader can't check a figure whose predecessor has been deleted.
:::

## The install instructions said not to install it

**It said:** don't run `pip install senbonzakura`. The newest PyPI release was 0.3.0, its numbers were withdrawn,
and `senbonzakura-check` was not published, so the install failed. The route offered instead was a pair of `git+` URLs.

**It says:** `pip install senbonzakura` is the whole install. It has been since 0.4.0.

## The install size disagreed with itself

**It said:** three different sizes, on three pages.

**It says:** 70 packages and 5.9 GB, measured on 2026-09-28 in an empty virtualenv on Python 3.14, with the
breakdown on the [install page](/guide/install). A test fails when the pages disagree.

## No abliterated checkpoints were published, and that correction was the error

**It said:** first, that abliterated checkpoints from this work are published on Hugging Face. Then, after that was
judged an overclaim, that no model weights are in this git tree and none are published.

**It says:** no model weights are in this git tree, and abliterated checkpoints are published separately on the Hub,
public and ungated, each under its base model's licence. Most of those cards carry withdrawn evaluation numbers, so
check the card before you trust a figure on it.

The first sentence was true, and the correction turned it into a false one. Checked on 2026-10-01 with no credentials,
there were fourteen public repositories, `gated` false on every one, with real weight files. The original sentence gave
no link, and the missing link was read as the absence of the checkpoints. A dual-use disclosure had inverted into a
reassurance, and the weights are the one thing a reader most wants an honest sentence about.

## The first-run guide timed the wrong model

**It said:** the command took 108 minutes end to end.

**It says:** the 108 minutes were measured on Qwen3-0.6B. The model in the command has not been timed at the default budget.

## The demo model hardly refused anything

**It said:** abliterate Qwen3-1.7B, on every page that shows a first command.

**It says:** abliterate Qwen2.5-0.5B-Instruct. Qwen3-0.6B measures 4.7% refusal on the bundled evaluation set, which is
below the floor the tool now refuses to edit below. The earlier demo model would have given a reader nothing to watch.

## A three-minute CPU demo was not measured

**It said, in planning rather than in public:** the whole edit could be shown on a CPU in under three minutes.

**It says:** 857 seconds, so 14.3 minutes, on sixteen modern cores, at a budget cut to the bone. An older CPU is
likely several times that.

The 857 seconds was timed once, and that is all the repository can tell you. It names neither the model, the track
nor the flags, and it is not in `evidence/`, so nobody can repeat it. Treat it as one machine's reading of a run you
cannot reconstruct. An earlier estimate for an older CPU had no measurement behind it, and it was removed.

## Limits said a model could not be run at all

**It said:** Qwen3-4B doesn't fit the 6 GB card, so a corrected run needs hardware we don't own.

**It says:** the model loads. What won't fit on the card goes to host RAM, so the run works and generates at host speed for
that share. The blocker is a day of wall clock, not capability. The tool now reports what fraction of a model is off the card,
and projects the cost.

## The help text's own arithmetic did not add up

**It said:** the page shows 16 flags a run needs, 55 more control the search, and there are 69 in total.

**It says:** all three numbers are counted from the parser when the help is built.

The total had been typed in by hand while the other two were computed, so they diverged as soon as a flag was added. The
count of shown flags also included the model positional, so it was wrong on the day it was written.

## Two pages disagreed about whether our numbers carry intervals

**It said:** the README said every measurement reports an interval, and the comparison page said `score`, `coherence` and
`drift` print bare numbers. Both cannot be right.

**It says:** every measurement carries an interval, with two exceptions named on the spot. `score` stamps a Wilson interval
into its result file and prints a bare percentage on its finishing line. `coherence` carries no interval, because one forward
pass over one fixed passage returns the same number every time. `drift` prints a seeded bootstrap interval beside its KL.

## The norm-restore residue was presented as a property of the field

**It said:** "Everyone has the norm-restore problem."

**It says:** three tools preserve each weight row's original length after removing a direction, and none of the three measures
the residue this leaves behind. A fourth tool bakes its projection with no per-row restoration, so it doesn't inherit the
problem. Whether that costs something else is open.

## `drift` was described as the distance between two models

**It said:** "How far did the output distribution move."

**It says:** how far the **first-token** distribution moved, one position per prompt, at the end of the rendered prompt before
the model has written anything. The gap matters most on a reasoning model, which can open identically and then reason differently.

## "Nobody validates the judge" was broader than the evidence

**It said:** nobody validates the judge.

**It says:** nobody computes agreement above chance. One tool grades delivery with an externally validated classifier pinned by
revision, which is a real answer to "is your grader any good", even though it isn't the answer this project gives.

## Where next

- [What is and is not established](/guide/what-we-know) for the research claims, which matter more than this page.
- [Limits and known defects](/guide/limits) for what is still true and still wrong.
- [REPRODUCING.md](https://github.com/elementmerc/senbonzakura/blob/dev/REPRODUCING.md) maps every published figure to the file it came from, and the command that makes it.
