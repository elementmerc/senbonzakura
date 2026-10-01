# What we got wrong

This page is the running record of things the documentation said that were not true, and what
they say now.

It exists because the corrections were piling up inside the pages they corrected. A README
paragraph that spends four lines explaining what it used to say is a paragraph that has stopped
being a README, and the correction is the part a returning reader needs while the new reader
needs only the current truth. So the current truth stays where it was and the history moves here.

[What is and is not established](/guide/what-we-know) is the sister page and the more important
one: it covers the research claims, including the headline claim this project is named for, which
we tested and could not support. This page is narrower. It is about the documentation.

::: tip Why keep it at all
A project that quietly edits its own claims is asking to be trusted twice: once about the new
number, and once about there having been no old one. The whole argument here is that the figures
can be checked, and a reader cannot check a figure whose predecessor has been deleted.
:::

## The install instructions told you not to install it

**It said:** don't run `pip install senbonzakura`. The newest thing on PyPI was 0.3.0, its
numbers were withdrawn, and `senbonzakura-check` was not published at all, so the dependency
resolved to nothing and the install failed for a reason that read like something else entirely.
The route it offered instead was a pair of `git+` URLs.

**It says:** `pip install senbonzakura` is the whole install.

**What was actually wrong:** nothing, while it lasted. It was true from July 2026 until 0.4.0
went out, and then it was false for as long as nobody re-read it. The lesson is not about
accuracy, it is about where a time-limited claim goes: one that expires on a release needs to be
checked by the release, not by memory.

## It said no abliterated checkpoints were published, and that correction was the error

**It said:** first, that abliterated checkpoints from this work are published separately on
HuggingFace. Then, after that was judged an overclaim, that no model weights are in this git tree
and none are published.

**It says:** no model weights are in this git tree, and abliterated checkpoints are published
separately on the Hub, public and ungated, each under its base model's own licence. The
evaluation numbers on most of those cards are withdrawn, so a reader should check the card rather
than trust a figure on it.

**What was actually wrong:** the first sentence was true and was corrected into a false one.
Checked on 2026-10-01 with no credentials, the way a stranger would: fourteen public repositories,
`gated` false on every one, real weight files, and downloads in the hundreds on some. The original
sentence gave no link, and the absence of a link was read as the absence of the checkpoints.

**Why this one is worth its own entry rather than a quiet fix.** It is a dual-use disclosure that
inverted into a reassurance, which is the second time that has happened here; the evaluation track
card did the same thing at 0.4.0. And it inverted *inside the page that exists to record
inversions*, which is the part worth sitting with: a correction mechanism with nothing checking it
is another unchecked claim. The weights are the one thing in this project a reader might most want
an honest sentence about, and for a release and a half they got the opposite of one in both
directions.

## The first-run guide timed the wrong model

**It said:** run the command on Qwen3-1.7B, and underneath, that it took 108 minutes end to end.

**It says:** the 108 minutes were measured on Qwen3-0.6B, and the model in the command has not
been timed at the default budget.

**What was actually wrong:** the figure was real and the attribution was not. A reader takes a
number printed under a command to be about that command. The README had a third version of the
same claim, "about an hour for a 1.7B", which was the error this guide already carried a whole
paragraph correcting: the correction had landed on one page and not the other.

## The demo model hardly refused anything

**It said:** abliterate Qwen3-1.7B, on every page that shows a first command.

**It says:** abliterate Qwen2.5-0.5B-Instruct.

**What was actually wrong:** the demo had almost nothing to demonstrate. That model's refusal rate
on the bundled evaluation set was never measured, and its sibling Qwen3-0.6B measures 4.7%, which
is below the floor the tool now refuses to edit below. So the flagship command was pointed at a
model that would give a reader nothing to watch, and had become a command the tool itself would
decline.

## A three-minute CPU demo was not possible

**It said, in planning rather than in public:** the whole edit could be shown on a CPU in under
three minutes.

**It says:** 857 seconds, so 14.3 minutes, on sixteen modern cores, at a budget cut to the bone.
An older CPU is 45 to 90 minutes.

**What was actually wrong:** the figure was an extrapolation from a GPU run presented with the
confidence of a measurement. It is on this page even though it never shipped, because it was one
review away from shipping and the reason it did not is that somebody timed it.

## Limits said a model could not be run at all

**It said:** Qwen3-4B doesn't fit the 6 GB card this project is built around, so a corrected run
needs hardware we don't own.

**It says:** the model loads. What will not fit in the card goes to host RAM, so the run works and
generates at host speed for that share. The blocker is a day of wall clock, not capability.

**What was actually wrong:** an unwillingness to spend the time had been written up as a hardware
impossibility. The tool now says what fraction of a model ended up off the card, and projects what
that costs, rather than leaving you to infer it from how slowly the progress line moves.

## The help text's own arithmetic did not add up

**It said:** this page shows the 16 flags a run needs, 55 more control the search, and there are
69 in total.

**It says:** all three numbers are counted from the parser when the help is built.

**What was actually wrong:** 16 plus 55 is 71. Two faults in one sentence. The total was typed in
by hand while the other two were computed, so they diverged the moment anybody added a flag; and
the count of shown flags included the model positional, so it was wrong on the day it was written.
A first-time reader added them up, which is the entire point: the argument here is that the
numbers can be checked, and the first one a stranger checked was wrong.

## The install size disagreed with itself

**It said:** three different sizes, on three pages.

**It says:** 70 packages and 5.9 GB, measured on 2026-09-28 in an empty virtualenv on Python 3.14,
with the breakdown beside it, and a test that fails when the pages disagree.

**What was actually wrong:** a measured fact copied into prose in three places, where nothing can
notice it drifting. The same shape as the flag count above, and as the AUC figures this project
withdrew.

## Where next

- [What is and is not established](/guide/what-we-know) for the research claims, which is the
  page that matters more than this one.
- [Limits and known defects](/guide/limits) for what is still true and still wrong.
- [REPRODUCING.md](https://github.com/elementmerc/senbonzakura/blob/dev/REPRODUCING.md) maps every
  published figure to the file it came from and the command that makes it.
