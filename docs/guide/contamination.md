# Contamination

Whether the corpus you fitted on already contains the benchmark you are about to quote.

## The problem

Suppose you want to report a number on a public benchmark. If any of that benchmark's
requests are in the part of your track the directions were fitted on, the model was
tuned on the questions it is being marked against, and the number says nothing. This
is easy to do by accident: public corpora are assembled from each other, so a benchmark
can be inside yours without anybody choosing to put it there.

```sh
python -m senbonzakura.track --out mytrack \
    --contamination advbench.txt --contamination-name AdvBench \
    --contamination-report contamination-advbench.json
```

One prompt per line, same as the builder. It reads the track and writes nothing to it.

```
contamination: AdvBench against this track
  520 rows, 520 distinct requests (templates discovered: 7)
  in fit 2, in search 3, in measure 7, absent 508
  CONTAMINATED: 5 of its requests were fitted or searched on.
```

Three things to know about how it counts:

- **It matches requests, not strings.** A benchmark that phrases the same question
  differently from your corpus still counts as overlap, for the reason in the paragraph
  above. Comparing text alone would report a clean result and be wrong.
- **`measure` is not contamination.** That part is held out by design, so a benchmark
  row found only there can be reported. The tool tells you how many rows qualify.
- **It reports counts and never a prompt**, so the output is safe to paste anywhere.

Add `--fail-on-contamination` to make it exit non-zero instead of just reporting, which
is what you want in a script. Without the flag it always exits cleanly, because a check
that refuses cannot be run to find out.

It finishes by telling you the flags that put the compass on the held-out rows:

```
TRACK_BUILT mytrack  harmful {'fit': 256, 'search': 128, 'measure': 616}
  measure with: --skip-harmful 128 --skip-harmless 384 --n 616
```

