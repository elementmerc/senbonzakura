# Limits and known defects

Every condition attached to a number this project has published, in one place.

## Reproducibility and status

- **The [Why multi-direction](#why-multi-direction) table is NOT current.** It was measured on
  2026-07-14, before two scoring fixes, on an evaluation that is not held out. The caveats are
  printed beside the table itself rather than here, because a qualification three hundred lines
  from the thing it qualifies is not a qualification. Every cell was produced by the shipped
  tools (`python -m senbonzakura.score --load-in-4bit` for the refusal columns,
  `python -m senbonzakura.coherence --load-in-4bit` for perplexity), which is what makes it
  reproducible; it is comparability with today's code that it lacks.
- **Every figure this project measured before 2026-07-30 is affected.** Two bugs: the refusal
  scanner read a 240-character window (`71cc119`, 2026-07-21), and the prompt renderer existed
  in three copies that had drifted, so a configuration selected under one prompt format was
  reported under another (`d5a16e0`, 2026-07-30). Numbers either side of those dates are not
  comparable, and this project would rather say so than quietly restate them.
- **Determinism, measured rather than assumed.** On a fixed batch size the forward-only paths
  are reproducible: each compass command was run three times on the same card at batch 16 and
  produced AUCs identical to four decimal places. That says nothing about generation, which is
  sampled, and nothing about a different batch size, which changes reduction order.
- **Every Gemma figure is withdrawn (2026-08-05).** The weight edit did not reach the residual
  stream on that architecture. Gemma 2 and Gemma 3 pass each sublayer's output through a
  learned-gain normalisation *before* adding it to the residual stream, and this tool edited the
  weights upstream of that step, so the normalisation partly undid the edit before it took
  effect. Measured: on gemma-2-2b-it the weight edit and an equivalent activation-space
  intervention disagreed by 0.578 in refusal rate, against 0.016 on Qwen3, and no Gemma
  configuration moved KL above 0.021 at any setting. Directions chosen by the tool scored no
  better than random ones. The cause is fixed and the numbers will be re-measured; until they
  are, treat every Gemma result here as unmeasured rather than as a result. Qwen models are
  unaffected: the architecture difference is the whole mechanism, which is why the same code
  worked on one family and not the other.
- **The model-size ceiling.** Every model this tool has been run on is **under 3B, and six of
  the seven are under 2B**; gemma-2-2b-it is the largest at 2.61B. Nothing here is evidence about
  how the method behaves at 7B, 30B or beyond. The streaming work exists to make those sizes
  reachable on hardware we own.
- **The comparison is runnable by anyone, and that is deliberate.** It used to exist only as our
  private runner configuration, which meant nobody outside this machine could reproduce it. As of
  2026-08-06 it is `senbonzakura bench`, ships in the wheel, and has its own tests. A table you
  cannot re-derive is a claim, not a measurement.
- **No head-to-head against Heretic is published yet** (see [Benchmark](#benchmark)). Recent
  correctness fixes to direction extraction, the multi-direction basis, and knee selection moved the
  numbers substantially in Senbonzakura's favour on the keyword axis, so any comparison is run under
  the corrected code, on identical ground, before it goes in.
