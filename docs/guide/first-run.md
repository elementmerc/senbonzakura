# Your first run


The fast path, if you just want the best result and no knob-twiddling:

```sh
senbonzakura kageyoshi --model <hf-model-or-path> --out <dir> --track <dir> --device cuda
```

`kageyoshi` is the ultimate balanced-effort mode. It detects the
architecture (dense, fused MoE, or expert-list) and parameter count, scales the
search budget accordingly, and switches on every quality lever, so you supply only
the paths. "Balanced" is the point here. It picks the most uncensored config that stays
coherent (the KL ceiling and coherence penalty guard it), not the most aggressive
one. It owns the search knobs; manual `--trials` / `--max-directions` and friends
are ignored in this mode. If a `hedge_ds/` sits in your track directory it folds the
hedging axis in automatically.

For full manual control:

```sh
senbonzakura --model <hf-model-or-path> --out <dir> \
    --track <dir-holding-bad_ds-good_ds-bad_eval_ds> \
    --search pareto --max-directions 6 --trials 200 --device cuda \
    --eval-refusal-final 256 --patience 40
```

Score any model on a held-out evaluation set with the same ruler:

```sh
python -m senbonzakura.score --model <dir> --eval <eval-dataset> --out results.json --label mymodel
```

Measure its coherence on the same loader, the other half of the ruler (the
perplexity the model assigns to one fixed neutral passage, lower is better):

```sh
python -m senbonzakura.coherence --model <dir> --out coherence.json --label mymodel --load-in-4bit
```

`senbonzakura.metrics` is that shared ruler: hard refusal, soft refusal (the "I
can't, but here's a lecture" hedge), broken output, and the Heretic keyword rate
(copied verbatim, so the numbers are directly comparable to Heretic's published
figures).

