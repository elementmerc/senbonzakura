# Flags worth knowing

## The ones that change what a run means

- `--max-directions K` — size of the refusal subspace to ablate (1 = single-direction).
- `--mlp-off` — attention-only ablation (leave `mlp.down_proj` untouched); tests the
  "attention carries refusal, MLP carries capability" hypothesis.
- `--hedge-ds DIR` — fold a hedged-vs-clean contrast direction into the basis, so the
  search can strip the disclaimer/hedging axis the difference-of-means direction misses.
- `--patience N` — stop the search once the frontier stalls for N trials.
- `--eval-refusal-final N` — re-score the top frontier candidates on a larger eval before
  picking the knee, so the choice isn't overfit to the small search eval.
- `--inspect LAYER STRENGTH` — print real generations before and after a cut.

The search minimises **three** objectives at once: strict non-compliance
(hard refusal plus hedging), the Heretic keyword rate as its own axis, and KL
divergence (coherence). Earlier versions optimised only hard-refusal-versus-KL and
left the keyword/hedging axis to chance.
