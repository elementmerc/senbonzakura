# Benchmarking against another tool


A matched comparison against [Heretic](https://github.com/p-e-w/heretic) on
gemma-3-12b-it, the model Heretic reports in its own README, is planned so the two
methods can be read side by side on identical ground: same base model, same
evaluation, same keyword ruler. It will be added to this section when run.

### Run it yourself

The comparison is a command, not a script we keep. It runs on one machine, needs
no orchestrator, and produces the arms, the scores and the report:

```sh
# 1. Cut the prompt slices every tool is scored on, from one corpus.
python -m senbonzakura.bench stage --track mytrack --out slices

# 2. Run every tool over every seed, score every model, print the verdict.
python -m senbonzakura.bench head-to-head \
    --tools senbon,heretic --seeds 42,43,44,45,46 --trials 200 \
    --model Qwen/Qwen3-1.7B --track mytrack --eval-slices slices \
    --harmful mytrack/bad_eval_ds --harmless mytrack/good_ds \
    --out results/h2h

# 3. Read a finished run again later, without re-running anything.
python -m senbonzakura.bench report results/h2h
```

**What makes it a comparison rather than two runs.** Both tools get the same
corpus, the same trial budget, and the same prompt slices, and every model either
tool produces is scored afterwards by one instrument: our compass, on held-out
prompts, run by us. The slices record which corpus they were cut from, and the
run refuses to start if that does not match the corpus you passed. Each tool's
own reported numbers are printed too, in separate rows labelled with whose
estimator produced them, and no gap between those rows is ever called a win.

**Sandbox anything you did not write.** Add `--isolate docker --image
senbon=IMAGE --image heretic=IMAGE` and each arm runs with no network, read-only
inputs, no capabilities and no credentials. Without it you get a warning, because
a third-party abliteration tool otherwise runs with your network and your keys.

**Stopping and starting is safe.** An arm is skipped only when a manifest agrees
with this run's tool, seed, model and budget *and* every artefact it declared is
present. An arm that exits cleanly having produced nothing counts as a failure and
leaves no manifest, so the next run retries it rather than inheriting the silence.

**Fewer than three seeds gets no verdict.** A spread from two points is arithmetic
dressed as statistics, and the report says so rather than naming a winner. A gap
smaller than the spread is reported as a tie.

**Adding another tool** is an adapter: how to invoke it, what proves it ran, where
it leaves a model, and how to read its own reported figures. See `ADAPTERS` in
`senbonzakura/bench.py`; it is data rather than logic, and a pull request adding
one is welcome.

