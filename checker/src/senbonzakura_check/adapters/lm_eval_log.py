# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""lm-evaluation-harness result files.

SHAPE, read out of `lm_eval/evaluator_utils.py::_to_eval_results` and `lm_eval/evaluator.py` on
2026-09-11 rather than from memory:

    {
      "results":          {task: {"acc,none": 0.51, "acc_stderr,none": 0.02, "alias": "..."}},
      "groups":           {...}                      # only when a group aggregates
      "group_subtasks":   {group: [task, ...]},
      "configs":          {task: {...}},
      "versions":         {task: 1.0},
      "n-shot":           {task: 5},
      "higher_is_better": {task: {"acc": true}},
      "n-samples":        {task: {"original": 1319, "effective": 1319}},
      "samples":          {...}                      # only with --log_samples
      "config":           {"model": ..., "model_args": ..., "batch_size": ..., "device": ...,
                           "use_cache": ..., "limit": ..., "bootstrap_iters": ...,
                           "gen_kwargs": ..., "random_seed": ..., "fewshot_seed": ...},
      "git_hash": ..., "date": ...
    }

TWO THINGS WORTH KNOWING BEFORE READING THE CODE

**A metric key carries its filter.** `metric_key = f"{metric},{filter_key}"`, so `acc,none` is
accuracy under the `none` filter and `acc,strict-match` is the same metric under a different
extraction rule. That comma is provenance, and splitting it away would throw away the one piece
of estimator information this format carries.

**`limit` is the field that decides whether a headline number means anything.** It is how many
documents were actually evaluated, and a run with `limit: 10` produces a figure that looks
exactly like a full run's in every other respect.
"""
from __future__ import annotations

#: Keys that, together, are not plausibly anything else. `results` alone is far too common.
_SIGNATURE = ("results", "configs", "versions")


class LmEvalAdapter:
    name = "lm-evaluation-harness"

    @staticmethod
    def detects(doc) -> bool:
        """Structural, not a version string.

        lm-eval writes no format marker into the file, so detection has to be by shape. Three
        co-occurring keys with `results` a mapping of mappings is specific enough in practice and
        deliberately not clever: a detector that guesses is how a file gets read as something it
        is not, and the checks then answer the wrong question confidently.
        """
        if not all(k in doc for k in _SIGNATURE):
            return False
        results = doc.get("results")
        return isinstance(results, dict) and all(
            isinstance(v, dict) for v in results.values())

    @staticmethod
    def normalise(doc) -> dict:
        cfg = doc.get("config") or {}
        results = doc.get("results") or {}
        higher = doc.get("higher_is_better") or {}
        n_samples = doc.get("n-samples") or {}

        metrics = {}
        for task, task_metrics in results.items():
            if not isinstance(task_metrics, dict):
                continue
            counts = n_samples.get(task) if isinstance(n_samples, dict) else None
            for key, value in task_metrics.items():
                if key == "alias" or not isinstance(value, (int, float)):
                    continue
                if key.startswith("_") or "_stderr" in key:
                    continue
                base, _, filter_key = key.partition(",")
                metrics[f"{task}.{key}"] = {
                    "metric": base,
                    "value": value,
                    # The filter is the only estimator-shaped thing this format carries, and it
                    # is real: `acc,none` and `acc,strict-match` are the same metric under
                    # different extraction rules and are not interchangeable figures.
                    "estimator": f"lm-eval filter {filter_key}" if filter_key else None,
                    "units": None,
                    "task": task,
                    "higher_is_better": (higher.get(task) or {}).get(base)
                    if isinstance(higher.get(task), dict) else None,
                    "n": (counts or {}).get("effective") if isinstance(counts, dict) else None,
                    "stderr": task_metrics.get(f"{base}_stderr,{filter_key}"),
                }

        return {
            "model": cfg.get("model"),
            "model_args": cfg.get("model_args"),
            "tasks": sorted(results),
            "metrics": metrics,
            # `limit` truncates the evaluation set. A figure from `limit: 10` is indistinguishable
            # from a full run's in every other field, so it is lifted where a check can see it.
            "limit": cfg.get("limit"),
            "few_shot": doc.get("n-shot"),
            "random_seed": cfg.get("random_seed"),
            "bootstrap_iters": cfg.get("bootstrap_iters"),
            "harness_version": doc.get("git_hash"),
            "date": doc.get("date"),
            # lm-eval applies a chat template only when asked, and records the request in
            # `model_args`/`gen_kwargs` rather than as a field of its own, so there is nothing
            # honest to lift here. Left absent rather than guessed at: the chat-template check
            # skips on an artefact that says nothing, which is the correct outcome.
        }
