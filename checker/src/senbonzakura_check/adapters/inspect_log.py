# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Inspect eval logs.

SHAPE, read out of `src/inspect_ai/log/_log.py` on 2026-09-11: the `EvalLog`, `EvalResults`,
`EvalScore` and `EvalMetric` models, not from memory.

    {
      "version": 2,
      "status": "started" | "success" | "cancelled" | "error",
      "eval":    {"run_id", "created", "task", "task_version", "dataset", "model",
                  "model_args", "config", "revision", ...},
      "plan":    {...},
      "results": {"total_samples", "completed_samples", "logged_samples",
                  "scores": [{"name", "scorer", "reducer", "scored_samples",
                              "unscored_samples", "params",
                              "metrics": {name: {"name", "group", "value", "params"}}}],
                  "headline": {...}},
      "stats":   {...},
      "error":   {...} | null,
      "samples": [...] | null
    }

WHY THIS FORMAT IS THE MORE INFORMATIVE OF THE TWO

**Inspect records the scorer beside every metric.** `EvalScore.scorer` is the name of the
function that produced the number, which is exactly the estimator this project spent a withdrawal
learning to record. So an Inspect log arrives with the provenance the 2026-08-05 incident was
about, and the adapter's job is to carry it across rather than to invent it.

**`status` and `completed_samples` are load bearing.** A log can be `success` with fewer
completed samples than total, because `--fail-on-error` and early stopping both truncate. A
headline metric computed over a truncated run looks exactly like one computed over a full run,
and this is the field that tells them apart.
"""
from __future__ import annotations


class InspectAdapter:
    name = "inspect"

    @staticmethod
    def detects(doc) -> bool:
        """`version` plus an `eval` block naming a task. Inspect writes both on every log.

        Checked structurally rather than on `version` alone, which is a field name common enough
        to appear anywhere.
        """
        spec = doc.get("eval")
        return (
            isinstance(spec, dict)
            and "task" in spec
            and "version" in doc
            and isinstance(doc.get("version"), int)
        )

    @staticmethod
    def normalise(doc) -> dict:
        spec = doc.get("eval") or {}
        results = doc.get("results") or {}
        scores = results.get("scores") or []

        metrics = {}
        for score in scores:
            if not isinstance(score, dict):
                continue
            scorer = score.get("scorer")
            score_name = score.get("name")
            for metric_name, metric in (score.get("metrics") or {}).items():
                if not isinstance(metric, dict) or not isinstance(
                        metric.get("value"), (int, float)):
                    continue
                key = f"{score_name}.{metric_name}" if score_name else str(metric_name)
                metrics[key] = {
                    "metric": metric.get("name", metric_name),
                    "value": metric["value"],
                    # THE FIELD THAT MAKES THIS FORMAT WORTH READING. `scorer` names the function
                    # that produced the number, which is the provenance a bare `kl` lacks.
                    "estimator": scorer,
                    "units": None,
                    "task": spec.get("task"),
                    "n": score.get("scored_samples"),
                    "unscored": score.get("unscored_samples"),
                    "reducer": score.get("reducer"),
                    "group": metric.get("group"),
                }

        total = results.get("total_samples")
        completed = results.get("completed_samples")
        return {
            "model": spec.get("model"),
            "model_args": spec.get("model_args"),
            "tasks": [spec["task"]] if spec.get("task") else [],
            "metrics": metrics,
            "status": doc.get("status"),
            "total_samples": total,
            "completed_samples": completed,
            # Lifted into one field because a check reading two numbers and comparing them is a
            # check that has to know this format. A truncated run is the thing worth noticing,
            # and it is noticeable here regardless of which harness wrote the file.
            "truncated": (
                None if total is None or completed is None else completed < total),
            "dataset": spec.get("dataset"),
            "limit": (spec.get("config") or {}).get("limit"),
            "epochs": (spec.get("config") or {}).get("epochs"),
            "harness_version": doc.get("version"),
            "date": spec.get("created"),
            "error": doc.get("error"),
        }
