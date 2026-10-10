#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Count which checkpoints keep a prediction head inside the decoder stack, and which keep it apart.

FOUR DIFFERENT ANSWERS WERE REPORTED TO THIS ONE QUESTION over three days (25, then 15, then 13,
then 15 again), each from a hand-written query over the corpus, and each read as a finding about
checkpoints rather than as a reading of whatever that query happened to match. This file exists so
the count has one implementation and a rerun is a command rather than a fresh regex.

The distinction matters to an editor and not only to a catalogue. A head stored INSIDE the stack
occupies trailing block indices, so a pass that edits "every block" reaches it, and editing a
prediction head changes what the model predicts next rather than how it behaves. A head in its own
stack is simply not visited. One flag covering both conventions while reporting on both is the
defect the two cases exist to separate.

Two traps are handled here because both produced a published wrong number:

1. `num_hidden_layers` IS NOT ALWAYS A TOP-LEVEL KEY. On a multimodal checkpoint the decoder's own
   count sits under `text_config`, so reading the top level returns `None`. A count of 13 came from
   treating those as not-in-stack rather than as not-read.
2. A HEAD CAN BE PRESENT WITH NO CONFIG KEY. Nine records carry `eh_proj`, `enorm` and `hnorm` in
   their tensor patterns while declaring neither count field, so asking the config alone whether a
   head exists misses all of them.

And the stem compared against the declared count has to be the LANGUAGE stem, not the longest one:
a vision tower of 47 blocks says nothing about a prediction head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The three tensors a multi-token prediction head brings with it. Their presence is the second
#: witness, used where the config declares no count.
HEAD_MARKERS = ("eh_proj", "enorm", "hnorm")

#: Where a multimodal config hides the decoder's own layer count, in preference order.
NESTED_CONFIGS = ("text_config", "language_config", "llm_config")

#: Stem name fragments that belong to something other than the decoder: a vision tower, or the
#: n-gram embedders LongCat carries. Counting these as the decoder stack is how a 47-block image
#: encoder became evidence about a prediction head.
NOT_THE_DECODER = ("vision", "visual", "resblocks", "ngram")


def declared_layers(config: dict) -> tuple[int | None, str | None]:
    """The decoder's declared block count, and where it was found."""
    if config.get("num_hidden_layers") is not None:
        return config["num_hidden_layers"], "top level"
    for nested in NESTED_CONFIGS:
        value = (config.get(nested) or {}).get("num_hidden_layers")
        if value is not None:
            return value, nested
    return None, None


def decoder_stem(counts: dict) -> tuple[str | None, int | None]:
    """The longest stem that is actually the decoder stack."""
    candidates = {
        stem: n for stem, n in counts.items()
        if not any(word in stem for word in NOT_THE_DECODER)
        # A stem with `.{i}.` inside it is a per-block sub-pattern, not the stack itself, and a
        # separate head stack is the thing being distinguished rather than a candidate for it.
        and ".{i}." not in stem and not stem.endswith(".mtp.layers")}
    if not candidates:
        return None, None
    stem, n = max(candidates.items(), key=lambda kv: kv[1])
    return stem, n


def separate_head_stems(counts: dict) -> list[str]:
    return [stem for stem in counts if stem.endswith("mtp.layers") or "mtp_layers" in stem]


def classify(record: dict) -> tuple[str, dict]:
    """One of `in_stack`, `own_stack`, `undetermined` or `no_head`, with what the call rests on."""
    config = record.get("config") or {}
    tensors = record.get("tensors") or {}
    counts = tensors.get("layer_counts_by_stem") or {}
    patterns = " ".join((tensors.get("pattern_layer_indices") or {}).keys())

    by_config = [k for k in ("num_nextn_predict_layers", "mtp_num_hidden_layers") if config.get(k)]
    by_marker = [m for m in HEAD_MARKERS if m in patterns]
    if not (by_config or by_marker):
        return "no_head", {}
    witness = {"config_keys": by_config, "markers": by_marker}

    separate = separate_head_stems(counts)
    if separate:
        return "own_stack", {**witness, "reason": f"separate stem {separate}"}
    if not counts:
        return "undetermined", {**witness, "reason": "no layer counts in the record"}

    nhl, where = declared_layers(config)
    stem, top = decoder_stem(counts)
    if nhl is None or top is None:
        return "undetermined", {**witness, "reason": "no declared layer count found, nested or not"}
    detail = {**witness, "declared": nhl, "declared_from": where, "stem": stem, "stem_count": top}
    if top > nhl:
        return "in_stack", {**detail, "extra": top - nhl}
    return "own_stack", {**detail, "reason": "decoder stem does not run past the declared count"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpus", type=Path,
                    help="the corpus directory holding one JSON record per model")
    ap.add_argument("--json", action="store_true", help="emit the full classification as JSON")
    args = ap.parse_args(argv)

    models = args.corpus / "models" if (args.corpus / "models").is_dir() else args.corpus
    records = sorted(models.glob("*.json"))
    if not records:
        print(f"no records under {models}, so nothing was counted. This is not a finding about "
              f"prediction heads.", file=sys.stderr)
        return 2

    buckets: dict[str, list] = {"in_stack": [], "own_stack": [], "undetermined": [], "no_head": []}
    skipped = {"partial": 0, "synthetic": 0}
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            skipped["partial"] += 1
            continue
        if record.get("synthetic_fixture"):
            skipped["synthetic"] += 1
            continue
        verdict, detail = classify(record)
        buckets[verdict].append((record.get("model") or path.stem, detail))

    if args.json:
        print(json.dumps({k: dict(v) for k, v in buckets.items()}, indent=2,
                         sort_keys=True))
        return 0

    print(f"{len(records)} records, {skipped['partial']} partial and {skipped['synthetic']} "
          f"synthetic skipped\n")
    for verdict in ("in_stack", "own_stack", "undetermined"):
        rows = sorted(buckets[verdict])
        print(f"{verdict.upper().replace('_', ' ')}: {len(rows)}")
        for name, detail in rows:
            where = detail.get("declared_from")
            nested = "  (count was NESTED)" if where and where != "top level" else ""
            if verdict == "in_stack":
                print(f"   {name}  declared={detail['declared']} stem={detail['stem_count']} "
                      f"extra={detail['extra']}{nested}")
            else:
                print(f"   {name}  {detail.get('reason', '')}{nested}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
