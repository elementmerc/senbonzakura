# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A published GGUF file, read as a claim about itself.

WHAT THIS ADAPTER IS FOR, WHICH IS NOT WHAT THE OTHER THREE ARE FOR

The other adapters read a harness's RESULT file: a number somebody measured. A GGUF carries no
measurement at all. What it carries is a set of claims about what the file is, and those claims are
checkable against each other, which is the whole value here. `quantisation.md` states the finding
this exists to surface: **two files can both say `Q4_K_M` and be measurably different models.**

So `metrics` comes back empty and that is correct rather than a gap. The checks that read this
artefact are about agreement between a label and its contents, not about a figure and its interval.

THE THREE DISAGREEMENTS IT CAN SEE

1. **The filename against the file's own `general.file_type`.** A file renamed on the way out of a
   pipeline, or quantised twice and labelled once, disagrees here.
2. **The declared recipe against the tensors actually inside.** This is the interesting one. A
   recipe like `Q4_K_M` is a mixture by design, so a mixture is not a defect; what a reader can be
   misled by is the recipe's own nominal type being a small minority of the file. Measured on a
   real published file during this adapter's development:
   `unsloth/SmolLM2-135M-Instruct-Q4_K_M.gguf` carries 272 tensors of which **16 are Q4_K and 166
   are Q5_0**. Nothing is wrong with that file; a reader who assumes the name describes the bulk of
   its weights is wrong about it, and no existing tool tells them.
3. **A missing chat template.** Every runtime that loads a GGUF reads the template from the file's
   own metadata, so a file without one is handed raw text where the model expects turn markers.

WHY THE NOMINAL SHARE IS REPORTED AND NOT CHECKED, WHICH IS A DESIGN THAT CHANGED UNDER MEASUREMENT

The plan was a check firing when the recipe's nominal type is a minority of the file. It was
measured against two real published files before being written, and it would have fired on both:

    unsloth/SmolLM2-135M-Instruct-Q4_K_M.gguf   Q4_K is  16/272 =  5.9%   dominant: Q5_0 at 61.0%
    unsloth/SmolLM2-135M-Instruct-Q6_K.gguf     Q6_K is  30/272 = 11.0%   dominant: Q8_0 at 66.5%

Neither file is defective. `llama-quantize` falls back to other types for tensors whose dimensions
do not suit a K-quant block, and on a small model that is most of them. **So a nominal-minority
check would fire on essentially every real GGUF, which is cry-wolf on the format this adapter
exists to read**, and a check that fires on everything is switched off within a fortnight and then
no longer protects anything.

The fact is still worth a reader's attention, so it is reported as a field and not as a finding:
the recipe name genuinely does not describe the bulk of the file. That belongs in a sentence
somebody reads, not in an alarm. Distinguishing a legitimate fallback mixture from a wrong one
would need a table of each recipe's expected mixture, which means reverse-engineering
`llama-quantize`, and that is out of scope here.

WHAT IS CHECKED, THEREFORE, IS ONLY WHAT DISCRIMINATES

Two things: a filename that disagrees with the file's own metadata, and a file using tensor types
this build cannot name, which makes any statement about its precision unverified rather than
wrong. Both are properties a sound file does not have.

A missing chat template is deliberately NOT a finding on a standalone GGUF. A base model is
supposed to have none, nothing in the header reliably says whether this is a base or an instruct
model, and firing on absence would flag every base model in the field. The existing
`a-chat-template-lost-in-conversion` check fires on a DISAGREEMENT because a conversion record has
both sides; a published file on its own has one side, and one side is not a disagreement.
"""
from __future__ import annotations

#: The loader stamps this. A GGUF is binary, so it cannot arrive as a parsed JSON document the way
#: the other three artefacts do, and `detect` only ever sees dicts. The loader reads the header and
#: presents it as a document under this key, which keeps the one-dispatcher contract intact instead
#: of adding a second detection path for binary inputs.
FORMAT_KEY = "artefact_format"
FORMAT_VALUE = "gguf"

#: The per-tensor type a whole-file recipe is named after. `Q4_K_M` is a mixture whose nominal type
#: is `Q4_K`; the trailing size letter is the variant, not a different type. Derived by stripping
#: the variant suffix rather than by a second lookup table, so a recipe this build has never heard
#: of still resolves to something sensible.
_VARIANT_SUFFIXES = ("_XXS", "_XS", "_S", "_M", "_L")


def nominal_type(file_type: str | None) -> str | None:
    """The per-tensor type a recipe name is built on, or None.

    `Q4_K_M` -> `Q4_K`, `Q3_K_L` -> `Q3_K`, `Q6_K` -> `Q6_K`, `IQ2_XXS` -> `IQ2_XXS`.

    The IQ family is left alone: its names ARE the tensor type, so stripping `_XXS` from `IQ2_XXS`
    would invent a type called `IQ2` that no file contains and then report every IQ file as
    disagreeing with itself.
    """
    if not isinstance(file_type, str) or not file_type:
        return None
    if file_type.startswith("IQ"):
        return file_type
    for suffix in _VARIANT_SUFFIXES:
        if file_type.endswith(suffix):
            return file_type[: -len(suffix)]
    return file_type


class GgufAdapter:
    name = "gguf file"

    @staticmethod
    def detects(doc) -> bool:
        """Exact, because the loader stamps it.

        No structural guessing here, unlike the harness adapters: a binary file either parsed as a
        GGUF header or it did not, and the loader already decided. A detector that tried to guess
        would be guessing about a document this package itself constructed.
        """
        return doc.get(FORMAT_KEY) == FORMAT_VALUE

    @staticmethod
    def normalise(doc) -> dict:
        header = doc.get("header") or {}
        census = header.get("tensor_type_census") or {}
        total = sum(census.values()) if census else 0

        declared = header.get("file_type")
        nominal = nominal_type(declared)
        nominal_count = census.get(nominal, 0) if nominal else 0
        # None rather than 0.0 when there is nothing to divide by, so a check reading this cannot
        # mistake "no tensors found" for "none of them match".
        nominal_share = (nominal_count / total) if total else None

        unknown = header.get("unknown_tensor_types") or {}
        # The type carrying the most tensors, which on a real file is routinely not the one the
        # recipe is named after. None when the census is empty rather than a guess.
        dominant = max(census, key=census.get) if census else None

        claimed_by_name = doc.get("claimed_quant_from_name")
        # Three-valued on purpose. True and False are both findings; None means the filename made
        # no claim, and a file whose name says nothing is not a file whose name disagrees.
        name_matches_metadata = (
            None if not (claimed_by_name and declared) else claimed_by_name == declared)

        return {
            "artefact_kind": "gguf",
            "model": header.get("name"),
            "architecture": header.get("architecture"),
            "gguf_version": header.get("version"),
            "tensor_count": header.get("tensor_count"),
            "declared_file_type": declared,
            "declared_file_type_number": header.get("file_type_number"),
            "claimed_quant_from_name": claimed_by_name,
            "name_matches_metadata": name_matches_metadata,
            "nominal_tensor_type": nominal,
            "nominal_tensor_share": nominal_share,
            # Reported so a reader can see what the file is mostly made of, which the recipe name
            # does not tell them. See the docstring: information, not an alarm.
            "dominant_tensor_type": dominant,
            "dominant_tensor_share": (census[dominant] / total) if dominant and total else None,
            "distinct_tensor_types": len(census),
            "tensor_type_census": census,
            "unknown_tensor_types": unknown,
            # A COUNT RATHER THAN THE MAPPING, because the rule vocabulary has no numeric
            # comparison for a fixed path and `truthy` on a count says exactly what is meant: some
            # tensors could not be named, so the census is incomplete.
            "unknown_tensor_type_count": sum(unknown.values()),
            "target_carries_chat_template": header.get("has_chat_template"),
            "quantization_version": header.get("quantization_version"),
            # An importance matrix leaves no mark in the header. llama.cpp records nothing about
            # one, so a file built with `--imatrix` and one built without it are identical here.
            # Reported as unknown rather than absent, because absent would read as "no matrix was
            # used" and that is a claim this file cannot support either way.
            "imatrix_recorded": None,
            # NO MEASUREMENTS, and this is the honest answer rather than an empty slot. A GGUF
            # holds no figure anybody measured, so every check gated on `metrics` skips, which is
            # what should happen.
            "metrics": {},
            "source_path": doc.get("source_path"),
        }


__all__ = ["FORMAT_KEY", "FORMAT_VALUE", "GgufAdapter", "nominal_type"]
