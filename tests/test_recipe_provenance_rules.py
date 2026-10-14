# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Every refusal the provenance, base identity and measured rules can make, checked in both implementations.

The writer (`senbonzakura.recipe`) and the reference applier (`recipes/apply_recipe.py`) each carry their own copy of
these rules, so a rule can drift in one of them without anything else noticing. Each defect below is applied to one
valid recipe, and both implementations must refuse it with the same code. A defect that only one of them catches is a
failure, and so is a valid recipe that either of them refuses.
"""

import copy
import json
import pathlib

import pytest

from senbonzakura import recipe as recipe_mod
from tests.test_recipe_roundtrip import load_applier

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "recipes" / "conformance"


def _valid_document():
    doc = json.loads((FIXTURE / "recipe.json").read_text(encoding="utf-8"))
    doc["measured"] = {
        "status": "measured",
        "instrument": {
            "scorer": "scorer/1",
            "corpus": {"id": "held-out", "revision": "rev-1"},
            "partition": "test",
            "seed": 7,
        },
        "refusal_before": {"point": 0.4, "n": 100, "ci": [0.3, 0.5]},
        "refusal_after": {"point": 0.0, "n": 100, "ci": [0.0, 0.04]},
        "kl": {"point": 0.02, "n": 64, "ci": [0.01, 0.03]},
        "capability": [
            {
                "name": "bench",
                "before": {"point": 0.5, "n": 150, "ci": [0.42, 0.58]},
                "after": {"point": 0.5, "n": 150, "ci": [0.42, 0.58]},
            }
        ],
        "self_invalidated": None,
        "limits": ["Measured on one corpus partition only."],
        "edit_sha256": recipe_mod.edit_sha256(doc["edit"]),
    }
    return doc


def _set(*path_and_value):
    *path, value = path_and_value

    def mutate(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = copy.deepcopy(value)

    return mutate


def _delete(*path):
    def mutate(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return mutate


def _measured(*path_and_value):
    return _set("measured", *path_and_value)


#: (name, mutation, the code both implementations must give). Each mutation starts from `_valid_document()`.
DEFECTS = [
    ("provenance-absent", _delete("provenance"), "provenance-missing"),
    ("provenance-not-an-object", _set("provenance", "x"), "provenance-invalid"),
    ("tool-not-an-object", _set("provenance", "tool", "x"), "provenance-invalid"),
    ("tool-without-name", _delete("provenance", "tool", "name"), "provenance-invalid"),
    ("tool-without-version", _delete("provenance", "tool", "version"), "provenance-invalid"),
    ("source-not-an-object", _set("provenance", "source", "x"), "provenance-invalid"),
    (
        "no-commit-and-no-reason",
        _set("provenance", "source", {"commit": None, "dirty": None, "via": None}),
        "provenance-invalid",
    ),
    (
        "commit-without-origin",
        _set("provenance", "source", {"commit": "abc1234", "dirty": False}),
        "provenance-invalid",
    ),
    ("dirty-not-a-flag", _set("provenance", "source", "dirty", "yes"), "provenance-invalid"),
    ("base-without-id", _delete("base", "id"), "base-invalid"),
    ("null-revision-without-reason", _delete("base", "revision_reason"), "base-invalid"),
    ("revision-blank", _set("base", "revision", "  "), "base-invalid"),
    ("measured-status-unknown", _set("measured", "status", "maybe"), "measured-invalid"),
    ("not-measured-without-why", _set("measured", {"status": "not_measured"}), "measured-invalid"),
    ("instrument-not-an-object", _measured("instrument", "x"), "measured-invalid"),
    ("scorer-missing", _measured("instrument", "scorer", ""), "measured-invalid"),
    ("partition-missing", _delete("measured", "instrument", "partition"), "measured-invalid"),
    ("corpus-not-an-object", _measured("instrument", "corpus", "x"), "measured-invalid"),
    ("corpus-unpinned", _measured("instrument", "corpus", {"id": "held-out"}), "measured-invalid"),
    ("seed-is-a-flag", _measured("instrument", "seed", True), "measured-invalid"),
    ("seed-negative", _measured("instrument", "seed", -1), "measured-invalid"),
    ("figure-not-an-object", _measured("refusal_before", "x"), "measured-invalid"),
    ("point-not-finite", _measured("refusal_before", "point", float("nan")), "measured-invalid"),
    ("figure-without-n", _delete("measured", "refusal_before", "n"), "measured-invalid"),
    ("n-is-a-flag", _measured("refusal_before", "n", True), "measured-invalid"),
    ("ci-wrong-length", _measured("refusal_before", "ci", [0.3, 0.4, 0.5]), "measured-invalid"),
    ("ci-excludes-point", _measured("refusal_before", "ci", [0.6, 0.9]), "measured-invalid"),
    ("rate-above-one", _measured("refusal_before", "ci", [0.3, 1.5]), "measured-invalid"),
    ("kl-interval-excludes-point", _measured("kl", "ci", [0.5, 0.6]), "measured-invalid"),
    ("self-invalidated-absent", _delete("measured", "self_invalidated"), "measured-invalid"),
    ("self-invalidated-blank", _measured("self_invalidated", " "), "measured-invalid"),
    ("limits-not-a-list", _measured("limits", "none"), "measured-invalid"),
    ("limits-empty", _measured("limits", []), "measured-invalid"),
    ("limits-blank-entry", _measured("limits", [""]), "measured-invalid"),
    ("capability-not-measured-without-why", _measured("capability", {"status": "not_measured"}), "measured-invalid"),
    ("capability-empty-list", _measured("capability", []), "measured-invalid"),
    ("capability-entry-not-an-object", _measured("capability", ["x"]), "measured-invalid"),
    ("capability-entry-unnamed", _measured("capability", [{"before": {}, "after": {}}]), "measured-invalid"),
    (
        "capability-figure-not-an-object",
        _measured("capability", [{"name": "b", "before": "x", "after": {}}]),
        "measured-invalid",
    ),
    (
        "capability-figure-without-n",
        _measured(
            "capability",
            [
                {
                    "name": "b",
                    "before": {"point": 0.5, "ci": [0.4, 0.6]},
                    "after": {"point": 0.5, "n": 9, "ci": [0.4, 0.6]},
                }
            ],
        ),
        "measured-invalid",
    ),
    ("bound-to-another-edit", _measured("edit_sha256", "0" * 64), "measured-mismatch"),
]


def _refusal_of_the_writer(doc):
    try:
        recipe_mod.validate_document(doc)
    except recipe_mod.RecipeError as e:
        return e.code
    return None


def _refusal_of_the_applier(doc, tmp_path, applier):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    try:
        recipe = applier.load_recipe(path)
        applier.check_base(recipe, FIXTURE / "model")
    except applier.RecipeError as e:
        return e.code
    return None


def test_a_valid_recipe_is_accepted_by_both_implementations(tmp_path):
    doc = _valid_document()
    assert _refusal_of_the_writer(doc) is None
    assert _refusal_of_the_applier(doc, tmp_path, load_applier()) is None


@pytest.mark.parametrize(("name", "mutate", "code"), DEFECTS, ids=[d[0] for d in DEFECTS])
def test_the_writer_refuses_each_defect_with_the_code_the_applier_gives(name, mutate, code, tmp_path):
    doc = _valid_document()
    mutate(doc)
    applier = load_applier()
    writer_code = _refusal_of_the_writer(copy.deepcopy(doc))
    applier_code = _refusal_of_the_applier(copy.deepcopy(doc), tmp_path, applier)
    assert writer_code == code, f"the writer gave {writer_code!r}"
    assert applier_code == code, f"the applier gave {applier_code!r}"


def test_a_capability_recorded_as_not_measured_is_accepted_by_both_implementations(tmp_path):
    doc = _valid_document()
    doc["measured"]["capability"] = {"status": "not_measured", "why": "no benchmark was run"}
    assert _refusal_of_the_writer(doc) is None
    assert _refusal_of_the_applier(doc, tmp_path, load_applier()) is None


def test_the_writer_refuses_a_document_that_is_not_a_recipe_object():
    with pytest.raises(recipe_mod.RecipeError) as refused:
        recipe_mod.validate_document(["not", "a", "recipe"])
    assert refused.value.code == "recipe-invalid"
