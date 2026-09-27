# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Three messages that stated a total in one clause and contradicted it in the next.

WHAT PROMPTED IT, 2026-09-27

`say.some_of` was written because `doctor` reported "7 failed to import" and then named six of them,
`convert` named five of N, and a JSON object with twenty keys was described by eight of its keys. A
reader who counts the names and gets a different number from the count they were handed has to work
out which of the two is lying.

WHY THIS FILE EXISTS SEPARATELY FROM THE HELPER'S OWN TESTS

A mutation pass put the bare slices back at all three call sites and the whole suite still passed.
`say.some_of` had thorough tests; nothing asserted that any of the three messages used it. That is
the same defect as the helper was written to fix, one level up: a correct function wired to a call
site nobody checks. `envsetup`'s cut GPU line and both hashing sites were guarded; these three were
not.

So these drive each message with more items than its limit and read what comes out.
"""
from __future__ import annotations

import json

import pytest

from senbonzakura import convert, dataset, doctor, say

#: Comfortably more than the largest limit any of the three sites uses, so every one of them has to
#: leave something out and say so.
MANY = [f"item{i:02d}" for i in range(20)]


def test_the_fixture_is_bigger_than_every_limit_under_test():
    """Otherwise a message that cut nothing would pass for one that counted correctly."""
    assert len(MANY) > 8


class TestDoctorsBrokenModuleList:
    """`check_converter` imports inside itself, so the source modules are what get patched."""

    def _detail(self, monkeypatch, broken):
        from senbonzakura import vendored

        monkeypatch.setattr(vendored, "find_script", lambda _n: "/x/convert_hf_to_gguf.py")
        monkeypatch.setattr(convert, "supported_architectures",
                            lambda *_a, **_k: ({"Qwen3ForCausalLM"}, broken, None))
        rows = doctor.check_converter()
        modules = [r for r in rows if r.name == "architecture modules"]
        assert modules, f"no row about the architecture modules was produced: {rows}"
        return modules[0].detail

    def test_it_counts_what_it_did_not_name(self, monkeypatch):
        detail = self._detail(monkeypatch, MANY)
        assert f"{len(MANY)} failed to import" in detail
        assert "and 14 more" in detail, (
            f"it names six modules after claiming twenty, with nothing to say so: {detail}")

    def test_a_short_list_is_named_in_full_with_no_suffix(self, monkeypatch):
        detail = self._detail(monkeypatch, ["one", "two"])
        assert "one, two" in detail and "more" not in detail


class TestConvertsBrokenModuleList:
    def test_it_counts_what_it_did_not_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(convert, "supported_architectures",
                            lambda *_a, **_k: ({"Qwen3ForCausalLM"}, MANY, None))
        monkeypatch.setattr(convert, "find_script", lambda _n: tmp_path / "convert.py")
        model = tmp_path / "m"
        model.mkdir()
        (model / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}',
                                           encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"\0" * 64)

        with pytest.raises(convert.ConvertError) as e:
            convert.preflight(model, tmp_path / "o.gguf", force=True, skip_arch_check=False)
        said = " ".join(str(e.value).split())
        assert f"could not import {len(MANY)}" in said
        assert "and 15 more" in said, (
            f"it names five modules after claiming twenty, with nothing to say so: {said}")


class TestTheKeysOfAnUnreadableTable:
    def test_it_counts_the_keys_it_did_not_name(self, tmp_path):
        """A twenty-key object described by eight keys reads as an eight-key object."""
        path = tmp_path / "rows.json"
        path.write_text(json.dumps(dict.fromkeys(MANY, 1)), encoding="utf-8")
        with pytest.raises(dataset.DatasetError) as e:
            dataset._read_table(path, None)
        said = " ".join(str(e.value).split())
        assert "and 12 more" in said, (
            f"it names eight keys of twenty with nothing to say so: {said}")


def test_the_helper_is_what_all_three_use():
    """Named directly, so a future site that hand-rolls the suffix is still a finding here.

    Not a source scan: the three tests above already hold the behaviour down. This asserts only that
    the shared definition exists under the name they call, so removing it fails loudly rather than
    leaving three copies of the same sentence to drift apart.
    """
    assert callable(say.some_of)
    assert say.some_of(MANY, 6).endswith("and 14 more")
