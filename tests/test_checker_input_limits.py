# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The checker reads other people's files, so it has to survive hostile ones.

`SECURITY.md` says it in those words: "Code execution, file reads or file writes outside the
paths you pointed it at, including through a crafted ... result file. This tool loads other
people's files, so this is the category that matters most." `REPRODUCING.md` then invites exactly
that, telling people to point it at somebody else's lm-evaluation-harness and Inspect output.

Until 2026-09-25 both ingestion points read an entire caller-supplied path into memory and caught
only OSError and JSONDecodeError. A deeply nested document raises RecursionError, which is
neither, so a few kilobytes produced a traceback instead of a refusal, and there was no size
ceiling at all.

These tests are the control. A refusal nobody exercises is a claim.
"""
import json

import pytest
from senbonzakura_check.registry import (
    MAX_ARTEFACT_BYTES,
    MAX_ARTEFACT_DEPTH,
    ArtefactTooLargeError,
    read_json_bounded,
)


def test_an_ordinary_artefact_still_reads(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"metric": "refusal_rate", "value": 0.025}), encoding="utf-8")
    assert read_json_bounded(p) == {"metric": "refusal_rate", "value": 0.025}


def test_a_file_past_the_size_cap_is_refused_before_it_is_read(tmp_path):
    """The cap is checked with stat, so an over-large file is never held in memory at all."""
    p = tmp_path / "big.json"
    p.write_text("[" + "0," * 5000 + "0]", encoding="utf-8")
    with pytest.raises(ArtefactTooLargeError) as e:
        read_json_bounded(p, max_bytes=100)
    assert "100" in str(e.value), "the refusal has to name the limit it applied"


def test_a_deeply_nested_document_is_refused_rather_than_crashing(tmp_path):
    """The hazard a size cap does not cover: small on disk, unbounded in the parser."""
    p = tmp_path / "deep.json"
    p.write_text("[" * 400 + "]" * 400, encoding="utf-8")
    with pytest.raises(ArtefactTooLargeError):
        read_json_bounded(p, max_depth=50)


def test_nesting_at_the_limit_is_allowed(tmp_path):
    """The boundary, so the guard cannot quietly become stricter than it says."""
    p = tmp_path / "edge.json"
    p.write_text("[" * 10 + "]" * 10, encoding="utf-8")
    assert read_json_bounded(p, max_depth=10) is not None


def test_a_recursion_error_becomes_a_refusal_not_a_traceback(tmp_path):
    """Nesting deep enough to break the PARSER, which is the case that used to escape entirely.

    `json.loads` raises RecursionError before any depth check of ours can run, and RecursionError
    is not an OSError and not a JSONDecodeError, so both call sites let it through as a traceback.
    """
    p = tmp_path / "bomb.json"
    p.write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    with pytest.raises(ArtefactTooLargeError):
        read_json_bounded(p)


def test_the_cli_reports_an_over_large_file_as_a_refusal(tmp_path, monkeypatch):
    """End to end through the CLI's own reader, which must answer (None, why) rather than raise."""
    import functools

    from senbonzakura_check import cli, loaders, registry

    # THE CAP IS PATCHED WHERE THE LOADER CALLS IT, not on the module constant. `read_json_bounded`
    # binds its default at definition time, so patching `MAX_ARTEFACT_BYTES` leaves the real 64 MB
    # cap in force and the file is refused for being unrecognised: the assertion below would pass
    # on the wrong reason. Asserting on the size refusal's own wording is what makes this a test.
    monkeypatch.setattr(loaders, "read_json_bounded",
                        functools.partial(registry.read_json_bounded, max_bytes=10))
    p = tmp_path / "big.json"
    p.write_text(json.dumps({"metric": "x", "value": 1}), encoding="utf-8")
    doc, why = cli.read_artefact(p)
    assert doc is None and why, "an over-large file must come back as a refusal with a reason"
    assert why.startswith("this tool declines to read it") and "at most 10" in why, (
        f"refused, but not by the size cap: {why}")


def test_the_declared_limits_are_sane():
    """Guards against someone 'fixing' a failing test by setting a limit to infinity."""
    assert 1024 <= MAX_ARTEFACT_BYTES <= 1024 * 1024 * 1024
    assert 10 <= MAX_ARTEFACT_DEPTH <= 1000
