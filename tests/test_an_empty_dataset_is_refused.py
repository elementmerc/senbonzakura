# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A source that yields no rows must refuse, not measure nothing and report a number.

WHY THIS EXISTS, and it is not hypothetical. On 2026-09-25, while preparing an overnight run, the
ROG was found holding a Hugging Face cache entry for the evaluation track that contained a bare
`refs/main` and nothing behind it: no snapshot, no blobs, 12 KB in total. A directory listing says
the track is cached. There is no track.

That is the shape this project keeps finding in its own code, arriving from outside it: a thing
that looks present and is not, where the dangerous outcome is not a crash but a clean-looking zero.
A refusal rate measured over no prompts is 0.0%, and 0.0% is the number this tool exists to
produce. If an empty source reached the scorer, the run would report the best possible result for
the worst possible reason.

`resolve` and `resolve_labelled` both carry a guard for it. Neither guard was covered by a test,
which is why this file exists: an untested refusal is a refusal nobody has watched fire, and the
next person to restructure the function has nothing telling them it mattered.
"""
import json

import pytest

from senbonzakura import dataset


def _empty_jsonl(tmp_path, name="prompts.jsonl"):
    p = tmp_path / name
    p.write_text("", encoding="utf-8")
    return p


def test_an_empty_file_is_refused_rather_than_measured(tmp_path):
    with pytest.raises(dataset.DatasetError, match="is empty"):
        dataset.resolve(str(_empty_jsonl(tmp_path)))


def test_the_refusal_says_why_it_matters(tmp_path):
    """"Empty" alone reads as a file-format complaint. The point is that nothing can be measured."""
    with pytest.raises(dataset.DatasetError, match="nothing to measure"):
        dataset.resolve(str(_empty_jsonl(tmp_path)))


def test_a_file_of_only_blank_lines_is_refused_more_precisely(tmp_path):
    """Whitespace is not data, and this refusal is better than the empty one rather than the same.

    A file with rows whose text is blank is a different fault from a file with no rows: the reader
    found the rows and the column is wrong or the data is. So it says which column it looked in and
    offers `--text-column`, instead of the generic emptiness message. Asserted here because the
    first version of this test expected "is empty" and the real behaviour was more useful, which is
    worth pinning so a later simplification cannot quietly coarsen it.
    """
    p = tmp_path / "prompts.txt"
    p.write_text("\n   \n\t\n\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="every one of them is blank"):
        dataset.resolve(str(p))


def test_a_slice_that_selects_nothing_is_refused_separately(tmp_path):
    """A different fault with a different fix, so it gets its own sentence rather than "empty"."""
    p = tmp_path / "prompts.txt"
    p.write_text("one\ntwo\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="selected no rows"):
        dataset.resolve(f"{p}::train[0:0]")


def test_an_empty_labelled_source_is_refused(tmp_path):
    """The two-arm path has its own reader and therefore its own way to return nothing."""
    p = tmp_path / "labelled.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="is empty"):
        dataset.resolve_labelled(str(p))


def test_a_directory_with_no_rows_is_refused(tmp_path):
    """The case the ROG actually presented: the container exists and holds nothing usable."""
    d = tmp_path / "track"
    d.mkdir()
    (d / "rows.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(dataset.DatasetError):
        dataset.resolve(str(d / "rows.jsonl"))


def test_a_source_with_rows_still_resolves(tmp_path):
    """The guard must not be so eager that it refuses real data: without this, a test file that
    only proves refusals can be satisfied by a function that refuses everything.
    """
    p = tmp_path / "prompts.jsonl"
    p.write_text("".join(json.dumps({"text": t}) + "\n" for t in ("alpha", "beta")),
                 encoding="utf-8")
    assert dataset.resolve(str(p)) == ["alpha", "beta"]


def test_the_refusal_names_the_spec_the_user_typed(tmp_path):
    """So somebody with several sources on one command line knows which one was empty."""
    p = _empty_jsonl(tmp_path, "the-one-that-was-empty.jsonl")
    with pytest.raises(dataset.DatasetError, match="the-one-that-was-empty"):
        dataset.resolve(str(p))
