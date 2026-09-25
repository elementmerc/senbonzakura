# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A line number in a refusal has to be a line number in the file.

THE DEFECT. `_check_gradeable` enumerated the list of PARSED rows and called the index a line
number. That list has already dropped blank lines, unparseable ones, non-objects and rows missing
a required key, so the count drifts by exactly the number of earlier problems the file had. The
one file where the number is wrong is the file that had other problems too, which is the file
somebody is already trying to fix, and they were sent to the wrong row of it.

Every other message in the same function uses the true line number, so the two kinds of problem in
one refusal disagreed about how to count the same file.
"""
import pathlib

from senbonzakura import probe

MANIFEST = """[probe]
name = "t"
measures = "arithmetic"
task = "numeric"
items = "items.jsonl"
version = "1"

[content]
class = "benign"

[source]
url = "https://example.invalid/x"
licence = "MIT"
author = "somebody"
"""


def _probe_dir(tmp_path, items):
    (tmp_path / "probe.toml").write_text(MANIFEST, encoding="utf-8")
    (tmp_path / "items.jsonl").write_text(items, encoding="utf-8")
    return tmp_path


def _gradeability_problem(problems):
    return next((p for p in problems if "cannot read" in p), None)


def test_the_line_number_survives_earlier_skipped_lines(tmp_path):
    """Three lines are dropped before the first row, so a naive index would be three short."""
    items = (
        "\n"                                              # line 1: blank, skipped
        "not json\n"                                      # line 2: unparseable, skipped
        "[1, 2]\n"                                        # line 3: not an object, skipped
        '{"problem": "a", "reference": "4"}\n'            # line 4: fine
        '{"problem": "b", "reference": "no number"}\n'    # line 5: ungradeable
    )
    problems = probe.problems_with(_probe_dir(tmp_path, items))
    found = _gradeability_problem(problems)
    assert found is not None, problems
    assert "line 5" in found, (
        f"the ungradeable item is on line 5 of the file and the refusal said: {found}")
    # The number it used to print, which is its position among the rows that parsed.
    assert "line 2" not in found


def test_a_row_missing_a_key_also_shifts_nothing(tmp_path):
    items = (
        '{"problem": "a"}\n'                              # line 1: no reference, skipped
        '{"problem": "b", "reference": "7"}\n'            # line 2: fine
        '{"problem": "c", "reference": "nothing"}\n'      # line 3: ungradeable
    )
    found = _gradeability_problem(probe.problems_with(_probe_dir(tmp_path, items)))
    assert found is not None
    assert "line 3" in found


def test_several_ungradeable_rows_are_listed_and_counted(tmp_path):
    items = "".join(f'{{"problem": "p{i}", "reference": "none"}}\n' for i in range(8))
    found = _gradeability_problem(probe.problems_with(_probe_dir(tmp_path, items)))
    assert found is not None
    assert found.startswith("8 item(s)")
    # Five shown, three accounted for rather than silently dropped from the message.
    assert "lines 1, 2, 3, 4, 5" in found
    assert "3 more" in found


def test_one_ungradeable_row_is_not_called_lines(tmp_path):
    items = ('{"problem": "a", "reference": "4"}\n'
             '{"problem": "b", "reference": "none"}\n')
    found = _gradeability_problem(probe.problems_with(_probe_dir(tmp_path, items)))
    assert found is not None
    assert "(line 2)" in found


def test_a_clean_probe_still_loads_and_carries_its_items(tmp_path):
    """`load` consumes the same rows, so pairing them with line numbers must not reach it."""
    items = ('{"problem": "a", "reference": "4"}\n'
             '{"problem": "b", "reference": "7"}\n')
    loaded = probe.load(_probe_dir(tmp_path, items))
    assert [r["problem"] for r in loaded.items] == ["a", "b"]
    assert all(isinstance(r, dict) for r in loaded.items), (
        "a loaded probe's items are the rows themselves; the line numbers are an internal "
        "pairing for the refusal message and must not leak into what a run scores")


def test_the_containment_check_is_untouched(tmp_path):
    """The neighbouring guard in the same function, asserted here so this file cannot loosen it."""
    outside = tmp_path.parent / f"{tmp_path.name}-evil"
    outside.mkdir(exist_ok=True)
    (outside / "items.jsonl").write_text('{"problem": "a", "reference": "4"}\n', encoding="utf-8")
    (tmp_path / "probe.toml").write_text(
        MANIFEST.replace('items = "items.jsonl"',
                         f'items = "../{outside.name}/items.jsonl"'), encoding="utf-8")
    problems = probe.problems_with(tmp_path)
    assert any("points outside the probe directory" in p for p in problems), problems


def test_line_numbers_are_one_based(tmp_path):
    """The first line of a file is line 1, matching every other message this module writes."""
    found = _gradeability_problem(probe.problems_with(
        _probe_dir(tmp_path, '{"problem": "a", "reference": "none"}\n')))
    assert found is not None
    assert "(line 1)" in found


def test_the_probe_module_reads_its_own_directory_only(tmp_path):
    """A sanity check that the fixture above is a probe at all, so the tests are not vacuous."""
    items = '{"problem": "a", "reference": "4"}\n'
    assert probe.problems_with(_probe_dir(tmp_path, items)) == []
    assert pathlib.Path(tmp_path / "items.jsonl").is_file()
