# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""`senbonzakura track --audit` says what it examined, including the check it declined to run.

WHAT PROMPTED IT, 2026-10-01

A CLI surface sweep item: "`track --audit` prints one word on success." It printed
`TRACK_AUDIT_OK <path>` and returned, with the partition detail appearing only on the failure
path and only on stderr.

The reason that is worse than terse output: the strongest check this command has is the one it
silently skips. `check`'s strata check needs `--labels`, and `audit`'s own docstring has said since
it was written that "a hand-built track passes an audit that never asked the question". So the
pass line was evidence of a weaker audit than a reader would take it for, and `TRACK_AUDIT_OK` is
exactly the string somebody pastes into a release note.

This is the sweep's recurring shape in its purest form: a verdict read as broader than the question
the instrument asked. The fix is not a better check, it is the output admitting which checks ran.

WHY THE MARKER LINE IS ASSERTED SEPARATELY

`TRACK_AUDIT_OK` is a machine interface: `say.MARKER` exempts it from wrapping, and CI greps for
it. A readability fix that moved the path off that line, or indented it, would be a change to the
grep dressed as prose. So its exact shape is held as well as the new detail.
"""
from __future__ import annotations

import pytest

from senbonzakura import track

#: A track that passes every check, with two strata per side, as `load_partitions` returns one.
#: The numbers differ per partition on purpose: a summary that prints the same count everywhere
#: would pass a test built on equal partitions.
PARTITIONS = {
    "harmful": {"fit": [f"harmful fit {i}" for i in range(7)],
                "search": [f"harmful search {i}" for i in range(5)],
                "measure": [f"harmful measure {i}" for i in range(11)]},
    "harmless": {"fit": [f"harmless fit {i}" for i in range(6)],
                 "search": [f"harmless search {i}" for i in range(4)],
                 "measure": [f"harmless measure {i}" for i in range(9)]},
}


def _labels():
    """A label per prompt, two strata per side, every stratum present in measure.

    Built from `PARTITIONS` rather than written out, so a change to the fixture above cannot leave
    this silently labelling prompts that are no longer there, which would make the strata check
    pass for the wrong reason.
    """
    labels = {}
    for side, parts in PARTITIONS.items():
        for rows in parts.values():
            for index, row in enumerate(rows):
                labels[track.normalise(row)] = f"{side}-{index % 2}"
    return labels


@pytest.fixture
def a_track(tmp_path, monkeypatch):
    """`--audit` driven against stubbed partitions.

    The boundary slicing is `test_track.py`'s subject and needs a table-writing backend; what is
    under test here is what the command says about a track it has read, so `load_partitions` is
    the seam. The directory is real because `_refuse_build_flags_under_audit` and the manifest
    refusals look at the path.
    """
    out = tmp_path / "tr"
    out.mkdir()
    (out / "track.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(track, "load_partitions",
                        lambda path: (PARTITIONS["harmful"], PARTITIONS["harmless"]))
    return out


def _audit(out, capsys, *extra):
    code = track.main(["--audit", "--out", str(out), *extra])
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err.splitlines()


def test_the_marker_line_is_unchanged(a_track, capsys):
    """The one line a machine reads keeps its exact shape."""
    _code, out, _err = _audit(a_track, capsys)
    assert out[0] == f"TRACK_AUDIT_OK {a_track}"


def test_a_pass_names_every_partition_and_its_size(a_track, capsys):
    _code, out, _err = _audit(a_track, capsys)
    body = "\n".join(out)
    assert "harmful: fit 7, search 5, measure 11" in body, body
    assert "harmless: fit 6, search 4, measure 9" in body, body


def test_a_pass_without_labels_says_the_strata_check_did_not_run(a_track, capsys):
    """The defect this file exists for: a pass that was quieter than it was narrow."""
    _code, out, _err = _audit(a_track, capsys)
    body = "\n".join(out)
    assert "STRATA NOT CHECKED" in body, body
    assert "--labels" in body, "the output does not say how to ask the question it skipped"


def test_a_pass_with_labels_says_the_strata_check_ran_and_how_wide_it_was(
        a_track, capsys, tmp_path, monkeypatch):
    labels = _labels()
    monkeypatch.setattr(track, "read_labels", lambda path: labels)
    label_file = tmp_path / "labels.json"
    label_file.write_text("{}", encoding="utf-8")
    _code, out, _err = _audit(a_track, capsys, "--labels", str(label_file))
    body = "\n".join(out)
    assert "STRATA NOT CHECKED" not in body, body
    # Whitespace-collapsed, because this line is wrapped to the terminal and a substring test
    # against the raw output would pass or fail on where the break happens to land.
    flat = " ".join(body.split())
    assert f"{len(labels)} labelled prompts across 4 strata" in flat, body


def test_the_failure_path_says_what_it_examined_too(a_track, capsys, monkeypatch):
    """A failure had the detail and a pass did not; now both do, and neither has it only.

    Driven by making `check` refuse, rather than by corrupting the fixture, so the test keeps
    working if a new check is added.
    """
    monkeypatch.setattr(track, "check", lambda *a, **kw: ["a reason this track is not trustworthy"])
    with pytest.raises(SystemExit) as exit_info:
        _audit(a_track, capsys)
    assert exit_info.value.code == 1
    err = "\n".join(capsys.readouterr().err.splitlines())
    assert "TRACK_AUDIT_FAILED" in err
    assert "a reason this track is not trustworthy" in err
    assert "harmful: fit 7, search 5, measure 11" in err, err


@pytest.mark.parametrize("columns", [80, 120, 40])
def test_every_line_but_the_marker_fits_the_terminal(a_track, capsys, monkeypatch, columns):
    """The strata notice is 188 characters, which is what made this worth asserting.

    The marker line is exempt and the exemption is `say`'s, not this file's: it carries a path, it
    is read by a grep, and wrapping it would be a change to a machine interface. Every other line
    here is prose.
    """
    from senbonzakura import say
    monkeypatch.setenv("COLUMNS", str(columns))
    _code, out, _err = _audit(a_track, capsys)
    for line in out:
        if say.is_marker(line):
            continue
        assert len(line) <= say.CEILING, f"{len(line)} columns at COLUMNS={columns}: {line!r}"


def test_audit_still_returns_a_bare_list_of_failures(a_track):
    """`audit` has three call sites and a public signature; the summary is additive.

    Asserted because the refactor moved its body into `audit_report`, and a function that
    quietly started returning a tuple would break `promote` and `score` with a truthiness bug
    rather than an error: a non-empty tuple is always truthy, so `if failures:` would refuse
    every track.
    """
    failures = track.audit(a_track)
    assert isinstance(failures, list)
    assert failures == []


def test_the_summary_and_the_failures_come_from_one_read_of_the_partitions(a_track, monkeypatch):
    """Two reads would be two chances to disagree, which is this file's own recurring defect.

    `load_partitions`'s docstring records why: the boundaries are recorded rather than derivable,
    so two copies of the slicing is the shape that has already put published numbers on the wrong
    rows here.
    """
    reads = []
    real = track.load_partitions
    monkeypatch.setattr(track, "load_partitions",
                        lambda path: (reads.append(path), real(path))[1])
    track.audit_report(a_track)
    assert len(reads) == 1, f"the partitions were read {len(reads)} times for one audit"
