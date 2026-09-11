# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""`senbonzakura check`: the thirty-second finding, and the three outcomes it must tell apart.

Item G of the shortlist, property 1 of the six. `strategy-2026-08-02.md` on why the 10x axis
scored "partly" and what would fix it: *"the axis is trust, and trust does not demo. Time to
find a real bug does."*

MOST OF WHAT IS TESTED HERE IS THE OUTPUT, not the arithmetic, and that is the right emphasis.
The checks are tested in `test_check_registry.py` and the adapters in `test_check_adapters.py`.
What this command adds is a report somebody has to be able to act on and an exit code a CI job
has to be able to branch on, and both of those are where a checker quietly becomes useless.
"""
import io
import json

import pytest

from senbonzakura.check import cli

GOOD = {"version": 2, "status": "success",
        "eval": {"task": "t", "model": "m"},
        "results": {"total_samples": 10, "completed_samples": 10,
                    "scores": [{"name": "s", "scorer": "choice", "scored_samples": 10,
                                "metrics": {"accuracy": {"name": "accuracy", "value": 0.5}}}]}}

BAD = json.loads(json.dumps(GOOD))
BAD["results"]["scores"][0]["scorer"] = None


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _run(argv):
    out = io.StringIO()
    code = cli.main(argv, out=out)
    return code, out.getvalue()


# ── the three outcomes, which must never be confused ─────────────────────────────────────────

def test_a_clean_file_exits_zero(tmp_path):
    code, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    assert code == 0
    assert "nothing found" in text


def test_a_file_with_a_finding_exits_one(tmp_path):
    code, text = _run([str(_write(tmp_path, "bad.json", BAD))])
    assert code == 1
    assert "metric-reported-without-its-estimator" in text


def test_a_file_that_cannot_be_read_exits_two(tmp_path):
    """A CI GATE THAT TREATS "could not check" THE SAME AS "checked, nothing found" IS WORSE
    THAN NO GATE, because it goes green on the day the format changes under it.
    """
    p = tmp_path / "strange.json"
    p.write_text(json.dumps({"unrelated": "json"}), encoding="utf-8")
    code, text = _run([str(p)])
    assert code == 2
    assert "UNCHECKED" in text


def test_unreadable_beats_a_finding_in_the_exit_code(tmp_path):
    """When both happen, the louder problem wins: a file nobody could parse is the one that
    invalidates the run, and 1 would let a CI job treat the report as complete.
    """
    code, _ = _run([
        str(_write(tmp_path, "bad.json", BAD)),
        str(_write(tmp_path, "strange.json", {"unrelated": 1})),
    ])
    assert code == 2


@pytest.mark.parametrize(("content", "why"), [
    ("{not json", "malformed JSON"),
    ("[1, 2, 3]", "valid JSON that is not an object"),
    ("null", "valid JSON that is nothing at all"),
])
def test_every_unreadable_shape_is_reported_rather_than_crashing(content, why, tmp_path):
    p = tmp_path / "x.json"
    p.write_text(content, encoding="utf-8")
    code, text = _run([str(p)])
    assert code == 2, why
    assert "UNCHECKED" in text


def test_a_missing_file_is_reported_rather_than_crashing(tmp_path):
    code, text = _run([str(tmp_path / "absent.json")])
    assert code == 2
    assert "could not read it" in text


# ── what the report has to say ───────────────────────────────────────────────────────────────

def test_a_finding_carries_everything_needed_to_judge_it(tmp_path):
    """THE WHOLE VALUE OF THE COMMAND IS IN THIS ASSERTION.

    A finding that says only what fired sends the reader to the source. Each of these four is a
    separate requirement from the v0.8 plan: the incident is the citation without which a finding
    is an opinion, and the false-positive sentence is what lets a reader decide whether to
    believe it.
    """
    code, text = _run([str(_write(tmp_path, "bad.json", BAD))])
    assert code == 1
    for label in ("what it is:", "seen before:", "what to do:",
                  "when this check is wrong:"):
        assert label in text, f"a finding printed without {label!r}"
    assert "2026-08-05" in text, "the incident has to be quoted, not just referenced"


def test_the_summary_says_how_many_checks_did_not_apply(tmp_path):
    """SKIPPED IS NOT PASSED. "No findings" over a set of checks that mostly could not run is a
    different statement from "no findings", and the reader is entitled to tell them apart.
    """
    _, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    assert "did not apply" in text


def test_a_clean_report_refuses_to_read_as_a_certificate(tmp_path):
    """Loophole 7 of the v0.8 plan, and it is in the OUTPUT rather than the README because
    somebody will otherwise quote a clean report as a claim of correctness.
    """
    _, text = _run([str(_write(tmp_path, "ok.json", GOOD))])
    assert "cannot tell you a number is right" in text


def test_quiet_prints_findings_and_nothing_else(tmp_path):
    _, text = _run(["--quiet", str(_write(tmp_path, "ok.json", GOOD)),
                    str(_write(tmp_path, "bad.json", BAD))])
    assert "bad.json" in text
    assert "ok.json" not in text
    assert "cannot tell you" not in text


def test_json_output_is_machine_readable_and_complete(tmp_path):
    """The shape a CI action consumes. Every field a human sees is in here too, because an
    action that has to re-derive the caveat from the check id will not bother.
    """
    code, text = _run(["--json", str(_write(tmp_path, "bad.json", BAD))])
    assert code == 1
    got = json.loads(text)
    assert len(got) == 1
    entry = got[0]
    assert entry["unchecked"] is None
    assert entry["findings"][0]["check"] == "metric-reported-without-its-estimator"
    for key in ("title", "confidence", "detects", "incident", "remedy", "false_positive"):
        assert entry["findings"][0][key]


def test_json_output_records_an_unchecked_file_too(tmp_path):
    p = tmp_path / "strange.json"
    p.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
    code, text = _run(["--json", str(p)])
    assert code == 2
    assert json.loads(text)[0]["unchecked"]


# ── walking a directory ──────────────────────────────────────────────────────────────────────

def test_a_directory_is_searched_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    _write(tmp_path, "a.json", GOOD)
    _write(tmp_path / "nested", "b.json", BAD)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    code, text = _run([str(tmp_path)])
    assert code == 1
    assert "2 file(s)" in text, "the .txt must not be read as a result artefact"


def test_files_are_visited_in_a_stable_order(tmp_path):
    """Two runs over the same tree produce the same report, per baseline section 2.1. A report
    whose order depends on the filesystem cannot be diffed between runs.
    """
    for name in ("c.json", "a.json", "b.json"):
        _write(tmp_path, name, BAD)
    first = _run([str(tmp_path)])[1]
    second = _run([str(tmp_path)])[1]
    assert first == second
    assert first.index("a.json") < first.index("b.json") < first.index("c.json")


def test_an_empty_directory_is_not_an_error(tmp_path):
    code, text = _run([str(tmp_path)])
    assert code == 0
    assert "0 file(s)" in text


# ── the command is wired in, and stays cheap ─────────────────────────────────────────────────

def test_check_is_a_delegated_command():
    from senbonzakura import entry

    assert entry.DELEGATED["check"] == ("check.cli", "main")
    assert entry.dispatch("check") is cli.main


def test_the_command_runs_from_the_entry_point(tmp_path, capsys):
    """Through `entry.main`, which is the path a user's shell actually takes."""
    from senbonzakura import entry

    code = entry.main(["check", str(_write(tmp_path, "bad.json", BAD))])
    assert code == 1
    assert "metric-reported-without-its-estimator" in capsys.readouterr().out


def test_the_checker_does_not_write_anything_where_it_looked(tmp_path):
    """It is pointed at other people's evidence and the one unforgivable behaviour is changing
    it. Asserted on the directory as well as the file, because a report written beside the
    input would be just as wrong.
    """
    p = _write(tmp_path, "bad.json", BAD)
    before = (p.read_bytes(), p.stat().st_mtime_ns, sorted(q.name for q in tmp_path.iterdir()))
    _run([str(tmp_path)])
    assert (p.read_bytes(), p.stat().st_mtime_ns,
            sorted(q.name for q in tmp_path.iterdir())) == before
