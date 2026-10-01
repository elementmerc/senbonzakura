# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A regression caught in March is only evidence in June if March wrote something down.

The gate read two files, compared, logged and returned a status. Nothing was written, while the
design said a history existed, and the project already carries the bill for that shape: a
published K-sweep headline whose per-seed values live in a session transcript and nowhere in the
tree, so the number cannot be re-judged by anybody including its author.

TWO PROPERTIES, AND THEY PULL AGAINST EACH OTHER. A history has to be complete, so every run
writes; and a build step has to be safe to re-run after an interruption, so writing twice must
not corrupt anything. Naming each record after its own contents gives both: the same comparison
lands on the same file and the first recording stands, a different comparison can never collide,
and nothing is ever overwritten. These tests pin both halves, because an implementation that
satisfies one by breaking the other looks correct from either side on its own.
"""
from __future__ import annotations

import json

import pytest

from senbonzakura import baseline as b
from senbonzakura import gate

COMMON = {
    "model": "Qwen/Qwen3-1.7B",
    "metric": "refusal_rate",
    "direction": b.LOWER_IS_BETTER,
    "input_digest": "sha256:aaa",
    "partition": "measure",
    "prompt_format": "renderer:v3",
    "tool_version": "0.4.1",
    "estimator": "senbonzakura-ruler",
    "precision": "bfloat16",
    "seeds": [42],
    "n": 200,
}

STEADY = {"point": 0.102, "interval": (0.068, 0.142)}
WORSE = {"point": 0.284, "interval": (0.231, 0.341)}
DRIFTED = {"point": 0.102, "interval": (0.068, 0.142), "prompt_format": "renderer:v4"}


@pytest.fixture
def made(tmp_path):
    paths = {}
    for name, over in (("base", {"point": 0.094, "interval": (0.061, 0.133)}),
                       ("steady", STEADY), ("worse", WORSE), ("drifted", DRIFTED)):
        path = tmp_path / f"{name}.json"
        b.write(path, b.record(**{**COMMON, **over}))
        paths[name] = path
    return paths


def _gate(made, which, history=None, *extra, log=None):
    lines = [] if log is None else log
    args = ["--baseline", str(made["base"]), "--current", str(made[which])]
    if history is not None:
        args += ["--history", str(history)]
    status = gate.run([*args, *extra], log=lines.append)
    return status, "\n".join(lines)


def _records(directory):
    return sorted(p for p in directory.rglob("*.json"))


class TestEveryRunWritesItsMeasurement:
    @pytest.mark.parametrize(("which", "status"), [
        ("steady", gate.OK), ("worse", gate.REGRESSED), ("drifted", gate.REFUSED)])
    def test_pass_fail_and_refusal_are_all_recorded(self, made, tmp_path, which, status):
        """A refusal is recorded too, and that is the one worth arguing for.

        "The gate refused every day that fortnight" is the answer to why nothing was caught, and
        a history with no row for a refusal reads as though the gate had been passing.
        """
        history = tmp_path / "history"
        got, _said = _gate(made, which, history)
        assert got == status
        written = _records(history)
        assert len(written) == 1, f"{which} wrote {len(written)} record(s)"

    def test_the_record_carries_the_whole_measurement_and_not_just_the_number(self, made, tmp_path):
        """A history holding the figure and not the conditions is a history nobody can re-judge,
        and re-judging is what somebody disputing a verdict in June will want to do.
        """
        history = tmp_path / "history"
        _gate(made, "worse", history)
        doc = json.loads(_records(history)[0].read_text(encoding="utf-8"))
        assert doc["schema"] == gate.RUN_SCHEMA
        assert doc["status"] == gate.REGRESSED
        assert doc["verdict"] == "REGRESSED"
        assert doc["metric"] == "refusal_rate"
        assert doc["partition"] == "measure"
        assert doc["recorded_at"].endswith("Z")
        for field in b.PINNED:
            assert field in doc["measurement"], (
                f"the recorded measurement has no {field}, so this row cannot be re-judged")
        assert doc["measurement"]["point"] == WORSE["point"]

    def test_the_record_names_both_measurements_by_address(self, made, tmp_path):
        """Not by path. A path is where a file happened to be on one runner."""
        history = tmp_path / "history"
        _gate(made, "steady", history)
        doc = json.loads(_records(history)[0].read_text(encoding="utf-8"))
        assert doc["baseline"] == b.read(made["base"])[b.ADDRESS_FIELD]
        assert doc["current"] == b.read(made["steady"])[b.ADDRESS_FIELD]

    def test_the_run_says_where_it_wrote(self, made, tmp_path):
        """A record nobody can find is a record nobody will read."""
        history = tmp_path / "history"
        _, said = _gate(made, "steady", history)
        path = _records(history)[0]
        assert str(path) in said, said

    def test_nothing_is_written_when_no_history_was_asked_for(self, made, tmp_path):
        """The gate's default is still to write nothing, so wiring it in costs nobody a directory
        they did not ask for.
        """
        _gate(made, "steady")
        assert not list(tmp_path.rglob("*/*.json")), "a record appeared with no --history"


class TestReRunningIsSafe:
    def test_the_same_comparison_records_once(self, made, tmp_path):
        history = tmp_path / "history"
        _gate(made, "steady", history)
        first = _records(history)
        _gate(made, "steady", history)
        assert _records(history) == first, "a re-run of one comparison wrote a second record"

    def test_the_second_run_says_it_was_already_recorded(self, made, tmp_path):
        """Rather than printing a path it did not write, which reads as a fresh recording."""
        history = tmp_path / "history"
        _gate(made, "steady", history)
        _, said = _gate(made, "steady", history)
        assert "already recorded" in said, said

    def test_the_first_recording_is_not_touched(self, made, tmp_path, monkeypatch):
        """The first run's recording is the one with the date that matters: it is when the thing
        was caught, and "caught in March" is the whole reason a history exists.

        THE CLOCK IS STUBBED RATHER THAN TRUSTED. Comparing the two files' bytes with the real
        clock passes whenever both runs land inside the same second, which in a test suite is
        always, so the assertion would hold over an implementation that rewrites the file every
        time. Measured: with the real clock, removing the already-recorded branch left this test
        green.
        """
        history = tmp_path / "history"
        monkeypatch.setattr(gate, "_now", lambda: "2026-03-01T09:00:00Z")
        _gate(made, "steady", history)
        path = _records(history)[0]
        monkeypatch.setattr(gate, "_now", lambda: "2026-06-01T09:00:00Z")
        _gate(made, "steady", history)
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["recorded_at"] == "2026-03-01T09:00:00Z", (
            "a later run rewrote the record, so the date the regression was caught is gone")

    def test_different_comparisons_do_not_collide(self, made, tmp_path):
        history = tmp_path / "history"
        for which in ("steady", "worse", "drifted"):
            _gate(made, which, history)
        assert len(_records(history)) == 3, [p.name for p in _records(history)]


class TestWhenTheHistoryCannotBeWritten:
    def test_a_regression_is_still_a_regression(self, made, tmp_path):
        """A full disk is a fact about the disk. It must not change the verdict in either
        direction, and the status is what a build system reads.
        """
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("I am a file", encoding="utf-8")
        status, said = _gate(made, "worse", blocked)
        assert status == gate.REGRESSED, (
            "an unwritable history changed the verdict, which is the one thing it must never do")
        assert "could NOT be recorded" in said, said
        assert "leaves no evidence behind" in said, said

    def test_a_pass_is_still_a_pass_and_says_it_recorded_nothing(self, made, tmp_path):
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("I am a file", encoding="utf-8")
        status, said = _gate(made, "steady", blocked)
        assert status == gate.OK
        assert "could NOT be recorded" in said, said

    def test_an_unreadable_baseline_records_nothing_and_says_so(self, made, tmp_path):
        """A row saying "a run happened and we cannot say what it compared" looks like evidence
        and is not, so none is written. The reader is told that rather than left to assume.
        """
        history = tmp_path / "history"
        lines = []
        status = gate.run(["--baseline", str(tmp_path / "absent.json"),
                           "--current", str(made["steady"]),
                           "--history", str(history)], log=lines.append)
        assert status == gate.REFUSED
        assert not history.exists() or not _records(history)
        assert "nothing was recorded" in "\n".join(lines), lines


class TestTheSliceIsNamedBesideTheVerdict:
    """Loophole 5's other half. "One property, on one track" does not say which."""

    @pytest.mark.parametrize("which", ["steady", "worse"])
    def test_the_verdict_names_the_metric_the_partition_and_the_sample_size(self, made, which):
        _, said = _gate(made, which)
        assert "Checked: refusal_rate on the measure partition" in said, said
        assert "n=200" in said, said
        assert "senbonzakura-ruler" in said, said

    @pytest.mark.parametrize("which", ["steady", "worse"])
    def test_the_verdict_says_what_it_did_not_cover(self, made, which):
        _, said = _gate(made, which)
        assert "Not checked" in said, said

    def test_the_caveat_matches_the_verdict_it_follows(self, made):
        _, passed = _gate(made, "steady")
        _, failed = _gate(made, "worse")
        assert "A pass means" in passed and "A pass means" not in failed
        assert "A failure means" in failed and "A failure means" not in passed


class TestAddressesAndTheStore:
    def test_two_identical_measurements_share_one_address(self, tmp_path):
        one = b.record(**{**COMMON, **STEADY})
        two = b.record(**{**COMMON, **STEADY})
        assert one[b.ADDRESS_FIELD] == two[b.ADDRESS_FIELD]
        assert b.is_address(one[b.ADDRESS_FIELD])

    def test_a_different_measurement_has_a_different_address(self):
        assert (b.record(**{**COMMON, **STEADY})[b.ADDRESS_FIELD]
                != b.record(**{**COMMON, **WORSE})[b.ADDRESS_FIELD])

    def test_addressing_a_document_that_already_carries_one_is_stable(self):
        """Without this, stamping would change the content and the content would change the
        stamp, and no document would ever address to what it says it addresses to.
        """
        doc = b.record(**{**COMMON, **STEADY})
        assert b.address(doc) == doc[b.ADDRESS_FIELD]

    def test_a_path_is_not_mistaken_for_an_address(self):
        for not_an_address in ("base.json", "sha256:short", "sha256:" + "Z" * 64,
                               "/tmp/sha256:" + "a" * 64, ""):
            assert not b.is_address(not_an_address), not_an_address

    def test_the_store_lays_measurements_out_by_address(self, tmp_path):
        """`hash[0:2]/hash[2:4]/hash`, per baseline section 12.3, so the layout works unchanged
        on a local disk and on object storage.
        """
        doc = b.record(**{**COMMON, **STEADY})
        path = b.write_addressed(tmp_path / "store", doc)
        digest = doc[b.ADDRESS_FIELD].split(":", 1)[1]
        assert path.relative_to(tmp_path / "store").as_posix() == \
            f"{digest[:2]}/{digest[2:4]}/{digest}.json"

    def test_writing_the_same_measurement_twice_is_one_file(self, tmp_path):
        store = tmp_path / "store"
        doc = b.record(**{**COMMON, **STEADY})
        first = b.write_addressed(store, doc)
        before = first.read_bytes()
        assert b.write_addressed(store, b.record(**{**COMMON, **STEADY})) == first
        assert first.read_bytes() == before
        assert len(list(store.rglob("*.json"))) == 1

    def test_the_gate_takes_an_address_when_it_is_told_where_to_look(self, tmp_path):
        store = tmp_path / "store"
        base = b.record(**{**COMMON, "point": 0.094, "interval": (0.061, 0.133)})
        now = b.record(**{**COMMON, **WORSE})
        b.write_addressed(store, base)
        b.write_addressed(store, now)
        lines = []
        status = gate.run(["--store", str(store),
                           "--baseline", base[b.ADDRESS_FIELD],
                           "--current", now[b.ADDRESS_FIELD]], log=lines.append)
        assert status == gate.REGRESSED, "\n".join(lines)
        assert "REGRESSED" in "\n".join(lines)

    def test_an_address_with_no_store_says_there_is_nowhere_to_look(self, made):
        lines = []
        status = gate.run(["--baseline", "sha256:" + "a" * 64,
                           "--current", str(made["steady"])], log=lines.append)
        assert status == gate.REFUSED
        assert "nowhere to look" in "\n".join(lines), lines

    def test_an_address_the_store_does_not_hold_refuses_rather_than_crashing(self, tmp_path, made):
        lines = []
        status = gate.run(["--store", str(tmp_path / "store"),
                           "--baseline", "sha256:" + "a" * 64,
                           "--current", str(made["steady"])], log=lines.append)
        assert status == gate.REFUSED
        assert "no baseline at" in "\n".join(lines), lines

    def test_an_edited_measurement_no_longer_matches_its_own_address(self, tmp_path, made):
        doc = json.loads(made["steady"].read_text(encoding="utf-8"))
        doc["point"] = 0.5
        edited = tmp_path / "edited.json"
        edited.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(b.BaselineError, match="changed since it was recorded"):
            b.read(edited)

    def test_a_measurement_recorded_before_addresses_existed_still_reads(self, tmp_path, made):
        """`SCHEMA` is bumped only for a change that makes an older file mean something
        different, and this is not one: an absent address is unknown rather than wrong.
        """
        doc = json.loads(made["steady"].read_text(encoding="utf-8"))
        del doc[b.ADDRESS_FIELD]
        older = tmp_path / "older.json"
        older.write_text(json.dumps(doc), encoding="utf-8")
        assert b.read(older)["point"] == STEADY["point"]
