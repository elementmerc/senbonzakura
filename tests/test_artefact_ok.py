"""Tests for tools/artefact_ok.py, the resume guard that checks WHICH configuration made a file.

The defect this replaces cost a day: every guard tested whether a result file parsed, so a run
that existed to measure three code changes reused the previous night's outputs and reported
success. These tests are written around that failure rather than around the happy path.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "artefact_ok", Path(__file__).resolve().parent.parent / "tools" / "artefact_ok.py")
ao = importlib.util.module_from_spec(_SPEC)
sys.modules["artefact_ok"] = ao
_SPEC.loader.exec_module(ao)


@pytest.fixture
def result(tmp_path):
    def write(doc, name="out.json"):
        p = tmp_path / name
        p.write_text(json.dumps(doc), encoding="utf-8")
        return str(p)
    return write


# ── the failure that motivated the whole file ─────────────────────────────────────────
def test_a_file_from_a_different_configuration_is_rejected(result, capsys):
    """The 2026-08-04 failure, exactly: valid JSON, wrong provenance, reported as complete."""
    p = result({"model": "Qwen/Qwen3-1.7B", "seed": 42, "start_layer": 15})
    assert ao.main([p, "start_layer=9"]) == 1
    out = capsys.readouterr().out
    assert "different configuration" in out and "start_layer" in out


def test_a_file_from_the_same_configuration_is_reused(result):
    p = result({"model": "Qwen/Qwen3-1.7B", "seed": 42, "init": "mean-diff"})
    assert ao.main([p, "model=Qwen/Qwen3-1.7B", "seed=42", "init=mean-diff"]) == 0


def test_a_guard_with_no_expectations_is_refused(result, capsys):
    """That call IS the old broken guard, so it must not be expressible."""
    p = result({"model": "x"})
    assert ao.main([p]) == 2
    assert "proves nothing" in capsys.readouterr().err


def test_a_key_the_file_never_recorded_cannot_be_shown_to_match(result, capsys):
    """Absence is not agreement. An older writer that never wrote the key must not pass."""
    p = result({"model": "Qwen/Qwen3-1.7B", "seed": 42})
    assert ao.main([p, "induce=0.2"]) == 1
    assert "does not record it" in capsys.readouterr().out


# ── reading the file at all ───────────────────────────────────────────────────────────
def test_a_missing_file_means_rebuild_not_crash(tmp_path, capsys):
    assert ao.main([str(tmp_path / "nope.json")]) == 1
    assert "does not exist" in capsys.readouterr().out


def test_a_truncated_file_means_rebuild(tmp_path, capsys):
    """A run killed mid-write leaves half a JSON document; that is a rebuild, not a failure."""
    p = tmp_path / "half.json"
    p.write_text('{"model": "Qwen/Qwen3-1.7', encoding="utf-8")
    assert ao.main([str(p), "model=Qwen/Qwen3-1.7B"]) == 1
    assert "unreadable" in capsys.readouterr().out


# ── lookup behaviour ──────────────────────────────────────────────────────────────────
def test_values_compare_as_strings_so_the_spec_need_not_know_the_json_type(result):
    p = result({"seed": 42, "preserve": 1.0})
    assert ao.main([p, "seed=42"]) == 0


def test_a_float_written_as_1_0_does_not_match_the_string_1(result):
    """Compared as strings, so 1.0 and 1 differ. Better a spurious rebuild than a false reuse."""
    p = result({"preserve": 1.0})
    assert ao.main([p, "preserve=1"]) == 1
    assert ao.main([p, "preserve=1.0"]) == 0


def test_a_dotted_path_reaches_into_nested_dicts(result):
    p = result({"e4": {"strengths": [0.2], "degenerate_reason": None}})
    assert ao.main([p, "e4.degenerate_reason=null"]) == 0
    assert ao.main([p, "e4.degenerate_reason=something"]) == 1


def test_json_null_is_spelled_null_and_a_literal_none_string_does_not_satisfy_it(result):
    """`degenerate_reason=null` means "the grid was readable" and is a guard worth writing.

    Comparing str(got) alone made a real null and the recorded TEXT "None" indistinguishable, so a
    file that had recorded the string would have satisfied a guard asking for the null.
    """
    real_null = result({"degenerate_reason": None}, "a.json")
    literal = result({"degenerate_reason": "None"}, "b.json")
    assert ao.main([real_null, "degenerate_reason=null"]) == 0
    assert ao.main([literal, "degenerate_reason=null"]) == 1, (
        "the string 'None' satisfied a guard asking for JSON null")
    assert ao.main([literal, "degenerate_reason=None"]) == 0


def test_booleans_may_be_written_the_json_way_or_the_python_way(result):
    p = result({"warm_started": True})
    assert ao.main([p, "warm_started=true"]) == 0
    assert ao.main([p, "warm_started=True"]) == 0
    assert ao.main([p, "warm_started=false"]) == 1


def test_only_the_root_is_searched_so_no_answer_can_come_from_elsewhere(result):
    """The container fallback could satisfy a guard from a place the real container never held.

    With a `meta` fallback, `arms.K` died inside the root's real `arms` (a list, not a dict) and
    then silently retried under `meta`, where a planted value satisfied it. Every writer in this
    repo puts configuration at the root, so the fallback bought nothing and cost exactly this.
    """
    p = result({"arms": [{"K": 1}, {"K": 2}], "meta": {"arms": {"K": "9"}}})
    assert ao.main([p, "arms.K=9"]) == 1, "an expectation was satisfied from the wrong container"

    q = result({"meta": {"seed": 43}}, "q.json")
    assert ao.main([q, "seed=43"]) == 1, "a root-only lookup answered from meta"


def test_a_list_rooted_artefact_says_why_it_cannot_be_checked(result, capsys):
    """cli.py writes trials.json as a bare array; the guard must explain, not list every key."""
    p = result([{"trial": 1}, {"trial": 2}])
    assert ao.main([p, "seed=42"]) == 1
    assert "not an object" in capsys.readouterr().out


def test_lookup_survives_a_path_that_runs_into_a_non_dict(result):
    """`a.b` where `a` is a list must report missing, not raise."""
    p = result({"a": [1, 2, 3]})
    assert ao.main([p, "a.b=1"]) == 1


def test_every_failing_key_is_reported_not_just_the_first(result, capsys):
    """A run rebuilt for three reasons should say three reasons; one at a time wastes a day."""
    p = result({"model": "A", "seed": 1, "init": "random"})
    assert ao.main([p, "model=B", "seed=2", "init=mean-diff"]) == 1
    out = capsys.readouterr().out
    assert "model" in out and "seed" in out and "init" in out


# ── argument handling ─────────────────────────────────────────────────────────────────
def test_a_malformed_expectation_is_a_usage_error_not_a_silent_pass(result, capsys):
    p = result({"seed": 42})
    assert ao.main([p, "seed"]) == 2
    assert "expected key=value" in capsys.readouterr().err


def test_an_empty_key_is_refused(result):
    p = result({"seed": 42})
    assert ao.main([p, "=42"]) == 2


def test_a_value_containing_an_equals_sign_survives_the_split(result):
    p = result({"note": "a=b"})
    assert ao.main([p, "note=a=b"]) == 0


def test_parse_expectations_keeps_order_and_pairs():
    assert ao.parse_expectations(["a=1", "b=2"]) == [("a", "1"), ("b", "2")]


def test_mismatches_is_empty_when_everything_agrees():
    assert ao.mismatches({"a": 1}, [("a", "1")]) == []
