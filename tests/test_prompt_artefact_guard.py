"""Tests for the retention guard (tools/check_prompt_artefacts.py).

The guard is the layer that does not trust `.gitignore`. An ignore rule stops an
accidental `git add .` and nothing else: not `git add -f`, not a path that matches
no pattern, not a file moved into a tracked directory. So the guard reads what is
actually staged, and these tests are mostly about the ways a file can slip past a
naive check: a nested key, a key inside a list, a mixed-case key, and a file that
cannot be parsed at all.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_prompt_artefacts",
    Path(__file__).resolve().parent.parent / "tools" / "check_prompt_artefacts.py")
guard = importlib.util.module_from_spec(_SPEC)
sys.modules["check_prompt_artefacts"] = guard
_SPEC.loader.exec_module(guard)


# ── the recursive walk ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("obj", [
    {"prompt": "x"},
    {"rows": [{"generation": "x"}]},                       # inside a list
    {"a": {"b": {"c": {"prompt": "x"}}}},                  # deeply nested
    {"PROMPT": "x"},                                       # case-folded
    {"Generations": ["x"]},
    [{"prompt": "x"}],                                     # a bare top-level list
])
def test_banned_keys_are_found_wherever_they_hide(obj):
    assert guard.banned_keys_in(obj)


@pytest.mark.parametrize("obj", [
    {"auc": 0.83, "n_harmful": 200},
    {"label": "after", "margins_path": "results/x.jsonl"},   # a path is not a prompt
    {"i": 0, "set": "harmful", "margin": -1.25},
    {},
    [],
    "a bare string",
    None,
])
def test_a_stripped_artefact_is_clean(obj):
    assert not guard.banned_keys_in(obj)


def test_the_walk_is_depth_capped():
    """A structure deeper than the cap stops rather than recursing without bound."""
    deep = {"prompt": "buried"}
    for _ in range(guard.MAX_DEPTH + 5):
        deep = {"n": deep}
    assert not guard.banned_keys_in(deep)


def test_a_key_named_after_a_prompt_field_is_reported_by_name():
    assert guard.banned_keys_in({"prompt": "a", "generations": ["b"]}) == {"prompt", "generations"}


# ── scanning files ─────────────────────────────────────────────────────────────────
def test_a_clean_json_passes(tmp_path):
    p = tmp_path / "res.json"
    p.write_text(json.dumps({"auc": 0.9, "skip_harmful": 128}), encoding="utf-8")
    assert guard.scan_file(p) == []


def test_a_result_file_with_prompts_is_caught(tmp_path):
    p = tmp_path / "res.json"
    p.write_text(json.dumps({"auc": 0.9, "rows": [{"prompt": "do the bad thing"}]}), encoding="utf-8")
    (finding,) = guard.scan_file(p)
    assert "carries prompt" in finding


def test_jsonl_reports_the_offending_line_number(tmp_path):
    """Which row matters: a 200-line artefact with one unstripped row is the likely case."""
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join([
        json.dumps({"i": 0, "margin": 1.0}),
        json.dumps({"i": 1, "margin": 2.0}),
        json.dumps({"i": 2, "margin": 3.0, "prompt": "do the bad thing"}),
    ]), encoding="utf-8")
    (finding,) = guard.scan_file(p)
    assert ":3:" in finding


def test_blank_lines_in_jsonl_are_not_findings(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"margin": 1.0}) + "\n\n   \n", encoding="utf-8")
    assert guard.scan_file(p) == []


def test_unparseable_is_reported_rather_than_skipped(tmp_path):
    """"Cannot be read" is not "harmless", and a truncated artefact is exactly the case."""
    p = tmp_path / "broken.json"
    p.write_text('{"auc": 0.9, "rows": [{"prom', encoding="utf-8")
    (finding,) = guard.scan_file(p)
    assert "cannot be cleared" in finding


def test_an_overlong_jsonl_line_is_reported_not_parsed(tmp_path):
    """A multi-megabyte single line is itself the thing worth looking at."""
    p = tmp_path / "huge.jsonl"
    p.write_text(json.dumps({"margin": 1.0, "pad": "x" * (guard.MAX_LINE_BYTES + 10)}), encoding="utf-8")
    (finding,) = guard.scan_file(p)
    assert "too large to check" in finding


def test_a_file_that_cannot_be_read_is_a_finding(tmp_path):
    assert "could not be read" in guard.scan_file(tmp_path / "absent.json")[0]


# ── collecting targets ─────────────────────────────────────────────────────────────
def test_collect_walks_directories_and_ignores_other_suffixes(tmp_path):
    (tmp_path / "nested").mkdir()
    for name in ("a.json", "nested/b.jsonl", "c.txt", "d.safetensors"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    got = {p.name for p in guard.collect([str(tmp_path)])}
    assert got == {"a.json", "b.jsonl"}


def test_collect_takes_named_files_directly(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}", encoding="utf-8")
    assert guard.collect([str(f), str(tmp_path / "skip.txt")]) == [f]


# ── the command ────────────────────────────────────────────────────────────────────
def test_main_returns_zero_on_a_clean_tree(tmp_path, capsys):
    (tmp_path / "res.json").write_text(json.dumps({"auc": 0.9}), encoding="utf-8")
    assert guard.main([str(tmp_path)]) == 0
    assert "clean" in capsys.readouterr().out


def test_main_returns_one_and_explains_why(tmp_path, capsys):
    (tmp_path / "res.jsonl").write_text(json.dumps({"prompt": "x"}), encoding="utf-8")
    assert guard.main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "FAILED" in err
    assert "public remote" in err          # the reason, not just the verdict


def test_main_refuses_both_staged_and_paths(tmp_path):
    """Ambiguous invocation is a usage error, not a silent preference for one."""
    with pytest.raises(SystemExit) as e:
        guard.main(["--staged", str(tmp_path)])
    assert e.value.code == 2


def test_main_refuses_neither_staged_nor_paths():
    with pytest.raises(SystemExit) as e:
        guard.main([])
    assert e.value.code == 2


def test_staged_paths_keeps_only_json_and_jsonl(monkeypatch):
    """Git reports every staged path; only two suffixes are this guard's business."""
    class _Done:
        stdout = b"a.json\0b.jsonl\0c.py\0d.safetensors\0"

    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: _Done())
    assert [p.name for p in guard.staged_paths()] == ["a.json", "b.jsonl"]


def test_staged_paths_fails_loudly_when_git_cannot_be_run(monkeypatch):
    def boom(*a, **k):
        raise OSError("no git here")

    monkeypatch.setattr(guard.subprocess, "run", boom)
    with pytest.raises(SystemExit) as e:
        guard.staged_paths()
    assert e.value.code == 2


def test_main_scans_what_is_staged(monkeypatch, tmp_path, capsys):
    f = tmp_path / "staged.jsonl"
    f.write_text(json.dumps({"generation": "here you go"}), encoding="utf-8")
    monkeypatch.setattr(guard, "staged_paths", lambda: [f])
    assert guard.main(["--staged"]) == 1
    assert "carries generation" in capsys.readouterr().err


# ── the guard against its own repository ───────────────────────────────────────────
def test_this_repository_is_clean():
    """The check CI runs, run here too, so a bad commit fails before it is written."""
    root = Path(__file__).resolve().parent.parent
    assert guard.main([str(root / "evidence"), str(root / "docs"), str(root / "holst")]) == 0
